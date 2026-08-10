#!/usr/bin/env python3
"""Evaluate one validation-selected checkpoint on the untouched test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from mso_recovery import RECONSTRUCTION_STATUS  # noqa: E402
from mso_recovery.common import sha256_file, write_json  # noqa: E402
from mso_recovery.data import FloorplanManifestDataset, read_manifest  # noqa: E402
from sensemap.explore_model.SenseMapNet import (  # noqa: E402
    DistillMapNetDeconv,
    TeacherMapNet2,
)


METRICS = (
    "unknown_bce",
    "unknown_brier",
    "unknown_precision",
    "unknown_obstacle_recall",
    "unknown_f1",
    "unknown_iou",
    "unknown_false_free_rate",
    "frontier_obstacle_recall",
    "frontier_false_free_rate",
    "free_component_count_abs_error",
    "reachable_free_iou",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=9137)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def component_count(free: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    return max(0, int(count) - 1)


def reachable_component(free: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    count, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(free, dtype=bool)
    label = int(labels[seed])
    if label == 0:
        return np.zeros_like(free, dtype=bool)
    return labels == label


def nearest_known_free_seed(known_free: np.ndarray) -> tuple[int, int] | None:
    points = np.argwhere(known_free)
    if not len(points):
        return None
    center = np.asarray(known_free.shape, dtype=np.float64) / 2.0
    index = int(np.argmin(np.square(points - center).sum(axis=1)))
    return int(points[index, 0]), int(points[index, 1])


def sample_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    unknown: np.ndarray,
    known_occupied: np.ndarray,
    known_free: np.ndarray,
) -> dict[str, float | int]:
    selected_probability = np.clip(probability[unknown], 1e-6, 1.0 - 1e-6)
    selected_target = target[unknown]
    decision = selected_probability >= 0.5
    tp = int(np.sum(decision & selected_target))
    fp = int(np.sum(decision & ~selected_target))
    fn = int(np.sum(~decision & selected_target))
    tn = int(np.sum(~decision & ~selected_target))
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision) and math.isfinite(recall) and precision + recall
        else float("nan")
    )
    iou = safe_ratio(tp, tp + fp + fn)

    frontier = unknown & (
        cv2.dilate(known_free.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    )
    frontier_truth = target[frontier]
    frontier_decision = probability[frontier] >= 0.5
    frontier_tp = int(np.sum(frontier_decision & frontier_truth))
    frontier_fn = int(np.sum(~frontier_decision & frontier_truth))

    predicted_occupied = known_occupied | (unknown & (probability >= 0.5))
    target_occupied = target
    predicted_free = ~predicted_occupied
    target_free = ~target_occupied
    target_components = component_count(target_free)
    predicted_components = component_count(predicted_free)
    seed = nearest_known_free_seed(known_free)
    reachable_iou = float("nan")
    if seed is not None:
        predicted_reachable = reachable_component(predicted_free, seed)
        target_reachable = reachable_component(target_free, seed)
        union = int(np.sum(predicted_reachable | target_reachable))
        if union:
            reachable_iou = float(
                np.sum(predicted_reachable & target_reachable) / union
            )

    return {
        "unknown_cells": int(unknown.sum()),
        "unknown_positive_cells": int(selected_target.sum()),
        "unknown_bce": float(
            functional.binary_cross_entropy(
                torch.from_numpy(selected_probability),
                torch.from_numpy(selected_target.astype(np.float32)),
            ).item()
        ),
        "unknown_brier": float(np.mean(np.square(selected_probability - selected_target))),
        "unknown_precision": precision,
        "unknown_obstacle_recall": recall,
        "unknown_f1": f1,
        "unknown_iou": iou,
        "unknown_false_free_rate": safe_ratio(fn, tp + fn),
        "frontier_cells": int(frontier.sum()),
        "frontier_obstacle_recall": safe_ratio(frontier_tp, frontier_tp + frontier_fn),
        "frontier_false_free_rate": safe_ratio(frontier_fn, frontier_tp + frontier_fn),
        "target_free_component_count": target_components,
        "predicted_free_component_count": predicted_components,
        "free_component_count_abs_error": abs(predicted_components - target_components),
        "reachable_free_iou": reachable_iou,
    }


def finite_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(finite.mean()) if len(finite) else float("nan")


def building_cluster_bootstrap(
    rows: list[dict[str, object]], replicates: int, seed: int
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["building_id"])].append(row)
    building_ids = sorted(grouped)
    per_building = {
        metric: np.asarray(
            [
                finite_mean([float(row[metric]) for row in grouped[building_id]])
                for building_id in building_ids
            ],
            dtype=np.float64,
        )
        for metric in METRICS
    }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(building_ids), size=(replicates, len(building_ids)))
    result: dict[str, dict[str, float | int]] = {}
    for metric, values in per_building.items():
        valid = np.isfinite(values)
        observed = float(np.nanmean(values))
        if valid.all():
            bootstrap = values[indices].mean(axis=1)
        else:
            sampled = values[indices]
            bootstrap = np.nanmean(sampled, axis=1)
            bootstrap = bootstrap[np.isfinite(bootstrap)]
        result[metric] = {
            "building_equal_weight_mean": observed,
            "bootstrap_95ci_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_95ci_high": float(np.quantile(bootstrap, 0.975)),
            "evaluated_buildings": int(valid.sum()),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("status") != RECONSTRUCTION_STATUS:
        raise ValueError("Checkpoint is not from the declared replacement experiment")
    if payload.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("Checkpoint and evaluation manifest hashes differ")
    stage = payload["stage"]
    if stage == "student":
        model = DistillMapNetDeconv(image_size=256, dim=4)
    elif stage == "teacher":
        model = TeacherMapNet2(image_size=256, dim=4)
    else:
        raise ValueError(f"Unsupported checkpoint stage: {stage}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().to(args.device)

    dataset = FloorplanManifestDataset(
        args.dataset_root, args.manifest, "test", verify_hashes=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    manifest_rows = read_manifest(args.manifest)
    building_by_sample = {
        row["sample_id"]: row["group_key"] for row in manifest_rows if row["split"] == "test"
    }
    results: list[dict[str, object]] = []
    with torch.inference_mode():
        for features, labels, unknown, sample_ids, floorplans in loader:
            prediction = model(features.to(args.device))[0].float().cpu().numpy()[:, 0]
            labels_np = labels.numpy()[:, 0].astype(bool)
            unknown_np = unknown.numpy()[:, 0].astype(bool)
            features_np = features.numpy().astype(bool)
            for index, sample_id in enumerate(sample_ids):
                metrics = sample_metrics(
                    prediction[index],
                    labels_np[index],
                    unknown_np[index],
                    features_np[index, 0],
                    features_np[index, 2],
                )
                results.append(
                    {
                        "sample_id": sample_id,
                        "floorplan_id": floorplans[index],
                        "building_id": building_by_sample[sample_id],
                        **metrics,
                    }
                )

    per_sample_path = args.output_dir / "per_sample_metrics.csv"
    with per_sample_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "status": RECONSTRUCTION_STATUS,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_sha256": sha256_file(args.manifest),
        "checkpoint_stage": stage,
        "checkpoint_seed": payload["seed"],
        "checkpoint_epoch": payload["epoch"],
        "selection_metric": "validation_unknown_bce",
        "test_samples": len(results),
        "test_buildings": len({row["building_id"] for row in results}),
        "cluster_bootstrap_unit": "building",
        "bootstrap_replicates": args.bootstrap_replicates,
        "metrics": building_cluster_bootstrap(
            results, args.bootstrap_replicates, args.bootstrap_seed
        ),
        "per_sample_metrics_sha256": sha256_file(per_sample_path),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
