#!/usr/bin/env python3
"""Describe validation prediction collapse without accessing the test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Subset

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from mso_recovery import RECONSTRUCTION_STATUS  # noqa: E402
from mso_recovery.common import sha256_file, write_json  # noqa: E402
from mso_recovery.data import FloorplanManifestDataset  # noqa: E402
from sensemap.explore_model.SenseMapNet import (  # noqa: E402
    DistillMapNetDeconv,
    TeacherMapNet2,
)


HISTOGRAM_BINS = 10001
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
THRESHOLDS = tuple(value / 100.0 for value in range(1, 100))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Deterministic validation subset size for a low-impact diagnostic.",
    )
    parser.add_argument("--subset-seed", type=int, default=3020)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def histogram(values: np.ndarray) -> np.ndarray:
    indices = np.minimum(
        (np.clip(values, 0.0, 1.0) * (HISTOGRAM_BINS - 1)).astype(np.int64),
        HISTOGRAM_BINS - 1,
    )
    return np.bincount(indices, minlength=HISTOGRAM_BINS)


def histogram_quantiles(counts: np.ndarray) -> dict[str, float]:
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1])
    if total == 0:
        return {f"q{int(round(q * 100)):02d}": float("nan") for q in QUANTILES}
    output = {}
    for quantile in QUANTILES:
        rank = min(total - 1, max(0, int(round(quantile * (total - 1)))))
        index = int(np.searchsorted(cumulative, rank + 1, side="left"))
        output[f"q{int(round(quantile * 100)):02d}"] = index / (HISTOGRAM_BINS - 1)
    return output


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("status") != RECONSTRUCTION_STATUS:
        raise ValueError("Checkpoint provenance status is incompatible")
    manifest_sha = sha256_file(args.manifest)
    if payload.get("manifest_sha256") != manifest_sha:
        raise ValueError("Checkpoint and validation manifest hashes differ")
    if payload["stage"] == "teacher":
        model = TeacherMapNet2(image_size=256, dim=4)
    elif payload["stage"] == "student":
        model = DistillMapNetDeconv(image_size=256, dim=4)
    else:
        raise ValueError(f"Unsupported stage: {payload['stage']}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().to(args.device)
    full_dataset = FloorplanManifestDataset(
        args.dataset_root, args.manifest, "val", verify_hashes=True
    )
    selected_indices = np.arange(len(full_dataset), dtype=np.int64)
    if args.max_samples is not None and args.max_samples < len(full_dataset):
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        rng = np.random.default_rng(args.subset_seed)
        selected_indices = np.sort(
            rng.choice(len(full_dataset), size=args.max_samples, replace=False)
        )
        dataset = Subset(full_dataset, selected_indices.tolist())
    else:
        dataset = full_dataset
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    all_histogram = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    positive_histogram = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    negative_histogram = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    threshold_positive_counts = {threshold: 0 for threshold in THRESHOLDS}
    threshold_confusion = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for threshold in THRESHOLDS
    }
    total_cells = 0
    positive_cells = 0
    probability_sum = 0.0
    probability_square_sum = 0.0
    bce_sum = 0.0
    per_sample: list[dict[str, object]] = []

    with torch.inference_mode():
        for features, labels, unknown, sample_ids, floorplans in loader:
            prediction = model(features.to(args.device))[0].float().cpu()
            for index, sample_id in enumerate(sample_ids):
                mask = unknown[index, 0].bool()
                values = prediction[index, 0][mask].numpy()
                truth = labels[index, 0][mask].bool().numpy()
                all_histogram += histogram(values)
                positive_histogram += histogram(values[truth])
                negative_histogram += histogram(values[~truth])
                total_cells += len(values)
                positive_cells += int(truth.sum())
                probability_sum += float(values.sum(dtype=np.float64))
                probability_square_sum += float(
                    np.square(values, dtype=np.float64).sum(dtype=np.float64)
                )
                bce_sum += float(
                    functional.binary_cross_entropy(
                        torch.from_numpy(np.clip(values, 1e-6, 1.0 - 1e-6)),
                        torch.from_numpy(truth.astype(np.float32)),
                        reduction="sum",
                    ).item()
                )
                for threshold in THRESHOLDS:
                    decision = values >= threshold
                    threshold_positive_counts[threshold] += int(np.sum(decision))
                    threshold_confusion[threshold]["tp"] += int(
                        np.sum(decision & truth)
                    )
                    threshold_confusion[threshold]["fp"] += int(
                        np.sum(decision & ~truth)
                    )
                    threshold_confusion[threshold]["fn"] += int(
                        np.sum(~decision & truth)
                    )
                    threshold_confusion[threshold]["tn"] += int(
                        np.sum(~decision & ~truth)
                    )
                per_sample.append(
                    {
                        "sample_id": sample_id,
                        "floorplan_id": floorplans[index],
                        "unknown_cells": len(values),
                        "target_positive_fraction": float(truth.mean()),
                        "prediction_mean": float(values.mean()),
                        "prediction_median": float(np.median(values)),
                        "prediction_minimum": float(values.min()),
                        "prediction_maximum": float(values.max()),
                        "prediction_positive_fraction_at_0.5": float(
                            np.mean(values >= 0.5)
                        ),
                    }
                )

    per_sample_path = args.output_dir / "validation_per_sample_diagnostics.csv"
    with per_sample_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(per_sample[0]))
        writer.writeheader()
        writer.writerows(per_sample)
    mean = probability_sum / total_cells
    variance = max(0.0, probability_square_sum / total_cells - mean * mean)
    threshold_table = {}
    for threshold in THRESHOLDS:
        counts = threshold_confusion[threshold]
        precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        threshold_table[f"{threshold:.2f}"] = {
            "prediction_positive_fraction": threshold_positive_counts[threshold]
            / total_cells,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **counts,
        }
    descriptive_best_threshold = max(
        THRESHOLDS, key=lambda threshold: threshold_table[f"{threshold:.2f}"]["f1"]
    )
    report = {
        "status": RECONSTRUCTION_STATUS,
        "diagnostic_boundary": "validation_only_descriptive_no_parameter_selection",
        "test_split_accessed": False,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_stage": payload["stage"],
        "checkpoint_seed": payload["seed"],
        "checkpoint_epoch": payload["epoch"],
        "manifest_sha256": manifest_sha,
        "validation_samples_total": len(full_dataset),
        "validation_samples_diagnosed": len(dataset),
        "validation_subset_seed": args.subset_seed if len(dataset) < len(full_dataset) else None,
        "validation_subset_sample_ids_sha256": hashlib.sha256(
            "\n".join(per_sample_row["sample_id"] for per_sample_row in per_sample).encode(
                "utf-8"
            )
        ).hexdigest(),
        "validation_unknown_cells": total_cells,
        "validation_target_positive_cells": positive_cells,
        "validation_target_positive_fraction": positive_cells / total_cells,
        "validation_unknown_bce": bce_sum / total_cells,
        "prediction_mean": mean,
        "prediction_standard_deviation": variance**0.5,
        "prediction_quantiles_histogram_resolution_1e-4": histogram_quantiles(
            all_histogram
        ),
        "positive_target_prediction_quantiles": histogram_quantiles(
            positive_histogram
        ),
        "negative_target_prediction_quantiles": histogram_quantiles(
            negative_histogram
        ),
        "descriptive_threshold_table_not_used_for_selection": threshold_table,
        "descriptive_max_f1_threshold_not_adopted": descriptive_best_threshold,
        "per_sample_diagnostics_sha256": sha256_file(per_sample_path),
    }
    write_json(args.output_dir / "validation_diagnostics.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
