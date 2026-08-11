#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


MODEL_SHAPE = (256, 256)
ECE_BINS = 15


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_metrics(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    target_values = target[mask]
    prediction_values = prediction[mask]
    tp = int(np.logical_and(target_values, prediction_values).sum())
    fp = int(np.logical_and(~target_values, prediction_values).sum())
    fn = int(np.logical_and(target_values, ~prediction_values).sum())
    tn = int(np.logical_and(~target_values, ~prediction_values).sum())
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2.0 * precision * recall, precision + recall),
        "iou": safe_divide(tp, tp + fp + fn),
    }


def expected_calibration_error(
    target: np.ndarray, probability: np.ndarray, mask: np.ndarray
) -> float:
    target_values = target[mask].astype(np.float64)
    probability_values = probability[mask].astype(np.float64)
    if not len(target_values):
        return 0.0
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    bins = np.minimum(np.digitize(probability_values, edges[1:-1]), ECE_BINS - 1)
    total = float(len(target_values))
    result = 0.0
    for bin_value in range(ECE_BINS):
        selected = bins == bin_value
        count = int(selected.sum())
        if not count:
            continue
        confidence = float(probability_values[selected].mean())
        frequency = float(target_values[selected].mean())
        result += count / total * abs(confidence - frequency)
    return result


def boundary_map(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    values = binary.astype(np.uint8)
    return cv2.morphologyEx(values, cv2.MORPH_GRADIENT, kernel).astype(bool)


def boundary_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    evaluation_mask: np.ndarray,
    tolerance_px: int = 2,
) -> dict[str, float | None]:
    target_boundary = boundary_map(target) & evaluation_mask
    prediction_boundary = boundary_map(prediction) & evaluation_mask
    target_count = int(target_boundary.sum())
    prediction_count = int(prediction_boundary.sum())
    if target_count == 0 and prediction_count == 0:
        return {"boundary_f1": 1.0, "boundary_chamfer_px": 0.0}
    if target_count == 0 or prediction_count == 0:
        return {"boundary_f1": 0.0, "boundary_chamfer_px": None}
    kernel_size = 2 * tolerance_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    target_near = cv2.dilate(target_boundary.astype(np.uint8), kernel).astype(bool)
    prediction_near = cv2.dilate(
        prediction_boundary.astype(np.uint8), kernel
    ).astype(bool)
    precision = safe_divide(
        int((prediction_boundary & target_near).sum()), prediction_count
    )
    recall = safe_divide(int((target_boundary & prediction_near).sum()), target_count)
    boundary_f1 = safe_divide(2.0 * precision * recall, precision + recall)
    distance_to_target = cv2.distanceTransform(
        (~target_boundary).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_prediction = cv2.distanceTransform(
        (~prediction_boundary).astype(np.uint8), cv2.DIST_L2, 3
    )
    chamfer = 0.5 * (
        float(distance_to_target[prediction_boundary].mean())
        + float(distance_to_prediction[target_boundary].mean())
    )
    return {"boundary_f1": boundary_f1, "boundary_chamfer_px": chamfer}


def free_topology(binary_occupied: np.ndarray) -> tuple[int, float]:
    free = (~binary_occupied).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        free, connectivity=4
    )
    foreground_components = component_count - 1
    total_free = int(free.sum())
    if not total_free or foreground_components == 0:
        return foreground_components, 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return foreground_components, largest / total_free


def load_observation_and_target(
    observation_path: Path, target_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(observation_path) as image:
        observation = np.asarray(
            image.convert("RGB").resize(MODEL_SHAPE[::-1], Image.Resampling.NEAREST)
        )
    with Image.open(target_path) as image:
        target = np.asarray(
            image.convert("L").resize(MODEL_SHAPE[::-1], Image.Resampling.NEAREST)
        )
    return observation, target >= 128


def evaluate_event(
    common: dict[str, str], prediction_row: dict[str, str], prediction_root: Path
) -> dict[str, object]:
    probability_path = (prediction_root / prediction_row["probability_path"]).resolve()
    if not probability_path.is_relative_to(prediction_root.resolve()):
        raise ValueError("probability path escapes its run directory")
    expected_hash = prediction_row["probability_sha256"].lower()
    actual_hash = sha256(probability_path)
    if actual_hash != expected_hash:
        raise ValueError(f"probability hash mismatch for {probability_path.name}")
    probability = np.load(probability_path, allow_pickle=False)
    if probability.shape != MODEL_SHAPE or probability.dtype != np.float32:
        raise ValueError(
            f"invalid probability contract for {probability_path}: "
            f"shape={probability.shape}, dtype={probability.dtype}"
        )
    if not np.isfinite(probability).all():
        raise ValueError(f"non-finite probability in {probability_path}")
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise ValueError(f"out-of-range probability in {probability_path}")
    observation, target = load_observation_and_target(
        Path(common["observation_path"]), Path(common["target_path"])
    )
    measured_occupied = np.all(
        observation == np.asarray([255, 0, 0], dtype=np.uint8), axis=2
    )
    unknown = np.all(
        observation == np.asarray([0, 255, 0], dtype=np.uint8), axis=2
    )
    measured_free = np.all(
        observation == np.asarray([0, 0, 255], dtype=np.uint8), axis=2
    )
    if not np.all(measured_occupied | unknown | measured_free):
        raise ValueError(f"invalid observation categories for {common['sample_id']}")
    if int(unknown.sum()) == 0:
        raise ValueError(f"no unknown evaluation cells for {common['sample_id']}")
    copied_probability = probability.copy()
    copied_probability[measured_occupied] = 1.0
    copied_probability[measured_free] = 0.0
    predicted = copied_probability >= 0.5
    confusion = confusion_metrics(target, predicted, unknown)
    brier = float(np.mean((probability[unknown] - target[unknown]) ** 2))
    ece = expected_calibration_error(target, probability, unknown)
    boundaries = boundary_metrics(target, predicted, unknown)
    target_components, target_largest = free_topology(target)
    predicted_components, predicted_largest = free_topology(predicted)
    return {
        "method": prediction_row["method"],
        "seed": int(prediction_row["seed"]),
        "split": prediction_row["split"],
        "sample_id": prediction_row["sample_id"],
        "unknown_cells": int(unknown.sum()),
        "unknown_target_occupied_fraction": float(target[unknown].mean()),
        "unknown_occupied_precision": confusion["precision"],
        "unknown_occupied_recall": confusion["recall"],
        "unknown_occupied_f1": confusion["f1"],
        "unknown_occupied_iou": confusion["iou"],
        "unknown_occupied_brier": brier,
        "unknown_occupied_ece": ece,
        "boundary_f1_tolerance_2px": boundaries["boundary_f1"],
        "boundary_chamfer_px": boundaries["boundary_chamfer_px"],
        "target_free_components": target_components,
        "predicted_free_components": predicted_components,
        "free_component_count_error": abs(predicted_components - target_components),
        "target_largest_free_component_fraction": target_largest,
        "predicted_largest_free_component_fraction": predicted_largest,
        "largest_free_component_fraction_error": abs(
            predicted_largest - target_largest
        ),
        "tp": confusion["tp"],
        "fp": confusion["fp"],
        "fn": confusion["fn"],
        "tn": confusion["tn"],
        "inference_ms": float(prediction_row["inference_ms"]),
        "probability_sha256": actual_hash,
    }


def finite_mean(values: Iterable[object]) -> float | None:
    parsed = [float(value) for value in values if value is not None]
    parsed = [value for value in parsed if math.isfinite(value)]
    return float(np.mean(parsed)) if parsed else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-manifest", required=True, type=Path)
    parser.add_argument("--prediction-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    common_rows = read_tsv(args.common_manifest)
    common_by_id = {
        row["sample_id"]: row for row in common_rows if row["split"] == args.split
    }
    prediction_rows = [
        row
        for row in read_tsv(args.prediction_manifest)
        if row["split"] == args.split
    ]
    required_columns = {
        "method",
        "seed",
        "split",
        "sample_id",
        "probability_path",
        "probability_sha256",
        "inference_ms",
    }
    if prediction_rows and not required_columns.issubset(prediction_rows[0]):
        raise ValueError("prediction manifest does not follow the output contract")
    prediction_ids = [row["sample_id"] for row in prediction_rows]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("duplicate prediction sample IDs")
    missing = sorted(set(common_by_id) - set(prediction_ids))
    extra = sorted(set(prediction_ids) - set(common_by_id))
    if missing or extra:
        raise ValueError(
            f"prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    events = [
        evaluate_event(common_by_id[row["sample_id"]], row, args.prediction_manifest.parent)
        for row in sorted(prediction_rows, key=lambda item: item["sample_id"])
    ]
    event_path = args.output / "event_metrics.tsv"
    with event_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(events[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(events)
    metric_fields = [
        "unknown_occupied_precision",
        "unknown_occupied_recall",
        "unknown_occupied_f1",
        "unknown_occupied_iou",
        "unknown_occupied_brier",
        "unknown_occupied_ece",
        "boundary_f1_tolerance_2px",
        "boundary_chamfer_px",
        "free_component_count_error",
        "largest_free_component_fraction_error",
        "inference_ms",
    ]
    summary = {
        "status": "prospective_uniform_evaluation",
        "method": events[0]["method"],
        "seed": events[0]["seed"],
        "split": args.split,
        "n_samples": len(events),
        "macro_means": {
            field: finite_mean(event[field] for event in events)
            for field in metric_fields
        },
        "common_manifest_sha256": sha256(args.common_manifest),
        "prediction_manifest_sha256": sha256(args.prediction_manifest),
        "event_metrics_sha256": sha256(event_path),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "SHA256SUMS").write_text(
        f"{sha256(event_path)}  {event_path.name}\n"
        f"{sha256(summary_path)}  {summary_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
