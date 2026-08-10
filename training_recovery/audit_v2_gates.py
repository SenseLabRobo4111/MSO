#!/usr/bin/env python3
"""Atomically evaluate all five prospective-v2 go/no-go gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mso_recovery.v2_lock import atomic_write_json, load_lock, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--positive-control", type=Path, required=True)
    parser.add_argument("--teacher-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        raw = list(csv.DictReader(stream))
    if not raw:
        raise ValueError("Missing teacher smoke metrics")
    rows = []
    for row in raw:
        parsed = {
            "epoch": float(row["epoch"]),
            "validation_unknown_bce": float(row["validation_unknown_bce"]),
            "validation_unknown_f1": float(row["validation_unknown_f1"]),
        }
        if not all(math.isfinite(value) for value in parsed.values()):
            raise ValueError("Non-finite teacher smoke metric")
        rows.append(parsed)
    return rows


def evaluate_gates(
    lock: dict[str, Any],
    lock_sha: str,
    validation: dict[str, Any],
    positive: dict[str, Any],
    teacher_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    for name, artifact in (("validation", validation), ("positive", positive)):
        if artifact.get("v2_lock_sha256") != lock_sha:
            raise ValueError(f"{name} artifact is not bound to the v2 lock")
        if artifact.get("test_outcomes_accessed") is not False:
            raise ValueError(f"{name} artifact lacks an explicit no-test record")
        if artifact.get("test_images_decoded") != 0:
            raise ValueError(f"{name} artifact decoded test images")
    if positive.get("dataset_manifest_sha256") != validation.get(
        "dataset_manifest_sha256"
    ):
        raise ValueError("Positive control and validation use different datasets")
    thresholds = lock["go_no_go"]
    integrity_values = {
        "one_hot_fraction": validation["one_hot_fraction"],
        "known_target_agreement": validation["known_target_agreement"],
        "cross_split_group_leakage": validation[
            "cross_split_operational_group_leakage"
        ],
        "cross_split_source_hash_leakage": validation[
            "cross_split_source_hash_leakage"
        ],
        "cross_split_processed_pair_leakage": validation[
            "cross_split_processed_pair_leakage"
        ],
    }
    integrity_pass = all(
        integrity_values[key] == expected
        for key, expected in thresholds["integrity"].items()
    )
    coverage_values = validation["floorplan_full_coverage_fraction"]
    coverage_pass = (
        coverage_values["train"]
        >= thresholds["coverage"]["train_floorplan_fraction"]
        and coverage_values["val"]
        >= thresholds["coverage"]["val_floorplan_fraction"]
    )
    marginal_thresholds = thresholds["marginals"]
    observation_deltas = {
        split: [
            abs(value - reference)
            for value, reference in zip(
                validation["pooled_observation_channel_fraction"][split],
                marginal_thresholds["reference_observation_channels"],
                strict=True,
            )
        ]
        for split in ("train", "val")
    }
    target_deltas = {
        split: abs(
            validation["pooled_target_fraction"][split]
            - marginal_thresholds["reference_target"]
        )
        for split in ("train", "val")
    }
    train_val_target_delta = abs(
        validation["pooled_target_fraction"]["train"]
        - validation["pooled_target_fraction"]["val"]
    )
    marginal_pass = (
        max(value for values in observation_deltas.values() for value in values)
        <= marginal_thresholds["maximum_absolute_channel_delta"]
        and max(target_deltas.values())
        <= marginal_thresholds["maximum_absolute_target_delta"]
        and train_val_target_delta
        <= marginal_thresholds["maximum_train_val_target_delta"]
    )
    positive_thresholds = thresholds["positive_control"]
    positive_pass = (
        positive["candidate_sha256"] == lock["positive_control"]["sha256"]
        and positive["validation_unknown_f1"]
        >= positive_thresholds["minimum_unknown_f1"]
        and positive["validation_unknown_iou"]
        >= positive_thresholds["minimum_unknown_iou"]
    )
    first_epoch = min(teacher_metrics, key=lambda row: row["epoch"])
    best_bce = min(row["validation_unknown_bce"] for row in teacher_metrics)
    best_f1 = max(row["validation_unknown_f1"] for row in teacher_metrics)
    if first_epoch["validation_unknown_bce"] <= 0:
        raise ValueError("Epoch-1 validation BCE must be positive")
    relative_reduction = (
        first_epoch["validation_unknown_bce"] - best_bce
    ) / first_epoch["validation_unknown_bce"]
    learnability_thresholds = thresholds["learnability"]
    observed_epochs = sorted(int(row["epoch"]) for row in teacher_metrics)
    expected_epochs = list(
        range(1, int(learnability_thresholds["maximum_epochs"]) + 1)
    )
    learnability_pass = (
        observed_epochs == expected_epochs
        and relative_reduction
        >= learnability_thresholds[
            "minimum_validation_unknown_bce_relative_reduction_from_epoch_1"
        ]
        and best_f1 >= learnability_thresholds["minimum_validation_unknown_f1"]
    )
    gates = {
        "integrity": {"pass": integrity_pass, "observed": integrity_values},
        "coverage": {"pass": coverage_pass, "observed": coverage_values},
        "marginals": {
            "pass": marginal_pass,
            "observation_absolute_deltas": observation_deltas,
            "target_absolute_deltas": target_deltas,
            "train_val_target_delta": train_val_target_delta,
        },
        "positive_control": {
            "pass": positive_pass,
            "candidate_sha256": positive["candidate_sha256"],
            "validation_unknown_f1": positive["validation_unknown_f1"],
            "validation_unknown_iou": positive["validation_unknown_iou"],
        },
        "learnability": {
            "pass": learnability_pass,
            "epoch_1_validation_unknown_bce": first_epoch[
                "validation_unknown_bce"
            ],
            "best_validation_unknown_bce": best_bce,
            "relative_reduction": relative_reduction,
            "best_validation_unknown_f1": best_f1,
            "observed_epochs": observed_epochs,
        },
    }
    all_pass = all(gate["pass"] for gate in gates.values())
    return {
        "schema": "mso.prospective_v2.atomic_gate_audit/1",
        "status": "GO" if all_pass else "NO_GO",
        "v2_lock_sha256": lock_sha,
        "all_five_gates_pass": all_pass,
        "gates": gates,
        "test_decode_allowed": False,
        "test_outcomes_accessed": False,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("Atomic gate decision already exists and is immutable")
    lock, lock_sha = load_lock(args.lock_root)
    validation = json.loads(args.dataset_validation.read_text(encoding="utf-8"))
    positive = json.loads(args.positive_control.read_text(encoding="utf-8"))
    metrics = load_metrics(args.teacher_metrics)
    run_dir = args.teacher_metrics.resolve().parent
    run_manifest_path = run_dir / "run_manifest.json"
    completion_path = run_dir / "completion.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    teacher_seed = int(lock["training"]["teacher_seed"])
    expected_job = f"smoke_teacher_{teacher_seed}"
    manifest_sha = validation.get("dataset_manifest_sha256")
    required_run = {
        "job_id": expected_job,
        "v2_lock_sha256": lock_sha,
        "dataset_manifest_sha256": manifest_sha,
        "source_split_sha256": lock["artifacts"]["source_split"]["sha256"],
        "training_configuration_sha256": lock["configuration_sha256"],
        "parent_teacher_checkpoint_sha256": None,
        "test_outcomes_accessed": False,
    }
    for key, expected in required_run.items():
        if run_manifest.get(key) != expected:
            raise ValueError(f"Smoke-teacher run binding failed: {key}")
    expected_job_record = {
        "stage": "teacher",
        "seed": teacher_seed,
        "dataset_kind": "smoke",
        "maximum_epochs": int(lock["training"]["smoke_teacher"]["maximum_epochs"]),
    }
    if run_manifest.get("job") != expected_job_record:
        raise ValueError("Smoke-teacher job record differs from the lock")
    required_completion = {
        "job_id": expected_job,
        "stage": "teacher",
        "seed": teacher_seed,
        "last_epoch": int(lock["training"]["smoke_teacher"]["maximum_epochs"]),
        "v2_lock_sha256": lock_sha,
        "dataset_manifest_sha256": manifest_sha,
        "source_split_sha256": lock["artifacts"]["source_split"]["sha256"],
        "training_configuration_sha256": lock["configuration_sha256"],
        "parent_teacher_checkpoint_sha256": None,
        "test_outcomes_accessed": False,
    }
    for key, expected in required_completion.items():
        if completion.get(key) != expected:
            raise ValueError(f"Smoke-teacher completion binding failed: {key}")
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    if sha256_file(best_path) != completion.get("best_checkpoint_sha256"):
        raise ValueError("Smoke-teacher best checkpoint hash mismatch")
    if sha256_file(last_path) != completion.get("last_checkpoint_sha256"):
        raise ValueError("Smoke-teacher last checkpoint hash mismatch")
    if sha256_file(args.teacher_metrics) != completion.get("metrics_sha256"):
        raise ValueError("Smoke-teacher metrics hash mismatch")
    result = evaluate_gates(lock, lock_sha, validation, positive, metrics)
    result["input_sha256"] = {
        "dataset_validation": sha256_file(args.dataset_validation),
        "positive_control": sha256_file(args.positive_control),
        "teacher_metrics": sha256_file(args.teacher_metrics),
        "teacher_run_manifest": sha256_file(run_manifest_path),
        "teacher_completion": sha256_file(completion_path),
        "teacher_best_checkpoint": sha256_file(best_path),
        "teacher_last_checkpoint": sha256_file(last_path),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
