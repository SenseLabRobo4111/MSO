#!/usr/bin/env python3
"""Summarize all five locked student seeds without exclusion or best-seed claims."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mso_recovery.v2_lock import atomic_write_json, load_lock, sha256_file  # noqa: E402


METRICS = (
    "validation_unknown_bce",
    "validation_unknown_precision",
    "validation_unknown_recall",
    "validation_unknown_f1",
    "validation_unknown_iou",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_row(metrics_path: Path, epoch: int) -> dict[str, float]:
    with metrics_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if int(row["epoch"]) == epoch]
    if len(matches) != 1:
        raise ValueError(f"Expected one selected metric row for epoch {epoch}")
    if int(matches[0]["selected_best"]) != 1:
        raise ValueError("Selected checkpoint epoch is not marked as a new best")
    values = {name: float(matches[0][name]) for name in METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite selected seed metric")
    return values


def summarize(values: list[float], t_critical: float) -> dict[str, float]:
    if len(values) != 5:
        raise ValueError("Locked v2 statistics require exactly five seeds")
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    half_width = t_critical * std / math.sqrt(len(values))
    return {
        "mean": mean,
        "sample_standard_deviation_ddof_1": std,
        "two_sided_t_95_percent_CI_lower": mean - half_width,
        "two_sided_t_95_percent_CI_upper": mean + half_width,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("Locked multiseed summary already exists")
    lock, lock_sha = load_lock(args.lock_root)
    seeds = [int(seed) for seed in lock["training"]["student_seeds"]]
    if seeds != [11, 23, 37, 53, 71]:
        raise ValueError("Student seed set differs from the lock")
    full_manifest = args.campaign_root / "datasets" / "full" / "manifest.csv"
    manifest_sha = sha256_file(full_manifest)
    teacher_dir = (
        args.campaign_root
        / "runs"
        / f"teacher_{int(lock['training']['teacher_seed'])}"
    )
    teacher_completion = json.loads(
        (teacher_dir / "completion.json").read_text(encoding="utf-8")
    )
    teacher_sha = sha256_file(teacher_dir / "best.pt")
    if teacher_completion.get("best_checkpoint_sha256") != teacher_sha:
        raise ValueError("Teacher selection hash differs from completion")
    teacher_required = {
        "v2_lock_sha256": lock_sha,
        "dataset_manifest_sha256": manifest_sha,
        "source_split_sha256": lock["artifacts"]["source_split"]["sha256"],
        "training_configuration_sha256": lock["configuration_sha256"],
        "stage": "teacher",
        "seed": int(lock["training"]["teacher_seed"]),
        "parent_teacher_checkpoint_sha256": None,
        "test_outcomes_accessed": False,
    }
    for key, expected in teacher_required.items():
        if teacher_completion.get(key) != expected:
            raise ValueError(f"Teacher completion binding failed: {key}")
    teacher_checkpoint = torch.load(
        teacher_dir / "best.pt", map_location="cpu", weights_only=False
    )
    teacher_checkpoint_required = {
        key: expected
        for key, expected in teacher_required.items()
        if key != "test_outcomes_accessed"
    }
    teacher_checkpoint_required["epoch"] = int(teacher_completion["best_epoch"])
    for key, expected in teacher_checkpoint_required.items():
        if teacher_checkpoint.get(key) != expected:
            raise ValueError(f"Teacher checkpoint binding failed: {key}")
    if teacher_checkpoint.get("selected_validation_metric") != float(
        teacher_completion["best_validation_unknown_bce"]
    ):
        raise ValueError("Teacher checkpoint selection metric mismatch")
    seed_rows = []
    for seed in seeds:
        run_dir = args.campaign_root / "runs" / f"student_{seed}"
        completion = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
        if completion.get("v2_lock_sha256") != lock_sha:
            raise ValueError(f"Seed {seed} completion is not lock-bound")
        if completion.get("dataset_manifest_sha256") != manifest_sha:
            raise ValueError(f"Seed {seed} completion has another dataset")
        if completion.get("seed") != seed or completion.get("stage") != "student":
            raise ValueError(f"Seed {seed} completion identity mismatch")
        if completion.get("source_split_sha256") != lock["artifacts"][
            "source_split"
        ]["sha256"]:
            raise ValueError(f"Seed {seed} source-split binding mismatch")
        if completion.get("training_configuration_sha256") != lock[
            "configuration_sha256"
        ]:
            raise ValueError(f"Seed {seed} configuration binding mismatch")
        if completion.get("parent_teacher_checkpoint_sha256") != teacher_sha:
            raise ValueError(f"Seed {seed} uses a different teacher")
        if completion.get("test_outcomes_accessed") is not False:
            raise ValueError(f"Seed {seed} lacks the no-test binding")
        best_path = run_dir / "best.pt"
        if sha256_file(best_path) != completion["best_checkpoint_sha256"]:
            raise ValueError(f"Seed {seed} best checkpoint hash mismatch")
        metrics_path = run_dir / "metrics.csv"
        if sha256_file(metrics_path) != completion.get("metrics_sha256"):
            raise ValueError(f"Seed {seed} metrics hash mismatch")
        values = selected_row(metrics_path, int(completion["best_epoch"]))
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        checkpoint_required = {
            "v2_lock_sha256": lock_sha,
            "dataset_manifest_sha256": manifest_sha,
            "source_split_sha256": lock["artifacts"]["source_split"]["sha256"],
            "training_configuration_sha256": lock["configuration_sha256"],
            "stage": "student",
            "seed": seed,
            "epoch": int(completion["best_epoch"]),
            "parent_teacher_checkpoint_sha256": teacher_sha,
        }
        for key, expected in checkpoint_required.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"Seed {seed} checkpoint binding failed: {key}")
        if checkpoint.get("selected_validation_metric") != float(
            completion["best_validation_unknown_bce"]
        ):
            raise ValueError(f"Seed {seed} checkpoint selection metric mismatch")
        if values["validation_unknown_bce"] != float(
            completion["best_validation_unknown_bce"]
        ):
            raise ValueError(f"Seed {seed} selected metric row mismatch")
        seed_rows.append(
            {
                "seed": seed,
                "best_epoch": completion["best_epoch"],
                "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
                "parent_teacher_checkpoint_sha256": completion[
                    "parent_teacher_checkpoint_sha256"
                ],
                **values,
            }
        )
    t_critical = float(lock["statistics"]["t_critical_df_4"])
    result = {
        "schema": "mso.prospective_v2.multiseed_summary/1",
        "status": "prospective_v2_validation_statistics_not_historical_recovery",
        "v2_lock_sha256": lock_sha,
        "dataset_manifest_sha256": manifest_sha,
        "primary_experimental_unit": "student_seed",
        "seeds_included": seeds,
        "seeds_excluded": [],
        "best_seed_selected": False,
        "per_seed": seed_rows,
        "statistics": {
            metric: summarize([float(row[metric]) for row in seed_rows], t_critical)
            for metric in METRICS
        },
        "spatial_cluster_unit": "operational_prefix_group_not_semantically_verified_building",
        "test_images_decoded": 0,
        "test_outcomes_accessed": False,
    }
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
