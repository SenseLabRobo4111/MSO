#!/usr/bin/env python3
"""Aggregate the teacher and five reconstructed-student runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (11, 23, 37, 53, 71)
T_CRITICAL_95_DF4 = 2.7764451051977987


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def verify_sha_file(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise ValueError(f"missing SHA256SUMS: {root}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"hash mismatch: {path}")


def mean_sd_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if len(array) != len(SEEDS) or not np.isfinite(array).all():
        raise ValueError("five finite seed estimates are required")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = T_CRITICAL_95_DF4 * sd / math.sqrt(len(array))
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--uniform-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing non-empty aggregate output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    teacher = args.campaign / "runs" / "teacher_seed_101"
    verify_sha_file(teacher)
    teacher_completion = json.loads(
        (teacher / "completion.json").read_text(encoding="utf-8")
    )
    if not teacher_completion.get("completed") or teacher_completion.get("stage") != "teacher":
        raise ValueError("teacher run is not complete")

    curve_rows: list[dict[str, Any]] = []
    for row in read_tsv(teacher / "metrics_by_epoch.tsv"):
        curve_rows.append({"stage": "teacher", "seed": 101, **row})

    per_seed: list[dict[str, Any]] = []
    metric_names: list[str] | None = None
    for seed in SEEDS:
        run = args.campaign / "runs" / f"student_seed_{seed}"
        verify_sha_file(run)
        completion = json.loads((run / "completion.json").read_text(encoding="utf-8"))
        if not completion.get("completed") or completion.get("stage") != "student":
            raise ValueError(f"student seed {seed} is not complete")
        validation_path = (
            args.campaign / "evaluation" / f"student_seed_{seed}"
            / "validation" / "summary.json"
        )
        test_path = (
            args.campaign / "evaluation" / f"student_seed_{seed}"
            / "test" / "summary.json"
        )
        verify_sha_file(validation_path.parent)
        verify_sha_file(test_path.parent)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        test = json.loads(test_path.read_text(encoding="utf-8"))
        resource = json.loads((run / "resource_profile.json").read_text(encoding="utf-8"))
        if validation["seed"] != seed or test["seed"] != seed:
            raise ValueError("evaluation seed differs from its run")
        row: dict[str, Any] = {
            "method": "MSO-Paper-Equation-Reconstructed",
            "seed": seed,
            "best_epoch": completion["best_epoch"],
            "best_validation_unknown_bce": completion[
                "best_validation_unknown_bce"
            ],
            "training_time_s": resource["training_time_s"],
            "latency_batch1_median_ms": resource["latency_batch1"]["median_ms"],
            "latency_batch1_p10_ms": resource["latency_batch1"]["p10_ms"],
            "latency_batch1_p90_ms": resource["latency_batch1"]["p90_ms"],
        }
        for split, payload in (("validation", validation), ("test", test)):
            for name, value in payload["macro_means"].items():
                row[f"{split}_{name}"] = value
        metric_names = [
            name for name in row if name not in {"method", "seed", "best_epoch"}
        ]
        per_seed.append(row)
        for curve in read_tsv(run / "metrics_by_epoch.tsv"):
            curve_rows.append({"stage": "student", "seed": seed, **curve})

    if metric_names is None:
        raise ValueError("no student metrics were found")
    write_tsv(args.output / "per_seed_results.tsv", per_seed)
    write_tsv(args.output / "training_curves.tsv", curve_rows)

    summary_rows = []
    for metric in metric_names:
        values = [float(row[metric]) for row in per_seed]
        summary_rows.append({"metric": metric, **mean_sd_ci(values)})
    write_tsv(args.output / "five_seed_summary.tsv", summary_rows)

    verify_sha_file(args.uniform_aggregate)
    uniform_path = args.uniform_aggregate / "per_seed_results.tsv"
    uniform_rows = [
        row for row in read_tsv(uniform_path) if row["method"] == "MSO"
    ]
    if {int(row["seed"]) for row in uniform_rows} != set(SEEDS):
        raise ValueError("uniform-objective MSO does not contain the matched five seeds")
    uniform = {int(row["seed"]): row for row in uniform_rows}
    paired_metrics = (
        "best_epoch",
        "best_validation_unknown_bce",
        "test_unknown_occupied_f1",
        "test_unknown_occupied_iou",
        "test_unknown_occupied_brier",
        "test_unknown_occupied_ece",
        "latency_batch1_median_ms",
    )
    paired_rows = []
    for row in per_seed:
        seed = int(row["seed"])
        for metric in paired_metrics:
            full_value = float(row[metric])
            uniform_value = float(uniform[seed][metric])
            paired_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "uniform_objective": uniform_value,
                    "paper_equation_reconstruction": full_value,
                    "paper_minus_uniform": full_value - uniform_value,
                    "comparison_scope": (
                        "descriptive_protocol_comparison_different_training_budgets"
                    ),
                }
            )
    write_tsv(args.output / "paired_protocol_comparison.tsv", paired_rows)

    outputs = (
        "per_seed_results.tsv",
        "training_curves.tsv",
        "five_seed_summary.tsv",
        "paired_protocol_comparison.tsv",
    )
    manifest = {
        "status": "prospective_paper_equation_reconstruction_aggregate",
        "teacher_seed": 101,
        "student_seeds": list(SEEDS),
        "n_student_seeds": len(SEEDS),
        "center": "arithmetic mean over seeds",
        "spread": "sample standard deviation over seeds",
        "interval": "two-sided 95% t interval with 4 degrees of freedom",
        "best_seed_selection_used": False,
        "uniform_comparison_is_matched_by_seed": True,
        "uniform_comparison_has_identical_training_budget": False,
        "uniform_aggregate_sha256_manifest": sha256(
            args.uniform_aggregate / "SHA256SUMS"
        ),
        "aggregate_script_sha256": sha256(Path(__file__)),
        "files": {name: sha256(args.output / name) for name in outputs},
    }
    write_json = args.output / "aggregate_manifest.json"
    write_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = (*outputs, "aggregate_manifest.json")
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(args.output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
