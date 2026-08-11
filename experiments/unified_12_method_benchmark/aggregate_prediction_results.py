"""Aggregate five-seed prediction results without dropping failed methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METHODS = (
    "MSO",
    "U-Net",
    "LaMa-Fourier",
    "MI-GAN",
    "PartialConv",
    "GatedConv",
    "EdgeConnect",
    "AOT-GAN",
    "MAT",
    "ZITS++",
    "RePaint",
    "HINT",
)
SEEDS = (11, 23, 37, 53, 71)
T_CRITICAL_95_DF4 = 2.7764451051977987


def safe_name(method: str) -> str:
    return method.replace("+", "_plus_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten(prefix: str, value: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested numeric records into stable TSV columns.

    Resource profiles contain a nested ``latency_batch1`` record.  Keeping the
    mapping as a single cell would make the five-seed numeric aggregation fail
    after the expensive training queue has completed, so nested keys are
    expanded recursively (for example, ``latency_batch1_median_ms``).
    """

    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            flattened.update(flatten(f"{name}_", item))
        else:
            flattened[name] = item
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite aggregate output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for method in METHODS:
        completed = 0
        for seed in SEEDS:
            run = args.root / "runs" / safe_name(method) / f"seed_{seed}"
            validation = (
                args.root / "evaluation" / safe_name(method) / f"seed_{seed}"
                / "validation" / "summary.json"
            )
            test = (
                args.root / "evaluation" / safe_name(method) / f"seed_{seed}"
                / "test" / "summary.json"
            )
            resource = run / "resource_profile.json"
            completion = run / "completion.json"
            if not all(path.is_file() for path in (validation, test, resource, completion)):
                continue
            validation_value = json.loads(validation.read_text(encoding="utf-8"))
            test_value = json.loads(test.read_text(encoding="utf-8"))
            resource_value = json.loads(resource.read_text(encoding="utf-8"))
            completion_value = json.loads(completion.read_text(encoding="utf-8"))
            seed_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    **flatten("validation_", validation_value["macro_means"]),
                    **flatten("test_", test_value["macro_means"]),
                    **flatten("", resource_value),
                    "best_epoch": completion_value["best_epoch"],
                    "best_validation_unknown_bce": completion_value[
                        "best_validation_unknown_bce"
                    ],
                }
            )
            completed += 1
        status_rows.append(
            {
                "method": method,
                "registered_seeds": len(SEEDS),
                "completed_seeds": completed,
                "status": "complete" if completed == len(SEEDS) else "incomplete",
                "exclusion_from_complete_case_contrasts": int(completed != len(SEEDS)),
            }
        )

    seed_path = args.output / "per_seed_results.tsv"
    if seed_rows:
        with seed_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(seed_rows[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(seed_rows)
    else:
        raise ValueError("no completed seed result was found")

    status_path = args.output / "method_status.tsv"
    with status_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(status_rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(status_rows)

    numeric_fields = [
        field for field in seed_rows[0]
        if field not in {"method", "seed"}
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in seed_rows if row["method"] == method]
        output: dict[str, Any] = {"method": method, "n_seeds": len(selected)}
        for field in numeric_fields:
            values = np.asarray(
                [float(row[field]) for row in selected if row[field] is not None],
                dtype=float,
            )
            if len(values) != len(SEEDS) or not np.isfinite(values).all():
                output[f"{field}_mean"] = ""
                output[f"{field}_sd"] = ""
                output[f"{field}_ci95_low"] = ""
                output[f"{field}_ci95_high"] = ""
                continue
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            half = T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
            output[f"{field}_mean"] = mean
            output[f"{field}_sd"] = sd
            output[f"{field}_ci95_low"] = mean - half
            output[f"{field}_ci95_high"] = mean + half
        aggregate_rows.append(output)

    aggregate_path = args.output / "method_summary.tsv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(aggregate_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    manifest = {
        "status": "prospective_five_seed_aggregate",
        "methods_registered": len(METHODS),
        "methods_complete": sum(row["status"] == "complete" for row in status_rows),
        "seed_level_uncertainty": "two-sided 95% t interval over five seed estimates",
        "semantic_building_identity_available": False,
        "best_seed_selection_used": False,
        "per_seed_results_sha256": sha256(seed_path),
        "method_status_sha256": sha256(status_path),
        "method_summary_sha256": sha256(aggregate_path),
    }
    manifest_path = args.output / "aggregate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "SHA256SUMS").write_text(
        f"{sha256(seed_path)}  {seed_path.name}\n"
        f"{sha256(status_path)}  {status_path.name}\n"
        f"{sha256(aggregate_path)}  {aggregate_path.name}\n"
        f"{sha256(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
