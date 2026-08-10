#!/usr/bin/env python3
"""Combine five independently trained student test summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from mso_recovery.common import sha256_file, write_json


T_CRITICAL_95_DF4 = 2.7764451051977987


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, nargs=5, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    manifest_hashes = {payload["manifest_sha256"] for payload in payloads}
    seeds = [int(payload["checkpoint_seed"]) for payload in payloads]
    if len(manifest_hashes) != 1 or len(set(seeds)) != 5:
        raise ValueError("Five unique seeds must share one evaluation manifest")
    metric_names = sorted(payloads[0]["metrics"])
    rows = []
    combined: dict[str, dict[str, object]] = {}
    for metric in metric_names:
        values = [
            float(payload["metrics"][metric]["building_equal_weight_mean"])
            for payload in payloads
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite seed-level result for {metric}")
        mean = statistics.mean(values)
        standard_deviation = statistics.stdev(values)
        half_width = T_CRITICAL_95_DF4 * standard_deviation / math.sqrt(5)
        combined[metric] = {
            "seed_mean": mean,
            "seed_standard_deviation": standard_deviation,
            "seed_t_95ci_low": mean - half_width,
            "seed_t_95ci_high": mean + half_width,
            "per_seed": {str(seed): value for seed, value in zip(seeds, values)},
        }
        rows.append({"metric": metric, **combined[metric]})

    csv_path = args.output_dir / "multiseed_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "metric",
                "seed_mean",
                "seed_standard_deviation",
                "seed_t_95ci_low",
                "seed_t_95ci_high",
                "per_seed",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    output = {
        "status": "new_reconstructed_experiment_not_historical",
        "manifest_sha256": next(iter(manifest_hashes)),
        "student_seeds": seeds,
        "seed_count": 5,
        "seed_interval": "two-sided Student-t 95% interval across five run-level means",
        "input_summaries": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.summaries
        ],
        "metrics": combined,
        "csv_sha256": sha256_file(csv_path),
    }
    write_json(args.output_dir / "multiseed_summary.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
