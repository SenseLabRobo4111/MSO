#!/usr/bin/env python3
"""Summarize generated coverage and occupancy distributions by split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mso_recovery.common import sha256_file, write_json
from mso_recovery.data import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite output: {args.output}")
    rows = read_manifest(args.manifest)
    floorplan_sample_counts = Counter(row["floorplan_id"] for row in rows)
    split_summary = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        split_summary[split] = {
            "samples": len(selected),
            "buildings": len({row["group_key"] for row in selected}),
            "floorplans": len({row["floorplan_id"] for row in selected}),
            "unknown_fraction": describe(
                [float(row["unknown_fraction"]) for row in selected]
            ),
            "occupied_fraction": describe(
                [float(row["occupied_fraction"]) for row in selected]
            ),
            "scan_count": describe([float(row["scan_count"]) for row in selected]),
        }
    report = {
        "manifest_sha256": sha256_file(args.manifest),
        "samples": len(rows),
        "floorplan_sample_count_minimum": min(floorplan_sample_counts.values()),
        "floorplan_sample_count_maximum": max(floorplan_sample_counts.values()),
        "floorplan_sample_count_unique_values": sorted(set(floorplan_sample_counts.values())),
        "split_summary": split_summary,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
