#!/usr/bin/env python3
"""Audit the map-size filter visible in the preserved HouseExpo generator.

This is source-code forensics for a new experiment design.  It does not claim
that the preserved archive used this exact source list or split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixels-per-meter", type=int, default=16)
    parser.add_argument("--world-padding", type=int, default=10)
    parser.add_argument("--minimum-shape-sum", type=int, default=1500)
    return parser.parse_args()


def exact_shape(payload: dict, pixels_per_meter: int, padding: int) -> tuple[int, int]:
    vertices = (np.asarray(payload["verts"], dtype=np.float64) * pixels_per_meter).astype(
        np.int32
    )
    width = int(vertices[:, 0].max() - vertices[:, 0].min() + 2 + 2 * padding)
    height = int(vertices[:, 1].max() - vertices[:, 1].min() + 2 + 2 * padding)
    return height, width


def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts: Counter[str] = Counter()
    sums: list[int] = []
    eligible_ids: list[str] = []
    for index, row in enumerate(rows):
        payload = json.loads(
            (args.json_root / row["json_relpath"]).read_text(encoding="utf-8")
        )
        height, width = exact_shape(
            payload, args.pixels_per_meter, args.world_padding
        )
        shape_sum = height + width
        sums.append(shape_sum)
        if shape_sum >= args.minimum_shape_sum:
            counts[row["split"]] += 1
            eligible_ids.append(row["house_id"])
        if (index + 1) % 5000 == 0:
            print(json.dumps({"audited": index + 1, "total": len(rows)}), flush=True)
    array = np.asarray(sums, dtype=np.float64)
    result = {
        "status": "visible_filter_audit_for_new_experiment_not_historical_recovery",
        "houses": len(rows),
        "pixels_per_meter": args.pixels_per_meter,
        "world_padding_each_side_px": args.world_padding,
        "minimum_world_height_plus_width_px": args.minimum_shape_sum,
        "eligible_houses": len(eligible_ids),
        "eligible_by_split": dict(sorted(counts.items())),
        "eligible_fraction": len(eligible_ids) / len(rows),
        "shape_sum_px": {
            "minimum": float(array.min()),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.9)),
            "p99": float(np.quantile(array, 0.99)),
            "p999": float(np.quantile(array, 0.999)),
            "maximum": float(array.max()),
        },
        "test_outcomes_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
