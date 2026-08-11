#!/usr/bin/env python3
"""Read-only partial metric summary for the running official simulator smoke."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--house-manifest", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    args = parser.parse_args()
    with args.house_manifest.open("r", encoding="utf-8", newline="") as stream:
        splits = {row["house_id"]: row["split"] for row in csv.DictReader(stream)}
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"obs0": [], "obs1": [], "obs2": [], "target": []}
    )
    for sample_dir in sorted(args.samples_root.iterdir()):
        if not sample_dir.is_dir() or "__" not in sample_dir.name:
            continue
        house_id = sample_dir.name.split("__", 1)[0]
        split = splits[house_id]
        observation = np.asarray(Image.open(sample_dir / "raw_obs_480.png").convert("RGB"))
        target = np.asarray(Image.open(sample_dir / "raw_target_480.png").convert("L"))
        for channel in range(3):
            values[split][f"obs{channel}"].append(float(np.mean(observation[:, :, channel] == 255)))
        values[split]["target"].append(float(np.mean(target == 255)))
    output: dict[str, object] = {"status": "read_only_partial_running_smoke", "test_accessed": False}
    for split, split_values in values.items():
        summary: dict[str, object] = {
            "samples": len(split_values["target"]),
        }
        for key, items in split_values.items():
            summary[key] = {
                "mean": float(np.mean(items)),
                "median": float(np.median(items)),
                "p10": float(np.quantile(items, 0.1)),
                "p90": float(np.quantile(items, 0.9)),
            }
        output[split] = summary
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
