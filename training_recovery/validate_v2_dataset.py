#!/usr/bin/env python3
"""Fail-closed validator for locked v2 train/validation datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mso_recovery.v2_data import read_v2_manifest, safe_child  # noqa: E402
from mso_recovery.v2_lock import (  # noqa: E402
    atomic_write_json,
    load_lock,
    read_csv_artifact,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-kind", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock, lock_sha = load_lock(args.lock_root)
    dataset_root = args.dataset_root.resolve()
    summary = json.loads((dataset_root / "dataset_summary.json").read_text(encoding="utf-8"))
    if summary.get("v2_lock_sha256") != lock_sha:
        raise ValueError("Dataset was not generated under this v2 lock")
    if summary.get("dataset_kind") != args.dataset_kind:
        raise ValueError("Dataset kind mismatch")
    if summary.get("test_source_rows_decoded") != 0 or summary.get("test_samples_generated") != 0:
        raise ValueError("Test decode/generation prohibition was violated")
    manifest_path = dataset_root / "manifest.csv"
    if sha256_file(manifest_path) != summary.get("manifest_sha256"):
        raise ValueError("Dataset manifest hash differs from generation summary")
    rows = read_v2_manifest(manifest_path)
    locked_split = {
        row["floorplan_id"]: row
        for row in read_csv_artifact(args.lock_root, lock["artifacts"]["source_split"])
    }
    expected_per_floorplan = int(
        lock["datasets"][args.dataset_kind][
            "samples_per_train_or_validation_floorplan"
        ]
    )
    counts: Counter[tuple[str, str]] = Counter()
    group_splits: dict[str, set[str]] = {}
    source_splits: dict[str, set[str]] = {}
    pair_splits: dict[tuple[str, str], set[str]] = {}
    white_counts = {"train": np.zeros(3, dtype=np.int64), "val": np.zeros(3, dtype=np.int64)}
    target_counts = {"train": 0, "val": 0}
    pixel_counts = {"train": 0, "val": 0}
    one_hot_cells = 0
    known_cells = 0
    known_agree_cells = 0
    for row in rows:
        split = row["split"]
        floorplan_id = row["floorplan_id"]
        locked = locked_split.get(floorplan_id)
        if locked is None or locked["split"] != split:
            raise ValueError(f"Sample split differs from lock: {row['sample_id']}")
        if locked["split"] == "test":
            raise ValueError("Test sample found in v2 dataset")
        if locked["operational_group_id"] != row["group_key"]:
            raise ValueError(f"Operational group differs from lock: {row['sample_id']}")
        if locked["gt_sha256"] != row["source_sha256"]:
            raise ValueError(f"Source SHA differs from lock: {row['sample_id']}")
        counts[(split, floorplan_id)] += 1
        group_splits.setdefault(row["group_key"], set()).add(split)
        source_splits.setdefault(row["source_sha256"], set()).add(split)
        pair = (row["obs_sha256"], row["target_sha256"])
        pair_splits.setdefault(pair, set()).add(split)
        obs_path = safe_child(dataset_root, row["obs_relpath"])
        target_path = safe_child(dataset_root, row["target_relpath"])
        raw_obs_path = safe_child(dataset_root, row["raw_obs_relpath"])
        raw_target_path = safe_child(dataset_root, row["raw_target_relpath"])
        for path, field in (
            (obs_path, "obs_sha256"),
            (target_path, "target_sha256"),
            (raw_obs_path, "raw_obs_sha256"),
            (raw_target_path, "raw_target_sha256"),
        ):
            if sha256_file(path) != row[field]:
                raise ValueError(f"Generated hash mismatch: {row['sample_id']} {field}")
        observation = np.asarray(Image.open(obs_path).convert("RGB"), dtype=np.uint8)
        target = np.asarray(Image.open(target_path).convert("L"), dtype=np.uint8)
        raw_observation = np.asarray(Image.open(raw_obs_path).convert("RGB"), dtype=np.uint8)
        raw_target = np.asarray(Image.open(raw_target_path).convert("L"), dtype=np.uint8)
        if observation.shape != (256, 256, 3) or target.shape != (256, 256):
            raise ValueError(f"Model shape mismatch: {row['sample_id']}")
        if raw_observation.shape != (480, 480, 3) or raw_target.shape != (480, 480):
            raise ValueError(f"Raw shape mismatch: {row['sample_id']}")
        binary = observation == 255
        exact_one_hot = binary.sum(axis=2) == 1
        one_hot_cells += int(exact_one_hot.sum())
        known = ~binary[:, :, 1]
        agreement = binary[:, :, 0] == (target == 255)
        known_cells += int(known.sum())
        known_agree_cells += int((known & agreement).sum())
        white_counts[split] += binary.sum(axis=(0, 1))
        target_counts[split] += int((target == 255).sum())
        pixel_counts[split] += target.size
    group_leakage = sum(len(splits) > 1 for splits in group_splits.values())
    source_leakage = sum(len(splits) > 1 for splits in source_splits.values())
    pair_leakage = sum(len(splits) > 1 for splits in pair_splits.values())
    expected_floorplans = {"train": 110, "val": 23}
    full_coverage = {
        split: sum(
            counts[(split, floorplan_id)] == expected_per_floorplan
            for floorplan_id, locked in locked_split.items()
            if locked["split"] == split
        )
        / expected_floorplans[split]
        for split in ("train", "val")
    }
    total_cells = sum(pixel_counts.values())
    result = {
        "schema": "mso.prospective_v2.dataset_validation/1",
        "status": "prospective_v2_validation_not_historical_recovery",
        "dataset_kind": args.dataset_kind,
        "v2_lock_sha256": lock_sha,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "source_split_sha256": lock["artifacts"]["source_split"]["sha256"],
        "samples": len(rows),
        "expected_samples_per_floorplan": expected_per_floorplan,
        "floorplan_full_coverage_fraction": full_coverage,
        "one_hot_fraction": one_hot_cells / total_cells,
        "known_target_agreement": known_agree_cells / known_cells,
        "cross_split_operational_group_leakage": group_leakage,
        "cross_split_source_hash_leakage": source_leakage,
        "cross_split_processed_pair_leakage": pair_leakage,
        "pooled_observation_channel_fraction": {
            split: (white_counts[split] / pixel_counts[split]).tolist()
            for split in ("train", "val")
        },
        "pooled_target_fraction": {
            split: target_counts[split] / pixel_counts[split]
            for split in ("train", "val")
        },
        "group_semantics": "operational_prefix_group_not_semantically_verified_building",
        "test_rows_in_manifest": 0,
        "test_images_decoded": 0,
        "test_outcomes_accessed": False,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
