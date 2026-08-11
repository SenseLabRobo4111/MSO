#!/usr/bin/env python3
"""Audit split isolation, hashes, and image contracts before training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from mso_recovery.common import sha256_file  # noqa: E402
from mso_recovery.data import FloorplanManifestDataset, read_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-buildings", type=int)
    parser.add_argument("--expected-floorplans", type=int)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--verify-images", action="store_true")
    return parser.parse_args()


def split_sets(rows: list[dict[str, str]], key: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        result[row[key]].add(row["split"])
    return result


def cross_split_count(values: dict[object, set[str]]) -> int:
    return sum(len(splits) > 1 for splits in values.values())


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    buildings = split_sets(rows, "group_key")
    floorplans = split_sets(rows, "floorplan_id")
    source_hashes = split_sets(rows, "source_sha256")
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        pairs[(row["obs_sha256"], row["target_sha256"])].add(row["split"])

    checks = {
        "building_cross_split": cross_split_count(buildings),
        "floorplan_cross_split": cross_split_count(floorplans),
        "source_gt_hash_cross_split": cross_split_count(source_hashes),
        "processed_pair_hash_cross_split": cross_split_count(pairs),
    }
    if any(checks.values()):
        raise SystemExit(f"Split-isolation audit failed: {checks}")
    if args.expected_buildings is not None and len(buildings) != args.expected_buildings:
        raise SystemExit(f"Expected {args.expected_buildings} buildings, found {len(buildings)}")
    if args.expected_floorplans is not None and len(floorplans) != args.expected_floorplans:
        raise SystemExit(
            f"Expected {args.expected_floorplans} floorplans, found {len(floorplans)}"
        )
    if args.expected_samples is not None and len(rows) != args.expected_samples:
        raise SystemExit(f"Expected {args.expected_samples} samples, found {len(rows)}")

    image_contract_samples = 0
    integrity: dict[str, str] = {}
    if args.verify_images:
        for split in ("train", "val", "test"):
            dataset = FloorplanManifestDataset(
                args.dataset_root, args.manifest, split, verify_hashes=True
            )
            for index in range(len(dataset)):
                dataset[index]
                image_contract_samples += 1
            integrity[split] = dataset.integrity

    split_counts = {
        split: {
            "samples": sum(row["split"] == split for row in rows),
            "floorplans": len(
                {row["floorplan_id"] for row in rows if row["split"] == split}
            ),
            "buildings": len(
                {row["group_key"] for row in rows if row["split"] == split}
            ),
        }
        for split in ("train", "val", "test")
    }
    report = {
        "audit_passed": True,
        "manifest_sha256": sha256_file(args.manifest),
        "samples": len(rows),
        "floorplans": len(floorplans),
        "buildings": len(buildings),
        "split_counts": split_counts,
        "cross_split_checks": checks,
        "image_contract_samples_verified": image_contract_samples,
        "integrity": integrity,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
