#!/usr/bin/env python3
"""Build a reconstructed split from explicit semantic group metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ALLOWED_GROUP_KINDS = {"building", "floorplan", "scene", "environment"}
FORBIDDEN_GROUP_KINDS = {"worker", "thread", "process", "prefix"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build(inventory_path: Path, metadata_path: Path):
    inventory = read_csv(inventory_path)
    metadata = read_csv(metadata_path)
    required_inventory = {
        "sample_id",
        "source_worker_prefix",
        "obs_relpath",
        "target_relpath",
        "obs_sha256",
        "target_sha256",
        "processed_pair_sha256",
    }
    missing_inventory = required_inventory.difference(inventory[0])
    if missing_inventory:
        raise ValueError(f"Inventory lacks columns: {sorted(missing_inventory)}")
    required_metadata = {"sample_id", "split", "group_key", "group_kind"}
    missing = required_metadata.difference(metadata[0])
    if missing:
        raise ValueError(f"Metadata lacks columns: {sorted(missing)}")

    by_sample = {}
    for row in metadata:
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in by_sample:
            raise ValueError(f"Missing or duplicate metadata sample_id: {sample_id!r}")
        by_sample[sample_id] = row

    inventory_ids = {row["sample_id"] for row in inventory}
    metadata_ids = set(by_sample)
    if inventory_ids != metadata_ids:
        missing_meta = sorted(inventory_ids - metadata_ids)[:10]
        extra_meta = sorted(metadata_ids - inventory_ids)[:10]
        raise ValueError(
            f"Metadata must cover the inventory exactly; missing={missing_meta}, extra={extra_meta}"
        )

    rows = []
    group_splits: dict[tuple[str, str], set[str]] = {}
    prefix_equivalent = 0
    for source in inventory:
        meta = by_sample[source["sample_id"]]
        split = meta["split"].strip().lower()
        kind = meta["group_kind"].strip().lower()
        key = meta["group_key"].strip()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {split!r} for {source['sample_id']}")
        if kind in FORBIDDEN_GROUP_KINDS or kind not in ALLOWED_GROUP_KINDS:
            raise ValueError(
                f"Invalid group_kind {kind!r}; worker/thread/process/prefix identifiers "
                f"are provenance only. Allowed semantic kinds: {sorted(ALLOWED_GROUP_KINDS)}"
            )
        if not key:
            raise ValueError(f"Missing semantic group_key for {source['sample_id']}")
        if key == source.get("source_worker_prefix", ""):
            prefix_equivalent += 1
        group_splits.setdefault((kind, key), set()).add(split)
        row = dict(source)
        row.update(
            {
                "split": split,
                "group_key": key,
                "group_kind": kind,
                "manifest_status": "reconstructed_canonical",
            }
        )
        rows.append(row)

    if prefix_equivalent / len(rows) >= 0.90:
        raise ValueError(
            "Proposed semantic group_key reproduces the source worker prefix for at least "
            "90% of samples; worker prefixes cannot stand in for buildings/floorplans."
        )
    leaks = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaks:
        raise ValueError(f"Semantic groups cross splits: {list(leaks.items())[:10]}")
    pair_splits: dict[str, set[str]] = {}
    for row in rows:
        pair_hash = row["processed_pair_sha256"].strip()
        if not pair_hash:
            raise ValueError(f"Missing processed_pair_sha256 for {row['sample_id']}")
        pair_splits.setdefault(pair_hash, set()).add(row["split"])
    duplicate_leaks = {
        pair_hash: splits for pair_hash, splits in pair_splits.items() if len(splits) > 1
    }
    if duplicate_leaks:
        preview = [(digest, sorted(splits)) for digest, splits in duplicate_leaks.items()][:10]
        raise ValueError(f"Exact processed pairs cross splits: {preview}")
    return rows


def write_split(
    inventory_path: Path, metadata_path: Path, output: Path, summary: Path
) -> dict[str, object]:
    rows = build(inventory_path, metadata_path)
    preferred = [
        "sample_id",
        "split",
        "group_key",
        "group_kind",
        "manifest_status",
        "split_status",
        "archive_partition",
        "source_worker_prefix",
        "obs_relpath",
        "target_relpath",
        "obs_sha256",
        "target_sha256",
        "processed_pair_sha256",
        "obs_shape",
        "target_shape",
        "processed_obs_shape",
        "processed_target_shape",
    ]
    extras = sorted(set(rows[0]).difference(preferred))
    fields = [field for field in preferred if field in rows[0]] + extras
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["sample_id"]))

    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "val", "test")
    }
    payload = {
        "schema": "reconstructed_canonical_split_v1",
        "status": "reconstructed_not_historical",
        "sample_count": len(rows),
        "split_counts": counts,
        "semantic_group_count": len(
            {(row["group_kind"], row["group_key"]) for row in rows}
        ),
        "source_inventory_sha256": sha256_file(inventory_path),
        "source_metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": sha256_file(output),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_split(args.inventory, args.metadata, args.output, args.summary)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
