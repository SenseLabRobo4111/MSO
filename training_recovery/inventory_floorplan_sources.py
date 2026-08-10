#!/usr/bin/env python3
"""Freeze GT image and resolution-metadata provenance for all source maps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from generate_floorplan_dataset import building_id_from_floorplan  # noqa: E402
from mso_recovery.common import sha256_file, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    for path in (args.output_csv, args.output_summary):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for gt_path in sorted(source_root.glob("*/GT.bmp")):
        metadata_path = gt_path.with_name("GT.json")
        resolution = 0.1
        metadata_hash = "missing_used_declared_default_0.1"
        metadata_relpath = ""
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            resolution = float(metadata.get("resolution", resolution))
            metadata_hash = sha256_file(metadata_path)
            metadata_relpath = metadata_path.relative_to(source_root).as_posix()
        floorplan_id = gt_path.parent.name
        rows.append(
            {
                "building_id": building_id_from_floorplan(floorplan_id),
                "floorplan_id": floorplan_id,
                "gt_relpath": gt_path.relative_to(source_root).as_posix(),
                "gt_sha256": sha256_file(gt_path),
                "metadata_relpath": metadata_relpath,
                "metadata_sha256_or_status": metadata_hash,
                "resolution_m_per_cell": resolution,
            }
        )
    if len(rows) != 156:
        raise SystemExit(f"Expected 156 floorplans, found {len(rows)}")
    if len({row["building_id"] for row in rows}) != 146:
        raise SystemExit("Expected 146 building groups")
    if len({row["gt_sha256"] for row in rows}) != len(rows):
        raise SystemExit("Exact source GT duplicates require explicit review")
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_root_recorded_at_inventory": str(source_root),
        "floorplans": len(rows),
        "buildings": len({row["building_id"] for row in rows}),
        "metadata_files": sum(bool(row["metadata_relpath"]) for row in rows),
        "missing_metadata_using_default_resolution": sum(
            not bool(row["metadata_relpath"]) for row in rows
        ),
        "exact_gt_duplicate_count": len(rows) - len({row["gt_sha256"] for row in rows}),
        "inventory_sha256": sha256_file(args.output_csv),
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
