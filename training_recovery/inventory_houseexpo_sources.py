#!/usr/bin/env python3
"""Inventory explicit HouseExpo house IDs and deterministic renderability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from mso_recovery.common import sha256_file, write_json  # noqa: E402


HEX_ID = re.compile(r"^[0-9a-f]{32}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-smoke-count", type=int, default=256)
    parser.add_argument("--render-seed", type=int, default=20260807)
    parser.add_argument("--split-seed", type=int, default=20260807)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        name: float(np.quantile(array, quantile))
        for name, quantile in (
            ("minimum", 0.0),
            ("p10", 0.1),
            ("median", 0.5),
            ("p90", 0.9),
            ("p99", 0.99),
            ("maximum", 1.0),
        )
    }


def stable_id_sha256(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def render_polygon(vertices: np.ndarray, pixels_per_meter: int) -> np.ndarray:
    scaled = np.rint(vertices * pixels_per_meter).astype(np.int32)
    minimum = scaled.min(axis=0)
    shifted = scaled - minimum + 1
    maximum = shifted.max(axis=0)
    canvas = np.zeros((int(maximum[1]) + 1, int(maximum[0]) + 1), dtype=np.uint8)
    cv2.drawContours(canvas, [shifted], 0, 255, thickness=-1)
    return canvas


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = [
        line.strip()
        for line in args.id_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("House ID list contains duplicates")
    invalid_ids = [value for value in ids if not HEX_ID.fullmatch(value)]
    if invalid_ids:
        raise ValueError(f"Non-canonical house IDs: {invalid_ids[:5]}")
    rng = np.random.default_rng(args.split_seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    validation_count = int(round(len(shuffled) * args.validation_fraction))
    validation = set(shuffled[:validation_count])

    rows: list[dict[str, object]] = []
    aggregate_json_hash = hashlib.sha256()
    key_sets: Counter[str] = Counter()
    room_categories: Counter[str] = Counter()
    missing: list[str] = []
    id_mismatches: list[str] = []
    invalid_vertices: list[str] = []
    file_hashes: Counter[str] = Counter()
    for index, house_id in enumerate(ids):
        path = args.json_root / f"{house_id}.json"
        if not path.is_file():
            missing.append(house_id)
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        file_hashes[digest] += 1
        aggregate_json_hash.update(house_id.encode("ascii"))
        aggregate_json_hash.update(b"\0")
        aggregate_json_hash.update(bytes.fromhex(digest))
        payload = json.loads(raw)
        key_sets[",".join(sorted(payload))] += 1
        if payload.get("id") != house_id:
            id_mismatches.append(house_id)
        vertices = np.asarray(payload.get("verts", []), dtype=np.float64)
        valid = (
            vertices.ndim == 2
            and vertices.shape[1:] == (2,)
            and len(vertices) >= 3
            and bool(np.isfinite(vertices).all())
        )
        if not valid:
            invalid_vertices.append(house_id)
            continue
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        width, height = maximum - minimum
        polygon_area = float(abs(cv2.contourArea(vertices.astype(np.float32))))
        categories = payload.get("room_category", {})
        if isinstance(categories, dict):
            room_categories.update(str(key).lower() for key in categories)
        row = {
            "house_id": house_id,
            "split": "val" if house_id in validation else "train",
            "json_relpath": f"{house_id}.json",
            "json_sha256": digest,
            "json_size_bytes": len(raw),
            "vertex_count": len(vertices),
            "width_m": float(width),
            "height_m": float(height),
            "polygon_area_m2": polygon_area,
            "room_num_declared": payload.get("room_num"),
            "room_category_count": len(categories) if isinstance(categories, dict) else 0,
            "predicted_width_px_at_8ppm": int(math.ceil(width * 8)) + 3,
            "predicted_height_px_at_8ppm": int(math.ceil(height * 8)) + 3,
            "predicted_width_px_at_16ppm": int(math.ceil(width * 16)) + 3,
            "predicted_height_px_at_16ppm": int(math.ceil(height * 16)) + 3,
            "predicted_width_px_at_25ppm": int(math.ceil(width * 25)) + 3,
            "predicted_height_px_at_25ppm": int(math.ceil(height * 25)) + 3,
        }
        rows.append(row)
        if (index + 1) % 5000 == 0:
            print(json.dumps({"parsed": index + 1, "total": len(ids)}), flush=True)

    if missing or id_mismatches or invalid_vertices:
        raise ValueError(
            f"Invalid source inventory: missing={len(missing)}, "
            f"id_mismatch={len(id_mismatches)}, invalid_vertices={len(invalid_vertices)}"
        )
    if len(rows) != len(ids):
        raise RuntimeError(f"Inventory is incomplete: {len(rows)} of {len(ids)}")

    csv_path = args.output_dir / "house_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    smoke_rng = np.random.default_rng(args.render_seed)
    smoke_indices = np.sort(
        smoke_rng.choice(
            len(rows), size=min(args.render_smoke_count, len(rows)), replace=False
        )
    )
    render_rows: list[dict[str, object]] = []
    for index in smoke_indices:
        row = rows[int(index)]
        payload = json.loads(
            (args.json_root / str(row["json_relpath"])).read_text(encoding="utf-8")
        )
        vertices = np.asarray(payload["verts"], dtype=np.float64)
        rendered = render_polygon(vertices, 16)
        render_rows.append(
            {
                "house_id": row["house_id"],
                "height_px": rendered.shape[0],
                "width_px": rendered.shape[1],
                "free_fraction": float(np.mean(rendered == 255)),
                "obstacle_fraction": float(np.mean(rendered == 0)),
            }
        )
    render_csv = args.output_dir / "render_smoke_16ppm.csv"
    with render_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(render_rows[0]))
        writer.writeheader()
        writer.writerows(render_rows)

    duplicate_hash_groups = sum(count > 1 for count in file_hashes.values())
    summary = {
        "status": "new_experiment_source_inventory_not_historical_recovery",
        "semantic_unit": "one JSON ID is one HouseExpo house/building",
        "json_root": str(args.json_root),
        "id_list_sha256": sha256_file(args.id_list),
        "ordered_house_ids_sha256": stable_id_sha256(ids),
        "aggregate_json_content_sha256": aggregate_json_hash.hexdigest(),
        "houses": len(rows),
        "train_houses": sum(row["split"] == "train" for row in rows),
        "validation_houses": sum(row["split"] == "val" for row in rows),
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "building_overlap": 0,
        "missing_json": 0,
        "json_id_mismatches": 0,
        "invalid_vertex_arrays": 0,
        "duplicate_json_hash_groups": duplicate_hash_groups,
        "schema_key_sets": dict(sorted(key_sets.items())),
        "top_room_categories": room_categories.most_common(20),
        "vertex_count": quantiles([float(row["vertex_count"]) for row in rows]),
        "width_m": quantiles([float(row["width_m"]) for row in rows]),
        "height_m": quantiles([float(row["height_m"]) for row in rows]),
        "polygon_area_m2": quantiles([float(row["polygon_area_m2"]) for row in rows]),
        "render_smoke": {
            "pixels_per_meter": 16,
            "houses": len(render_rows),
            "seed": args.render_seed,
            "house_ids_sha256": stable_id_sha256(
                [str(row["house_id"]) for row in render_rows]
            ),
            "height_px": quantiles([float(row["height_px"]) for row in render_rows]),
            "width_px": quantiles([float(row["width_px"]) for row in render_rows]),
            "free_fraction": quantiles(
                [float(row["free_fraction"]) for row in render_rows]
            ),
            "render_failures": 0,
            "render_rows_sha256": sha256_file(render_csv),
        },
        "house_manifest_sha256": sha256_file(csv_path),
        "reserved_external_test_source": "KTH is outside this source inventory and is not evaluated here",
        "test_outcomes_accessed": False,
    }
    write_json(args.output_dir / "house_inventory_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
