#!/usr/bin/env python3
"""Pixel-check a new renderer against the accessible HouseExpo jsonReader."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np


STATUS = "renderer_validation_for_new_experiment_not_historical_recovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--simulator-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixels-per-meter", type=int, default=16)
    parser.add_argument("--world-padding", type=int, default=10)
    parser.add_argument("--minimum-shape-sum", type=int, default=1500)
    parser.add_argument("--houses", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def new_render(vertices: np.ndarray, pixels_per_meter: int) -> np.ndarray:
    scaled = (vertices * pixels_per_meter).astype(np.int32)
    minimum = scaled.min(axis=0)
    shifted = scaled - minimum + 1
    maximum = shifted.max(axis=0)
    output = np.zeros((int(maximum[1]) + 1, int(maximum[0]) + 1), dtype=np.float64)
    cv2.drawContours(output, [shifted], 0, 255, thickness=-1)
    return output


def old_world_chain(input_world: np.ndarray, padding: int) -> tuple[np.ndarray, np.ndarray]:
    world = np.zeros_like(input_world)
    world[input_world == 0] = 100
    world[input_world == 255] = 0
    obstacle = np.where(world == 100)
    world = world[
        int(obstacle[0].min()) : int(obstacle[0].max()) + 1,
        int(obstacle[1].min()) : int(obstacle[1].max()) + 1,
    ]
    padded = np.pad(world, padding, mode="constant", constant_values=100)
    dilated = cv2.dilate(padded, np.ones((3, 3), np.uint8), iterations=3)
    return padded, dilated


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.simulator_root))
    import jsonReader  # type: ignore  # noqa: PLC0415

    reader = jsonReader.jsonReader(str(args.json_root) + "/", args.pixels_per_meter)
    with args.manifest.open("r", encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    eligible: list[dict[str, str]] = []
    rejected_by_split = {"train": 0, "val": 0}
    for row in manifest:
        payload = json.loads((args.json_root / row["json_relpath"]).read_text(encoding="utf-8"))
        vertices = np.asarray(payload["verts"], dtype=np.float64)
        scaled = (vertices * args.pixels_per_meter).astype(np.int32)
        shape_sum = int(
            scaled[:, 0].max()
            - scaled[:, 0].min()
            + 2
            + 2 * args.world_padding
            + scaled[:, 1].max()
            - scaled[:, 1].min()
            + 2
            + 2 * args.world_padding
        )
        if shape_sum >= args.minimum_shape_sum:
            eligible.append(row)
        else:
            rejected_by_split[row["split"]] += 1
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(eligible), size=min(args.houses, len(eligible)), replace=False))
    results: list[dict[str, object]] = []
    for index in indices:
        row = eligible[int(index)]
        house_id = row["house_id"]
        payload = json.loads((args.json_root / row["json_relpath"]).read_text(encoding="utf-8"))
        vertices = np.asarray(payload["verts"], dtype=np.float64)
        official_input, _ = reader.read_json(house_id)
        candidate_input = new_render(vertices, args.pixels_per_meter)
        padded, dilated = old_world_chain(official_input, args.world_padding)
        candidate_equal = bool(
            official_input.shape == candidate_input.shape
            and np.array_equal(official_input, candidate_input)
        )
        prefix = args.output_dir / house_id
        cv2.imwrite(str(prefix.with_name(prefix.name + "_official_input.png")), official_input)
        cv2.imwrite(str(prefix.with_name(prefix.name + "_candidate_input.png")), candidate_input)
        cv2.imwrite(
            str(prefix.with_name(prefix.name + "_input_xor.png")),
            (official_input != candidate_input).astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(prefix.with_name(prefix.name + "_official_world_padded.png")),
            np.where(padded == 100, 255, 0).astype(np.uint8),
        )
        cv2.imwrite(
            str(prefix.with_name(prefix.name + "_official_world_dilated.png")),
            np.where(dilated == 100, 255, 0).astype(np.uint8),
        )
        results.append(
            {
                "house_id": house_id,
                "split": row["split"],
                "input_shape": list(official_input.shape),
                "candidate_pixel_exact": candidate_equal,
                "different_pixels": int(np.sum(official_input != candidate_input)),
                "official_input_sha256": array_sha256(official_input),
                "candidate_input_sha256": array_sha256(candidate_input),
                "official_input_free_fraction": float(np.mean(official_input == 255)),
                "official_padded_obstacle_fraction": float(np.mean(padded == 100)),
                "official_dilated_obstacle_fraction": float(np.mean(dilated == 100)),
                "official_padded_sha256": array_sha256(padded),
                "official_dilated_sha256": array_sha256(dilated),
            }
        )
    summary = {
        "status": STATUS,
        "source_houses": len(manifest),
        "eligible_houses": len(eligible),
        "eligible_by_split": {
            split: sum(row["split"] == split for row in eligible)
            for split in ("train", "val")
        },
        "rejected_below_visible_size_filter_by_split": rejected_by_split,
        "visible_size_filter": {
            "pixels_per_meter": args.pixels_per_meter,
            "world_padding_each_side_px": args.world_padding,
            "minimum_world_height_plus_width_px": args.minimum_shape_sum,
        },
        "sample_seed": args.seed,
        "sampled_houses": len(results),
        "candidate_pixel_exact_houses": sum(
            bool(row["candidate_pixel_exact"]) for row in results
        ),
        "rows": results,
        "conclusion": "candidate polygon raster is checked against the accessible official jsonReader; padded/dilated world semantics are recorded separately",
        "test_outcomes_accessed": False,
    }
    (args.output_dir / "renderer_validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
