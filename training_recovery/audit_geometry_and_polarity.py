#!/usr/bin/env python3
"""Audit legacy/replacement geometry, class marginals, and source-map polarity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from mso_recovery.common import sha256_file, write_json  # noqa: E402
from mso_recovery.data import read_manifest, safe_child  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preserved-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--replacement-manifest", type=Path, required=True)
    parser.add_argument("--floorplan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preserved-samples", type=int, default=1000)
    parser.add_argument("--replacement-samples-per-split", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=20260807)
    return parser.parse_args()


def sampled_indices(count: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or maximum >= count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=maximum, replace=False))


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        name: float(np.quantile(array, value))
        for name, value in (
            ("min", 0.0),
            ("p10", 0.1),
            ("median", 0.5),
            ("p90", 0.9),
            ("max", 1.0),
        )
    }


def stable_list_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class PairMarginals:
    def __init__(self) -> None:
        self.samples = 0
        self.obs_shapes: Counter[str] = Counter()
        self.target_shapes: Counter[str] = Counter()
        self.obs_unique: list[Counter[int]] = [Counter(), Counter(), Counter()]
        self.target_unique: Counter[int] = Counter()
        self.channel_fractions: list[list[float]] = [[], [], []]
        self.target_fractions: list[float] = []
        self.one_hot_exact_cells = 0
        self.total_cells = 0

    def update(self, observation: np.ndarray, target: np.ndarray) -> None:
        if observation.ndim != 3 or observation.shape[2] != 3 or target.ndim != 2:
            raise ValueError(
                f"Unexpected pair shapes: observation={observation.shape}, target={target.shape}"
            )
        self.samples += 1
        self.obs_shapes["x".join(str(value) for value in observation.shape)] += 1
        self.target_shapes["x".join(str(value) for value in target.shape)] += 1
        for channel in range(3):
            values, counts = np.unique(observation[:, :, channel], return_counts=True)
            self.obs_unique[channel].update(
                {int(value): int(count) for value, count in zip(values, counts)}
            )
            self.channel_fractions[channel].append(
                float(np.mean(observation[:, :, channel] == 255))
            )
        values, counts = np.unique(target, return_counts=True)
        self.target_unique.update(
            {int(value): int(count) for value, count in zip(values, counts)}
        )
        self.target_fractions.append(float(np.mean(target == 255)))
        white_count = (observation == 255).sum(axis=2)
        self.one_hot_exact_cells += int(np.sum(white_count == 1))
        self.total_cells += int(white_count.size)

    def summary(self) -> dict[str, object]:
        total_channel_pixels = [sum(counter.values()) for counter in self.obs_unique]
        target_pixels = sum(self.target_unique.values())
        return {
            "samples": self.samples,
            "observation_shapes": dict(sorted(self.obs_shapes.items())),
            "target_shapes": dict(sorted(self.target_shapes.items())),
            "observation_channel_white_fraction_global": [
                counter[255] / max(1, total)
                for counter, total in zip(self.obs_unique, total_channel_pixels)
            ],
            "observation_channel_white_fraction_per_sample": [
                quantiles(values) for values in self.channel_fractions
            ],
            "target_white_fraction_global": self.target_unique[255]
            / max(1, target_pixels),
            "target_white_fraction_per_sample": quantiles(self.target_fractions),
            "one_hot_exact_cell_fraction": self.one_hot_exact_cells
            / max(1, self.total_cells),
            "observation_channel_value_counts": [
                {str(key): value for key, value in sorted(counter.items())}
                for counter in self.obs_unique
            ],
            "target_value_counts": {
                str(key): value for key, value in sorted(self.target_unique.items())
            },
        }


def nearest_256(observation: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resampling = Image.Resampling.NEAREST
    resized_observation = np.asarray(
        Image.fromarray(observation, mode="RGB").resize((256, 256), resampling),
        dtype=np.uint8,
    )
    resized_target = np.asarray(
        Image.fromarray(target, mode="L").resize((256, 256), resampling),
        dtype=np.uint8,
    )
    return resized_observation, resized_target


def audit_preserved(root: Path, sample_count: int, seed: int) -> dict[str, object]:
    train_root = root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)
    directories = sorted(path for path in train_root.iterdir() if path.is_dir())
    indices = sampled_indices(len(directories), sample_count, seed)
    chosen = [directories[int(index)] for index in indices]
    raw = PairMarginals()
    processed = PairMarginals()
    for directory in chosen:
        observation = np.asarray(
            Image.open(directory / "local_obs_0.png").convert("RGB"), dtype=np.uint8
        )
        target = np.asarray(
            Image.open(directory / "local_map_0.png").convert("L"), dtype=np.uint8
        )
        raw.update(observation, target)
        processed.update(*nearest_256(observation, target))
    return {
        "status": "preserved_archive_not_manuscript_split",
        "partition_accessed": "train_only",
        "test_partition_accessed": False,
        "train_directories_total": len(directories),
        "sampled_train_directories": len(chosen),
        "sample_seed": seed,
        "sample_ids_sha256": stable_list_sha256(
            [f"train/{path.name}" for path in chosen]
        ),
        "raw": raw.summary(),
        "legacy_loader_nearest_256": processed.summary(),
    }


def audit_replacement(
    root: Path, manifest: Path, samples_per_split: int, seed: int
) -> dict[str, object]:
    rows = read_manifest(manifest)
    result: dict[str, object] = {
        "status": "geometry_mismatched_pilot_not_historical",
        "manifest_sha256": sha256_file(manifest),
        "splits_accessed": ["train", "val"],
        "test_images_accessed": False,
    }
    for split_index, split in enumerate(("train", "val")):
        split_rows = [row for row in rows if row["split"] == split]
        indices = sampled_indices(
            len(split_rows), samples_per_split, seed + 1009 * split_index
        )
        selected = [split_rows[int(index)] for index in indices]
        marginals = PairMarginals()
        for row in selected:
            observation = np.asarray(
                Image.open(safe_child(root, row["obs_relpath"])).convert("RGB"),
                dtype=np.uint8,
            )
            target = np.asarray(
                Image.open(safe_child(root, row["target_relpath"])).convert("L"),
                dtype=np.uint8,
            )
            marginals.update(observation, target)
        result[split] = {
            "samples_total": len(split_rows),
            "samples_audited": len(selected),
            "sample_ids_sha256": stable_list_sha256(
                [row["sample_id"] for row in selected]
            ),
            "raw_and_model_input_256": marginals.summary(),
        }
    return result


def largest_component_fraction(mask: np.ndarray) -> float:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(np.max(areas) / mask.size)


def border_values(array: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (array[0, :], array[-1, :], array[1:-1, 0], array[1:-1, -1])
    )


def audit_floorplans(root: Path, rows_path: Path) -> dict[str, object]:
    floorplan_rows: list[dict[str, object]] = []
    shape_counts: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    for bitmap in sorted(root.glob("*/GT.bmp")):
        array = np.asarray(Image.open(bitmap).convert("L"), dtype=np.uint8)
        dark = array < 128
        light = ~dark
        border = border_values(array)
        json_path = bitmap.with_name("GT.json")
        resolution: float | None = None
        if json_path.is_file():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if "resolution" in payload:
                resolution = float(payload["resolution"])
                resolutions[f"{resolution:.12g}"] += 1
        else:
            resolutions["missing"] += 1
        shape = "x".join(str(value) for value in array.shape)
        shape_counts[shape] += 1
        row = {
            "floorplan_id": bitmap.parent.name,
            "gt_sha256": sha256_file(bitmap),
            "shape": shape,
            "json_present": int(json_path.is_file()),
            "declared_resolution_m_per_cell": resolution,
            "dark_fraction": float(np.mean(dark)),
            "light_fraction": float(np.mean(light)),
            "border_dark_fraction": float(np.mean(border < 128)),
            "border_light_fraction": float(np.mean(border >= 128)),
            "corner_values": ";".join(
                str(int(array[y, x]))
                for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1))
            ),
            "dark_largest_component_fraction": largest_component_fraction(dark),
            "light_largest_component_fraction": largest_component_fraction(light),
            "minimum_value": int(array.min()),
            "maximum_value": int(array.max()),
        }
        floorplan_rows.append(row)

    if not floorplan_rows:
        raise ValueError(f"No GT.bmp floorplans found under {root}")
    with rows_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(floorplan_rows[0]))
        writer.writeheader()
        writer.writerows(floorplan_rows)

    def field_summary(name: str) -> dict[str, float]:
        return quantiles([float(row[name]) for row in floorplan_rows])

    extremes = sorted(floorplan_rows, key=lambda row: float(row["dark_fraction"]))
    return {
        "floorplans": len(floorplan_rows),
        "shape_counts": dict(sorted(shape_counts.items())),
        "declared_resolution_counts": dict(sorted(resolutions.items())),
        "dark_fraction": field_summary("dark_fraction"),
        "light_fraction": field_summary("light_fraction"),
        "border_dark_fraction": field_summary("border_dark_fraction"),
        "dark_largest_component_fraction": field_summary(
            "dark_largest_component_fraction"
        ),
        "light_largest_component_fraction": field_summary(
            "light_largest_component_fraction"
        ),
        "lowest_dark_fraction_floorplans": [
            {"floorplan_id": row["floorplan_id"], "dark_fraction": row["dark_fraction"]}
            for row in extremes[:10]
        ],
        "highest_dark_fraction_floorplans": [
            {"floorplan_id": row["floorplan_id"], "dark_fraction": row["dark_fraction"]}
            for row in extremes[-10:]
        ],
        "per_floorplan_csv_sha256": sha256_file(rows_path),
        "polarity_conclusion": (
            "not inferred from grayscale alone; generator/semantic evidence is required"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    floorplan_rows_path = args.output_dir / "floorplan_polarity_rows.csv"
    result = {
        "status": "read_only_geometry_and_polarity_audit",
        "preserved": audit_preserved(
            args.preserved_root, args.preserved_samples, args.sample_seed
        ),
        "replacement_v1": audit_replacement(
            args.replacement_root,
            args.replacement_manifest,
            args.replacement_samples_per_split,
            args.sample_seed,
        ),
        "source_floorplans": audit_floorplans(args.floorplan_root, floorplan_rows_path),
        "test_data_evaluated": False,
    }
    write_json(args.output_dir / "geometry_and_polarity_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
