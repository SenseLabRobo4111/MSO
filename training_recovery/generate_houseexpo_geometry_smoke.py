#!/usr/bin/env python3
"""Generate a train/validation-only HouseExpo geometry smoke dataset.

This is a newly declared experiment.  It uses explicit HouseExpo house IDs and
must not be described as a recovery of the preserved manuscript training run.
KTH is reserved as an external test source and is not opened by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from generate_floorplan_dataset import (  # noqa: E402
    centered_crop,
    move_random_walk,
    ray_offsets,
    scan_from_pose,
    select_navigable_component,
)
from mso_recovery.common import sha256_file, stable_seed, write_json  # noqa: E402


STATUS = "new_houseexpo_experiment_geometry_smoke_not_historical_recovery"


@dataclass(frozen=True)
class SmokeConfig:
    pixels_per_meter: int = 16
    physical_window_m: int = 30
    raw_size: int = 480
    model_size: int = 256
    interpolation: str = "nearest"
    sensor_range_m: float = 5.0
    laser_range_cells: int = 80
    laser_rays: int = 360
    scans_per_sample: int = 15
    movement_m_per_scan: float = 1.0
    movement_cells_per_scan: int = 16
    world_padding_cells: int = 10
    visible_source_minimum_world_shape_sum_px: int = 1500
    samples_per_house: int = 8
    generation_seed: int = 20260807
    minimum_unknown_fraction: float = 0.20
    maximum_unknown_fraction: float = 0.98
    minimum_target_occupied_fraction: float = 0.002
    maximum_target_occupied_fraction: float = 0.95
    maximum_attempts_per_house: int = 320


MANIFEST_FIELDS = (
    "sample_id",
    "split",
    "group_key",
    "group_kind",
    "building_id",
    "floorplan_id",
    "source_relpath",
    "source_sha256",
    "generation_seed",
    "scan_count",
    "rotation_quadrants",
    "raw_obs_relpath",
    "raw_target_relpath",
    "raw_obs_sha256",
    "raw_target_sha256",
    "obs_relpath",
    "target_relpath",
    "obs_sha256",
    "target_sha256",
    "raw_unknown_fraction",
    "raw_target_occupied_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--house-manifest", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-house", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def render_house(payload: dict, config: SmokeConfig) -> tuple[np.ndarray, tuple[int, int]]:
    vertices = (np.asarray(payload["verts"], dtype=np.float64) * config.pixels_per_meter).astype(
        np.int32
    )
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("Invalid HouseExpo vertex array")
    minimum = vertices.min(axis=0)
    shifted = vertices - minimum + 1
    maximum = shifted.max(axis=0)
    free_map = np.zeros((int(maximum[1]) + 1, int(maximum[0]) + 1), dtype=np.uint8)
    cv2.drawContours(free_map, [shifted], 0, 255, thickness=-1)
    occupied = free_map == 0
    occupied = np.pad(
        occupied,
        config.world_padding_cells,
        mode="constant",
        constant_values=True,
    )
    return occupied, occupied.shape


def pad_for_local_window(occupied: np.ndarray, config: SmokeConfig) -> np.ndarray:
    pad = config.raw_size // 2 + config.laser_range_cells + 4
    return np.pad(occupied, pad, mode="constant", constant_values=True)


def create_sample(
    occupied: np.ndarray,
    component: np.ndarray,
    starts: np.ndarray,
    rays: list[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    config: SmokeConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    pose = starts[int(rng.integers(0, len(starts)))].astype(np.int32)
    known = np.zeros(occupied.shape, dtype=bool)
    for _ in range(config.scans_per_sample):
        scan_from_pose(occupied, known, pose, rays)
        pose = move_random_walk(
            pose, component, rng, config.movement_cells_per_scan
        )
    scan_from_pose(occupied, known, pose, rays)
    target = centered_crop(occupied, pose, config.raw_size)
    observed = centered_crop(known, pose, config.raw_size)
    unknown_fraction = float((~observed).mean())
    target_fraction = float(target.mean())
    if not (
        config.minimum_unknown_fraction
        <= unknown_fraction
        <= config.maximum_unknown_fraction
    ):
        return None
    if not (
        config.minimum_target_occupied_fraction
        <= target_fraction
        <= config.maximum_target_occupied_fraction
    ):
        return None
    if int((observed & ~target).sum()) < 32 or int((observed & target).sum()) < 4:
        return None
    observation = np.zeros((config.raw_size, config.raw_size, 3), dtype=np.uint8)
    observation[:, :, 0] = (observed & target).astype(np.uint8) * 255
    observation[:, :, 1] = (~observed).astype(np.uint8) * 255
    observation[:, :, 2] = (observed & ~target).astype(np.uint8) * 255
    target_image = target.astype(np.uint8) * 255
    rotation = int(rng.integers(0, 4))
    if rotation:
        observation = np.rot90(observation, rotation).copy()
        target_image = np.rot90(target_image, rotation).copy()
    return observation, target_image, {
        "unknown_fraction": unknown_fraction,
        "target_occupied_fraction": target_fraction,
        "rotation_quadrants": rotation,
    }


def image_metrics(observations: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    obs_fractions = np.asarray(
        [[float(np.mean(obs[:, :, channel] == 255)) for channel in range(3)] for obs in observations],
        dtype=np.float64,
    )
    target_fractions = np.asarray(
        [float(np.mean(target == 255)) for target in targets], dtype=np.float64
    )

    def describe(values: np.ndarray) -> dict[str, float]:
        return {
            "minimum": float(values.min()),
            "p10": float(np.quantile(values, 0.1)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "maximum": float(values.max()),
            "global_mean": float(values.mean()),
        }

    return {
        "samples": len(observations),
        "observation_channel_white_fraction": [
            describe(obs_fractions[:, channel]) for channel in range(3)
        ],
        "target_white_fraction": describe(target_fractions),
        "one_hot_exact_cell_fraction": float(
            np.mean(
                [
                    np.mean(np.sum(observation == 255, axis=2) == 1)
                    for observation in observations
                ]
            )
        ),
    }


def stable_rows_sha256(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = SmokeConfig(
        samples_per_house=args.samples_per_house,
        generation_seed=args.seed,
    )
    if config.raw_size != config.physical_window_m * config.pixels_per_meter:
        raise ValueError("Raw size is inconsistent with physical window and resolution")
    if config.laser_range_cells != round(
        config.sensor_range_m * config.pixels_per_meter
    ):
        raise ValueError("Laser range cells are inconsistent with physical range")
    if config.movement_cells_per_scan != round(
        config.movement_m_per_scan * config.pixels_per_meter
    ):
        raise ValueError("Movement cells are inconsistent with physical distance")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    incomplete = args.output_root / "GENERATION_INCOMPLETE"
    incomplete.write_text(STATUS + "\n", encoding="utf-8")
    with args.house_manifest.open("r", encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    if not source_rows:
        raise ValueError("Empty HouseExpo source manifest")
    if {row["split"] for row in source_rows} != {"train", "val"}:
        raise ValueError("Source manifest must contain train and val only")

    eligible: list[tuple[dict[str, str], dict, np.ndarray]] = []
    render_failures: list[dict[str, str]] = []
    exact_shape_sums: list[int] = []
    for source_row in source_rows:
        source_path = args.json_root / source_row["json_relpath"]
        if sha256_file(source_path) != source_row["json_sha256"]:
            raise ValueError(f"Source hash mismatch: {source_row['house_id']}")
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            occupied, world_shape = render_house(payload, config)
        except Exception as error:  # evidence is written before failing below
            render_failures.append(
                {"house_id": source_row["house_id"], "error": repr(error)}
            )
            continue
        shape_sum = int(sum(world_shape))
        exact_shape_sums.append(shape_sum)
        if shape_sum >= config.visible_source_minimum_world_shape_sum_px:
            eligible.append((source_row, payload, occupied))
    if render_failures:
        write_json(args.output_root / "render_failures.json", render_failures)
        raise RuntimeError(f"House rendering failures: {len(render_failures)}")
    split_houses = {
        split: sum(row[0]["split"] == split for row in eligible)
        for split in ("train", "val")
    }
    if min(split_houses.values()) < 2:
        raise RuntimeError(f"Too few eligible houses for smoke split: {split_houses}")

    rays = ray_offsets(config.laser_rays, config.laser_range_cells)
    manifest_rows: list[dict[str, object]] = []
    raw_by_split: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        "train": ([], []),
        "val": ([], []),
    }
    model_by_split: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        "train": ([], []),
        "val": ([], []),
    }
    rejected_attempts = 0
    failed_houses: list[dict[str, object]] = []
    for house_index, (source_row, _payload, base_occupied) in enumerate(eligible, start=1):
        house_id = source_row["house_id"]
        split = source_row["split"]
        occupied = pad_for_local_window(base_occupied, config)
        component, starts = select_navigable_component(occupied)
        accepted = 0
        attempts = 0
        while accepted < config.samples_per_house and attempts < config.maximum_attempts_per_house:
            attempts += 1
            sample_seed = stable_seed(config.generation_seed, house_id, attempts)
            rng = np.random.default_rng(sample_seed)
            generated = create_sample(occupied, component, starts, rays, rng, config)
            if generated is None:
                rejected_attempts += 1
                continue
            raw_observation, raw_target, details = generated
            model_observation = cv2.resize(
                raw_observation,
                (config.model_size, config.model_size),
                interpolation=cv2.INTER_NEAREST,
            )
            model_target = cv2.resize(
                raw_target,
                (config.model_size, config.model_size),
                interpolation=cv2.INTER_NEAREST,
            )
            sample_id = f"{house_id}__{accepted:02d}"
            sample_dir = args.output_root / "samples" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=False)
            raw_obs_path = sample_dir / "raw_obs_480.png"
            raw_target_path = sample_dir / "raw_target_480.png"
            obs_path = sample_dir / "local_obs_0.png"
            target_path = sample_dir / "local_map_0.png"
            Image.fromarray(raw_observation, mode="RGB").save(raw_obs_path)
            Image.fromarray(raw_target, mode="L").save(raw_target_path)
            Image.fromarray(model_observation, mode="RGB").save(obs_path)
            Image.fromarray(model_target, mode="L").save(target_path)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "group_key": house_id,
                    "group_kind": "building",
                    "building_id": house_id,
                    "floorplan_id": house_id,
                    "source_relpath": source_row["json_relpath"],
                    "source_sha256": source_row["json_sha256"],
                    "generation_seed": sample_seed,
                    "scan_count": config.scans_per_sample + 1,
                    "rotation_quadrants": details["rotation_quadrants"],
                    "raw_obs_relpath": raw_obs_path.relative_to(args.output_root).as_posix(),
                    "raw_target_relpath": raw_target_path.relative_to(args.output_root).as_posix(),
                    "raw_obs_sha256": sha256_file(raw_obs_path),
                    "raw_target_sha256": sha256_file(raw_target_path),
                    "obs_relpath": obs_path.relative_to(args.output_root).as_posix(),
                    "target_relpath": target_path.relative_to(args.output_root).as_posix(),
                    "obs_sha256": sha256_file(obs_path),
                    "target_sha256": sha256_file(target_path),
                    "raw_unknown_fraction": f"{details['unknown_fraction']:.9f}",
                    "raw_target_occupied_fraction": f"{details['target_occupied_fraction']:.9f}",
                }
            )
            raw_by_split[split][0].append(raw_observation)
            raw_by_split[split][1].append(raw_target)
            model_by_split[split][0].append(model_observation)
            model_by_split[split][1].append(model_target)
            accepted += 1
        if accepted != config.samples_per_house:
            failed_houses.append(
                {"house_id": house_id, "split": split, "accepted": accepted, "attempts": attempts}
            )
        print(
            json.dumps(
                {
                    "house": house_index,
                    "eligible_houses": len(eligible),
                    "house_id": house_id,
                    "split": split,
                    "accepted": accepted,
                    "attempts": attempts,
                }
            ),
            flush=True,
        )
    if failed_houses:
        write_json(args.output_root / "failed_houses.json", failed_houses)
        raise RuntimeError(f"Could not generate all smoke samples for {len(failed_houses)} houses")

    manifest_path = args.output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    source_splits: dict[str, set[str]] = {}
    pair_splits: dict[tuple[str, str], set[str]] = {}
    for row in manifest_rows:
        source_splits.setdefault(str(row["source_sha256"]), set()).add(str(row["split"]))
        pair = (str(row["obs_sha256"]), str(row["target_sha256"]))
        pair_splits.setdefault(pair, set()).add(str(row["split"]))
    source_leakage = sum(len(splits) > 1 for splits in source_splits.values())
    pair_leakage = sum(len(splits) > 1 for splits in pair_splits.values())
    if source_leakage or pair_leakage:
        raise RuntimeError(
            f"Cross-split leakage: source={source_leakage}, processed_pair={pair_leakage}"
        )
    shape_array = np.asarray(exact_shape_sums, dtype=np.float64)
    summary = {
        "status": STATUS,
        "configuration": asdict(config),
        "source_manifest_sha256": sha256_file(args.house_manifest),
        "source_houses": len(source_rows),
        "source_render_failures": 0,
        "eligible_houses": len(eligible),
        "eligible_houses_by_split": split_houses,
        "eligible_fraction": len(eligible) / len(source_rows),
        "generated_samples_by_split": {
            split: len(raw_by_split[split][0]) for split in ("train", "val")
        },
        "rejected_generation_attempts": rejected_attempts,
        "building_overlap": 0,
        "cross_split_source_hash_overlap": source_leakage,
        "cross_split_processed_pair_overlap": pair_leakage,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows_logical_sha256": stable_rows_sha256(manifest_rows),
        "source_world_shape_sum_px": {
            "minimum": float(shape_array.min()),
            "median": float(np.median(shape_array)),
            "p90": float(np.quantile(shape_array, 0.9)),
            "p99": float(np.quantile(shape_array, 0.99)),
            "maximum": float(shape_array.max()),
        },
        "raw_480_metrics": {
            split: image_metrics(*raw_by_split[split]) for split in ("train", "val")
        },
        "model_256_metrics": {
            split: image_metrics(*model_by_split[split]) for split in ("train", "val")
        },
        "preserved_reference": {
            "status": "train_only_deterministic_1000_sample_audit",
            "raw_shape": "640x640",
            "raw_observation_channel_white_fraction_global": [
                0.01555701904296875,
                0.7289040576171875,
                0.25553892333984374,
            ],
            "raw_target_white_fraction_global": 0.09820650634765625,
            "model_256_observation_channel_white_fraction_global": [
                0.01554852294921875,
                0.728926513671875,
                0.25552496337890623,
            ],
            "model_256_target_white_fraction_global": 0.09820632934570313,
        },
        "external_test_source": "KTH is reserved and was not opened or evaluated",
        "test_outcomes_accessed": False,
    }
    write_json(args.output_root / "geometry_smoke_summary.json", summary)
    incomplete.unlink()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
