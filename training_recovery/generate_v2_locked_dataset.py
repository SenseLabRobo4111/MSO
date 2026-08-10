#!/usr/bin/env python3
"""Generate the locked v2 train/validation dataset; test is never decoded."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mso_recovery.v2_geometry import (  # noqa: E402
    SmokeConfig,
    describe,
    generate_sample,
    load_source_occupied,
    ray_offsets,
    restricted_starts,
    select_navigable_component,
)
from mso_recovery.common import stable_seed  # noqa: E402
from mso_recovery.v2_lock import (  # noqa: E402
    atomic_write_json,
    load_lock,
    read_csv_artifact,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-kind", choices=("smoke", "full"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock, lock_sha = load_lock(args.lock_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    incomplete = args.output_root / "GENERATION_INCOMPLETE"
    incomplete.write_text(lock_sha + "\n", encoding="ascii")
    inventory = {
        row["floorplan_id"]: row
        for row in read_csv_artifact(
            args.lock_root, lock["artifacts"]["source_inventory"]
        )
    }
    split_rows = read_csv_artifact(
        args.lock_root, lock["artifacts"]["source_split"]
    )
    dataset_config = lock["datasets"][args.dataset_kind]
    samples_per_floorplan = int(
        dataset_config["samples_per_train_or_validation_floorplan"]
    )
    maximum_attempts = int(dataset_config["maximum_attempts_per_floorplan"])
    geometry = lock["geometry"]
    config = SmokeConfig(
        pixels_per_meter=int(geometry["working_pixels_per_meter"]),
        working_resolution_m_per_cell=float(
            geometry["working_resolution_m_per_cell"]
        ),
        physical_window_m=int(geometry["field_m"][0]),
        raw_size=int(geometry["raw_shape"][0]),
        model_size=int(geometry["model_shape"][0]),
        sensor_range_m=int(geometry["sensor"]["range_m"]),
        laser_range_cells=int(geometry["sensor"]["range_cells"]),
        laser_rays=int(geometry["sensor"]["rays"]),
        actions_before_capture=int(geometry["trajectory"]["actions_before_capture"]),
        forward_step_m=float(geometry["trajectory"]["forward_m"]),
        forward_step_cells=int(geometry["trajectory"]["forward_cells"]),
        turn_degrees=int(geometry["trajectory"]["turn_degrees"]),
        source_obstacle_dilation_iterations=int(
            geometry["source_obstacle_dilation"]["iterations"]
        ),
        observation_known_dilation_iterations=int(
            geometry["known_mask_dilation"]["iterations"]
        ),
        content_padding_source_cells=int(geometry["content_padding_source_cells"]),
        samples_per_floorplan=samples_per_floorplan,
        maximum_attempts_per_floorplan=maximum_attempts,
        split_seed=int(lock["grouping"]["split_seed"]),
        generation_seed=int(geometry["protocol_seed"]),
        minimum_observed_obstacle_fraction=float(
            geometry["save_gate"]["minimum_observed_obstacle_fraction"]
        ),
        minimum_observed_free_fraction=float(
            geometry["save_gate"]["minimum_observed_free_fraction"]
        ),
        maximum_free_to_obstacle_ratio=float(
            geometry["save_gate"]["maximum_free_to_obstacle_ratio"]
        ),
    )
    if config.raw_size != config.physical_window_m * config.pixels_per_meter:
        raise ValueError("Locked field size and working resolution disagree")
    if config.laser_range_cells != config.sensor_range_m * config.pixels_per_meter:
        raise ValueError("Locked sensor range and working resolution disagree")
    if geometry["raw_shape"] != [config.raw_size, config.raw_size]:
        raise ValueError("Locked raw field must be square")
    if geometry["model_shape"] != [config.model_size, config.model_size]:
        raise ValueError("Locked model field must be square")
    implementation_contract = {
        "source_image_mode": "Pillow_L_8bit",
        "source_occupied_rule": "gray_less_than_128",
        "source_to_working_interpolation": "opencv_INTER_NEAREST",
        "boundary_rule": "outermost_row_and_column_occupied",
        "component_connectivity": 8,
        "component_minimum_area_cells": 64,
        "start_clearance_cells": 2.0,
        "rotation_interpolation": "opencv_INTER_NEAREST",
        "target_rotation_border": "occupied",
        "known_rotation_border": "unknown",
        "observation_rgb_channel_order": ["occupied", "unknown", "free"],
        "target_white_semantics": "occupied",
        "raw_to_model_interpolation": "opencv_INTER_NEAREST",
        "image_encoding": "Pillow_PNG_default",
    }
    for key, expected in implementation_contract.items():
        if geometry.get(key) != expected:
            raise ValueError(f"Locked geometry is unsupported by this implementation: {key}")
    rays = ray_offsets(config.laser_rays, config.laser_range_cells)
    manifest_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    metrics = {
        "train": {"obs0": [], "obs1": [], "obs2": [], "target": []},
        "val": {"obs0": [], "obs1": [], "obs2": [], "target": []},
    }
    # This filter is the decode prohibition: test rows are never passed to
    # Pillow, OpenCV, or any generation helper.
    generation_rows = [row for row in split_rows if row["split"] in {"train", "val"}]
    if any(row["split"] == "test" for row in generation_rows):
        raise AssertionError("Internal test decode guard failed")
    for index, split_row in enumerate(generation_rows, start=1):
        source = inventory[split_row["floorplan_id"]]
        record = {
            "floorplan_id": source["floorplan_id"],
            "operational_group_id": source["operational_group_id"],
            "safe_id": source["floorplan_id"],
            "gt_path": Path(lock["source_root"]) / source["gt_relpath"],
            "source_relpath": source["gt_relpath"],
            "source_sha256": source["gt_sha256"],
            "resolution_m_per_cell": float(source["declared_resolution_m_per_cell"]),
        }
        if sha256_file(Path(record["gt_path"])) != record["source_sha256"]:
            raise ValueError(f"Locked source changed: {record['floorplan_id']}")
        occupied = load_source_occupied(record, config)
        component, _ = select_navigable_component(occupied)
        starts = restricted_starts(component)
        accepted = 0
        attempts = 0
        while accepted < samples_per_floorplan and attempts < maximum_attempts:
            attempts += 1
            sample_seed = stable_seed(
                config.generation_seed, record["floorplan_id"], attempts
            )
            generated = generate_sample(
                occupied,
                component,
                starts,
                rays,
                np.random.default_rng(sample_seed),
                config,
            )
            if generated is None:
                continue
            raw_observation, raw_target, details = generated
            model_observation = cv2.resize(
                raw_observation, (256, 256), interpolation=cv2.INTER_NEAREST
            )
            model_target = cv2.resize(
                raw_target, (256, 256), interpolation=cv2.INTER_NEAREST
            )
            sample_id = f"{record['safe_id']}__{accepted:03d}"
            sample_dir = args.output_root / "samples" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=False)
            raw_obs = sample_dir / "raw_obs_480.png"
            raw_target_path = sample_dir / "raw_target_480.png"
            obs_path = sample_dir / "local_obs_0.png"
            target_path = sample_dir / "local_map_0.png"
            Image.fromarray(raw_observation, mode="RGB").save(raw_obs)
            Image.fromarray(raw_target, mode="L").save(raw_target_path)
            Image.fromarray(model_observation, mode="RGB").save(obs_path)
            Image.fromarray(model_target, mode="L").save(target_path)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split_row["split"],
                    "group_key": split_row["operational_group_id"],
                    "group_kind": "operational_prefix_group",
                    "floorplan_id": record["floorplan_id"],
                    "source_sha256": record["source_sha256"],
                    "generation_seed": sample_seed,
                    "obs_relpath": obs_path.relative_to(args.output_root).as_posix(),
                    "target_relpath": target_path.relative_to(args.output_root).as_posix(),
                    "obs_sha256": sha256_file(obs_path),
                    "target_sha256": sha256_file(target_path),
                    "raw_obs_relpath": raw_obs.relative_to(args.output_root).as_posix(),
                    "raw_target_relpath": raw_target_path.relative_to(args.output_root).as_posix(),
                    "raw_obs_sha256": sha256_file(raw_obs),
                    "raw_target_sha256": sha256_file(raw_target_path),
                    "raw_obstacle_fraction": f"{details['obstacle_fraction']:.9f}",
                    "raw_unknown_fraction": f"{details['unknown_fraction']:.9f}",
                    "raw_free_fraction": f"{details['free_fraction']:.9f}",
                    "raw_target_fraction": f"{details['target_fraction']:.9f}",
                }
            )
            split = split_row["split"]
            for channel, key in enumerate(("obs0", "obs1", "obs2")):
                metrics[split][key].append(
                    float(np.mean(model_observation[:, :, channel] == 255))
                )
            metrics[split]["target"].append(float(np.mean(model_target == 255)))
            accepted += 1
        if accepted != samples_per_floorplan:
            failures.append(
                {
                    "floorplan_id": record["floorplan_id"],
                    "split": split_row["split"],
                    "accepted": accepted,
                    "attempts": attempts,
                }
            )
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(generation_rows),
                    "floorplan_id": record["floorplan_id"],
                    "accepted": accepted,
                    "attempts": attempts,
                }
            ),
            flush=True,
        )
    if not manifest_rows:
        raise RuntimeError("Locked v2 generation accepted no samples")
    manifest_path = args.output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "schema": "mso.prospective_v2.dataset/1",
        "status": "prospective_v2_unrun_not_historical_recovery",
        "dataset_kind": args.dataset_kind,
        "v2_lock_sha256": lock_sha,
        "source_split_sha256": lock["artifacts"]["source_split"]["sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "samples_per_floorplan_requested": samples_per_floorplan,
        "maximum_attempts_per_floorplan": maximum_attempts,
        "failed_floorplans": failures,
        "generated_samples": {
            split: len(metrics[split]["target"]) for split in ("train", "val")
        },
        "model_256_metrics": {
            split: {
                "observation_channel_white_fraction": [
                    describe(metrics[split]["obs0"]),
                    describe(metrics[split]["obs1"]),
                    describe(metrics[split]["obs2"]),
                ],
                "target_white_fraction": describe(metrics[split]["target"]),
            }
            for split in ("train", "val")
        },
        "test_source_rows_decoded": 0,
        "test_samples_generated": 0,
    }
    atomic_write_json(args.output_root / "dataset_summary.json", summary)
    incomplete.unlink()


if __name__ == "__main__":
    main()
