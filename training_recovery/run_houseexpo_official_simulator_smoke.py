#!/usr/bin/env python3
"""Run a train/validation-only smoke through the accessible old simulator.

The purpose is diagnostic: determine whether the visible 30 m, 16 px/m
simulator chain reproduces preserved train marginals.  It is not evidence that
the preserved archive used this exact source list, split, or checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from mso_recovery.common import sha256_file, stable_seed, write_json  # noqa: E402


STATUS = "accessible_official_simulator_smoke_not_historical_recovery"


@dataclass(frozen=True)
class DiagnosticConfig:
    pixels_per_meter: int = 16
    physical_window_m: int = 30
    raw_size: int = 480
    model_size: int = 256
    laser_range_m: int = 15
    actions_before_capture: int = 14
    source_minimum_world_shape_sum_px: int = 1500
    samples_per_house: int = 8
    maximum_attempts_per_house: int = 80
    generation_seed: int = 20260807
    minimum_observed_obstacle_fraction: float = 0.0125
    minimum_observed_free_fraction: float = 0.10
    maximum_free_to_obstacle_ratio: float = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--house-manifest", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--houseexpo-repo-root", type=Path, required=True)
    parser.add_argument("--simulator-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-house", type=int, default=8)
    parser.add_argument("--maximum-attempts-per-house", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def local_observation(simulator) -> tuple[np.ndarray, np.ndarray]:
    observation = simulator.dslamMap.copy()
    world_gt = simulator.world.copy()
    pose = simulator.get_pose()
    rot_y, rot_x = int(pose[0]), int(pose[1])
    rot_theta = -pose[2] * 180.0 / np.pi + 90.0
    pad_x = int(simulator.state_size[0] / 2.0 * 1.5)
    pad_y = int(simulator.state_size[1] / 2.0 * 1.5)
    state_size_x = int(simulator.state_size[0])
    state_size_y = int(simulator.state_size[1])
    if rot_y - pad_y < 0:
        observation = cv2.copyMakeBorder(
            observation,
            pad_y,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=simulator.map_color["uncertain"],
        )
        world_gt = cv2.copyMakeBorder(
            world_gt,
            pad_y,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=simulator.map_color["obstacle"],
        )
        rot_y += pad_y
    if rot_x - pad_x < 0:
        observation = cv2.copyMakeBorder(
            observation,
            0,
            0,
            pad_x,
            0,
            cv2.BORDER_CONSTANT,
            value=simulator.map_color["uncertain"],
        )
        world_gt = cv2.copyMakeBorder(
            world_gt,
            0,
            0,
            pad_x,
            0,
            cv2.BORDER_CONSTANT,
            value=simulator.map_color["obstacle"],
        )
        rot_x += pad_x
    if rot_y + pad_y > observation.shape[0]:
        observation = cv2.copyMakeBorder(
            observation,
            0,
            pad_y,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=simulator.map_color["uncertain"],
        )
        world_gt = cv2.copyMakeBorder(
            world_gt,
            0,
            pad_y,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=simulator.map_color["obstacle"],
        )
    if rot_x + pad_x > observation.shape[1]:
        observation = cv2.copyMakeBorder(
            observation,
            0,
            0,
            0,
            pad_x,
            cv2.BORDER_CONSTANT,
            value=simulator.map_color["uncertain"],
        )
        world_gt = cv2.copyMakeBorder(
            world_gt,
            0,
            0,
            0,
            pad_x,
            cv2.BORDER_CONSTANT,
            value=simulator.map_color["obstacle"],
        )
    local = observation[
        rot_y - pad_y : rot_y + pad_y, rot_x - pad_x : rot_x + pad_x
    ]
    matrix = cv2.getRotationMatrix2D((pad_y, pad_x), rot_theta, 1)
    local = cv2.warpAffine(
        local,
        matrix,
        (pad_y * 2, pad_x * 2),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=simulator.map_color["uncertain"],
    )
    local = local[
        pad_y - state_size_y // 2 : pad_y + state_size_y // 2,
        pad_x - state_size_x // 2 : pad_x + state_size_x // 2,
    ]
    real_world = world_gt[
        rot_y - pad_y : rot_y + pad_y, rot_x - pad_x : rot_x + pad_x
    ]
    target = cv2.warpAffine(
        real_world,
        matrix,
        (pad_y * 2, pad_x * 2),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=simulator.map_color["obstacle"],
    )
    target = target[
        pad_y - state_size_y // 2 : pad_y + state_size_y // 2,
        pad_x - state_size_x // 2 : pad_x + state_size_x // 2,
    ]
    return local, target


def exact_shape_sum(payload: dict, pixels_per_meter: int, padding: int = 10) -> int:
    vertices = (np.asarray(payload["verts"], dtype=np.float64) * pixels_per_meter).astype(
        np.int32
    )
    width = int(vertices[:, 0].max() - vertices[:, 0].min() + 2 + 2 * padding)
    height = int(vertices[:, 1].max() - vertices[:, 1].min() + 2 + 2 * padding)
    return height + width


def action_for_step(step: int) -> int:
    if step > 5 and step not in {7, 9, 11, 13}:
        return int(np.random.randint(2)) + 1
    return 0


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
        "global_mean": float(array.mean()),
    }


def main() -> None:
    args = parse_args()
    config = DiagnosticConfig(
        samples_per_house=args.samples_per_house,
        maximum_attempts_per_house=args.maximum_attempts_per_house,
        generation_seed=args.seed,
    )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "DIAGNOSTIC_INCOMPLETE").write_text(
        STATUS + "\n", encoding="utf-8"
    )
    sys.path.insert(0, str(args.houseexpo_repo_root))
    from pseudoslam.envs.simulator.pseudoSlam import pseudoSlam  # type: ignore  # noqa: PLC0415

    with args.house_manifest.open("r", encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    eligible: list[dict[str, str]] = []
    for row in source_rows:
        source_path = args.json_root / row["json_relpath"]
        if sha256_file(source_path) != row["json_sha256"]:
            raise ValueError(f"Source hash mismatch: {row['house_id']}")
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if exact_shape_sum(payload, config.pixels_per_meter) >= config.source_minimum_world_shape_sum_px:
            eligible.append(row)
    simulator = pseudoSlam(str(args.simulator_config))
    if int(simulator.m2p) != config.pixels_per_meter:
        raise ValueError(f"Unexpected simulator pixels/m: {simulator.m2p}")
    if tuple(int(value) for value in simulator.state_size) != (config.raw_size, config.raw_size):
        raise ValueError(f"Unexpected simulator state size: {simulator.state_size}")
    if int(simulator.laser_range) != config.laser_range_m * config.pixels_per_meter:
        raise ValueError(f"Unexpected simulator laser range: {simulator.laser_range}")

    manifest_rows: list[dict[str, object]] = []
    metrics = {
        "train": {"obs0": [], "obs1": [], "obs2": [], "target": []},
        "val": {"obs0": [], "obs1": [], "obs2": [], "target": []},
    }
    rejection_counts = {
        "observed_obstacle_below_minimum": 0,
        "observed_free_below_minimum": 0,
        "free_to_obstacle_ratio_above_maximum": 0,
        "early_exploration_completion": 0,
    }
    failed_houses: list[dict[str, object]] = []
    commands = ["forward", "left", "right"]
    for house_index, row in enumerate(eligible, start=1):
        house_id = row["house_id"]
        split = row["split"]
        accepted = 0
        attempts = 0
        while accepted < config.samples_per_house and attempts < config.maximum_attempts_per_house:
            attempts += 1
            episode_seed = stable_seed(config.generation_seed, house_id, attempts)
            np.random.seed(episode_seed)
            simulator.map_id_set = np.asarray([house_id], dtype=str)
            simulator.reset(order=True)
            early_done = False
            for step in range(1, config.actions_before_capture + 1):
                simulator.moveRobot(commands[action_for_step(step)])
                if simulator.measure_ratio() > 0.95:
                    early_done = True
                    break
            if early_done:
                rejection_counts["early_exploration_completion"] += 1
                continue
            raw_state, raw_world = local_observation(simulator)
            raw_observation = np.zeros((config.raw_size, config.raw_size, 3), dtype=np.uint8)
            raw_observation[:, :, 0] = (raw_state == simulator.map_color["obstacle"]).astype(np.uint8) * 255
            raw_observation[:, :, 1] = (raw_state == simulator.map_color["uncertain"]).astype(np.uint8) * 255
            raw_observation[:, :, 2] = (raw_state == simulator.map_color["free"]).astype(np.uint8) * 255
            if not np.all(np.sum(raw_observation == 255, axis=2) == 1):
                raise RuntimeError("Simulator observation did not map to one-hot channels")
            raw_target = (raw_world > 50).astype(np.uint8) * 255
            obstacle_fraction = float(np.mean(raw_observation[:, :, 0] == 255))
            unknown_fraction = float(np.mean(raw_observation[:, :, 1] == 255))
            free_fraction = float(np.mean(raw_observation[:, :, 2] == 255))
            if obstacle_fraction < config.minimum_observed_obstacle_fraction:
                rejection_counts["observed_obstacle_below_minimum"] += 1
                continue
            if free_fraction < config.minimum_observed_free_fraction:
                rejection_counts["observed_free_below_minimum"] += 1
                continue
            if free_fraction > obstacle_fraction * config.maximum_free_to_obstacle_ratio:
                rejection_counts["free_to_obstacle_ratio_above_maximum"] += 1
                continue
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
                    "floorplan_id": house_id,
                    "source_relpath": row["json_relpath"],
                    "source_sha256": row["json_sha256"],
                    "episode_seed": episode_seed,
                    "obs_relpath": obs_path.relative_to(args.output_root).as_posix(),
                    "target_relpath": target_path.relative_to(args.output_root).as_posix(),
                    "obs_sha256": sha256_file(obs_path),
                    "target_sha256": sha256_file(target_path),
                    "raw_obs_relpath": raw_obs_path.relative_to(args.output_root).as_posix(),
                    "raw_target_relpath": raw_target_path.relative_to(args.output_root).as_posix(),
                    "raw_obs_sha256": sha256_file(raw_obs_path),
                    "raw_target_sha256": sha256_file(raw_target_path),
                    "raw_obstacle_fraction": f"{obstacle_fraction:.9f}",
                    "raw_unknown_fraction": f"{unknown_fraction:.9f}",
                    "raw_free_fraction": f"{free_fraction:.9f}",
                    "raw_target_fraction": f"{float(np.mean(raw_target == 255)):.9f}",
                }
            )
            metrics[split]["obs0"].append(obstacle_fraction)
            metrics[split]["obs1"].append(unknown_fraction)
            metrics[split]["obs2"].append(free_fraction)
            metrics[split]["target"].append(float(np.mean(raw_target == 255)))
            accepted += 1
        if accepted < config.samples_per_house:
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
    manifest_path = args.output_root / "manifest.csv"
    if not manifest_rows:
        raise RuntimeError("Official simulator smoke accepted no samples")
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "status": STATUS,
        "configuration": config.__dict__,
        "simulator_config_sha256": sha256_file(args.simulator_config),
        "source_manifest_sha256": sha256_file(args.house_manifest),
        "eligible_houses": len(eligible),
        "eligible_houses_by_split": {
            split: sum(row["split"] == split for row in eligible)
            for split in ("train", "val")
        },
        "generated_samples_by_split": {
            split: len(metrics[split]["target"]) for split in ("train", "val")
        },
        "failed_houses": failed_houses,
        "rejection_counts": rejection_counts,
        "raw_480_metrics": {
            split: {
                "observation_channel_white_fraction": [
                    describe(metrics[split]["obs0"]),
                    describe(metrics[split]["obs1"]),
                    describe(metrics[split]["obs2"]),
                ],
                "target_white_fraction": describe(metrics[split]["target"]),
            }
            for split in ("train", "val")
            if metrics[split]["target"]
        },
        "manifest_sha256": sha256_file(manifest_path),
        "preserved_train_only_reference": {
            "raw_shape": "640x640",
            "observation_channel_white_fraction_global": [
                0.01555701904296875,
                0.7289040576171875,
                0.25553892333984374,
            ],
            "target_white_fraction_global": 0.09820650634765625,
        },
        "interpretation_boundary": "marginal agreement cannot prove this is the historical generator; raw size remains 480 here versus 640 in the preserved archive",
        "test_outcomes_accessed": False,
    }
    write_json(args.output_root / "official_simulator_smoke_summary.json", summary)
    (args.output_root / "DIAGNOSTIC_INCOMPLETE").unlink()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
