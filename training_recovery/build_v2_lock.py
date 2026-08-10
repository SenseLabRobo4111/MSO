#!/usr/bin/env python3
"""Build the immutable prospective-v2 lock without decoding source images."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL
import skimage
import torch

SOURCE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_ROOT))

from mso_recovery.v2_lock import (  # noqa: E402
    LOCK_DIGEST_FILENAME,
    LOCK_FILENAME,
    LOCK_SCHEMA,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)


SOURCE_FILES = (
    "build_v2_lock.py",
    "generate_v2_locked_dataset.py",
    "run_v2_locked_campaign.py",
    "validate_v2_dataset.py",
    "audit_v2_gates.py",
    "evaluate_v2_positive_control.py",
    "train_v2_locked.py",
    "summarize_v2_multiseed.py",
    "mso_recovery/__init__.py",
    "mso_recovery/common.py",
    "mso_recovery/objective.py",
    "mso_recovery/v2_data.py",
    "mso_recovery/v2_geometry.py",
    "mso_recovery/v2_lock.py",
    "sensemap/__init__.py",
    "sensemap/explore_model/__init__.py",
    "sensemap/explore_model/ffc.py",
    "sensemap/explore_model/SenseMapNet.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--lock-root", type=Path, required=True)
    return parser.parse_args()


def operational_group(floorplan_id: str) -> str:
    match = re.match(r"^(\d+)(?:_|$)", floorplan_id)
    return match.group(1) if match else f"named:{floorplan_id}"


def assign_operational_splits(rows: list[dict[str, str]]) -> dict[str, str]:
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row["operational_group_id"], []).append(row["floorplan_id"])
    group_ids = np.asarray(sorted(groups), dtype=object)
    np.random.default_rng(20260807).shuffle(group_ids)

    def exact_subset(candidates: list[str], target_floorplans: int) -> set[str]:
        paths: dict[int, tuple[str, ...]] = {0: ()}
        for group_id in candidates:
            size = len(groups[group_id])
            for total, path in sorted(paths.items(), reverse=True):
                new_total = total + size
                if new_total <= target_floorplans and new_total not in paths:
                    paths[new_total] = path + (group_id,)
        if target_floorplans not in paths:
            raise RuntimeError(f"Cannot form atomic split of {target_floorplans} floorplans")
        return set(paths[target_floorplans])

    ordered = [str(value) for value in group_ids]
    validation_groups = exact_subset(ordered, 23)
    remaining = [value for value in ordered if value not in validation_groups]
    test_groups = exact_subset(remaining, 23)
    assignments: dict[str, str] = {}
    for group_id, floorplans in groups.items():
        split = (
            "val"
            if group_id in validation_groups
            else "test"
            if group_id in test_groups
            else "train"
        )
        for floorplan_id in floorplans:
            assignments[floorplan_id] = split
    return assignments


def inventory_sources(source_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for gt_path in sorted(source_root.glob("*/GT.bmp")):
        floorplan_id = gt_path.parent.name
        metadata_path = gt_path.with_name("GT.json")
        metadata_sha = sha256_file(metadata_path) if metadata_path.is_file() else ""
        resolution = "0.1"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            resolution = str(float(metadata.get("resolution", 0.1)))
        rows.append(
            {
                "floorplan_id": floorplan_id,
                "operational_group_id": operational_group(floorplan_id),
                "operational_group_rule": "leading_numeric_prefix_before_underscore_else_named_singleton",
                "semantic_building_identity_verified": "false",
                "gt_relpath": gt_path.relative_to(source_root).as_posix(),
                "gt_sha256": sha256_file(gt_path),
                "metadata_relpath": (
                    metadata_path.relative_to(source_root).as_posix()
                    if metadata_path.is_file()
                    else ""
                ),
                "metadata_sha256": metadata_sha,
                "declared_resolution_m_per_cell": resolution,
                "image_decoded_during_inventory": "false",
            }
        )
    if len(rows) != 156:
        raise RuntimeError(f"Expected 156 floorplans, found {len(rows)}")
    if len({row["floorplan_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate floorplan IDs")
    if len({row["operational_group_id"] for row in rows}) != 146:
        raise RuntimeError("Expected 146 operational groups")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment_snapshot() -> dict:
    distributions = sorted(
        (
            distribution.metadata.get("Name", "unknown"),
            distribution.version,
        )
        for distribution in importlib.metadata.distributions()
    )
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "count": torch.cuda.device_count(),
        }
    try:
        nvidia_smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        nvidia_smi = None
    return {
        "schema": "mso.prospective_v2.environment/1",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "nvidia_smi": nvidia_smi,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "skimage": skimage.__version__,
        "cublas_workspace_config_required": ":4096:8",
        "pythonhashseed_required": "0",
        "installed_distributions": distributions,
    }


def main() -> None:
    args = parse_args()
    lock_root = args.lock_root.resolve()
    if (lock_root / LOCK_FILENAME).exists() or (lock_root / LOCK_DIGEST_FILENAME).exists():
        raise FileExistsError("Refusing to replace an existing v2 lock")
    source_dir = lock_root / "source"
    input_dir = lock_root / "input"
    if not source_dir.is_dir() or not input_dir.is_dir():
        raise FileNotFoundError("Lock root must already contain source/ and input/")

    inventory = inventory_sources(args.source_root.resolve())
    assignments = assign_operational_splits(inventory)
    split_rows = [
        {
            "floorplan_id": row["floorplan_id"],
            "operational_group_id": row["operational_group_id"],
            "group_semantics": "operational_only_not_verified_building_identity",
            "split": assignments[row["floorplan_id"]],
            "gt_sha256": row["gt_sha256"],
        }
        for row in inventory
    ]
    group_splits: dict[str, set[str]] = {}
    for row in split_rows:
        group_splits.setdefault(row["operational_group_id"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise RuntimeError("Operational group leakage")
    floorplan_counts = {
        split: sum(row["split"] == split for row in split_rows)
        for split in ("train", "val", "test")
    }
    group_counts = {
        split: len(
            {
                row["operational_group_id"]
                for row in split_rows
                if row["split"] == split
            }
        )
        for split in ("train", "val", "test")
    }
    if floorplan_counts != {"train": 110, "val": 23, "test": 23}:
        raise RuntimeError(f"Unexpected floorplan counts: {floorplan_counts}")
    if group_counts != {"train": 102, "val": 22, "test": 22}:
        raise RuntimeError(f"Unexpected operational-group counts: {group_counts}")

    inventory_path = lock_root / "source_inventory.csv"
    split_path = lock_root / "source_split.csv"
    environment_path = lock_root / "environment.json"
    write_csv(inventory_path, inventory)
    write_csv(split_path, split_rows)
    atomic_write_json(environment_path, environment_snapshot())

    source_records = []
    for relative in SOURCE_FILES:
        path = source_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked source file: {relative}")
        source_records.append(
            {"path": f"source/{relative}", "sha256": sha256_file(path)}
        )
    positive_path = input_dir / "recovered_candidate_deconv_a.pt"
    positive_sha = sha256_file(positive_path)
    if positive_sha != "da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8":
        raise RuntimeError("Positive-control A hash mismatch")

    geometry = {
        "source_image_mode": "Pillow_L_8bit",
        "source_occupied_rule": "gray_less_than_128",
        "content_bbox_locator": "minority_of_dark_mask_only_for_bbox_target_polarity_unchanged",
        "content_padding_source_cells": 8,
        "content_bbox_indexing": "inclusive_min_inclusive_max_plus_one_clipped_to_source",
        "minimum_resized_axis_cells": 8,
        "source_resolution_origin": "per_source_GT_json_resolution_else_0.1_m_per_cell",
        "source_to_working_interpolation": "opencv_INTER_NEAREST",
        "working_pixels_per_meter": 16,
        "working_resolution_m_per_cell": 0.0625,
        "resized_dimension_rule": "max_8_int_python_round_ties_to_even(source_cells_times_source_resolution_div_0.0625)",
        "boundary_rule": "outermost_row_and_column_occupied",
        "source_obstacle_dilation": {"kernel": [3, 3], "iterations": 3},
        "outer_padding": {
            "cells_each_side": 728,
            "formula": "raw_size_480_plus_sensor_range_240_plus_8",
            "mode": "constant",
            "value": "occupied",
        },
        "component_connectivity": 8,
        "component_minimum_area_cells": 64,
        "component_score": "area/(1+0.02*centroid_distance_to_array_center)",
        "component_score_tie_break": "higher_opencv_component_label",
        "distance_transform": {"type": "opencv_DIST_L2", "mask_size": 5},
        "start_clearance_cells": 2.0,
        "start_bbox_fraction": {"minimum": 0.10, "maximum_exclusive": 0.80},
        "start_fallback": "component_cells_inside_same_bbox_if_clearance_set_empty_else_error",
        "start_sampling": "uniform_over_candidate_cells",
        "initial_heading_radians": "numpy_uniform_half_open_[0,2*pi)",
        "field_m": [30, 30],
        "raw_shape": [480, 480],
        "oriented_patch_rule": "ceil(raw_size*sqrt(2))+4_then_next_even",
        "rotation_degrees": "-heading_radians*180/pi+90",
        "rotation_center": "integer_patch_half_half",
        "rotation_scale": 1.0,
        "rotation_interpolation": "opencv_INTER_NEAREST",
        "target_rotation_border": "occupied",
        "known_rotation_border": "unknown",
        "post_rotation_crop": "center_start_floor((patch_size-raw_size)/2)_half_open_480",
        "sensor": {
            "range_m": 15,
            "range_cells": 240,
            "rays": 360,
            "fov_degrees": 360,
            "angle_endpoint": False,
            "radial_cells": "integers_1_through_240_inclusive",
            "coordinate_rounding": "numpy_rint_ties_to_even",
            "duplicate_ray_cells": "remove_consecutive_duplicates",
            "first_hit": "inclusive",
            "pose_center_known_patch": [3, 3],
        },
        "trajectory": {
            "initial_scan": True,
            "actions_before_capture": 14,
            "forward_steps": [1, 2, 3, 4, 5, 7, 9, 11, 13],
            "turn_steps": [6, 8, 10, 12, 14],
            "turn_choice": "equiprobable_plus_or_minus_30_degrees_from_numpy_integers_1_3",
            "forward_m": 1.0,
            "forward_cells": 16,
            "turn_degrees": 30,
            "collision_rule": "advance_cellwise_until_next_cell_not_in_component",
            "scan_timing": "after_initial_pose_and_after_each_of_14_actions",
        },
        "known_mask_dilation": {"kernel": [3, 3], "iterations": 1},
        "observation_rgb_channel_order": ["occupied", "unknown", "free"],
        "observation_values": {"negative": 0, "positive": 255},
        "target_white_semantics": "occupied",
        "save_gate": {
            "minimum_observed_obstacle_fraction": 0.0125,
            "minimum_observed_free_fraction": 0.10,
            "maximum_free_to_obstacle_ratio": 25.0,
            "lower_bound_comparison": "reject_if_strictly_less",
            "ratio_comparison": "reject_if_free_strictly_greater_than_25_times_obstacle",
        },
        "attempt_policy": "attempts_1_through_limit_accept_first_N_passing_in_attempt_order",
        "model_shape": [256, 256],
        "raw_to_model_interpolation": "opencv_INTER_NEAREST",
        "image_encoding": "Pillow_PNG_default",
        "sample_seed": "sha256(protocol_seed,floorplan_id,attempt)_first_8_bytes_mod_2^32",
        "protocol_seed": 20260807,
    }
    training = {
        "batch_size": 16,
        "num_workers": 8,
        "maximum_epochs": 500,
        "minimum_epochs_before_stopping": 50,
        "early_stopping_patience": 30,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "adam_betas": [0.5, 0.999],
        "adam_epsilon": 1e-08,
        "adam_amsgrad": False,
        "adam_foreach": False,
        "adam_fused": False,
        "adam_capturable": False,
        "adam_differentiable": False,
        "weight_decay": 0.0,
        "precision": "float32",
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
            "pythonhashseed": "0",
            "train_shuffle": True,
            "train_shuffle_seed_each_epoch": "job_seed_plus_one_based_epoch",
            "validation_shuffle": False,
            "worker_seed": "torch_initial_seed_mod_2^32",
        },
        "models": {
            "teacher": {
                "class": "sensemap.explore_model.SenseMapNet.TeacherMapNet2",
                "image_size": 256,
                "dim": 4,
            },
            "student": {
                "class": "sensemap.explore_model.SenseMapNet.DistillMapNetDeconv",
                "image_size": 256,
                "dim": 4,
                "parameter_count": 342771,
            },
        },
        "teacher_seed": 101,
        "student_seeds": [11, 23, 37, 53, 71],
        "loss_weights": {
            "unknown_bce": 1.0,
            "unknown_soft_dice": 0.5,
            "full_bce": 0.25,
            "feature_distillation": 0.25,
        },
        "checkpoint_selection": {
            "metric": "validation_unknown_bce",
            "direction": "minimize",
            "tie_break": "earliest_epoch",
            "teacher_checkpoint_for_distillation": "selected_full_teacher_best.pt_SHA_bound",
            "test_used": False,
        },
        "smoke_teacher": {"seed": 101, "maximum_epochs": 30},
    }
    lock = {
        "schema": LOCK_SCHEMA,
        "status": "prospective_v2_unrun_not_historical_recovery",
        "historical_claim": False,
        "source_root": str(args.source_root.resolve()),
        "grouping": {
            "kind": "operational_prefix_group",
            "semantic_building_identity_verified": False,
            "claim_allowed": "operational-group-disjoint only",
            "floorplan_counts": floorplan_counts,
            "operational_group_counts": group_counts,
            "split_seed": 20260807,
        },
        "artifacts": {
            "source_inventory": {
                "path": inventory_path.name,
                "sha256": sha256_file(inventory_path),
            },
            "source_split": {
                "path": split_path.name,
                "sha256": sha256_file(split_path),
            },
            "environment": {
                "path": environment_path.name,
                "sha256": sha256_file(environment_path),
            },
            "source_files": source_records,
        },
        "positive_control": {
            "path": "input/recovered_candidate_deconv_a.pt",
            "sha256": positive_sha,
            "logical_name": "recovered_candidate_deconv_a",
            "role": "validation_only_positive_control_not_manuscript_checkpoint",
            "parameter_count": 342771,
        },
        "geometry": geometry,
        "datasets": {
            "smoke": {
                "samples_per_train_or_validation_floorplan": 4,
                "maximum_attempts_per_floorplan": 160,
                "test_samples": 0,
            },
            "full": {
                "samples_per_train_or_validation_floorplan": 48,
                "maximum_attempts_per_floorplan": 1920,
                "test_samples": 0,
            },
        },
        "go_no_go": {
            "integrity": {
                "one_hot_fraction": 1.0,
                "known_target_agreement": 1.0,
                "cross_split_group_leakage": 0,
                "cross_split_source_hash_leakage": 0,
                "cross_split_processed_pair_leakage": 0,
            },
            "coverage": {"train_floorplan_fraction": 0.90, "val_floorplan_fraction": 0.90},
            "marginals": {
                "reference_observation_channels": [0.01554852294921875, 0.728926513671875, 0.25552496337890623],
                "maximum_absolute_channel_delta": 0.05,
                "reference_target": 0.09820632934570313,
                "maximum_absolute_target_delta": 0.05,
                "maximum_train_val_target_delta": 0.03,
            },
            "positive_control": {"minimum_unknown_f1": 0.45, "minimum_unknown_iou": 0.30},
            "learnability": {
                "minimum_validation_unknown_bce_relative_reduction_from_epoch_1": 0.05,
                "minimum_validation_unknown_f1": 0.05,
                "maximum_epochs": 30,
                "threshold": 0.5,
            },
        },
        "training": training,
        "statistics": {
            "primary_experimental_unit": "student_seed",
            "number_of_student_seeds": 5,
            "summary": ["mean", "sample_standard_deviation_ddof_1", "two_sided_t_95_percent_CI_df_4", "median", "minimum", "maximum"],
            "t_critical_df_4": 2.7764451051977987,
            "spatial_cluster_unit": "operational_prefix_group_not_semantically_verified_building",
            "seed_exclusion_allowed": False,
            "best_seed_reporting_allowed": False,
        },
        "checkpoint_binding_required_fields": [
            "v2_lock_sha256",
            "dataset_manifest_sha256",
            "source_split_sha256",
            "training_configuration_sha256",
            "stage",
            "seed",
            "epoch",
            "selected_validation_metric",
            "parent_teacher_checkpoint_sha256",
        ],
        "execution_order": [
            "preflight_lock_environment_and_source_hashes",
            "generate_smoke_train_val_only",
            "validate_smoke",
            "evaluate_positive_control_A_on_validation_only",
            "train_smoke_teacher_seed_101_train_val_only",
            "atomic_five_gate_audit",
            "stop_on_any_gate_failure",
            "generate_full_train_val_only",
            "validate_full",
            "train_and_validation_select_teacher_seed_101",
            "train_and_validation_select_students_11_23_37_53_71",
            "summarize_all_five_student_seeds_without_exclusion",
            "stop_with_test_still_disabled",
        ],
        "test_policy": {
            "decode_allowed": False,
            "generated_samples_allowed": 0,
            "evaluation_entrypoint": None,
        },
        "configuration_sha256": canonical_json_sha256(
            {"geometry": geometry, "training": training}
        ),
    }
    lock_path = lock_root / LOCK_FILENAME
    atomic_write_json(lock_path, lock)
    digest = sha256_file(lock_path)
    (lock_root / LOCK_DIGEST_FILENAME).write_text(
        f"{digest}  {LOCK_FILENAME}\n", encoding="ascii"
    )
    os.chmod(lock_path, 0o444)
    os.chmod(lock_root / LOCK_DIGEST_FILENAME, 0o444)
    print(json.dumps({"lock_sha256": digest, "floorplans": floorplan_counts, "groups": group_counts}, indent=2))


if __name__ == "__main__":
    main()
