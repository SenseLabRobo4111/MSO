#!/usr/bin/env python3
"""Fail-closed validator for the Tongfang 27F four-robot campaign."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

try:
    import yaml
except ImportError as error:  # pragma: no cover - exercised only without dependency
    raise SystemExit("PyYAML is required") from error


PREDICTOR_MODES = {"mso_342771", "observed_only"}
NETWORK_CONDITIONS = {"nominal", "isolated_robot_impairment"}
LABEL_TO_CELL = {
    "A": ("mso_342771", "nominal"),
    "B": ("observed_only", "nominal"),
    "C": ("mso_342771", "isolated_robot_impairment"),
    "D": ("observed_only", "isolated_robot_impairment"),
}
ALLOWED_SEQUENCES = {
    ("A", "B", "D", "C"),
    ("B", "C", "A", "D"),
    ("C", "D", "B", "A"),
    ("D", "A", "C", "B"),
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
RUN_ID = re.compile(r"^tf27_n4_b(0[1-8])_([ABCD])$")
COLLECTION_ENABLEMENT_IMPLEMENTED = False
EXPECTED_LOCKS = {
    "team": "config/team_4.yaml",
    "arena": "config/arena.yaml",
    "start_poses": "config/start_poses.yaml",
    "model": "config/model_lock.yaml",
    "online_stack": "config/online_stack_lock.yaml",
    "network": "config/network_lock.yaml",
    "pilot_plan": "config/pilot_plan.yaml",
    "reference": "config/reference.yaml",
}
EXPECTED_NETWORK_PHASES = [
    {
        "phase_id": "baseline", "start_s": 180, "end_s": 210,
        "loss_probability": 0.0,
    },
    {
        "phase_id": "degraded", "start_s": 210, "end_s": 255,
        "loss_probability": 0.2,
    },
    {
        "phase_id": "disconnected", "start_s": 255, "end_s": 275,
        "loss_probability": 1.0,
    },
    {
        "phase_id": "recovery_monitor", "start_s": 275, "end_s": 320,
        "loss_probability": 0.0,
    },
]
SELECTED_MODEL_ARTIFACT_SHA256 = (
    "da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8"
)
MODEL_ID = "mso_deconv_342771_candidate_a"
MODEL_LOADER_ID = "distill_map_net_deconv_raw_state_v1"
MODEL_ARCHITECTURE = (
    "sensemap.explore_model.SenseMapNet."
    "DistillMapNetDeconv(image_size=256, dim=4)"
)
MODEL_PARAMETERS = 342771
REFERENCE_RUNTIME = {
    "python": "3.12.6",
    "torch": "2.12.0+cpu",
    "device": "cpu",
    "output_fingerprint_scope": "exact_reference_environment",
    "deployment_requires_same_runtime_or_reviewed_equivalence": True,
}
CAMPAIGN_ID = "tf27_n4_mso_342771_factorial_v1"
PILOT_PLAN_ID = "tf27_n4_mso_342771_integration_pilots_v1"
MODEL_BINDING = {
    "model_id": MODEL_ID,
    "artifact_sha256": SELECTED_MODEL_ARTIFACT_SHA256,
    "loader_id": MODEL_LOADER_ID,
    "input_contract": "occupied_unknown_free_one_hot_256_v1",
    "output_contract": "sigmoid_occupancy_probability_256_v1",
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk(value: Any, prefix: str = ""):
    """Yield dotted paths and scalar leaves from a nested value."""
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from walk(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            yield from walk(item, child)
    else:
        yield prefix, value


def placeholder_paths(value: Any) -> list[str]:
    """Return all unresolved `SET_...` scalar locations."""
    return [
        path for path, item in walk(value)
        if isinstance(item, str) and item.startswith("SET_")
    ]


def forbidden_key_paths(value: Any, key_name: str, prefix: str = "") -> list[str]:
    """Find a forbidden mapping key recursively."""
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            if key == key_name:
                paths.append(child)
            paths.extend(forbidden_key_paths(item, key_name, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(forbidden_key_paths(
                item, key_name, f"{prefix}[{index}]"))
    return paths


def _error(errors: list[dict[str, Any]], check: str, detail: Any) -> None:
    errors.append({"check": check, "detail": detail})


def validate_pilot_plan(
    pilot: dict[str, Any], confirmatory_run_ids: set[str]
) -> list[dict[str, Any]]:
    """Validate four fixed pilots that can never enter confirmatory analysis."""
    errors: list[dict[str, Any]] = []
    if pilot.get("pilot_plan_id") != PILOT_PLAN_ID:
        _error(errors, "pilot plan identity", pilot.get("pilot_plan_id"))
    if pilot.get("permanently_excluded_from_confirmatory_analysis") is not True:
        _error(errors, "pilot exclusion lock", None)
    result_status = pilot.get("results_status")
    completion_status = pilot.get("pilot_completion_status")
    if result_status not in {"not_collected", "completed"}:
        _error(errors, "pilot result status", pilot.get("results_status"))
    if completion_status == "not_run" and result_status != "not_collected":
        _error(errors, "unrun pilot status consistency", result_status)
    if completion_status == "verified" and result_status != "completed":
        _error(errors, "completed pilot status consistency", result_status)
    if pilot.get("claim_authorized") is not False:
        _error(errors, "pilot claim authorization", pilot.get("claim_authorized"))
    rows = pilot.get("pilot_runs") or []
    if len(rows) != 4:
        _error(errors, "pilot run count", len(rows))
        return errors
    if [row.get("pilot_order") for row in rows] != [1, 2, 3, 4]:
        _error(errors, "pilot order", [row.get("pilot_order") for row in rows])
    identifiers = [row.get("run_id") for row in rows]
    if len(set(identifiers)) != 4 or set(identifiers) & confirmatory_run_ids:
        _error(errors, "pilot run identity separation", identifiers)
    labels = [row.get("treatment_label") for row in rows]
    if labels != ["A", "B", "C", "D"]:
        _error(errors, "pilot treatment order", labels)
    for row in rows:
        cell = (row.get("predictor_mode"), row.get("network_condition"))
        if LABEL_TO_CELL.get(row.get("treatment_label")) != cell:
            _error(errors, "pilot treatment mismatch", row.get("run_id"))
        isolated = row.get("isolated_robot_id")
        if cell[1] == "nominal" and isolated is not None:
            _error(errors, "nominal pilot isolation", row.get("run_id"))
        if cell[1] == "isolated_robot_impairment" and isolated not in range(4):
            _error(errors, "impaired pilot isolation", row.get("run_id"))
    return errors


def validate_campaign_structure(
    campaign: dict[str, Any], pose_set_ids: set[str]
) -> list[dict[str, Any]]:
    """Validate the frozen 32-run factorial and its balancing properties."""
    errors: list[dict[str, Any]] = []
    if campaign.get("campaign_id") != CAMPAIGN_ID:
        _error(errors, "campaign identity", campaign.get("campaign_id"))
    if campaign.get("model_binding") != MODEL_BINDING:
        _error(errors, "campaign model binding", campaign.get("model_binding"))
    if campaign.get("team_size") != 4:
        _error(errors, "team_size", campaign.get("team_size"))
    if campaign.get("run_duration_s") != 600:
        _error(errors, "run duration", campaign.get("run_duration_s"))
    if campaign.get("required_complete_blocks") != 8:
        _error(errors, "required blocks", campaign.get("required_complete_blocks"))
    primary = campaign.get("primary_endpoint") or {}
    if primary.get("primary_contrast") != (
            "mso_342771_minus_observed_only_under_nominal_network"):
        _error(errors, "primary contrast identity", primary.get(
            "primary_contrast"))
    if campaign.get("results_status") != "not_collected":
        _error(errors, "results status", campaign.get("results_status"))
    if campaign.get("claim_authorized") is not False:
        _error(errors, "claim must remain unauthorized", campaign.get("claim_authorized"))
    if campaign.get("locks") != EXPECTED_LOCKS:
        _error(errors, "campaign lock path map", campaign.get("locks"))
    impairment = campaign.get("network_impairment") or {}
    if impairment.get("profile_id") != "isolate_one_robot_udp_v1":
        _error(errors, "network impairment profile", impairment.get("profile_id"))
    if impairment.get("transport") != "udp":
        _error(errors, "network impairment transport", impairment.get("transport"))
    if impairment.get("data_plane_only") is not True:
        _error(errors, "data-plane-only impairment", impairment.get("data_plane_only"))
    if impairment.get("out_of_band_safety_required") is not True:
        _error(errors, "out-of-band safety requirement", impairment.get(
            "out_of_band_safety_required"))
    if impairment.get("start_offset_s") != 180:
        _error(errors, "network impairment start offset", impairment.get(
            "start_offset_s"))
    if impairment.get("phases") != EXPECTED_NETWORK_PHASES:
        _error(errors, "network impairment phases", impairment.get("phases"))
    forbidden = forbidden_key_paths(campaign, "condition_id")
    if forbidden:
        _error(errors, "legacy condition_id is forbidden", forbidden)

    factors = campaign.get("factors") or {}
    if set(factors.get("predictor_mode") or []) != PREDICTOR_MODES:
        _error(errors, "predictor factor", factors.get("predictor_mode"))
    if set(factors.get("network_condition") or []) != NETWORK_CONDITIONS:
        _error(errors, "network factor", factors.get("network_condition"))
    allocation = campaign.get("balanced_allocation") or {}
    if allocation.get("design") != (
            "four_condition_williams_sequences_within_start_pose_block"):
        _error(errors, "balanced allocation design", allocation.get("design"))
    if allocation.get("frozen_before_collection") is not True:
        _error(errors, "allocation frozen before collection", allocation.get(
            "frozen_before_collection"))
    declared_sequences = {
        tuple(item) for item in allocation.get("allowed_williams_sequences") or []
        if isinstance(item, list)
    }
    if declared_sequences != ALLOWED_SEQUENCES:
        _error(errors, "allowed Williams sequences", allocation.get(
            "allowed_williams_sequences"))
    if allocation.get("required_repetitions_per_sequence") != 2:
        _error(errors, "Williams sequence repetitions", allocation.get(
            "required_repetitions_per_sequence"))

    runs = campaign.get("ordered_runs")
    if not isinstance(runs, list):
        _error(errors, "ordered runs", "missing list")
        return errors
    if len(runs) != 32:
        _error(errors, "run count", len(runs))

    orders = [run.get("order") for run in runs if isinstance(run, dict)]
    if orders != list(range(1, 33)):
        _error(errors, "unique sequential order", orders)
    run_ids = [run.get("run_id") for run in runs if isinstance(run, dict)]
    if len(run_ids) != len(set(run_ids)) or any(not item for item in run_ids):
        _error(errors, "unique non-empty run IDs", run_ids)
    for run in runs:
        if not isinstance(run, dict):
            continue
        match = RUN_ID.fullmatch(str(run.get("run_id", "")))
        expected = (int(match.group(1)), match.group(2)) if match else None
        actual = (run.get("block_id"), run.get("treatment_label"))
        if expected != actual:
            _error(errors, "portable run ID contract", {
                "run_id": run.get("run_id"), "expected": actual})

    expected_cells = {
        (predictor, network)
        for predictor in PREDICTOR_MODES for network in NETWORK_CONDITIONS
    }
    blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    overall_cells: Counter[tuple[str, str]] = Counter()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            _error(errors, "run mapping", {"index": index, "value": run})
            continue
        predictor = run.get("predictor_mode")
        network = run.get("network_condition")
        cell = (predictor, network)
        if predictor not in PREDICTOR_MODES or network not in NETWORK_CONDITIONS:
            _error(errors, "run factor value", {"run_id": run.get("run_id"), "cell": cell})
        else:
            overall_cells[cell] += 1
        label = run.get("treatment_label")
        if LABEL_TO_CELL.get(label) != cell:
            _error(errors, "treatment label mismatch", {
                "run_id": run.get("run_id"), "label": label, "cell": cell})
        if run.get("duration_s") != 600:
            _error(errors, "per-run duration", run.get("run_id"))
        block_id = run.get("block_id")
        if not isinstance(block_id, int):
            _error(errors, "block ID", {"run_id": run.get("run_id"), "block_id": block_id})
        else:
            blocks[block_id].append(run)
        assignment = run.get("hardware_pose_assignment")
        if not isinstance(assignment, list) or sorted(assignment) != [0, 1, 2, 3]:
            _error(errors, "hardware-to-pose permutation", run.get("run_id"))
        pose_set = run.get("start_pose_set_id")
        if pose_set not in pose_set_ids:
            _error(errors, "unknown start-pose set", {
                "run_id": run.get("run_id"), "pose_set": pose_set})
        isolated = run.get("isolated_robot_id")
        if network == "nominal" and isolated is not None:
            _error(errors, "nominal run isolates a robot", run.get("run_id"))
        if network == "isolated_robot_impairment" and isolated not in range(4):
            _error(errors, "impaired run lacks valid isolated robot", run.get("run_id"))

    if set(blocks) != set(range(1, 9)):
        _error(errors, "block IDs", sorted(blocks))
    for cell in expected_cells:
        if overall_cells[cell] != 8:
            _error(errors, "cell replication", {"cell": cell, "count": overall_cells[cell]})

    sequence_counts: Counter[tuple[str, ...]] = Counter()
    isolated_counts: Counter[int] = Counter()
    assignment_counts: Counter[tuple[int, ...]] = Counter()
    for block_id in sorted(blocks):
        rows = blocks[block_id]
        if len(rows) != 4:
            _error(errors, "runs per block", {"block_id": block_id, "count": len(rows)})
            continue
        rows = sorted(rows, key=lambda row: row.get("sequence_position", -1))
        block_orders = sorted(row.get("order") for row in rows)
        expected_orders = list(range((block_id - 1) * 4 + 1, block_id * 4 + 1))
        if block_orders != expected_orders:
            _error(errors, "consecutive within-block order", {
                "block_id": block_id, "orders": block_orders})
        if [row.get("sequence_position") for row in rows] != [1, 2, 3, 4]:
            _error(errors, "sequence positions", block_id)
        sequence = tuple(row.get("treatment_label") for row in rows)
        if sequence not in ALLOWED_SEQUENCES:
            _error(errors, "Williams sequence", {"block_id": block_id, "sequence": sequence})
        else:
            sequence_counts[sequence] += 1
        cells = {(row.get("predictor_mode"), row.get("network_condition")) for row in rows}
        if cells != expected_cells:
            _error(errors, "four factorial cells per block", block_id)
        pose_sets = {row.get("start_pose_set_id") for row in rows}
        expected_pose_set = f"P{block_id:02d}"
        if pose_sets != {expected_pose_set}:
            _error(errors, "one designated pose set per block", {
                "block_id": block_id,
                "pose_sets": sorted(str(item) for item in pose_sets)})
        assignments = {
            tuple(row.get("hardware_pose_assignment") or []) for row in rows
        }
        trials = {row.get("trial_id") for row in rows}
        if len(pose_sets) != 1 or len(assignments) != 1 or trials != {block_id}:
            _error(errors, "within-block lock mismatch", block_id)
        elif assignments:
            assignment_counts[next(iter(assignments))] += 1
        impaired_ids = {
            row.get("isolated_robot_id") for row in rows
            if row.get("network_condition") == "isolated_robot_impairment"
        }
        if len(impaired_ids) != 1:
            _error(errors, "one isolated robot per block", block_id)
        else:
            isolated = next(iter(impaired_ids))
            if isolated in range(4):
                isolated_counts[isolated] += 1

    if set(sequence_counts) != ALLOWED_SEQUENCES or any(
            count != 2 for count in sequence_counts.values()):
        _error(errors, "Williams sequence balance", {
            str(key): value for key, value in sequence_counts.items()})
    expected_rotations = {
        (0, 1, 2, 3), (1, 2, 3, 0),
        (2, 3, 0, 1), (3, 0, 1, 2),
    }
    if set(assignment_counts) != expected_rotations or any(
            count != 2 for count in assignment_counts.values()):
        _error(errors, "hardware-pose rotation balance", {
            str(key): value for key, value in assignment_counts.items()})
    if isolated_counts != Counter({0: 2, 1: 2, 2: 2, 3: 2}):
        _error(errors, "isolated robot balance", dict(isolated_counts))
    return errors


def validate_start_poses(start_poses: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    """Validate start-pose identities and resolved geometric separation."""
    errors: list[dict[str, Any]] = []
    pose_sets = start_poses.get("pose_sets") or []
    identifiers = [item.get("pose_set_id") for item in pose_sets if isinstance(item, dict)]
    expected = {f"P{index:02d}" for index in range(1, 9)}
    if set(identifiers) != expected or len(identifiers) != 8:
        _error(errors, "start-pose set IDs", identifiers)
    minimum = start_poses.get("minimum_start_separation_m", 0.0)
    for item in pose_sets:
        if not isinstance(item, dict):
            _error(errors, "start-pose mapping", item)
            continue
        poses = item.get("poses") or []
        slots = [pose.get("slot") for pose in poses if isinstance(pose, dict)]
        if slots != [0, 1, 2, 3]:
            _error(errors, "pose slots", item.get("pose_set_id"))
            continue
        if all(
            isinstance(pose.get(axis), (int, float)) and math.isfinite(pose[axis])
            for pose in poses for axis in ("x_m", "y_m", "yaw_deg")
        ):
            for first in range(4):
                for second in range(first + 1, 4):
                    dx = poses[first]["x_m"] - poses[second]["x_m"]
                    dy = poses[first]["y_m"] - poses[second]["y_m"]
                    if math.hypot(dx, dy) < float(minimum):
                        _error(errors, "start-pose separation", {
                            "pose_set_id": item.get("pose_set_id"),
                            "slots": [first, second]})
    return set(identifiers), errors


def validate_team(team: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the four distinct hardware identities."""
    errors: list[dict[str, Any]] = []
    robots = team.get("robots") or []
    if team.get("team_size") != 4 or len(robots) != 4:
        _error(errors, "four-robot roster", {
            "team_size": team.get("team_size"), "count": len(robots)})
        return errors
    if [robot.get("id") for robot in robots] != [0, 1, 2, 3]:
        _error(errors, "robot IDs", [robot.get("id") for robot in robots])
    for field in ("host", "base_serial", "lidar_serial", "compute_serial"):
        values = [robot.get(field) for robot in robots]
        if len(set(values)) != 4:
            _error(errors, f"unique {field}", values)
    return errors


def resolve_artifact(path_value: Any, lock_path: Path) -> Path | None:
    """Resolve a non-placeholder artifact path relative to its lock file."""
    if not isinstance(path_value, str) or not path_value or path_value.startswith("SET_"):
        return None
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (lock_path.parent / path).resolve()


def verified_json_report(
    lock: dict[str, Any], lock_path: Path, path_field: str,
    hash_field: str, label: str, blockers: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Load a digest-addressed report or add a readiness blocker."""
    report_path = resolve_artifact(lock.get(path_field), lock_path)
    if report_path is None or not report_path.is_file():
        blockers.append({
            "check": f"{label} report exists",
            "detail": str(report_path) if report_path else None})
        return None
    expected = lock.get(hash_field)
    if not isinstance(expected, str) or not HEX64.fullmatch(expected):
        blockers.append({
            "check": f"{label} report SHA-256 format", "detail": expected})
        return None
    actual = sha256(report_path)
    if actual != expected:
        blockers.append({
            "check": f"{label} report SHA-256 match",
            "detail": {"expected": expected, "actual": actual}})
        return None
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        blockers.append({
            "check": f"{label} report JSON", "detail": str(error)})
        return None
    if not isinstance(value, dict):
        blockers.append({
            "check": f"{label} report mapping", "detail": type(value).__name__})
        return None
    return value


def require_hashed_file(
    lock: dict[str, Any], lock_path: Path, path_field: str,
    hash_field: str, label: str, blockers: list[dict[str, Any]],
) -> Path | None:
    """Require an existing file whose bytes match the frozen SHA-256."""
    target = resolve_artifact(lock.get(path_field), lock_path)
    if target is None or not target.is_file():
        blockers.append({
            "check": f"{label} exists",
            "detail": str(target) if target else None})
        return None
    expected = lock.get(hash_field)
    if not isinstance(expected, str) or not HEX64.fullmatch(expected):
        blockers.append({
            "check": f"{label} SHA-256 format", "detail": expected})
        return None
    actual = sha256(target)
    if actual != expected:
        blockers.append({
            "check": f"{label} SHA-256 match",
            "detail": {"expected": expected, "actual": actual}})
        return None
    return target


def require_positive_number(
    value: Any, label: str, blockers: list[dict[str, Any]], *,
    allow_zero: bool = False,
) -> None:
    """Require a finite positive (or optionally non-negative) real number."""
    valid = all((
        isinstance(value, (int, float)) and not isinstance(value, bool),
        math.isfinite(float(value)) if isinstance(value, (int, float)) else False,
        (value >= 0 if allow_zero else value > 0)
        if isinstance(value, (int, float)) else False,
    ))
    if not valid:
        blockers.append({"check": label, "detail": value})


def readiness_blockers(
    documents: dict[str, dict[str, Any]], paths: dict[str, Path]
) -> list[dict[str, Any]]:
    """Return every reason why physical collection must remain disabled."""
    blockers: list[dict[str, Any]] = []
    if not COLLECTION_ENABLEMENT_IMPLEMENTED:
        blockers.append({
            "check": "trusted collection enablement implementation",
            "detail": (
                "not implemented in this protocol-only branch; reviewed "
                "executable model, online-stack, network and run-audit "
                "verifiers are required"),
        })
    for name, document in documents.items():
        if document.get("collection_ready") is not True:
            blockers.append({
                "check": f"{name}.collection_ready",
                "detail": document.get("collection_ready")})
        unresolved = placeholder_paths(document)
        if unresolved:
            blockers.append({"check": f"{name} unresolved placeholders", "detail": unresolved})

    pilot = documents["pilot"]
    for field, expected in (
        ("pilot_completion_status", "verified"),
        ("all_four_pilots_protocol_passed", True),
        ("no_critical_safety_faults", True),
        ("evidence_pipeline_verified", True),
    ):
        if pilot.get(field) != expected:
            blockers.append({
                "check": f"pilot completion {field}",
                "detail": pilot.get(field)})
    pilot_report = verified_json_report(
        pilot, paths["pilot"], "pilot_completion_report_path",
        "pilot_completion_report_sha256", "pilot completion", blockers)
    if pilot_report is not None:
        expected_pilot_ids = [
            "tf27_n4_pilot_A", "tf27_n4_pilot_B",
            "tf27_n4_pilot_C", "tf27_n4_pilot_D"]
        required_pilot_values = {
            "schema_version": "mso_n4_pilot_completion_v1",
            "verified": True,
            "pilot_plan_id": PILOT_PLAN_ID,
            "pilot_run_ids": expected_pilot_ids,
            "all_four_pilots_protocol_passed": True,
            "no_critical_safety_faults": True,
            "evidence_pipeline_verified": True,
            "permanently_excluded_from_confirmatory_analysis": True,
        }
        for field, expected in required_pilot_values.items():
            if pilot_report.get(field) != expected:
                blockers.append({
                    "check": f"pilot completion report {field}",
                    "detail": {
                        "expected": expected,
                        "actual": pilot_report.get(field)}})

    arena = documents["arena"]
    gt = arena.get("ground_truth_map") or {}
    require_positive_number(
        gt.get("resolution_m"), "finite positive GT resolution", blockers)
    for path_field, hash_field, label in (
        ("occupancy_image_path", "occupancy_image_sha256", "GT occupancy image"),
        ("occupancy_metadata_path", "occupancy_metadata_sha256", "GT occupancy metadata"),
        ("accessible_free_mask_path", "accessible_free_mask_sha256",
         "GT accessible-free mask"),
        ("dynamic_exclusion_mask_path", "dynamic_exclusion_mask_sha256",
         "GT dynamic exclusion mask"),
    ):
        require_hashed_file(
            gt, paths["arena"], path_field, hash_field, label, blockers)

    reference = documents["reference"]
    require_positive_number(
        reference.get("pose_rate_hz"), "finite positive reference pose rate", blockers)
    require_positive_number(
        reference.get("maximum_event_join_tolerance_ms"),
        "finite positive reference event-join tolerance", blockers)
    calibration = reference.get("calibration") or {}
    extrinsics = reference.get("robot_extrinsics") or {}
    require_positive_number(
        calibration.get("translation_uncertainty_m"),
        "finite positive translation uncertainty", blockers)
    require_positive_number(
        calibration.get("yaw_uncertainty_deg"),
        "finite positive yaw uncertainty", blockers)
    require_hashed_file(
        calibration, paths["reference"], "path", "sha256",
        "reference calibration", blockers)
    require_hashed_file(
        extrinsics, paths["reference"], "path", "sha256",
        "robot reference extrinsics", blockers)
    require_positive_number(
        documents["start_poses"].get("minimum_start_separation_m"),
        "finite positive start-pose separation", blockers)
    team_lock = documents["team"].get("protocol_lock") or {}
    require_positive_number(
        team_lock.get("maximum_clock_offset_ms"),
        "finite positive maximum clock offset", blockers)
    require_positive_number(
        team_lock.get("maximum_clock_rtt_ms"),
        "finite positive maximum clock RTT", blockers)

    model = documents["model"]
    if model.get("schema_version") != "2.0":
        blockers.append({
            "check": "model lock schema",
            "detail": model.get("schema_version")})
    artifact = resolve_artifact(model.get("artifact_path"), paths["model"])
    if artifact is None or not artifact.is_file():
        blockers.append({
            "check": "verified 342771-parameter artifact exists",
            "detail": str(artifact) if artifact else None})
    else:
        expected_hash = model.get("artifact_sha256")
        if not isinstance(expected_hash, str) or not HEX64.fullmatch(expected_hash):
            blockers.append({"check": "model SHA-256 format", "detail": expected_hash})
        elif sha256(artifact) != expected_hash:
            blockers.append({"check": "model SHA-256 match", "detail": str(artifact)})
    if model.get("model_id") != MODEL_ID:
        blockers.append({"check": "model identity", "detail": model.get("model_id")})
    if model.get("artifact_status") != (
            "verified_recovered_deployment_candidate"):
        blockers.append({
            "check": "model provenance status",
            "detail": model.get("artifact_status")})
    if model.get("artifact_role") != (
            "recovered_candidate_from_checkpoint_retained_in_deployment_copy"):
        blockers.append({"check": "model artifact role", "detail": model.get(
            "artifact_role")})
    if model.get("historical_manuscript_checkpoint_claim") is not False:
        blockers.append({
            "check": "model historical-claim boundary",
            "detail": model.get("historical_manuscript_checkpoint_claim")})
    if model.get("architecture_loader_id") != MODEL_LOADER_ID:
        blockers.append({
            "check": "model loader identity",
            "detail": model.get("architecture_loader_id")})
    if model.get("architecture_class") != MODEL_ARCHITECTURE:
        blockers.append({
            "check": "model architecture identity",
            "detail": model.get("architecture_class")})
    parameters = model.get("exact_trainable_parameter_count")
    if (not isinstance(parameters, int) or isinstance(parameters, bool)
            or parameters != MODEL_PARAMETERS):
        blockers.append({"check": "exact model parameter count", "detail": parameters})
    for field, expected in (
        ("ffc_block_count", 4),
        ("state_tensor_count", 594),
        ("state_value_count", 347690),
    ):
        if model.get(field) != expected:
            blockers.append({
                "check": f"model {field}",
                "detail": model.get(field)})
    if model.get("reference_runtime") != REFERENCE_RUNTIME:
        blockers.append({
            "check": "model reference runtime",
            "detail": model.get("reference_runtime")})
    artifact_hash = model.get("artifact_sha256")
    if artifact_hash != SELECTED_MODEL_ARTIFACT_SHA256:
        blockers.append({
            "check": "selected model artifact identity",
            "detail": artifact_hash})
    if model.get("strict_load_verified") is not True:
        blockers.append({
            "check": "strict model load",
            "detail": model.get("strict_load_verified")})
    if model.get("missing_keys") != [] or model.get("unexpected_keys") != []:
        blockers.append({"check": "strict-load key sets", "detail": {
            "missing_keys": model.get("missing_keys"),
            "unexpected_keys": model.get("unexpected_keys")}})
    fingerprint = model.get("fixed_fixture_output_fingerprint")
    if not isinstance(fingerprint, str) or not HEX64.fullmatch(fingerprint):
        blockers.append({"check": "model output fingerprint", "detail": fingerprint})
    require_hashed_file(
        model, paths["model"], "fixed_fixture_path", "fixed_fixture_sha256",
        "fixed 256x256 model fixture", blockers)
    require_hashed_file(
        model, paths["model"], "training_provenance_path",
        "training_provenance_sha256", "model training provenance", blockers)
    require_hashed_file(
        model, paths["model"], "selection_record_path",
        "selection_record_sha256", "model selection record", blockers)
    require_hashed_file(
        model, paths["model"], "architecture_source_path",
        "architecture_source_sha256", "model architecture source", blockers)
    require_hashed_file(
        model, paths["model"], "ffc_source_path",
        "ffc_source_sha256", "model FFC source", blockers)

    model_report = verified_json_report(
        model, paths["model"], "verifier_report_path",
        "verifier_report_sha256", "model verifier", blockers)
    if model_report is not None:
        verifier_script = Path(__file__).resolve().parent.joinpath(
            "verify_model_artifact.py")
        required_model_values = {
            "schema_version": "mso_model_verification_v2",
            "verified": True,
            "model_id": MODEL_ID,
            "artifact_role": (
                "recovered_candidate_from_checkpoint_retained_in_deployment_copy"),
            "historical_manuscript_checkpoint_claim": False,
            "artifact_filename": Path(
                str(model.get("artifact_path"))).name,
            "artifact_sha256": model.get("artifact_sha256"),
            "loader_id": model.get("architecture_loader_id"),
            "architecture_class": model.get("architecture_class"),
            "strict_state_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "trainable_parameter_count": MODEL_PARAMETERS,
            "state_tensor_count": 594,
            "state_value_count": 347690,
            "finite_state": True,
            "ffc_block_count": 4,
            "input_shape": [1, 3, 256, 256],
            "output_shape": [1, 1, 256, 256],
            "output_count": 5,
            "all_output_shapes": [
                [1, 1, 256, 256],
                [1, 8, 128, 128],
                [1, 16, 64, 64],
                [1, 16, 64, 64],
                [1, 8, 128, 128],
            ],
            "all_outputs_finite": True,
            "finite_forward": True,
            "python_version": REFERENCE_RUNTIME["python"],
            "torch_version": REFERENCE_RUNTIME["torch"],
            "device": REFERENCE_RUNTIME["device"],
            "fixture_sha256": model.get("fixed_fixture_sha256"),
            "output_fingerprint": model.get(
                "fixed_fixture_output_fingerprint"),
            "verifier_implementation_sha256": sha256(verifier_script),
            "architecture_source_sha256": model.get(
                "architecture_source_sha256"),
            "ffc_source_sha256": model.get("ffc_source_sha256"),
        }
        for field, expected in required_model_values.items():
            if model_report.get(field) != expected:
                blockers.append({
                    "check": f"model verifier report {field}",
                    "detail": {
                        "expected": expected,
                        "actual": model_report.get(field)}})

    stack = documents["online_stack"]
    stack_model_binding = {
        "predictor_model_id": model.get("model_id"),
        "predictor_artifact_sha256": model.get("artifact_sha256"),
        "predictor_loader_id": model.get("architecture_loader_id"),
        "predictor_input_contract": model.get("input_contract"),
        "predictor_output_contract": model.get("output_contract"),
        "predictor_architecture_source_sha256": model.get(
            "architecture_source_sha256"),
        "predictor_ffc_source_sha256": model.get("ffc_source_sha256"),
    }
    for field, expected in stack_model_binding.items():
        if stack.get(field) != expected:
            blockers.append({
                "check": f"online stack model binding {field}",
                "detail": {"expected": expected, "actual": stack.get(field)}})
    if stack.get("stack_status") != "verified":
        blockers.append({
            "check": "full online stack status",
            "detail": stack.get("stack_status")})
    stack_commit = stack.get("external_repository_commit")
    if not isinstance(stack_commit, str) or not HEX40.fullmatch(stack_commit):
        blockers.append({
            "check": "external full-stack commit", "detail": stack_commit})
    repository = resolve_artifact(
        stack.get("external_repository_path"), paths["online_stack"])
    if repository is None or not repository.is_dir():
        blockers.append({
            "check": "external full-stack repository exists",
            "detail": str(repository) if repository else None})
    else:
        launch_value = stack.get("launch_entrypoint")
        launch_is_resolved = all((
            isinstance(launch_value, str),
            bool(launch_value),
            not launch_value.startswith("SET_")
            if isinstance(launch_value, str) else False,
        ))
        launch_path = repository / launch_value if launch_is_resolved else None
        if launch_path is None or not launch_path.is_file():
            blockers.append({
                "check": "full-stack launch entrypoint exists",
                "detail": str(launch_path) if launch_path else None})
        try:
            actual_commit = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
                timeout=10).stdout.strip()
            if actual_commit != stack_commit:
                blockers.append({
                    "check": "external repository commit match",
                    "detail": {"expected": stack_commit, "actual": actual_commit}})
            dirty = subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain"],
                check=True, capture_output=True, text=True,
                timeout=10).stdout.strip()
            if dirty:
                blockers.append({
                    "check": "external repository clean worktree",
                    "detail": dirty.splitlines()})
        except (OSError, subprocess.SubprocessError) as error:
            blockers.append({
                "check": "external repository git verification",
                "detail": str(error)})
    required_stages = {
        "predictor", "registration", "gate", "commit",
        "persistent_map", "planner"}
    stages = stack.get("required_stages") or {}
    if set(stages) != required_stages or any(
            stages.get(stage) is not True for stage in required_stages):
        blockers.append({
            "check": "complete online pipeline stages", "detail": stages})
    stack_requirements = stack.get("requirements") or {}
    required_stack_requirements = {
        "four_robot_supported",
        "full_pipeline_integration_test_passed",
        "registration_event_interface_verified",
        "persistent_map_revision_hash_verified",
        "measured_only_collision_authority_verified",
        "prediction_shadow_write_barrier_verified",
        "predictor_model_binding_verified",
        "predictor_strict_raw_state_load_verified",
        "predictor_runtime_equivalence_verified",
    }
    if set(stack_requirements) != required_stack_requirements or any(
            stack_requirements.get(item) is not True
            for item in required_stack_requirements):
        blockers.append({
            "check": "complete online stack requirements",
            "detail": stack_requirements})
    stack_report = verified_json_report(
        stack, paths["online_stack"], "verification_report_path",
        "verification_report_sha256", "online stack verifier", blockers)
    if stack_report is not None:
        required_stack_values = {
            "schema_version": "mso_online_stack_verification_v1",
            "verified": True,
            "stack_id": "mso_full_online_n4_v1",
            "repository_commit": stack_commit,
            "launch_entrypoint": stack.get("launch_entrypoint"),
            "four_robot_supported": True,
            "full_pipeline_integration_test_passed": True,
            "registration_event_interface_verified": True,
            "persistent_map_revision_hash_verified": True,
            "measured_only_collision_authority_verified": True,
            "prediction_shadow_write_barrier_verified": True,
            "predictor_model_binding_verified": True,
            "predictor_strict_raw_state_load_verified": True,
            "predictor_runtime_equivalence_verified": True,
            "pipeline_stages": sorted(required_stages),
            **stack_model_binding,
        }
        for field, expected in required_stack_values.items():
            if stack_report.get(field) != expected:
                blockers.append({
                    "check": f"online stack report {field}",
                    "detail": {
                        "expected": expected,
                        "actual": stack_report.get(field)}})

    network = documents["network"]
    if network.get("network_status") != "verified":
        blockers.append({
            "check": "N4 network impairment status",
            "detail": network.get("network_status")})
    require_hashed_file(
        network, paths["network"], "dds_profile_path", "dds_profile_sha256",
        "DDS unicast profile", blockers)
    topology = network.get("topology_contract") or {}
    evidence = network.get("evidence_contract") or {}
    required_topology = {
        "isolated_robot_peer_count": 3,
        "required_directed_rule_count": 6,
        "peer_addressed_unicast": True,
        "multicast_data_path_disabled": True,
        "independent_safety_plane": True,
    }
    required_evidence = {
        "probe_traffic_verified_all_six_directions": True,
        "rule_packet_counters_verified_all_six_directions": True,
        "phase_timing_verified": True,
        "exact_rule_cleanup_verified": True,
        "no_compounded_input_rules_verified": True,
    }
    for field, expected in required_topology.items():
        if topology.get(field) != expected:
            blockers.append({
                "check": f"network topology {field}",
                "detail": topology.get(field)})
    for field, expected in required_evidence.items():
        if evidence.get(field) != expected:
            blockers.append({
                "check": f"network evidence {field}",
                "detail": evidence.get(field)})
    network_report = verified_json_report(
        network, paths["network"], "verification_report_path",
        "verification_report_sha256", "N4 network verifier", blockers)
    if network_report is not None:
        required_network_values = {
            "schema_version": "mso_n4_network_verification_v1",
            "verified": True,
            "network_lock_id": "tf27_n4_isolate_one_robot_udp_v1",
            "dds_profile_sha256": network.get("dds_profile_sha256"),
            "isolated_robot_peer_count": 3,
            "directed_rule_count": 6,
            "dds_peer_addressed_unicast": True,
            "multicast_data_path_disabled": True,
            "independent_safety_plane": True,
            "probe_traffic_verified_all_six_directions": True,
            "rule_packet_counters_verified_all_six_directions": True,
            "phase_timing_verified": True,
            "exact_rule_cleanup_verified": True,
            "no_compounded_input_rules_verified": True,
        }
        for field, expected in required_network_values.items():
            if network_report.get(field) != expected:
                blockers.append({
                    "check": f"N4 network report {field}",
                    "detail": {
                        "expected": expected,
                        "actual": network_report.get(field)}})

    commit = (documents["team"].get("protocol_lock") or {}).get("software_commit")
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        blockers.append({"check": "deployed software commit", "detail": commit})
    return blockers


def validate_package(
    campaign_path: Path,
    team_path: Path,
    arena_path: Path,
    start_poses_path: Path,
    model_path: Path,
    online_stack_path: Path,
    network_path: Path,
    pilot_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    """Validate structure and independently compute collection readiness."""
    paths = {
        "campaign": campaign_path.resolve(),
        "team": team_path.resolve(),
        "arena": arena_path.resolve(),
        "start_poses": start_poses_path.resolve(),
        "model": model_path.resolve(),
        "online_stack": online_stack_path.resolve(),
        "network": network_path.resolve(),
        "pilot": pilot_path.resolve(),
        "reference": reference_path.resolve(),
    }
    documents: dict[str, dict[str, Any]] = {}
    load_errors: list[dict[str, Any]] = []
    for name, path in paths.items():
        try:
            documents[name] = load_yaml(path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            load_errors.append({"check": f"load {name}", "detail": str(error)})
    if load_errors:
        return {
            "schema_version": "1.0", "plan_valid": False,
            "collection_ready": False, "claim_authorized": False,
            "errors": load_errors, "readiness_blockers": load_errors,
        }

    pose_ids, pose_errors = validate_start_poses(documents["start_poses"])
    errors = pose_errors
    package_root = paths["campaign"].parent.parent
    path_key_map = {
        "team": "team", "arena": "arena", "start_poses": "start_poses",
        "model": "model", "online_stack": "online_stack",
        "network": "network", "pilot_plan": "pilot",
        "reference": "reference",
    }
    for lock_key, path_key in path_key_map.items():
        expected_path = (package_root / EXPECTED_LOCKS[lock_key]).resolve()
        if paths[path_key] != expected_path:
            _error(errors, "CLI input path differs from campaign lock", {
                "lock": lock_key,
                "expected": str(expected_path),
                "actual": str(paths[path_key]),
            })
    errors.extend(validate_team(documents["team"]))
    errors.extend(validate_campaign_structure(documents["campaign"], pose_ids))
    confirmatory_ids = {
        run.get("run_id") for run in documents["campaign"].get("ordered_runs") or []
        if isinstance(run, dict) and run.get("run_id")}
    errors.extend(validate_pilot_plan(documents["pilot"], confirmatory_ids))
    if documents["arena"].get("run_duration_s") != 600:
        _error(errors, "arena run duration", documents["arena"].get("run_duration_s"))
    campaign = documents["campaign"]
    arena = documents["arena"]
    start_poses = documents["start_poses"]
    reference = documents["reference"]
    if campaign.get("site_id") != arena.get("site_id"):
        _error(errors, "campaign and arena site identity", {
            "campaign": campaign.get("site_id"), "arena": arena.get("site_id")})
    if campaign.get("arena_id") != arena.get("arena_id"):
        _error(errors, "campaign and arena identity", {
            "campaign": campaign.get("arena_id"), "arena": arena.get("arena_id")})
    frames = {
        arena.get("coordinate_frame"), start_poses.get("frame_id"),
        reference.get("static_map_frame")}
    if len(frames) != 1 or None in frames:
        _error(errors, "GT, start-pose and reference frame agreement", sorted(
            str(item) for item in frames))
    required_reference = {
        "independent_from_mso": True,
        "external_pose_not_consumed_by_planner": True,
        "reference_for_every_registration_decision": True,
        "frozen_gt_to_global_transform": True,
    }
    if reference.get("requirements") != required_reference:
        _error(errors, "independent reference requirements", reference.get(
            "requirements"))
    expected_safety = {
        "maximum_linear_velocity_m_s": 0.25,
        "maximum_angular_velocity_rad_s": 0.60,
        "minimum_robot_separation_m": 0.35,
        "independent_emergency_stop_required": True,
        "observers_required": 3,
    }
    if arena.get("safety") != expected_safety:
        _error(errors, "frozen conservative safety limits", arena.get("safety"))
    campaign_phases = (campaign.get("network_impairment") or {}).get("phases")
    network = documents["network"]
    if network.get("network_lock_id") != "tf27_n4_isolate_one_robot_udp_v1":
        _error(errors, "N4 network lock identity", network.get("network_lock_id"))
    if network.get("phase_schedule") != campaign_phases:
        _error(errors, "campaign and network phase schedule agreement", {
            "campaign": campaign_phases,
            "network": network.get("phase_schedule")})
    for name, document in documents.items():
        if document.get("claim_authorized") is not False:
            _error(errors, f"{name} claim must remain unauthorized", {
                "claim_authorized": document.get("claim_authorized")})
        forbidden = forbidden_key_paths(document, "condition_id")
        if forbidden:
            _error(errors, f"legacy condition_id in {name}", forbidden)

    blockers = readiness_blockers(documents, paths)
    ready = not errors and not blockers and COLLECTION_ENABLEMENT_IMPLEMENTED
    declared_ready = all(
        document.get("collection_ready") is True for document in documents.values())
    if declared_ready and not ready:
        _error(errors, "declared ready package has unresolved blockers", blockers)
        ready = False
    return {
        "schema_version": "1.0",
        "campaign_id": documents["campaign"].get("campaign_id"),
        "plan_valid": not errors,
        "collection_ready": ready,
        "claim_authorized": False,
        "planned_run_count": len(documents["campaign"].get("ordered_runs") or []),
        "planned_block_count": len({
            run.get("block_id")
            for run in documents["campaign"].get("ordered_runs") or []
            if isinstance(run, dict)}),
        "errors": errors,
        "readiness_blockers": blockers,
        "input_sha256": {
            name: sha256(path) for name, path in paths.items() if path.is_file()
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=base / "config/campaign.yaml")
    parser.add_argument("--team", type=Path, default=base / "config/team_4.yaml")
    parser.add_argument("--arena", type=Path, default=base / "config/arena.yaml")
    parser.add_argument("--start-poses", type=Path, default=base / "config/start_poses.yaml")
    parser.add_argument("--model-lock", type=Path, default=base / "config/model_lock.yaml")
    parser.add_argument(
        "--online-stack-lock", type=Path,
        default=base / "config/online_stack_lock.yaml")
    parser.add_argument(
        "--network-lock", type=Path,
        default=base / "config/network_lock.yaml")
    parser.add_argument(
        "--pilot-plan", type=Path,
        default=base / "config/pilot_plan.yaml")
    parser.add_argument("--reference", type=Path, default=base / "config/reference.yaml")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-collection-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Validate the package and use a distinct exit for readiness failure."""
    args = parse_args()
    report = validate_package(
        args.campaign, args.team, args.arena, args.start_poses,
        args.model_lock, args.online_stack_lock, args.network_lock,
        args.pilot_plan, args.reference)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["plan_valid"]:
        return 1
    if args.require_collection_ready and not report["collection_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
