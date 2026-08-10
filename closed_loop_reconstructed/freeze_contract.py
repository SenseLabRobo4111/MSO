"""Checkpoint, training-manifest, and validation-selection freeze contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Callable

from .artifact_chain import (
    audit_train_validation_dataset,
    strict_model_validation,
    validate_checkpoint_payload as validate_v2_checkpoint_payload,
    validate_training_and_selection,
    validate_v2_lock,
)
from .environment_lock import capture_runtime_environment, verify_environment_lock
from .hashing import canonical_json_sha256, sha256_file


SHA256 = re.compile(r"^[0-9a-f]{64}$")
validate_checkpoint_payload = validate_v2_checkpoint_payload


def _integer_equal(value: Any, expected: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value == int(expected)
    )


def _number_equal(value: Any, expected: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-12)
    )


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def validate_protocol_implementation_contract(
    protocol: dict[str, Any], *, require_freeze_ready: bool = False
) -> None:
    """Reject protocol fields that the reconstructed runner does not implement."""

    declared_arms = protocol.get("arms")
    supported_arm_sets = (
        ["prediction_on", "observed_only"],
        [
            "observed_only",
            "prediction_registration_only",
            "prediction_frontier_only",
            "prediction_both",
            "oracle_completion",
            "adversarial_prediction",
        ],
    )
    if declared_arms not in supported_arm_sets:
        raise ValueError("the declared arms differ from implemented arm contracts")
    experiment = protocol.get("experiment", {})
    supported_team_sizes = {2, 3, 5, 8, 10}
    team_size = experiment.get("team_size")
    if not isinstance(team_size, int) or isinstance(team_size, bool) or (
        team_size not in supported_team_sizes
    ):
        raise ValueError("the campaign team size is outside the implemented grid")
    if not _integer_equal(experiment.get("protocol_version"), 1):
        raise ValueError("unsupported protocol version")
    if experiment.get("historical_ros_reproduction") is not False or (
        experiment.get("physical_experiment") is not False
    ):
        raise ValueError(
            "reconstructed runner cannot claim historical or physical data"
        )
    checkpoint_policy = protocol.get("checkpoint_policy", {})
    if checkpoint_policy.get("historical_claim") is not False or (
        checkpoint_policy.get("mock_allowed_splits") != ["train", "val"]
    ) or checkpoint_policy.get(
        "test_requires_validation_selected_checkpoint"
    ) is not True:
        raise ValueError("checkpoint split/provenance policy is incompatible")
    expected_checkpoint = {
        "required_schema": "mso.prospective_v2.checkpoint/1",
        "required_status": "prospective_v2_not_historical_recovery",
        "required_stage": "student",
        "selection_metric": "validation_unknown_bce",
        "selection_direction": "minimize",
        "runtime_model_class": (
            "sensemap.explore_model.SenseMapNet.DistillMapNetDeconv"
        ),
        "runtime_model_dim": 4,
        "runtime_parameter_count": 342771,
    }
    if any(
        checkpoint_policy.get(key) != value
        for key, value in expected_checkpoint.items()
    ):
        raise ValueError("checkpoint runtime/provenance contract is incompatible")
    randomization = protocol.get("randomization")
    if not isinstance(randomization, dict) or randomization.get("streams") != [
        "team_starts",
        "private_frames",
    ]:
        raise ValueError("protocol declares unimplemented or missing random streams")
    deterministic = randomization.get("deterministic_shared_components")
    if not isinstance(deterministic, list) or set(deterministic) != {
        "sensor ray casting",
        "motion and collision transitions",
        "encounter range scheduling",
    }:
        raise ValueError("deterministic shared-component contract is incomplete")

    planning = protocol.get("planning", {})
    simulation = protocol.get("simulation", {})
    if not _positive_number(planning.get("distance_budget_m")):
        raise ValueError("fixed distance budget is invalid")
    if not _number_equal(planning.get("distance_budget_m"), 180.0):
        raise ValueError("implemented prospective analysis is locked to 180 m")
    for name in (
        "obstacle_inflation_m",
        "frontier_rank_radius_m",
        "prediction_rank_weight_m",
    ):
        if not _positive_number(planning.get(name)):
            raise ValueError(f"planning field {name} is invalid")
    if any((
        planning.get("authoritative_cells") != "measured only",
        planning.get("prediction_use") != "temporary frontier ranking only",
        planning.get("traversability") != "measured free cells only",
    )):
        raise ValueError("planning authority contract differs from implementation")
    for name in (
        "maximum_ticks",
        "laser_range_cells",
        "laser_rays",
        "registration_period_ticks",
        "verification_period_ticks",
        "registration_canvas_px",
        "planning_canvas_px",
        "prediction_model_input_px",
        "prediction_training_raw_px",
        "planner_period_ticks",
        "prediction_period_ticks",
    ):
        if not _positive_integer(simulation.get(name)):
            raise ValueError(f"simulation field {name} is invalid")
    for name in (
        "working_resolution_m_per_cell",
        "encounter_range_m",
        "prediction_world_crop_side_m",
    ):
        if not _positive_number(simulation.get(name)):
            raise ValueError(f"simulation field {name} is invalid")
    stop_coverage = simulation.get("stop_coverage")
    if not _positive_number(stop_coverage) or float(stop_coverage) > 1.0:
        raise ValueError("stop coverage is outside (0, 1]")
    crop_cells = float(simulation["prediction_world_crop_side_m"]) / float(
        simulation["working_resolution_m_per_cell"]
    )
    if not math.isclose(crop_cells, round(crop_cells), abs_tol=1e-12):
        raise ValueError("prediction world crop does not align to the simulator grid")
    stuck_window = simulation.get("stuck_window_ticks")
    if not _positive_integer(stuck_window):
        raise ValueError("stuck episode window is invalid")
    if not _integer_equal(simulation.get("motion_cells_per_tick"), 1):
        raise ValueError("runner implements exactly one requested cell per tick")
    if simulation.get("prediction_input_semantics") != (
        "three channels: occupied, unknown, free"
    ) or simulation.get("prediction_output_semantics") != (
        "occupied probability in [0, 1]"
    ):
        raise ValueError("predictor tensor semantics differ from the implementation")
    gate = protocol.get("gate", {})
    if not _positive_integer(gate.get("temporal_consensus_count")):
        raise ValueError("temporal consensus count is invalid")
    for name in (
        "cycle_absolute_yaw_limit_deg",
        "cycle_translation_limit_m",
        "temporal_yaw_limit_deg",
        "temporal_translation_limit_m",
    ):
        if not _positive_number(gate.get(name)):
            raise ValueError(f"gate field {name} is invalid")
    if protocol.get("freeze", {}).get(
        "stopping_rule_locked_before_freeze"
    ) is not True:
        raise ValueError("predeclared stopping-rule lock is absent")
    logging = protocol.get("logging", {})
    if any((
        not _integer_equal(logging.get("schema_version"), 1),
        logging.get("log_every_scan") is not True,
        logging.get("retain_registration_rasters") is not False,
        logging.get("event_format")
        != "canonical JSONL with a SHA-256 hash chain",
    )):
        raise ValueError("logging protocol differs from the implementation")
    primary = protocol.get("primary_endpoints")
    if not isinstance(primary, dict) or "persistent-team-map" not in str(
        primary.get("coverage_auc", "")
    ) or "persistent-team-map" not in str(
        primary.get("distance_to_80_percent", "")
    ):
        raise ValueError("primary endpoints are not persistent-team-map endpoints")

    safety = protocol.get("safety_endpoints")
    if not isinstance(safety, dict) or safety.get(
        "zero_denominator_policy"
    ) != "undefined_null_never_zero":
        raise ValueError("safety zero-denominator rule is not implemented")
    if safety.get("analysis_role") != (
        "joint descriptive safety endpoints; no standalone inferential safety claim"
    ):
        raise ValueError("safety analysis role differs from the implementation")
    correctness = safety.get("registration_correctness", {})
    if not _number_equal(correctness.get("translation_m"), 0.25) or not (
        _number_equal(correctness.get("absolute_yaw_deg"), 5.0)
    ):
        raise ValueError("registration correctness boundary differs from evaluator")
    if not _integer_equal(safety.get("reject_state_hash_violations_allowed"), 0):
        raise ValueError("reject-state hash violation allowance must remain zero")
    required_joint = {
        "collision_count",
        "collision_rate_per_motion_attempt",
        "motion_attempt_count",
        "motion_attempt_rate_per_robot_tick",
        "no_route_count",
        "replan_count",
        "stuck_episode_count",
        "cumulative_team_distance_m",
        "persistent_map_coverage_auc_normalized",
        "persistent_map_coverage_at_budget",
        "gate_wrong_accept_rate",
        "final_decision_false_accept_rate",
        "final_decision_false_reject_count",
        "unsafe_commit_unrecovered_count",
    }
    if set(safety.get("episode_level_joint_reporting", [])) != required_joint:
        raise ValueError("joint safety endpoint contract is incomplete")

    statistics = protocol.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("statistics contract is absent")
    alpha = statistics.get("alpha")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not (
        0.0 < float(alpha) < 1.0
    ):
        raise ValueError("statistics alpha is invalid")
    expected_statistics = {
        "alternative": "two_sided",
        "bootstrap_clusters": "building",
        "independent_unit": "building",
        "paired_arms": True,
        "seeds_are_repeated_measures": True,
        "primary_endpoint_names": [
            "coverage_auc_normalized",
            "restricted_distance_to_80_m",
        ],
        "required_complete_pair_keys": [
            "building_id",
            "floorplan_id",
            "seed",
        ],
        "zero_denominator_policy": (
            "retain null and report defined paired observations and independent "
            "buildings; never impute zero"
        ),
        "effect_directions": {
            "coverage_auc_normalized": "prediction_on_minus_observed_only",
            "restricted_distance_to_80_m": "observed_only_minus_prediction_on",
        },
        "primary_test": (
            "paired randomization test on building-level seed-averaged differences"
        ),
        "multiple_primary_endpoints": (
            "Holm family-wise correction across the two primary two-sided tests"
        ),
        "censoring_policy": (
            "replace unreached distance-to-80 values by the fixed 180 m "
            "restriction point; estimate no effect beyond the restriction point"
        ),
    }
    mismatches = {
        key: statistics.get(key)
        for key, expected in expected_statistics.items()
        if statistics.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"statistics contract differs from implementation: {mismatches}"
        )
    for name in ("bootstrap_seed", "randomization_seed"):
        value = statistics.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"statistics {name} must be a nonnegative integer")
    for name in ("bootstrap_replicates", "randomization_replicates"):
        value = statistics.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"statistics {name} must be a positive integer")

    freeze = protocol.get("freeze", {})
    if any((
        freeze.get("protocol_and_software_hashes_required_before_test") is not True,
        freeze.get("test_output_forbidden_before_unlock") is not True,
        freeze.get("validation_only_tuning") is not True,
    )):
        raise ValueError("freeze boundary contract is incomplete")
    if require_freeze_ready and any((
        freeze.get("current_protocol_state") != "ready_for_freeze",
        freeze.get("test_execution_enabled") is not True,
        protocol.get("checkpoint_policy", {}).get("geometry_frozen") is not True,
        "default" in str(
            protocol.get("checkpoint_policy", {}).get("geometry_status", "")
        ).lower(),
        "default" in str(
            protocol.get("simulation", {}).get("prediction_geometry_claim", "")
        ).lower(),
    )):
        raise ValueError("protocol is not explicitly ready for freeze")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_freeze_artifacts(
    *,
    checkpoint_path: Path,
    training_manifest_path: Path,
    selection_trace_path: Path,
    environment_lock_path: Path,
    v2_lock_path: Path,
    dataset_root: Path,
    dataset_manifest_path: Path,
    execution_device: str,
    protocol: dict[str, Any],
    checkpoint_loader: Callable[[Path], Any] | None = None,
    environment_snapshot_provider: Callable[[str], dict[str, Any]] = (
        capture_runtime_environment
    ),
    model_validator: Callable[
        [dict[str, Any], Path, str], dict[str, Any]
    ] = strict_model_validation,
) -> dict[str, Any]:
    """Hash and execute the complete train/val-to-runtime artifact contract."""
    paths = {
        "checkpoint": checkpoint_path.resolve(),
        "training_manifest": training_manifest_path.resolve(),
        "selection_trace": selection_trace_path.resolve(),
        "environment_lock": environment_lock_path.resolve(),
        "v2_lock": v2_lock_path.resolve(),
        "dataset_manifest": dataset_manifest_path.resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"freeze artifacts are missing: {missing}")
    validate_protocol_implementation_contract(protocol)
    checkpoint_sha = sha256_file(paths["checkpoint"])
    training_sha = sha256_file(paths["training_manifest"])
    selection_sha = sha256_file(paths["selection_trace"])
    environment_sha = sha256_file(paths["environment_lock"])
    v2 = validate_v2_lock(paths["v2_lock"])
    dataset = audit_train_validation_dataset(
        dataset_root.resolve(),
        paths["dataset_manifest"],
        source_split_path=Path(v2["source_split_path"]),
        source_split_sha256=str(v2["source_split_sha256"]),
    )
    _, _, selection_metadata = validate_training_and_selection(
        training_manifest_path=paths["training_manifest"],
        selection_trace_path=paths["selection_trace"],
        checkpoint_path=paths["checkpoint"],
        v2=v2,
        dataset=dataset,
    )
    environment = verify_environment_lock(
        paths["environment_lock"],
        execution_device,
        snapshot_provider=environment_snapshot_provider,
    )
    if checkpoint_loader is None:
        import torch

        checkpoint_loader = lambda path: torch.load(  # noqa: E731
            path, map_location="cpu", weights_only=False)
    payload = checkpoint_loader(paths["checkpoint"])
    metadata = validate_v2_checkpoint_payload(
        payload,
        checkpoint_sha256=checkpoint_sha,
        v2=v2,
        dataset=dataset,
        selection_metadata=selection_metadata,
    )
    model_contract = model_validator(
        payload,
        Path(dataset["validation_observation_path"]),
        str(environment["execution_device"]),
    )
    return {
        "checkpoint_sha256": checkpoint_sha,
        "training_manifest_sha256": training_sha,
        "selection_trace_sha256": selection_sha,
        "environment_lock_sha256": environment_sha,
        "v2_lock_sha256": str(v2["lock_sha256"]),
        "dataset_manifest_sha256": str(dataset["dataset_manifest_sha256"]),
        "dataset_file_inventory_sha256": str(dataset["file_inventory_sha256"]),
        "training_source_closure_sha256": canonical_json_sha256(
            v2["locked_training_sources"]
        ),
        "checkpoint_contract": metadata,
        "selection_contract": selection_metadata,
        "runtime_model_contract": model_contract,
        "runtime_environment_contract": environment,
    }
