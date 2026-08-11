"""Matched, headless, three-robot reconstructed closed-loop experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import cv2
import numpy as np

from training_recovery.mso_recovery.common import stable_seed

from .consensus import TemporalCycleGate, se2_delta
from .events import EventWriter
from .hashing import array_sha256, canonical_json_sha256, sha256_file
from .mapping import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    PersistentMeasuredMap,
    blank_grid,
    crop_with_origin,
    make_private_frames,
    paste_probability,
    transform_point_xy,
    update_local_measurement,
    warp_probability,
    warp_semantic,
)
from .planner import plan_measured_safe
from .predictor import Predictor, validate_probability_output
from .registrar import ReferenceRegistrarAdapter, RegistrationAttempt
from .world import FloorplanRecord, HeadlessFloorplanWorld


ARM_POLICIES = {
    "observed_only": {"registration": False, "ranking": False, "mode": "observed"},
    "prediction_on": {"registration": True, "ranking": True, "mode": "model"},
    "prediction_registration_only": {
        "registration": True,
        "ranking": False,
        "mode": "model",
    },
    "prediction_frontier_only": {
        "registration": False,
        "ranking": True,
        "mode": "model",
    },
    "prediction_both": {"registration": True, "ranking": True, "mode": "model"},
    "oracle_completion": {"registration": True, "ranking": True, "mode": "oracle"},
    "adversarial_prediction": {
        "registration": True,
        "ranking": True,
        "mode": "adversarial",
    },
}


def matched_stream_seeds(
    protocol_sha256: str,
    floorplan_id: str,
    seed: int,
    stream_names: list[str],
) -> dict[str, int]:
    """Derive arm-invariant random streams for a paired run."""

    return {
        name: stable_seed(protocol_sha256, floorplan_id, int(seed), name)
        for name in stream_names
    }


def fixed_budget_coverage_endpoints(
    curve: list[tuple[float, float]],
    *,
    budget_m: float,
    threshold: float = 0.80,
) -> dict[str, Any]:
    """Compute fixed-budget endpoints without substituting a post-budget value.

    A point beyond the budget may only provide the far endpoint for linear
    interpolation exactly at the budget.  It is never clamped to the budget.
    """

    if not math.isfinite(budget_m) or budget_m <= 0.0:
        raise ValueError("distance budget must be finite and positive")
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("coverage threshold must be in (0, 1]")
    raw: list[tuple[float, float]] = []
    for distance, coverage in curve:
        distance = float(distance)
        coverage = float(coverage)
        if not math.isfinite(distance) or not math.isfinite(coverage):
            raise ValueError("coverage curve contains a non-finite point")
        if distance < 0.0 or not 0.0 <= coverage <= 1.0:
            raise ValueError("coverage curve point is outside its valid range")
        if raw and distance < raw[-1][0] - 1e-12:
            raise ValueError("coverage curve distance must be nondecreasing")
        if raw and math.isclose(distance, raw[-1][0], abs_tol=1e-12):
            raw[-1] = (distance, coverage)
        else:
            raw.append((distance, coverage))
    if not raw:
        raw = [(0.0, 0.0)]
    elif raw[0][0] > 0.0:
        raw.insert(0, (0.0, 0.0))

    restricted: list[tuple[float, float]] = []
    post_budget: tuple[float, float] | None = None
    for point in raw:
        if point[0] <= budget_m:
            restricted.append(point)
        else:
            post_budget = point
            break
    if not restricted:
        restricted = [(0.0, 0.0)]
    if restricted[-1][0] < budget_m:
        left_distance, left_coverage = restricted[-1]
        if post_budget is not None and post_budget[0] > left_distance:
            fraction = (budget_m - left_distance) / (
                post_budget[0] - left_distance
            )
            budget_coverage = left_coverage + fraction * (
                post_budget[1] - left_coverage
            )
        else:
            budget_coverage = left_coverage
        restricted.append((budget_m, float(budget_coverage)))

    reached: float | None = None
    for index, (distance, coverage) in enumerate(restricted):
        if coverage < threshold:
            continue
        if index == 0:
            reached = distance
        else:
            left_distance, left_coverage = restricted[index - 1]
            if coverage <= left_coverage + 1e-15:
                reached = distance
            else:
                fraction = (threshold - left_coverage) / (
                    coverage - left_coverage
                )
                reached = left_distance + fraction * (distance - left_distance)
        break

    x = np.asarray([point[0] for point in restricted], dtype=float)
    y = np.asarray([point[1] for point in restricted], dtype=float)
    auc = float(np.trapezoid(y, x) / budget_m)
    return {
        "coverage_auc_normalized": auc,
        "coverage_at_budget": float(restricted[-1][1]),
        "distance_to_80_percent_m": reached,
        "distance_to_80_percent_censored": reached is None,
        "restricted_distance_to_80_m": (
            float(reached) if reached is not None else float(budget_m)
        ),
        "distance_budget_m": float(budget_m),
        "budget_value_method": (
            "linear_interpolation" if post_budget is not None else "carry_forward"
        ),
    }


def assess_registration_transforms(
    *,
    relative_estimate: np.ndarray | None,
    relative_truth: np.ndarray,
    target_root_estimate: np.ndarray,
    target_root_truth: np.ndarray,
    source_root_actual: np.ndarray | None,
    source_root_truth: np.ndarray,
    resolution_m: float,
    translation_limit_m: float,
    yaw_limit_deg: float,
) -> dict[str, Any]:
    """Assess pairwise and actual root-frame transforms independently."""

    def assess(
        estimate: np.ndarray | None, truth: np.ndarray
    ) -> dict[str, Any]:
        if estimate is None:
            return {
                "translation_error_m": None,
                "absolute_yaw_error_deg": None,
                "within_limits": None,
            }
        translation, yaw = se2_delta(estimate, truth, resolution_m)
        return {
            "translation_error_m": translation,
            "absolute_yaw_error_deg": yaw,
            "within_limits": bool(
                translation <= translation_limit_m and yaw <= yaw_limit_deg
            ),
        }

    relative = assess(relative_estimate, relative_truth)
    target_root = assess(target_root_estimate, target_root_truth)
    root_candidate = (
        target_root_estimate @ relative_estimate
        if relative_estimate is not None else None
    )
    candidate = assess(root_candidate, source_root_truth)
    actual = assess(source_root_actual, source_root_truth)
    return {
        "relative": relative,
        "target_root": target_root,
        "root_candidate": candidate,
        "actual_source_root": actual,
        "cascade_from_incorrect_target_root": bool(
            relative["within_limits"] is True
            and target_root["within_limits"] is False
            and candidate["within_limits"] is False
        ),
        "root_candidate_transform": root_candidate,
    }


def dependency_cascade(
    parent_by_robot: dict[int, int | None], invalidated_robot: int
) -> list[int]:
    """Return a validated transform-dependency subtree in parent-first order."""

    if invalidated_robot not in parent_by_robot:
        raise ValueError("invalidated robot is absent from the dependency graph")
    if parent_by_robot[invalidated_robot] is None:
        raise ValueError("the persistent root transform cannot be invalidated")
    for robot_id, parent_id in parent_by_robot.items():
        if parent_id is not None and (
            parent_id not in parent_by_robot or parent_id == robot_id
        ):
            raise ValueError("transform dependency graph has an invalid parent")
        visited: set[int] = set()
        cursor: int | None = robot_id
        while cursor is not None:
            if cursor in visited:
                raise ValueError("transform dependency graph contains a cycle")
            visited.add(cursor)
            cursor = parent_by_robot[cursor]
    result: list[int] = []
    queue = [invalidated_robot]
    while queue:
        current = queue.pop(0)
        result.append(current)
        queue.extend(
            sorted(
                robot_id
                for robot_id, parent_id in parent_by_robot.items()
                if parent_id == current
            )
        )
    return result


@dataclass
class RobotState:
    robot_id: int
    local_from_world: np.ndarray
    measured_local: np.ndarray
    probability_local: np.ndarray
    probability_valid: np.ndarray
    root_from_local: np.ndarray | None = None
    root_transform_event_id: str | None = None
    root_transform_unsafe: bool = False
    root_parent_robot_id: int | None = None
    planned_route: list[tuple[int, int]] = field(default_factory=list)
    planned_route_index: int = 0
    plan_frame: str = "local"

    @property
    def connected(self) -> bool:
        return self.root_from_local is not None


@dataclass(frozen=True)
class EngineParameters:
    team_size: int
    maximum_ticks: int
    working_resolution_m: float
    laser_range_cells: int
    laser_rays: int
    encounter_range_m: float
    registration_period_ticks: int
    verification_period_ticks: int
    registration_canvas_px: int
    planning_canvas_px: int
    prediction_world_crop_px: int
    prediction_model_input_px: int
    planner_period_ticks: int
    prediction_period_ticks: int
    obstacle_inflation_m: float
    frontier_rank_radius_m: float
    prediction_rank_weight_m: float
    distance_budget_m: float
    stop_coverage: float

    @classmethod
    def from_protocol(
        cls, protocol: dict[str, Any], maximum_ticks: int | None = None
    ) -> "EngineParameters":
        simulation = protocol["simulation"]
        planning = protocol["planning"]
        return cls(
            team_size=int(protocol["experiment"]["team_size"]),
            maximum_ticks=int(
                simulation["maximum_ticks"] if maximum_ticks is None else maximum_ticks
            ),
            working_resolution_m=float(
                simulation["working_resolution_m_per_cell"]
            ),
            laser_range_cells=int(simulation["laser_range_cells"]),
            laser_rays=int(simulation["laser_rays"]),
            encounter_range_m=float(simulation["encounter_range_m"]),
            registration_period_ticks=int(simulation["registration_period_ticks"]),
            verification_period_ticks=int(simulation["verification_period_ticks"]),
            registration_canvas_px=int(simulation["registration_canvas_px"]),
            planning_canvas_px=int(simulation["planning_canvas_px"]),
            prediction_world_crop_px=int(
                round(
                    float(simulation["prediction_world_crop_side_m"])
                    / float(simulation["working_resolution_m_per_cell"])
                )
            ),
            prediction_model_input_px=int(simulation["prediction_model_input_px"]),
            planner_period_ticks=int(simulation["planner_period_ticks"]),
            prediction_period_ticks=int(simulation["prediction_period_ticks"]),
            obstacle_inflation_m=float(planning["obstacle_inflation_m"]),
            frontier_rank_radius_m=float(planning["frontier_rank_radius_m"]),
            prediction_rank_weight_m=float(planning["prediction_rank_weight_m"]),
            distance_budget_m=float(planning["distance_budget_m"]),
            stop_coverage=float(simulation["stop_coverage"]),
        )


class ClosedLoopRun:
    def __init__(
        self,
        *,
        protocol: dict[str, Any],
        protocol_path: Path,
        record: FloorplanRecord,
        split: str,
        arm: str,
        seed: int,
        predictor: Predictor,
        registrar: ReferenceRegistrarAdapter,
        output_dir: Path,
        maximum_ticks: int | None = None,
        runtime_environment_contract: dict[str, Any] | None = None,
    ) -> None:
        self.protocol = protocol
        self.protocol_path = protocol_path.resolve()
        self.protocol_sha256 = sha256_file(self.protocol_path)
        self.record = record
        self.split = split
        self.arm = arm
        if arm not in ARM_POLICIES:
            raise ValueError(f"unknown closed-loop arm: {arm!r}")
        self.arm_policy = ARM_POLICIES[arm]
        self.seed = int(seed)
        self.predictor = predictor
        self.registrar = registrar
        self.output_dir = output_dir.resolve()
        if split == "test" and runtime_environment_contract is None:
            raise ValueError("test runs require a locked runtime environment")
        self.runtime_environment_contract = runtime_environment_contract
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.parameters = EngineParameters.from_protocol(protocol, maximum_ticks)
        self.run_id = f"{record.floorplan_id}__seed{seed}__{arm}"
        self.stream_seeds = matched_stream_seeds(
            self.protocol_sha256,
            record.floorplan_id,
            seed,
            list(protocol["randomization"]["streams"]),
        )
        self.world = HeadlessFloorplanWorld(
            record,
            team_size=self.parameters.team_size,
            start_rng=np.random.default_rng(self.stream_seeds["team_starts"]),
            working_resolution_m=self.parameters.working_resolution_m,
            laser_range_cells=self.parameters.laser_range_cells,
            laser_rays=self.parameters.laser_rays,
            encounter_range_m=self.parameters.encounter_range_m,
        )
        frames, canvas_size, quadrants = make_private_frames(
            self.world.occupied.shape,
            self.parameters.team_size,
            np.random.default_rng(self.stream_seeds["private_frames"]),
        )
        self.canvas_size = canvas_size
        self.frame_quadrants = quadrants
        self.robots = [
            RobotState(
                robot_id=index,
                local_from_world=frames[index],
                measured_local=blank_grid(canvas_size),
                probability_local=np.full(
                    (canvas_size, canvas_size), 0.5, dtype=np.float32
                ),
                probability_valid=np.zeros(
                    (canvas_size, canvas_size), dtype=bool
                ),
                root_from_local=np.eye(3, dtype=float) if index == 0 else None,
            )
            for index in range(self.parameters.team_size)
        ]
        self.persistent = PersistentMeasuredMap(blank_grid(canvas_size))
        gate = protocol["gate"]
        self.consistency = TemporalCycleGate(
            resolution_m=self.parameters.working_resolution_m,
            consensus_count=int(gate["temporal_consensus_count"]),
            temporal_translation_limit_m=float(
                gate["temporal_translation_limit_m"]
            ),
            temporal_yaw_limit_deg=float(gate["temporal_yaw_limit_deg"]),
            cycle_translation_limit_m=float(gate["cycle_translation_limit_m"]),
            cycle_yaw_limit_deg=float(gate["cycle_absolute_yaw_limit_deg"]),
        )
        self.coverage_curve: list[tuple[float, float]] = []
        self.seen_coverage_curve: list[tuple[float, float]] = []
        self.erroneous_revisions: set[int] = set()
        self.registration_counts = {
            "encounter_triggered": 0,
            "encounter_out_of_range": 0,
            "candidate": 0,
            "gate_accept": 0,
            "gate_reject": 0,
            "temporal_ready": 0,
            "cycle_reject": 0,
            "safe_accept": 0,
            "false_accept": 0,
            "false_reject": 0,
            "true_reject": 0,
            "registration_commit": 0,
            "unsafe_registration_commit": 0,
            "recovered_unsafe_registration_commit": 0,
            "gate_correct_accept": 0,
            "gate_wrong_accept": 0,
            "gate_false_reject": 0,
            "gate_true_reject": 0,
            "final_decision_correct_accept": 0,
            "final_decision_false_accept": 0,
            "final_decision_false_reject": 0,
            "transform_invalidated_recovery": 0,
            "incorrect_candidate_safe_noncommit": 0,
            "hash_preserving_reject": 0,
            "dependency_cascade_invalidations": 0,
        }
        self.motion_counts = {
            "attempted": 0,
            "moved": 0,
            "collision_rejected": 0,
            "no_route": 0,
            "held": 0,
            "motion_stage_robot_ticks": 0,
            "stuck_episode_count": 0,
        }
        self.planning_counts = {"replans": 0, "route_available": 0, "no_route": 0}
        self._ticks_without_team_motion = 0
        self._inside_stuck_episode = False
        self.writer = EventWriter(
            self.output_dir / "events.jsonl",
            run_id=self.run_id,
            arm=arm,
            split=split,
            seed=seed,
            building_id=record.building_id,
            floorplan_id=record.floorplan_id,
            protocol_sha256=self.protocol_sha256,
        )
        self.initial_state_manifest = {
            "team_start_yx": [pose.tolist() for pose in self.world.poses],
            "private_frames": [robot.local_from_world.tolist()
                               for robot in self.robots],
            "measured_local_sha256": [array_sha256(robot.measured_local)
                                      for robot in self.robots],
            "persistent_map_sha256": self.persistent.sha256,
        }
        self.initial_state_sha256 = canonical_json_sha256(
            self.initial_state_manifest)
        self._write_run_manifest()

    def _write_run_manifest(self) -> None:
        software_path = self.protocol_path.with_name("SOFTWARE_SHA256SUMS.json")
        manifest = {
            "status": "new_reconstructed_experiment_not_historical",
            "run_id": self.run_id,
            "split": self.split,
            "arm": self.arm,
            "seed": self.seed,
            "building_id": self.record.building_id,
            "floorplan_id": self.record.floorplan_id,
            "source_relpath": self.record.source_relpath,
            "source_sha256": self.record.source_sha256,
            "protocol_sha256": self.protocol_sha256,
            "software_manifest_sha256": (
                sha256_file(software_path) if software_path.is_file() else None
            ),
            "parameters": asdict(self.parameters),
            "random_stream_seeds": self.stream_seeds,
            "random_stream_manifest_sha256": canonical_json_sha256(self.stream_seeds),
            "initial_state_sha256": self.initial_state_sha256,
            "predictor": {
                "name": self.predictor.name,
                "artifact_sha256": self.predictor.artifact_sha256,
            },
            "registrar": self.registrar.name,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
            },
            "runtime_environment_contract": self.runtime_environment_contract,
            "truth_boundary": {
                "world_truth_consumers": [
                    "sensor simulation",
                    "motion collision simulation",
                    "post-decision evaluator"
                    ,
                    "persistent-map quality evaluator",
                    "stopping controller"
                ],
                "world_truth_forbidden_consumers": [
                    "predictor",
                    "registration gate",
                    "temporal/cycle gate",
                    "persistent commit",
                    "frontier planner"
                ]
            },
        }
        (self.output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _local_pose_xy(self, robot: RobotState) -> tuple[float, float]:
        world_y, world_x = self.world.poses[robot.robot_id]
        return transform_point_xy(
            robot.local_from_world, (float(world_x), float(world_y))
        )

    def _scan_all(self, tick: int) -> None:
        for robot in self.robots:
            pose_before = self.world.poses[robot.robot_id].copy()
            newly_known = self.world.scan(robot.robot_id)
            changed = update_local_measurement(
                robot.measured_local,
                robot.local_from_world,
                self.world.occupied,
                newly_known,
            )
            ys, xs = np.nonzero(newly_known)
            values = np.where(
                self.world.occupied[ys, xs], OCCUPIED_VALUE, FREE_VALUE
            ).astype(np.int32)
            delta = (
                np.stack((ys, xs, values), axis=1).astype(np.int32)
                if len(ys)
                else np.empty((0, 3), dtype=np.int32)
            )
            self.writer.emit(
                "scan",
                tick,
                inputs={
                    "pose_yx": canonical_json_sha256(pose_before.tolist()),
                    "sensor_contract": canonical_json_sha256(
                        {
                            "laser_range_cells": self.parameters.laser_range_cells,
                            "laser_rays": self.parameters.laser_rays,
                        }
                    ),
                },
                outputs={"measured_delta": array_sha256(delta)},
                outcome="measured",
                robot_id=robot.robot_id,
                newly_known_cells=int(len(ys)),
                local_cells_changed=changed,
                ground_truth_consumer="sensor_only",
            )

    def _prediction_crop(
        self, robot: RobotState
    ) -> tuple[np.ndarray, tuple[int, int]]:
        return crop_with_origin(
            robot.measured_local,
            self._local_pose_xy(robot),
            self.parameters.prediction_world_crop_px,
            UNKNOWN_VALUE,
        )

    def _update_predictions(self, tick: int) -> None:
        for robot in self.robots:
            measured_crop, origin = self._prediction_crop(robot)
            mode = str(self.arm_policy["mode"])
            if mode == "oracle":
                semantic_world = np.where(
                    self.world.occupied, OCCUPIED_VALUE, FREE_VALUE
                ).astype(np.uint8)
                semantic_local = warp_semantic(
                    semantic_world,
                    robot.local_from_world,
                    robot.measured_local.shape,
                )
                truth_crop, truth_origin = crop_with_origin(
                    semantic_local,
                    self._local_pose_xy(robot),
                    self.parameters.prediction_world_crop_px,
                    UNKNOWN_VALUE,
                )
                if truth_origin != origin:
                    raise AssertionError("oracle and measured crop origins differ")
                prediction = np.where(
                    truth_crop == OCCUPIED_VALUE,
                    1.0,
                    np.where(truth_crop == FREE_VALUE, 0.0, 0.5),
                ).astype(np.float32)
            else:
                prediction = validate_probability_output(
                    self.predictor.predict(measured_crop), measured_crop.shape
                )
                if mode == "adversarial":
                    prediction = 1.0 - np.roll(prediction, shift=(37, -29), axis=(0, 1))
                    prediction = validate_probability_output(
                        prediction.astype(np.float32), measured_crop.shape
                    )
            paste_probability(
                robot.probability_local,
                robot.probability_valid,
                prediction,
                origin,
            )
            self.writer.emit(
                "observed_control" if mode == "observed" else "predictor",
                tick,
                inputs={"measured_crop": array_sha256(measured_crop)},
                outputs={"occupied_probability": array_sha256(prediction)},
                outcome="provisional_only",
                robot_id=robot.robot_id,
                crop_origin_xy=list(origin),
                world_crop_shape=list(measured_crop.shape),
                model_input_shape=[
                    self.parameters.prediction_model_input_px,
                    self.parameters.prediction_model_input_px,
                ],
                predictor_name=self.predictor.name,
                arm_prediction_mode=mode,
                persistent_write_allowed=False,
            )

    def _registration_crop(
        self, robot: RobotState
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        observed, origin = crop_with_origin(
            robot.measured_local,
            self._local_pose_xy(robot),
            self.parameters.registration_canvas_px,
            UNKNOWN_VALUE,
        )
        if not bool(self.arm_policy["registration"]):
            return observed.copy(), observed, origin
        probability, probability_origin = crop_with_origin(
            robot.probability_local,
            self._local_pose_xy(robot),
            self.parameters.registration_canvas_px,
            0.5,
        )
        valid, valid_origin = crop_with_origin(
            robot.probability_valid,
            self._local_pose_xy(robot),
            self.parameters.registration_canvas_px,
            False,
        )
        if probability_origin != origin or valid_origin != origin:
            raise AssertionError("registration crops do not share an origin")
        registrar_input = observed.copy()
        unknown = np.abs(observed.astype(np.int16) - UNKNOWN_VALUE) <= 3
        use = unknown & valid
        registrar_input[use] = np.rint(probability[use] * 255.0).astype(np.uint8)
        return registrar_input, observed, origin

    def _state_snapshot(self) -> tuple[int, str]:
        return self.persistent.revision, self.persistent.sha256

    def _truth_root_from_local(self, robot_id: int) -> np.ndarray:
        return self.robots[0].local_from_world @ np.linalg.inv(
            self.robots[robot_id].local_from_world)

    def _root_pose_error(self, robot_id: int) -> dict[str, Any]:
        estimate = self.robots[robot_id].root_from_local
        if estimate is None:
            return {
                "connected": False,
                "translation_error_m": None,
                "absolute_yaw_error_deg": None,
                "within_limits": None,
            }
        translation, yaw = se2_delta(
            estimate, self._truth_root_from_local(robot_id),
            self.parameters.working_resolution_m)
        limits = self.protocol["safety_endpoints"]["registration_correctness"]
        return {
            "connected": True,
            "translation_error_m": translation,
            "absolute_yaw_error_deg": yaw,
            "within_limits": bool(
                translation <= float(limits["translation_m"])
                and yaw <= float(limits["absolute_yaw_deg"])),
        }

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    def _persistent_map_quality(self) -> dict[str, Any]:
        semantic_world = np.where(
            self.world.occupied, OCCUPIED_VALUE, FREE_VALUE).astype(np.uint8)
        truth_root = warp_semantic(
            semantic_world,
            self.robots[0].local_from_world,
            self.persistent.grid.shape)
        adjacent = cv2.dilate(
            self.world.component.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        evaluation_world = self.world.component | (
            self.world.occupied & adjacent
        )
        valid = cv2.warpPerspective(
            evaluation_world.astype(np.uint8),
            self.robots[0].local_from_world,
            (self.persistent.grid.shape[1], self.persistent.grid.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        predicted_known = self.persistent.grid != UNKNOWN_VALUE
        evaluation = valid & predicted_known
        truth_free = valid & (truth_root == FREE_VALUE)
        truth_occupied = valid & (truth_root == OCCUPIED_VALUE)
        predicted_free = self.persistent.grid == FREE_VALUE
        predicted_occupied = self.persistent.grid == OCCUPIED_VALUE
        free_true_positive = int(np.count_nonzero(predicted_free & truth_free))
        occupied_true_positive = int(
            np.count_nonzero(predicted_occupied & truth_occupied))
        incorrect_in_domain = int(np.count_nonzero(
            evaluation & (self.persistent.grid != truth_root)))
        off_domain_known = int(np.count_nonzero(predicted_known & ~valid))
        contamination = incorrect_in_domain + off_domain_known
        valid_cells = int(np.count_nonzero(valid))
        known_cells = int(np.count_nonzero(evaluation))
        correct_known_cells = known_cells - incorrect_in_domain
        return {
            "valid_truth_cells": valid_cells,
            "known_cells": known_cells,
            "known_cell_coverage": self._ratio(known_cells, valid_cells),
            "correct_known_cells": correct_known_cells,
            "correct_known_cell_coverage": self._ratio(
                correct_known_cells, valid_cells),
            "semantic_accuracy": self._ratio(
                correct_known_cells, known_cells),
            "free_precision": self._ratio(
                free_true_positive, int(np.count_nonzero(predicted_free))),
            "free_recall": self._ratio(
                free_true_positive, int(np.count_nonzero(truth_free))),
            "occupied_precision": self._ratio(
                occupied_true_positive,
                int(np.count_nonzero(predicted_occupied))),
            "occupied_recall": self._ratio(
                occupied_true_positive, int(np.count_nonzero(truth_occupied))),
            "incorrect_in_domain_cells": incorrect_in_domain,
            "off_domain_known_cells": off_domain_known,
            "contaminated_cells": contamination,
            "contaminated_area_m2": contamination
            * self.parameters.working_resolution_m ** 2,
            "revision": self.persistent.revision,
            "persistent_map_sha256": self.persistent.sha256,
        }

    def _rebuilt_connected_map(self) -> np.ndarray:
        rebuilt = PersistentMeasuredMap(blank_grid(self.canvas_size))
        for robot in self.robots:
            if not robot.connected:
                continue
            warped = (
                robot.measured_local if robot.robot_id == 0 else warp_semantic(
                    robot.measured_local, robot.root_from_local,
                    rebuilt.grid.shape))
            rebuilt.apply(warped, True)
        return rebuilt.grid

    def _recover_robot(
        self,
        tick: int,
        robot: RobotState,
        *,
        event_id: str,
        parent_sequence: int,
        transaction_target_id: int,
    ) -> dict[str, Any]:
        if robot.root_from_local is None:
            raise ValueError("cannot recover a disconnected transform")
        parents = {
            state.robot_id: state.root_parent_robot_id
            for state in self.robots
        }
        invalidated_ids = dependency_cascade(parents, robot.robot_id)
        invalidated_states = [self.robots[robot_id] for robot_id in invalidated_ids]
        transform_hashes = {
            str(state.robot_id): self._matrix_hash(state.root_from_local)
            for state in invalidated_states
        }
        invalidated_event_ids = {
            str(state.robot_id): state.root_transform_event_id
            for state in invalidated_states
        }
        unsafe_ids = [
            state.robot_id for state in invalidated_states
            if state.root_transform_unsafe
        ]
        dependency_edges = [
            [state.robot_id, state.root_parent_robot_id]
            for state in invalidated_states
        ]
        for state in invalidated_states:
            state.root_from_local = None
            state.root_transform_event_id = None
            state.root_transform_unsafe = False
            state.root_parent_robot_id = None
            state.plan_frame = "local"
            state.planned_route = []
            state.planned_route_index = 0
        rebuilt = self._rebuilt_connected_map()
        result = self.persistent.replace_with_rebuild(rebuilt)
        for robot_id in invalidated_ids:
            self.consistency.reset((0, robot_id))
        self.registration_counts["transform_invalidated_recovery"] += len(
            invalidated_ids
        )
        self.registration_counts["dependency_cascade_invalidations"] += max(
            0, len(invalidated_ids) - 1
        )
        self.registration_counts[
            "recovered_unsafe_registration_commit"
        ] += len(unsafe_ids)
        return self.writer.emit(
            "recovery",
            tick,
            inputs={
                "invalidated_transforms": canonical_json_sha256(transform_hashes),
                "invalidated_persistent_map": str(result["hash_before"]),
            },
            outputs={"rebuilt_persistent_map": str(result["hash_after"])},
            map_revision_before=int(result["revision_before"]),
            map_revision_after=int(result["revision_after"]),
            persistent_hash_before=str(result["hash_before"]),
            persistent_hash_after=str(result["hash_after"]),
            outcome="transform_invalidated_and_map_rebuilt",
            event_id=event_id,
            parent_sequence=parent_sequence,
            source_robot=robot.robot_id,
            target_robot=transaction_target_id,
            persistent_root_robot=0,
            invalidated_robot_ids=invalidated_ids,
            invalidated_transform_event_ids=invalidated_event_ids,
            invalidated_unsafe_robot_ids=unsafe_ids,
            invalidated_dependency_edges=dependency_edges,
            changed_cells=int(result["changed_cells"]),
            ground_truth_used_for_recovery=False,
        )

    def _emit_noncommit(
        self,
        tick: int,
        *,
        source_id: int,
        target_id: int,
        reason: str,
        input_hashes: dict[str, str],
        event_id: str,
        parent_sequence: int,
    ) -> dict[str, Any]:
        revision, state = self._state_snapshot()
        self.registration_counts["hash_preserving_reject"] += 1
        return self.writer.emit(
            "commit",
            tick,
            inputs=input_hashes,
            outputs={},
            map_revision_before=revision,
            map_revision_after=revision,
            persistent_hash_before=state,
            persistent_hash_after=state,
            outcome=reason,
            source_robot=source_id,
            target_robot=target_id,
            changed_cells=0,
            measured_cells_only=True,
            event_id=event_id,
            parent_sequence=parent_sequence,
        )

    def _commit_robot(
        self,
        tick: int,
        robot: RobotState,
        reason: str,
        *,
        event_id: str | None = None,
        parent_sequence: int | None = None,
        transaction_target_id: int = 0,
    ) -> dict[str, Any]:
        if robot.root_from_local is None:
            raise ValueError("cannot commit a disconnected robot")
        warped = (
            robot.measured_local
            if robot.robot_id == 0
            else warp_semantic(
                robot.measured_local,
                robot.root_from_local,
                self.persistent.grid.shape,
            )
        )
        measured_hash = array_sha256(robot.measured_local)
        warped_hash = array_sha256(warped)
        result = self.persistent.apply(warped, True)
        return self.writer.emit(
            "commit",
            tick,
            inputs={
                "measured_local": measured_hash,
                "accepted_transform": canonical_json_sha256(
                    robot.root_from_local.tolist()
                ),
            },
            outputs={"warped_measured": warped_hash},
            map_revision_before=int(result["revision_before"]),
            map_revision_after=int(result["revision_after"]),
            persistent_hash_before=str(result["hash_before"]),
            persistent_hash_after=str(result["hash_after"]),
            outcome="committed",
            source_robot=robot.robot_id,
            target_robot=transaction_target_id,
            persistent_root_robot=0,
            changed_cells=int(result["changed_cells"]),
            commit_reason=reason,
            measured_cells_only=True,
            prediction_persisted=False,
            event_id=event_id,
            parent_sequence=parent_sequence,
        )

    def _sync_connected_measurements(
        self, tick: int, reason: str = "periodic_measured_sync"
    ) -> None:
        for robot in self.robots:
            if robot.connected:
                self._commit_robot(tick, robot, reason)

    def _select_encounter_target(self, source_id: int) -> int | None:
        candidates = [
            robot.robot_id
            for robot in self.robots
            if robot.connected
            and robot.robot_id != source_id
            and self.world.encounter(
                robot.robot_id, source_id, self.parameters.encounter_range_m
            )
        ]
        return min(candidates) if candidates else None

    @staticmethod
    def _matrix_hash(matrix: np.ndarray | None) -> str:
        return (
            "none"
            if matrix is None
            else array_sha256(np.asarray(matrix, dtype=np.float64))
        )

    def _evaluate_registration(
        self,
        tick: int,
        target_id: int,
        source_id: int,
        attempt: RegistrationAttempt,
        committed: bool,
        *,
        event_id: str,
        parent_sequence: int,
        root_candidate: np.ndarray | None,
    ) -> dict[str, Any]:
        gate = self.protocol["safety_endpoints"]["registration_correctness"]
        relative_truth = (
            self.robots[target_id].local_from_world @ np.linalg.inv(
                self.robots[source_id].local_from_world
            )
        )
        target_root_estimate = self.robots[target_id].root_from_local
        if target_root_estimate is None:
            raise AssertionError("registration target is disconnected at evaluation")
        assessment = assess_registration_transforms(
            relative_estimate=attempt.full_transform_target_from_source,
            relative_truth=relative_truth,
            target_root_estimate=target_root_estimate,
            target_root_truth=self._truth_root_from_local(target_id),
            source_root_actual=self.robots[source_id].root_from_local,
            source_root_truth=self._truth_root_from_local(source_id),
            resolution_m=self.parameters.working_resolution_m,
            translation_limit_m=float(gate["translation_m"]),
            yaw_limit_deg=float(gate["absolute_yaw_deg"]),
        )
        computed_candidate = assessment.pop("root_candidate_transform")
        if root_candidate is None:
            if computed_candidate is not None:
                raise AssertionError("root candidate was dropped before evaluation")
        elif computed_candidate is None or not np.allclose(
            root_candidate, computed_candidate, atol=1e-9, rtol=0.0
        ):
            raise AssertionError("evaluated root candidate differs from the decision")
        relative_translation = assessment["relative"]["translation_error_m"]
        relative_yaw = assessment["relative"]["absolute_yaw_error_deg"]
        relative_correct = assessment["relative"]["within_limits"]
        root_candidate_translation = assessment["root_candidate"][
            "translation_error_m"
        ]
        root_candidate_yaw = assessment["root_candidate"][
            "absolute_yaw_error_deg"
        ]
        root_candidate_correct = assessment["root_candidate"]["within_limits"]
        source_root = {
            "connected": self.robots[source_id].connected,
            **assessment["actual_source_root"],
        }
        target_root = {"connected": True, **assessment["target_root"]}
        committed_root_correct = (
            source_root["within_limits"] if committed else None)
        cascade_from_target = assessment["cascade_from_incorrect_target_root"]
        map_quality = self._persistent_map_quality()
        if map_quality["contaminated_cells"] > 0:
            self.erroneous_revisions.add(self.persistent.revision)
        self.writer.emit(
            "post_decision_evaluation",
            tick,
            inputs={
                "candidate_transform": self._matrix_hash(
                    attempt.full_transform_target_from_source
                ),
                "root_candidate": self._matrix_hash(root_candidate),
                "persistent_map": self.persistent.sha256,
            },
            outputs={
                "evaluation": canonical_json_sha256({
                    "relative_translation_m": relative_translation,
                    "relative_absolute_yaw_deg": relative_yaw,
                    "root_candidate_translation_m": root_candidate_translation,
                    "root_candidate_absolute_yaw_deg": root_candidate_yaw,
                    "source_root_pose": source_root,
                    "target_root_pose": target_root,
                    "persistent_map_quality": map_quality,
                })
            },
            outcome=(
                "committed_root_correct" if committed_root_correct is True
                else "committed_root_incorrect" if committed_root_correct is False
                else "safe_noncommit"),
            event_id=event_id,
            parent_sequence=parent_sequence,
            source_robot=source_id,
            target_robot=target_id,
            relative_translation_error_m=relative_translation,
            relative_absolute_yaw_error_deg=relative_yaw,
            relative_candidate_correct=relative_correct,
            root_candidate_translation_error_m=root_candidate_translation,
            root_candidate_absolute_yaw_error_deg=root_candidate_yaw,
            root_candidate_correct=root_candidate_correct,
            source_root_pose=source_root,
            target_root_pose=target_root,
            cascade_from_incorrect_target_root=cascade_from_target,
            persistent_map_quality=map_quality,
            erroneous_revision=map_quality["contaminated_cells"] > 0,
            committed=committed,
            committed_root_correct=committed_root_correct,
            evaluator_only=True,
            available_to_gate=False,
            available_to_planner=False,
        )
        return {
            "relative_correct": relative_correct,
            "root_candidate_correct": root_candidate_correct,
            "committed_root_correct": committed_root_correct,
            "cascade_from_incorrect_target_root": cascade_from_target,
            "persistent_map_quality": map_quality,
        }

    def _count_registration_decision(
        self,
        attempt: RegistrationAttempt,
        evaluation: dict[str, Any],
        *,
        committed: bool,
        was_connected: bool,
    ) -> None:
        relative_correct = evaluation["relative_correct"]
        if attempt.gate_accepted:
            key = "gate_correct_accept" if relative_correct is True else (
                "gate_wrong_accept" if relative_correct is False else None)
            if key:
                self.registration_counts[key] += 1
            legacy = "safe_accept" if relative_correct is True else (
                "false_accept" if relative_correct is False else None)
            if legacy:
                self.registration_counts[legacy] += 1
        else:
            key = "gate_false_reject" if relative_correct is True else (
                "gate_true_reject" if relative_correct is False else None)
            if key:
                self.registration_counts[key] += 1
            if relative_correct is True:
                self.registration_counts["false_reject"] += 1
            elif relative_correct is False:
                self.registration_counts["true_reject"] += 1
        if not committed and relative_correct is False:
            self.registration_counts["incorrect_candidate_safe_noncommit"] += 1
        if committed:
            if evaluation["committed_root_correct"] is True:
                self.registration_counts["final_decision_correct_accept"] += 1
            elif evaluation["committed_root_correct"] is False:
                self.registration_counts["final_decision_false_accept"] += 1
                self.registration_counts["unsafe_registration_commit"] += 1
        elif not was_connected and evaluation["root_candidate_correct"] is True:
            self.registration_counts["final_decision_false_reject"] += 1

    def _attempt_registration(
        self,
        tick: int,
        target_id: int,
        source_id: int,
        encounter_event: dict[str, Any],
    ) -> None:
        target = self.robots[target_id]
        source = self.robots[source_id]
        target_input, target_observed, target_origin = self._registration_crop(target)
        source_input, source_observed, source_origin = self._registration_crop(source)
        event_id = (
            f"{self.record.floorplan_id}:{self.seed}:{tick}:"
            f"{target_id}:{source_id}"
        )
        attempt = self.registrar.register(
            event_id=event_id,
            target_input=target_input,
            source_input=source_input,
            target_observed=target_observed,
            source_observed=source_observed,
            target_origin_xy=target_origin,
            source_origin_xy=source_origin,
        )
        if attempt.candidate_returned:
            self.registration_counts["candidate"] += 1
        registration_event = self.writer.emit(
            "registration",
            tick,
            inputs=attempt.input_hashes,
            outputs={
                "crop_transform": self._matrix_hash(
                    attempt.crop_transform_target_from_source
                ),
                "full_transform": self._matrix_hash(
                    attempt.full_transform_target_from_source
                ),
            },
            outcome="candidate" if attempt.candidate_returned else "no_candidate",
            source_robot=source_id,
            target_robot=target_id,
            registrar=self.registrar.name,
            diagnostics=attempt.diagnostics,
            ground_truth_used=False,
            event_id=event_id,
            parent_sequence=int(encounter_event["sequence"]),
        )
        gate_outcome = "accepted" if attempt.gate_accepted else "rejected"
        self.registration_counts[
            "gate_accept" if attempt.gate_accepted else "gate_reject"
        ] += 1
        gate_event = self.writer.emit(
            "gate",
            tick,
            inputs={
                "candidate_transform": self._matrix_hash(
                    attempt.full_transform_target_from_source
                ),
                "target_observed": attempt.input_hashes[
                    "target_observed_gate_input"
                ],
                "source_observed": attempt.input_hashes[
                    "source_observed_gate_input"
                ],
            },
            outputs={"gate_decision": canonical_json_sha256(gate_outcome)},
            outcome=gate_outcome,
            source_robot=source_id,
            target_robot=target_id,
            reject_reason=attempt.reject_reason,
            gate_score=attempt.diagnostics.get("gate_score"),
            gate_threshold=attempt.diagnostics.get("gate_threshold"),
            inlier_count=attempt.diagnostics.get("inlier_count"),
            inlier_ratio=attempt.diagnostics.get("inlier_ratio"),
            ground_truth_used=False,
            event_id=event_id,
            parent_sequence=int(registration_event["sequence"]),
        )
        if target.root_from_local is None:
            raise AssertionError("registration target is not connected to root")
        root_candidate = (
            target.root_from_local @ attempt.full_transform_target_from_source
            if attempt.full_transform_target_from_source is not None else None)
        was_connected = source.connected
        if (
            not attempt.gate_accepted
            or attempt.full_transform_target_from_source is None
        ):
            commit_event = self._emit_noncommit(
                tick,
                source_id=source_id,
                target_id=target_id,
                reason="gate_rejected",
                input_hashes=attempt.input_hashes,
                event_id=event_id,
                parent_sequence=int(gate_event["sequence"]),
            )
            evaluation = self._evaluate_registration(
                tick, target_id, source_id, attempt, False,
                event_id=event_id,
                parent_sequence=int(commit_event["sequence"]),
                root_candidate=root_candidate,
            )
            self._count_registration_decision(
                attempt, evaluation, committed=False,
                was_connected=was_connected)
            return

        if root_candidate is None:
            raise AssertionError("accepted registration lacks a root candidate")
        decision = self.consistency.observe(
            (0, source_id), root_candidate, source.root_from_local
        )
        consistency_event = self.writer.emit(
            "temporal_cycle_consistency",
            tick,
            inputs={"root_candidate": self._matrix_hash(root_candidate)},
            outputs={
                "consensus_transform": self._matrix_hash(decision.transform)
            },
            outcome=decision.reason,
            source_robot=source_id,
            target_robot=target_id,
            accepted=decision.accepted,
            ready=decision.ready,
            temporal_count=decision.temporal_count,
            maximum_translation_delta_m=decision.maximum_translation_delta_m,
            maximum_yaw_delta_deg=decision.maximum_yaw_delta_deg,
            ground_truth_used=False,
            event_id=event_id,
            parent_sequence=int(gate_event["sequence"]),
        )
        if not decision.accepted or (not decision.ready and not was_connected):
            if decision.reason == "cycle_inconsistent_consensus":
                self.registration_counts["cycle_reject"] += 1
            if (
                    was_connected and decision.ready
                    and decision.reason == "cycle_inconsistent_consensus"):
                terminal_event = self._recover_robot(
                    tick, source, event_id=event_id,
                    parent_sequence=int(consistency_event["sequence"]),
                    transaction_target_id=target_id)
            else:
                terminal_event = self._emit_noncommit(
                    tick,
                    source_id=source_id,
                    target_id=target_id,
                    reason=decision.reason,
                    input_hashes={
                        "root_candidate": self._matrix_hash(root_candidate)},
                    event_id=event_id,
                    parent_sequence=int(consistency_event["sequence"]),
                )
            evaluation = self._evaluate_registration(
                tick, target_id, source_id, attempt, False,
                event_id=event_id,
                parent_sequence=int(terminal_event["sequence"]),
                root_candidate=root_candidate)
            self._count_registration_decision(
                attempt, evaluation, committed=False,
                was_connected=was_connected)
            return

        if not was_connected:
            if decision.transform is None:
                raise AssertionError("ready consensus did not return a transform")
            source.root_from_local = decision.transform.copy()
            source.root_transform_event_id = event_id
            source.root_parent_robot_id = target_id
            self.registration_counts["temporal_ready"] += 1
            terminal_event = self._commit_robot(
                tick, source, "registration_temporal_consensus",
                event_id=event_id,
                parent_sequence=int(consistency_event["sequence"]),
                transaction_target_id=target_id)
            self.registration_counts["registration_commit"] += 1
            committed = True
        else:
            terminal_event = self._emit_noncommit(
                tick,
                source_id=source_id,
                target_id=target_id,
                reason="cycle_verified_no_transform_change",
                input_hashes={"root_candidate": self._matrix_hash(root_candidate)},
                event_id=event_id,
                parent_sequence=int(consistency_event["sequence"]),
            )
            committed = False
        evaluation = self._evaluate_registration(
            tick, target_id, source_id, attempt, committed,
            event_id=event_id,
            parent_sequence=int(terminal_event["sequence"]),
            root_candidate=root_candidate)
        if committed:
            source.root_transform_unsafe = (
                evaluation["committed_root_correct"] is False)
        self._count_registration_decision(
            attempt, evaluation, committed=committed,
            was_connected=was_connected)

    def _registration_stage(self, tick: int) -> None:
        for source in self.robots[1:]:
            due = (
                tick % self.parameters.registration_period_ticks == 0
                if not source.connected
                else tick % self.parameters.verification_period_ticks == 0
            )
            if not due:
                continue
            target_id = self._select_encounter_target(source.robot_id)
            if target_id is None:
                self.registration_counts["encounter_out_of_range"] += 1
                revision, state = self._state_snapshot()
                self.writer.emit(
                    "encounter",
                    tick,
                    inputs={"encounter_contract": canonical_json_sha256({
                        "deterministic": True,
                        "range_m": self.parameters.encounter_range_m,
                    })},
                    outputs={},
                    map_revision_before=revision,
                    map_revision_after=revision,
                    persistent_hash_before=state,
                    persistent_hash_after=state,
                    outcome="out_of_range",
                    source_robot=source.robot_id,
                    connected=source.connected,
                    ground_truth_consumer="range_sensor_only",
                )
                continue
            self.registration_counts["encounter_triggered"] += 1
            encounter_event = self.writer.emit(
                "encounter",
                tick,
                inputs={
                    "encounter_contract": canonical_json_sha256({
                        "deterministic": True,
                        "range_m": self.parameters.encounter_range_m,
                    })
                },
                outputs={
                    "pair": canonical_json_sha256([target_id, source.robot_id])
                },
                outcome="triggered",
                source_robot=source.robot_id,
                target_robot=target_id,
                connected=source.connected,
                ground_truth_consumer="range_sensor_only",
                event_id=(
                    f"{self.record.floorplan_id}:{self.seed}:{tick}:"
                    f"{target_id}:{source.robot_id}"),
            )
            self._attempt_registration(
                tick, target_id, source.robot_id, encounter_event)

    def _rank_crop(
        self,
        robot: RobotState,
        *,
        plan_frame: str,
        origin_xy: tuple[int, int],
        size: int,
    ) -> np.ndarray:
        if not bool(self.arm_policy["ranking"]):
            return np.full((size, size), 0.5, dtype=np.float32)
        if plan_frame == "local":
            probability, probability_origin = crop_with_origin(
                robot.probability_local,
                (origin_xy[0] + size / 2.0, origin_xy[1] + size / 2.0),
                size,
                0.5,
            )
            if probability_origin != origin_xy:
                raise AssertionError("local rank crop origin changed")
            return probability

        output = np.full((size, size), 0.5, dtype=np.float32)
        confidence = np.zeros((size, size), dtype=np.float32)
        crop_from_root = np.array(
            [[1, 0, -origin_xy[0]], [0, 1, -origin_xy[1]], [0, 0, 1]],
            dtype=float,
        )
        for source in self.robots:
            if source.root_from_local is None:
                continue
            crop_from_source = crop_from_root @ source.root_from_local
            probability, valid = warp_probability(
                source.probability_local,
                source.probability_valid,
                crop_from_source,
                (size, size),
            )
            candidate_confidence = np.abs(probability - 0.5)
            use = valid & (candidate_confidence > confidence)
            output[use] = probability[use]
            confidence[use] = candidate_confidence[use]
        return output

    def _plan_robot(self, tick: int, robot: RobotState) -> None:
        local_pose = self._local_pose_xy(robot)
        if robot.root_from_local is None:
            measured_full = robot.measured_local
            start_xy = local_pose
            plan_frame = "local"
        else:
            measured_full = self.persistent.grid
            start_xy = transform_point_xy(robot.root_from_local, local_pose)
            plan_frame = "root"
        measured, origin = crop_with_origin(
            measured_full,
            start_xy,
            self.parameters.planning_canvas_px,
            UNKNOWN_VALUE,
        )
        rank = self._rank_crop(
            robot,
            plan_frame=plan_frame,
            origin_xy=origin,
            size=self.parameters.planning_canvas_px,
        )
        start_yx = (
            int(round(start_xy[1] - origin[1])),
            int(round(start_xy[0] - origin[0])),
        )
        result, route = plan_measured_safe(
            measured,
            start_yx,
            resolution_m=self.parameters.working_resolution_m,
            inflation_cells=max(
                0,
                int(
                    math.ceil(
                        self.parameters.obstacle_inflation_m
                        / self.parameters.working_resolution_m
                    )
                ),
            ),
            rank_probability=rank,
            rank_radius_cells=max(
                1,
                int(
                    round(
                        self.parameters.frontier_rank_radius_m
                        / self.parameters.working_resolution_m
                    )
                ),
            ),
            prediction_rank_weight_cells=(
                self.parameters.prediction_rank_weight_m
                / self.parameters.working_resolution_m
            ),
        )
        self.planning_counts["replans"] += 1
        self.planning_counts[
            "route_available" if result.route_available else "no_route"] += 1
        route_full = [(y + origin[1], x + origin[0]) for y, x in route]
        robot.planned_route = route_full
        robot.planned_route_index = 1 if len(route_full) > 1 else len(route_full)
        robot.plan_frame = plan_frame
        route_array = (
            np.asarray(route_full, dtype=np.int32)
            if route_full
            else np.empty((0, 2), dtype=np.int32)
        )
        self.writer.emit(
            "planner",
            tick,
            inputs={
                "measured_safe_grid": array_sha256(measured),
                "temporary_rank_probability": array_sha256(rank),
                "start_yx": canonical_json_sha256(start_yx),
            },
            outputs={"route": array_sha256(route_array)},
            map_revision_before=self.persistent.revision,
            map_revision_after=self.persistent.revision,
            persistent_hash_before=self.persistent.sha256,
            persistent_hash_after=self.persistent.sha256,
            outcome="route" if result.route_available else "no_route",
            robot_id=robot.robot_id,
            plan_frame=plan_frame,
            planner=result.to_dict(),
            candidate_set_measured_only=True,
            traversability_measured_only=True,
            prediction_use="ranking_only",
            ground_truth_used=False,
        )

    def _planning_stage(self, tick: int) -> None:
        if tick % self.parameters.planner_period_ticks != 0:
            return
        for robot in self.robots:
            self._plan_robot(tick, robot)

    def _route_waypoint_world(self, robot: RobotState) -> tuple[int, int] | None:
        while robot.planned_route_index < len(robot.planned_route):
            y, x = robot.planned_route[robot.planned_route_index]
            point_xy = (float(x), float(y))
            if robot.plan_frame == "root":
                if robot.root_from_local is None:
                    return None
                point_xy = transform_point_xy(
                    np.linalg.inv(robot.root_from_local), point_xy
                )
            world_xy = transform_point_xy(
                np.linalg.inv(robot.local_from_world), point_xy
            )
            target = (int(round(world_xy[1])), int(round(world_xy[0])))
            current = tuple(int(value) for value in self.world.poses[robot.robot_id])
            if target == current:
                robot.planned_route_index += 1
                continue
            return target
        return None

    @staticmethod
    def _one_cell_request(
        current_yx: tuple[int, int], target_yx: tuple[int, int]
    ) -> tuple[int, int]:
        dy = target_yx[0] - current_yx[0]
        dx = target_yx[1] - current_yx[1]
        if abs(dy) >= abs(dx) and dy:
            return current_yx[0] + int(np.sign(dy)), current_yx[1]
        if dx:
            return current_yx[0], current_yx[1] + int(np.sign(dx))
        return current_yx

    def _motion_stage(self, tick: int) -> None:
        moved_any = False
        for robot in self.robots:
            self.motion_counts["motion_stage_robot_ticks"] += 1
            target = self._route_waypoint_world(robot)
            current = tuple(int(value) for value in self.world.poses[robot.robot_id])
            if target is None:
                self.motion_counts["no_route"] += 1
                self.writer.emit(
                    "motion",
                    tick,
                    inputs={"pose_yx": canonical_json_sha256(current)},
                    outputs={"pose_yx": canonical_json_sha256(current)},
                    outcome="no_route",
                    robot_id=robot.robot_id,
                    moved=False,
                    collision=False,
                    ground_truth_consumer="motion_only",
                )
                continue
            request = self._one_cell_request(current, target)
            self.motion_counts["attempted"] += 1
            result = self.world.move(robot.robot_id, request)
            if result.collision:
                self.motion_counts["collision_rejected"] += 1
            if result.moved:
                moved_any = True
                self.motion_counts["moved"] += 1
                robot.planned_route_index += 1
            elif not result.collision:
                self.motion_counts["held"] += 1
            self.writer.emit(
                "motion",
                tick,
                inputs={
                    "pose_yx": canonical_json_sha256(current),
                    "next_planned_waypoint": canonical_json_sha256(target),
                },
                outputs={"pose_yx": canonical_json_sha256(result.final_yx)},
                outcome=(
                    "moved"
                    if result.moved
                    else "collision_rejected"
                    if result.collision
                    else "held"
                ),
                robot_id=robot.robot_id,
                requested_yx=list(result.requested_yx),
                final_yx=list(result.final_yx),
                moved=result.moved,
                collision=result.collision,
                distance_m=result.distance_m,
                planner_did_not_receive_truth=True,
                ground_truth_consumer="motion_only",
            )
        if moved_any:
            self._ticks_without_team_motion = 0
            self._inside_stuck_episode = False
        else:
            self._ticks_without_team_motion += 1
            window = int(self.protocol["simulation"]["stuck_window_ticks"])
            if self._ticks_without_team_motion >= window and not (
                    self._inside_stuck_episode):
                self.motion_counts["stuck_episode_count"] += 1
                self._inside_stuck_episode = True

    def _record_coverage(self, tick: int) -> float:
        quality = self._persistent_map_quality()
        coverage_value = quality["correct_known_cell_coverage"]
        coverage = float(coverage_value) if coverage_value is not None else 0.0
        seen_coverage = self.world.measured_coverage()
        distance = self.world.cumulative_team_distance_m()
        self.coverage_curve.append((distance, coverage))
        self.seen_coverage_curve.append((distance, seen_coverage))
        if int(quality["contaminated_cells"]) > 0:
            self.erroneous_revisions.add(self.persistent.revision)
        self.writer.emit(
            "metric",
            tick,
            inputs={
                "persistent_map": self.persistent.sha256,
                "truth_reference": canonical_json_sha256(
                    {
                        "source_floorplan": self.record.source_sha256,
                        "root_frame": 0,
                    }
                ),
            },
            outputs={
                "coverage_distance": canonical_json_sha256(
                    {"distance_m": distance, "coverage": coverage}
                ),
                "persistent_map_quality": canonical_json_sha256(quality),
            },
            outcome="recorded",
            persistent_map_correct_known_cell_coverage=coverage,
            persistent_map_known_cell_coverage=quality["known_cell_coverage"],
            persistent_map_quality=quality,
            seen_coverage_diagnostic=seen_coverage,
            cumulative_team_distance_m=distance,
            evaluator_only=True,
            ground_truth_consumer="persistent_map_quality_and_stopping_only",
            available_to_predictor=False,
            available_to_registration=False,
            available_to_planner=False,
        )
        return coverage

    def _coverage_endpoints(self) -> dict[str, Any]:
        return fixed_budget_coverage_endpoints(
            self.coverage_curve,
            budget_m=self.parameters.distance_budget_m,
            threshold=0.80,
        )

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        world_metadata = self.world.public_metadata()
        self.writer.emit(
            "world_init",
            0,
            inputs={"source_floorplan": self.record.source_sha256},
            outputs={
                "world_contract": canonical_json_sha256(world_metadata),
                "initial_state": self.initial_state_sha256,
            },
            outcome="initialized",
            world=world_metadata,
            team_size=self.parameters.team_size,
            initial_state_manifest=self.initial_state_manifest,
            initial_state_sha256=self.initial_state_sha256,
            private_frame_quadrants=self.frame_quadrants,
            random_stream_manifest_sha256=canonical_json_sha256(self.stream_seeds),
            ground_truth_consumer="initialization_only",
        )
        final_tick = 0
        for tick in range(self.parameters.maximum_ticks + 1):
            final_tick = tick
            self._scan_all(tick)
            if tick % self.parameters.prediction_period_ticks == 0:
                self._update_predictions(tick)
            self._registration_stage(tick)
            self._sync_connected_measurements(tick, "tick_measured_sync")
            self._planning_stage(tick)
            coverage = self._record_coverage(tick)
            if coverage >= self.parameters.stop_coverage:
                break
            if tick < self.parameters.maximum_ticks:
                self._motion_stage(tick)

        runtime = time.perf_counter() - started
        endpoints = self._coverage_endpoints()
        gate_accepted = self.registration_counts["gate_accept"]
        final_accepted = self.registration_counts[
            "final_decision_correct_accept"
        ] + self.registration_counts["final_decision_false_accept"]
        motion_attempts = self.motion_counts["attempted"]
        motion_robot_ticks = self.motion_counts["motion_stage_robot_ticks"]
        final_quality = self._persistent_map_quality()
        root_pose_errors = {
            str(robot.robot_id): self._root_pose_error(robot.robot_id)
            for robot in self.robots
        }
        connected_pose_errors = [
            value for value in root_pose_errors.values() if value["connected"]
        ]
        unrecovered_unsafe = max(
            0,
            self.registration_counts["unsafe_registration_commit"]
            - self.registration_counts["recovered_unsafe_registration_commit"],
        )
        collision_count = int(sum(self.world.collision_count))
        summary = {
            "status": "new_reconstructed_experiment_not_historical",
            "run_id": self.run_id,
            "split": self.split,
            "arm": self.arm,
            "seed": self.seed,
            "building_id": self.record.building_id,
            "floorplan_id": self.record.floorplan_id,
            "ticks_completed": final_tick,
            "runtime_seconds": runtime,
            "persistent_map_quality_final": final_quality,
            "persistent_map_known_cell_coverage_final": final_quality[
                "known_cell_coverage"
            ],
            "persistent_map_correct_known_cell_coverage_final": final_quality[
                "correct_known_cell_coverage"
            ],
            "seen_coverage_diagnostic_final": self.world.measured_coverage(),
            "cumulative_team_distance_m": self.world.cumulative_team_distance_m(),
            "distance_per_robot_m": self.world.distance_m,
            "collisions_per_robot": self.world.collision_count,
            "collision_count": collision_count,
            "connected_robots": sum(robot.connected for robot in self.robots),
            "root_pose_errors": root_pose_errors,
            "root_transform_dependency_graph": {
                str(robot.robot_id): robot.root_parent_robot_id
                for robot in self.robots
            },
            "maximum_connected_root_translation_error_m": max(
                (float(value["translation_error_m"])
                 for value in connected_pose_errors), default=None),
            "maximum_connected_root_absolute_yaw_error_deg": max(
                (float(value["absolute_yaw_error_deg"])
                 for value in connected_pose_errors), default=None),
            "map_revision": self.persistent.revision,
            "persistent_map_sha256": self.persistent.sha256,
            "erroneous_persistent_map_revisions": sorted(
                self.erroneous_revisions),
            "erroneous_persistent_map_revision_count": len(
                self.erroneous_revisions),
            "cascade_pollution_area_m2_final": final_quality[
                "contaminated_area_m2"
            ],
            "registration": self.registration_counts,
            "gate_wrong_accept_rate": self._ratio(
                self.registration_counts["gate_wrong_accept"], gate_accepted),
            "gate_wrong_accept_rate_denominator": gate_accepted,
            "final_decision_false_accept_rate": self._ratio(
                self.registration_counts["final_decision_false_accept"],
                final_accepted,
            ),
            "final_decision_false_accept_rate_denominator": final_accepted,
            "final_decision_false_reject_count": self.registration_counts[
                "final_decision_false_reject"
            ],
            "unsafe_commit_unrecovered_count": unrecovered_unsafe,
            "run_safety_failure": unrecovered_unsafe > 0,
            "motion": self.motion_counts,
            "planning": self.planning_counts,
            "collision_rate_per_motion_attempt": self._ratio(
                collision_count, motion_attempts),
            "motion_attempt_rate_per_robot_tick": self._ratio(
                motion_attempts, motion_robot_ticks),
            "zero_denominator_policy": "undefined_null_never_zero",
            "joint_safety_interpretation_required": True,
            "initial_state_sha256": self.initial_state_sha256,
            "random_stream_manifest_sha256": canonical_json_sha256(self.stream_seeds),
            **endpoints,
        }
        summary["event_count"] = self.writer.event_count + 1
        summary_sha256 = canonical_json_sha256(summary)
        self.writer.emit(
            "run_complete",
            final_tick,
            inputs={"persistent_map": self.persistent.sha256},
            outputs={"summary": summary_sha256},
            map_revision_before=self.persistent.revision,
            map_revision_after=self.persistent.revision,
            persistent_hash_before=self.persistent.sha256,
            persistent_hash_after=self.persistent.sha256,
            outcome="complete",
            runtime_seconds=runtime,
            summary_sha256=summary_sha256,
        )
        if self.writer.event_count != summary["event_count"]:
            raise AssertionError("summary event count differs from the final log")
        self.writer.close()
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary
