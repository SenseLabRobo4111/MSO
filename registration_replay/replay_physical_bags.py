"""Replay the legacy two-map registrar and evaluate a guarded SE(2) commit path.

The script separates three facts that must not be conflated.

1. A rosbag contains recorded ``/merged_map`` publications but no transform or
   decision message.
2. A transform recovered by rerunning the historical matcher is an offline
   reconstruction.  It is marked as output-verified only when its synthesised
   occupancy grid exactly equals a recorded publication in sequence.
3. The proposed gate and temporal consensus are counterfactual safety analyses.
   They were not deployed in the recorded experiments.

UAV-derived frame alignment is loaded only after matching and gating.  It is
used for evaluation, never for candidate selection or commit decisions.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registration_replay.se2_gate import (
    CandidateDiagnostics,
    GateThresholds,
    gate_se2_candidate,
)
from registration_replay.rigid_estimator import estimate_from_keypoint_matches


PREDICTED_TOPICS = {
    "/robot_0/predicted_map": "robot_0",
    "/robot_1/predicted_map": "robot_1",
}
MERGED_TOPIC = "/merged_map"


@dataclass
class GridObservation:
    robot_id: str
    topic: str
    index: int
    record_time_ns: int
    image: np.ndarray
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    width: int
    height: int


@dataclass
class RecordedOutput:
    bag: str
    output_index: int
    record_time_ns: int
    width: int
    height: int
    occupied_cells: int
    raster_sha256: str

    @property
    def signature(self) -> Tuple[int, int, str]:
        return self.height, self.width, self.raster_sha256


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _grid_signature(array: np.ndarray) -> Tuple[int, int, str]:
    return int(array.shape[0]), int(array.shape[1]), _sha256_bytes(array.tobytes())


def _occupancy_to_matcher_image(msg) -> np.ndarray:
    raw = np.asarray(msg.data, dtype=np.int8).reshape(int(msg.info.height), int(msg.info.width))
    # Use an intermediate integer type because newer NumPy versions reject the
    # historical assignment of 255 into an int8 view.
    image = np.flipud(raw).astype(np.int16)
    image[image == -1] = 0
    image[image == 100] = 255
    return image.astype(np.uint8)


def _recorded_array(msg) -> np.ndarray:
    return np.asarray(msg.data, dtype=np.int8).reshape(int(msg.info.height), int(msg.info.width))


def _published_array(merged_image: np.ndarray) -> np.ndarray:
    # This exactly reproduces the historical node's conversion order.  The node
    # first changes image value 255 to 100, so only zero becomes occupied (100)
    # and every nonzero value becomes unknown (-1).
    return np.flipud(np.where(merged_image == 0, 100, -1).astype(np.int8))


def _closest_proper_se2(affine: np.ndarray) -> np.ndarray:
    h = np.asarray(affine, dtype=float)
    u, _, vt = np.linalg.svd(h[:2, :2])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    result = np.eye(3, dtype=float)
    result[:2, :2] = rotation
    result[:2, 2] = h[:2, 2]
    return result


def _feature_residuals(kp0, kp1, matches, affine: np.ndarray) -> np.ndarray:
    if not matches:
        return np.empty(0, dtype=float)
    source = np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    transformed = cv2.transform(source, np.asarray(affine[:2], dtype=np.float32))
    return np.linalg.norm(transformed.reshape(-1, 2) - target.reshape(-1, 2), axis=1)


def _match_minor_spreads(kp0, kp1, matches) -> Tuple[float, float]:
    if matches is None or len(matches) < 3:
        return 0.0, 0.0
    source = np.float64([kp0[m.queryIdx].pt for m in matches])
    target = np.float64([kp1[m.trainIdx].pt for m in matches])

    def spread(points):
        eigenvalues = np.linalg.eigvalsh(np.cov(points.T, bias=True))
        return float(np.sqrt(max(0.0, eigenvalues[0])))

    return spread(source), spread(target)


def _occupancy_agreement(
    image0: np.ndarray,
    image1: np.ndarray,
    se2_image_0_to_1: np.ndarray,
) -> Tuple[float, float, float, int, int]:
    source = (image0 > 0).astype(np.uint8)
    target = (image1 > 0).astype(np.uint8)
    warped = cv2.warpAffine(
        source,
        np.asarray(se2_image_0_to_1[:2], dtype=np.float32),
        (target.shape[1], target.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    source_count = int(np.count_nonzero(warped))
    target_count = int(np.count_nonzero(target))
    intersection = int(np.count_nonzero((warped > 0) & (target > 0)))
    union = int(np.count_nonzero((warped > 0) | (target > 0)))
    iou = intersection / max(1, union)
    overlap = intersection / max(1, min(source_count, target_count))
    distance = cv2.distanceTransform((1 - target).astype(np.uint8), cv2.DIST_L2, 3)
    samples = distance[warped > 0]
    chamfer_median = float(np.median(samples)) if samples.size else float("inf")
    return float(iou), float(overlap), chamfer_median, source_count, target_count


def _image_se2_to_metric_0_from_1(
    se2_image_0_to_1: np.ndarray,
    map0: GridObservation,
    map1: GridObservation,
) -> Tuple[np.ndarray, float, float, float]:
    if map0.width != map1.width or map0.height != map1.height:
        raise ValueError("metric conversion currently requires equal grid dimensions")
    if not np.isclose(map0.resolution_m, map1.resolution_m, rtol=0.0, atol=1e-9):
        raise ValueError("metric conversion currently requires equal grid resolution")

    flip0 = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, map0.height - 1.0], [0.0, 0.0, 1.0]]
    )
    flip1_inverse = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, map1.height - 1.0], [0.0, 0.0, 1.0]]
    )
    grid_1_from_0 = flip1_inverse @ se2_image_0_to_1 @ flip0
    rotation_1_from_0 = grid_1_from_0[:2, :2]
    origin0 = np.array([map0.origin_x_m, map0.origin_y_m], dtype=float)
    origin1 = np.array([map1.origin_x_m, map1.origin_y_m], dtype=float)
    translation_1_from_0 = (
        origin1
        + map1.resolution_m * grid_1_from_0[:2, 2]
        - rotation_1_from_0 @ origin0
    )
    rotation_0_from_1 = rotation_1_from_0.T
    translation_0_from_1 = -rotation_0_from_1 @ translation_1_from_0
    transform = np.eye(3, dtype=float)
    transform[:2, :2] = rotation_0_from_1
    transform[:2, 2] = translation_0_from_1
    yaw = float(math.atan2(rotation_0_from_1[1, 0], rotation_0_from_1[0, 0]))
    return transform, float(translation_0_from_1[0]), float(translation_0_from_1[1]), yaw


def _wrap_radians(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _reference_error(
    x_m: float,
    y_m: float,
    yaw_rad: float,
    reference: dict,
) -> Tuple[float, float]:
    reference_xy = np.asarray(reference["inter_robot_offset_joint"], dtype=float)
    translation_error = float(np.linalg.norm(np.array([x_m, y_m]) - reference_xy))
    yaw_error = abs(math.degrees(_wrap_radians(yaw_rad - float(reference["theta_rad"]))))
    return translation_error, float(yaw_error)


def _lcs_exact_matches(
    replay: Sequence[Tuple[Tuple[int, int, str], int]],
    recorded: Sequence[RecordedOutput],
) -> List[Tuple[int, int]]:
    """Maximise ordered exact raster matches, then minimise timestamp distance."""

    n, m = len(replay), len(recorded)
    count = np.zeros((n + 1, m + 1), dtype=np.int16)
    cost = np.zeros((n + 1, m + 1), dtype=np.float64)
    direction = np.zeros((n + 1, m + 1), dtype=np.uint8)  # 1 up, 2 left, 3 match

    def better(c1, k1, c2, k2):
        return c1 > c2 or (c1 == c2 and k1 < k2)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_count = int(count[i - 1, j])
            best_cost = float(cost[i - 1, j])
            best_direction = 1
            left_count = int(count[i, j - 1])
            left_cost = float(cost[i, j - 1])
            if better(left_count, left_cost, best_count, best_cost):
                best_count, best_cost, best_direction = left_count, left_cost, 2
            if replay[i - 1][0] == recorded[j - 1].signature:
                match_count = int(count[i - 1, j - 1]) + 1
                dt = abs(replay[i - 1][1] - recorded[j - 1].record_time_ns) / 1e9
                match_cost = float(cost[i - 1, j - 1]) + dt
                if better(match_count, match_cost, best_count, best_cost):
                    best_count, best_cost, best_direction = match_count, match_cost, 3
            count[i, j] = best_count
            cost[i, j] = best_cost
            direction[i, j] = best_direction

    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        move = int(direction[i, j])
        if move == 3:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _load_bag(
    bag_dir: Path,
    typestore,
) -> Tuple[List[GridObservation], List[RecordedOutput], List[dict], Path]:
    observations: List[GridObservation] = []
    recorded: List[RecordedOutput] = []
    inventory: List[dict] = []
    counters = Counter()
    db_files = sorted(bag_dir.glob("*.db3"))
    if len(db_files) != 1:
        raise RuntimeError(f"expected one db3 file in {bag_dir}, found {len(db_files)}")

    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        for connection in reader.connections:
            inventory.append(
                {
                    "bag": bag_dir.name,
                    "topic": connection.topic,
                    "message_type": connection.msgtype,
                    "message_count": int(connection.msgcount),
                }
            )
        relevant = [
            connection
            for connection in reader.connections
            if connection.topic in PREDICTED_TOPICS or connection.topic == MERGED_TOPIC
        ]
        for connection, record_time_ns, raw in reader.messages(connections=relevant):
            msg = reader.deserialize(raw, connection.msgtype)
            if connection.topic == MERGED_TOPIC:
                array = _recorded_array(msg)
                signature = _grid_signature(array)
                recorded.append(
                    RecordedOutput(
                        bag=bag_dir.name,
                        output_index=len(recorded),
                        record_time_ns=int(record_time_ns),
                        width=int(msg.info.width),
                        height=int(msg.info.height),
                        occupied_cells=int(np.count_nonzero(array == 100)),
                        raster_sha256=signature[2],
                    )
                )
                continue

            robot_id = PREDICTED_TOPICS[connection.topic]
            counters[robot_id] += 1
            observations.append(
                GridObservation(
                    robot_id=robot_id,
                    topic=connection.topic,
                    index=int(counters[robot_id] - 1),
                    record_time_ns=int(record_time_ns),
                    image=_occupancy_to_matcher_image(msg),
                    resolution_m=float(msg.info.resolution),
                    origin_x_m=float(msg.info.origin.position.x),
                    origin_y_m=float(msg.info.origin.position.y),
                    width=int(msg.info.width),
                    height=int(msg.info.height),
                )
            )
    return observations, recorded, inventory, db_files[0]


def _blank_attempt(bag: str, attempt_index: int, trigger: GridObservation, map0, map1) -> dict:
    return {
        "bag": bag,
        "attempt_index": attempt_index,
        "trigger_robot": trigger.robot_id,
        "trigger_topic": trigger.topic,
        "trigger_record_time_ns": trigger.record_time_ns,
        "robot_0_map_index": map0.index,
        "robot_1_map_index": map1.index,
        "robot_0_map_time_ns": map0.record_time_ns,
        "robot_1_map_time_ns": map1.record_time_ns,
        "map_pair_time_gap_s": abs(map0.record_time_ns - map1.record_time_ns) / 1e9,
        "candidate_found": 0,
        "legacy_replay_publish": 0,
        "legacy_replay_reject": 1,
        "matcher_error": "",
        "raw_match_count": 0,
        "filtered_match_count": 0,
        "ransac_inlier_count": 0,
        "ransac_inlier_ratio": 0.0,
        "feature_residual_median_px": "",
        "feature_residual_p90_px": "",
        "legacy_visual_inlier_ratio": "",
        "legacy_blend_weight": "",
        "h00": "",
        "h01": "",
        "h02": "",
        "h10": "",
        "h11": "",
        "h12": "",
        "determinant": "",
        "singular_value_min": "",
        "singular_value_max": "",
        "isotropic_scale": "",
        "anisotropy": "",
        "orthogonality_error": "",
        "occupancy_iou": "",
        "occupancy_overlap": "",
        "occupancy_chamfer_median_px": "",
        "warped_occupied_cells": "",
        "target_occupied_cells": "",
        "structure_se2_pass": 0,
        "legacy_candidate_gate_accept": 0,
        "legacy_candidate_gate_reasons": "no_candidate",
        "estimate_x_0_from_1_m": "",
        "estimate_y_0_from_1_m": "",
        "estimate_yaw_0_from_1_deg": "",
        "reference_x_0_from_1_m": "",
        "reference_y_0_from_1_m": "",
        "reference_yaw_0_from_1_deg": "",
        "translation_error_m": "",
        "yaw_error_deg": "",
        "within_0p25m_5deg": 0,
        "within_0p50m_10deg": 0,
        "replay_output_height": "",
        "replay_output_width": "",
        "replay_output_sha256": "",
        "legacy_output_verified_exact": 0,
        "recorded_output_index": "",
        "recorded_output_time_ns": "",
        "recorded_output_delay_s": "",
        "replay_hash_multiplicity": "",
        "recorded_hash_multiplicity": "",
        "legacy_candidate_consensus_support": 0,
        "legacy_gated_counterfactual_commit": 0,
        "legacy_gated_counterfactual_reject": 1,
        "legacy_gated_counterfactual_reason": "no_candidate",
        "rigid_candidate_found": 0,
        "rigid_input_correspondence_count": 0,
        "rigid_inlier_count": 0,
        "rigid_inlier_ratio": 0.0,
        "rigid_inlier_residual_median_px": "",
        "rigid_inlier_residual_p90_px": "",
        "rigid_source_minor_spread_px": "",
        "rigid_target_minor_spread_px": "",
        "rigid_estimator_reject_reason": "no_correspondences",
        "rigid_h00": "",
        "rigid_h01": "",
        "rigid_h02": "",
        "rigid_h10": "",
        "rigid_h11": "",
        "rigid_h12": "",
        "rigid_occupancy_iou": "",
        "rigid_occupancy_overlap": "",
        "rigid_occupancy_chamfer_median_px": "",
        "rigid_local_gate_accept": 0,
        "rigid_local_gate_reasons": "no_candidate",
        "rigid_estimate_x_0_from_1_m": "",
        "rigid_estimate_y_0_from_1_m": "",
        "rigid_estimate_yaw_0_from_1_deg": "",
        "rigid_translation_error_m": "",
        "rigid_yaw_error_deg": "",
        "rigid_within_0p25m_5deg": 0,
        "rigid_within_0p50m_10deg": 0,
        "rigid_consensus_support": 0,
        "rigid_gated_counterfactual_commit": 0,
        "rigid_gated_counterfactual_reject": 1,
        "rigid_gated_counterfactual_reason": "no_candidate",
    }


def _replay_bag(
    bag: str,
    observations: Sequence[GridObservation],
    recorded: Sequence[RecordedOutput],
    reference: dict,
    matcher,
    merger,
    thresholds: GateThresholds,
) -> Tuple[List[dict], List[dict]]:
    cache: Dict[str, GridObservation] = {}
    attempts: List[dict] = []
    replay_publications: List[Tuple[Tuple[int, int, str], int, int]] = []

    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        for observation in observations:
            cache[observation.robot_id] = observation
            if "robot_0" not in cache or "robot_1" not in cache:
                continue
            map0, map1 = cache["robot_0"], cache["robot_1"]
            row = _blank_attempt(bag, len(attempts), observation, map0, map1)
            row["reference_x_0_from_1_m"] = float(reference["inter_robot_offset_joint"][0])
            row["reference_y_0_from_1_m"] = float(reference["inter_robot_offset_joint"][1])
            row["reference_yaw_0_from_1_deg"] = float(reference["theta_deg"])
            try:
                h, kp0, kp1, matches, _, filtered, inliers = matcher(map0.image, map1.image)
            except Exception as exc:  # preserve the attempt and continue the audit
                row["matcher_error"] = f"{type(exc).__name__}: {exc}"
                row["legacy_candidate_gate_reasons"] = "matcher_exception"
                row["legacy_gated_counterfactual_reason"] = "matcher_exception"
                row["rigid_gated_counterfactual_reason"] = "matcher_exception"
                attempts.append(row)
                continue

            row["raw_match_count"] = len(matches) if matches is not None else 0
            row["filtered_match_count"] = len(filtered) if filtered is not None else 0
            if inliers is not None:
                mask = np.asarray(inliers).reshape(-1)
                row["ransac_inlier_count"] = int(np.count_nonzero(mask))
                row["ransac_inlier_ratio"] = float(np.count_nonzero(mask) / max(1, mask.size))
            if h is None or filtered is None or len(filtered) == 0:
                attempts.append(row)
                continue

            h = np.asarray(h, dtype=float)
            row["candidate_found"] = 1
            row["legacy_replay_publish"] = 1
            row["legacy_replay_reject"] = 0
            row.update(
                {
                    "h00": h[0, 0], "h01": h[0, 1], "h02": h[0, 2],
                    "h10": h[1, 0], "h11": h[1, 1], "h12": h[1, 2],
                }
            )

            residuals = _feature_residuals(kp0, kp1, filtered, h)
            median_residual = float(np.median(residuals)) if residuals.size else float("inf")
            row["feature_residual_median_px"] = median_residual
            row["feature_residual_p90_px"] = (
                float(np.quantile(residuals, 0.9)) if residuals.size else ""
            )
            visual_inlier_ratio = float(np.mean(residuals < 3.0)) if residuals.size else 0.0
            blend_weight = float(np.clip(visual_inlier_ratio, 0.3, 0.7))
            row["legacy_visual_inlier_ratio"] = visual_inlier_ratio
            row["legacy_blend_weight"] = blend_weight

            projected = _closest_proper_se2(h)
            iou, overlap, chamfer, warped_count, target_count = _occupancy_agreement(
                map0.image, map1.image, projected
            )
            row["occupancy_iou"] = iou
            row["occupancy_overlap"] = overlap
            row["occupancy_chamfer_median_px"] = chamfer
            row["warped_occupied_cells"] = warped_count
            row["target_occupied_cells"] = target_count
            legacy_source_spread, legacy_target_spread = _match_minor_spreads(
                kp0, kp1, filtered
            )
            diagnostics = CandidateDiagnostics(
                inlier_count=int(row["ransac_inlier_count"]),
                inlier_ratio=float(row["ransac_inlier_ratio"]),
                median_residual_px=median_residual,
                occupancy_iou=iou,
                occupancy_chamfer_median_px=chamfer,
                source_minor_spread_px=legacy_source_spread,
                target_minor_spread_px=legacy_target_spread,
            )
            decision = gate_se2_candidate(h, diagnostics, thresholds)
            row["determinant"] = decision.determinant
            row["singular_value_min"] = decision.singular_value_min
            row["singular_value_max"] = decision.singular_value_max
            row["isotropic_scale"] = decision.isotropic_scale
            row["anisotropy"] = decision.anisotropy
            row["orthogonality_error"] = decision.orthogonality_error
            structural_reasons = {
                "reflection_or_singular", "scale", "anisotropy_or_shear", "nonorthogonal"
            }
            row["structure_se2_pass"] = int(not structural_reasons.intersection(decision.reasons))
            row["legacy_candidate_gate_accept"] = int(decision.accepted)
            row["legacy_candidate_gate_reasons"] = "|".join(decision.reasons)

            _, x_m, y_m, yaw_rad = _image_se2_to_metric_0_from_1(projected, map0, map1)
            translation_error, yaw_error = _reference_error(x_m, y_m, yaw_rad, reference)
            row["estimate_x_0_from_1_m"] = x_m
            row["estimate_y_0_from_1_m"] = y_m
            row["estimate_yaw_0_from_1_deg"] = math.degrees(yaw_rad)
            row["translation_error_m"] = translation_error
            row["yaw_error_deg"] = yaw_error
            row["within_0p25m_5deg"] = int(translation_error <= 0.25 and yaw_error <= 5.0)
            row["within_0p50m_10deg"] = int(translation_error <= 0.50 and yaw_error <= 10.0)

            # Counterfactual repair.  Estimate unit-scale SE(2) directly from the
            # feature correspondences rather than projecting the legacy affine.
            rigid = estimate_from_keypoint_matches(kp0, kp1, matches)
            row["rigid_input_correspondence_count"] = rigid.input_correspondences
            row["rigid_inlier_count"] = rigid.inlier_count
            row["rigid_inlier_ratio"] = rigid.inlier_ratio
            row["rigid_inlier_residual_median_px"] = rigid.median_inlier_residual_px
            row["rigid_inlier_residual_p90_px"] = rigid.p90_inlier_residual_px
            row["rigid_source_minor_spread_px"] = rigid.source_minor_spread_px
            row["rigid_target_minor_spread_px"] = rigid.target_minor_spread_px
            row["rigid_estimator_reject_reason"] = rigid.reject_reason
            if rigid.se2_image is not None:
                rigid_h = rigid.se2_image
                row["rigid_candidate_found"] = 1
                row.update(
                    {
                        "rigid_h00": rigid_h[0, 0], "rigid_h01": rigid_h[0, 1],
                        "rigid_h02": rigid_h[0, 2], "rigid_h10": rigid_h[1, 0],
                        "rigid_h11": rigid_h[1, 1], "rigid_h12": rigid_h[1, 2],
                    }
                )
                rigid_iou, rigid_overlap, rigid_chamfer, _, _ = _occupancy_agreement(
                    map0.image, map1.image, rigid_h
                )
                row["rigid_occupancy_iou"] = rigid_iou
                row["rigid_occupancy_overlap"] = rigid_overlap
                row["rigid_occupancy_chamfer_median_px"] = rigid_chamfer
                rigid_diagnostics = CandidateDiagnostics(
                    inlier_count=rigid.inlier_count,
                    inlier_ratio=rigid.inlier_ratio,
                    median_residual_px=rigid.median_inlier_residual_px,
                    occupancy_iou=rigid_iou,
                    occupancy_chamfer_median_px=rigid_chamfer,
                    source_minor_spread_px=rigid.source_minor_spread_px,
                    target_minor_spread_px=rigid.target_minor_spread_px,
                )
                rigid_decision = gate_se2_candidate(rigid_h, rigid_diagnostics, thresholds)
                row["rigid_local_gate_accept"] = int(rigid_decision.accepted)
                row["rigid_local_gate_reasons"] = "|".join(rigid_decision.reasons)
                _, rigid_x, rigid_y, rigid_yaw = _image_se2_to_metric_0_from_1(
                    rigid_h, map0, map1
                )
                rigid_translation_error, rigid_yaw_error = _reference_error(
                    rigid_x, rigid_y, rigid_yaw, reference
                )
                row["rigid_estimate_x_0_from_1_m"] = rigid_x
                row["rigid_estimate_y_0_from_1_m"] = rigid_y
                row["rigid_estimate_yaw_0_from_1_deg"] = math.degrees(rigid_yaw)
                row["rigid_translation_error_m"] = rigid_translation_error
                row["rigid_yaw_error_deg"] = rigid_yaw_error
                row["rigid_within_0p25m_5deg"] = int(
                    rigid_translation_error <= 0.25 and rigid_yaw_error <= 5.0
                )
                row["rigid_within_0p50m_10deg"] = int(
                    rigid_translation_error <= 0.50 and rigid_yaw_error <= 10.0
                )

            merged = merger(map0.image, map1.image, h, blend_weight)
            publication = _published_array(merged)
            signature = _grid_signature(publication)
            row["replay_output_height"] = signature[0]
            row["replay_output_width"] = signature[1]
            row["replay_output_sha256"] = signature[2]
            replay_publications.append((signature, observation.record_time_ns, len(attempts)))
            attempts.append(row)

    exact_pairs = _lcs_exact_matches(
        [(signature, timestamp) for signature, timestamp, _ in replay_publications], recorded
    )
    replay_counts = Counter(signature for signature, _, _ in replay_publications)
    recorded_counts = Counter(output.signature for output in recorded)
    matched_recorded = set()
    for replay_index, recorded_index in exact_pairs:
        signature, trigger_time, attempt_index = replay_publications[replay_index]
        output = recorded[recorded_index]
        row = attempts[attempt_index]
        row["legacy_output_verified_exact"] = 1
        row["recorded_output_index"] = recorded_index
        row["recorded_output_time_ns"] = output.record_time_ns
        row["recorded_output_delay_s"] = (output.record_time_ns - trigger_time) / 1e9
        row["replay_hash_multiplicity"] = replay_counts[signature]
        row["recorded_hash_multiplicity"] = recorded_counts[signature]
        matched_recorded.add(recorded_index)

    output_rows = []
    match_by_recorded = {recorded_index: replay_index for replay_index, recorded_index in exact_pairs}
    for output in recorded:
        replay_index = match_by_recorded.get(output.output_index)
        attempt_index = replay_publications[replay_index][2] if replay_index is not None else ""
        output_rows.append(
            {
                **asdict(output),
                "exact_replay_match": int(replay_index is not None),
                "matched_attempt_index": attempt_index,
                "hash_multiplicity_in_recorded": recorded_counts[output.signature],
                "startup_or_unrecovered": int(replay_index is None),
            }
        )
    return attempts, output_rows


def _apply_temporal_consensus(
    attempts: List[dict],
    *,
    local_accept_key: str,
    local_reason_key: str,
    estimate_x_key: str,
    estimate_y_key: str,
    estimate_yaw_key: str,
    support_key: str,
    commit_key: str,
    reject_key: str,
    reason_key: str,
    confirmations: int = 3,
    min_confirmation_interval_s: float = 0.5,
    max_gap_s: float = 5.0,
    translation_tolerance_m: float = 0.5,
    yaw_tolerance_deg: float = 5.0,
) -> None:
    support = 0
    last_time_ns: Optional[int] = None
    last_xy: Optional[np.ndarray] = None
    last_yaw: Optional[float] = None
    last_pair: Optional[Tuple[int, int]] = None
    for row in attempts:
        if not row[local_accept_key]:
            row[reason_key] = row[local_reason_key] or "local_gate"
            support = 0
            last_time_ns = None
            last_xy = None
            last_yaw = None
            last_pair = None
            continue
        timestamp = int(row["trigger_record_time_ns"])
        pair = (int(row["robot_0_map_index"]), int(row["robot_1_map_index"]))
        xy = np.array(
            [float(row[estimate_x_key]), float(row[estimate_y_key])],
            dtype=float,
        )
        yaw = math.radians(float(row[estimate_yaw_key]))
        if last_time_ns is not None:
            dt_s = (timestamp - last_time_ns) / 1e9
            both_revised = pair[0] != last_pair[0] and pair[1] != last_pair[1]
            if dt_s < min_confirmation_interval_s or not both_revised:
                row[support_key] = support
                row[reason_key] = "nonindependent_confirmation"
                continue
        consistent = (
            last_time_ns is not None
            and dt_s <= max_gap_s
            and float(np.linalg.norm(xy - last_xy)) <= translation_tolerance_m
            and abs(math.degrees(_wrap_radians(yaw - last_yaw))) <= yaw_tolerance_deg
        )
        support = support + 1 if consistent else 1
        last_time_ns, last_xy, last_yaw, last_pair = timestamp, xy, yaw, pair
        row[support_key] = support
        if support >= confirmations:
            row[commit_key] = 1
            row[reject_key] = 0
            row[reason_key] = ""
        else:
            row[reason_key] = "temporal_confirmation"


def _later_good_candidate_statistics(
    rows: Sequence[dict], tight: bool
) -> Tuple[int, int, float, float]:
    verified = [row for row in rows if row["legacy_output_verified_exact"]]
    key = "within_0p25m_5deg" if tight else "within_0p50m_10deg"
    unsafe_indices = [
        i for i, row in enumerate(verified)
        if not row["structure_se2_pass"] or not int(row[key])
    ]
    later_good = []
    for index in unsafe_indices:
        for later in range(index + 1, len(verified)):
            row = verified[later]
            if row["structure_se2_pass"] and int(row[key]):
                later_good.append(
                    (
                        later - index,
                        (int(row["trigger_record_time_ns"]) - int(verified[index]["trigger_record_time_ns"])) / 1e9,
                    )
                )
                break
    if not later_good:
        return len(unsafe_indices), 0, float("nan"), float("nan")
    return (
        len(unsafe_indices),
        len(later_good),
        float(np.median([item[0] for item in later_good])),
        float(np.median([item[1] for item in later_good])),
    )


def _summarise_bag(bag: str, rows: Sequence[dict], outputs: Sequence[dict], reference: dict) -> dict:
    verified = [row for row in rows if row["legacy_output_verified_exact"]]
    rigid_candidates = [row for row in rows if row["rigid_candidate_found"]]
    commits = [row for row in rows if row["rigid_gated_counterfactual_commit"]]
    tight_bad, tight_later, tight_events, tight_seconds = _later_good_candidate_statistics(rows, True)
    relaxed_bad, relaxed_later, relaxed_events, relaxed_seconds = _later_good_candidate_statistics(rows, False)

    def count_where(items, predicate):
        return sum(1 for item in items if predicate(item))

    return {
        "bag": bag,
        "registration_attempts_replayed": len(rows),
        "legacy_replay_accepts": count_where(rows, lambda x: x["legacy_replay_publish"]),
        "legacy_replay_rejects": count_where(rows, lambda x: x["legacy_replay_reject"]),
        "recorded_merged_publications": len(outputs),
        "recorded_publications_exactly_reproduced": count_where(outputs, lambda x: x["exact_replay_match"]),
        "recorded_publication_reproduction_fraction": (
            count_where(outputs, lambda x: x["exact_replay_match"]) / max(1, len(outputs))
        ),
        "recorded_publications_without_exact_replay": count_where(outputs, lambda x: not x["exact_replay_match"]),
        "verified_legacy_structure_invalid": count_where(verified, lambda x: not x["structure_se2_pass"]),
        "verified_legacy_unsafe_proxy_0p25m_5deg": count_where(
            verified, lambda x: (not x["structure_se2_pass"]) or (not x["within_0p25m_5deg"])
        ),
        "verified_legacy_unsafe_proxy_0p50m_10deg": count_where(
            verified, lambda x: (not x["structure_se2_pass"]) or (not x["within_0p50m_10deg"])
        ),
        "legacy_candidate_local_gate_accepts": count_where(
            rows, lambda x: x["legacy_candidate_gate_accept"]
        ),
        "legacy_candidate_local_gate_rejects": count_where(
            rows, lambda x: not x["legacy_candidate_gate_accept"]
        ),
        "legacy_gated_counterfactual_commits": count_where(
            rows, lambda x: x["legacy_gated_counterfactual_commit"]
        ),
        "rigid_candidates": len(rigid_candidates),
        "rigid_local_gate_accepts": count_where(rows, lambda x: x["rigid_local_gate_accept"]),
        "rigid_local_gate_rejects": count_where(rows, lambda x: not x["rigid_local_gate_accept"]),
        "rigid_gated_counterfactual_commits": len(commits),
        "rigid_gated_counterfactual_rejects": len(rows) - len(commits),
        "rigid_gated_commit_within_0p25m_5deg": count_where(
            commits, lambda x: x["rigid_within_0p25m_5deg"]
        ),
        "rigid_gated_commit_outside_0p25m_5deg": count_where(
            commits, lambda x: not x["rigid_within_0p25m_5deg"]
        ),
        "rigid_gated_commit_within_0p50m_10deg": count_where(
            commits, lambda x: x["rigid_within_0p50m_10deg"]
        ),
        "rigid_gated_commit_outside_0p50m_10deg": count_where(
            commits, lambda x: not x["rigid_within_0p50m_10deg"]
        ),
        "rigid_gated_rejected_bad_candidate_proxy_0p25m_5deg": count_where(
            rows,
            lambda x: x["rigid_gated_counterfactual_reject"]
            and (not x["rigid_candidate_found"] or not x["rigid_within_0p25m_5deg"]),
        ),
        "rigid_gated_rejected_good_candidate_proxy_0p25m_5deg": count_where(
            rigid_candidates,
            lambda x: x["rigid_gated_counterfactual_reject"]
            and x["rigid_within_0p25m_5deg"],
        ),
        "rigid_gated_rejected_bad_candidate_proxy_0p50m_10deg": count_where(
            rows,
            lambda x: x["rigid_gated_counterfactual_reject"]
            and (not x["rigid_candidate_found"] or not x["rigid_within_0p50m_10deg"]),
        ),
        "rigid_gated_rejected_good_candidate_proxy_0p50m_10deg": count_where(
            rigid_candidates,
            lambda x: x["rigid_gated_counterfactual_reject"]
            and x["rigid_within_0p50m_10deg"],
        ),
        "recommended_fail_closed_commits": 0,
        "recommended_fail_closed_rejects": len(rows),
        "legacy_bad_verified_events_tight": tight_bad,
        "legacy_bad_events_with_later_good_candidate_tight": tight_later,
        "legacy_later_good_candidate_median_event_gap_tight": tight_events,
        "legacy_later_good_candidate_median_seconds_tight": tight_seconds,
        "legacy_bad_verified_events_relaxed": relaxed_bad,
        "legacy_bad_events_with_later_good_candidate_relaxed": relaxed_later,
        "legacy_later_good_candidate_median_event_gap_relaxed": relaxed_events,
        "legacy_later_good_candidate_median_seconds_relaxed": relaxed_seconds,
        "uav_reference_pairs": int(reference["n_pairs"]),
        "uav_reported_rms_first_label_m": float(reference["r0_rms"]),
        "uav_reported_rms_second_label_m": float(reference["r1_rms"]),
    }


def _aggregate_summaries(summaries: Sequence[dict]) -> dict:
    result = {"bag": "ALL"}
    additive = [
        key for key in summaries[0]
        if key not in {
            "bag",
            "recorded_publication_reproduction_fraction",
            "legacy_later_good_candidate_median_event_gap_tight",
            "legacy_later_good_candidate_median_seconds_tight",
            "legacy_later_good_candidate_median_event_gap_relaxed",
            "legacy_later_good_candidate_median_seconds_relaxed",
            "uav_reported_rms_first_label_m",
            "uav_reported_rms_second_label_m",
        }
    ]
    for key in additive:
        result[key] = sum(item[key] for item in summaries)
    result["recorded_publication_reproduction_fraction"] = (
        result["recorded_publications_exactly_reproduced"]
        / max(1, result["recorded_merged_publications"])
    )
    for key in (
        "legacy_later_good_candidate_median_event_gap_tight",
        "legacy_later_good_candidate_median_seconds_tight",
        "legacy_later_good_candidate_median_event_gap_relaxed",
        "legacy_later_good_candidate_median_seconds_relaxed",
        "uav_reported_rms_first_label_m",
        "uav_reported_rms_second_label_m",
    ):
        result[key] = float("nan")
    # Preserve the same column order as per-bag rows.
    return {key: result[key] for key in summaries[0]}


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _git_value(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _bag_sort_key(name: str) -> Tuple[int, int]:
    scene, trial = name.removeprefix("sense").split("-")
    return int(scene), int(trial)


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--bag-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--matcher-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bags", nargs="*")
    parser.add_argument("--skip-file-hashes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    bag_root = (args.bag_root or workspace / "Exp" / "experiment_video" / "rosbag").resolve()
    reference_root = (
        args.reference_root or workspace / "corl_2026" / "figs" / "real_world" / "_calib"
    ).resolve()
    matcher_root = (args.matcher_root or workspace / "map_merge_private").resolve()
    output_root = (
        args.output or workspace / "MSO_public" / "registration_replay" / "results"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(matcher_root))
    from map_merge.map_merge_core import adaptive_warp_and_merge, match_with_multi_level_features

    requested = args.bags or [f"sense{scene}-{trial}" for scene in range(1, 6) for trial in (1, 2)]
    requested = sorted(requested, key=_bag_sort_key)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    thresholds = GateThresholds()

    all_attempts: List[dict] = []
    all_outputs: List[dict] = []
    all_inventory: List[dict] = []
    summaries: List[dict] = []
    references: List[dict] = []
    input_files: List[dict] = []

    for bag in requested:
        print(f"[{bag}] loading", flush=True)
        bag_dir = bag_root / bag
        reference_path = reference_root / f"{bag}.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        observations, recorded, inventory, db_path = _load_bag(bag_dir, typestore)
        print(
            f"[{bag}] replaying {len(observations)} map inputs against {len(recorded)} recorded outputs",
            flush=True,
        )
        attempts, output_rows = _replay_bag(
            bag,
            observations,
            recorded,
            reference,
            match_with_multi_level_features,
            adaptive_warp_and_merge,
            thresholds,
        )
        _apply_temporal_consensus(
            attempts,
            local_accept_key="legacy_candidate_gate_accept",
            local_reason_key="legacy_candidate_gate_reasons",
            estimate_x_key="estimate_x_0_from_1_m",
            estimate_y_key="estimate_y_0_from_1_m",
            estimate_yaw_key="estimate_yaw_0_from_1_deg",
            support_key="legacy_candidate_consensus_support",
            commit_key="legacy_gated_counterfactual_commit",
            reject_key="legacy_gated_counterfactual_reject",
            reason_key="legacy_gated_counterfactual_reason",
        )
        _apply_temporal_consensus(
            attempts,
            local_accept_key="rigid_local_gate_accept",
            local_reason_key="rigid_local_gate_reasons",
            estimate_x_key="rigid_estimate_x_0_from_1_m",
            estimate_y_key="rigid_estimate_y_0_from_1_m",
            estimate_yaw_key="rigid_estimate_yaw_0_from_1_deg",
            support_key="rigid_consensus_support",
            commit_key="rigid_gated_counterfactual_commit",
            reject_key="rigid_gated_counterfactual_reject",
            reason_key="rigid_gated_counterfactual_reason",
        )
        summary = _summarise_bag(bag, attempts, output_rows, reference)
        print(
            f"[{bag}] exact {summary['recorded_publications_exactly_reproduced']}/"
            f"{summary['recorded_merged_publications']}; rigid gated commits "
            f"{summary['rigid_gated_counterfactual_commits']}",
            flush=True,
        )
        all_attempts.extend(attempts)
        all_outputs.extend(output_rows)
        all_inventory.extend(inventory)
        summaries.append(summary)
        references.append(
            {
                "bag": bag,
                "n_pairs": int(reference["n_pairs"]),
                "theta_deg_1_to_0": float(reference["theta_deg"]),
                "offset_x_1_to_0_m": float(reference["inter_robot_offset_joint"][0]),
                "offset_y_1_to_0_m": float(reference["inter_robot_offset_joint"][1]),
                "reported_r0_rms_m": float(reference["r0_rms"]),
                "reported_r1_rms_m": float(reference["r1_rms"]),
                "warning": (
                    "The source calibration script interleaves robot residuals before splitting "
                    "the RMS labels; the joint fit is used, but the two labelled RMS values must "
                    "not be interpreted as robot-specific uncertainty estimates."
                ),
                "reference_json_sha256": _sha256_file(reference_path),
            }
        )
        input_files.append(
            {
                "bag": bag,
                "relative_path": str(db_path.relative_to(workspace)).replace("\\", "/"),
                "bytes": db_path.stat().st_size,
                "sha256": "" if args.skip_file_hashes else _sha256_file(db_path),
            }
        )

    total = _aggregate_summaries(summaries)
    summaries.append(total)

    _write_csv(output_root / "registration_attempts.csv", all_attempts)
    _write_csv(output_root / "recorded_merged_outputs.csv", all_outputs)
    _write_csv(output_root / "bag_topic_inventory.csv", all_inventory)
    _write_csv(output_root / "uav_reference_summary.csv", references)
    _write_csv(output_root / "bag_summary.csv", summaries)

    source_commit = _git_value(matcher_root, "rev-parse", "HEAD")
    source_date = _git_value(matcher_root, "show", "-s", "--format=%aI", "HEAD")
    manifest = {
        "analysis_type": "offline replay with exact recorded-output cross-check",
        "not_an_online_transform_log": True,
        "strict_gate_was_deployed": False,
        "legacy_source": {
            "commit": source_commit,
            "author_date": source_date,
            "core_sha256": _sha256_file(matcher_root / "map_merge" / "map_merge_core.py"),
            "node_sha256": _sha256_file(matcher_root / "map_merge" / "map_merge_node.py"),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "rosbags": __import__("rosbags").__version__ if hasattr(__import__("rosbags"), "__version__") else "unknown",
        },
        "gate_thresholds": asdict(thresholds),
        "rigid_estimator": {
            "model": "proper unit-scale SE(2)",
            "method": "deterministic two-point RANSAC followed by Kabsch refinement",
            "residual_threshold_px": 3.0,
            "iterations": 384,
            "maximum_one_to_one_correspondences": 500,
        },
        "threshold_development": {
            "development_bag": "sense1-1",
            "held_out_bags": [bag for bag in requested if bag != "sense1-1"],
            "status": (
                "Exploratory replay analysis. Gate thresholds were fixed after inspecting "
                "sense1-1 and before evaluating the listed held-out bags."
            ),
        },
        "temporal_consensus": {
            "confirmations": 3,
            "min_confirmation_interval_s": 0.5,
            "max_gap_s": 5.0,
            "translation_tolerance_m": 0.5,
            "yaw_tolerance_deg": 5.0,
            "both_cached_map_revisions_must_change": True,
            "a_local_reject_resets_support": True,
        },
        "recommended_operating_policy": {
            "commit_enabled": False,
            "reason": (
                "The frozen rigid candidate and gate failed the held-out error audit; "
                "the public manager therefore defaults to fail-closed."
            ),
        },
        "evaluation_thresholds": [
            {"translation_m": 0.25, "yaw_deg": 5.0, "label": "tight diagnostic"},
            {"translation_m": 0.50, "yaw_deg": 10.0, "label": "reference-aware sensitivity"},
        ],
        "input_bags": input_files,
        "limitations": [
            "The bags contain merged rasters but no transform or decision topic.",
            "Recovered transforms are offline reruns; exact raster equality verifies outputs, not a recorded H value.",
            "OpenCV version and callback scheduling can change RANSAC output, so unreproduced publications remain unevaluable.",
            "Every cached-map callback is called an attempt, not an externally labelled communication encounter.",
            "The UAV fit is independent of map matching but has non-negligible calibration residual and no stored bootstrap interval.",
            "Strict-gate results are counterfactual and exploratory; they are not evidence of deployed online behaviour.",
            "Only two-robot physical bags are present.  The robot-ID-agnostic queue has synthetic unit tests, not three- or five-robot physical validation.",
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote results to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
