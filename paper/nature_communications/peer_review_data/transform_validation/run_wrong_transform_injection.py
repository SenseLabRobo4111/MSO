#!/usr/bin/env python3
"""Offline wrong-SE(2) injection against the frozen synthetic benchmark.

This is a gate/commit harness, not a replay of the deployed MSO system.  It
constructs one deliberately wrong, analytically rigid proposal for every
GT-positive manifest event, applies a GT-free occupied-support gate, and then
simulates an atomic map commit.  Ground truth is used only to construct the
controlled perturbation and to score the resulting state; it is never an input
to the gate decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


DEFAULT_TRANSLATION_M = 1.0
DEFAULT_YAW_DEG = 15.0
DEFAULT_GATE_THRESHOLD = 0.20
DEFAULT_TAU_TRANSLATION_M = 0.25
DEFAULT_TAU_YAW_DEG = 5.0
OCCUPIED_VALUE_THRESHOLD = 200

# The first fields are the complete evaluate_transform_events.py schema.  The
# remaining audit fields are deliberately retained by the injection CSV even
# though the evaluator ignores unknown columns.
EVENT_FIELDS = [
    "event_id", "run_id", "scene_id", "cluster_id", "seed", "input_type",
    "method", "gt_positive", "candidate_returned", "rigid_valid", "accepted",
    "committed", "hat_tx_m", "hat_ty_m", "hat_yaw_deg", "gt_tx_m", "gt_ty_m",
    "gt_yaw_deg", "raw_scale", "raw_shear", "raw_determinant", "reject_reason",
    "inlier_count", "inlier_ratio", "ransac_stage", "gate_score", "gate_threshold",
    "injected_wrong_transform", "injection_id", "detection_latency_events",
    "detection_latency_s", "recovery_success", "recovery_horizon_events",
    "recovery_events", "recovery_time_s", "contaminated_map_cycles",
    "contaminated_cells",
    "source_event_id", "base_frame_cluster", "source_manifest", "target_path",
    "source_path", "target_gate_path", "source_gate_path", "proposal_rule",
    "delta_tx_target_m", "delta_ty_target_m", "delta_translation_norm_m",
    "delta_yaw_deg", "measured_translation_error_m", "measured_yaw_error_deg",
    "target_support_px", "warped_source_support_px", "support_intersection_px",
    "pre_map_sha256", "post_map_sha256", "map_state_unchanged", "changed_cells",
    "counterfactual_wrong_vs_gt_cells", "T_hat_i_from_j_metric_json",
    "T_gt_i_from_j_metric_json", "H_hat_i_from_j_pixel_json",
    "H_gt_i_from_j_pixel_json", "evaluation_scope",
]


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    rigid_valid: bool
    reason: str
    score: float
    scale: float
    shear: float
    determinant: float
    warped_occupied: np.ndarray
    target_support_px: int
    warped_source_support_px: int
    intersection_px: int


def se2(tx_m: float, ty_m: float, yaw_deg: float) -> np.ndarray:
    theta = math.radians(yaw_deg)
    cosine, sine = math.cos(theta), math.sin(theta)
    return np.array(
        [[cosine, -sine, tx_m], [sine, cosine, ty_m], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def components(matrix: np.ndarray) -> tuple[float, float, float]:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("SE(2) matrix must be finite and 3x3")
    yaw = math.degrees(math.atan2(float(value[1, 0]), float(value[0, 0])))
    return float(value[0, 2]), float(value[1, 2]), wrap_degrees(yaw)


def wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def pixel_to_metric(width: int, height: int, resolution: float) -> np.ndarray:
    """Match the frozen benchmark/adapter centred-canvas convention."""
    return np.array(
        [
            [resolution, 0.0, -resolution * (width - 1) / 2.0],
            [0.0, -resolution, resolution * (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def metric_to_pixel(
    transform_metric: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    source_width: int,
    source_height: int,
    resolution: float,
) -> np.ndarray:
    g_target = pixel_to_metric(target_width, target_height, resolution)
    g_source = pixel_to_metric(source_width, source_height, resolution)
    return np.linalg.inv(g_target) @ transform_metric @ g_source


def manifest_ground_truth(event: Mapping[str, Any]) -> np.ndarray:
    value = event.get("T_gt_i_from_j_metric")
    if value is None:
        required = ("gt_tx_m", "gt_ty_m", "gt_yaw_deg")
        if any(event.get(name) is None for name in required):
            raise ValueError(f"{event.get('event_id')}: GT-positive event has no complete transform")
        value = se2(
            float(event["gt_tx_m"]),
            float(event["gt_ty_m"]),
            float(event["gt_yaw_deg"]),
        )
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{event.get('event_id')}: invalid T_gt_i_from_j_metric")
    tx_m, ty_m, yaw_deg = components(matrix)
    scalar = (event.get("gt_tx_m"), event.get("gt_ty_m"), event.get("gt_yaw_deg"))
    if all(item is not None for item in scalar):
        supplied = np.asarray([float(item) for item in scalar], dtype=float)
        derived = np.asarray([tx_m, ty_m, yaw_deg], dtype=float)
        if not np.allclose(supplied[:2], derived[:2], atol=1e-9) or abs(
            wrap_degrees(float(supplied[2] - derived[2]))
        ) > 1e-9:
            raise ValueError(f"{event.get('event_id')}: scalar and matrix GT disagree")
    return matrix


def frozen_wrong_proposal(
    event_id: str,
    ground_truth: np.ndarray,
    translation_m: float = DEFAULT_TRANSLATION_M,
    yaw_deg: float = DEFAULT_YAW_DEG,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a hash-frozen target-coordinate translation and yaw error.

    The GT-transformed source origin is shifted by a target-frame vector of
    exactly ``translation_m``.  Its two signs and the yaw sign come from the
    SHA-256 of ``event_id``.  Rotation is applied about that source origin, so
    the proposal's translation and yaw errors are exactly the requested values
    under the evaluator's ``inv(T_gt) @ T_hat`` convention.
    """
    if not event_id:
        raise ValueError("event_id must be non-empty")
    if translation_m <= 0 or yaw_deg <= 0:
        raise ValueError("injected translation and yaw magnitudes must be positive")
    digest = hashlib.sha256(("wrong-se2-v1:" + event_id).encode("utf-8")).digest()
    sign_x = -1.0 if digest[0] & 1 else 1.0
    sign_y = -1.0 if digest[1] & 1 else 1.0
    sign_yaw = -1.0 if digest[2] & 1 else 1.0
    delta_tx = sign_x * translation_m / math.sqrt(2.0)
    delta_ty = sign_y * translation_m / math.sqrt(2.0)
    delta_yaw = sign_yaw * yaw_deg

    proposal = np.eye(3, dtype=float)
    yaw_total = components(ground_truth)[2] + delta_yaw
    proposal[:2, :2] = se2(0.0, 0.0, yaw_total)[:2, :2]
    proposal[:2, 2] = ground_truth[:2, 2] + [delta_tx, delta_ty]
    metadata = {
        "injection_id": "offline_wrong_se2_v1_" + digest.hex()[:20],
        "delta_tx_target_m": delta_tx,
        "delta_ty_target_m": delta_ty,
        "delta_translation_norm_m": math.hypot(delta_tx, delta_ty),
        "delta_yaw_deg": delta_yaw,
    }
    return proposal, metadata


def se2_error(estimate: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    """Match evaluate_transform_events.py exactly."""
    hat_tx, hat_ty, hat_yaw = components(estimate)
    gt_tx, gt_ty, gt_yaw = components(ground_truth)
    delta_x, delta_y = hat_tx - gt_tx, hat_ty - gt_ty
    theta = math.radians(gt_yaw)
    error_x = math.cos(theta) * delta_x + math.sin(theta) * delta_y
    error_y = -math.sin(theta) * delta_x + math.cos(theta) * delta_y
    return math.hypot(error_x, error_y), abs(wrap_degrees(hat_yaw - gt_yaw))


def load_occupied(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read gate map: {path}")
    return image >= OCCUPIED_VALUE_THRESHOLD


def rigid_occupied_support_gate(
    target_occupied: np.ndarray,
    source_occupied: np.ndarray,
    h_i_from_j_pixel: np.ndarray,
    threshold: float = DEFAULT_GATE_THRESHOLD,
) -> GateResult:
    """Apply the adapter's rigid validity and occupied-support rules.

    Injected proposals have no feature matches, so the archived registrar's
    inlier-count rules are not applicable.  The remaining rules are identical:
    positive determinant, singular values within 2% of one, shear <= 0.01,
    and occupied overlap coefficient >= ``threshold``.
    """
    if target_occupied.ndim != 2 or source_occupied.ndim != 2:
        raise ValueError("occupied masks must be two-dimensional")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("gate threshold must lie in [0, 1]")
    matrix = np.asarray(h_i_from_j_pixel, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("pixel transform must be a finite 3x3 matrix")
    linear = matrix[:2, :2]
    singular = np.linalg.svd(linear, compute_uv=False)
    determinant = float(np.linalg.det(linear))
    scale = float(math.sqrt(abs(determinant)))
    shear = float(abs(singular[0] - singular[1]))
    rigid_valid = bool(
        determinant > 0.0
        and np.max(np.abs(singular - 1.0)) <= 0.02
        and shear <= 0.01
    )

    warped = np.zeros(target_occupied.shape, dtype=bool)
    score = 0.0
    intersection = 0
    warped_support = 0
    target_support = int(np.count_nonzero(target_occupied))
    if rigid_valid:
        warped = cv2.warpPerspective(
            source_occupied.astype(np.uint8),
            matrix,
            (target_occupied.shape[1], target_occupied.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        warped_support = int(np.count_nonzero(warped))
        intersection = int(np.count_nonzero(warped & target_occupied))
        smaller_support = min(warped_support, target_support)
        score = intersection / smaller_support if smaller_support else 0.0

    accepted = bool(rigid_valid and score >= threshold)
    reason = (
        "accepted"
        if accepted
        else "non_rigid_candidate"
        if not rigid_valid
        else "low_occupied_support_overlap"
    )
    return GateResult(
        accepted=accepted,
        rigid_valid=rigid_valid,
        reason=reason,
        score=score,
        scale=scale,
        shear=shear,
        determinant=determinant,
        warped_occupied=warped,
        target_support_px=target_support,
        warped_source_support_px=warped_support,
        intersection_px=intersection,
    )


def map_state_sha256(occupied: np.ndarray) -> str:
    state = np.ascontiguousarray(occupied, dtype=np.uint8)
    header = f"shape={state.shape};dtype=uint8;".encode("ascii")
    return hashlib.sha256(header + state.tobytes()).hexdigest()


def atomic_commit(
    persistent_target: np.ndarray,
    warped_source: np.ndarray,
    accepted: bool,
) -> tuple[np.ndarray, bool, int]:
    """Gate first, then produce a new persistent state in one operation."""
    if persistent_target.shape != warped_source.shape:
        raise ValueError("persistent and warped maps must have equal shapes")
    before = np.asarray(persistent_target, dtype=bool)
    if accepted:
        after = before | np.asarray(warped_source, dtype=bool)
        committed = True
    else:
        # Return a copy to prevent alias-based mutation after a rejected event.
        after = before.copy()
        committed = False
    changed = int(np.count_nonzero(after != before))
    if not accepted and changed != 0:
        raise AssertionError("rejected proposal modified the persistent map")
    return after, committed, changed


def json_matrix(matrix: np.ndarray) -> str:
    return json.dumps(np.asarray(matrix, dtype=float).tolist(), separators=(",", ":"))


def _path(event: Mapping[str, Any], preferred: str, fallback: str) -> Path:
    value = event.get(preferred) or event.get(fallback)
    if not value:
        raise ValueError(f"{event.get('event_id')}: missing {preferred}/{fallback}")
    return Path(str(value))


def evaluate_positive_event(
    event: Mapping[str, Any],
    *,
    manifest_path: Path,
    translation_m: float,
    yaw_deg: float,
    gate_threshold: float,
    tau_translation_m: float,
    tau_yaw_deg: float,
) -> dict[str, Any]:
    source_event_id = str(event.get("event_id", "")).strip()
    if not source_event_id or not bool(event.get("gt_positive")):
        raise ValueError("injection requires a named GT-positive event")
    ground_truth = manifest_ground_truth(event)
    proposal, injection = frozen_wrong_proposal(
        source_event_id, ground_truth, translation_m=translation_m, yaw_deg=yaw_deg
    )
    translation_error, yaw_error = se2_error(proposal, ground_truth)
    if translation_error <= tau_translation_m or yaw_error <= tau_yaw_deg:
        raise ValueError(
            f"{source_event_id}: injection is not beyond both analysis limits "
            f"({translation_error:g} m, {yaw_error:g} deg)"
        )

    target_gate_path = _path(event, "target_observed_path", "target_path")
    source_gate_path = _path(event, "source_observed_path", "source_path")
    target_occupied = load_occupied(target_gate_path)
    source_occupied = load_occupied(source_gate_path)
    target_height, target_width = target_occupied.shape
    source_height, source_width = source_occupied.shape
    declared_width = int(event.get("canvas_width_px", target_width))
    declared_height = int(event.get("canvas_height_px", target_height))
    if (declared_width, declared_height) != (target_width, target_height):
        raise ValueError(
            f"{source_event_id}: target gate shape disagrees with frozen manifest "
            f"({target_width}x{target_height} vs {declared_width}x{declared_height})"
        )
    resolution = float(event["resolution_m_per_px"])
    h_hat = metric_to_pixel(
        proposal,
        target_width=target_width,
        target_height=target_height,
        source_width=source_width,
        source_height=source_height,
        resolution=resolution,
    )
    h_gt = metric_to_pixel(
        ground_truth,
        target_width=target_width,
        target_height=target_height,
        source_width=source_width,
        source_height=source_height,
        resolution=resolution,
    )
    gate = rigid_occupied_support_gate(
        target_occupied, source_occupied, h_hat, threshold=gate_threshold
    )

    pre_hash = map_state_sha256(target_occupied)
    persistent_after, committed, changed_cells = atomic_commit(
        target_occupied, gate.warped_occupied, gate.accepted
    )
    post_hash = map_state_sha256(persistent_after)
    state_unchanged = pre_hash == post_hash
    if not gate.accepted and (committed or not state_unchanged or changed_cells != 0):
        raise AssertionError(f"{source_event_id}: rejected proposal violated atomic-commit contract")

    warped_gt = cv2.warpPerspective(
        source_occupied.astype(np.uint8),
        h_gt,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    correct_counterfactual = target_occupied | warped_gt
    wrong_counterfactual = target_occupied | gate.warped_occupied
    wrong_vs_gt = int(np.count_nonzero(wrong_counterfactual != correct_counterfactual))
    contaminated_cells: int | str = wrong_vs_gt if committed else 0
    hat_tx, hat_ty, hat_yaw = components(proposal)
    gt_tx, gt_ty, gt_yaw = components(ground_truth)

    # ``committed`` here describes only this harness's simulated state update.
    # Recovery fields remain blank because the frozen snapshots contain no
    # temporal rollback/recovery implementation to evaluate.
    return {
        "event_id": source_event_id + "__offline_wrong_se2_v1",
        "run_id": str(event.get("run_id", "")) + "__offline_wrong_se2_v1",
        "scene_id": str(event.get("scene_id", "")),
        "cluster_id": str(event.get("base_frame_cluster", "")),
        "seed": str(event.get("seed", "")),
        "input_type": str(event.get("input_type", "")),
        "method": f"OFFLINE_RIGID_OCCUPIED_SUPPORT_GATE_{gate_threshold:.2f}",
        "gt_positive": "true",
        "candidate_returned": "true",
        "rigid_valid": str(gate.rigid_valid).lower(),
        "accepted": str(gate.accepted).lower(),
        "committed": str(committed).lower(),
        "hat_tx_m": hat_tx, "hat_ty_m": hat_ty, "hat_yaw_deg": hat_yaw,
        "gt_tx_m": gt_tx, "gt_ty_m": gt_ty, "gt_yaw_deg": gt_yaw,
        "raw_scale": gate.scale, "raw_shear": gate.shear,
        "raw_determinant": gate.determinant, "reject_reason": gate.reason,
        "inlier_count": "", "inlier_ratio": "",
        "ransac_stage": "not_applicable_injected_proposal",
        "gate_score": gate.score, "gate_threshold": gate_threshold,
        "injected_wrong_transform": "true",
        "injection_id": injection["injection_id"],
        "detection_latency_events": 0 if not gate.accepted else "",
        "detection_latency_s": "", "recovery_success": "",
        "recovery_horizon_events": "", "recovery_events": "",
        "recovery_time_s": "", "contaminated_map_cycles": "",
        "contaminated_cells": contaminated_cells,
        "source_event_id": source_event_id,
        "base_frame_cluster": str(event.get("base_frame_cluster", "")),
        "source_manifest": str(manifest_path),
        "target_path": str(event.get("target_path", "")),
        "source_path": str(event.get("source_path", "")),
        "target_gate_path": str(target_gate_path),
        "source_gate_path": str(source_gate_path),
        "proposal_rule": "target-coordinate origin shift plus yaw; SHA256(event_id)-fixed signs",
        **injection,
        "measured_translation_error_m": translation_error,
        "measured_yaw_error_deg": yaw_error,
        "target_support_px": gate.target_support_px,
        "warped_source_support_px": gate.warped_source_support_px,
        "support_intersection_px": gate.intersection_px,
        "pre_map_sha256": pre_hash, "post_map_sha256": post_hash,
        "map_state_unchanged": str(state_unchanged).lower(),
        "changed_cells": changed_cells,
        "counterfactual_wrong_vs_gt_cells": wrong_vs_gt,
        "T_hat_i_from_j_metric_json": json_matrix(proposal),
        "T_gt_i_from_j_metric_json": json_matrix(ground_truth),
        "H_hat_i_from_j_pixel_json": json_matrix(h_hat),
        "H_gt_i_from_j_pixel_json": json_matrix(h_gt),
        "evaluation_scope": "offline gate injection; not deployed/online system evidence",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    manifest_path: Path,
    output_path: Path,
    *,
    translation_m: float = DEFAULT_TRANSLATION_M,
    yaw_deg: float = DEFAULT_YAW_DEG,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
    tau_translation_m: float = DEFAULT_TAU_TRANSLATION_M,
    tau_yaw_deg: float = DEFAULT_TAU_YAW_DEG,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("test_only") or not manifest.get("selection_locked_without_registrar_outputs"):
        raise ValueError("manifest is not a frozen test-only benchmark")
    positives = [event for event in manifest.get("events", []) if bool(event.get("gt_positive"))]
    if not positives:
        raise ValueError("manifest contains no GT-positive events")
    rows = [
        evaluate_positive_event(
            event,
            manifest_path=manifest_path,
            translation_m=translation_m,
            yaw_deg=yaw_deg,
            gate_threshold=gate_threshold,
            tau_translation_m=tau_translation_m,
            tau_yaw_deg=tau_yaw_deg,
        )
        for event in positives
    ]
    injection_ids = [str(row["injection_id"]) for row in rows]
    if len(set(injection_ids)) != len(injection_ids):
        raise AssertionError("injection_id collision")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    accepted = sum(row["accepted"] == "true" for row in rows)
    rejected = len(rows) - accepted
    summary = {
        "scope": "offline gate injection; not deployed/online system evidence",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "output_csv": output_path.as_posix(),
        "events_n": len(rows),
        "accepted_n": accepted,
        "rejected_n": rejected,
        "safe_rejected_n": sum(
            row["accepted"] == "false" and row["committed"] == "false" for row in rows
        ),
        "incorrect_commits_n": sum(row["committed"] == "true" for row in rows),
        "translation_injection_m": translation_m,
        "yaw_injection_deg": yaw_deg,
        "gate_threshold": gate_threshold,
        "analysis_correctness_limits": {
            "translation_m": tau_translation_m,
            "yaw_deg": tau_yaw_deg,
        },
        "proposal_rule": "target-coordinate origin shift plus yaw; SHA256(event_id)-fixed signs",
        "gate_rule": (
            "proper rigid pixel transform (singular values within 2%, shear <= 0.01) "
            f"and observed occupied-support overlap coefficient >= {gate_threshold:.6g}"
        ),
        "atomic_commit": "rejected candidate preserves pre-state; accepted candidate OR-merges warped support",
        "contaminated_cells_definition": (
            "symmetric-difference cells between immediate wrong-transform state and "
            "counterfactual GT-transform state; zero when no wrong transform is committed"
        ),
        "recovery_fields": "blank: no rollback/recovery sequence is present in the frozen snapshots",
    }
    summary_path = output_path.with_suffix(".run.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translation-m", type=float, default=DEFAULT_TRANSLATION_M)
    parser.add_argument("--yaw-deg", type=float, default=DEFAULT_YAW_DEG)
    parser.add_argument("--gate-threshold", type=float, default=DEFAULT_GATE_THRESHOLD)
    parser.add_argument("--tau-translation-m", type=float, default=DEFAULT_TAU_TRANSLATION_M)
    parser.add_argument("--tau-yaw-deg", type=float, default=DEFAULT_TAU_YAW_DEG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(
            args.manifest,
            args.output,
            translation_m=args.translation_m,
            yaw_deg=args.yaw_deg,
            gate_threshold=args.gate_threshold,
            tau_translation_m=args.tau_translation_m,
            tau_yaw_deg=args.tau_yaw_deg,
        )
    except (OSError, ValueError, AssertionError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
