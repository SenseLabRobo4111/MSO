#!/usr/bin/env python3
"""Run a frozen synthetic-SE(2) manifest through an external registrar adapter."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


EVENT_FIELDS = [
    "event_id", "run_id", "scene_id", "seed", "input_type", "method",
    "gt_positive", "candidate_returned", "accepted", "committed",
    "hat_tx_m", "hat_ty_m", "hat_yaw_deg", "gt_tx_m", "gt_ty_m", "gt_yaw_deg",
    "injected_wrong_transform", "injection_id", "detection_latency_events",
    "detection_latency_s", "recovery_success", "recovery_horizon_events",
    "recovery_events", "recovery_time_s", "contaminated_map_cycles", "contaminated_cells",
    "cluster_id", "base_frame_cluster", "registrar_adapter", "target_path", "source_path",
    "rigid_valid", "reject_reason", "raw_scale", "raw_shear", "inlier_count",
    "inlier_ratio", "ransac_stage", "raw_determinant", "gate_score", "gate_threshold",
    "raw_h_i_from_j_pixel_json",
]


def parse_adapter(specification: str) -> Callable[..., Mapping[str, Any] | None]:
    if ":" not in specification:
        raise ValueError("registrar must be written as module:function")
    module_name, function_name = specification.rsplit(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError(f"{specification!r} does not resolve to a callable")
    return function


def as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"adapter field {field!r} must be Boolean")


def homogeneous_matrix(value: Any, field: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, [0.0, 0.0, 1.0]))
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"adapter field {field!r} must be a finite 2x3 or 3x3 matrix")
    if abs(matrix[2, 2]) < 1e-12:
        raise ValueError(f"adapter field {field!r} has invalid homogeneous scale")
    matrix = matrix / matrix[2, 2]
    if not np.allclose(matrix[2, :], [0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"adapter field {field!r} is projective; an explicit SE(2) estimate is required")
    return matrix


def pixel_to_metric(width: int, height: int, resolution: float) -> np.ndarray:
    return np.array(
        [
            [resolution, 0.0, -resolution * (width - 1) / 2.0],
            [0.0, -resolution, resolution * (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def rigid_components(matrix: np.ndarray) -> tuple[float, float, float]:
    rotation = matrix[:2, :2]
    if not np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-3) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-3
    ):
        raise ValueError(
            "registrar returned scale/shear/reflection; project to SE(2) explicitly in the adapter "
            "and audit the discarded components"
        )
    yaw_deg = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    return float(matrix[0, 2]), float(matrix[1, 2]), yaw_deg


def estimate_from_result(result: Mapping[str, Any], event: Mapping[str, Any]) -> tuple[float, float, float] | None:
    scalar_names = ("hat_tx_m", "hat_ty_m", "hat_yaw_deg")
    scalar_present = [name in result and result[name] is not None for name in scalar_names]
    if any(scalar_present):
        if not all(scalar_present):
            raise ValueError("adapter returned a partial scalar metric transform")
        values = tuple(float(result[name]) for name in scalar_names)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("adapter returned a non-finite scalar metric transform")
        return values  # type: ignore[return-value]

    if "T_i_from_j_metric" in result and result["T_i_from_j_metric"] is not None:
        return rigid_components(homogeneous_matrix(result["T_i_from_j_metric"], "T_i_from_j_metric"))

    if "H_i_from_j_pixel" in result and result["H_i_from_j_pixel"] is not None:
        pixel = homogeneous_matrix(result["H_i_from_j_pixel"], "H_i_from_j_pixel")
        resolution = float(event["resolution_m_per_px"])
        g_target = pixel_to_metric(
            int(event["canvas_width_px"]), int(event["canvas_height_px"]), resolution
        )
        g_source = pixel_to_metric(
            int(event.get("source_canvas_width_px", event["canvas_width_px"])),
            int(event.get("source_canvas_height_px", event["canvas_height_px"])),
            resolution,
        )
        metric = g_target @ pixel @ np.linalg.inv(g_source)
        return rigid_components(metric)
    return None


def optional_bool(value: Any, field: str) -> bool | None:
    if value is None or value == "":
        return None
    return as_bool(value, field)


def raw_matrix_json(result: Mapping[str, Any]) -> str:
    value = result.get("raw_H_i_from_j_pixel")
    if value is None:
        value = result.get("H_i_from_j_pixel")
    if value is None:
        return ""
    matrix = homogeneous_matrix(value, "H_i_from_j_pixel")
    return json.dumps(matrix.tolist(), separators=(",", ":"))


def blank_row(event: Mapping[str, Any], method: str, adapter_spec: str) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "run_id": event["run_id"],
        "scene_id": event["scene_id"],
        "seed": event["seed"],
        "input_type": event["input_type"],
        "method": method,
        "gt_positive": str(bool(event["gt_positive"])).lower(),
        "candidate_returned": "false",
        "accepted": "false",
        "committed": "false",
        "hat_tx_m": "", "hat_ty_m": "", "hat_yaw_deg": "",
        "gt_tx_m": "" if event.get("gt_tx_m") is None else event["gt_tx_m"],
        "gt_ty_m": "" if event.get("gt_ty_m") is None else event["gt_ty_m"],
        "gt_yaw_deg": "" if event.get("gt_yaw_deg") is None else event["gt_yaw_deg"],
        "injected_wrong_transform": "false", "injection_id": "",
        "detection_latency_events": "", "detection_latency_s": "", "recovery_success": "",
        "recovery_horizon_events": "", "recovery_events": "", "recovery_time_s": "",
        "contaminated_map_cycles": "", "contaminated_cells": "",
        "cluster_id": event["base_frame_cluster"],
        "base_frame_cluster": event["base_frame_cluster"],
        "registrar_adapter": adapter_spec,
        "target_path": event["target_path"],
        "source_path": event["source_path"],
        "rigid_valid": "", "reject_reason": "", "raw_scale": "", "raw_shear": "",
        "inlier_count": "", "inlier_ratio": "", "ransac_stage": "", "raw_determinant": "",
        "gate_score": "", "gate_threshold": "",
        "raw_h_i_from_j_pixel_json": "",
    }


def run_manifest(
    manifest: Mapping[str, Any],
    registrar: Callable[..., Mapping[str, Any] | None],
    adapter_spec: str,
    method: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not manifest.get("test_only") or not manifest.get("selection_locked_without_registrar_outputs"):
        raise ValueError("manifest is not a frozen test-only benchmark")
    events: Sequence[Mapping[str, Any]] = manifest.get("events", [])
    if limit is not None:
        events = events[:limit]
    rows: list[dict[str, Any]] = []
    for event in events:
        target = Path(str(event["target_path"]))
        source = Path(str(event["source_path"]))
        if not target.is_file() or not source.is_file():
            raise FileNotFoundError(f"missing event asset for {event['event_id']}: {target} or {source}")
        context = {
            "event": dict(event),
            "benchmark": {key: value for key, value in manifest.items() if key != "events"},
        }
        result = registrar(target_path=str(target), source_path=str(source), context=context)
        if result is None:
            result = {"candidate_returned": False, "accepted": False, "committed": False}
        if not isinstance(result, Mapping):
            raise ValueError(f"{event['event_id']}: adapter must return a mapping or None")
        rigid_valid = optional_bool(result.get("rigid_valid"), "rigid_valid")
        raw_h = raw_matrix_json(result)
        if rigid_valid is False:
            # Keep a raw affine/projective candidate observable without silently
            # turning it into an SE(2) pose.  The evaluator treats it as wrong.
            estimate = None
        else:
            estimate = estimate_from_result(result, event)
            if rigid_valid is None and estimate is not None:
                rigid_valid = True
        raw_candidate_present = raw_h != "" or result.get("T_i_from_j_metric") is not None
        returned = as_bool(
            result.get("candidate_returned", estimate is not None or raw_candidate_present),
            "candidate_returned",
        )
        if returned and estimate is None and rigid_valid is not False:
            raise ValueError(
                f"{event['event_id']}: candidate has no metric SE(2); adapter must set rigid_valid=false "
                "for a raw non-rigid candidate"
            )
        if not returned and (estimate is not None or raw_candidate_present):
            raise ValueError(f"{event['event_id']}: candidate_returned=false but a transform was supplied")
        accepted = as_bool(result.get("accepted", False), "accepted")
        committed = as_bool(result.get("committed", False), "committed")
        if accepted and not returned:
            raise ValueError(f"{event['event_id']}: accepted=true without a candidate")
        if committed and not accepted:
            raise ValueError(f"{event['event_id']}: committed=true without acceptance")
        row = blank_row(event, method, adapter_spec)
        row["candidate_returned"] = str(returned).lower()
        row["accepted"] = str(accepted).lower()
        row["committed"] = str(committed).lower()
        if estimate is not None:
            row["hat_tx_m"], row["hat_ty_m"], row["hat_yaw_deg"] = estimate
        row["rigid_valid"] = "" if rigid_valid is None else str(rigid_valid).lower()
        row["reject_reason"] = str(result.get("reject_reason", ""))
        for field in (
            "raw_scale", "raw_shear", "inlier_count", "inlier_ratio", "ransac_stage",
            "raw_determinant", "gate_score", "gate_threshold",
        ):
            row[field] = result.get(field, "")
        row["raw_h_i_from_j_pixel_json"] = raw_h
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--registrar", help="external adapter as module:function")
    parser.add_argument("--method", default="USER_REGISTRAR")
    parser.add_argument("--output", type=Path, default=Path("events.csv"))
    parser.add_argument("--limit", type=int, help="run only the first N manifest events")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    count = len(manifest.get("events", []))
    if not args.registrar:
        print(f"Validated manifest with {count} events. No registrar supplied; no event CSV was written.")
        print("Use --registrar module:function to call the actual matcher through a thin adapter.")
        return 0
    registrar = parse_adapter(args.registrar)
    rows = run_manifest(manifest, registrar, args.registrar, args.method, args.limit)
    if not rows:
        raise ValueError("manifest/limit selected no events")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} registrar events to {args.output.resolve()}")
    print("These are test-only offline stress-test events; run evaluate_transform_events.py separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
