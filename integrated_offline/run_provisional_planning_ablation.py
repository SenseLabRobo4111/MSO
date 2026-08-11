#!/usr/bin/env python3
"""Explore saved prediction maps as a disposable planning layer.

Both arms share the observed-only registrar result, observed-support gate, and
atomic measured-map commit.  Prediction is unavailable to registration and the
gate, is never persisted as measured state, and differs only at planner input.
This is a reconstructed offline single-scene audit, not an online navigation run
or a complete execution of the historical system.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrated_offline import reference_registrar
from integrated_offline.run_paired_ablation import (
    PREDICTED_INPUT_TYPE,
    TAU_TRANSLATION_M,
    TAU_YAW_DEG,
    UNKNOWN_VALUE,
    atomic_measured_commit,
    choose_start,
    cluster_groups,
    cluster_metric_distribution,
    cluster_rate_distribution,
    finite_values,
    known_mask,
    load_exploratory_manifest,
    matrix_from_registrar,
    metric_to_pixel,
    pixel_to_metric_transform,
    plan_frontier,
    pct,
    preferred_manifest_path,
    read_gray,
    require_equal_canvases,
    route_against_oracle,
    sha256_file,
    sign_summary,
    state_hash,
    transform_error,
    warp_semantic,
)


PREDICTION_UNKNOWN_VALUE = 0
PREDICTION_OCCUPIED_THRESHOLD = 128
ARMS = ("measured_only", "measured_plus_provisional")


FIELDS = (
    "event_id", "base_frame_cluster", "seed", "target_step", "source_step", "arm",
    "observed_gate_accepted", "observed_gate_score", "observed_gate_threshold",
    "observed_reject_reason", "observed_translation_error_m",
    "observed_yaw_error_deg", "observed_transform_within_thresholds",
    "measured_atomic_committed", "measured_changed_cells",
    "measured_persistent_hash", "planner_input_hash", "prediction_persisted",
    "source_prediction_included", "measured_known_support_cells",
    "planner_known_support_cells", "provisional_support_gain_cells",
    "provisional_free_gain_cells", "provisional_occupied_gain_cells",
    "reachable_free_cells", "reachable_frontiers", "route_available",
    "route_length_m", "route_oracle_safe", "route_oracle_collision_cells",
    "route_oracle_unknown_cells", "persistent_hash_after_planning",
    "persistent_state_changed_by_planning",
)


def prediction_to_categorical(raw_probability: np.ndarray) -> np.ndarray:
    """Apply the deployment 0.5 threshold while retaining zero as unknown."""
    raw = np.asarray(raw_probability, dtype=np.uint8)
    output = np.full(raw.shape, UNKNOWN_VALUE, dtype=np.uint8)
    support = raw != PREDICTION_UNKNOWN_VALUE
    output[support & (raw < PREDICTION_OCCUPIED_THRESHOLD)] = 0
    output[support & (raw >= PREDICTION_OCCUPIED_THRESHOLD)] = 255
    return output


def merge_probability_support(target_raw: np.ndarray, source_raw: np.ndarray) -> np.ndarray:
    """Form a conservative probability union without inventing support."""
    target = np.asarray(target_raw, dtype=np.uint8)
    source = np.asarray(source_raw, dtype=np.uint8)
    if target.shape != source.shape:
        raise ValueError("predicted probability canvases must have equal shapes")
    merged = target.copy()
    target_support = target != PREDICTION_UNKNOWN_VALUE
    source_support = source != PREDICTION_UNKNOWN_VALUE
    source_only = ~target_support & source_support
    both = target_support & source_support
    merged[source_only] = source[source_only]
    merged[both] = np.maximum(target[both], source[both])
    return merged


def measured_authoritative_overlay(
    provisional: np.ndarray, measured_persistent: np.ndarray
) -> np.ndarray:
    if provisional.shape != measured_persistent.shape:
        raise ValueError("provisional and measured grids must have equal shapes")
    output = np.asarray(provisional, dtype=np.uint8).copy()
    measured = np.asarray(measured_persistent, dtype=np.uint8)
    mask = known_mask(measured)
    output[mask] = measured[mask]
    return output


def warp_raw_prediction(
    source_raw: np.ndarray, transform: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    return cv2.warpPerspective(
        source_raw,
        np.asarray(transform, dtype=float),
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=PREDICTION_UNKNOWN_VALUE,
    )


def validate_prediction_canvases(event: Mapping[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    target = read_gray(event["target_path"])
    source = read_gray(event["source_path"])
    if target.shape != shape or source.shape != shape:
        raise ValueError(
            f"{event['event_id']}: predicted and observed rasters do not share one canvas"
        )
    if sha256_file(Path(str(event["target_path"]))) != str(event["target_sha256"]).lower():
        raise ValueError(f"{event['event_id']}: target prediction hash mismatch")
    if sha256_file(Path(str(event["source_path"]))) != str(event["source_sha256"]).lower():
        raise ValueError(f"{event['event_id']}: source prediction hash mismatch")
    return target, source


def event_rows(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    target_observed, source_observed = require_equal_canvases(event)
    target_prediction, source_prediction = validate_prediction_canvases(
        event, target_observed.shape
    )

    observed_event = dict(event)
    observed_event["target_path"] = observed_event["target_observed_path"]
    observed_event["source_path"] = observed_event["source_observed_path"]
    registration = reference_registrar.register(
        target_path=str(observed_event["target_path"]),
        source_path=str(observed_event["source_path"]),
        context={"event": observed_event, "benchmark": {}},
    )
    accepted = bool(registration.get("accepted"))
    transform = matrix_from_registrar(registration)
    if accepted and transform is None:
        raise AssertionError("observed gate accepted without a rigid transform")

    translation_error = None
    yaw_error = None
    transform_correct = False
    if transform is not None:
        truth = np.asarray(event["T_gt_i_from_j_metric"], dtype=float)
        translation_error, yaw_error = transform_error(
            pixel_to_metric_transform(transform, event), truth
        )
        transform_correct = (
            translation_error <= TAU_TRANSLATION_M and yaw_error <= TAU_YAW_DEG
        )

    if transform is None:
        warped_observed = np.full(
            target_observed.shape, UNKNOWN_VALUE, dtype=np.uint8
        )
    else:
        warped_observed = warp_semantic(
            source_observed, transform, target_observed.shape
        )
    persistent, committed, changed = atomic_measured_commit(
        target_observed, warped_observed, accepted
    )
    persistent_hash = state_hash(persistent)

    truth = np.asarray(event["T_gt_i_from_j_metric"], dtype=float)
    oracle_source = warp_semantic(
        source_observed, metric_to_pixel(truth, event), target_observed.shape
    )
    oracle, _, _ = atomic_measured_commit(target_observed, oracle_source, True)

    merged_probability = target_prediction.copy()
    source_prediction_included = accepted and transform is not None
    if source_prediction_included:
        warped_prediction = warp_raw_prediction(
            source_prediction, transform, target_observed.shape
        )
        merged_probability = merge_probability_support(
            target_prediction, warped_prediction
        )
    provisional = prediction_to_categorical(merged_probability)
    provisional = measured_authoritative_overlay(provisional, persistent)

    start = choose_start(persistent)
    measured_known = int(np.count_nonzero(known_mask(persistent)))
    rows = []
    for arm, planner_grid in (
        ("measured_only", persistent),
        ("measured_plus_provisional", provisional),
    ):
        plan, route = plan_frontier(
            planner_grid, start, float(event["resolution_m_per_px"])
        )
        safety = route_against_oracle(route, oracle)
        planner_known = int(np.count_nonzero(known_mask(planner_grid)))
        provisional_cells = known_mask(planner_grid) & ~known_mask(persistent)
        row = {
            "event_id": event["event_id"],
            "base_frame_cluster": event["base_frame_cluster"],
            "seed": event["seed"],
            "target_step": event["target_step"],
            "source_step": event["source_step"],
            "arm": arm,
            "observed_gate_accepted": accepted,
            "observed_gate_score": registration.get("gate_score"),
            "observed_gate_threshold": registration.get("gate_threshold"),
            "observed_reject_reason": registration.get("reject_reason", ""),
            "observed_translation_error_m": translation_error,
            "observed_yaw_error_deg": yaw_error,
            "observed_transform_within_thresholds": (
                transform_correct if transform is not None else None
            ),
            "measured_atomic_committed": committed,
            "measured_changed_cells": changed,
            "measured_persistent_hash": persistent_hash,
            "planner_input_hash": state_hash(planner_grid),
            "prediction_persisted": False,
            "source_prediction_included": source_prediction_included,
            "measured_known_support_cells": measured_known,
            "planner_known_support_cells": planner_known,
            "provisional_support_gain_cells": planner_known - measured_known,
            "provisional_free_gain_cells": int(
                np.count_nonzero(provisional_cells & (planner_grid == 0))
            ),
            "provisional_occupied_gain_cells": int(
                np.count_nonzero(provisional_cells & (planner_grid == 255))
            ),
            **plan,
            **safety,
            "persistent_hash_after_planning": state_hash(persistent),
            "persistent_state_changed_by_planning": (
                state_hash(persistent) != persistent_hash
            ),
        }
        if row["persistent_state_changed_by_planning"]:
            raise AssertionError("planner mutated measured persistent state")
        rows.append(row)
    if rows[0]["measured_persistent_hash"] != rows[1]["measured_persistent_hash"]:
        raise AssertionError("planning arms do not share one measured persistent map")
    return rows


def distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "mean": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def summarise(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = {
        arm: [row for row in rows if row["arm"] == arm]
        for arm in ARMS
    }
    arm_summary = {}
    for arm, arm_rows in selected.items():
        collision_rows = [
            dict(row, collision_route=row["route_oracle_collision_cells"] not in (None, 0))
            for row in arm_rows
        ]
        unknown_rows = [
            dict(row, unknown_route=row["route_oracle_unknown_cells"] not in (None, 0))
            for row in arm_rows
        ]
        fully_known_rows = [
            dict(row, fully_known_free=row["route_oracle_safe"] is True)
            for row in arm_rows
        ]
        arm_summary[arm] = {
            "events": len(arm_rows),
            "clusters": len(cluster_groups(arm_rows)),
            "route_available_cluster_rate": cluster_rate_distribution(
                arm_rows, "route_available"
            ),
            "known_support_gain_cells_cluster_mean": cluster_metric_distribution(
                arm_rows, "provisional_support_gain_cells"
            ),
            "reachable_free_cells_cluster_mean": cluster_metric_distribution(
                arm_rows, "reachable_free_cells"
            ),
            "reachable_frontiers_cluster_mean": cluster_metric_distribution(
                arm_rows, "reachable_frontiers"
            ),
            "route_length_m_cluster_mean": cluster_metric_distribution(
                arm_rows, "route_length_m"
            ),
            "oracle_collision_route_cluster_rate": cluster_rate_distribution(
                collision_rows, "collision_route"
            ),
            "oracle_unknown_traversal_cluster_rate": cluster_rate_distribution(
                unknown_rows, "unknown_route"
            ),
            "oracle_fully_known_free_cluster_rate": cluster_rate_distribution(
                fully_known_rows, "fully_known_free"
            ),
            "persistent_mutation_violation_clusters": sum(
                any(bool(row["persistent_state_changed_by_planning"]) for row in grouped_rows)
                for grouped_rows in cluster_groups(arm_rows).values()
            ),
        }

    indexed = {
        (str(row["event_id"]), str(row["arm"])): row
        for row in rows
    }
    event_differences = []
    for event_id in sorted({str(row["event_id"]) for row in rows}):
        measured = indexed[(event_id, "measured_only")]
        provisional = indexed[(event_id, "measured_plus_provisional")]
        measured_collision = measured["route_oracle_collision_cells"] not in (None, 0)
        provisional_collision = provisional["route_oracle_collision_cells"] not in (None, 0)
        measured_unknown = measured["route_oracle_unknown_cells"] not in (None, 0)
        provisional_unknown = provisional["route_oracle_unknown_cells"] not in (None, 0)
        route_length_difference = None
        if measured["route_length_m"] is not None and provisional["route_length_m"] is not None:
            route_length_difference = (
                float(provisional["route_length_m"])
                - float(measured["route_length_m"])
            )
        event_differences.append(
            {
                "event_id": event_id,
                "base_frame_cluster": measured["base_frame_cluster"],
                "observed_gate_accepted": measured["observed_gate_accepted"],
                "provisional_support_gain_cells": provisional["provisional_support_gain_cells"],
                "reachable_free_difference_cells": (
                    int(provisional["reachable_free_cells"])
                    - int(measured["reachable_free_cells"])
                ),
                "reachable_frontier_difference": (
                    int(provisional["reachable_frontiers"])
                    - int(measured["reachable_frontiers"])
                ),
                "route_length_difference_m": route_length_difference,
                "route_availability_difference": (
                    int(bool(provisional["route_available"]))
                    - int(bool(measured["route_available"]))
                ),
                "collision_route_difference": (
                    int(provisional_collision) - int(measured_collision)
                ),
                "unknown_traversal_difference": (
                    int(provisional_unknown) - int(measured_unknown)
                ),
            }
        )
    differences = []
    aggregate_fields = (
        "observed_gate_accepted",
        "provisional_support_gain_cells",
        "reachable_free_difference_cells",
        "reachable_frontier_difference",
        "route_length_difference_m",
        "route_availability_difference",
        "collision_route_difference",
        "unknown_traversal_difference",
    )
    for cluster, cluster_rows in sorted(cluster_groups(event_differences).items()):
        record: dict[str, Any] = {
            "base_frame_cluster": cluster,
            "event_count": len(cluster_rows),
        }
        for field in aggregate_fields:
            values = finite_values(cluster_rows, field)
            record[field] = float(np.mean(values)) if values else None
        differences.append(record)

    summary = {
        "scope": "exploratory saved-probability-map provisional-planning audit on one archived scene",
        "shared_observed_gate": {
            "events": len(selected["measured_only"]),
            "clusters": len(cluster_groups(selected["measured_only"])),
            "accepted_cluster_rate": cluster_rate_distribution(
                selected["measured_only"], "observed_gate_accepted"
            ),
            "accepted_outside_pose_threshold_cluster_rate": cluster_rate_distribution(
                [
                    dict(
                        row,
                        accepted_outside=(
                            bool(row["observed_gate_accepted"])
                            and not bool(row["observed_transform_within_thresholds"])
                        ),
                    )
                    for row in selected["measured_only"]
                ],
                "accepted_outside",
            ),
        },
        "arms": arm_summary,
        "cluster_aware_paired": {
            "events": len(event_differences),
            "clusters": len(differences),
            "route_availability_sign": sign_summary(
                differences, "route_availability_difference"
            ),
            "collision_route_sign": sign_summary(
                differences, "collision_route_difference"
            ),
            "unknown_traversal_sign": sign_summary(
                differences, "unknown_traversal_difference"
            ),
            "support_gain_cluster_mean_cells": distribution(
                finite_values(differences, "provisional_support_gain_cells")
            ),
            "reachable_free_cluster_mean_difference_cells": distribution(
                finite_values(differences, "reachable_free_difference_cells")
            ),
            "reachable_frontier_cluster_mean_difference": distribution(
                finite_values(differences, "reachable_frontier_difference")
            ),
            "route_length_cluster_mean_difference_m": distribution(
                finite_values(differences, "route_length_difference_m")
            ),
        },
    }
    return summary, differences


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: csv_value(row.get(field)) for field in fields}
            for row in rows
        )


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def render_report(summary: Mapping[str, Any], manifest_hash: str) -> str:
    measured = summary["arms"]["measured_only"]
    provisional = summary["arms"]["measured_plus_provisional"]
    paired = summary["cluster_aware_paired"]
    gate = summary["shared_observed_gate"]
    route_sign = paired["route_availability_sign"]
    collision_sign = paired["collision_route_sign"]
    unknown_sign = paired["unknown_traversal_sign"]
    lines = [
        "# Exploratory provisional-planning audit",
        "",
        "This reconstructed offline audit starts from saved probability maps and isolates prediction at a planner proxy input. Both arms share one observed-only reference transform, observed-support decision, atomic measured-map commit, persistent-state hash, start cell, and planner proxy. Prediction is never used by registration or the gate and is never persisted as measured occupancy.",
        "",
        "## Shared upstream path",
        "",
        f"The 50 perturbations form {gate['clusters']} independent `base_frame_cluster` units. The observed-only gate's mean within-cluster acceptance rate was {pct(gate['accepted_cluster_rate'])}; its mean within-cluster rate of accepted transforms outside 0.25 m / 5 deg was {pct(gate['accepted_outside_pose_threshold_cluster_rate'])}.",
        "",
        "## Cluster-aware planner result",
        "",
        "| Planner input | Route-available mean cluster rate | Median cluster-mean reachable frontiers | Median cluster-mean route length, m | Collision mean cluster rate | Unknown-traversal mean cluster rate | Fully known-free mean cluster rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Measured persistent only | {pct(measured['route_available_cluster_rate'])} | {fmt(measured['reachable_frontiers_cluster_mean']['median'], 1)} | {fmt(measured['route_length_m_cluster_mean']['median'])} | {pct(measured['oracle_collision_route_cluster_rate'])} | {pct(measured['oracle_unknown_traversal_cluster_rate'])} | {pct(measured['oracle_fully_known_free_cluster_rate'])} |",
        f"| Measured + disposable prediction | {pct(provisional['route_available_cluster_rate'])} | {fmt(provisional['reachable_frontiers_cluster_mean']['median'], 1)} | {fmt(provisional['route_length_m_cluster_mean']['median'])} | {pct(provisional['oracle_collision_route_cluster_rate'])} | {pct(provisional['oracle_unknown_traversal_cluster_rate'])} | {pct(provisional['oracle_fully_known_free_cluster_rate'])} |",
        "",
        f"Across {paired['clusters']} clusters, the disposable layer added a median {fmt(paired['support_gain_cluster_mean_cells']['median'], 1)} cells to the within-cluster support mean and changed the within-cluster reachable-free mean by a median {fmt(paired['reachable_free_cluster_mean_difference_cells']['median'], 1)} cells. The median cluster-mean route-length difference was {fmt(paired['route_length_cluster_mean_difference_m']['median'])} m.",
        "",
        f"Provisional-minus-measured cluster-mean route availability was positive/negative/zero in {route_sign['positive']}/{route_sign['negative']}/{route_sign['zero']} clusters (two-sided exact sign p={fmt(route_sign['exact_sign_p'])}). Collision-route signs were {collision_sign['positive']}/{collision_sign['negative']}/{collision_sign['zero']} (p={fmt(collision_sign['exact_sign_p'])}); unknown-traversal signs were {unknown_sign['positive']}/{unknown_sign['negative']}/{unknown_sign['zero']} (p={fmt(unknown_sign['exact_sign_p'])}). No event-level significance test is reported.",
        "",
        "These results do not demonstrate a safe or statistically resolved planning benefit from prediction. Support gain is not equivalent to explored area, and the oracle is only the union of two saved measured maps rather than complete environment ground truth.",
        "",
        "## Exploratory construction",
        "",
        "- Only the 50 known-positive `predicted_raw` A3 perturbations are used; they aggregate to 30 `base_frame_cluster` units.",
        "- A3 uses one common 1020 x 424 canvas and 0.05 m resolution. All predicted and observed file hashes and shapes are checked before processing.",
        "- Raw prediction value 0 is unsupported/unknown, 1--127 is provisional free, and 128--255 is provisional occupied. The 128 threshold follows the archived 0.5 rule. The design is declared exploratory, not preregistered or prelocked.",
        "- Accepted source probabilities are warped only by the shared observed-only estimate. Target and source probabilities use a conservative maximum on overlapping support. Measured cells then overwrite the provisional layer.",
        "- The measured persistent map is hashed before and after both planner calls. Prediction is disposable and cannot mutate this state.",
        "",
        "## Boundaries",
        "",
        "The cluster units remain snapshots from one archived A3 scene. The planner is a deterministic grid proxy, not the historical MRPB planner. This is not an online, physical, temporal-consensus, communication-impairment, or exploration-coverage experiment, and it is not a complete system execution.",
        "",
        f"Input manifest SHA-256: `{manifest_hash}`.",
        "",
    ]
    return "\n".join(lines)


def run(manifest_path: Path, output: Path, limit: int | None = None) -> dict[str, Any]:
    manifest_path, manifest, manifest_events = load_exploratory_manifest(manifest_path)
    events = [
        event
        for event in manifest_events
        if event.get("input_type") == PREDICTED_INPUT_TYPE
        and bool(event.get("gt_positive"))
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        events = events[:limit]
    if not events:
        raise ValueError("manifest contains no selected positive saved probability maps")

    rows = []
    for index, event in enumerate(events, start=1):
        rows.extend(event_rows(event))
        if index % 10 == 0 or index == len(events):
            print(f"processed {index}/{len(events)} planning pairs", flush=True)
    summary, differences = summarise(rows)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "planning_events.csv", rows, FIELDS)
    write_csv(
        output / "planning_differences.csv",
        differences,
        tuple(differences[0].keys()),
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_hash = sha256_file(manifest_path)
    (output / "REPORT.md").write_text(
        render_report(summary, manifest_hash), encoding="utf-8"
    )
    run_manifest = {
        "claim_scope": summary["scope"],
        "created_by": "integrated_offline/run_provisional_planning_ablation.py",
        "input_manifest_name": manifest_path.name,
        "input_manifest_sha256": manifest_hash,
        "selected_events": len(events),
        "base_frame_clusters": summary["cluster_aware_paired"]["clusters"],
        "output_rows": len(rows),
        "pipeline_script_sha256": sha256_file(Path(__file__)),
        "reference_registrar_sha256": sha256_file(
            Path(reference_registrar.__file__)
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "exploratory_rules": {
            "prediction_unknown_value": PREDICTION_UNKNOWN_VALUE,
            "prediction_occupied_threshold": PREDICTION_OCCUPIED_THRESHOLD,
            "probability_overlap": "maximum on supported cells",
            "measured_cells_authoritative": True,
            "prediction_persisted": False,
            "registrar_input": "observed only",
            "observed_support_gate": reference_registrar.MIN_OBSERVED_SUPPORT_OVERLAP,
            "selection_lock_self_declaration_used_for_inference": False,
            "relative_asset_paths_supported": True,
        },
        "limitations": [
            "exploratory reconstruction from saved probability maps",
            "single archived A3 scene with repeated synthetic perturbations",
            "oracle is two-map measured union, not full environment ground truth",
            "deterministic planner proxy; not historical MRPB planner",
            "no online, physical, scaling, or communication claim",
        ],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result_files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in result_files),
        encoding="ascii",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, nargs="?", default=preferred_manifest_path(),
        help="portable A3 manifest (defaults to the Nature submission asset bundle)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("integrated_offline/planning_results"),
    )
    parser.add_argument("--limit", type=int, help="debug-only prefix")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    summary = run(arguments.manifest, arguments.output, arguments.limit)
    print(json.dumps(summary, indent=2))
    print("EXPLORATORY OFFLINE ONLY: saved prediction maps are disposable planner input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    load_exploratory_manifest,
    preferred_manifest_path,
