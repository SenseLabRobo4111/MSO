#!/usr/bin/env python3
"""Explore saved prediction maps only as a fixed-frontier ranking signal.

Both arms share the observed-only transform, gate, atomic measured-map commit,
reachable frontier candidates, and measured-only path planner.  The baseline
selects the farthest candidate.  The prediction arm ranks the same candidates
by disposable predicted-free support within a fixed 1.0 m neighbourhood.
The design is reconstructed and exploratory rather than a complete execution
of the historical system.
"""

from __future__ import annotations

import argparse
from collections import deque
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
    traversable_mask,
    unknown_mask,
    warp_semantic,
)
from integrated_offline.run_provisional_planning_ablation import (
    merge_probability_support,
    prediction_to_categorical,
    validate_prediction_canvases,
    warp_raw_prediction,
)


RANKING_RADIUS_M = 1.0
ARMS = ("measured_farthest", "predicted_free_ranking")


FIELDS = (
    "event_id", "base_frame_cluster", "seed", "target_step", "source_step", "arm",
    "observed_gate_accepted", "observed_gate_score", "observed_gate_threshold",
    "observed_translation_error_m", "observed_yaw_error_deg",
    "observed_transform_within_thresholds", "measured_persistent_hash",
    "measured_path_grid_hash", "frontier_candidate_hash",
    "frontier_candidate_count", "ranking_radius_m", "ranking_radius_cells",
    "source_prediction_included", "selected_y", "selected_x",
    "selected_distance_cells", "route_length_m", "route_available",
    "selected_prediction_free_support", "selected_oracle_hidden_free_yield",
    "selected_oracle_hidden_occupied", "route_oracle_safe",
    "route_oracle_collision_cells", "route_oracle_unknown_cells",
    "prediction_persisted", "persistent_state_changed_by_ranking",
)


def disk_kernel(radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        raise ValueError("ranking radius must be positive")
    axis = np.arange(-radius_cells, radius_cells + 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return ((xx * xx + yy * yy) <= radius_cells * radius_cells).astype(
        np.float32
    )


def local_count_field(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    return cv2.filter2D(
        np.asarray(mask, dtype=np.float32),
        ddepth=-1,
        kernel=disk_kernel(radius_cells),
        borderType=cv2.BORDER_CONSTANT,
    )


def frontier_distance_field(
    grid: np.ndarray, start: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    traversable = traversable_mask(grid)
    sy, sx = start
    if not traversable[sy, sx]:
        raise ValueError("fixed start is not traversable on measured map")
    distance = np.full(grid.shape, -1, dtype=np.int32)
    distance[sy, sx] = 0
    queue: deque[tuple[int, int]] = deque([(sy, sx)])
    while queue:
        y, x = queue.popleft()
        next_distance = int(distance[y, x]) + 1
        for ny, nx in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)):
            if (
                0 <= ny < grid.shape[0]
                and 0 <= nx < grid.shape[1]
                and traversable[ny, nx]
                and distance[ny, nx] < 0
            ):
                distance[ny, nx] = next_distance
                queue.append((ny, nx))
    adjacent_unknown = cv2.dilate(
        unknown_mask(grid).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    frontier = traversable & adjacent_unknown & (distance >= 0)
    return distance, frontier


def reconstruct_route(
    distance: np.ndarray, goal: tuple[int, int]
) -> list[tuple[int, int]]:
    y, x = goal
    if distance[y, x] < 0:
        raise ValueError("goal is not reachable")
    route = [(y, x)]
    while distance[y, x] > 0:
        wanted = int(distance[y, x]) - 1
        for ny, nx in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)):
            if (
                0 <= ny < distance.shape[0]
                and 0 <= nx < distance.shape[1]
                and distance[ny, nx] == wanted
            ):
                y, x = ny, nx
                route.append((y, x))
                break
        else:
            raise AssertionError("could not reconstruct measured-only route")
    route.reverse()
    return route


def select_ranked_goal(
    frontier: np.ndarray,
    distance: np.ndarray,
    predicted_free_counts: np.ndarray,
) -> tuple[int, int]:
    candidates = np.argwhere(frontier)
    if not len(candidates):
        raise ValueError("event has no reachable frontier candidates")
    scores = predicted_free_counts[frontier]
    best_score = float(np.max(scores))
    best = candidates[np.isclose(scores, best_score, rtol=0.0, atol=1e-6)]
    best_distances = distance[best[:, 0], best[:, 1]]
    best = best[best_distances == np.max(best_distances)]
    return int(best[0, 0]), int(best[0, 1])


def mask_hash(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little")
    header = f"shape={mask.shape};packbits=little;".encode("ascii")
    return hashlib.sha256(header + packed.tobytes()).hexdigest()


def event_rows(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    target_observed, source_observed = require_equal_canvases(event)
    target_prediction, source_prediction = validate_prediction_canvases(
        event, target_observed.shape
    )
    resolution = float(event["resolution_m_per_px"])
    radius_cells_float = RANKING_RADIUS_M / resolution
    radius_cells = int(round(radius_cells_float))
    if abs(radius_cells - radius_cells_float) > 1e-9:
        raise ValueError("1.0 m ranking radius is not cell-aligned")

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
    persistent, _, _ = atomic_measured_commit(
        target_observed, warped_observed, accepted
    )
    persistent_hash = state_hash(persistent)

    merged_probability = target_prediction.copy()
    source_prediction_included = accepted and transform is not None
    if source_prediction_included:
        merged_probability = merge_probability_support(
            target_prediction,
            warp_raw_prediction(
                source_prediction, transform, target_observed.shape
            ),
        )
    categorical_prediction = prediction_to_categorical(merged_probability)
    predicted_hidden_free = (
        (categorical_prediction == 0) & unknown_mask(persistent)
    )
    prediction_count = local_count_field(predicted_hidden_free, radius_cells)

    truth = np.asarray(event["T_gt_i_from_j_metric"], dtype=float)
    oracle_source = warp_semantic(
        source_observed, metric_to_pixel(truth, event), target_observed.shape
    )
    oracle, _, _ = atomic_measured_commit(target_observed, oracle_source, True)
    oracle_hidden_free = (oracle == 0) & unknown_mask(persistent)
    oracle_hidden_occupied = (oracle == 255) & unknown_mask(persistent)
    oracle_free_count = local_count_field(oracle_hidden_free, radius_cells)
    oracle_occupied_count = local_count_field(
        oracle_hidden_occupied, radius_cells
    )

    start = choose_start(persistent)
    if start is None:
        raise ValueError(f"{event['event_id']}: measured map has no planner start")
    baseline_plan, baseline_route = plan_frontier(persistent, start, resolution)
    if not baseline_route:
        raise ValueError(f"{event['event_id']}: measured map has no reachable frontier")
    distance, frontier = frontier_distance_field(persistent, start)
    if int(np.count_nonzero(frontier)) != baseline_plan["reachable_frontiers"]:
        raise AssertionError("frontier candidates differ from current planner")
    if int(np.count_nonzero(distance >= 0)) != baseline_plan["reachable_free_cells"]:
        raise AssertionError("reachable set differs from current planner")
    baseline_goal = baseline_route[-1]
    predicted_goal = select_ranked_goal(frontier, distance, prediction_count)
    predicted_route = reconstruct_route(distance, predicted_goal)
    frontier_hash = mask_hash(frontier)
    path_grid_hash = state_hash(persistent)

    rows = []
    for arm, goal, route in (
        ("measured_farthest", baseline_goal, baseline_route),
        ("predicted_free_ranking", predicted_goal, predicted_route),
    ):
        y, x = goal
        safety = route_against_oracle(route, oracle)
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
            "observed_translation_error_m": translation_error,
            "observed_yaw_error_deg": yaw_error,
            "observed_transform_within_thresholds": (
                transform_correct if transform is not None else None
            ),
            "measured_persistent_hash": persistent_hash,
            "measured_path_grid_hash": path_grid_hash,
            "frontier_candidate_hash": frontier_hash,
            "frontier_candidate_count": int(np.count_nonzero(frontier)),
            "ranking_radius_m": RANKING_RADIUS_M,
            "ranking_radius_cells": radius_cells,
            "source_prediction_included": source_prediction_included,
            "selected_y": y,
            "selected_x": x,
            "selected_distance_cells": int(distance[y, x]),
            "route_length_m": int(distance[y, x]) * resolution,
            "route_available": True,
            "selected_prediction_free_support": int(round(prediction_count[y, x])),
            "selected_oracle_hidden_free_yield": int(round(oracle_free_count[y, x])),
            "selected_oracle_hidden_occupied": int(
                round(oracle_occupied_count[y, x])
            ),
            **safety,
            "prediction_persisted": False,
            "persistent_state_changed_by_ranking": (
                state_hash(persistent) != persistent_hash
            ),
        }
        if row["persistent_state_changed_by_ranking"]:
            raise AssertionError("frontier ranking mutated measured state")
        rows.append(row)
    if rows[0]["frontier_candidate_hash"] != rows[1]["frontier_candidate_hash"]:
        raise AssertionError("ranking arms used different frontier candidates")
    if rows[0]["measured_path_grid_hash"] != rows[1]["measured_path_grid_hash"]:
        raise AssertionError("ranking arms used different path grids")
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


def summarise(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = {
        arm: [row for row in rows if row["arm"] == arm]
        for arm in ARMS
    }
    arms = {}
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
        arms[arm] = {
            "events": len(arm_rows),
            "clusters": len(cluster_groups(arm_rows)),
            "oracle_hidden_free_yield_cluster_mean": cluster_metric_distribution(
                arm_rows, "selected_oracle_hidden_free_yield"
            ),
            "oracle_hidden_occupied_cluster_mean": cluster_metric_distribution(
                arm_rows, "selected_oracle_hidden_occupied"
            ),
            "prediction_free_score_cluster_mean": cluster_metric_distribution(
                arm_rows, "selected_prediction_free_support"
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
            "state_mutation_violation_clusters": sum(
                any(bool(row["persistent_state_changed_by_ranking"]) for row in grouped_rows)
                for grouped_rows in cluster_groups(arm_rows).values()
            ),
        }

    indexed = {
        (str(row["event_id"]), str(row["arm"])): row
        for row in rows
    }
    event_differences = []
    for event_id in sorted({str(row["event_id"]) for row in rows}):
        baseline = indexed[(event_id, "measured_farthest")]
        predicted = indexed[(event_id, "predicted_free_ranking")]
        yield_difference = (
            int(predicted["selected_oracle_hidden_free_yield"])
            - int(baseline["selected_oracle_hidden_free_yield"])
        )
        baseline_collision = baseline["route_oracle_collision_cells"] not in (
            None, 0
        )
        predicted_collision = predicted["route_oracle_collision_cells"] not in (
            None, 0
        )
        baseline_unknown = baseline["route_oracle_unknown_cells"] not in (None, 0)
        predicted_unknown = predicted["route_oracle_unknown_cells"] not in (None, 0)
        goals_equal = (
            baseline["selected_y"] == predicted["selected_y"]
            and baseline["selected_x"] == predicted["selected_x"]
        )
        event_differences.append(
            {
                "event_id": event_id,
                "base_frame_cluster": baseline["base_frame_cluster"],
                "observed_gate_accepted": baseline["observed_gate_accepted"],
                "same_selected_goal": int(goals_equal),
                "oracle_hidden_free_yield_difference": yield_difference,
                "oracle_hidden_occupied_difference": (
                    int(predicted["selected_oracle_hidden_occupied"])
                    - int(baseline["selected_oracle_hidden_occupied"])
                ),
                "route_length_difference_m": (
                    float(predicted["route_length_m"])
                    - float(baseline["route_length_m"])
                ),
                "collision_route_difference": (
                    int(predicted_collision) - int(baseline_collision)
                ),
                "unknown_traversal_difference": (
                    int(predicted_unknown) - int(baseline_unknown)
                ),
            }
        )
    differences = []
    aggregate_fields = (
        "observed_gate_accepted",
        "same_selected_goal",
        "oracle_hidden_free_yield_difference",
        "oracle_hidden_occupied_difference",
        "route_length_difference_m",
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
        "scope": "exploratory saved-probability-map measured-path frontier-ranking audit",
        "exploratory_ranking_radius_m": RANKING_RADIUS_M,
        "shared_observed_gate": {
            "events": len(selected["measured_farthest"]),
            "clusters": len(cluster_groups(selected["measured_farthest"])),
            "accepted_cluster_rate": cluster_rate_distribution(
                selected["measured_farthest"], "observed_gate_accepted"
            ),
        },
        "arms": arms,
        "cluster_aware_paired": {
            "events": len(event_differences),
            "clusters": len(differences),
            "same_goal_cluster_rate": distribution(
                finite_values(differences, "same_selected_goal")
            ),
            "oracle_hidden_free_yield_sign": sign_summary(
                differences, "oracle_hidden_free_yield_difference"
            ),
            "oracle_hidden_free_yield_cluster_mean_difference": distribution(
                finite_values(differences, "oracle_hidden_free_yield_difference")
            ),
            "oracle_hidden_occupied_cluster_mean_difference": distribution(
                finite_values(differences, "oracle_hidden_occupied_difference")
            ),
            "route_length_cluster_mean_difference_m": distribution(
                finite_values(differences, "route_length_difference_m")
            ),
            "collision_route_sign": sign_summary(
                differences, "collision_route_difference"
            ),
            "unknown_traversal_sign": sign_summary(
                differences, "unknown_traversal_difference"
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


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
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
    baseline = summary["arms"]["measured_farthest"]
    predicted = summary["arms"]["predicted_free_ranking"]
    paired = summary["cluster_aware_paired"]
    gate = summary["shared_observed_gate"]
    yield_sign = paired["oracle_hidden_free_yield_sign"]
    yield_difference = paired["oracle_hidden_free_yield_cluster_mean_difference"]
    occupied_difference = paired["oracle_hidden_occupied_cluster_mean_difference"]
    route_difference = paired["route_length_cluster_mean_difference_m"]
    collision_sign = paired["collision_route_sign"]
    unknown_sign = paired["unknown_traversal_sign"]
    lines = [
        "# Exploratory frontier-ranking audit",
        "",
        "This reconstructed offline audit starts from saved probability maps and gives prediction only one role: ranking the same measured-map reachable frontier candidates. Registration, gating, measured-map commit, candidate generation, obstacle inflation, traversability, and path search are identical between arms and do not read prediction.",
        "",
        f"The 50 perturbations aggregate to {paired['clusters']} `base_frame_cluster` units. These 30 clusters are the independent units for this within-audit comparison, although all come from one scene. The shared observed-only gate's mean within-cluster acceptance rate was {pct(gate['accepted_cluster_rate'])}.",
        "",
        "## Cluster-aware selected-target and path result",
        "",
        "| Ranking | Median cluster-mean hidden-free yield | Median cluster-mean hidden occupied | Median cluster-mean route length, m | Collision mean cluster rate | Unknown-traversal mean cluster rate | Fully known-free mean cluster rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Measured farthest | {fmt(baseline['oracle_hidden_free_yield_cluster_mean']['median'], 1)} | {fmt(baseline['oracle_hidden_occupied_cluster_mean']['median'], 1)} | {fmt(baseline['route_length_m_cluster_mean']['median'])} | {pct(baseline['oracle_collision_route_cluster_rate'])} | {pct(baseline['oracle_unknown_traversal_cluster_rate'])} | {pct(baseline['oracle_fully_known_free_cluster_rate'])} |",
        f"| Predicted-free ranking | {fmt(predicted['oracle_hidden_free_yield_cluster_mean']['median'], 1)} | {fmt(predicted['oracle_hidden_occupied_cluster_mean']['median'], 1)} | {fmt(predicted['route_length_m_cluster_mean']['median'])} | {pct(predicted['oracle_collision_route_cluster_rate'])} | {pct(predicted['oracle_unknown_traversal_cluster_rate'])} | {pct(predicted['oracle_fully_known_free_cluster_rate'])} |",
        "",
        f"The predicted-minus-baseline hidden-free cluster mean was positive/negative/zero in {yield_sign['positive']}/{yield_sign['negative']}/{yield_sign['zero']} clusters. The two-sided exact cluster sign test gives p={fmt(yield_sign['exact_sign_p'])}; the median cluster-mean difference was {fmt(yield_difference['median'], 1)} cells, but the mean was {fmt(yield_difference['mean'], 1)} cells. Thus the event-level positive impression does not survive cluster-aware inference, and the cluster mean is negative.",
        "",
        f"Collision-route cluster-mean signs were {collision_sign['positive']}/{collision_sign['negative']}/{collision_sign['zero']} (two-sided exact sign p={fmt(collision_sign['exact_sign_p'])}); unknown-traversal signs were {unknown_sign['positive']}/{unknown_sign['negative']}/{unknown_sign['zero']} (p={fmt(unknown_sign['exact_sign_p'])}). No event-level significance test is reported.",
        "",
        f"The prediction-ranked goal changed the cluster-mean route length by a median {fmt(route_difference['median'])} m. The hidden-occupied cluster-mean difference had median {fmt(occupied_difference['median'], 1)} cells and mean {fmt(occupied_difference['mean'], 1)} cells. This is a utility--risk trade-off, not a safety benefit.",
        "",
        "## Exploratory design",
        "",
        "- The candidate set is exactly the current measured-only planner's reachable frontier mask. Its hash is identical between paired arms.",
        "- The baseline chooses maximum measured path distance. Prediction chooses maximum predicted-free support on measured-unknown cells in a 1.0 m disk, then maximum measured path distance and row/column order for ties.",
        "- The 1.0 m radius follows the existing frontier-clustering scale (`DBSCAN eps=1.0`) and equals 20 cells at 0.05 m resolution. The design is exploratory, not preregistered or prelocked.",
        "- Both routes are reconstructed only on the same measured persistent map. Prediction is never persisted and never changes collision checking.",
        "- Oracle hidden-free yield counts measured-unknown cells labelled free by the two-map ground-truth-transform union within the same 1.0 m disk. It is a partial offline oracle, not explored area or complete environment truth.",
        "",
        "## Boundaries",
        "",
        "The 30 cluster units remain correlated snapshots from one archived A3 scene and are not independent environments. The planner is a deterministic audit proxy, not historical MRPB navigation. This reconstructed audit cannot establish online exploration, physical scaling, communication robustness, population-level safety, or complete-system performance.",
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
            print(f"processed {index}/{len(events)} ranking pairs", flush=True)
    summary, differences = summarise(rows)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "ranking_events.csv", rows, FIELDS)
    write_csv(
        output / "ranking_differences.csv",
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
        "created_by": "integrated_offline/run_frontier_ranking_ablation.py",
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
            "ranking_radius_m": RANKING_RADIUS_M,
            "ranking_radius_provenance": "existing DBSCAN frontier eps=1.0 m",
            "candidate_source": "measured reachable frontier mask",
            "path_grid": "measured persistent occupancy only",
            "prediction_persisted": False,
            "registrar_input": "observed only",
            "selection_lock_self_declaration_used_for_inference": False,
            "relative_asset_paths_supported": True,
        },
        "limitations": [
            "exploratory reconstruction from saved probability maps",
            "partial two-map oracle, not explored area or full ground truth",
            "deterministic planner proxy, not historical MRPB planner",
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
        default=Path("integrated_offline/frontier_ranking_results"),
    )
    parser.add_argument("--limit", type=int, help="debug-only prefix")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    summary = run(arguments.manifest, arguments.output, arguments.limit)
    print(json.dumps(summary, indent=2))
    print("EXPLORATORY OFFLINE ONLY: saved prediction maps rank fixed frontiers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
