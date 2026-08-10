#!/usr/bin/env python3
"""Run a reconstructed predictor-map-versus-observed exploratory audit.

The experiment starts from saved probability maps.  It does not rerun the
historical trainer and it is not an online robot experiment.  Each archived A3
event is processed in two arms that share the event, synthetic transform,
observed-support gate, thresholds, random seed, commit rule, and planner rule.
Only the reconstructed registrar input differs.  This is not an end-to-end
execution of the historical system.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrated_offline import reference_registrar


PREDICTED_INPUT_TYPE = "predicted_raw"
TAU_TRANSLATION_M = 0.25
TAU_YAW_DEG = 5.0
UNKNOWN_VALUE = 100
UNKNOWN_TOLERANCE = 3
FREE_MAX_VALUE = 10
OCCUPIED_MIN_VALUE = 200
OBSTACLE_INFLATION_CELLS = 2
PORTABLE_MANIFEST_RELATIVE = Path(
    "paper/nature_communications/peer_review_data/transform_validation/manifest.json"
)
ASSET_PATH_FIELDS = (
    "target_path",
    "source_path",
    "source_original_path",
    "target_observed_path",
    "source_observed_path",
    "source_observed_original_path",
)


EVENT_FIELDS = (
    "event_id", "base_frame_cluster", "seed", "target_step", "source_step",
    "gt_positive", "evaluation_role", "known_error_injection", "arm",
    "candidate_returned", "rigid_valid", "gate_accepted", "gate_score",
    "gate_threshold", "reject_reason", "inlier_count", "inlier_ratio",
    "translation_error_m", "yaw_error_deg", "within_thresholds",
    "correct_positive_accept", "incorrect_positive_accept",
    "positive_abstention", "indeterminate_challenge_accept",
    "false_accept_known_injection", "atomic_committed",
    "changed_cells", "state_unchanged_on_reject", "state_hash_before",
    "state_hash_after", "known_cells_before", "known_cells_after",
    "known_coverage_gain_cells", "known_coverage_gain_fraction",
    "known_precision_vs_oracle", "known_recall_vs_oracle",
    "false_known_cells_vs_oracle", "missed_known_cells_vs_oracle",
    "semantic_accuracy_on_mutually_known", "added_cells_correct_fraction",
    "reachable_free_cells", "reachable_frontiers", "route_available",
    "route_length_m", "route_oracle_safe", "route_oracle_collision_cells",
    "route_oracle_unknown_cells", "oracle_known_cells",
    "oracle_reachable_free_cells", "oracle_reachable_frontiers",
    "oracle_route_available", "oracle_route_length_m",
)


def preferred_manifest_path() -> Path:
    """Return the repository's public portable manifest when it is available."""
    repository = Path(__file__).resolve().parents[1]
    return repository / PORTABLE_MANIFEST_RELATIVE


def resolve_event_paths(
    event: Mapping[str, Any], manifest_directory: Path
) -> dict[str, Any]:
    """Resolve portable relative asset paths without mutating manifest records."""
    resolved = dict(event)
    for field in ASSET_PATH_FIELDS:
        value = resolved.get(field)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = manifest_directory / path
        resolved[field] = str(path.resolve())
    cluster = str(resolved.get("base_frame_cluster", "")).strip()
    if not cluster:
        raise ValueError(f"{resolved.get('event_id', '<unknown>')}: missing base_frame_cluster")
    resolved["base_frame_cluster"] = cluster
    return resolved


def load_exploratory_manifest(
    manifest_path: Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Load one test-only A3 manifest and resolve its portable asset paths."""
    path = manifest_path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("test_only"):
        raise ValueError("input manifest is not marked test-only")
    if manifest.get("profile") != "a3":
        raise ValueError("this exploratory audit requires the equal-canvas A3 profile")
    events = [resolve_event_paths(event, path.parent) for event in manifest.get("events", [])]
    return path, manifest, events


def evaluation_role(event: Mapping[str, Any]) -> tuple[str, bool]:
    """Separate known positives, explicit injections, and indeterminate challenges."""
    if bool(event.get("gt_positive")):
        return "known_transform_positive", False
    known_injection = bool(event.get("known_error_injection", False))
    if known_injection:
        return "known_error_injection", True
    return "indeterminate_abstention_challenge", False


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def state_hash(grid: np.ndarray) -> str:
    array = np.ascontiguousarray(grid, dtype=np.uint8)
    header = f"shape={array.shape};dtype=uint8;".encode("ascii")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read occupancy raster: {path}")
    return image


def unknown_mask(grid: np.ndarray) -> np.ndarray:
    return np.abs(grid.astype(np.int16) - UNKNOWN_VALUE) <= UNKNOWN_TOLERANCE


def known_mask(grid: np.ndarray) -> np.ndarray:
    return ~unknown_mask(grid)


def free_mask(grid: np.ndarray) -> np.ndarray:
    return grid <= FREE_MAX_VALUE


def occupied_mask(grid: np.ndarray) -> np.ndarray:
    return grid >= OCCUPIED_MIN_VALUE


def pixel_to_metric(width: int, height: int, resolution: float) -> np.ndarray:
    return np.array(
        [
            [resolution, 0.0, -resolution * (width - 1) / 2.0],
            [0.0, -resolution, resolution * (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def metric_to_pixel(transform: np.ndarray, event: Mapping[str, Any]) -> np.ndarray:
    resolution = float(event["resolution_m_per_px"])
    target = pixel_to_metric(
        int(event["canvas_width_px"]), int(event["canvas_height_px"]), resolution
    )
    source = pixel_to_metric(
        int(event.get("source_canvas_width_px", event["canvas_width_px"])),
        int(event.get("source_canvas_height_px", event["canvas_height_px"])),
        resolution,
    )
    return np.linalg.inv(target) @ np.asarray(transform, dtype=float) @ source


def pixel_to_metric_transform(pixel: np.ndarray, event: Mapping[str, Any]) -> np.ndarray:
    resolution = float(event["resolution_m_per_px"])
    target = pixel_to_metric(
        int(event["canvas_width_px"]), int(event["canvas_height_px"]), resolution
    )
    source = pixel_to_metric(
        int(event.get("source_canvas_width_px", event["canvas_width_px"])),
        int(event.get("source_canvas_height_px", event["canvas_height_px"])),
        resolution,
    )
    return target @ np.asarray(pixel, dtype=float) @ np.linalg.inv(source)


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def transform_error(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    delta = np.linalg.inv(truth) @ estimate
    translation = float(np.linalg.norm(delta[:2, 2]))
    yaw = abs(wrap_degrees(math.degrees(math.atan2(delta[1, 0], delta[0, 0]))))
    return translation, yaw


def warp_semantic(source: np.ndarray, h_source_to_target: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(
        source,
        np.asarray(h_source_to_target, dtype=float),
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=UNKNOWN_VALUE,
    )


def atomic_measured_commit(
    persistent: np.ndarray,
    warped_measured_source: np.ndarray,
    accepted: bool,
) -> tuple[np.ndarray, bool, int]:
    """Fill only unknown target cells and never mutate state after rejection."""
    if persistent.shape != warped_measured_source.shape:
        raise ValueError("persistent and source grids must have equal shapes")
    before = np.asarray(persistent, dtype=np.uint8)
    if not accepted:
        after = before.copy()
        changed = int(np.count_nonzero(after != before))
        if changed:
            raise AssertionError("a rejected proposal changed persistent state")
        return after, False, changed

    after = before.copy()
    fill = unknown_mask(before) & known_mask(warped_measured_source)
    after[fill] = warped_measured_source[fill]
    return after, True, int(np.count_nonzero(after != before))


def traversable_mask(grid: np.ndarray) -> np.ndarray:
    inflated = cv2.dilate(
        occupied_mask(grid).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=OBSTACLE_INFLATION_CELLS,
    ).astype(bool)
    return free_mask(grid) & ~inflated


def choose_start(grid: np.ndarray) -> tuple[int, int] | None:
    traversable = traversable_mask(grid)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        traversable.astype(np.uint8), connectivity=4
    )
    if count <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == label)
    center_y = (grid.shape[0] - 1) / 2.0
    center_x = (grid.shape[1] - 1) / 2.0
    order = np.lexsort((xs, ys, (ys - center_y) ** 2 + (xs - center_x) ** 2))
    index = int(order[0])
    return int(ys[index]), int(xs[index])


def plan_frontier(
    grid: np.ndarray,
    start: tuple[int, int] | None,
    resolution_m: float,
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    """Run a deterministic four-connected farthest-frontier grid planner."""
    if start is None:
        return {
            "reachable_free_cells": 0,
            "reachable_frontiers": 0,
            "route_available": False,
            "route_length_m": None,
        }, []
    traversable = traversable_mask(grid)
    sy, sx = start
    if not (0 <= sy < grid.shape[0] and 0 <= sx < grid.shape[1] and traversable[sy, sx]):
        return {
            "reachable_free_cells": 0,
            "reachable_frontiers": 0,
            "route_available": False,
            "route_length_m": None,
        }, []

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
        unknown_mask(grid).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ).astype(bool)
    frontier = traversable & adjacent_unknown & (distance >= 0)
    frontier_count = int(np.count_nonzero(frontier))
    result = {
        "reachable_free_cells": int(np.count_nonzero(distance >= 0)),
        "reachable_frontiers": frontier_count,
        "route_available": bool(frontier_count),
        "route_length_m": None,
    }
    if not frontier_count:
        return result, []

    maximum = int(distance[frontier].max())
    goals = np.argwhere(frontier & (distance == maximum))
    goal_y, goal_x = (int(value) for value in goals[0])
    route = [(goal_y, goal_x)]
    y, x = goal_y, goal_x
    while distance[y, x] > 0:
        wanted = int(distance[y, x]) - 1
        candidates = (
            (y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)
        )
        for ny, nx in candidates:
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and distance[ny, nx] == wanted:
                y, x = ny, nx
                route.append((y, x))
                break
        else:
            raise AssertionError("planner could not reconstruct its breadth-first route")
    route.reverse()
    result["route_length_m"] = maximum * resolution_m
    return result, route


def route_against_oracle(route: Sequence[tuple[int, int]], oracle: np.ndarray) -> dict[str, Any]:
    if not route:
        return {
            "route_oracle_safe": None,
            "route_oracle_collision_cells": None,
            "route_oracle_unknown_cells": None,
        }
    inflated = cv2.dilate(
        occupied_mask(oracle).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=OBSTACLE_INFLATION_CELLS,
    ).astype(bool)
    collision = sum(bool(inflated[y, x]) for y, x in route)
    unseen = sum(bool(unknown_mask(oracle)[y, x]) for y, x in route)
    return {
        "route_oracle_safe": collision == 0 and unseen == 0,
        "route_oracle_collision_cells": collision,
        "route_oracle_unknown_cells": unseen,
    }


def oracle_comparison(
    before: np.ndarray,
    after: np.ndarray,
    oracle: np.ndarray,
) -> dict[str, Any]:
    after_known = known_mask(after)
    oracle_known = known_mask(oracle)
    intersection = after_known & oracle_known
    false_known = after_known & ~oracle_known
    missed_known = ~after_known & oracle_known
    changed = after != before
    correct_added = changed & oracle_known & (after == oracle)
    common_count = int(np.count_nonzero(intersection))
    changed_count = int(np.count_nonzero(changed))
    return {
        "known_precision_vs_oracle": (
            common_count / int(np.count_nonzero(after_known)) if np.any(after_known) else None
        ),
        "known_recall_vs_oracle": (
            common_count / int(np.count_nonzero(oracle_known)) if np.any(oracle_known) else None
        ),
        "false_known_cells_vs_oracle": int(np.count_nonzero(false_known)),
        "missed_known_cells_vs_oracle": int(np.count_nonzero(missed_known)),
        "semantic_accuracy_on_mutually_known": (
            float(np.mean(after[intersection] == oracle[intersection])) if common_count else None
        ),
        "added_cells_correct_fraction": (
            int(np.count_nonzero(correct_added)) / changed_count if changed_count else None
        ),
    }


def matrix_from_registrar(result: Mapping[str, Any]) -> np.ndarray | None:
    value = result.get("H_i_from_j_pixel")
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, [0.0, 0.0, 1.0]))
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("registrar returned an invalid pixel transform")
    return matrix


def require_equal_canvases(event: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    hash_fields = (
        ("target_path", "target_sha256"),
        ("source_path", "source_sha256"),
        ("target_observed_path", "target_observed_sha256"),
        ("source_observed_path", "source_observed_sha256"),
    )
    for path_field, hash_field in hash_fields:
        expected_hash = str(event.get(hash_field, "")).lower()
        if not expected_hash:
            raise ValueError(f"{event['event_id']}: missing archived asset hash {hash_field}")
        actual_hash = sha256_file(Path(str(event[path_field])))
        if actual_hash != expected_hash:
            raise ValueError(
                f"{event['event_id']}: {path_field} hash does not match the source manifest"
            )
    target = read_gray(event["target_observed_path"])
    source = read_gray(event["source_observed_path"])
    expected = (int(event["canvas_height_px"]), int(event["canvas_width_px"]))
    source_expected = (
        int(event.get("source_canvas_height_px", event["canvas_height_px"])),
        int(event.get("source_canvas_width_px", event["canvas_width_px"])),
    )
    if target.shape != expected or source.shape != source_expected or target.shape != source.shape:
        raise ValueError(
            f"{event['event_id']}: paired audit requires equal, manifest-consistent canvases"
        )
    return target, source


def evaluate_arm(
    event: Mapping[str, Any],
    arm: str,
    target_observed: np.ndarray,
    source_observed: np.ndarray,
    start: tuple[int, int] | None,
    oracle: np.ndarray | None,
    oracle_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    arm_event = dict(event)
    if arm == "observed_only":
        arm_event["target_path"] = arm_event["target_observed_path"]
        arm_event["source_path"] = arm_event["source_observed_path"]
    elif arm != "predicted":
        raise ValueError(f"unsupported arm: {arm}")

    result = reference_registrar.register(
        target_path=str(arm_event["target_path"]),
        source_path=str(arm_event["source_path"]),
        context={"event": arm_event, "benchmark": {}},
    )
    candidate = bool(result.get("candidate_returned"))
    accepted = bool(result.get("accepted"))
    rigid_valid = bool(result.get("rigid_valid"))
    pixel = matrix_from_registrar(result)
    if accepted and pixel is None:
        raise AssertionError("gate accepted without a rigid transform")

    positive = bool(event["gt_positive"])
    role, known_error_injection = evaluation_role(event)
    translation_error = None
    yaw_error = None
    correct = False
    if positive and pixel is not None:
        truth = np.asarray(event["T_gt_i_from_j_metric"], dtype=float)
        translation_error, yaw_error = transform_error(
            pixel_to_metric_transform(pixel, event), truth
        )
        correct = translation_error <= TAU_TRANSLATION_M and yaw_error <= TAU_YAW_DEG

    if pixel is None:
        warped = np.full(target_observed.shape, UNKNOWN_VALUE, dtype=np.uint8)
    else:
        warped = warp_semantic(source_observed, pixel, target_observed.shape)
    before_hash = state_hash(target_observed)
    after, committed, changed = atomic_measured_commit(target_observed, warped, accepted)
    after_hash = state_hash(after)
    if (not accepted) and before_hash != after_hash:
        raise AssertionError("rejected gate altered the persistent-map hash")

    resolution = float(event["resolution_m_per_px"])
    plan, route = plan_frontier(after, start, resolution)
    known_before = int(np.count_nonzero(known_mask(target_observed)))
    known_after = int(np.count_nonzero(known_mask(after)))
    row: dict[str, Any] = {
        "event_id": event["event_id"],
        "base_frame_cluster": event["base_frame_cluster"],
        "seed": event["seed"],
        "target_step": event["target_step"],
        "source_step": event["source_step"],
        "gt_positive": positive,
        "evaluation_role": role,
        "known_error_injection": known_error_injection,
        "arm": arm,
        "candidate_returned": candidate,
        "rigid_valid": rigid_valid,
        "gate_accepted": accepted,
        "gate_score": result.get("gate_score"),
        "gate_threshold": result.get("gate_threshold"),
        "reject_reason": result.get("reject_reason", ""),
        "inlier_count": result.get("inlier_count"),
        "inlier_ratio": result.get("inlier_ratio"),
        "translation_error_m": translation_error,
        "yaw_error_deg": yaw_error,
        "within_thresholds": correct if positive and pixel is not None else None,
        "correct_positive_accept": accepted and positive and correct,
        "incorrect_positive_accept": accepted and positive and not correct,
        "positive_abstention": positive and not accepted,
        "indeterminate_challenge_accept": (
            accepted and role == "indeterminate_abstention_challenge"
        ),
        "false_accept_known_injection": accepted and known_error_injection,
        "atomic_committed": committed,
        "changed_cells": changed,
        "state_unchanged_on_reject": (not accepted and before_hash == after_hash),
        "state_hash_before": before_hash,
        "state_hash_after": after_hash,
        "known_cells_before": known_before,
        "known_cells_after": known_after,
        "known_coverage_gain_cells": known_after - known_before,
        "known_coverage_gain_fraction": (
            (known_after - known_before) / known_before if known_before else None
        ),
        **plan,
    }
    if oracle is not None and oracle_plan is not None:
        row.update(oracle_comparison(target_observed, after, oracle))
        row.update(route_against_oracle(route, oracle))
        row.update(
            {
                "oracle_known_cells": int(np.count_nonzero(known_mask(oracle))),
                "oracle_reachable_free_cells": oracle_plan["reachable_free_cells"],
                "oracle_reachable_frontiers": oracle_plan["reachable_frontiers"],
                "oracle_route_available": oracle_plan["route_available"],
                "oracle_route_length_m": oracle_plan["route_length_m"],
            }
        )
    else:
        for field in EVENT_FIELDS:
            row.setdefault(field, None)
    return row


def paired_rows(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events, start=1):
        target_observed, source_observed = require_equal_canvases(event)
        start = choose_start(target_observed)
        oracle = None
        oracle_plan = None
        if bool(event["gt_positive"]):
            truth = np.asarray(event["T_gt_i_from_j_metric"], dtype=float)
            truth_pixel = metric_to_pixel(truth, event)
            oracle_warped = warp_semantic(source_observed, truth_pixel, target_observed.shape)
            oracle, _, _ = atomic_measured_commit(target_observed, oracle_warped, True)
            oracle_plan, _ = plan_frontier(
                oracle, start, float(event["resolution_m_per_px"])
            )
        for arm in ("predicted", "observed_only"):
            rows.append(
                evaluate_arm(
                    event, arm, target_observed, source_observed, start, oracle, oracle_plan
                )
            )
        if index % 10 == 0 or index == len(events):
            print(f"processed {index}/{len(events)} paired events", flush=True)
    return rows


def finite_values(rows: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is not None and value != "" and np.isfinite(float(value)):
            values.append(float(value))
    return values


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


def exact_sign_p(positive: int, negative: int) -> float:
    """Two-sided exact sign test over independent cluster summaries."""
    nonzero = positive + negative
    if nonzero == 0:
        return 1.0
    tail = sum(math.comb(nonzero, k) for k in range(0, min(positive, negative) + 1))
    return min(1.0, 2.0 * tail / (2 ** nonzero))


def cluster_groups(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_frame_cluster"])].append(row)
    return dict(grouped)


def cluster_metric_distribution(
    rows: Iterable[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    means = []
    for grouped_rows in cluster_groups(rows).values():
        values = finite_values(grouped_rows, field)
        if values:
            means.append(float(np.mean(values)))
    return distribution(means)


def cluster_rate_distribution(
    rows: Iterable[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    rates = [
        float(np.mean([bool(row.get(field)) for row in grouped_rows]))
        for grouped_rows in cluster_groups(rows).values()
    ]
    return distribution(rates)


def clusters_with_violation(
    rows: Iterable[Mapping[str, Any]], field: str
) -> int:
    return sum(
        any(bool(row.get(field)) for row in grouped_rows)
        for grouped_rows in cluster_groups(rows).values()
    )


def sign_summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = finite_values(rows, field)
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    zero = sum(value == 0 for value in values)
    return {
        "clusters": len(values),
        "positive": int(positive),
        "negative": int(negative),
        "zero": int(zero),
        "exact_sign_p": exact_sign_p(int(positive), int(negative)),
    }


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ("predicted", "observed_only")}
    arms = {}
    for arm, selected in by_arm.items():
        positives = [
            row for row in selected
            if row["evaluation_role"] == "known_transform_positive"
        ]
        challenges = [
            row for row in selected
            if row["evaluation_role"] == "indeterminate_abstention_challenge"
        ]
        injections = [
            row for row in selected
            if row["evaluation_role"] == "known_error_injection"
        ]
        arms[arm] = {
            "events": len(selected),
            "clusters": len(cluster_groups(selected)),
            "positive_events": len(positives),
            "positive_clusters": len(cluster_groups(positives)),
            "indeterminate_challenge_events": len(challenges),
            "indeterminate_challenge_clusters": len(cluster_groups(challenges)),
            "known_error_injection_events": len(injections),
            "known_error_injection_clusters": len(cluster_groups(injections)),
            "candidate_returned_cluster_rate": cluster_rate_distribution(
                selected, "candidate_returned"
            ),
            "correct_positive_accept_cluster_rate": cluster_rate_distribution(
                positives, "correct_positive_accept"
            ),
            "incorrect_positive_accept_cluster_rate": cluster_rate_distribution(
                positives, "incorrect_positive_accept"
            ),
            "positive_abstention_cluster_rate": cluster_rate_distribution(
                positives, "positive_abstention"
            ),
            "indeterminate_challenge_accept_cluster_rate": cluster_rate_distribution(
                challenges, "indeterminate_challenge_accept"
            ),
            "false_accept_known_injection_cluster_rate": cluster_rate_distribution(
                injections, "false_accept_known_injection"
            ),
            "rejected_state_hash_violation_clusters": sum(
                any(
                    not bool(row["gate_accepted"])
                    and not bool(row["state_unchanged_on_reject"])
                    for row in grouped_rows
                )
                for grouped_rows in cluster_groups(selected).values()
            ),
            "translation_error_m_cluster_mean": cluster_metric_distribution(
                positives, "translation_error_m"
            ),
            "yaw_error_deg_cluster_mean": cluster_metric_distribution(
                positives, "yaw_error_deg"
            ),
            "coverage_gain_cells_cluster_mean": cluster_metric_distribution(
                positives, "known_coverage_gain_cells"
            ),
            "known_recall_vs_oracle_cluster_mean": cluster_metric_distribution(
                positives, "known_recall_vs_oracle"
            ),
            "route_length_m_cluster_mean": cluster_metric_distribution(
                positives, "route_length_m"
            ),
            "oracle_fully_known_free_cluster_rate": cluster_rate_distribution(
                [dict(row, oracle_fully_known_free=row.get("route_oracle_safe") is True) for row in positives],
                "oracle_fully_known_free",
            ),
            "oracle_collision_route_cluster_rate": cluster_rate_distribution(
                [dict(row, oracle_collision=row.get("route_oracle_collision_cells") not in (None, 0)) for row in positives],
                "oracle_collision",
            ),
            "oracle_unknown_traversal_cluster_rate": cluster_rate_distribution(
                [dict(row, oracle_unknown=row.get("route_oracle_unknown_cells") not in (None, 0)) for row in positives],
                "oracle_unknown",
            ),
        }

    indexed = {
        (str(row["event_id"]), str(row["arm"])): row
        for row in rows
    }
    pairs = sorted({str(row["event_id"]) for row in rows})
    event_differences = []
    for event_id in pairs:
        predicted = indexed[(event_id, "predicted")]
        observed = indexed[(event_id, "observed_only")]
        predicted_collision = predicted.get("route_oracle_collision_cells") not in (None, 0)
        observed_collision = observed.get("route_oracle_collision_cells") not in (None, 0)
        event_differences.append(
            {
                "event_id": event_id,
                "base_frame_cluster": predicted["base_frame_cluster"],
                "evaluation_role": predicted["evaluation_role"],
                "gt_positive": predicted["gt_positive"],
                "correct_positive_accept_difference": (
                    int(bool(predicted["correct_positive_accept"]))
                    - int(bool(observed["correct_positive_accept"]))
                ),
                "incorrect_positive_accept_difference": (
                    int(bool(predicted["incorrect_positive_accept"]))
                    - int(bool(observed["incorrect_positive_accept"]))
                ),
                "indeterminate_challenge_accept_difference": (
                    int(bool(predicted["indeterminate_challenge_accept"]))
                    - int(bool(observed["indeterminate_challenge_accept"]))
                ),
                "false_accept_known_injection_difference": (
                    int(bool(predicted["false_accept_known_injection"]))
                    - int(bool(observed["false_accept_known_injection"]))
                ),
                "predicted_coverage_gain_cells": predicted["known_coverage_gain_cells"],
                "observed_coverage_gain_cells": observed["known_coverage_gain_cells"],
                "coverage_gain_difference_cells": (
                    int(predicted["known_coverage_gain_cells"])
                    - int(observed["known_coverage_gain_cells"])
                ),
                "collision_route_difference": (
                    int(predicted_collision) - int(observed_collision)
                ),
            }
        )
    differences = []
    difference_fields = (
        "correct_positive_accept_difference",
        "incorrect_positive_accept_difference",
        "indeterminate_challenge_accept_difference",
        "false_accept_known_injection_difference",
        "coverage_gain_difference_cells",
        "collision_route_difference",
    )
    for cluster, cluster_rows in sorted(cluster_groups(event_differences).items()):
        record: dict[str, Any] = {
            "base_frame_cluster": cluster,
            "evaluation_role": cluster_rows[0]["evaluation_role"],
            "event_count": len(cluster_rows),
        }
        for field in difference_fields:
            values = finite_values(cluster_rows, field)
            record[field] = float(np.mean(values)) if values else None
        differences.append(record)
    positive_differences = [
        row for row in differences
        if row["evaluation_role"] == "known_transform_positive"
    ]
    challenge_differences = [
        row for row in differences
        if row["evaluation_role"] == "indeterminate_abstention_challenge"
    ]
    injection_differences = [
        row for row in differences
        if row["evaluation_role"] == "known_error_injection"
    ]
    return {
        "scope": "reconstructed exploratory audit of saved probability maps on one archived A3 scene",
        "correctness_thresholds": {
            "translation_m": TAU_TRANSLATION_M,
            "absolute_yaw_deg": TAU_YAW_DEG,
        },
        "arms": arms,
        "cluster_aware_paired": {
            "events": len(pairs),
            "clusters": len(differences),
            "known_positive_clusters": len(positive_differences),
            "indeterminate_challenge_clusters": len(challenge_differences),
            "known_error_injection_clusters": len(injection_differences),
            "correct_positive_accept_sign": sign_summary(
                positive_differences, "correct_positive_accept_difference"
            ),
            "incorrect_positive_accept_sign": sign_summary(
                positive_differences, "incorrect_positive_accept_difference"
            ),
            "indeterminate_challenge_accept_sign": sign_summary(
                challenge_differences, "indeterminate_challenge_accept_difference"
            ),
            "false_accept_known_injection_sign": sign_summary(
                injection_differences, "false_accept_known_injection_difference"
            ),
            "collision_route_sign": sign_summary(
                positive_differences, "collision_route_difference"
            ),
            "positive_coverage_gain_cluster_mean_difference_cells": distribution(
                finite_values(positive_differences, "coverage_gain_difference_cells")
            ),
        },
        "paired_differences": differences,
    }


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
        writer.writerows({field: csv_value(row.get(field)) for field in fields} for row in rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def pct(summary: Mapping[str, Any], digits: int = 1) -> str:
    value = summary.get("mean")
    return "NA" if value is None else f"{100.0 * float(value):.{digits}f}%"


def render_report(summary: Mapping[str, Any], manifest_hash: str) -> str:
    predicted = summary["arms"]["predicted"]
    observed = summary["arms"]["observed_only"]
    paired = summary["cluster_aware_paired"]
    correct_sign = paired["correct_positive_accept_sign"]
    incorrect_sign = paired["incorrect_positive_accept_sign"]
    challenge_sign = paired["indeterminate_challenge_accept_sign"]
    collision_sign = paired["collision_route_sign"]
    coverage = paired["positive_coverage_gain_cluster_mean_difference_cells"]
    lines = [
        "# Exploratory reconstructed component-chain audit",
        "",
        "This is a single-scene exploratory audit reconstructed from saved probability maps. It does not rerun the historical trainer or historical online implementation, and it is not a complete system execution, an online robot experiment, or evidence of real-world exploration improvement.",
        "",
        "## Reconstructed audit chain",
        "",
        "`saved probability map / observed-only control -> reconstructed reference registration -> observed-support gate -> isolated atomic measured-map commit -> categorical occupancy state -> deterministic farthest-frontier grid proxy`",
        "",
        f"The raw trace contains {paired['events']} paired perturbations grouped by `base_frame_cluster`: {paired['known_positive_clusters']} known-positive clusters and {paired['indeterminate_challenge_clusters']} indeterminate abstention-challenge clusters. All summaries below first average repeated perturbations within a cluster; exact sign tests then use cluster means. No event-level significance test is reported.",
        "",
        "## Cluster-aware result",
        "",
        "| Arm | Correct-positive accept, mean cluster rate | Incorrect-positive accept, mean cluster rate | Positive abstention, mean cluster rate | Challenge accept, mean cluster rate | Collision route, mean positive-cluster rate | Unknown traversal, mean positive-cluster rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Predicted maps | {pct(predicted['correct_positive_accept_cluster_rate'])} | {pct(predicted['incorrect_positive_accept_cluster_rate'])} | {pct(predicted['positive_abstention_cluster_rate'])} | {pct(predicted['indeterminate_challenge_accept_cluster_rate'])} | {pct(predicted['oracle_collision_route_cluster_rate'])} | {pct(predicted['oracle_unknown_traversal_cluster_rate'])} |",
        f"| Observed only | {pct(observed['correct_positive_accept_cluster_rate'])} | {pct(observed['incorrect_positive_accept_cluster_rate'])} | {pct(observed['positive_abstention_cluster_rate'])} | {pct(observed['indeterminate_challenge_accept_cluster_rate'])} | {pct(observed['oracle_collision_route_cluster_rate'])} | {pct(observed['oracle_unknown_traversal_cluster_rate'])} |",
        "",
        f"Across the 30 known-positive clusters, the predicted-minus-observed correct-accept cluster mean was positive/negative/zero in {correct_sign['positive']}/{correct_sign['negative']}/{correct_sign['zero']} clusters (two-sided exact sign p={fmt(correct_sign['exact_sign_p'])}). The corresponding incorrect-positive accept signs were {incorrect_sign['positive']}/{incorrect_sign['negative']}/{incorrect_sign['zero']} (p={fmt(incorrect_sign['exact_sign_p'])}).",
        "",
        f"The median predicted-minus-observed difference between cluster-mean known-coverage gains was {fmt(coverage['median'], 1)} cells and the mean was {fmt(coverage['mean'], 1)} cells. Collision-route cluster-mean signs were {collision_sign['positive']}/{collision_sign['negative']}/{collision_sign['zero']} (p={fmt(collision_sign['exact_sign_p'])}).",
        "",
        f"The low-support temporal-mismatch cases are indeterminate abstention challenges, not verified negatives. Challenge-accept cluster-mean signs were {challenge_sign['positive']}/{challenge_sign['negative']}/{challenge_sign['zero']} (p={fmt(challenge_sign['exact_sign_p'])}); these accepts are not counted as errors or false accepts. This manifest contains {paired['known_error_injection_clusters']} explicit known-error-injection clusters, so a false-accept rate is not estimable here.",
        "",
        "The cluster-aware result does not demonstrate a safety advantage from prediction. The clusters are snapshots from one archived scene, not independent environments.",
        "",
        "## Pose errors on known-positive clusters with rigid candidates",
        "",
        "| Arm | Cluster-mean translation median / P90, m | Cluster-mean absolute yaw median / P90, deg |",
        "|---|---:|---:|",
        (
            f"| Predicted maps | {fmt(predicted['translation_error_m_cluster_mean']['median'])} / {fmt(predicted['translation_error_m_cluster_mean']['p90'])} | "
            f"{fmt(predicted['yaw_error_deg_cluster_mean']['median'])} / {fmt(predicted['yaw_error_deg_cluster_mean']['p90'])} |"
        ),
        (
            f"| Observed only | {fmt(observed['translation_error_m_cluster_mean']['median'])} / {fmt(observed['translation_error_m_cluster_mean']['p90'])} | "
            f"{fmt(observed['yaw_error_deg_cluster_mean']['median'])} / {fmt(observed['yaw_error_deg_cluster_mean']['p90'])} |"
        ),
        "",
        "## Boundaries",
        "",
        "- The predictor stage consists only of saved probability maps; it is not a recovered training-to-inference execution.",
        "- Each event has an isolated persistent state. The synthetic perturbations are not a coherent temporal trajectory, so three-frame consensus is not invented.",
        "- The planner is a deterministic four-connected farthest-frontier audit proxy with 0.10 m obstacle inflation, not the historical MRPB planner or navigation stack.",
        "- MRPB predicted and observed rasters are excluded because their dynamic canvases differ and the retained files lack the metric origins needed for a valid pairing.",
        "- Physical bags are excluded from this causal analysis because they contain no transform, gate, commit, or planner-event topics.",
        "",
        f"Input manifest SHA-256: `{manifest_hash}`.",
        "",
    ]
    return "\n".join(lines)


def run(manifest_path: Path, output: Path, limit: int | None = None) -> dict[str, Any]:
    manifest_path, manifest, manifest_events = load_exploratory_manifest(manifest_path)
    events = [
        event for event in manifest_events
        if event.get("input_type") == PREDICTED_INPUT_TYPE
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        events = events[:limit]
    if not events:
        raise ValueError("manifest contains no selected saved probability-map events")

    rows = paired_rows(events)
    summary = summarise(rows)
    differences = summary.pop("paired_differences")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "paired_events.csv", rows, EVENT_FIELDS)
    difference_fields = tuple(differences[0].keys())
    write_csv(output / "paired_differences.csv", differences, difference_fields)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_hash = sha256_file(manifest_path)
    (output / "REPORT.md").write_text(render_report(summary, manifest_hash), encoding="utf-8")

    run_manifest = {
        "claim_scope": summary["scope"],
        "created_by": "integrated_offline/run_paired_ablation.py",
        "input_manifest_name": manifest_path.name,
        "input_manifest_sha256": manifest_hash,
        "selected_input_type": PREDICTED_INPUT_TYPE,
        "paired_events": len(events),
        "base_frame_clusters": summary["cluster_aware_paired"]["clusters"],
        "output_rows": len(rows),
        "reference_registrar_sha256": sha256_file(Path(reference_registrar.__file__)),
        "pipeline_script_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "exploratory_rules": {
            "translation_correct_m": TAU_TRANSLATION_M,
            "absolute_yaw_correct_deg": TAU_YAW_DEG,
            "observed_support_gate": reference_registrar.MIN_OBSERVED_SUPPORT_OVERLAP,
            "minimum_inlier_ratio": reference_registrar.MIN_GATE_INLIER_RATIO,
            "unit_scale_tolerance": reference_registrar.UNIT_SCALE_TOLERANCE,
            "obstacle_inflation_cells": OBSTACLE_INFLATION_CELLS,
            "resolution_m_per_px": manifest.get("resolution_m_per_px"),
            "selection_lock_self_declaration_used_for_inference": False,
            "relative_asset_paths_supported": True,
        },
        "limitations": [
            "exploratory reconstruction from saved probability maps; no historical trainer execution",
            "single archived A3 scene with repeated snapshots",
            "isolated per-event commit; no temporal consensus claim",
            "deterministic planner proxy; not the historical planner",
            "no physical or online claim",
        ],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result_files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    sums = "".join(f"{sha256_file(path)}  {path.name}\n" for path in result_files)
    (output / "SHA256SUMS").write_text(sums, encoding="ascii")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, nargs="?", default=preferred_manifest_path(),
        help="portable A3 manifest (defaults to the Nature submission asset bundle)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("integrated_offline/results"),
        help="output directory",
    )
    parser.add_argument("--limit", type=int, help="debug-only prefix of selected events")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    summary = run(arguments.manifest, arguments.output, arguments.limit)
    print(json.dumps(summary["arms"], indent=2))
    print("EXPLORATORY OFFLINE ONLY: saved probability maps and reconstructed proxies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
