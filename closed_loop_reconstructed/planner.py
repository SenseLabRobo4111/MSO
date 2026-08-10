"""Measured-safe frontier planner with an optional temporary ranking layer."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np


UNKNOWN_VALUE = 100
UNKNOWN_TOLERANCE = 3
FREE_MAX_VALUE = 10
OCCUPIED_MIN_VALUE = 200


@dataclass(frozen=True)
class PlannerResult:
    route_available: bool
    route_cells: int
    route_length_m: float | None
    reachable_free_cells: int
    reachable_frontiers: int
    selected_goal_yx: tuple[int, int] | None
    selected_base_distance_cells: int | None
    selected_prediction_support: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def unknown_mask(grid: np.ndarray) -> np.ndarray:
    return np.abs(grid.astype(np.int16) - UNKNOWN_VALUE) <= UNKNOWN_TOLERANCE


def traversable_mask(grid: np.ndarray, inflation_cells: int) -> np.ndarray:
    occupied = (grid >= OCCUPIED_MIN_VALUE).astype(np.uint8)
    if inflation_cells > 0:
        occupied = cv2.dilate(
            occupied,
            np.ones((3, 3), dtype=np.uint8),
            iterations=inflation_cells,
        )
    return (grid <= FREE_MAX_VALUE) & (occupied == 0)


def nearest_traversable(
    traversable: np.ndarray, start_yx: tuple[int, int], radius: int = 20
) -> tuple[int, int] | None:
    sy, sx = start_yx
    if (
        0 <= sy < traversable.shape[0]
        and 0 <= sx < traversable.shape[1]
        and traversable[sy, sx]
    ):
        return sy, sx
    y0, y1 = max(0, sy - radius), min(traversable.shape[0], sy + radius + 1)
    x0, x1 = max(0, sx - radius), min(traversable.shape[1], sx + radius + 1)
    ys, xs = np.nonzero(traversable[y0:y1, x0:x1])
    if not len(ys):
        return None
    ys, xs = ys + y0, xs + x0
    order = np.lexsort((xs, ys, (ys - sy) ** 2 + (xs - sx) ** 2))
    index = int(order[0])
    return int(ys[index]), int(xs[index])


def plan_measured_safe(
    measured: np.ndarray,
    start_yx: tuple[int, int],
    *,
    resolution_m: float,
    inflation_cells: int,
    rank_probability: np.ndarray | None,
    rank_radius_cells: int,
    prediction_rank_weight_cells: float,
) -> tuple[PlannerResult, list[tuple[int, int]]]:
    """Plan only through measured free cells; prediction changes ranking only."""
    grid = np.asarray(measured, dtype=np.uint8)
    traversable = traversable_mask(grid, inflation_cells)
    start = nearest_traversable(traversable, start_yx)
    if start is None:
        return PlannerResult(False, 0, None, 0, 0, None, None, None), []
    sy, sx = start
    distance = np.full(grid.shape, -1, dtype=np.int32)
    parent_direction = np.full(grid.shape, -1, dtype=np.int8)
    distance[sy, sx] = 0
    queue: deque[tuple[int, int]] = deque([(sy, sx)])
    moves = ((-1, 0), (0, -1), (0, 1), (1, 0))
    while queue:
        y, x = queue.popleft()
        next_distance = int(distance[y, x]) + 1
        for direction, (dy, dx) in enumerate(moves):
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < grid.shape[0]
                and 0 <= nx < grid.shape[1]
                and traversable[ny, nx]
                and distance[ny, nx] < 0
            ):
                distance[ny, nx] = next_distance
                parent_direction[ny, nx] = direction ^ 3
                queue.append((ny, nx))

    adjacent_unknown = cv2.dilate(
        unknown_mask(grid).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    frontier = traversable & adjacent_unknown & (distance >= 0)
    frontier_yx = np.argwhere(frontier)
    reachable_count = int(np.count_nonzero(distance >= 0))
    if not len(frontier_yx):
        return PlannerResult(False, 0, None, reachable_count, 0, None, None, None), []

    if rank_probability is None:
        probability = np.full(grid.shape, 0.5, dtype=np.float32)
    else:
        probability = np.asarray(rank_probability, dtype=np.float32)
        if probability.shape != grid.shape:
            raise ValueError(
                "rank probability and measured grid must have equal shapes"
            )
    probability = np.clip(probability, 0.0, 1.0).copy()
    probability[grid <= FREE_MAX_VALUE] = 0.0
    probability[grid >= OCCUPIED_MIN_VALUE] = 1.0
    free_support = cv2.boxFilter(
        1.0 - probability,
        ddepth=-1,
        ksize=(2 * rank_radius_cells + 1, 2 * rank_radius_cells + 1),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    fy, fx = frontier_yx[:, 0], frontier_yx[:, 1]
    base_distance = distance[fy, fx].astype(np.float64)
    support = free_support[fy, fx].astype(np.float64)
    score = base_distance + prediction_rank_weight_cells * support
    order = np.lexsort((fx, fy, -base_distance, -score))
    selected = int(order[0])
    goal_y, goal_x = int(fy[selected]), int(fx[selected])

    route = [(goal_y, goal_x)]
    y, x = goal_y, goal_x
    while distance[y, x] > 0:
        direction = int(parent_direction[y, x])
        if direction < 0:
            raise AssertionError("planner parent chain is incomplete")
        dy, dx = moves[direction]
        y, x = y + dy, x + dx
        route.append((y, x))
    route.reverse()
    cells = int(distance[goal_y, goal_x])
    result = PlannerResult(
        True,
        cells,
        cells * resolution_m,
        reachable_count,
        len(frontier_yx),
        (goal_y, goal_x),
        cells,
        float(support[selected]),
    )
    return result, route
