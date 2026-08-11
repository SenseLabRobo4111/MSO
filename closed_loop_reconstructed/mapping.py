"""Private frames, semantic warps, and measured-only persistent commits."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from integrated_offline.run_paired_ablation import (
    atomic_measured_commit,
    state_hash,
)


UNKNOWN_VALUE = 100
FREE_VALUE = 0
OCCUPIED_VALUE = 255


def transform_point_xy(
    transform: np.ndarray, point_xy: tuple[float, float]
) -> tuple[float, float]:
    value = np.asarray(transform, dtype=float) @ np.array(
        [point_xy[0], point_xy[1], 1.0], dtype=float
    )
    return float(value[0] / value[2]), float(value[1] / value[2])


def private_frame_matrix(
    world_shape: tuple[int, int], quadrant: int, margin: int
) -> np.ndarray:
    height, width = world_shape
    quadrant %= 4
    if quadrant == 0:
        return np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1]], float)
    if quadrant == 1:
        return np.array(
            [[0, -1, height - 1 + margin], [1, 0, margin], [0, 0, 1]],
            float,
        )
    if quadrant == 2:
        return np.array(
            [
                [-1, 0, width - 1 + margin],
                [0, -1, height - 1 + margin],
                [0, 0, 1],
            ],
            float,
        )
    return np.array(
        [[0, 1, margin], [-1, 0, width - 1 + margin], [0, 0, 1]],
        float,
    )


def make_private_frames(
    world_shape: tuple[int, int],
    team_size: int,
    rng: np.random.Generator,
    margin: int = 16,
) -> tuple[list[np.ndarray], int, list[int]]:
    if team_size < 1:
        raise ValueError("team_size must be positive")
    quadrants = [0]
    while len(quadrants) < team_size:
        choices = np.array([1, 2, 3, 0], dtype=int)
        rng.shuffle(choices)
        quadrants.extend(int(value) for value in choices)
    quadrants = quadrants[:team_size]
    tile_size = max(world_shape) + 2 * margin
    grid_side = int(math.ceil(math.sqrt(team_size)))
    size = tile_size if team_size <= 4 else grid_side * tile_size
    frames: list[np.ndarray] = []
    for index, quadrant in enumerate(quadrants):
        transform = private_frame_matrix(world_shape, quadrant, margin)
        if team_size > 4:
            tile_x = (index % grid_side) * tile_size
            tile_y = (index // grid_side) * tile_size
            transform = np.array(
                [[1, 0, tile_x], [0, 1, tile_y], [0, 0, 1]], dtype=float
            ) @ transform
        frames.append(transform)
    return (
        frames,
        size,
        quadrants,
    )


def blank_grid(size: int) -> np.ndarray:
    return np.full((size, size), UNKNOWN_VALUE, dtype=np.uint8)


def update_local_measurement(
    local_grid: np.ndarray,
    transform_local_from_world: np.ndarray,
    world_occupied: np.ndarray,
    newly_known: np.ndarray,
) -> int:
    ys, xs = np.nonzero(newly_known)
    if not len(xs):
        return 0
    points = np.stack((xs, ys, np.ones_like(xs)), axis=0).astype(float)
    mapped = transform_local_from_world @ points
    local_x = np.rint(mapped[0] / mapped[2]).astype(int)
    local_y = np.rint(mapped[1] / mapped[2]).astype(int)
    valid = (
        (local_x >= 0)
        & (local_x < local_grid.shape[1])
        & (local_y >= 0)
        & (local_y < local_grid.shape[0])
    )
    values = np.where(
        world_occupied[ys[valid], xs[valid]], OCCUPIED_VALUE, FREE_VALUE
    ).astype(np.uint8)
    before = local_grid[local_y[valid], local_x[valid]].copy()
    local_grid[local_y[valid], local_x[valid]] = values
    return int(np.count_nonzero(before != values))


def warp_semantic(
    grid: np.ndarray,
    transform_target_from_source: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    return cv2.warpPerspective(
        np.asarray(grid, dtype=np.uint8),
        np.asarray(transform_target_from_source, dtype=float),
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=UNKNOWN_VALUE,
    )


def warp_probability(
    probability: np.ndarray,
    valid: np.ndarray,
    transform_target_from_source: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    warped_probability = cv2.warpPerspective(
        np.asarray(probability, dtype=np.float32),
        np.asarray(transform_target_from_source, dtype=float),
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.5,
    )
    warped_valid = cv2.warpPerspective(
        np.asarray(valid, dtype=np.uint8),
        np.asarray(transform_target_from_source, dtype=float),
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return warped_probability, warped_valid


def crop_with_origin(
    grid: np.ndarray,
    center_xy: tuple[float, float],
    size: int,
    fill_value: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    x0 = int(round(center_xy[0])) - size // 2
    y0 = int(round(center_xy[1])) - size // 2
    output = np.full((size, size), fill_value, dtype=grid.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(grid.shape[1], x0 + size), min(grid.shape[0], y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x0, sy0 - y0
        output[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = grid[
            sy0:sy1, sx0:sx1
        ]
    return output, (x0, y0)


def paste_probability(
    probability: np.ndarray,
    valid: np.ndarray,
    crop_probability: np.ndarray,
    origin_xy: tuple[int, int],
) -> None:
    x0, y0 = origin_xy
    height, width = crop_probability.shape
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1 = min(width, probability.shape[1] - x0)
    sy1 = min(height, probability.shape[0] - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    tx0, ty0 = x0 + sx0, y0 + sy0
    probability[ty0 : ty0 + sy1 - sy0, tx0 : tx0 + sx1 - sx0] = crop_probability[
        sy0:sy1, sx0:sx1
    ]
    valid[ty0 : ty0 + sy1 - sy0, tx0 : tx0 + sx1 - sx0] = True


def full_transform_from_crop_estimate(
    crop_target_from_crop_source: np.ndarray,
    target_origin_xy: tuple[int, int],
    source_origin_xy: tuple[int, int],
) -> np.ndarray:
    target = np.array(
        [[1, 0, target_origin_xy[0]], [0, 1, target_origin_xy[1]], [0, 0, 1]],
        float,
    )
    source = np.array(
        [[1, 0, -source_origin_xy[0]], [0, 1, -source_origin_xy[1]], [0, 0, 1]],
        float,
    )
    return target @ np.asarray(crop_target_from_crop_source, dtype=float) @ source


@dataclass
class PersistentMeasuredMap:
    grid: np.ndarray
    revision: int = 0

    @property
    def sha256(self) -> str:
        return state_hash(self.grid)

    def apply(self, warped_measured: np.ndarray, accepted: bool) -> dict[str, object]:
        before_hash = self.sha256
        before_revision = self.revision
        after, committed, changed = atomic_measured_commit(
            self.grid, warped_measured, accepted
        )
        if committed and changed:
            self.grid = after
            self.revision += 1
        elif committed:
            self.grid = after
        after_hash = self.sha256
        if not accepted and (
            before_hash != after_hash
            or before_revision != self.revision
            or changed != 0
        ):
            raise AssertionError("rejected commit changed persistent state")
        return {
            "committed": bool(committed),
            "changed_cells": int(changed),
            "revision_before": before_revision,
            "revision_after": self.revision,
            "hash_before": before_hash,
            "hash_after": after_hash,
        }

    def replace_with_rebuild(self, rebuilt: np.ndarray) -> dict[str, object]:
        """Atomically replace state after invalidating a committed transform."""
        candidate = np.asarray(rebuilt, dtype=np.uint8)
        if candidate.shape != self.grid.shape:
            raise ValueError("rebuilt persistent map has a different shape")
        if not set(np.unique(candidate)).issubset(
                {FREE_VALUE, UNKNOWN_VALUE, OCCUPIED_VALUE}):
            raise ValueError("rebuilt persistent map is not semantic occupancy")
        before_hash = self.sha256
        before_revision = self.revision
        changed = int(np.count_nonzero(candidate != self.grid))
        self.grid = candidate.copy()
        if changed:
            self.revision += 1
        return {
            "changed_cells": changed,
            "revision_before": before_revision,
            "revision_after": self.revision,
            "hash_before": before_hash,
            "hash_after": self.sha256,
        }
