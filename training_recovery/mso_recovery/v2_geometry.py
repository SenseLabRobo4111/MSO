"""Closed geometry implementation for the prospective-v2 generator.

This module intentionally has no command-line entrypoint.  Every value is
supplied by the immutable lock through ``SmokeConfig``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SmokeConfig:
    pixels_per_meter: int
    working_resolution_m_per_cell: float
    physical_window_m: int
    raw_size: int
    model_size: int
    sensor_range_m: int
    laser_range_cells: int
    laser_rays: int
    actions_before_capture: int
    forward_step_m: float
    forward_step_cells: int
    turn_degrees: int
    source_obstacle_dilation_iterations: int
    observation_known_dilation_iterations: int
    content_padding_source_cells: int
    samples_per_floorplan: int
    maximum_attempts_per_floorplan: int
    split_seed: int
    generation_seed: int
    minimum_observed_obstacle_fraction: float
    minimum_observed_free_fraction: float
    maximum_free_to_obstacle_ratio: float


def load_source_occupied(record: dict[str, object], config: SmokeConfig) -> np.ndarray:
    gray = np.asarray(Image.open(Path(record["gt_path"])).convert("L"), dtype=np.uint8)
    occupied = gray < 128
    content = occupied if float(occupied.mean()) < 0.5 else ~occupied
    ys, xs = np.where(content)
    if not len(ys):
        raise ValueError(f"No content in {record['floorplan_id']}")
    pad = config.content_padding_source_cells
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(gray.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(gray.shape[1], int(xs.max()) + pad + 1)
    occupied = occupied[y0:y1, x0:x1]
    source_resolution = float(record["resolution_m_per_cell"])
    scale = source_resolution / config.working_resolution_m_per_cell
    occupied = cv2.resize(
        occupied.astype(np.uint8),
        (
            max(8, int(round(occupied.shape[1] * scale))),
            max(8, int(round(occupied.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    occupied = cv2.dilate(
        occupied.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=config.source_obstacle_dilation_iterations,
    ).astype(bool)
    outer_pad = config.raw_size + config.laser_range_cells + 8
    return np.pad(occupied, outer_pad, mode="constant", constant_values=True)


def select_navigable_component(occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    free = (~occupied).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(free, 8)
    if count <= 1:
        raise ValueError("Floorplan has no free-space component")
    height, width = occupied.shape
    center = np.array([width / 2.0, height / 2.0])
    candidates: list[tuple[float, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 64:
            continue
        distance = float(np.linalg.norm(centroids[label] - center))
        candidates.append((area / (1.0 + 0.02 * distance), label))
    if not candidates:
        raise ValueError("No navigable free-space component with at least 64 cells")
    selected = max(candidates)[1]
    component = labels == selected
    clearance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    starts = np.argwhere(clearance >= 1.5)
    if not len(starts):
        starts = np.argwhere(component)
    return component, starts


def restricted_starts(component: np.ndarray) -> np.ndarray:
    clearance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.where(component)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bbox_mask = np.zeros(component.shape, dtype=bool)
    bbox_mask[
        y0 + int(0.10 * (y1 - y0)) : y0 + int(0.80 * (y1 - y0)),
        x0 + int(0.10 * (x1 - x0)) : x0 + int(0.80 * (x1 - x0)),
    ] = True
    starts = np.argwhere((clearance >= 2.0) & bbox_mask)
    if not len(starts):
        starts = np.argwhere(component & bbox_mask)
    if not len(starts):
        raise ValueError("No navigable start in the declared 10--80% box")
    return starts


def ray_offsets(ray_count: int, radius: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rays: list[tuple[np.ndarray, np.ndarray]] = []
    radii = np.arange(1, radius + 1, dtype=np.float64)
    for angle in np.linspace(0.0, 2.0 * np.pi, ray_count, endpoint=False):
        dy = np.rint(np.sin(angle) * radii).astype(np.int32)
        dx = np.rint(np.cos(angle) * radii).astype(np.int32)
        keep = np.ones(len(dy), dtype=bool)
        keep[1:] = (dy[1:] != dy[:-1]) | (dx[1:] != dx[:-1])
        rays.append((dy[keep], dx[keep]))
    return rays


def scan_from_pose(
    occupied: np.ndarray,
    known: np.ndarray,
    pose: np.ndarray,
    rays: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    y, x = int(pose[0]), int(pose[1])
    known[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = True
    height, width = occupied.shape
    for dy, dx in rays:
        yy, xx = y + dy, x + dx
        valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        yy, xx = yy[valid], xx[valid]
        if not len(yy):
            continue
        hits = np.flatnonzero(occupied[yy, xx])
        end = int(hits[0]) + 1 if len(hits) else len(yy)
        known[yy[:end], xx[:end]] = True


def action_for_step(step: int, rng: np.random.Generator) -> int:
    if step > 5 and step not in {7, 9, 11, 13}:
        return int(rng.integers(1, 3))
    return 0


def attempt_forward(
    pose: np.ndarray, heading: float, component: np.ndarray, cells: int
) -> np.ndarray:
    current = pose.copy()
    for distance in range(1, cells + 1):
        candidate = np.asarray(
            [
                int(round(pose[0] - math.sin(heading) * distance)),
                int(round(pose[1] + math.cos(heading) * distance)),
            ],
            dtype=np.int32,
        )
        if not component[int(candidate[0]), int(candidate[1])]:
            break
        current = candidate
    return current


def oriented_crop(
    array: np.ndarray,
    pose: np.ndarray,
    heading: float,
    size: int,
    border_value: int | float,
) -> np.ndarray:
    patch_size = int(math.ceil(size * math.sqrt(2.0))) + 4
    if patch_size % 2:
        patch_size += 1
    half = patch_size // 2
    y, x = int(pose[0]), int(pose[1])
    patch = array[y - half : y + half, x - half : x + half]
    if patch.shape != (patch_size, patch_size):
        raise ValueError(f"Unexpected oriented crop patch: {patch.shape}")
    angle = -heading * 180.0 / math.pi + 90.0
    matrix = cv2.getRotationMatrix2D((half, half), angle, 1.0)
    rotated = cv2.warpAffine(
        patch,
        matrix,
        (patch_size, patch_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    start = (patch_size - size) // 2
    return rotated[start : start + size, start : start + size]


def generate_sample(
    occupied: np.ndarray,
    component: np.ndarray,
    starts: np.ndarray,
    rays: list[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    config: SmokeConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
    pose = starts[int(rng.integers(0, len(starts)))].astype(np.int32)
    heading = float(rng.random() * 2.0 * math.pi)
    known = np.zeros(occupied.shape, dtype=bool)
    scan_from_pose(occupied, known, pose, rays)
    for step in range(1, config.actions_before_capture + 1):
        action = action_for_step(step, rng)
        if action == 0:
            pose = attempt_forward(pose, heading, component, config.forward_step_cells)
        elif action == 1:
            heading += math.radians(config.turn_degrees)
        else:
            heading -= math.radians(config.turn_degrees)
        scan_from_pose(occupied, known, pose, rays)
    known = cv2.dilate(
        known.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=config.observation_known_dilation_iterations,
    ).astype(bool)
    target = oriented_crop(
        occupied.astype(np.uint8), pose, heading, config.raw_size, 1
    ).astype(bool)
    observed = oriented_crop(
        known.astype(np.uint8), pose, heading, config.raw_size, 0
    ).astype(bool)
    observation = np.zeros((config.raw_size, config.raw_size, 3), dtype=np.uint8)
    observation[:, :, 0] = (observed & target).astype(np.uint8) * 255
    observation[:, :, 1] = (~observed).astype(np.uint8) * 255
    observation[:, :, 2] = (observed & ~target).astype(np.uint8) * 255
    obstacle_fraction = float(np.mean(observation[:, :, 0] == 255))
    unknown_fraction = float(np.mean(observation[:, :, 1] == 255))
    free_fraction = float(np.mean(observation[:, :, 2] == 255))
    if obstacle_fraction < config.minimum_observed_obstacle_fraction:
        return None
    if free_fraction < config.minimum_observed_free_fraction:
        return None
    if free_fraction > obstacle_fraction * config.maximum_free_to_obstacle_ratio:
        return None
    return observation, target.astype(np.uint8) * 255, {
        "obstacle_fraction": obstacle_fraction,
        "unknown_fraction": unknown_fraction,
        "free_fraction": free_fraction,
        "target_fraction": float(target.mean()),
    }


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
        "global_mean": float(array.mean()),
    }
