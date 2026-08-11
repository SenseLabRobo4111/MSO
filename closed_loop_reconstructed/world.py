"""Headless floorplan world built from the replacement-study data loaders."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from training_recovery.generate_floorplan_dataset import (
    GenerationConfig,
    assign_splits,
    building_id_from_floorplan,
    load_occupied_map,
    ray_offsets,
    scan_from_pose,
    select_navigable_component,
)

from .hashing import sha256_file


@dataclass(frozen=True)
class FloorplanRecord:
    floorplan_id: str
    building_id: str
    split: str
    gt_path: Path
    source_relpath: str
    source_sha256: str
    source_resolution_m: float


def split_floorplans(
    source_root: Path,
    split_seed: int,
    readable_splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, list[FloorplanRecord]]:
    """Assign all floorplans while opening GT pixels only for allowed splits.

    Directory names and small resolution metadata are sufficient for the
    building-disjoint assignment.  The source bitmap digest is delayed until
    its split is explicitly allowed, which prevents validation preparation
    from reading held-out bitmap outcomes.
    """

    source_root = source_root.resolve()
    records: list[dict[str, object]] = []
    for gt_path in sorted(source_root.glob("*/GT.bmp")):
        floorplan_id = gt_path.parent.name
        metadata_path = gt_path.with_name("GT.json")
        resolution = 0.1
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            resolution = float(metadata.get("resolution", resolution))
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError(f"invalid resolution metadata for {floorplan_id}")
        records.append(
            {
                "floorplan_id": floorplan_id,
                "building_id": building_id_from_floorplan(floorplan_id),
                "gt_path": gt_path,
                "source_relpath": gt_path.relative_to(source_root).as_posix(),
                "resolution_m_per_cell": resolution,
            }
        )
    if not records:
        raise ValueError(f"no floorplan directories found below {source_root}")
    config = GenerationConfig(split_seed=split_seed)
    assignments = assign_splits(records, config)
    output = {"train": [], "val": [], "test": []}
    for record in records:
        split = assignments[str(record["floorplan_id"])]
        output[split].append(
            FloorplanRecord(
                floorplan_id=str(record["floorplan_id"]),
                building_id=str(record["building_id"]),
                split=split,
                gt_path=Path(record["gt_path"]),
                source_relpath=str(record["source_relpath"]),
                source_sha256=(
                    sha256_file(Path(record["gt_path"]))
                    if split in readable_splits
                    else "unread-until-split-unlock"
                ),
                source_resolution_m=float(record["resolution_m_per_cell"]),
            )
        )
    for rows in output.values():
        rows.sort(key=lambda item: item.floorplan_id)
    return output


@dataclass(frozen=True)
class MotionResult:
    moved: bool
    collision: bool
    previous_yx: tuple[int, int]
    requested_yx: tuple[int, int]
    final_yx: tuple[int, int]
    distance_m: float


class HeadlessFloorplanWorld:
    """A deterministic sensor and motion environment.

    Ground truth is held here.  Registration gates and planners receive only
    measured or provisional arrays constructed outside this class.
    """

    def __init__(
        self,
        record: FloorplanRecord,
        *,
        team_size: int,
        start_rng: np.random.Generator,
        working_resolution_m: float,
        laser_range_cells: int,
        laser_rays: int,
        encounter_range_m: float,
    ) -> None:
        config = GenerationConfig(
            working_resolution_m_per_cell=working_resolution_m,
            laser_range_cells=laser_range_cells,
            laser_rays=laser_rays,
        )
        self.record = record
        self.resolution_m = working_resolution_m
        self.occupied = load_occupied_map(
            record.gt_path, record.source_resolution_m, config
        )
        self.component, navigable_starts = select_navigable_component(self.occupied)
        self.rays = ray_offsets(laser_rays, laser_range_cells)
        self.poses = self._select_team_starts(
            navigable_starts,
            team_size,
            start_rng,
            max_pair_distance_cells=max(
                4, int(round(encounter_range_m / working_resolution_m * 0.70))
            ),
        )
        self.known = [
            np.zeros(self.occupied.shape, dtype=bool) for _ in range(team_size)
        ]
        self.distance_m = [0.0 for _ in range(team_size)]
        self.collision_count = [0 for _ in range(team_size)]
        self.scan_count = [0 for _ in range(team_size)]
        self._accessible_free = self.component & ~self.occupied

    @staticmethod
    def _select_team_starts(
        candidates: np.ndarray,
        team_size: int,
        rng: np.random.Generator,
        max_pair_distance_cells: int,
    ) -> list[np.ndarray]:
        if len(candidates) < team_size:
            raise ValueError("floorplan has fewer navigable starts than robots")
        center = np.mean(candidates, axis=0)
        distance_to_center = np.sum((candidates - center) ** 2, axis=1)
        central = np.argsort(distance_to_center)[: min(1024, len(candidates))]
        root = candidates[int(central[int(rng.integers(0, len(central)))])].astype(
            np.int32
        )
        radius_sq = float(max_pair_distance_cells**2)
        root_distance = np.sum((candidates - root) ** 2, axis=1)
        pool = candidates[(root_distance > 25) & (root_distance <= radius_sq)]
        if len(pool) < team_size - 1:
            pool = candidates[np.argsort(root_distance)[1 : max(team_size * 8, 32)]]
        selected = [root]
        available = pool.copy()
        while len(selected) < team_size:
            if not len(available):
                raise ValueError("could not place the full team")
            minimum_distance = np.min(
                np.stack(
                    [np.sum((available - value) ** 2, axis=1) for value in selected],
                    axis=0,
                ),
                axis=0,
            )
            best = np.flatnonzero(minimum_distance == minimum_distance.max())
            index = int(best[int(rng.integers(0, len(best)))])
            selected.append(available[index].astype(np.int32))
            available = np.delete(available, index, axis=0)
        return selected

    def scan(self, robot_id: int) -> np.ndarray:
        before = self.known[robot_id].copy()
        scan_from_pose(
            self.occupied,
            self.known[robot_id],
            self.poses[robot_id],
            self.rays,
        )
        self.scan_count[robot_id] += 1
        return self.known[robot_id] & ~before

    def move(self, robot_id: int, requested_yx: tuple[int, int]) -> MotionResult:
        previous = tuple(int(value) for value in self.poses[robot_id])
        requested = tuple(int(value) for value in requested_yx)
        y, x = requested
        valid = (
            0 <= y < self.occupied.shape[0]
            and 0 <= x < self.occupied.shape[1]
            and self.component[y, x]
            and not self.occupied[y, x]
        )
        if not valid:
            self.collision_count[robot_id] += 1
            return MotionResult(False, True, previous, requested, previous, 0.0)
        distance_cells = float(
            np.linalg.norm(np.asarray(requested) - np.asarray(previous))
        )
        self.poses[robot_id] = np.asarray(requested, dtype=np.int32)
        distance = distance_cells * self.resolution_m
        self.distance_m[robot_id] += distance
        return MotionResult(True, False, previous, requested, requested, distance)

    def encounter(self, robot_a: int, robot_b: int, range_m: float) -> bool:
        distance_cells = float(
            np.linalg.norm(self.poses[robot_a] - self.poses[robot_b])
        )
        return distance_cells * self.resolution_m <= range_m

    def measured_coverage(self) -> float:
        union = np.logical_or.reduce(self.known)
        denominator = int(np.count_nonzero(self._accessible_free))
        if denominator == 0:
            return 0.0
        return float(np.count_nonzero(union & self._accessible_free) / denominator)

    def cumulative_team_distance_m(self) -> float:
        return float(sum(self.distance_m))

    def public_metadata(self) -> dict[str, Any]:
        return {
            "shape": list(self.occupied.shape),
            "resolution_m_per_cell": self.resolution_m,
            "accessible_free_cells": int(np.count_nonzero(self._accessible_free)),
            "team_start_yx": [pose.tolist() for pose in self.poses],
        }
