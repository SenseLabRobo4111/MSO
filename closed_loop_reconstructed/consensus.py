"""Ground-truth-free temporal and cycle-consistency checks for SE(2)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


def yaw_deg(transform: np.ndarray) -> float:
    matrix = np.asarray(transform, dtype=float)
    return math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))


def wrap_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def se2_delta(
    estimate: np.ndarray, reference: np.ndarray, resolution_m: float
) -> tuple[float, float]:
    delta = np.linalg.inv(np.asarray(reference, dtype=float)) @ np.asarray(
        estimate, dtype=float
    )
    translation_m = float(np.linalg.norm(delta[:2, 2])) * resolution_m
    absolute_yaw_deg = abs(wrap_deg(yaw_deg(delta)))
    return translation_m, absolute_yaw_deg


def median_se2(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("cannot average an empty transform list")
    angles = np.radians([yaw_deg(matrix) for matrix in transforms])
    reference = angles[0]
    unwrapped = reference + np.unwrap(angles - reference)
    angle = float(np.median(unwrapped))
    output = np.eye(3, dtype=float)
    output[0, 0] = math.cos(angle)
    output[0, 1] = -math.sin(angle)
    output[1, 0] = math.sin(angle)
    output[1, 1] = math.cos(angle)
    output[0, 2] = float(np.median([matrix[0, 2] for matrix in transforms]))
    output[1, 2] = float(np.median([matrix[1, 2] for matrix in transforms]))
    return output


@dataclass(frozen=True)
class ConsistencyDecision:
    ready: bool
    accepted: bool
    reason: str
    transform: np.ndarray | None
    temporal_count: int
    maximum_translation_delta_m: float
    maximum_yaw_delta_deg: float


@dataclass
class TemporalCycleGate:
    resolution_m: float
    consensus_count: int = 3
    temporal_translation_limit_m: float = 0.25
    temporal_yaw_limit_deg: float = 5.0
    cycle_translation_limit_m: float = 0.25
    cycle_yaw_limit_deg: float = 5.0
    _history: dict[tuple[int, int], list[np.ndarray]] = field(default_factory=dict)

    def reset(self, edge: tuple[int, int]) -> None:
        """Require a fresh temporal window after invalidating a transform."""
        self._history.pop(edge, None)

    def observe(
        self,
        edge: tuple[int, int],
        candidate: np.ndarray,
        existing_transform: np.ndarray | None = None,
    ) -> ConsistencyDecision:
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != (3, 3) or not np.isfinite(candidate).all():
            return ConsistencyDecision(
                False, False, "invalid_candidate", None, 0, 0.0, 0.0
            )
        history = self._history.setdefault(edge, [])
        history.append(candidate.copy())
        if len(history) > self.consensus_count:
            del history[:-self.consensus_count]
        consensus = median_se2(history)
        deltas = [se2_delta(value, consensus, self.resolution_m) for value in history]
        maximum_translation = max((value[0] for value in deltas), default=0.0)
        maximum_yaw = max((value[1] for value in deltas), default=0.0)
        consistent = (
            maximum_translation <= self.temporal_translation_limit_m
            and maximum_yaw <= self.temporal_yaw_limit_deg
        )
        if not consistent:
            self._history[edge] = [candidate.copy()]
            return ConsistencyDecision(
                False,
                False,
                "temporal_inconsistent_reset",
                None,
                1,
                maximum_translation,
                maximum_yaw,
            )
        ready = len(history) >= self.consensus_count
        if existing_transform is not None:
            translation, yaw = se2_delta(
                consensus, existing_transform, self.resolution_m)
            cycle_accepted = (
                translation <= self.cycle_translation_limit_m
                and yaw <= self.cycle_yaw_limit_deg)
            if cycle_accepted:
                return ConsistencyDecision(
                    ready, True, "cycle_consistent",
                    np.asarray(existing_transform, dtype=float), len(history),
                    translation, yaw)
            return ConsistencyDecision(
                ready,
                False,
                "cycle_inconsistent_consensus" if ready
                else "awaiting_cycle_recheck_consensus",
                consensus if ready else None,
                len(history),
                translation,
                yaw,
            )
        return ConsistencyDecision(
            ready,
            True,
            "temporal_consensus" if ready else "awaiting_temporal_consensus",
            consensus if ready else None,
            len(history),
            maximum_translation,
            maximum_yaw,
        )
