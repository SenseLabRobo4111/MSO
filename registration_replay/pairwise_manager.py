"""Robot-ID-agnostic pairwise registration queue with an explicit commit barrier."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from itertools import combinations
from typing import Callable, Deque, Dict, Iterable, Optional, Tuple

import numpy as np

from .se2_gate import CandidateDiagnostics, GateDecision, GateThresholds, gate_se2_candidate


@dataclass(frozen=True)
class MapObservation:
    robot_id: str
    stamp_ns: int
    image: np.ndarray
    resolution_m: float
    origin_xy_m: Tuple[float, float]
    revision: int = 0


@dataclass(frozen=True)
class Encounter:
    first: MapObservation
    second: MapObservation


@dataclass(frozen=True)
class MatchCandidate:
    affine_first_to_second: Optional[np.ndarray]
    diagnostics: CandidateDiagnostics


@dataclass(frozen=True)
class CommitRecord:
    encounter: Encounter
    decision: GateDecision
    consensus_support: int
    consensus_reason: str
    committed: bool


@dataclass(frozen=True)
class ConsensusThresholds:
    confirmations: int = 3
    min_interval_s: float = 0.5
    max_gap_s: float = 5.0
    translation_tolerance_m: float = 0.5
    yaw_tolerance_deg: float = 5.0


@dataclass
class _ConsensusState:
    first_revision: int
    second_revision: int
    stamp_ns: int
    translation_xy_m: np.ndarray
    yaw_rad: float
    support: int


Matcher = Callable[[MapObservation, MapObservation], MatchCandidate]
Committer = Callable[[CommitRecord], None]


class PairwiseRegistrationManager:
    """FIFO encounter processor supporting any configured robot-ID set.

    `update` creates unordered encounters between the arriving observation and
    each other robot with a cached observation.  `enqueue` is available when a
    communication or proximity layer supplies explicit encounters.  A commit
    callback is invoked only after the local gate and pair-isolated temporal
    consensus both accept the candidate.  Commit remains disabled by default.
    """

    def __init__(
        self,
        robot_ids: Iterable[str],
        matcher: Matcher,
        committer: Committer,
        thresholds: GateThresholds = GateThresholds(),
        consensus_thresholds: ConsensusThresholds = ConsensusThresholds(),
        *,
        commit_enabled: bool = False,
    ) -> None:
        ids = tuple(dict.fromkeys(str(x) for x in robot_ids))
        if len(ids) < 2:
            raise ValueError("at least two unique robot IDs are required")
        self.robot_ids = ids
        self._allowed = set(ids)
        self._matcher = matcher
        self._committer = committer
        self._thresholds = thresholds
        self._consensus_thresholds = consensus_thresholds
        self._commit_enabled = bool(commit_enabled)
        self._latest: Dict[str, MapObservation] = {}
        self._queue: Deque[Encounter] = deque()
        self._consensus: Dict[Tuple[str, str], _ConsensusState] = {}

    @property
    def pending(self) -> int:
        return len(self._queue)

    def configured_pairs(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(combinations(self.robot_ids, 2))

    def _ordered_encounter(
        self, first: MapObservation, second: MapObservation
    ) -> Encounter:
        if self.robot_ids.index(first.robot_id) <= self.robot_ids.index(second.robot_id):
            return Encounter(first, second)
        return Encounter(second, first)

    def update(self, observation: MapObservation) -> None:
        if observation.robot_id not in self._allowed:
            raise KeyError(f"unconfigured robot ID {observation.robot_id!r}")
        for other_id in self.robot_ids:
            if other_id == observation.robot_id or other_id not in self._latest:
                continue
            other = self._latest[other_id]
            self._queue.append(self._ordered_encounter(observation, other))
        self._latest[observation.robot_id] = observation

    def enqueue(self, first: MapObservation, second: MapObservation) -> None:
        if first.robot_id == second.robot_id:
            raise ValueError("an encounter requires two different robots")
        if first.robot_id not in self._allowed or second.robot_id not in self._allowed:
            raise KeyError("encounter contains an unconfigured robot ID")
        self._queue.append(self._ordered_encounter(first, second))

    @staticmethod
    def _wrap_radians(value: float) -> float:
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _metric_pose_first_to_second(
        se2_image: np.ndarray,
        first: MapObservation,
        second: MapObservation,
    ) -> Tuple[np.ndarray, float]:
        if first.image.shape != second.image.shape:
            raise ValueError("consensus requires equal grid dimensions")
        if not np.isclose(first.resolution_m, second.resolution_m, rtol=0.0, atol=1e-9):
            raise ValueError("consensus requires equal grid resolution")
        height = int(first.image.shape[0])
        flip = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]]
        )
        grid_second_from_first = flip @ se2_image @ flip
        rotation = grid_second_from_first[:2, :2]
        origin_first = np.asarray(first.origin_xy_m, dtype=float)
        origin_second = np.asarray(second.origin_xy_m, dtype=float)
        translation = (
            origin_second
            + second.resolution_m * grid_second_from_first[:2, 2]
            - rotation @ origin_first
        )
        yaw = float(math.atan2(rotation[1, 0], rotation[0, 0]))
        return translation, yaw

    def _update_consensus(
        self,
        encounter: Encounter,
        decision: GateDecision,
    ) -> Tuple[int, str]:
        key = (encounter.first.robot_id, encounter.second.robot_id)
        if not decision.accepted or decision.se2_image is None:
            self._consensus.pop(key, None)
            return 0, "local_gate_reject"
        try:
            translation, yaw = self._metric_pose_first_to_second(
                decision.se2_image, encounter.first, encounter.second
            )
        except ValueError as exc:
            self._consensus.pop(key, None)
            return 0, str(exc)

        stamp_ns = max(encounter.first.stamp_ns, encounter.second.stamp_ns)
        state = self._consensus.get(key)
        thresholds = self._consensus_thresholds
        if state is None:
            self._consensus[key] = _ConsensusState(
                encounter.first.revision,
                encounter.second.revision,
                stamp_ns,
                translation,
                yaw,
                1,
            )
            return 1, "temporal_confirmation"

        both_revised = (
            encounter.first.revision != state.first_revision
            and encounter.second.revision != state.second_revision
        )
        elapsed_s = (stamp_ns - state.stamp_ns) / 1e9
        if not both_revised or elapsed_s < thresholds.min_interval_s:
            return state.support, "nonindependent_confirmation"

        consistent = (
            elapsed_s <= thresholds.max_gap_s
            and float(np.linalg.norm(translation - state.translation_xy_m))
            <= thresholds.translation_tolerance_m
            and abs(math.degrees(self._wrap_radians(yaw - state.yaw_rad)))
            <= thresholds.yaw_tolerance_deg
        )
        support = state.support + 1 if consistent else 1
        self._consensus[key] = _ConsensusState(
            encounter.first.revision,
            encounter.second.revision,
            stamp_ns,
            translation,
            yaw,
            support,
        )
        reason = "confirmed" if support >= thresholds.confirmations else "temporal_confirmation"
        return support, reason

    def process_one(self) -> Optional[CommitRecord]:
        if not self._queue:
            return None
        encounter = self._queue.popleft()
        candidate = self._matcher(encounter.first, encounter.second)
        decision = gate_se2_candidate(
            candidate.affine_first_to_second,
            candidate.diagnostics,
            self._thresholds,
        )
        support, consensus_reason = self._update_consensus(encounter, decision)
        committed = bool(
            decision.accepted
            and support >= self._consensus_thresholds.confirmations
            and self._commit_enabled
        )
        record = CommitRecord(
            encounter=encounter,
            decision=decision,
            consensus_support=support,
            consensus_reason=consensus_reason,
            committed=committed,
        )
        if committed:
            self._committer(record)
        return record

    def drain(self) -> Tuple[CommitRecord, ...]:
        records = []
        while self._queue:
            record = self.process_one()
            if record is not None:
                records.append(record)
        return tuple(records)
