"""Append-only canonical event log with state-transition invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .hashing import canonical_json_sha256


class EventWriter:
    """Write a tamper-evident JSONL stream.

    Every record includes the previous record hash.  Non-committing commit
    events are rejected if their persistent-map revision or hash changes.
    """

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        arm: str,
        split: str,
        seed: int,
        building_id: str,
        floorplan_id: str,
        protocol_sha256: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("w", encoding="utf-8", newline="\n")
        self._sequence = 0
        self._previous_hash: str | None = None
        self._base = {
            "schema_version": 1,
            "run_id": run_id,
            "arm": arm,
            "split": split,
            "seed": int(seed),
            "building_id": building_id,
            "floorplan_id": floorplan_id,
            "protocol_sha256": protocol_sha256,
        }

    def emit(
        self,
        stage: str,
        tick: int,
        *,
        inputs: Mapping[str, str] | None = None,
        outputs: Mapping[str, str] | None = None,
        map_revision_before: int | None = None,
        map_revision_after: int | None = None,
        persistent_hash_before: str | None = None,
        persistent_hash_after: str | None = None,
        outcome: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        if not isinstance(stage, str) or not stage:
            raise ValueError("event stage must be a non-empty string")
        if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
            raise ValueError("event tick must be a nonnegative integer")
        parent_sequence = fields.get("parent_sequence")
        if parent_sequence is not None and (
            not isinstance(parent_sequence, int)
            or isinstance(parent_sequence, bool)
            or parent_sequence < 0
            or parent_sequence >= self._sequence
        ):
            raise ValueError("event parent must identify an earlier sequence")
        if stage == "commit" and outcome != "committed":
            if map_revision_before != map_revision_after:
                raise AssertionError("non-committing event changed map revision")
            if persistent_hash_before != persistent_hash_after:
                raise AssertionError("non-committing event changed persistent-map hash")
        event: dict[str, Any] = {
            **self._base,
            "sequence": self._sequence,
            "previous_event_sha256": self._previous_hash,
            "stage": stage,
            "tick": int(tick),
            "inputs": dict(sorted((inputs or {}).items())),
            "outputs": dict(sorted((outputs or {}).items())),
            "map_revision_before": map_revision_before,
            "map_revision_after": map_revision_after,
            "persistent_hash_before": persistent_hash_before,
            "persistent_hash_after": persistent_hash_after,
            "outcome": outcome,
            **fields,
        }
        event_hash = canonical_json_sha256(event)
        event["event_sha256"] = event_hash
        self._stream.write(
            json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
        self._stream.flush()
        self._previous_hash = event_hash
        self._sequence += 1
        return event

    @property
    def final_hash(self) -> str | None:
        return self._previous_hash

    @property
    def event_count(self) -> int:
        return self._sequence

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
