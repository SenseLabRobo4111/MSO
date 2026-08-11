"""Verify event-chain, transaction, coverage, and summary invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .hashing import canonical_json_sha256


BASE_FIELDS = (
    "schema_version",
    "run_id",
    "arm",
    "split",
    "seed",
    "building_id",
    "floorplan_id",
    "protocol_sha256",
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected an object in {path}")
    return value


def _phase(event: dict[str, Any]) -> int:
    stage = event["stage"]
    if stage == "world_init":
        return 0
    if stage == "scan":
        return 10
    if stage in {"predictor", "observed_control"}:
        return 20
    if stage in {
        "encounter",
        "registration",
        "gate",
        "temporal_cycle_consistency",
        "recovery",
        "post_decision_evaluation",
    }:
        return 30
    if stage == "commit":
        return 30 if event.get("event_id") else 40
    if stage == "planner":
        return 50
    if stage == "metric":
        return 60
    if stage == "motion":
        return 70
    if stage == "run_complete":
        return 80
    raise AssertionError(f"unknown event stage: {stage!r}")


def _robot_ids(
    events: list[dict[str, Any]], stages: set[str], tick: int
) -> list[int]:
    return [
        int(event["robot_id"])
        for event in events
        if event["stage"] in stages and int(event["tick"]) == tick
    ]


def _verify_parent(
    event: dict[str, Any], rows: list[dict[str, Any]], allowed: set[str]
) -> None:
    parent_sequence = event.get("parent_sequence")
    if not isinstance(parent_sequence, int) or isinstance(parent_sequence, bool):
        raise AssertionError(
            f"{event['stage']} event {event['sequence']} lacks an integer parent"
        )
    if parent_sequence < 0 or parent_sequence >= int(event["sequence"]):
        raise AssertionError("event parent is not an earlier sequence")
    parent = rows[parent_sequence]
    if parent["stage"] not in allowed:
        raise AssertionError(
            f"{event['stage']} parent stage {parent['stage']} is invalid"
        )
    for field in ("tick", "event_id", "source_robot", "target_robot"):
        if event.get(field) != parent.get(field):
            raise AssertionError(
                f"{event['stage']} and parent disagree on {field}"
            )


def _verify_transactions(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in rows:
        event_id = event.get("event_id")
        stage = event["stage"]
        if stage == "encounter" and event.get("outcome") == "triggered" and (
            event_id is None
        ):
            raise AssertionError("triggered encounter lacks an event_id")
        if stage in {
            "registration",
            "gate",
            "temporal_cycle_consistency",
            "recovery",
            "post_decision_evaluation",
        } and event_id is None:
            raise AssertionError(f"{stage} lacks a registration event_id")
        if stage == "commit" and event.get("parent_sequence") is not None and (
            event_id is None
        ):
            raise AssertionError("transaction commit lacks an event_id")
        if event_id is not None:
            if not isinstance(event_id, str) or not event_id:
                raise AssertionError("event_id must be a non-empty string")
            grouped.setdefault(event_id, []).append(event)
    for event_id, events in grouped.items():
        stages = [event["stage"] for event in events]
        expected_prefix = ["encounter", "registration", "gate"]
        if stages[:3] != expected_prefix:
            raise AssertionError(
                f"registration transaction {event_id} has invalid prefix {stages}"
            )
        tail = stages[3:]
        if tail not in (
            ["commit", "post_decision_evaluation"],
            ["temporal_cycle_consistency", "commit", "post_decision_evaluation"],
            ["temporal_cycle_consistency", "recovery", "post_decision_evaluation"],
        ):
            raise AssertionError(
                f"registration transaction {event_id} has invalid tail {tail}"
            )
        first = events[0]
        if first.get("outcome") != "triggered":
            raise AssertionError("only triggered encounters may start a transaction")
        for event in events:
            for field in ("tick", "source_robot", "target_robot"):
                if event.get(field) != first.get(field):
                    raise AssertionError(
                        f"registration transaction {event_id} changes {field}"
                    )
        terminal = events[-2]
        evaluation = events[-1]
        terminal_hash = terminal.get("persistent_hash_after")
        terminal_revision = terminal.get("map_revision_after")
        quality = evaluation.get("persistent_map_quality")
        if not isinstance(terminal_hash, str) or not terminal_hash:
            raise AssertionError("transaction terminal state hash is absent")
        if not isinstance(terminal_revision, int) or isinstance(
            terminal_revision, bool
        ):
            raise AssertionError("transaction terminal map revision is absent")
        if evaluation.get("inputs", {}).get("persistent_map") != terminal_hash:
            raise AssertionError("post-decision evaluator read a different map state")
        if not isinstance(quality, dict) or quality.get(
            "persistent_map_sha256"
        ) != terminal_hash or quality.get("revision") != terminal_revision:
            raise AssertionError("post-decision map-state evaluation is inconsistent")


def verify_event_log(path: Path) -> dict[str, Any]:
    previous: str | None = None
    stages: dict[str, int] = {}
    reject_count = 0
    rows: list[dict[str, Any]] = []
    base: dict[str, Any] | None = None
    last_tick = -1
    tick_phase = -1
    persistent_hash: str | None = None
    persistent_revision: int | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            event_with_digest = json.loads(line)
            if not isinstance(event_with_digest, dict):
                raise AssertionError(f"non-object event at line {line_number}")
            event = dict(event_with_digest)
            digest = event.pop("event_sha256", None)
            if event.get("sequence") != len(rows):
                raise AssertionError(f"sequence discontinuity at line {line_number}")
            if event.get("previous_event_sha256") != previous:
                raise AssertionError(f"hash-chain discontinuity at line {line_number}")
            if canonical_json_sha256(event) != digest:
                raise AssertionError(f"event digest mismatch at line {line_number}")
            identity = {field: event.get(field) for field in BASE_FIELDS}
            if base is None:
                base = identity
            elif identity != base:
                raise AssertionError(f"run identity changed at line {line_number}")
            tick = event.get("tick")
            if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
                raise AssertionError(f"invalid tick at line {line_number}")
            phase = _phase(event)
            if tick < last_tick:
                raise AssertionError(f"tick order regressed at line {line_number}")
            if tick != last_tick:
                last_tick = tick
                tick_phase = -1
            if phase < tick_phase:
                raise AssertionError(f"stage order regressed at line {line_number}")
            tick_phase = phase
            if not isinstance(event.get("inputs"), dict) or not isinstance(
                event.get("outputs"), dict
            ):
                raise AssertionError(
                    f"missing input/output digest maps at line {line_number}"
                )
            stage = str(event["stage"])
            if stage == "world_init":
                initial = event.get("initial_state_manifest")
                if not isinstance(initial, dict):
                    raise AssertionError("world_init lacks the initial state manifest")
                persistent_hash = initial.get("persistent_map_sha256")
                persistent_revision = 0
                if not isinstance(persistent_hash, str) or not persistent_hash:
                    raise AssertionError("world_init lacks the initial persistent map")
            map_fields = (
                "map_revision_before",
                "map_revision_after",
                "persistent_hash_before",
                "persistent_hash_after",
            )
            present = [event.get(field) is not None for field in map_fields]
            if any(present) and not all(present):
                raise AssertionError("persistent-map transition fields are incomplete")
            if all(present):
                before_revision = event["map_revision_before"]
                after_revision = event["map_revision_after"]
                before_hash = event["persistent_hash_before"]
                after_hash = event["persistent_hash_after"]
                if any(
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in (before_revision, after_revision)
                ) or any(
                    not isinstance(value, str) or not value
                    for value in (before_hash, after_hash)
                ):
                    raise AssertionError("persistent-map transition fields are malformed")
                if before_revision != persistent_revision or before_hash != persistent_hash:
                    raise AssertionError(
                        "persistent-map transition is detached from the global chain"
                    )
                expected_revision = before_revision + int(after_hash != before_hash)
                if after_revision != expected_revision:
                    raise AssertionError(
                        "persistent-map revision does not match the state transition"
                    )
                persistent_revision = after_revision
                persistent_hash = after_hash
            stages[stage] = stages.get(stage, 0) + 1
            if stage == "commit" and event.get("outcome") != "committed":
                reject_count += 1
                if event.get("map_revision_before") != event.get(
                    "map_revision_after"
                ):
                    raise AssertionError(
                        f"rejected revision changed at line {line_number}"
                    )
                if event.get("persistent_hash_before") != event.get(
                    "persistent_hash_after"
                ):
                    raise AssertionError(
                        f"rejected state hash changed at line {line_number}"
                    )
            if stage == "commit" and event.get("outcome") == "committed" and any((
                event.get("measured_cells_only") is not True,
                event.get("prediction_persisted") is not False,
            )):
                raise AssertionError(
                    f"committed map authority is invalid at line {line_number}"
                )
            if stage == "recovery" and event.get(
                "ground_truth_used_for_recovery"
            ) is not False:
                raise AssertionError(
                    f"recovery consumed ground truth at line {line_number}"
                )
            if stage == "recovery":
                invalidated = event.get("invalidated_robot_ids")
                if not isinstance(invalidated, list) or not invalidated or any(
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in invalidated
                ) or len(set(invalidated)) != len(invalidated) or event.get(
                    "source_robot"
                ) not in invalidated:
                    raise AssertionError(
                        "recovery invalidated-robot dependency closure is malformed"
                    )
                edges = event.get("invalidated_dependency_edges")
                if not isinstance(edges, list) or len(edges) != len(invalidated):
                    raise AssertionError("recovery dependency-edge evidence is absent")
                parent_by_robot: dict[int, int | None] = {}
                for edge in edges:
                    if not isinstance(edge, list) or len(edge) != 2 or edge[0] in (
                        parent_by_robot
                    ):
                        raise AssertionError("recovery dependency-edge evidence is malformed")
                    parent_by_robot[edge[0]] = edge[1]
                if set(parent_by_robot) != set(invalidated):
                    raise AssertionError("recovery dependency closure and edges disagree")
                positions = {robot_id: index for index, robot_id in enumerate(invalidated)}
                for robot_id in invalidated[1:]:
                    parent_id = parent_by_robot[robot_id]
                    if parent_id not in positions or positions[parent_id] >= positions[robot_id]:
                        raise AssertionError("recovery descendants are not parent-first")
                unsafe_ids = event.get("invalidated_unsafe_robot_ids")
                if not isinstance(unsafe_ids, list) or not set(unsafe_ids).issubset(
                    invalidated
                ):
                    raise AssertionError("recovery unsafe-transform evidence is malformed")
            rows.append(event_with_digest)
            previous = str(digest)
    if not rows:
        raise AssertionError("event log is empty")
    if rows[0]["stage"] != "world_init" or rows[0]["tick"] != 0:
        raise AssertionError("world_init must be the first event at tick zero")
    if sum(event["stage"] == "world_init" for event in rows) != 1:
        raise AssertionError("world_init must occur exactly once")
    if rows[-1]["stage"] != "run_complete":
        raise AssertionError("run_complete must be the final event")
    if sum(event["stage"] == "run_complete" for event in rows) != 1:
        raise AssertionError("run_complete must occur exactly once")

    manifest_path = path.with_name("run_manifest.json")
    summary_path = path.with_name("summary.json")
    if not manifest_path.is_file() or not summary_path.is_file():
        raise AssertionError(
            "event verification requires run_manifest.json and summary.json"
        )
    manifest = _read_object(manifest_path)
    summary = _read_object(summary_path)
    if any(manifest.get(field) != base.get(field) for field in (
        "run_id", "arm", "split", "seed", "building_id", "floorplan_id",
        "protocol_sha256",
    )):
        raise AssertionError("run manifest identity differs from the event log")
    world_init = rows[0]
    if manifest.get("initial_state_sha256") != world_init.get(
        "initial_state_sha256"
    ) or world_init.get("outputs", {}).get("initial_state") != manifest.get(
        "initial_state_sha256"
    ):
        raise AssertionError("initial map/start/private-frame digest is inconsistent")
    team_size = int(manifest.get("parameters", {}).get("team_size", 0))
    prediction_period = int(
        manifest.get("parameters", {}).get("prediction_period_ticks", 0)
    )
    planner_period = int(
        manifest.get("parameters", {}).get("planner_period_ticks", 0)
    )
    if team_size <= 0 or prediction_period <= 0 or planner_period <= 0:
        raise AssertionError("run manifest has invalid coverage parameters")
    expected_robots = list(range(team_size))
    final_tick = int(rows[-1]["tick"])
    for tick in range(final_tick + 1):
        if sorted(_robot_ids(rows, {"scan"}, tick)) != expected_robots:
            raise AssertionError(f"scan robot coverage is incomplete at tick {tick}")
        metrics = [
            event for event in rows
            if event["stage"] == "metric" and int(event["tick"]) == tick
        ]
        if len(metrics) != 1:
            raise AssertionError(f"metric coverage is incomplete at tick {tick}")
        prediction_events = sorted(_robot_ids(
            rows, {"predictor", "observed_control"}, tick
        ))
        expected_prediction = (
            expected_robots if tick % prediction_period == 0 else []
        )
        if prediction_events != expected_prediction:
            raise AssertionError(f"predictor robot coverage is invalid at tick {tick}")
        planner_events = sorted(_robot_ids(rows, {"planner"}, tick))
        expected_planner = expected_robots if tick % planner_period == 0 else []
        if planner_events != expected_planner:
            raise AssertionError(f"planner robot coverage is invalid at tick {tick}")
        motion_events = sorted(_robot_ids(rows, {"motion"}, tick))
        expected_motion = expected_robots if tick < final_tick else []
        if motion_events != expected_motion:
            raise AssertionError(f"motion robot coverage is invalid at tick {tick}")

    stripped_rows = [dict(event) for event in rows]
    for event in stripped_rows:
        event.pop("event_sha256", None)
    for event in stripped_rows:
        stage = event["stage"]
        if stage == "registration":
            _verify_parent(event, stripped_rows, {"encounter"})
        elif stage == "gate":
            _verify_parent(event, stripped_rows, {"registration"})
        elif stage == "temporal_cycle_consistency":
            _verify_parent(event, stripped_rows, {"gate"})
        elif stage == "recovery":
            _verify_parent(event, stripped_rows, {"temporal_cycle_consistency"})
        elif stage == "post_decision_evaluation":
            _verify_parent(event, stripped_rows, {"commit", "recovery"})
        elif stage == "commit" and event.get("event_id"):
            _verify_parent(event, stripped_rows, {"gate", "temporal_cycle_consistency"})
    _verify_transactions(stripped_rows)

    run_complete = rows[-1]
    summary_sha = canonical_json_sha256(summary)
    if run_complete.get("outputs", {}).get("summary") != summary_sha or (
        run_complete.get("summary_sha256") != summary_sha
    ):
        raise AssertionError("run_complete does not hash the exact summary file")
    if summary.get("event_count") != len(rows):
        raise AssertionError("summary event count is inconsistent")
    for field in (
        "run_id",
        "arm",
        "split",
        "seed",
        "building_id",
        "floorplan_id",
    ):
        if summary.get(field) != base.get(field):
            raise AssertionError(f"summary identity differs on {field}")
    if summary.get("ticks_completed") != final_tick:
        raise AssertionError("summary final tick is inconsistent")
    if summary.get("persistent_map_sha256") != run_complete.get(
        "persistent_hash_after"
    ) or summary.get("map_revision") != run_complete.get("map_revision_after"):
        raise AssertionError("summary persistent-map state is inconsistent")
    registration = summary.get("registration")
    if isinstance(registration, dict):
        unrecovered = max(
            0,
            int(registration.get("unsafe_registration_commit", 0))
            - int(registration.get("recovered_unsafe_registration_commit", 0)),
        )
        if summary.get("unsafe_commit_unrecovered_count") != unrecovered or (
            summary.get("run_safety_failure") is not (unrecovered > 0)
        ):
            raise AssertionError("summary unsafe-recovery accounting is inconsistent")
    return {
        "event_count": len(rows),
        "final_event_sha256": previous,
        "hash_preserving_rejects": reject_count,
        "stage_counts": dict(sorted(stages.items())),
        "summary_sha256": summary_sha,
        "robot_coverage_verified": True,
        "registration_transactions_verified": len(
            {event.get("event_id") for event in rows if event.get("event_id")}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_log", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_event_log(args.event_log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
