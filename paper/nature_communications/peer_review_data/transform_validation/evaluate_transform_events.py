#!/usr/bin/env python3
"""Evaluate event-level SE(2) registration and fusion decisions.

This tool intentionally does not infer transform correctness from map overlap.
It expects one CSV or JSONL row per encounter, with an explicit ground-truth
label and, for positive encounters with a returned candidate, estimated and
reference SE(2) transforms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}
REQUIRED_FIELDS = (
    "event_id",
    "run_id",
    "scene_id",
    "input_type",
    "method",
    "gt_positive",
    "candidate_returned",
    "accepted",
    "committed",
)


class ValidationError(ValueError):
    """Raised when an input event is internally inconsistent."""


@dataclass(frozen=True)
class Transform:
    tx_m: float
    ty_m: float
    yaw_deg: float


@dataclass(frozen=True)
class Event:
    event_id: str
    run_id: str
    scene_id: str
    cluster_id: str
    seed: str
    input_type: str
    method: str
    gt_positive: bool
    candidate_returned: bool
    rigid_valid: bool
    accepted: bool
    committed: bool
    estimate: Transform | None
    ground_truth: Transform | None
    injected_wrong_transform: bool
    injection_id: str
    detection_latency_events: float | None
    detection_latency_s: float | None
    recovery_success: bool | None
    recovery_horizon_events: float | None
    recovery_events: float | None
    recovery_time_s: float | None
    contaminated_map_cycles: float | None
    contaminated_cells: float | None
    raw_scale: float | None
    raw_shear: float | None
    raw_determinant: float | None
    reject_reason: str
    inlier_count: int | None
    inlier_ratio: float | None
    ransac_stage: str
    gate_score: float | None
    gate_threshold: float | None


@dataclass(frozen=True)
class EvaluatedEvent:
    event: Event
    translation_error_m: float | None
    yaw_error_deg: float | None
    correct: bool
    outcome: str


def parse_bool(value: Any, field: str, *, optional: bool = False) -> bool | None:
    if value is None or str(value).strip() == "":
        if optional:
            return None
        raise ValidationError(f"missing Boolean field {field!r}")
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValidationError(f"invalid Boolean {field}={value!r}")


def parse_float(value: Any, field: str, *, optional: bool = True) -> float | None:
    if value is None or str(value).strip() == "":
        if optional:
            return None
        raise ValidationError(f"missing numeric field {field!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"invalid number {field}={value!r}") from exc
    if not math.isfinite(result):
        raise ValidationError(f"non-finite number {field}={value!r}")
    return result


def parse_int(value: Any, field: str, *, optional: bool = True) -> int | None:
    number = parse_float(value, field, optional=optional)
    if number is None:
        return None
    if not number.is_integer() or number < 0:
        raise ValidationError(f"{field} must be a non-negative integer, got {value!r}")
    return int(number)


def parse_transform(row: Mapping[str, Any], prefix: str) -> Transform | None:
    names = (f"{prefix}_tx_m", f"{prefix}_ty_m", f"{prefix}_yaw_deg")
    present = [str(row.get(name, "")).strip() != "" for name in names]
    if not any(present):
        return None
    if not all(present):
        missing = [name for name, exists in zip(names, present) if not exists]
        raise ValidationError(f"partial {prefix} transform; missing {', '.join(missing)}")
    return Transform(
        parse_float(row[names[0]], names[0], optional=False),  # type: ignore[arg-type]
        parse_float(row[names[1]], names[1], optional=False),  # type: ignore[arg-type]
        parse_float(row[names[2]], names[2], optional=False),  # type: ignore[arg-type]
    )


def event_from_row(row: Mapping[str, Any], row_number: int) -> Event:
    missing = [name for name in REQUIRED_FIELDS if str(row.get(name, "")).strip() == ""]
    if missing:
        raise ValidationError(f"row {row_number}: missing required fields: {', '.join(missing)}")
    try:
        candidate_returned = bool(parse_bool(row["candidate_returned"], "candidate_returned"))
        estimate = parse_transform(row, "hat")
        ground_truth = parse_transform(row, "gt")
        rigid_value = parse_bool(row.get("rigid_valid"), "rigid_valid", optional=True)
        # Backward-compatible inference for pre-schema logs: a complete hat
        # transform is treated as rigid-valid unless explicitly marked false.
        rigid_valid = estimate is not None if rigid_value is None else rigid_value
        event = Event(
            event_id=str(row["event_id"]).strip(),
            run_id=str(row["run_id"]).strip(),
            scene_id=str(row["scene_id"]).strip(),
            cluster_id=str(row.get("cluster_id", "")).strip(),
            seed=str(row.get("seed", "")).strip(),
            input_type=str(row["input_type"]).strip(),
            method=str(row["method"]).strip(),
            gt_positive=bool(parse_bool(row["gt_positive"], "gt_positive")),
            candidate_returned=candidate_returned,
            rigid_valid=rigid_valid,
            accepted=bool(parse_bool(row["accepted"], "accepted")),
            committed=bool(parse_bool(row["committed"], "committed")),
            estimate=estimate,
            ground_truth=ground_truth,
            injected_wrong_transform=bool(
                parse_bool(row.get("injected_wrong_transform", "false"), "injected_wrong_transform")
            ),
            injection_id=str(row.get("injection_id", "")).strip(),
            detection_latency_events=parse_float(row.get("detection_latency_events"), "detection_latency_events"),
            detection_latency_s=parse_float(row.get("detection_latency_s"), "detection_latency_s"),
            recovery_success=parse_bool(row.get("recovery_success"), "recovery_success", optional=True),
            recovery_horizon_events=parse_float(row.get("recovery_horizon_events"), "recovery_horizon_events"),
            recovery_events=parse_float(row.get("recovery_events"), "recovery_events"),
            recovery_time_s=parse_float(row.get("recovery_time_s"), "recovery_time_s"),
            contaminated_map_cycles=parse_float(row.get("contaminated_map_cycles"), "contaminated_map_cycles"),
            contaminated_cells=parse_float(row.get("contaminated_cells"), "contaminated_cells"),
            raw_scale=parse_float(row.get("raw_scale"), "raw_scale"),
            raw_shear=parse_float(row.get("raw_shear"), "raw_shear"),
            raw_determinant=parse_float(row.get("raw_determinant"), "raw_determinant"),
            reject_reason=str(row.get("reject_reason", "")).strip(),
            inlier_count=parse_int(row.get("inlier_count"), "inlier_count"),
            inlier_ratio=parse_float(row.get("inlier_ratio"), "inlier_ratio"),
            ransac_stage=str(row.get("ransac_stage", "")).strip(),
            gate_score=parse_float(row.get("gate_score"), "gate_score"),
            gate_threshold=parse_float(row.get("gate_threshold"), "gate_threshold"),
        )
    except ValidationError as exc:
        raise ValidationError(f"row {row_number} ({row.get('event_id', '<unknown>')}): {exc}") from exc

    if event.accepted and not event.candidate_returned:
        raise ValidationError(f"row {row_number} ({event.event_id}): accepted=true requires candidate_returned=true")
    if event.committed and not event.accepted:
        raise ValidationError(f"row {row_number} ({event.event_id}): committed=true requires accepted=true")
    if event.rigid_valid and not event.candidate_returned:
        raise ValidationError(f"row {row_number} ({event.event_id}): rigid_valid=true requires candidate_returned=true")
    if event.rigid_valid and event.estimate is None:
        raise ValidationError(f"row {row_number} ({event.event_id}): rigid_valid=true requires a hat transform")
    if event.accepted and event.estimate is None:
        raise ValidationError(f"row {row_number} ({event.event_id}): accepted=true requires a hat transform")
    if event.gt_positive and event.rigid_valid and event.ground_truth is None:
        raise ValidationError(f"row {row_number} ({event.event_id}): positive rigid-valid candidate has no gt transform")
    if event.inlier_ratio is not None and not 0.0 <= event.inlier_ratio <= 1.0:
        raise ValidationError(f"row {row_number} ({event.event_id}): inlier_ratio must be within [0, 1]")
    if event.injected_wrong_transform and not event.injection_id:
        raise ValidationError(f"row {row_number} ({event.event_id}): injected event needs injection_id")
    if event.recovery_success is True and not event.committed:
        raise ValidationError(f"row {row_number} ({event.event_id}): recovery_success=true requires committed=true")
    if (
        event.recovery_success is True
        and event.recovery_horizon_events is not None
        and event.recovery_events is not None
        and event.recovery_events > event.recovery_horizon_events
    ):
        raise ValidationError(
            f"row {row_number} ({event.event_id}): recovery_success=true but "
            "recovery_events exceeds recovery_horizon_events"
        )
    return event


def load_events(path: Path) -> list[Event]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows: Iterable[Mapping[str, Any]] = csv.DictReader(handle)
            events = [event_from_row(row, i) for i, row in enumerate(rows, start=2)]
    elif suffix in {".jsonl", ".ndjson"}:
        parsed_rows = []
        with path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"line {i}: invalid JSON: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValidationError(f"line {i}: each JSONL record must be an object")
                parsed_rows.append((i, obj))
        events = [event_from_row(row, i) for i, row in parsed_rows]
    else:
        raise ValidationError("input must be .csv, .jsonl, or .ndjson")

    if not events:
        raise ValidationError("input contains no events")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for event in events:
        if event.event_id in seen:
            duplicates.add(event.event_id)
        seen.add(event.event_id)
    if duplicates:
        raise ValidationError(f"duplicate event_id values: {', '.join(sorted(duplicates))}")
    return events


def wrap_degrees(angle: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def se2_error(estimate: Transform, ground_truth: Transform) -> tuple[float, float]:
    """Return translation and yaw of inv(T_gt) @ T_hat.

    Translation is expressed in the GT frame. Its Euclidean norm is invariant
    to that rotation, but the explicit computation documents the convention.
    """
    delta_x = estimate.tx_m - ground_truth.tx_m
    delta_y = estimate.ty_m - ground_truth.ty_m
    theta = math.radians(ground_truth.yaw_deg)
    error_x = math.cos(theta) * delta_x + math.sin(theta) * delta_y
    error_y = -math.sin(theta) * delta_x + math.cos(theta) * delta_y
    translation_error = math.hypot(error_x, error_y)
    yaw_error = abs(wrap_degrees(estimate.yaw_deg - ground_truth.yaw_deg))
    return translation_error, yaw_error


def evaluate_events(events: Sequence[Event], tau_translation_m: float, tau_yaw_deg: float) -> list[EvaluatedEvent]:
    if tau_translation_m < 0 or tau_yaw_deg < 0:
        raise ValidationError("correctness thresholds must be non-negative")
    evaluated: list[EvaluatedEvent] = []
    for event in events:
        translation_error = yaw_error = None
        if event.candidate_returned and event.rigid_valid and event.gt_positive:
            assert event.estimate is not None and event.ground_truth is not None
            translation_error, yaw_error = se2_error(event.estimate, event.ground_truth)
        correct = bool(
            event.gt_positive
            and event.candidate_returned
            and event.rigid_valid
            and translation_error is not None
            and yaw_error is not None
            and translation_error <= tau_translation_m
            and yaw_error <= tau_yaw_deg
        )
        if event.accepted:
            outcome = "TA" if correct else "FA"
        else:
            outcome = "FR" if event.gt_positive else "TR"
        evaluated.append(EvaluatedEvent(event, translation_error, yaw_error, correct, outcome))
    return evaluated


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
    }


def group_evaluated(events: Sequence[EvaluatedEvent]) -> dict[tuple[str, str], list[EvaluatedEvent]]:
    groups: dict[tuple[str, str], list[EvaluatedEvent]] = defaultdict(list)
    for item in events:
        groups[(item.event.input_type, item.event.method)].append(item)
    return dict(sorted(groups.items()))


def cluster_value(event: Event, field: str) -> str:
    supported = {"scene_id", "cluster_id", "run_id", "seed", "event_id"}
    if field not in supported:
        raise ValidationError(f"unsupported cluster field {field!r}; choose one of {', '.join(sorted(supported))}")
    value = str(getattr(event, field)).strip()
    if not value:
        raise ValidationError(f"event {event.event_id}: selected cluster field {field!r} is empty")
    return value


def bootstrap_intervals(
    events: Sequence[EvaluatedEvent], cluster_field: str, reps: int, seed: int
) -> dict[tuple[str, str], dict[str, tuple[float | None, float | None, int]]]:
    """Cluster bootstrap percentile intervals, sampled independently per group."""
    if reps < 0:
        raise ValidationError("bootstrap_reps must be non-negative")
    result: dict[tuple[str, str], dict[str, tuple[float | None, float | None, int]]] = {}
    rng = random.Random(seed)
    for key, group in group_evaluated(events).items():
        clusters: dict[str, list[EvaluatedEvent]] = defaultdict(list)
        for item in group:
            clusters[cluster_value(item.event, cluster_field)].append(item)
        names = sorted(clusters)
        samples: dict[str, list[float]] = defaultdict(list)
        for _ in range(reps):
            resampled: list[EvaluatedEvent] = []
            for _ in names:
                resampled.extend(clusters[rng.choice(names)])
            translation = [x.translation_error_m for x in resampled if x.event.gt_positive and x.event.candidate_returned and x.translation_error_m is not None]
            yaw = [x.yaw_error_deg for x in resampled if x.event.gt_positive and x.event.candidate_returned and x.yaw_error_deg is not None]
            if translation:
                samples["translation_m_median"].append(statistics.median(translation))  # type: ignore[arg-type]
            if yaw:
                samples["yaw_deg_median"].append(statistics.median(yaw))  # type: ignore[arg-type]
            accepted = sum(x.event.accepted for x in resampled)
            positives = sum(x.event.gt_positive for x in resampled)
            if accepted:
                samples["fa_per_accepted"].append(sum(x.outcome == "FA" for x in resampled) / accepted)
            if positives:
                samples["fr_per_positive"].append(sum(x.outcome == "FR" for x in resampled) / positives)
            injected = [x for x in resampled if x.event.injected_wrong_transform]
            if injected:
                samples["safe_reject_rate"].append(
                    sum((not x.event.accepted) and (not x.event.committed) for x in injected) / len(injected)
                )
                samples["incorrect_commit_rate"].append(sum(x.event.committed for x in injected) / len(injected))
        intervals: dict[str, tuple[float | None, float | None, int]] = {}
        for metric in (
            "translation_m_median", "yaw_deg_median", "fa_per_accepted",
            "fr_per_positive", "safe_reject_rate", "incorrect_commit_rate",
        ):
            values = samples.get(metric, [])
            intervals[metric] = (percentile(values, 0.025), percentile(values, 0.975), len(values))
        result[key] = intervals
    return result


def add_ci(row: dict[str, Any], intervals: Mapping[str, tuple[float | None, float | None, int]], metric: str) -> None:
    low, high, valid = intervals.get(metric, (None, None, 0))
    row[f"{metric}_ci95_low"] = low
    row[f"{metric}_ci95_high"] = high
    row[f"{metric}_bootstrap_valid_n"] = valid


def build_accuracy_rows(
    events: Sequence[EvaluatedEvent],
    intervals: Mapping[tuple[str, str], Mapping[str, tuple[float | None, float | None, int]]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (input_type, method), group in group_evaluated(events).items():
        positives = [x for x in group if x.event.gt_positive]
        returned = [x for x in positives if x.event.candidate_returned]
        rigid_valid = [x for x in returned if x.event.rigid_valid]
        translation = [x.translation_error_m for x in rigid_valid if x.translation_error_m is not None]
        yaw = [x.yaw_error_deg for x in rigid_valid if x.yaw_error_deg is not None]
        t = distribution(translation)  # type: ignore[arg-type]
        y = distribution(yaw)  # type: ignore[arg-type]
        row = {
            "input_type": input_type,
            "method": method,
            "positive_n": len(positives),
            "candidate_returned_n": len(returned),
            "rigid_valid_returned_n": len(rigid_valid),
            "numeric_error_n": len(translation),
            "translation_m_median": t["median"],
            "translation_m_q1": t["q1"],
            "translation_m_q3": t["q3"],
            "translation_m_p90": t["p90"],
            "yaw_deg_median": y["median"],
            "yaw_deg_q1": y["q1"],
            "yaw_deg_q3": y["q3"],
            "yaw_deg_p90": y["p90"],
        }
        group_ci = (intervals or {}).get((input_type, method), {})
        add_ci(row, group_ci, "translation_m_median")
        add_ci(row, group_ci, "yaw_deg_median")
        rows.append(row)
    return rows


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_reliability_rows(
    events: Sequence[EvaluatedEvent],
    intervals: Mapping[tuple[str, str], Mapping[str, tuple[float | None, float | None, int]]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (input_type, method), group in group_evaluated(events).items():
        positive_n = sum(x.event.gt_positive for x in group)
        negative_n = len(group) - positive_n
        accepted = sum(x.event.accepted for x in group)
        rejected = len(group) - accepted
        counts = {label: sum(x.outcome == label for x in group) for label in ("TA", "FA", "FR", "TR")}
        row = {
            "input_type": input_type,
            "method": method,
            "events_n": len(group),
            "positive_n": positive_n,
            "negative_n": negative_n,
            "accepted_n": accepted,
            "rejected_n": rejected,
            "ta_n": counts["TA"],
            "fa_n": counts["FA"],
            "fa_per_accepted": ratio(counts["FA"], accepted),
            "fr_n": counts["FR"],
            "fr_per_positive": ratio(counts["FR"], positive_n),
            "tr_n": counts["TR"],
            "gate_contract_violation_n": sum(x.event.accepted and not x.event.rigid_valid for x in group),
        }
        group_ci = (intervals or {}).get((input_type, method), {})
        add_ci(row, group_ci, "fa_per_accepted")
        add_ci(row, group_ci, "fr_per_positive")
        rows.append(row)
    return rows


def build_injection_rows(
    events: Sequence[EvaluatedEvent],
    intervals: Mapping[tuple[str, str], Mapping[str, tuple[float | None, float | None, int]]] | None = None,
) -> list[dict[str, Any]]:
    injected = [x for x in events if x.event.injected_wrong_transform]
    groups: dict[tuple[str, str], list[EvaluatedEvent]] = defaultdict(list)
    for item in injected:
        groups[(item.event.input_type, item.event.method)].append(item)
    rows: list[dict[str, Any]] = []
    for (input_type, method), group in sorted(groups.items()):
        safe_reject = sum((not x.event.accepted) and (not x.event.committed) for x in group)
        incorrect_commit = sum(x.event.committed for x in group)
        committed = [x for x in group if x.event.committed]
        recovery_observed = [x for x in committed if x.event.recovery_success is not None]
        recovered = [x for x in recovery_observed if x.event.recovery_success]
        numeric_fields = {
            "detection_latency_events": [x.event.detection_latency_events for x in group if x.event.detection_latency_events is not None],
            "detection_latency_s": [x.event.detection_latency_s for x in group if x.event.detection_latency_s is not None],
            "recovery_events": [x.event.recovery_events for x in recovered if x.event.recovery_events is not None],
            "recovery_time_s": [x.event.recovery_time_s for x in recovered if x.event.recovery_time_s is not None],
            "contaminated_map_cycles": [x.event.contaminated_map_cycles for x in group if x.event.contaminated_map_cycles is not None],
            "contaminated_cells": [x.event.contaminated_cells for x in group if x.event.contaminated_cells is not None],
        }
        row: dict[str, Any] = {
            "input_type": input_type,
            "method": method,
            "injections_n": len(group),
            "accepted_n": sum(x.event.accepted for x in group),
            "rejected_n": sum(not x.event.accepted for x in group),
            "safe_reject_n": safe_reject,
            "safe_reject_rate": ratio(safe_reject, len(group)),
            "incorrect_commit_n": incorrect_commit,
            "incorrect_commit_rate": ratio(incorrect_commit, len(group)),
            "recovery_observed_n": len(recovery_observed),
            "recovered_within_horizon_n": len(recovered),
            "recovery_rate_per_observed_commit": ratio(len(recovered), len(recovery_observed)),
        }
        for name, values in numeric_fields.items():
            stats = distribution(values)  # type: ignore[arg-type]
            row[f"{name}_n"] = stats["n"]
            row[f"{name}_median"] = stats["median"]
            row[f"{name}_p90"] = stats["p90"]
        group_ci = (intervals or {}).get((input_type, method), {})
        add_ci(row, group_ci, "safe_reject_rate")
        add_ci(row, group_ci, "incorrect_commit_rate")
        rows.append(row)
    return rows


def build_event_rows(events: Sequence[EvaluatedEvent]) -> list[dict[str, Any]]:
    rows = []
    for item in events:
        event = item.event
        rows.append({
            "event_id": event.event_id,
            "run_id": event.run_id,
            "scene_id": event.scene_id,
            "cluster_id": event.cluster_id,
            "seed": event.seed,
            "input_type": event.input_type,
            "method": event.method,
            "gt_positive": event.gt_positive,
            "candidate_returned": event.candidate_returned,
            "rigid_valid": event.rigid_valid,
            "accepted": event.accepted,
            "committed": event.committed,
            "translation_error_m": item.translation_error_m,
            "yaw_error_deg": item.yaw_error_deg,
            "correct_within_thresholds": item.correct,
            "outcome": item.outcome,
            "injected_wrong_transform": event.injected_wrong_transform,
            "injection_id": event.injection_id,
            "raw_scale": event.raw_scale,
            "raw_shear": event.raw_shear,
            "raw_determinant": event.raw_determinant,
            "reject_reason": event.reject_reason,
            "inlier_count": event.inlier_count,
            "inlier_ratio": event.inlier_ratio,
            "ransac_stage": event.ransac_stage,
            "gate_score": event.gate_score,
            "gate_threshold": event.gate_threshold,
            "gate_contract_valid": not (event.accepted and not event.rigid_valid),
        })
    return rows


def format_number(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def med_iqr_p90(row: Mapping[str, Any], prefix: str) -> str:
    if row.get(f"{prefix}_median") is None:
        return "NA"
    return (
        f"{format_number(row[f'{prefix}_median'])} "
        f"[{format_number(row[f'{prefix}_q1'])}, {format_number(row[f'{prefix}_q3'])}]; "
        f"{format_number(row[f'{prefix}_p90'])}"
    )


def ci95(row: Mapping[str, Any], metric: str, *, percent: bool = False) -> str:
    low = row.get(f"{metric}_ci95_low")
    high = row.get(f"{metric}_ci95_high")
    if low is None or high is None:
        return "NA"
    scale = 100.0 if percent else 1.0
    return f"[{format_number(scale * low)}, {format_number(scale * high)}]"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return format_number(value).replace("|", "\\|")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def latex_escape(value: Any) -> str:
    text = format_number(value)
    for old, new in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        text = text.replace(old, new)
    return text


def latex_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], caption: str, label: str) -> str:
    columns = "l" * len(headers)
    body = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(latex_escape(x) for x in headers) + " \\\\",
        "\\midrule",
    ]
    body.extend(" & ".join(latex_escape(x) for x in row) + " \\\\" for row in rows)
    body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(body)


def render_report(
    accuracy: Sequence[Mapping[str, Any]],
    reliability: Sequence[Mapping[str, Any]],
    injections: Sequence[Mapping[str, Any]],
    tau_translation_m: float,
    tau_yaw_deg: float,
    cluster_field: str,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> tuple[str, str]:
    accuracy_headers = ["Input", "Method", "GT+ N", "Returned/rigid-valid", "Translation m: median [IQR]; P90; median 95% CI", "Yaw deg: median [IQR]; P90; median 95% CI"]
    accuracy_values = [[
        row["input_type"], row["method"], row["positive_n"], f"{row['candidate_returned_n']}/{row['rigid_valid_returned_n']}",
        f"{med_iqr_p90(row, 'translation_m')}; {ci95(row, 'translation_m_median')}",
        f"{med_iqr_p90(row, 'yaw_deg')}; {ci95(row, 'yaw_deg_median')}",
    ] for row in accuracy]
    reliability_headers = ["Input", "Method", "Pos/neg", "Accepted", "Rejected", "TA", "FA (% accepted)", "FR (% positive)", "TR", "Gate violations"]
    reliability_values = [[
        row["input_type"], row["method"], f"{row['positive_n']}/{row['negative_n']}", row["accepted_n"], row["rejected_n"], row["ta_n"],
        f"{row['fa_n']} ({format_number(None if row['fa_per_accepted'] is None else 100 * row['fa_per_accepted'], 1)}%; CI {ci95(row, 'fa_per_accepted', percent=True)})",
        f"{row['fr_n']} ({format_number(None if row['fr_per_positive'] is None else 100 * row['fr_per_positive'], 1)}%; CI {ci95(row, 'fr_per_positive', percent=True)})", row["tr_n"], row["gate_contract_violation_n"],
    ] for row in reliability]
    injection_headers = ["Input", "Method", "Injected N", "Accept/reject", "Safe reject", "Incorrect commit", "Recovered/observed commits", "Detection latency events med/P90", "Recovery events med/P90", "Contaminated cycles med/P90"]
    injection_values = [[
        row["input_type"], row["method"], row["injections_n"],
        f"{row['accepted_n']}/{row['rejected_n']}",
        f"{row['safe_reject_n']} ({format_number(100 * row['safe_reject_rate'], 1)}%; CI {ci95(row, 'safe_reject_rate', percent=True)})",
        f"{row['incorrect_commit_n']} ({format_number(100 * row['incorrect_commit_rate'], 1)}%; CI {ci95(row, 'incorrect_commit_rate', percent=True)})",
        f"{row['recovered_within_horizon_n']}/{row['recovery_observed_n']}",
        f"{format_number(row['detection_latency_events_median'])}/{format_number(row['detection_latency_events_p90'])}",
        f"{format_number(row['recovery_events_median'])}/{format_number(row['recovery_events_p90'])}",
        f"{format_number(row['contaminated_map_cycles_median'])}/{format_number(row['contaminated_map_cycles_p90'])}",
    ] for row in injections]

    threshold_text = f"tau_t = {tau_translation_m:g} m and tau_yaw = {tau_yaw_deg:g} deg"
    markdown = "\n\n".join([
        "# Transform validation report",
        f"Correctness thresholds: **{threshold_text}**. Errors use `E = inv(T_gt) @ T_hat` under the `T_i<-j` convention.",
        f"Uncertainty: **95% percentile cluster bootstrap**, cluster `{cluster_field}`, {bootstrap_reps} replicates, seed {bootstrap_seed}. Clusters, not individual encounters, are sampled with replacement.",
        "## A. SE(2) accuracy on returned positive candidates\n\n" + markdown_table(accuracy_headers, accuracy_values),
        "## B. Selective reliability\n\n" + markdown_table(reliability_headers, reliability_values),
        "## C. Wrong-transform injection safety\n\n" + (markdown_table(injection_headers, injection_values) if injection_values else "No injected events were present."),
        "FA is reported per accepted encounter; FR is reported per GT-positive encounter. A pre-commit injection rejection has both `accepted=false` and `committed=false`. CIs are cluster-bootstrap percentile intervals.",
    ]) + "\n"

    latex_parts = [
        "% Requires \\usepackage{booktabs}. Generated by evaluate_transform_events.py.",
        f"% Correctness thresholds: {threshold_text}.",
        f"% 95% percentile cluster bootstrap: cluster={cluster_field}, reps={bootstrap_reps}, seed={bootstrap_seed}.",
        latex_table(accuracy_headers, accuracy_values, "SE(2) accuracy on returned candidates. Values are median [IQR]; P90; cluster-bootstrap 95\\% CI for the median.", "tab:transform-accuracy"),
        latex_table(reliability_headers, reliability_values, "Selective reliability of encounter decisions.", "tab:transform-reliability"),
    ]
    if injection_values:
        latex_parts.append(latex_table(injection_headers, injection_values, "Safety under injected wrong transforms.", "tab:transform-injection"))
    return markdown, "\n\n".join(latex_parts) + "\n"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(
    input_path: Path,
    output_dir: Path,
    tau_translation_m: float,
    tau_yaw_deg: float,
    cluster_field: str = "scene_id",
    bootstrap_reps: int = 10_000,
    bootstrap_seed: int = 20_260_807,
) -> dict[str, Any]:
    events = load_events(input_path)
    evaluated = evaluate_events(events, tau_translation_m, tau_yaw_deg)
    intervals = bootstrap_intervals(evaluated, cluster_field, bootstrap_reps, bootstrap_seed)
    accuracy = build_accuracy_rows(evaluated, intervals)
    reliability = build_reliability_rows(evaluated, intervals)
    injections = build_injection_rows(evaluated, intervals)
    markdown, latex = render_report(
        accuracy,
        reliability,
        injections,
        tau_translation_m,
        tau_yaw_deg,
        cluster_field,
        bootstrap_reps,
        bootstrap_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "evaluated_events.csv", build_event_rows(evaluated))
    write_csv(output_dir / "transform_accuracy.csv", accuracy)
    write_csv(output_dir / "selective_reliability.csv", reliability)
    output_names = [
        "evaluated_events.csv",
        "transform_accuracy.csv",
        "selective_reliability.csv",
        "report.md",
        "tables.tex",
        "manifest.json",
    ]
    if injections:
        write_csv(output_dir / "injection_safety.csv", injections)
        output_names.insert(3, "injection_safety.csv")
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    (output_dir / "tables.tex").write_text(latex, encoding="utf-8")
    manifest = {
        "input": input_path.as_posix(),
        "event_count": len(events),
        "tau_translation_m": tau_translation_m,
        "tau_yaw_deg": tau_yaw_deg,
        "cluster_field": cluster_field,
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": bootstrap_seed,
        "confidence_interval": "cluster-bootstrap percentile 95%",
        "transform_convention": "T_i<-j; E=inv(T_gt)@T_hat",
        "outputs": output_names,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="event-level .csv or .jsonl file")
    parser.add_argument("--output-dir", type=Path, default=Path("transform_validation_results"))
    parser.add_argument("--tau-translation-m", type=float, required=True, help="translation correctness limit used for analysis")
    parser.add_argument("--tau-yaw-deg", type=float, required=True, help="absolute yaw correctness limit used for analysis")
    parser.add_argument("--cluster-field", default="scene_id", choices=("scene_id", "cluster_id", "run_id", "seed", "event_id"), help="whole cluster resampling unit")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_807)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(
            args.input,
            args.output_dir,
            args.tau_translation_m,
            args.tau_yaw_deg,
            cluster_field=args.cluster_field,
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (OSError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
