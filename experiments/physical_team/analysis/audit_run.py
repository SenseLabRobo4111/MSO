#!/usr/bin/env python3
"""Audit one physical-team run and derive reviewer-facing metrics."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import sys

try:
    import yaml
except ImportError as error:
    raise SystemExit('PyYAML is required: install python3-yaml') from error
try:
    from jsonschema import Draft202012Validator, FormatChecker
    from jsonschema.exceptions import SchemaError
except ImportError as error:
    raise SystemExit(
        'jsonschema is required: install python3-jsonschema') from error


ROBOT_TOPIC = re.compile(r'^/robot_(\d+)/')
MAP_HASH = re.compile(r'^[0-9a-f]{64}$')
IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')
GIT_HASH = re.compile(r'^[0-9a-f]{40}$')
SCHEMA_DIR = Path(__file__).resolve().parents[1] / 'schema'
SCHEMA_FILES = {
    'manifest': 'run_manifest.schema.json',
    'registration': 'registration_event.schema.json',
    'reference': 'reference_transform.schema.json',
    'recovery_evidence': 'recovery_evidence.schema.json',
    'safety': 'safety_event.schema.json',
    'coverage': 'coverage_event.schema.json',
    'network': 'network_event.schema.json',
    'traffic_probe': 'traffic_probe_event.schema.json',
}


def load_json(path: Path):
    """Load one JSON object."""
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path: Path):
    """Load an append-only JSONL file and report malformed rows."""
    values = []
    errors = []
    if not path.exists():
        return values, errors
    for line_number, line in enumerate(
            path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('row is not an object')
            values.append(value)
        except (json.JSONDecodeError, ValueError) as error:
            errors.append({'line': line_number, 'error': str(error)})
    return values, errors


def build_schema_validators():
    """Load and check every physical-evidence JSON schema before auditing."""
    validators = {}
    for name, filename in SCHEMA_FILES.items():
        schema = load_json(SCHEMA_DIR / filename)
        Draft202012Validator.check_schema(schema)
        validators[name] = Draft202012Validator(
            schema, format_checker=FormatChecker())
    return validators


def validate_schema(value, validator, source):
    """Return stable, serialisable validation errors for one JSON value."""
    errors = []
    for error in sorted(
            validator.iter_errors(value),
            key=lambda item: tuple(str(part) for part in item.absolute_path)):
        errors.append({
            'source': str(source),
            'path': '/'.join(str(part) for part in error.absolute_path),
            'error': error.message,
        })
    return errors


def validate_rows(rows, validator, source):
    """Validate every JSONL row and retain its one-based line identity."""
    errors = []
    for line_number, row in enumerate(rows, 1):
        errors.extend(validate_schema(
            row, validator, f'{source}:{line_number}'))
    return errors


def resolved_value(value):
    """Reject nulls, blank strings, empty containers, and non-finite numbers."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return bool(value) and all(
            resolved_value(key) and resolved_value(item)
            for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(resolved_value(item) for item in value)
    return False


def frozen_lock_errors(manifest):
    """Validate every field whose identity is frozen across the campaign."""
    ground_truth = manifest.get('ground_truth_reference') or {}
    protocol = manifest.get('protocol_lock') or {}
    values = {
        'site_id': manifest.get('site_id'),
        'arena_id': manifest.get('arena_id'),
        'git_commit': manifest.get('git_commit'),
        'checkpoint_sha256': manifest.get('checkpoint_sha256'),
        'stopping_rule_id': protocol.get('stopping_rule_id'),
        'motion_limits_id': protocol.get('motion_limits_id'),
        'sensing_policy_id': protocol.get('sensing_policy_id'),
        'reference_timestamp_tolerance_ms': protocol.get(
            'reference_timestamp_tolerance_ms'),
        'calibration_sha256': ground_truth.get('calibration_sha256'),
        'extrinsics_sha256': ground_truth.get('extrinsics_sha256'),
        'uncertainty_translation_m': ground_truth.get(
            'uncertainty_translation_m'),
        'uncertainty_yaw_deg': ground_truth.get('uncertainty_yaw_deg'),
    }
    errors = []
    for field, value in values.items():
        valid = resolved_value(value)
        if field in {
                'site_id', 'arena_id', 'stopping_rule_id',
                'motion_limits_id', 'sensing_policy_id'}:
            valid = valid and isinstance(value, str) and bool(
                IDENTIFIER.fullmatch(value))
        elif field == 'git_commit':
            valid = valid and isinstance(value, str) and bool(
                GIT_HASH.fullmatch(value))
        elif field.endswith('_sha256'):
            valid = valid and isinstance(value, str) and bool(
                MAP_HASH.fullmatch(value))
        elif field.startswith('uncertainty_'):
            valid = valid and isinstance(value, (int, float)) and not isinstance(
                value, bool) and value >= 0
        elif field == 'reference_timestamp_tolerance_ms':
            valid = valid and isinstance(value, (int, float)) and not isinstance(
                value, bool) and value > 0
        if not valid:
            errors.append({'field': field, 'value': value})
    return errors


def reference_identifier_errors(references):
    """Return duplicate and empty independent-reference identifiers."""
    identifiers = [row.get('event_id') for row in references]
    duplicates = sorted(
        event_id for event_id, count in Counter(identifiers).items()
        if isinstance(event_id, str) and event_id and count > 1)
    empty_count = sum(
        not isinstance(event_id, str) or not event_id.strip()
        for event_id in identifiers)
    return identifiers, duplicates, empty_count


def required_topics(config):
    """Expand required topics for every configured robot."""
    templates = config['topic_templates']
    topics = list(templates.get('required_global', []))
    for robot in config['robots']:
        topics.extend(
            template.format(id=robot['id'])
            for template in templates.get('required_per_robot', []))
    return sorted(set(topics))


def read_sqlite_topics(paths):
    """Aggregate message counts and time spans from split SQLite bags."""
    stats = defaultdict(lambda: {'count': 0, 'start_ns': None, 'end_ns': None})
    errors = []
    for path in paths:
        try:
            uri = f'file:{path.resolve().as_posix()}?mode=ro'
            connection = sqlite3.connect(uri, uri=True)
            rows = connection.execute(
                'SELECT topics.name, COUNT(messages.id), '
                'MIN(messages.timestamp), MAX(messages.timestamp) '
                'FROM topics LEFT JOIN messages '
                'ON topics.id = messages.topic_id GROUP BY topics.id'
            ).fetchall()
            connection.close()
            for name, count, start_ns, end_ns in rows:
                item = stats[name]
                item['count'] += int(count or 0)
                if start_ns is not None:
                    item['start_ns'] = (
                        int(start_ns) if item['start_ns'] is None
                        else min(item['start_ns'], int(start_ns)))
                    item['end_ns'] = (
                        int(end_ns) if item['end_ns'] is None
                        else max(item['end_ns'], int(end_ns)))
        except (sqlite3.Error, OSError) as error:
            errors.append({'path': str(path), 'error': str(error)})
    return dict(stats), errors


def count_topic_messages_after(paths, topic_name, timestamp_ns):
    """Count messages for one topic at or after an absolute bag timestamp."""
    count = 0
    errors = []
    for path in paths:
        try:
            uri = f'file:{path.resolve().as_posix()}?mode=ro'
            connection = sqlite3.connect(uri, uri=True)
            value = connection.execute(
                'SELECT COUNT(messages.id) FROM messages JOIN topics '
                'ON topics.id = messages.topic_id '
                'WHERE topics.name = ? AND messages.timestamp >= ?',
                (topic_name, int(timestamp_ns))).fetchone()
            connection.close()
            count += int(value[0] or 0)
        except (sqlite3.Error, OSError) as error:
            errors.append({'path': str(path), 'error': str(error)})
    return count, errors


def wrap_yaw_error(degrees):
    """Return the absolute wrapped yaw error in degrees."""
    return abs((degrees + 180.0) % 360.0 - 180.0)


def registration_lifecycle_errors(events):
    """Return invalid decisions, mutations, commits, and recoveries."""
    errors = []
    committed_by_id = {
        event.get('event_id'): event for event in events
        if event.get('state') == 'committed'}
    grouped = defaultdict(list)
    for event in events:
        grouped[event.get('event_id')].append(event)
    for event_id, rows in grouped.items():
        if not isinstance(event_id, str) or not event_id.strip():
            errors.append({'event_id': event_id, 'reason': 'empty event_id'})
            continue
        ordered = sorted(rows, key=lambda row: int(row.get('timestamp_ns', 0)))
        states = [row.get('state') for row in ordered]
        timestamps = [int(row.get('timestamp_ns', 0)) for row in ordered]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            errors.append({'event_id': event_id, 'reason': 'non-increasing time'})
        if states == ['recovered']:
            recovery = ordered[0]
            required = (
                recovery.get('parent_event_id'),
                recovery.get('map_revision_before'),
                recovery.get('map_revision_after'),
                recovery.get('map_hash_before'),
                recovery.get('map_hash_after'),
                recovery.get('recovery_verification'))
            if any(value is None for value in required):
                errors.append({
                    'event_id': event_id,
                    'reason': 'recovery proof fields are incomplete'})
                continue
            if not MAP_HASH.fullmatch(str(recovery['map_hash_before'])) or not (
                    MAP_HASH.fullmatch(str(recovery['map_hash_after']))):
                errors.append({
                    'event_id': event_id,
                    'reason': 'recovery map hash is invalid'})
            if recovery['map_revision_after'] <= recovery['map_revision_before']:
                errors.append({
                    'event_id': event_id,
                    'reason': 'recovery did not create a newer revision'})
            if recovery['map_hash_after'] == recovery['map_hash_before']:
                errors.append({
                    'event_id': event_id,
                    'reason': 'recovery did not replace the invalid map hash'})
            verification = recovery['recovery_verification']
            if not isinstance(verification, dict) or any((
                    verification.get('invalidated_commit_event_id')
                    != recovery.get('parent_event_id'),
                    verification.get('invalidated_revision')
                    != recovery.get('map_revision_before'),
                    verification.get('invalidated_map_hash')
                    != recovery.get('map_hash_before'),
                    verification.get('active_revision')
                    != recovery.get('map_revision_after'),
                    verification.get('active_map_hash')
                    != recovery.get('map_hash_after'))):
                errors.append({
                    'event_id': event_id,
                    'reason': 'recovery verification contradicts map state'})
            if not isinstance(verification, dict) or not isinstance(
                    verification.get('method'), str) or not verification.get(
                        'method', '').strip() or not isinstance(
                            verification.get('verified_by'), str) or not (
                                verification.get('verified_by', '').strip()):
                errors.append({
                    'event_id': event_id,
                    'reason': 'recovery method or verifier is missing'})
            parent = committed_by_id.get(recovery.get('parent_event_id'))
            if parent is None:
                errors.append({
                    'event_id': event_id,
                    'reason': 'recovery parent commit is missing'})
            elif (
                    recovery.get('source_robot_id'),
                    recovery.get('target_robot_id')) != (
                        parent.get('source_robot_id'),
                        parent.get('target_robot_id')):
                errors.append({
                    'event_id': event_id,
                    'reason': 'recovery robot pair differs from parent commit'})
            continue
        allowed = (
            ['candidate', 'accepted'],
            ['candidate', 'accepted', 'committed'],
            ['candidate', 'rejected'])
        if states not in allowed:
            errors.append({
                'event_id': event_id,
                'reason': f'invalid lifecycle: {states}'})
        if len(states) != len(set(states)):
            errors.append({'event_id': event_id, 'reason': 'duplicate state'})
        pairs = {
            (row.get('source_robot_id'), row.get('target_robot_id'))
            for row in ordered
        }
        if len(pairs) != 1:
            errors.append({'event_id': event_id, 'reason': 'robot pair changed'})
        decisions = [
            row for row in ordered if row.get('state') in ('accepted', 'rejected')]
        for decision in decisions:
            if any(decision.get(key) is None for key in (
                    'map_revision_before', 'map_revision_after',
                    'map_hash_before', 'map_hash_after')):
                errors.append({
                    'event_id': event_id,
                    'reason': 'decision lacks pre/post map proof'})
                continue
            if not MAP_HASH.fullmatch(str(decision['map_hash_before'])) or not (
                    MAP_HASH.fullmatch(str(decision['map_hash_after']))):
                errors.append({
                    'event_id': event_id,
                    'reason': 'decision map hash is invalid'})
            if decision['map_revision_before'] != decision['map_revision_after']:
                errors.append({
                    'event_id': event_id,
                    'reason': 'pre-commit decision changed map revision'})
            if decision['map_hash_before'] != decision['map_hash_after']:
                reason = (
                    'rejected decision changed map hash'
                    if decision['state'] == 'rejected'
                    else 'accepted decision changed map hash before commit')
                errors.append({'event_id': event_id, 'reason': reason})
            expected_gate = 'accept' if decision['state'] == 'accepted' else 'reject'
            if (decision.get('gate') or {}).get('decision') != expected_gate:
                errors.append({
                    'event_id': event_id,
                    'reason': 'decision state and gate label disagree'})
        commits = [row for row in ordered if row.get('state') == 'committed']
        for commit in commits:
            if any(commit.get(key) is None for key in (
                    'map_revision_before', 'map_revision_after',
                    'map_hash_before', 'map_hash_after')):
                errors.append({
                    'event_id': event_id,
                    'reason': 'commit lacks pre/post map proof'})
                continue
            if not MAP_HASH.fullmatch(str(commit['map_hash_before'])) or not (
                    MAP_HASH.fullmatch(str(commit['map_hash_after']))):
                errors.append({
                    'event_id': event_id,
                    'reason': 'commit map hash is invalid'})
            if commit['map_revision_after'] <= commit['map_revision_before']:
                errors.append({
                    'event_id': event_id,
                    'reason': 'commit did not create a newer revision'})
            if commit['map_hash_after'] == commit['map_hash_before']:
                errors.append({
                    'event_id': event_id,
                    'reason': 'commit did not change map hash'})
            acceptance = next(
                (row for row in decisions if row['state'] == 'accepted'), None)
            if acceptance and (
                    commit['map_revision_before']
                    != acceptance['map_revision_after']
                    or commit['map_hash_before']
                    != acceptance['map_hash_after']):
                errors.append({
                    'event_id': event_id,
                    'reason': 'commit pre-state differs from accepted state'})
    return errors


def reference_timestamp_errors(references, registration_by_id, tolerance_ns):
    """Require positive reference times near every decision and commit."""
    errors = []
    for reference in references:
        event_id = reference.get('event_id')
        timestamp_ns = reference.get('timestamp_ns')
        if not isinstance(timestamp_ns, int) or isinstance(
                timestamp_ns, bool) or timestamp_ns <= 0:
            errors.append({
                'event_id': event_id,
                'reason': 'reference timestamp is not a positive integer'})
            continue
        compared = [
            event for event in registration_by_id.get(event_id, [])
            if event.get('state') in ('accepted', 'rejected', 'committed')]
        if not compared:
            errors.append({
                'event_id': event_id,
                'reason': 'reference has no matching decision or commit'})
            continue
        for event in compared:
            event_timestamp = event.get('timestamp_ns')
            if not isinstance(event_timestamp, int) or isinstance(
                    event_timestamp, bool) or event_timestamp <= 0 or abs(
                        timestamp_ns - event_timestamp) > tolerance_ns:
                errors.append({
                    'event_id': event_id,
                    'state': event.get('state'),
                    'reason': 'reference timestamp exceeds frozen tolerance'})
    return errors


def recovery_evidence_errors(events, run_dir, run_id, validator):
    """Verify hashed, in-run, post-recovery independent map-state records."""
    errors = []
    for event in events:
        if event.get('state') != 'recovered':
            continue
        event_id = event.get('event_id')
        verification = event.get('recovery_verification')
        if not isinstance(verification, dict):
            errors.append({
                'event_id': event_id,
                'reason': 'recovery verification is not an object'})
            continue
        relative = verification.get('evidence_path')
        declared_hash = verification.get('evidence_sha256')
        if not isinstance(relative, str) or not relative.strip() or Path(
                relative).is_absolute():
            errors.append({
                'event_id': event_id,
                'reason': 'recovery evidence path must be relative and non-empty'})
            continue
        try:
            path = (run_dir / relative).resolve()
            path.relative_to(run_dir.resolve())
            path.relative_to((run_dir / 'evidence').resolve())
        except (OSError, ValueError):
            errors.append({
                'event_id': event_id,
                'reason': 'recovery evidence path is outside run evidence'})
            continue
        if not path.is_file():
            errors.append({
                'event_id': event_id,
                'reason': 'recovery evidence file is missing'})
            continue
        if not isinstance(declared_hash, str) or not MAP_HASH.fullmatch(
                declared_hash) or sha256(path) != declared_hash:
            errors.append({
                'event_id': event_id,
                'reason': 'recovery evidence SHA-256 is invalid or differs'})
            continue
        try:
            evidence = load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            errors.append({
                'event_id': event_id,
                'reason': f'recovery evidence JSON is invalid: {error}'})
            continue
        schema_failures = validate_schema(evidence, validator, relative)
        if schema_failures:
            errors.append({
                'event_id': event_id,
                'reason': 'recovery evidence schema validation failed',
                'schema_errors': schema_failures})
            continue
        expected = {
            'run_id': run_id,
            'recovery_event_id': event_id,
            'parent_event_id': event.get('parent_event_id'),
            'source_robot_id': event.get('source_robot_id'),
            'target_robot_id': event.get('target_robot_id'),
            'active_revision': event.get('map_revision_after'),
            'active_map_hash': event.get('map_hash_after'),
            'method': verification.get('method'),
            'verified_by': verification.get('verified_by'),
        }
        mismatches = {
            key: {'event': value, 'evidence': evidence.get(key)}
            for key, value in expected.items() if evidence.get(key) != value}
        observed_timestamp = evidence.get('observed_timestamp_ns')
        if mismatches:
            errors.append({
                'event_id': event_id,
                'reason': 'recovery evidence identity, pair, or map state differs',
                'mismatches': mismatches})
        if not isinstance(observed_timestamp, int) or isinstance(
                observed_timestamp, bool) or observed_timestamp <= int(
                    event.get('timestamp_ns', 0)):
            errors.append({
                'event_id': event_id,
                'reason': 'active map state was not observed after recovery'})
    return errors


def registration_metrics(events, references, limits, recovery_deadline_s):
    """Compute event-level transform, decision, and recovery metrics."""
    reference_by_id = {row['event_id']: row for row in references}
    states = Counter(str(event.get('state')) for event in events)
    rows = []
    decision_events = [
        event for event in events
        if event.get('state') in ('accepted', 'rejected', 'committed')]
    for event in decision_events:
        reference = reference_by_id.get(event.get('event_id'))
        row = {
            'event_id': event.get('event_id'),
            'state': event.get('state'),
            'source_robot_id': event.get('source_robot_id'),
            'target_robot_id': event.get('target_robot_id'),
            'translation_error_m': None,
            'yaw_error_deg': None,
            'within_limits': None,
            'map_revision_before': event.get('map_revision_before'),
            'map_revision_after': event.get('map_revision_after'),
            'map_hash_before': event.get('map_hash_before'),
            'map_hash_after': event.get('map_hash_after'),
        }
        if reference:
            estimate = event['estimate']
            translation_error = math.hypot(
                float(estimate['tx_m']) - float(reference['gt_tx_m']),
                float(estimate['ty_m']) - float(reference['gt_ty_m']))
            yaw_error = wrap_yaw_error(
                float(estimate['yaw_deg']) - float(reference['gt_yaw_deg']))
            row['translation_error_m'] = translation_error
            row['yaw_error_deg'] = yaw_error
            row['within_limits'] = (
                translation_error <= limits['translation_limit_m']
                and yaw_error <= limits['yaw_limit_deg'])
        rows.append(row)

    accepted = [row for row in rows if row['state'] == 'accepted']
    rejected = [row for row in rows if row['state'] == 'rejected']
    committed = [row for row in rows if row['state'] == 'committed']
    referenced_decisions = [
        row for row in accepted + rejected if row['within_limits'] is not None]
    errors_t = [
        row['translation_error_m'] for row in referenced_decisions
        if row['translation_error_m'] is not None]
    errors_y = [
        row['yaw_error_deg'] for row in referenced_decisions
        if row['yaw_error_deg'] is not None]
    false_accept = sum(row['within_limits'] is False for row in accepted)
    false_reject = sum(row['within_limits'] is True for row in rejected)
    wrong_commits = [row for row in committed if row['within_limits'] is False]
    correct_edges = {
        tuple(sorted((int(row['source_robot_id']), int(row['target_robot_id']))))
        for row in committed if row['within_limits'] is True
    }
    connected_ids = set()
    if correct_edges:
        adjacency = defaultdict(set)
        for source, target in correct_edges:
            adjacency[source].add(target)
            adjacency[target].add(source)
        frontier = [next(iter(adjacency))]
        while frontier:
            current = frontier.pop()
            if current in connected_ids:
                continue
            connected_ids.add(current)
            frontier.extend(adjacency[current] - connected_ids)

    recovered_events = [event for event in events if event.get('state') == 'recovered']
    recovered_wrong_commits = 0
    recovery_errors = []
    wrong_ids = {row['event_id'] for row in wrong_commits}
    recovery_parent_counts = Counter(
        event.get('parent_event_id') for event in recovered_events)
    for recovery in recovered_events:
        parent_id = recovery.get('parent_event_id')
        if parent_id not in wrong_ids:
            recovery_errors.append({
                'event_id': recovery.get('event_id'),
                'parent_event_id': parent_id,
                'reason': 'recovery parent is not an independently wrong commit'})
        if recovery_parent_counts[parent_id] > 1:
            recovery_errors.append({
                'event_id': recovery.get('event_id'),
                'parent_event_id': parent_id,
                'reason': 'duplicate recovery proof for one commit'})
    for wrong in wrong_commits:
        original = next(
            event for event in events
            if event.get('event_id') == wrong['event_id']
            and event.get('state') == 'committed')
        deadline_ns = int(recovery_deadline_s * 1e9)
        valid_recovery = False
        for event in recovered_events:
            if event.get('parent_event_id') != wrong['event_id']:
                continue
            verification = event.get('recovery_verification') or {}
            proof_valid = all((
                event.get('map_revision_before')
                == original.get('map_revision_after'),
                event.get('map_hash_before') == original.get('map_hash_after'),
                isinstance(event.get('map_revision_after'), int),
                event.get('map_revision_after', -1)
                > event.get('map_revision_before', -1),
                event.get('map_hash_after') != event.get('map_hash_before'),
                verification.get('invalidated_commit_event_id')
                == wrong['event_id'],
                verification.get('invalidated_revision')
                == original.get('map_revision_after'),
                verification.get('invalidated_map_hash')
                == original.get('map_hash_after'),
                verification.get('active_revision')
                == event.get('map_revision_after'),
                verification.get('active_map_hash')
                == event.get('map_hash_after'),
                isinstance(verification.get('method'), str),
                bool(verification.get('method', '').strip()),
                isinstance(verification.get('verified_by'), str),
                bool(verification.get('verified_by', '').strip()),
                (event.get('source_robot_id'), event.get('target_robot_id'))
                == (original.get('source_robot_id'),
                    original.get('target_robot_id')),
                0 <= int(event['timestamp_ns']) - int(original['timestamp_ns'])
                <= deadline_ns))
            if proof_valid:
                valid_recovery = True
                break
            recovery_errors.append({
                'event_id': event.get('event_id'),
                'parent_event_id': wrong['event_id'],
                'reason': 'recovery does not invalidate the wrong committed state'})
        if valid_recovery:
            recovered_wrong_commits += 1

    return {
        'state_counts': dict(states),
        'accepted': len(accepted),
        'rejected': len(rejected),
        'reference_coverage': (
            len(referenced_decisions) / (len(accepted) + len(rejected))
            if accepted or rejected else 0.0),
        'false_accept': false_accept,
        'false_reject': false_reject,
        'safe_reject': sum(row['within_limits'] is False for row in rejected),
        'correct_committed_pairs': [list(pair) for pair in sorted(correct_edges)],
        'connected_robot_ids': sorted(connected_ids),
        'wrong_commits': len(wrong_commits),
        'recovered_wrong_commits': recovered_wrong_commits,
        'unrecovered_wrong_commits': (
            len(wrong_commits) - recovered_wrong_commits),
        'recovery_errors': recovery_errors,
        'translation_error_median_m': statistics.median(errors_t) if errors_t else None,
        'translation_error_p95_m': percentile(errors_t, 0.95),
        'yaw_error_median_deg': statistics.median(errors_y) if errors_y else None,
        'yaw_error_p95_deg': percentile(errors_y, 0.95),
        'rows': rows,
    }


def percentile(values, fraction):
    """Return a linearly interpolated percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def coverage_metrics(events):
    """Derive final coverage, time to 80 percent, and normalised AUC."""
    ordered = sorted(events, key=lambda row: int(row['timestamp_ns']))
    if not ordered:
        return None, []
    start_ns = int(ordered[0]['timestamp_ns'])
    trace = []
    for event in ordered:
        trace.append({
            'time_s': (int(event['timestamp_ns']) - start_ns) / 1e9,
            'coverage': float(event['coverage']),
            'observed_free_m2': event.get('observed_free_m2'),
            'occupied_precision': event.get('occupied_precision'),
        })
    duration_s = trace[-1]['time_s']
    auc = 0.0
    for left, right in zip(trace, trace[1:]):
        auc += 0.5 * (left['coverage'] + right['coverage']) * (
            right['time_s'] - left['time_s'])
    t80 = next(
        (row['time_s'] for row in trace if row['coverage'] >= 0.8), None)
    metrics = {
        'samples': len(trace),
        'duration_s': duration_s,
        'final_coverage': trace[-1]['coverage'],
        't80_s': t80,
        'normalised_auc': auc / duration_s if duration_s > 0 else None,
        'largest_decrease': min(
            (right['coverage'] - left['coverage']
             for left, right in zip(trace, trace[1:])), default=0.0),
    }
    return metrics, trace


def sha256(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(run_dir, output_path):
    """Hash every captured input artifact outside the analysis directory."""
    rows = []
    for path in sorted(run_dir.rglob('*')):
        relative = path.relative_to(run_dir)
        if not path.is_file() or relative.parts[0] == 'analysis':
            continue
        rows.append((sha256(path), relative.as_posix()))
    output_path.write_text(
        ''.join(f'{digest}  {relative}\n' for digest, relative in rows),
        encoding='utf-8')
    return len(rows)


def add_check(checks, name, passed, detail):
    """Append one serialisable audit check."""
    checks.append({'name': name, 'passed': bool(passed), 'detail': detail})


def write_csv(path, rows, fieldnames):
    """Write a table with a stable column order."""
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument(
        '--skip-hash', action='store_true',
        help='diagnostic only; makes the integrity check fail')
    return parser.parse_args()


def main():
    """Audit one run and write metrics only when supported by evidence."""
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    analysis_dir = run_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    checks = []

    try:
        validators = build_schema_validators()
    except (OSError, json.JSONDecodeError, ValueError, SchemaError) as error:
        raise SystemExit(f'JSON schema setup failed: {error}') from error
    manifest = load_json(run_dir / 'manifest.json')
    manifest_schema_errors = validate_schema(
        manifest, validators['manifest'], run_dir / 'manifest.json')
    add_check(
        checks, 'manifest JSON schema', not manifest_schema_errors,
        {'errors': manifest_schema_errors,
         'validated_schema_count': len(validators)})
    config = yaml.safe_load(
        (run_dir / 'run_config.yaml').read_text(encoding='utf-8'))
    team_size = int(manifest['team_size'])
    expected_ids = set(range(team_size))
    add_check(
        checks, 'manifest and config agree',
        team_size == int(config['team_size'])
        and len(manifest['robots']) == team_size,
        {'manifest_team_size': team_size,
         'config_team_size': config['team_size']})
    config_digest = sha256(run_dir / 'run_config.yaml')
    add_check(
        checks, 'manifest integrity fields',
        manifest.get('config_sha256') == config_digest
        and bool(re.fullmatch(r'[0-9a-f]{40}', manifest.get('git_commit', '')))
        and bool(re.fullmatch(
            r'[0-9a-f]{64}', manifest.get('checkpoint_sha256', ''))),
        {'config_sha256': config_digest,
         'recorded_config_sha256': manifest.get('config_sha256')})
    lock_errors = frozen_lock_errors(manifest)
    add_check(
        checks, 'frozen protocol lock values', not lock_errors,
        {'errors': lock_errors})
    ground_truth = manifest.get('ground_truth_reference') or {}
    calibration_path = run_dir / str(ground_truth.get('calibration_path', ''))
    extrinsics_path = run_dir / str(ground_truth.get('extrinsics_path', ''))
    reference_file_errors = []
    reference_paths_inside_run = True
    for name, path in (
            ('calibration', calibration_path), ('extrinsics', extrinsics_path)):
        try:
            path.resolve().relative_to((run_dir / 'evidence').resolve())
            value = load_json(path)
            if not isinstance(value, dict) or not value:
                raise ValueError('reference file must contain a non-empty object')
        except (OSError, ValueError, json.JSONDecodeError) as error:
            reference_paths_inside_run = False
            reference_file_errors.append({'file': name, 'error': str(error)})
    reference_files_pass = bool(
        ground_truth.get('calibration_id')
        and ground_truth.get('extrinsics_id')
        and reference_paths_inside_run
        and not reference_file_errors
        and calibration_path.is_file() and extrinsics_path.is_file()
        and sha256(calibration_path) == ground_truth.get('calibration_sha256')
        and sha256(extrinsics_path) == ground_truth.get('extrinsics_sha256')
        and isinstance(ground_truth.get('uncertainty_translation_m'), (int, float))
        and ground_truth.get('uncertainty_translation_m') >= 0
        and isinstance(ground_truth.get('uncertainty_yaw_deg'), (int, float))
        and ground_truth.get('uncertainty_yaw_deg') >= 0)
    add_check(
        checks, 'reference calibration and extrinsics integrity',
        reference_files_pass,
        {'calibration_path': str(calibration_path),
         'extrinsics_path': str(extrinsics_path),
         'errors': reference_file_errors})
    manifest_robots = manifest.get('robots', [])
    add_check(
        checks, 'distinct physical hardware roster',
        {robot.get('id') for robot in manifest_robots} == expected_ids
        and len({robot.get('host') for robot in manifest_robots}) == team_size
        and len({robot.get('base_serial') for robot in manifest_robots})
        == team_size
        and len({robot.get('lidar_serial') for robot in manifest_robots})
        == team_size,
        {'robots': manifest_robots})

    preflight_path = run_dir / 'preflight.json'
    preflight = load_json(preflight_path) if preflight_path.exists() else None
    add_check(
        checks, 'preflight passed',
        bool(preflight and preflight.get('overall_pass')),
        str(preflight_path))

    db3_paths = sorted(run_dir.rglob('*.db3'))
    topic_stats, bag_errors = read_sqlite_topics(db3_paths)
    add_check(
        checks, 'readable SQLite rosbag', bool(db3_paths) and not bag_errors,
        {'files': [str(path.relative_to(run_dir)) for path in db3_paths],
         'errors': bag_errors})

    required = required_topics(config)
    zero_allowed = {'/mso/safety_event'}
    missing = [
        topic for topic in required
        if topic not in zero_allowed
        and topic_stats.get(topic, {}).get('count', 0) == 0]
    add_check(checks, 'required recorded topics', not missing, {'missing': missing})
    add_check(
        checks, 'physical time evidence',
        topic_stats.get('/clock', {}).get('count', 0) == 0,
        {'clock_messages': topic_stats.get('/clock', {}).get('count', 0)})

    discovered_ids = {
        int(match.group(1))
        for topic in topic_stats
        for match in [ROBOT_TOPIC.match(topic)] if match
    }
    add_check(
        checks, 'robot namespaces', discovered_ids == expected_ids,
        {'expected': sorted(expected_ids), 'discovered': sorted(discovered_ids)})

    analysis_config = config.get('analysis', {})
    min_sensor = int(analysis_config.get('min_sensor_messages', 100))
    min_pose = int(analysis_config.get('min_pose_messages', 100))
    min_map = int(analysis_config.get('min_map_messages', 10))
    message_failures = []
    sensor_spans = []
    reference_spans = []
    robot_rows = []
    for robot_id in sorted(expected_ids):
        lidar_topic = f'/robot_{robot_id}/lidar/points'
        odom_topic = f'/robot_{robot_id}/odom'
        map_topic = f'/robot_{robot_id}/map'
        reference_topic = f'/ground_truth/robot_{robot_id}/pose'
        lidar = topic_stats.get(lidar_topic, {})
        odom = topic_stats.get(odom_topic, {})
        map_stats = topic_stats.get(map_topic, {})
        reference_stats = topic_stats.get(reference_topic, {})
        if lidar.get('count', 0) < min_sensor:
            message_failures.append(lidar_topic)
        if odom.get('count', 0) < min_pose:
            message_failures.append(odom_topic)
        if map_stats.get('count', 0) < min_map:
            message_failures.append(map_topic)
        if reference_stats.get('count', 0) < min_pose:
            message_failures.append(reference_topic)
        if lidar.get('start_ns') is not None:
            sensor_spans.append((lidar['start_ns'], lidar['end_ns']))
        if reference_stats.get('start_ns') is not None:
            reference_spans.append((
                reference_stats['start_ns'], reference_stats['end_ns']))
        robot_rows.append({
            'robot_id': robot_id,
            'lidar_messages': lidar.get('count', 0),
            'lidar_duration_s': (
                (lidar['end_ns'] - lidar['start_ns']) / 1e9
                if lidar.get('start_ns') is not None else None),
            'odom_messages': odom.get('count', 0),
            'map_messages': map_stats.get('count', 0),
            'reference_pose_messages': reference_stats.get('count', 0),
        })
    add_check(
        checks, 'per-robot message minima', not message_failures,
        {'below_minimum': message_failures})

    common_duration_s = 0.0
    if len(sensor_spans) == team_size:
        common_duration_s = max(
            0.0, (min(end for _, end in sensor_spans)
                  - max(start for start, _ in sensor_spans)) / 1e9)
    minimum_common = float(
        analysis_config.get('min_common_duration_s', 60.0))
    add_check(
        checks, 'simultaneous physical sensor interval',
        common_duration_s >= minimum_common,
        {'common_duration_s': common_duration_s,
         'minimum_s': minimum_common})

    events_dir = run_dir / 'events'
    campaign = manifest.get('campaign')
    network_summary = None
    if campaign:
        network_logs = sorted(events_dir.glob('network_events*.jsonl'))
        network_rows = []
        network_parse_errors = []
        for path in network_logs:
            values, errors = load_jsonl(path)
            network_rows.extend(values)
            network_parse_errors.extend({
                **error, 'path': str(path)} for error in errors)
        network_schema_errors = validate_rows(
            network_rows, validators['network'], 'network event')
        condition = campaign.get('condition_id')
        profile = campaign.get('network_profile')
        if condition == 'impaired':
            network_audit_path = analysis_dir / 'network_audit.json'
            network_audit = (
                load_json(network_audit_path)
                if network_audit_path.exists() else None)
            declared_probe_logs = (
                network_audit.get('probe_logs', []) if network_audit else [])
            probe_logs_valid = bool(declared_probe_logs)
            probe_parse_errors = []
            probe_schema_errors = []
            probe_rows = []
            for declared in declared_probe_logs:
                try:
                    declared_path = Path(declared).expanduser()
                    if not declared_path.is_absolute():
                        declared_path = run_dir / declared_path
                    declared_path = declared_path.resolve()
                    declared_path.relative_to(events_dir.resolve())
                    if not declared_path.is_file():
                        probe_logs_valid = False
                        continue
                    values, row_errors = load_jsonl(declared_path)
                    probe_rows.extend(values)
                    probe_parse_errors.extend({
                        **error, 'path': str(declared_path)}
                        for error in row_errors)
                    probe_schema_errors.extend(validate_rows(
                        values, validators['traffic_probe'], declared_path))
                except (OSError, ValueError):
                    probe_logs_valid = False
            if probe_parse_errors or probe_schema_errors or any(
                    row.get('run_id') != manifest.get('run_id')
                    for row in probe_rows):
                probe_logs_valid = False
            common_start_ns = (
                max(start for start, _ in sensor_spans)
                if len(sensor_spans) == team_size else None)
            common_end_ns = (
                min(end for _, end in sensor_spans)
                if len(sensor_spans) == team_size else None)
            timed_events = [
                row for row in network_rows
                if row.get('event') in ('phase_started', 'phase_completed')
                and isinstance(row.get('timestamp_ns'), int)]
            inside_capture = bool(
                timed_events and common_start_ns is not None
                and common_end_ns is not None
                and min(row['timestamp_ns'] for row in timed_events)
                >= common_start_ns
                and max(row['timestamp_ns'] for row in timed_events)
                <= common_end_ns)
            network_pass = bool(
                profile and profile != 'none'
                and network_logs and not network_parse_errors
                and not network_schema_errors
                and network_audit
                and network_audit.get('run_id') == manifest['run_id']
                and network_audit.get('profile_id') == profile
                and network_audit.get('profile_sha256')
                == campaign.get('network_profile_sha256')
                and network_audit.get('physical_impairment_evidence') is True
                and network_audit.get('rule_execution_pass') is True
                and network_audit.get('traffic_measurement_pass') is True
                and probe_logs_valid
                and inside_capture)
            network_summary = {
                'condition_id': condition,
                'profile_id': profile,
                'profile_sha256': (
                    network_audit.get('profile_sha256')
                    if network_audit else None),
                'log_files': [str(path.relative_to(run_dir))
                              for path in network_logs],
                'event_count': len(network_rows),
                'probe_logs': declared_probe_logs,
                'probe_logs_inside_run': probe_logs_valid,
                'parse_errors': network_parse_errors,
                'schema_errors': network_schema_errors,
                'probe_parse_errors': probe_parse_errors,
                'probe_schema_errors': probe_schema_errors,
                'inside_common_sensor_interval': inside_capture,
                'physical_impairment_evidence': bool(
                    network_audit and network_audit.get(
                        'physical_impairment_evidence')),
                'rule_execution_pass': bool(
                    network_audit and network_audit.get(
                        'rule_execution_pass')),
                'traffic_measurement_pass': bool(
                    network_audit and network_audit.get(
                        'traffic_measurement_pass')),
                'recovery_start_ns': max(
                    (row['timestamp_ns'] for row in network_rows
                     if row.get('event') == 'phase_started'
                     and row.get('phase_id') == 'recovery'
                     and isinstance(row.get('timestamp_ns'), int)),
                    default=None),
            }
            add_check(
                checks, 'executed communication impairment', network_pass,
                network_summary)
        else:
            executed = [
                row for row in network_rows if row.get('mode') == 'execute']
            network_pass = (
                condition == 'reference' and profile == 'none'
                and not network_parse_errors and not network_schema_errors
                and not executed)
            network_summary = {
                'condition_id': condition,
                'profile_id': profile,
                'event_count': len(network_rows),
                'executed_event_count': len(executed),
                'parse_errors': network_parse_errors,
                'schema_errors': network_schema_errors,
            }
            add_check(
                checks, 'unimpaired reference condition', network_pass,
                network_summary)
    reference_common_s = 0.0
    if len(reference_spans) == team_size:
        reference_common_s = max(
            0.0, (min(end for _, end in reference_spans)
                  - max(start for start, _ in reference_spans)) / 1e9)
    add_check(
        checks, 'simultaneous independent pose interval',
        reference_common_s >= minimum_common,
        {'common_duration_s': reference_common_s,
         'minimum_s': minimum_common})

    evidence_missing = [
        relative for relative in config.get('required_evidence', [])
        if not (run_dir / relative).is_file()
        or (run_dir / relative).stat().st_size == 0]
    add_check(
        checks, 'physical evidence files', not evidence_missing,
        {'missing_or_empty': evidence_missing})

    registration, registration_errors = load_jsonl(
        events_dir / 'registration_events.jsonl')
    safety, safety_errors = load_jsonl(events_dir / 'safety_events.jsonl')
    coverage, coverage_errors = load_jsonl(events_dir / 'coverage_events.jsonl')
    invalid, invalid_errors = load_jsonl(events_dir / 'invalid_events.jsonl')
    registration_schema_errors = validate_rows(
        registration, validators['registration'],
        events_dir / 'registration_events.jsonl')
    safety_schema_errors = validate_rows(
        safety, validators['safety'], events_dir / 'safety_events.jsonl')
    coverage_schema_errors = validate_rows(
        coverage, validators['coverage'], events_dir / 'coverage_events.jsonl')
    event_schema_errors = (
        registration_schema_errors + safety_schema_errors
        + coverage_schema_errors)
    event_identity_errors = [
        {'event_id': row.get('event_id'), 'reason': 'event run_id mismatch'}
        for row in registration + safety + coverage
        if row.get('run_id') != manifest.get('run_id')]
    parse_errors = (
        registration_errors + safety_errors + coverage_errors + invalid_errors)
    add_check(
        checks, 'event log validity',
        not parse_errors and not invalid and not event_schema_errors
        and not event_identity_errors,
        {'parse_errors': parse_errors,
         'schema_errors': event_schema_errors,
         'identity_errors': event_identity_errors,
         'invalid_event_count': len(invalid)})
    if registration_errors or registration_schema_errors:
        registration = []
    if safety_errors or safety_schema_errors:
        safety = []
    if coverage_errors or coverage_schema_errors:
        coverage = []

    if campaign and campaign.get('condition_id') == 'impaired':
        recovery_start_ns = (
            network_summary.get('recovery_start_ns')
            if network_summary else None)
        if recovery_start_ns is None:
            post_recovery_map_messages = 0
            map_query_errors = []
            post_recovery_coverage = []
            post_recovery_registration = []
            post_recovery_safety = []
        else:
            post_recovery_map_messages, map_query_errors = (
                count_topic_messages_after(
                    db3_paths, '/merged_map', recovery_start_ns))
            post_recovery_coverage = [
                row for row in coverage
                if int(row.get('timestamp_ns', 0)) >= recovery_start_ns]
            post_recovery_registration = [
                row for row in registration
                if int(row.get('timestamp_ns', 0)) >= recovery_start_ns]
            post_recovery_safety = [
                row for row in safety
                if int(row.get('timestamp_ns', 0)) >= recovery_start_ns]
        recovery_observation = {
            'recovery_start_ns': recovery_start_ns,
            'post_recovery_merged_map_messages': post_recovery_map_messages,
            'post_recovery_coverage_samples': len(post_recovery_coverage),
            'post_recovery_registration_events': len(
                post_recovery_registration),
            'post_recovery_safety_events': len(post_recovery_safety),
            'map_query_errors': map_query_errors,
            'coverage_at_recovery': (
                float(post_recovery_coverage[0]['coverage'])
                if post_recovery_coverage else None),
            'coverage_at_run_end': (
                float(post_recovery_coverage[-1]['coverage'])
                if post_recovery_coverage else None),
        }
        recovery_observed = bool(
            recovery_start_ns is not None and not map_query_errors
            and post_recovery_map_messages >= 3
            and len(post_recovery_coverage) >= 3)
        network_summary['recovery_observation'] = recovery_observation
        add_check(
            checks, 'post-impairment recovery observation',
            recovery_observed, recovery_observation)
    lifecycle_errors = registration_lifecycle_errors(registration)
    add_check(
        checks, 'registration event lifecycle', not lifecycle_errors,
        {'errors': lifecycle_errors})
    recovery_file_errors = recovery_evidence_errors(
        registration, run_dir, manifest.get('run_id'),
        validators['recovery_evidence'])
    add_check(
        checks, 'post-recovery map-state evidence', not recovery_file_errors,
        {'errors': recovery_file_errors})

    references, reference_errors = load_jsonl(
        run_dir / 'evidence' / 'reference_transforms.jsonl')
    reference_schema_errors = validate_rows(
        references, validators['reference'],
        run_dir / 'evidence' / 'reference_transforms.jsonl')
    if reference_errors or reference_schema_errors:
        references = []
    reference_ids, duplicate_reference_ids, empty_reference_ids = (
        reference_identifier_errors(references))
    registration_by_id = defaultdict(list)
    for event in registration:
        registration_by_id[event.get('event_id')].append(event)
    decision_ids = {
        event.get('event_id') for event in registration
        if event.get('state') in ('accepted', 'rejected')}
    reference_metadata_errors = []
    for reference in references:
        event_id = reference.get('event_id')
        matching = registration_by_id.get(event_id, [])
        pairs = {
            (event.get('source_robot_id'), event.get('target_robot_id'))
            for event in matching}
        reference_pair = (
            reference.get('source_robot_id'), reference.get('target_robot_id'))
        metadata_matches = all((
            reference.get('source') == manifest.get('ground_truth_source'),
            reference.get('calibration_id')
            == ground_truth.get('calibration_id'),
            reference.get('calibration_sha256')
            == ground_truth.get('calibration_sha256'),
            reference.get('extrinsics_id') == ground_truth.get('extrinsics_id'),
            reference.get('extrinsics_sha256')
            == ground_truth.get('extrinsics_sha256'),
            reference.get('uncertainty_translation_m')
            == ground_truth.get('uncertainty_translation_m'),
            reference.get('uncertainty_yaw_deg')
            == ground_truth.get('uncertainty_yaw_deg'),
            reference_pair in pairs))
        if not metadata_matches:
            reference_metadata_errors.append({
                'event_id': event_id,
                'reason': 'reference metadata/pair differs from frozen source'})
    tolerance_ms = (manifest.get('protocol_lock') or {}).get(
        'reference_timestamp_tolerance_ms')
    tolerance_ns = (
        int(tolerance_ms * 1e6)
        if isinstance(tolerance_ms, (int, float))
        and not isinstance(tolerance_ms, bool)
        and math.isfinite(tolerance_ms) and tolerance_ms > 0 else 0)
    reference_time_errors = reference_timestamp_errors(
        references, registration_by_id, tolerance_ns)
    reference_identity_pass = bool(
        not reference_errors and not reference_schema_errors
        and not duplicate_reference_ids
        and empty_reference_ids == 0
        and set(reference_ids) == decision_ids
        and not reference_metadata_errors and not reference_time_errors)
    limits = {
        'translation_limit_m': float(
            analysis_config.get('translation_limit_m', 0.15)),
        'yaw_limit_deg': float(analysis_config.get('yaw_limit_deg', 5.0)),
    }
    registration_summary = registration_metrics(
        registration, references, limits,
        float(analysis_config.get('recovery_deadline_s', 5.0)))
    add_check(
        checks, 'independent registration reference',
        reference_identity_pass
        and registration_summary['accepted'] + registration_summary['rejected'] > 0
        and registration_summary['reference_coverage'] == 1.0,
        {'reference_parse_errors': reference_errors,
         'reference_schema_errors': reference_schema_errors,
         'duplicate_reference_ids': duplicate_reference_ids,
         'empty_reference_ids': empty_reference_ids,
         'metadata_errors': reference_metadata_errors,
         'timestamp_tolerance_ms': tolerance_ms,
         'timestamp_errors': reference_time_errors,
         'decision_count': registration_summary['accepted']
         + registration_summary['rejected'],
         'reference_coverage': registration_summary['reference_coverage']})
    add_check(
        checks, 'registration graph connects the physical fleet',
        set(registration_summary['connected_robot_ids']) == expected_ids,
        {'connected_robot_ids': registration_summary['connected_robot_ids'],
         'correct_committed_pairs': registration_summary[
             'correct_committed_pairs']})
    add_check(
        checks, 'wrong-commit recovery',
        registration_summary['wrong_commits']
        == registration_summary['recovered_wrong_commits']
        and not registration_summary['recovery_errors'],
        {'wrong_commits': registration_summary['wrong_commits'],
         'recovered': registration_summary['recovered_wrong_commits'],
         'errors': registration_summary['recovery_errors']})

    coverage_summary, coverage_trace = coverage_metrics(coverage)
    coverage_valid = bool(
        coverage_summary
        and coverage_summary['samples'] >= 10
        and coverage_summary['duration_s'] >= minimum_common
        and coverage_summary['largest_decrease'] >= -0.01)
    add_check(
        checks, 'independent coverage trace', coverage_valid,
        coverage_summary or {'samples': 0})

    if args.skip_hash:
        checksum_count = 0
    else:
        checksum_count = write_checksums(
            run_dir, analysis_dir / 'checksums.sha256')
    add_check(
        checks, 'artifact integrity hashes', checksum_count > 0,
        {'hashed_files': checksum_count, 'skipped': args.skip_hash})

    write_csv(
        analysis_dir / 'robot_topic_summary.csv', robot_rows,
        ('robot_id', 'lidar_messages', 'lidar_duration_s', 'odom_messages',
         'map_messages', 'reference_pose_messages'))
    write_csv(
        analysis_dir / 'registration_metrics.csv',
        registration_summary.pop('rows'),
        ('event_id', 'state', 'source_robot_id', 'target_robot_id',
         'translation_error_m', 'yaw_error_deg', 'within_limits',
         'map_revision_before', 'map_revision_after', 'map_hash_before',
         'map_hash_after'))
    write_csv(
        analysis_dir / 'coverage_trace.csv', coverage_trace,
        ('time_s', 'coverage', 'observed_free_m2', 'occupied_precision'))

    protocol_diagnostic_pass = all(item['passed'] for item in checks)
    report = {
        'schema_version': '1.0',
        'run_id': manifest['run_id'],
        'method': manifest['method'],
        'team_size': team_size,
        'site_id': manifest['site_id'],
        'arena_id': manifest['arena_id'],
        'trial_id': manifest['trial_id'],
        'checks': checks,
        'registration': registration_summary,
        'coverage': coverage_summary,
        'safety_event_counts': dict(Counter(
            row.get('event_type', 'unknown') for row in safety)),
        'common_sensor_duration_s': common_duration_s,
        'bag_start_ns': (
            min(start for start, _ in sensor_spans)
            if sensor_spans else None),
        'bag_end_ns': (
            max(end for _, end in sensor_spans)
            if sensor_spans else None),
        'common_reference_pose_duration_s': reference_common_s,
        'campaign': campaign,
        'protocol_lock': manifest.get('protocol_lock'),
        'network': network_summary,
        'evidence_class': (
            'physical' if protocol_diagnostic_pass
            else 'unverified_physical_candidate'),
        'protocol_diagnostic_pass': protocol_diagnostic_pass,
        'claim_authorized': False,
        'claim_authorization_note': (
            'This audit is a protocol diagnostic only. It does not authorize '
            'a manuscript claim; independent evidence review is required.'),
    }
    output = analysis_dir / 'run_audit.json'
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['protocol_diagnostic_pass'] else 2


if __name__ == '__main__':
    sys.exit(main())
