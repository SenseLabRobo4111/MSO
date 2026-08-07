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


ROBOT_TOPIC = re.compile(r'^/robot_(\d+)/')


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


def wrap_yaw_error(degrees):
    """Return the absolute wrapped yaw error in degrees."""
    return abs((degrees + 180.0) % 360.0 - 180.0)


def registration_lifecycle_errors(events):
    """Return invalid registration state transitions."""
    errors = []
    grouped = defaultdict(list)
    for event in events:
        grouped[event.get('event_id')].append(event)
    for event_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row.get('timestamp_ns', 0)))
        states = [row.get('state') for row in ordered]
        if states == ['recovered'] and ordered[0].get('parent_event_id'):
            continue
        if not states or states[0] != 'candidate':
            errors.append({'event_id': event_id, 'reason': 'missing candidate'})
        if len(states) != len(set(states)):
            errors.append({'event_id': event_id, 'reason': 'duplicate state'})
        if 'accepted' in states and 'rejected' in states:
            errors.append({'event_id': event_id, 'reason': 'two decisions'})
        if 'committed' in states and (
                'accepted' not in states
                or states.index('committed') < states.index('accepted')):
            errors.append({
                'event_id': event_id,
                'reason': 'commit without a preceding acceptance'})
        if 'rejected' in states and 'committed' in states:
            errors.append({'event_id': event_id, 'reason': 'rejected commit'})
        pairs = {
            (row.get('source_robot_id'), row.get('target_robot_id'))
            for row in ordered
        }
        if len(pairs) != 1:
            errors.append({'event_id': event_id, 'reason': 'robot pair changed'})
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
        for row in accepted if row['within_limits'] is True
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
    for wrong in wrong_commits:
        original = next(
            event for event in events
            if event.get('event_id') == wrong['event_id']
            and event.get('state') == 'committed')
        deadline_ns = int(recovery_deadline_s * 1e9)
        if any(
            event.get('parent_event_id') == wrong['event_id']
            and 0 <= int(event['timestamp_ns']) - int(original['timestamp_ns'])
            <= deadline_ns
            for event in recovered_events
        ):
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
        'correct_accepted_pairs': [list(pair) for pair in sorted(correct_edges)],
        'connected_robot_ids': sorted(connected_ids),
        'wrong_commits': len(wrong_commits),
        'recovered_wrong_commits': recovered_wrong_commits,
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
    """Hash every immutable input artifact outside the analysis directory."""
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

    manifest = load_json(run_dir / 'manifest.json')
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

    events_dir = run_dir / 'events'
    registration, registration_errors = load_jsonl(
        events_dir / 'registration_events.jsonl')
    safety, safety_errors = load_jsonl(events_dir / 'safety_events.jsonl')
    coverage, coverage_errors = load_jsonl(events_dir / 'coverage_events.jsonl')
    invalid, invalid_errors = load_jsonl(events_dir / 'invalid_events.jsonl')
    parse_errors = (
        registration_errors + safety_errors + coverage_errors + invalid_errors)
    add_check(
        checks, 'event log validity', not parse_errors and not invalid,
        {'parse_errors': parse_errors, 'invalid_event_count': len(invalid)})
    lifecycle_errors = registration_lifecycle_errors(registration)
    add_check(
        checks, 'registration event lifecycle', not lifecycle_errors,
        {'errors': lifecycle_errors})

    references, reference_errors = load_jsonl(
        run_dir / 'evidence' / 'reference_transforms.jsonl')
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
        not reference_errors
        and registration_summary['accepted'] + registration_summary['rejected'] > 0
        and registration_summary['reference_coverage'] == 1.0,
        {'reference_parse_errors': reference_errors,
         'decision_count': registration_summary['accepted']
         + registration_summary['rejected'],
         'reference_coverage': registration_summary['reference_coverage']})
    add_check(
        checks, 'registration graph connects the physical fleet',
        set(registration_summary['connected_robot_ids']) == expected_ids,
        {'connected_robot_ids': registration_summary['connected_robot_ids'],
         'correct_accepted_pairs': registration_summary[
             'correct_accepted_pairs']})
    add_check(
        checks, 'wrong-commit recovery',
        registration_summary['wrong_commits']
        == registration_summary['recovered_wrong_commits'],
        {'wrong_commits': registration_summary['wrong_commits'],
         'recovered': registration_summary['recovered_wrong_commits']})

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
         'translation_error_m', 'yaw_error_deg', 'within_limits'))
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
        'common_reference_pose_duration_s': reference_common_s,
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
