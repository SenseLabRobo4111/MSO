#!/usr/bin/env python3
"""Audit rule execution and measured topic traffic as separate evidence."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

try:
    import yaml
except ImportError as error:
    raise SystemExit('PyYAML is required: install python3-yaml') from error


NETWORK_REQUIRED = {
    'schema_version', 'run_id', 'session_id', 'robot_id', 'peer_robot_id',
    'peer_ip', 'interface', 'profile_id', 'rule_scope', 'loss_definition',
    'mode', 'event', 'timestamp_ns',
}
PROBE_REQUIRED = {
    'schema_version', 'run_id', 'session_id', 'event', 'timestamp_ns',
    'source_robot_id', 'target_robot_id', 'topic', 'clock_basis',
}


def sha256(path):
    """Return the SHA-256 digest of the exact frozen profile file."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, fraction):
    """Return a linearly interpolated percentile or null."""
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


def load_profile(path):
    """Load the frozen per-direction OUTPUT-only impairment profile."""
    profile = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(profile, dict) or profile.get('schema_version') != '1.0':
        raise ValueError('invalid profile schema')
    if profile.get('transport') != 'udp':
        raise ValueError('profile transport must be udp')
    if profile.get('rule_scope') != 'output_only':
        raise ValueError('profile rule_scope must be output_only')
    if profile.get('loss_definition') != 'per_direction':
        raise ValueError('profile loss must be defined per direction')
    phases = profile.get('phases')
    if not isinstance(phases, list) or len(phases) < 4:
        raise ValueError('profile has too few phases')
    cleaned = []
    identifiers = set()
    for row in phases:
        cleaned_row = {
            'phase_id': str(row['phase_id']),
            'duration_s': float(row['duration_s']),
            'loss_probability': float(row['loss_probability']),
        }
        if cleaned_row['phase_id'] in identifiers:
            raise ValueError('profile has duplicate phase identifiers')
        identifiers.add(cleaned_row['phase_id'])
        if cleaned_row['duration_s'] <= 0 or not (
                0.0 <= cleaned_row['loss_probability'] <= 1.0):
            raise ValueError('profile has an invalid duration or loss value')
        cleaned.append(cleaned_row)
    if cleaned[0] != {
            'phase_id': 'baseline', 'duration_s': 30.0,
            'loss_probability': 0.0}:
        raise ValueError('profile baseline must be 30 s without loss')
    degraded = [row for row in cleaned if row['phase_id'] == 'degraded']
    if len(degraded) != 1 or degraded[0]['loss_probability'] != 0.2:
        raise ValueError('degraded loss must be 20% per direction')
    disconnected = [
        row for row in cleaned if row['phase_id'] == 'disconnected']
    if len(disconnected) != 1 or disconnected[0][
            'loss_probability'] != 1.0:
        raise ValueError('profile must contain one complete disconnection')
    if cleaned[-1]['phase_id'] != 'recovery' or cleaned[-1][
            'loss_probability'] != 0.0:
        raise ValueError('profile must end with an unimpaired recovery')
    probe = profile.get('probe') or {}
    probe_config = {
        'rate_hz': float(probe.get('rate_hz', 0)),
        'payload_bytes': int(probe.get('payload_bytes', 0)),
        'minimum_send_fraction': float(
            probe.get('minimum_send_fraction', 0)),
    }
    if probe_config['rate_hz'] <= 0 or probe_config['payload_bytes'] < 64 or not (
            0 < probe_config['minimum_send_fraction'] <= 1):
        raise ValueError('profile probe configuration is invalid')
    tolerances = profile.get('measurement_tolerances')
    expected_phase_ids = {row['phase_id'] for row in cleaned}
    if not isinstance(tolerances, dict) or set(tolerances) != expected_phase_ids:
        raise ValueError('profile must lock one measurement tolerance per phase')
    cleaned_tolerances = {}
    for phase_id, value in tolerances.items():
        if not isinstance(value, dict) or set(value) != {
                'minimum_loss_fraction', 'maximum_loss_fraction'}:
            raise ValueError(f'{phase_id}: invalid measurement tolerance fields')
        minimum = float(value['minimum_loss_fraction'])
        maximum = float(value['maximum_loss_fraction'])
        if not all(map(math.isfinite, (minimum, maximum))) or not (
                0.0 <= minimum <= maximum <= 1.0):
            raise ValueError(f'{phase_id}: invalid measured-loss tolerance')
        cleaned_tolerances[phase_id] = {
            'minimum_loss_fraction': minimum,
            'maximum_loss_fraction': maximum,
        }
    if cleaned_tolerances['baseline']['minimum_loss_fraction'] != 0.0 or (
            cleaned_tolerances['baseline']['maximum_loss_fraction'] > 0.05):
        raise ValueError('baseline tolerance must require low measured loss')
    if cleaned_tolerances['recovery']['minimum_loss_fraction'] != 0.0 or (
            cleaned_tolerances['recovery']['maximum_loss_fraction'] > 0.05):
        raise ValueError('recovery tolerance must require low measured loss')
    degraded_tolerance = cleaned_tolerances['degraded']
    if not (
            0.10 <= degraded_tolerance['minimum_loss_fraction'] <= 0.20
            <= degraded_tolerance['maximum_loss_fraction'] <= 0.30):
        raise ValueError('degraded tolerance must be compatible with 20% loss')
    if cleaned_tolerances['disconnected']['minimum_loss_fraction'] < 0.95 or (
            cleaned_tolerances['disconnected']['maximum_loss_fraction'] != 1.0):
        raise ValueError('disconnected tolerance must require near-full loss')
    return {
        'profile_id': str(profile['profile_id']),
        'phases': cleaned,
        'probe': probe_config,
        'measurement_tolerances': cleaned_tolerances,
    }


def load_jsonl(paths, required):
    """Read JSONL and attach exact source locations."""
    rows = []
    errors = []
    for path in paths:
        for line_number, line in enumerate(
                path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('row is not an object')
                missing = sorted(required - set(row))
                if missing:
                    raise ValueError(f'missing fields: {missing}')
                row['_source'] = f'{path}:{line_number}'
                rows.append(row)
            except (json.JSONDecodeError, ValueError) as error:
                errors.append({
                    'source': f'{path}:{line_number}', 'error': str(error)})
    return rows, errors


def command_is_one_output_rule(command, row, loss_probability):
    """Accept only the exact peer-targeted OUTPUT rule for this endpoint."""
    if not isinstance(command, list) or not all(
            isinstance(token, str) for token in command):
        return False
    prefix = [
        'iptables', '-w', '5', '-I', 'OUTPUT', '1',
        '-o', row.get('interface'), '-d', row.get('peer_ip'), '-p', 'udp']
    if 0.0 < loss_probability < 1.0:
        prefix.extend([
            '-m', 'statistic', '--mode', 'random', '--probability',
            f'{loss_probability:.6f}'])
    prefix.extend(['-m', 'comment', '--comment'])
    comment = (
        f'mso-{row.get("run_id")}-r{row.get("robot_id")}'
        f'-p{row.get("peer_robot_id")}')[:240]
    return command == [*prefix, comment, '-j', 'DROP']


def check_session(rows, profile, duration_tolerance_s):
    """Validate one endpoint's scheduled and executed rule lifecycle."""
    errors = []
    ordered = sorted(rows, key=lambda row: int(row['timestamp_ns']))
    modes = {row.get('mode') for row in ordered}
    robot_ids = {row.get('robot_id') for row in ordered}
    peers = {row.get('peer_robot_id') for row in ordered}
    scopes = {row.get('rule_scope') for row in ordered}
    definitions = {row.get('loss_definition') for row in ordered}
    if len(modes) != 1:
        errors.append('mode changed within session')
    if len(robot_ids) != 1 or len(peers) != 1:
        errors.append('robot identity changed within session')
    if scopes != {'output_only'} or definitions != {'per_direction'}:
        errors.append('rule scope or loss definition changed')
    lifecycle = (
        'session_started', 'cleanup_started', 'cleanup_completed',
        'session_completed')
    for event in lifecycle:
        if sum(row.get('event') == event for row in ordered) != 1:
            errors.append(f'{event} must occur exactly once')
    if any(row.get('event') == 'command_failed' for row in ordered):
        errors.append('command_failed was recorded')
    completed = [
        row for row in ordered if row.get('event') == 'session_completed']
    if completed and completed[0].get('success') is not True:
        errors.append('session did not complete successfully')
    cleanup = [row for row in ordered if row.get('event') == 'cleanup_completed']
    if cleanup and cleanup[0].get('cleanup_ok') is not True:
        errors.append('rule cleanup did not complete successfully')

    phases = profile['phases']
    expected_ids = [phase['phase_id'] for phase in phases]
    expected_events = ['session_started']
    for _phase in phases:
        expected_events.extend(['phase_started', 'phase_completed'])
    expected_events.extend([
        'cleanup_started', 'cleanup_completed', 'session_completed'])
    if [row.get('event') for row in ordered] != expected_events:
        errors.append('session event order or cardinality differs from profile')
    started = [row for row in ordered if row.get('event') == 'phase_started']
    ended = [row for row in ordered if row.get('event') == 'phase_completed']
    if [row.get('phase_id') for row in started] != expected_ids:
        errors.append('phase-start order does not match profile')
    if [row.get('phase_id') for row in ended] != expected_ids:
        errors.append('phase-completion order does not match profile')
    starts = {row.get('phase_id'): row for row in started}
    ends = {row.get('phase_id'): row for row in ended}
    mode = next(iter(modes)) if len(modes) == 1 else None
    first_start = starts.get(phases[0]['phase_id'], {}).get(
        'scheduled_timestamp_ns')
    previous_loss = 0.0
    counters = {}
    cumulative_s = 0.0
    for phase in phases:
        phase_id = phase['phase_id']
        start = starts.get(phase_id)
        end = ends.get(phase_id)
        if not start or not end:
            continue
        expected_start = (
            first_start + int(cumulative_s * 1e9)
            if isinstance(first_start, int) else None)
        cumulative_s += phase['duration_s']
        expected_end = (
            first_start + int(cumulative_s * 1e9)
            if isinstance(first_start, int) else None)
        if start.get('scheduled_timestamp_ns') != expected_start or end.get(
                'scheduled_timestamp_ns') != expected_end:
            errors.append(f'{phase_id}: scheduled duration/order differs')
        if float(start.get('duration_s', -1)) != phase['duration_s'] or float(
                end.get('duration_s', -1)) != phase['duration_s']:
            errors.append(f'{phase_id}: duration field differs from profile')
        if float(start.get('loss_probability', -1)) != phase[
                'loss_probability'] or float(end.get(
                    'loss_probability', -1)) != phase['loss_probability']:
            errors.append(f'{phase_id}: loss differs from profile')
        actual_duration_s = (
            (int(end['timestamp_ns']) - int(start['timestamp_ns'])) / 1e9)
        if int(end['timestamp_ns']) <= int(start['timestamp_ns']):
            errors.append(f'{phase_id}: phase completion precedes start')
        if mode == 'execute' and abs(
                actual_duration_s - phase['duration_s']) > duration_tolerance_s:
            errors.append(f'{phase_id}: executed duration exceeds tolerance')

        expected_rules = 0 if phase['loss_probability'] == 0 else 1
        planned = start.get('planned_commands')
        if not isinstance(planned, list) or len(planned) != expected_rules:
            errors.append(f'{phase_id}: expected one endpoint OUTPUT rule')
        elif any(not command_is_one_output_rule(
                     command, start, phase['loss_probability'])
                 for command in planned):
            errors.append(f'{phase_id}: rule is duplicated or too broad')
        if int(start.get('planned_rule_count', -1)) != expected_rules:
            errors.append(f'{phase_id}: planned rule count differs')

        command_results = start.get('command_results', [])
        if any(result.get('returncode') != 0 for result in command_results):
            errors.append(f'{phase_id}: rule command returned non-zero')
        if mode == 'execute':
            expected_results = (
                (1 if previous_loss > 0 else 0)
                + (1 if phase['loss_probability'] > 0 else 0))
            if len(command_results) != expected_results:
                errors.append(f'{phase_id}: executed command count differs')

        snapshot = end.get('counter_snapshot')
        counter_packets = None
        counter_bytes = None
        if not isinstance(snapshot, dict):
            errors.append(f'{phase_id}: counter snapshot is missing')
        else:
            rules = snapshot.get('rules')
            if not isinstance(rules, list):
                errors.append(f'{phase_id}: counter rules are malformed')
                rules = []
            if mode == 'execute':
                if snapshot.get('dry_run') is not False or snapshot.get(
                        'returncode') != 0 or len(rules) != expected_rules:
                    errors.append(f'{phase_id}: live counter snapshot failed')
                for rule in rules:
                    text = str(rule.get('rule', ''))
                    if '-A OUTPUT' not in text or '-A INPUT' in text:
                        errors.append(f'{phase_id}: counter is not OUTPUT-only')
                counter_packets = sum(int(rule.get('packets', 0)) for rule in rules)
                counter_bytes = sum(int(rule.get('bytes', 0)) for rule in rules)
            elif snapshot.get('dry_run') is not True:
                errors.append(f'{phase_id}: dry-run counter label is false')
        counters[phase_id] = {
            'packets': counter_packets,
            'bytes': counter_bytes,
            'expected_rule_count': expected_rules,
        }
        previous_loss = phase['loss_probability']

    return {
        'session_id': ordered[0].get('session_id') if ordered else None,
        'robot_id': next(iter(robot_ids)) if len(robot_ids) == 1 else None,
        'peer_robot_id': next(iter(peers)) if len(peers) == 1 else None,
        'mode': next(iter(modes)) if len(modes) == 1 else None,
        'errors': errors,
        'counters': counters,
        'phase_starts': {
            row['phase_id']: {
                'scheduled_timestamp_ns': row.get('scheduled_timestamp_ns'),
                'actual_timestamp_ns': row.get('timestamp_ns'),
            } for row in started
        },
    }


def probe_metrics(rows, profile, expected_robot_ids, phase_start_ns):
    """Measure per-direction topic throughput, loss, and one-way delay."""
    errors = []
    invalid = [row for row in rows if row.get('event') == 'invalid_received']
    if invalid:
        errors.append('probe recorded invalid received payloads')
    starts = [row for row in rows if row.get('event') == 'session_started']
    completions = [
        row for row in rows if row.get('event') == 'session_completed']
    for robot_id in expected_robot_ids:
        robot_starts = [
            row for row in starts if row.get('source_robot_id') == robot_id]
        if len(robot_starts) != 1:
            errors.append(f'robot {robot_id}: probe session start is missing')
        else:
            total_duration = sum(
                phase['duration_s'] for phase in profile['phases'])
            if float(robot_starts[0].get('rate_hz', -1)) != profile[
                    'probe']['rate_hz'] or float(robot_starts[0].get(
                        'duration_s', -1)) != total_duration:
                errors.append(f'robot {robot_id}: probe rate/duration differs')
            if int(robot_starts[0].get('timestamp_ns', 0)) > phase_start_ns:
                errors.append(f'robot {robot_id}: probe started after baseline')
        robot_completions = [
            row for row in completions
            if row.get('source_robot_id') == robot_id]
        if len(robot_completions) != 1:
            errors.append(f'robot {robot_id}: probe session completion is missing')
        elif int(robot_completions[0].get('timestamp_ns', 0)) < (
                phase_start_ns + int(sum(
                    phase['duration_s'] for phase in profile['phases']) * 1e9)):
            errors.append(f'robot {robot_id}: probe ended before recovery completed')

    sent = {}
    received = {}
    duplicate_sent = []
    duplicate_received = []
    for row in rows:
        if row.get('event') not in ('sent', 'received'):
            continue
        try:
            key = (
                int(row['source_robot_id']), int(row['target_robot_id']),
                int(row['sequence']))
        except (KeyError, TypeError, ValueError):
            errors.append(f'{row.get("_source")}: malformed probe identity')
            continue
        target = sent if row['event'] == 'sent' else received
        duplicates = duplicate_sent if row['event'] == 'sent' else duplicate_received
        if key in target:
            duplicates.append(key)
        else:
            target[key] = row
    if duplicate_sent:
        errors.append('duplicate sent probe labels')
    if duplicate_received:
        errors.append('duplicate received probe labels')
    if any(row.get('clock_basis') != 'system_utc' for row in rows):
        errors.append('probe clock basis is not synchronized system time')
    if any(int(row.get('payload_bytes', 0)) < profile['probe']['payload_bytes']
           for row in sent.values()):
        errors.append('sent probe payload is smaller than frozen profile')

    expected_directions = {
        (source, target)
        for source in expected_robot_ids for target in expected_robot_ids
        if source != target}
    observed_directions = {(key[0], key[1]) for key in sent}
    if observed_directions != expected_directions:
        errors.append('probe directions are missing or unexpected')
    received_directions = {(key[0], key[1]) for key in received}
    if not received_directions.issubset(expected_directions):
        errors.append('received probe directions are unexpected')
    orphan_received = sorted(set(received) - set(sent))
    if orphan_received:
        errors.append('received probe labels lack matching sent records')
    for key in sorted(set(sent) & set(received)):
        if int(received[key].get('sender_timestamp_ns', -1)) != int(
                sent[key].get('sender_timestamp_ns', -2)):
            errors.append(f'probe sender timestamp differs for {key}')
        if int(received[key].get('payload_bytes', -1)) != int(
                sent[key].get('payload_bytes', -2)):
            errors.append(f'probe payload size differs for {key}')

    phase_windows = []
    cursor_ns = phase_start_ns
    for phase in profile['phases']:
        end_ns = cursor_ns + int(phase['duration_s'] * 1e9)
        phase_windows.append((phase, cursor_ns, end_ns))
        cursor_ns = end_ns
    metrics = []
    traffic_pass = True
    for source, target in sorted(expected_directions):
        for phase, start_ns, end_ns in phase_windows:
            phase_sent = {
                key: row for key, row in sent.items()
                if key[:2] == (source, target)
                and start_ns <= int(row.get('sender_timestamp_ns', 0)) < end_ns}
            phase_received = {
                key: received[key] for key in phase_sent if key in received}
            sent_count = len(phase_sent)
            received_count = len(phase_received)
            minimum_sent = math.floor(
                phase['duration_s'] * profile['probe']['rate_hz']
                * profile['probe']['minimum_send_fraction'])
            delays_ms = [
                (int(row['timestamp_ns'])
                 - int(row['sender_timestamp_ns'])) / 1e6
                for row in phase_received.values()]
            payload_bytes = sum(
                int(row.get('payload_bytes', 0))
                for row in phase_received.values())
            measured_loss = (
                1.0 - received_count / sent_count if sent_count else None)
            tolerance = profile['measurement_tolerances'][phase['phase_id']]
            phase_ok = bool(
                sent_count >= minimum_sent
                and measured_loss is not None
                and tolerance['minimum_loss_fraction'] <= measured_loss
                <= tolerance['maximum_loss_fraction'])
            if any(delay < -10.0 for delay in delays_ms):
                phase_ok = False
                errors.append(
                    f'{source}->{target} {phase["phase_id"]}: clock-invalid delay')
            if not phase_ok:
                traffic_pass = False
                errors.append(
                    f'{source}->{target} {phase["phase_id"]}: insufficient traffic')
            metrics.append({
                'source_robot_id': source,
                'target_robot_id': target,
                'phase_id': phase['phase_id'],
                'configured_loss_probability_per_direction': phase[
                    'loss_probability'],
                'sent': sent_count,
                'received': received_count,
                'minimum_sent': minimum_sent,
                'measured_loss_fraction': measured_loss,
                'minimum_accepted_loss_fraction': tolerance[
                    'minimum_loss_fraction'],
                'maximum_accepted_loss_fraction': tolerance[
                    'maximum_loss_fraction'],
                'received_throughput_bytes_s': (
                    payload_bytes / phase['duration_s']),
                'one_way_delay_median_ms': (
                    statistics.median(delays_ms) if delays_ms else None),
                'one_way_delay_p95_ms': percentile(delays_ms, 0.95),
                'traffic_present': sent_count > 0,
                'phase_measurement_pass': phase_ok,
            })
    return {
        'errors': errors,
        'invalid_received_count': len(invalid),
        'duplicate_sent_labels': [list(key) for key in duplicate_sent],
        'duplicate_received_labels': [list(key) for key in duplicate_received],
        'metrics': metrics,
        'traffic_measurement_pass': traffic_pass and not errors,
    }


def parse_robot_ids(value):
    try:
        identifiers = [int(item) for item in value.split(',') if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            'robot IDs must be comma-separated integers') from error
    if len(identifiers) != 2 or len(set(identifiers)) != 2:
        raise argparse.ArgumentTypeError('exactly two distinct robot IDs are required')
    return identifiers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--expected-robot-ids', type=parse_robot_ids, default=[0, 1])
    parser.add_argument('--logs', type=Path, nargs='+', required=True)
    parser.add_argument('--probe-logs', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--start-tolerance-ms', type=float, default=250.0)
    parser.add_argument('--duration-tolerance-s', type=float, default=0.5)
    parser.add_argument('--allow-dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        profile_path = args.profile.expanduser().resolve()
        profile = load_profile(profile_path)
        network_paths = [path.expanduser().resolve() for path in args.logs]
        network_rows, network_parse_errors = load_jsonl(
            network_paths, NETWORK_REQUIRED)
        probe_paths = [path.expanduser().resolve() for path in args.probe_logs]
        probe_rows, probe_parse_errors = load_jsonl(
            probe_paths, PROBE_REQUIRED)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise SystemExit(str(error)) from error

    errors = []
    if network_parse_errors:
        errors.append('network JSONL parse errors')
    if any(row.get('schema_version') != '1.0' for row in network_rows):
        errors.append('unsupported network event schema')
    if any(row.get('run_id') != args.run_id for row in network_rows):
        errors.append('network run_id mismatch')
    if any(row.get('profile_id') != profile['profile_id']
           for row in network_rows):
        errors.append('network profile_id mismatch')

    grouped = defaultdict(list)
    for row in network_rows:
        grouped[row.get('session_id')].append(row)
    sessions = [
        check_session(rows, profile, args.duration_tolerance_s)
        for rows in grouped.values()]
    session_errors = [
        {'session_id': row['session_id'], 'errors': row['errors']}
        for row in sessions if row['errors']]
    observed_ids = {row['robot_id'] for row in sessions}
    if observed_ids != set(args.expected_robot_ids):
        errors.append('expected endpoint logs are missing')
    session_by_robot = {row['robot_id']: row for row in sessions}
    if len(session_by_robot) == 2:
        left, right = args.expected_robot_ids
        if session_by_robot[left]['peer_robot_id'] != right or (
                session_by_robot[right]['peer_robot_id'] != left):
            errors.append('endpoint logs are not reciprocal')

    modes = {session['mode'] for session in sessions}
    execute_mode = modes == {'execute'}
    tolerance_ns = int(args.start_tolerance_ms * 1e6)
    alignment = {}
    phase_start_ns = None
    for phase in profile['phases']:
        phase_id = phase['phase_id']
        starts = [
            session['phase_starts'].get(phase_id) for session in sessions]
        starts = [row for row in starts if row]
        scheduled = [row['scheduled_timestamp_ns'] for row in starts]
        actual = [row['actual_timestamp_ns'] for row in starts]
        scheduled_spread = max(scheduled) - min(scheduled) if scheduled else None
        actual_spread = max(actual) - min(actual) if actual else None
        scheduled_ok = (
            len(scheduled) == 2 and scheduled_spread <= tolerance_ns)
        actual_ok = len(actual) == 2 and actual_spread <= tolerance_ns
        on_schedule = (
            len(actual) == len(scheduled) == 2
            and all(abs(observed - planned) <= tolerance_ns
                    for observed, planned in zip(actual, scheduled)))
        if not scheduled_ok or (execute_mode and not (actual_ok and on_schedule)):
            errors.append(f'{phase_id}: endpoint schedules are not aligned')
        if phase_id == 'baseline' and scheduled:
            phase_start_ns = min(scheduled)
        alignment[phase_id] = {
            'scheduled_spread_ms': (
                scheduled_spread / 1e6 if scheduled_spread is not None else None),
            'actual_spread_ms': (
                actual_spread / 1e6 if actual_spread is not None else None),
            'scheduled_alignment_pass': scheduled_ok,
            'actual_alignment_pass': actual_ok,
            'executed_on_schedule': on_schedule,
        }

    structure_pass = bool(network_rows) and not (
        network_parse_errors or errors or session_errors)
    rule_execution_pass = bool(
        structure_pass and execute_mode
        and all(
            session['counters'][phase_id]['packets'] is not None
            for session in sessions for phase_id in session['counters']))

    if probe_parse_errors:
        traffic = {
            'errors': ['probe JSONL parse errors'], 'metrics': [],
            'traffic_measurement_pass': False}
    elif not probe_rows or phase_start_ns is None:
        traffic = {
            'errors': ['measured probe traffic is absent'], 'metrics': [],
            'traffic_measurement_pass': False}
    elif any(row.get('run_id') != args.run_id for row in probe_rows):
        traffic = {
            'errors': ['probe run_id mismatch'], 'metrics': [],
            'traffic_measurement_pass': False}
    else:
        traffic = probe_metrics(
            probe_rows, profile, args.expected_robot_ids, phase_start_ns)

    if execute_mode and traffic['traffic_measurement_pass']:
        metric_by_source_phase = {
            (row['source_robot_id'], row['phase_id']): row
            for row in traffic['metrics']}
        for session in sessions:
            for phase_id in ('degraded', 'disconnected'):
                counter = session['counters'].get(phase_id) or {}
                metric = metric_by_source_phase.get(
                    (session['robot_id'], phase_id)) or {}
                if counter.get('packets', 0) <= 0 or metric.get('sent', 0) <= 0:
                    traffic['errors'].append(
                        f'robot {session["robot_id"]} {phase_id}: '
                        'no measured attempted/dropped traffic')
                    traffic['traffic_measurement_pass'] = False

    physical_impairment_evidence = bool(
        structure_pass and rule_execution_pass
        and traffic['traffic_measurement_pass'])
    diagnostic_pass = structure_pass and (
        physical_impairment_evidence
        or (not execute_mode and args.allow_dry_run))
    report = {
        'schema_version': '1.0',
        'run_id': args.run_id,
        'profile_id': profile['profile_id'],
        'profile_sha256': sha256(profile_path),
        'loss_definition': '20% per direction in degraded phase',
        'rule_scope': 'one OUTPUT rule per endpoint',
        'measurement_tolerances': profile['measurement_tolerances'],
        'input_logs': [str(path) for path in network_paths],
        'probe_logs': [str(path) for path in probe_paths],
        'network_parse_errors': network_parse_errors,
        'probe_parse_errors': probe_parse_errors,
        'errors': errors,
        'session_errors': session_errors,
        'sessions': sessions,
        'alignment': alignment,
        'structure_pass': structure_pass,
        'execute_mode': execute_mode,
        'rule_execution_pass': rule_execution_pass,
        'traffic_measurement': traffic,
        'traffic_measurement_pass': traffic['traffic_measurement_pass'],
        'physical_impairment_evidence': physical_impairment_evidence,
        'diagnostic_pass': diagnostic_pass,
        'claim_authorized': False,
        'claim_authorization_note': (
            'Rule counters and measured probe traffic are necessary but not '
            'sufficient for a robustness claim; matched physical outcomes '
            'still require independent review.'),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    return 0 if diagnostic_pass else 2


if __name__ == '__main__':
    sys.exit(main())
