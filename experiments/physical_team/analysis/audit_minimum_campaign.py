#!/usr/bin/env python3
"""Audit the frozen ordered 15-run physical evidence campaign."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

try:
    import yaml
except ImportError as error:
    raise SystemExit('PyYAML is required: install python3-yaml') from error


IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')
GIT_HASH = re.compile(r'^[0-9a-f]{40}$')
SHA256 = re.compile(r'^[0-9a-f]{64}$')
REQUIRED_LOCK_FIELDS = {
    'site_id', 'arena_id', 'git_commit', 'checkpoint_sha256',
    'stopping_rule_id', 'motion_limits_id', 'sensing_policy_id',
    'reference_timestamp_tolerance_ms', 'calibration_sha256',
    'extrinsics_sha256', 'uncertainty_translation_m',
    'uncertainty_yaw_deg',
}


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def lock_value(manifest, field):
    if field in manifest:
        return manifest.get(field)
    if field in (manifest.get('protocol_lock') or {}):
        return manifest['protocol_lock'].get(field)
    return (manifest.get('ground_truth_reference') or {}).get(field)


def resolved_value(value):
    """Reject nulls, blank strings, empty containers, and non-finite values."""
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


def lock_value_valid(field, value):
    """Apply the field-specific format frozen by the campaign plan."""
    if not resolved_value(value):
        return False
    if field in {
            'site_id', 'arena_id', 'stopping_rule_id',
            'motion_limits_id', 'sensing_policy_id'}:
        return isinstance(value, str) and bool(IDENTIFIER.fullmatch(value))
    if field == 'git_commit':
        return isinstance(value, str) and bool(GIT_HASH.fullmatch(value))
    if field.endswith('_sha256'):
        return isinstance(value, str) and bool(SHA256.fullmatch(value))
    if field.startswith('uncertainty_'):
        return isinstance(value, (int, float)) and not isinstance(
            value, bool) and value >= 0
    if field == 'reference_timestamp_tolerance_ms':
        return isinstance(value, (int, float)) and not isinstance(
            value, bool) and value > 0
    return True


def hardware_set(manifest):
    return sorted(
        (robot.get('base_serial'), robot.get('lidar_serial'))
        for robot in manifest.get('robots', []))


def write_csv(path, rows):
    fields = (
        'run_id', 'cell_id', 'trial_id', 'team_size', 'condition_id',
        'pair_id', 'start_pose_set_id', 'randomization_order',
        'bag_start_ns',
        'protocol_diagnostic_pass', 'evidence_class',
        'rule_execution_pass', 'traffic_measurement_pass',
        'physical_impairment_evidence')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--network-profile', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def validate_plan(plan, profile_hash):
    """Return explicit entries/cells after validating the frozen plan."""
    errors = []
    if not isinstance(plan, dict) or plan.get('schema_version') != '1.0':
        raise ValueError('invalid campaign plan schema')
    campaign_id = plan.get('campaign_id')
    if not isinstance(campaign_id, str) or not IDENTIFIER.fullmatch(campaign_id):
        raise ValueError('campaign identifier is empty or invalid')
    if plan.get('evidence_class') != 'physical':
        raise ValueError('plan does not require physical evidence')
    if plan.get('network_profile_sha256') != profile_hash:
        raise ValueError('network profile hash differs from frozen plan')
    required_lock_fields = plan.get('required_lock_fields')
    if not isinstance(required_lock_fields, list) or len(
            required_lock_fields) != len(set(required_lock_fields)) or not (
                REQUIRED_LOCK_FIELDS.issubset(set(required_lock_fields))):
        raise ValueError('campaign required_lock_fields are incomplete or invalid')
    cells = plan.get('cells')
    if not isinstance(cells, list) or len(cells) != 3:
        raise ValueError('plan must contain exactly three evidence cells')
    signatures = {}
    cell_trials = {}
    for cell in cells:
        cell_id = cell.get('cell_id')
        if not isinstance(cell_id, str) or not IDENTIFIER.fullmatch(cell_id):
            errors.append('empty or invalid cell identifier')
            continue
        signature = (
            int(cell['team_size']), str(cell['condition_id']),
            str(cell['network_profile']))
        if signature in signatures:
            errors.append(f'duplicate cell signature: {signature}')
        signatures[signature] = cell_id
        trials = [int(value) for value in cell.get('trial_ids', [])]
        if len(trials) != 5 or len(set(trials)) != 5:
            errors.append(f'{cell_id}: expected five unique trials')
        cell_trials[cell_id] = set(trials)
    required = {
        (2, 'reference', 'none'),
        (2, 'impaired', 'udp_pair_loss_disconnect_v1'),
        (3, 'reference', 'none')}
    if set(signatures) != required:
        errors.append('plan evidence cells differ from required minimum')

    ordered = plan.get('ordered_runs')
    if not isinstance(ordered, list) or len(ordered) != 15:
        raise ValueError('plan must contain 15 explicit ordered runs')
    orders = [entry.get('order') for entry in ordered]
    run_ids = [entry.get('run_id') for entry in ordered]
    if orders != list(range(1, 16)):
        errors.append('ordered runs are not the exact 1..15 sequence')
    if len(set(run_ids)) != 15 or any(
            not isinstance(value, str) or not IDENTIFIER.fullmatch(value)
            for value in run_ids):
        errors.append('ordered run identifiers are empty or duplicate')
    entry_keys = set()
    annotated = []
    for entry in ordered:
        required_ids = (
            entry.get('pair_id'), entry.get('start_pose_set_id'),
            entry.get('network_profile'))
        if any(not isinstance(value, str) or not value.strip()
               for value in required_ids):
            errors.append(f'{entry.get("run_id")}: empty ordered identifier')
        signature = (
            int(entry.get('team_size', -1)), str(entry.get('condition_id')),
            str(entry.get('network_profile')))
        cell_id = signatures.get(signature)
        trial_id = int(entry.get('trial_id', -1))
        if cell_id is None or trial_id not in cell_trials.get(cell_id, set()):
            errors.append(f'{entry.get("run_id")}: outside declared cells')
        key = (cell_id, trial_id)
        if key in entry_keys:
            errors.append(f'duplicate ordered cell/trial: {key}')
        entry_keys.add(key)
        annotated.append({**entry, 'cell_id': cell_id})
    if errors:
        raise ValueError('; '.join(errors))
    return campaign_id, annotated


def main():
    args = parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    plan_path = args.plan.expanduser().resolve()
    profile_path = args.network_profile.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        plan = yaml.safe_load(plan_path.read_text(encoding='utf-8'))
        profile = yaml.safe_load(profile_path.read_text(encoding='utf-8'))
        profile_hash = sha256(profile_path)
        plan_hash = sha256(plan_path)
        campaign_id, ordered = validate_plan(plan, profile_hash)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise SystemExit(str(error)) from error
    if not isinstance(profile, dict) or profile.get(
            'profile_id') != plan.get('network_profile_id'):
        raise SystemExit('profile identifier differs from frozen campaign')

    expected_by_run = {entry['run_id']: entry for entry in ordered}
    errors = []
    records = []
    observed = Counter()
    for manifest_path in sorted(campaign_root.rglob('manifest.json')):
        manifest = load_json(manifest_path)
        campaign = manifest.get('campaign') or {}
        if campaign.get('campaign_id') != campaign_id:
            continue
        run_id = manifest.get('run_id')
        observed[run_id] += 1
        expected = expected_by_run.get(run_id)
        if expected is None:
            errors.append(f'{run_id}: run is outside frozen ordered plan')
            continue
        run_dir = manifest_path.parent
        audit_path = run_dir / 'analysis' / 'run_audit.json'
        if not audit_path.is_file():
            errors.append(f'{run_id}: missing run audit')
            audit = {}
        else:
            audit = load_json(audit_path)
        network = audit.get('network') or {}
        row = {
            'run_id': run_id,
            'cell_id': expected['cell_id'],
            'trial_id': manifest.get('trial_id'),
            'team_size': manifest.get('team_size'),
            'condition_id': campaign.get('condition_id'),
            'pair_id': campaign.get('pair_id'),
            'start_pose_set_id': campaign.get('start_pose_set_id'),
            'randomization_order': campaign.get('randomization_order'),
            'bag_start_ns': audit.get('bag_start_ns'),
            'protocol_diagnostic_pass': bool(
                audit.get('protocol_diagnostic_pass')),
            'evidence_class': audit.get('evidence_class'),
            'rule_execution_pass': bool(network.get('rule_execution_pass')),
            'traffic_measurement_pass': bool(
                network.get('traffic_measurement_pass')),
            'physical_impairment_evidence': bool(
                network.get('physical_impairment_evidence')),
        }
        records.append({'row': row, 'manifest': manifest, 'audit': audit})
        expected_values = {
            'team_size': manifest.get('team_size'),
            'trial_id': manifest.get('trial_id'),
            'condition_id': campaign.get('condition_id'),
            'pair_id': campaign.get('pair_id'),
            'start_pose_set_id': campaign.get('start_pose_set_id'),
            'network_profile': campaign.get('network_profile'),
            'order': campaign.get('randomization_order')}
        mismatches = {
            key: {'plan': expected.get(key), 'manifest': value}
            for key, value in expected_values.items()
            if expected.get(key) != value}
        if mismatches:
            errors.append(f'{run_id}: ordered manifest mismatch {mismatches}')
        if campaign.get('campaign_plan_sha256') != plan_hash or campaign.get(
                'network_profile_sha256') != profile_hash:
            errors.append(f'{run_id}: campaign/profile hash mismatch')
        if manifest.get('collection_mode') != 'physical' or not row[
                'protocol_diagnostic_pass'] or row['evidence_class'] != 'physical':
            errors.append(f'{run_id}: physical run audit did not pass')
        if manifest.get('method') != plan.get('method'):
            errors.append(f'{run_id}: method differs from plan')
        ground_truth = manifest.get('ground_truth_reference') or {}
        if any(not ground_truth.get(field) for field in (
                'calibration_id', 'calibration_sha256', 'extrinsics_id',
                'extrinsics_sha256')):
            errors.append(f'{run_id}: frozen ground-truth identity is incomplete')
        if campaign.get('condition_id') == 'impaired' and not all((
                row['rule_execution_pass'], row['traffic_measurement_pass'],
                row['physical_impairment_evidence'])):
            errors.append(f'{run_id}: live rule/traffic impairment evidence failed')
        if campaign.get('condition_id') == 'reference' and network.get(
                'executed_event_count', 0) != 0:
            errors.append(f'{run_id}: reference contains an executed fault')

    missing = sorted(set(expected_by_run) - set(observed))
    duplicates = sorted(run_id for run_id, count in observed.items() if count != 1)
    if missing:
        errors.append(f'missing planned runs: {missing}')
    if duplicates:
        errors.append(f'duplicate planned runs: {duplicates}')

    chronological = sorted(
        records, key=lambda record: (
            record['row']['randomization_order']
            if isinstance(record['row']['randomization_order'], int)
            else 10 ** 9))
    bag_starts = [record['row']['bag_start_ns'] for record in chronological]
    if len(bag_starts) != 15 or any(
            not isinstance(value, int) for value in bag_starts) or any(
                right <= left for left, right in zip(bag_starts, bag_starts[1:])):
        errors.append('physical bag start times do not follow frozen run order')

    lock_summary = {}
    for field in [str(value) for value in plan.get('required_lock_fields', [])]:
        field_values = [
            (record['row']['run_id'], lock_value(record['manifest'], field))
            for record in records]
        values = {
            json.dumps(value, sort_keys=True) for _, value in field_values}
        lock_summary[field] = sorted(values)
        invalid_runs = [
            run_id for run_id, value in field_values
            if not lock_value_valid(field, value)]
        if len(values) != 1 or invalid_runs:
            errors.append(
                f'protocol lock differs, is empty, or is malformed: {field}; '
                f'invalid_runs={invalid_runs}')

    by_cell_trial = {
        (record['row']['cell_id'], record['row']['trial_id']): record
        for record in records}
    paired_checks = []
    for trial_id in range(1, 6):
        current = by_cell_trial.get(('n2_impaired', trial_id))
        reference = by_cell_trial.get(('n2_reference', trial_id))
        if not current or not reference:
            continue
        current_campaign = current['manifest']['campaign']
        reference_campaign = reference['manifest']['campaign']
        passed = all((
            current_campaign.get('pair_id') == reference_campaign.get('pair_id'),
            current_campaign.get('start_pose_set_id')
            == reference_campaign.get('start_pose_set_id'),
            hardware_set(current['manifest'])
            == hardware_set(reference['manifest'])))
        paired_checks.append({'trial_id': trial_id, 'passed': passed})
        if not passed:
            errors.append(f'N=2 trial {trial_id}: pairing differs')

    complete = not errors and len(records) == 15
    write_csv(output_dir / 'minimum_campaign_runs.csv', [
        record['row'] for record in records])
    report = {
        'schema_version': '1.0',
        'campaign_id': campaign_id,
        'campaign_plan_sha256': plan_hash,
        'network_profile_sha256': profile_hash,
        'expected_run_count': 15,
        'observed_run_count': len(records),
        'missing_runs': missing,
        'duplicate_runs': duplicates,
        'protocol_locks': lock_summary,
        'paired_checks': paired_checks,
        'errors': errors,
        'minimum_campaign_complete': complete,
        'claim_authorized': False,
        'claim_authorization_note': (
            'Completeness is a protocol diagnostic. Scientific claims still '
            'require independent evidence and statistical review.'),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    (output_dir / 'minimum_campaign_audit.json').write_text(
        rendered, encoding='utf-8')
    print(rendered, end='')
    return 0 if complete else 2


if __name__ == '__main__':
    sys.exit(main())
