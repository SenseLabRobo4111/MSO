#!/usr/bin/env python3
"""Create a pre-capture run skeleton with hashed, frozen inputs."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

try:
    import yaml
except ImportError as error:
    raise SystemExit('PyYAML is required: install python3-yaml') from error


IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    """Return the full commit identifier of a clean deployed repository."""
    status = subprocess.run(
        ['git', '-C', str(repo_root), 'status', '--porcelain'],
        check=True, capture_output=True, text=True)
    if status.stdout.strip():
        raise ValueError('deployed system repository must be clean')
    result = subprocess.run(
        ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True)
    commit = result.stdout.strip()
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('git did not return a full commit identifier')
    return commit


def contains_placeholder(value) -> bool:
    """Return whether nested configuration contains an unresolved value."""
    if isinstance(value, str):
        return not value.strip() or value.startswith('SET_')
    if isinstance(value, dict):
        return any(contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_placeholder(item) for item in value)
    return False


def require_identifier(name, value):
    """Reject empty or non-portable identifiers."""
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f'invalid {name}')


def validate_campaign(args, config, plan_path, profile_path):
    """Match this run to one exact entry in a frozen ordered campaign."""
    plan = yaml.safe_load(plan_path.read_text(encoding='utf-8'))
    profile = yaml.safe_load(profile_path.read_text(encoding='utf-8'))
    if not isinstance(plan, dict) or contains_placeholder(plan):
        raise ValueError('campaign plan is unresolved or invalid')
    if not isinstance(profile, dict) or contains_placeholder(profile):
        raise ValueError('network profile is unresolved or invalid')
    if plan.get('campaign_id') != args.campaign_id:
        raise ValueError('campaign-id differs from frozen plan')
    ordered = plan.get('ordered_runs')
    if not isinstance(ordered, list) or len(ordered) != 15:
        raise ValueError('campaign must contain 15 explicit ordered runs')
    orders = [row.get('order') for row in ordered]
    run_ids = [row.get('run_id') for row in ordered]
    if orders != list(range(1, 16)) or len(set(run_ids)) != 15:
        raise ValueError('campaign order or run identifiers are not unique')
    entry = ordered[args.randomization_order - 1]
    expected = {
        'run_id': args.run_id,
        'team_size': int(config['team_size']),
        'trial_id': args.trial_id,
        'condition_id': args.condition_id,
        'pair_id': args.pair_id,
        'start_pose_set_id': args.start_pose_set_id,
        'network_profile': args.network_profile,
    }
    mismatches = {
        key: {'plan': entry.get(key), 'requested': value}
        for key, value in expected.items() if entry.get(key) != value}
    if entry.get('order') != args.randomization_order or mismatches:
        raise ValueError(f'run differs from ordered campaign entry: {mismatches}')
    profile_hash = sha256(profile_path)
    if plan.get('network_profile_sha256') != profile_hash:
        raise ValueError('network profile hash differs from frozen campaign')
    if profile.get('profile_id') != 'udp_pair_loss_disconnect_v1':
        raise ValueError('unexpected network profile identifier')
    return plan, sha256(plan_path), profile_hash


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--site-id', required=True)
    parser.add_argument('--arena-id', required=True)
    parser.add_argument('--trial-id', type=int, required=True)
    parser.add_argument('--operator', required=True)
    parser.add_argument(
        '--ground-truth-source', required=True,
        choices=('motion_capture', 'overhead_tracking', 'surveyed_fiducials'))
    parser.add_argument('--reference-calibration', type=Path, required=True)
    parser.add_argument('--calibration-id', required=True)
    parser.add_argument('--reference-extrinsics', type=Path, required=True)
    parser.add_argument('--extrinsics-id', required=True)
    parser.add_argument('--uncertainty-translation-m', type=float, required=True)
    parser.add_argument('--uncertainty-yaw-deg', type=float, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument(
        '--repo-root', type=Path, required=True,
        help='clean repository containing the deployed complete system')
    parser.add_argument('--notes', default='')
    parser.add_argument('--campaign-id')
    parser.add_argument('--campaign-plan', type=Path)
    parser.add_argument('--condition-id', choices=('reference', 'impaired'))
    parser.add_argument('--pair-id')
    parser.add_argument('--randomization-order', type=int)
    parser.add_argument('--start-pose-set-id')
    parser.add_argument('--network-profile')
    parser.add_argument('--network-profile-file', type=Path)
    return parser.parse_args()


def main() -> int:
    """Prepare one pre-capture directory after validating every input."""
    args = parse_args()
    for name, value in (
            ('run-id', args.run_id), ('site-id', args.site_id),
            ('arena-id', args.arena_id), ('operator', args.operator),
            ('calibration-id', args.calibration_id),
            ('extrinsics-id', args.extrinsics_id)):
        require_identifier(name, value)
    if args.trial_id < 1:
        raise SystemExit('trial-id must be positive')
    if args.uncertainty_translation_m < 0 or args.uncertainty_yaw_deg < 0:
        raise SystemExit('reference uncertainties must be non-negative')

    campaign_values = (
        args.campaign_id, args.campaign_plan, args.condition_id, args.pair_id,
        args.randomization_order, args.start_pose_set_id,
        args.network_profile, args.network_profile_file)
    if any(value is not None for value in campaign_values) and not all(
            value is not None for value in campaign_values):
        raise SystemExit('all frozen campaign arguments must be supplied together')
    if args.randomization_order is not None and not (
            1 <= args.randomization_order <= 15):
        raise SystemExit('randomization-order must be within 1..15')
    if args.condition_id == 'reference' and args.network_profile != 'none':
        raise SystemExit('reference condition requires network-profile none')
    if args.condition_id == 'impaired' and args.network_profile == 'none':
        raise SystemExit('impaired condition requires a named network profile')

    paths = {
        'config': args.config.expanduser().resolve(),
        'checkpoint': args.checkpoint.expanduser().resolve(),
        'calibration': args.reference_calibration.expanduser().resolve(),
        'extrinsics': args.reference_extrinsics.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f'{name} file not found: {path}')
    for name in ('calibration', 'extrinsics'):
        try:
            value = json.loads(paths[name].read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f'{name} must be valid JSON: {error}') from error
        if not isinstance(value, dict) or not value or contains_placeholder(value):
            raise SystemExit(f'{name} JSON must be a resolved non-empty object')
    config = yaml.safe_load(paths['config'].read_text(encoding='utf-8'))
    if contains_placeholder(config):
        raise SystemExit('replace all placeholders before preparing a run')
    team_size = int(config.get('team_size', 0))
    robots = config.get('robots', [])
    if team_size not in (2, 3, 4, 5) or len(robots) != team_size:
        raise SystemExit('config team_size and robot roster are inconsistent')

    campaign_plan_hash = None
    network_profile_hash = None
    if args.campaign_id is not None:
        require_identifier('campaign-id', args.campaign_id)
        for name, value in (
                ('pair-id', args.pair_id),
                ('start-pose-set-id', args.start_pose_set_id)):
            require_identifier(name, value)
        plan_path = args.campaign_plan.expanduser().resolve()
        profile_path = args.network_profile_file.expanduser().resolve()
        if not plan_path.is_file() or not profile_path.is_file():
            raise SystemExit('campaign plan or network profile file is missing')
        try:
            _, campaign_plan_hash, network_profile_hash = validate_campaign(
                args, config, plan_path, profile_path)
        except (KeyError, OSError, ValueError, yaml.YAMLError) as error:
            raise SystemExit(str(error)) from error

    protocol_lock = config.get('protocol_lock')
    if args.campaign_id is not None:
        required_lock = {
            'stopping_rule_id', 'motion_limits_id', 'sensing_policy_id',
            'reference_timestamp_tolerance_ms'}
        if not isinstance(protocol_lock, dict) or not required_lock.issubset(
                protocol_lock):
            raise SystemExit('campaign config lacks required protocol locks')
        for name in (
                'stopping_rule_id', 'motion_limits_id', 'sensing_policy_id'):
            require_identifier(name.replace('_', '-'), protocol_lock.get(name))
        reference_tolerance = protocol_lock.get(
            'reference_timestamp_tolerance_ms')
        if not isinstance(reference_tolerance, (int, float)) or isinstance(
                reference_tolerance, bool) or not math.isfinite(
                    reference_tolerance) or reference_tolerance <= 0:
            raise SystemExit(
                'reference-timestamp-tolerance-ms must be positive and finite')

    try:
        deployed_commit = git_commit(args.repo_root.expanduser().resolve())
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(str(error)) from error
    digests = {name: sha256(path) for name, path in paths.items()}
    run_dir = args.output_root.expanduser().resolve() / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f'run directory is not empty: {run_dir}')
    for child in ('analysis', 'events', 'evidence'):
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    copied_config = run_dir / 'run_config.yaml'
    calibration_copy = run_dir / 'evidence' / 'reference_calibration.json'
    extrinsics_copy = run_dir / 'evidence' / 'reference_extrinsics.json'
    shutil.copyfile(paths['config'], copied_config)
    shutil.copyfile(paths['calibration'], calibration_copy)
    shutil.copyfile(paths['extrinsics'], extrinsics_copy)
    manifest = {
        'schema_version': '1.0',
        'run_id': args.run_id,
        'method': str(config.get('method', 'mso')),
        'team_size': team_size,
        'site_id': args.site_id,
        'arena_id': args.arena_id,
        'trial_id': args.trial_id,
        'operator': args.operator,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': deployed_commit,
        'checkpoint_sha256': digests['checkpoint'],
        'config_sha256': digests['config'],
        'ground_truth_source': args.ground_truth_source,
        'ground_truth_reference': {
            'calibration_id': args.calibration_id,
            'calibration_path': 'evidence/reference_calibration.json',
            'calibration_sha256': digests['calibration'],
            'extrinsics_id': args.extrinsics_id,
            'extrinsics_path': 'evidence/reference_extrinsics.json',
            'extrinsics_sha256': digests['extrinsics'],
            'uncertainty_translation_m': args.uncertainty_translation_m,
            'uncertainty_yaw_deg': args.uncertainty_yaw_deg,
        },
        'robots': robots,
        'notes': args.notes,
        'collection_mode': 'physical',
        'evidence_status': 'unverified_pre_capture',
        'manifest_stage': 'pre_capture',
    }
    if args.campaign_id is not None:
        manifest['campaign'] = {
            'campaign_id': args.campaign_id,
            'condition_id': args.condition_id,
            'pair_id': args.pair_id,
            'randomization_order': args.randomization_order,
            'start_pose_set_id': args.start_pose_set_id,
            'network_profile': args.network_profile,
            'campaign_plan_sha256': campaign_plan_hash,
            'network_profile_sha256': network_profile_hash,
        }
        manifest['protocol_lock'] = protocol_lock
    (run_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')

    with (run_dir / 'evidence' / 'robot_roster.csv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=('id', 'namespace', 'host', 'base_serial', 'lidar_serial'))
        writer.writeheader()
        writer.writerows(robots)
    print(run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
