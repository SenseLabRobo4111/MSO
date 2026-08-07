#!/usr/bin/env python3
"""Create an immutable physical-run manifest and directory skeleton."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

try:
    import yaml
except ImportError as error:
    raise SystemExit('PyYAML is required: install python3-yaml') from error


RUN_ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$')


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
    """Return whether a nested configuration still has a SET_ value."""
    if isinstance(value, str):
        return value.startswith('SET_')
    if isinstance(value, dict):
        return any(contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_placeholder(item) for item in value)
    return False


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
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument(
        '--repo-root', type=Path, required=True,
        help='clean repository containing the deployed complete system')
    parser.add_argument('--notes', default='')
    return parser.parse_args()


def main() -> int:
    """Prepare one immutable pre-capture run directory."""
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise SystemExit('invalid run identifier')
    if args.trial_id < 1:
        raise SystemExit('trial-id must be positive')
    if not config_path.is_file():
        raise SystemExit(f'config not found: {config_path}')
    if not checkpoint_path.is_file():
        raise SystemExit(f'checkpoint not found: {checkpoint_path}')

    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if contains_placeholder(config):
        raise SystemExit('replace all SET_ placeholders before preparing a run')
    team_size = int(config.get('team_size', 0))
    robots = config.get('robots', [])
    if team_size not in (2, 3, 5) or len(robots) != team_size:
        raise SystemExit('config team_size and robot roster are inconsistent')

    repo_root = args.repo_root.expanduser().resolve()
    run_dir = args.output_root.expanduser().resolve() / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f'run directory is not empty: {run_dir}')
    for child in ('analysis', 'events', 'evidence'):
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    copied_config = run_dir / 'run_config.yaml'
    shutil.copyfile(config_path, copied_config)
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
        'git_commit': git_commit(repo_root),
        'checkpoint_sha256': sha256(checkpoint_path),
        'config_sha256': sha256(copied_config),
        'ground_truth_source': args.ground_truth_source,
        'robots': robots,
        'notes': args.notes,
        'manifest_stage': 'pre_capture',
    }
    manifest_path = run_dir / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')

    roster_path = run_dir / 'evidence' / 'robot_roster.csv'
    with roster_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=('id', 'namespace', 'host', 'base_serial', 'lidar_serial'))
        writer.writeheader()
        writer.writerows(robots)

    print(run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
