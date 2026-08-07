#!/usr/bin/env python3
"""Read-only checks before a physical multirobot capture."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

try:
    import yaml
except ImportError as error:
    raise SystemExit('PyYAML is required: install python3-yaml') from error


def contains_placeholder(value) -> bool:
    """Return whether a nested configuration still has a SET_ value."""
    if isinstance(value, str):
        return value.startswith('SET_')
    if isinstance(value, dict):
        return any(contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_placeholder(item) for item in value)
    return False


def required_topics(config):
    """Expand all required topic templates for the configured team."""
    templates = config['topic_templates']
    topics = list(templates.get('required_global', []))
    for robot in config['robots']:
        topics.extend(
            template.format(id=robot['id'])
            for template in templates.get('required_per_robot', []))
    return sorted(set(topics))


def discover_topics(topics_file):
    """Read an offline topic list or query the active ROS graph."""
    if topics_file:
        return {
            line.strip().split()[0]
            for line in topics_file.read_text(encoding='utf-8').splitlines()
            if line.strip()
        }
    result = subprocess.run(
        ['ros2', 'topic', 'list'], check=True, capture_output=True, text=True)
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def clock_check(host, connect_timeout_s):
    """Estimate remote clock offset and confirm NTP state over read-only SSH."""
    command = [
        'ssh', '-o', 'BatchMode=yes',
        '-o', f'ConnectTimeout={connect_timeout_s}', host,
        'date +%s%N; timedatectl show -p NTPSynchronized --value 2>/dev/null'
    ]
    start_ns = time.time_ns()
    result = subprocess.run(command, capture_output=True, text=True)
    end_ns = time.time_ns()
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    output = {
        'host': host,
        'reachable': result.returncode == 0,
        'rtt_ms': (end_ns - start_ns) / 1e6,
        'offset_ms': None,
        'ntp_synchronised': None,
        'stderr': result.stderr.strip(),
    }
    if result.returncode == 0 and lines:
        remote_ns = int(lines[0])
        midpoint_ns = (start_ns + end_ns) // 2
        output['offset_ms'] = (remote_ns - midpoint_ns) / 1e6
        if len(lines) > 1:
            output['ntp_synchronised'] = lines[1].lower() == 'yes'
    return output


def check(name, passed, detail):
    """Build one serialisable preflight check."""
    return {'name': name, 'passed': bool(passed), 'detail': detail}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--topics-file', type=Path)
    parser.add_argument('--connect-timeout-s', type=int, default=5)
    parser.add_argument(
        '--skip-ssh', action='store_true',
        help='diagnostic only; causes the clock-evidence check to fail')
    return parser.parse_args()


def main() -> int:
    """Run all read-only preflight checks."""
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    checks = []
    team_size = int(config.get('team_size', 0))
    robots = config.get('robots', [])
    checks.append(check(
        'team roster',
        team_size in (2, 3, 5) and len(robots) == team_size,
        {'team_size': team_size, 'roster_count': len(robots)}))
    checks.append(check(
        'placeholders replaced', not contains_placeholder(config),
        'all host and hardware identifiers must be real values'))

    ids = [robot.get('id') for robot in robots]
    hosts = [robot.get('host') for robot in robots]
    base_serials = [robot.get('base_serial') for robot in robots]
    lidar_serials = [robot.get('lidar_serial') for robot in robots]
    checks.append(check(
        'unique robot identities',
        ids == list(range(team_size))
        and len(set(hosts)) == team_size
        and len(set(base_serials)) == team_size
        and len(set(lidar_serials)) == team_size,
        {'ids': ids, 'hosts': hosts}))

    try:
        topics = discover_topics(args.topics_file)
        missing = sorted(set(required_topics(config)) - topics)
        checks.append(check('required ROS topics', not missing, {'missing': missing}))
        checks.append(check(
            'physical clock source', '/clock' not in topics,
            'a published /clock topic indicates simulation time'))
    except (OSError, subprocess.CalledProcessError) as error:
        topics = set()
        checks.append(check('required ROS topics', False, str(error)))
        checks.append(check('physical clock source', False, 'topic discovery failed'))

    recorder = config.get('recorder', {})
    output_root = (
        args.output_root or Path(recorder.get('output_root', '.'))
    ).expanduser().resolve()
    existing = output_root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    free_gb = shutil.disk_usage(existing).free / (1024 ** 3)
    min_free_gb = float(recorder.get('min_free_gb', 100))
    checks.append(check(
        'recording disk capacity', free_gb >= min_free_gb,
        {'path': str(existing), 'free_gb': round(free_gb, 2),
         'minimum_gb': min_free_gb}))

    clock_results = []
    if not args.skip_ssh:
        for host in hosts:
            clock_results.append(clock_check(host, args.connect_timeout_s))
    max_offset = float(recorder.get('max_clock_offset_ms', 5.0))
    max_rtt = float(recorder.get('max_clock_rtt_ms', 20.0))
    clock_pass = len(clock_results) == team_size and all(
        result['reachable']
        and result['ntp_synchronised'] is True
        and result['offset_ms'] is not None
        and abs(result['offset_ms']) <= max_offset
        and result['rtt_ms'] <= max_rtt
        for result in clock_results)
    checks.append(check(
        'multi-host clock synchronisation', clock_pass,
        {'limits_ms': {'offset': max_offset, 'rtt': max_rtt},
         'hosts': clock_results,
         'skipped': args.skip_ssh}))

    report = {
        'schema_version': '1.0',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': str(config_path),
        'team_size': team_size,
        'checks': checks,
        'overall_pass': all(item['passed'] for item in checks),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.report:
        args.report.expanduser().resolve().write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    return 0 if report['overall_pass'] else 2


if __name__ == '__main__':
    sys.exit(main())
