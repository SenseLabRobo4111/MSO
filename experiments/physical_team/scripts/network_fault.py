#!/usr/bin/env python3
"""Apply a bounded, logged UDP impairment profile between two robots.

Dry-run is the default. Live execution is deliberately difficult to enable and
must use a dedicated robot-data interface; emergency stop, operator control,
and tracking must remain on an independent path.
"""

import argparse
from datetime import datetime, timezone
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

try:
    import yaml
except ImportError as error:
    raise SystemExit('PyYAML is required: install python3-yaml') from error


IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')
INTERFACE = re.compile(r'^[A-Za-z0-9_.:-]{1,32}$')
LIVE_CONFIRMATION = 'APPLY_IMPAIRMENT'


def load_profile(path: Path):
    """Load and strictly validate one impairment schedule."""
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('schema_version') != '1.0':
        raise ValueError('profile schema_version must be 1.0')
    profile_id = value.get('profile_id')
    if not isinstance(profile_id, str) or not IDENTIFIER.fullmatch(profile_id):
        raise ValueError('invalid profile_id')
    if value.get('transport') != 'udp':
        raise ValueError('only a targeted UDP profile is supported')
    if value.get('rule_scope') != 'output_only':
        raise ValueError('rule_scope must be output_only')
    if value.get('loss_definition') != 'per_direction':
        raise ValueError('loss_definition must be per_direction')
    phases = value.get('phases')
    if not isinstance(phases, list) or len(phases) < 4:
        raise ValueError('profile requires baseline, degraded, disconnect, recovery')
    seen = set()
    cleaned = []
    for phase in phases:
        if not isinstance(phase, dict):
            raise ValueError('each phase must be an object')
        phase_id = phase.get('phase_id')
        if not isinstance(phase_id, str) or not IDENTIFIER.fullmatch(phase_id):
            raise ValueError('invalid phase_id')
        if phase_id in seen:
            raise ValueError(f'duplicate phase_id: {phase_id}')
        seen.add(phase_id)
        duration_s = float(phase.get('duration_s', -1))
        loss_probability = float(phase.get('loss_probability', -1))
        if duration_s <= 0:
            raise ValueError(f'phase {phase_id} duration must be positive')
        if not 0.0 <= loss_probability <= 1.0:
            raise ValueError(f'phase {phase_id} loss must be within [0, 1]')
        cleaned.append({
            'phase_id': phase_id,
            'duration_s': duration_s,
            'loss_probability': loss_probability,
        })
    if cleaned[0]['phase_id'] != 'baseline' or cleaned[0][
            'loss_probability'] != 0.0:
        raise ValueError('first phase must be an unimpaired baseline')
    if cleaned[-1]['phase_id'] != 'recovery' or cleaned[-1][
            'loss_probability'] != 0.0:
        raise ValueError('last phase must be an unimpaired recovery')
    if not any(0.0 < row['loss_probability'] < 1.0 for row in cleaned):
        raise ValueError('profile lacks a degraded-loss phase')
    degraded = [row for row in cleaned if row['phase_id'] == 'degraded']
    if len(degraded) != 1 or degraded[0]['loss_probability'] != 0.2:
        raise ValueError('degraded loss must be 20% per direction')
    if not any(row['loss_probability'] == 1.0 for row in cleaned):
        raise ValueError('profile lacks a complete-disconnection phase')
    return {
        'profile_id': profile_id,
        'transport': 'udp',
        'rule_scope': 'output_only',
        'loss_definition': 'per_direction',
        'phases': cleaned,
    }


def validate_peer_ip(value: str) -> str:
    """Accept only a unicast IPv4 peer address."""
    address = ipaddress.ip_address(value)
    if not isinstance(address, ipaddress.IPv4Address):
        raise ValueError('peer-ip must be IPv4')
    if address.is_loopback or address.is_multicast or address.is_unspecified:
        raise ValueError('peer-ip must be a unicast non-loopback address')
    return str(address)


def rule_commands(interface, peer_ip, comment, loss_probability):
    """Return one OUTPUT rule; the peer owns the reverse direction."""
    if loss_probability <= 0:
        return []
    matcher = []
    if loss_probability < 1.0:
        matcher = [
            '-m', 'statistic', '--mode', 'random', '--probability',
            f'{loss_probability:.6f}',
        ]
    common = ['-p', 'udp', *matcher, '-m', 'comment', '--comment', comment]
    return [[
        'iptables', '-w', '5', '-I', 'OUTPUT', '1', '-o', interface,
        '-d', peer_ip, *common, '-j', 'DROP']]


def deletion_command(add_command):
    """Convert an exact insertion command to an exact deletion command."""
    command = list(add_command)
    index = command.index('-I')
    command[index] = '-D'
    del command[index + 2]
    return command


class EventWriter:
    """Append events to JSONL and mirror them to stdout."""

    def __init__(self, path, common):
        self.common = common
        self.handle = None
        if path != '-':
            output = Path(path).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            self.handle = output.open('a', encoding='utf-8', buffering=1)

    def write(self, event, **fields):
        row = {
            **self.common,
            'event': event,
            'timestamp_ns': time.time_ns(),
            'created_utc': datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        line = json.dumps(row, sort_keys=True)
        if self.handle:
            self.handle.write(line + '\n')
            self.handle.flush()
            os.fsync(self.handle.fileno())
        print(line, flush=True)

    def close(self):
        if self.handle:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()


class RuleManager:
    """Install and remove only the exact rules created by this process."""

    def __init__(self, interface, peer_ip, comment, execute):
        self.interface = interface
        self.peer_ip = peer_ip
        self.comment = comment
        self.execute = execute
        self.active = []

    @staticmethod
    def _run(command):
        result = subprocess.run(command, capture_output=True, text=True)
        return {
            'operation': ' '.join(command[:7]),
            'command': command,
            'returncode': result.returncode,
            'stderr': result.stderr.strip(),
        }

    def clear(self):
        results = []
        remaining = []
        if self.execute:
            for command in reversed(self.active):
                result = self._run(deletion_command(command))
                results.append(result)
                if result['returncode'] != 0:
                    remaining.append(command)
        self.active = list(reversed(remaining))
        return results

    def set_loss(self, probability):
        results = self.clear()
        if any(row['returncode'] != 0 for row in results):
            raise RuntimeError('failed to remove a preceding impairment rule')
        commands = rule_commands(
            self.interface, self.peer_ip, self.comment, probability)
        if not self.execute:
            return results, commands
        for command in commands:
            result = self._run(command)
            results.append(result)
            if result['returncode'] != 0:
                self.clear()
                raise RuntimeError(
                    f'iptables rule failed: {result["stderr"] or result}')
            self.active.append(command)
        return results, commands

    def counter_snapshot(self):
        """Return packet/byte counters for this process's active rule."""
        if not self.execute:
            return {
                'command': ['iptables-save', '-c', '-t', 'filter'],
                'returncode': None,
                'rules': [],
                'dry_run': True,
            }
        result = subprocess.run(
            ['iptables-save', '-c', '-t', 'filter'],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f'iptables counter snapshot failed: {result.stderr.strip()}')
        rules = []
        counter_pattern = re.compile(r'^\[(\d+):(\d+)\]\s+(.*)$')
        for line in result.stdout.splitlines():
            if self.comment not in line:
                continue
            match = counter_pattern.match(line.strip())
            if not match:
                raise RuntimeError('could not parse iptables counter line')
            rules.append({
                'packets': int(match.group(1)),
                'bytes': int(match.group(2)),
                'rule': match.group(3),
            })
        if self.active and len(rules) != len(self.active):
            raise RuntimeError('active impairment rule counter is missing')
        return {
            'command': ['iptables-save', '-c', '-t', 'filter'],
            'returncode': result.returncode,
            'rules': rules,
            'dry_run': False,
        }


def wait_until(timestamp_ns):
    """Wait in short intervals so interruption remains responsive."""
    while True:
        remaining = (timestamp_ns - time.time_ns()) / 1e9
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


def terminate_on_signal(signum, _frame):
    """Convert normal termination signals into a cleanup path."""
    raise InterruptedError(f'received signal {signum}')


def live_preflight(args):
    """Fail before touching the network unless all safety gates pass."""
    if os.name != 'posix' or not hasattr(os, 'geteuid'):
        raise ValueError('live execution is supported only on Linux')
    if os.geteuid() != 0:
        raise ValueError('live execution requires root privileges')
    if not args.confirm_out_of_band_safety:
        raise ValueError('confirm the independent safety/control link')
    if args.confirmation != LIVE_CONFIRMATION:
        raise ValueError('live confirmation token is missing')
    if args.no_wait or args.time_scale != 1.0:
        raise ValueError('live execution forbids no-wait and time scaling')
    if args.start_at_ns is None or args.start_at_ns < time.time_ns() + 5_000_000_000:
        raise ValueError('live start-at-ns must be at least five seconds ahead')
    checks = (
        ['ip', 'link', 'show', 'dev', args.interface],
        ['ip', 'route', 'get', args.peer_ip],
        ['iptables', '--version'],
    )
    for command in checks:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(
                f'live preflight failed: {" ".join(command)}: '
                f'{result.stderr.strip()}')
        if command[:3] == ['ip', 'route', 'get'] and (
                f'dev {args.interface}' not in result.stdout):
            raise ValueError('peer route does not use the declared interface')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--session-id')
    parser.add_argument('--robot-id', type=int, required=True)
    parser.add_argument('--peer-robot-id', type=int, required=True)
    parser.add_argument('--peer-ip', required=True)
    parser.add_argument('--interface', required=True)
    parser.add_argument('--log-path', required=True)
    parser.add_argument('--start-at-ns', type=int)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm-out-of-band-safety', action='store_true')
    parser.add_argument('--confirmation', default='')
    parser.add_argument('--no-wait', action='store_true')
    parser.add_argument('--time-scale', type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not IDENTIFIER.fullmatch(args.run_id):
        raise SystemExit('invalid run-id')
    if args.robot_id < 0 or args.peer_robot_id < 0 or (
            args.robot_id == args.peer_robot_id):
        raise SystemExit('robot identifiers must be distinct non-negative integers')
    if not INTERFACE.fullmatch(args.interface):
        raise SystemExit('invalid interface')
    if args.time_scale < 0:
        raise SystemExit('time-scale must be non-negative')
    try:
        args.peer_ip = validate_peer_ip(args.peer_ip)
        profile = load_profile(args.profile.expanduser().resolve())
        if args.execute:
            live_preflight(args)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise SystemExit(str(error)) from error

    session_id = args.session_id or (
        f'{args.run_id}.r{args.robot_id}.p{args.peer_robot_id}')
    if not IDENTIFIER.fullmatch(session_id):
        raise SystemExit('invalid session-id')
    start_ns = args.start_at_ns or time.time_ns()
    mode = 'execute' if args.execute else 'dry_run'
    common = {
        'schema_version': '1.0',
        'run_id': args.run_id,
        'session_id': session_id,
        'robot_id': args.robot_id,
        'peer_robot_id': args.peer_robot_id,
        'peer_ip': args.peer_ip,
        'interface': args.interface,
        'profile_id': profile['profile_id'],
        'rule_scope': profile['rule_scope'],
        'loss_definition': profile['loss_definition'],
        'mode': mode,
    }
    writer = EventWriter(args.log_path, common)
    comment = f'mso-{args.run_id}-r{args.robot_id}-p{args.peer_robot_id}'[:240]
    manager = RuleManager(args.interface, args.peer_ip, comment, args.execute)
    for signal_name in ('SIGTERM', 'SIGHUP'):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), terminate_on_signal)
    success = False
    try:
        writer.write('session_started', scheduled_timestamp_ns=start_ns)
        if not args.no_wait:
            wait_until(start_ns)
        cumulative_s = 0.0
        for phase in profile['phases']:
            scheduled_ns = start_ns + int(cumulative_s * 1e9)
            if not args.no_wait:
                wait_until(scheduled_ns)
            results, commands = manager.set_loss(phase['loss_probability'])
            writer.write(
                'phase_started',
                scheduled_timestamp_ns=scheduled_ns,
                phase_id=phase['phase_id'],
                duration_s=phase['duration_s'],
                loss_probability=phase['loss_probability'],
                command_results=results,
                planned_rule_count=len(commands),
                planned_commands=commands,
            )
            cumulative_s += phase['duration_s']
            if not args.no_wait:
                if args.execute:
                    phase_end_target_ns = start_ns + int(cumulative_s * 1e9)
                else:
                    phase_end_target_ns = time.time_ns() + int(
                        phase['duration_s'] * args.time_scale * 1e9)
                wait_until(phase_end_target_ns)
            counter_snapshot = manager.counter_snapshot()
            writer.write(
                'phase_completed',
                scheduled_timestamp_ns=start_ns + int(cumulative_s * 1e9),
                phase_id=phase['phase_id'],
                duration_s=phase['duration_s'],
                loss_probability=phase['loss_probability'],
                counter_snapshot=counter_snapshot,
            )
        success = True
    except (KeyboardInterrupt, InterruptedError, OSError, RuntimeError) as error:
        writer.write('command_failed', detail=str(error))
    finally:
        writer.write('cleanup_started')
        cleanup = manager.clear()
        cleanup_ok = all(row['returncode'] == 0 for row in cleanup)
        writer.write(
            'cleanup_completed', command_results=cleanup,
            cleanup_ok=cleanup_ok)
        success = success and cleanup_ok
        writer.write('session_completed', success=success)
        writer.close()
    return 0 if success else 2


if __name__ == '__main__':
    sys.exit(main())
