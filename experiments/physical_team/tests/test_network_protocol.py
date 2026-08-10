#!/usr/bin/env python3
"""Synthetic protocol tests; generated logs are diagnostics, never evidence."""

import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PHYSICAL_TEAM = Path(__file__).resolve().parents[1]
PROFILE = PHYSICAL_TEAM / 'config' / 'network_impairment.yaml'
FAULT_TOOL = PHYSICAL_TEAM / 'scripts' / 'network_fault.py'
AUDIT_TOOL = PHYSICAL_TEAM / 'analysis' / 'audit_network.py'
CAMPAIGN_AUDIT = PHYSICAL_TEAM / 'analysis' / 'audit_minimum_campaign.py'
PLAN = PHYSICAL_TEAM / 'config' / 'minimum_campaign.yaml'


class NetworkProtocolTest(unittest.TestCase):
    def run_fault(self, log_path, robot_id, peer_id, peer_ip):
        return subprocess.run([
            sys.executable, str(FAULT_TOOL),
            '--profile', str(PROFILE),
            '--run-id', 'synthetic_replay_check',
            '--robot-id', str(robot_id),
            '--peer-robot-id', str(peer_id),
            '--peer-ip', peer_ip,
            '--interface', f'replay{robot_id}',
            '--log-path', str(log_path),
            '--start-at-ns', '2000000000000000000',
            '--no-wait', '--time-scale', '0',
        ], capture_output=True, text=True)

    def test_profile_is_twenty_percent_per_direction(self):
        import yaml
        profile = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))
        self.assertEqual(profile['rule_scope'], 'output_only')
        self.assertEqual(profile['loss_definition'], 'per_direction')
        degraded = next(
            phase for phase in profile['phases']
            if phase['phase_id'] == 'degraded')
        self.assertEqual(float(degraded['loss_probability']), 0.2)
        self.assertEqual(
            profile['measurement_tolerances']['degraded'], {
                'minimum_loss_fraction': 0.15,
                'maximum_loss_fraction': 0.25})
        self.assertLessEqual(
            profile['measurement_tolerances']['baseline'][
                'maximum_loss_fraction'], 0.05)
        self.assertGreaterEqual(
            profile['measurement_tolerances']['disconnected'][
                'minimum_loss_fraction'], 0.95)

    def test_dry_run_is_structurally_valid_but_not_physical_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log0 = root / 'robot0.jsonl'
            log1 = root / 'robot1.jsonl'
            self.assertEqual(
                self.run_fault(log0, 0, 1, '192.0.2.11').returncode, 0)
            self.assertEqual(
                self.run_fault(log1, 1, 0, '192.0.2.10').returncode, 0)
            command = [
                sys.executable, str(AUDIT_TOOL),
                '--profile', str(PROFILE),
                '--run-id', 'synthetic_replay_check',
                '--expected-robot-ids', '0,1',
                '--logs', str(log0), str(log1),
            ]
            strict = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(strict.returncode, 2)
            strict_report = json.loads(strict.stdout)
            self.assertTrue(strict_report['structure_pass'])
            self.assertEqual(
                strict_report['profile_sha256'],
                hashlib.sha256(PROFILE.read_bytes()).hexdigest())
            self.assertFalse(strict_report['physical_impairment_evidence'])
            diagnostic = subprocess.run(
                [*command, '--allow-dry-run'], capture_output=True, text=True)
            self.assertEqual(diagnostic.returncode, 0)
            diagnostic_report = json.loads(diagnostic.stdout)
            self.assertTrue(diagnostic_report['diagnostic_pass'])
            self.assertFalse(
                diagnostic_report['physical_impairment_evidence'])
            for log_path in (log0, log1):
                rows = [
                    json.loads(line) for line in log_path.read_text(
                        encoding='utf-8').splitlines()]
                for row in rows:
                    if row.get('event') != 'phase_started':
                        continue
                    for command_row in row.get('planned_commands', []):
                        self.assertIn('OUTPUT', command_row)
                        self.assertNotIn('INPUT', command_row)
            tampered = [
                json.loads(line) for line in log0.read_text(
                    encoding='utf-8').splitlines()]
            phase = next(
                row for row in tampered
                if row.get('event') == 'phase_started'
                and row.get('phase_id') == 'degraded')
            input_rule = list(phase['planned_commands'][0])
            input_rule[input_rule.index('OUTPUT')] = 'INPUT'
            phase['planned_commands'].append(input_rule)
            phase['planned_rule_count'] = 2
            bad_log = root / 'robot0_duplicate_rule.jsonl'
            bad_log.write_text(
                ''.join(json.dumps(row) + '\n' for row in tampered),
                encoding='utf-8')
            rejected = subprocess.run([
                sys.executable, str(AUDIT_TOOL), '--profile', str(PROFILE),
                '--run-id', 'synthetic_replay_check',
                '--expected-robot-ids', '0,1', '--logs', str(bad_log),
                str(log1), '--allow-dry-run'], capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 2)
            self.assertFalse(json.loads(rejected.stdout)['structure_pass'])

            wrong_probability = [
                json.loads(line) for line in log0.read_text(
                    encoding='utf-8').splitlines()]
            degraded = next(
                row for row in wrong_probability
                if row.get('event') == 'phase_started'
                and row.get('phase_id') == 'degraded')
            command_row = degraded['planned_commands'][0]
            probability_index = command_row.index('--probability') + 1
            command_row[probability_index] = '0.100000'
            wrong_probability_path = root / 'robot0_wrong_probability.jsonl'
            wrong_probability_path.write_text(
                ''.join(json.dumps(row) + '\n' for row in wrong_probability),
                encoding='utf-8')
            rejected_probability = subprocess.run([
                sys.executable, str(AUDIT_TOOL), '--profile', str(PROFILE),
                '--run-id', 'synthetic_replay_check',
                '--expected-robot-ids', '0,1',
                '--logs', str(wrong_probability_path), str(log1),
                '--allow-dry-run'], capture_output=True, text=True)
            self.assertEqual(rejected_probability.returncode, 2)
            self.assertFalse(
                json.loads(rejected_probability.stdout)['structure_pass'])

    def test_live_mode_fails_without_explicit_safety_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([
                sys.executable, str(FAULT_TOOL),
                '--profile', str(PROFILE),
                '--run-id', 'synthetic_live_gate',
                '--robot-id', '0', '--peer-robot-id', '1',
                '--peer-ip', '192.0.2.11', '--interface', 'data0',
                '--log-path', str(Path(directory) / 'events.jsonl'),
                '--start-at-ns', '4102444800000000000',
                '--execute',
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(
                'independent safety/control link' in result.stderr
                or 'supported only on Linux' in result.stderr)

    def test_executed_rules_without_probe_flow_are_not_physical_evidence(self):
        """Separate successful rule commands from measured traffic evidence."""
        import yaml
        profile = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start_ns = 2_000_000_000_000_000_000
            logs = []
            for robot_id, peer_id, peer_ip in (
                    (0, 1, '192.0.2.11'), (1, 0, '192.0.2.10')):
                common = {
                    'schema_version': '1.0',
                    'run_id': 'synthetic_zero_flow',
                    'session_id': f'zero.r{robot_id}',
                    'robot_id': robot_id, 'peer_robot_id': peer_id,
                    'peer_ip': peer_ip, 'interface': f'data{robot_id}',
                    'profile_id': profile['profile_id'],
                    'rule_scope': 'output_only',
                    'loss_definition': 'per_direction', 'mode': 'execute'}
                rows = [{
                    **common, 'event': 'session_started',
                    'timestamp_ns': start_ns - 1,
                    'scheduled_timestamp_ns': start_ns}]
                cursor = start_ns
                previous_loss = 0.0
                for phase in profile['phases']:
                    loss = float(phase['loss_probability'])
                    add_rule = [
                        'iptables', '-w', '5', '-I', 'OUTPUT', '1',
                        '-o', f'data{robot_id}', '-d', peer_ip,
                        '-p', 'udp']
                    if 0.0 < loss < 1.0:
                        add_rule.extend([
                            '-m', 'statistic', '--mode', 'random',
                            '--probability', f'{loss:.6f}'])
                    add_rule.extend([
                        '-m', 'comment', '--comment',
                        f'mso-synthetic_zero_flow-r{robot_id}-p{peer_id}',
                        '-j', 'DROP'])
                    planned = [add_rule] if loss > 0 else []
                    result_count = (
                        (1 if previous_loss > 0 else 0)
                        + (1 if loss > 0 else 0))
                    rows.append({
                        **common, 'event': 'phase_started',
                        'timestamp_ns': cursor,
                        'scheduled_timestamp_ns': cursor,
                        'phase_id': phase['phase_id'],
                        'duration_s': float(phase['duration_s']),
                        'loss_probability': loss,
                        'planned_rule_count': len(planned),
                        'planned_commands': planned,
                        'command_results': [
                            {'returncode': 0} for _ in range(result_count)]})
                    cursor += int(float(phase['duration_s']) * 1e9)
                    rules = ([{
                        'packets': 20, 'bytes': 5120,
                        'rule': (
                            f'-A OUTPUT -o data{robot_id} -d {peer_ip} '
                            f'-p udp -j DROP')}]
                        if loss > 0 else [])
                    rows.append({
                        **common, 'event': 'phase_completed',
                        'timestamp_ns': cursor,
                        'scheduled_timestamp_ns': cursor,
                        'phase_id': phase['phase_id'],
                        'duration_s': float(phase['duration_s']),
                        'loss_probability': loss,
                        'counter_snapshot': {
                            'command': ['iptables-save', '-c', '-t', 'filter'],
                            'returncode': 0, 'dry_run': False, 'rules': rules}})
                    previous_loss = loss
                rows.extend((
                    {**common, 'event': 'cleanup_started',
                     'timestamp_ns': cursor + 1},
                    {**common, 'event': 'cleanup_completed',
                     'timestamp_ns': cursor + 2, 'cleanup_ok': True,
                     'command_results': []},
                    {**common, 'event': 'session_completed',
                     'timestamp_ns': cursor + 3, 'success': True}))
                path = root / f'network_{robot_id}.jsonl'
                path.write_text(
                    ''.join(json.dumps(row) + '\n' for row in rows),
                    encoding='utf-8')
                logs.append(path)
            probe_logs = []
            for robot_id, peer_id in ((0, 1), (1, 0)):
                rows = [
                    {'schema_version': '1.0',
                     'run_id': 'synthetic_zero_flow',
                     'session_id': f'probe.r{robot_id}',
                     'event': 'session_started', 'timestamp_ns': start_ns,
                     'source_robot_id': robot_id,
                     'target_robot_id': peer_id,
                     'topic': f'/probe/{robot_id}',
                     'clock_basis': 'system_utc',
                     'rate_hz': 20.0, 'duration_s': 140.0},
                    {'schema_version': '1.0',
                     'run_id': 'synthetic_zero_flow',
                     'session_id': f'probe.r{robot_id}',
                     'event': 'session_completed',
                     'timestamp_ns': cursor,
                     'source_robot_id': robot_id,
                     'target_robot_id': peer_id,
                     'topic': f'/probe/{robot_id}',
                     'clock_basis': 'system_utc'}]
                path = root / f'probe_{robot_id}.jsonl'
                path.write_text(
                    ''.join(json.dumps(row) + '\n' for row in rows),
                    encoding='utf-8')
                probe_logs.append(path)
            result = subprocess.run([
                sys.executable, str(AUDIT_TOOL), '--profile', str(PROFILE),
                '--run-id', 'synthetic_zero_flow',
                '--expected-robot-ids', '0,1', '--logs',
                *(str(path) for path in logs), '--probe-logs',
                *(str(path) for path in probe_logs)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            report = json.loads(result.stdout)
            self.assertTrue(report['structure_pass'])
            self.assertTrue(report['rule_execution_pass'])
            self.assertFalse(report['traffic_measurement_pass'])
            self.assertFalse(report['physical_impairment_evidence'])
            permissive = subprocess.run([
                sys.executable, str(AUDIT_TOOL), '--profile', str(PROFILE),
                '--run-id', 'synthetic_zero_flow',
                '--expected-robot-ids', '0,1', '--logs',
                *(str(path) for path in logs), '--probe-logs',
                *(str(path) for path in probe_logs), '--allow-dry-run'],
                capture_output=True, text=True)
            self.assertEqual(permissive.returncode, 2)
            self.assertFalse(json.loads(permissive.stdout)['diagnostic_pass'])
            bad_rows = [
                json.loads(line) for line in logs[0].read_text(
                    encoding='utf-8').splitlines()]
            degraded_end = next(
                row for row in bad_rows
                if row.get('event') == 'phase_completed'
                and row.get('phase_id') == 'degraded')
            degraded_end['timestamp_ns'] -= 10_000_000_000
            bad_path = root / 'network_bad_duration.jsonl'
            bad_path.write_text(
                ''.join(json.dumps(row) + '\n' for row in bad_rows),
                encoding='utf-8')
            bad_result = subprocess.run([
                sys.executable, str(AUDIT_TOOL), '--profile', str(PROFILE),
                '--run-id', 'synthetic_zero_flow',
                '--expected-robot-ids', '0,1', '--logs', str(bad_path),
                str(logs[1]), '--probe-logs',
                *(str(path) for path in probe_logs)],
                capture_output=True, text=True)
            self.assertEqual(bad_result.returncode, 2)
            self.assertFalse(json.loads(bad_result.stdout)['structure_pass'])

            full_delivery_logs = []
            for source, target in ((0, 1), (1, 0)):
                session_id = f'full.r{source}'
                rows = [{
                    'schema_version': '1.0',
                    'run_id': 'synthetic_zero_flow',
                    'session_id': session_id,
                    'event': 'session_started',
                    'timestamp_ns': start_ns - 1,
                    'source_robot_id': source,
                    'target_robot_id': target,
                    'topic': f'/probe/{source}',
                    'clock_basis': 'system_utc',
                    'rate_hz': 20.0,
                    'duration_s': 140.0}]
                sequence = 0
                phase_cursor = start_ns
                for phase in profile['phases']:
                    sample_count = int(float(phase['duration_s']) * 20)
                    for sample_index in range(sample_count):
                        sent_ns = phase_cursor + int(sample_index * 50_000_000)
                        common = {
                            'schema_version': '1.0',
                            'run_id': 'synthetic_zero_flow',
                            'session_id': session_id,
                            'source_robot_id': source,
                            'target_robot_id': target,
                            'topic': f'/probe/{source}',
                            'clock_basis': 'system_utc',
                            'sequence': sequence,
                            'sender_timestamp_ns': sent_ns,
                            'payload_bytes': 256}
                        rows.append({
                            **common, 'event': 'sent',
                            'timestamp_ns': sent_ns})
                        rows.append({
                            **common, 'event': 'received',
                            'timestamp_ns': sent_ns + 1_000_000})
                        sequence += 1
                    phase_cursor += int(float(phase['duration_s']) * 1e9)
                rows.append({
                    'schema_version': '1.0',
                    'run_id': 'synthetic_zero_flow',
                    'session_id': session_id,
                    'event': 'session_completed',
                    'timestamp_ns': cursor,
                    'source_robot_id': source,
                    'target_robot_id': target,
                    'topic': f'/probe/{source}',
                    'clock_basis': 'system_utc'})
                path = root / f'probe_full_delivery_{source}.jsonl'
                path.write_text(
                    ''.join(json.dumps(row) + '\n' for row in rows),
                    encoding='utf-8')
                full_delivery_logs.append(path)
            full_delivery = subprocess.run([
                sys.executable, str(AUDIT_TOOL), '--profile', str(PROFILE),
                '--run-id', 'synthetic_zero_flow',
                '--expected-robot-ids', '0,1', '--logs',
                *(str(path) for path in logs), '--probe-logs',
                *(str(path) for path in full_delivery_logs)],
                capture_output=True, text=True)
            self.assertEqual(full_delivery.returncode, 2)
            full_report = json.loads(full_delivery.stdout)
            self.assertTrue(full_report['structure_pass'])
            self.assertTrue(full_report['rule_execution_pass'])
            self.assertFalse(full_report['traffic_measurement_pass'])
            self.assertFalse(full_report['physical_impairment_evidence'])
            by_phase = {
                row['phase_id']: row
                for row in full_report['traffic_measurement']['metrics']
                if row['source_robot_id'] == 0}
            self.assertTrue(by_phase['baseline']['phase_measurement_pass'])
            self.assertFalse(by_phase['degraded']['phase_measurement_pass'])
            self.assertFalse(
                by_phase['disconnected']['phase_measurement_pass'])

    def test_transient_complete_campaign_fixture_remains_non_authorizing(self):
        """Exercise campaign matching with no persistent experimental rows."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            import yaml
            plan = yaml.safe_load(PLAN.read_text(encoding='utf-8'))
            plan_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
            profile_hash = hashlib.sha256(PROFILE.read_bytes()).hexdigest()
            for entry in plan['ordered_runs']:
                run_id = entry['run_id']
                team_size = int(entry['team_size'])
                condition = entry['condition_id']
                run_dir = root / run_id
                (run_dir / 'analysis').mkdir(parents=True)
                robots = [
                    {'id': robot_id, 'base_serial': f'base{robot_id}',
                     'lidar_serial': f'lidar{robot_id}'}
                    for robot_id in range(team_size)]
                manifest = {
                    'run_id': run_id, 'method': 'mso',
                    'team_size': team_size, 'trial_id': entry['trial_id'],
                    'site_id': 'site_a', 'arena_id': 'arena_a',
                    'git_commit': 'a' * 40,
                    'checkpoint_sha256': 'b' * 64,
                    'collection_mode': 'physical', 'robots': robots,
                    'ground_truth_reference': {
                        'calibration_id': 'cal_v1',
                        'calibration_sha256': 'c' * 64,
                        'extrinsics_id': 'ext_v1',
                        'extrinsics_sha256': 'd' * 64,
                        'uncertainty_translation_m': 0.01,
                        'uncertainty_yaw_deg': 0.2},
                    'protocol_lock': {
                        'stopping_rule_id': 'stop_v1',
                        'motion_limits_id': 'motion_v1',
                        'sensing_policy_id': 'sense_v1',
                        'reference_timestamp_tolerance_ms': 100.0},
                    'campaign': {
                        'campaign_id': plan['campaign_id'],
                        'condition_id': condition,
                        'pair_id': entry['pair_id'],
                        'randomization_order': entry['order'],
                        'start_pose_set_id': entry['start_pose_set_id'],
                        'network_profile': entry['network_profile'],
                        'campaign_plan_sha256': plan_hash,
                        'network_profile_sha256': profile_hash},
                }
                audit = {
                    'protocol_diagnostic_pass': True,
                    'evidence_class': 'physical',
                    'bag_start_ns': 2_000_000_000_000_000_000 + entry['order'],
                    'network': ({
                        'rule_execution_pass': True,
                        'traffic_measurement_pass': True,
                        'physical_impairment_evidence': True}
                        if condition == 'impaired'
                        else {'executed_event_count': 0}),
                }
                (run_dir / 'manifest.json').write_text(
                    json.dumps(manifest), encoding='utf-8')
                (run_dir / 'analysis' / 'run_audit.json').write_text(
                    json.dumps(audit), encoding='utf-8')
            output = root / 'output'
            result = subprocess.run([
                sys.executable, str(CAMPAIGN_AUDIT),
                '--campaign-root', str(root), '--plan', str(PLAN),
                '--network-profile', str(PROFILE),
                '--output-dir', str(output),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report['minimum_campaign_complete'])
            self.assertFalse(report['claim_authorized'])

            for manifest_path in root.glob('*/manifest.json'):
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                manifest['site_id'] = ''
                manifest['arena_id'] = {}
                manifest['git_commit'] = ''
                manifest['checkpoint_sha256'] = []
                manifest['protocol_lock'] = {
                    'stopping_rule_id': '',
                    'motion_limits_id': {},
                    'sensing_policy_id': [],
                    'reference_timestamp_tolerance_ms': []}
                manifest['ground_truth_reference'][
                    'calibration_sha256'] = 'not-a-sha256'
                manifest['ground_truth_reference']['extrinsics_sha256'] = {}
                manifest['ground_truth_reference'][
                    'uncertainty_translation_m'] = []
                manifest['ground_truth_reference']['uncertainty_yaw_deg'] = {}
                manifest_path.write_text(
                    json.dumps(manifest), encoding='utf-8')
            rejected_output = root / 'empty_lock_output'
            rejected = subprocess.run([
                sys.executable, str(CAMPAIGN_AUDIT),
                '--campaign-root', str(root), '--plan', str(PLAN),
                '--network-profile', str(PROFILE),
                '--output-dir', str(rejected_output),
            ], capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 2)
            rejected_report = json.loads(rejected.stdout)
            self.assertFalse(rejected_report['minimum_campaign_complete'])
            self.assertTrue(any(
                'empty, or is malformed' in error
                for error in rejected_report['errors']))
            self.assertFalse(rejected_report['claim_authorized'])


if __name__ == '__main__':
    unittest.main()
