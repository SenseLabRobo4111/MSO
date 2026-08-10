#!/usr/bin/env python3
"""Pre-capture manifest tests using only transient protocol fixtures."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PHYSICAL_TEAM = Path(__file__).resolve().parents[1]
PREPARE = PHYSICAL_TEAM / 'scripts' / 'prepare_run.py'
PLAN = PHYSICAL_TEAM / 'config' / 'minimum_campaign.yaml'
PROFILE = PHYSICAL_TEAM / 'config' / 'network_impairment.yaml'


class PrepareRunTest(unittest.TestCase):
    def test_manifest_embeds_campaign_profile_and_reference_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / 'deployed'
            repository.mkdir()
            subprocess.run(
                ['git', 'init', str(repository)], check=True,
                capture_output=True, text=True)
            (repository / 'tracked.txt').write_text('frozen\n', encoding='utf-8')
            subprocess.run(
                ['git', '-C', str(repository), 'add', 'tracked.txt'],
                check=True, capture_output=True, text=True)
            subprocess.run([
                'git', '-C', str(repository), '-c', 'user.name=Protocol Test',
                '-c', 'user.email=protocol@example.invalid', 'commit',
                '-m', 'frozen'], check=True, capture_output=True, text=True)

            config = root / 'team2.yaml'
            config.write_text(json.dumps({
                'team_size': 2, 'method': 'mso',
                'protocol_lock': {
                    'stopping_rule_id': 'stop_v1',
                    'motion_limits_id': 'motion_v1',
                    'sensing_policy_id': 'sense_v1',
                    'reference_timestamp_tolerance_ms': 100.0},
                'robots': [
                    {'id': 0, 'namespace': '/robot_0', 'host': 'robot0',
                     'base_serial': 'base0', 'lidar_serial': 'lidar0'},
                    {'id': 1, 'namespace': '/robot_1', 'host': 'robot1',
                     'base_serial': 'base1', 'lidar_serial': 'lidar1'}]}),
                encoding='utf-8')
            checkpoint = root / 'model.ckpt'
            checkpoint.write_bytes(b'synthetic checkpoint fixture')
            calibration = root / 'calibration.json'
            calibration.write_text(
                json.dumps({'camera': 'overhead', 'version': 1}),
                encoding='utf-8')
            extrinsics = root / 'extrinsics.json'
            extrinsics.write_text(
                json.dumps({'frames': ['world', 'robot_0', 'robot_1']}),
                encoding='utf-8')
            output = root / 'runs'
            command = [
                sys.executable, str(PREPARE), '--config', str(config),
                '--checkpoint', str(checkpoint),
                '--run-id', 'mso_n2_reference_t01', '--site-id', 'site_a',
                '--arena-id', 'arena_a', '--trial-id', '1',
                '--operator', 'operator1',
                '--ground-truth-source', 'overhead_tracking',
                '--reference-calibration', str(calibration),
                '--calibration-id', 'cal_v1',
                '--reference-extrinsics', str(extrinsics),
                '--extrinsics-id', 'ext_v1',
                '--uncertainty-translation-m', '0.01',
                '--uncertainty-yaw-deg', '0.2',
                '--repo-root', str(repository), '--output-root', str(output),
                '--campaign-id', 'mso_physical_minimum_v1',
                '--campaign-plan', str(PLAN), '--condition-id', 'reference',
                '--pair-id', 'pair_01', '--randomization-order', '4',
                '--start-pose-set-id', 'poses_01',
                '--network-profile', 'none',
                '--network-profile-file', str(PROFILE)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((
                output / 'mso_n2_reference_t01' / 'manifest.json').read_text(
                    encoding='utf-8'))
            self.assertEqual(manifest['manifest_stage'], 'pre_capture')
            self.assertEqual(
                manifest['campaign']['campaign_plan_sha256'],
                hashlib.sha256(PLAN.read_bytes()).hexdigest())
            self.assertEqual(
                manifest['campaign']['network_profile_sha256'],
                hashlib.sha256(PROFILE.read_bytes()).hexdigest())
            self.assertEqual(
                manifest['ground_truth_reference']['calibration_sha256'],
                hashlib.sha256(calibration.read_bytes()).hexdigest())
            self.assertEqual(
                manifest['ground_truth_reference']['extrinsics_sha256'],
                hashlib.sha256(extrinsics.read_bytes()).hexdigest())

            bad_command = list(command)
            bad_command[bad_command.index('--campaign-id') + 1] = ''
            bad_command[bad_command.index('--output-root') + 1] = str(
                root / 'empty_campaign_runs')
            rejected = subprocess.run(
                bad_command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('invalid campaign-id', rejected.stderr)
            self.assertFalse((root / 'empty_campaign_runs').exists())

    def test_empty_identifiers_fail_before_capture(self):
        import importlib.util
        specification = importlib.util.spec_from_file_location(
            'prepare_physical_run', PREPARE)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        with self.assertRaisesRegex(ValueError, 'invalid calibration-id'):
            module.require_identifier('calibration-id', '')


if __name__ == '__main__':
    unittest.main()
