#!/usr/bin/env python3
"""Four-robot shared protocol tests using transient fixtures."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import jsonschema
import yaml


PHYSICAL_TEAM = Path(__file__).resolve().parents[1]
REPOSITORY = PHYSICAL_TEAM.parents[1]
TEAM_CONFIG = PHYSICAL_TEAM / 'config' / 'team_4.yaml'
PREPARE = PHYSICAL_TEAM / 'scripts' / 'prepare_run.py'
PREFLIGHT = PHYSICAL_TEAM / 'scripts' / 'preflight.py'
MANIFEST_SCHEMA = PHYSICAL_TEAM / 'schema' / 'run_manifest.schema.json'
CAPTURE_LAUNCH = REPOSITORY / 'launch' / 'physical_team_capture.launch.py'
TEAM_LAUNCH = REPOSITORY / 'launch' / 'physical_team_4.launch.py'


def load_module(name, path):
    """Load a script module without requiring ROS package installation."""
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def resolved_config():
    """Return a runnable four-robot config derived from the template."""
    config = yaml.safe_load(TEAM_CONFIG.read_text(encoding='utf-8'))
    config['protocol_lock']['stopping_rule_id'] = 'fixed_600s_v1'
    config['protocol_lock']['motion_limits_id'] = 'sensebeetle_safe_v1'
    config['protocol_lock']['sensing_policy_id'] = 'lidar_fixed_v1'
    for robot in config['robots']:
        robot_id = robot['id']
        robot['host'] = f'robot{robot_id}'
        robot['base_serial'] = f'base{robot_id}'
        robot['lidar_serial'] = f'lidar{robot_id}'
    return config


class FourRobotProtocolTest(unittest.TestCase):
    def test_template_and_topic_expansion_cover_exactly_four_robots(self):
        config = resolved_config()
        self.assertEqual(config['team_size'], 4)
        self.assertEqual(
            [robot['id'] for robot in config['robots']], [0, 1, 2, 3])
        self.assertEqual(
            [robot['namespace'] for robot in config['robots']],
            ['/robot_0', '/robot_1', '/robot_2', '/robot_3'])

        preflight = load_module('n4_preflight', PREFLIGHT)
        topics = set(preflight.required_topics(config))
        for robot_id in range(4):
            self.assertIn(f'/robot_{robot_id}/lidar/points', topics)
            self.assertIn(
                f'/robot_{robot_id}/provisional_predicted_map', topics)
            self.assertIn(f'/ground_truth/robot_{robot_id}/pose', topics)
        self.assertNotIn('/robot_4/lidar/points', topics)

    def test_manifest_schema_accepts_four_robot_roster(self):
        config = resolved_config()
        schema = json.loads(MANIFEST_SCHEMA.read_text(encoding='utf-8'))
        manifest = {
            'schema_version': '1.0',
            'run_id': 'mso_n4_reference_t01',
            'method': 'mso',
            'team_size': 4,
            'site_id': 'site_a',
            'arena_id': 'arena_a',
            'trial_id': 1,
            'operator': 'operator1',
            'created_utc': '2026-08-11T00:00:00+00:00',
            'git_commit': 'a' * 40,
            'checkpoint_sha256': 'b' * 64,
            'config_sha256': 'c' * 64,
            'ground_truth_source': 'overhead_tracking',
            'ground_truth_reference': {
                'calibration_id': 'cal_v1',
                'calibration_path': 'evidence/reference_calibration.json',
                'calibration_sha256': 'd' * 64,
                'extrinsics_id': 'ext_v1',
                'extrinsics_path': 'evidence/reference_extrinsics.json',
                'extrinsics_sha256': 'e' * 64,
                'uncertainty_translation_m': 0.01,
                'uncertainty_yaw_deg': 0.2,
            },
            'robots': config['robots'],
            'collection_mode': 'physical',
            'evidence_status': 'unverified_pre_capture',
        }
        jsonschema.Draft202012Validator(schema).validate(manifest)

    def test_prepare_run_accepts_resolved_four_robot_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / 'deployed'
            repository.mkdir()
            subprocess.run(
                ['git', 'init', str(repository)], check=True,
                capture_output=True, text=True)
            (repository / 'tracked.txt').write_text(
                'frozen\n', encoding='utf-8')
            subprocess.run(
                ['git', '-C', str(repository), 'add', 'tracked.txt'],
                check=True, capture_output=True, text=True)
            subprocess.run([
                'git', '-C', str(repository), '-c',
                'user.name=Protocol Test',
                '-c', 'user.email=protocol@example.invalid', 'commit',
                '-m', 'frozen'], check=True, capture_output=True, text=True)

            config_path = root / 'team4.yaml'
            config_path.write_text(
                yaml.safe_dump(resolved_config(), sort_keys=False),
                encoding='utf-8')
            checkpoint = root / 'model.pt'
            checkpoint.write_bytes(b'synthetic checkpoint fixture')
            calibration = root / 'calibration.json'
            calibration.write_text(
                json.dumps({'camera': 'overhead', 'version': 1}),
                encoding='utf-8')
            extrinsics = root / 'extrinsics.json'
            extrinsics.write_text(
                json.dumps({'frames': [
                    'world', 'robot_0', 'robot_1', 'robot_2', 'robot_3']}),
                encoding='utf-8')
            output = root / 'runs'
            command = [
                sys.executable, str(PREPARE),
                '--config', str(config_path),
                '--checkpoint', str(checkpoint),
                '--run-id', 'mso_n4_reference_t01',
                '--site-id', 'site_a', '--arena-id', 'arena_a',
                '--trial-id', '1', '--operator', 'operator1',
                '--ground-truth-source', 'overhead_tracking',
                '--reference-calibration', str(calibration),
                '--calibration-id', 'cal_v1',
                '--reference-extrinsics', str(extrinsics),
                '--extrinsics-id', 'ext_v1',
                '--uncertainty-translation-m', '0.01',
                '--uncertainty-yaw-deg', '0.2',
                '--repo-root', str(repository),
                '--output-root', str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(
                result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((
                output / 'mso_n4_reference_t01' / 'manifest.json'
            ).read_text(encoding='utf-8'))
            self.assertEqual(manifest['team_size'], 4)
            self.assertEqual(len(manifest['robots']), 4)

    def test_capture_launch_and_wrapper_declare_four_robots(self):
        capture_source = CAPTURE_LAUNCH.read_text(encoding='utf-8')
        wrapper_source = TEAM_LAUNCH.read_text(encoding='utf-8')
        self.assertIn('team_size not in (2, 3, 4, 5)', capture_source)
        self.assertIn("'team_size': '4'", wrapper_source)


if __name__ == '__main__':
    unittest.main()
