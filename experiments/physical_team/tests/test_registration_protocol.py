#!/usr/bin/env python3
"""Registration audit tests with synthetic labels; never experimental data."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


AUDIT_PATH = Path(__file__).resolve().parents[1] / 'analysis' / 'audit_run.py'
SPEC = importlib.util.spec_from_file_location('physical_audit_run', AUDIT_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def event(event_id, state, timestamp_ns, estimate=(0.0, 0.0, 0.0), **extra):
    """Build one synthetic lifecycle event."""
    decision = {
        'candidate': 'not_applicable', 'accepted': 'accept',
        'rejected': 'reject', 'committed': 'accept',
        'recovered': 'not_applicable'}[state]
    value = {
        'schema_version': '1.0', 'event_id': event_id,
        'run_id': 'synthetic_registration', 'timestamp_ns': timestamp_ns,
        'source_robot_id': 0, 'target_robot_id': 1, 'state': state,
        'estimate': {
            'tx_m': estimate[0], 'ty_m': estimate[1],
            'yaw_deg': estimate[2]},
        'gate': {'decision': decision, 'reason': 'synthetic'},
    }
    value.update(extra)
    return value


def reference(event_id, tx=0.0, ty=0.0, yaw=0.0):
    return {
        'event_id': event_id, 'gt_tx_m': tx, 'gt_ty_m': ty,
        'gt_yaw_deg': yaw}


class RegistrationProtocolTest(unittest.TestCase):
    limits = {'translation_limit_m': 0.15, 'yaw_limit_deg': 5.0}

    def test_only_correct_committed_edges_connect_fleet(self):
        candidate = event('edge1', 'candidate', 1)
        accepted = event(
            'edge1', 'accepted', 2, map_revision_before=1,
            map_revision_after=1, map_hash_before='a' * 64,
            map_hash_after='a' * 64)
        accepted_only = AUDIT.registration_metrics(
            [candidate, accepted], [reference('edge1')], self.limits, 5.0)
        self.assertEqual(accepted_only['connected_robot_ids'], [])
        committed = event(
            'edge1', 'committed', 3, map_revision_before=1,
            map_revision_after=2, map_hash_before='a' * 64,
            map_hash_after='b' * 64)
        summary = AUDIT.registration_metrics(
            [candidate, accepted, committed], [reference('edge1')],
            self.limits, 5.0)
        self.assertEqual(summary['connected_robot_ids'], [0, 1])
        self.assertEqual(summary['correct_committed_pairs'], [[0, 1]])

    def test_empty_and_duplicate_ground_truth_labels_are_rejected(self):
        identifiers, duplicates, empty_count = AUDIT.reference_identifier_errors([
            {'event_id': 'edge1'}, {'event_id': 'edge1'}, {'event_id': ''}])
        self.assertEqual(identifiers, ['edge1', 'edge1', ''])
        self.assertEqual(duplicates, ['edge1'])
        self.assertEqual(empty_count, 1)

    def test_pseudo_ground_truth_exposes_false_accept_and_wrong_commit(self):
        events = [
            event('edge_bad', 'candidate', 1, estimate=(1.0, 0.0, 0.0)),
            event(
                'edge_bad', 'accepted', 2, estimate=(1.0, 0.0, 0.0),
                map_revision_before=4, map_revision_after=4,
                map_hash_before='a' * 64, map_hash_after='a' * 64),
            event(
                'edge_bad', 'committed', 3, estimate=(1.0, 0.0, 0.0),
                map_revision_before=4, map_revision_after=5,
                map_hash_before='a' * 64, map_hash_after='b' * 64),
            event(
                'recovery1', 'recovered', 4, estimate=(1.0, 0.0, 0.0),
                parent_event_id='edge_bad', map_revision_before=5,
                map_revision_after=6, map_hash_before='b' * 64,
                map_hash_after='c' * 64,
                recovery_verification={
                    'invalidated_commit_event_id': 'edge_bad',
                    'invalidated_revision': 5,
                    'invalidated_map_hash': 'b' * 64,
                    'active_revision': 6, 'active_map_hash': 'c' * 64,
                    'method': 'rollback', 'verified_by': 'independent_map'})]
        summary = AUDIT.registration_metrics(
            events, [reference('edge_bad')], self.limits, 5.0)
        self.assertEqual(summary['false_accept'], 1)
        self.assertEqual(summary['wrong_commits'], 1)
        self.assertEqual(summary['recovered_wrong_commits'], 1)
        self.assertEqual(summary['connected_robot_ids'], [])
        self.assertFalse(AUDIT.registration_lifecycle_errors(events))

    def test_reject_map_mutation_and_unproven_recovery_fail(self):
        rejected = [
            event('reject1', 'candidate', 1),
            event(
                'reject1', 'rejected', 2, map_revision_before=2,
                map_revision_after=2, map_hash_before='a' * 64,
                map_hash_after='b' * 64)]
        reasons = {
            row['reason'] for row in AUDIT.registration_lifecycle_errors(rejected)}
        self.assertIn('rejected decision changed map hash', reasons)

        wrong = [
            event('edge_bad', 'candidate', 1, estimate=(1.0, 0.0, 0.0)),
            event(
                'edge_bad', 'accepted', 2, estimate=(1.0, 0.0, 0.0),
                map_revision_before=1, map_revision_after=1,
                map_hash_before='a' * 64, map_hash_after='a' * 64),
            event(
                'edge_bad', 'committed', 3, estimate=(1.0, 0.0, 0.0),
                map_revision_before=1, map_revision_after=2,
                map_hash_before='a' * 64, map_hash_after='b' * 64),
            event(
                'bad_recovery', 'recovered', 4, parent_event_id='edge_bad',
                map_revision_before=2, map_revision_after=3,
                map_hash_before='b' * 64, map_hash_after='b' * 64,
                recovery_verification={
                    'invalidated_commit_event_id': 'edge_bad',
                    'invalidated_revision': 2,
                    'invalidated_map_hash': 'b' * 64,
                    'active_revision': 3, 'active_map_hash': 'b' * 64,
                    'method': 'rollback', 'verified_by': 'synthetic'})]
        lifecycle = AUDIT.registration_lifecycle_errors(wrong)
        self.assertTrue(lifecycle)
        summary = AUDIT.registration_metrics(
            wrong, [reference('edge_bad')], self.limits, 5.0)
        self.assertEqual(summary['recovered_wrong_commits'], 0)
        self.assertTrue(summary['recovery_errors'])

    def test_recovery_requires_method_and_independent_verifier(self):
        committed = event(
            'edge_bad', 'committed', 3, estimate=(1.0, 0.0, 0.0),
            map_revision_before=1, map_revision_after=2,
            map_hash_before='a' * 64, map_hash_after='b' * 64)
        base = event(
            'recovery1', 'recovered', 4, parent_event_id='edge_bad',
            map_revision_before=2, map_revision_after=3,
            map_hash_before='b' * 64, map_hash_after='c' * 64,
            recovery_verification={
                'invalidated_commit_event_id': 'edge_bad',
                'invalidated_revision': 2,
                'invalidated_map_hash': 'b' * 64,
                'active_revision': 3,
                'active_map_hash': 'c' * 64,
                'method': 'rollback',
                'verified_by': 'independent_tracking',
                'evidence_path': 'evidence/recovery/recovery1.json',
                'evidence_sha256': 'd' * 64})
        for field in ('method', 'verified_by'):
            recovery = json.loads(json.dumps(base))
            del recovery['recovery_verification'][field]
            reasons = {
                row['reason'] for row in AUDIT.registration_lifecycle_errors(
                    [committed, recovery])}
            self.assertIn('recovery method or verifier is missing', reasons)
            validators = AUDIT.build_schema_validators()
            self.assertTrue(AUDIT.validate_schema(
                recovery, validators['registration'], 'synthetic'))

    def test_hashed_in_run_post_recovery_map_state_evidence(self):
        validators = AUDIT.build_schema_validators()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            evidence_dir = run_dir / 'evidence' / 'recovery'
            evidence_dir.mkdir(parents=True)
            recovery = event(
                'recovery1', 'recovered', 40,
                parent_event_id='edge_bad', map_revision_before=2,
                map_revision_after=3, map_hash_before='b' * 64,
                map_hash_after='c' * 64)
            payload = {
                'schema_version': '1.0',
                'run_id': 'synthetic_registration',
                'recovery_event_id': 'recovery1',
                'parent_event_id': 'edge_bad',
                'source_robot_id': 0,
                'target_robot_id': 1,
                'observed_timestamp_ns': 41,
                'active_revision': 3,
                'active_map_hash': 'c' * 64,
                'method': 'rollback',
                'verified_by': 'independent_tracking'}
            path = evidence_dir / 'recovery1.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            recovery['recovery_verification'] = {
                'invalidated_commit_event_id': 'edge_bad',
                'invalidated_revision': 2,
                'invalidated_map_hash': 'b' * 64,
                'active_revision': 3,
                'active_map_hash': 'c' * 64,
                'method': 'rollback',
                'verified_by': 'independent_tracking',
                'evidence_path': 'evidence/recovery/recovery1.json',
                'evidence_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertFalse(AUDIT.recovery_evidence_errors(
                [recovery], run_dir, 'synthetic_registration',
                validators['recovery_evidence']))

            payload['target_robot_id'] = 2
            path.write_text(json.dumps(payload), encoding='utf-8')
            recovery['recovery_verification'][
                'evidence_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            mismatch = AUDIT.recovery_evidence_errors(
                [recovery], run_dir, 'synthetic_registration',
                validators['recovery_evidence'])
            self.assertTrue(any('pair' in row['reason'] for row in mismatch))

            payload['target_robot_id'] = 1
            payload['observed_timestamp_ns'] = 40
            path.write_text(json.dumps(payload), encoding='utf-8')
            recovery['recovery_verification'][
                'evidence_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            stale = AUDIT.recovery_evidence_errors(
                [recovery], run_dir, 'synthetic_registration',
                validators['recovery_evidence'])
            self.assertTrue(any('after recovery' in row['reason'] for row in stale))

            outside = run_dir.parent / 'outside_recovery.json'
            outside.write_text(json.dumps(payload), encoding='utf-8')
            recovery['recovery_verification'][
                'evidence_path'] = '../outside_recovery.json'
            recovery['recovery_verification'][
                'evidence_sha256'] = hashlib.sha256(
                    outside.read_bytes()).hexdigest()
            outside_errors = AUDIT.recovery_evidence_errors(
                [recovery], run_dir, 'synthetic_registration',
                validators['recovery_evidence'])
            self.assertTrue(any(
                'outside run evidence' in row['reason']
                for row in outside_errors))
            outside.unlink()

    def test_reference_timestamp_must_be_positive_and_within_frozen_window(self):
        accepted = event(
            'edge1', 'accepted', 200, map_revision_before=1,
            map_revision_after=1, map_hash_before='a' * 64,
            map_hash_after='a' * 64)
        committed = event(
            'edge1', 'committed', 210, map_revision_before=1,
            map_revision_after=2, map_hash_before='a' * 64,
            map_hash_after='b' * 64)
        grouped = {'edge1': [accepted, committed]}
        valid = {'event_id': 'edge1', 'timestamp_ns': 205}
        self.assertFalse(AUDIT.reference_timestamp_errors(
            [valid], grouped, tolerance_ns=5))
        self.assertEqual(len(AUDIT.reference_timestamp_errors(
            [valid], grouped, tolerance_ns=4)), 2)
        invalid = {'event_id': 'edge1', 'timestamp_ns': 0}
        self.assertTrue(AUDIT.reference_timestamp_errors(
            [invalid], grouped, tolerance_ns=5))

    def test_all_declared_json_schemas_compile(self):
        self.assertEqual(
            set(AUDIT.build_schema_validators()), set(AUDIT.SCHEMA_FILES))

    def test_audit_run_rejects_empty_locks_and_malformed_hashes(self):
        manifest = {
            'site_id': 'site_a', 'arena_id': 'arena_a',
            'git_commit': 'a' * 40, 'checkpoint_sha256': 'b' * 64,
            'protocol_lock': {
                'stopping_rule_id': 'stop_v1',
                'motion_limits_id': 'motion_v1',
                'sensing_policy_id': 'sense_v1',
                'reference_timestamp_tolerance_ms': 100.0},
            'ground_truth_reference': {
                'calibration_sha256': 'c' * 64,
                'extrinsics_sha256': 'd' * 64,
                'uncertainty_translation_m': 0.01,
                'uncertainty_yaw_deg': 0.2}}
        self.assertFalse(AUDIT.frozen_lock_errors(manifest))
        manifest['protocol_lock']['stopping_rule_id'] = ''
        manifest['protocol_lock']['motion_limits_id'] = {}
        manifest['protocol_lock'][
            'reference_timestamp_tolerance_ms'] = []
        manifest['checkpoint_sha256'] = 'not-a-sha256'
        invalid = {
            row['field'] for row in AUDIT.frozen_lock_errors(manifest)}
        self.assertTrue({
            'stopping_rule_id', 'motion_limits_id',
            'reference_timestamp_tolerance_ms',
            'checkpoint_sha256'}.issubset(invalid))


if __name__ == '__main__':
    unittest.main()
