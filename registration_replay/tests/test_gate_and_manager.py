from __future__ import annotations

import unittest

import numpy as np

from registration_replay.pairwise_manager import (
    MapObservation,
    MatchCandidate,
    PairwiseRegistrationManager,
)
from registration_replay.se2_gate import CandidateDiagnostics, GateThresholds, gate_se2_candidate
from registration_replay.rigid_estimator import estimate_se2_ransac
from registration_replay.replay_physical_bags import _apply_temporal_consensus


GOOD = CandidateDiagnostics(
    inlier_count=40,
    inlier_ratio=0.4,
    median_residual_px=1.0,
    occupancy_iou=0.5,
    occupancy_chamfer_median_px=1.0,
    source_minor_spread_px=15.0,
    target_minor_spread_px=15.0,
)


class GateTests(unittest.TestCase):
    def test_valid_candidate_is_projected_to_se2(self) -> None:
        angle = np.deg2rad(25.0)
        affine = np.array(
            [[1.005 * np.cos(angle), -1.005 * np.sin(angle), 8.0],
             [1.005 * np.sin(angle), 1.005 * np.cos(angle), -3.0],
             [0.0, 0.0, 1.0]]
        )
        decision = gate_se2_candidate(affine, GOOD)
        self.assertTrue(decision.accepted)
        self.assertIsNotNone(decision.se2_image)
        rotation = decision.se2_image[:2, :2]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(2), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_scale_reflection_and_shear_are_rejected(self) -> None:
        cases = (
            np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        for affine in cases:
            with self.subTest(affine=affine):
                self.assertFalse(gate_se2_candidate(affine, GOOD).accepted)


class RigidEstimatorTests(unittest.TestCase):
    def test_recovers_unit_scale_transform_with_outliers(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.uniform(-50.0, 50.0, size=(80, 2))
        angle = np.deg2rad(-31.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        target = source @ rotation.T + np.array([17.0, -9.0])
        target += rng.normal(0.0, 0.25, size=target.shape)
        target[:20] = rng.uniform(-80.0, 80.0, size=(20, 2))
        estimate = estimate_se2_ransac(source, target, random_seed=11)
        self.assertIsNotNone(estimate.se2_image)
        self.assertGreaterEqual(estimate.inlier_count, 55)
        np.testing.assert_allclose(estimate.se2_image[:2, :2], rotation, atol=0.01)
        np.testing.assert_allclose(estimate.se2_image[:2, 2], [17.0, -9.0], atol=0.5)
        self.assertAlmostEqual(float(np.linalg.det(estimate.se2_image[:2, :2])), 1.0)

    def test_scaled_correspondences_do_not_turn_into_scaled_output(self) -> None:
        rng = np.random.default_rng(3)
        source = rng.uniform(-20.0, 20.0, size=(50, 2))
        target = 0.2 * source + np.array([5.0, 4.0])
        estimate = estimate_se2_ransac(source, target, random_seed=9)
        self.assertIsNotNone(estimate.se2_image)
        self.assertAlmostEqual(float(np.linalg.det(estimate.se2_image[:2, :2])), 1.0)
        self.assertLess(estimate.inlier_ratio, 0.25)


class ManagerTests(unittest.TestCase):
    def test_three_robot_queue_has_no_hard_wired_pair(self) -> None:
        committed = []

        def matcher(first, second):
            del first, second
            return MatchCandidate(np.eye(3), GOOD)

        manager = PairwiseRegistrationManager(
            ["alpha", "beta", "gamma"], matcher, committed.append, commit_enabled=True
        )
        image = np.zeros((8, 8), dtype=np.uint8)
        for i, robot_id in enumerate(manager.robot_ids):
            manager.update(MapObservation(robot_id, i, image, 0.05, (0.0, 0.0)))
        records = manager.drain()
        self.assertEqual(manager.configured_pairs(), (("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")))
        self.assertEqual(len(records), 3)
        self.assertEqual(len(committed), 0)
        self.assertTrue(all(record.consensus_support == 1 for record in records))

    def test_commit_is_fail_closed_by_default(self) -> None:
        committed = []

        def matcher(first, second):
            del first, second
            return MatchCandidate(np.eye(3), GOOD)

        manager = PairwiseRegistrationManager(["left", "right"], matcher, committed.append)
        image = np.zeros((8, 8), dtype=np.uint8)
        manager.update(MapObservation("left", 0, image, 0.05, (0.0, 0.0)))
        manager.update(MapObservation("right", 1, image, 0.05, (0.0, 0.0)))
        records = manager.drain()
        self.assertTrue(records[0].decision.accepted)
        self.assertFalse(records[0].committed)
        self.assertEqual(committed, [])

    def test_enabled_commit_still_requires_three_independent_confirmations(self) -> None:
        committed = []

        def matcher(first, second):
            del first, second
            return MatchCandidate(np.eye(3), GOOD)

        manager = PairwiseRegistrationManager(
            ["left", "right"], matcher, committed.append, commit_enabled=True
        )
        image = np.zeros((8, 8), dtype=np.uint8)
        records = []
        for revision in range(3):
            first = MapObservation(
                "left", int((revision + 1) * 1e9), image, 0.05, (0.0, 0.0), revision
            )
            second = MapObservation(
                "right", int((revision + 1) * 1e9), image, 0.05, (0.0, 0.0), revision
            )
            manager.enqueue(first, second)
            records.append(manager.process_one())
        self.assertEqual([record.consensus_support for record in records], [1, 2, 3])
        self.assertEqual([record.committed for record in records], [False, False, True])
        self.assertEqual(len(committed), 1)

    def test_local_reject_resets_online_pair_consensus(self) -> None:
        committed = []
        candidates = iter(
            [
                MatchCandidate(np.eye(3), GOOD),
                MatchCandidate(np.diag([0.1, 0.1, 1.0]), GOOD),
                MatchCandidate(np.eye(3), GOOD),
                MatchCandidate(np.eye(3), GOOD),
            ]
        )

        def matcher(first, second):
            del first, second
            return next(candidates)

        manager = PairwiseRegistrationManager(
            ["left", "right"], matcher, committed.append, commit_enabled=True
        )
        image = np.zeros((8, 8), dtype=np.uint8)
        supports = []
        for revision in range(4):
            manager.enqueue(
                MapObservation("left", int((revision + 1) * 1e9), image, 0.05, (0.0, 0.0), revision),
                MapObservation("right", int((revision + 1) * 1e9), image, 0.05, (0.0, 0.0), revision),
            )
            supports.append(manager.process_one().consensus_support)
        self.assertEqual(supports, [1, 0, 1, 2])
        self.assertEqual(committed, [])

    def test_online_consensus_is_isolated_per_robot_pair(self) -> None:
        committed = []

        def matcher(first, second):
            del first, second
            return MatchCandidate(np.eye(3), GOOD)

        manager = PairwiseRegistrationManager(
            ["a", "b", "c"], matcher, committed.append, commit_enabled=True
        )
        image = np.zeros((8, 8), dtype=np.uint8)
        for revision in range(2):
            stamp = int((revision + 1) * 1e9)
            manager.enqueue(
                MapObservation("a", stamp, image, 0.05, (0.0, 0.0), revision),
                MapObservation("b", stamp, image, 0.05, (0.0, 0.0), revision),
            )
            manager.enqueue(
                MapObservation("a", stamp, image, 0.05, (0.0, 0.0), revision),
                MapObservation("c", stamp, image, 0.05, (0.0, 0.0), revision),
            )
        records = manager.drain()
        supports = {(r.encounter.first.robot_id, r.encounter.second.robot_id): r.consensus_support for r in records[-2:]}
        self.assertEqual(supports[("a", "b")], 2)
        self.assertEqual(supports[("a", "c")], 2)
        self.assertEqual(committed, [])


class TemporalConsensusTests(unittest.TestCase):
    def test_reject_resets_and_both_map_revisions_are_required(self) -> None:
        def row(t, pair, accepted=True):
            return {
                "trigger_record_time_ns": int(t * 1e9),
                "robot_0_map_index": pair[0],
                "robot_1_map_index": pair[1],
                "accept": int(accepted),
                "local_reason": "rejected" if not accepted else "",
                "x": 1.0,
                "y": 2.0,
                "yaw": 10.0,
                "support": 0,
                "commit": 0,
                "reject": 1,
                "reason": "",
            }

        rows = [
            row(0.0, (0, 0)),
            row(1.0, (1, 0)),  # only one cached map changed
            row(2.0, (1, 1)),
            row(3.0, (2, 2), accepted=False),
            row(4.0, (2, 2)),
            row(5.0, (3, 3)),
            row(6.0, (4, 4)),
        ]
        _apply_temporal_consensus(
            rows,
            local_accept_key="accept",
            local_reason_key="local_reason",
            estimate_x_key="x",
            estimate_y_key="y",
            estimate_yaw_key="yaw",
            support_key="support",
            commit_key="commit",
            reject_key="reject",
            reason_key="reason",
        )
        self.assertEqual(rows[1]["reason"], "nonindependent_confirmation")
        self.assertEqual(rows[2]["support"], 2)
        self.assertEqual(rows[4]["support"], 1)
        self.assertEqual(rows[5]["support"], 2)
        self.assertEqual(rows[6]["support"], 3)
        self.assertEqual(rows[6]["commit"], 1)

    def test_rejected_candidate_never_reaches_committer(self) -> None:
        committed = []

        def matcher(first, second):
            del first, second
            return MatchCandidate(np.diag([0.1, 0.1, 1.0]), GOOD)

        manager = PairwiseRegistrationManager(["r2", "r7"], matcher, committed.append)
        image = np.zeros((8, 8), dtype=np.uint8)
        manager.update(MapObservation("r2", 0, image, 0.05, (0.0, 0.0)))
        manager.update(MapObservation("r7", 1, image, 0.05, (0.0, 0.0)))
        records = manager.drain()
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].decision.accepted)
        self.assertEqual(committed, [])


if __name__ == "__main__":
    unittest.main()
