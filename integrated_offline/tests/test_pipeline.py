from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import tempfile

import numpy as np

from integrated_offline.run_paired_ablation import (
    UNKNOWN_VALUE,
    atomic_measured_commit,
    choose_start,
    cluster_metric_distribution,
    evaluation_role,
    exact_sign_p,
    load_exploratory_manifest,
    metric_to_pixel,
    pixel_to_metric_transform,
    plan_frontier,
    route_against_oracle,
    state_hash,
)
from integrated_offline.reference_registrar import FeaturePool, _candidate_matches
from integrated_offline.run_paired_ablation import run as run_paired_audit
from integrated_offline.run_provisional_planning_ablation import (
    measured_authoritative_overlay,
    merge_probability_support,
    prediction_to_categorical,
)
from integrated_offline.run_frontier_ranking_ablation import (
    disk_kernel,
    select_ranked_goal,
)


class AtomicCommitTests(unittest.TestCase):
    def test_rejection_is_hash_stable_and_non_aliasing(self) -> None:
        persistent = np.array(
            [[0, UNKNOWN_VALUE], [255, UNKNOWN_VALUE]], dtype=np.uint8
        )
        source = np.array([[255, 0], [0, 255]], dtype=np.uint8)
        before = state_hash(persistent)
        after, committed, changed = atomic_measured_commit(
            persistent, source, accepted=False
        )
        self.assertFalse(committed)
        self.assertEqual(changed, 0)
        self.assertEqual(state_hash(after), before)
        after[0, 0] = 123
        self.assertEqual(int(persistent[0, 0]), 0)

    def test_acceptance_fills_only_unknown_target_cells(self) -> None:
        persistent = np.array(
            [[0, UNKNOWN_VALUE], [255, UNKNOWN_VALUE]], dtype=np.uint8
        )
        source = np.array([[255, 0], [0, 255]], dtype=np.uint8)
        after, committed, changed = atomic_measured_commit(
            persistent, source, accepted=True
        )
        self.assertTrue(committed)
        self.assertEqual(changed, 2)
        np.testing.assert_array_equal(after, [[0, 0], [255, 255]])


class GeometryTests(unittest.TestCase):
    def test_metric_pixel_conversion_round_trip(self) -> None:
        angle = np.deg2rad(17.0)
        transform = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.7],
                [np.sin(angle), np.cos(angle), -1.2],
                [0.0, 0.0, 1.0],
            ]
        )
        event = {
            "canvas_width_px": 80,
            "canvas_height_px": 60,
            "source_canvas_width_px": 80,
            "source_canvas_height_px": 60,
            "resolution_m_per_px": 0.05,
        }
        pixel = metric_to_pixel(transform, event)
        recovered = pixel_to_metric_transform(pixel, event)
        np.testing.assert_allclose(recovered, transform, atol=1e-12)


class MatcherRegressionTests(unittest.TestCase):
    def test_direct_fallback_accepts_tuple_for_three_descriptors(self) -> None:
        descriptors = np.zeros((3, 32), dtype=np.uint8)
        descriptors[1, 0] = 15
        descriptors[2, 0] = 240
        pool = FeaturePool(
            points=np.asarray([[0, 0], [10, 0], [0, 10]], dtype=np.float32),
            descriptors=descriptors,
            ambiguity=np.zeros(3, dtype=np.float32),
            angles=np.zeros(3, dtype=np.float32),
            sizes=np.ones(3, dtype=np.float32),
            families=np.zeros(3, dtype=np.int16),
        )
        source, target, distances = _candidate_matches(pool, pool)
        self.assertEqual(len(source), 3)
        self.assertEqual(len(target), 3)
        self.assertEqual(len(distances), 3)
        np.testing.assert_array_equal(source, [0, 1, 2])
        np.testing.assert_array_equal(target, [0, 1, 2])


class ClusterAwareStatisticsTests(unittest.TestCase):
    def test_cluster_means_not_events_are_summary_units(self) -> None:
        rows = [
            {"base_frame_cluster": "a", "value": 1.0},
            {"base_frame_cluster": "a", "value": 3.0},
            {"base_frame_cluster": "b", "value": 10.0},
        ]
        summary = cluster_metric_distribution(rows, "value")
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["mean"], 6.0)

    def test_frontier_cluster_sign_reference_value(self) -> None:
        self.assertAlmostEqual(exact_sign_p(17, 9), 0.16863754391670227)

    def test_low_support_case_is_indeterminate_not_known_error(self) -> None:
        role, known_error = evaluation_role(
            {
                "gt_positive": False,
                "negative_kind": "temporal_mismatch_low_support_iou",
            }
        )
        self.assertEqual(role, "indeterminate_abstention_challenge")
        self.assertFalse(known_error)


class PortableBundleReplayTests(unittest.TestCase):
    def test_public_tree_supports_relative_manifest_and_one_pair_replay(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        source = (
            repository
            / "paper/nature_communications/peer_review_data/transform_validation"
        )
        if not source.is_dir():
            self.skipTest("public peer-review data are not present in this checkout")
        self.assertGreater(sum(path.is_file() for path in source.rglob("*")), 400)
        with tempfile.TemporaryDirectory() as temporary:
            extract_root = Path(temporary)
            portable = extract_root / "peer_review_data/transform_validation"
            shutil.copytree(source, portable)
            manifest_path = (
                portable / "manifest.json"
            )
            _, _, events = load_exploratory_manifest(manifest_path)
            self.assertEqual(len(events), 300)
            self.assertTrue(Path(events[0]["target_path"]).is_file())
            output = extract_root / "replay_output"
            summary = run_paired_audit(manifest_path, output, limit=1)
            self.assertEqual(summary["cluster_aware_paired"]["clusters"], 1)
            self.assertTrue((output / "SHA256SUMS").is_file())


class PlannerTests(unittest.TestCase):
    def test_planner_reaches_a_frontier_and_oracle_detects_collision(self) -> None:
        grid = np.full((15, 15), UNKNOWN_VALUE, dtype=np.uint8)
        grid[3:12, 3:12] = 0
        grid[7, 7] = 255
        start = choose_start(grid)
        self.assertIsNotNone(start)
        result, route = plan_frontier(grid, start, 0.05)
        self.assertTrue(result["route_available"])
        self.assertGreater(result["reachable_free_cells"], 0)
        self.assertGreater(result["reachable_frontiers"], 0)
        self.assertGreater(len(route), 1)

        oracle = grid.copy()
        y, x = route[len(route) // 2]
        oracle[y, x] = 255
        safety = route_against_oracle(route, oracle)
        self.assertFalse(safety["route_oracle_safe"])
        self.assertGreater(safety["route_oracle_collision_cells"], 0)


class ProvisionalLayerTests(unittest.TestCase):
    def test_zero_probability_is_unsupported_and_threshold_is_fixed(self) -> None:
        raw = np.array([[0, 1, 127, 128, 255]], dtype=np.uint8)
        categorical = prediction_to_categorical(raw)
        np.testing.assert_array_equal(
            categorical,
            [[UNKNOWN_VALUE, 0, 0, 255, 255]],
        )

    def test_probability_union_uses_supported_maximum(self) -> None:
        target = np.array([[0, 10, 220, 70]], dtype=np.uint8)
        source = np.array([[0, 80, 120, 0]], dtype=np.uint8)
        merged = merge_probability_support(target, source)
        np.testing.assert_array_equal(merged, [[0, 80, 220, 70]])

    def test_measured_cells_override_and_provisional_cells_remain_disposable(self) -> None:
        provisional = np.array(
            [[0, 255], [255, 0]], dtype=np.uint8
        )
        measured = np.array(
            [[UNKNOWN_VALUE, 0], [255, UNKNOWN_VALUE]], dtype=np.uint8
        )
        measured_before = state_hash(measured)
        planner_grid = measured_authoritative_overlay(provisional, measured)
        np.testing.assert_array_equal(planner_grid, [[0, 0], [255, 0]])
        self.assertEqual(state_hash(measured), measured_before)


class FrontierRankingTests(unittest.TestCase):
    def test_one_cell_radius_kernel_is_a_cross(self) -> None:
        kernel = disk_kernel(1)
        np.testing.assert_array_equal(
            kernel,
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        )

    def test_prediction_score_precedes_distance_and_ties_use_distance(self) -> None:
        frontier = np.zeros((3, 5), dtype=bool)
        frontier[1, 1] = True
        frontier[1, 3] = True
        distance = np.zeros((3, 5), dtype=np.int32)
        distance[1, 1] = 8
        distance[1, 3] = 3
        scores = np.zeros((3, 5), dtype=np.float32)
        scores[1, 1] = 2
        scores[1, 3] = 5
        self.assertEqual(
            select_ranked_goal(frontier, distance, scores),
            (1, 3),
        )
        scores[1, 1] = 5
        self.assertEqual(
            select_ranked_goal(frontier, distance, scores),
            (1, 1),
        )


if __name__ == "__main__":
    unittest.main()
