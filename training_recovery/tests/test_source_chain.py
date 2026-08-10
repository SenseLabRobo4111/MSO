from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_floorplan_dataset import ray_offsets, select_navigable_component  # noqa: E402
from generate_houseexpo_geometry_smoke import (  # noqa: E402
    SmokeConfig as HouseConfig,
    render_house,
)
from generate_kth_source_aligned_smoke import (  # noqa: E402
    SmokeConfig as KthConfig,
    generate_sample,
    restricted_starts,
)
from run_houseexpo_official_simulator_smoke import action_for_step  # noqa: E402


class SourceChainTests(unittest.TestCase):
    def test_house_renderer_uses_official_border_shape(self):
        payload = {
            "verts": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]
        }
        occupied, shape = render_house(payload, HouseConfig())
        self.assertEqual(shape, (38, 54))
        self.assertEqual(occupied.dtype, np.bool_)
        self.assertTrue(occupied[0, 0])
        self.assertFalse(occupied[11, 11])

    def test_visible_action_schedule_is_fixed(self):
        np.random.seed(7)
        actions = [action_for_step(step) for step in range(1, 15)]
        self.assertEqual(
            [index for index, action in enumerate(actions, start=1) if action == 0],
            [1, 2, 3, 4, 5, 7, 9, 11, 13],
        )
        self.assertTrue(all(action in {1, 2} for action in actions if action != 0))

    def test_prospective_sample_is_one_hot_and_known_exact(self):
        occupied = np.ones((900, 900), dtype=bool)
        occupied[100:800, 100:800] = False
        occupied[100:800, 445:455] = True
        occupied[430:470, 445:455] = False
        component, _ = select_navigable_component(occupied)
        starts = restricted_starts(component)
        config = KthConfig(
            raw_size=128,
            model_size=64,
            laser_range_cells=80,
            laser_rays=90,
            actions_before_capture=6,
            forward_step_cells=8,
            minimum_observed_obstacle_fraction=0.0,
            minimum_observed_free_fraction=0.0,
            maximum_free_to_obstacle_ratio=float("inf"),
        )
        generated = generate_sample(
            occupied,
            component,
            starts,
            ray_offsets(config.laser_rays, config.laser_range_cells),
            np.random.default_rng(11),
            config,
        )
        self.assertIsNotNone(generated)
        observation, target, _ = generated
        self.assertTrue(np.all(np.sum(observation == 255, axis=2) == 1))
        known = observation[:, :, 1] == 0
        measured_occupied = observation[:, :, 0] == 255
        self.assertTrue(np.array_equal(measured_occupied[known], (target == 255)[known]))


if __name__ == "__main__":
    unittest.main()
