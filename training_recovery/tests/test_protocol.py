from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_floorplan_dataset import (  # noqa: E402
    GenerationConfig,
    assign_splits,
    building_id_from_floorplan,
    generate_sample,
    ray_offsets,
    select_navigable_component,
)
from mso_recovery.objective import unknown_bce  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_building_identifier_keeps_layouts_together(self):
        self.assertEqual(building_id_from_floorplan("50015850_PLAN4"), "50015850")
        self.assertEqual(
            building_id_from_floorplan("0510025537_Layout1_PLAN2"), "0510025537"
        )
        self.assertEqual(building_id_from_floorplan("tianda"), "named:tianda")

    def test_split_is_atomic_at_building_level(self):
        floorplan_ids = [f"{index:08d}" for index in range(16)]
        floorplan_ids += ["90000001_A", "90000001_B", "90000002_A", "90000002_B"]
        records = [
            {
                "floorplan_id": floorplan_id,
                "building_id": building_id_from_floorplan(floorplan_id),
            }
            for floorplan_id in floorplan_ids
        ]
        config = GenerationConfig(validation_fraction=0.15, test_fraction=0.15)
        assignments = assign_splits(records, config)
        counts = {
            split: sum(value == split for value in assignments.values())
            for split in ("train", "val", "test")
        }
        self.assertEqual(counts, {"train": 14, "val": 3, "test": 3})
        self.assertEqual(assignments["90000001_A"], assignments["90000001_B"])
        self.assertEqual(assignments["90000002_A"], assignments["90000002_B"])

    def test_generated_observation_is_one_hot_and_known_cells_are_exact(self):
        occupied = np.ones((600, 600), dtype=bool)
        occupied[180:420, 180:420] = False
        occupied[180:420, 298:302] = True
        occupied[290:310, 298:302] = False
        component, starts = select_navigable_component(occupied)
        config = GenerationConfig(
            laser_range_cells=80,
            laser_rays=180,
            minimum_scans=8,
            maximum_scans=8,
            movement_cells_per_scan=8,
            minimum_unknown_fraction=0.0,
            maximum_unknown_fraction=1.0,
        )
        generated = generate_sample(
            occupied,
            component,
            starts,
            np.random.default_rng(7),
            ray_offsets(config.laser_rays, config.laser_range_cells),
            config,
        )
        self.assertIsNotNone(generated)
        observation, target, _ = generated
        self.assertTrue(np.all((observation == 255).sum(axis=2) == 1))
        known = observation[:, :, 1] == 0
        measured_occupied = observation[:, :, 0] == 255
        self.assertTrue(np.array_equal(measured_occupied[known], (target == 255)[known]))

    def test_unknown_bce_ignores_known_cells(self):
        prediction = torch.tensor([[[[0.9, 0.2]]]])
        target = torch.tensor([[[[1.0, 1.0]]]])
        unknown = torch.tensor([[[[1.0, 0.0]]]])
        loss = unknown_bce(prediction, target, unknown)
        self.assertAlmostEqual(float(loss), -float(np.log(0.9)), places=6)


if __name__ == "__main__":
    unittest.main()
