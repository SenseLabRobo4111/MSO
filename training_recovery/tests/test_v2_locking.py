from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from audit_v2_gates import evaluate_gates
from mso_recovery.v2_data import read_v2_manifest
from summarize_v2_multiseed import summarize
from train_v2_locked import authorized_job


class V2LockingTests(unittest.TestCase):
    def setUp(self):
        self.lock_sha = "a" * 64
        self.lock = {
            "positive_control": {"sha256": "b" * 64},
            "training": {
                "teacher_seed": 101,
                "student_seeds": [11, 23, 37, 53, 71],
                "maximum_epochs": 500,
                "smoke_teacher": {"maximum_epochs": 30},
            },
            "go_no_go": {
                "integrity": {
                    "one_hot_fraction": 1.0,
                    "known_target_agreement": 1.0,
                    "cross_split_group_leakage": 0,
                    "cross_split_source_hash_leakage": 0,
                    "cross_split_processed_pair_leakage": 0,
                },
                "coverage": {
                    "train_floorplan_fraction": 0.9,
                    "val_floorplan_fraction": 0.9,
                },
                "marginals": {
                    "reference_observation_channels": [0.015, 0.729, 0.256],
                    "maximum_absolute_channel_delta": 0.05,
                    "reference_target": 0.098,
                    "maximum_absolute_target_delta": 0.05,
                    "maximum_train_val_target_delta": 0.03,
                },
                "positive_control": {
                    "minimum_unknown_f1": 0.45,
                    "minimum_unknown_iou": 0.30,
                },
                "learnability": {
                    "minimum_validation_unknown_bce_relative_reduction_from_epoch_1": 0.05,
                    "minimum_validation_unknown_f1": 0.05,
                    "maximum_epochs": 30,
                },
            },
        }
        self.validation = {
            "v2_lock_sha256": self.lock_sha,
            "test_outcomes_accessed": False,
            "test_images_decoded": 0,
            "dataset_manifest_sha256": "c" * 64,
            "one_hot_fraction": 1.0,
            "known_target_agreement": 1.0,
            "cross_split_operational_group_leakage": 0,
            "cross_split_source_hash_leakage": 0,
            "cross_split_processed_pair_leakage": 0,
            "floorplan_full_coverage_fraction": {"train": 0.91, "val": 0.91},
            "pooled_observation_channel_fraction": {
                "train": [0.015, 0.729, 0.256],
                "val": [0.016, 0.728, 0.256],
            },
            "pooled_target_fraction": {"train": 0.098, "val": 0.099},
        }
        self.positive = {
            "v2_lock_sha256": self.lock_sha,
            "test_outcomes_accessed": False,
            "test_images_decoded": 0,
            "dataset_manifest_sha256": "c" * 64,
            "candidate_sha256": "b" * 64,
            "validation_unknown_f1": 0.50,
            "validation_unknown_iou": 0.35,
        }
        self.metrics = [
            {
                "epoch": float(epoch),
                "validation_unknown_bce": 1.0 - 0.1 * (epoch - 1) / 29,
                "validation_unknown_f1": 0.01 + 0.05 * (epoch - 1) / 29,
            }
            for epoch in range(1, 31)
        ]

    def test_atomic_five_gate_go_requires_every_gate(self):
        result = evaluate_gates(
            self.lock,
            self.lock_sha,
            self.validation,
            self.positive,
            self.metrics,
        )
        self.assertEqual(result["status"], "GO")
        self.assertEqual(len(result["gates"]), 5)
        self.positive["validation_unknown_f1"] = 0.449
        result = evaluate_gates(
            self.lock,
            self.lock_sha,
            self.validation,
            self.positive,
            self.metrics,
        )
        self.assertEqual(result["status"], "NO_GO")

    def test_v2_manifest_rejects_test_before_decode(self):
        fields = [
            "sample_id",
            "split",
            "group_key",
            "group_kind",
            "floorplan_id",
            "source_sha256",
            "obs_relpath",
            "target_relpath",
            "obs_sha256",
            "target_sha256",
            "raw_obs_relpath",
            "raw_target_relpath",
            "raw_obs_sha256",
            "raw_target_sha256",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "forbidden",
                        "split": "test",
                        "group_key": "g",
                        "group_kind": "operational_prefix_group",
                        "floorplan_id": "f",
                        "source_sha256": "a" * 64,
                        "obs_relpath": "missing.png",
                        "target_relpath": "missing2.png",
                        "obs_sha256": "b" * 64,
                        "target_sha256": "c" * 64,
                        "raw_obs_relpath": "missing3.png",
                        "raw_target_relpath": "missing4.png",
                        "raw_obs_sha256": "d" * 64,
                        "raw_target_sha256": "e" * 64,
                    }
                )
            with self.assertRaisesRegex(ValueError, "train/val samples only"):
                read_v2_manifest(path)

    def test_job_ids_are_closed_over_locked_seeds(self):
        self.assertEqual(authorized_job(self.lock, "teacher_101")["seed"], 101)
        self.assertEqual(authorized_job(self.lock, "student_53")["stage"], "student")
        with self.assertRaises(ValueError):
            authorized_job(self.lock, "student_52")

    def test_statistics_require_all_five_units(self):
        result = summarize([1.0, 2.0, 3.0, 4.0, 5.0], 2.7764451051977987)
        self.assertEqual(result["mean"], 3.0)
        with self.assertRaises(ValueError):
            summarize([1.0, 2.0, 3.0, 4.0], 2.7764451051977987)


if __name__ == "__main__":
    unittest.main()
