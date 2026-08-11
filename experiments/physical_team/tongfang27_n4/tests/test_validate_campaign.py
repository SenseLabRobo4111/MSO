"""Tests for the prospective N=4 protocol validator."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import shutil
import sys
import tempfile
import unittest

import yaml


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import validate_campaign as validator  # noqa: E402


def load_shared_preflight():
    """Load the shared preflight without requiring a ROS installation."""
    path = BASE.parent / "scripts" / "preflight.py"
    specification = importlib.util.spec_from_file_location(
        "n4_dedicated_preflight", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class CampaignValidationTests(unittest.TestCase):
    """Exercise structural locks and current fail-closed readiness."""

    def paths(self, config: Path | None = None):
        config = config or BASE / "config"
        return (
            config / "campaign.yaml",
            config / "team_4.yaml",
            config / "arena.yaml",
            config / "start_poses.yaml",
            config / "model_lock.yaml",
            config / "online_stack_lock.yaml",
            config / "network_lock.yaml",
            config / "pilot_plan.yaml",
            config / "reference.yaml",
        )

    def test_frozen_plan_is_valid_but_not_collection_ready(self):
        report = validator.validate_package(*self.paths())
        self.assertTrue(report["plan_valid"])
        self.assertFalse(report["collection_ready"])
        self.assertFalse(report["claim_authorized"])
        self.assertEqual(report["planned_run_count"], 32)
        self.assertEqual(report["planned_block_count"], 8)
        checks = {item["check"] for item in report["readiness_blockers"]}
        self.assertIn("trusted collection enablement implementation", checks)
        self.assertIn("verified 304K artifact exists", checks)
        self.assertIn("full online stack status", checks)
        self.assertIn("N4 network impairment status", checks)

    def test_all_cells_have_eight_runs_and_pilots_are_separate(self):
        campaign = validator.load_yaml(BASE / "config/campaign.yaml")
        cells = {}
        main_ids = set()
        for run in campaign["ordered_runs"]:
            cell = (run["predictor_mode"], run["network_condition"])
            cells[cell] = cells.get(cell, 0) + 1
            main_ids.add(run["run_id"])
        self.assertEqual(set(cells.values()), {8})
        pilot = validator.load_yaml(BASE / "config/pilot_plan.yaml")
        pilot_ids = {run["run_id"] for run in pilot["pilot_runs"]}
        self.assertFalse(main_ids & pilot_ids)
        self.assertTrue(pilot["permanently_excluded_from_confirmatory_analysis"])

    def test_dedicated_team_lock_is_shared_preflight_compatible(self):
        team = validator.load_yaml(BASE / "config/team_4.yaml")
        topics = set(load_shared_preflight().required_topics(team))
        for robot_id in range(4):
            self.assertIn(f"/robot_{robot_id}/lidar/points", topics)
            self.assertIn(f"/ground_truth/robot_{robot_id}/pose", topics)
        self.assertNotIn("/robot_4/lidar/points", topics)

    def test_duplicate_run_id_invalidates_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config"
            shutil.copytree(BASE / "config", config)
            campaign = yaml.safe_load(
                (config / "campaign.yaml").read_text(encoding="utf-8"))
            campaign["ordered_runs"][1]["run_id"] = campaign["ordered_runs"][0]["run_id"]
            (config / "campaign.yaml").write_text(
                yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
            report = validator.validate_package(*self.paths(config))
            self.assertFalse(report["plan_valid"])
            self.assertTrue(any(
                item["check"] == "unique non-empty run IDs"
                for item in report["errors"]))

    def test_legacy_condition_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config"
            shutil.copytree(BASE / "config", config)
            campaign = yaml.safe_load(
                (config / "campaign.yaml").read_text(encoding="utf-8"))
            campaign["ordered_runs"][0]["condition_id"] = "reference"
            (config / "campaign.yaml").write_text(
                yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
            report = validator.validate_package(*self.paths(config))
            self.assertFalse(report["plan_valid"])
            self.assertTrue(any(
                "condition_id" in item["check"] for item in report["errors"]))

    def test_342771_parameter_substitute_is_explicitly_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config"
            shutil.copytree(BASE / "config", config)
            model = yaml.safe_load(
                (config / "model_lock.yaml").read_text(encoding="utf-8"))
            model["exact_trainable_parameter_count"] = 342771
            model["artifact_sha256"] = next(iter(
                validator.KNOWN_342771_CANDIDATE_HASHES))
            (config / "model_lock.yaml").write_text(
                yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
            report = validator.validate_package(*self.paths(config))
            checks = {item["check"] for item in report["readiness_blockers"]}
            self.assertIn("exact model parameter count", checks)
            self.assertIn("reject known recovered candidate artifact", checks)
            self.assertFalse(report["collection_ready"])

    def test_each_williams_sequence_occurs_twice(self):
        campaign = validator.load_yaml(BASE / "config/campaign.yaml")
        blocks = {}
        for run in campaign["ordered_runs"]:
            blocks.setdefault(run["block_id"], []).append(run)
        counts = {}
        for rows in blocks.values():
            rows.sort(key=lambda row: row["sequence_position"])
            sequence = tuple(row["treatment_label"] for row in rows)
            counts[sequence] = counts.get(sequence, 0) + 1
        self.assertEqual(set(counts), validator.ALLOWED_SEQUENCES)
        self.assertEqual(set(counts.values()), {2})

    def test_structural_safety_frame_lock_and_network_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config"
            shutil.copytree(BASE / "config", config)
            campaign_path = config / "campaign.yaml"
            arena_path = config / "arena.yaml"
            reference_path = config / "reference.yaml"
            campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
            arena = yaml.safe_load(arena_path.read_text(encoding="utf-8"))
            reference = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
            campaign["locks"]["team"] = "config/unrelated.yaml"
            campaign["network_impairment"]["phases"][1]["loss_probability"] = 0.0
            arena["safety"]["maximum_linear_velocity_m_s"] = 99.0
            arena["coordinate_frame"] = "wrong_frame"
            reference["requirements"][
                "external_pose_not_consumed_by_planner"] = False
            campaign_path.write_text(
                yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
            arena_path.write_text(
                yaml.safe_dump(arena, sort_keys=False), encoding="utf-8")
            reference_path.write_text(
                yaml.safe_dump(reference, sort_keys=False), encoding="utf-8")
            report = validator.validate_package(*self.paths(config))
            self.assertFalse(report["plan_valid"])
            checks = {item["check"] for item in report["errors"]}
            self.assertIn("campaign lock path map", checks)
            self.assertIn("network impairment phases", checks)
            self.assertIn("frozen conservative safety limits", checks)
            self.assertIn("GT, start-pose and reference frame agreement", checks)
            self.assertIn("independent reference requirements", checks)

    def test_pose_set_and_run_identity_are_locked_to_each_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config"
            shutil.copytree(BASE / "config", config)
            campaign_path = config / "campaign.yaml"
            campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
            campaign["ordered_runs"][0]["run_id"] = "arbitrary_run"
            campaign["ordered_runs"][0]["start_pose_set_id"] = "P08"
            campaign_path.write_text(
                yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
            report = validator.validate_package(*self.paths(config))
            self.assertFalse(report["plan_valid"])
            checks = {item["check"] for item in report["errors"]}
            self.assertIn("portable run ID contract", checks)
            self.assertIn("one designated pose set per block", checks)

    def test_cli_lock_paths_must_match_campaign_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config"
            shutil.copytree(BASE / "config", config)
            alternate_team = root / "alternate_team.yaml"
            alternate_team.write_bytes((config / "team_4.yaml").read_bytes())
            paths = list(self.paths(config))
            paths[1] = alternate_team
            report = validator.validate_package(*paths)
            self.assertFalse(report["plan_valid"])
            self.assertTrue(any(
                item["check"] == "CLI input path differs from campaign lock"
                for item in report["errors"]))


if __name__ == "__main__":
    unittest.main()
