"""Tests for fail-closed model verification and atomic run preparation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import yaml


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import prepare_n4_run as preparer  # noqa: E402
import verify_model_artifact as model_verifier  # noqa: E402
from build_model_fixture import build_fixture  # noqa: E402


def config_paths():
    config = BASE / "config"
    return {
        "campaign_path": config / "campaign.yaml",
        "team_path": config / "team_4.yaml",
        "arena_path": config / "arena.yaml",
        "start_poses_path": config / "start_poses.yaml",
        "model_path": config / "model_lock.yaml",
        "online_stack_path": config / "online_stack_lock.yaml",
        "network_path": config / "network_lock.yaml",
        "pilot_path": config / "pilot_plan.yaml",
        "reference_path": config / "reference.yaml",
    }


def synthetic_ready_validation(paths=None):
    """Return a mocked ready report bound to the current lock bytes."""
    paths = paths or config_paths()
    key_map = {
        "campaign": "campaign_path",
        "team": "team_path",
        "arena": "arena_path",
        "start_poses": "start_poses_path",
        "model": "model_path",
        "online_stack": "online_stack_path",
        "network": "network_path",
        "pilot": "pilot_path",
        "reference": "reference_path",
    }
    return {
        "plan_valid": True,
        "collection_ready": True,
        "input_sha256": {
            name: preparer.sha256(paths[path_key])
            for name, path_key in key_map.items()
        },
    }


class PreparationAndModelTests(unittest.TestCase):
    """Check current refusal and temporary-directory cleanup."""

    def test_model_verifier_refuses_unsupported_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "candidate.pt"
            fixture = root / "fixture.npy"
            report_path = root / "report.json"
            artifact.write_bytes(b"not-a-model")
            fixture.write_bytes(b"not-a-fixture")
            report, status = model_verifier.verify(
                artifact, "unreviewed_loader", fixture, report_path)
            self.assertEqual(status, 2)
            self.assertFalse(report["verified"])
            self.assertTrue(report_path.is_file())
            stored = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(stored["verified"])
            self.assertEqual(stored["artifact_filename"], "candidate.pt")
            self.assertNotIn("artifact_path", stored)
            self.assertNotIn(str(root), report_path.read_text(encoding="utf-8"))
            self.assertIn("not explicitly supported", " ".join(stored["errors"]))

    def test_selected_model_strict_load_and_forward_are_verified(self):
        artifact = (
            BASE.parents[2] / "repro_reconstructed" / "checkpoints" /
            "recovered_candidate_deconv_a.pt"
        )
        fixture = (
            BASE / "deployment_models" /
            "fixed_fixture_occ_unknown_free_v1.npy"
        )
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "verification.json"
            report, status = model_verifier.verify(
                artifact, model_verifier.LOADER_ID, fixture, report_path)
            self.assertEqual(status, 0)
            self.assertTrue(report["verified"])
            self.assertEqual(
                report["trainable_parameter_count"], 342771)
            self.assertEqual(report["missing_keys"], [])
            self.assertEqual(report["unexpected_keys"], [])
            self.assertTrue(report["finite_state"])
            self.assertEqual(report["output_shape"], [1, 1, 256, 256])
            self.assertTrue(report["all_outputs_finite"])
            self.assertEqual(report["all_output_shapes"], [
                [1, 1, 256, 256],
                [1, 8, 128, 128],
                [1, 16, 64, 64],
                [1, 16, 64, 64],
                [1, 8, 128, 128],
            ])
            self.assertEqual(len(report["all_output_fingerprints"]), 5)
            self.assertEqual(
                report["output_fingerprint"],
                "31e3cd4ec395b615a9d5dafd1024286fd9860f58acaffc1df09d23bc624778e0")

    def test_nonselected_candidate_is_rejected_by_identity_hash(self):
        artifact = (
            BASE.parents[2] / "repro_reconstructed" / "checkpoints" /
            "recovered_candidate_deconv_label_b.pt"
        )
        fixture = (
            BASE / "deployment_models" /
            "fixed_fixture_occ_unknown_free_v1.npy"
        )
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "verification.json"
            report, status = model_verifier.verify(
                artifact, model_verifier.LOADER_ID, fixture, report_path)
            self.assertEqual(status, 2)
            self.assertFalse(report["verified"])
            self.assertIn(
                "does not identify the locked candidate A",
                " ".join(report["errors"]),
            )

    def test_committed_fixture_matches_deterministic_builder(self):
        import numpy as np

        fixture = np.load(
            BASE / "deployment_models" /
            "fixed_fixture_occ_unknown_free_v1.npy",
            allow_pickle=False)
        self.assertTrue(np.array_equal(fixture, build_fixture()))

    def test_current_source_gated_package_cannot_prepare_a_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "runs"
            report, status = preparer.prepare_run(
                run_id="tf27_n4_b01_B", operator="test_operator",
                output_root=output, **config_paths())
            self.assertEqual(status, 2)
            self.assertFalse(report["prepared"])
            self.assertFalse((output / "tf27_n4_b01_B").exists())

    def test_staging_failure_leaves_no_partial_final_or_temp_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "runs"
            artifact = root / "model.pt"
            artifact.write_bytes(b"synthetic-test-artifact")
            model_hash = preparer.sha256(artifact)
            original_copy = preparer.shutil.copy2
            original_loader = preparer.load_yaml
            calls = {"count": 0}

            def fail_second_copy(source, target):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("injected copy failure")
                return original_copy(source, target)

            def load_with_model_hash(path):
                value = original_loader(path)
                if Path(path) == config_paths()["model_path"]:
                    value["artifact_sha256"] = model_hash
                return value

            with mock.patch.object(
                    preparer, "validate_package",
                    return_value=synthetic_ready_validation()), \
                    mock.patch.object(
                        preparer, "resolve_artifact", return_value=artifact), \
                    mock.patch.object(
                        preparer, "load_yaml", side_effect=load_with_model_hash), \
                    mock.patch.object(
                        preparer.shutil, "copy2", side_effect=fail_second_copy):
                report, status = preparer.prepare_run(
                    run_id="tf27_n4_b01_B", operator="test_operator",
                    output_root=output, **config_paths())
            self.assertEqual(status, 2)
            self.assertFalse(report["prepared"])
            self.assertFalse((output / "tf27_n4_b01_B").exists())
            self.assertEqual(list(output.glob(".*.preparing-*")), [])

    def test_successful_synthetic_staging_renames_complete_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "runs"
            artifact = root / "model.pt"
            artifact.write_bytes(b"synthetic-test-artifact")
            model_hash = preparer.sha256(artifact)
            original_loader = preparer.load_yaml

            def load_with_model_hash(path):
                value = original_loader(path)
                if Path(path) == config_paths()["model_path"]:
                    value["artifact_sha256"] = model_hash
                return value
            with mock.patch.object(
                    preparer, "validate_package",
                    return_value=synthetic_ready_validation()), \
                    mock.patch.object(
                        preparer, "resolve_artifact", return_value=artifact), \
                    mock.patch.object(
                        preparer, "load_yaml", side_effect=load_with_model_hash):
                report, status = preparer.prepare_run(
                    run_id="tf27_n4_b01_B", operator="test_operator",
                    output_root=output, **config_paths())
            final = output / "tf27_n4_b01_B"
            self.assertEqual(status, 0)
            self.assertTrue(report["prepared"])
            self.assertTrue((final / "run_lock.json").is_file())
            run_lock = json.loads(
                (final / "run_lock.json").read_text(encoding="utf-8"))
            self.assertEqual(
                run_lock["model_loader_id"],
                model_verifier.LOADER_ID)
            self.assertEqual(
                run_lock["model_selection_record_sha256"],
                "c79ec419309620d52fc3a6d353cb5622e23bb65d99d6add4753641f0f2788b41")
            self.assertEqual(
                run_lock["model_architecture_source_sha256"],
                "c84ff05fa168c2e856002f12a903e53896da70cd8854a152b85f407d176727b3")
            self.assertTrue((final / "PRECAPTURE_SHA256SUMS").is_file())
            self.assertTrue((final / "locks").is_dir())
            self.assertFalse((final / "bag").exists())
            self.assertEqual(list(output.glob(".*.preparing-*")), [])

    def test_lock_hash_change_after_validation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "runs"
            validation = synthetic_ready_validation()
            validation["input_sha256"]["campaign"] = "0" * 64
            with mock.patch.object(
                    preparer, "validate_package", return_value=validation):
                report, status = preparer.prepare_run(
                    run_id="tf27_n4_b01_B", operator="test_operator",
                    output_root=output, **config_paths())
            self.assertEqual(status, 2)
            self.assertFalse(report["prepared"])
            self.assertIn("changed after validation", report["reason"])
            self.assertFalse(output.exists())

    def test_source_lock_change_during_staging_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config"
            shutil.copytree(BASE / "config", config)
            artifact = root / "model.pt"
            artifact.write_bytes(b"synthetic-test-artifact")
            model_path = config / "model_lock.yaml"
            model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
            model["artifact_path"] = str(artifact)
            model["artifact_sha256"] = preparer.sha256(artifact)
            model_path.write_text(
                yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
            paths = {
                "campaign_path": config / "campaign.yaml",
                "team_path": config / "team_4.yaml",
                "arena_path": config / "arena.yaml",
                "start_poses_path": config / "start_poses.yaml",
                "model_path": model_path,
                "online_stack_path": config / "online_stack_lock.yaml",
                "network_path": config / "network_lock.yaml",
                "pilot_path": config / "pilot_plan.yaml",
                "reference_path": config / "reference.yaml",
            }
            validation = synthetic_ready_validation(paths)
            original_loader = preparer.load_yaml
            changed = {"done": False}

            def mutate_after_first_load(path):
                value = original_loader(path)
                if not changed["done"]:
                    with model_path.open("a", encoding="utf-8") as handle:
                        handle.write("\n# injected replacement window\n")
                    changed["done"] = True
                return value

            output = root / "runs"
            with mock.patch.object(
                    preparer, "validate_package", return_value=validation), \
                    mock.patch.object(
                        preparer, "load_yaml", side_effect=mutate_after_first_load):
                report, status = preparer.prepare_run(
                    run_id="tf27_n4_b01_B", operator="test_operator",
                    output_root=output, **paths)
            self.assertEqual(status, 2)
            self.assertFalse(report["prepared"])
            self.assertIn("changed during staging", report["reason"])
            self.assertFalse((output / "tf27_n4_b01_B").exists())
            self.assertEqual(list(output.glob(".*.preparing-*")), [])


if __name__ == "__main__":
    unittest.main()
