"""Tests for complete-block paired analysis."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import analyse_paired_campaign as analysis  # noqa: E402


FIELDS = [
    "run_id", "block_id", "predictor_mode", "network_condition",
    "protocol_pass", "normalized_coverage_auc", "final_coverage",
    "registration_decisions", "false_accepts", "false_rejects",
    "wrong_commits", "safety_stops",
]


def synthetic_rows():
    """Create temporary test fixtures, not empirical results."""
    campaign = yaml.safe_load(
        (BASE / "config/campaign.yaml").read_text(encoding="utf-8"))
    rows = []
    for planned in campaign["ordered_runs"]:
        block = planned["block_id"]
        predictor = planned["predictor_mode"]
        network = planned["network_condition"]
        observed_nominal = 0.55 + 0.005 * block
        if predictor == "observed_only" and network == "nominal":
            auc = observed_nominal
        elif predictor == "mso_304k" and network == "nominal":
            auc = observed_nominal + 0.10
        elif predictor == "observed_only":
            auc = observed_nominal - 0.03
        else:
            auc = observed_nominal + 0.05
        rows.append({
            "run_id": planned["run_id"],
            "block_id": block,
            "predictor_mode": predictor,
            "network_condition": network,
            "protocol_pass": "true",
            "normalized_coverage_auc": f"{auc:.6f}",
            "final_coverage": f"{min(auc + 0.12, 0.99):.6f}",
            "registration_decisions": "20",
            "false_accepts": "0",
            "false_rejects": "1",
            "wrong_commits": "0",
            "safety_stops": "0",
        })
    return rows


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class PairedAnalysisTests(unittest.TestCase):
    """Ensure analysis never proceeds with an incomplete frozen matrix."""

    def test_complete_synthetic_matrix_produces_paired_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "runs.csv"
            output = root / "analysis"
            write_rows(results, synthetic_rows())
            report, status = analysis.analyse(
                BASE / "config/campaign.yaml", results, output,
                bootstrap_resamples=200, bootstrap_seed=7)
            self.assertEqual(status, 0)
            self.assertTrue(report["analysis_complete"])
            self.assertFalse(report["claim_authorized"])
            self.assertEqual(report["complete_block_count"], 8)
            self.assertAlmostEqual(
                report["primary_summary"]["mean_difference"], 0.10)
            filenames = (
                "paired_differences.csv", "paired_summary.csv",
                "cell_summary.csv", "analysis_audit.json")
            for filename in filenames:
                self.assertTrue((output / filename).is_file())

    def test_missing_run_fails_without_paired_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "runs.csv"
            output = root / "analysis"
            write_rows(results, synthetic_rows()[:-1])
            report, status = analysis.analyse(
                BASE / "config/campaign.yaml", results, output,
                bootstrap_resamples=20)
            self.assertEqual(status, 2)
            self.assertFalse(report["analysis_complete"])
            self.assertFalse((output / "paired_summary.csv").exists())
            self.assertTrue((output / "analysis_audit.json").is_file())

    def test_factor_metadata_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = synthetic_rows()
            rows[0]["predictor_mode"] = "mso_304k"
            results = root / "runs.csv"
            write_rows(results, rows)
            report, status = analysis.analyse(
                BASE / "config/campaign.yaml", results, root / "analysis",
                bootstrap_resamples=20)
            self.assertEqual(status, 2)
            self.assertTrue(any(
                "differs from frozen campaign" in item
                for item in report["errors"]))

    def test_logged_safety_stop_can_remain_protocol_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = synthetic_rows()
            rows[0]["safety_stops"] = "1"
            results = root / "runs.csv"
            write_rows(results, rows)
            report, status = analysis.analyse(
                BASE / "config/campaign.yaml", results, root / "analysis",
                bootstrap_resamples=20)
            self.assertEqual(status, 0)
            self.assertTrue(report["analysis_complete"])

    def test_existing_output_is_immutable_after_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_results = root / "valid.csv"
            invalid_results = root / "invalid.csv"
            output = root / "analysis"
            write_rows(valid_results, synthetic_rows())
            write_rows(invalid_results, synthetic_rows()[:-1])
            first, first_status = analysis.analyse(
                BASE / "config/campaign.yaml", valid_results, output,
                bootstrap_resamples=20, bootstrap_seed=3)
            self.assertEqual(first_status, 0)
            self.assertTrue(first["analysis_complete"])
            hashes_before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.iterdir() if path.is_file()
            }
            second, second_status = analysis.analyse(
                BASE / "config/campaign.yaml", invalid_results, output,
                bootstrap_resamples=20, bootstrap_seed=3)
            hashes_after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.iterdir() if path.is_file()
            }
            self.assertEqual(second_status, 2)
            self.assertFalse(second["analysis_complete"])
            self.assertIn("immutable", " ".join(second["errors"]))
            self.assertEqual(hashes_before, hashes_after)

    def test_invalid_bootstrap_count_leaves_no_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "runs.csv"
            output = root / "analysis"
            write_rows(results, synthetic_rows())
            report, status = analysis.analyse(
                BASE / "config/campaign.yaml", results, output,
                bootstrap_resamples=0)
            self.assertEqual(status, 2)
            self.assertFalse(report["analysis_complete"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
