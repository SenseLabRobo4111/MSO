#!/usr/bin/env python3
"""Block-paired analysis for a complete Tongfang 27F N=4 campaign."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

try:
    import yaml
except ImportError as error:  # pragma: no cover
    raise SystemExit("PyYAML is required") from error


CELLS = (
    ("mso_342771", "nominal"),
    ("observed_only", "nominal"),
    ("mso_342771", "isolated_robot_impairment"),
    ("observed_only", "isolated_robot_impairment"),
)
CAMPAIGN_ID = "tf27_n4_mso_342771_factorial_v1"
MODEL_BINDING = {
    "model_id": "mso_deconv_342771_candidate_a",
    "artifact_sha256": (
        "da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8"),
    "loader_id": "distill_map_net_deconv_raw_state_v1",
    "input_contract": "occupied_unknown_free_one_hot_256_v1",
    "output_contract": "sigmoid_occupancy_probability_256_v1",
}
REQUIRED_COLUMNS = {
    "run_id", "block_id", "predictor_mode", "network_condition",
    "protocol_pass", "normalized_coverage_auc", "final_coverage",
    "registration_decisions", "false_accepts", "false_rejects",
    "wrong_commits", "safety_stops",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_campaign(path: Path) -> dict[str, Any]:
    """Load the frozen campaign mapping."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("campaign YAML must contain a mapping")
    return value


def parse_bool(value: str) -> bool:
    """Parse an explicit CSV boolean without truthy shortcuts."""
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"expected true or false, received {value!r}")


def parse_fraction(value: str, field: str) -> float:
    """Parse a finite fraction in [0, 1]."""
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return result


def parse_count(value: str, field: str) -> int:
    """Parse a non-negative integer count."""
    result = int(value)
    if result < 0 or str(result) != value.strip():
        raise ValueError(f"{field} must be a canonical non-negative integer")
    return result


def read_results(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read and strongly type one-row-per-run endpoint data."""
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - columns)
            if missing:
                return [], [f"missing columns: {missing}"]
            for line_number, raw in enumerate(reader, 2):
                try:
                    row = {
                        "run_id": raw["run_id"].strip(),
                        "block_id": int(raw["block_id"]),
                        "predictor_mode": raw["predictor_mode"].strip(),
                        "network_condition": raw["network_condition"].strip(),
                        "protocol_pass": parse_bool(raw["protocol_pass"]),
                        "normalized_coverage_auc": parse_fraction(
                            raw["normalized_coverage_auc"], "normalized_coverage_auc"),
                        "final_coverage": parse_fraction(
                            raw["final_coverage"], "final_coverage"),
                    }
                    for field in (
                        "registration_decisions", "false_accepts", "false_rejects",
                        "wrong_commits", "safety_stops",
                    ):
                        row[field] = parse_count(raw[field], field)
                    if row["false_accepts"] + row["false_rejects"] > row["registration_decisions"]:
                        raise ValueError("gate error counts exceed registration decisions")
                    rows.append(row)
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(f"line {line_number}: {error}")
    except OSError as error:
        errors.append(str(error))
    return rows, errors


def exact_sign_flip_p(differences: list[float]) -> float:
    """Return the exhaustive two-sided paired sign-flip p-value."""
    if not differences:
        raise ValueError("at least one paired difference is required")
    observed = abs(statistics.mean(differences))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(statistics.mean(
            sign * value for sign, value in zip(signs, differences)))
        total += 1
        if statistic + 1e-15 >= observed:
            extreme += 1
    return extreme / total


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of a non-empty sorted sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_interval(
    differences: list[float], resamples: int, seed: int
) -> tuple[float, float]:
    """Percentile interval from block-level bootstrap resampling."""
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    generator = random.Random(seed)
    estimates = [
        statistics.mean(generator.choice(differences) for _ in differences)
        for _ in range(resamples)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def contrast_summary(
    name: str, endpoint: str, values: list[float], resamples: int, seed: int
) -> dict[str, Any]:
    """Summarize one eight-block paired contrast."""
    lower, upper = bootstrap_mean_interval(values, resamples, seed)
    return {
        "contrast": name,
        "endpoint": endpoint,
        "n_blocks": len(values),
        "mean_difference": statistics.mean(values),
        "median_difference": statistics.median(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "bootstrap_95_ci_lower": lower,
        "bootstrap_95_ci_upper": upper,
        "exact_sign_flip_p_two_sided": exact_sign_flip_p(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write a stable CSV table."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyse(
    campaign_path: Path,
    results_path: Path,
    output_dir: Path,
    *,
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 20260811,
) -> tuple[dict[str, Any], int]:
    """Audit completeness, then calculate block-paired descriptive outputs."""
    if bootstrap_resamples < 1:
        return {
            "schema_version": "1.0",
            "analysis_complete": False,
            "claim_authorized": False,
            "errors": ["bootstrap_resamples must be positive"],
        }, 2
    if output_dir.exists():
        return {
            "schema_version": "1.0",
            "analysis_complete": False,
            "claim_authorized": False,
            "errors": [
                "output directory already exists; analysis outputs are immutable"],
        }, 2
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    errors: list[str] = []
    try:
        campaign = load_campaign(campaign_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        campaign = {}
        errors.append(str(error))
    rows, row_errors = read_results(results_path)
    errors.extend(row_errors)
    if campaign.get("campaign_id") != CAMPAIGN_ID:
        errors.append("campaign identity differs from the locked analysis")
    if campaign.get("model_binding") != MODEL_BINDING:
        errors.append("model binding differs from the locked analysis")
    planned = {
        run.get("run_id"): run
        for run in campaign.get("ordered_runs") or [] if isinstance(run, dict)
    }
    if len(planned) != 32:
        errors.append(f"campaign must contain 32 unique planned runs, found {len(planned)}")
    observed_ids = [row["run_id"] for row in rows]
    duplicate_ids = sorted({item for item in observed_ids if observed_ids.count(item) > 1})
    if duplicate_ids:
        errors.append(f"duplicate result run IDs: {duplicate_ids}")
    missing_ids = sorted(set(planned) - set(observed_ids))
    unplanned_ids = sorted(set(observed_ids) - set(planned))
    if missing_ids:
        errors.append(f"missing planned runs: {missing_ids}")
    if unplanned_ids:
        errors.append(f"unplanned result runs: {unplanned_ids}")

    for row in rows:
        plan = planned.get(row["run_id"])
        if not plan:
            continue
        for field in ("block_id", "predictor_mode", "network_condition"):
            if row[field] != plan.get(field):
                errors.append(
                    f"{row['run_id']}: {field} differs from frozen campaign")
        if not row["protocol_pass"]:
            errors.append(f"{row['run_id']}: protocol_pass is false")

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "campaign_id": campaign.get("campaign_id"),
        "model_binding": campaign.get("model_binding"),
        "analysis_complete": False,
        "claim_authorized": False,
        "formal_primary_endpoint": "normalized_coverage_auc",
        "formal_primary_contrast": (
            "mso_342771_minus_observed_only_under_nominal_network"),
        "result_row_count": len(rows),
        "errors": errors,
        "input_sha256": {
            "campaign": sha256(campaign_path) if campaign_path.is_file() else None,
            "endpoint_table": sha256(results_path) if results_path.is_file() else None,
            "analysis_implementation": sha256(Path(__file__).resolve()),
        },
        "input_contract": (
            "The endpoint table must be produced only after independent N4 "
            "run audits. This analysis does not verify ROS bags, GT traces, "
            "600-s carry-forward or collection eligibility."),
        "inference_note": (
            "The exact sign-flip p-value relies on exchangeability and "
            "symmetry of the eight paired block differences; it is not a "
            "generated randomization-test claim."),
        "note": (
            "Analysis completeness never authorizes a manuscript claim; "
            "independent review remains required."),
    }
    audit_path = output_dir / "analysis_audit.json"
    if errors:
        audit_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return report, 2

    by_block: dict[int, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_block[row["block_id"]][
            (row["predictor_mode"], row["network_condition"])] = row
    if set(by_block) != set(range(1, 9)) or any(
            set(cells) != set(CELLS) for cells in by_block.values()):
        report["errors"].append("results do not form eight complete four-cell blocks")
        audit_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return report, 2

    difference_rows: list[dict[str, Any]] = []
    for block_id in range(1, 9):
        cells = by_block[block_id]
        row: dict[str, Any] = {"block_id": block_id}
        for endpoint in ("normalized_coverage_auc", "final_coverage"):
            a = cells[("mso_342771", "nominal")][endpoint]
            b = cells[("observed_only", "nominal")][endpoint]
            c = cells[("mso_342771", "isolated_robot_impairment")][endpoint]
            d = cells[("observed_only", "isolated_robot_impairment")][endpoint]
            row[f"{endpoint}_prediction_nominal"] = a - b
            row[f"{endpoint}_prediction_impaired"] = c - d
            row[f"{endpoint}_network_mso"] = c - a
            row[f"{endpoint}_network_observed"] = d - b
            row[f"{endpoint}_interaction"] = (c - d) - (a - b)
        difference_rows.append(row)

    contrast_names = (
        "prediction_nominal", "prediction_impaired", "network_mso",
        "network_observed", "interaction")
    summary_rows: list[dict[str, Any]] = []
    for endpoint_index, endpoint in enumerate(
            ("normalized_coverage_auc", "final_coverage")):
        for contrast_index, contrast in enumerate(contrast_names):
            differences = [
                row[f"{endpoint}_{contrast}"] for row in difference_rows]
            summary_rows.append(contrast_summary(
                contrast, endpoint, differences, bootstrap_resamples,
                bootstrap_seed + endpoint_index * 100 + contrast_index))

    cell_rows: list[dict[str, Any]] = []
    for predictor, network in CELLS:
        cell = [
            row for row in rows
            if all((
                row["predictor_mode"] == predictor,
                row["network_condition"] == network))]
        cell_rows.append({
            "predictor_mode": predictor,
            "network_condition": network,
            "n_runs": len(cell),
            "normalized_coverage_auc_mean": statistics.mean(
                row["normalized_coverage_auc"] for row in cell),
            "normalized_coverage_auc_sd": statistics.stdev(
                row["normalized_coverage_auc"] for row in cell),
            "final_coverage_mean": statistics.mean(
                row["final_coverage"] for row in cell),
            "registration_decisions_total": sum(
                row["registration_decisions"] for row in cell),
            "false_accepts_total": sum(row["false_accepts"] for row in cell),
            "false_rejects_total": sum(row["false_rejects"] for row in cell),
            "wrong_commits_total": sum(row["wrong_commits"] for row in cell),
            "safety_stops_total": sum(row["safety_stops"] for row in cell),
        })

    write_csv(
        output_dir / "paired_differences.csv", difference_rows,
        list(difference_rows[0]))
    write_csv(
        output_dir / "paired_summary.csv", summary_rows,
        list(summary_rows[0]))
    write_csv(output_dir / "cell_summary.csv", cell_rows, list(cell_rows[0]))
    report["analysis_complete"] = True
    report["complete_block_count"] = 8
    report["primary_summary"] = next(
        row for row in summary_rows
        if all((
            row["endpoint"] == "normalized_coverage_auc",
            row["contrast"] == "prediction_nominal")))
    audit_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return report, 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=base / "config/campaign.yaml")
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> int:
    """Run the fail-closed paired analysis."""
    args = parse_args()
    report, status = analyse(
        args.campaign, args.runs_csv, args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    sys.exit(main())
