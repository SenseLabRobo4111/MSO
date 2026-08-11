#!/usr/bin/env python3
"""Aggregate V4 primary-selected teacher and five student runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from aggregate_full_mso_reconstruction import (
    SEEDS,
    mean_sd_ci,
    read_tsv,
    sha256,
    verify_sha_file,
    write_tsv,
)


V4_METHOD = "MSO-Paper-Equation-Reconstructed-V4"
BASELINE_METHODS = (
    "MSO",
    "U-Net",
    "LaMa-Fourier",
    "MI-GAN",
    "PartialConv",
    "GatedConv",
    "EdgeConnect",
    "AOT-GAN",
    "MAT",
    "ZITS++",
    "RePaint",
    "HINT",
)
INCOMPLETE_BASELINE = "RePaint"
PRIMARY_TEST_METRIC = "test_unknown_occupied_f1"
CLAIM_SCOPE = "frozen_archive_benchmark_only_not_universal_sota"
PROTOCOL_ID = "mso_paper_equation_reconstruction_v4"
SELECTION_SOURCE = "independent_full_metrics_tsv_recomputation"
PRIMARY_CHECKPOINT_KIND = "validation_best_primary_generator"
CALIBRATION_CHECKPOINT_KIND = "validation_best_calibration_generator"


def finite_float(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing finite {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def rows_by_seed(
    rows: list[dict[str, Any]], method: str
) -> dict[int, dict[str, Any]]:
    selected = [row for row in rows if row.get("method") == method]
    output: dict[int, dict[str, Any]] = {}
    for row in selected:
        seed = int(row["seed"])
        if seed in output:
            raise ValueError(f"duplicate seed {seed} for {method}")
        output[seed] = row
    return output


def validate_uniform_archive(
    uniform_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    status = {row["method"]: row for row in status_rows}
    if len(status) != len(status_rows) or set(status) != set(BASELINE_METHODS):
        raise ValueError("uniform method-status registry differs")
    if any(row.get("method") not in BASELINE_METHODS for row in uniform_rows):
        raise ValueError("uniform per-seed table contains an unregistered method")
    complete: list[str] = []
    for method in BASELINE_METHODS:
        row = status[method]
        if int(row["registered_seeds"]) != len(SEEDS):
            raise ValueError(f"registered seed count differs for {method}")
        completed = int(row["completed_seeds"])
        seeds = rows_by_seed(uniform_rows, method)
        if method == INCOMPLETE_BASELINE:
            if completed != 0 or seeds:
                raise ValueError("RePaint must remain explicitly incomplete at 0/5")
            if row["status"] != "incomplete":
                raise ValueError("RePaint status must be incomplete")
            continue
        if completed != len(SEEDS) or row["status"] != "complete":
            raise ValueError(f"completed baseline status differs for {method}")
        if set(seeds) != set(SEEDS):
            raise ValueError(f"completed baseline seed grid differs for {method}")
        complete.append(method)
    return tuple(complete)


def validate_uniform_summary(
    uniform_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    summary = {row["method"]: row for row in summary_rows}
    if len(summary) != len(summary_rows) or set(summary) != set(BASELINE_METHODS):
        raise ValueError("uniform method-summary registry differs")
    for method in BASELINE_METHODS:
        row = summary[method]
        selected = rows_by_seed(uniform_rows, method)
        if method == INCOMPLETE_BASELINE:
            if int(row["n_seeds"]) != 0:
                raise ValueError("RePaint summary must report zero seeds")
            if row.get(f"{PRIMARY_TEST_METRIC}_mean", "") not in {"", None}:
                raise ValueError("RePaint summary must not report a primary mean")
            continue
        values = [finite_float(selected[seed], PRIMARY_TEST_METRIC) for seed in SEEDS]
        expected = mean_sd_ci(values)
        if int(row["n_seeds"]) != len(SEEDS):
            raise ValueError(f"uniform summary seed count differs for {method}")
        for suffix, expected_value in expected.items():
            recorded = finite_float(row, f"{PRIMARY_TEST_METRIC}_{suffix}")
            if not math.isclose(recorded, expected_value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"uniform primary summary differs for {method}")


def combined_seed_rows(
    uniform_rows: list[dict[str, Any]],
    v4_rows: list[dict[str, Any]],
    complete_baselines: tuple[str, ...],
) -> list[dict[str, Any]]:
    source_rows: list[tuple[str, dict[str, Any]]] = []
    for method in complete_baselines:
        by_seed = rows_by_seed(uniform_rows, method)
        source_rows.extend(("uniform_objective", by_seed[seed]) for seed in SEEDS)
    v4_by_seed = rows_by_seed(v4_rows, V4_METHOD)
    if set(v4_by_seed) != set(SEEDS):
        raise ValueError("V4 aggregate lacks the exact five predeclared seeds")
    source_rows.extend(("paper_equation_reconstruction_v4", v4_by_seed[seed]) for seed in SEEDS)
    data_fields: list[str] = []
    seen = {"method", "seed"}
    for _, row in source_rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                data_fields.append(field)
    combined: list[dict[str, Any]] = []
    for protocol, row in source_rows:
        combined.append(
            {
                "method": row["method"],
                "seed": int(row["seed"]),
                "protocol": protocol,
                "comparison_scope": CLAIM_SCOPE,
                **{field: row.get(field, "") for field in data_fields},
            }
        )
    return combined


def build_archive_benchmark_tables(
    v4_rows: list[dict[str, Any]],
    uniform_rows: list[dict[str, Any]],
    uniform_status_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build the complete-case frozen-archive ranking without seed removal."""

    complete_baselines = validate_uniform_archive(uniform_rows, uniform_status_rows)
    combined = combined_seed_rows(uniform_rows, v4_rows, complete_baselines)
    eligible_methods = (*complete_baselines, V4_METHOD)
    status_lookup = {row["method"]: row for row in uniform_status_rows}
    status_rows: list[dict[str, Any]] = []
    for method in (*BASELINE_METHODS, V4_METHOD):
        if method == V4_METHOD:
            registered = completed = len(SEEDS)
            source_status = "complete"
        else:
            source = status_lookup[method]
            registered = int(source["registered_seeds"])
            completed = int(source["completed_seeds"])
            source_status = source["status"]
        eligible = method in eligible_methods
        status_rows.append(
            {
                "method": method,
                "registered_seeds": registered,
                "completed_seeds": completed,
                "status": source_status,
                "ranking_eligible": int(eligible),
                "ranking_exclusion_reason": (
                    "" if eligible else "zero_of_five_seed_runs_complete"
                ),
                "seed_exclusions": 0,
                "comparison_scope": CLAIM_SCOPE,
            }
        )

    metric_fields = [
        field
        for field in combined[0]
        if field
        not in {"method", "seed", "protocol", "comparison_scope"}
    ]
    summary_rows: list[dict[str, Any]] = []
    for status_row in status_rows:
        method = status_row["method"]
        eligible = bool(status_row["ranking_eligible"])
        metric_summary: dict[str, Any] = {}
        if eligible:
            selected = rows_by_seed(combined, method)
            if set(selected) != set(SEEDS):
                raise ValueError(f"ranking seed grid differs for {method}")
            for field in metric_fields:
                raw_values = [selected[seed].get(field, "") for seed in SEEDS]
                present = [value for value in raw_values if value not in {"", None}]
                if present and len(present) != len(SEEDS):
                    raise ValueError(
                        f"partial five-seed metric {field} for {method}"
                    )
                statistics = (
                    mean_sd_ci(
                        [finite_float(selected[seed], field) for seed in SEEDS]
                    )
                    if present
                    else {"mean": "", "sd": "", "ci95_low": "", "ci95_high": ""}
                )
                for suffix, value in statistics.items():
                    metric_summary[f"{field}_{suffix}"] = value
        else:
            for field in metric_fields:
                for suffix in ("mean", "sd", "ci95_low", "ci95_high"):
                    metric_summary[f"{field}_{suffix}"] = ""
        summary_rows.append(
            {
                "method": method,
                "n_seeds": int(status_row["completed_seeds"]),
                "ranking_eligible": int(eligible),
                "primary_metric": PRIMARY_TEST_METRIC,
                "comparison_scope": CLAIM_SCOPE,
                **metric_summary,
            }
        )

    eligible_summaries = [row for row in summary_rows if row["ranking_eligible"]]
    eligible_summaries.sort(
        key=lambda row: (
            -float(row[f"{PRIMARY_TEST_METRIC}_mean"]),
            str(row["method"]),
        )
    )
    ranking_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(eligible_summaries, start=1):
        ranking_rows.append(
            {
                "rank": rank,
                "method": row["method"],
                "n_seeds": row["n_seeds"],
                "ranking_eligible": 1,
                "primary_metric": PRIMARY_TEST_METRIC,
                "primary_metric_mean": row[f"{PRIMARY_TEST_METRIC}_mean"],
                "primary_metric_sd": row[f"{PRIMARY_TEST_METRIC}_sd"],
                "primary_metric_ci95_low": row[
                    f"{PRIMARY_TEST_METRIC}_ci95_low"
                ],
                "primary_metric_ci95_high": row[
                    f"{PRIMARY_TEST_METRIC}_ci95_high"
                ],
                "comparison_scope": CLAIM_SCOPE,
                "rank_rule": "descending_arithmetic_mean_over_all_five_seeds",
                "ranking_exclusion_reason": "",
            }
        )
    excluded = next(row for row in summary_rows if row["method"] == INCOMPLETE_BASELINE)
    ranking_rows.append(
        {
            "rank": "",
            "method": excluded["method"],
            "n_seeds": excluded["n_seeds"],
            "ranking_eligible": 0,
            "primary_metric": PRIMARY_TEST_METRIC,
            "primary_metric_mean": "",
            "primary_metric_sd": "",
            "primary_metric_ci95_low": "",
            "primary_metric_ci95_high": "",
            "comparison_scope": CLAIM_SCOPE,
            "rank_rule": "not_ranked_incomplete_zero_of_five",
            "ranking_exclusion_reason": "zero_of_five_seed_runs_complete",
        }
    )

    v4_by_seed = rows_by_seed(combined, V4_METHOD)
    difference_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for method in complete_baselines:
        baseline_by_seed = rows_by_seed(combined, method)
        differences: list[float] = []
        baseline_values: list[float] = []
        v4_values: list[float] = []
        for seed in SEEDS:
            baseline_value = finite_float(baseline_by_seed[seed], PRIMARY_TEST_METRIC)
            v4_value = finite_float(v4_by_seed[seed], PRIMARY_TEST_METRIC)
            difference = v4_value - baseline_value
            baseline_values.append(baseline_value)
            v4_values.append(v4_value)
            differences.append(difference)
            difference_rows.append(
                {
                    "baseline_method": method,
                    "seed": seed,
                    "baseline_test_unknown_occupied_f1": baseline_value,
                    "v4_test_unknown_occupied_f1": v4_value,
                    "v4_minus_baseline": difference,
                    "seed_excluded": 0,
                    "comparison_scope": CLAIM_SCOPE,
                }
            )
        difference_statistics = mean_sd_ci(differences)
        contrast_rows.append(
            {
                "baseline_method": method,
                "n_matched_seeds": len(SEEDS),
                "baseline_mean": mean_sd_ci(baseline_values)["mean"],
                "v4_mean": mean_sd_ci(v4_values)["mean"],
                "v4_minus_baseline_mean": difference_statistics["mean"],
                "v4_minus_baseline_sd": difference_statistics["sd"],
                "v4_minus_baseline_ci95_low": difference_statistics["ci95_low"],
                "v4_minus_baseline_ci95_high": difference_statistics["ci95_high"],
                "positive_direction": "favours_v4",
                "comparison_type": "descriptive_matched_seed_contrast",
                "comparison_scope": CLAIM_SCOPE,
            }
        )
    return {
        "combined_per_seed_results.tsv": combined,
        "combined_method_status.tsv": status_rows,
        "combined_method_summary.tsv": summary_rows,
        "primary_metric_ranking.tsv": ranking_rows,
        "v4_vs_baseline_matched_seed_differences.tsv": difference_rows,
        "v4_vs_baseline_matched_seed_summary.tsv": contrast_rows,
    }


def read_completion(
    run: Path,
    stage: str,
    seed: int,
    expected_parent_teacher_sha256: str | None,
) -> dict[str, Any]:
    verify_sha_file(run)
    value = json.loads((run / "completion.json").read_text(encoding="utf-8"))
    run_manifest = json.loads(
        (run / "run_manifest.json").read_text(encoding="utf-8")
    )
    if not value.get("completed") or value.get("stage") != stage:
        raise ValueError(f"incomplete V4 {stage} run: {run}")
    if int(value.get("seed", -1)) != seed:
        raise ValueError("V4 run seed differs")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V4 completion protocol differs")
    if (
        run_manifest.get("protocol_id") != PROTOCOL_ID
        or run_manifest.get("stage") != stage
        or int(run_manifest.get("seed", -1)) != seed
    ):
        raise ValueError("V4 run manifest identity differs")
    configuration = run_manifest.get("configuration")
    if not isinstance(configuration, dict) or configuration.get(
        "protocol_id"
    ) != PROTOCOL_ID:
        raise ValueError("V4 run-manifest configuration protocol differs")
    if value.get("selection_metric") != (
        "validation_unknown_occupied_micro_f1_at_0.5"
    ):
        raise ValueError("V4 run does not use the primary F1 selector")
    selection_path = run / "selection_manifest.json"
    if value.get("selection_manifest_sha256") != sha256(selection_path):
        raise ValueError("V4 selection-manifest SHA differs")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("protocol_id") != PROTOCOL_ID
        or selection.get("stage") != stage
        or int(selection.get("seed", -1)) != seed
        or selection.get("source") != SELECTION_SOURCE
    ):
        raise ValueError("V4 selection-manifest identity or source differs")
    provenance = selection.get("run_provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("V4 selection-manifest provenance is missing")
    for key, expected_value in provenance.items():
        if run_manifest.get(key) != expected_value:
            raise ValueError(f"V4 selection provenance differs for {key}")
    expected_parent_kind = (
        PRIMARY_CHECKPOINT_KIND if stage == "student" else None
    )
    if stage == "student" and not expected_parent_teacher_sha256:
        raise ValueError("student aggregate requires the campaign teacher SHA")
    if stage == "teacher" and expected_parent_teacher_sha256 is not None:
        raise ValueError("teacher aggregate must not receive a parent SHA")
    for source_name, source in (
        ("completion", value),
        ("run_manifest", run_manifest),
        ("selection_provenance", provenance),
    ):
        if source.get("parent_teacher_checkpoint_kind") != expected_parent_kind:
            raise ValueError(f"V4 {source_name} parent checkpoint kind differs")
        if source.get("parent_teacher_checkpoint_sha256") != (
            expected_parent_teacher_sha256
        ):
            raise ValueError(f"V4 {source_name} parent teacher SHA differs")
    primary = selection["primary"]
    calibration = selection["calibration"]
    if primary.get("checkpoint_kind") != PRIMARY_CHECKPOINT_KIND:
        raise ValueError("V4 primary selection checkpoint kind differs")
    if calibration.get("checkpoint_kind") != CALIBRATION_CHECKPOINT_KIND:
        raise ValueError("V4 calibration selection checkpoint kind differs")
    if primary["checkpoint_sha256"] != value["best_primary_checkpoint_sha256"]:
        raise ValueError("V4 primary checkpoint SHA differs")
    if calibration["checkpoint_sha256"] != value[
        "best_calibration_checkpoint_sha256"
    ]:
        raise ValueError("V4 calibration checkpoint SHA differs")
    if int(primary["epoch"]) != int(value["best_primary_epoch"]):
        raise ValueError("V4 primary epoch differs")
    if int(calibration["epoch"]) != int(value["best_calibration_epoch"]):
        raise ValueError("V4 calibration epoch differs")
    value["audited_selection"] = selection
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--uniform-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing non-empty aggregate output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    teacher_run = args.campaign / "runs" / "teacher_seed_101"
    teacher = read_completion(teacher_run, "teacher", 101, None)
    campaign_teacher_sha = teacher["best_primary_checkpoint_sha256"]
    curve_rows: list[dict[str, Any]] = [
        {"stage": "teacher", "seed": 101, **row}
        for row in read_tsv(teacher_run / "metrics_by_epoch.tsv")
    ]
    teacher_selection = teacher["audited_selection"]
    teacher_summary = {
        "stage": "teacher",
        "seed": 101,
        "best_primary_epoch": teacher_selection["primary"]["epoch"],
        **{
            f"best_primary_validation_{key}": value
            for key, value in teacher_selection["primary"][
                "validation_metrics"
            ].items()
            if key != "epoch"
        },
        "best_calibration_epoch": teacher_selection["calibration"]["epoch"],
        **{
            f"best_calibration_validation_{key}": value
            for key, value in teacher_selection["calibration"][
                "validation_metrics"
            ].items()
            if key != "epoch"
        },
        "best_primary_checkpoint_sha256": teacher[
            "best_primary_checkpoint_sha256"
        ],
        "best_calibration_checkpoint_sha256": teacher[
            "best_calibration_checkpoint_sha256"
        ],
    }
    write_tsv(args.output / "teacher_selection.tsv", [teacher_summary])

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        run = args.campaign / "runs" / f"student_seed_{seed}"
        completion = read_completion(
            run, "student", seed, campaign_teacher_sha
        )
        selection = completion["audited_selection"]
        primary_metrics = selection["primary"]["validation_metrics"]
        calibration_metrics = selection["calibration"]["validation_metrics"]
        validation_path = (
            args.campaign
            / "evaluation"
            / f"student_seed_{seed}"
            / "validation"
            / "summary.json"
        )
        test_path = (
            args.campaign
            / "evaluation"
            / f"student_seed_{seed}"
            / "test"
            / "summary.json"
        )
        verify_sha_file(validation_path.parent)
        verify_sha_file(test_path.parent)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        test = json.loads(test_path.read_text(encoding="utf-8"))
        resource = json.loads(
            (run / "resource_profile.json").read_text(encoding="utf-8")
        )
        if validation["seed"] != seed or test["seed"] != seed:
            raise ValueError("evaluation seed differs from its V4 run")
        row: dict[str, Any] = {
            "method": V4_METHOD,
            "seed": seed,
            "best_primary_epoch": int(selection["primary"]["epoch"]),
            "best_primary_validation_unknown_f1": primary_metrics["unknown_f1"],
            "best_primary_validation_unknown_iou": primary_metrics["unknown_iou"],
            "best_primary_validation_unknown_bce": primary_metrics["unknown_bce"],
            "best_primary_validation_unknown_brier": primary_metrics[
                "unknown_brier"
            ],
            "best_primary_validation_unknown_ece15": primary_metrics[
                "unknown_ece15"
            ],
            "best_calibration_epoch": int(selection["calibration"]["epoch"]),
            "best_calibration_validation_unknown_brier": calibration_metrics[
                "unknown_brier"
            ],
            "best_calibration_validation_unknown_bce": calibration_metrics[
                "unknown_bce"
            ],
            "training_time_s": resource["training_time_s"],
            "latency_batch1_median_ms": resource["latency_batch1"]["median_ms"],
            "latency_batch1_p10_ms": resource["latency_batch1"]["p10_ms"],
            "latency_batch1_p90_ms": resource["latency_batch1"]["p90_ms"],
        }
        for split, payload in (("validation", validation), ("test", test)):
            for name, value in payload["macro_means"].items():
                row[f"{split}_{name}"] = value
        per_seed.append(row)
        curve_rows.extend(
            {"stage": "student", "seed": seed, **curve}
            for curve in read_tsv(run / "metrics_by_epoch.tsv")
        )

    write_tsv(args.output / "per_seed_results.tsv", per_seed)
    write_tsv(args.output / "training_curves.tsv", curve_rows)
    numeric_metrics = [
        name for name in per_seed[0] if name not in {"method", "seed"}
    ]
    summary_rows = [
        {
            "metric": metric,
            **mean_sd_ci([float(row[metric]) for row in per_seed]),
        }
        for metric in numeric_metrics
    ]
    write_tsv(args.output / "five_seed_summary.tsv", summary_rows)

    verify_sha_file(args.uniform_aggregate)
    uniform_manifest = json.loads(
        (args.uniform_aggregate / "aggregate_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        uniform_manifest.get("methods_registered") != len(BASELINE_METHODS)
        or uniform_manifest.get("methods_complete") != len(BASELINE_METHODS) - 1
        or uniform_manifest.get("best_seed_selection_used") is not False
    ):
        raise ValueError("uniform aggregate manifest differs from the frozen archive")
    all_uniform_rows = read_tsv(args.uniform_aggregate / "per_seed_results.tsv")
    uniform_status_rows = read_tsv(args.uniform_aggregate / "method_status.tsv")
    uniform_summary_rows = read_tsv(args.uniform_aggregate / "method_summary.tsv")
    validate_uniform_summary(all_uniform_rows, uniform_summary_rows)
    benchmark_tables = build_archive_benchmark_tables(
        per_seed, all_uniform_rows, uniform_status_rows
    )
    for filename, rows in benchmark_tables.items():
        write_tsv(args.output / filename, rows)

    uniform_rows = rows_by_seed(all_uniform_rows, "MSO")
    uniform = uniform_rows
    if set(uniform) != set(SEEDS):
        raise ValueError("uniform-objective MSO lacks the five matched seeds")
    paired_mapping = {
        "selected_epoch": ("best_epoch", "best_primary_epoch"),
        "validation_bce_at_selected_checkpoint": (
            "best_validation_unknown_bce",
            "best_primary_validation_unknown_bce",
        ),
        "test_unknown_occupied_f1": (
            "test_unknown_occupied_f1",
            "test_unknown_occupied_f1",
        ),
        "test_unknown_occupied_iou": (
            "test_unknown_occupied_iou",
            "test_unknown_occupied_iou",
        ),
        "test_unknown_occupied_brier": (
            "test_unknown_occupied_brier",
            "test_unknown_occupied_brier",
        ),
        "test_unknown_occupied_ece": (
            "test_unknown_occupied_ece",
            "test_unknown_occupied_ece",
        ),
        "latency_batch1_median_ms": (
            "latency_batch1_median_ms",
            "latency_batch1_median_ms",
        ),
    }
    paired_rows = []
    for row in per_seed:
        seed = int(row["seed"])
        for label, (uniform_key, v4_key) in paired_mapping.items():
            uniform_value = float(uniform[seed][uniform_key])
            v4_value = float(row[v4_key])
            paired_rows.append(
                {
                    "seed": seed,
                    "metric": label,
                    "uniform_objective": uniform_value,
                    "paper_equation_reconstruction_v4": v4_value,
                    "v4_minus_uniform": v4_value - uniform_value,
                    "v4_selection_semantics": (
                        "primary_unknown_occupied_micro_f1_at_0.5"
                    ),
                    "comparison_scope": (
                        "descriptive_protocol_comparison_different_training_budgets"
                    ),
                }
            )
    write_tsv(args.output / "paired_protocol_comparison.tsv", paired_rows)

    outputs = (
        "teacher_selection.tsv",
        "per_seed_results.tsv",
        "training_curves.tsv",
        "five_seed_summary.tsv",
        "paired_protocol_comparison.tsv",
        *benchmark_tables.keys(),
    )
    ranking_rows = benchmark_tables["primary_metric_ranking.tsv"]
    v4_rank = int(next(row["rank"] for row in ranking_rows if row["method"] == V4_METHOD))
    manifest = {
        "status": "prospective_paper_equation_reconstruction_v4_aggregate",
        "selection_semantics": (
            "primary_unknown_occupied_micro_f1_at_0.5_not_minimum_bce"
        ),
        "teacher_seed": 101,
        "student_seeds": list(SEEDS),
        "best_seed_selection_used": False,
        "seed_exclusions": 0,
        "ranking_primary_metric": PRIMARY_TEST_METRIC,
        "ranking_rule": "descending_arithmetic_mean_over_all_five_seeds",
        "comparison_scope": CLAIM_SCOPE,
        "universal_sota_claim_supported": False,
        "registered_methods_including_v4": len(BASELINE_METHODS) + 1,
        "ranking_eligible_methods": len(BASELINE_METHODS),
        "all_completed_baselines_included": True,
        "repaint_completed_seeds": 0,
        "repaint_ranking_eligible": False,
        "v4_frozen_archive_rank": v4_rank,
        "v4_ranked_first_within_frozen_archive_benchmark": v4_rank == 1,
        "matched_seed_contrast_interval": (
            "two-sided_95_percent_t_interval_df4_over_all_five_seed_differences"
        ),
        "cross_method_contrasts_are_descriptive": True,
        "uniform_aggregate_sha256_manifest": sha256(
            args.uniform_aggregate / "SHA256SUMS"
        ),
        "aggregate_script_sha256": sha256(Path(__file__)),
        "files": {name: sha256(args.output / name) for name in outputs},
    }
    manifest_path = args.output / "aggregate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = (*outputs, "aggregate_manifest.json")
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(args.output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
