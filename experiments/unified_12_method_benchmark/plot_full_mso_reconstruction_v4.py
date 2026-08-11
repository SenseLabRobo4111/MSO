#!/usr/bin/env python3
"""Render the V4 checkpoint-selection and frozen-archive ranking audit.

Figure contract
---------------
Core conclusion: the figure determines where full V4 MSO ranks on the frozen
archive benchmark while separately auditing its predeclared checkpoint rule.
Evidence chain: teacher F1 and Brier trajectories; all student F1 trajectories;
primary-versus-calibration epochs; the complete five-seed test-F1 ranking; and
V4-minus-each-baseline matched-seed contrasts. Archetype: quantitative grid.
Reviewer risk: training budgets differ, RePaint has 0/5 completed seeds, and
the archive ranking cannot support a universal state-of-the-art claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (11, 23, 37, 53, 71)
BLUE = "#72A7E2"
BLUE_DARK = "#4F8FD6"
BLUE_LIGHT = "#B8D5F2"
PURPLE = "#8A7BD1"
PURPLE_DARK = "#6F63B6"
PURPLE_LIGHT = "#C9C0EC"
GRID = "#E6EBF3"
TEXT = "#28324A"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.4,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.7,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha_file(root: Path) -> None:
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"aggregate hash mismatch: {relative}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axis.tick_params(length=2.5, width=0.6, color=TEXT)
    for spine in axis.spines.values():
        spine.set_color(TEXT)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    axis.title.set_color(TEXT)


def panel_label(axis: plt.Axes, value: str) -> None:
    axis.text(
        0.01,
        0.98,
        value,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=TEXT,
    )


def interval_forest(
    axis: plt.Axes,
    labels: list[str],
    means: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    colors: list[str],
    xlabel: str,
    raw_values: list[list[float]],
) -> None:
    positions = np.arange(len(labels), dtype=float)
    for position, mean, low, high, color, raw in zip(
        positions, means, lows, highs, colors, raw_values
    ):
        raw_array = np.asarray(raw, dtype=float)
        if len(raw_array) != len(SEEDS) or not np.isfinite(raw_array).all():
            raise ValueError("forest row requires all five finite seed values")
        axis.scatter(
            raw_array,
            position + np.linspace(-0.12, 0.12, len(raw_array)),
            s=7,
            color=color,
            alpha=0.42,
            linewidth=0,
            zorder=2,
        )
        axis.errorbar(
            mean,
            position,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            markersize=3.8,
            color=color,
            capsize=2.1,
            linewidth=0.9,
            zorder=3,
        )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    style_axis(axis)
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_sha_file(args.aggregate)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing non-empty figure output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    curves = read_tsv(args.aggregate / "training_curves.tsv")
    per_seed = read_tsv(args.aggregate / "per_seed_results.tsv")
    teacher = read_tsv(args.aggregate / "teacher_selection.tsv")
    ranking = read_tsv(args.aggregate / "primary_metric_ranking.tsv")
    contrasts = read_tsv(
        args.aggregate / "v4_vs_baseline_matched_seed_summary.tsv"
    )
    method_status = read_tsv(args.aggregate / "combined_method_status.tsv")
    combined_seeds = read_tsv(
        args.aggregate / "combined_per_seed_results.tsv"
    )
    matched_differences = read_tsv(
        args.aggregate / "v4_vs_baseline_matched_seed_differences.tsv"
    )
    if len(per_seed) != len(SEEDS) or len(teacher) != 1:
        raise ValueError("V4 figure requires one teacher and five student seeds")
    by_seed = {int(row["seed"]): row for row in per_seed}
    if set(by_seed) != set(SEEDS):
        raise ValueError("V4 figure seed set differs")
    eligible_ranking = [row for row in ranking if row["ranking_eligible"] == "1"]
    eligible_ranking.sort(key=lambda row: int(row["rank"]))
    expected_ranks = list(range(1, 13))
    if (
        len(eligible_ranking) != len(expected_ranks)
        or [int(row["rank"]) for row in eligible_ranking] != expected_ranks
    ):
        raise ValueError("V4 figure requires the complete 12-method ranking")
    v4_name = "MSO-Paper-Equation-Reconstructed-V4"
    if sum(row["method"] == v4_name for row in eligible_ranking) != 1:
        raise ValueError("V4 occurs incorrectly in the archive ranking")
    status = {row["method"]: row for row in method_status}
    if (
        len(status) != 13
        or len(status) != len(method_status)
        or status.get("RePaint", {}).get("completed_seeds") != "0"
        or status.get("RePaint", {}).get("ranking_eligible") != "0"
        or status.get("RePaint", {}).get("ranking_exclusion_reason")
        != "zero_of_five_seed_runs_complete"
        or any(row.get("seed_exclusions") != "0" for row in method_status)
    ):
        raise ValueError("method status or RePaint 0/5 exclusion differs")
    excluded_ranking = [
        row for row in ranking if row["ranking_eligible"] == "0"
    ]
    if (
        len(excluded_ranking) != 1
        or excluded_ranking[0]["method"] != "RePaint"
        or excluded_ranking[0]["rank"] != ""
    ):
        raise ValueError("ranking must retain RePaint as the sole unranked method")
    if {row["method"] for row in eligible_ranking} != {
        method for method, row in status.items() if row["ranking_eligible"] == "1"
    }:
        raise ValueError("ranking and method-status eligibility differ")
    contrast_lookup = {row["baseline_method"]: row for row in contrasts}
    expected_baselines = {
        row["method"] for row in eligible_ranking if row["method"] != v4_name
    }
    if set(contrast_lookup) != expected_baselines or len(contrasts) != 11:
        raise ValueError(
            "V4 figure requires one contrast for every complete baseline"
        )
    seed_grid = {
        method: {
            int(row["seed"]): row
            for row in combined_seeds
            if row["method"] == method
        }
        for method in {row["method"] for row in eligible_ranking}
    }
    if len(combined_seeds) != 60 or any(
        set(rows) != set(SEEDS) for rows in seed_grid.values()
    ):
        raise ValueError("V4 figure requires all 60 complete per-seed rows")
    difference_grid = {
        method: {
            int(row["seed"]): row
            for row in matched_differences
            if row["baseline_method"] == method
        }
        for method in expected_baselines
    }
    if len(matched_differences) != 55 or any(
        set(rows) != set(SEEDS) for rows in difference_grid.values()
    ) or any(row.get("seed_excluded") != "0" for row in matched_differences):
        raise ValueError("V4 figure requires all 55 matched-seed differences")

    figure = plt.figure(figsize=(7.204724, 6.692913), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.05, 1.55))
    axes = [
        figure.add_subplot(grid[row_index, column_index])
        for row_index in range(3)
        for column_index in range(2)
    ]
    teacher_rows = [row for row in curves if row["stage"] == "teacher"]
    teacher_epochs = np.asarray([int(row["epoch"]) for row in teacher_rows])
    teacher_f1 = np.asarray(
        [float(row["validation_unknown_f1"]) for row in teacher_rows]
    )
    teacher_brier = np.asarray(
        [float(row["validation_unknown_brier"]) for row in teacher_rows]
    )
    primary_epoch = int(teacher[0]["best_primary_epoch"])
    calibration_epoch = int(teacher[0]["best_calibration_epoch"])

    axes[0].plot(teacher_epochs, teacher_f1, color=BLUE_DARK, linewidth=1.15)
    axes[0].scatter(
        primary_epoch,
        teacher_f1[primary_epoch],
        s=22,
        color=PURPLE,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label=f"Primary epoch {primary_epoch}",
    )
    axes[0].set_title("Teacher occupied-map discrimination")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation unknown micro-F1")
    axes[0].legend(loc="lower right")
    style_axis(axes[0])
    panel_label(axes[0], "a")

    axes[1].plot(teacher_epochs, teacher_brier, color=BLUE_DARK, linewidth=1.15)
    axes[1].scatter(
        calibration_epoch,
        teacher_brier[calibration_epoch],
        s=22,
        color=PURPLE,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label=f"Calibration epoch {calibration_epoch}",
    )
    axes[1].scatter(
        primary_epoch,
        teacher_brier[primary_epoch],
        s=18,
        facecolor="white",
        edgecolor=PURPLE_DARK,
        linewidth=0.8,
        zorder=3,
        label="Primary checkpoint",
    )
    axes[1].set_title("Teacher probability calibration")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation unknown Brier")
    axes[1].legend(loc="upper right")
    style_axis(axes[1])
    panel_label(axes[1], "b")

    seed_colors = ("#B6D5F2", "#9FC3EA", "#91AEE1", "#A99AD9", "#8878CA")
    for seed, color in zip(SEEDS, seed_colors):
        rows = [
            row
            for row in curves
            if row["stage"] == "student" and int(row["seed"]) == seed
        ]
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        values = np.asarray([float(row["validation_unknown_f1"]) for row in rows])
        selected_epoch = int(by_seed[seed]["best_primary_epoch"])
        axes[2].plot(epochs, values, color=color, linewidth=0.85, label=f"Seed {seed}")
        axes[2].scatter(
            selected_epoch,
            values[selected_epoch],
            s=10,
            color=color,
            edgecolor=TEXT,
            linewidth=0.25,
            zorder=3,
        )
    axes[2].set_title("Student trajectories and primary checkpoints")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Validation unknown micro-F1")
    axes[2].legend(loc="lower right", ncol=2, columnspacing=0.7)
    style_axis(axes[2])
    panel_label(axes[2], "c")

    for seed, color in zip(SEEDS, seed_colors):
        epochs = [
            float(by_seed[seed]["best_primary_epoch"]),
            float(by_seed[seed]["best_calibration_epoch"]),
        ]
        axes[3].plot(
            [0, 1], epochs, color=color, linewidth=0.9, label=f"Seed {seed}"
        )
        axes[3].scatter([0, 1], epochs, color=color, s=16, zorder=3)
    axes[3].set_xlim(-0.15, 1.15)
    axes[3].set_xticks([0, 1], ["Primary\nF1", "Calibration\nBrier"])
    axes[3].set_ylabel("Selected epoch")
    axes[3].set_title("Predeclared checkpoint trade-off")
    axes[3].legend(loc="upper right", ncol=2, columnspacing=0.7)
    style_axis(axes[3])
    panel_label(axes[3], "d")

    rank_labels = [
        f"{row['rank']}  "
        + ("MSO (full V4)" if row["method"] == v4_name else row["method"])
        for row in eligible_ranking
    ]
    rank_means = np.asarray(
        [float(row["primary_metric_mean"]) for row in eligible_ranking]
    )
    rank_lows = np.asarray(
        [float(row["primary_metric_ci95_low"]) for row in eligible_ranking]
    )
    rank_highs = np.asarray(
        [float(row["primary_metric_ci95_high"]) for row in eligible_ranking]
    )
    interval_forest(
        axes[4],
        rank_labels,
        rank_means,
        rank_lows,
        rank_highs,
        [PURPLE if row["method"] == v4_name else BLUE for row in eligible_ranking],
        "Test unknown occupied F1 (mean, 95% t CI; n = 5)",
        [
            [
                float(seed_grid[row["method"]][seed]["test_unknown_occupied_f1"])
                for seed in SEEDS
            ]
            for row in eligible_ranking
        ],
    )
    axes[4].set_title("Frozen-archive ranking; RePaint unranked (0/5)")
    panel_label(axes[4], "e")

    ordered_baselines = [
        row["method"] for row in eligible_ranking if row["method"] != v4_name
    ]
    contrast_rows = [contrast_lookup[method] for method in ordered_baselines]
    contrast_means = np.asarray(
        [float(row["v4_minus_baseline_mean"]) for row in contrast_rows]
    )
    contrast_lows = np.asarray(
        [float(row["v4_minus_baseline_ci95_low"]) for row in contrast_rows]
    )
    contrast_highs = np.asarray(
        [float(row["v4_minus_baseline_ci95_high"]) for row in contrast_rows]
    )
    interval_forest(
        axes[5],
        ordered_baselines,
        contrast_means,
        contrast_lows,
        contrast_highs,
        [PURPLE] * len(ordered_baselines),
        "V4 - baseline test F1 (mean, 95% t CI; n = 5)",
        [
            [
                float(difference_grid[method][seed]["v4_minus_baseline"])
                for seed in SEEDS
            ]
            for method in ordered_baselines
        ],
    )
    axes[5].axvline(0.0, color=TEXT, linewidth=0.7, linestyle="--", zorder=1)
    axes[5].set_title("Matched-seed contrasts to all complete baselines")
    panel_label(axes[5], "f")

    base = args.output / "full_mso_reconstruction_v4_selection_audit"
    figure.savefig(base.with_suffix(".svg"), facecolor="white")
    figure.savefig(base.with_suffix(".pdf"), facecolor="white")
    figure.savefig(base.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(base.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(figure)

    outputs = [base.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png", ".tiff")]
    qa = {
        "status": "generated_pending_final_human_visual_inspection",
        "backend": "Python matplotlib only",
        "figure_contract": (
            "V4 selection audit plus frozen-archive primary-F1 ranking"
        ),
        "reuse_level": "build anew with palette and typography inheritance only",
        "final_width_mm": 183,
        "final_height_mm": 170,
        "minimum_source_font_pt": 5.8,
        "student_seed_count": 5,
        "seed_exclusions": 0,
        "raw_seed_values_visible_in_ranking_and_contrasts": True,
        "ranked_method_count": 12,
        "complete_baseline_count": 11,
        "repaint_completed_seeds": 0,
        "repaint_ranking_eligible": False,
        "center": "arithmetic mean over all five seeds where shown",
        "spread": "two-sided 95% t interval with 4 degrees of freedom",
        "cross_protocol_inference_performed": False,
        "universal_sota_claim_supported": False,
        "known_risk": (
            "training budgets differ; ranking is limited to the frozen archive"
        ),
        "plot_script_sha256": sha256(Path(__file__)),
        "aggregate_sha256_manifest": sha256(args.aggregate / "SHA256SUMS"),
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    qa_path = args.output / "FIGURE_QA.json"
    qa_path.write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = [path.name for path in outputs] + [qa_path.name]
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(args.output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
