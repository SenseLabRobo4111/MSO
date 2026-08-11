#!/usr/bin/env python3
"""Plot the completed MSO paper-equation reconstruction audit.

Figure contract
---------------
Core conclusion: show whether the complete reconstructed objective changes
five-seed optimisation stability and predictive metrics, without assuming a
favourable direction.
Archetype: quantitative grid.
Evidence: raw 500-epoch curves, matched-seed protocol contrasts, raw seed
metrics with mean plus sample SD, and latency-versus-F1 operating points.
Reviewer risk: the uniform-objective comparator used a different training
budget, so all cross-protocol contrasts are explicitly descriptive.
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


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
plt.rcParams["font.size"] = 6.4
plt.rcParams["axes.titlesize"] = 7.0
plt.rcParams["axes.labelsize"] = 6.7
plt.rcParams["xtick.labelsize"] = 6.2
plt.rcParams["ytick.labelsize"] = 6.2
plt.rcParams["legend.fontsize"] = 6.0
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


SEEDS = (11, 23, 37, 53, 71)
BLUE = "#72A7E2"
BLUE_DARK = "#4F8FD6"
BLUE_LIGHT = "#B8D5F2"
PURPLE = "#8A7BD1"
PURPLE_DARK = "#6F63B6"
PURPLE_LIGHT = "#C9C0EC"
GRID = "#E6EBF3"
TEXT = "#28324A"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha_file(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise ValueError(f"missing SHA256SUMS: {root}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"hash mismatch: {path}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axis.tick_params(length=2.5, width=0.6, color=TEXT)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    axis.title.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color(TEXT)


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


def raw_seed_summary(
    axis: plt.Axes,
    labels: list[str],
    uniform: list[list[float]],
    complete: list[list[float]],
    ylabel: str,
) -> None:
    centers = np.arange(len(labels), dtype=float)
    offsets = (-0.15, 0.15)
    for values, offset, color, name in (
        (uniform, offsets[0], BLUE, "Uniform objective"),
        (complete, offsets[1], PURPLE, "Complete reconstruction"),
    ):
        for index, series in enumerate(values):
            array = np.asarray(series, dtype=float)
            jitter = np.linspace(-0.045, 0.045, len(array))
            axis.scatter(
                np.full(len(array), centers[index] + offset) + jitter,
                array,
                s=12,
                color=color,
                alpha=0.75,
                linewidth=0,
                zorder=3,
            )
            axis.errorbar(
                centers[index] + offset,
                array.mean(),
                yerr=array.std(ddof=1),
                fmt="o",
                markersize=3.8,
                color=color,
                capsize=2.4,
                linewidth=1.0,
                label=name if index == 0 else None,
                zorder=4,
            )
    axis.set_xticks(centers, labels)
    axis.set_ylabel(ylabel)
    axis.legend(loc="best", borderaxespad=0.3)
    style_axis(axis)


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
    paired = read_tsv(args.aggregate / "paired_protocol_comparison.tsv")
    per_seed = read_tsv(args.aggregate / "per_seed_results.tsv")
    if len(per_seed) != len(SEEDS):
        raise ValueError("plot requires five completed student seeds")
    by_seed = {int(row["seed"]): row for row in per_seed}
    if set(by_seed) != set(SEEDS):
        raise ValueError("plot seed set differs from the frozen contract")

    figure = plt.figure(figsize=(7.204724, 5.748031), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(1.05, 1.0, 1.0))
    axes = [figure.add_subplot(grid[row, column]) for row in range(3) for column in range(2)]

    teacher_rows = [row for row in curves if row["stage"] == "teacher"]
    teacher_epoch = np.asarray([int(row["epoch"]) for row in teacher_rows])
    teacher_bce = np.asarray(
        [float(row["validation_unknown_bce"]) for row in teacher_rows]
    )
    axes[0].plot(teacher_epoch, teacher_bce, color=BLUE_DARK, linewidth=1.2)
    teacher_best = int(np.argmin(teacher_bce))
    axes[0].scatter(
        teacher_epoch[teacher_best], teacher_bce[teacher_best],
        s=18, color=PURPLE, edgecolor="white", linewidth=0.5, zorder=3,
        label=f"Best epoch {teacher_epoch[teacher_best]}",
    )
    axes[0].set_title("Teacher optimisation on the current split")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation unknown BCE")
    axes[0].legend(loc="upper right")
    style_axis(axes[0])
    panel_label(axes[0], "a")

    seed_colors = ("#B6D5F2", "#9FC3EA", "#91AEE1", "#A99AD9", "#8878CA")
    for seed, color in zip(SEEDS, seed_colors):
        rows = [
            row for row in curves
            if row["stage"] == "student" and int(row["seed"]) == seed
        ]
        epochs = [int(row["epoch"]) for row in rows]
        values = [float(row["validation_unknown_bce"]) for row in rows]
        axes[1].plot(epochs, values, color=color, linewidth=0.9, label=f"Seed {seed}")
    axes[1].set_title("Student optimisation, all five seeds")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation unknown BCE")
    axes[1].legend(loc="upper right", ncol=2, columnspacing=0.6, handlelength=1.2)
    style_axis(axes[1])
    panel_label(axes[1], "b")

    bce_rows = [row for row in paired if row["metric"] == "best_validation_unknown_bce"]
    for row in sorted(bce_rows, key=lambda item: int(item["seed"])):
        values = [float(row["uniform_objective"]), float(row["paper_equation_reconstruction"])]
        axes[2].plot([0, 1], values, color=PURPLE_LIGHT, linewidth=0.8, zorder=1)
        axes[2].scatter([0, 1], values, color=(BLUE, PURPLE), s=16, zorder=2)
        axes[2].text(1.04, values[1], row["seed"], fontsize=5.8, va="center", color=TEXT)
    axes[2].set_xlim(-0.2, 1.22)
    axes[2].set_xticks([0, 1], ["Uniform\nobjective", "Complete\nreconstruction"])
    axes[2].set_ylabel("Best validation unknown BCE")
    axes[2].set_title("Matched-seed protocol contrast")
    axes[2].text(
        0.02, 0.04, "Different training budgets; descriptive only",
        transform=axes[2].transAxes, fontsize=5.8, color=TEXT,
    )
    style_axis(axes[2])
    panel_label(axes[2], "c")

    metric_lookup: dict[str, dict[int, tuple[float, float]]] = {}
    for row in paired:
        metric_lookup.setdefault(row["metric"], {})[int(row["seed"])] = (
            float(row["uniform_objective"]),
            float(row["paper_equation_reconstruction"]),
        )
    quality_metrics = ["test_unknown_occupied_f1", "test_unknown_occupied_iou"]
    quality_uniform = [
        [metric_lookup[name][seed][0] for seed in SEEDS] for name in quality_metrics
    ]
    quality_complete = [
        [metric_lookup[name][seed][1] for seed in SEEDS] for name in quality_metrics
    ]
    raw_seed_summary(
        axes[3], ["F1", "IoU"], quality_uniform, quality_complete,
        "Test unknown-region score",
    )
    axes[3].set_title("Predictive quality across seeds")
    panel_label(axes[3], "d")

    calibration_metrics = [
        "test_unknown_occupied_brier",
        "test_unknown_occupied_ece",
    ]
    calibration_uniform = [
        [metric_lookup[name][seed][0] for seed in SEEDS]
        for name in calibration_metrics
    ]
    calibration_complete = [
        [metric_lookup[name][seed][1] for seed in SEEDS]
        for name in calibration_metrics
    ]
    raw_seed_summary(
        axes[4], ["Brier", "ECE"], calibration_uniform, calibration_complete,
        "Test calibration error",
    )
    axes[4].set_title("Probability calibration across seeds")
    panel_label(axes[4], "e")

    for seed in SEEDS:
        full_epoch = float(by_seed[seed]["best_epoch"])
        full_f1 = float(by_seed[seed]["test_unknown_occupied_f1"])
        uniform_epoch = metric_lookup["best_epoch"][seed][0]
        uniform_f1 = metric_lookup["test_unknown_occupied_f1"][seed][0]
        axes[5].plot(
            [uniform_epoch, full_epoch], [uniform_f1, full_f1],
            color=PURPLE_LIGHT, linewidth=0.7, zorder=1,
        )
        axes[5].scatter(uniform_epoch, uniform_f1, s=14, color=BLUE, zorder=2)
        axes[5].scatter(full_epoch, full_f1, s=18, color=PURPLE, zorder=3)
        axes[5].text(full_epoch, full_f1, f" {seed}", fontsize=5.7, va="center")
    axes[5].scatter([], [], s=14, color=BLUE, label="Uniform objective")
    axes[5].scatter([], [], s=18, color=PURPLE, label="Complete reconstruction")
    axes[5].set_xlabel("Validation-selected epoch")
    axes[5].set_ylabel("Test unknown-region F1")
    axes[5].set_title("Selection trajectory and test quality")
    axes[5].legend(loc="best")
    style_axis(axes[5])
    panel_label(axes[5], "f")

    base = args.output / "full_mso_reconstruction_audit"
    figure.savefig(base.with_suffix(".svg"), facecolor="white")
    figure.savefig(base.with_suffix(".pdf"), facecolor="white")
    figure.savefig(base.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(base.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(figure)

    outputs = [base.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png", ".tiff")]
    qa = {
        "status": "generated_pending_final_human_visual_inspection",
        "backend": "Python matplotlib only",
        "final_width_mm": 183,
        "final_height_mm": 146,
        "minimum_source_font_pt": 6.0,
        "student_seed_count": len(SEEDS),
        "student_seeds": list(SEEDS),
        "line_smoothing_or_interpolation": False,
        "seed_exclusions": 0,
        "center": "arithmetic mean over five seeds where shown",
        "spread": "sample standard deviation over five seeds where shown",
        "cross_protocol_inference_performed": False,
        "known_risk": "uniform and complete objectives have different training budgets",
        "plot_script_sha256": sha256(Path(__file__)),
        "aggregate_sha256_manifest": sha256(args.aggregate / "SHA256SUMS"),
        "source_files": {
            "training_curves.tsv": sha256(args.aggregate / "training_curves.tsv"),
            "paired_protocol_comparison.tsv": sha256(
                args.aggregate / "paired_protocol_comparison.tsv"
            ),
            "per_seed_results.tsv": sha256(args.aggregate / "per_seed_results.tsv"),
        },
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    qa_path = args.output / "FIGURE_QA.json"
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    names = [path.name for path in outputs] + [qa_path.name]
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(args.output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
