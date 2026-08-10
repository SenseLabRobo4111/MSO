"""Render traceable final figures from the frozen aggregate tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = (
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

PALETTE = {
    "mso": "#806FC1",
    "blue": "#79A9DC",
    "blue_dark": "#526F9E",
    "purple": "#A89AD6",
    "grey": "#CBD2DC",
    "text": "#263347",
    "grid": "#E8ECF2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def finite_value(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", None}:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def metric_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    xlabel: str,
    higher_is_better: bool,
) -> None:
    selected = []
    for row in rows:
        mean = finite_value(row, f"{metric}_mean")
        low = finite_value(row, f"{metric}_ci95_low")
        high = finite_value(row, f"{metric}_ci95_high")
        if mean is not None and low is not None and high is not None:
            selected.append((row["method"], mean, low, high))
    selected.sort(key=lambda item: item[1], reverse=higher_is_better)
    if not selected:
        ax.text(0.5, 0.5, "Pending aggregate", ha="center", va="center")
        ax.set_axis_off()
        return
    y = np.arange(len(selected))
    means = np.asarray([item[1] for item in selected])
    lows = np.asarray([item[2] for item in selected])
    highs = np.asarray([item[3] for item in selected])
    colors = [PALETTE["mso"] if item[0] == "MSO" else PALETTE["blue"] for item in selected]
    ax.errorbar(
        means,
        y,
        xerr=np.vstack((means - lows, highs - means)),
        fmt="none",
        ecolor=PALETTE["grey"],
        elinewidth=1.4,
        capsize=2.2,
        zorder=1,
    )
    ax.scatter(means, y, s=30, c=colors, edgecolor="white", linewidth=0.5, zorder=2)
    ax.set_yticks(y, [item[0] for item in selected])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)


def render(aggregate: Path, output: Path) -> dict[str, object]:
    status_path = aggregate / "method_status.tsv"
    summary_path = aggregate / "method_summary.tsv"
    status_rows = read_tsv(status_path)
    summary_rows = read_tsv(summary_path)
    status_by_method = {row["method"]: row for row in status_rows}
    summary_by_method = {row["method"]: row for row in summary_rows}
    ordered_status = [status_by_method[name] for name in METHOD_ORDER]
    ordered_summary = [summary_by_method[name] for name in METHOD_ORDER]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.4,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    # 7.205 in is the 183 mm double-column target width.
    fig = plt.figure(figsize=(7.205, 6.457), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1, 1), hspace=0.44, wspace=0.55)

    ax = fig.add_subplot(grid[0, 0])
    completed = np.asarray([int(row["completed_seeds"]) for row in ordered_status])
    y = np.arange(len(METHOD_ORDER))
    colors = [PALETTE["mso"] if name == "MSO" else PALETTE["blue"] for name in METHOD_ORDER]
    colors = [PALETTE["grey"] if value == 0 else color for color, value in zip(colors, completed)]
    ax.barh(y, completed, color=colors, height=0.68)
    ax.set_yticks(y, METHOD_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, 5.25)
    ax.set_xticks(range(6))
    ax.set_xlabel("Completed seeds")
    ax.set_title("a  Registered-method status", loc="left", fontweight="bold")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)
    for yy, value in zip(y, completed):
        ax.text(value + 0.08, yy, f"{value}/5", va="center", fontsize=5.6)

    panels = (
        ("test_unknown_occupied_f1", "b  Unknown-region F1", "F1 (higher is better)", True),
        ("test_unknown_occupied_iou", "c  Unknown-region IoU", "IoU (higher is better)", True),
        ("test_unknown_occupied_brier", "d  Probability calibration", "Brier score (lower is better)", False),
        ("test_boundary_f1_tolerance_2px", "e  Boundary fidelity", "Boundary F1 (higher is better)", True),
    )
    for slot, panel in zip((grid[0, 1], grid[0, 2], grid[1, 0], grid[1, 1]), panels):
        metric_panel(fig.add_subplot(slot), ordered_summary, *panel)

    ax = fig.add_subplot(grid[1, 2])
    points = []
    for row in ordered_summary:
        params = finite_value(row, "parameter_count_mean")
        latency = finite_value(row, "latency_batch1_median_ms_mean")
        f1 = finite_value(row, "test_unknown_occupied_f1_mean")
        if params is not None and latency is not None and f1 is not None and params > 0 and latency > 0:
            points.append((row["method"], params / 1e6, latency, f1))
    if points:
        resource_values = np.asarray([(point[1], point[2]) for point in points])
        if np.any(resource_values <= 0):
            raise ValueError("log-scale resource values must be strictly positive")
        f1_values = np.asarray([point[3] for point in points])
        sizes = 25 + 70 * (f1_values - f1_values.min()) / max(np.ptp(f1_values), 1e-12)
        for point, size in zip(points, sizes):
            color = PALETTE["mso"] if point[0] == "MSO" else PALETTE["blue"]
            ax.scatter(point[1], point[2], s=size, color=color, edgecolor="white", linewidth=0.5)
            ax.annotate(point[0], (point[1], point[2]), xytext=(3, 2), textcoords="offset points", fontsize=5.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Use plain-number major ticks so logarithmic mathtext does not shrink
        # exponent glyphs below the 5 pt publication floor at final size.
        ax.set_xticks((1.0, 10.0), ("1", "10"))
        ax.set_yticks((3.0, 4.0, 6.0), ("3", "4", "6"))
        ax.minorticks_off()
        ax.set_xlabel("Parameters (millions, log scale)")
        ax.set_ylabel("Batch-1 latency (ms, log scale)")
        ax.grid(color=PALETTE["grid"], linewidth=0.7)
    else:
        ax.text(0.5, 0.5, "Pending resource aggregate", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("f  Accuracy–resource context", loc="left", fontweight="bold")

    fig.suptitle(
        "Common-objective diagnostic: completion, predictive quality and resource context",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["text"],
    )
    fig.text(
        0.06,
        0.018,
        "Shared BCE/Dice objective; this diagnostic does not evaluate the complete MSO training method.",
        ha="left",
        fontsize=5.4,
        color=PALETTE["text"],
    )
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.12, top=0.945)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "uniform_benchmark_summary"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    result = {
        "status": "rendered_from_final_aggregate",
        "methods_registered": len(METHOD_ORDER),
        "methods_complete": sum(int(row["completed_seeds"]) == 5 for row in ordered_status),
        "uncertainty": "two-sided 95% t interval over five seed estimates",
        "input_sha256": {
            status_path.name: sha256(status_path),
            summary_path.name: sha256(summary_path),
        },
        "outputs": {},
    }
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        path = stem.with_suffix(suffix)
        result["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = output / "visualization_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_lines = [
        f"{sha256(output / name)}  {name}"
        for name in (*result["outputs"], manifest.name)
    ]
    (output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite visualization output: {args.output}")
    print(json.dumps(render(args.aggregate, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
