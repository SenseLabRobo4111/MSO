"""Render source-backed MRPB and physical endpoint panels.

The exploration traces are read directly from the retained aggregate CSV
records.  Every recorded row through step 1800 is drawn in file order: there
is no interpolation, smoothing, resampling or synthetic endpoint.  The
frontier-comparator quality field is identically zero in the retained file and
is treated as an unavailable-value sentinel rather than a measured trace.

The physical endpoint bars are read from the packaged tab-separated record.
They are descriptive endpoints from one run per method and scene; this script
does not construct uncertainty intervals or infer replicate-level variation.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "all_metrics"
FIG_DIR = ROOT / "figs"
PHYSICAL_SOURCE = ROOT.parent / "tables_and_figures" / "physical_endpoint_values.txt"

MAX_STEP = 1800.0
width_mm = 183.0
EXPLORATION_HEIGHT_MM = 86.0
PHYSICAL_HEIGHT_MM = 82.0
MM_PER_INCH = 25.4
RASTER_DPI = 600

EXPLORATION_PREFIX = FIG_DIR / "fig5_exploration"
PHYSICAL_PREFIX = FIG_DIR / "fig6_real_world"

# Cool blue--violet palette shared with the manuscript's system overview.
# All nine retained coverage records are shown.  The classical-ORB and
# no-merge records are numerically identical through the displayed horizon;
# sparse markers keep both records visible without offsetting either trace.
# Occupied-cell precision is restricted to methods with a retained predictive
# completion output.  The all-zero and observed-only fields are not silently
# reinterpreted as prediction-quality measurements.
SERIES = (
    # filename, label, colour, line style, marker, precision available
    ("ours_multi_ours_orb.csv", "MSO multi", "#5B3F9B", "-", None, True),
    ("ours_multi_ours_orb_nomerge.csv", "MSO no merge", "#8A78C2", "--", None, True),
    ("ours_multi_orb.csv", "MSO multi, ORB", "#5B5F96", ":", "o", True),
    ("nearest-multi-our-orb.csv", "Frontier multi", "#586A83", "-.", None, False),
    ("ours_single.csv", "MSO single", "#7966A5", "-", "s", True),
    ("MapEx_single.csv", "MapEx", "#2F6FB2", "-", None, True),
    ("upen_single.csv", "UPEN", "#6C8EBF", "--", None, True),
    ("ig-hector_single.csv", "IG-Hector", "#303B5F", ":", None, False),
    ("nearest.csv", "Frontier single", "#747985", "-.", None, False),
)


def configure_style() -> None:
    """Use editable, journal-scale typography without altering data."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_metric(path: Path, metric: str) -> tuple[list[float], list[float]]:
    """Return all finite retained values through ``MAX_STEP`` in file order."""
    steps: list[float] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", metric}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {path.name}")

        for row_number, row in enumerate(reader, start=2):
            step = float(row["step"])
            value = float(row[metric])
            if not math.isfinite(step) or not math.isfinite(value):
                raise ValueError(f"Non-finite value in {path.name}, row {row_number}")
            if step > MAX_STEP:
                continue
            if steps and step < steps[-1]:
                raise ValueError(f"Non-monotone step order in {path.name}, row {row_number}")
            steps.append(step)
            values.append(value)

    if not steps:
        raise ValueError(f"No retained {metric} values through step {MAX_STEP:g}: {path.name}")
    return steps, values


def save_exports(fig: plt.Figure, prefix: Path) -> None:
    """Write editable vectors plus high-resolution raster companions."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=RASTER_DPI, facecolor="white")
    fig.savefig(
        prefix.with_suffix(".tiff"),
        dpi=RASTER_DPI,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def draw_exploration() -> None:
    """Plot all nine coverage traces and six comparable precision traces."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(width_mm / MM_PER_INCH, EXPLORATION_HEIGHT_MM / MM_PER_INCH),
    )
    metrics = (
        ("coverage", "(c) Coverage"),
        ("predicted_map_quality", "(d) Occupied-cell precision"),
    )

    coverage_handles = []
    for ax, (metric, title) in zip(axes, metrics):
        for filename, label, colour, line_style, marker, precision_available in SERIES:
            steps, values = read_metric(DATA_DIR / filename, metric)
            if metric == "predicted_map_quality" and not precision_available:
                continue
            (line,) = ax.plot(
                steps,
                values,
                color=colour,
                linewidth=1.25 if label == "MSO multi" else 1.0,
                linestyle=line_style,
                marker=marker,
                markevery=150 if marker else None,
                markersize=2.4 if marker else 0,
                markerfacecolor="white" if marker else colour,
                markeredgewidth=0.55,
                solid_capstyle="round",
                label=label,
            )
            if metric == "coverage":
                coverage_handles.append(line)

        ax.set_xlim(0, MAX_STEP)
        ax.set_ylim(0, 1.03)
        ax.set_xticks([0, 600, 1200, 1800])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Exploration step")
        ax.set_ylabel("Fraction")
        ax.set_title(title, loc="left", fontweight="bold", pad=3)
        ax.grid(True, color="#D7DCE2", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(width=0.6, length=2.5, pad=1.5)

    legend_labels = [label for _, label, _, _, _, _ in SERIES]
    fig.legend(
        coverage_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#888888",
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=0.78,
    )
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0), pad=0.45, w_pad=0.85)
    save_exports(fig, EXPLORATION_PREFIX)
    plt.close(fig)


def read_physical_endpoints() -> list[dict[str, str]]:
    """Read only the tabular block preceding the explanatory prose."""
    table_text = PHYSICAL_SOURCE.read_text(encoding="utf-8").split("\n\n", maxsplit=1)[0]
    rows = list(csv.DictReader(io.StringIO(table_text), delimiter="\t"))
    required = {"scene", "method", "coverage", "occupied_cell_precision", "duration_s"}
    if not rows or required.difference(rows[0]):
        raise ValueError(f"Malformed physical endpoint table: {PHYSICAL_SOURCE}")
    return rows


def draw_physical_endpoints() -> None:
    """Plot source-backed endpoints; one run per method and arena."""
    rows = read_physical_endpoints()
    scenes = sorted({int(row["scene"]) for row in rows})
    methods = ("MSO", "Frontier comparator")
    colours = {"MSO": "#6F4DA1", "Frontier comparator": "#2E5C8A"}
    labels = {"MSO": "MSO", "Frontier comparator": "Frontier comparator"}
    indexed = {(int(row["scene"]), row["method"]): row for row in rows}

    positions = list(range(len(scenes)))
    width = 0.34
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(width_mm / MM_PER_INCH, PHYSICAL_HEIGHT_MM / MM_PER_INCH),
    )
    metrics = (
        ("coverage", "(a) Coverage"),
        ("occupied_cell_precision", "(b) Occupied-cell precision"),
    )

    legend_handles = []
    for ax, (metric, title) in zip(axes, metrics):
        for method_index, method in enumerate(methods):
            offset = -width / 2 if method_index == 0 else width / 2
            values = [float(indexed[(scene, method)][metric]) for scene in scenes]
            bars = ax.bar(
                [position + offset for position in positions],
                values,
                width,
                label=labels[method],
                color=colours[method],
                edgecolor="#1F2933",
                linewidth=0.45,
            )
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=5.5)
            if metric == "coverage":
                legend_handles.append(bars)

        tick_labels = []
        for scene in scenes:
            duration = indexed[(scene, "MSO")]["duration_s"]
            tick_labels.append(f"Scene {scene}\nMSO {duration} s")
        ax.set_xticks(positions, tick_labels)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Fraction")
        ax.set_title(title, loc="left", fontweight="bold", pad=3)
        ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(width=0.6, length=2.5, pad=1.5)

    fig.legend(
        legend_handles,
        [labels[method] for method in methods],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#888888",
        borderpad=0.35,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0), pad=0.45, w_pad=0.85)
    save_exports(fig, PHYSICAL_PREFIX)
    plt.close(fig)


def main() -> None:
    configure_style()
    draw_exploration()
    draw_physical_endpoints()
    print(f"Wrote exact aggregate plots under {FIG_DIR}")


if __name__ == "__main__":
    main()
