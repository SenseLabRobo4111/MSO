from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SUBMISSION = HERE.parent
SOURCE = HERE / "source_data"
PEER = SUBMISSION / "peer_review_data"

# Bright blue--violet family sampled from the Fig. 1 visual vocabulary.  The
# darkest categorical tone remains print-safe, while the former charcoal navy
# and muted purple are replaced by brighter fills used in the system overview.
NAVY = "#5659BD"
BLUE = "#5A7DBE"
BLUE_MID = "#7EB6E0"
BLUE_SOFT = "#B1DDF0"
INDIGO = "#6A6CCB"
PURPLE = "#7E57C2"
PURPLE_SOFT = "#B9A6D9"
GREY = "#747B8D"
LIGHT_GREY = "#E8EDF5"
BLACK = "#272727"


def text_colour_on(fill: str) -> str:
    """Choose readable annotation text for the declared solid fill."""

    red, green, blue = mpl.colors.to_rgb(fill)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.54 else BLACK

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(SUBMISSION / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(SUBMISSION / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(SUBMISSION / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        SUBMISSION / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def save_figure_fixed_canvas(fig: plt.Figure, stem: str) -> None:
    """Save the declared 183-mm canvas without tight-bbox width expansion."""
    fig.savefig(SUBMISSION / f"{stem}.svg", facecolor="white")
    fig.savefig(SUBMISSION / f"{stem}.pdf", facecolor="white")
    fig.savefig(SUBMISSION / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(
        SUBMISSION / f"{stem}.tiff",
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _annotated_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    annotations: np.ndarray,
    rows: list[str],
    columns: list[str],
    cmap: str,
    norm=None,
) -> None:
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(columns)), columns)
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.tick_params(length=0)
    ax.set_frame_on(False)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            rgba = image.cmap(image.norm(values[row, col]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                col,
                row,
                annotations[row, col],
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if luminance < 0.47 else BLACK,
            )


def _figure_resource_profile_legacy() -> None:
    data = pd.read_csv(SOURCE / "model_profile.csv")
    primary = data[data["group"] == "baseline"].copy()
    ablation = data[data["group"] == "ablation"].copy()
    runtime = pd.read_csv(SOURCE / "runtime_profile.csv")
    if (primary[["parameters", "operations_million"]] <= 0).any().any():
        raise ValueError("Log-scale resource quantities must be strictly positive")

    fig = plt.figure(figsize=(7.2, 7.0))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.92, 1.12, 0.90],
        hspace=0.56,
        wspace=0.38,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    d_grid = grid[1, 1].subgridspec(1, 3, wspace=0.28)
    ax_d = [fig.add_subplot(d_grid[0, index]) for index in range(3)]
    ax_e = fig.add_subplot(grid[2, 0])
    ax_f = fig.add_subplot(grid[2, 1])

    method_colors = {
        "U-Net": NAVY,
        "LaMa-Fourier": BLUE,
        "MI-GAN": BLUE_SOFT,
        "MSO": PURPLE,
    }
    for _, row in primary.iterrows():
        method = row["method"]
        ax_a.scatter(
            row["parameters"] / 1e6,
            row["operations_million"],
            s=55 if method == "MSO" else 35,
            color=method_colors[method],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    offsets = {
        "U-Net": (5, -8),
        "LaMa-Fourier": (5, 5),
        "MI-GAN": (5, 5),
        "MSO": (5, 5),
    }
    for _, row in primary.iterrows():
        method = row["method"]
        ax_a.annotate(
            method,
            (row["parameters"] / 1e6, row["operations_million"]),
            xytext=offsets[method],
            textcoords="offset points",
            fontsize=6.2,
            fontweight="bold" if method == "MSO" else "normal",
        )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Parameters (millions, log scale)")
    ax_a.set_ylabel("Profiler operation entries (millions, log scale)")
    # Log tick exponents render at roughly 70% of the parent size; 7.2 pt keeps
    # every glyph above the 5 pt final-size floor.
    # Log-axis exponents render at roughly 70% of the parent tick size; keep
    # those superscripts above 5 pt after the manuscript scales 183 mm to its
    # 160 mm review-column width.
    ax_a.tick_params(axis="both", labelsize=8.6)
    ax_a.set_title("Computational operating points", loc="left", fontweight="bold")
    panel_label(ax_a, "a")

    mso_profile = primary[primary["method"] == "MSO"].iloc[0]
    comparators = primary[primary["method"] != "MSO"].copy()
    parameter_ratio = comparators["parameters"].to_numpy(float) / float(mso_profile["parameters"])
    operation_ratio = (
        comparators["operations_million"].to_numpy(float)
        / float(mso_profile["operations_million"])
    )
    y_ratio = np.arange(len(comparators))[::-1]
    for index, y_pos in enumerate(y_ratio):
        ax_b.plot(
            [parameter_ratio[index], operation_ratio[index]],
            [y_pos, y_pos],
            color=LIGHT_GREY,
            lw=1.2,
            zorder=1,
        )
    ax_b.scatter(
        parameter_ratio,
        y_ratio,
        s=24,
        color=BLUE,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
        label="Parameters",
    )
    ax_b.scatter(
        operation_ratio,
        y_ratio,
        s=24,
        color=PURPLE,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
        label="Profiler entries",
    )
    for index, y_pos in enumerate(y_ratio):
        ax_b.annotate(
            f"{parameter_ratio[index]:.1f}×",
            (parameter_ratio[index], y_pos),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=BLUE,
        )
        ax_b.annotate(
            f"{operation_ratio[index]:.1f}×",
            (operation_ratio[index], y_pos),
            xytext=(0, -7),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=6.2,
            color=PURPLE,
        )
    ax_b.set_xscale("log")
    ax_b.set_xlim(10, 245)
    ax_b.set_yticks(y_ratio, comparators["method"].tolist())
    ax_b.tick_params(axis="x", labelsize=8.6)
    ax_b.set_xlabel("Comparator-to-MSO ratio (log scale)")
    ax_b.grid(axis="x", color="#E6E8ED", lw=0.55)
    ax_b.legend(loc="lower right", ncol=2, columnspacing=0.9, handletextpad=0.4)
    ax_b.set_title("Direct resource ratios", loc="left", fontweight="bold")
    panel_label(ax_b, "b")

    metrics = ["PSNR", "SSIM", "LPIPS", "FID", "KID"]
    raw = primary[metrics].to_numpy(float)
    annotations = np.empty(raw.shape, dtype=object)
    formats = ["{:.2f}", "{:.3f}", "{:.3f}", "{:.1f}", "{:.3f}"]
    for col, fmt in enumerate(formats):
        annotations[:, col] = [fmt.format(value) for value in raw[:, col]]
    ax_c.axis("off")
    table = ax_c.table(
        cellText=annotations,
        rowLabels=primary["method"].tolist(),
        colLabels=["PSNR ↑", "SSIM ↑", "LPIPS ↓", "FID ↓", "KID ↓"],
        cellLoc="center",
        rowLoc="right",
        bbox=[0.0, 0.02, 1.0, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C7C7C7")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#E8EEF6")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        else:
            cell.set_facecolor("white")
        if row == 4:
            cell.get_text().set_fontweight("bold")
    ax_c.set_title("Whole-crop point estimates", loc="left", fontweight="bold")
    panel_label(ax_c, "c")

    full = ablation[ablation["method"] == "Full"].iloc[0]
    ablation_labels = {
        "Full": "Full",
        "w/o adversarial": "– adversarial",
        "w/o distillation": "– distillation",
        "w/o adversarial + distillation": "– both",
        "w/o FFC": "– FFC",
    }
    ablation_metrics = [("LPIPS", "LPIPS ↓"), ("FID", "FID ↓"), ("KID", "KID ↓")]
    y_ablation = np.arange(len(ablation))[::-1]
    for metric_index, ((metric, title), axis) in enumerate(zip(ablation_metrics, ax_d)):
        axis.scatter(
            ablation[metric],
            y_ablation,
            s=20,
            color=[PURPLE if method == "Full" else BLUE_MID for method in ablation["method"]],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axis.axvline(float(full[metric]), color=GREY, ls="--", lw=0.8)
        axis.set_xlabel(title)
        axis.set_ylim(-0.55, len(ablation) - 0.45)
        if metric_index == 0:
            axis.set_yticks(y_ablation, [ablation_labels[name] for name in ablation["method"]])
        else:
            axis.set_yticks(y_ablation, [])
        axis.grid(axis="x", color="#E6E6E6", lw=0.55)
        axis.tick_params(axis="both", labelsize=6.2)
    ax_d[0].set_title("Ablation point estimates", loc="left", fontweight="bold")
    panel_label(ax_d[0], "d", x=-0.58)

    recurring = runtime[runtime["pathway"] == "recurring"].copy()
    recurring_y = np.arange(len(recurring))[::-1]
    for index, (_, row) in enumerate(recurring.iterrows()):
        pos = recurring_y[index]
        latency = float(row["latency_high_ms"])
        ax_e.hlines(pos, 0, latency, color=BLUE_SOFT, linewidth=3.6, zorder=1)
        ax_e.plot(latency, pos, "o", color=BLUE, markersize=4.4, zorder=3)
        ax_e.text(
            latency + 0.14,
            pos,
            f"{latency:g} ms",
            va="center",
            ha="left",
            fontsize=6.2,
            fontweight="bold",
            color=NAVY,
        )
        schedule = str(row["schedule"]).replace("approximately ", "≈")
        ax_e.text(4.75, pos, schedule, va="center", ha="right", fontsize=6.2, color=GREY)
    ax_e.set_yticks(recurring_y, recurring["stage"].tolist())
    ax_e.set_xlim(0, 5.0)
    ax_e.set_ylim(-0.55, len(recurring) - 0.45)
    ax_e.set_xlabel("Archived stage latency (ms)")
    ax_e.set_title("Recurring onboard path", loc="left", fontweight="bold")
    panel_label(ax_e, "e")

    encounter = runtime[runtime["pathway"] == "encounter"].copy()
    encounter_y = np.arange(len(encounter))[::-1]
    for index, (_, row) in enumerate(encounter.iterrows()):
        pos = encounter_y[index]
        low = float(row["latency_low_ms"])
        high = float(row["latency_high_ms"])
        is_total = row["stage"] == "Fusion event total"
        color = PURPLE if is_total else INDIGO
        ax_f.hlines(pos, low, high, color=color, linewidth=4.2, alpha=0.72)
        ax_f.plot((low + high) / 2.0, pos, "o", color=color, markersize=4.2)
        ax_f.text(
            high + 1.1,
            pos,
            f"{low:g}–{high:g} ms",
            va="center",
            fontsize=6.2,
            fontweight="bold" if is_total else "normal",
        )
    ax_f.axhline(0.5, color=LIGHT_GREY, lw=0.7)
    ax_f.set_yticks(encounter_y, encounter["stage"].tolist())
    ax_f.set_xlim(0, 67)
    ax_f.set_ylim(-0.55, len(encounter) - 0.45)
    ax_f.set_xlabel("Archived latency range (ms)")
    ax_f.set_title("Event-driven peer encounter", loc="left", fontweight="bold")
    panel_label(ax_f, "f")

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.075, top=0.965)
    save_figure(fig, "figure_resource_profile")


def figure_resource_profile() -> None:
    """Render a dense, evidence-bounded model and runtime profile."""
    data = pd.read_csv(SOURCE / "model_profile.csv")
    primary = data[data["group"] == "baseline"].copy()
    ablation = data[data["group"] == "ablation"].copy()
    runtime = pd.read_csv(SOURCE / "runtime_profile.csv")
    if (primary[["parameters", "operations_million"]] <= 0).any().any():
        raise ValueError("Log-scale resource quantities must be strictly positive")

    fig = plt.figure(figsize=(7.2, 8.10))
    grid = fig.add_gridspec(
        4,
        12,
        height_ratios=[0.95, 1.05, 0.78, 0.90],
        hspace=0.68,
        wspace=0.78,
    )
    top_grid = grid[0, 0:12].subgridspec(1, 2, wspace=0.42)
    ax_a = fig.add_subplot(top_grid[0, 0])
    ax_b = fig.add_subplot(top_grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0:4])
    ax_d = fig.add_subplot(grid[1, 5:12])
    ax_e = fig.add_subplot(grid[2, 0:12])
    ax_f = fig.add_subplot(grid[3, 0:12])

    method_colors = {
        "U-Net": NAVY,
        "LaMa-Fourier": BLUE,
        "MI-GAN": BLUE_SOFT,
        "MSO": PURPLE,
    }

    # a, operations versus FID.  Lower-left is favourable for both retained
    # fields; no combined efficiency score is constructed.
    for _, row in primary.iterrows():
        method = row["method"]
        ax_a.scatter(
            row["operations_million"],
            row["FID"],
            s=58 if method == "MSO" else 38,
            color=method_colors[method],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    offsets_a = {
        "U-Net": (-7, 6),
        "LaMa-Fourier": (-8, 5),
        "MI-GAN": (-7, -12),
        "MSO": (7, -12),
    }
    for _, row in primary.iterrows():
        method = row["method"]
        ax_a.annotate(
            method,
            (row["operations_million"], row["FID"]),
            xytext=offsets_a[method],
            textcoords="offset points",
            ha="right" if method in {"U-Net", "MI-GAN", "LaMa-Fourier"} else "left",
            fontsize=6.3,
            fontweight="bold" if method == "MSO" else "normal",
        )
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Profiler operation entries (millions, log scale)")
    ax_a.set_ylabel("FID ↓")
    ax_a.tick_params(axis="x", labelsize=8.6)
    ax_a.grid(color="#E8EAF0", lw=0.55)
    ax_a.set_title("Quality versus recorded computation", loc="left", fontweight="bold")
    panel_label(ax_a, "a")

    # b, parameter count versus LPIPS.
    for _, row in primary.iterrows():
        method = row["method"]
        ax_b.scatter(
            row["parameters"] / 1e6,
            row["LPIPS"],
            s=58 if method == "MSO" else 38,
            color=method_colors[method],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    offsets_b = {
        "U-Net": (-7, 6),
        "LaMa-Fourier": (-7, 6),
        "MI-GAN": (7, 5),
        "MSO": (7, -11),
    }
    for _, row in primary.iterrows():
        method = row["method"]
        ax_b.annotate(
            method,
            (row["parameters"] / 1e6, row["LPIPS"]),
            xytext=offsets_b[method],
            textcoords="offset points",
            ha="right" if method in {"U-Net", "LaMa-Fourier"} else "left",
            fontsize=6.3,
            fontweight="bold" if method == "MSO" else "normal",
        )
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Parameters (millions, log scale)")
    ax_b.set_ylabel("LPIPS ↓")
    ax_b.tick_params(axis="x", labelsize=8.6)
    ax_b.grid(color="#E8EAF0", lw=0.55)
    ax_b.set_title("Quality versus model size", loc="left", fontweight="bold")
    panel_label(ax_b, "b")

    # c, exact comparator-to-MSO resource ratios.
    mso_profile = primary[primary["method"] == "MSO"].iloc[0]
    comparators = primary[primary["method"] != "MSO"].copy()
    parameter_ratio = comparators["parameters"].to_numpy(float) / float(
        mso_profile["parameters"]
    )
    operation_ratio = comparators["operations_million"].to_numpy(float) / float(
        mso_profile["operations_million"]
    )
    ratio_values = np.column_stack([parameter_ratio, operation_ratio])
    ratio_annotations = np.asarray(
        [[f"{value:.1f}×" for value in row] for row in ratio_values],
        dtype=object,
    )
    ratio_log = np.log10(ratio_values)
    ratio_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "ratio_cool", ["#F1F4F9", BLUE_SOFT, PURPLE]
    )
    _annotated_heatmap(
        ax_c,
        ratio_log,
        ratio_annotations,
        comparators["method"].tolist(),
        ["Parameters", "Profiler entries"],
        ratio_cmap,
    )
    ax_c.tick_params(axis="x", labelsize=6.1, pad=3)
    ax_c.tick_params(axis="y", labelsize=6.1, pad=2)
    ax_c.set_title("Resource ratios to MSO", loc="left", fontweight="bold", pad=4)
    panel_label(ax_c, "c", x=-0.22)

    # d, all five retained baseline metrics.  Colour is a within-column,
    # direction-aware guide; annotations remain the untransformed point data.
    metrics = ["PSNR", "SSIM", "LPIPS", "FID", "KID"]
    raw = primary[metrics].to_numpy(float)
    directions = np.asarray([1.0, 1.0, -1.0, -1.0, -1.0])
    score = np.zeros_like(raw)
    for col in range(raw.shape[1]):
        values = raw[:, col]
        span = float(values.max() - values.min())
        if span <= 0:
            score[:, col] = 0.5
        elif directions[col] > 0:
            score[:, col] = (values - values.min()) / span
        else:
            score[:, col] = (values.max() - values) / span
    annotations = np.empty(raw.shape, dtype=object)
    formats = ["{:.2f}", "{:.3f}", "{:.3f}", "{:.1f}", "{:.3f}"]
    for col, fmt in enumerate(formats):
        annotations[:, col] = [fmt.format(value) for value in raw[:, col]]
    baseline_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "baseline_cool", ["#F4F6FA", BLUE_SOFT, PURPLE]
    )
    _annotated_heatmap(
        ax_d,
        score,
        annotations,
        primary["method"].tolist(),
        ["PSNR ↑", "SSIM ↑", "LPIPS ↓", "FID ↓", "KID ↓"],
        baseline_cmap,
    )
    ax_d.tick_params(axis="y", labelsize=6.1, pad=2)
    ax_d.set_title("Whole-crop point estimates", loc="left", fontweight="bold", pad=4)
    panel_label(ax_d, "d", x=-0.09)

    # e, all seven retained ablation fields.  Positive signed values indicate
    # a favourable direction relative to Full; negative values are adverse.
    full = ablation[ablation["method"] == "Full"].iloc[0]
    ablation_fields = [
        "parameters",
        "operations_million",
        "PSNR",
        "SSIM",
        "LPIPS",
        "FID",
        "KID",
    ]
    ablation_directions = np.asarray([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    ablation_raw = ablation[ablation_fields].to_numpy(float)
    full_raw = full[ablation_fields].to_numpy(float)
    ablation_delta = 100.0 * (ablation_raw - full_raw) / np.abs(full_raw)
    ablation_delta *= ablation_directions
    delta_annotations = np.empty(ablation_delta.shape, dtype=object)
    for row in range(ablation_delta.shape[0]):
        for col in range(ablation_delta.shape[1]):
            value = ablation_delta[row, col]
            delta_annotations[row, col] = "0" if abs(value) < 0.005 else f"{value:+.1f}%"
    max_abs = max(1.0, float(np.max(np.abs(ablation_delta))))
    delta_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "delta_cool", [BLUE, "#F5F6F9", PURPLE]
    )
    _annotated_heatmap(
        ax_e,
        ablation_delta,
        delta_annotations,
        ["Full", "− adversarial", "− distillation", "− both", "− FFC"],
        ["Params", "Entries", "PSNR", "SSIM", "LPIPS", "FID", "KID"],
        delta_cmap,
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
    )
    ax_e.set_title(
        "Direction-adjusted change from Full (single-run point estimates)",
        loc="left",
        fontweight="bold",
        pad=4,
    )
    panel_label(ax_e, "e", x=-0.06)

    # f, one shared logarithmic axis for recurring and encounter stages.
    runtime_y = np.arange(len(runtime))[::-1]
    hardware_markers = {"Cortex-A78AE": "o", "Ampere GPU": "s", "mixed": "D"}
    for index, (_, row) in enumerate(runtime.iterrows()):
        pos = runtime_y[index]
        low = float(row["latency_low_ms"])
        high = float(row["latency_high_ms"])
        recurring = row["pathway"] == "recurring"
        total = row["stage"] == "Fusion event total"
        colour = BLUE if recurring else (PURPLE if total else INDIGO)
        if high > low:
            ax_f.hlines(pos, low, high, color=colour, linewidth=4.0, alpha=0.72)
        midpoint = (low + high) / 2.0
        ax_f.plot(
            midpoint,
            pos,
            marker=hardware_markers[str(row["hardware"])],
            color=colour,
            markersize=4.1,
            linestyle="None",
        )
        label = f"{low:g} ms" if low == high else f"{low:g}–{high:g} ms"
        ax_f.text(
            high * 1.08,
            pos,
            label,
            va="center",
            fontsize=6.2,
            fontweight="bold" if total else "normal",
        )
    runtime_labels = [
        "LiDAR raster",
        "Prediction",
        "ORB",
        "RANSAC",
        "Blending",
        "Fusion total",
    ]
    ax_f.axhline(3.5, color=LIGHT_GREY, lw=0.7)
    ax_f.set_xscale("log")
    ax_f.set_xlim(1.4, 85)
    ax_f.set_yticks(runtime_y, runtime_labels)
    ax_f.set_xlabel("Archived latency (ms, log scale)")
    ax_f.tick_params(axis="x", labelsize=8.6)
    ax_f.grid(axis="x", color="#E6E8ED", lw=0.55)
    ax_f.set_title("Recurring and encounter paths", loc="left", fontweight="bold")
    panel_label(ax_f, "f", x=-0.06)

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.065, top=0.965)
    save_figure_fixed_canvas(fig, "figure_resource_profile")


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def figure_transform_audit() -> None:
    evaluated = pd.read_csv(
        PEER / "transform_validation" / "results_manuscript_reference" / "evaluated_events.csv"
    )
    decisions = pd.read_csv(
        PEER / "transform_validation" / "table3_decision_summary.csv"
    )
    injected = pd.read_csv(
        PEER / "transform_validation" / "wrong_transform_injection_evaluator" / "evaluated_events.csv"
    )

    labels = {
        "observed": "Observed",
        "predicted": "Thresholded prediction",
        "predicted_raw": "Probability preserving",
    }
    colors = {"observed": NAVY, "predicted": BLUE_MID, "predicted_raw": PURPLE}

    fig = plt.figure(figsize=(7.2, 7.35))
    grid = fig.add_gridspec(3, 2, hspace=0.48, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_e = fig.add_subplot(grid[2, 0])
    ax_f = fig.add_subplot(grid[2, 1])

    positive = evaluated[(evaluated["gt_positive"] == True) & (evaluated["rigid_valid"] == True)].copy()  # noqa: E712
    for input_type in ["observed", "predicted", "predicted_raw"]:
        subset = positive[positive["input_type"] == input_type]
        errors = subset["translation_error_m"].to_numpy(float)
        if not np.isfinite(errors).all():
            raise ValueError(f"Missing translation error for {input_type}")
        x, y = _ecdf(errors)
        ax_a.step(x, y, where="post", color=colors[input_type], lw=1.7, label=labels[input_type])
    ax_a.axvline(0.25, color=GREY, ls="--", lw=1.0)
    ax_a.text(
        0.27,
        0.08,
        "Translation analysis limit: 0.25 m",
        va="bottom",
        ha="left",
        color=GREY,
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.92},
    )
    ax_a.set_xlabel("Translation error (m)")
    ax_a.set_ylabel("Empirical cumulative fraction")
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(0, 1.02)
    ax_a.set_title("Returned positive candidates", loc="left", fontweight="bold")
    panel_label(ax_a, "a")

    for input_type in ["observed", "predicted", "predicted_raw"]:
        subset = positive[positive["input_type"] == input_type]
        errors = subset["yaw_error_deg"].to_numpy(float)
        if not np.isfinite(errors).all():
            raise ValueError(f"Missing yaw error for {input_type}")
        x, y = _ecdf(errors)
        ax_b.step(x, y, where="post", color=colors[input_type], lw=1.7, label=labels[input_type])
    ax_b.text(
        0.97,
        0.84,
        "Yaw analysis limit: 5°\n(outside plotted range)",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        color=GREY,
        fontsize=6.2,
        linespacing=1.15,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.94},
    )
    ax_b.set_xlabel("Absolute yaw error (degrees)")
    ax_b.set_ylabel("Empirical cumulative fraction")
    ax_b.set_xlim(0, 0.65)
    ax_b.set_ylim(0, 1.02)
    ax_b.legend(loc="lower right")
    ax_b.set_title("Returned positive candidates", loc="left", fontweight="bold")
    panel_label(ax_b, "b")

    decision_labels = ["Observed", "Thresholded", "Probability\npreserving"]
    y_base = np.arange(3)[::-1] * 2.4
    for idx, row in decisions.iterrows():
        y_pos = y_base[idx] + 0.38
        correct = row["positive_correct_accept_n"]
        out_limit = row["positive_out_of_limit_accept_n"]
        rejected = row["positive_reject_n"]
        ax_c.barh(y_pos, correct, height=0.62, color=BLUE, edgecolor="none")
        ax_c.barh(y_pos, out_limit, left=correct, height=0.62, color=PURPLE, edgecolor="none")
        ax_c.barh(y_pos, rejected, left=correct + out_limit, height=0.62, color=LIGHT_GREY, edgecolor="none")
        low_y = y_base[idx] - 0.38
        accepted = row["low_support_accept_n"]
        low_reject = row["low_support_reject_n"]
        ax_c.barh(low_y, accepted, height=0.62, color=PURPLE_SOFT, edgecolor="none")
        ax_c.barh(low_y, low_reject, left=accepted, height=0.62, color="#ECECEC", edgecolor="none")
        ax_c.text(51.5, y_pos, "Positive", va="center", fontsize=6.2, color=GREY)
        ax_c.text(51.5, low_y, "Low support", va="center", fontsize=6.2, color=GREY)
        if accepted > 0:
            ax_c.text(
                accepted / 2,
                low_y,
                str(int(accepted)),
                ha="center",
                va="center",
                fontsize=6.2,
                color=text_colour_on(PURPLE_SOFT),
            )
    ax_c.set_yticks(y_base, decision_labels)
    ax_c.set_xlim(0, 66)
    ax_c.set_ylim(-0.9, 7.25)
    ax_c.set_xlabel("Events per 50-case stratum")
    ax_c.set_title("Offline pre-commit decisions", loc="left", fontweight="bold")
    ax_c.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=6, label="Correct positive accept"),
            Line2D([0], [0], color=PURPLE, lw=6, label="Out-of-limit positive accept"),
            Line2D([0], [0], color=LIGHT_GREY, lw=6, label="Reject"),
            Line2D([0], [0], color=PURPLE_SOFT, lw=6, label="Low-support accept"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
        borderpad=0.35,
        handlelength=2.1,
        columnspacing=0.9,
        labelspacing=0.4,
    )
    panel_label(ax_c, "c")

    scores_raw = injected["gate_score"].to_numpy(float)
    thresholds_raw = injected["gate_threshold"].to_numpy(float)
    if not np.isfinite(scores_raw).all() or not np.isfinite(thresholds_raw).all():
        raise ValueError("Injection audit contains a missing gate value")
    threshold_values = np.unique(thresholds_raw)
    if len(threshold_values) != 1:
        raise ValueError("Injection audit must use one analysis gate threshold")
    threshold = float(threshold_values[0])
    input_order = ["observed", "predicted", "predicted_raw"]
    for category_index, input_type in enumerate(input_order):
        subset = injected[injected["input_type"] == input_type]
        scores = subset["gate_score"].to_numpy(float)
        jitter = np.linspace(-0.12, 0.12, len(scores), dtype=float)
        ax_d.scatter(
            np.full(len(scores), category_index) + jitter,
            scores,
            s=9,
            alpha=0.72,
            color=colors[input_type],
            edgecolor="none",
        )
    ax_d.axhline(threshold, color=BLACK, ls="--", lw=1.0, label=f"Gate threshold {threshold:.2f}")
    ax_d.text(
        0.98,
        0.74,
        "150/150 rejected before mutation\n0 wrong commits",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        fontweight="bold",
        color=BLUE,
    )
    ax_d.set_xticks(range(len(input_order)), ["Observed", "Thresholded", "Probability\npreserving"])
    ax_d.set_xlabel("Input representation (50 proposals each)")
    ax_d.set_ylabel("Observed-support gate score")
    ax_d.set_ylim(0, max(0.225, threshold * 1.08))
    ax_d.legend(loc="upper left")
    ax_d.set_title("Fixed 1.0 m and 15° injection", loc="left", fontweight="bold")
    panel_label(ax_d, "d")

    # Show score separation under the same fixed gate.  The low-support
    # stratum remains an abstention challenge and is not relabelled as a
    # negative pose-error example.  Deterministic jitter reveals density.
    score_positions: list[float] = []
    score_values: list[np.ndarray] = []
    score_colours: list[str] = []
    for input_index, input_type in enumerate(input_order):
        subset = evaluated[evaluated["input_type"] == input_type]
        for stratum_index, (is_positive, stratum_label) in enumerate(
            ((True, "positive"), (False, "low support"))
        ):
            values = subset[subset["gt_positive"] == is_positive][
                "gate_score"
            ].to_numpy(float)
            if len(values) != 50 or not np.isfinite(values).all():
                raise ValueError(
                    f"Unexpected gate-score stratum for {input_type}/{stratum_label}"
                )
            position = input_index * 2.5 + stratum_index * 0.78
            score_positions.append(position)
            score_values.append(values)
            score_colours.append(colors[input_type])
            jitter = np.linspace(-0.13, 0.13, len(values), dtype=float)
            ax_e.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=5.5,
                alpha=0.34,
                color=colors[input_type],
                edgecolor="none",
                zorder=1,
            )
    boxes = ax_e.boxplot(
        score_values,
        positions=score_positions,
        widths=0.42,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": BLACK, "linewidth": 1.1},
        whiskerprops={"color": GREY, "linewidth": 0.8},
        capprops={"color": GREY, "linewidth": 0.8},
        boxprops={"linewidth": 0.8, "edgecolor": GREY},
    )
    for box, colour in zip(boxes["boxes"], score_colours):
        box.set_facecolor(colour)
        box.set_alpha(0.36)
    ax_e.axhline(0.20, color=BLACK, ls="--", lw=1.0)
    ax_e.text(
        0.98,
        0.22,
        "Gate threshold 0.20",
        transform=ax_e.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=GREY,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
    )
    ax_e.set_xticks(
        [0.39, 2.89, 5.39],
        ["Observed", "Thresholded", "Probability\npreserving"],
    )
    ax_e.set_ylabel("Observed-support gate score")
    ax_e.set_ylim(-0.03, 1.0)
    ax_e.set_title("Gate-score separation", loc="left", fontweight="bold")
    ax_e.text(
        0.02,
        0.98,
        "left: positive   right: low support",
        transform=ax_e.transAxes,
        va="top",
        ha="left",
        fontsize=6.2,
        color=GREY,
    )
    panel_label(ax_e, "e")

    stage_order = ["3px_30pct", "5px_20pct", "10px_3pct"]
    stage_labels = ["3 px / 30%", "5 px / 20%", "10 px / 3%"]
    stage_colours = [NAVY, BLUE_MID, PURPLE_SOFT]
    stage_left = np.zeros(len(input_order), dtype=float)
    y_positions = np.arange(len(input_order))[::-1]
    for stage, stage_label, stage_colour in zip(
        stage_order, stage_labels, stage_colours
    ):
        counts = np.asarray(
            [
                int(
                    (
                        (evaluated["input_type"] == input_type)
                        & (evaluated["ransac_stage"] == stage)
                    ).sum()
                )
                for input_type in input_order
            ],
            dtype=float,
        )
        ax_f.barh(
            y_positions,
            counts,
            left=stage_left,
            height=0.54,
            color=stage_colour,
            edgecolor="white",
            linewidth=0.45,
            label=stage_label,
        )
        for y_value, left_value, count in zip(y_positions, stage_left, counts):
            if count >= 8:
                ax_f.text(
                    left_value + count / 2,
                    y_value,
                    f"{int(count)}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=text_colour_on(stage_colour),
                )
        stage_left += counts
    if not np.array_equal(stage_left, np.full(len(input_order), 100.0)):
        raise ValueError("RANSAC-stage counts do not sum to 100 per representation")
    ax_f.set_yticks(
        y_positions,
        ["Observed", "Thresholded", "Probability\npreserving"],
    )
    ax_f.set_xlim(0, 100)
    # Reserve a legend band inside the panel so that the shared RANSAC-stage
    # key does not sit in the inter-panel/caption margin or cover any bar.
    ax_f.set_ylim(-1.25, 2.55)
    ax_f.set_xlabel("Events (n = 100 per representation)")
    ax_f.set_title("RANSAC stage reached", loc="left", fontweight="bold")
    ax_f.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.5,
    )
    panel_label(ax_f, "f")

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.10, top=0.965)
    save_figure(fig, "figure_transform_audit")


def _read_curve(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"step", "coverage"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required columns in {path.name}")
    if not frame["step"].is_monotonic_increasing:
        raise ValueError(f"Non-monotonic step axis in {path.name}")
    return frame


def figure_scaling_and_scene() -> None:
    scaling_dir = PEER / "scaling_kth"
    team_paths = {
        2: scaling_dir / "A2_N2_coverage.csv",
        3: scaling_dir / "A2_N3_coverage.csv",
        5: scaling_dir / "A2_N5_coverage.csv",
    }
    team_frames = {team: _read_curve(path) for team, path in team_paths.items()}
    kth_paths = {
        "Nearest frontier": scaling_dir / "A3_nearest-multi-our-orb_coverage.csv",
        "MSO": scaling_dir / "A3_ours_multi_ours_orb_coverage.csv",
        "MSO without merge": scaling_dir / "A3_ours_multi_ours_orb_nomerge_coverage.csv",
    }
    kth_frames = {label: _read_curve(path) for label, path in kth_paths.items()}
    physical = pd.read_csv(
        PEER / "tables_and_figures" / "physical_endpoint_values.txt",
        sep="\t",
        nrows=6,
    )

    fig = plt.figure(figsize=(7.2, 6.75))
    grid = fig.add_gridspec(3, 2, hspace=0.56, wspace=0.36)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_e = fig.add_subplot(grid[2, 0])
    ax_f = fig.add_subplot(grid[2, 1])

    team_colors = {2: BLUE, 3: BLUE_MID, 5: PURPLE}
    team_styles = {2: (0, (3.0, 1.2)), 3: "--", 5: "-"}
    team_thresholds = (0.5, 0.8)
    crossings = {team: {} for team in team_frames}
    endpoints = {}
    horizons = {}
    for team, frame in team_frames.items():
        ax_a.plot(
            frame["step"],
            frame["coverage"],
            color=team_colors[team],
            ls=team_styles[team],
            lw=1.7,
            label=f"N = {team}",
        )
        for threshold in team_thresholds:
            reached = frame[frame["coverage"] >= threshold]
            if reached.empty:
                raise ValueError(f"Team {team} never reaches coverage {threshold}")
            crossings[team][threshold] = int(reached.iloc[0]["step"])
        endpoints[team] = float(frame.iloc[-1]["coverage"])
        horizons[team] = int(frame.iloc[-1]["step"])
        ax_a.scatter(frame.iloc[-1]["step"], frame.iloc[-1]["coverage"], color=team_colors[team], s=18, zorder=3)
    ax_a.axhline(0.8, color=GREY, ls="--", lw=0.9)
    ax_a.text(
        0.97,
        0.84,
        "0.8 coverage",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=GREY,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.9, "alpha": 0.92},
    )
    ax_a.set_xlim(0, 400)
    ax_a.set_ylim(0, 1.0)
    ax_a.set_xlabel("Logged exploration step")
    ax_a.set_ylabel("Coverage")
    ax_a.legend(loc="lower right", ncol=3)
    ax_a.set_title("Team-size traces on one MRPB layout", loc="left", fontweight="bold")
    panel_label(ax_a, "a")

    teams = np.array(sorted(crossings))
    threshold_markers = {0.5: "o", 0.8: "D"}
    for threshold in team_thresholds:
        for team in teams:
            value = crossings[int(team)][threshold]
            ax_b.scatter(
                team,
                value,
                s=28,
                marker=threshold_markers[threshold],
                color=team_colors[int(team)],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
            ax_b.annotate(
                f"{value}",
                (team, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.2,
                fontweight="bold" if threshold == 0.8 else "normal",
            )
    ax_b.set_xticks(teams)
    ax_b.set_xlim(1.6, 5.4)
    ax_b.set_ylim(0, 96)
    ax_b.set_xlabel("Number of robots")
    ax_b.set_ylabel("First logged step")
    ax_b.grid(axis="y", color=LIGHT_GREY, linewidth=0.6, zorder=0)
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY, label="Coverage 0.5"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=GREY, label="Coverage 0.8"),
        ],
        loc="upper right",
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    ax_b.set_title("Recorded coverage milestones", loc="left", fontweight="bold")
    panel_label(ax_b, "b")

    support_offsets = {2: (-7, 6), 3: (6, 6), 5: (6, -2)}
    support_align = {2: "right", 3: "left", 5: "left"}
    for team in teams:
        team = int(team)
        ax_c.scatter(
            horizons[team],
            endpoints[team],
            s=36,
            color=team_colors[team],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax_c.annotate(
            f"N = {team}  |  {endpoints[team]:.3f}",
            (horizons[team], endpoints[team]),
            xytext=support_offsets[team],
            textcoords="offset points",
            ha=support_align[team],
            va="bottom",
            fontsize=6.2,
            color=team_colors[team],
            fontweight="bold" if team == 5 else "normal",
        )
    ax_c.axhline(0.8, color=LIGHT_GREY, ls="--", lw=0.8, zorder=0)
    ax_c.set_xlim(60, 415)
    ax_c.set_ylim(0.80, 0.975)
    ax_c.set_xlabel("Last retained logged step")
    ax_c.set_ylabel("Endpoint coverage")
    ax_c.set_title("Unequal retained trace horizons", loc="left", fontweight="bold")
    ax_c.text(
        0.98,
        0.08,
        "One aggregate trace per team",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=GREY,
    )
    panel_label(ax_c, "c")

    kth_colors = {"Nearest frontier": GREY, "MSO": PURPLE, "MSO without merge": BLUE_MID}
    kth_styles = {"Nearest frontier": (0, (2.2, 1.1)), "MSO": "-", "MSO without merge": "--"}
    for label, frame in kth_frames.items():
        ax_d.plot(
            frame["step"],
            frame["coverage"],
            color=kth_colors[label],
            ls=kth_styles[label],
            lw=1.7,
            label=label,
        )
        ax_d.scatter(frame.iloc[-1]["step"], frame.iloc[-1]["coverage"], color=kth_colors[label], s=16, zorder=3)
    ax_d.set_xlim(left=0)
    ax_d.set_ylim(0, 1.0)
    ax_d.set_xlabel("Logged exploration step")
    ax_d.set_ylabel("Coverage")
    ax_d.legend(loc="lower right")
    ax_d.set_title("One retained KTH floorplan", loc="left", fontweight="bold")
    panel_label(ax_d, "d")

    kth_thresholds = (0.5, 0.8, 0.9)
    kth_markers = {"Nearest frontier": "o", "MSO": "D", "MSO without merge": "^"}
    kth_y_offsets = {"Nearest frontier": -0.12, "MSO": 0.0, "MSO without merge": 0.12}
    for row, threshold in enumerate(kth_thresholds):
        for label, frame in kth_frames.items():
            reached = frame[frame["coverage"] >= threshold]
            if reached.empty:
                raise ValueError(f"{label} never reaches coverage {threshold}")
            value = int(reached.iloc[0]["step"])
            y_value = row + kth_y_offsets[label]
            ax_e.scatter(
                value,
                y_value,
                s=25,
                marker=kth_markers[label],
                color=kth_colors[label],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            ax_e.annotate(
                f"{value}",
                (value, y_value),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6.2,
                color=kth_colors[label],
            )
    ax_e.set_xlim(0, 53)
    ax_e.set_ylim(-0.35, 2.35)
    ax_e.set_yticks(range(len(kth_thresholds)), [f"{threshold:.1f}" for threshold in kth_thresholds])
    ax_e.set_xlabel("First logged step")
    ax_e.set_ylabel("Coverage threshold")
    ax_e.grid(axis="x", color=LIGHT_GREY, linewidth=0.6, zorder=0)
    ax_e.set_title("KTH coverage milestones", loc="left", fontweight="bold")
    panel_label(ax_e, "e")

    scene_colors = {1: NAVY, 2: BLUE_MID, 3: PURPLE}
    for scene, subset in physical.groupby("scene", sort=True):
        frontier = subset[subset["method"] == "Frontier comparator"].iloc[0]
        mso = subset[subset["method"] == "MSO"].iloc[0]
        ax_f.plot(
            [frontier["coverage"], mso["coverage"]],
            [frontier["occupied_cell_precision"], mso["occupied_cell_precision"]],
            color=scene_colors[int(scene)],
            lw=1.1,
            alpha=0.85,
        )
        ax_f.scatter(
            frontier["coverage"],
            frontier["occupied_cell_precision"],
            s=24,
            facecolor="white",
            edgecolor=scene_colors[int(scene)],
            linewidth=1.1,
            zorder=3,
        )
        ax_f.scatter(
            mso["coverage"],
            mso["occupied_cell_precision"],
            s=28,
            color=scene_colors[int(scene)],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax_f.text(
            mso["coverage"] - 0.01,
            mso["occupied_cell_precision"] + 0.012,
            f"Scene {int(scene)}",
            ha="right",
            va="bottom",
            fontsize=6.2,
            color=scene_colors[int(scene)],
        )
    ax_f.set_xlim(0.32, 1.02)
    ax_f.set_ylim(0.25, 0.60)
    ax_f.set_xlabel("Endpoint coverage")
    ax_f.set_ylabel("Occupied-cell precision")
    ax_f.set_title("Physical endpoint trade-off", loc="left", fontweight="bold")
    ax_f.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=GREY, label="Frontier"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY, markeredgecolor="white", label="MSO"),
        ],
        loc="lower left",
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    panel_label(ax_f, "f")

    fig.text(
        0.5,
        0.012,
        "Panels a-e derive from one retained aggregate trace per condition; f shows one run per scene and method.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=GREY,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.095, top=0.965)
    save_figure(fig, "figure_scaling_scene")


def main() -> None:
    figure_resource_profile()
    figure_transform_audit()
    figure_scaling_and_scene()


if __name__ == "__main__":
    main()
