"""Multi-robot scaling of full MSO on the large MRPB scene at team sizes
N in {2, 3, 5}.

Reads the 5-seed-aggregated coverage curves
``../Exp/A2/A2/tianda/N{2,3,5}/centre/coverage.csv`` (columns step,coverage,...).
Produces figs/appendix/scaling_multirobot.png/pdf:
  (a) coverage over exploration step (one mean curve per team size)
  (b) final coverage per team size.
Palette: blue -> purple matching the predicted-map colour family.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
A2_DIR = ROOT.parent / "Exp" / "A2" / "A2" / "tianda"
FIG_DIR = ROOT / "figs" / "appendix"
OUT_PNG = FIG_DIR / "scaling_multirobot.png"
OUT_PDF = FIG_DIR / "scaling_multirobot.pdf"

TEAMS = {
    2: {"label": "2 robots", "color": "#A6B8D8", "lw": 2.2, "zorder": 5},
    3: {"label": "3 robots", "color": "#4C72B0", "lw": 2.5, "zorder": 7},
    5: {"label": "5 robots", "color": "#734AAD", "lw": 2.8, "zorder": 9},
}


def load_cov(n: int) -> np.ndarray:
    f = A2_DIR / f"N{n}" / "centre" / "coverage.csv"
    rows = list(csv.DictReader(f.open(encoding="utf-8")))
    return np.array([float(r["coverage"]) for r in rows], dtype=float)


def steps_to(cov: np.ndarray, thr: float) -> int | None:
    idx = np.where(cov >= thr)[0]
    return int(idx[0]) if idx.size else None


def main() -> None:
    curves = {n: load_cov(n) for n in TEAMS}
    finals = {n: curves[n][-1] for n in TEAMS}
    lengths = {n: len(curves[n]) for n in TEAMS}
    max_len = max(lengths.values())

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 13, "axes.titlesize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3),
                             gridspec_kw={"width_ratios": [1.7, 1.0]})

    ax = axes[0]
    for n, spec in TEAMS.items():
        y = curves[n]
        ax.plot(np.arange(len(y)), y, label=spec["label"], color=spec["color"],
                linewidth=spec["lw"], alpha=0.9, zorder=spec["zorder"],
                solid_capstyle="round")
    ax.set_xlim(0, max_len - 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Exploration step")
    ax.set_ylabel("Coverage")
    ax.set_title("MSO scaling on MRPB")
    ax.grid(True, color="#D7DCE2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.04, "(a)", transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="left")
    ax.legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor="#888",
              borderpad=0.4, labelspacing=0.3)

    ax = axes[1]
    ns = list(TEAMS.keys())
    x = np.arange(len(ns))
    vals = [finals[n] for n in ns]
    colors = [TEAMS[n]["color"] for n in ns]
    b = ax.bar(x, vals, 0.62, color=colors, edgecolor="#1F2933", linewidth=0.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Final coverage")
    ax.set_xticks(x, [TEAMS[n]["label"] for n in ns], fontsize=10)
    ax.bar_label(b, fmt="%.2f", padding=2, fontsize=10)
    ax.set_title("Final coverage")
    ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(-0.18, 1.04, "(b)", transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=350, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print("=" * 52)
    print("MRPB SCALING SUMMARY (A2, 5-seed aggregated)")
    print("=" * 52)
    for n in ns:
        print(f"  N={n}: final coverage={finals[n]:.3f}  steps={lengths[n]}  "
              f"to0.8={steps_to(curves[n], 0.8)}")
    print(f"  Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
