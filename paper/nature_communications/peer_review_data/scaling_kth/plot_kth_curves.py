"""Render the retained KTH coverage trace without unsourced metrics.

The packaged record contains one aggregate coverage CSV.  Historical
run-level seed traces and the old prediction-precision series were not
retained, so this script deliberately renders coverage only and does not add
an uncertainty band.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "A3_ours_multi_ours_orb_coverage.csv"
MSO_COLOR = "#734AAD"


def read_coverage(path: Path) -> tuple[list[int], list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return (
        [int(row["step"]) for row in rows],
        [float(row["coverage"]) for row in rows],
    )


def render(output_prefix: Path) -> None:
    steps, coverage = read_coverage(DATA)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(4.8, 3.15))
    ax.plot(
        steps,
        coverage,
        color=MSO_COLOR,
        linewidth=2.2,
        solid_capstyle="round",
        label="MSO",
    )
    ax.set_xlim(0, max(steps))
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Exploration step")
    ax.set_ylabel("Coverage")
    ax.set_title("KTH coverage")
    ax.grid(True, color="#D7DCE2", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        output_prefix.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "figs" / "appendix" / "kth_coverage",
    )
    args = parser.parse_args()
    render(args.output_prefix)
    print(
        f"Rendered {len(read_coverage(DATA)[0])} retained aggregate rows; "
        "run-level n unavailable."
    )


if __name__ == "__main__":
    main()
