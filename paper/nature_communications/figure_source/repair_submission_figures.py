"""Apply evidence-preserving corrections to the existing submission figures.

This script does not redesign the image plates.  It makes four bounded edits:

* exports the author-supplied Fig. 1 and Fig. 2 draw.io pages on white;
* replaces the two approximate MRPB plots with raw retained aggregate traces;
* corrects the physical endpoint metric name; and
* exports TIFF companions for the corrected raster composites.

All quantitative lines are read directly from packaged CSV files.  No
smoothing, interpolation, subsampling or synthetic values are used.
"""

from __future__ import annotations

import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from export_drawio_figures import export_submission_figures


HERE = Path(__file__).resolve().parent
SUBMISSION = HERE.parent
MRPB_DATA = SUBMISSION / "peer_review_data" / "mrpb" / "all_metrics"
GENERATED = HERE / "generated"
SOURCE_ASSETS = HERE / "source_assets"

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

MRPB_SERIES = (
    ("ours_multi_ours_orb.csv", "MSO multi", "#5B3F9B", "-"),
    ("ours_multi_ours_orb_nomerge.csv", "MSO no merge", "#8A78C2", "--"),
    ("nearest-multi-our-orb.csv", "Frontier multi", "#586A83", ":"),
    ("ours_single.csv", "MSO single", "#5B3F9B", "-."),
    ("MapEx_single.csv", "MapEx", "#2F6FB2", "-"),
    ("ig-hector_single.csv", "IG-Hector", "#2B91A3", "--"),
)


def _font_path(condensed: bool = False) -> str:
    if condensed:
        candidate = Path(fm.findfont("DejaVu Sans"))
        condensed_path = candidate.with_name("DejaVuSansCondensed-Bold.ttf")
        if condensed_path.exists():
            return str(condensed_path)
    return fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))


def _fit_font(text: str, max_width: int, max_size: int, min_size: int = 24) -> ImageFont.FreeTypeFont:
    path = _font_path(condensed=True)
    for size in range(max_size, min_size - 1, -1):
        font = ImageFont.truetype(path, size=size)
        box = font.getbbox(text)
        if box[2] - box[0] <= max_width:
            return font
    return ImageFont.truetype(path, size=min_size)


def _replace_profiler_cells(source_svg: Path, output_svg: Path) -> None:
    tree = ET.parse(source_svg)
    root = tree.getroot()

    metadata = root.attrib.get("content", "")
    metadata = metadata.replace("Multi-Robot", "Multirobot")
    metadata = metadata.replace("Model MACs (MFLOPs)", "Profiler operation entries")
    metadata = metadata.replace("660.005 ¡Á 10^2 MFLOPS", "66,000.5")
    metadata = metadata.replace("547.400 ¡Á 10^2 MFLOPS", "54,740.0")
    metadata = metadata.replace("336.2 MFLOPS", "336.2")
    metadata = metadata.replace("MFLOPS", "operation entries")
    root.attrib["content"] = metadata

    # Preserve the original rounded title box, replacing only its visible
    # wording.  Rebuilding this one group also updates the SVG fallback path,
    # which otherwise contains a rasterised copy of the old hyphenated label.
    for group in root.iter(f"{{{SVG_NS}}}g"):
        if group.attrib.get("data-cell-id") != "BBy7hOv_NdyBTfu9mbVX-2":
            continue
        group.clear()
        group.attrib["data-cell-id"] = "BBy7hOv_NdyBTfu9mbVX-2"
        ET.SubElement(
            group,
            f"{{{SVG_NS}}}rect",
            {
                "x": "177.13",
                "y": "7",
                "width": "75.87",
                "height": "33.5",
                "rx": "5.02",
                "ry": "5.02",
                "fill": "#cce5ff",
                "stroke": "#36393d",
            },
        )
        for y, value in (("20", "Multirobot"), ("33", "System")):
            title = ET.SubElement(
                group,
                f"{{{SVG_NS}}}text",
                {
                    "x": "215.065",
                    "y": y,
                    "text-anchor": "middle",
                    "font-family": "Helvetica, Arial, sans-serif",
                    "font-size": "10",
                    "font-weight": "700",
                    "fill": "#000000",
                },
            )
            title.text = value
        break

    replacement = {
        "HPAiXndjdLM7GzWBDHrP-3": (86.0, 143.0, "Profiler operation entries", 8.5, "#1685E5"),
        "HPAiXndjdLM7GzWBDHrP-14": (103.5, 157.0, "66,000.5", 7.0, "#000000"),
        "HPAiXndjdLM7GzWBDHrP-12": (51.0, 166.0, "54,740.0", 7.0, "#000000"),
        "HPAiXndjdLM7GzWBDHrP-10": (143.0, 226.0, "336.2", 7.0, "#1685E5"),
    }

    for group in root.iter(f"{{{SVG_NS}}}g"):
        cell_id = group.attrib.get("data-cell-id")
        if cell_id not in replacement:
            continue
        group.clear()
        group.attrib["data-cell-id"] = cell_id
        x, y, value, size, colour = replacement[cell_id]
        text = ET.SubElement(
            group,
            f"{{{SVG_NS}}}text",
            {
                "x": f"{x:g}",
                "y": f"{y:g}",
                "text-anchor": "middle",
                "font-family": "Helvetica, Arial, sans-serif",
                "font-size": f"{size:g}",
                "font-weight": "700",
                "fill": colour,
            },
        )
        text.text = value

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_svg, encoding="utf-8", xml_declaration=True)

    revised = output_svg.read_text(encoding="utf-8")
    if "MFLOP" in revised or "Model MACs" in revised or "Multi-Robot" in revised:
        raise RuntimeError("Legacy operation labels remain in the revised Fig. 1 SVG")


def _patch_figure1_png(source_path: Path, output_path: Path) -> None:
    image = Image.open(source_path).convert("RGBA")
    if image.size != (4409, 2615):
        raise ValueError(f"Unexpected Fig. 1 raster dimensions: {image.size}")
    draw = ImageDraw.Draw(image)

    # Cover only the interior of the supplied rounded title box; its fill,
    # outline, corner radius and position remain untouched.
    title_fill = image.getpixel((1290, 120))
    draw.rectangle((1082, 60, 1498, 135), fill=title_fill)
    draw.rectangle((1140, 136, 1440, 225), fill=title_fill)
    title_font = _fit_font("Multirobot", 410, 64, 50)
    system_font = _fit_font("System", 360, 64, 50)
    draw.text((1290, 105), "Multirobot", font=title_font, fill="black", anchor="mm")
    draw.text((1290, 185), "System", font=system_font, fill="black", anchor="mm")

    blue = "#1685E5"
    black = (0, 0, 0, 255)
    draw.rectangle((45, 780, 1000, 875), fill=black)
    header = _fit_font("Profiler operation entries", 900, 54, 42)
    units = _fit_font("(millions)", 300, 38, 30)
    draw.text((522, 810), "Profiler operation entries", font=header, fill=blue, anchor="mm")
    draw.text((522, 855), "(millions)", font=units, fill=blue, anchor="mm")

    # Cover only the legacy glyph areas; the bar geometry and colours remain unchanged.
    u_colour = image.getpixel((300, 1120))
    lama_colour = image.getpixel((560, 1120))
    draw.rectangle((198, 890, 426, 1075), fill=u_colour)
    draw.rectangle((472, 870, 698, 1045), fill=lama_colour)
    u_font = _fit_font("54,740.0", 214, 48, 40)
    lama_font = _fit_font("66,000.5", 214, 48, 40)
    draw.text((312, 950), "54,740.0", font=u_font, fill="black", anchor="mm")
    draw.text((585, 935), "66,000.5", font=lama_font, fill="black", anchor="mm")

    draw.rectangle((755, 1292, 925, 1383), fill=black)
    mso_font = _fit_font("336.2", 160, 50, 40)
    draw.text((840, 1338), "336.2", font=mso_font, fill=blue, anchor="mm")

    dpi = image.info.get("dpi", (600, 600))
    image.save(output_path, dpi=dpi)
    image.convert("RGB").save(
        output_path.with_suffix(".tiff"),
        dpi=dpi,
        compression="tiff_lzw",
    )


def _read_metric(path: Path, metric: str, max_step: int = 1800) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = float(row["step"])
            if step > max_step:
                continue
            value = float(row[metric])
            x.append(step)
            y.append(value)
    if not x:
        raise ValueError(f"No {metric} observations at step <= {max_step}: {path}")
    return x, y


def _render_mrpb_metrics(output_prefix: Path) -> None:
    width_px, height_px, dpi = 2210, 3290, 600
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            # The source strip uses a 5--7 pt glyph floor.  Once embedded in
            # the author-supplied composite, its effective placed size is
            # smaller because the strip occupies only 14.3% of the width.
            # That preserved-layout limitation is recorded in FIGURE_QA.md.
            "font.size": 6.5,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    axes = [
        fig.add_axes([0.18, 0.595, 0.76, 0.320]),
        fig.add_axes([0.18, 0.105, 0.76, 0.320]),
    ]
    metrics = (
        ("coverage", "Coverage", "(c)"),
        ("predicted_map_quality", "Occupied-cell precision", "(d)"),
    )

    top_handles = []
    top_labels = []
    for ax, (metric, title, panel) in zip(axes, metrics):
        for filename, label, colour, line_style in MRPB_SERIES:
            x, y = _read_metric(MRPB_DATA / filename, metric)
            if metric == "predicted_map_quality" and max(y) == 0.0:
                # The all-zero frontier-multi field is an unavailable/sentinel
                # quality record, not a measured zero-precision trace.
                continue
            (line,) = ax.plot(
                x,
                y,
                color=colour,
                linewidth=1.25,
                linestyle=line_style,
                solid_capstyle="round",
                label=label,
            )
            if metric == "coverage":
                top_handles.append(line)
                top_labels.append(label)

        ax.set_xlim(0, 1800)
        ax.set_ylim(0, 1.03)
        ax.set_xticks([0, 600, 1200, 1800])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, color="#D7DCE2", linewidth=1.0)
        ax.set_axisbelow(True)
        ax.set_title(title, pad=2, linespacing=0.84)
        ax.text(
            0.025,
            0.965,
            panel,
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="bold",
            va="top",
            ha="left",
        )
        ax.tick_params(width=0.9, length=2.5, pad=1.5)
    axes[0].set_xlabel("Exploration step")
    axes[1].set_xlabel("Exploration step")
    axes[0].set_ylabel("Fraction")
    axes[1].set_ylabel("Fraction")

    short_labels = [
        "MSO multi",
        "No merge",
        "Frontier (c only)",
        "MSO single",
        "MapEx",
        "IG-Hector",
    ]
    fig.legend(
        top_handles,
        short_labels,
        loc="center",
        bbox_to_anchor=(0.50, 0.510),
        ncol=3,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#888888",
        borderpad=0.30,
        labelspacing=0.20,
        handlelength=1.45,
        handletextpad=0.40,
        columnspacing=0.65,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(output_prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(
        output_prefix.with_suffix(".tiff"),
        dpi=dpi,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    if Image.open(output_prefix.with_suffix(".png")).size != (width_px, height_px):
        raise RuntimeError("MRPB metrics export changed the fixed composite dimensions")


def _patch_mrpb_composite(source_path: Path, output_path: Path, panel_path: Path) -> None:
    image = Image.open(source_path).convert("RGBA")
    if image.size != (15460, 3290):
        raise ValueError(f"Unexpected MRPB composite dimensions: {image.size}")
    panel = Image.open(panel_path).convert("RGBA")
    if panel.size != (2210, 3290):
        raise ValueError(f"Unexpected MRPB metric-panel dimensions: {panel.size}")
    image.paste(panel, (13250, 0), panel)
    dpi = image.info.get("dpi", (144, 144))
    image.save(output_path, dpi=dpi)
    image.convert("RGB").save(
        output_path.with_suffix(".tiff"),
        dpi=dpi,
        compression="tiff_lzw",
    )


def _patch_physical_metric(source_path: Path, output_path: Path) -> None:
    image = Image.open(source_path).convert("RGBA")
    if image.size != (6619, 1907):
        raise ValueError(f"Unexpected physical composite dimensions: {image.size}")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5590, 940, 6618, 1042), fill="white")
    label = "(b) Occupied-cell precision"
    font = _fit_font(label, 995, 64, 48)
    draw.text((5608, 952), label, font=font, fill="black")
    dpi = image.info.get("dpi", (72, 72))
    image.save(output_path, dpi=dpi)
    image.convert("RGB").save(
        output_path.with_suffix(".tiff"),
        dpi=dpi,
        compression="tiff_lzw",
    )


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    SOURCE_ASSETS.mkdir(parents=True, exist_ok=True)

    export_submission_figures()

    for name in ("figure4.png", "figure5.png"):
        source = SUBMISSION / name
        frozen = SOURCE_ASSETS / name.replace(".png", "_base.png")
        if not frozen.exists():
            shutil.copy2(source, frozen)

    mrpb_prefix = GENERATED / "figure4_exact_metrics"
    _render_mrpb_metrics(mrpb_prefix)
    _patch_mrpb_composite(
        SOURCE_ASSETS / "figure4_base.png",
        SUBMISSION / "figure4.png",
        mrpb_prefix.with_suffix(".png"),
    )
    _patch_physical_metric(
        SOURCE_ASSETS / "figure5_base.png",
        SUBMISSION / "figure5.png",
    )

    print("Exported white originals and corrected the retained quantitative plates.")


if __name__ == "__main__":
    main()
