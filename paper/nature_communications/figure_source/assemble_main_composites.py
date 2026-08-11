"""Assemble evidence-preserving main-text figure composites.

The script changes layout, not experimental content.  It keeps the supplied
architecture, registration, progression, arena and map panels intact apart
from deterministic scaling/cropping needed for page composition.  No image is
used to derive a numerical result.

Outputs
-------
``figure2_composite``
    The original predictor/distillation architecture followed by the supplied
    predictive registration schematic that previously appeared in the SI.
``figure4_qualitative``
    The two original MRPB progression blocks, excluding the obsolete narrow
    metric strip.  Source-backed vector metric panels are included separately
    from LaTeX at full text width.
``figure5_composite``
    The three supplied arena views above the corrected physical progression
    and endpoint panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SUBMISSION = HERE.parent
SOURCE_ASSETS = HERE / "source_assets"

INK = "#202A44"
NAVY = "#334E78"
PALE_COOL = "#E9ECF7"
FINAL_WIDTH_MM = 183.0
RASTER_DPI = 600
MM_PER_INCH = 25.4
FINAL_WIDTH_PX = round(FINAL_WIDTH_MM / MM_PER_INCH * RASTER_DPI)


def _font(size: int, *, bold: bool = True) -> ImageFont.FreeTypeFont:
    properties = fm.FontProperties(
        family="DejaVu Sans",
        weight="bold" if bold else "normal",
    )
    return ImageFont.truetype(fm.findfont(properties), size=size)


def _save(image: Image.Image, stem: str) -> None:
    png_path = SUBMISSION / f"{stem}.png"
    tiff_path = SUBMISSION / f"{stem}.tiff"
    rgb = image.convert("RGB")
    if rgb.width != FINAL_WIDTH_PX:
        final_height = round(rgb.height * FINAL_WIDTH_PX / rgb.width)
        rgb = rgb.resize((FINAL_WIDTH_PX, final_height), Image.Resampling.LANCZOS)
    dpi = (RASTER_DPI, RASTER_DPI)
    rgb.save(png_path, dpi=dpi)
    rgb.save(tiff_path, dpi=dpi, compression="tiff_lzw")


def _fit_exact(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.width * height != image.height * width:
        raise ValueError(
            f"Requested geometry {width}x{height} changes aspect of {image.size}"
        )
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _input_asset(name: str) -> Path:
    """Resolve a submission asset, falling back to the packaged source copy."""
    submission_path = SUBMISSION / name
    if submission_path.is_file():
        return submission_path
    packaged_path = SOURCE_ASSETS / name
    if packaged_path.is_file():
        return packaged_path
    raise FileNotFoundError(f"Missing figure input: {name}")


def assemble_method_figure() -> None:
    architecture = Image.open(_input_asset("figure2.png")).convert("RGB")
    registration = Image.open(_input_asset("supplementary_figure1.png")).convert("RGB")
    if architecture.size != (7695, 2354):
        raise ValueError(f"Unexpected predictor architecture size: {architecture.size}")
    if registration.size != (2534, 674):
        raise ValueError(f"Unexpected registration schematic size: {registration.size}")

    width = architecture.width
    registration_height = round(registration.height * width / registration.width)
    registration_scaled = registration.resize(
        (width, registration_height),
        Image.Resampling.LANCZOS,
    )
    header_height = 150
    inter_panel_gap = 55
    bottom_margin = 35
    height = (
        header_height
        + architecture.height
        + inter_panel_gap
        + header_height
        + registration_scaled.height
        + bottom_margin
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    panel_font = _font(110)
    title_font = _font(76)

    draw.text((35, 12), "a", fill=INK, font=panel_font)
    draw.text(
        (190, 34),
        "Local completion and feature distillation",
        fill=NAVY,
        font=title_font,
    )
    y = header_height
    canvas.paste(architecture, (0, y))

    y += architecture.height + inter_panel_gap
    draw.text((35, y + 12), "b", fill=INK, font=panel_font)
    draw.text(
        (190, y + 34),
        "Predictive registration and provisional fusion",
        fill=NAVY,
        font=title_font,
    )
    y += header_height
    canvas.paste(registration_scaled, (0, y))
    _save(canvas, "figure2_composite")


def assemble_mrpb_qualitative() -> None:
    source = Image.open(SOURCE_ASSETS / "figure4_base.png").convert("RGB")
    if source.size != (15460, 3290):
        raise ValueError(f"Unexpected MRPB source size: {source.size}")

    # Pixel boundaries follow the supplied plate: x=13,330 is the right edge
    # of the second five-frame progression block and precedes the old plot.
    plate = source.crop((0, 0, 13330, 3290))
    header_height = 215
    canvas = Image.new("RGB", (plate.width, plate.height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    panel_font = _font(150)
    title_font = _font(140)
    draw.text((30, 12), "a", fill=INK, font=panel_font)
    draw.text((220, 42), "Team exploration", fill=NAVY, font=title_font)
    draw.text((6690, 12), "b", fill=INK, font=panel_font)
    draw.text((6880, 42), "Single-robot baselines", fill=NAVY, font=title_font)
    canvas.paste(plate, (0, header_height))
    _save(canvas, "figure4_qualitative")


def assemble_physical_figure() -> None:
    physical = Image.open(_input_asset("figure5.png")).convert("RGB")
    if physical.size != (6619, 1907):
        raise ValueError(f"Unexpected physical result size: {physical.size}")

    scene_paths = [_input_asset(f"supplementary_scene{i}.png") for i in (1, 2, 3)]
    scenes = [Image.open(path).convert("RGB") for path in scene_paths]
    if any(scene.size != (1200, 675) for scene in scenes):
        raise ValueError(f"Unexpected arena-view sizes: {[scene.size for scene in scenes]}")

    gap = 20
    panel_width = (physical.width - 2 * gap) // 3
    if panel_width * 3 + 2 * gap != physical.width:
        raise RuntimeError("Arena-strip width does not tile the physical figure")
    panel_height = round(panel_width * scenes[0].height / scenes[0].width)
    label_height = 120
    section_gap = 50
    canvas_height = panel_height + label_height + section_gap + physical.height
    canvas = Image.new("RGB", (physical.width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = _font(78)

    for index, scene in enumerate(scenes):
        x = index * (panel_width + gap)
        scaled = scene.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        canvas.paste(scaled, (x, 0))
        draw.rectangle(
            (x, panel_height, x + panel_width, panel_height + label_height),
            fill=PALE_COOL,
        )
        text = f"Scene {index + 1}  |  controlled arena"
        text_box = draw.textbbox((0, 0), text, font=label_font)
        text_width = text_box[2] - text_box[0]
        draw.text(
            (x + (panel_width - text_width) // 2, panel_height + 17),
            text,
            fill=NAVY,
            font=label_font,
        )

    y = panel_height + label_height + section_gap
    canvas.paste(physical, (0, y))
    _save(canvas, "figure5_composite")


def main() -> None:
    assemble_method_figure()
    assemble_mrpb_qualitative()
    assemble_physical_figure()
    print("Assembled method, MRPB and physical main-text composites.")


if __name__ == "__main__":
    main()
