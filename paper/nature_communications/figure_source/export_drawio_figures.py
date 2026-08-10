"""Export the author-supplied Fig. 1 and Fig. 2 pages on a white canvas.

The diagrams are exported directly from ``figs.drawio``.  No diagram element,
text label, line, colour or layout coordinate is edited here.  The only raster
post-processing is compositing any exporter alpha channel over true white and
writing publication-resolution metadata plus TIFF companions.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image


HERE = Path(__file__).resolve().parent
SUBMISSION = HERE.parent
PROJECT_ROOT = HERE.parents[2]
PROJECT_DRAWIO_SOURCE = PROJECT_ROOT / "figs.drawio"
PACKAGED_DRAWIO_SOURCE = HERE / "source_assets" / "figs.drawio"
DRAWIO_SOURCE = (
    PROJECT_DRAWIO_SOURCE
    if PROJECT_DRAWIO_SOURCE.is_file()
    else PACKAGED_DRAWIO_SOURCE
)
RASTER_DPI = 600


@dataclass(frozen=True)
class PageExport:
    page_name: str
    page_index: int
    width_px: int
    height_px: int
    output_stem: str


EXPORTS = (
    PageExport("fig1", 0, 4409, 2616, "figure1"),
    PageExport("fig2", 1, 7695, 2354, "figure2"),
)


def _find_drawio() -> Path:
    configured = os.environ.get("DRAWIO_CLI")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))

    for command in ("drawio", "draw.io"):
        resolved = shutil.which(command)
        if resolved:
            candidates.append(Path(resolved))

    candidates.extend(
        [
            Path(r"C:\Program Files\draw.io\draw.io.exe"),
            Path(r"C:\Program Files (x86)\draw.io\draw.io.exe"),
            Path.home() / "AppData/Local/Programs/draw.io/draw.io.exe",
            Path("/Applications/draw.io.app/Contents/MacOS/draw.io"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "draw.io CLI was not found. Set DRAWIO_CLI to the draw.io executable."
    )


def _validate_page_order(source: Path) -> None:
    root = ET.parse(source).getroot()
    names = [diagram.attrib.get("name", "") for diagram in root.findall("diagram")]
    for spec in EXPORTS:
        if spec.page_index >= len(names) or names[spec.page_index] != spec.page_name:
            raise RuntimeError(
                f"Expected page {spec.page_index} to be {spec.page_name!r}; "
                f"found {names!r}"
            )


def _flatten_to_white(source: Path, output_stem: Path, spec: PageExport) -> None:
    with Image.open(source) as opened:
        rgba = opened.convert("RGBA")
        if rgba.size != (spec.width_px, spec.height_px):
            raise RuntimeError(
                f"Unexpected {spec.page_name} dimensions: {rgba.size}; "
                f"expected {(spec.width_px, spec.height_px)}"
            )
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.alpha_composite(rgba)
        rgb = white.convert("RGB")

    corners = (
        rgb.getpixel((0, 0)),
        rgb.getpixel((rgb.width - 1, 0)),
        rgb.getpixel((0, rgb.height - 1)),
        rgb.getpixel((rgb.width - 1, rgb.height - 1)),
    )
    if any(pixel != (255, 255, 255) for pixel in corners):
        raise RuntimeError(f"{spec.page_name} does not have a true-white canvas")
    if all(low == 255 for low, _ in rgb.getextrema()):
        raise RuntimeError(f"{spec.page_name} export is blank")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(output_stem.with_suffix(".png"), dpi=(RASTER_DPI, RASTER_DPI))
    rgb.save(
        output_stem.with_suffix(".tiff"),
        dpi=(RASTER_DPI, RASTER_DPI),
        compression="tiff_lzw",
    )


def export_submission_figures() -> None:
    if not DRAWIO_SOURCE.is_file():
        raise FileNotFoundError(f"Missing diagram source: {DRAWIO_SOURCE}")
    _validate_page_order(DRAWIO_SOURCE)
    drawio = _find_drawio()

    with tempfile.TemporaryDirectory(prefix="mso_drawio_export_") as temp_dir:
        temp = Path(temp_dir)
        for spec in EXPORTS:
            raw_path = temp / f"{spec.output_stem}.png"
            command = [
                str(drawio),
                "--export",
                "--format",
                "png",
                "--page-index",
                str(spec.page_index),
                "--width",
                str(spec.width_px),
                "--border",
                "0",
                "--output",
                str(raw_path),
                str(DRAWIO_SOURCE),
            ]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0 or not raw_path.is_file():
                details = "\n".join(
                    part.strip() for part in (result.stdout, result.stderr) if part.strip()
                )
                raise RuntimeError(
                    f"draw.io failed to export {spec.page_name} "
                    f"(exit {result.returncode}).\n{details}"
                )
            _flatten_to_white(
                raw_path,
                SUBMISSION / spec.output_stem,
                spec,
            )

    print(
        "Exported white-canvas originals: "
        + ", ".join(
            f"{spec.output_stem}.png ({spec.width_px}x{spec.height_px})"
            for spec in EXPORTS
        )
    )


def main() -> None:
    export_submission_figures()


if __name__ == "__main__":
    main()
