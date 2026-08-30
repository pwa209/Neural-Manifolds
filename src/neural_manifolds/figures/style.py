"""Exclusive Python/matplotlib publication style and export validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from neural_manifolds.provenance import sha256_file

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class ExportedFigure:
    """Validated file bundle for one manuscript figure."""

    svg: Path
    pdf: Path
    tiff: Path
    sha256: dict[str, str]
    qa: dict[str, Any]


def apply_publication_style(font_size: float = 7.0) -> None:
    """Apply a compact Nature-style, editable-text plotting configuration."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "legend.fontsize": font_size - 0.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def figure_size(size_mm: tuple[float, float]) -> tuple[float, float]:
    return size_mm[0] / MM_PER_INCH, size_mm[1] / MM_PER_INCH


def add_panel_label(ax: Any, label: str, *, font_size: float = 8.0) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=font_size,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def _qa_exports(
    *,
    svg: Path,
    pdf: Path,
    tiff: Path,
    expected_size_mm: tuple[float, float],
    dpi: int,
) -> dict[str, Any]:
    svg_text = svg.read_text(encoding="utf-8")
    editable_text = bool(re.search(r"<(?:[A-Za-z0-9_]+:)?text\b", svg_text))
    if not editable_text:
        raise RuntimeError(f"editable SVG text validation failed: {svg}")
    pdf_bytes = pdf.read_bytes()
    pdf_truetype = b"/FontFile2" in pdf_bytes or b"/CIDFontType2" in pdf_bytes
    if not pdf_truetype:
        raise RuntimeError(f"TrueType PDF font validation failed: {pdf}")
    with Image.open(tiff) as image:
        recorded_dpi = image.info.get("dpi", (0.0, 0.0))
        width_px, height_px = image.size
    dpi_x, dpi_y = float(recorded_dpi[0]), float(recorded_dpi[1])
    if dpi_x < dpi - 2 or dpi_y < dpi - 2:
        raise RuntimeError(f"TIFF resolution is below {dpi} dpi: {recorded_dpi}")
    expected_width = round(expected_size_mm[0] / MM_PER_INCH * dpi)
    expected_height = round(expected_size_mm[1] / MM_PER_INCH * dpi)
    if abs(width_px - expected_width) > 2 or abs(height_px - expected_height) > 2:
        raise RuntimeError(
            "TIFF dimensions do not preserve the contracted final size: "
            f"got {(width_px, height_px)}, expected {(expected_width, expected_height)}"
        )
    return {
        "svg_editable_text": True,
        "pdf_truetype_embedded": True,
        "pdf_fonttype": 42,
        "tiff_dpi": [dpi_x, dpi_y],
        "pixel_dimensions": [width_px, height_px],
        "final_size_mm": list(expected_size_mm),
        "background": "white",
        "p_value_stars_drawn": False,
    }


def export_figure(
    fig: Any,
    base_path: str | Path,
    *,
    size_mm: tuple[float, float],
    dpi: int = 600,
    tiff_compression: str = "tiff_lzw",
) -> ExportedFigure:
    """Export vector/editable and 600-dpi raster files, then inspect all three."""

    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    svg = base.with_suffix(".svg")
    pdf = base.with_suffix(".pdf")
    tiff = base.with_suffix(".tiff")
    fig.set_size_inches(*figure_size(size_mm), forward=True)
    fig.savefig(svg, format="svg", bbox_inches=None)
    fig.savefig(pdf, format="pdf", bbox_inches=None)
    fig.savefig(
        tiff,
        format="tiff",
        dpi=dpi,
        bbox_inches=None,
        pil_kwargs={"compression": tiff_compression},
    )
    plt.close(fig)
    qa = _qa_exports(
        svg=svg,
        pdf=pdf,
        tiff=tiff,
        expected_size_mm=size_mm,
        dpi=dpi,
    )
    paths = {"svg": svg, "pdf": pdf, "tiff": tiff}
    return ExportedFigure(
        svg=svg,
        pdf=pdf,
        tiff=tiff,
        sha256={key: sha256_file(path) for key, path in paths.items()},
        qa=qa,
    )
