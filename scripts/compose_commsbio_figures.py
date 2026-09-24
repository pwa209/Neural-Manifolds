"""Compose original-results-only four-figure draft from reviewed figure PDFs.

This changes panel order/numbering only. It does not read participant data,
recompute statistics, or incorporate later LODE/DREAM validation results.
Figure 1a is intentionally blank for author-supplied BioRender artwork.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

import pdfplumber
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


MAP = (
    ("fig02_prediction", "figure01_framework_prediction"),
    ("fig03_null_controls", "figure02_null_controls"),
    ("fig04_context_subjective", "figure03_context_subjective"),
    ("fig05_measurement", "figure04_measurement"),
)
SLOT_HEIGHT_PT = 180.0
NULL_STRIP_PT = 100.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_panel_letters(path: Path) -> tuple[float, list[dict[str, float | str]]]:
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 1:
            raise ValueError("Expected a single-page prediction figure")
        page = pdf.pages[0]
        targets = (("a", "b", 13, 26), ("b", "c", 13, 212),
                   ("c", "d", 262, 212), ("d", "e", 13, 374))
        found = []
        for old, new, x, top in targets:
            hits = [
                char for char in page.chars
                if char["text"] == old
                and abs(char["x0"] - x) < 2
                and abs(char["top"] - top) < 2
            ]
            if len(hits) != 1:
                raise ValueError(f"Cannot uniquely locate original panel {old}")
            found.append(dict(hits[0], replacement=new))
        return float(page.height), found


def overlay(width: float, height: float, original_height: float,
            letters: list[dict[str, float | str]]) -> bytes:
    buffer = io.BytesIO()
    drawing = canvas.Canvas(buffer, pagesize=(width, height), pageCompression=1)
    for char in letters:
        x0, x1 = float(char["x0"]), float(char["x1"])
        top, bottom = float(char["top"]), float(char["bottom"])
        drawing.setFillColorRGB(1, 1, 1)
        drawing.rect(x0 - 1.2, original_height - bottom - 1.0,
                     x1 - x0 + 2.4, bottom - top + 2.0, stroke=0, fill=1)
        drawing.setFillColorRGB(0, 0, 0)
        drawing.setFont("Helvetica-Bold", 11)
        drawing.drawString(x0, original_height - bottom + 1.0,
                           str(char["replacement"]))

    # Deliberately empty BioRender slot; only the panel label and alignment
    # boundary are provided. There are no result icons or inferred mechanisms.
    drawing.setStrokeColorRGB(0.78, 0.81, 0.83)
    drawing.setLineWidth(0.5)
    drawing.rect(12.5, original_height + 16, width - 25,
                 SLOT_HEIGHT_PT - 28, stroke=1, fill=0)
    drawing.setFillColorRGB(0, 0, 0)
    drawing.setFont("Helvetica-Bold", 11)
    drawing.drawString(13, height - 35, "a")
    drawing.save()
    return buffer.getvalue()


def compose_figure_one(source: Path, destination: Path) -> None:
    original_height, letters = original_panel_letters(source)
    reader = PdfReader(source)
    original = reader.pages[0]
    width = float(original.mediabox.width)
    if abs(float(original.mediabox.height) - original_height) > 0.1:
        raise ValueError("Prediction PDF geometry disagrees between readers")
    height = original_height + SLOT_HEIGHT_PT
    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=height)
    page.merge_page(original)
    page.merge_page(PdfReader(io.BytesIO(overlay(width, height, original_height, letters))).pages[0])
    with destination.open("wb") as stream:
        writer.write(stream)


def compose_figure_two(source: Path, destination: Path) -> None:
    with pdfplumber.open(source) as pdf:
        if len(pdf.pages) != 1:
            raise ValueError("Expected a single-page null-control figure")
        originals = pdf.pages[0]
        old_height = float(originals.height)
        targets = (("a", "b", 14.5), ("b", "c", 270.3))
        letters = []
        for old, new, x in targets:
            hits = [char for char in originals.chars
                    if char["text"] == old and abs(char["x0"] - x) < 2
                    and abs(char["top"] - 14.8) < 2]
            if len(hits) != 1:
                raise ValueError(f"Cannot uniquely locate null panel {old}")
            letters.append(dict(hits[0], replacement=new))
    original = PdfReader(source).pages[0]
    width = float(original.mediabox.width)
    height = old_height + NULL_STRIP_PT
    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=height)
    page.merge_page(original)
    buffer = io.BytesIO()
    drawing = canvas.Canvas(buffer, pagesize=(width, height), pageCompression=1)
    for char in letters:
        x0, x1 = float(char["x0"]), float(char["x1"])
        top, bottom = float(char["top"]), float(char["bottom"])
        drawing.setFillColorRGB(1, 1, 1)
        drawing.rect(x0 - 1.2, old_height - bottom - 1,
                     x1 - x0 + 2.4, bottom - top + 2, stroke=0, fill=1)
        drawing.setFillColorRGB(0, 0, 0)
        drawing.setFont("Helvetica-Bold", 11)
        drawing.drawString(x0, old_height - bottom + 1, str(char["replacement"]))

    drawing.setFont("Helvetica-Bold", 11)
    drawing.drawString(14.5, height - 26, "a")
    drawing.setFont("Helvetica-Bold", 8.5)
    drawing.drawString(35, height - 25, "Matched null-control logic")
    box_y = old_height + 22
    for x, w, text in ((36, 119, "Observed representation"),
                       (220, 118, "Label permutation"),
                       (365, 130, "Temporal-order shuffle")):
        drawing.setStrokeColorRGB(.69, .74, .77)
        drawing.setLineWidth(.65)
        drawing.roundRect(x, box_y, w, 25, 3, stroke=1, fill=0)
        drawing.setFillColorRGB(.12, .16, .19)
        drawing.setFont("Helvetica", 7.3)
        drawing.drawCentredString(x + w / 2, box_y + 9, text)
    drawing.setStrokeColorRGB(.3, .42, .46)
    drawing.line(155, box_y + 12.5, 218, box_y + 12.5)
    drawing.line(214, box_y + 15, 218, box_y + 12.5)
    drawing.line(214, box_y + 10, 218, box_y + 12.5)
    drawing.line(95.5, box_y + 25, 95.5, old_height + 59)
    drawing.line(95.5, old_height + 59, 430, old_height + 59)
    drawing.line(430, old_height + 59, 430, box_y + 27)
    drawing.line(427.5, box_y + 31, 430, box_y + 27)
    drawing.line(432.5, box_y + 31, 430, box_y + 27)
    drawing.setFillColorRGB(.35, .39, .42)
    drawing.setFont("Helvetica", 7.1)
    drawing.drawString(36, old_height + 6,
                       "Each transformed dataset repeats the complete nested fit and held-out evaluation.")
    drawing.save()
    page.merge_page(PdfReader(io.BytesIO(buffer.getvalue())).pages[0])
    with destination.open("wb") as stream:
        writer.write(stream)


def render_png(pdf_path: Path, png_path: Path, pdftoppm: Path) -> None:
    prefix = png_path.with_suffix("")
    subprocess.run(
        [str(pdftoppm), "-f", "1", "-singlefile", "-r", "300", "-png",
         str(pdf_path), str(prefix)],
        check=True,
    )
    if not png_path.is_file():
        raise RuntimeError(f"Poppler did not create {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdftoppm", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing figure draft")
    for old, _ in MAP:
        for suffix in ("pdf", "png"):
            if not (args.source / f"{old}.{suffix}").is_file():
                raise FileNotFoundError(args.source / f"{old}.{suffix}")
    args.output.mkdir(parents=True)
    source = args.source / "fig02_prediction.pdf"
    figure_one = args.output / "figure01_framework_prediction.pdf"
    compose_figure_one(source, figure_one)
    render_png(figure_one, args.output / "figure01_framework_prediction.png", args.pdftoppm)
    figure_two = args.output / "figure02_null_controls.pdf"
    compose_figure_two(args.source / "fig03_null_controls.pdf", figure_two)
    render_png(figure_two, args.output / "figure02_null_controls.png", args.pdftoppm)
    for old, new in MAP[2:]:
        for suffix in ("pdf", "png"):
            shutil.copy2(args.source / f"{old}.{suffix}", args.output / f"{new}.{suffix}")

    bundle = PdfWriter()
    for _, new in MAP:
        bundle.append(PdfReader(args.output / f"{new}.pdf"))
    with (args.output / "Figures_1-4_original_results.pdf").open("wb") as stream:
        bundle.write(stream)
    manifest = {
        "scope": "Original study results only; later LODE and DREAM validations excluded at user request",
        "figure_1a": "Blank BioRender slot; no scientific content drawn",
        "statistics_recomputed": False,
        "original_source_sha256": {
            f"{old}.pdf": sha256(args.source / f"{old}.pdf") for old, _ in MAP
        },
        "output_sha256": {
            path.name: sha256(path) for path in sorted(args.output.glob("*.pdf"))
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
