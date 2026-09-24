"""Read-only figure export checks; write a separate QA record, never modify artwork.

Run with the bundled PDF runtime (pypdf and pdfplumber). Visual inspection of
actual rasterized PDF pages remains a separate, required check.
"""

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


def audit(root, renderer):
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["renderer_sha256"] == hashlib.sha256(renderer.read_bytes()).hexdigest()
    for record in manifest["outputs"]:
        assert hashlib.sha256((root / record["file"]).read_bytes()).hexdigest() == record["sha256"]
    reader = PdfReader(root / "Figures_2-5_draft.pdf")
    assert len(reader.pages) == 4
    fonts = {}
    for page in reader.pages:
        for ref in page["/Resources"]["/Font"].values():
            font = ref.get_object()
            base = str(font["/BaseFont"])
            children = font.get("/DescendantFonts", [font])
            descriptors = [child.get_object().get("/FontDescriptor") for child in children]
            embedded = all(
                d is not None
                and any(key in d.get_object() for key in ["/FontFile", "/FontFile2", "/FontFile3"])
                for d in descriptors
            )
            fonts[base] = embedded
    assert all(fonts.values()), fonts
    pdf_pages = []
    with pdfplumber.open(root / "Figures_2-5_draft.pdf") as pdf:
        for number, page in enumerate(pdf.pages, 2):
            # pdfplumber's character-box height is not font size for rotated labels.
            # Read declared text sizes from PDF text operators instead.
            sizes = []

            def record_size(text, _cm, _tm, _font, size, _sizes=sizes):
                if text.strip():
                    _sizes.append(float(size))

            reader.pages[number - 2].extract_text(visitor_text=record_size)
            assert min(sizes) >= 6.99
            assert f"FIGURE {number}" in page.extract_text()
            pdf_pages.append(
                {
                    "figure": number,
                    "width_mm": page.width * 25.4 / 72,
                    "height_mm": page.height * 25.4 / 72,
                    "minimum_font_pt": min(sizes),
                    "text_characters": len(page.chars),
                }
            )
    vectors = []
    for path in sorted(root.glob("fig*.svg")):
        tree = ET.parse(path)
        texts = tree.findall(".//{http://www.w3.org/2000/svg}text")
        assert len(texts) > 30
        vectors.append({"file": path.name, "editable_text_elements": len(texts)})
    counts = {}
    for name, expected in [
        ("context_interactions.csv", 27),
        ("subjective_all_396.csv", 396),
        ("simulation_all_settings.csv", 36),
        ("null_controls.csv", 72),
    ]:
        with (root / "source_tables" / name).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == expected, (name, len(rows))
        counts[name] = len(rows)
    layout = json.loads((root / "layout_checks.json").read_text())
    assert not any(r["outside_canvas_text"] for r in layout)
    result = {
        "artifact_hashes_verified": len(manifest["outputs"]),
        "renderer_matches_local_source": True,
        "embedded_fonts": fonts,
        "pdf_pages": pdf_pages,
        "svg_text": vectors,
        "source_table_counts": counts,
        "canvas_checks": "pass",
        "visual_inspection": "separate manual record",
    }
    target = root / "export_audit.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    args = parser.parse_args()
    audit(args.directory, args.renderer)
