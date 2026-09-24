"""Verify a private aggregate-results export and build its portable ZIP package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.export.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["outputs"]:
        path = root / record["file"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Source-export checksum mismatch: {path}")
    for name, expected in manifest["table_rows"].items():
        with (root / "tables" / name).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            if any("participant_id" in h.lower() or "paired_values" in h.lower() for h in headers):
                raise ValueError(f"Participant-level column in {name}")
            count = sum(1 for _ in reader)
        if count != expected:
            raise ValueError(f"Row-count mismatch: {name}: {count} != {expected}")

    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    for source in (args.methods, args.results):
        destination = reports / source.name
        if destination.exists() and digest(destination) != digest(source):
            raise ValueError(f"Existing report differs: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)

    skip = {"manifest-final.json", "Statistical_results_20260924.zip", "workbook_overview_qa.png"}
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.name not in skip
        and not path.name.endswith(".inspect.ndjson")
    )
    final = {
        "description": "Reviewed aggregate statistical export; no participant-level records",
        "source_manifest": "manifest.json",
        "file_count_excluding_this_manifest": len(files),
        "files": [{"file": path.relative_to(root).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size} for path in files],
    }
    final_path = root / "manifest-final.json"
    final_path.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    archive = root / "Statistical_results_20260924.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as bundle:
        for path in [*files, final_path]:
            bundle.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP integrity test failed")
        if len(bundle.namelist()) != len(files) + 1:
            raise ValueError("ZIP member-count mismatch")
    print(json.dumps({"archive": str(archive), "files": len(files) + 1, "bytes": archive.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
