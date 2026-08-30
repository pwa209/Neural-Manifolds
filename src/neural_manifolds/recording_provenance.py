"""Content-bound inventories for single- and multi-file electrophysiology recordings."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from neural_manifolds.provenance import sha256_file

_BRAINVISION_REFERENCE = re.compile(
    r"^\s*(DataFile|MarkerFile)\s*=\s*(.*?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _safe_companion(*, declaring_file: Path, boundary: Path, declared: str) -> Path:
    value = declared.strip().strip('"').strip("'")
    if not value:
        raise ValueError(f"empty recording companion declared by {declaring_file}")
    relative = Path(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe recording companion declared by {declaring_file}: {value!r}")
    candidate = declaring_file.parent.joinpath(relative)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"recording companion is missing or not a regular file: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(boundary):
        raise ValueError(f"recording companion escapes its source directory: {candidate}")
    return resolved


def _brainvision_members(primary: Path) -> set[Path]:
    pending = [primary]
    observed: set[Path] = set()
    primary_reference_keys: set[str] = set()
    boundary = primary.parent.resolve(strict=True)
    while pending:
        current = pending.pop()
        if current in observed:
            continue
        observed.add(current)
        if current.suffix.lower() not in {".vhdr", ".vmrk"}:
            continue
        try:
            text = current.read_text(encoding="utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            text = current.read_text(encoding="latin-1", errors="strict")
        references = _BRAINVISION_REFERENCE.findall(text)
        if current == primary and current.suffix.lower() == ".vhdr":
            primary_reference_keys.update(key.casefold() for key, _ in references)
        for _, declared in references:
            companion = _safe_companion(
                declaring_file=current,
                boundary=boundary,
                declared=declared,
            )
            if companion not in observed:
                pending.append(companion)
    if primary.suffix.lower() == ".vhdr" and primary_reference_keys != {
        "datafile",
        "markerfile",
    }:
        raise ValueError(
            f"BrainVision header does not declare both DataFile and MarkerFile: {primary}"
        )
    return observed


def recording_members(primary_path: str | Path, *, raw: Any | None = None) -> tuple[Path, ...]:
    """Enumerate every regular file that physically backs one recording.

    BrainVision header/marker references are parsed directly. MNE-reported filenames
    are additionally accepted when a header object is already available. EEGLAB's
    conventional external ``.fdt`` companion is included when present.
    """

    primary = Path(primary_path)
    if primary.is_symlink() or not primary.is_file():
        raise ValueError(f"recording source is missing or not a regular file: {primary}")
    primary = primary.resolve(strict=True)
    members = (
        _brainvision_members(primary) if primary.suffix.lower() in {".vhdr", ".vmrk"} else {primary}
    )
    if primary.suffix.lower() == ".set":
        external = primary.with_suffix(".fdt")
        if external.exists():
            if external.is_symlink() or not external.is_file():
                raise ValueError(f"EEGLAB companion is not a regular file: {external}")
            members.add(external.resolve(strict=True))
    filenames = getattr(raw, "filenames", ()) if raw is not None else ()
    for value in filenames or ():
        if value is None:
            continue
        candidate = Path(str(value))
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"MNE-reported recording member is not a regular file: {candidate}")
        members.add(candidate.resolve(strict=True))
    return tuple(sorted(members, key=lambda path: str(path).casefold()))


def recording_inventory(primary_path: str | Path, *, raw: Any | None = None) -> dict[str, Any]:
    """Return a deterministic SHA-256 inventory and combined content identity."""

    primary = Path(primary_path).resolve(strict=True)
    files = []
    for member in recording_members(primary, raw=raw):
        files.append(
            {
                "path": str(member),
                "role": "primary" if member == primary else "companion",
                "size": member.stat().st_size,
                "sha256": sha256_file(member),
            }
        )
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "schema_version": 1,
        "primary_path": str(primary),
        "file_count": len(files),
        "files": files,
        "combined_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
