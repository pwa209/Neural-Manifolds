"""Explicit QC-only compatibility copies; never rewrite sealed raw recordings."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from neural_manifolds.provenance import sha256_file


def invalid_clock(source: Path) -> bool:
    if source.suffix.lower() != ".edf":
        return False
    with source.open("rb") as stream:
        header = stream.read(256)
    if len(header) != 256:
        return False
    value = header[176:184]
    try:
        hour, minute, second = (int(v) for v in value.split(b"."))
    except ValueError:
        return False
    # This narrow defect was observed in the source. Do not guess other clocks.
    return 0 <= hour < 24 and 0 <= minute < 60 and second == 60


def body_sha256(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        stream.seek(256)
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def qc_clock_copy(source: Path, scratch: Path):
    """Replace an invalid clock with a parser placeholder, NOT a recovered time.

    Every byte outside the eight clock bytes remains unchanged. The caller must
    clear measurement date and mark absolute clock unavailable before inspection.
    This is only for signal diagnostics, never absolute-time event alignment.
    """
    if not invalid_clock(source):
        yield source, None
        return
    scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch).free < source.stat().st_size + 2 * 1024**3:
        raise ValueError("Insufficient scratch for QC compatibility copy")
    with tempfile.TemporaryDirectory(prefix="nm-edf-clock-", dir=scratch) as temporary:
        target = Path(temporary) / "recording.edf"
        shutil.copyfile(source, target)
        with source.open("rb") as stream:
            original_header = stream.read(256)
        with target.open("r+b") as stream:
            stream.seek(176)
            stream.write(b"00.00.00")
        with target.open("rb") as stream:
            fixed_header = stream.read(256)
        if (
            original_header[:176] != fixed_header[:176]
            or original_header[184:] != fixed_header[184:]
            or body_sha256(source) != body_sha256(target)
        ):
            raise ValueError("QC clock copy changed bytes outside the clock field")
        provenance = {
            "operation": "qc_only_parser_clock_placeholder",
            "source_sha256": sha256_file(source),
            "qc_copy_sha256": sha256_file(target),
            "changed_byte_offsets": [176, 184],
            "original_clock": original_header[176:184].decode("ascii"),
            "placeholder_clock": "00.00.00",
            "absolute_clock_available": False,
            "all_other_bytes_verified_identical": True,
            "permitted_use": "sampled_signal_qc_only_not_absolute_time_alignment",
        }
        target.chmod(0o444)
        try:
            yield target, provenance
        finally:
            target.chmod(0o600)
