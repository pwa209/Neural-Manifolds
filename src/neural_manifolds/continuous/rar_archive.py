"""Read RAR members through the cluster's 7-Zip, never extract archive paths."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import zlib
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from neural_manifolds.provenance import sha256_file


def backend() -> str:
    configured = os.environ.get("NEURAL_MANIFOLDS_7ZIP")
    if configured:
        expected = os.environ.get("NEURAL_MANIFOLDS_7ZIP_SHA256")
        if not expected or sha256_file(configured) != expected:
            raise ValueError("Pinned archive decoder checksum missing or mismatched")
        return str(Path(configured).resolve(strict=True))
    program = shutil.which("7zz") or shutil.which("7z")
    if not program:
        raise RuntimeError("RAR requires the qualified cluster 7z/7zz executable")
    return program


def safe_name(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return (
        bool(path.parts)
        and not path.is_absolute()
        and ".." not in path.parts
        and ":" not in name
        and not name.startswith("@")
        and all(ord(c) >= 32 for c in name)
    )


def parse_listing(text: str) -> list[dict]:
    entries, seen = [], set()
    for block in text.replace("\r\n", "\n").strip("\n").split("\n\n"):
        fields = {}
        for line in block.splitlines():
            if " = " not in line:
                raise ValueError("Unexpected 7-Zip technical listing format")
            key, value = line.split(" = ", 1)
            if key in fields:
                raise ValueError("Duplicate archive metadata field")
            fields[key] = value
        name = fields.get("Path", "")
        if not safe_name(name) or name in seen:
            raise ValueError("Unsafe or duplicate archive member")
        seen.add(name)
        if any(value for key, value in fields.items() if "Link" in key):
            raise ValueError("Archive links are not supported")
        if fields.get("Encrypted") == "+":
            raise ValueError("Encrypted archive members are not open-access inputs")
        if fields.get("Folder") == "+" or fields.get("Attributes", "").startswith("D"):
            continue
        size = int(fields["Size"])
        if size < 0:
            raise ValueError("Invalid archive member size")
        crc = fields.get("CRC", "")
        if len(crc) != 8 or any(c not in "0123456789ABCDEFabcdef" for c in crc):
            raise ValueError("RAR member lacks a supported CRC32 integrity field")
        entries.append({"member": name, "bytes": size, "crc32": crc})
    if not entries:
        raise ValueError("Archive contains no supported file entries")
    return entries


def members(path: Path) -> list[dict]:
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as error:
        result = subprocess.run(
            [backend(), "l", "-slt", "-ba", "-sccUTF-8", "--", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=error,
            timeout=60,
        )
        error.seek(0)
        if result.returncode:
            raise ValueError("RAR listing failed: " + error.read(2048).decode(errors="replace"))
        if out.tell() > 32 * 1024**2:
            raise ValueError("RAR metadata exceeds bounded listing size")
        out.seek(0)
        return parse_listing(out.read().decode("utf-8"))


@contextmanager
def member_stream(path: Path, name: str):
    """Caller must consume the member; nonzero exit (including CRC error) fails."""
    if not safe_name(name) or any(c in name for c in "*?[]"):
        raise ValueError("Unsafe archive member")
    # The cluster's p7zip can list RAR5 but was built without its decoder.
    # libarchive supports that decoder. Never let its pattern matcher select
    # multiple members: wildcard names are rejected, and inventory rejects duplicates.
    tar = None if os.environ.get("NEURAL_MANIFOLDS_7ZIP") else shutil.which("bsdtar")
    command = (
        [tar, "-xOf", str(path), "--", name]
        if tar
        else [backend(), "x", "-so", "-bd", "-y", "-spd", "-ssc", "-bso0", "--", str(path), name]
    )
    with tempfile.TemporaryFile() as error:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=error,
        )
        watchdog = threading.Timer(600, process.kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            yield process.stdout
            process.stdout.close()
            code = process.wait(timeout=10)
            if code:
                error.seek(0)
                raise ValueError(
                    f"RAR extraction/integrity failure ({code}): "
                    + error.read(2048).decode(errors="replace")
                )
        finally:
            watchdog.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


def metadata_member(path: Path, item: dict) -> bytes:
    size = item["bytes"]
    if size > 16 * 1024**2:
        raise ValueError("Archive metadata member exceeds size limit")
    with member_stream(path, item["member"]) as stream:
        payload = stream.read(size + 1)
    if len(payload) != size:
        raise ValueError("Archive metadata member size differs from inventory")
    if f"{zlib.crc32(payload):08X}" != item["crc32"].upper():
        raise ValueError("Archive metadata member CRC32 mismatch")
    return payload
