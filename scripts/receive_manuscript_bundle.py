"""Receive generated figure artifacts through an already-authenticated SSH log."""

import argparse
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath


def receive(log, version, destination):
    if destination.exists():
        raise FileExistsError("Preserve historical figure deliveries")
    pattern = rf"^NM_FIG_BUNDLE {re.escape(version)} ([0-9a-f]{{64}}) ([A-Za-z0-9+/=]+)$"
    matches = re.findall(pattern, log.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected one complete bundle marker, found {len(matches)}")
    checksum, encoded = matches[0]
    payload = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("Transfer checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or ":" in item.filename
                or "\\" in item.filename
            ):
                raise ValueError("Unsafe archive path")
            if item.file_size > 30_000_000:
                raise ValueError("Unexpectedly large figure artifact")
        manifest = json.loads(archive.read("manifest.json"))
        for record in manifest["outputs"]:
            raw = archive.read(record["file"])
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {record['file']}")
        destination.mkdir(parents=True)
        archive.extractall(destination)
    receipt = {
        "version": version,
        "bundle_sha256": checksum,
        "bytes": len(payload),
        "verified_artifacts": len(manifest["outputs"]),
    }
    (destination / "transfer_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    receive(args.log, args.version, args.destination)
