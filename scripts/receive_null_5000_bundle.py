"""Extract a checksum-verified final null report and aggregate statistics from SSH log."""

import argparse
import base64
import hashlib
import io
import re
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    matches = re.findall(
        r"^NM_FINAL_NULL_BUNDLE ([0-9a-f]{64}) ([A-Za-z0-9+/=]+)$",
        args.log.read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one complete transfer; found {len(matches)}")
    expected, encoded = matches[0]
    payload = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Transfer hash mismatch")
    allowed = {"null_review_5000.json", "null_refit_statistics_5000.csv"}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if set(archive.namelist()) != allowed:
            raise ValueError("Unexpected archive members")
        for item in archive.infolist():
            if item.file_size > 60_000_000:
                raise ValueError("Unexpectedly large artifact")
        args.destination.mkdir(parents=True)
        for name in sorted(allowed):
            (args.destination / name).write_bytes(archive.read(name))
    print(f"Verified {len(allowed)} files; zip sha256={expected}")


if __name__ == "__main__":
    main()
