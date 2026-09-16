"""Verify the exact source snapshot before installation and each cluster job."""

import hashlib
import json
from pathlib import Path


def verify(root: Path) -> str:
    root = root.resolve()
    d = json.loads((root / "release_manifest.json").read_text())
    identity = hashlib.sha256(json.dumps(d["files"], sort_keys=True).encode()).hexdigest()
    if identity != d["source_digest"]:
        raise ValueError("Source manifest identity mismatch")
    for name, expected in d["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.is_symlink():
            raise ValueError("Unsafe source path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source mismatch: {name}")
    return identity


if __name__ == "__main__":
    print(verify(Path(__file__).resolve().parents[2]))
