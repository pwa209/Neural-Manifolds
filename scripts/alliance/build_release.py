"""Package source only; data, credentials and operational records are never included."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORIES = ("src", "workflow", "scripts", "configs", "tests", "docs", "requirements")
SUFFIXES = {".py", ".sh", ".sbatch", ".yaml", ".yml", ".json", ".md", ".txt", ".lock", ".toml"}


def build() -> dict:
    files = [ROOT / n for n in ("pyproject.toml", "README.md", "STATUS.md", ".gitignore")]
    for folder in DIRECTORIES:
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
                continue
            if path.suffix not in SUFFIXES:
                continue
            if (
                path.parent.name == "configs"
                and path.name.startswith("server")
                and path.name != "server.yaml"
            ):
                continue
            files.append(path)
    contents = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(files)}
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in contents.items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source_digest": digest,
        "files": hashes,
        "git_base": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "identity": "source_digest_includes_uncommitted_revision",
    }
    output = ROOT / "work" / "alliance-deploy" / f"source-{digest}.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    contents["release_manifest.json"] = json.dumps(manifest, indent=2).encode()
    with tarfile.open(output, "w:gz") as archive:
        for name, payload in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    return {
        "archive": str(output),
        "source_digest": digest,
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "files": len(hashes),
    }


if __name__ == "__main__":
    print(json.dumps(build()))
