import json
from pathlib import Path

from neural_manifolds.provenance import atomic_write_json, sha256_file


def test_atomic_json_and_checksum(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "record.json"
    atomic_write_json(target, {"phase": "audit", "ok": True})
    assert json.loads(target.read_text(encoding="utf-8"))["ok"] is True
    assert len(sha256_file(target)) == 64
