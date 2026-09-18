import errno
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "supervisor", Path(__file__).resolve().parents[1] / "scripts/alliance/supervise_controller.py"
)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


@pytest.mark.parametrize("prefix", ["", "/project/example "])
def test_quota_headroom(prefix):
    assert supervisor.quota_headroom(
        "header\n" + prefix + "426146703 1000000000 1000000000 - 475313* 500000 500000 -\n"
    ) == (573853297, 24687)


def test_quota_parse_fails_closed():
    with pytest.raises(ValueError):
        supervisor.quota_headroom("quota service unavailable")


def test_health_survives_full_project_quota(tmp_path, monkeypatch):
    project, scratch = tmp_path / "project.json", tmp_path / "scratch.json"
    real = supervisor.atomic_write_json

    def write(path, health):
        if path == project:
            raise OSError(errno.EDQUOT, "quota exceeded")
        real(path, health)

    monkeypatch.setattr(supervisor, "atomic_write_json", write)
    supervisor.publish_health(project, scratch, {"status": "waiting_for_quota_recovery"})
    assert scratch.is_file()
    assert not project.exists()
    assert supervisor.is_capacity_error(OSError(errno.EDQUOT, "quota"))
    assert supervisor.is_capacity_error(OSError(errno.ENOSPC, "space"))
    assert not supervisor.is_capacity_error(ValueError("bad science input"))


def test_repaired_task_uses_its_own_source_and_restores_controller_cwd(tmp_path, monkeypatch):
    original = tmp_path / ("a" * 64)
    repaired = tmp_path / ("b" * 64)
    original.mkdir()
    repaired.mkdir()
    (repaired / "release_manifest.json").write_text(json.dumps({"source_digest": repaired.name}))
    task = tmp_path / "spec.json"
    task.write_text(json.dumps({"source_digest": repaired.name}))
    monkeypatch.chdir(original)

    def submit(self, spec, name, root, account):
        assert Path.cwd() == repaired
        raise RuntimeError("uncertain scheduler response")

    monkeypatch.setattr(supervisor.Slurm, "submit", submit)
    with pytest.raises(RuntimeError):
        supervisor.ReleaseScheduler().submit(task, "test", tmp_path, "test")
    assert Path.cwd() == original
