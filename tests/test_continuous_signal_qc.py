import json
import zipfile

import pytest

from neural_manifolds.continuous.signal_qc import materialize_recording, run
from neural_manifolds.provenance import atomic_write_json, sha256_file


def test_one_archive_member_is_bounded_and_cleaned(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    with zipfile.ZipFile(release / "data.zip", "w") as archive:
        archive.writestr("Data/a.edf", b"recording")
    item = {"member": "Data/a.edf", "container": "data.zip", "bytes": 9}
    with materialize_recording(release, item, tmp_path / "scratch") as path:
        assert path.read_bytes() == b"recording"
        assert not path.stat().st_mode & 0o222
        generated = path
    assert not generated.exists()
    with (
        pytest.raises(ValueError, match="Unsafe"),
        materialize_recording(release, {**item, "member": "../outside"}, tmp_path),
    ):
        pass
    with (
        pytest.raises(ValueError, match="size differs"),
        materialize_recording(release, {**item, "bytes": 10}, tmp_path),
    ):
        pass


def test_qc_does_not_consume_report_labels_and_resumes(tmp_path, monkeypatch):
    from neural_manifolds.continuous import signal_qc

    release = tmp_path / "dataset" / "1"
    (release / ".acquisition").mkdir(parents=True)
    (release / ".acquisition/COMPLETE.json").write_text("{}")
    (release / "a.edf").write_bytes(b"synthetic")
    audit = tmp_path / "audit.json"
    atomic_write_json(
        audit,
        {
            "release": str(release),
            "completion_marker_sha256": sha256_file(release / ".acquisition/COMPLETE.json"),
            "files": [{"container": None, "member": "a.edf", "bytes": 9}],
            "records": [{"Experience": "2", "Subject ID": "secret-label"}],
            "issues": [],
        },
    )
    calls = []

    def inspect(row, *, study):
        assert set(row) == {
            "recording_id",
            "dataset_id",
            "source_path",
            "events_path",
            "channels_path",
        }
        calls.append(row)
        return {**row, "technically_eligible": True}, []

    monkeypatch.setattr(signal_qc, "_inspect_recording", inspect)
    monkeypatch.setenv("NM_SCRATCH_ROOT", str(tmp_path / "scratch"))
    run(audit, tmp_path / "qc")
    run(audit, tmp_path / "qc")
    assert len(calls) == 1
    result = json.loads((tmp_path / "qc/signal_qc.json").read_text())
    assert result["labels_consumed"] == []
    assert result["analysis_ready"] is False
    assert "source_path" not in result["recordings"][0]
