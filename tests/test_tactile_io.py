import hashlib
from types import SimpleNamespace

import pytest

from neural_manifolds.revised import tactile_io


def fake_reader(monkeypatch, path, *, samples=10, channels=2):
    closed = []
    raw = SimpleNamespace(
        filenames=[path],
        n_times=samples,
        ch_names=["x"] * channels,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(tactile_io, "read_raw_recording", lambda _: raw)
    return raw, closed


def test_complete_payload_remains_open(tmp_path, monkeypatch):
    path = tmp_path / "eeg.fdt"
    path.write_bytes(bytes(80))
    expected, closed = fake_reader(monkeypatch, path)
    raw, audit = tactile_io.open_tactile_recording(tmp_path / "eeg.set")
    assert raw is expected and not closed
    assert audit["expected_payload_bytes"] == audit["actual_payload_bytes"] == 80


def test_verified_source_truncation_is_distinct_and_closes_file(tmp_path, monkeypatch):
    data = bytes(40)
    key = f"MD5E-s40--{hashlib.md5(data).hexdigest()}.fdt"
    path = tmp_path / key
    path.write_bytes(data)
    _, closed = fake_reader(monkeypatch, path)
    with pytest.raises(tactile_io.IncompleteSourceRecordingError) as exc:
        tactile_io.open_tactile_recording(tmp_path / "eeg.set")
    assert closed
    assert exc.value.audit["source_checksum_verified"]
    assert exc.value.audit["expected_payload_bytes"] == 80
    assert exc.value.audit["actual_payload_bytes"] == 40


@pytest.mark.parametrize(
    "name", ["eeg.fdt", "MD5E-s40--" + "0" * 32 + ".fdt", "MD5E-s80--" + "0" * 32 + ".fdt"]
)
def test_unverified_short_payload_remains_fatal(tmp_path, monkeypatch, name):
    path = tmp_path / name
    path.write_bytes(bytes(40))
    _, closed = fake_reader(monkeypatch, path)
    with pytest.raises(ValueError) as exc:
        tactile_io.open_tactile_recording(tmp_path / "eeg.set")
    assert not isinstance(exc.value, tactile_io.IncompleteSourceRecordingError)
    assert closed


def test_extra_bytes_not_silently_accepted(tmp_path, monkeypatch):
    path = tmp_path / "eeg.fdt"
    path.write_bytes(bytes(88))
    fake_reader(monkeypatch, path)
    with pytest.raises(ValueError, match="larger_than_header"):
        tactile_io.open_tactile_recording(tmp_path / "eeg.set")
