from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_manifolds.recording_provenance import recording_inventory, recording_members


def _brainvision_recording(root: Path) -> tuple[Path, Path, Path]:
    header = root / "recording.vhdr"
    marker = root / "recording.vmrk"
    signal = root / "recording.eeg"
    signal.write_bytes(b"signal-v1")
    marker.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\nDataFile=recording.eeg\n",
        encoding="utf-8",
    )
    header.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "DataFile=recording.eeg\n"
        "MarkerFile=recording.vmrk\n",
        encoding="utf-8",
    )
    return header, marker, signal


def test_brainvision_inventory_binds_header_marker_and_signal(tmp_path: Path) -> None:
    header, marker, signal = _brainvision_recording(tmp_path)
    first = recording_inventory(header)
    assert first["file_count"] == 3
    assert {Path(row["path"]) for row in first["files"]} == {
        header.resolve(),
        marker.resolve(),
        signal.resolve(),
    }

    signal.write_bytes(b"signal-v2")
    second = recording_inventory(header)
    assert second["combined_sha256"] != first["combined_sha256"]


def test_brainvision_inventory_rejects_missing_or_escaping_companions(tmp_path: Path) -> None:
    header, marker, signal = _brainvision_recording(tmp_path)
    signal.unlink()
    with pytest.raises(ValueError, match="missing or not a regular file"):
        recording_inventory(header)

    outside = tmp_path.parent / "outside.eeg"
    outside.write_bytes(b"outside")
    marker.write_text("DataFile=../outside.eeg\n", encoding="utf-8")
    signal.write_bytes(b"restored")
    with pytest.raises(ValueError, match="unsafe recording companion"):
        recording_inventory(header)


def test_mne_reported_and_eeglab_external_members_are_included(tmp_path: Path) -> None:
    eeglab = tmp_path / "recording.set"
    external = tmp_path / "recording.fdt"
    auxiliary = tmp_path / "auxiliary.bin"
    eeglab.write_bytes(b"set")
    external.write_bytes(b"fdt")
    auxiliary.write_bytes(b"aux")

    members = recording_members(
        eeglab,
        raw=SimpleNamespace(filenames=(str(eeglab), str(auxiliary))),
    )
    assert set(members) == {eeglab.resolve(), external.resolve(), auxiliary.resolve()}
