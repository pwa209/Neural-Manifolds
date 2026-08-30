import json
from pathlib import Path

from neural_manifolds.inventory import scan_recordings


def test_scan_recordings_respects_complete_releases(tmp_path: Path) -> None:
    release = tmp_path / "example" / "1.0.0"
    (release / ".acquisition").mkdir(parents=True)
    (release / ".acquisition" / "COMPLETE.json").write_text(
        json.dumps({"ok": True}), encoding="utf-8"
    )
    eeg = release / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    source = eeg / "sub-01_task-rest_eeg.vhdr"
    source.write_text("Brain Vision Data Exchange Header File Version 1.0", encoding="utf-8")
    records = scan_recordings(tmp_path)
    assert len(records) == 1
    assert records[0].participant_id == "example:sub-01"
    assert records[0].task == "rest"
