from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neural_manifolds.config import load_study
from neural_manifolds.stages import qc


class _FakeRaw:
    def __init__(self, data: np.ndarray, *, sfreq: float = 100.0) -> None:
        self._data = np.asarray(data, dtype=float)
        self.n_times = self._data.shape[1]
        self.ch_names = ["F3", "Cz", "Pz", "EOG"][: self._data.shape[0]]
        self._types = ["eeg", "eeg", "eeg", "eog"][: self._data.shape[0]]
        self.info = {
            "sfreq": sfreq,
            "line_freq": 50.0,
            "custom_ref_applied": "off",
            "projs": [],
            "chs": [
                {"loc": np.r_[np.asarray([index + 1.0, 1.0, 1.0]), np.zeros(9)]}
                for index in range(self._data.shape[0])
            ],
        }
        self.annotations = []
        self.filenames: tuple[str, ...] = ()
        self.closed = False

    def get_channel_types(self) -> list[str]:
        return list(self._types)

    def copy(self) -> _FakeRaw:
        copied = _FakeRaw(self._data.copy(), sfreq=float(self.info["sfreq"]))
        copied.ch_names = list(self.ch_names)
        copied._types = list(self._types)
        copied.info["chs"] = list(self.info["chs"])
        return copied

    def pick(self, selection: str) -> _FakeRaw:
        assert selection == "eeg"
        indices = [index for index, kind in enumerate(self._types) if kind == "eeg"]
        self._data = self._data[indices]
        self.ch_names = [self.ch_names[index] for index in indices]
        self._types = [self._types[index] for index in indices]
        self.info["chs"] = [self.info["chs"][index] for index in indices]
        return self

    def get_data(self, *, start: int = 0, stop: int | None = None) -> np.ndarray:
        return self._data[:, start:stop]

    def close(self) -> None:
        self.closed = True


def _study():
    study = load_study(Path("configs/study.yaml"))
    return study.model_copy(
        update={
            "signal_qc": study.signal_qc.model_copy(
                update={
                    "sample_segments": 3,
                    "seconds_per_segment": 2.0,
                    "diagnostic_window_seconds": 1.0,
                }
            )
        }
    )


def _signal(*, nonfinite: bool = False) -> np.ndarray:
    times = np.arange(1_000, dtype=float) / 100.0
    data = np.vstack(
        [
            10e-6 * np.sin(2 * np.pi * 8 * times),
            12e-6 * np.sin(2 * np.pi * 10 * times + 0.2),
            9e-6 * np.sin(2 * np.pi * 12 * times + 0.4),
            15e-6 * np.sin(2 * np.pi * 1 * times),
        ]
    )
    if nonfinite:
        data[0, 0] = np.nan
    return data


def _inventory_row(tmp_path: Path, recording_id: str) -> dict[str, object]:
    source = tmp_path / f"{recording_id}.edf"
    source.write_bytes(recording_id.encode("utf-8"))
    events = tmp_path / f"{recording_id}_events.tsv"
    events.write_text("onset\tduration\ttrial_type\n1.0\t0.2\tsecret-label\n", encoding="utf-8")
    channels = tmp_path / f"{recording_id}_channels.tsv"
    channels.write_text("name\tstatus\nF3\tgood\nCz\tgood\nPz\tbad\n", encoding="utf-8")
    return {
        "recording_id": recording_id,
        "dataset_id": "synthetic",
        "release_version": "v1",
        "participant_id": f"synthetic:{recording_id}",
        "session": None,
        "task": "rest",
        "acquisition": None,
        "run": None,
        "source_path": str(source),
        "events_path": str(events),
        "channels_path": str(channels),
        "modality": "eeg",
    }


def test_signal_qc_is_label_blind_and_writes_recording_channel_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _inventory_row(tmp_path, "recording-1")
    inventory = tmp_path / "recordings.parquet"
    pd.DataFrame([row]).to_parquet(inventory, index=False)
    raw = _FakeRaw(_signal())
    monkeypatch.setattr(qc, "read_raw_recording", lambda _path: raw)
    original_read_csv = pd.read_csv
    event_usecols: list[object] = []

    def tracked_read_csv(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if Path(path) == Path(str(row["events_path"])):
            event_usecols.append(kwargs.get("usecols"))
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(qc.pd, "read_csv", tracked_read_csv)

    recording_path, channel_path, audit_path = qc.run_signal_qc(
        inventory_path=inventory,
        output_root=tmp_path / "qc",
        study=_study(),
    )

    recording = pd.read_parquet(recording_path).iloc[0]
    assert bool(recording["technically_eligible"]) is True
    assert recording["eeg_channel_count"] == 3
    assert recording["eog_channel_count"] == 1
    assert recording["event_rows"] == 1
    assert json.loads(recording["event_value_columns_consumed_json"]) == [
        "onset",
        "duration",
    ]
    assert event_usecols == [None, ["onset", "duration"]]
    assert "secret-label" not in recording_path.read_bytes().decode("latin-1")
    channels = pd.read_parquet(channel_path)
    assert channels["canonical_channel_name"].tolist() == ["F3", "Cz", "Pz"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["label_fields_consumed"] == []
    assert audit["condition_or_outcome_values_written"] is False
    assert audit["review_flags_are_exclusions"] is False
    assert audit["scientific_gate_applied"] is False
    assert raw.closed is True


def test_signal_qc_preserves_technical_exclusion_without_losing_valid_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_inventory_row(tmp_path, name) for name in ("valid", "nonfinite")]
    inventory = tmp_path / "recordings.parquet"
    pd.DataFrame(rows).to_parquet(inventory, index=False)
    raws = {
        "valid.edf": _FakeRaw(_signal()),
        "nonfinite.edf": _FakeRaw(_signal(nonfinite=True)),
    }
    monkeypatch.setattr(qc, "read_raw_recording", lambda path: raws[Path(path).name])

    recording_path, _channel_path, audit_path = qc.run_signal_qc(
        inventory_path=inventory,
        output_root=tmp_path / "qc",
        study=_study(),
    )

    flow = pd.read_parquet(recording_path).set_index("recording_id")
    assert bool(flow.loc["valid", "technically_eligible"]) is True
    assert bool(flow.loc["nonfinite", "technically_eligible"]) is False
    assert flow.loc["nonfinite", "qc_status"] == "excluded_technical"
    assert "non-finite" in flow.loc["nonfinite", "technical_exclusion_reason"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["recordings_technically_eligible"] == 1
    assert audit["recordings_technically_excluded"] == 1


def test_signal_qc_rejects_label_columns_before_reading_signal(tmp_path: Path) -> None:
    row = {**_inventory_row(tmp_path, "recording"), "condition": "secret"}
    inventory = tmp_path / "recordings.parquet"
    pd.DataFrame([row]).to_parquet(inventory, index=False)
    with pytest.raises(ValueError, match="forbidden fields"):
        qc.run_signal_qc(
            inventory_path=inventory,
            output_root=tmp_path / "qc",
            study=_study(),
        )
