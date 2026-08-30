from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from neural_manifolds.config import load_study
from neural_manifolds.preprocessing import tms as preprocessing_tms
from neural_manifolds.stages import tms as tms_stage


def _test_study() -> Any:
    study = load_study(Path("configs/study.yaml"))
    preprocessing = study.preprocessing.model_copy(
        update={
            "canonical_channels": ["Cz", "Pz"],
            "minimum_canonical_channels": 2,
            "minimum_event_trials_per_condition": 2,
        }
    )
    return study.model_copy(update={"preprocessing": preprocessing})


class _FakeRaw:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self._data = np.zeros((2, 2_000), dtype=float)
        self._data[:, 995:1_016] = 100.0
        self.first_samp = 0
        self.info = {"sfreq": 1_000.0, "bads": []}
        self.ch_names = ["Cz", "Pz"]
        self.annotations: list[Any] = []

    def load_data(self) -> None:
        self._calls.append("load")

    def pick(self, selection: str | list[str]) -> None:
        self._calls.append(f"pick:{selection}")

    def rename_channels(self, rename: dict[str, str]) -> None:
        self.ch_names = [rename[name] for name in self.ch_names]

    def get_data(self) -> np.ndarray:
        self._calls.append("get_continuous_data")
        return self._data.copy()


class _FakeCleanRaw:
    def __init__(
        self,
        data: np.ndarray,
        info: dict[str, Any],
        calls: list[str],
    ) -> None:
        self._data = np.asarray(data)
        self.info = dict(info)
        self.info.setdefault("bads", [])
        self.ch_names = ["Cz", "Pz"]
        self._calls = calls

    def set_annotations(self, annotations: list[Any]) -> None:
        self.annotations = annotations

    def set_montage(self, *_args: Any, **_kwargs: Any) -> None:
        self._calls.append("montage")

    def interpolate_bads(self, *_args: Any, **_kwargs: Any) -> None:
        self._calls.append("interpolate_bad_channels")

    def set_eeg_reference(self, *_args: Any, **_kwargs: Any) -> None:
        self._calls.append("reference")

    def filter(self, *_args: Any, **_kwargs: Any) -> None:
        self._calls.append("filter")

    def notch_filter(self, *_args: Any, **_kwargs: Any) -> None:
        self._calls.append("notch")

    def resample(
        self,
        sampling_hz: float,
        *,
        events: np.ndarray,
        **_kwargs: Any,
    ) -> tuple[_FakeCleanRaw, np.ndarray]:
        self._calls.append("resample")
        self.info["sfreq"] = float(sampling_hz)
        return self, events


class _FakeEpochs:
    def __init__(self, calls: list[str], event_count: int) -> None:
        calls.append("epoch")
        self._data = np.zeros((event_count, 2, 301), dtype=float)
        self.info = {"sfreq": 200.0}
        self.times = np.linspace(-0.5, 1.0, 301)

    def get_data(self, *, copy: bool) -> np.ndarray:
        return self._data.copy() if copy else self._data

    def __len__(self) -> int:
        return len(self._data)


def test_tms_pulse_interpolation_precedes_filter_and_epoch_and_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    raw = _FakeRaw(calls)

    def interpolate_once(
        data: np.ndarray,
        pulse_samples: np.ndarray,
        sampling_hz: float,
        **_kwargs: Any,
    ) -> np.ndarray:
        calls.append("continuous_pulse_interpolation")
        np.testing.assert_array_equal(pulse_samples, np.asarray([1_000, 1_500]))
        assert sampling_hz == 1_000.0
        assert np.max(data[:, 995:1_016]) == 100.0
        return np.zeros_like(data)

    def raw_array(
        data: np.ndarray,
        info: dict[str, Any],
        *,
        first_samp: int,
        verbose: str,
    ) -> _FakeCleanRaw:
        del first_samp, verbose
        calls.append("construct_clean_continuous_raw")
        assert np.max(data) == 0.0
        return _FakeCleanRaw(data, info, calls)

    fake_mne = SimpleNamespace(
        events_from_annotations=lambda *_args, **_kwargs: (
            np.asarray([[1_000, 0, 128], [1_500, 0, 128]], dtype=int),
            {"Response/R128": 128},
        ),
        io=SimpleNamespace(RawArray=raw_array),
        Epochs=lambda _clean, events, **_kwargs: _FakeEpochs(calls, len(events)),
    )
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    monkeypatch.setattr(tms_stage, "read_raw_recording", lambda _path: raw)
    monkeypatch.setattr(tms_stage, "interpolate_continuous_pulses", interpolate_once)
    monkeypatch.setattr(
        preprocessing_tms,
        "interpolate_pulse_interval",
        lambda *_args, **_kwargs: pytest.fail("post-epoch pulse interpolation was repeated"),
    )
    monkeypatch.setattr(
        tms_stage,
        "detect_bad_channels",
        lambda *_args, **_kwargs: SimpleNamespace(bad_indices=np.asarray([], dtype=int)),
    )
    monkeypatch.setattr(
        tms_stage,
        "detect_artifact_windows",
        lambda values, _sampling_hz: SimpleNamespace(keep=np.ones(values.shape[0], dtype=bool)),
    )
    monkeypatch.setattr(tms_stage, "infer_mains_frequency", lambda _raw: 50.0)
    labels = tmp_path / "labels.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "tms-unit",
                "participant_id": "p1",
                "dataset_id": "propofol_tms_eeg",
                "modality": "tms-eeg",
                "condition": "awake",
                "source_path": "unused.vhdr",
            }
        ]
    ).to_parquet(labels, index=False)

    manifest_path, audit_path = tms_stage.build_tms_epoch_manifest(
        cohort_labels=labels,
        output_root=tmp_path / "prepared",
        study=_test_study(),
    )

    assert calls.count("continuous_pulse_interpolation") == 1
    assert calls.index("continuous_pulse_interpolation") < calls.index("filter")
    assert calls.index("continuous_pulse_interpolation") < calls.index("epoch")
    assert calls.index("filter") < calls.index("epoch")
    assert len(pd.read_parquet(manifest_path)) == 1
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["pulse_interpolation_domain"] == ("continuous_eeg_before_filtering_and_epoching")


def _write_epoch_archive(path: Path, marker: float) -> None:
    times = np.linspace(-0.5, 1.0, 9)
    np.savez_compressed(
        path,
        epochs=np.full((2, 3, len(times)), marker, dtype=np.float64),
        times_seconds=times,
    )


def test_tms_linkage_excludes_direct_acquisition_and_correlates_participant_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    passive_baselines = [50.0, -20.0, 100.0, 0.0, 30.0]
    direct_baselines = [-100.0, 80.0, 5.0, 60.0, -40.0]
    manifest_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for index, (passive_base, direct_base) in enumerate(
        zip(passive_baselines, direct_baselines, strict=True), start=1
    ):
        participant = f"p{index}"
        condition_values = {
            "awake": (direct_base + index, passive_base + index),
            "propofol_sedation": (direct_base, passive_base),
        }
        for condition, (direct_value, passive_value) in condition_values.items():
            archive = tmp_path / f"{participant}-{condition}.npz"
            _write_epoch_archive(archive, direct_value)
            manifest_rows.append(
                {
                    "participant_id": participant,
                    "condition": condition,
                    "epochs_path": str(archive),
                }
            )
            # Two passive runs must collapse to one participant-condition predictor.
            profile_rows.extend(
                {
                    "participant_id": participant,
                    "condition": condition,
                    "dataset_id": "propofol_tms_eeg",
                    "modality": "eeg",
                    "acquisition": acquisition,
                    "reachability": value,
                }
                for acquisition, value in (
                    ("rest", passive_value),
                    ("EC", passive_value),
                    ("tms", 10_000.0 + direct_value),
                )
            )
            profile_rows.append(
                {
                    "participant_id": participant,
                    "condition": condition,
                    "dataset_id": "propofol_tms_eeg",
                    "modality": "tms-eeg",
                    "acquisition": "tms",
                    "reachability": -10_000.0 - direct_value,
                }
            )

    manifest_path = tmp_path / "tms-manifest.parquet"
    profiles_path = tmp_path / "profiles.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    pd.DataFrame(profile_rows).to_parquet(profiles_path, index=False)

    def fake_trajectories(
        conditions: dict[str, np.ndarray],
        times: np.ndarray,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for condition, values in conditions.items():
            marker = float(np.mean(values))
            output[condition] = SimpleNamespace(
                marker=marker,
                mean_trajectory=np.column_stack(
                    [np.full(len(times), marker), np.zeros(len(times))]
                ),
                baseline_centroid=np.zeros(2),
            )
        return output

    def fake_outcomes(trajectory: Any, _times: np.ndarray, **_kwargs: Any) -> dict[str, float]:
        marker = float(trajectory.marker)
        return {
            "maximum_displacement": marker,
            "occupied_log_volume": marker,
            "spatial_differentiation": marker,
            "recovery_half_time_seconds": marker,
        }

    monkeypatch.setattr(tms_stage, "fit_shared_perturbational_trajectories", fake_trajectories)
    monkeypatch.setattr(tms_stage, "trajectory_outcomes", fake_outcomes)

    outcomes_path, associations_path, _trajectory_path, audit_path = tms_stage.run_tms_validation(
        tms_manifest=manifest_path,
        profiles_path=profiles_path,
        output_root=tmp_path / "validation",
        study=_test_study(),
    )

    outcomes = pd.read_parquet(outcomes_path)
    assert len(outcomes) == 10
    for index, passive_base in enumerate(passive_baselines, start=1):
        participant = outcomes[outcomes["participant_id"] == f"p{index}"].set_index("condition")
        assert participant.loc["awake", "reachability"] == pytest.approx(passive_base + index)
        assert participant.loc["propofol_sedation", "reachability"] == pytest.approx(passive_base)

    associations = pd.read_parquet(associations_path)
    assert len(associations) == 4
    assert set(associations["n_participants"]) == {5}
    assert np.allclose(associations["estimate"], 1.0)
    assert set(associations["contrast"]) == {"awake_minus_propofol_sedation"}
    assert set(associations["test"]) == {"spearman_participant_level_within_condition_delta"}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["passive_profile_rule"]["acquisition_excluded"] == "tms"
    assert audit["association_unit"] == ("participant_awake_minus_propofol_sedation_delta")
    assert audit["pulse_interpolation_repeated_after_filtering"] is False
