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
from neural_manifolds.provenance import sha256_file
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
    source = tmp_path / "unused.vhdr"
    source.write_text("TMS source-lineage fixture\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "unit_id": "tms-unit",
                "participant_id": "p1",
                "dataset_id": "propofol_tms_eeg",
                "modality": "tms-eeg",
                "condition": "awake",
                "source_path": str(source),
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
    manifest = pd.read_parquet(manifest_path).iloc[0]
    assert manifest["source_path"] == str(source.resolve(strict=True))
    assert json.loads(manifest["channel_order_json"]) == ["Cz", "Pz"]
    assert manifest["pulse_trials_total"] == 2
    assert manifest["pulse_trials_retained"] == 2
    assert manifest["pulse_trials_rejected"] == 0
    epochs, _, channels = tms_stage._load_epochs(
        Path(manifest["epochs_path"]),
        expected_sha256=manifest["epochs_sha256"],
        expected_channel_order=("Cz", "Pz"),
    )
    assert epochs.shape[0] == 2
    assert channels == ("Cz", "Pz")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["pulse_interpolation_domain"] == ("continuous_eeg_before_filtering_and_epoching")
    assert audit["general_encoder_status"] == ("omitted_requires_dedicated_tms_preprocessing")
    assert audit["ica_status"] == "not_implemented"
    assert audit["ica_execution_status"] == (
        "not_executed_no_validated_two_pass_component_selection"
    )
    assert audit["ssp_execution_status"] == ("not_executed_no_validated_tms_projector_definition")
    assert audit["epoch_archives_include_channel_order"] is True
    assert audit["raw_lineage_retained"] is True
    channel_provenance = pd.read_parquet(audit["channel_provenance"]["path"])
    trial_provenance = pd.read_parquet(audit["trial_provenance"]["path"])
    trial_channel_qc = pd.read_parquet(audit["trial_channel_early_burden"]["path"])
    assert len(channel_provenance) == 2
    assert set(channel_provenance["canonical_channel_name"]) == {"Cz", "Pz"}
    assert len(trial_provenance) == 2
    assert set(trial_provenance["trial_status"]) == {"retained"}
    assert set(trial_provenance["epoch_constructor_status"]) == {"unavailable_drop_log_not_exposed"}
    assert trial_provenance["retained_archive_index"].tolist() == [0, 1]
    assert len(trial_channel_qc) == 4
    assert set(trial_channel_qc["trial_status"]) == {"retained"}
    assert manifest["auxiliary_channel_status"] == "unavailable_channel_type_metadata"
    assert not bool(manifest["ica_executed"])
    assert not bool(manifest["ssp_executed"])
    assert manifest["early_post_pulse_interval_seconds"] == "[0.02,0.05]"
    assert audit["scientific_gate_applied"] is False


def _write_epoch_archive(
    path: Path,
    marker: float,
    *,
    channels: tuple[str, ...] = ("F3", "Cz", "Pz"),
    trials: int = 2,
) -> str:
    times = np.linspace(-0.5, 1.0, 9)
    np.savez_compressed(
        path,
        epochs=np.full((trials, len(channels), len(times)), marker, dtype=np.float64),
        times_seconds=times,
        channel_names=np.asarray(channels, dtype="U"),
    )
    return sha256_file(path)


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
            archive_sha256 = _write_epoch_archive(archive, direct_value)
            manifest_rows.append(
                {
                    "unit_id": f"{participant}-{condition}",
                    "participant_id": participant,
                    "condition": condition,
                    "epochs_path": str(archive),
                    "epochs_sha256": archive_sha256,
                    "channel_order_json": json.dumps(["F3", "Cz", "Pz"]),
                    "pulse_trials_total": 3,
                    "pulse_trials_retained": 2,
                    "pulse_trials_rejected": 1,
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
    monkeypatch.setattr(
        tms_stage,
        "conventional_tms_eeg_outcomes",
        lambda epochs, _times, **_kwargs: {
            **{outcome: float(np.mean(epochs)) for outcome in tms_stage.CONVENTIONAL_TMS_OUTCOMES},
            "sensor_propagation_status": "available_sensor_level_temporal_spread",
        },
    )

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
    assert len(associations) == len(tms_stage.DIRECT_TRAJECTORY_OUTCOMES) + len(
        tms_stage.CONVENTIONAL_TMS_OUTCOMES
    )
    assert set(associations["outcome"]) == set(tms_stage.DIRECT_TRAJECTORY_OUTCOMES) | set(
        tms_stage.CONVENTIONAL_TMS_OUTCOMES
    )
    assert set(associations["n_participants"]) == {5}
    assert np.allclose(associations["estimate"], 1.0)
    assert set(associations["status"]) == {"available"}
    assert set(associations["contrast"]) == {"awake_minus_propofol_sedation"}
    assert set(associations["test"]) == {"spearman_participant_level_within_condition_delta"}
    assert set(outcomes["source_run_count"]) == {1}
    assert set(outcomes["pulse_trials_total"]) == {3}
    assert set(outcomes["pulse_trials_retained"]) == {2}
    assert set(outcomes["pulse_trials_rejected"]) == {1}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["passive_profile_rule"]["acquisition_excluded"] == "tms"
    assert audit["association_unit"] == ("participant_awake_minus_propofol_sedation_delta")
    assert audit["pulse_interpolation_repeated_after_filtering"] is False
    assert audit["epoch_archive_sha256_verified_before_load"] is True
    assert audit["repeated_condition_runs_concatenated_before_trial_matching"] is True
    assert audit["association_rows_expected"] == len(associations)


def test_tms_epoch_loader_rejects_hash_and_channel_order_tampering(tmp_path: Path) -> None:
    archive = tmp_path / "epochs.npz"
    digest = _write_epoch_archive(archive, 1.0)
    with pytest.raises(ValueError, match="hash mismatch"):
        tms_stage._load_epochs(
            archive,
            expected_sha256="0" * 64,
            expected_channel_order=("F3", "Cz", "Pz"),
        )
    with pytest.raises(ValueError, match="channel order mismatch"):
        tms_stage._load_epochs(
            archive,
            expected_sha256=digest,
            expected_channel_order=("Pz", "Cz", "F3"),
        )


def test_tms_repeated_condition_runs_are_concatenated_before_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_rows: list[dict[str, Any]] = []
    for condition, base in (("awake", 10.0), ("propofol_sedation", 2.0)):
        for run_index, trials in enumerate((2, 3), start=1):
            archive = tmp_path / f"{condition}-run-{run_index}.npz"
            digest = _write_epoch_archive(archive, base + run_index, trials=trials)
            manifest_rows.append(
                {
                    "unit_id": f"p1-{condition}-run-{run_index}",
                    "participant_id": "p1",
                    "condition": condition,
                    "epochs_path": str(archive),
                    "epochs_sha256": digest,
                    "channel_order_json": json.dumps(["F3", "Cz", "Pz"]),
                    "pulse_trials_total": trials + 1,
                    "pulse_trials_retained": trials,
                    "pulse_trials_rejected": 1,
                }
            )
    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    profiles_path = tmp_path / "profiles.parquet"
    pd.DataFrame(
        [
            {
                "participant_id": "p1",
                "condition": condition,
                "dataset_id": "propofol_tms_eeg",
                "modality": "eeg",
                "acquisition": "rest",
                "reachability": value,
            }
            for condition, value in (("awake", 1.0), ("propofol_sedation", 0.5))
        ]
    ).to_parquet(profiles_path, index=False)
    observed_trials: dict[str, int] = {}

    def fake_trajectories(
        conditions: dict[str, np.ndarray],
        times: np.ndarray,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        observed_trials.update({name: value.shape[0] for name, value in conditions.items()})
        return {
            name: SimpleNamespace(
                marker=float(np.mean(values)),
                mean_trajectory=np.zeros((len(times), 2)),
                baseline_centroid=np.zeros(2),
            )
            for name, values in conditions.items()
        }

    monkeypatch.setattr(tms_stage, "fit_shared_perturbational_trajectories", fake_trajectories)
    monkeypatch.setattr(
        tms_stage,
        "trajectory_outcomes",
        lambda trajectory, _times, **_kwargs: {
            "maximum_displacement": trajectory.marker,
            "occupied_log_volume": trajectory.marker,
            "spatial_differentiation": trajectory.marker,
            "recovery_half_time_seconds": trajectory.marker,
        },
    )

    outcomes_path, associations_path, *_ = tms_stage.run_tms_validation(
        tms_manifest=manifest_path,
        profiles_path=profiles_path,
        output_root=tmp_path / "out",
        study=_test_study(),
    )

    assert observed_trials == {"awake": 5, "propofol_sedation": 5}
    outcomes = pd.read_parquet(outcomes_path)
    assert set(outcomes["source_run_count"]) == {2}
    assert set(outcomes["pulse_trials_total"]) == {7}
    assert set(outcomes["pulse_trials_retained"]) == {5}
    assert set(outcomes["pulse_trials_rejected"]) == {2}
    associations = pd.read_parquet(associations_path)
    assert len(associations) == len(tms_stage.DIRECT_TRAJECTORY_OUTCOMES) + len(
        tms_stage.CONVENTIONAL_TMS_OUTCOMES
    )
    assert set(associations["test"]) == {"spearman_participant_level_within_condition_delta"}
    assert set(associations["status"]) == {
        "unavailable_fewer_than_five_complete_participant_deltas"
    }
