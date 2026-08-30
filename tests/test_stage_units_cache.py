from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from neural_manifolds import stage_units
from neural_manifolds.config import StudyConfig, load_study
from neural_manifolds.preprocessing.eeg import (
    NATIVE_AVERAGE_REFERENCE_BRANCH,
    NATIVE_CSD_BRANCH,
    SLEEP_HIGHPASS_BRANCH,
    SensitivityBranchResult,
)
from neural_manifolds.provenance import sha256_file
from neural_manifolds.stage_units import (
    DerivativeIntegrityError,
    encode_analysis_units,
    preprocess_analysis_units,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeRaw:
    def __init__(self, payload: bytes, *, n_times: int = 20_000) -> None:
        self.payload = bytes(payload)
        self.n_times = int(n_times)
        self.info = {"sfreq": 200.0}
        self.ch_names = list(
            load_study(ROOT / "configs" / "study.yaml").preprocessing.canonical_channels
        )
        self.times = np.arange(self.n_times, dtype=float) / self.info["sfreq"]

    def copy(self) -> FakeRaw:
        return FakeRaw(self.payload, n_times=self.n_times)

    def crop(self, *, tmin: float, tmax: float, include_tmax: bool = True) -> FakeRaw:
        del include_tmax
        samples = max(1, round((tmax - tmin) * self.info["sfreq"]) + 1)
        return FakeRaw(self.payload, n_times=samples)

    def load_data(self) -> FakeRaw:
        return self

    def save(self, path: str | Path, *, overwrite: bool, verbose: str) -> None:
        del overwrite, verbose
        Path(path).write_bytes(
            b"fake-fif\0" + self.payload + b"\0" + str(self.n_times).encode("ascii")
        )


def _study() -> StudyConfig:
    study = load_study(ROOT / "configs" / "study.yaml")
    return study.model_copy(
        update={
            "preprocessing": study.preprocessing.model_copy(update={"minimum_valid_windows": 2})
        }
    )


def _install_fake_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    counters = {"preprocess": 0, "sensitivities": 0}

    def read_fif(path: str | Path, **kwargs: Any) -> FakeRaw:
        del kwargs
        return FakeRaw(Path(path).read_bytes())

    monkeypatch.setitem(
        sys.modules,
        "mne",
        SimpleNamespace(io=SimpleNamespace(read_raw_fif=read_fif)),
    )
    monkeypatch.setattr(
        stage_units,
        "read_raw_recording",
        lambda path: FakeRaw(Path(path).read_bytes()),
    )

    def preprocess(raw: FakeRaw, **kwargs: Any) -> tuple[FakeRaw, dict[str, Any]]:
        del kwargs
        counters["preprocess"] += 1
        return raw.copy(), {
            "backend": "synthetic",
            "parameters": "fixed",
            "auxiliary_channel_audit": {
                "metadata_status": "available",
                "channels": {"eog": ["EOG"], "ecg": [], "emg": []},
                "ica_support_status": "available_eog_or_ecg_reference",
                "ica_status": "not_performed_policy_report_only_with_auxiliary_support",
                "auxiliary_artifact_control_support_status": ("available_eog_ecg_or_emg_reference"),
                "auxiliary_artifact_control_status": (
                    "not_performed_policy_report_only_with_auxiliary_support"
                ),
                "auxiliary_channels_used_for_cleaning": False,
            },
        }

    def sensitivities(raw: FakeRaw, **kwargs: Any) -> dict[str, SensitivityBranchResult]:
        counters["sensitivities"] += 1
        sleep = bool(kwargs["is_sleep_recording"])
        return {
            NATIVE_AVERAGE_REFERENCE_BRANCH: SensitivityBranchResult(
                raw=raw.copy(),
                status="available",
                reason=None,
                metadata={"branch": NATIVE_AVERAGE_REFERENCE_BRANCH, "reference": "average"},
            ),
            NATIVE_CSD_BRANCH: SensitivityBranchResult(
                raw=None,
                status="unavailable",
                reason="insufficient_montage_positions",
                metadata={"branch": NATIVE_CSD_BRANCH, "position_fraction": 0.0},
            ),
            SLEEP_HIGHPASS_BRANCH: SensitivityBranchResult(
                raw=raw.copy() if sleep else None,
                status="available" if sleep else "not_applicable",
                reason=None if sleep else "unit_modality_not_configured_as_sleep",
                metadata={
                    "branch": SLEEP_HIGHPASS_BRANCH,
                    "configured_highpass_hz": kwargs["sleep_highpass_hz"],
                },
            ),
        }

    monkeypatch.setattr(stage_units, "preprocess_mne_raw", preprocess)
    monkeypatch.setattr(stage_units, "preprocess_mne_sensitivity_branches", sensitivities)
    monkeypatch.setattr(stage_units, "infer_mains_frequency", lambda raw: 50.0)
    return counters


def _preprocessed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[StudyConfig, Path, Path, Path, dict[str, int]]:
    study = _study()
    counters = _install_fake_preprocessing(monkeypatch)
    source = tmp_path / "source.edf"
    source.write_bytes(b"immutable-source-v1")
    inputs = tmp_path / "encoder-inputs.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "unit-1",
                "source_path": str(source),
                "modality": "eeg",
                "selector_json": json.dumps({"kind": "full_recording"}),
            }
        ]
    ).to_parquet(inputs, index=False)
    output = tmp_path / "preprocess"
    manifest, _flow = preprocess_analysis_units(
        encoder_inputs=inputs,
        output_root=output,
        study=study,
    )
    return study, source, inputs, manifest, counters


def _install_fake_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counters = {"encode_tracks": 0, "encoder_constructions": 0}
    monkeypatch.setattr(
        stage_units,
        "_model_environment",
        lambda: (tmp_path / "model", tmp_path / "checkpoint.pth", "f" * 64),
    )
    monkeypatch.setattr(
        stage_units,
        "_model_fingerprint",
        lambda *args: {"source_sha256": "e" * 64, "checkpoint_sha256": "f" * 64},
    )

    def encoder_factory(**kwargs: Any) -> object:
        del kwargs
        counters["encoder_constructions"] += 1
        return object()

    monkeypatch.setattr(stage_units, "OfficialLaBraMEncoder", encoder_factory)

    def encode_track(
        raw: FakeRaw,
        encoder: object,
        *,
        window_seconds: float,
        step_seconds: float,
    ) -> tuple[SimpleNamespace, np.ndarray, SimpleNamespace]:
        del raw, encoder, window_seconds, step_seconds
        counters["encode_tracks"] += 1
        starts = np.arange(6, dtype=np.int64) * 200
        states = np.arange(18, dtype=np.float32).reshape(6, 3)
        encoded = SimpleNamespace(
            global_states=states,
            regional_states={"anterior": states, "posterior": states + 1},
        )
        return encoded, starts, SimpleNamespace(keep=np.ones(6, dtype=bool))

    monkeypatch.setattr(stage_units, "_encode_windows", encode_track)
    return counters


def _encoded_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[StudyConfig, Path, Path, Path, dict[str, int]]:
    study, _source, _inputs, preprocessing_manifest, _preprocess_counters = _preprocessed_run(
        tmp_path, monkeypatch
    )
    counters = _install_fake_encoding(tmp_path, monkeypatch)
    labels = tmp_path / "labels.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "unit-1",
                "participant_id": "sub-1",
                "dataset_id": "synthetic",
                "condition": "wake",
                "selector_json": json.dumps({"kind": "full_recording"}),
            }
        ]
    ).to_parquet(labels, index=False)
    output = tmp_path / "encode"
    manifest, _flow = encode_analysis_units(
        preprocessing_manifest=preprocessing_manifest,
        labels_manifest=labels,
        output_root=output,
        study=study,
    )
    return study, preprocessing_manifest, labels, manifest, counters


def test_preprocessing_reuse_is_granular_and_source_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, source, inputs, manifest, counters = _preprocessed_run(tmp_path, monkeypatch)
    first_calls = counters["preprocess"]
    second_manifest, second_flow = preprocess_analysis_units(
        encoder_inputs=inputs,
        output_root=manifest.parent,
        study=study,
    )
    assert second_manifest == manifest
    assert counters["preprocess"] == first_calls
    flow = json.loads(second_flow.read_text(encoding="utf-8"))
    assert flow["reused_unit_derivatives"] == 1
    assert flow["generated_unit_derivatives"] == 0

    source.write_bytes(b"mutated-source")
    with pytest.raises(DerivativeIntegrityError, match="input fingerprint changed"):
        preprocess_analysis_units(
            encoder_inputs=inputs,
            output_root=manifest.parent,
            study=study,
        )


def test_preprocessing_reuse_is_bound_to_brainvision_signal_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study()
    _install_fake_preprocessing(monkeypatch)
    source = tmp_path / "source.vhdr"
    marker = tmp_path / "source.vmrk"
    signal = tmp_path / "source.eeg"
    signal.write_bytes(b"signal-v1")
    marker.write_text("DataFile=source.eeg\n", encoding="utf-8")
    source.write_text(
        "DataFile=source.eeg\nMarkerFile=source.vmrk\n",
        encoding="utf-8",
    )
    inputs = tmp_path / "encoder-inputs.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "unit-1",
                "source_path": str(source),
                "modality": "eeg",
                "selector_json": json.dumps({"kind": "full_recording"}),
            }
        ]
    ).to_parquet(inputs, index=False)
    output = tmp_path / "preprocess"
    preprocess_analysis_units(encoder_inputs=inputs, output_root=output, study=study)

    signal.write_bytes(b"signal-v2")
    with pytest.raises(DerivativeIntegrityError, match="input fingerprint changed"):
        preprocess_analysis_units(encoder_inputs=inputs, output_root=output, study=study)


def test_preprocessing_enforces_and_binds_label_blind_qc_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study()
    _install_fake_preprocessing(monkeypatch)
    source = tmp_path / "source.edf"
    source.write_bytes(b"source")
    inputs = tmp_path / "inputs.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "unit-1",
                "source_path": str(source),
                "modality": "eeg",
                "selector_json": json.dumps({"kind": "full_recording"}),
            }
        ]
    ).to_parquet(inputs, index=False)
    qc_flow = tmp_path / "recording-flow.parquet"
    pd.DataFrame(
        [
            {
                "source_path": str(source),
                "technically_eligible": True,
                "qc_status": "eligible_with_blind_review_flags",
                "technical_exclusion_reason": None,
                "review_flags_json": '["montage_position_fraction"]',
            }
        ]
    ).to_parquet(qc_flow, index=False)
    output = tmp_path / "preprocess"
    manifest, flow_path = preprocess_analysis_units(
        encoder_inputs=inputs,
        output_root=output,
        study=study,
        qc_recordings=qc_flow,
    )
    row = pd.read_parquet(manifest).iloc[0]
    assert row["qc_status"] == "eligible_with_blind_review_flags"
    assert bool(row["eligible"]) is True
    flow = json.loads(flow_path.read_text(encoding="utf-8"))
    assert flow["qc_flow_enforced"] is True

    qc_frame = pd.read_parquet(qc_flow)
    qc_frame.loc[0, "review_flags_json"] = "[]"
    qc_frame.to_parquet(qc_flow, index=False)
    with pytest.raises(DerivativeIntegrityError, match="input fingerprint changed"):
        preprocess_analysis_units(
            encoder_inputs=inputs,
            output_root=output,
            study=study,
            qc_recordings=qc_flow,
        )

    excluded_output = tmp_path / "excluded"
    qc_frame.loc[0, "technically_eligible"] = False
    qc_frame.loc[0, "qc_status"] = "excluded_technical"
    qc_frame.loc[0, "technical_exclusion_reason"] = "non-finite signal"
    qc_frame.to_parquet(qc_flow, index=False)
    with pytest.raises(RuntimeError, match="all analysis units failed preprocessing"):
        preprocess_analysis_units(
            encoder_inputs=inputs,
            output_root=excluded_output,
            study=study,
            qc_recordings=qc_flow,
        )
    excluded = pd.read_parquet(excluded_output / "preprocessing-manifest.parquet").iloc[0]
    assert bool(excluded["eligible"]) is False
    assert "label-blind technical QC" in excluded["exclusion_reason"]


def test_preprocessing_reuse_rejects_mutated_cached_fif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, _source, inputs, manifest, _counters = _preprocessed_run(tmp_path, monkeypatch)
    row = pd.read_parquet(manifest).iloc[0]
    cached = Path(row["preprocessed_path"])
    cached.write_bytes(cached.read_bytes() + b"tamper")

    with pytest.raises(DerivativeIntegrityError, match="output checksum changed"):
        preprocess_analysis_units(
            encoder_inputs=inputs,
            output_root=manifest.parent,
            study=study,
        )


def test_preprocessing_materialises_hash_bound_sensitivities_without_csd_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _study_value, _source, _inputs, manifest, _counters = _preprocessed_run(tmp_path, monkeypatch)
    row = pd.read_parquet(manifest).iloc[0]
    assert bool(row["eligible"]) is True
    assert row["average_reference_status"] == "available_primary_harmonised"
    assert row["native_average_reference_status"] == "available"
    assert Path(row["native_average_reference_path"]).is_file()
    assert (
        sha256_file(row["native_average_reference_path"]) == row["native_average_reference_sha256"]
    )
    sensitivity_receipt = Path(row["native_average_reference_provenance_path"])
    assert sha256_file(sensitivity_receipt) == row["native_average_reference_provenance_sha256"]
    sensitivity_payload = json.loads(sensitivity_receipt.read_text(encoding="utf-8"))
    assert sensitivity_payload["inputs"]["sensitivity_branch"] == NATIVE_AVERAGE_REFERENCE_BRANCH
    assert sensitivity_payload["output"]["sha256"] == row["native_average_reference_sha256"]
    assert row["native_csd_status"] == "unavailable"
    assert row["native_csd_reason"] == "insufficient_montage_positions"
    assert row["native_csd_path"] is None
    primary_receipt = json.loads(
        Path(row["preprocessing_provenance_path"]).read_text(encoding="utf-8")
    )
    assert (
        primary_receipt["metadata"]["preprocessing"]["preprocessing_sensitivities"][
            NATIVE_CSD_BRANCH
        ]["status"]
        == "unavailable"
    )
    assert row["sleep_highpass_status"] == "not_applicable"
    assert row["ica_status"] == "not_performed_policy_report_only_with_auxiliary_support"
    assert (
        row["auxiliary_artifact_control_status"]
        == "not_performed_policy_report_only_with_auxiliary_support"
    )
    assert bool(row["auxiliary_channels_used_for_cleaning"]) is False

    sensitivity = Path(row["native_average_reference_path"])
    sensitivity.write_bytes(sensitivity.read_bytes() + b"tamper")
    with pytest.raises(DerivativeIntegrityError, match="output checksum changed"):
        preprocess_analysis_units(
            encoder_inputs=_inputs,
            output_root=manifest.parent,
            study=_study_value,
        )


def test_preprocessing_cache_rejects_sensitivity_configuration_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, _source, inputs, manifest, _counters = _preprocessed_run(tmp_path, monkeypatch)
    changed = study.model_copy(
        update={
            "preprocessing": study.preprocessing.model_copy(
                update={"sleep_sensitivity_highpass_hz": 0.4}
            )
        }
    )
    with pytest.raises(DerivativeIntegrityError, match="input fingerprint changed"):
        preprocess_analysis_units(
            encoder_inputs=inputs,
            output_root=manifest.parent,
            study=changed,
        )


def test_psg_modality_auditably_enables_configured_sleep_highpass_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study()
    _install_fake_preprocessing(monkeypatch)
    source = tmp_path / "sleep.edf"
    source.write_bytes(b"sleep-source")
    inputs = tmp_path / "sleep-inputs.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "sleep-unit",
                "source_path": str(source),
                "modality": "psg",
                "selector_json": json.dumps({"kind": "full_recording"}),
            }
        ]
    ).to_parquet(inputs, index=False)
    manifest, flow_path = preprocess_analysis_units(
        encoder_inputs=inputs,
        output_root=tmp_path / "sleep-preprocess",
        study=study,
    )
    row = pd.read_parquet(manifest).iloc[0]
    assert row["sleep_highpass_status"] == "available"
    assert Path(row["sleep_highpass_path"]).is_file()
    metadata = json.loads(row["sleep_highpass_metadata_json"])
    assert metadata["configured_highpass_hz"] == pytest.approx(0.3)
    flow = json.loads(flow_path.read_text(encoding="utf-8"))
    assert flow["preprocessing_sensitivity_contract"]["sleep_highpass"] == {
        "branch": SLEEP_HIGHPASS_BRANCH,
        "highpass_hz": 0.3,
        "identification": "label_free_modality_membership",
        "modalities": ["psg"],
    }


def test_encoding_reuse_rejects_manifest_hash_field_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, preprocessing, labels, _manifest, _counters = _encoded_run(tmp_path, monkeypatch)
    frame = pd.read_parquet(preprocessing)
    frame.loc[0, "preprocessed_sha256"] = "0" * 64
    frame.to_parquet(preprocessing, index=False)

    with pytest.raises(DerivativeIntegrityError, match="differs from the manifest"):
        encode_analysis_units(
            preprocessing_manifest=preprocessing,
            labels_manifest=labels,
            output_root=tmp_path / "encode",
            study=study,
        )


def test_encoding_reuse_rejects_mutated_trajectory_and_skips_valid_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, preprocessing, labels, manifest, counters = _encoded_run(tmp_path, monkeypatch)
    first_track_calls = counters["encode_tracks"]
    encode_analysis_units(
        preprocessing_manifest=preprocessing,
        labels_manifest=labels,
        output_root=tmp_path / "encode",
        study=study,
    )
    assert counters["encode_tracks"] == first_track_calls

    trajectory = Path(pd.read_parquet(manifest).iloc[0]["trajectory_path"])
    trajectory.write_bytes(trajectory.read_bytes() + b"tamper")
    with pytest.raises(DerivativeIntegrityError, match="output checksum changed"):
        encode_analysis_units(
            preprocessing_manifest=preprocessing,
            labels_manifest=labels,
            output_root=tmp_path / "encode",
            study=study,
        )


def test_model_fingerprint_recursively_rehashes_source_inventory(tmp_path: Path) -> None:
    repository = tmp_path / "labram"
    repository.mkdir()
    model_file = repository / "modeling_finetune.py"
    model_file.write_text("MODEL = 1\n", encoding="utf-8")
    inventory = repository / stage_units.MODEL_SOURCE_INVENTORY
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": model_file.name,
                        "sha256": sha256_file(model_file),
                        "size": model_file.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    fingerprint = stage_units._model_fingerprint(repository, checkpoint, sha256_file(checkpoint))
    assert fingerprint["source_file_count"] == 1

    model_file.write_text("MODEL = 2\n", encoding="utf-8")
    with pytest.raises(DerivativeIntegrityError, match="source file checksum changed"):
        stage_units._model_fingerprint(repository, checkpoint, sha256_file(checkpoint))
