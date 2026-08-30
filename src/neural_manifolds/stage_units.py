"""Analysis-unit preprocessing, label-free encoding, and post-encoding label join."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from neural_manifolds.config import StudyConfig, canonical_json, config_sha256
from neural_manifolds.foundation.labram import DEFAULT_REGIONS, OfficialLaBraMEncoder
from neural_manifolds.foundation.overlap import (
    ensure_pretraining_overlap_columns,
    summarize_pretraining_overlap,
)
from neural_manifolds.preprocessing.eeg import (
    NATIVE_AVERAGE_REFERENCE_BRANCH,
    NATIVE_CSD_BRANCH,
    SENSITIVITY_BRANCHES,
    SLEEP_HIGHPASS_BRANCH,
    SensitivityBranchResult,
    detect_artifact_windows,
    make_windows,
    preprocess_mne_raw,
    preprocess_mne_sensitivity_branches,
)
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.recording_provenance import recording_inventory
from neural_manifolds.stage_processing import (
    _model_environment,
    infer_mains_frequency,
    read_raw_recording,
)
from neural_manifolds.tms_separation import assert_no_direct_tms

PREPROCESSED_SOURCE_MARKER = "neural_manifolds.preprocessed_source.v1"
PREPROCESSED_UNIT_MARKER = "neural_manifolds.preprocessed_unit.v1"
PREPROCESSED_SENSITIVITY_SOURCE_MARKER = "neural_manifolds.preprocessed_sensitivity_source.v1"
PREPROCESSED_SENSITIVITY_UNIT_MARKER = "neural_manifolds.preprocessed_sensitivity_unit.v1"
ENCODED_UNIT_MARKER = "neural_manifolds.encoded_unit.v1"
ENCODED_EVENT_GROUP_MARKER = "neural_manifolds.encoded_event_group.v1"
MODEL_SOURCE_INVENTORY = "SOURCE_MANIFEST.json"
STANDARD_HARMONISED_BRANCH = "harmonised_19_channel"
CLINICAL_LOW_CHANNEL_BRANCH = "clinical_low_channel_psg"
CLINICAL_LOW_CHANNEL_MIN_CHANNELS = 2
SENSITIVITY_COLUMN_PREFIX = {
    NATIVE_AVERAGE_REFERENCE_BRANCH: "native_average_reference",
    NATIVE_CSD_BRANCH: "native_csd",
    SLEEP_HIGHPASS_BRANCH: "sleep_highpass",
}


def _spatial_regions(channel_names: list[str]) -> list[str]:
    observed = set(channel_names)
    return sorted(
        region for region, names in DEFAULT_REGIONS.items() if observed.intersection(names)
    )


def _clinical_property_scope(channel_names: list[str]) -> dict[str, str]:
    observed = set(channel_names)
    regional_counts = {
        region: len(observed.intersection(names)) for region, names in DEFAULT_REGIONS.items()
    }
    sufficiently_sampled = sorted(region for region, count in regional_counts.items() if count >= 3)
    primary_limited = "available_primary_replication_limited_sparse_variable_montage"
    return {
        "repertoire": primary_limited,
        "metastability": primary_limited,
        "directionality": primary_limited,
        "alignment": (
            "available_secondary_limited_sparse_variable_montage_" + "_".join(sufficiently_sampled)
            if len(sufficiently_sampled) >= 2
            else "unavailable_requires_three_channels_in_each_of_two_modules"
        ),
        "reachability": "available_secondary_passive_only_limited_sparse_variable_montage",
    }


class DerivativeIntegrityError(RuntimeError):
    """A cached or manifest-referenced derivative failed its content binding."""


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise DerivativeIntegrityError(f"derivative is not a regular file: {path}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size": resolved.stat().st_size,
    }


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DerivativeIntegrityError(f"{label} is missing or not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DerivativeIntegrityError(f"cannot read valid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise DerivativeIntegrityError(f"{label} must contain a JSON object: {path}")
    return payload


def _load_cached_derivative(
    *,
    output: Path,
    receipt: Path,
    marker: str,
) -> dict[str, Any] | None:
    """Return a fully rehashed derivative receipt, or ``None`` if both are absent."""

    output_exists = output.exists() or output.is_symlink()
    receipt_exists = receipt.exists() or receipt.is_symlink()
    if not output_exists and not receipt_exists:
        return None
    if output_exists != receipt_exists:
        raise DerivativeIntegrityError(
            f"cached derivative/receipt pair is incomplete: {output}, {receipt}"
        )
    payload = _read_json_object(receipt, label="derivative provenance receipt")
    if payload.get("schema_version") != 1 or payload.get("provenance_marker") != marker:
        raise DerivativeIntegrityError(f"cached derivative has an invalid marker: {receipt}")
    inputs = payload.get("inputs")
    recorded_output = payload.get("output")
    if not isinstance(inputs, dict) or not isinstance(recorded_output, dict):
        raise DerivativeIntegrityError(f"cached derivative receipt is incomplete: {receipt}")
    if recorded_output.get("path") != str(output.resolve(strict=True)):
        raise DerivativeIntegrityError(f"cached derivative output path is not bound: {receipt}")
    expected_hash = recorded_output.get("sha256")
    expected_size = recorded_output.get("size")
    if (
        not _valid_sha256(expected_hash)
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
    ):
        raise DerivativeIntegrityError(
            f"cached derivative output fingerprint is invalid: {receipt}"
        )
    observed = _artifact(output)
    if observed["sha256"] != expected_hash or observed["size"] != expected_size:
        raise DerivativeIntegrityError(f"cached derivative output checksum changed: {output}")
    return payload


def _require_input_binding(
    payload: dict[str, Any],
    expected: dict[str, Any],
    *,
    receipt: Path,
    exact: bool = True,
) -> None:
    observed = payload.get("inputs")
    if not isinstance(observed, dict):
        raise DerivativeIntegrityError(f"cached derivative inputs are missing: {receipt}")
    if exact:
        matches = observed == expected
    else:
        matches = all(observed.get(key) == value for key, value in expected.items())
    if not matches:
        raise DerivativeIntegrityError(
            f"cached derivative input fingerprint changed; use a new run directory: {receipt}"
        )


def _write_derivative_receipt(
    *,
    receipt: Path,
    marker: str,
    inputs: dict[str, Any],
    output: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "provenance_marker": marker,
        "inputs": inputs,
        "output": _artifact(output),
        "metadata": metadata,
    }
    atomic_write_json(receipt, payload)
    return payload


def _save_raw_derivative(raw: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.fif")
    raw.save(temporary, overwrite=False, verbose="ERROR")
    os.replace(temporary, destination)


def _sensitivity_paths(
    destination: Path,
    *,
    identity: str,
    branch: str,
    source_level: bool,
) -> tuple[Path, Path]:
    if branch not in SENSITIVITY_COLUMN_PREFIX:
        raise ValueError(f"unknown preprocessing sensitivity branch: {branch}")
    if source_level:
        stem = f"{_safe_key(identity + ':' + branch)}-{branch}-raw.fif"
        output = destination / "source-recordings" / stem
        receipt = output.with_suffix(".provenance.json")
    else:
        output = destination / "units" / f"{identity}-{branch}-raw.fif"
        receipt = destination / "provenance" / f"{identity}-{branch}.json"
    return output, receipt


def _sensitivity_metadata_payload(result: SensitivityBranchResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "reason": result.reason,
        "metadata": result.metadata,
    }


def _validate_sensitivity_summary(value: Any, *, receipt: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or tuple(value) != SENSITIVITY_BRANCHES:
        raise DerivativeIntegrityError(
            f"cached preprocessing sensitivity contract is incomplete: {receipt}"
        )
    validated: dict[str, dict[str, Any]] = {}
    for branch in SENSITIVITY_BRANCHES:
        item = value.get(branch)
        if not isinstance(item, dict) or set(item) != {"status", "reason", "metadata"}:
            raise DerivativeIntegrityError(
                f"cached preprocessing sensitivity metadata is invalid: {receipt}"
            )
        status = item.get("status")
        reason = item.get("reason")
        metadata = item.get("metadata")
        if status not in {"available", "unavailable", "not_applicable", "disabled"}:
            raise DerivativeIntegrityError(
                f"cached preprocessing sensitivity status is invalid: {receipt}"
            )
        if status == "available" and reason is not None:
            raise DerivativeIntegrityError(
                f"cached available sensitivity has an unavailable reason: {receipt}"
            )
        if status != "available" and not isinstance(reason, str):
            raise DerivativeIntegrityError(
                f"cached unavailable sensitivity has no reason: {receipt}"
            )
        if not isinstance(metadata, dict):
            raise DerivativeIntegrityError(
                f"cached preprocessing sensitivity metadata is missing: {receipt}"
            )
        validated[branch] = item
    return validated


def _preprocessing_input_fingerprints(
    study: StudyConfig, selector: dict[str, Any]
) -> dict[str, str]:
    window_keys = (
        "kind",
        "start_seconds",
        "stop_seconds",
        "event_onset_seconds",
        "event_sample",
        "epoch_start_offset_seconds",
        "epoch_stop_offset_seconds",
        "trial_index",
    )
    return {
        "study_config_sha256": config_sha256(study),
        "preprocessing_config_sha256": _payload_sha256(study.preprocessing.model_dump(mode="json")),
        "selector_sha256": _payload_sha256(selector),
        "selector_window_sha256": _payload_sha256(
            {key: selector.get(key) for key in window_keys if key in selector}
        ),
    }


def _encoding_input_fingerprints(study: StudyConfig) -> dict[str, str]:
    representation = study.representation.model_dump(mode="json")
    window_keys = (
        "harmonised_window_seconds",
        "harmonised_step_seconds",
        "alignment_window_seconds",
        "alignment_step_seconds",
        "labram_patch_seconds",
    )
    return {
        "study_config_sha256": config_sha256(study),
        "representation_config_sha256": _payload_sha256(representation),
        "encoding_window_config_sha256": _payload_sha256(
            {
                **{key: representation[key] for key in window_keys},
                "target_sampling_hz": study.preprocessing.target_sampling_hz,
                "minimum_valid_windows": study.preprocessing.minimum_valid_windows,
            }
        ),
    }


def _validated_model_source_inventory(repository: Path) -> dict[str, Any]:
    if repository.is_symlink() or not repository.is_dir():
        raise DerivativeIntegrityError(
            f"LaBraM repository is missing or not a regular directory: {repository}"
        )
    root = repository.resolve(strict=True)
    inventory_path = root / MODEL_SOURCE_INVENTORY
    payload = _read_json_object(inventory_path, label="LaBraM source inventory")
    if set(payload) != {"schema_version", "files"} or payload.get("schema_version") != 1:
        raise DerivativeIntegrityError("LaBraM source inventory has an invalid document shape")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise DerivativeIntegrityError("LaBraM source inventory contains no files")
    recorded: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            raise DerivativeIntegrityError(
                f"LaBraM source inventory entry {index} has an invalid shape"
            )
        relative_value = item.get("path")
        expected_hash = item.get("sha256")
        expected_size = item.get("size")
        if not isinstance(relative_value, str):
            raise DerivativeIntegrityError("LaBraM source inventory path is invalid")
        relative = PurePosixPath(relative_value)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise DerivativeIntegrityError(
                f"LaBraM source inventory path is unsafe: {relative_value!r}"
            )
        canonical = relative.as_posix()
        if canonical in recorded or canonical == MODEL_SOURCE_INVENTORY:
            raise DerivativeIntegrityError(
                f"LaBraM source inventory path is duplicate or reserved: {canonical!r}"
            )
        if (
            not _valid_sha256(expected_hash)
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise DerivativeIntegrityError(
                f"LaBraM source inventory fingerprint is invalid: {canonical!r}"
            )
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise DerivativeIntegrityError(f"LaBraM source file is missing or unsafe: {path}")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise DerivativeIntegrityError(f"LaBraM source file checksum changed: {path}")
        recorded.add(canonical)
    physical: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            raise DerivativeIntegrityError(f"LaBraM source contains a symbolic link: {path}")
        if path.is_file() and relative.as_posix() != MODEL_SOURCE_INVENTORY:
            physical.add(relative.as_posix())
    if physical != recorded:
        raise DerivativeIntegrityError(
            "LaBraM source inventory coverage changed; "
            f"missing={sorted(recorded - physical)}, extra={sorted(physical - recorded)}"
        )
    return {
        "repository_path": str(root),
        "source_inventory_path": str(inventory_path),
        "source_inventory_sha256": sha256_file(inventory_path),
        "source_file_count": len(recorded),
    }


def _model_fingerprint(repository: Path, checkpoint: Path, checkpoint_hash: str) -> dict[str, Any]:
    if not _valid_sha256(checkpoint_hash):
        raise DerivativeIntegrityError("LaBraM checkpoint environment SHA-256 is invalid")
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise DerivativeIntegrityError(
            f"LaBraM checkpoint is missing or not a regular file: {checkpoint}"
        )
    resolved_checkpoint = checkpoint.resolve(strict=True)
    observed_checkpoint_hash = sha256_file(resolved_checkpoint)
    if observed_checkpoint_hash != checkpoint_hash:
        raise DerivativeIntegrityError(
            "LaBraM checkpoint checksum differs from its bootstrap fingerprint"
        )
    source = _validated_model_source_inventory(repository)
    manifest_value = os.environ.get("NEURAL_MANIFOLDS_MODEL_MANIFEST")
    model_manifest: dict[str, Any] | None = None
    if manifest_value:
        manifest_path = Path(manifest_value)
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise DerivativeIntegrityError(
                f"model manifest is missing or not a regular file: {manifest_path}"
            )
        manifest_path = manifest_path.resolve(strict=True)
        manifest = _read_json_object(manifest_path, label="model manifest")
        labram = manifest.get("models", {}).get("labram_base", {})
        checkpoint_record = labram.get("checkpoint", {}) if isinstance(labram, dict) else {}
        source_record = labram.get("source", {}) if isinstance(labram, dict) else {}
        if (
            checkpoint_record.get("path") != str(resolved_checkpoint)
            or checkpoint_record.get("sha256") != observed_checkpoint_hash
            or source_record.get("path") != source["source_inventory_path"]
            or source_record.get("sha256") != source["source_inventory_sha256"]
        ):
            raise DerivativeIntegrityError(
                "model manifest does not bind the active LaBraM source and checkpoint"
            )
        model_manifest = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
    return {
        **source,
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_sha256": observed_checkpoint_hash,
        "model_manifest": model_manifest,
        "factory": "modeling_finetune:labram_base_patch200_200",
        "weights_frozen": True,
    }


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def derivative_artifact_paths(
    output_root: str | Path, *, directories: tuple[str, ...]
) -> list[Path]:
    """Return every regular cached derivative/receipt for queue-level rehashing."""

    root = Path(output_root)
    paths: list[Path] = []
    for name in directories:
        directory = root / name
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise DerivativeIntegrityError(f"derivative cache directory is unsafe: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise DerivativeIntegrityError(f"derivative cache contains a symlink: {path}")
            if path.is_file():
                paths.append(path.resolve(strict=True))
    if len(paths) != len(set(paths)):
        raise DerivativeIntegrityError("derivative cache inventory contains duplicate paths")
    return paths


def _safe_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _selector(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("selector_json")
    if not isinstance(value, str):
        raise ValueError("encoder row has no serialized selector")
    selector = json.loads(value)
    if not isinstance(selector, dict) or not isinstance(selector.get("kind"), str):
        raise ValueError("serialized selector is malformed")
    return selector


def _crop_clean(raw: Any, selector: dict[str, Any]) -> Any:
    kind = selector["kind"]
    if kind == "full_recording":
        return raw.copy()
    if kind == "interval_seconds":
        start = float(selector["start_seconds"])
        stop = min(float(selector["stop_seconds"]), float(raw.times[-1]))
    elif kind == "event_epoch":
        onset = selector.get("event_onset_seconds")
        if onset is None and selector.get("event_sample") is not None:
            onset = float(selector["event_sample"]) / float(raw.info["sfreq"])
        if onset is None:
            raise ValueError("event selector has no onset")
        start = float(onset) + float(selector["epoch_start_offset_seconds"])
        stop = float(onset) + float(selector["epoch_stop_offset_seconds"])
        start = max(0.0, start)
        stop = min(stop, float(raw.times[-1]))
    else:
        raise ValueError(f"electrophysiology selector {kind!r} is not implemented")
    if stop <= start:
        raise ValueError("selector lies outside the recording")
    # MNE includes tmax; subtract one sample so adjacent intervals never share a sample.
    stop = max(start, stop - 1.0 / float(raw.info["sfreq"]))
    return raw.copy().crop(tmin=start, tmax=stop, include_tmax=True)


def preprocess_analysis_units(
    *,
    encoder_inputs: str | Path,
    output_root: str | Path,
    study: StudyConfig,
    clinical_low_channel_unit_ids: frozenset[str] = frozenset(),
    qc_recordings: str | Path | None = None,
) -> tuple[Path, Path]:
    """Preprocess source recordings once, then apply immutable unit selectors."""

    frame = pd.read_parquet(encoder_inputs)
    assert_no_direct_tms(frame, stage="general preprocessing input")
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - EEG runtime extra
        raise RuntimeError("install neural-manifolds[eeg]") from exc
    required = {"unit_id", "source_path", "modality", "selector_json"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"encoder inputs are missing {sorted(missing)}")
    if frame["unit_id"].duplicated().any():
        raise ValueError("encoder inputs contain duplicate unit_id values")
    qc_by_source: dict[str, dict[str, Any]] = {}
    qc_recordings_sha256: str | None = None
    if qc_recordings is not None:
        qc_path = Path(qc_recordings).resolve(strict=True)
        qc_frame = pd.read_parquet(qc_path)
        qc_required = {
            "source_path",
            "technically_eligible",
            "qc_status",
            "technical_exclusion_reason",
            "review_flags_json",
        }
        qc_missing = qc_required.difference(qc_frame.columns)
        if qc_missing:
            raise ValueError(f"recording QC flow is missing {sorted(qc_missing)}")
        for qc_row in qc_frame.to_dict(orient="records"):
            key = str(Path(str(qc_row["source_path"])).resolve(strict=True))
            if key in qc_by_source:
                raise ValueError(f"recording QC flow contains duplicate source path: {key}")
            qc_by_source[key] = qc_row
        qc_recordings_sha256 = sha256_file(qc_path)
    observed_unit_ids = set(frame["unit_id"].astype(str))
    unknown_low_channel_ids = sorted(
        set(clinical_low_channel_unit_ids).difference(observed_unit_ids)
    )
    if unknown_low_channel_ids:
        raise ValueError(
            f"clinical low-channel route contains unknown unit ids: {unknown_low_channel_ids}"
        )
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    source_cache: dict[str, dict[str, Any]] = {}
    source_inventory_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    reused_units = 0
    generated_units = 0
    reused_sources = 0
    generated_sources = 0
    for row in frame.to_dict(orient="records"):
        analysis_branch = (
            CLINICAL_LOW_CHANNEL_BRANCH
            if str(row["unit_id"]) in clinical_low_channel_unit_ids
            else STANDARD_HARMONISED_BRANCH
        )
        result = {
            "unit_id": row["unit_id"],
            "modality": row["modality"],
            "selector_json": row["selector_json"],
            "preprocessed_path": None,
            "eligible": False,
            "analysis_branch": analysis_branch,
            "qc_status": None,
            "qc_review_flags_json": "[]",
            "primary_reference": study.preprocessing.primary_reference,
            "average_reference_status": "unavailable_primary_preprocessing_failed",
            "average_reference_path": None,
            "average_reference_sha256": None,
            "auxiliary_channel_inventory_json": json.dumps(
                {"eog": [], "ecg": [], "emg": []}, separators=(",", ":")
            ),
            "ica_support_status": "unavailable_preprocessing_failed",
            "ica_status": "not_performed_preprocessing_failed",
            "auxiliary_artifact_control_support_status": ("unavailable_preprocessing_failed"),
            "auxiliary_artifact_control_status": "not_performed_preprocessing_failed",
            "auxiliary_channels_used_for_cleaning": False,
        }
        for branch, prefix in SENSITIVITY_COLUMN_PREFIX.items():
            result.update(
                {
                    f"{prefix}_status": "unavailable_primary_preprocessing_failed",
                    f"{prefix}_reason": "primary_preprocessing_failed",
                    f"{prefix}_path": None,
                    f"{prefix}_sha256": None,
                    f"{prefix}_provenance_path": None,
                    f"{prefix}_provenance_sha256": None,
                    f"{prefix}_metadata_json": json.dumps(
                        {"branch": branch}, sort_keys=True, separators=(",", ":")
                    ),
                }
            )
        try:
            source_path = Path(str(row["source_path"]))
            if source_path.is_symlink() or not source_path.is_file():
                raise DerivativeIntegrityError(
                    f"source recording is not a regular file: {source_path}"
                )
            source = source_path.resolve(strict=True)
            source_cache_id = str(source)
            qc_row = qc_by_source.get(source_cache_id) if qc_recordings is not None else None
            if qc_recordings is not None and qc_row is None:
                raise ValueError("source recording is absent from the label-blind QC flow")
            if qc_row is not None:
                result["qc_status"] = str(qc_row["qc_status"])
                result["qc_review_flags_json"] = str(qc_row["review_flags_json"])
                if not bool(qc_row["technically_eligible"]):
                    raise ValueError(
                        "recording excluded by label-blind technical QC: "
                        + str(qc_row["technical_exclusion_reason"])
                    )
            if source_cache_id not in source_inventory_cache:
                source_header = read_raw_recording(source)
                try:
                    source_inventory_cache[source_cache_id] = recording_inventory(
                        source,
                        raw=source_header,
                    )
                finally:
                    close = getattr(source_header, "close", None)
                    if callable(close):
                        close()
            source_inventory = source_inventory_cache[source_cache_id]
            source_hash = str(source_inventory["combined_sha256"])
            selector = _selector(row)
            fingerprints = _preprocessing_input_fingerprints(study, selector)
            is_sleep_recording = str(row["modality"]) in set(
                study.preprocessing.sleep_sensitivity_modalities
            )
            require_complete_canonical = bool(
                study.preprocessing.require_complete_harmonised_montage
                and analysis_branch != CLINICAL_LOW_CHANNEL_BRANCH
            )
            source_key = (
                f"{source}:{source_hash}:{fingerprints['preprocessing_config_sha256']}:"
                f"{analysis_branch}:sleep={is_sleep_recording}"
            )
            cache_key = source_key
            master = destination / "source-recordings" / f"{_safe_key(cache_key)}-raw.fif"
            master_receipt = master.with_suffix(".provenance.json")
            unit_path = destination / "units" / f"{row['unit_id']}-raw.fif"
            provenance_path = destination / "provenance" / f"{row['unit_id']}.json"
            unit_base_inputs: dict[str, Any] = {
                "unit_id": str(row["unit_id"]),
                "source_path": str(source),
                "source_sha256": source_hash,
                "source_recording_inventory": source_inventory,
                "qc_recordings_sha256": qc_recordings_sha256,
                "qc_status": result["qc_status"],
                "master_preprocessed_path": str(master.resolve()),
                "analysis_branch": analysis_branch,
                "is_sleep_recording": is_sleep_recording,
                "sleep_identification": "label_free_modality_membership",
                **fingerprints,
            }
            cached_unit = _load_cached_derivative(
                output=unit_path,
                receipt=provenance_path,
                marker=PREPROCESSED_UNIT_MARKER,
            )
            if cached_unit is not None:
                _require_input_binding(
                    cached_unit,
                    unit_base_inputs,
                    receipt=provenance_path,
                    exact=False,
                )
            if source_key not in source_cache:
                master_inputs = {
                    "source_path": str(source),
                    "source_sha256": source_hash,
                    "source_recording_inventory": source_inventory,
                    "study_config_sha256": fingerprints["study_config_sha256"],
                    "preprocessing_config_sha256": fingerprints["preprocessing_config_sha256"],
                    "analysis_branch": analysis_branch,
                    "is_sleep_recording": is_sleep_recording,
                    "sleep_identification": "label_free_modality_membership",
                }
                cached_master = _load_cached_derivative(
                    output=master,
                    receipt=master_receipt,
                    marker=PREPROCESSED_SOURCE_MARKER,
                )
                if cached_master is not None:
                    _require_input_binding(
                        cached_master,
                        master_inputs,
                        receipt=master_receipt,
                    )
                    provenance = cached_master.get("metadata", {}).get("preprocessing")
                    if not isinstance(provenance, dict):
                        raise DerivativeIntegrityError(
                            f"cached source preprocessing metadata is missing: {master_receipt}"
                        )
                    sensitivity_summary = _validate_sensitivity_summary(
                        provenance.get("preprocessing_sensitivities"),
                        receipt=master_receipt,
                    )
                    source_sensitivities: dict[str, dict[str, Any]] = {}
                    for branch in SENSITIVITY_BRANCHES:
                        sensitivity_path, sensitivity_receipt = _sensitivity_paths(
                            destination,
                            identity=source_key,
                            branch=branch,
                            source_level=True,
                        )
                        branch_inputs = {**master_inputs, "sensitivity_branch": branch}
                        cached_sensitivity = _load_cached_derivative(
                            output=sensitivity_path,
                            receipt=sensitivity_receipt,
                            marker=PREPROCESSED_SENSITIVITY_SOURCE_MARKER,
                        )
                        branch_summary = sensitivity_summary[branch]
                        if branch_summary["status"] == "available":
                            if cached_sensitivity is None:
                                raise DerivativeIntegrityError(
                                    "cached sensitivity is declared available but missing: "
                                    f"{sensitivity_path}"
                                )
                            _require_input_binding(
                                cached_sensitivity,
                                branch_inputs,
                                receipt=sensitivity_receipt,
                            )
                            source_sensitivities[branch] = {
                                **branch_summary,
                                "path": sensitivity_path,
                                "sha256": str(cached_sensitivity["output"]["sha256"]),
                                "provenance_path": sensitivity_receipt,
                                "provenance_sha256": sha256_file(sensitivity_receipt),
                            }
                        else:
                            if cached_sensitivity is not None:
                                raise DerivativeIntegrityError(
                                    "cached sensitivity derivative exists for a branch declared "
                                    f"{branch_summary['status']}: {sensitivity_path}"
                                )
                            source_sensitivities[branch] = {
                                **branch_summary,
                                "path": None,
                                "sha256": None,
                                "provenance_path": None,
                                "provenance_sha256": None,
                            }
                    reused_sources += 1
                else:
                    raw = read_raw_recording(source)
                    notch_hz = infer_mains_frequency(raw)
                    interpolation_fraction = (
                        0.0
                        if analysis_branch == CLINICAL_LOW_CHANNEL_BRANCH
                        else study.preprocessing.maximum_interpolation_fraction
                    )
                    clean, provenance = preprocess_mne_raw(
                        raw,
                        canonical_channels=study.preprocessing.canonical_channels,
                        target_sampling_hz=study.preprocessing.target_sampling_hz,
                        highpass_hz=study.preprocessing.highpass_hz,
                        lowpass_hz=study.preprocessing.lowpass_hz,
                        notch_hz=notch_hz,
                        maximum_interpolation_fraction=interpolation_fraction,
                        require_complete_canonical=require_complete_canonical,
                    )
                    minimum_channels = (
                        CLINICAL_LOW_CHANNEL_MIN_CHANNELS
                        if analysis_branch == CLINICAL_LOW_CHANNEL_BRANCH
                        else study.preprocessing.minimum_canonical_channels
                    )
                    if len(clean.ch_names) < minimum_channels:
                        raise ValueError(
                            f"only {len(clean.ch_names)} canonical channels after preprocessing; "
                            f"{analysis_branch} requires {minimum_channels}"
                        )
                    regions = _spatial_regions(list(clean.ch_names))
                    if analysis_branch == CLINICAL_LOW_CHANNEL_BRANCH and len(regions) < 2:
                        raise ValueError(
                            "clinical low-channel PSG requires at least two represented "
                            "frontal/central/occipital regions for the frozen metric implementation"
                        )
                    sensitivity_results = preprocess_mne_sensitivity_branches(
                        raw,
                        canonical_channels=study.preprocessing.canonical_channels,
                        target_sampling_hz=study.preprocessing.target_sampling_hz,
                        primary_highpass_hz=study.preprocessing.highpass_hz,
                        sleep_highpass_hz=study.preprocessing.sleep_sensitivity_highpass_hz,
                        lowpass_hz=study.preprocessing.lowpass_hz,
                        notch_hz=notch_hz,
                        maximum_interpolation_fraction=interpolation_fraction,
                        require_complete_canonical=require_complete_canonical,
                        native_montage_sensitivity=(study.preprocessing.native_montage_sensitivity),
                        csd_sensitivity=study.preprocessing.csd_sensitivity,
                        csd_minimum_channels=study.preprocessing.csd_minimum_channels,
                        csd_minimum_position_fraction=(
                            study.preprocessing.csd_minimum_position_fraction
                        ),
                        is_sleep_recording=is_sleep_recording,
                    )
                    sensitivity_summary = {
                        branch: _sensitivity_metadata_payload(sensitivity_results[branch])
                        for branch in SENSITIVITY_BRANCHES
                    }
                    provenance = {
                        **provenance,
                        "analysis_branch": analysis_branch,
                        "spatial_regions": regions,
                        "property_scope": (
                            _clinical_property_scope(list(clean.ch_names))
                            if analysis_branch == CLINICAL_LOW_CHANNEL_BRANCH
                            else {
                                axis: "available_primary_harmonised_track"
                                for axis in (
                                    "repertoire",
                                    "metastability",
                                    "directionality",
                                    "alignment",
                                    "reachability",
                                )
                            }
                        ),
                        "interpolation_allowed": (analysis_branch != CLINICAL_LOW_CHANNEL_BRANCH),
                        "preprocessing_sensitivities": sensitivity_summary,
                        "sleep_sensitivity_identification": {
                            "is_sleep_recording": is_sleep_recording,
                            "criterion": "label_free_modality_membership",
                            "configured_modalities": list(
                                study.preprocessing.sleep_sensitivity_modalities
                            ),
                        },
                        "auxiliary_ica_policy": study.preprocessing.auxiliary_ica_policy,
                    }
                    observed_source_inventory = recording_inventory(source, raw=raw)
                    if observed_source_inventory != source_inventory:
                        raise DerivativeIntegrityError(
                            f"source recording changed during preprocessing: {source}"
                        )
                    source_sensitivities = {}
                    for branch in SENSITIVITY_BRANCHES:
                        branch_result = sensitivity_results[branch]
                        sensitivity_path, sensitivity_receipt = _sensitivity_paths(
                            destination,
                            identity=source_key,
                            branch=branch,
                            source_level=True,
                        )
                        branch_inputs = {**master_inputs, "sensitivity_branch": branch}
                        if branch_result.status == "available":
                            _save_raw_derivative(branch_result.raw, sensitivity_path)
                            sensitivity_payload = _write_derivative_receipt(
                                receipt=sensitivity_receipt,
                                marker=PREPROCESSED_SENSITIVITY_SOURCE_MARKER,
                                inputs=branch_inputs,
                                output=sensitivity_path,
                                metadata={
                                    "preprocessing": branch_result.metadata,
                                    "availability_status": branch_result.status,
                                    "label_fields_consumed": [],
                                },
                            )
                            source_sensitivities[branch] = {
                                **sensitivity_summary[branch],
                                "path": sensitivity_path,
                                "sha256": str(sensitivity_payload["output"]["sha256"]),
                                "provenance_path": sensitivity_receipt,
                                "provenance_sha256": sha256_file(sensitivity_receipt),
                            }
                        else:
                            unavailable_cache = _load_cached_derivative(
                                output=sensitivity_path,
                                receipt=sensitivity_receipt,
                                marker=PREPROCESSED_SENSITIVITY_SOURCE_MARKER,
                            )
                            if unavailable_cache is not None:
                                raise DerivativeIntegrityError(
                                    "stale sensitivity derivative exists for unavailable branch: "
                                    f"{sensitivity_path}"
                                )
                            source_sensitivities[branch] = {
                                **sensitivity_summary[branch],
                                "path": None,
                                "sha256": None,
                                "provenance_path": None,
                                "provenance_sha256": None,
                            }
                    _save_raw_derivative(clean, master)
                    _write_derivative_receipt(
                        receipt=master_receipt,
                        marker=PREPROCESSED_SOURCE_MARKER,
                        inputs=master_inputs,
                        output=master,
                        metadata={
                            "preprocessing": provenance,
                            "label_fields_consumed": [],
                        },
                    )
                    generated_sources += 1
                master_hash = sha256_file(master)
                source_cache[source_key] = {
                    "master": master,
                    "provenance": provenance,
                    "source_hash": source_hash,
                    "master_hash": master_hash,
                    "sensitivities": source_sensitivities,
                }
            source_record = source_cache[source_key]
            master = Path(source_record["master"])
            provenance = dict(source_record["provenance"])
            source_hash = str(source_record["source_hash"])
            master_hash = str(source_record["master_hash"])
            source_sensitivities = dict(source_record["sensitivities"])
            unit_inputs = {
                **unit_base_inputs,
                "master_preprocessed_sha256": master_hash,
            }
            if cached_unit is not None:
                _require_input_binding(cached_unit, unit_inputs, receipt=provenance_path)
                metadata = cached_unit.get("metadata")
                if not isinstance(metadata, dict) or not isinstance(
                    metadata.get("duration_seconds"), (int, float)
                ):
                    raise DerivativeIntegrityError(
                        f"cached unit duration metadata is missing: {provenance_path}"
                    )
                duration = float(metadata["duration_seconds"])
                unit_hash = str(cached_unit["output"]["sha256"])
                reused_units += 1
            else:
                clean = mne.io.read_raw_fif(master, preload=False, verbose="ERROR")
                selected = _crop_clean(clean, selector)
                duration = float(selected.n_times / selected.info["sfreq"])
                if (
                    selector["kind"] == "full_recording"
                    and duration < study.preprocessing.minimum_rest_seconds
                ):
                    raise ValueError(
                        f"only {duration:.1f}s; need {study.preprocessing.minimum_rest_seconds}s"
                    )
                if sha256_file(master) != master_hash:
                    raise DerivativeIntegrityError(
                        f"cached source derivative changed before unit materialisation: {master}"
                    )
                _save_raw_derivative(selected.load_data(), unit_path)
                unit_receipt = _write_derivative_receipt(
                    receipt=provenance_path,
                    marker=PREPROCESSED_UNIT_MARKER,
                    inputs=unit_inputs,
                    output=unit_path,
                    metadata={
                        "preprocessing": provenance,
                        "selector": selector,
                        "duration_seconds": duration,
                        "label_fields_consumed": [],
                    },
                )
                unit_hash = str(unit_receipt["output"]["sha256"])
                generated_units += 1
            for branch in SENSITIVITY_BRANCHES:
                prefix = SENSITIVITY_COLUMN_PREFIX[branch]
                source_sensitivity = source_sensitivities[branch]
                result[f"{prefix}_status"] = str(source_sensitivity["status"])
                result[f"{prefix}_reason"] = source_sensitivity["reason"]
                result[f"{prefix}_metadata_json"] = json.dumps(
                    source_sensitivity["metadata"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                sensitivity_unit_path, sensitivity_unit_receipt = _sensitivity_paths(
                    destination,
                    identity=str(row["unit_id"]),
                    branch=branch,
                    source_level=False,
                )
                if source_sensitivity["status"] != "available":
                    stale = _load_cached_derivative(
                        output=sensitivity_unit_path,
                        receipt=sensitivity_unit_receipt,
                        marker=PREPROCESSED_SENSITIVITY_UNIT_MARKER,
                    )
                    if stale is not None:
                        raise DerivativeIntegrityError(
                            "unit sensitivity derivative exists for unavailable source branch: "
                            f"{sensitivity_unit_path}"
                        )
                    continue
                sensitivity_source_path = Path(source_sensitivity["path"])
                sensitivity_source_receipt = Path(source_sensitivity["provenance_path"])
                sensitivity_unit_inputs = {
                    **unit_base_inputs,
                    "sensitivity_branch": branch,
                    "source_sensitivity_path": str(sensitivity_source_path.resolve(strict=True)),
                    "source_sensitivity_sha256": str(source_sensitivity["sha256"]),
                    "source_sensitivity_provenance_sha256": str(
                        source_sensitivity["provenance_sha256"]
                    ),
                }
                cached_sensitivity_unit = _load_cached_derivative(
                    output=sensitivity_unit_path,
                    receipt=sensitivity_unit_receipt,
                    marker=PREPROCESSED_SENSITIVITY_UNIT_MARKER,
                )
                try:
                    if cached_sensitivity_unit is not None:
                        _require_input_binding(
                            cached_sensitivity_unit,
                            sensitivity_unit_inputs,
                            receipt=sensitivity_unit_receipt,
                        )
                        sensitivity_unit_hash = str(cached_sensitivity_unit["output"]["sha256"])
                    else:
                        if sha256_file(sensitivity_source_path) != source_sensitivity["sha256"]:
                            raise DerivativeIntegrityError(
                                "source sensitivity changed before unit materialisation: "
                                f"{sensitivity_source_path}"
                            )
                        if (
                            sha256_file(sensitivity_source_receipt)
                            != source_sensitivity["provenance_sha256"]
                        ):
                            raise DerivativeIntegrityError(
                                "source sensitivity receipt changed before unit materialisation: "
                                f"{sensitivity_source_receipt}"
                            )
                        sensitivity_raw = mne.io.read_raw_fif(
                            sensitivity_source_path,
                            preload=False,
                            verbose="ERROR",
                        )
                        sensitivity_selected = _crop_clean(sensitivity_raw, selector)
                        _save_raw_derivative(
                            sensitivity_selected.load_data(), sensitivity_unit_path
                        )
                        sensitivity_unit_payload = _write_derivative_receipt(
                            receipt=sensitivity_unit_receipt,
                            marker=PREPROCESSED_SENSITIVITY_UNIT_MARKER,
                            inputs=sensitivity_unit_inputs,
                            output=sensitivity_unit_path,
                            metadata={
                                "preprocessing": source_sensitivity["metadata"],
                                "selector": selector,
                                "duration_seconds": float(
                                    sensitivity_selected.n_times
                                    / sensitivity_selected.info["sfreq"]
                                ),
                                "availability_status": "available",
                                "label_fields_consumed": [],
                            },
                        )
                        sensitivity_unit_hash = str(sensitivity_unit_payload["output"]["sha256"])
                except DerivativeIntegrityError:
                    raise
                except (ValueError, RuntimeError, OSError) as error:
                    result[f"{prefix}_status"] = "unavailable"
                    result[f"{prefix}_reason"] = (
                        f"unit_materialisation_failed:{type(error).__name__}:{error}"
                    )
                    continue
                result.update(
                    {
                        f"{prefix}_status": "available",
                        f"{prefix}_reason": None,
                        f"{prefix}_path": str(sensitivity_unit_path),
                        f"{prefix}_sha256": sensitivity_unit_hash,
                        f"{prefix}_provenance_path": str(sensitivity_unit_receipt),
                        f"{prefix}_provenance_sha256": sha256_file(sensitivity_unit_receipt),
                    }
                )
            auxiliary_audit = provenance.get("auxiliary_channel_audit")
            if not isinstance(auxiliary_audit, dict):
                auxiliary_audit = {
                    "metadata_status": "unavailable_preprocessing_provenance",
                    "channels": {"eog": [], "ecg": [], "emg": []},
                    "ica_support_status": "unavailable_preprocessing_provenance",
                    "ica_status": "not_performed_unavailable_preprocessing_provenance",
                    "auxiliary_artifact_control_support_status": (
                        "unavailable_preprocessing_provenance"
                    ),
                    "auxiliary_artifact_control_status": (
                        "not_performed_unavailable_preprocessing_provenance"
                    ),
                    "auxiliary_channels_used_for_cleaning": False,
                }
            ica_status = str(auxiliary_audit.get("ica_status", ""))
            artifact_control_status = str(
                auxiliary_audit.get("auxiliary_artifact_control_status", "")
            )
            if (
                not ica_status.startswith("not_performed")
                or not artifact_control_status.startswith("not_performed")
                or bool(auxiliary_audit.get("auxiliary_channels_used_for_cleaning", False))
            ):
                raise DerivativeIntegrityError(
                    "generic EEG preprocessing provenance falsely claims auxiliary ICA cleaning"
                )
            result.update(
                {
                    "preprocessed_path": str(unit_path),
                    "preprocessed_sha256": unit_hash,
                    "source_recording_sha256": source_hash,
                    "source_recording_file_count": int(source_inventory["file_count"]),
                    "preprocessing_provenance_path": str(provenance_path),
                    "preprocessing_provenance_sha256": sha256_file(provenance_path),
                    "duration_seconds": duration,
                    "average_reference_status": "available_primary_harmonised",
                    "average_reference_path": str(unit_path),
                    "average_reference_sha256": unit_hash,
                    "auxiliary_channel_inventory_json": json.dumps(
                        auxiliary_audit.get("channels", {"eog": [], "ecg": [], "emg": []}),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "ica_support_status": str(auxiliary_audit.get("ica_support_status")),
                    "ica_status": ica_status,
                    "auxiliary_artifact_control_support_status": str(
                        auxiliary_audit.get("auxiliary_artifact_control_support_status")
                    ),
                    "auxiliary_artifact_control_status": artifact_control_status,
                    "auxiliary_channels_used_for_cleaning": False,
                    "property_scope_json": json.dumps(
                        provenance.get("property_scope", {}),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "spatial_regions": "|".join(
                        str(value) for value in provenance.get("spatial_regions", [])
                    ),
                    "eligible": True,
                    "exclusion_reason": None,
                }
            )
        except DerivativeIntegrityError:
            raise
        except (ValueError, RuntimeError, OSError) as error:
            result["exclusion_reason"] = f"{type(error).__name__}: {error}"
        rows.append(result)
    manifest = _atomic_parquet(pd.DataFrame(rows), destination / "preprocessing-manifest.parquet")
    flow = destination / "preprocessing-flow.json"
    eligible = sum(bool(row["eligible"]) for row in rows)
    derivative_inventory = [
        _artifact(path)
        for path in derivative_artifact_paths(
            destination, directories=("source-recordings", "units", "provenance")
        )
    ]
    atomic_write_json(
        flow,
        {
            "schema_version": 1,
            "analysis_units": len(rows),
            "eligible_units": eligible,
            "excluded_units": len(rows) - eligible,
            "unique_source_recordings": len(source_cache),
            "reused_source_derivatives": reused_sources,
            "generated_source_derivatives": generated_sources,
            "reused_unit_derivatives": reused_units,
            "generated_unit_derivatives": generated_units,
            "qc_recordings_sha256": qc_recordings_sha256,
            "qc_flow_enforced": qc_recordings is not None,
            "qc_status_counts": pd.Series([str(row.get("qc_status")) for row in rows], dtype=str)
            .value_counts()
            .sort_index()
            .to_dict(),
            "derivative_inventory": derivative_inventory,
            "analysis_branch_counts": pd.Series([row["analysis_branch"] for row in rows])
            .value_counts()
            .sort_index()
            .to_dict(),
            "preprocessing_sensitivity_contract": {
                "primary": {
                    "branch": STANDARD_HARMONISED_BRANCH,
                    "canonical_channel_count": len(study.preprocessing.canonical_channels),
                    "require_complete_harmonised_montage": (
                        study.preprocessing.require_complete_harmonised_montage
                    ),
                    "reference": study.preprocessing.primary_reference,
                },
                "native_full_montage": {
                    "enabled": study.preprocessing.native_montage_sensitivity,
                    "average_reference_branch": NATIVE_AVERAGE_REFERENCE_BRANCH,
                    "csd_branch": NATIVE_CSD_BRANCH,
                    "csd_enabled": study.preprocessing.csd_sensitivity,
                    "csd_minimum_channels": study.preprocessing.csd_minimum_channels,
                    "csd_minimum_position_fraction": (
                        study.preprocessing.csd_minimum_position_fraction
                    ),
                },
                "sleep_highpass": {
                    "branch": SLEEP_HIGHPASS_BRANCH,
                    "highpass_hz": study.preprocessing.sleep_sensitivity_highpass_hz,
                    "identification": "label_free_modality_membership",
                    "modalities": list(study.preprocessing.sleep_sensitivity_modalities),
                },
                "auxiliary_ica_policy": study.preprocessing.auxiliary_ica_policy,
                "scientific_result_gate": False,
            },
            "sensitivity_status_counts": {
                prefix: pd.Series([str(row.get(f"{prefix}_status")) for row in rows], dtype=str)
                .value_counts()
                .sort_index()
                .to_dict()
                for prefix in SENSITIVITY_COLUMN_PREFIX.values()
            },
            "clinical_low_channel_contract": {
                "branch": CLINICAL_LOW_CHANNEL_BRANCH,
                "minimum_canonical_channels": CLINICAL_LOW_CHANNEL_MIN_CHANNELS,
                "minimum_spatial_regions": 2,
                "interpolation_allowed": False,
                "healthy_fitted_objects_refit": False,
                "primary_replication_axes": [
                    "repertoire",
                    "metastability",
                    "directionality",
                ],
                "alignment_reporting_requirement": (
                    "at_least_three_channels_in_each_of_two_required_modules"
                ),
                "reachability_scope": "secondary_passive_only",
            },
            "exclusions": [
                {"unit_id": row["unit_id"], "reason": row["exclusion_reason"]}
                for row in rows
                if not row["eligible"]
            ],
            "label_fields_consumed": [],
        },
    )
    if eligible == 0:
        raise RuntimeError(f"all analysis units failed preprocessing; inspect {flow}")
    return manifest, flow


def _encode_windows(
    raw: Any,
    encoder: OfficialLaBraMEncoder,
    *,
    window_seconds: float,
    step_seconds: float,
) -> tuple[Any, np.ndarray, Any]:
    windows, starts = make_windows(
        raw.get_data(), float(raw.info["sfreq"]), window_seconds, step_seconds
    )
    if len(windows) == 0:
        raise ValueError(f"recording is shorter than the {window_seconds}s encoder window")
    artifact = detect_artifact_windows(windows, float(raw.info["sfreq"]))
    kept = windows[artifact.keep]
    if len(kept) == 0:
        raise ValueError("all encoder windows were rejected as artifacts")
    return encoder.encode(kept, raw.ch_names), starts[artifact.keep], artifact


def _segment_ids_from_retained_starts(
    starts: np.ndarray,
    *,
    expected_step_samples: int,
) -> np.ndarray:
    """Label contiguous retained-window runs after artifact rejection.

    A rejected window creates a discontinuity in the retained start indices.  A
    new segment at every such discontinuity prevents temporal estimators from
    joining observations on opposite sides of an artifact gap.
    """

    retained = np.asarray(starts)
    if retained.ndim != 1 or retained.size == 0:
        raise ValueError("retained starts must be a non-empty one-dimensional sequence")
    if retained.dtype.kind not in "iu":
        if retained.dtype.kind == "f" and np.all(retained == np.floor(retained)):
            retained = retained.astype(np.int64)
        else:
            raise ValueError("retained starts must contain integer sample indices")
    retained = np.asarray(retained, dtype=np.int64)
    if expected_step_samples <= 0:
        raise ValueError("expected step must contain at least one sample")
    if np.any(retained < 0) or np.any(np.diff(retained) <= 0):
        raise ValueError("retained starts must be strictly increasing and non-negative")
    boundaries = np.r_[True, np.diff(retained) != expected_step_samples]
    return (np.cumsum(boundaries, dtype=np.int64) - 1).astype(np.int32)


def _concatenate_event_segment_ids(
    archives: list[dict[str, np.ndarray]],
    *,
    key: str,
    lengths: list[int],
) -> np.ndarray:
    """Preserve every within-trial gap while assigning trial-unique run IDs."""

    combined: list[np.ndarray] = []
    next_segment = 0
    for archive, length in zip(archives, lengths, strict=True):
        source = archive.get(key)
        segments = np.zeros(length, dtype=np.int64) if source is None else np.asarray(source)
        if segments.ndim != 1 or len(segments) != length:
            raise ValueError(f"{key} must align with every event trajectory")
        if segments.dtype.kind == "f" and not np.all(np.isfinite(segments)):
            raise ValueError(f"{key} contains non-finite values")
        if segments.dtype.kind == "O" and any(value is None for value in segments):
            raise ValueError(f"{key} contains None")
        run_starts = np.r_[True, segments[1:] != segments[:-1]]
        local = np.cumsum(run_starts, dtype=np.int64) - 1
        local += next_segment
        combined.append(local)
        next_segment = int(local[-1]) + 1
    return np.concatenate(combined).astype(np.int32)


def _validate_preprocessed_references(
    frame: pd.DataFrame, study: StudyConfig
) -> dict[str, dict[str, Any]]:
    """Recursively rehash every eligible FIF and its content-binding receipt."""

    validated: dict[str, dict[str, Any]] = {}
    for row in frame[frame["eligible"].astype(bool)].to_dict(orient="records"):
        unit_id = str(row["unit_id"])
        if unit_id in validated:
            raise DerivativeIntegrityError(f"duplicate preprocessed unit_id: {unit_id}")
        path_value = row.get("preprocessed_path")
        expected_hash = row.get("preprocessed_sha256")
        receipt_value = row.get("preprocessing_provenance_path")
        expected_receipt_hash = row.get("preprocessing_provenance_sha256")
        if (
            not isinstance(path_value, str)
            or not _valid_sha256(expected_hash)
            or not isinstance(receipt_value, str)
            or not _valid_sha256(expected_receipt_hash)
        ):
            raise DerivativeIntegrityError(
                f"preprocessing manifest fingerprints are incomplete for {unit_id}"
            )
        path = Path(path_value)
        receipt = Path(receipt_value)
        payload = _load_cached_derivative(
            output=path,
            receipt=receipt,
            marker=PREPROCESSED_UNIT_MARKER,
        )
        if payload is None:
            raise DerivativeIntegrityError(f"preprocessed derivative is missing for {unit_id}")
        if payload["output"].get("sha256") != expected_hash:
            raise DerivativeIntegrityError(
                f"preprocessed SHA-256 differs from the manifest for {unit_id}"
            )
        if sha256_file(receipt) != expected_receipt_hash:
            raise DerivativeIntegrityError(
                f"preprocessing provenance SHA-256 differs from the manifest for {unit_id}"
            )
        selector = _selector(row)
        fingerprints = _preprocessing_input_fingerprints(study, selector)
        _require_input_binding(
            payload,
            {"unit_id": unit_id, **fingerprints},
            receipt=receipt,
            exact=False,
        )
        validated[unit_id] = {
            "path": str(path.resolve(strict=True)),
            "sha256": expected_hash,
            "provenance_path": str(receipt.resolve(strict=True)),
            "provenance_sha256": expected_receipt_hash,
            "selector": selector,
        }
    return validated


def _aggregate_event_rows(
    frame: pd.DataFrame,
    *,
    destination: Path,
    minimum_trials: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    selectors = frame["selector_json"].map(json.loads)
    event_mask = selectors.map(lambda item: item["kind"] in {"event_epoch", "pre_epoched"})
    retained = [row for row in frame[~event_mask].to_dict(orient="records")]
    issues: list[dict[str, Any]] = []
    event = frame[event_mask & frame["encoded"].astype(bool)].copy()
    group_columns = ["participant_id", "dataset_id", "condition"]
    for identity, group in event.groupby(group_columns, dropna=False):
        if len(group) < minimum_trials:
            issues.append(
                {
                    "group": dict(zip(group_columns, identity, strict=True)),
                    "status": "insufficient_trials",
                    "observed": len(group),
                    "required": minimum_trials,
                }
            )
            continue
        archives: list[dict[str, np.ndarray]] = []
        trial_inputs: list[dict[str, str]] = []
        for trial in group.to_dict(orient="records"):
            path_value = trial.get("trajectory_path")
            expected_hash = trial.get("trajectory_sha256")
            receipt_value = trial.get("encoding_provenance_path")
            expected_receipt_hash = trial.get("encoding_provenance_sha256")
            if (
                not isinstance(path_value, str)
                or not _valid_sha256(expected_hash)
                or not isinstance(receipt_value, str)
                or not _valid_sha256(expected_receipt_hash)
            ):
                raise DerivativeIntegrityError("event trial encoding fingerprints are incomplete")
            path = Path(path_value)
            receipt = Path(receipt_value)
            cached_trial = _load_cached_derivative(
                output=path,
                receipt=receipt,
                marker=ENCODED_UNIT_MARKER,
            )
            if cached_trial is None or cached_trial["output"].get("sha256") != expected_hash:
                raise DerivativeIntegrityError(
                    f"event trial encoding receipt is not bound: {trial.get('unit_id')}"
                )
            if sha256_file(receipt) != expected_receipt_hash:
                raise DerivativeIntegrityError(
                    f"event trial encoding changed before aggregation: {trial.get('unit_id')}"
                )
            path = path.resolve(strict=True)
            receipt = receipt.resolve(strict=True)
            trial_inputs.append(
                {
                    "unit_id": str(trial.get("unit_id")),
                    "trajectory_path": str(path),
                    "trajectory_sha256": expected_hash,
                    "encoding_provenance_path": str(receipt),
                    "encoding_provenance_sha256": expected_receipt_hash,
                }
            )
            with np.load(path, allow_pickle=False) as source:
                archives.append({name: np.asarray(source[name]) for name in source.files})
        region_names = set(name for name in archives[0] if name.startswith("alignment_regional_"))
        for archive in archives[1:]:
            region_names.intersection_update(
                name for name in archive if name.startswith("alignment_regional_")
            )
        if len(region_names) < 2:
            issues.append(
                {
                    "group": dict(zip(group_columns, identity, strict=True)),
                    "status": "insufficient_shared_regions",
                }
            )
            continue
        trajectories = [archive["alignment_global_states"] for archive in archives]
        lengths = [len(value) for value in trajectories]
        segments = _concatenate_event_segment_ids(
            archives,
            key="alignment_segment_ids",
            lengths=lengths,
        )
        arrays: dict[str, np.ndarray] = {
            "global_states": np.concatenate(trajectories).astype(np.float32),
            "alignment_global_states": np.concatenate(trajectories).astype(np.float32),
            "segment_ids": segments,
            "alignment_segment_ids": segments,
        }
        for encoded_name in sorted(region_names):
            name = encoded_name.removeprefix("alignment_")
            values = np.concatenate([archive[encoded_name] for archive in archives]).astype(
                np.float32
            )
            arrays[name] = values
            arrays[encoded_name] = values
        group_key = _safe_key("|".join(str(value) for value in identity))
        trajectory = destination / "trajectories" / f"group-{group_key}.npz"
        encoding_receipt = destination / "provenance" / f"group-{group_key}.encoding.json"
        aggregate_inputs: dict[str, Any] = {
            "group_identity": dict(zip(group_columns, identity, strict=True)),
            "minimum_trials": minimum_trials,
            "trials": trial_inputs,
            "alignment_regions": sorted(region_names),
            "aggregation_config_sha256": _payload_sha256(
                {
                    "minimum_trials": minimum_trials,
                    "global_track": "alignment_global_states",
                    "segment_track": "alignment_segment_ids",
                    "regions": sorted(region_names),
                }
            ),
        }
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        cached_group = _load_cached_derivative(
            output=trajectory,
            receipt=encoding_receipt,
            marker=ENCODED_EVENT_GROUP_MARKER,
        )
        if cached_group is not None:
            _require_input_binding(cached_group, aggregate_inputs, receipt=encoding_receipt)
            trajectory_hash = str(cached_group["output"]["sha256"])
        else:
            temporary = trajectory.with_name(f".{trajectory.name}.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, trajectory)
            group_receipt = _write_derivative_receipt(
                receipt=encoding_receipt,
                marker=ENCODED_EVENT_GROUP_MARKER,
                inputs=aggregate_inputs,
                output=trajectory,
                metadata={
                    "trial_count": len(group),
                    "temporal_segments": int(segments[-1]) + 1,
                    "coarse_windows": int(sum(lengths)),
                    "alignment_windows": int(sum(lengths)),
                },
            )
            trajectory_hash = str(group_receipt["output"]["sha256"])
        first = group.iloc[0].to_dict()
        row: dict[str, Any] = {
            key: value
            for key, value in first.items()
            if key
            not in {
                "unit_id",
                "trajectory_path",
                "trajectory_sha256",
                "encoding_provenance_path",
                "encoding_provenance_sha256",
                "source_path",
                "source_file",
                "preprocessed_path",
            }
        }
        for column in group.columns:
            if column in row or column in {
                "unit_id",
                "trajectory_path",
                "trajectory_sha256",
                "encoding_provenance_path",
                "encoding_provenance_sha256",
                "source_path",
                "source_file",
                "preprocessed_path",
            }:
                continue
            numeric = pd.to_numeric(group[column], errors="coerce")
            if numeric.notna().all():
                row[column] = float(numeric.mean())
        row.update(
            {
                "unit_id": group_key,
                "trajectory_path": str(trajectory),
                "trajectory_sha256": trajectory_hash,
                "encoding_provenance_path": str(encoding_receipt),
                "encoding_provenance_sha256": sha256_file(encoding_receipt),
                "encoded": True,
                "encoding_error": None,
                "trial_count": len(group),
                "event_aggregated": True,
                "temporal_segments": int(segments[-1]) + 1,
                "coarse_windows": int(sum(lengths)),
                "alignment_windows": int(sum(lengths)),
            }
        )
        retained.append(row)
    return pd.DataFrame(retained), issues


def encode_analysis_units(
    *,
    preprocessing_manifest: str | Path,
    labels_manifest: str | Path,
    output_root: str | Path,
    study: StudyConfig,
) -> tuple[Path, Path]:
    """Encode signal-only units and attach labels only after checkpoint inference."""

    frame = pd.read_parquet(preprocessing_manifest)
    assert_no_direct_tms(frame, stage="general encoder input")
    required = {
        "unit_id",
        "eligible",
        "selector_json",
        "preprocessed_path",
        "preprocessed_sha256",
        "preprocessing_provenance_path",
        "preprocessing_provenance_sha256",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise DerivativeIntegrityError(
            f"preprocessing manifest is missing integrity fields {sorted(missing)}"
        )
    validated_preprocessing = _validate_preprocessed_references(frame, study)
    try:
        import mne
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install neural-manifolds[eeg]") from exc
    repository, checkpoint, checkpoint_hash = _model_environment()
    model_fingerprint = _model_fingerprint(repository, checkpoint, checkpoint_hash)
    encoding_fingerprints = _encoding_input_fingerprints(study)
    encoder: OfficialLaBraMEncoder | None = None
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reused_units = 0
    generated_units = 0
    for row in frame[frame["eligible"].astype(bool)].to_dict(orient="records"):
        unit_id = str(row["unit_id"])
        preprocessed = validated_preprocessing[unit_id]
        result = {
            "unit_id": unit_id,
            "encoded": False,
            "trajectory_path": None,
            "trajectory_sha256": None,
            "preprocessed_path": preprocessed["path"],
            "preprocessed_sha256": preprocessed["sha256"],
            "analysis_branch": row.get("analysis_branch", STANDARD_HARMONISED_BRANCH),
            "property_scope_json": row.get("property_scope_json"),
            "spatial_regions": row.get("spatial_regions"),
        }
        try:
            selector = preprocessed["selector"]
            trajectory = destination / "trajectories" / f"{unit_id}.npz"
            encoding_receipt = destination / "provenance" / f"{unit_id}.encoding.json"
            encoding_inputs: dict[str, Any] = {
                "unit_id": unit_id,
                "preprocessed_path": preprocessed["path"],
                "preprocessed_sha256": preprocessed["sha256"],
                "preprocessing_provenance_path": preprocessed["provenance_path"],
                "preprocessing_provenance_sha256": preprocessed["provenance_sha256"],
                "selector_sha256": _payload_sha256(selector),
                **encoding_fingerprints,
                "model": model_fingerprint,
            }
            cached = _load_cached_derivative(
                output=trajectory,
                receipt=encoding_receipt,
                marker=ENCODED_UNIT_MARKER,
            )
            required_metadata = {
                "coarse_windows",
                "alignment_windows",
                "coarse_segments",
                "alignment_segments",
                "rejected_coarse_windows",
                "rejected_alignment_windows",
                "minimum_valid_windows_required",
            }
            if cached is not None:
                _require_input_binding(cached, encoding_inputs, receipt=encoding_receipt)
                metadata = cached.get("metadata")
                if not isinstance(metadata, dict) or not required_metadata <= set(metadata):
                    raise DerivativeIntegrityError(
                        f"cached encoding metadata is incomplete: {encoding_receipt}"
                    )
                trajectory_hash = str(cached["output"]["sha256"])
                reused_units += 1
            else:
                if encoder is None:
                    encoder = OfficialLaBraMEncoder(
                        repository=repository,
                        factory="modeling_finetune:labram_base_patch200_200",
                        checkpoint=checkpoint,
                        checkpoint_sha256=checkpoint_hash,
                        device="cuda",
                    )
                raw = mne.io.read_raw_fif(Path(preprocessed["path"]), preload=True, verbose="ERROR")
                is_event = selector["kind"] in {"event_epoch", "pre_epoched"}
                if is_event:
                    encoded, starts, artifact = _encode_windows(
                        raw,
                        encoder,
                        window_seconds=study.representation.alignment_window_seconds,
                        step_seconds=study.representation.alignment_step_seconds,
                    )
                    coarse = encoded
                    coarse_starts = starts
                    coarse_artifact = artifact
                    required_windows: int | None = None
                else:
                    coarse, coarse_starts, coarse_artifact = _encode_windows(
                        raw,
                        encoder,
                        window_seconds=study.representation.harmonised_window_seconds,
                        step_seconds=study.representation.harmonised_step_seconds,
                    )
                    encoded, starts, artifact = _encode_windows(
                        raw,
                        encoder,
                        window_seconds=study.representation.alignment_window_seconds,
                        step_seconds=study.representation.alignment_step_seconds,
                    )
                    planned_interval = (
                        float(selector["stop_seconds"]) - float(selector["start_seconds"])
                        if selector["kind"] == "interval_seconds"
                        else None
                    )
                    required_windows = (
                        min(10, study.preprocessing.minimum_valid_windows)
                        if planned_interval is not None
                        and planned_interval < study.preprocessing.minimum_rest_seconds
                        else study.preprocessing.minimum_valid_windows
                    )
                    if len(coarse.global_states) < required_windows:
                        raise ValueError(
                            f"only {len(coarse.global_states)} coarse windows; need "
                            f"{required_windows}"
                        )
                sampling_hz = float(raw.info["sfreq"])
                coarse_step_seconds = (
                    study.representation.alignment_step_seconds
                    if is_event
                    else study.representation.harmonised_step_seconds
                )
                coarse_segment_ids = _segment_ids_from_retained_starts(
                    coarse_starts,
                    expected_step_samples=round(coarse_step_seconds * sampling_hz),
                )
                alignment_segment_ids = _segment_ids_from_retained_starts(
                    starts,
                    expected_step_samples=round(
                        study.representation.alignment_step_seconds * sampling_hz
                    ),
                )
                arrays: dict[str, np.ndarray] = {
                    "global_states": np.asarray(coarse.global_states, dtype=np.float32),
                    "window_start_samples": coarse_starts,
                    "alignment_global_states": np.asarray(encoded.global_states, dtype=np.float32),
                    "alignment_window_start_samples": starts,
                    "segment_ids": coarse_segment_ids,
                    "alignment_segment_ids": alignment_segment_ids,
                }
                arrays.update(
                    {
                        f"regional_{name}": np.asarray(value, dtype=np.float32)
                        for name, value in coarse.regional_states.items()
                    }
                )
                arrays.update(
                    {
                        f"alignment_regional_{name}": np.asarray(value, dtype=np.float32)
                        for name, value in encoded.regional_states.items()
                    }
                )
                if sha256_file(preprocessed["path"]) != preprocessed["sha256"]:
                    raise DerivativeIntegrityError(
                        f"preprocessed derivative changed during encoding: {preprocessed['path']}"
                    )
                trajectory.parent.mkdir(parents=True, exist_ok=True)
                temporary = trajectory.with_name(f".{trajectory.name}.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                os.replace(temporary, trajectory)
                metadata = {
                    "coarse_windows": len(coarse.global_states),
                    "alignment_windows": len(encoded.global_states),
                    "coarse_segments": int(coarse_segment_ids[-1]) + 1,
                    "alignment_segments": int(alignment_segment_ids[-1]) + 1,
                    "rejected_coarse_windows": int(np.count_nonzero(~coarse_artifact.keep)),
                    "rejected_alignment_windows": int(np.count_nonzero(~artifact.keep)),
                    "minimum_valid_windows_required": required_windows,
                    "label_fields_consumed": [],
                }
                encoding_payload = _write_derivative_receipt(
                    receipt=encoding_receipt,
                    marker=ENCODED_UNIT_MARKER,
                    inputs=encoding_inputs,
                    output=trajectory,
                    metadata=metadata,
                )
                trajectory_hash = str(encoding_payload["output"]["sha256"])
                generated_units += 1
            result.update(
                {
                    "trajectory_path": str(trajectory),
                    "trajectory_sha256": trajectory_hash,
                    "encoding_provenance_path": str(encoding_receipt),
                    "encoding_provenance_sha256": sha256_file(encoding_receipt),
                    "encoded": True,
                    "encoding_error": None,
                    **{key: metadata[key] for key in required_metadata},
                    "label_fields_consumed": "",
                }
            )
        except DerivativeIntegrityError:
            raise
        except (ValueError, RuntimeError, OSError) as error:
            result["encoding_error"] = f"{type(error).__name__}: {error}"
        rows.append(result)
    signal = pd.DataFrame(rows)
    labels = pd.read_parquet(labels_manifest)
    if set(signal["unit_id"]) - set(labels["unit_id"]):
        raise RuntimeError("encoded units are absent from the post-encoding label manifest")
    joined = signal.merge(labels, on="unit_id", how="left", validate="one_to_one")
    joined = ensure_pretraining_overlap_columns(joined, default_model_id="labram_base")
    joined, aggregation_issues = _aggregate_event_rows(
        joined,
        destination=destination,
        minimum_trials=study.preprocessing.minimum_event_trials_per_condition,
    )
    manifest = _atomic_parquet(joined, destination / "encoding-manifest.parquet")
    flow = destination / "encoding-flow.json"
    encoded_count = int(signal["encoded"].sum()) if len(signal) else 0
    derivative_inventory = [
        _artifact(path)
        for path in derivative_artifact_paths(
            destination, directories=("trajectories", "provenance")
        )
    ]
    atomic_write_json(
        flow,
        {
            "schema_version": 1,
            "eligible_units": len(signal),
            "encoded_units": encoded_count,
            "failed_units": len(signal) - encoded_count,
            "checkpoint_sha256": checkpoint_hash,
            "model_fingerprint_sha256": _payload_sha256(model_fingerprint),
            "reused_unit_derivatives": reused_units,
            "generated_unit_derivatives": generated_units,
            "derivative_inventory": derivative_inventory,
            "labels_joined_after_encoding": True,
            "encoder_label_fields_consumed": [],
            "pretraining_overlap": summarize_pretraining_overlap(joined),
            "event_aggregation_issues": aggregation_issues,
            "errors": [
                {"unit_id": row["unit_id"], "error": row.get("encoding_error")}
                for row in rows
                if not row["encoded"]
            ],
        },
    )
    if encoded_count == 0:
        raise RuntimeError(f"no analysis units encoded; inspect {flow}")
    return manifest, flow
