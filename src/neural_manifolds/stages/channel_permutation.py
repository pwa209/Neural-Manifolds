"""True pre-encoder EEG sensor-row permutation control.

For every repeat, EEG signal rows are permuted while the original ordered
channel-name list is retained.  The mismatched signal/name pair is then passed
to the frozen LaBraM encoder.  This is deliberately distinct from rotating or
permuting an already encoded latent trajectory.

The stage has a hard two-phase label firewall:

1. all missing repeats are encoded from the label-free preprocessing manifest;
2. only after signal inference is complete is the label manifest opened for
   event-trial aggregation and scalar profile publication.

Each repeat is atomically published with content-hashed completion metadata.
Completed repeats are validated and reused; incomplete work directories are
never interpreted as results.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

from neural_manifolds.adapters.models import LABEL_FIELDS
from neural_manifolds.config import StudyConfig, config_sha256
from neural_manifolds.dynamics.state_dictionary import StateDictionary
from neural_manifolds.foundation.base import EncodedBatch, validate_frozen_model
from neural_manifolds.foundation.labram import OfficialLaBraMEncoder
from neural_manifolds.manifold.profile import AXIS_NAMES, FiveAxisProfileEstimator, ManifoldRecord
from neural_manifolds.preprocessing.eeg import detect_artifact_windows, make_windows
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stage_processing import _model_environment

FAMILY = "pre_encoder_sensor_row_permutation"
NULL_PROFILE_COLUMNS = (
    "unit_id",
    "participant_id",
    "dataset_id",
    "family",
    "repeat",
    "seed",
    *AXIS_NAMES,
)
COMPLETION_FILE = "COMPLETE.json"
SIGNAL_MANIFEST_FILE = "signal-manifest.parquet"
SIGNAL_AUDIT_FILE = "signal-audit.json"
REPEAT_PROFILES_FILE = "profiles.parquet"
REPEAT_AUDIT_FILE = "repeat-audit.json"

_PREENCODER_LABEL_COLUMNS = frozenset(LABEL_FIELDS).union(
    {
        "acquisition",
        "behavioral_responsiveness",
        "condition",
        "content",
        "dataset_id",
        "diagnosis",
        "drug",
        "group",
        "healthy_wake_reference",
        "outcome",
        "participant_id",
        "report",
        "response",
        "target",
        "task",
        "trial_type",
    }
)

_PUBLISHED_LABEL_COLUMNS = (
    "participant_id",
    "dataset_id",
    "condition",
    "modality",
    "healthy_wake_reference",
    "clinical_holdout",
    "explanatory_target",
    "task_relevance",
    "content",
    "metadata_status",
    "acquisition",
)


class ChannelPermutationError(RuntimeError):
    """Raised when the pre-encoder permutation control cannot remain auditable."""


class LaBraMBackend(Protocol):
    """Minimal frozen LaBraM inference contract used by this stage."""

    def encode(
        self,
        windows_volts: np.ndarray,
        channel_names: Sequence[str],
        *,
        metadata_fields: Sequence[str] = (),
    ) -> EncodedBatch: ...


@dataclass(frozen=True)
class ChannelPermutationArtifacts:
    """Published scalar control outputs and repeat cache root."""

    profiles_path: Path
    audit_path: Path
    repeats_root: Path
    repeats: int
    profile_rows: int


@dataclass(frozen=True)
class _EncodedTracks:
    global_states: np.ndarray
    regional_states: Mapping[str, np.ndarray]
    alignment_regional_states: Mapping[str, np.ndarray]
    window_start_samples: np.ndarray
    alignment_window_start_samples: np.ndarray
    rejected_coarse_windows: int
    rejected_alignment_windows: int
    encoder_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _ProfileSource:
    profile_id: str
    source_unit_ids: tuple[str, ...]
    participant_id: str
    dataset_id: str
    condition: str
    modality: str
    trajectory: np.ndarray
    regional: Mapping[str, np.ndarray]
    segment_ids: np.ndarray
    alignment_segment_ids: np.ndarray
    labels: Mapping[str, Any]
    event_aggregated: bool
    trial_count: int


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary_name, index=False)
        os.replace(temporary_name, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return destination


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChannelPermutationError(f"cannot read valid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ChannelPermutationError(f"{label} must be a JSON object: {path}")
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _derived_seed(base_seed: int, *, repeat: int, unit_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{repeat}:{unit_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _repeat_seed(study: StudyConfig, repeat: int) -> int:
    if not study.random_seeds:
        raise ChannelPermutationError("study configuration contains no deterministic seeds")
    base = int(study.random_seeds[repeat % len(study.random_seeds)])
    return int((base + 1_000_003 * repeat) % (2**32 - 1))


def _selector_kind(value: object) -> str:
    if not isinstance(value, str):
        raise ChannelPermutationError("preprocessing row has no serialized selector")
    try:
        selector = json.loads(value)
    except json.JSONDecodeError as error:
        raise ChannelPermutationError("serialized selector is invalid JSON") from error
    if not isinstance(selector, dict) or not isinstance(selector.get("kind"), str):
        raise ChannelPermutationError("serialized selector is malformed")
    kind = selector["kind"]
    allowed = {"full_recording", "interval_seconds", "event_epoch", "pre_epoched"}
    if kind not in allowed:
        raise ChannelPermutationError(f"unsupported EEG selector for this control: {kind!r}")
    return kind


def _read_preprocessed_raw(path: Path) -> Any:
    try:
        import mne
    except ImportError as error:  # pragma: no cover - EEG deployment environment
        raise ChannelPermutationError("install neural-manifolds[eeg]") from error
    return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")


def _safe_encoder_metadata(backend: LaBraMBackend) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "class": f"{type(backend).__module__}.{type(backend).__qualname__}",
    }
    declared = getattr(backend, "metadata", None)
    if isinstance(declared, Mapping):
        metadata["declared"] = {
            str(key): value
            for key, value in declared.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
    checkpoint_hash = getattr(backend, "checkpoint_hash", None)
    if isinstance(checkpoint_hash, str):
        metadata["checkpoint_sha256"] = checkpoint_hash
    repository = getattr(backend, "repository", None)
    if repository is not None:
        metadata["repository"] = str(repository)
    metadata["sha256"] = _canonical_hash(metadata)
    return metadata


def _validate_frozen_backend(backend: LaBraMBackend) -> None:
    if isinstance(backend, OfficialLaBraMEncoder):
        validate_frozen_model(backend.model)
        if not isinstance(backend.checkpoint_hash, str) or len(backend.checkpoint_hash) != 64:
            raise ChannelPermutationError("official LaBraM backend lacks a checkpoint SHA-256")
        return
    declared = getattr(backend, "metadata", None)
    if not isinstance(declared, Mapping) or declared.get("weights_frozen") is not True:
        raise ChannelPermutationError(
            "an injected LaBraM backend must declare metadata['weights_frozen']=True"
        )


def _build_official_encoder(*, device: str) -> OfficialLaBraMEncoder:
    repository, checkpoint, checkpoint_hash = _model_environment()
    return OfficialLaBraMEncoder(
        repository=repository,
        factory="modeling_finetune:labram_base_patch200_200",
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        device=device,
    )


def _reset_private_work_directory(path: Path, *, parent: Path) -> None:
    if path.parent != parent or not path.name.startswith("."):
        raise ChannelPermutationError(f"refusing to reset non-private work directory: {path}")
    if path.is_symlink():
        raise ChannelPermutationError(f"private work directory cannot be a symlink: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _validate_preprocessing_manifest(
    path: Path, *, modalities: tuple[str, ...]
) -> tuple[pd.DataFrame, dict[str, int]]:
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as error:
        raise ChannelPermutationError(f"cannot read preprocessing manifest: {path}") from error
    prohibited = sorted(set(frame.columns).intersection(_PREENCODER_LABEL_COLUMNS))
    if prohibited:
        raise ChannelPermutationError(
            f"label fields are forbidden in the pre-encoder manifest: {prohibited}"
        )
    required = {
        "eligible",
        "modality",
        "preprocessed_path",
        "preprocessed_sha256",
        "selector_json",
        "unit_id",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ChannelPermutationError(f"preprocessing manifest is missing {missing}")
    if frame["unit_id"].isna().any() or frame["unit_id"].astype(str).duplicated().any():
        raise ChannelPermutationError("preprocessing manifest unit IDs must be nonempty and unique")
    eligible = frame[frame["eligible"].astype(bool)].copy()
    counts = {
        str(name): int(count)
        for name, count in eligible["modality"].astype(str).value_counts().items()
    }
    selected = eligible[eligible["modality"].astype(str).isin(modalities)].copy()
    if selected.empty:
        raise ChannelPermutationError(
            f"no eligible units have requested EEG modalities {list(modalities)}; observed={counts}"
        )
    selected["unit_id"] = selected["unit_id"].astype(str)
    selected["selector_kind"] = selected["selector_json"].map(_selector_kind)
    return selected.sort_values("unit_id", kind="stable").reset_index(drop=True), counts


def _permutation(channel_count: int, *, seed: int) -> np.ndarray:
    if channel_count < 2:
        raise ChannelPermutationError("sensor-row permutation requires at least two EEG channels")
    order = np.random.default_rng(seed).permutation(channel_count)
    if np.array_equal(order, np.arange(channel_count)):
        order = np.roll(order, 1)
    if sorted(order.tolist()) != list(range(channel_count)):
        raise AssertionError("generated sensor-row order is not a permutation")
    return np.asarray(order, dtype=np.int64)


def _validate_encoded_batch(encoded: EncodedBatch, *, expected_windows: int, label: str) -> None:
    global_states = np.asarray(encoded.global_states)
    if (
        global_states.ndim != 2
        or global_states.shape[0] != expected_windows
        or global_states.shape[1] < 2
        or not np.all(np.isfinite(global_states))
    ):
        raise ChannelPermutationError(f"malformed {label} global LaBraM trajectory")
    regional = {str(name): np.asarray(values) for name, values in encoded.regional_states.items()}
    if len(regional) < 2:
        raise ChannelPermutationError(f"{label} LaBraM output has fewer than two regions")
    for name, values in regional.items():
        if (
            values.ndim != 2
            or values.shape[0] != expected_windows
            or not np.all(np.isfinite(values))
        ):
            raise ChannelPermutationError(f"malformed {label} regional trajectory: {name}")


def _encode_window_track(
    signal: np.ndarray,
    *,
    sampling_hz: float,
    window_seconds: float,
    step_seconds: float,
    channel_names: tuple[str, ...],
    encoder: LaBraMBackend,
    label: str,
) -> tuple[EncodedBatch, np.ndarray, int]:
    windows, starts = make_windows(signal, sampling_hz, window_seconds, step_seconds)
    if len(windows) == 0:
        raise ChannelPermutationError(
            f"signal is shorter than the {window_seconds}s {label} encoder window"
        )
    artifact = detect_artifact_windows(windows, sampling_hz)
    keep = np.asarray(artifact.keep, dtype=bool)
    if keep.shape != (len(windows),):
        raise ChannelPermutationError(f"artifact mask is malformed for {label} windows")
    retained = windows[keep]
    if len(retained) == 0:
        raise ChannelPermutationError(f"all {label} encoder windows were rejected")
    # The signal row order has changed, but channel_names intentionally has not.
    encoded = encoder.encode(retained, channel_names, metadata_fields=())
    _validate_encoded_batch(encoded, expected_windows=len(retained), label=label)
    return encoded, np.asarray(starts[keep], dtype=np.int64), int(np.count_nonzero(~keep))


def _encode_permuted_unit(
    raw: Any,
    *,
    selector_kind: str,
    encoder: LaBraMBackend,
    study: StudyConfig,
    seed: int,
) -> tuple[_EncodedTracks, dict[str, Any]]:
    try:
        signal = np.asarray(raw.get_data(), dtype=np.float64)
        sampling_hz = float(raw.info["sfreq"])
        channel_names = tuple(str(name) for name in raw.ch_names)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ChannelPermutationError(
            "preprocessed EEG object lacks data/channels/sampling rate"
        ) from error
    if signal.ndim != 2 or signal.shape[0] != len(channel_names) or signal.shape[1] < 2:
        raise ChannelPermutationError("preprocessed EEG must be channels x samples")
    if any(not name for name in channel_names) or len(set(channel_names)) != len(channel_names):
        raise ChannelPermutationError("preprocessed EEG channel names must be nonempty and unique")
    if not np.all(np.isfinite(signal)) or not np.isfinite(sampling_hz) or sampling_hz <= 0:
        raise ChannelPermutationError("preprocessed EEG contains invalid samples or sampling rate")
    expected_sampling_hz = float(study.preprocessing.target_sampling_hz)
    if not np.isclose(sampling_hz, expected_sampling_hz, rtol=0.0, atol=1e-8):
        raise ChannelPermutationError(
            f"preprocessed EEG sampling rate is {sampling_hz:g} Hz; expected "
            f"{expected_sampling_hz:g} Hz"
        )
    if not np.isclose(expected_sampling_hz, 200.0, rtol=0.0, atol=1e-8):
        raise ChannelPermutationError("the pinned LaBraM encoder contract requires 200 Hz EEG")
    order = _permutation(signal.shape[0], seed=seed)
    permuted_signal = signal[order, :]
    event_trial = selector_kind in {"event_epoch", "pre_epoched"}

    if event_trial:
        alignment, alignment_starts, rejected_alignment = _encode_window_track(
            permuted_signal,
            sampling_hz=sampling_hz,
            window_seconds=study.representation.alignment_window_seconds,
            step_seconds=study.representation.alignment_step_seconds,
            channel_names=channel_names,
            encoder=encoder,
            label="event-trial",
        )
        coarse = alignment
        coarse_starts = alignment_starts
        rejected_coarse = rejected_alignment
    else:
        coarse, coarse_starts, rejected_coarse = _encode_window_track(
            permuted_signal,
            sampling_hz=sampling_hz,
            window_seconds=study.representation.harmonised_window_seconds,
            step_seconds=study.representation.harmonised_step_seconds,
            channel_names=channel_names,
            encoder=encoder,
            label="coarse",
        )
        if len(coarse.global_states) < study.preprocessing.minimum_valid_windows:
            raise ChannelPermutationError(
                f"only {len(coarse.global_states)} coarse windows; need "
                f"{study.preprocessing.minimum_valid_windows}"
            )
        alignment, alignment_starts, rejected_alignment = _encode_window_track(
            permuted_signal,
            sampling_hz=sampling_hz,
            window_seconds=study.representation.alignment_window_seconds,
            step_seconds=study.representation.alignment_step_seconds,
            channel_names=channel_names,
            encoder=encoder,
            label="alignment",
        )
    permutation_hash = hashlib.sha256(order.astype("<i8", copy=False).tobytes()).hexdigest()
    channel_hash = hashlib.sha256("\0".join(channel_names).encode()).hexdigest()
    audit = {
        "seed": seed,
        "channel_count": len(channel_names),
        "channel_names_sha256": channel_hash,
        "permutation_sha256": permutation_hash,
        "permutation_nonidentity": not np.array_equal(order, np.arange(len(order))),
        "signal_rows_permuted": True,
        "channel_names_reordered": False,
        "channel_names_passed_unchanged": True,
        "label_fields_consumed": [],
        "selector_kind": selector_kind,
        "event_trial_encoded_independently": event_trial,
    }
    tracks = _EncodedTracks(
        global_states=np.asarray(coarse.global_states, dtype=np.float32),
        regional_states={
            str(name): np.asarray(values, dtype=np.float32)
            for name, values in coarse.regional_states.items()
        },
        alignment_regional_states={
            str(name): np.asarray(values, dtype=np.float32)
            for name, values in alignment.regional_states.items()
        },
        window_start_samples=coarse_starts,
        alignment_window_start_samples=alignment_starts,
        rejected_coarse_windows=rejected_coarse,
        rejected_alignment_windows=rejected_alignment,
        encoder_metadata=dict(coarse.metadata),
    )
    return tracks, audit


def _write_trajectory(path: Path, tracks: _EncodedTracks) -> Path:
    arrays: dict[str, np.ndarray] = {
        "global_states": tracks.global_states,
        "window_start_samples": tracks.window_start_samples,
        "alignment_window_start_samples": tracks.alignment_window_start_samples,
        "segment_ids": np.zeros(len(tracks.global_states), dtype=np.int32),
        "alignment_segment_ids": np.zeros(
            len(next(iter(tracks.alignment_regional_states.values()))), dtype=np.int32
        ),
    }
    arrays.update({f"regional_{name}": values for name, values in tracks.regional_states.items()})
    arrays.update(
        {
            f"alignment_regional_{name}": values
            for name, values in tracks.alignment_regional_states.items()
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    return path


def _signal_expected(
    *,
    repeat: int,
    repeat_seed: int,
    preprocessing_sha256: str,
    study_sha256: str,
    encoder_sha256: str,
    modalities: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "family": FAMILY,
        "repeat": repeat,
        "repeat_seed": repeat_seed,
        "preprocessing_manifest_sha256": preprocessing_sha256,
        "study_sha256": study_sha256,
        "encoder_sha256": encoder_sha256,
        "modalities": list(modalities),
        "labels_consumed": False,
    }


def _validate_completed_directory(
    directory: Path,
    *,
    expected: Mapping[str, Any],
    data_file: str,
    audit_file: str,
) -> tuple[Path, Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise ChannelPermutationError(f"completed repeat is not a safe directory: {directory}")
    marker_path = directory / COMPLETION_FILE
    marker = _read_json_object(marker_path, label="repeat completion marker")
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ChannelPermutationError(
                f"completed repeat fingerprint mismatch for {directory.name}.{key}: "
                f"{marker.get(key)!r} != {value!r}"
            )
    data_path = directory / data_file
    audit_path = directory / audit_file
    if not data_path.is_file() or not audit_path.is_file():
        raise ChannelPermutationError(f"completed repeat is missing outputs: {directory}")
    if marker.get("data_sha256") != sha256_file(data_path):
        raise ChannelPermutationError(f"completed repeat data checksum mismatch: {data_path}")
    if marker.get("audit_sha256") != sha256_file(audit_path):
        raise ChannelPermutationError(f"completed repeat audit checksum mismatch: {audit_path}")
    return data_path, audit_path


def _prepare_signal_repeat(
    *,
    frame: pd.DataFrame,
    preprocessing_manifest: Path,
    signals_root: Path,
    repeat: int,
    repeat_seed: int,
    study: StudyConfig,
    study_sha256: str,
    encoder: LaBraMBackend,
    encoder_metadata: Mapping[str, Any],
    modalities: tuple[str, ...],
    raw_loader: Callable[[Path], Any],
) -> Path:
    final = signals_root / f"repeat-{repeat:04d}"
    expected = _signal_expected(
        repeat=repeat,
        repeat_seed=repeat_seed,
        preprocessing_sha256=sha256_file(preprocessing_manifest),
        study_sha256=study_sha256,
        encoder_sha256=str(encoder_metadata["sha256"]),
        modalities=modalities,
    )
    if final.exists():
        _validate_completed_directory(
            final,
            expected=expected,
            data_file=SIGNAL_MANIFEST_FILE,
            audit_file=SIGNAL_AUDIT_FILE,
        )
        return final

    signals_root.mkdir(parents=True, exist_ok=True)
    work = signals_root / f".repeat-{repeat:04d}.work"
    _reset_private_work_directory(work, parent=signals_root)
    rows: list[dict[str, Any]] = []
    unit_audit: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        unit_id = str(row["unit_id"])
        try:
            source = Path(str(row["preprocessed_path"])).resolve(strict=True)
        except FileNotFoundError as error:
            raise ChannelPermutationError(
                f"preprocessed EEG does not exist for {unit_id}"
            ) from error
        expected_source_hash = row.get("preprocessed_sha256")
        if not isinstance(expected_source_hash, str) or len(expected_source_hash) != 64:
            raise ChannelPermutationError(f"preprocessed SHA-256 is missing for {unit_id}")
        source_hash = sha256_file(source)
        if source_hash != expected_source_hash:
            raise ChannelPermutationError(f"preprocessed checksum mismatch for {unit_id}")
        unit_seed = _derived_seed(repeat_seed, repeat=repeat, unit_id=unit_id)
        raw = raw_loader(source)
        tracks, permutation_audit = _encode_permuted_unit(
            raw,
            selector_kind=str(row["selector_kind"]),
            encoder=encoder,
            study=study,
            seed=unit_seed,
        )
        # Unit IDs are provenance identifiers, not safe filesystem names (global
        # prefixes commonly contain ':' and BIDS-derived IDs may contain '/').
        trajectory_name = f"{_profile_identifier((unit_id,))}.npz"
        trajectory = _write_trajectory(work / "trajectories" / trajectory_name, tracks)
        rows.append(
            {
                "unit_id": unit_id,
                "modality": str(row["modality"]),
                "selector_kind": str(row["selector_kind"]),
                "trajectory_path": trajectory.relative_to(work).as_posix(),
                "trajectory_sha256": sha256_file(trajectory),
                "coarse_windows": len(tracks.global_states),
                "alignment_windows": len(next(iter(tracks.alignment_regional_states.values()))),
            }
        )
        unit_audit.append(
            {
                "unit_id": unit_id,
                "preprocessed_sha256": source_hash,
                "trajectory_sha256": rows[-1]["trajectory_sha256"],
                "coarse_windows": rows[-1]["coarse_windows"],
                "alignment_windows": rows[-1]["alignment_windows"],
                "rejected_coarse_windows": tracks.rejected_coarse_windows,
                "rejected_alignment_windows": tracks.rejected_alignment_windows,
                "encoder_metadata": dict(tracks.encoder_metadata),
                **permutation_audit,
            }
        )
    manifest_path = _atomic_parquet(pd.DataFrame(rows), work / SIGNAL_MANIFEST_FILE)
    audit_path = work / SIGNAL_AUDIT_FILE
    atomic_write_json(
        audit_path,
        {
            **expected,
            "signal_units": len(rows),
            "encoder": dict(encoder_metadata),
            "unit_audit": unit_audit,
            "label_fields_consumed": [],
            "channel_names_retained_after_signal_row_permutation": True,
        },
    )
    atomic_write_json(
        work / COMPLETION_FILE,
        {
            **expected,
            "data_sha256": sha256_file(manifest_path),
            "audit_sha256": sha256_file(audit_path),
        },
    )
    os.replace(work, final)
    return final


def _load_trajectory(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    source = path.resolve(strict=True)
    if sha256_file(source) != expected_sha256:
        raise ChannelPermutationError(f"signal trajectory checksum mismatch: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if "global_states" not in archive:
            raise ChannelPermutationError(f"signal trajectory lacks global_states: {source}")
        global_states = np.asarray(archive["global_states"], dtype=np.float64)
        regional = {
            name.removeprefix("alignment_regional_"): np.asarray(archive[name], dtype=np.float64)
            for name in archive.files
            if name.startswith("alignment_regional_")
        }
    if global_states.ndim != 2 or global_states.shape[0] < 3:
        raise ChannelPermutationError(f"signal trajectory is too short or malformed: {source}")
    if len(regional) < 2 or len({len(value) for value in regional.values()}) != 1:
        raise ChannelPermutationError(f"signal trajectory lacks aligned regional tracks: {source}")
    return {"global": global_states, "regional": regional}


def _profile_identifier(identity: tuple[str, ...]) -> str:
    return hashlib.sha256("\0".join(identity).encode()).hexdigest()[:24]


def _constant_label_values(group: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in _PUBLISHED_LABEL_COLUMNS:
        if column not in group:
            continue
        values = group[column].drop_duplicates()
        if len(values) > 1:
            raise ChannelPermutationError(
                f"event trials disagree on profile label {column!r}: {values.tolist()}"
            )
        value = values.iloc[0] if len(values) else None
        if isinstance(value, np.generic):
            value = value.item()
        result[column] = value
    return result


def _build_profile_sources(
    signal_manifest: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    minimum_event_trials: int,
) -> list[_ProfileSource]:
    missing_labels = sorted(set(signal_manifest["unit_id"]) - set(labels["unit_id"]))
    if missing_labels:
        raise ChannelPermutationError(
            f"signal-only encoded units are absent from post-encoding labels: {missing_labels[:10]}"
        )
    signal_for_join = signal_manifest.rename(columns={"modality": "encoded_modality"})
    joined = signal_for_join.merge(labels, on="unit_id", how="left", validate="one_to_one")
    required = {"participant_id", "dataset_id", "condition", "modality"}
    missing = sorted(required.difference(joined.columns))
    if missing:
        raise ChannelPermutationError(f"post-encoding labels are missing {missing}")
    if joined[list(required)].isna().any().any():
        raise ChannelPermutationError("post-encoding profile labels contain missing values")
    modality_mismatch = joined["encoded_modality"].astype(str) != joined["modality"].astype(str)
    if modality_mismatch.any():
        units = joined.loc[modality_mismatch, "unit_id"].astype(str).tolist()
        raise ChannelPermutationError(
            f"post-encoding modality labels disagree with the encoded units: {units[:10]}"
        )

    sources: list[_ProfileSource] = []
    event_mask = joined["selector_kind"].isin({"event_epoch", "pre_epoched"})
    for row in joined[~event_mask].to_dict(orient="records"):
        trajectory = _load_trajectory(
            Path(str(row["trajectory_path"])), expected_sha256=str(row["trajectory_sha256"])
        )
        global_states = trajectory["global"]
        regional = trajectory["regional"]
        fine_length = len(next(iter(regional.values())))
        labels_for_row = {
            column: row.get(column) for column in _PUBLISHED_LABEL_COLUMNS if column in row
        }
        sources.append(
            _ProfileSource(
                profile_id=str(row["unit_id"]),
                source_unit_ids=(str(row["unit_id"]),),
                participant_id=str(row["participant_id"]),
                dataset_id=str(row["dataset_id"]),
                condition=str(row["condition"]),
                modality=str(row["modality"]),
                trajectory=global_states,
                regional=regional,
                segment_ids=np.zeros(len(global_states), dtype=np.int64),
                alignment_segment_ids=np.zeros(fine_length, dtype=np.int64),
                labels=labels_for_row,
                event_aggregated=False,
                trial_count=1,
            )
        )

    event = joined[event_mask].copy()
    group_columns = ["participant_id", "dataset_id", "condition", "modality"]
    for identity, group in event.groupby(group_columns, dropna=False, sort=True):
        if len(group) < minimum_event_trials:
            raise ChannelPermutationError(
                f"event group {identity} has {len(group)} trials; need {minimum_event_trials}"
            )
        loaded = [
            _load_trajectory(
                Path(str(row["trajectory_path"])),
                expected_sha256=str(row["trajectory_sha256"]),
            )
            for row in group.to_dict(orient="records")
        ]
        shared = set(loaded[0]["regional"])
        for item in loaded[1:]:
            shared.intersection_update(item["regional"])
        if len(shared) < 2:
            raise ChannelPermutationError(
                f"event group {identity} has fewer than two shared regions"
            )
        coarse_lengths = [len(item["global"]) for item in loaded]
        fine_lengths = [len(next(iter(item["regional"].values()))) for item in loaded]
        coarse_segments = np.concatenate(
            [np.full(length, index, dtype=np.int64) for index, length in enumerate(coarse_lengths)]
        )
        fine_segments = np.concatenate(
            [np.full(length, index, dtype=np.int64) for index, length in enumerate(fine_lengths)]
        )
        identity_strings = tuple(str(value) for value in identity)
        sources.append(
            _ProfileSource(
                profile_id=_profile_identifier(identity_strings),
                source_unit_ids=tuple(sorted(group["unit_id"].astype(str))),
                participant_id=identity_strings[0],
                dataset_id=identity_strings[1],
                condition=identity_strings[2],
                modality=identity_strings[3],
                trajectory=np.concatenate([item["global"] for item in loaded]),
                regional={
                    name: np.concatenate([item["regional"][name] for item in loaded])
                    for name in sorted(shared)
                },
                segment_ids=coarse_segments,
                alignment_segment_ids=fine_segments,
                labels=_constant_label_values(group),
                event_aggregated=True,
                trial_count=len(group),
            )
        )
    if not sources:
        raise ChannelPermutationError("no profile sources remain after post-encoding label join")
    return sorted(sources, key=lambda source: source.profile_id)


def _profile_scalars(profile: Any) -> dict[str, float]:
    output: dict[str, float] = {}
    for index, axis in enumerate(AXIS_NAMES):
        output[axis] = float(profile.values[index])
        output[f"{axis}_raw"] = float(profile.raw_values[index])
    details = profile.details
    output.update(
        {
            "repertoire_effective_rank": float(details.repertoire.effective_rank),
            "metastability_median_dwell_seconds": float(details.metastability.median_dwell),
            "metastability_switching_rate": float(details.metastability.switching_rate),
            "metastability_recurrence_probability": float(
                details.metastability.recurrence_probability
            ),
            "directionality_flux_asymmetry": float(details.directionality.flux_asymmetry),
            "alignment_best_pair_mean": float(details.alignment.mean_shared_predictive_variance),
            "reachability_effective_rank": float(details.reachability.effective_rank),
        }
    )
    if not all(np.isfinite(value) for value in output.values()):
        raise ChannelPermutationError("frozen profiler produced a non-finite scalar")
    return output


def _profile_source(
    source: _ProfileSource,
    *,
    dictionary: StateDictionary,
    estimator: FiveAxisProfileEstimator,
) -> tuple[dict[str, float], dict[str, Any]]:
    projected = dictionary.project(source.trajectory)
    states = dictionary.predict_projected(projected, segment_ids=source.segment_ids)
    record = ManifoldRecord(
        trajectory=projected,
        states=states,
        regional_trajectories=source.regional,
        repertoire_trajectory=source.trajectory,
        segment_ids=source.segment_ids,
        alignment_segment_ids=source.alignment_segment_ids,
        name=source.profile_id,
    )
    profile = estimator.profile(record)
    coarse_boundaries = np.r_[
        0,
        np.flatnonzero(source.segment_ids[1:] != source.segment_ids[:-1]) + 1,
        len(source.segment_ids),
    ]
    fine_boundaries = np.r_[
        0,
        np.flatnonzero(source.alignment_segment_ids[1:] != source.alignment_segment_ids[:-1]) + 1,
        len(source.alignment_segment_ids),
    ]
    segment_audit = {
        "profile_id": source.profile_id,
        "source_unit_ids": list(source.source_unit_ids),
        "event_aggregated": source.event_aggregated,
        "trial_count": source.trial_count,
        "coarse_segment_lengths": np.diff(coarse_boundaries).astype(int).tolist(),
        "alignment_segment_lengths": np.diff(fine_boundaries).astype(int).tolist(),
        "cross_trial_transitions_allowed": False,
        "cross_trial_alignment_pairs_allowed": False,
    }
    return _profile_scalars(profile), segment_audit


def _profile_expected(
    signal_expected: Mapping[str, Any],
    *,
    labels_sha256: str,
    dictionary_sha256: str,
    estimator_sha256: str,
    minimum_event_trials: int,
) -> dict[str, Any]:
    return {
        **signal_expected,
        "labels_manifest_sha256": labels_sha256,
        "state_dictionary_sha256": dictionary_sha256,
        "profile_estimator_sha256": estimator_sha256,
        "minimum_event_trials": minimum_event_trials,
        "labels_consumed": True,
        "labels_joined_after_encoding": True,
    }


def _profile_signal_repeat(
    *,
    signal_directory: Path,
    repeats_root: Path,
    labels: pd.DataFrame,
    repeat: int,
    expected: Mapping[str, Any],
    dictionary: StateDictionary,
    estimator: FiveAxisProfileEstimator,
    minimum_event_trials: int,
) -> tuple[Path, Path]:
    final = repeats_root / f"repeat-{repeat:04d}"
    if final.exists():
        return _validate_completed_directory(
            final,
            expected=expected,
            data_file=REPEAT_PROFILES_FILE,
            audit_file=REPEAT_AUDIT_FILE,
        )
    signal_manifest_path, signal_audit_path = _validate_completed_directory(
        signal_directory,
        expected={
            key: value
            for key, value in expected.items()
            if key
            not in {
                "labels_manifest_sha256",
                "state_dictionary_sha256",
                "profile_estimator_sha256",
                "minimum_event_trials",
            }
            and key != "labels_consumed"
            and key != "labels_joined_after_encoding"
        }
        | {"labels_consumed": False},
        data_file=SIGNAL_MANIFEST_FILE,
        audit_file=SIGNAL_AUDIT_FILE,
    )
    signal_manifest = pd.read_parquet(signal_manifest_path)
    resolved_trajectories: list[str] = []
    for relative_value in signal_manifest["trajectory_path"]:
        relative = Path(str(relative_value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ChannelPermutationError(
                f"signal cache contains an unsafe trajectory path: {relative_value}"
            )
        resolved_trajectories.append(str((signal_directory / relative).resolve(strict=True)))
    signal_manifest = signal_manifest.copy()
    signal_manifest["trajectory_path"] = resolved_trajectories
    sources = _build_profile_sources(
        signal_manifest, labels, minimum_event_trials=minimum_event_trials
    )
    signal_audit = _read_json_object(signal_audit_path, label="signal audit")
    rows: list[dict[str, Any]] = []
    segment_audit: list[dict[str, Any]] = []
    for source in sources:
        scalars, segments = _profile_source(source, dictionary=dictionary, estimator=estimator)
        label_values = {
            key: value
            for key, value in source.labels.items()
            if value is None or isinstance(value, (str, int, float, bool, np.generic))
        }
        rows.append(
            {
                **label_values,
                "unit_id": source.profile_id,
                "profile_id": source.profile_id,
                "participant_id": source.participant_id,
                "dataset_id": source.dataset_id,
                "condition": source.condition,
                "modality": source.modality,
                "family": FAMILY,
                "repeat": repeat,
                "seed": int(expected["repeat_seed"]),
                "repeat_seed": int(expected["repeat_seed"]),
                "event_aggregated": source.event_aggregated,
                "trial_count": source.trial_count,
                "n_windows": len(source.trajectory),
                "source_unit_ids_json": json.dumps(list(source.source_unit_ids)),
                "state_method": dictionary.method,
                **scalars,
            }
        )
        segment_audit.append(segments)
    profile_frame = pd.DataFrame(rows).sort_values("profile_id", kind="stable")
    forbidden_paths = {
        "preprocessed_path",
        "source_path",
        "trajectory_path",
        "permutation",
    }
    if forbidden_paths.intersection(profile_frame.columns):
        raise AssertionError("published scalar profile table contains signal/path payloads")

    repeats_root.mkdir(parents=True, exist_ok=True)
    work = repeats_root / f".repeat-{repeat:04d}.work"
    _reset_private_work_directory(work, parent=repeats_root)
    profiles_path = _atomic_parquet(
        profile_frame.reset_index(drop=True), work / REPEAT_PROFILES_FILE
    )
    audit_path = work / REPEAT_AUDIT_FILE
    atomic_write_json(
        audit_path,
        {
            **expected,
            "profile_rows": len(profile_frame),
            "state_dictionary_refit": False,
            "profile_estimator_refit": False,
            "encoder_label_fields_consumed": [],
            "profile_segment_audit": segment_audit,
            "unit_permutation_audit": signal_audit["unit_audit"],
        },
    )
    atomic_write_json(
        work / COMPLETION_FILE,
        {
            **expected,
            "data_sha256": sha256_file(profiles_path),
            "audit_sha256": sha256_file(audit_path),
        },
    )
    os.replace(work, final)
    return final / REPEAT_PROFILES_FILE, final / REPEAT_AUDIT_FILE


def _canonical_null_profile_table(path: Path, *, label: str) -> pd.DataFrame:
    try:
        source = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ChannelPermutationError(f"{label} null-profile table is missing: {path}") from error
    try:
        frame = pd.read_parquet(source)
    except (OSError, ValueError) as error:
        raise ChannelPermutationError(
            f"cannot read {label} null-profile table: {source}"
        ) from error
    missing = sorted(set(NULL_PROFILE_COLUMNS).difference(frame.columns))
    if missing:
        raise ChannelPermutationError(f"{label} null-profile table is missing {missing}")
    canonical = frame.loc[:, NULL_PROFILE_COLUMNS].copy()
    for column in ("unit_id", "participant_id", "dataset_id", "family"):
        if canonical[column].isna().any():
            raise ChannelPermutationError(f"{label} null-profile {column} contains missing values")
        canonical[column] = canonical[column].astype(str)
        if canonical[column].str.len().eq(0).any():
            raise ChannelPermutationError(f"{label} null-profile {column} contains empty values")
    for column in ("repeat", "seed"):
        try:
            numeric = pd.to_numeric(canonical[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ChannelPermutationError(
                f"{label} null-profile {column} is not numeric"
            ) from error
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ChannelPermutationError(
                f"{label} null-profile {column} must contain finite integers"
            )
        canonical[column] = numeric.astype(np.int64)
    for axis in AXIS_NAMES:
        try:
            values = pd.to_numeric(canonical[axis], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ChannelPermutationError(
                f"{label} null-profile axis {axis} is not numeric"
            ) from error
        if not np.all(np.isfinite(values)):
            raise ChannelPermutationError(
                f"{label} null-profile axis {axis} contains non-finite values"
            )
        canonical[axis] = values
    return canonical


def combine_null_profile_tables(
    base: str | Path,
    channel: str | Path,
    destination: str | Path,
) -> Path:
    """Atomically combine canonical metrics and pre-encoder null-profile rows.

    Extra audit/detail columns in the channel-control table are intentionally
    excluded so the result has exactly the schema of ``metrics/null-profiles``.
    """

    base_frame = _canonical_null_profile_table(Path(base), label="base")
    channel_frame = _canonical_null_profile_table(Path(channel), label="channel")
    unexpected = sorted(set(channel_frame["family"]) - {FAMILY})
    if unexpected:
        raise ChannelPermutationError(
            f"channel null-profile table contains unexpected families: {unexpected}"
        )
    combined = pd.concat((base_frame, channel_frame), ignore_index=True)
    key = ["unit_id", "family", "repeat"]
    if combined.duplicated(key).any():
        duplicates = combined.loc[combined.duplicated(key, keep=False), key]
        raise ChannelPermutationError(
            "combined null-profile table has duplicate unit/family/repeat keys: "
            f"{duplicates.head(10).to_dict(orient='records')}"
        )
    return _atomic_parquet(combined, Path(destination).resolve())


def run_preencoder_channel_permutation_control(
    *,
    preprocessing_manifest: str | Path,
    labels_manifest: str | Path,
    state_dictionary_path: str | Path,
    profile_estimator_path: str | Path,
    output_root: str | Path,
    study: StudyConfig,
    repeats: int,
    modalities: Sequence[str] = ("eeg",),
    encoder: LaBraMBackend | None = None,
    device: str = "cuda",
    raw_loader: Callable[[Path], Any] | None = None,
    minimum_event_trials: int | None = None,
) -> ChannelPermutationArtifacts:
    """Run a deterministic, restart-safe pre-LaBraM sensor-row null.

    ``repeats`` is mandatory and must be positive; the function never silently
    inherits a computational repeat count.  ``preprocessing_manifest`` must be
    the label-free output of analysis-unit preprocessing.  ``labels_manifest``
    is not opened until all missing repeat encodings have completed.

    An injected ``encoder`` and ``raw_loader`` support dependency-free tests.
    Production calls should omit them so the pinned official LaBraM checkpoint
    and MNE FIF reader are used.
    """

    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ChannelPermutationError("repeats must be an explicit positive integer")
    requested_modalities = tuple(str(value) for value in modalities)
    if not requested_modalities or any(not value for value in requested_modalities):
        raise ChannelPermutationError("modalities must contain at least one nonempty value")
    if len(set(requested_modalities)) != len(requested_modalities):
        raise ChannelPermutationError("modalities contain duplicates")
    if requested_modalities != ("eeg",):
        raise ChannelPermutationError(
            "the pre-encoder sensor-row permutation control is defined only for modality 'eeg'"
        )
    event_minimum = (
        int(study.preprocessing.minimum_event_trials_per_condition)
        if minimum_event_trials is None
        else int(minimum_event_trials)
    )
    if event_minimum <= 0:
        raise ChannelPermutationError("minimum_event_trials must be positive")

    try:
        preprocessing_path = Path(preprocessing_manifest).resolve(strict=True)
        dictionary_path = Path(state_dictionary_path).resolve(strict=True)
        estimator_path = Path(profile_estimator_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise ChannelPermutationError(
            f"required pre-encoding/profiler artifact is missing: {error}"
        ) from error
    frame, modality_counts = _validate_preprocessing_manifest(
        preprocessing_path, modalities=requested_modalities
    )
    dictionary = joblib.load(dictionary_path)
    estimator = joblib.load(estimator_path)
    if not isinstance(dictionary, StateDictionary):
        raise ChannelPermutationError("state dictionary artifact has the wrong type")
    if not isinstance(estimator, FiveAxisProfileEstimator):
        raise ChannelPermutationError("profile estimator artifact has the wrong type")
    dictionary_hash = sha256_file(dictionary_path)
    estimator_hash = sha256_file(estimator_path)

    backend = encoder if encoder is not None else _build_official_encoder(device=device)
    _validate_frozen_backend(backend)
    encoder_metadata = _safe_encoder_metadata(backend)
    loader = raw_loader or _read_preprocessed_raw
    destination = Path(output_root).resolve()
    signals_root = destination / ".signal-repeats"
    repeats_root = destination / "repeats"
    study_hash = config_sha256(study)
    preprocessing_hash = sha256_file(preprocessing_path)

    signal_directories: dict[int, Path] = {}
    # Phase one: do all required signal inference before labels are opened.
    for repeat in range(repeats):
        final_repeat = repeats_root / f"repeat-{repeat:04d}"
        if final_repeat.exists():
            continue
        repeat_seed = _repeat_seed(study, repeat)
        signal_directories[repeat] = _prepare_signal_repeat(
            frame=frame,
            preprocessing_manifest=preprocessing_path,
            signals_root=signals_root,
            repeat=repeat,
            repeat_seed=repeat_seed,
            study=study,
            study_sha256=study_hash,
            encoder=backend,
            encoder_metadata=encoder_metadata,
            modalities=requested_modalities,
            raw_loader=loader,
        )

    # Phase two: only now may post-encoding labels be resolved and read.
    try:
        labels_path = Path(labels_manifest).resolve(strict=True)
    except FileNotFoundError as error:
        raise ChannelPermutationError(
            f"post-encoding label manifest is missing: {labels_manifest}"
        ) from error
    try:
        labels = pd.read_parquet(labels_path)
    except (OSError, ValueError) as error:
        raise ChannelPermutationError(f"cannot read post-encoding labels: {labels_path}") from error
    if "unit_id" not in labels:
        raise ChannelPermutationError("post-encoding labels have no unit_id")
    if labels["unit_id"].isna().any() or labels["unit_id"].astype(str).duplicated().any():
        raise ChannelPermutationError("post-encoding label unit IDs must be nonempty and unique")
    labels = labels.copy()
    labels["unit_id"] = labels["unit_id"].astype(str)
    labels_hash = sha256_file(labels_path)

    repeat_frames: list[pd.DataFrame] = []
    repeat_artifacts: list[dict[str, Any]] = []
    for repeat in range(repeats):
        repeat_seed = _repeat_seed(study, repeat)
        signal_expected = _signal_expected(
            repeat=repeat,
            repeat_seed=repeat_seed,
            preprocessing_sha256=preprocessing_hash,
            study_sha256=study_hash,
            encoder_sha256=str(encoder_metadata["sha256"]),
            modalities=requested_modalities,
        )
        expected = _profile_expected(
            signal_expected,
            labels_sha256=labels_hash,
            dictionary_sha256=dictionary_hash,
            estimator_sha256=estimator_hash,
            minimum_event_trials=event_minimum,
        )
        final_repeat = repeats_root / f"repeat-{repeat:04d}"
        reused = final_repeat.exists()
        if reused:
            profiles_path, repeat_audit_path = _validate_completed_directory(
                final_repeat,
                expected=expected,
                data_file=REPEAT_PROFILES_FILE,
                audit_file=REPEAT_AUDIT_FILE,
            )
        else:
            signal_directory = signal_directories.get(repeat)
            if signal_directory is None:
                raise AssertionError("missing signal-only repeat after label firewall")
            profiles_path, repeat_audit_path = _profile_signal_repeat(
                signal_directory=signal_directory,
                repeats_root=repeats_root,
                labels=labels,
                repeat=repeat,
                expected=expected,
                dictionary=dictionary,
                estimator=estimator,
                minimum_event_trials=event_minimum,
            )
        repeat_frame = pd.read_parquet(profiles_path)
        repeat_frames.append(repeat_frame)
        repeat_artifacts.append(
            {
                "repeat": repeat,
                "repeat_seed": repeat_seed,
                "profiles_path": str(profiles_path),
                "profiles_sha256": sha256_file(profiles_path),
                "audit_path": str(repeat_audit_path),
                "audit_sha256": sha256_file(repeat_audit_path),
                "reused": reused,
            }
        )
        signal_directory = signals_root / f"repeat-{repeat:04d}"
        if signal_directory.exists():
            if signal_directory.is_symlink() or signal_directory.parent != signals_root:
                raise ChannelPermutationError(f"unsafe signal-cache directory: {signal_directory}")
            shutil.rmtree(signal_directory)

    if (
        sha256_file(dictionary_path) != dictionary_hash
        or sha256_file(estimator_path) != estimator_hash
    ):
        raise ChannelPermutationError("frozen profiler artifacts changed during the control")
    combined = pd.concat(repeat_frames, ignore_index=True).sort_values(
        ["repeat", "profile_id"], kind="stable"
    )
    profiles_path = _atomic_parquet(
        combined.reset_index(drop=True),
        destination / "preencoder-channel-permutation-null-profiles.parquet",
    )
    audit_path = destination / "preencoder-channel-permutation-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "family": FAMILY,
            "intervention": {
                "signal_rows_permuted_before_encoder": True,
                "original_channel_position_names_retained": True,
                "post_encoder_latent_rotation": False,
            },
            "study_sha256": study_hash,
            "study_random_seeds": [int(value) for value in study.random_seeds],
            "repeats": repeats,
            "repeat_count_is_explicit": True,
            "modalities": list(requested_modalities),
            "eligible_modality_counts": modality_counts,
            "preprocessing_manifest": str(preprocessing_path),
            "preprocessing_manifest_sha256": preprocessing_hash,
            "labels_manifest": str(labels_path),
            "labels_manifest_sha256": labels_hash,
            "labels_opened_after_all_missing_repeat_encodings": True,
            "encoder_label_fields_consumed": [],
            "encoder": encoder_metadata,
            "state_dictionary": str(dictionary_path),
            "state_dictionary_sha256": dictionary_hash,
            "state_dictionary_refit": False,
            "profile_estimator": str(estimator_path),
            "profile_estimator_sha256": estimator_hash,
            "profile_estimator_refit": False,
            "profile_input_spaces": {
                "repertoire": {
                    "record_field": "repertoire_trajectory",
                    "space": "untruncated_frozen_encoder_embedding",
                    "dimension": int(dictionary.projection.n_features_in_),
                    "discovery_projection_applied": False,
                },
                "dynamics": {
                    "record_field": "trajectory",
                    "space": "discovery_fitted_pca_projection",
                    "dimension": int(dictionary.projection.n_components_),
                    "discovery_projection_applied": True,
                },
            },
            "minimum_event_trials": event_minimum,
            "event_trials_encoded_independently": True,
            "event_transitions_cross_boundaries": False,
            "event_alignment_pairs_cross_boundaries": False,
            "profiles": {
                "path": str(profiles_path),
                "sha256": sha256_file(profiles_path),
                "rows": len(combined),
                "scalar_axes": list(AXIS_NAMES),
                "canonical_null_profile_columns": list(NULL_PROFILE_COLUMNS),
                "unit_id_semantics": (
                    "original unit ID for standalone recordings; deterministic profile ID for "
                    "aggregated event trials, whose members are in source_unit_ids_json"
                ),
            },
            "repeat_artifacts": repeat_artifacts,
            "restart_safety": "atomic content-hashed completion marker per repeat",
            "scientific_gate_applied": False,
        },
    )
    return ChannelPermutationArtifacts(
        profiles_path=profiles_path,
        audit_path=audit_path,
        repeats_root=repeats_root,
        repeats=repeats,
        profile_rows=len(combined),
    )


__all__ = [
    "FAMILY",
    "NULL_PROFILE_COLUMNS",
    "ChannelPermutationArtifacts",
    "ChannelPermutationError",
    "LaBraMBackend",
    "combine_null_profile_tables",
    "run_preencoder_channel_permutation_control",
]
