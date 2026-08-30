"""Secondary fMRI triangulation with frozen BrainLM coordinates.

This stage is descriptive and modality-specific.  It computes a BrainLM latent
repertoire and transition summary plus lagged alignment in preprocessed A424
BOLD space.  It deliberately does not estimate perturbational reachability from
passive fMRI.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from neural_manifolds.config import StudyConfig, config_sha256
from neural_manifolds.foundation.brainlm import (
    N_PARCELS,
    PARCELLATION,
    USAGE_LICENSE,
    BrainLMEncoding,
    load_ukb424_coordinates,
    validate_parcel_timeseries,
)
from neural_manifolds.manifold.alignment import pairwise_module_alignment
from neural_manifolds.manifold.directionality import estimate_directionality
from neural_manifolds.manifold.repertoire import estimate_repertoire
from neural_manifolds.provenance import atomic_write_json, sha256_file


class BrainLMBackend(Protocol):
    metadata: Mapping[str, Any]

    def encode(
        self,
        parcel_timeseries: np.ndarray,
        coordinates: np.ndarray,
        *,
        metadata_fields: Sequence[str] = (),
    ) -> BrainLMEncoding: ...


class FMRIManifestError(ValueError):
    """Raised when an fMRI input cannot satisfy the audited stage contract."""


DEFAULT_CONFOUNDS = (
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "trans_x_derivative1",
    "trans_y_derivative1",
    "trans_z_derivative1",
    "rot_x_derivative1",
    "rot_y_derivative1",
    "rot_z_derivative1",
    "white_matter",
    "csf",
    "framewise_displacement",
)


def _read_manifest(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise FMRIManifestError("fMRI manifest must be parquet, CSV, TSV, or JSONL")


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _atomic_npz(destination: Path, **arrays: np.ndarray) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".npz"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _atomic_npy(destination: Path, values: np.ndarray) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".npy"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _present(value: object) -> bool:
    return (
        value is not None
        and not (isinstance(value, float) and np.isnan(value))
        and str(value) != ""
    )


def _resolve_input(value: object, base: Path, *, field: str) -> Path:
    if not _present(value):
        raise FMRIManifestError(f"missing {field}")
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FMRIManifestError(f"{field} does not exist: {path}") from exc


def _load_timeseries(path: Path, *, array_key: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = array_key or "timeseries"
            if key not in archive.files:
                raise FMRIManifestError(f"{path} has no NPZ array {key!r}")
            values = archive[key]
    elif suffix in {".csv", ".tsv", ".txt", ".dat"}:
        delimiter = "," if suffix == ".csv" else None
        values = np.genfromtxt(path, delimiter=delimiter, dtype=np.float64)
    else:
        raise FMRIManifestError(f"unsupported parcel time-series format: {path}")
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != N_PARCELS:
        raise FMRIManifestError(
            f"preprocessed parcel time series must be time x {N_PARCELS}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise FMRIManifestError(f"non-finite parcel time series: {path}")
    return array


def _parse_confound_columns(value: object) -> tuple[str, ...]:
    if not _present(value):
        return DEFAULT_CONFOUNDS
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise FMRIManifestError("confound_columns_json must be a JSON list") from exc
    else:
        decoded = value
    if (
        not isinstance(decoded, list)
        or not decoded
        or not all(isinstance(item, str) for item in decoded)
    ):
        raise FMRIManifestError("confound columns must be a non-empty string list")
    result = tuple(decoded)
    if len(set(result)) != len(result):
        raise FMRIManifestError("confound columns contain duplicates")
    return result


def _numeric_confounds(path: Path, columns: tuple[str, ...], n_volumes: int) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t")
    missing = [column for column in columns if column not in frame]
    # Derivatives and physiological regressors are useful but release-dependent;
    # the six rigid-body terms are the non-negotiable minimum.
    required = set(DEFAULT_CONFOUNDS[:6])
    if required.intersection(missing):
        raise FMRIManifestError(
            f"confounds file is missing rigid-body terms: {sorted(required.intersection(missing))}"
        )
    selected_columns = [column for column in columns if column in frame]
    selected = frame[selected_columns].apply(pd.to_numeric, errors="coerce")
    if len(selected) != n_volumes:
        raise FMRIManifestError("confound rows do not match BOLD volumes")
    for column in selected:
        missing_indices = np.flatnonzero(selected[column].isna().to_numpy())
        if missing_indices.size:
            if np.array_equal(missing_indices, np.asarray([0])):
                selected.loc[selected.index[0], column] = 0.0
            else:
                raise FMRIManifestError(f"confound {column!r} has non-initial missing values")
    values = selected.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FMRIManifestError("confounds remain non-finite after the audited first-row rule")
    return values


def extract_ukb424_timeseries(
    *,
    bold_path: str | Path,
    confounds_path: str | Path,
    atlas_path: str | Path,
    tr_seconds: float,
    confound_columns: tuple[str, ...] = DEFAULT_CONFOUNDS,
) -> np.ndarray:
    """Extract denoised A424 parcel time series when Nilearn is available."""

    try:
        import nibabel as nib
        from nilearn.maskers import NiftiLabelsMasker
    except ImportError as exc:  # pragma: no cover - optional server environment
        raise RuntimeError(
            "install neural-manifolds[fmri] to extract BOLD parcel time series"
        ) from exc
    bold = Path(bold_path).resolve(strict=True)
    confounds = Path(confounds_path).resolve(strict=True)
    atlas = Path(atlas_path).resolve(strict=True)
    if not np.isfinite(tr_seconds) or tr_seconds <= 0:
        raise FMRIManifestError("tr_seconds must be finite and positive")
    atlas_image = nib.load(str(atlas))
    atlas_data = np.asanyarray(atlas_image.dataobj)
    rounded = np.rint(atlas_data)
    if not np.allclose(atlas_data, rounded, atol=1e-6):
        raise FMRIManifestError("UKB_424 atlas must contain integer parcel labels")
    labels = set(np.unique(rounded.astype(np.int64)).tolist())
    if labels != set(range(N_PARCELS + 1)):
        raise FMRIManifestError(
            "UKB_424 atlas labels must be exactly background 0 and parcels 1..424"
        )
    bold_image = nib.load(str(bold))
    if len(bold_image.shape) != 4:
        raise FMRIManifestError("BOLD image must be four-dimensional")
    n_volumes = int(bold_image.shape[3])
    nuisance = _numeric_confounds(confounds, confound_columns, n_volumes)
    nyquist = 0.5 / float(tr_seconds)
    low_pass = min(0.09, 0.8 * nyquist)
    high_pass = 0.008 if low_pass > 0.008 else None
    masker = NiftiLabelsMasker(
        labels_img=str(atlas),
        standardize=False,
        detrend=True,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=float(tr_seconds),
        smoothing_fwhm=None,
        strategy="mean",
        reports=False,
    )
    values = np.asarray(masker.fit_transform(str(bold), confounds=nuisance), dtype=np.float32)
    if values.shape != (n_volumes, N_PARCELS):
        raise FMRIManifestError(f"A424 extraction returned unexpected shape {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(np.std(values, axis=0) <= 0):
        raise FMRIManifestError("A424 extraction produced non-finite or constant parcel signals")
    return values


def _validate_manifest(frame: pd.DataFrame) -> None:
    required = {"unit_id", "participant_id", "dataset_id", "parcellation", "tr_seconds"}
    missing = required.difference(frame.columns)
    if missing:
        raise FMRIManifestError(f"fMRI manifest is missing {sorted(missing)}")
    if frame.empty:
        raise FMRIManifestError("fMRI manifest is empty")
    if frame["unit_id"].astype(str).duplicated().any():
        raise FMRIManifestError("fMRI manifest contains duplicate unit_id values")
    if set(frame["dataset_id"].astype(str)) != {"propofol_fmri"}:
        raise FMRIManifestError("fMRI stage accepts only the audited propofol_fmri dataset")
    if set(frame["parcellation"].astype(str)) != {PARCELLATION}:
        raise FMRIManifestError(f"every fMRI row must declare parcellation {PARCELLATION}")
    tr = pd.to_numeric(frame["tr_seconds"], errors="coerce")
    if tr.isna().any() or (tr <= 0).any():
        raise FMRIManifestError("tr_seconds must be finite and positive")
    for row in frame.to_dict(orient="records"):
        has_timeseries = _present(row.get("timeseries_path"))
        has_bold = _present(row.get("bold_path"))
        if has_timeseries == has_bold:
            raise FMRIManifestError("each row requires exactly one of timeseries_path or bold_path")
        if has_timeseries:
            if row.get("preprocessed") is not True:
                raise FMRIManifestError("timeseries_path rows must declare preprocessed=true")
            if str(row.get("timeseries_scope")) != "run":
                raise FMRIManifestError(
                    "parcel time series must cover the full run before segmentation"
                )
        else:
            if not _present(row.get("confounds_path")):
                raise FMRIManifestError("BOLD extraction requires confounds_path")


def _assign_partitions(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.copy()
    if "partition" in result and result["partition"].notna().all():
        allowed = {"discovery", "validation", "test"}
        if not set(result["partition"].astype(str)) <= allowed:
            raise FMRIManifestError(f"partition must lie in {sorted(allowed)}")
    else:
        participants = sorted(set(result["participant_id"].astype(str)))
        ranked = sorted(
            participants,
            key=lambda participant: hashlib.sha256(f"{seed}:{participant}".encode()).hexdigest(),
        )
        n_participants = len(ranked)
        n_discovery = max(1, int(np.floor(0.70 * n_participants)))
        n_validation = 1 if n_participants >= 2 else 0
        if n_discovery + n_validation > n_participants:
            n_discovery = n_participants - n_validation
        assignments: dict[str, str] = {}
        for index, participant in enumerate(ranked):
            if index < n_discovery:
                assignments[participant] = "discovery"
            elif index < n_discovery + n_validation:
                assignments[participant] = "validation"
            else:
                assignments[participant] = "test"
        result["partition"] = result["participant_id"].astype(str).map(assignments)
    counts = result.groupby("participant_id")["partition"].nunique()
    if (counts != 1).any():
        leaked = sorted(counts[counts != 1].index.astype(str))
        raise FMRIManifestError(f"participants cross fMRI partitions: {leaked}")
    return result


def _encoder_metadata(encoder: BrainLMBackend) -> dict[str, Any]:
    metadata = dict(encoder.metadata)
    expected = {
        "weights_frozen": True,
        "label_free": True,
        "parcellation": PARCELLATION,
        "usage_license": USAGE_LICENSE,
        "commercial_use": False,
        "derivative_redistribution": False,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"BrainLM backend metadata has invalid {field}")
    for field in ("checkpoint_config_sha256", "checkpoint_weights_sha256"):
        value = metadata.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"BrainLM backend metadata requires {field}")
    revision = metadata.get("source_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("BrainLM backend metadata requires the exact source revision")
    return metadata


def _module_trajectories(timeseries: np.ndarray, coordinates: np.ndarray) -> dict[str, np.ndarray]:
    masks = {
        "left": coordinates[:, 0] < 0.0,
        "right": coordinates[:, 0] >= 0.0,
        "posterior": coordinates[:, 1] < np.median(coordinates[:, 1]),
        "anterior": coordinates[:, 1] >= np.median(coordinates[:, 1]),
    }
    modules: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        if np.count_nonzero(mask) < 3:
            raise ValueError(f"UKB_424 coordinate module {name} has fewer than three parcels")
        values = timeseries[:, mask]
        rank = min(5, values.shape[0] - 1, values.shape[1])
        modules[name] = PCA(n_components=rank, svd_solver="full").fit_transform(values)
    return modules


def _summarise_unit(
    encoding: BrainLMEncoding,
    timeseries: np.ndarray,
    coordinates: np.ndarray,
    *,
    tr_seconds: float,
    random_state: int,
) -> dict[str, Any]:
    states = np.asarray(encoding.global_states, dtype=np.float64)
    repertoire = estimate_repertoire(states, shrinkage="oas")
    if states.shape[0] < 4:
        raise ValueError("at least four BrainLM windows are required for transition summaries")
    rank = min(8, states.shape[0] - 1, states.shape[1])
    projected = PCA(n_components=rank, svd_solver="full").fit_transform(states)
    n_states = min(6, max(2, states.shape[0] // 3), states.shape[0] - 1)
    labels = KMeans(n_clusters=n_states, n_init=20, random_state=random_state).fit_predict(
        projected
    )
    sample_interval = float(np.median(np.diff(encoding.window_starts))) * tr_seconds
    directionality = estimate_directionality(
        labels,
        pseudocount=0.5,
        sample_interval=sample_interval,
    )
    counts = np.bincount(labels, minlength=n_states).astype(np.float64)
    probabilities = counts[counts > 0] / np.sum(counts)
    occupancy_entropy = -float(np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = occupancy_entropy / np.log(n_states) if n_states > 1 else 0.0
    switching = float(np.mean(labels[1:] != labels[:-1]))

    modules = _module_trajectories(timeseries, coordinates)
    maximum_lag = min(3, max(1, timeseries.shape[0] // 20))
    lags = tuple(range(1, maximum_lag + 1))
    n_pairs = timeseries.shape[0] - maximum_lag
    cv = min(5, n_pairs // 2)
    if cv < 2:
        raise ValueError("too few fMRI volumes for blocked alignment validation")
    alignment = pairwise_module_alignment(
        modules,
        module_pairs=(("left", "right"), ("posterior", "anterior")),
        lags=lags,
        rank=min(3, *(value.shape[1] for value in modules.values())),
        ridge=1e-6,
        cv=cv,
        bidirectional=True,
    )
    pair_values = dict(zip(alignment.pair_names, alignment.pair_values, strict=True))
    return {
        "brainlm_repertoire_participation_ratio": repertoire.participation_ratio,
        "brainlm_repertoire_effective_rank": repertoire.effective_rank,
        "brainlm_repertoire_total_variance": repertoire.total_variance,
        "brainlm_repertoire_leading_variance_fraction": repertoire.leading_variance_fraction,
        "transition_n_states": n_states,
        "transition_occupancy_entropy_normalized": normalized_entropy,
        "transition_switching_fraction": switching,
        "transition_entropy_production": directionality.entropy_production,
        "transition_entropy_production_rate": directionality.entropy_production_rate,
        "transition_flux_asymmetry": directionality.flux_asymmetry,
        "alignment_shared_predictive_variance": alignment.mean_shared_predictive_variance,
        "alignment_left_right": pair_values["left<->right"],
        "alignment_anterior_posterior": pair_values["posterior<->anterior"],
        "alignment_maximum_lag_seconds": maximum_lag * tr_seconds,
        "brainlm_windows": states.shape[0],
        "fmri_volumes": timeseries.shape[0],
    }


def _segment(values: np.ndarray, row: Mapping[str, Any]) -> tuple[np.ndarray, int, int]:
    start = int(row["volume_start"]) if _present(row.get("volume_start")) else 0
    stop = int(row["volume_stop"]) if _present(row.get("volume_stop")) else values.shape[0]
    if not 0 <= start < stop <= values.shape[0]:
        raise FMRIManifestError(
            f"volume interval [{start}, {stop}) is outside run length {values.shape[0]}"
        )
    return values[start:stop], start, stop


def _calibrate_secondary_axes(
    participant: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map fMRI-compatible summaries onto discovery-calibrated R/M/D/A axes."""

    required = {
        "partition",
        "brainlm_repertoire_effective_rank",
        "transition_occupancy_entropy_normalized",
        "transition_switching_fraction",
        "transition_entropy_production_rate",
        "alignment_shared_predictive_variance",
    }
    missing = required.difference(participant.columns)
    if missing:
        raise FMRIManifestError(f"fMRI summaries cannot form R/M/D/A: missing {sorted(missing)}")
    output = participant.copy()
    raw = {
        "R": np.log1p(output["brainlm_repertoire_effective_rank"].to_numpy(dtype=float)),
        "M": np.mean(
            np.column_stack(
                [
                    output["transition_occupancy_entropy_normalized"].to_numpy(dtype=float),
                    output["transition_switching_fraction"].to_numpy(dtype=float),
                ]
            ),
            axis=1,
        ),
        "D": output["transition_entropy_production_rate"].to_numpy(dtype=float),
        "A": output["alignment_shared_predictive_variance"].to_numpy(dtype=float),
    }
    discovery = output["partition"].astype(str).eq("discovery").to_numpy()
    if np.count_nonzero(discovery) < 2:
        raise FMRIManifestError("R/M/D/A calibration requires two discovery participants")
    audit: dict[str, Any] = {}
    sources = {
        "R": "log1p(brainlm_repertoire_effective_rank)",
        "M": "mean(transition_occupancy_entropy_normalized,transition_switching_fraction)",
        "D": "transition_entropy_production_rate",
        "A": "alignment_shared_predictive_variance",
    }
    for axis, values in raw.items():
        if not np.all(np.isfinite(values)):
            raise FMRIManifestError(f"fMRI {axis} source contains non-finite values")
        fit = values[discovery]
        center = float(np.mean(fit))
        observed_scale = float(np.std(fit, ddof=1))
        degenerate = bool(observed_scale <= np.finfo(float).eps)
        scale = 1.0 if degenerate else observed_scale
        output[f"{axis}_raw"] = values
        output[axis] = (values - center) / scale
        audit[axis] = {
            "source": sources[axis],
            "center": center,
            "scale": scale,
            "observed_discovery_scale": observed_scale,
            "degenerate_discovery_scale": degenerate,
        }
    return output, audit


def run_fmri_triangulation(
    *,
    manifest_path: str | Path,
    coordinates_path: str | Path,
    output_root: str | Path,
    encoder: BrainLMBackend,
    study: StudyConfig,
    atlas_path: str | Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Encode ds006623 units and write participant-safe secondary summaries."""

    source_manifest = Path(manifest_path).resolve(strict=True)
    frame = _assign_partitions(_read_manifest(source_manifest), study.random_seeds[0])
    _validate_manifest(frame)
    encoder_metadata = _encoder_metadata(encoder)
    coordinates_source = Path(coordinates_path).resolve(strict=True)
    coordinates = load_ukb424_coordinates(coordinates_source)
    atlas_source = Path(atlas_path).resolve(strict=True) if atlas_path is not None else None
    manifest_base = source_manifest.parent
    destination = Path(output_root)
    trajectory_root = destination / "trajectories"
    extracted_root = destination / "preprocessed-runs"

    cache: dict[tuple[str, ...], np.ndarray] = {}
    run_provenance: dict[tuple[str, ...], dict[str, Any]] = {}

    def load_run(row: Mapping[str, Any]) -> tuple[tuple[str, ...], np.ndarray]:
        if _present(row.get("timeseries_path")):
            path = _resolve_input(row["timeseries_path"], manifest_base, field="timeseries_path")
            key = ("timeseries", str(path), str(row.get("array_key", "timeseries")))
            if key not in cache:
                cache[key] = _load_timeseries(
                    path,
                    array_key=str(row["array_key"]) if _present(row.get("array_key")) else None,
                )
                run_provenance[key] = {
                    "mode": "preprocessed_manifest",
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            return key, cache[key]
        if atlas_source is None:
            raise FMRIManifestError("raw BOLD mode requires an explicit UKB_424 atlas_path")
        bold = _resolve_input(row.get("bold_path"), manifest_base, field="bold_path")
        confounds = _resolve_input(row.get("confounds_path"), manifest_base, field="confounds_path")
        columns = _parse_confound_columns(row.get("confound_columns_json"))
        key = ("bold", str(bold), str(confounds), *columns)
        if key not in cache:
            values = extract_ukb424_timeseries(
                bold_path=bold,
                confounds_path=confounds,
                atlas_path=atlas_source,
                tr_seconds=float(row["tr_seconds"]),
                confound_columns=columns,
            )
            digest = hashlib.sha256("\0".join(key).encode()).hexdigest()[:24]
            generated = _atomic_npy(extracted_root / f"{digest}.npy", values)
            cache[key] = values
            run_provenance[key] = {
                "mode": "nilearn_bold_confounds",
                "bold_path": str(bold),
                "bold_sha256": sha256_file(bold),
                "confounds_path": str(confounds),
                "confounds_sha256": sha256_file(confounds),
                "confound_columns": list(columns),
                "generated_path": str(generated.resolve()),
                "generated_sha256": sha256_file(generated),
            }
        return key, cache[key]

    row_runs: dict[str, tuple[str, ...]] = {}
    for row in frame.to_dict(orient="records"):
        key, _ = load_run(row)
        row_runs[str(row["unit_id"])] = key

    normalization_values = (
        set(frame["normalization"].fillna("unscaled_denoised").astype(str))
        if "normalization" in frame
        else {"unscaled_denoised"}
    )
    allowed_normalization = {"unscaled_denoised", "brainlm_all_patient_all_voxel"}
    if not normalization_values <= allowed_normalization or len(normalization_values) != 1:
        raise FMRIManifestError(
            "all runs must consistently use unscaled_denoised or brainlm_all_patient_all_voxel"
        )
    normalization = next(iter(normalization_values))
    fit_participants: list[str] = []
    scale_min: float | None = None
    scale_max: float | None = None
    if normalization == "unscaled_denoised":
        discovery = frame[frame["partition"].astype(str) == "discovery"]
        if discovery.empty:
            raise FMRIManifestError("normalization requires at least one discovery participant")
        discovery_keys = {
            row_runs[str(row["unit_id"])] for row in discovery.to_dict(orient="records")
        }
        scale_min = min(float(np.min(cache[key])) for key in discovery_keys)
        scale_max = max(float(np.max(cache[key])) for key in discovery_keys)
        if not np.isfinite(scale_min) or not np.isfinite(scale_max) or scale_max <= scale_min:
            raise FMRIManifestError("discovery-only BrainLM normalization range is degenerate")
        fit_participants = sorted(set(discovery["participant_id"].astype(str)))

    encoded_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    label_columns = [
        column
        for column in (
            "condition",
            "task",
            "run_id",
            "behavioral_responsiveness",
            "effect_site_concentration_min",
            "effect_site_concentration_mean",
            "effect_site_concentration_max",
            "metadata_status",
        )
        if column in frame
    ]
    for index, row in enumerate(frame.to_dict(orient="records")):
        unit_id = str(row["unit_id"])
        try:
            run_values = cache[row_runs[unit_id]]
            if normalization == "unscaled_denoised":
                assert scale_min is not None and scale_max is not None
                run_values = (run_values - scale_min) / (scale_max - scale_min)
            segment, volume_start, volume_stop = _segment(run_values, row)
            segment = validate_parcel_timeseries(segment)
            # No participant, condition, concentration, or response field crosses
            # this call boundary: only signal and fixed atlas coordinates do.
            encoding = encoder.encode(segment, coordinates, metadata_fields=())
            trajectory_path = _atomic_npz(
                trajectory_root / f"{unit_id}.npz",
                global_states=np.asarray(encoding.global_states, dtype=np.float32),
                window_starts=np.asarray(encoding.window_starts, dtype=np.int64),
                window_stops=np.asarray(encoding.window_stops, dtype=np.int64),
            )
            common = {
                "unit_id": unit_id,
                "participant_id": str(row["participant_id"]),
                "dataset_id": "propofol_fmri",
                "partition": str(row["partition"]),
                "parcellation": PARCELLATION,
                "tr_seconds": float(row["tr_seconds"]),
                "volume_start": volume_start,
                "volume_stop": volume_stop,
                **{column: row.get(column) for column in label_columns},
            }
            encoded_rows.append(
                {
                    **common,
                    "trajectory_path": str(trajectory_path.resolve()),
                    "trajectory_sha256": sha256_file(trajectory_path),
                    "encoded_windows": int(encoding.global_states.shape[0]),
                    "labels_joined_after_encoding": True,
                }
            )
            summary_rows.append(
                {
                    **common,
                    **_summarise_unit(
                        encoding,
                        segment,
                        coordinates,
                        tr_seconds=float(row["tr_seconds"]),
                        random_state=study.random_seeds[index % len(study.random_seeds)],
                    ),
                }
            )
        except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as error:
            failures.append({"unit_id": unit_id, "error": f"{type(error).__name__}: {error}"})
    if not summary_rows:
        raise RuntimeError(f"no fMRI unit completed BrainLM triangulation; failures={failures}")

    encoded_path = _atomic_parquet(
        pd.DataFrame(encoded_rows), destination / "fmri-encoded-manifest.parquet"
    )
    summaries = pd.DataFrame(summary_rows)
    summaries_path = _atomic_parquet(summaries, destination / "fmri-unit-summaries.parquet")
    grouping = ["participant_id", "dataset_id", "partition"]
    if "condition" in summaries:
        grouping.append("condition")
    metric_columns = [
        column
        for column in summaries.columns
        if column.startswith(("brainlm_", "transition_", "alignment_", "fmri_"))
        and pd.api.types.is_numeric_dtype(summaries[column])
    ]
    participant = summaries.groupby(grouping, as_index=False)[metric_columns].mean()
    participant, axis_calibration = _calibrate_secondary_axes(participant)
    participant_path = _atomic_parquet(
        participant, destination / "fmri-participant-summaries.parquet"
    )
    partition_sets = {
        name: sorted(set(group["participant_id"].astype(str)))
        for name, group in frame.groupby("partition")
    }
    overlaps: dict[str, list[str]] = {}
    names = sorted(partition_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps[f"{left}|{right}"] = sorted(
                set(partition_sets[left]).intersection(partition_sets[right])
            )
    audit_path = destination / "fmri-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "study_sha256": config_sha256(study),
            "manifest": str(source_manifest),
            "manifest_sha256": sha256_file(source_manifest),
            "coordinates": str(coordinates_source),
            "coordinates_sha256": sha256_file(coordinates_source),
            "atlas": str(atlas_source) if atlas_source is not None else None,
            "atlas_sha256": sha256_file(atlas_source) if atlas_source is not None else None,
            "encoder": encoder_metadata,
            "normalization": normalization,
            "normalization_fit_partition": "discovery" if scale_min is not None else "external",
            "normalization_fit_participants": fit_participants,
            "normalization_min": scale_min,
            "normalization_max": scale_max,
            "secondary_axes": {
                "included": ["R", "M", "D", "A"],
                "excluded": ["reachability"],
                "fit_partition": "discovery",
                "calibration": axis_calibration,
            },
            "partition_participants": partition_sets,
            "partition_overlaps": overlaps,
            "run_provenance": list(run_provenance.values()),
            "units_attempted": len(frame),
            "units_completed": len(summary_rows),
            "failures": failures,
            "labels_joined_after_encoding": True,
            "representation_refit": False,
            "weights_frozen": True,
            "fMRI_scope": "secondary_cross_modal_triangulation",
            "direct_perturbational_inference": False,
            "project_status": "exploratory_non_preregistered",
            "scientific_gate_applied": False,
        },
    )
    return encoded_path, summaries_path, participant_path, audit_path
