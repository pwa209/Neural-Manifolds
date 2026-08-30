"""Conventional EEG scalar benchmarks on label-defined analysis units.

The stage consumes only non-event, non-clinical encoded rows.  It reads the
preprocessed FIF signal, publishes scalar summaries, and deliberately does not
write samples, spectra, connectivity matrices, or other signal-derived arrays.
Methods without an audited implementation remain explicit unavailable values.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_manifolds.benchmarks import (
    DEFAULT_BANDS,
    normalized_lempel_ziv,
    permutation_entropy,
    relative_band_power,
    spectral_exponent,
    weighted_phase_lag_index,
)
from neural_manifolds.provenance import atomic_write_json, sha256_file

BAND_POWER_FEATURES = tuple(f"relative_band_power_{name}" for name in DEFAULT_BANDS)
CONVENTIONAL_FEATURES = (
    *BAND_POWER_FEATURES,
    "spectral_exponent",
    "permutation_entropy_median",
    "normalized_lempel_ziv",
    "wpli_mean",
)
UNAVAILABLE_METHODS = {
    "wsmi": "unavailable_no_validated_backend",
    "microstates": "unavailable_no_validated_backend",
    "pcist": "unavailable_no_validated_backend",
}
_EVENT_SELECTOR_KINDS = frozenset({"event_epoch", "pre_epoched"})
_PATH_OR_SIGNAL_FIELDS = frozenset(
    {
        "trajectory_path",
        "preprocessed_path",
        "source_path",
        "source_file",
        "events_path",
        "channels_path",
        "selector_json",
    }
)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _read_raw_fif(path: Path) -> Any:
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("install neural-manifolds[eeg] for MNE FIF loading") from exc
    return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")


def _selector_kind(row: Mapping[str, Any]) -> str:
    value = row.get("selector_json")
    if isinstance(value, str) and value:
        try:
            selector = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("selector_json is not valid JSON") from error
        if not isinstance(selector, dict) or not isinstance(selector.get("kind"), str):
            raise ValueError("selector_json has no string kind")
        return str(selector["kind"])
    if isinstance(value, Mapping) and isinstance(value.get("kind"), str):
        return str(value["kind"])
    raise ValueError("encoding manifest row has no audited selector kind")


def _is_true(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _safe_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if (
            key in _PATH_OR_SIGNAL_FIELDS
            or key in CONVENTIONAL_FEATURES
            or key.endswith("_path")
            or key.endswith("_file")
        ):
            continue
        if value is None or isinstance(value, (str, int, float, bool, np.generic)):
            output[key] = value.item() if isinstance(value, np.generic) else value
    return output


def _features(data: np.ndarray, sfreq: float) -> dict[str, float]:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("benchmark data must contain at least two channels")
    if values.shape[1] < max(256, round(4 * sfreq)):
        raise ValueError("benchmark data must contain at least four seconds")
    if not np.all(np.isfinite(values)):
        raise ValueError("benchmark data contain non-finite samples")
    powers = relative_band_power(values, sfreq)
    phase_entropy = np.asarray([permutation_entropy(channel) for channel in values])
    wpli = weighted_phase_lag_index(values, sfreq)
    upper = wpli[np.triu_indices(wpli.shape[0], k=1)]
    if upper.size == 0:
        raise ValueError("wPLI requires at least two channels")
    result = {
        **{f"relative_band_power_{name}": float(value) for name, value in powers.items()},
        "spectral_exponent": spectral_exponent(values, sfreq),
        "permutation_entropy_median": float(np.median(phase_entropy)),
        "normalized_lempel_ziv": normalized_lempel_ziv(values),
        "wpli_mean": float(np.mean(upper)),
    }
    if set(result) != set(CONVENTIONAL_FEATURES) or not np.all(np.isfinite(list(result.values()))):
        raise RuntimeError("conventional benchmark calculation is incomplete or non-finite")
    return result


def _empty_frame() -> pd.DataFrame:
    columns = [
        "unit_id",
        "participant_id",
        "dataset_id",
        "condition",
        *CONVENTIONAL_FEATURES,
        "wsmi",
        "wsmi_status",
        "microstates",
        "microstates_status",
        "pcist",
        "pcist_status",
        "benchmark_status",
    ]
    return pd.DataFrame(columns=columns)


def run_benchmarks(
    *,
    encoding_manifest: str | Path,
    output_root: str | Path,
) -> tuple[Path, Path]:
    """Compute fixed conventional scalar comparators without scientific gating."""

    manifest_path = Path(encoding_manifest).resolve(strict=True)
    frame = pd.read_parquet(manifest_path)
    required = {
        "unit_id",
        "participant_id",
        "dataset_id",
        "condition",
        "encoded",
        "preprocessed_path",
        "selector_json",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"encoding manifest is missing {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    skipped = {
        "not_encoded": 0,
        "event_unit": 0,
        "clinical_holdout": 0,
        "secondary_fmri": 0,
        "missing_preprocessed_path": 0,
    }
    for row in frame.to_dict(orient="records"):
        identity = {
            "unit_id": str(row.get("unit_id", "")),
            "participant_id": str(row.get("participant_id", "")),
            "dataset_id": str(row.get("dataset_id", "")),
        }
        try:
            if not _is_true(row.get("encoded")):
                skipped["not_encoded"] += 1
                continue
            if _selector_kind(row) in _EVENT_SELECTOR_KINDS or _is_true(
                row.get("event_aggregated")
            ):
                skipped["event_unit"] += 1
                continue
            if _is_true(row.get("clinical_holdout")):
                skipped["clinical_holdout"] += 1
                continue
            if _is_true(row.get("secondary_fmri")) or row.get("modality") == "fmri":
                skipped["secondary_fmri"] += 1
                continue
            raw_path_value = row.get("preprocessed_path")
            if not isinstance(raw_path_value, str) or not raw_path_value:
                skipped["missing_preprocessed_path"] += 1
                continue
            raw_path = Path(raw_path_value).resolve(strict=True)
            expected_hash = row.get("preprocessed_sha256")
            if isinstance(expected_hash, str) and expected_hash:
                observed_hash = sha256_file(raw_path)
                if observed_hash != expected_hash:
                    raise ValueError("preprocessed FIF checksum mismatch")
            raw = _read_raw_fif(raw_path)
            try:
                data = np.asarray(raw.get_data(), dtype=np.float64)
                sfreq = float(raw.info["sfreq"])
                features = _features(data, sfreq)
                n_channels, n_samples = data.shape
            finally:
                close = getattr(raw, "close", None)
                if callable(close):
                    close()
            result = {
                **_safe_metadata(row),
                **features,
                "n_benchmark_channels": int(n_channels),
                "n_benchmark_samples": int(n_samples),
                "benchmark_duration_seconds": float(n_samples / sfreq),
                "wsmi": np.nan,
                "wsmi_status": UNAVAILABLE_METHODS["wsmi"],
                "microstates": np.nan,
                "microstates_status": UNAVAILABLE_METHODS["microstates"],
                "pcist": np.nan,
                "pcist_status": UNAVAILABLE_METHODS["pcist"],
                "benchmark_status": "computed",
            }
            rows.append(result)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            issues.append(
                {
                    **identity,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    destination = Path(output_root)
    output_frame = pd.DataFrame(rows) if rows else _empty_frame()
    benchmark_path = _atomic_parquet(output_frame, destination / "benchmarks.parquet")
    audit_path = destination / "benchmark-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "encoding_manifest_sha256": sha256_file(manifest_path),
            "manifest_rows": len(frame),
            "completed_rows": len(rows),
            "failed_rows": len(issues),
            "skipped_rows": skipped,
            "issues": issues,
            "published_features": list(CONVENTIONAL_FEATURES),
            "unavailable_methods": UNAVAILABLE_METHODS,
            "analysis_unit": "participant_condition_analysis_unit",
            "raw_or_array_artifacts_published": False,
            "path_fields_published": False,
            "scientific_gate_applied": False,
        },
    )
    return benchmark_path, audit_path
