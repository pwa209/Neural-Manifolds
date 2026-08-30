"""Contrast-specific equal-window and duration-reliability sensitivity stage.

The stage consumes already encoded trajectories and already fitted state/profile
objects.  Labels are used only to form post-encoding contrast groups; neither
fitted object is updated.  Sampling operates on contiguous synchronized spans,
and every selected span receives a new segment id so transitions are never
created across recordings, artifact gaps, or event trials.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from neural_manifolds.config import StudyConfig, config_sha256
from neural_manifolds.dynamics.state_dictionary import StateDictionary
from neural_manifolds.foundation.overlap import (
    OVERLAP_OUTPUT_COLUMNS,
    ensure_pretraining_overlap_columns,
    overlap_output_fields,
    summarize_pretraining_overlap,
)
from neural_manifolds.manifold.profile import (
    AXIS_NAMES,
    FiveAxisProfileEstimator,
    ManifoldRecord,
)
from neural_manifolds.provenance import atomic_write_json, sha256_file

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class SynchronizedSegment:
    """One transition-safe coarse span and its temporally overlapping fine span."""

    unit_id: str
    segment_key: str
    order: int
    trajectory: FloatMatrix
    coarse_starts: NDArray[np.float64]
    regional: Mapping[str, FloatMatrix]
    fine_starts: NDArray[np.float64]
    coarse_window_seconds: float
    coarse_step_seconds: float
    fine_window_seconds: float
    fine_step_seconds: float


@dataclass(frozen=True)
class UnitSegments:
    unit_id: str
    participant_id: str
    dataset_id: str
    row: Mapping[str, Any]
    segments: tuple[SynchronizedSegment, ...]


@dataclass(frozen=True)
class ProfileSource:
    source_id: str
    participant_id: str
    dataset_id: str
    contrast_arm: str
    condition_levels: tuple[str, ...]
    stratum_id: str
    units: tuple[UnitSegments, ...]
    segments: tuple[SynchronizedSegment, ...]


@dataclass(frozen=True)
class SampledSpan:
    segment: SynchronizedSegment
    coarse_start: int
    coarse_count: int
    trajectory: FloatMatrix
    regional: Mapping[str, FloatMatrix]
    coarse_starts: NDArray[np.float64]
    fine_starts: NDArray[np.float64]


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = json.dumps([int(base_seed), *parts], sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _is_true(value: Any) -> bool:
    """Interpret manifest flags without treating NaN/unknown values as true."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) == 1
    return False


def _unit_key(row: Mapping[str, Any]) -> str:
    for column in ("unit_id", "recording_id"):
        value = row.get(column)
        if isinstance(value, str) and value:
            return value
    raise ValueError("encoding manifest row has no unit_id or recording_id")


def _selector_kind(row: Mapping[str, Any]) -> str | None:
    if _is_true(row.get("event_aggregated", False)):
        return "event_aggregated"
    value = row.get("selector")
    if isinstance(value, Mapping):
        kind = value.get("kind")
        return str(kind) if kind is not None else None
    for key in ("selector_json", "selector"):
        value = row.get(key)
        if isinstance(value, str) and value:
            try:
                payload = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping) and payload.get("kind") is not None:
                return str(payload["kind"])
    return None


def _validated_matrix(value: Any, *, name: str) -> FloatMatrix:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty time-by-feature matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _validated_segments(value: Any | None, length: int, *, name: str) -> NDArray[Any]:
    if value is None:
        return np.zeros(length, dtype=np.int64)
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must align with its trajectory")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validated_starts(
    value: Any | None,
    length: int,
    *,
    sampling_hz: float,
    name: str,
) -> NDArray[np.float64] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or len(array) != length or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and align with its trajectory")
    return array / float(sampling_hz)


def _track_runs(
    segment_ids: NDArray[Any],
    starts: NDArray[np.float64],
    *,
    expected_step: float,
) -> list[tuple[str, int, int]]:
    if len(segment_ids) != len(starts):
        raise ValueError("track starts and segment ids differ in length")
    boundaries = [0]
    tolerance = max(1e-9, expected_step * 0.25)
    for index in range(1, len(starts)):
        changed_segment = segment_ids[index] != segment_ids[index - 1]
        discontinuity = abs((starts[index] - starts[index - 1]) - expected_step) > tolerance
        if changed_segment or discontinuity:
            boundaries.append(index)
    boundaries.append(len(starts))
    return [
        (str(segment_ids[start]), start, stop)
        for start, stop in pairwise(boundaries)
        if stop > start
    ]


def _event_segments_without_starts(
    *,
    unit_id: str,
    trajectory: FloatMatrix,
    regional: Mapping[str, FloatMatrix],
    coarse_ids: NDArray[Any],
    fine_ids: NDArray[Any],
    window_seconds: float,
    step_seconds: float,
) -> tuple[SynchronizedSegment, ...]:
    if len(trajectory) != len(next(iter(regional.values()))):
        raise ValueError("start-free coarse and fine event tracks differ in length")
    coarse_boundaries = np.r_[
        0, np.flatnonzero(coarse_ids[1:] != coarse_ids[:-1]) + 1, len(coarse_ids)
    ]
    fine_boundaries = np.r_[0, np.flatnonzero(fine_ids[1:] != fine_ids[:-1]) + 1, len(fine_ids)]
    if not np.array_equal(coarse_boundaries, fine_boundaries):
        raise ValueError("start-free event tracks have different trial boundaries")
    output: list[SynchronizedSegment] = []
    for order, (start, stop) in enumerate(pairwise(coarse_boundaries)):
        length = int(stop - start)
        local_starts = np.arange(length, dtype=np.float64) * step_seconds
        output.append(
            SynchronizedSegment(
                unit_id=unit_id,
                segment_key=f"{unit_id}:{order}:{coarse_ids[start]}",
                order=order,
                trajectory=trajectory[start:stop],
                coarse_starts=local_starts,
                regional={name: values[start:stop] for name, values in regional.items()},
                fine_starts=local_starts.copy(),
                coarse_window_seconds=window_seconds,
                coarse_step_seconds=step_seconds,
                fine_window_seconds=window_seconds,
                fine_step_seconds=step_seconds,
            )
        )
    return tuple(output)


def _synchronize_tracks(
    *,
    unit_id: str,
    trajectory: FloatMatrix,
    regional: Mapping[str, FloatMatrix],
    coarse_ids: NDArray[Any],
    fine_ids: NDArray[Any],
    coarse_starts: NDArray[np.float64],
    fine_starts: NDArray[np.float64],
    coarse_window: float,
    coarse_step: float,
    fine_window: float,
    fine_step: float,
) -> tuple[SynchronizedSegment, ...]:
    coarse_runs = _track_runs(coarse_ids, coarse_starts, expected_step=coarse_step)
    fine_runs = _track_runs(fine_ids, fine_starts, expected_step=fine_step)
    tolerance = max(1e-9, min(coarse_step, fine_step) * 0.25)
    output: list[SynchronizedSegment] = []
    order = 0
    for coarse_label, coarse_start, coarse_stop in coarse_runs:
        coarse_run_starts = coarse_starts[coarse_start:coarse_stop]
        coarse_run_end = coarse_run_starts[-1] + coarse_window
        for fine_label, fine_start, fine_stop in fine_runs:
            if fine_label != coarse_label:
                continue
            fine_run_starts = fine_starts[fine_start:fine_stop]
            fine_run_end = fine_run_starts[-1] + fine_window
            overlap_start = max(coarse_run_starts[0], fine_run_starts[0])
            overlap_stop = min(coarse_run_end, fine_run_end)
            if overlap_stop <= overlap_start:
                continue
            coarse_mask = (coarse_run_starts >= overlap_start - tolerance) & (
                coarse_run_starts + coarse_window <= overlap_stop + tolerance
            )
            fine_mask = (fine_run_starts >= overlap_start - tolerance) & (
                fine_run_starts + fine_window <= overlap_stop + tolerance
            )
            coarse_local = np.flatnonzero(coarse_mask)
            fine_local = np.flatnonzero(fine_mask)
            if len(coarse_local) == 0 or len(fine_local) == 0:
                continue
            coarse_indices = coarse_start + coarse_local
            fine_indices = fine_start + fine_local
            output.append(
                SynchronizedSegment(
                    unit_id=unit_id,
                    segment_key=f"{unit_id}:{order}:{coarse_label}",
                    order=order,
                    trajectory=trajectory[coarse_indices],
                    coarse_starts=coarse_starts[coarse_indices],
                    regional={name: values[fine_indices] for name, values in regional.items()},
                    fine_starts=fine_starts[fine_indices],
                    coarse_window_seconds=coarse_window,
                    coarse_step_seconds=coarse_step,
                    fine_window_seconds=fine_window,
                    fine_step_seconds=fine_step,
                )
            )
            order += 1
    if not output:
        raise ValueError("coarse and fine tracks have no complete synchronized span")
    return tuple(output)


def _load_unit(row: Mapping[str, Any], study: StudyConfig) -> UnitSegments:
    path = Path(str(row.get("trajectory_path"))).resolve(strict=True)
    expected = row.get("trajectory_sha256")
    if isinstance(expected, str) and expected and sha256_file(path) != expected:
        raise ValueError("trajectory checksum differs from the encoding manifest")
    unit_id = _unit_key(row)
    with np.load(path, allow_pickle=False) as archive:
        if "global_states" not in archive:
            raise ValueError("trajectory archive has no global_states")
        trajectory = _validated_matrix(archive["global_states"], name="global_states")
        fine_names = sorted(
            name for name in archive.files if name.startswith("alignment_regional_")
        )
        prefix = "alignment_regional_" if len(fine_names) >= 2 else "regional_"
        regional = {
            name.removeprefix(prefix): _validated_matrix(archive[name], name=name)
            for name in archive.files
            if name.startswith(prefix)
        }
        if len(regional) < 2:
            raise ValueError("trajectory archive has fewer than two regional tracks")
        fine_length = len(next(iter(regional.values())))
        if len({len(value) for value in regional.values()}) != 1:
            raise ValueError("regional tracks differ in length")
        coarse_ids = _validated_segments(
            archive.get("segment_ids"),
            len(trajectory),
            name="segment_ids",
        )
        fine_ids = _validated_segments(
            archive.get("alignment_segment_ids"),
            fine_length,
            name="alignment_segment_ids",
        )
        coarse_starts = _validated_starts(
            archive.get("window_start_samples"),
            len(trajectory),
            sampling_hz=study.preprocessing.target_sampling_hz,
            name="window_start_samples",
        )
        fine_starts = _validated_starts(
            archive.get("alignment_window_start_samples"),
            fine_length,
            sampling_hz=study.preprocessing.target_sampling_hz,
            name="alignment_window_start_samples",
        )
    event = _selector_kind(row) in {"event_epoch", "pre_epoched", "event_aggregated"}
    coarse_window = (
        study.representation.alignment_window_seconds
        if event
        else study.representation.harmonised_window_seconds
    )
    coarse_step = (
        study.representation.alignment_step_seconds
        if event
        else study.representation.harmonised_step_seconds
    )
    if coarse_starts is None or fine_starts is None:
        if coarse_starts is not None or fine_starts is not None:
            raise ValueError("only one temporal track supplies start samples")
        segments = _event_segments_without_starts(
            unit_id=unit_id,
            trajectory=trajectory,
            regional=regional,
            coarse_ids=coarse_ids,
            fine_ids=fine_ids,
            window_seconds=coarse_window,
            step_seconds=coarse_step,
        )
    else:
        segments = _synchronize_tracks(
            unit_id=unit_id,
            trajectory=trajectory,
            regional=regional,
            coarse_ids=coarse_ids,
            fine_ids=fine_ids,
            coarse_starts=coarse_starts,
            fine_starts=fine_starts,
            coarse_window=coarse_window,
            coarse_step=coarse_step,
            fine_window=study.representation.alignment_window_seconds,
            fine_step=study.representation.alignment_step_seconds,
        )
    return UnitSegments(
        unit_id=unit_id,
        participant_id=str(row["participant_id"]),
        dataset_id=str(row["dataset_id"]),
        # Preserve nested ``variables`` mappings for post-encoding selectors.
        # These labels are never passed to either frozen fitted object.
        row=dict(row),
        segments=segments,
    )


def _load_contrasts(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("contrast configuration must use schema_version 1")
    output: list[dict[str, Any]] = []
    top_level = payload.get("contrasts")
    if isinstance(top_level, list):
        for item in top_level:
            if not isinstance(item, Mapping):
                raise ValueError("top-level contrast entries must be mappings")
            output.append(dict(item))
    datasets = payload.get("datasets")
    if isinstance(datasets, Mapping):
        for dataset_id, dataset_payload in datasets.items():
            if not isinstance(dataset_payload, Mapping):
                continue
            for item in dataset_payload.get("contrasts", []):
                if not isinstance(item, Mapping):
                    raise ValueError(f"contrast for {dataset_id} must be a mapping")
                output.append({"dataset_id": str(dataset_id), **dict(item)})
    if not output:
        raise ValueError("contrast configuration has no contrasts")
    identifiers = [str(item.get("id", "")) for item in output]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("contrast ids must be present and unique")
    return output


def _contrast_arms(contrast: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    if contrast.get("positive") is not None and contrast.get("reference") is not None:
        return {
            "positive": tuple(str(value) for value in contrast["positive"]),
            "reference": tuple(str(value) for value in contrast["reference"]),
        }
    conditions = contrast.get("conditions")
    if (
        isinstance(conditions, Sequence)
        and not isinstance(conditions, str)
        and len(conditions) >= 2
    ):
        return {str(value): (str(value),) for value in conditions}
    return {}


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    variables = row.get("variables")
    if isinstance(variables, Mapping):
        return variables.get(key)
    return None


def _subset_matches(row: Mapping[str, Any], subset: Mapping[str, Any]) -> bool:
    for key, expected in subset.items():
        observed = _row_value(row, str(key))
        if isinstance(expected, Sequence) and not isinstance(expected, str):
            if observed not in expected:
                return False
        elif observed != expected:
            return False
    return True


def _stratum_id(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    if not keys:
        return "all"
    values = [(key, _safe_scalar(_row_value(row, key))) for key in keys]
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _profile_sources(
    units: Sequence[UnitSegments],
    contrast: Mapping[str, Any],
    arms: Mapping[str, tuple[str, ...]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, list[ProfileSource]]:
    contrast_id = str(contrast["id"])
    dataset_id = contrast.get("dataset_id")
    match_on = contrast.get("match_on")
    if isinstance(match_on, str):
        match_keys = [match_on]
    elif isinstance(match_on, Sequence):
        match_keys = [str(value) for value in match_on]
    elif contrast.get("match_within") is not None:
        match_keys = [str(contrast["match_within"])]
    else:
        match_keys = []
    subset = contrast.get("subset") or {}
    if not isinstance(subset, Mapping):
        raise ValueError(f"contrast {contrast_id} subset must be a mapping")
    condition_column = str(contrast.get("label_column", "condition"))
    selected: list[tuple[UnitSegments, str, str]] = []
    for unit in units:
        if dataset_id is not None and unit.dataset_id != str(dataset_id):
            continue
        if not _subset_matches(unit.row, subset):
            continue
        condition = str(_row_value(unit.row, condition_column))
        membership = [arm for arm, levels in arms.items() if condition in levels]
        if len(membership) == 1:
            missing_match_keys = [
                key
                for key in match_keys
                if _row_value(unit.row, key) is None or bool(pd.isna(_row_value(unit.row, key)))
            ]
            if missing_match_keys:
                audit_rows.append(
                    {
                        "status": "unavailable",
                        "contrast_id": contrast_id,
                        "unit_id": unit.unit_id,
                        "reason": "required matching value is unavailable",
                        "missing_match_keys": missing_match_keys,
                    }
                )
                continue
            selected.append((unit, membership[0], condition))
        elif len(membership) > 1:
            audit_rows.append(
                {
                    "status": "error",
                    "contrast_id": contrast_id,
                    "unit_id": unit.unit_id,
                    "reason": "condition belongs to multiple contrast arms",
                }
            )
    grouped: dict[tuple[str, str, str, str], list[UnitSegments]] = defaultdict(list)
    for unit, arm, condition in selected:
        stratum = _stratum_id(unit.row, match_keys)
        grouped[(stratum, unit.participant_id, arm, condition)].append(unit)
    by_stratum: dict[str, list[ProfileSource]] = defaultdict(list)
    for (stratum, participant, arm, condition), grouped_units in sorted(grouped.items()):
        ordered_units = tuple(sorted(grouped_units, key=lambda value: value.unit_id))
        segments = tuple(
            segment
            for unit in ordered_units
            for segment in sorted(unit.segments, key=lambda value: (value.unit_id, value.order))
        )
        source_id = hashlib.sha256(
            f"{contrast_id}|{stratum}|{participant}|{arm}|{condition}".encode()
        ).hexdigest()[:20]
        by_stratum[stratum].append(
            ProfileSource(
                source_id=source_id,
                participant_id=participant,
                dataset_id=ordered_units[0].dataset_id,
                contrast_arm=arm,
                condition_levels=(condition,),
                stratum_id=stratum,
                units=ordered_units,
                segments=segments,
            )
        )
    return dict(by_stratum)


def _common_template(sources: Sequence[ProfileSource]) -> tuple[int, ...]:
    if not sources or any(not source.segments for source in sources):
        return ()
    ranked = [
        sorted((len(segment.trajectory) for segment in source.segments), reverse=True)
        for source in sources
    ]
    count = min(len(values) for values in ranked)
    return tuple(min(values[index] for values in ranked) for index in range(count))


def _duration_template(
    template: Sequence[int],
    *,
    duration_seconds: float,
    window_seconds: float,
    step_seconds: float,
) -> tuple[int, ...]:
    if duration_seconds <= 0:
        raise ValueError("reliability duration must be positive")
    available_seconds = sum(
        window_seconds + (int(count) - 1) * step_seconds for count in template if count > 0
    )
    if available_seconds + 1e-9 < duration_seconds:
        return ()
    remaining = float(duration_seconds)
    output: list[int] = []
    tolerance = 1e-9
    for available in template:
        block_seconds = window_seconds + (int(available) - 1) * step_seconds
        if remaining <= window_seconds + tolerance:
            selected = 1
        elif remaining <= block_seconds + tolerance:
            selected = math.ceil((remaining - window_seconds - tolerance) / step_seconds) + 1
        else:
            selected = int(available)
        output.append(selected)
        remaining -= window_seconds + (selected - 1) * step_seconds
        if remaining <= tolerance:
            break
    if remaining > tolerance:
        return ()
    return tuple(output)


def _sample_source(
    source: ProfileSource,
    template: Sequence[int],
    *,
    seed: int,
) -> list[SampledSpan]:
    rng = np.random.default_rng(seed)
    remaining = list(range(len(source.segments)))
    output: list[SampledSpan] = []
    tolerance = 1e-9
    for count in template:
        eligible = [index for index in remaining if len(source.segments[index].trajectory) >= count]
        if not eligible:
            raise ValueError("profile source cannot realize the common segment template")
        selected_index = eligible[int(rng.integers(0, len(eligible)))]
        remaining.remove(selected_index)
        segment = source.segments[selected_index]
        maximum_start = len(segment.trajectory) - int(count)
        coarse_start = int(rng.integers(0, maximum_start + 1)) if maximum_start else 0
        coarse_stop = coarse_start + int(count)
        selected_coarse_starts = segment.coarse_starts[coarse_start:coarse_stop]
        span_start = float(selected_coarse_starts[0])
        span_stop = float(selected_coarse_starts[-1] + segment.coarse_window_seconds)
        fine_mask = (segment.fine_starts >= span_start - tolerance) & (
            segment.fine_starts + segment.fine_window_seconds <= span_stop + tolerance
        )
        if not np.any(fine_mask):
            raise ValueError("sampled coarse span has no complete synchronized fine windows")
        output.append(
            SampledSpan(
                segment=segment,
                coarse_start=coarse_start,
                coarse_count=int(count),
                trajectory=segment.trajectory[coarse_start:coarse_stop],
                regional={name: values[fine_mask] for name, values in segment.regional.items()},
                coarse_starts=selected_coarse_starts,
                fine_starts=segment.fine_starts[fine_mask],
            )
        )
    return output


def _equalize_fine_spans(samples: Mapping[str, list[SampledSpan]]) -> dict[str, list[SampledSpan]]:
    if not samples:
        return {}
    block_counts = {len(value) for value in samples.values()}
    if len(block_counts) != 1:
        raise AssertionError("matched samples have different segment counts")
    n_blocks = block_counts.pop()
    output: dict[str, list[SampledSpan]] = {key: [] for key in samples}
    for block in range(n_blocks):
        fine_count = min(
            len(next(iter(spans[block].regional.values()))) for spans in samples.values()
        )
        if fine_count <= 0:
            raise ValueError("matched fine-alignment span is empty")
        for source_id, spans in samples.items():
            span = spans[block]
            observed = len(next(iter(span.regional.values())))
            offset = (observed - fine_count) // 2
            stop = offset + fine_count
            output[source_id].append(
                SampledSpan(
                    segment=span.segment,
                    coarse_start=span.coarse_start,
                    coarse_count=span.coarse_count,
                    trajectory=span.trajectory,
                    regional={name: values[offset:stop] for name, values in span.regional.items()},
                    coarse_starts=span.coarse_starts,
                    fine_starts=span.fine_starts[offset:stop],
                )
            )
    return output


def _record_from_spans(
    source: ProfileSource,
    spans: Sequence[SampledSpan],
    dictionary: StateDictionary,
) -> tuple[ManifoldRecord, str]:
    if not spans:
        raise ValueError("cannot profile an empty matched sample")
    # Sampling chooses segments randomly, but their contents and their order in
    # the reconstructed record remain chronological/deterministic.  Segment ids
    # still prevent transitions between independently selected spans.
    spans = tuple(
        sorted(
            spans,
            key=lambda span: (
                span.segment.unit_id,
                span.segment.order,
                span.coarse_start,
            ),
        )
    )
    region_names = set(spans[0].regional)
    for span in spans[1:]:
        region_names.intersection_update(span.regional)
    if len(region_names) < 2:
        raise ValueError("matched sample has fewer than two shared regions")
    trajectory = np.concatenate([span.trajectory for span in spans], axis=0)
    coarse_segment_ids = np.concatenate(
        [np.full(len(span.trajectory), index, dtype=np.int64) for index, span in enumerate(spans)]
    )
    regional = {
        name: np.concatenate([span.regional[name] for span in spans], axis=0)
        for name in sorted(region_names)
    }
    fine_segment_ids = np.concatenate(
        [
            np.full(len(next(iter(span.regional.values()))), index, dtype=np.int64)
            for index, span in enumerate(spans)
        ]
    )
    projected = dictionary.project(trajectory)
    states = dictionary.predict_projected(projected, segment_ids=coarse_segment_ids)
    fingerprint_payload = [
        [span.segment.segment_key, span.coarse_start, span.coarse_count, len(span.fine_starts)]
        for span in spans
    ]
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, separators=(",", ":")).encode()
    ).hexdigest()
    return (
        ManifoldRecord(
            trajectory=projected,
            states=states,
            regional_trajectories=regional,
            repertoire_trajectory=trajectory,
            segment_ids=coarse_segment_ids,
            alignment_segment_ids=fine_segment_ids,
            name=f"sampling:{source.source_id}:{fingerprint[:12]}",
        ),
        fingerprint,
    )


def _repeat_group(
    *,
    contrast_id: str,
    sources: Sequence[ProfileSource],
    template: Sequence[int],
    estimator: FiveAxisProfileEstimator,
    dictionary: StateDictionary,
    study: StudyConfig,
    repeats: int,
    analysis: str,
    duration_seconds: float | None,
    audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not template:
        audit_rows.append(
            {
                "status": "unavailable",
                "contrast_id": contrast_id,
                "stratum_id": sources[0].stratum_id if sources else None,
                "analysis": analysis,
                "duration_seconds": duration_seconds,
                "reason": "no common transition-safe synchronized duration",
            }
        )
        return rows
    for repeat in range(repeats):
        try:
            sampled: dict[str, list[SampledSpan]] = {}
            seeds: dict[str, int] = {}
            for source in sorted(sources, key=lambda value: value.source_id):
                seed = _stable_seed(
                    study.random_seeds[repeat % len(study.random_seeds)],
                    contrast_id,
                    source.stratum_id,
                    source.source_id,
                    analysis,
                    duration_seconds,
                    repeat,
                )
                seeds[source.source_id] = seed
                sampled[source.source_id] = _sample_source(source, template, seed=seed)
            sampled = _equalize_fine_spans(sampled)
            pending: list[dict[str, Any]] = []
            for source in sorted(sources, key=lambda value: value.source_id):
                spans = sampled[source.source_id]
                record, fingerprint = _record_from_spans(source, spans, dictionary)
                profile = estimator.profile(record)
                coarse_windows = int(sum(len(span.trajectory) for span in spans))
                fine_windows = int(sum(len(next(iter(span.regional.values()))) for span in spans))
                effective_seconds = float(
                    sum(
                        span.segment.coarse_window_seconds
                        + (len(span.trajectory) - 1) * span.segment.coarse_step_seconds
                        for span in spans
                    )
                )
                row: dict[str, Any] = {
                    "analysis": analysis,
                    "contrast_id": contrast_id,
                    "dataset_id": source.dataset_id,
                    "stratum_id": source.stratum_id,
                    "profile_id": source.source_id,
                    "participant_id": source.participant_id,
                    "contrast_arm": source.contrast_arm,
                    "condition_levels": "|".join(source.condition_levels),
                    "repeat": repeat,
                    "seed": seeds[source.source_id],
                    "duration_seconds": duration_seconds,
                    "matched_segments": len(spans),
                    "matched_coarse_windows": coarse_windows,
                    "matched_fine_windows": fine_windows,
                    "matched_effective_seconds": effective_seconds,
                    "sample_fingerprint": fingerprint,
                    "source_unit_count": len(source.units),
                    "source_unit_ids": "|".join(unit.unit_id for unit in source.units),
                    **overlap_output_fields(
                        pd.DataFrame([dict(unit.row) for unit in source.units])
                    ),
                }
                for index, axis in enumerate(AXIS_NAMES):
                    row[axis] = float(profile.values[index])
                    row[f"{axis}_raw"] = float(profile.raw_values[index])
                pending.append(row)
            rows.extend(pending)
        except (OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            audit_rows.append(
                {
                    "status": "error",
                    "contrast_id": contrast_id,
                    "stratum_id": sources[0].stratum_id if sources else None,
                    "analysis": analysis,
                    "duration_seconds": duration_seconds,
                    "repeat": repeat,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    return rows


def _average_profiles(repeats: pd.DataFrame) -> pd.DataFrame:
    frame = repeats[repeats["analysis"] == "equal_window"].copy()
    identity = [
        "contrast_id",
        "dataset_id",
        "stratum_id",
        "profile_id",
        "participant_id",
        "contrast_arm",
        "condition_levels",
        "source_unit_count",
        "source_unit_ids",
        *OVERLAP_OUTPUT_COLUMNS,
    ]
    columns = [
        *identity,
        "successful_repeats",
        "matched_segments",
        "matched_coarse_windows",
        "matched_fine_windows",
        "matched_effective_seconds",
        *[
            f"{axis}_{suffix}"
            for axis in AXIS_NAMES
            for suffix in ("mean", "sd", "q025", "q975", "raw_mean", "raw_sd")
        ],
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(identity, dropna=False, sort=True):
        row = dict(zip(identity, keys, strict=True))
        row.update(
            {
                "successful_repeats": int(group["repeat"].nunique()),
                "matched_segments": int(group["matched_segments"].iloc[0]),
                "matched_coarse_windows": int(group["matched_coarse_windows"].iloc[0]),
                "matched_fine_windows": int(group["matched_fine_windows"].iloc[0]),
                "matched_effective_seconds": float(group["matched_effective_seconds"].iloc[0]),
            }
        )
        for axis in AXIS_NAMES:
            values = group[axis].to_numpy(dtype=float)
            raw = group[f"{axis}_raw"].to_numpy(dtype=float)
            row.update(
                {
                    f"{axis}_mean": float(np.mean(values)),
                    f"{axis}_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    f"{axis}_q025": float(np.quantile(values, 0.025)),
                    f"{axis}_q975": float(np.quantile(values, 0.975)),
                    f"{axis}_raw_mean": float(np.mean(raw)),
                    f"{axis}_raw_sd": float(np.std(raw, ddof=1)) if len(raw) > 1 else 0.0,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _icc3_1(matrix: NDArray[np.float64]) -> float:
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("ICC requires at least two profiles and two repeats")
    n_profiles, n_repeats = matrix.shape
    profile_means = np.mean(matrix, axis=1)
    repeat_means = np.mean(matrix, axis=0)
    grand_mean = float(np.mean(matrix))
    between = n_repeats * float(np.sum((profile_means - grand_mean) ** 2)) / (n_profiles - 1)
    residual_sum = float(
        np.sum((matrix - profile_means[:, None] - repeat_means[None, :] + grand_mean) ** 2)
    )
    residual = residual_sum / ((n_profiles - 1) * (n_repeats - 1))
    denominator = between + (n_repeats - 1) * residual
    if denominator <= np.finfo(float).eps:
        raise ValueError("ICC is undefined because between-profile variance is zero")
    return float((between - residual) / denominator)


def _reliability_curves(repeats: pd.DataFrame, audit_rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = repeats[repeats["analysis"] == "reliability"].copy()
    columns = [
        "contrast_id",
        "dataset_id",
        "contrast_arm",
        "duration_seconds",
        *OVERLAP_OUTPUT_COLUMNS,
        "axis",
        "status",
        "profile_count",
        "repeat_count",
        "icc3_1",
        "median_within_profile_sd",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_columns = [
        "contrast_id",
        "dataset_id",
        "contrast_arm",
        "duration_seconds",
        *OVERLAP_OUTPUT_COLUMNS,
    ]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        for axis in AXIS_NAMES:
            pivot = group.pivot_table(index="profile_id", columns="repeat", values=axis)
            pivot = pivot.dropna(axis=0, how="any")
            row = dict(zip(group_columns, keys, strict=True))
            row.update(
                {
                    "axis": axis,
                    "profile_count": int(pivot.shape[0]),
                    "repeat_count": int(pivot.shape[1]),
                }
            )
            try:
                values = pivot.to_numpy(dtype=float)
                row.update(
                    {
                        "status": "available",
                        "icc3_1": _icc3_1(values),
                        "median_within_profile_sd": float(
                            np.median(np.std(values, axis=1, ddof=1))
                        ),
                    }
                )
            except ValueError as error:
                row.update(
                    {
                        "status": "unavailable",
                        "icc3_1": np.nan,
                        "median_within_profile_sd": np.nan,
                    }
                )
                audit_rows.append(
                    {
                        "status": "unavailable",
                        **dict(zip(group_columns, keys, strict=True)),
                        "axis": axis,
                        "analysis": "reliability_curve",
                        "reason": str(error),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def run_sampling_sensitivity(
    *,
    encoding_manifest: str | Path,
    state_dictionary_path: str | Path,
    profile_estimator_path: str | Path,
    contrasts_path: str | Path,
    output_root: str | Path,
    study: StudyConfig,
    repeats: int | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Run contrast-specific matched-window and reliability sensitivity analyses.

    Returns ``(repeat_profiles, averaged_profiles, reliability_curves, audit)``.
    Per-contrast data limitations and numerical failures are written to the
    audit and do not act as scientific gates.
    """

    repeat_count = study.sampling.repeats if repeats is None else int(repeats)
    if repeat_count <= 0:
        raise ValueError("repeats must be positive")
    if not study.sampling.equalise_windows:
        raise ValueError("sampling sensitivity requires sampling.equalise_windows=true")
    dictionary = joblib.load(Path(state_dictionary_path).resolve(strict=True))
    estimator = joblib.load(Path(profile_estimator_path).resolve(strict=True))
    if not isinstance(dictionary, StateDictionary):
        raise TypeError("state dictionary artifact has the wrong type")
    if not isinstance(estimator, FiveAxisProfileEstimator):
        raise TypeError("profile estimator artifact has the wrong type")
    frame = pd.read_parquet(encoding_manifest)
    required = {"participant_id", "dataset_id", "trajectory_path", "encoded"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"encoding manifest is missing {sorted(missing)}")
    frame = ensure_pretraining_overlap_columns(frame, default_model_id="labram_base")
    audit_rows: list[dict[str, Any]] = []
    units: list[UnitSegments] = []
    eligible = frame[frame["encoded"].fillna(False).astype(bool)].copy()
    ordered = eligible.assign(_unit_order=eligible.apply(_unit_key, axis=1)).sort_values(
        ["dataset_id", "participant_id", "_unit_order"]
    )
    for row in ordered.drop(columns="_unit_order").to_dict(orient="records"):
        try:
            units.append(_load_unit(row, study))
        except (OSError, RuntimeError, ValueError) as error:
            audit_rows.append(
                {
                    "status": "error",
                    "unit_id": row.get("unit_id") or row.get("recording_id"),
                    "dataset_id": row.get("dataset_id"),
                    "analysis": "load_synchronized_tracks",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    contrasts = _load_contrasts(contrasts_path)
    repeat_rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        contrast_id = str(contrast["id"])
        arms = _contrast_arms(contrast)
        if not arms:
            audit_rows.append(
                {
                    "status": "unavailable",
                    "contrast_id": contrast_id,
                    "analysis": "equal_window",
                    "reason": "contrast has no explicit categorical arms",
                }
            )
            continue
        by_stratum = _profile_sources(units, contrast, arms, audit_rows)
        if not by_stratum:
            audit_rows.append(
                {
                    "status": "unavailable",
                    "contrast_id": contrast_id,
                    "analysis": "equal_window",
                    "reason": "no encoded rows match the contrast",
                }
            )
            continue
        for stratum, sources in sorted(by_stratum.items()):
            present = {source.contrast_arm for source in sources}
            if present != set(arms):
                audit_rows.append(
                    {
                        "status": "unavailable",
                        "contrast_id": contrast_id,
                        "stratum_id": stratum,
                        "analysis": "equal_window",
                        "reason": "matched stratum does not contain every contrast arm",
                        "present_arms": sorted(present),
                        "required_arms": sorted(arms),
                    }
                )
                continue
            steps = {
                (
                    source.segments[0].coarse_window_seconds,
                    source.segments[0].coarse_step_seconds,
                )
                for source in sources
                if source.segments
            }
            if len(steps) != 1:
                audit_rows.append(
                    {
                        "status": "unavailable",
                        "contrast_id": contrast_id,
                        "stratum_id": stratum,
                        "analysis": "equal_window",
                        "reason": "matched sources use different coarse temporal grids",
                    }
                )
                continue
            template = _common_template(sources)
            repeat_rows.extend(
                _repeat_group(
                    contrast_id=contrast_id,
                    sources=sources,
                    template=template,
                    estimator=estimator,
                    dictionary=dictionary,
                    study=study,
                    repeats=repeat_count,
                    analysis="equal_window",
                    duration_seconds=None,
                    audit_rows=audit_rows,
                )
            )
            coarse_window, coarse_step = next(iter(steps))
            for duration in study.sampling.reliability_seconds:
                duration_template = _duration_template(
                    template,
                    duration_seconds=float(duration),
                    window_seconds=float(coarse_window),
                    step_seconds=float(coarse_step),
                )
                repeat_rows.extend(
                    _repeat_group(
                        contrast_id=contrast_id,
                        sources=sources,
                        template=duration_template,
                        estimator=estimator,
                        dictionary=dictionary,
                        study=study,
                        repeats=repeat_count,
                        analysis="reliability",
                        duration_seconds=float(duration),
                        audit_rows=audit_rows,
                    )
                )
    repeat_columns = [
        "analysis",
        "contrast_id",
        "dataset_id",
        "stratum_id",
        "profile_id",
        "participant_id",
        "contrast_arm",
        "condition_levels",
        "repeat",
        "seed",
        "duration_seconds",
        "matched_segments",
        "matched_coarse_windows",
        "matched_fine_windows",
        "matched_effective_seconds",
        "sample_fingerprint",
        "source_unit_count",
        "source_unit_ids",
        *OVERLAP_OUTPUT_COLUMNS,
        *AXIS_NAMES,
        *(f"{axis}_raw" for axis in AXIS_NAMES),
    ]
    repeat_frame = pd.DataFrame(repeat_rows, columns=repeat_columns)
    average_frame = _average_profiles(repeat_frame)
    reliability_frame = _reliability_curves(repeat_frame, audit_rows)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    repeat_path = _atomic_parquet(repeat_frame, destination / "sampling-repeat-profiles.parquet")
    average_path = _atomic_parquet(average_frame, destination / "sampling-matched-profiles.parquet")
    reliability_path = _atomic_parquet(
        reliability_frame, destination / "sampling-reliability-curves.parquet"
    )
    audit_path = destination / "sampling-audit.json"
    status_counts = Counter(str(row.get("status", "unknown")) for row in audit_rows)
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "study_sha256": config_sha256(study),
            "encoding_manifest": str(Path(encoding_manifest).resolve(strict=True)),
            "encoding_manifest_sha256": sha256_file(encoding_manifest),
            "state_dictionary_sha256": sha256_file(state_dictionary_path),
            "profile_estimator_sha256": sha256_file(profile_estimator_path),
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
            "pretraining_overlap": summarize_pretraining_overlap(eligible),
            "contrasts_sha256": sha256_file(contrasts_path),
            "encoded_units_seen": len(eligible),
            "synchronized_units_available": len(units),
            "configured_contrasts": len(contrasts),
            "configured_repeats": repeat_count,
            "configured_reliability_seconds": list(study.sampling.reliability_seconds),
            "repeat_profile_rows": len(repeat_frame),
            "averaged_profile_rows": len(average_frame),
            "reliability_curve_rows": len(reliability_frame),
            "audit_status_counts": dict(sorted(status_counts.items())),
            "rows": audit_rows,
            "labels_used_only_for_post_encoding_group_selection": True,
            "state_dictionary_refit": False,
            "profile_estimator_refit": False,
            "temporal_order_preserved": True,
            "cross_segment_transitions_forbidden": True,
            "coarse_and_fine_spans_synchronized": True,
            "scientific_gate_applied": False,
        },
    )
    return repeat_path, average_path, reliability_path, audit_path


__all__ = ["run_sampling_sensitivity"]
