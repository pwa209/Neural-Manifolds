"""Context-specific public PsiConnect FieldTrip boundary analysis.

This is an amplitude-normalized sensor analysis, not an unverified voltage input
to LaBraM and not a simultaneous EEG/fMRI validation.
"""

from __future__ import annotations

import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat

from neural_manifolds.preprocessing.eeg import resample_array
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.revised.cohort import normalized_channel
from neural_manifolds.revised.inference import DYNAMICS, FoldDynamics
from neural_manifolds.revised.measurement import conventional_features, sensor_trajectory

FILE = re.compile(r"sub-(PC\d+)_ses-(0[12])_task-(meditation|movie|music|rest)_Clean-ft\.mat$")


def fieldtrip(path):
    try:
        document = loadmat(path, simplify_cells=True)
        candidates = [
            v
            for v in document.values()
            if isinstance(v, dict) and {"trial", "time", "label", "fsample"} <= set(v)
        ]
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as handle:

            def decode(node):
                if isinstance(node, h5py.Group):
                    return {
                        k: decode(node[k])
                        for k in ("trial", "time", "label", "fsample")
                        if k in node
                    }
                values = node[()]
                if h5py.check_dtype(ref=node.dtype):
                    return [decode(handle[r]) for r in values.ravel()]
                if node.attrs.get("MATLAB_class") == b"char":
                    return "".join(chr(int(x)) for x in values.ravel())
                return values.T.squeeze()

            candidates = [
                decode(v)
                for v in handle.values()
                if isinstance(v, h5py.Group) and {"trial", "time", "label", "fsample"} <= set(v)
            ]
    if len(candidates) != 1:
        raise ValueError("FieldTrip_structure_not_unique")
    data = candidates[0]
    labels = [str(x) for x in np.asarray(data["label"]).ravel()]
    sfreq = float(np.asarray(data["fsample"]).squeeze())

    def blocks(value):
        if isinstance(value, np.ndarray) and value.dtype != object:
            return [value]
        return [np.asarray(x) for x in value]

    trials, times = blocks(data["trial"]), blocks(data["time"])
    if len(trials) != len(times) or sfreq <= 0 or len(labels) != len(set(labels)):
        raise ValueError("FieldTrip_axes_or_frequency_invalid")
    for x, t in zip(trials, times, strict=True):
        if x.ndim != 2 or x.shape != (len(labels), np.asarray(t).size):
            raise ValueError("FieldTrip_trial_channel_time_mismatch")
    return labels, sfreq, trials, times


def context_measure(path, policy):
    labels, sfreq, trials, times = fieldtrip(path)
    normalized = [normalized_channel(x) for x in labels]
    requested = policy["high_density_channels"]
    if any(normalized.count(c.upper()) != 1 for c in requested):
        raise ValueError("boundary_observed_montage_unavailable")
    picks = [normalized.index(c.upper()) for c in requested]
    windows, segments = [], []
    segment = 0
    for x, t in zip(trials, times, strict=True):
        x = np.asarray(x, float)[picks]
        t = np.asarray(t, float).ravel()
        finite = np.isfinite(x).all(axis=0)
        # Respect explicit FieldTrip time discontinuities before filtering.
        starts = np.r_[
            0,
            np.flatnonzero((np.abs(np.diff(t) - 1 / sfreq) > 1e-6) | (~finite[1:]) | (~finite[:-1]))
            + 1,
            len(t),
        ]
        for a, b in pairwise(starts):
            if b - a < sfreq * 2 or not finite[a:b].all():
                continue
            values = x[:, a:b] - x[:, a:b].mean(axis=0, keepdims=True)
            sos = signal.butter(4, [0.5, 30], btype="bandpass", fs=sfreq, output="sos")
            values = resample_array(signal.sosfiltfilt(sos, values, axis=1), sfreq, 200)
            count = values.shape[1] // 200
            block = values[:, : count * 200].reshape(len(picks), count, 200).transpose(1, 0, 2)
            for w in block:
                if len(windows) >= 120:
                    break
                rms = np.sqrt(np.mean(w * w))
                if (
                    not np.isfinite(w).all()
                    or rms <= 0
                    or np.max(np.abs(w)) / rms > 30
                    or np.any(w.std(axis=1) == 0)
                ):
                    segment += 1
                    continue
                windows.append(w)
                segments.append(segment)
            segment += 1
    if len(windows) < 10:
        raise ValueError("boundary_insufficient_clean_seconds")
    windows = np.stack(windows)
    # Explicitly eliminate unverified amplitude units; never imply absolute power.
    scale = float(np.sqrt(np.mean(windows**2)))
    windows /= scale
    trajectory = sensor_trajectory(windows)
    return conventional_features(windows, 200), {
        "trajectory": trajectory,
        "segments": np.array(segments),
        "regions": {str(i): trajectory[:, i : i + 1] for i in range(trajectory.shape[1])},
    }


def paired_context_summary(frame, policy):
    rng = np.random.default_rng(policy["seed"])
    rows = []
    deltas = {}
    for (context, feature), block in frame.groupby(["context", "feature"]):
        wide = block.pivot(index="participant_id", columns="session", values="value")
        if not {"01", "02"} <= set(wide):
            continue
        delta = (wide["02"] - wide["01"]).dropna()
        deltas[(context, feature)] = delta
        if len(delta) < 2:
            continue
        draws = [
            rng.choice(delta, len(delta), replace=True).mean()
            for _ in range(policy["bootstrap_repetitions"])
        ]
        rows.append(
            {
                "context": context,
                "feature": feature,
                "participants": len(delta),
                "post_minus_baseline": float(delta.mean()),
                "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
            }
        )
    interactions = []
    for (context, feature), delta in deltas.items():
        if context == "rest" or ("rest", feature) not in deltas:
            continue
        change = (delta - deltas[("rest", feature)]).dropna()
        if len(change) < 2:
            continue
        draws = [
            rng.choice(change, len(change), replace=True).mean()
            for _ in range(policy["bootstrap_repetitions"])
        ]
        interactions.append(
            {
                "context_minus_rest": context,
                "feature": feature,
                "participants": len(change),
                "difference_in_change": float(change.mean()),
                "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
            }
        )
    return {"paired_changes": rows, "context_interactions": interactions}


def run_boundary(marker: Path, output: Path, policy: dict):
    subset = json.loads(marker.read_text())
    release = Path(subset["release"])
    rows, arrays, unavailable = [], {}, []
    for item in subset["files"]:
        match = FILE.fullmatch(Path(item["path"]).name)
        if not match:
            continue
        path = release / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("boundary_source_checksum_changed")
        person, session, context = match.groups()
        unit = hashlib.sha256(item["path"].encode()).hexdigest()[:24]
        try:
            conventional, trajectory = context_measure(path, policy)
            rows.append(
                dict(
                    unit_id=unit,
                    participant_id=person,
                    study_group="psiconnect",
                    session=session,
                    context=context,
                    conventional=conventional,
                )
            )
            arrays[unit] = trajectory
        except (ValueError, OSError, RuntimeError) as exc:
            unavailable.append({"unit_id": unit, "reason": str(exc)})
    frame = pd.DataFrame(rows)
    measured = []
    if not frame.empty:
        for held in sorted(frame.participant_id.unique()):
            train = frame[(frame.participant_id != held) & (frame.session == "01")]
            test = frame[frame.participant_id == held]
            try:
                transform = FoldDynamics(seed=policy["seed"]).fit(train, arrays)
                fit, _ = transform.transform(train, arrays)
                values, _ = transform.transform(test, arrays)
                center, scale = fit.mean(), fit.std(ddof=1).replace(0, np.nan)
                standardized = (values - center) / scale
                for index, row in test.iterrows():
                    for feature in DYNAMICS:
                        value = standardized.loc[index, feature]
                        if np.isfinite(value):
                            measured.append(
                                {
                                    **row.drop("conventional").to_dict(),
                                    "feature": feature,
                                    "value": float(value),
                                }
                            )
                    measured.append(
                        {
                            **row.drop("conventional").to_dict(),
                            "feature": "lz",
                            "value": row.conventional["lz"],
                        }
                    )
            except ValueError as exc:
                unavailable.append({"participant_id": held, "reason": str(exc)})
    summary = paired_context_summary(pd.DataFrame(measured), policy) if measured else {}
    result = {
        "measurements": measured,
        **summary,
        "unavailable": unavailable,
        "scientific_gates": False,
        "source": "https://doi.org/10.1038/s41597-026-07312-1",
        "normalization": "baseline_training_participants_only_axis_standardization",
        "limitations": [
            "sensor_only_amplitude_normalized_due_to_unverified_MAT_units",
            "public_cleaned_data_include_upstream_interpolation_and_zero_phase_filtering",
            "no_placebo_drug_session_and_order_confounded",
            "subjective_scale_mapping_not_yet_adjudicated_no_invented_join",
            "encoder_and_fMRI_not_in_this_boundary_test",
        ],
    }
    atomic_write_json(output / "boundary.json", result)
    return result
