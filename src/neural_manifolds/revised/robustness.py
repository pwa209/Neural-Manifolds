"""Matched physical-signal perturbations with a frozen encoder and full refits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.revised.inference import load_measurements, run_transfer
from neural_manifolds.revised.measurement import (
    conventional_features,
    sensor_trajectory,
    time_frequency_trajectory,
)


def perturb(windows, channels, kind, rng):
    if kind == "independent_phase":
        coefficients = np.fft.rfft(windows, axis=-1)
        phase = rng.uniform(-np.pi, np.pi, size=coefficients.shape)
        phase[..., 0] = 0
        if windows.shape[-1] % 2 == 0:
            phase[..., -1] = 0
        return np.fft.irfft(
            coefficients * np.exp(1j * phase), n=windows.shape[-1], axis=-1
        ), channels
    if kind == "noise_10_percent":
        sd = windows.std(axis=-1, keepdims=True)
        return windows + rng.normal(size=windows.shape) * sd * 0.1, channels
    if kind == "channel_dropout":
        keep = [i for i in range(len(channels)) if i % 3 != 2]
        return windows[:, keep, :], [channels[i] for i in keep]
    raise ValueError("unknown_signal_perturbation")


def measure_perturbations(paths, output, policy, encoder):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for row in load_measurements(paths):
        if sha256_file(row["array_path"]) != row["array_sha256"]:
            raise ValueError("perturbation_input_checksum_changed")
        with np.load(row["array_path"], allow_pickle=False) as source:
            windows = source["windows_volts"]
            channels = source["channels"].tolist()
            seconds, segments = source["seconds"], source["segments"]
        for kind in ("independent_phase", "noise_10_percent", "channel_dropout"):
            key = f"{row['unit_id']}-{row['track']}-{kind}"
            checkpoint = output / "records" / f"{key}.json"
            if checkpoint.exists():
                cached = json.loads(checkpoint.read_text())
                if (
                    cached["original_array_sha256"] != row["array_sha256"]
                    or sha256_file(cached["array_path"]) != cached["array_sha256"]
                ):
                    raise ValueError("perturbation_checkpoint_changed")
                records.append(cached)
                continue
            seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
            x, names = perturb(windows, channels, kind, np.random.default_rng(seed))
            encoded = encoder.encode(x, names, metadata_fields=())
            data = {
                "sensor": sensor_trajectory(x),
                "time_frequency": time_frequency_trajectory(x, 200),
                "encoder": encoded.global_states,
                "segments": segments,
                "seconds": seconds,
            }
            data.update(
                {
                    f"regional__encoder__{name}": value
                    for name, value in encoded.regional_states.items()
                }
            )
            path = output / "arrays" / f"{key}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream, **data)
            path.with_suffix(".tmp").replace(path)
            result = {
                **row,
                "track": f"{row['track']}_{kind}",
                "array_path": str(path),
                "array_sha256": sha256_file(path),
                "conventional": conventional_features(x, 200),
                "perturbation": kind,
                "original_array_sha256": row["array_sha256"],
                "channels": names,
                "matched_original_qc_windows": True,
                "new_QC_selection": False,
                "label_fields_seen_by_encoder": [],
            }
            atomic_write_json(checkpoint, result)
            records.append(result)
    result = {
        "records": records,
        "scope": "matched_signal_perturbations_not_new_biological_data",
        "phase_null": "independent_channel_phase_within_each_one_second_window_power_spectrum_preserved",
        "dropout": "fixed_every_third_channel_removed_no_interpolation",
        "scientific_gates": False,
    }
    atomic_write_json(output / "perturbation_measurements.json", result)
    return result


def run_robustness(path: Path, output: Path, policy):
    result = run_transfer([path], output, policy)
    result["scope"] = "signal_perturbation_sensitivity_full_nested_refits_on_matched_observations"
    atomic_write_json(output / "robustness.json", result)
    return result
