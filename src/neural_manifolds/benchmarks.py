"""Conventional scalar and connectivity benchmarks on matched EEG samples."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations
from math import factorial

import numpy as np
from scipy import signal, stats

DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(frozen=True)
class WeightedSymbolicMutualInformation:
    value: float
    matrix: np.ndarray
    order: int
    lag_samples: int
    symbol_samples: int
    channel_pairs: int
    minimum_symbol_samples: int
    lowpass_hz: float

    def audit(self, *, sfreq: float) -> dict[str, object]:
        return {
            "status": "available",
            "method": "weighted_symbolic_mutual_information",
            "symbolization": "stable_ordinal_patterns",
            "order": self.order,
            "lag_samples": self.lag_samples,
            "lag_seconds": self.lag_samples / sfreq,
            "lowpass_hz": self.lowpass_hz,
            "lowpass": "fourth_order_zero_phase_butterworth",
            "alphabet_size": factorial(self.order),
            "symbol_samples": self.symbol_samples,
            "minimum_symbol_samples": self.minimum_symbol_samples,
            "channel_pairs": self.channel_pairs,
            "normalization": "natural_log_mi_divided_by_log_factorial_order",
            "excluded_pair_weights": ["identical_patterns", "sign_reversed_patterns"],
            "channel_pair_aggregation": "median_upper_triangle",
            "tie_rule": "stable_time_index_order",
        }


def _ordinal_symbol_ids(data: np.ndarray, *, order: int, lag_samples: int) -> np.ndarray:
    samples = data.shape[1] - (order - 1) * lag_samples
    if samples < 1:
        raise ValueError("symbol order/lag leaves no complete ordinal patterns")
    offsets = lag_samples * np.arange(order)
    embedded = np.stack([data[:, offset : offset + samples] for offset in offsets], axis=2)
    ordinal = np.argsort(embedded, axis=2, kind="stable")
    alphabet = np.asarray(list(permutations(range(order))), dtype=np.int64)
    symbols = np.full((data.shape[0], samples), -1, dtype=np.int64)
    for symbol_id, pattern in enumerate(alphabet):
        symbols[np.all(ordinal == pattern[None, None, :], axis=2)] = symbol_id
    if np.any(symbols < 0):
        raise RuntimeError("ordinal symbolization produced an unknown pattern")
    return symbols


def weighted_symbolic_mutual_information(
    data: np.ndarray,
    sfreq: float,
    *,
    order: int = 3,
    lag_seconds: float = 0.032,
    lowpass_hz: float = 10.0,
    minimum_symbol_samples: int | None = None,
) -> WeightedSymbolicMutualInformation:
    """Weighted symbolic mutual information across all EEG channel pairs.

    Identical ordinal patterns and their sign-reversed counterparts receive
    zero weight.  All other joint symbols receive unit weight, and each pair's
    result is normalized by the maximum entropy of the ordinal alphabet.
    """

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("wSMI data must be finite channels x samples with at least two channels")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("wSMI sampling frequency must be finite and positive")
    if order < 3 or order > 5:
        raise ValueError("wSMI ordinal order must lie between 3 and 5")
    if not np.isfinite(lag_seconds) or lag_seconds <= 0:
        raise ValueError("wSMI symbol lag must be finite and positive")
    if not np.isfinite(lowpass_hz) or lowpass_hz <= 0 or lowpass_hz >= sfreq / 2:
        raise ValueError("wSMI low-pass cutoff must lie strictly below Nyquist")
    lag_samples = max(1, round(lag_seconds * sfreq))
    alphabet = np.asarray(list(permutations(range(order))), dtype=np.int64)
    alphabet_size = len(alphabet)
    required = (
        5 * alphabet_size * alphabet_size
        if minimum_symbol_samples is None
        else int(minimum_symbol_samples)
    )
    if required < alphabet_size * alphabet_size:
        raise ValueError("wSMI minimum symbols must cover the joint alphabet")
    filter_sos = signal.butter(4, lowpass_hz, btype="lowpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(filter_sos, values, axis=1)
    symbols = _ordinal_symbol_ids(filtered, order=order, lag_samples=lag_samples)
    if symbols.shape[1] < required:
        raise ValueError(
            f"wSMI has {symbols.shape[1]} symbol samples; requires at least {required}"
        )
    unique_counts = [len(np.unique(channel)) for channel in symbols]
    if any(count < 2 for count in unique_counts):
        raise ValueError("wSMI requires at least two occupied ordinal symbols per channel")
    weights = np.ones((alphabet_size, alphabet_size), dtype=np.float64)
    for first, pattern in enumerate(alphabet):
        for second, comparison in enumerate(alphabet):
            if first == second or np.array_equal(comparison, pattern[::-1]):
                weights[first, second] = 0.0
    matrix = np.zeros((values.shape[0], values.shape[0]), dtype=np.float64)
    pair_values: list[float] = []
    normalization = np.log(alphabet_size)
    for first in range(values.shape[0]):
        for second in range(first + 1, values.shape[0]):
            joint = np.bincount(
                symbols[first] * alphabet_size + symbols[second],
                minlength=alphabet_size * alphabet_size,
            ).reshape(alphabet_size, alphabet_size)
            probability = joint / joint.sum()
            first_probability = probability.sum(axis=1, keepdims=True)
            second_probability = probability.sum(axis=0, keepdims=True)
            expected = first_probability * second_probability
            occupied = probability > 0
            value = float(
                np.sum(
                    weights[occupied]
                    * probability[occupied]
                    * np.log(probability[occupied] / expected[occupied])
                )
                / normalization
            )
            if not np.isfinite(value):
                raise RuntimeError("wSMI channel-pair estimate is non-finite")
            matrix[first, second] = matrix[second, first] = value
            pair_values.append(value)
    return WeightedSymbolicMutualInformation(
        value=float(np.median(pair_values)),
        matrix=matrix,
        order=order,
        lag_samples=lag_samples,
        symbol_samples=symbols.shape[1],
        channel_pairs=len(pair_values),
        minimum_symbol_samples=required,
        lowpass_hz=float(lowpass_hz),
    )


def microstate_peak_maps(
    data: np.ndarray,
    sfreq: float,
    *,
    minimum_peak_distance_seconds: float = 0.010,
) -> np.ndarray:
    """Return polarity-normalized EEG maps at local global-field-power peaks."""

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("microstate data must be finite channels x samples")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("microstate sampling frequency must be finite and positive")
    centred = values - np.mean(values, axis=0, keepdims=True)
    global_field_power = np.sqrt(np.mean(np.square(centred), axis=0))
    peaks, _ = signal.find_peaks(
        global_field_power,
        distance=max(1, round(minimum_peak_distance_seconds * sfreq)),
    )
    maps = centred[:, peaks].T
    norms = np.linalg.norm(maps, axis=1)
    valid = norms > np.finfo(np.float64).eps
    maps = maps[valid] / norms[valid, None]
    if len(maps) == 0:
        raise ValueError("microstate extraction found no non-flat GFP peaks")
    return maps


class FrozenMicrostateModel:
    """Deterministic polarity-invariant prototypes fitted without condition labels."""

    def __init__(self, *, n_states: int = 4, max_iterations: int = 100) -> None:
        if n_states < 2 or max_iterations < 1:
            raise ValueError("microstate state count/iterations are invalid")
        self.n_states = int(n_states)
        self.max_iterations = int(max_iterations)

    @staticmethod
    def _canonical_polarity(prototypes: np.ndarray) -> np.ndarray:
        output = prototypes.copy()
        for index, prototype in enumerate(output):
            anchor = int(np.argmax(np.abs(prototype)))
            if prototype[anchor] < 0:
                output[index] *= -1.0
        return output

    def fit(self, participant_maps: Mapping[str, np.ndarray]) -> FrozenMicrostateModel:
        if getattr(self, "fitted_", False):
            raise RuntimeError("microstate prototypes are already frozen")
        normalized_input = {
            str(participant): np.asarray(maps, dtype=np.float64)
            for participant, maps in participant_maps.items()
        }
        if len(normalized_input) != len(participant_maps):
            raise ValueError("microstate participant identifiers are not unique as strings")
        participants = sorted(normalized_input)
        if len(participants) < 3:
            raise ValueError("microstate fitting requires at least three discovery participants")
        balanced: list[np.ndarray] = []
        dimensions: set[int] = set()
        for participant in participants:
            maps = normalized_input[participant]
            if maps.ndim != 2 or not np.all(np.isfinite(maps)):
                raise ValueError("microstate participant maps must be finite maps x channels")
            dimensions.add(maps.shape[1])
            if len(maps) < self.n_states:
                raise ValueError("a discovery participant has too few microstate maps")
            norms = np.linalg.norm(maps, axis=1)
            if np.any(norms <= np.finfo(np.float64).eps):
                raise ValueError("microstate participant maps contain a flat topography")
            maps = maps / norms[:, None]
            take = min(500, len(maps))
            indices = np.linspace(0, len(maps) - 1, take, dtype=np.int64)
            balanced.append(maps[indices])
        if len(dimensions) != 1:
            raise ValueError("microstate discovery maps use inconsistent channel dimensions")
        maps = np.concatenate(balanced, axis=0)
        minimum = 25 * self.n_states
        if len(maps) < minimum:
            raise ValueError(f"microstate fitting requires at least {minimum} balanced GFP maps")
        prototypes = [maps[0]]
        while len(prototypes) < self.n_states:
            similarity = np.max(np.abs(maps @ np.stack(prototypes).T), axis=1)
            prototypes.append(maps[int(np.argmin(similarity))])
        current = np.stack(prototypes)
        for _ in range(self.max_iterations):
            correlation = maps @ current.T
            labels = np.argmax(np.abs(correlation), axis=1)
            updated: list[np.ndarray] = []
            for state in range(self.n_states):
                assigned = maps[labels == state]
                if len(assigned) == 0:
                    raise ValueError("microstate fitting produced an empty state")
                signs = np.sign(assigned @ current[state])
                signs[signs == 0] = 1.0
                prototype = np.mean(assigned * signs[:, None], axis=0)
                norm = np.linalg.norm(prototype)
                if norm <= np.finfo(np.float64).eps:
                    raise ValueError("microstate prototype update collapsed to zero")
                updated.append(prototype / norm)
            candidate = np.stack(updated)
            change = 1.0 - np.min(np.abs(np.sum(current * candidate, axis=1)))
            current = candidate
            if change < 1e-8:
                break
        self.prototypes_ = self._canonical_polarity(current)
        self.n_channels_in_ = self.prototypes_.shape[1]
        self.n_discovery_participants_ = len(participants)
        self.discovery_participant_set_sha256_ = hashlib.sha256(
            json.dumps(participants, ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.prototype_sha256_ = hashlib.sha256(self.prototypes_.tobytes()).hexdigest()
        self.fitted_ = True
        return self

    def score(self, data: np.ndarray, sfreq: float) -> dict[str, float]:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("microstate prototypes are not fitted")
        values = np.asarray(data, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[0] != self.n_channels_in_
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("microstate scoring data do not match the frozen channel dimension")
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("microstate sampling frequency must be finite and positive")
        centred = values - np.mean(values, axis=0, keepdims=True)
        norms = np.linalg.norm(centred, axis=0)
        valid = norms > np.finfo(np.float64).eps
        if np.count_nonzero(valid) < 10:
            raise ValueError("microstate scoring has fewer than ten non-flat samples")
        maps = (centred[:, valid] / norms[valid]).T
        correlation = maps @ self.prototypes_.T
        labels = np.argmax(np.abs(correlation), axis=1)
        maximum = np.max(np.abs(correlation), axis=1)
        weights = np.square(norms[valid])
        explained = float(np.sum(weights * np.square(maximum)) / np.sum(weights))
        run_starts = np.r_[True, labels[1:] != labels[:-1]]
        starts = np.flatnonzero(run_starts)
        stops = np.r_[starts[1:], len(labels)]
        median_duration = float(np.median(stops - starts) / sfreq)
        transitions = labels[1:] * self.n_states + labels[:-1]
        _, counts = np.unique(transitions, return_counts=True)
        probabilities = counts / counts.sum()
        transition_entropy = float(
            -np.sum(probabilities * np.log(probabilities)) / np.log(self.n_states**2)
        )
        result = {
            "microstate_transition_entropy": transition_entropy,
            "microstate_global_explained_variance": explained,
            "microstate_median_duration_seconds": median_duration,
        }
        if not np.all(np.isfinite(list(result.values()))):
            raise RuntimeError("microstate frozen-model score is non-finite")
        return result

    def audit(self) -> dict[str, object]:
        if not getattr(self, "fitted_", False):
            return {"status": "unavailable", "reason": "prototypes_not_fitted"}
        return {
            "status": "frozen",
            "method": "polarity_invariant_spherical_microstates",
            "n_states": self.n_states,
            "n_channels": self.n_channels_in_,
            "discovery_participants": self.n_discovery_participants_,
            "discovery_participant_set_sha256": self.discovery_participant_set_sha256_,
            "prototype_sha256": self.prototype_sha256_,
            "fit_label_fields": [],
            "fit_partition": "representation_discovery",
            "participant_balancing": "maximum_500_evenly_spaced_gfp_peak_maps_per_participant",
            "refit_after_discovery": False,
        }


def relative_band_power(
    data: np.ndarray,
    sfreq: float,
    *,
    bands: dict[str, tuple[float, float]] = DEFAULT_BANDS,
) -> dict[str, float]:
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("data must be finite channels x samples")
    frequencies, psd = signal.welch(
        x, fs=sfreq, nperseg=min(x.shape[1], max(256, int(4 * sfreq))), axis=1
    )
    full = (frequencies >= min(low for low, _ in bands.values())) & (
        frequencies <= max(high for _, high in bands.values())
    )
    denominator = np.trapezoid(psd[:, full], frequencies[full], axis=1)
    output: dict[str, float] = {}
    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)
        power = np.trapezoid(psd[:, mask], frequencies[mask], axis=1)
        output[name] = float(np.mean(power / np.maximum(denominator, np.finfo(float).tiny)))
    return output


def spectral_exponent(
    data: np.ndarray,
    sfreq: float,
    *,
    frequency_range: tuple[float, float] = (2.0, 45.0),
    exclude: tuple[tuple[float, float], ...] = ((7.0, 14.0),),
) -> float:
    """Robust log-log aperiodic slope, excluding the alpha peak by default."""

    x = np.asarray(data, dtype=float)
    frequencies, psd = signal.welch(
        x, fs=sfreq, nperseg=min(x.shape[1], max(256, int(4 * sfreq))), axis=1
    )
    spectrum = np.median(psd, axis=0)
    mask = (frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1])
    for low, high in exclude:
        mask &= ~((frequencies >= low) & (frequencies <= high))
    if np.count_nonzero(mask) < 10:
        raise ValueError("too few frequencies for spectral exponent")
    slope, _, _, _ = stats.theilslopes(
        np.log10(np.maximum(spectrum[mask], np.finfo(float).tiny)),
        np.log10(frequencies[mask]),
    )
    return float(-slope)


def permutation_entropy(
    series: np.ndarray,
    *,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    x = np.asarray(series, dtype=float).ravel()
    if order < 2 or delay < 1 or x.size <= (order - 1) * delay:
        raise ValueError("series/order/delay combination is invalid")
    patterns = np.asarray(
        [
            np.argsort(x[start + delay * np.arange(order)], kind="stable")
            for start in range(x.size - (order - 1) * delay)
        ]
    )
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    if normalize:
        entropy /= np.log(factorial(order))
    return entropy


def _lz76_complexity(bits: np.ndarray) -> int:
    sequence = "".join("1" if bit else "0" for bit in bits)
    n = len(sequence)
    if n == 0:
        return 0
    i, k, cursor, complexity = 0, 1, 1, 1
    while True:
        if sequence[i + k - 1] == sequence[cursor + k - 1]:
            k += 1
            if cursor + k > n:
                complexity += 1
                break
        else:
            if k > 1:
                i += 1
                if i == cursor:
                    complexity += 1
                    cursor += k
                    if cursor + 1 > n:
                        break
                    i, k = 0, 1
                else:
                    k = 1
            else:
                i += 1
                if i == cursor:
                    complexity += 1
                    cursor += 1
                    if cursor + 1 > n:
                        break
                    i = 0
    return complexity


def normalized_lempel_ziv(data: np.ndarray) -> float:
    """Median-binarised, time-concatenated LZ76 complexity."""

    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] < 10:
        raise ValueError("data must be channels x samples")
    standardized = (x - np.median(x, axis=1, keepdims=True)) > 0
    bits = standardized.T.reshape(-1)
    complexity = _lz76_complexity(bits)
    n = bits.size
    return float(complexity * np.log2(n) / n)


def weighted_phase_lag_index(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Debiased-like weighted phase-lag matrix from Fourier cross spectra."""

    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("data must be channels x samples")
    frequencies, _, spectra = signal.stft(
        x,
        fs=sfreq,
        nperseg=min(x.shape[1], max(128, int(2 * sfreq))),
        noverlap=min(x.shape[1] // 2, int(sfreq)),
        axis=-1,
    )
    mask = (frequencies >= 1.0) & (frequencies <= min(45.0, sfreq / 2 - 1))
    spectra = spectra[:, mask]
    channels = x.shape[0]
    result = np.eye(channels)
    for first in range(channels):
        for second in range(first + 1, channels):
            imaginary = np.imag(spectra[first] * np.conjugate(spectra[second])).ravel()
            numerator = abs(np.mean(imaginary))
            denominator = np.mean(np.abs(imaginary))
            value = numerator / denominator if denominator > 0 else 0.0
            result[first, second] = result[second, first] = value
    return result
