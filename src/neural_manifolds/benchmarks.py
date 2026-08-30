"""Conventional scalar and connectivity benchmarks on matched EEG samples."""

from __future__ import annotations

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
