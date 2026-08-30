import numpy as np

from neural_manifolds.benchmarks import (
    normalized_lempel_ziv,
    permutation_entropy,
    relative_band_power,
    spectral_exponent,
    weighted_phase_lag_index,
)


def test_benchmarks_are_finite() -> None:
    rng = np.random.default_rng(18)
    times = np.arange(2000) / 200
    data = np.stack([np.sin(2 * np.pi * 10 * times), np.sin(2 * np.pi * 10 * times + 0.3)])
    data += rng.normal(scale=0.1, size=data.shape)
    powers = relative_band_power(data, 200)
    assert powers["alpha"] > powers["delta"]
    assert np.isfinite(spectral_exponent(data, 200))
    assert 0 <= permutation_entropy(data[0]) <= 1
    assert normalized_lempel_ziv(data) > 0
    assert weighted_phase_lag_index(data, 200).shape == (2, 2)
