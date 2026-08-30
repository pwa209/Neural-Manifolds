from collections import Counter

import numpy as np
import pytest

from neural_manifolds.manifold import (
    block_permutation,
    covariance_matched_surrogate,
    dwell_matched_state_surrogate,
    fit_local_linear_dynamics,
    iaaft_surrogate,
    matched_sample_indices,
    phase_randomized_surrogate,
    reachability_energy,
    sample_matched_metric,
    segment_ids_from_indices,
    stochastic_reachability_gramian,
    summarize_reachability,
)


def _run_multiset(states: np.ndarray) -> Counter[tuple[int, int]]:
    starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    stops = np.r_[starts[1:], states.size]
    return Counter(
        (int(states[start]), int(stop - start)) for start, stop in zip(starts, stops, strict=True)
    )


def test_reachability_matches_scalar_closed_form_and_orders_memory() -> None:
    dimension, horizon = 3, 20
    innovation = np.eye(dimension)
    slow = 0.8 * np.eye(dimension)
    fast = 0.2 * np.eye(dimension)

    gramian = stochastic_reachability_gramian(slow, innovation, horizon=horizon)
    expected = (1.0 - 0.8 ** (2 * horizon)) / (1.0 - 0.8**2)
    slow_summary = summarize_reachability(slow, innovation, horizon=horizon)
    fast_summary = summarize_reachability(fast, innovation, horizon=horizon)

    assert np.allclose(gramian, expected * np.eye(dimension), atol=1e-12)
    assert slow_summary.log_determinant > fast_summary.log_determinant
    assert slow_summary.effective_rank == pytest.approx(dimension)


def test_unreachable_target_has_infinite_energy() -> None:
    gramian = np.diag([2.0, 0.0])
    energy = reachability_energy(gramian, [[1.0, 0.0], [0.0, 1.0]])
    assert energy[0] == pytest.approx(0.5)
    assert np.isinf(energy[1])


def test_local_linear_fit_recovers_known_ar_dynamics() -> None:
    rng = np.random.default_rng(31)
    transition = np.array([[0.75, 0.1], [-0.05, 0.55]])
    innovation = np.array([[0.08, 0.02], [0.02, 0.05]])
    trajectory = np.zeros((5_000, 2))
    for index in range(trajectory.shape[0] - 1):
        trajectory[index + 1] = transition @ trajectory[index] + rng.multivariate_normal(
            np.zeros(2), innovation
        )

    fit = fit_local_linear_dynamics(trajectory, ridge=1e-7)

    assert np.allclose(fit.transition_matrices[0], transition, atol=0.035)
    assert np.allclose(fit.innovation_covariances[0], innovation, atol=0.008)


def test_sample_matching_is_equal_ordered_and_deterministic() -> None:
    sizes = {"wake": 15, "sedation": 9}
    first = matched_sample_indices(sizes, n_repeats=4, random_state=42)
    second = matched_sample_indices(sizes, n_repeats=4, random_state=42)

    for left, right in zip(first, second, strict=True):
        for group in sizes:
            assert len(left[group]) == 9
            assert np.all(np.diff(left[group]) >= 0)
            assert np.array_equal(left[group], right[group])

    result = sample_matched_metric(
        {"wake": np.arange(15.0), "sedation": np.arange(9.0)},
        np.mean,
        n_repeats=10,
        random_state=1,
    )
    assert result.n_samples == 9
    assert result.estimates["wake"].shape == (10,)
    assert result.sampling_variances["wake"] > 0.0
    assert segment_ids_from_indices([1, 2, 5, 6, 9]).tolist() == [0, 0, 1, 1, 2]


def test_phase_and_iaaft_surrogates_preserve_their_locked_marginals() -> None:
    rng = np.random.default_rng(32)
    signal = rng.standard_t(df=4, size=(512, 2))

    phase = phase_randomized_surrogate(signal, random_state=2)
    iaaft = iaaft_surrogate(signal[:, 0], n_iterations=300, random_state=2)

    assert np.allclose(
        np.abs(np.fft.rfft(phase, axis=0)),
        np.abs(np.fft.rfft(signal, axis=0)),
        rtol=1e-11,
        atol=1e-11,
    )
    assert np.array_equal(np.sort(iaaft), np.sort(signal[:, 0]))
    relative_spectral_error = np.linalg.norm(
        np.abs(np.fft.rfft(iaaft)) - np.abs(np.fft.rfft(signal[:, 0]))
    ) / np.linalg.norm(np.abs(np.fft.rfft(signal[:, 0])))
    assert relative_spectral_error < 0.08


def test_covariance_and_dwell_matched_nulls_preserve_exact_targets() -> None:
    rng = np.random.default_rng(33)
    trajectory = rng.normal(size=(300, 4)) @ np.array(
        [[1.0, 0.2, 0.0], [0.0, 0.8, 0.4], [0.3, 0.0, 0.7], [0.2, 0.1, 0.0]]
    )
    covariance_null = covariance_matched_surrogate(trajectory, random_state=4)
    states = np.concatenate(
        [
            np.full(length, state)
            for state, length in [(0, 2), (1, 3), (2, 1), (0, 4), (2, 2), (1, 1)]
        ]
    )
    dwell_null = dwell_matched_state_surrogate(states, random_state=5)

    assert np.allclose(np.mean(covariance_null, axis=0), np.mean(trajectory, axis=0))
    assert np.allclose(
        np.cov(covariance_null, rowvar=False),
        np.cov(trajectory, rowvar=False),
        atol=1e-12,
    )
    assert _run_multiset(dwell_null) == _run_multiset(states)


def test_block_permutation_preserves_samples() -> None:
    signal = np.arange(40.0).reshape(20, 2)
    surrogate = block_permutation(signal, block_size=4, random_state=9)
    assert sorted(map(tuple, surrogate)) == sorted(map(tuple, signal))
