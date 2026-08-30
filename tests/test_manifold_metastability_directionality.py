import numpy as np
import pytest

from neural_manifolds.manifold import (
    MetastabilityReference,
    entropy_production,
    estimate_directionality,
    estimate_metastability,
    estimate_stationary_distribution,
)


def _simulate_markov(transition: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = np.empty(n_samples, dtype=int)
    states[0] = 0
    for index in range(1, n_samples):
        states[index] = rng.choice(transition.shape[0], p=transition[states[index - 1]])
    return states


def test_metastability_recovers_known_dwell_recurrence_and_exit_entropy() -> None:
    states = np.array([0, 0, 1, 1, 0, 2, 2, 0])

    summary = estimate_metastability(states, sample_interval=0.5)

    assert summary.median_dwell == pytest.approx(1.0)
    assert summary.n_runs == 5
    assert summary.recurrence_probability == pytest.approx(0.5)
    assert summary.exit_entropy == pytest.approx(0.5)
    assert summary.transition_counts.tolist() == [[0, 1, 1], [1, 0, 0], [1, 0, 0]]


def test_segment_boundaries_do_not_create_dwell_or_transition_artifacts() -> None:
    states = np.array([0, 0, 0, 0])
    segments = np.array([0, 0, 1, 1])

    summary = estimate_metastability(states, segment_ids=segments)
    directionality = estimate_directionality(states, segment_ids=segments, pseudocount=0.0)

    assert summary.n_runs == 2
    assert summary.median_dwell == 2.0
    assert directionality.n_transitions == 2


def test_longer_runs_are_detected_without_calling_them_more_conscious() -> None:
    short = np.tile([0, 1, 2], 30)
    long = np.repeat(np.tile([0, 1, 2], 10), 3)

    short_summary = estimate_metastability(short)
    long_summary = estimate_metastability(long)

    assert short_summary.median_dwell == 1.0
    assert long_summary.median_dwell == 3.0
    assert long_summary.switching_rate < short_summary.switching_rate


def test_reference_score_is_optimal_near_healthy_persistence_revisability() -> None:
    reference_summaries = [
        estimate_metastability(np.repeat(np.tile([0, 1, 2], 20), dwell)) for dwell in (2, 3, 3, 4)
    ]
    reference = MetastabilityReference(covariance_shrinkage=0.2).fit(reference_summaries)
    near = estimate_metastability(np.repeat(np.tile([0, 1, 2], 20), 3))
    far = estimate_metastability(np.repeat(np.tile([0, 1, 2], 5), 12))

    assert reference.score(near) > reference.score(far)
    assert reference.score(near) <= 0.0


def test_entropy_production_separates_reversible_and_irreversible_chains() -> None:
    reversible = np.array([[0.10, 0.45, 0.45], [0.45, 0.10, 0.45], [0.45, 0.45, 0.10]])
    irreversible = np.array([[0.05, 0.90, 0.05], [0.05, 0.05, 0.90], [0.90, 0.05, 0.05]])

    pi_reversible = estimate_stationary_distribution(reversible)
    pi_irreversible = estimate_stationary_distribution(irreversible)

    assert entropy_production(reversible, pi_reversible) == pytest.approx(0.0, abs=1e-12)
    assert entropy_production(irreversible, pi_irreversible) > 2.0


def test_empirical_entropy_production_recovers_directed_cycle() -> None:
    reversible = np.array([[0.10, 0.45, 0.45], [0.45, 0.10, 0.45], [0.45, 0.45, 0.10]])
    irreversible = np.array([[0.05, 0.90, 0.05], [0.05, 0.05, 0.90], [0.90, 0.05, 0.05]])
    reversible_states = _simulate_markov(reversible, 20_000, seed=2)
    irreversible_states = _simulate_markov(irreversible, 20_000, seed=3)

    reversible_summary = estimate_directionality(reversible_states, pseudocount=0.5)
    irreversible_summary = estimate_directionality(irreversible_states, pseudocount=0.5)

    assert reversible_summary.entropy_production < 0.01
    assert irreversible_summary.entropy_production > 1.5
    assert irreversible_summary.flux_asymmetry > reversible_summary.flux_asymmetry


def test_one_way_structural_flux_has_infinite_entropy_production() -> None:
    one_way_cycle = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert np.isinf(entropy_production(one_way_cycle))
