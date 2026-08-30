import numpy as np

from neural_manifolds.manifold import (
    communication_subspace_alignment,
    fit_communication_subspace,
    principal_angles,
)


def test_recovers_imposed_lag_and_communication_subspace() -> None:
    rng = np.random.default_rng(21)
    n_samples = 1_200
    source = rng.normal(size=(n_samples, 6))
    weights = np.array([[1.2, -0.4, 0.8, 0.0, 0.2], [-0.3, 1.1, 0.0, 0.7, -0.5]])
    target = rng.normal(scale=0.08, size=(n_samples, 5))
    target[2:] += source[:-2, :2] @ weights

    summary = communication_subspace_alignment(
        source, target, lags=(1, 2, 3), rank=2, ridge=1e-5, cv=5
    )
    model = fit_communication_subspace(source, target, lag=2, rank=2, ridge=1e-5)
    true_source_basis = np.eye(6)[:, :2]
    angles = principal_angles(model.source_basis, true_source_basis, degrees=True)

    assert summary.best_lag == 2
    assert summary.best_lag_shared_predictive_variance > 0.97
    assert summary.shared_predictive_variance > 0.25
    assert np.max(angles) < 3.0


def test_independent_modules_have_no_held_out_alignment() -> None:
    rng = np.random.default_rng(22)
    source = rng.normal(size=(800, 5))
    target = rng.normal(size=(800, 4))

    summary = communication_subspace_alignment(
        source, target, lags=(1, 2), rank=2, ridge=1e-4, cv=5
    )

    assert summary.best_lag_shared_predictive_variance < 0.03


def test_segment_boundaries_are_excluded_from_lagged_pairs() -> None:
    rng = np.random.default_rng(23)
    source = rng.normal(size=(100, 3))
    target = np.roll(source, 1, axis=0)
    segments = np.repeat(np.arange(10), 10)

    model = fit_communication_subspace(
        source,
        target,
        lag=1,
        rank=3,
        segment_ids=segments,
    )

    assert model.n_pairs == 90
