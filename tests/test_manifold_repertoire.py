import numpy as np
import pytest

from neural_manifolds.manifold import (
    RepertoireEstimator,
    estimate_repertoire,
    knn_intrinsic_dimension,
    participation_ratio,
)


def test_participation_ratio_recovers_isotropic_latent_rank() -> None:
    rng = np.random.default_rng(7)
    n_samples, ambient_dimension, intrinsic_dimension = 6_000, 12, 4
    basis, _ = np.linalg.qr(rng.normal(size=(ambient_dimension, intrinsic_dimension)))
    latent = rng.normal(size=(n_samples, intrinsic_dimension))
    trajectory = latent @ basis.T

    summary = estimate_repertoire(trajectory, shrinkage="none")

    assert summary.participation_ratio == pytest.approx(intrinsic_dimension, abs=0.12)
    assert summary.effective_rank == pytest.approx(intrinsic_dimension, abs=0.12)
    assert np.count_nonzero(summary.eigenvalues > 1e-10) == intrinsic_dimension


def test_participation_ratio_is_not_raw_variance() -> None:
    spectrum = np.array([10.0, 10.0, 0.0, 0.0])
    assert participation_ratio(spectrum) == pytest.approx(2.0)
    assert participation_ratio(100.0 * spectrum) == pytest.approx(2.0)
    assert participation_ratio([10.0, 1.0, 0.0]) < 1.25


def test_knn_dimension_recovers_gaussian_dimension() -> None:
    rng = np.random.default_rng(11)
    intrinsic = rng.normal(size=(1_200, 3))
    embedding = rng.normal(size=(3, 8))
    trajectory = intrinsic @ embedding

    estimate = knn_intrinsic_dimension(trajectory, k=20)

    assert 2.3 < estimate < 3.8


def test_repertoire_estimator_obeys_parameter_and_transform_protocol() -> None:
    rng = np.random.default_rng(13)
    trajectory = rng.normal(size=(100, 5))
    estimator = RepertoireEstimator(shrinkage="oas", noise_variance=0.0)

    transformed = estimator.fit_transform(trajectory)

    assert transformed.shape == trajectory.shape
    assert estimator.get_params(deep=False) == {
        "noise_variance": 0.0,
        "shrinkage": "oas",
    }
    assert np.allclose(np.mean(transformed, axis=0), 0.0, atol=1e-12)
    assert 1.0 <= estimator.score() <= trajectory.shape[1]


def test_repertoire_rejects_nonfinite_values_and_negative_spectrum() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        estimate_repertoire([[0.0, 1.0], [np.nan, 2.0]])
    with pytest.raises(ValueError, match="non-negative"):
        participation_ratio([1.0, -0.1])
