import numpy as np

from neural_manifolds.manifold import (
    FiveAxisProfileEstimator,
    LocalLinearDynamics,
    ManifoldRecord,
)


def _make_record(seed: int, alpha: float) -> ManifoldRecord:
    rng = np.random.default_rng(seed)
    n_samples, dimension = 360, 4
    trajectory = rng.normal(size=(n_samples, dimension))
    states = np.empty(n_samples, dtype=int)
    states[0] = 0
    transition = np.array([[0.80, 0.17, 0.03], [0.05, 0.80, 0.15], [0.15, 0.05, 0.80]])
    for index in range(1, n_samples):
        states[index] = rng.choice(3, p=transition[states[index - 1]])
    posterior = rng.normal(size=(n_samples, 3))
    frontal = rng.normal(scale=0.15, size=(n_samples, 3))
    frontal[1:] += posterior[:-1] @ np.diag([0.9, 0.6, 0.3])
    occupancy = np.bincount(states, minlength=3).astype(float)
    occupancy /= occupancy.sum()
    matrices = np.repeat((alpha * np.eye(dimension))[None, :, :], 3, axis=0)
    innovations = np.repeat((0.2 * np.eye(dimension))[None, :, :], 3, axis=0)
    dynamics = LocalLinearDynamics(
        state_labels=np.arange(3),
        transition_matrices=matrices,
        innovation_covariances=innovations,
        intercepts=np.zeros((3, dimension)),
        occupancy=occupancy,
        n_transitions_by_state=np.full(3, 50),
        spectral_radii=np.full(3, alpha),
        ridge=1e-4,
        innovation_regularization=1e-8,
    )
    return ManifoldRecord(
        trajectory=trajectory,
        states=states,
        regional_trajectories={"posterior": posterior, "frontal": frontal},
        local_dynamics=dynamics,
        name=f"reference-{seed}",
    )


def test_five_axis_profile_is_finite_auditable_and_deterministic() -> None:
    references = [_make_record(seed, 0.45 + 0.03 * seed) for seed in range(4)]
    estimator = FiveAxisProfileEstimator(
        alignment_lags=(1,),
        alignment_rank=2,
        alignment_cv=3,
        module_pairs=(("posterior", "frontal"),),
        alignment_bidirectional=False,
        reachability_horizon=8,
        standardization="zscore",
    )

    reference_values = estimator.fit_transform(references)
    target = _make_record(10, 0.75)
    first = estimator.profile(target)
    second = estimator.profile(target)

    assert reference_values.shape == (4, 5)
    assert np.allclose(np.mean(reference_values, axis=0), 0.0, atol=1e-10)
    assert first.values.shape == (5,)
    assert np.all(np.isfinite(first.values))
    assert np.array_equal(first.values, second.values)
    assert first.as_dict().keys() == {
        "repertoire",
        "metastability",
        "directionality",
        "alignment",
        "reachability",
    }
    assert first.details.metastability.feature_vector().shape == (4,)
    assert first.raw_values[4] > estimator.reference_raw_values_[0, 4]
