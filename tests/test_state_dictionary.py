from __future__ import annotations

import numpy as np

from neural_manifolds.dynamics.state_dictionary import fit_state_dictionary


def test_fixed_fallback_is_deterministic_and_signal_only() -> None:
    rng = np.random.default_rng(17)
    discovery = [rng.normal(loc=index, size=(80, 8)) for index in range(4)]
    validation = [rng.normal(loc=index, size=(50, 8)) for index in range(2)]
    first = fit_state_dictionary(
        discovery,
        validation,
        rank=5,
        state_counts=(3, 4, 5),
        seeds=(101, 103),
        minimum_stability_ami=0.7,
        force_fallback=True,
    )
    second = fit_state_dictionary(
        discovery,
        validation,
        rank=5,
        state_counts=(3, 4, 5),
        seeds=(101, 103),
        minimum_stability_ami=0.7,
        force_fallback=True,
    )
    assert first.method == "fixed_k_robust_kmeans"
    assert first.primary_status == "unavailable"
    np.testing.assert_array_equal(first.predict(validation[0]), second.predict(validation[0]))
    assert set(first.audit()) >= {"projection_components", "primary_error", "n_states"}


def test_state_dictionary_rejects_dimension_mismatch() -> None:
    rng = np.random.default_rng(9)
    try:
        fit_state_dictionary(
            [rng.normal(size=(20, 5))],
            [rng.normal(size=(20, 4))],
            rank=3,
            state_counts=(3,),
            seeds=(1, 2),
            minimum_stability_ami=0.7,
        )
    except ValueError as error:
        assert "dimensions differ" in str(error)
    else:
        raise AssertionError("dimension mismatch was accepted")
