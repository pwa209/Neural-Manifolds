import importlib.util

import numpy as np
import pytest

from neural_manifolds.dynamics.hmm import fit_stable_gaussian_hmm


@pytest.mark.skipif(importlib.util.find_spec("hmmlearn") is None, reason="optional hmmlearn")
def test_hmm_recovers_two_state_sequences() -> None:
    rng = np.random.default_rng(12)
    sequences = []
    for _ in range(12):
        state = np.repeat([0, 1, 0, 1], 30)
        sequence = rng.normal(scale=0.2, size=(state.size, 2)) + state[:, None] * 3
        sequences.append(sequence)
    selection = fit_stable_gaussian_hmm(
        sequences[:8],
        sequences[8:],
        state_counts=[2, 3],
        seeds=[1, 2, 3],
        minimum_stability_ami=0.5,
    )
    assert selection.n_states == 2
