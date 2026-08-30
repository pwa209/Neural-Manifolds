from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_manifolds.statistics.mixed import fit_participant_mixed_model


def test_random_intercept_mixed_model_recovers_repeated_observation_effect() -> None:
    rng = np.random.default_rng(31)
    rows = []
    for participant in range(20):
        intercept = rng.normal(scale=0.8)
        for target in (0.0, 1.0, 2.0):
            rows.append(
                {
                    "participant_id": f"sub-{participant:02d}",
                    "target": target,
                    "axis": intercept + 0.6 * target + rng.normal(scale=0.08),
                }
            )

    result = fit_participant_mixed_model(
        pd.DataFrame(rows),
        outcome="axis",
        fixed_effects=["target"],
    )

    assert result.converged
    assert result.parameters["target"] == pytest.approx(0.6, abs=0.05)
    assert result.n_participants == 20
    assert result.n_observations == 60
    assert result.random_intercept_variance > 0
    assert result.confidence_intervals["target"][0] < result.parameters["target"]
    assert result.confidence_intervals["target"][1] > result.parameters["target"]


def test_mixed_model_rejects_nonfinite_repeated_outcome() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["a", "a", "b", "b", "c", "c"],
            "target": [0, 1, 0, 1, 0, 1],
            "axis": [0.0, 1.0, 0.0, np.nan, 0.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="outcome contains non-finite"):
        fit_participant_mixed_model(frame, outcome="axis", fixed_effects=["target"])
