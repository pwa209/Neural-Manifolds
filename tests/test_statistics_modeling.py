from __future__ import annotations

import numpy as np

from neural_manifolds.statistics.equivalence import participant_bootstrap_tost_interval
from neural_manifolds.statistics.prediction import (
    calibration_slope_intercept,
    participant_bootstrap_prediction_metrics,
)


def test_tost_equivalence_uses_configured_interval_not_nonsignificance() -> None:
    rng = np.random.default_rng(17)
    near_zero = participant_bootstrap_tost_interval(
        rng.normal(0.0, 0.03, size=500),
        estimate=0.0,
        smallest_effect_size=0.20,
    )
    outside = participant_bootstrap_tost_interval(
        rng.normal(0.30, 0.03, size=500),
        estimate=0.30,
        smallest_effect_size=0.20,
    )

    assert near_zero.equivalent
    assert near_zero.ci_low > -0.20
    assert near_zero.ci_high < 0.20
    assert not outside.equivalent
    assert outside.smallest_effect_size == 0.20
    assert outside.method == "participant_cluster_bootstrap_percentile_tost"


def test_prediction_bootstrap_is_deterministic_and_clustered_by_participant() -> None:
    participants = np.repeat([f"sub-{index}" for index in range(8)], 2)
    labels = np.tile([0, 1], 8)
    probabilities = np.tile([0.2, 0.8], 8) + np.repeat(np.linspace(-0.05, 0.05, 8), 2)

    first = participant_bootstrap_prediction_metrics(
        labels,
        probabilities,
        participants,
        repetitions=99,
        seed=23,
    )
    second = participant_bootstrap_prediction_metrics(
        labels,
        probabilities,
        participants,
        repetitions=99,
        seed=23,
    )

    assert first == second
    assert first["participant_bootstrap_unit"] == "participant"
    assert first["participant_bootstrap_successful_binary_resamples"] == 99
    assert first["auroc_bootstrap_successful_repetitions"] == 99
    assert first["auroc_ci_low"] <= first["auroc"] <= first["auroc_ci_high"]


def test_calibration_reports_explicit_unavailable_status_without_logit_variation() -> None:
    diagnostic = calibration_slope_intercept(
        np.asarray([0, 1, 0, 1]),
        np.asarray([0.5, 0.5, 0.5, 0.5]),
    )

    assert diagnostic.status == "unavailable"
    assert diagnostic.reason == "predicted logits have no finite variation"
    assert np.isnan(diagnostic.intercept)
    assert np.isnan(diagnostic.slope)
