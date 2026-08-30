from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_manifolds.manifold.clinical_reference import (
    WAKE_REGIME_LLR,
    FrozenWakePropofolLikelihoodRatio,
)
from neural_manifolds.manifold.profile import AXIS_NAMES
from neural_manifolds.stages.clinical import _association_rows


def _reference_profiles() -> pd.DataFrame:
    rows = []
    for index in range(8):
        participant = f"p{index:02d}"
        offset = (index - 3.5) * 0.04
        rows.extend(
            [
                {
                    "participant_id": participant,
                    "dataset_id": "propofol_tms_eeg",
                    "condition": "awake",
                    **{
                        axis: 1.0 + offset + axis_index * 0.03
                        for axis_index, axis in enumerate(AXIS_NAMES)
                    },
                },
                {
                    "participant_id": participant,
                    "dataset_id": "propofol_tms_eeg",
                    "condition": "propofol_sedation",
                    **{
                        axis: -1.0 + offset - axis_index * 0.02
                        for axis_index, axis in enumerate(AXIS_NAMES)
                    },
                },
            ]
        )
    return pd.DataFrame(rows)


def test_frozen_wake_propofol_likelihood_ratio_has_declared_direction() -> None:
    model = FrozenWakePropofolLikelihoodRatio().fit(_reference_profiles())
    wake = np.full(len(AXIS_NAMES), 1.0)
    propofol = np.full(len(AXIS_NAMES), -1.0)
    assert model.score(wake) > 0
    assert model.score(propofol) < 0
    assert model.audit()["class_priors"] == "equal"
    assert model.audit()["paired_participants"] == 8
    assert WAKE_REGIME_LLR == "wake_regime_log_likelihood_ratio"


def test_frozen_reference_rejects_missing_inputs_and_refit() -> None:
    model = FrozenWakePropofolLikelihoodRatio().fit(_reference_profiles())
    missing = np.zeros(len(AXIS_NAMES))
    missing[2] = np.nan
    with pytest.raises(ValueError, match="likelihood ratio is unavailable"):
        model.score(missing)
    with pytest.raises(RuntimeError, match="already frozen"):
        model.fit(_reference_profiles())


def test_reference_uses_paired_participants_only_and_fails_when_too_few() -> None:
    profiles = _reference_profiles()
    profiles = profiles[
        ~profiles["participant_id"].isin(["p02", "p03", "p04", "p05", "p06", "p07"])
    ]
    with pytest.raises(ValueError, match="three paired participants"):
        FrozenWakePropofolLikelihoodRatio().fit(profiles)


def test_clinical_associations_are_participant_resampled_fdr_corrected_and_deterministic() -> None:
    rows = []
    for index in range(12):
        score = float(index)
        rows.append(
            {
                "dataset_id": "clinical-a",
                "participant_id": f"c{index:02d}",
                "diagnosis": "MCS" if index < 6 else "UWS",
                "crs_r_total": score,
                **{
                    axis: score * (0.08 + axis_index * 0.01) + (-1) ** index * 0.02
                    for axis_index, axis in enumerate(AXIS_NAMES)
                },
                WAKE_REGIME_LLR: score * 0.1 + (-1) ** index * 0.03,
            }
        )
    arguments = {
        "bootstrap_repetitions": 199,
        "permutation_repetitions": 199,
        "seed": 1701,
        "fdr_alpha": 0.05,
    }
    first = _association_rows(pd.DataFrame(rows), **arguments)
    second = _association_rows(pd.DataFrame(rows), **arguments)

    assert first == second
    assert len(first) == 2 * (len(AXIS_NAMES) + 1)
    assert {row["p_value_method"] for row in first} == {"participant_label_permutation_plus_one"}
    assert all(0 <= row["p_value"] <= row["p_value_fdr"] <= 1 for row in first)
    assert all(row["bootstrap_valid_repetitions"] > 0 for row in first)
    assert all(np.isfinite([row["estimate"], row["ci_low"], row["ci_high"]]).all() for row in first)
    assert all(row["ci_low"] <= row["ci_high"] for row in first)
    assert all(row["scientific_gate_applied"] is False for row in first)
