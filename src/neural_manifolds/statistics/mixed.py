"""Participant-level mixed-effects model helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MixedModelResult:
    parameters: dict[str, float]
    standard_errors: dict[str, float]
    p_values: dict[str, float]
    converged: bool
    n_participants: int
    n_observations: int
    formula: str


def fit_participant_mixed_model(
    frame: pd.DataFrame,
    *,
    outcome: str,
    fixed_effects: list[str],
    participant: str = "participant_id",
    random_slope: str | None = None,
) -> MixedModelResult:
    """Fit a Gaussian mixed model to participant-level or nested summaries.

    Robust Student-t models are implemented in the optional Bayesian workflow;
    this frequentist model is the prespecified robustness analogue.
    """

    required = {outcome, participant, *fixed_effects}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame[participant].nunique() < 3:
        raise ValueError("at least three participants are required")
    if not np.all(np.isfinite(frame[outcome].to_numpy(dtype=float))):
        raise ValueError("outcome contains non-finite values")
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("statsmodels is required for mixed models") from exc
    formula = f"{outcome} ~ " + " + ".join(fixed_effects)
    re_formula = f"~{random_slope}" if random_slope else "1"
    fitted = smf.mixedlm(
        formula,
        frame,
        groups=frame[participant],
        re_formula=re_formula,
    ).fit(reml=False, method="lbfgs")
    fixed_names = list(fitted.fe_params.index)
    return MixedModelResult(
        parameters={name: float(fitted.fe_params[name]) for name in fixed_names},
        standard_errors={name: float(fitted.bse_fe[name]) for name in fixed_names},
        p_values={name: float(fitted.pvalues[name]) for name in fixed_names},
        converged=bool(fitted.converged),
        n_participants=int(frame[participant].nunique()),
        n_observations=len(frame),
        formula=formula,
    )
