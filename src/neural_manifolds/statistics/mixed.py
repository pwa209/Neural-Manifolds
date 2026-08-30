"""Participant-level mixed-effects model helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MixedModelResult:
    parameters: dict[str, float]
    standard_errors: dict[str, float]
    p_values: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    converged: bool
    n_participants: int
    n_observations: int
    formula: str
    random_intercept_variance: float
    optimizer: str


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
    if random_slope is not None:
        required.add(random_slope)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame[participant].nunique() < 3:
        raise ValueError("at least three participants are required")
    if frame[participant].isna().any():
        raise ValueError("participant identifiers contain missing values")
    if not np.all(np.isfinite(frame[outcome].to_numpy(dtype=float))):
        raise ValueError("outcome contains non-finite values")
    for fixed_effect in fixed_effects:
        if pd.api.types.is_numeric_dtype(frame[fixed_effect]) and not np.all(
            np.isfinite(frame[fixed_effect].to_numpy(dtype=float))
        ):
            raise ValueError(f"fixed effect {fixed_effect!r} contains non-finite values")
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("statsmodels is required for mixed models") from exc
    formula = f"{outcome} ~ " + " + ".join(fixed_effects)
    re_formula = f"~{random_slope}" if random_slope else "1"
    model = smf.mixedlm(
        formula,
        frame,
        groups=frame[participant],
        re_formula=re_formula,
    )
    fitted = None
    failures: list[str] = []
    optimizer = "unavailable"
    for candidate in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate_fit = model.fit(reml=False, method=candidate, disp=False)
            fitted = candidate_fit
            optimizer = candidate
            if bool(candidate_fit.converged):
                break
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failures.append(f"{candidate}: {type(error).__name__}: {error}")
    if fitted is None:
        raise RuntimeError("mixed model failed: " + " | ".join(failures))
    fixed_names = list(fitted.fe_params.index)
    confidence = fitted.conf_int().loc[fixed_names]
    covariance = np.asarray(fitted.cov_re, dtype=np.float64)
    random_intercept_variance = float(covariance[0, 0]) if covariance.size else np.nan
    return MixedModelResult(
        parameters={name: float(fitted.fe_params[name]) for name in fixed_names},
        standard_errors={name: float(fitted.bse_fe[name]) for name in fixed_names},
        p_values={name: float(fitted.pvalues[name]) for name in fixed_names},
        confidence_intervals={
            name: (float(confidence.loc[name, 0]), float(confidence.loc[name, 1]))
            for name in fixed_names
        },
        converged=bool(fitted.converged),
        n_participants=int(frame[participant].nunique()),
        n_observations=len(frame),
        formula=formula,
        random_intercept_variance=random_intercept_variance,
        optimizer=optimizer,
    )
