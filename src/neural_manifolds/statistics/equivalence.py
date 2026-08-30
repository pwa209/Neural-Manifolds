"""Smallest-effect equivalence intervals with explicit TOST semantics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class EquivalenceInterval:
    estimate: float
    smallest_effect_size: float
    alpha: float
    ci_low: float
    ci_high: float
    lower_test_p_value: float
    upper_test_p_value: float
    equivalent: bool
    method: str
    bootstrap_repetitions: int


def participant_bootstrap_tost_interval(
    bootstrap_estimates: np.ndarray,
    *,
    estimate: float,
    smallest_effect_size: float,
    alpha: float = 0.05,
) -> EquivalenceInterval:
    """Invert two one-sided tests using a participant-bootstrap interval.

    The percentile interval is ``100 * (1 - 2 * alpha)`` percent wide, as
    required for a two-one-sided equivalence test at level ``alpha``.  A
    conventional non-significant difference is never interpreted as evidence
    of equivalence.
    """

    values = np.asarray(bootstrap_estimates, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.ndim != 1 or len(values) < 20:
        raise ValueError("equivalence requires at least 20 finite bootstrap estimates")
    if not np.isfinite(estimate):
        raise ValueError("equivalence estimate must be finite")
    if not np.isfinite(smallest_effect_size) or smallest_effect_size <= 0:
        raise ValueError("smallest_effect_size must be finite and positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must lie strictly between zero and one half")

    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    standard_error = float(np.std(values, ddof=1))
    if standard_error <= np.finfo(float).eps:
        lower_p = 0.0 if estimate > -smallest_effect_size else 1.0
        upper_p = 0.0 if estimate < smallest_effect_size else 1.0
    else:
        normal = NormalDist()
        lower_z = (estimate + smallest_effect_size) / standard_error
        upper_z = (estimate - smallest_effect_size) / standard_error
        lower_p = 1.0 - normal.cdf(lower_z)
        upper_p = normal.cdf(upper_z)
    equivalent = bool(low > -smallest_effect_size and high < smallest_effect_size)
    return EquivalenceInterval(
        estimate=float(estimate),
        smallest_effect_size=float(smallest_effect_size),
        alpha=float(alpha),
        ci_low=float(low),
        ci_high=float(high),
        lower_test_p_value=float(lower_p),
        upper_test_p_value=float(upper_p),
        equivalent=equivalent,
        method="participant_cluster_bootstrap_percentile_tost",
        bootstrap_repetitions=len(values),
    )
