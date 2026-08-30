"""Frozen healthy wake-versus-propofol reference for clinical transfer."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import OAS

from neural_manifolds.manifold.profile import AXIS_NAMES

WAKE_REGIME_LLR = "wake_regime_log_likelihood_ratio"


class FrozenWakePropofolLikelihoodRatio:
    """OAS-regularised Gaussian log-likelihood ratio in the five-axis space.

    The reference is fit once from participant-level paired healthy wake and
    propofol profiles.  Equal class priors are fixed deliberately, so the score
    is not shifted by condition-specific recording counts.  Positive values are
    more likely under the frozen wake distribution; the value is not a
    probability of consciousness.
    """

    wake_condition = "awake"
    propofol_condition = "propofol_sedation"
    reference_dataset = "propofol_tms_eeg"
    axes = AXIS_NAMES

    def fit(self, profiles: pd.DataFrame) -> FrozenWakePropofolLikelihoodRatio:
        if getattr(self, "fitted_", False):
            raise RuntimeError("the wake-versus-propofol reference is already frozen")
        required = {"participant_id", "dataset_id", "condition", *self.axes}
        missing = required.difference(profiles.columns)
        if missing:
            raise ValueError(f"reference profiles lack columns {sorted(missing)}")
        selected = profiles[
            profiles["dataset_id"].astype(str).eq(self.reference_dataset)
            & profiles["condition"].astype(str).isin([self.wake_condition, self.propofol_condition])
        ].copy()
        for axis in self.axes:
            selected[axis] = pd.to_numeric(selected[axis], errors="coerce")
        if selected[list(self.axes)].isna().any(axis=None):
            raise ValueError("wake/propofol reference profiles contain missing axes")
        participant = selected.groupby(["participant_id", "condition"], as_index=False, sort=True)[
            list(self.axes)
        ].mean()
        counts = participant.groupby("participant_id")["condition"].nunique()
        paired_ids = sorted(counts[counts == 2].index.astype(str))
        participant = participant[participant["participant_id"].astype(str).isin(paired_ids)].copy()
        if len(paired_ids) < 3:
            raise ValueError(
                "wake-versus-propofol reference requires at least three paired participants"
            )
        fitted: dict[str, OAS] = {}
        for condition in (self.wake_condition, self.propofol_condition):
            values = participant.loc[
                participant["condition"].astype(str).eq(condition), list(self.axes)
            ].to_numpy(dtype=np.float64)
            if values.shape != (len(paired_ids), len(self.axes)) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid participant-level {condition} reference matrix")
            fitted[condition] = OAS(store_precision=True).fit(values)
        self.wake_mean_ = fitted[self.wake_condition].location_.copy()
        self.wake_covariance_ = fitted[self.wake_condition].covariance_.copy()
        self.wake_precision_ = fitted[self.wake_condition].precision_.copy()
        self.propofol_mean_ = fitted[self.propofol_condition].location_.copy()
        self.propofol_covariance_ = fitted[self.propofol_condition].covariance_.copy()
        self.propofol_precision_ = fitted[self.propofol_condition].precision_.copy()
        self.n_paired_participants_ = len(paired_ids)
        self.participant_set_sha256_ = hashlib.sha256(
            json.dumps(paired_ids, ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.equal_class_priors_ = True
        self.fitted_ = True
        return self

    @staticmethod
    def _log_density(
        value: np.ndarray,
        *,
        mean: np.ndarray,
        covariance: np.ndarray,
        precision: np.ndarray,
    ) -> float:
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(log_determinant):
            raise RuntimeError("frozen clinical reference covariance is not positive definite")
        delta = value - mean
        mahalanobis = float(delta @ precision @ delta)
        return float(-0.5 * (len(value) * np.log(2.0 * np.pi) + log_determinant + mahalanobis))

    def score(self, profile: np.ndarray | list[float] | tuple[float, ...]) -> float:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("wake-versus-propofol reference is not fitted")
        value = np.asarray(profile, dtype=np.float64)
        if value.shape != (len(self.axes),):
            raise ValueError(f"clinical profile must have shape ({len(self.axes)},)")
        if not np.all(np.isfinite(value)):
            raise ValueError(
                "clinical profile has missing/non-finite axes; likelihood ratio is unavailable"
            )
        wake = self._log_density(
            value,
            mean=self.wake_mean_,
            covariance=self.wake_covariance_,
            precision=self.wake_precision_,
        )
        propofol = self._log_density(
            value,
            mean=self.propofol_mean_,
            covariance=self.propofol_covariance_,
            precision=self.propofol_precision_,
        )
        return wake - propofol

    def audit(self) -> dict[str, Any]:
        if not getattr(self, "fitted_", False):
            return {
                "status": "unavailable",
                "reason": "reference_not_fitted",
            }
        return {
            "status": "frozen",
            "kind": "healthy_wake_vs_propofol_gaussian_log_likelihood_ratio",
            "reference_dataset": self.reference_dataset,
            "wake_condition": self.wake_condition,
            "propofol_condition": self.propofol_condition,
            "axes": list(self.axes),
            "covariance": "oas_shrinkage",
            "class_priors": "equal",
            "paired_participants": self.n_paired_participants_,
            "participant_set_sha256": self.participant_set_sha256_,
            "interpretation": "positive_is_more_wake_like_not_probability_of_consciousness",
        }
