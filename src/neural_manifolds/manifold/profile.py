"""Reference-calibrated five-axis manifold-regime profile API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._base import EstimatorMixin, require_fitted
from ._validation import safe_scale
from .alignment import PairwiseAlignmentSummary, pairwise_module_alignment
from .directionality import DirectionalitySummary, estimate_directionality
from .metastability import (
    MetastabilityReference,
    MetastabilitySummary,
    estimate_metastability,
)
from .reachability import (
    LocalLinearDynamics,
    StateWeightedReachabilitySummary,
    fit_local_linear_dynamics,
    state_weighted_reachability,
)
from .repertoire import RepertoireSummary, estimate_repertoire

AXIS_NAMES = ("repertoire", "metastability", "directionality", "alignment", "reachability")


@dataclass(frozen=True)
class ManifoldRecord:
    """Inputs required to estimate one participant-condition regime profile."""

    # ``trajectory`` is the discovery-fitted dynamics projection used for local
    # dynamics and state-dependent axes. ``repertoire_trajectory`` retains the
    # untruncated frozen-encoder embedding for R. Standalone callers may omit the
    # latter, in which case the same trajectory is used for both spaces.
    trajectory: ArrayLike
    states: ArrayLike
    regional_trajectories: Mapping[str, ArrayLike]
    repertoire_trajectory: ArrayLike | None = None
    segment_ids: ArrayLike | None = None
    alignment_segment_ids: ArrayLike | None = None
    local_dynamics: LocalLinearDynamics | None = None
    name: str | None = None


@dataclass(frozen=True)
class ManifoldProfileDetails:
    """Uncollapsed estimators retained so every axis remains auditable."""

    repertoire: RepertoireSummary
    metastability: MetastabilitySummary
    directionality: DirectionalitySummary
    alignment: PairwiseAlignmentSummary
    reachability: StateWeightedReachabilitySummary
    local_dynamics: LocalLinearDynamics


@dataclass(frozen=True)
class ManifoldProfile:
    """Five numeric axes plus raw, separately inspectable property summaries."""

    values: NDArray[np.float64]
    raw_values: NDArray[np.float64]
    details: ManifoldProfileDetails
    standardized: bool
    name: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        raw = np.asarray(self.raw_values, dtype=np.float64)
        if values.shape != (5,) or raw.shape != (5,):
            raise ValueError("ManifoldProfile values and raw_values must have shape (5,)")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(raw)):
            raise ValueError("ManifoldProfile axes must be finite")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "raw_values", raw.copy())

    @property
    def repertoire(self) -> float:
        return float(self.values[0])

    @property
    def metastability(self) -> float:
        return float(self.values[1])

    @property
    def directionality(self) -> float:
        return float(self.values[2])

    @property
    def alignment(self) -> float:
        return float(self.values[3])

    @property
    def reachability(self) -> float:
        return float(self.values[4])

    def as_array(self, *, raw: bool = False, copy: bool = True) -> NDArray[np.float64]:
        """Return axes in the fixed ``R, M, D, A, P`` order."""

        result = self.raw_values if raw else self.values
        return result.copy() if copy else result

    def as_dict(self, *, raw: bool = False) -> dict[str, float]:
        values = self.raw_values if raw else self.values
        return dict(zip(AXIS_NAMES, (float(value) for value in values), strict=True))


def _coerce_record(record: ManifoldRecord | Mapping[str, Any]) -> ManifoldRecord:
    if isinstance(record, ManifoldRecord):
        return record
    if not isinstance(record, Mapping):
        raise TypeError("each record must be a ManifoldRecord or mapping")
    required = {"trajectory", "states", "regional_trajectories"}
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"record is missing required fields {missing!r}")
    return ManifoldRecord(
        trajectory=record["trajectory"],
        states=record["states"],
        regional_trajectories=record["regional_trajectories"],
        repertoire_trajectory=record.get("repertoire_trajectory"),
        segment_ids=record.get("segment_ids"),
        alignment_segment_ids=record.get("alignment_segment_ids"),
        local_dynamics=record.get("local_dynamics"),
        name=record.get("name"),
    )


def _coerce_records(
    records: ManifoldRecord | Mapping[str, Any] | Sequence[ManifoldRecord | Mapping[str, Any]],
) -> list[ManifoldRecord]:
    if isinstance(records, (ManifoldRecord, Mapping)):
        return [_coerce_record(records)]
    result = [_coerce_record(record) for record in records]
    if not result:
        raise ValueError("records must not be empty")
    return result


class FiveAxisProfileEstimator(EstimatorMixin):
    """Estimate and healthy-reference-standardise the ``(R, M, D, A, P)`` profile.

    ``fit`` accepts healthy discovery records only. It learns (i) the
    persistence-revisability optimum required for a defensible M scalar and (ii)
    optional axis standardisation. It does not consume consciousness, diagnosis,
    drug, report or task labels.
    """

    def __init__(
        self,
        *,
        repertoire_shrinkage: str | float | None = "oas",
        repertoire_noise_variance: float = 0.0,
        sample_interval: float = 1.0,
        directionality_pseudocount: float = 0.5,
        alignment_lags: tuple[int, ...] = (1,),
        alignment_rank: int | None = None,
        alignment_ridge: float = 1e-6,
        alignment_cv: int = 5,
        module_pairs: tuple[tuple[str, str], ...] | None = None,
        alignment_bidirectional: bool = True,
        reachability_horizon: int = 10,
        dynamics_ridge: float = 1e-4,
        innovation_regularization: float = 1e-8,
        gramian_regularization: float | None = None,
        min_state_transitions: int = 5,
        metastability_covariance_shrinkage: float = 0.1,
        standardization: str = "zscore",
    ) -> None:
        self.repertoire_shrinkage = repertoire_shrinkage
        self.repertoire_noise_variance = repertoire_noise_variance
        self.sample_interval = sample_interval
        self.directionality_pseudocount = directionality_pseudocount
        self.alignment_lags = alignment_lags
        self.alignment_rank = alignment_rank
        self.alignment_ridge = alignment_ridge
        self.alignment_cv = alignment_cv
        self.module_pairs = module_pairs
        self.alignment_bidirectional = alignment_bidirectional
        self.reachability_horizon = reachability_horizon
        self.dynamics_ridge = dynamics_ridge
        self.innovation_regularization = innovation_regularization
        self.gramian_regularization = gramian_regularization
        self.min_state_transitions = min_state_transitions
        self.metastability_covariance_shrinkage = metastability_covariance_shrinkage
        self.standardization = standardization

    @staticmethod
    def _input_dimensions(record: ManifoldRecord) -> tuple[int, int]:
        dynamics = np.asarray(record.trajectory)
        repertoire = np.asarray(
            record.trajectory
            if record.repertoire_trajectory is None
            else record.repertoire_trajectory
        )
        if dynamics.ndim != 2 or repertoire.ndim != 2:
            raise ValueError("repertoire and dynamics trajectories must be two-dimensional")
        if dynamics.shape[0] != repertoire.shape[0]:
            raise ValueError("repertoire and dynamics trajectories must share temporal rows")
        return int(repertoire.shape[1]), int(dynamics.shape[1])

    def _validate_input_dimensions(self, record: ManifoldRecord) -> None:
        source_dimension, dynamics_dimension = self._input_dimensions(record)
        if hasattr(self, "repertoire_source_dimension_") and (
            source_dimension != self.repertoire_source_dimension_
        ):
            raise ValueError(
                f"repertoire trajectory has {source_dimension} features; expected "
                f"{self.repertoire_source_dimension_}"
            )
        if hasattr(self, "dynamics_projection_dimension_") and (
            dynamics_dimension != self.dynamics_projection_dimension_
        ):
            raise ValueError(
                f"dynamics trajectory has {dynamics_dimension} features; expected "
                f"{self.dynamics_projection_dimension_}"
            )

    def input_space_audit(self) -> dict[str, Any]:
        """Describe the two serialized input spaces used by the fitted profile."""

        require_fitted(
            self,
            "repertoire_source_dimension_",
            "dynamics_projection_dimension_",
        )
        return {
            "repertoire": {
                "record_field": "repertoire_trajectory",
                "space": "untruncated_frozen_encoder_embedding",
                "dimension": int(self.repertoire_source_dimension_),
                "discovery_projection_applied": False,
            },
            "dynamics": {
                "record_field": "trajectory",
                "space": "discovery_fitted_pca_projection",
                "dimension": int(self.dynamics_projection_dimension_),
                "discovery_projection_applied": True,
            },
        }

    def _estimate_details(self, record: ManifoldRecord) -> ManifoldProfileDetails:
        self._validate_input_dimensions(record)
        repertoire_trajectory = (
            record.trajectory
            if record.repertoire_trajectory is None
            else record.repertoire_trajectory
        )
        repertoire = estimate_repertoire(
            repertoire_trajectory,
            shrinkage=self.repertoire_shrinkage,
            noise_variance=self.repertoire_noise_variance,
        )
        metastability = estimate_metastability(
            record.states,
            segment_ids=record.segment_ids,
            sample_interval=self.sample_interval,
        )
        directionality = estimate_directionality(
            record.states,
            segment_ids=record.segment_ids,
            pseudocount=self.directionality_pseudocount,
            sample_interval=self.sample_interval,
        )
        alignment = pairwise_module_alignment(
            record.regional_trajectories,
            module_pairs=self.module_pairs,
            lags=self.alignment_lags,
            rank=self.alignment_rank,
            ridge=self.alignment_ridge,
            cv=self.alignment_cv,
            segment_ids=record.alignment_segment_ids,
            bidirectional=self.alignment_bidirectional,
        )
        dynamics = record.local_dynamics
        if dynamics is None:
            dynamics = fit_local_linear_dynamics(
                record.trajectory,
                record.states,
                segment_ids=record.segment_ids,
                ridge=self.dynamics_ridge,
                innovation_regularization=self.innovation_regularization,
                min_transitions=self.min_state_transitions,
            )
        reachability = state_weighted_reachability(
            dynamics.transition_matrices,
            dynamics.innovation_covariances,
            dynamics.occupancy,
            horizon=self.reachability_horizon,
            regularization=self.gramian_regularization,
        )
        return ManifoldProfileDetails(
            repertoire=repertoire,
            metastability=metastability,
            directionality=directionality,
            alignment=alignment,
            reachability=reachability,
            local_dynamics=dynamics,
        )

    def _raw_values(self, details: ManifoldProfileDetails) -> NDArray[np.float64]:
        require_fitted(self, "metastability_reference_")
        values = np.asarray(
            [
                details.repertoire.participation_ratio,
                self.metastability_reference_.score(details.metastability),
                details.directionality.entropy_production,
                details.alignment.mean_shared_predictive_variance,
                details.reachability.log_determinant,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "profile contains a non-finite axis; use finite transition "
                "pseudocounts and Gramian regularisation for profile estimation"
            )
        return values

    def _standardize(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.standardization == "none":
            return values.copy()
        require_fitted(self, "axis_center_", "axis_scale_")
        return (values - self.axis_center_) / self.axis_scale_

    def fit(
        self,
        records: Sequence[ManifoldRecord | Mapping[str, Any]],
        y: ArrayLike | None = None,
    ) -> FiveAxisProfileEstimator:
        del y
        reference_records = _coerce_records(records)
        if len(reference_records) < 2:
            raise ValueError("at least two healthy discovery records are required")
        if self.standardization not in {"zscore", "robust", "none"}:
            raise ValueError("standardization must be 'zscore', 'robust', or 'none'")
        dimensions = [self._input_dimensions(record) for record in reference_records]
        source_dimensions = {source for source, _ in dimensions}
        dynamics_dimensions = {dynamics for _, dynamics in dimensions}
        if len(source_dimensions) != 1 or len(dynamics_dimensions) != 1:
            raise ValueError("healthy discovery records use inconsistent input dimensions")
        self.repertoire_source_dimension_ = source_dimensions.pop()
        self.dynamics_projection_dimension_ = dynamics_dimensions.pop()
        details = [self._estimate_details(record) for record in reference_records]
        self.metastability_reference_ = MetastabilityReference(
            covariance_shrinkage=self.metastability_covariance_shrinkage
        ).fit([item.metastability for item in details])
        raw = np.stack([self._raw_values(item) for item in details], axis=0)
        if self.standardization == "zscore":
            center = np.mean(raw, axis=0)
            scale = np.std(raw, axis=0, ddof=1)
        elif self.standardization == "robust":
            center = np.median(raw, axis=0)
            scale = np.asarray([safe_scale(raw[:, index]) for index in range(5)])
        else:
            center = np.zeros(5, dtype=np.float64)
            scale = np.ones(5, dtype=np.float64)
        self.constant_axes_ = np.asarray(scale < 1e-12, dtype=bool)
        # A constant discovery axis has no estimable z-score denominator. Keep it
        # on its raw unit scale and expose ``constant_axes_`` rather than amplify
        # numerical noise by dividing through an arbitrary tiny value.
        scale = np.where(self.constant_axes_, 1.0, scale)
        self.axis_center_ = np.asarray(center, dtype=np.float64)
        self.axis_scale_ = np.asarray(scale, dtype=np.float64)
        self.reference_details_ = tuple(details)
        self.reference_raw_values_ = raw
        self.n_features_in_ = 5
        return self

    def transform(
        self,
        records: ManifoldRecord | Mapping[str, Any] | Sequence[ManifoldRecord | Mapping[str, Any]],
    ) -> NDArray[np.float64]:
        require_fitted(
            self,
            "metastability_reference_",
            "axis_center_",
            "axis_scale_",
            "repertoire_source_dimension_",
            "dynamics_projection_dimension_",
        )
        output = []
        for record in _coerce_records(records):
            details = self._estimate_details(record)
            output.append(self._standardize(self._raw_values(details)))
        return np.stack(output, axis=0)

    def fit_transform(
        self,
        records: Sequence[ManifoldRecord | Mapping[str, Any]],
        y: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        self.fit(records, y)
        return np.stack(
            [self._standardize(values) for values in self.reference_raw_values_], axis=0
        )

    def profile(self, record: ManifoldRecord | Mapping[str, Any]) -> ManifoldProfile:
        """Return axes together with all unstandardised component summaries."""

        require_fitted(
            self,
            "metastability_reference_",
            "axis_center_",
            "axis_scale_",
            "repertoire_source_dimension_",
            "dynamics_projection_dimension_",
        )
        coerced = _coerce_record(record)
        details = self._estimate_details(coerced)
        raw = self._raw_values(details)
        return ManifoldProfile(
            values=self._standardize(raw),
            raw_values=raw,
            details=details,
            standardized=self.standardization != "none",
            name=coerced.name,
        )


# A discoverable semantic alias for users who search for "manifold profile".
ManifoldProfileEstimator = FiveAxisProfileEstimator
