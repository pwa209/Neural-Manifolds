"""Participant-level resampling, multivariate inference, and transfer utilities."""

from .multivariate import (
    benjamini_hochberg,
    energy_distance,
    permutation_pvalue,
    profile_mahalanobis_distance,
)
from .resampling import participant_bootstrap, participant_folds, permute_within_dataset

__all__ = [
    "benjamini_hochberg",
    "energy_distance",
    "participant_bootstrap",
    "participant_folds",
    "permutation_pvalue",
    "permute_within_dataset",
    "profile_mahalanobis_distance",
]
