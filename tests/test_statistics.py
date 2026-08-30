import numpy as np

from neural_manifolds.statistics.multivariate import (
    benjamini_hochberg,
    energy_distance,
    permutation_pvalue,
    profile_mahalanobis_distance,
)
from neural_manifolds.statistics.resampling import participant_folds, permute_within_dataset


def test_folds_do_not_leak_participants() -> None:
    ids = np.repeat([f"p{i}" for i in range(10)], 3)
    for train, test in participant_folds(ids, n_splits=5, seed=3):
        assert not set(ids[train]).intersection(ids[test])


def test_permutation_preserves_dataset_label_counts() -> None:
    participants = np.repeat(["a", "b", "c", "d"], 2)
    datasets = np.repeat(["x", "x", "y", "y"], 2)
    labels = np.repeat([0, 1, 0, 1], 2)
    shuffled = permute_within_dataset(labels, participants, datasets, seed=5)
    for dataset in np.unique(datasets):
        assert sorted(shuffled[datasets == dataset]) == sorted(labels[datasets == dataset])


def test_multivariate_distances_and_fdr() -> None:
    rng = np.random.default_rng(9)
    a = rng.normal(size=(50, 5))
    b = rng.normal(loc=0.7, size=(50, 5))
    assert profile_mahalanobis_distance(a, b) > 0
    assert energy_distance(a, b) > 0
    assert permutation_pvalue(3.0, [0, 1, 2]) == 0.25
    adjusted, reject = benjamini_hochberg([0.001, 0.02, 0.8])
    assert reject[:2].all()
    assert not reject[2]
    assert np.all(adjusted >= 0)
