from __future__ import annotations

import numpy as np

from neural_manifolds.statistics.folds import (
    maximum_participant_stratified_splits,
    participant_stratified_test_sets,
)


def test_between_participant_folds_preserve_classes_and_whole_participants() -> None:
    participants = np.asarray(
        [f"low-{index}" for index in range(6)] + [f"high-{index}" for index in range(6)]
    )
    labels = np.asarray([0] * 6 + [1] * 6)

    folds = list(
        participant_stratified_test_sets(
            participants,
            labels,
            n_splits=3,
            seed=17,
        )
    )

    assert maximum_participant_stratified_splits(participants, labels) == 6
    assert len(folds) == 3
    assert set(np.concatenate(folds)) == set(participants)
    for test_participants in folds:
        test = np.isin(participants, test_participants)
        train = ~test
        assert set(participants[test]).isdisjoint(participants[train])
        assert set(labels[test]) == {0, 1}


def test_within_participant_signature_remains_stratifiable() -> None:
    participants = np.repeat([f"sub-{index}" for index in range(9)], 2)
    labels = np.tile([0, 1], 9)

    folds = list(
        participant_stratified_test_sets(
            participants,
            labels,
            n_splits=3,
            seed=19,
        )
    )

    assert len(folds) == 3
    for test_participants in folds:
        test = np.isin(participants, test_participants)
        assert set(labels[test]) == {0, 1}
        assert len(np.unique(participants[test])) == 3
