"""Label-stratified folds that keep every participant wholly in one partition."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np


def _participant_strata(
    participant_ids: Sequence[str], labels: Sequence[int]
) -> dict[tuple[int, ...], list[str]]:
    participants = np.asarray(participant_ids, dtype=str)
    targets = np.asarray(labels, dtype=int)
    if participants.ndim != 1 or targets.ndim != 1 or len(participants) != len(targets):
        raise ValueError("participant_ids and labels must be aligned one-dimensional arrays")
    strata: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for participant in np.unique(participants):
        signature = tuple(
            sorted(int(value) for value in np.unique(targets[participants == participant]))
        )
        strata[signature].append(str(participant))
    return dict(strata)


def maximum_participant_stratified_splits(
    participant_ids: Sequence[str], labels: Sequence[int]
) -> int:
    strata = _participant_strata(participant_ids, labels)
    return min((len(participants) for participants in strata.values()), default=0)


def participant_stratified_test_sets(
    participant_ids: Sequence[str],
    labels: Sequence[int],
    *,
    n_splits: int,
    seed: int,
) -> Iterator[np.ndarray]:
    """Yield stratified test-participant IDs without ever splitting a participant."""

    strata = _participant_strata(participant_ids, labels)
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if not strata or any(len(participants) < n_splits for participants in strata.values()):
        raise ValueError("every participant-label stratum must support every fold")
    rng = np.random.default_rng(seed)
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(n_splits)]
    for signature in sorted(strata):
        participants = np.asarray(strata[signature], dtype=str)
        rng.shuffle(participants)
        for fold, chunk in enumerate(np.array_split(participants, n_splits)):
            fold_parts[fold].append(chunk)
    test_sets = [np.concatenate(parts) for parts in fold_parts]
    all_participants = set(np.unique(np.asarray(participant_ids, dtype=str)))
    assigned = np.concatenate(test_sets)
    if len(set(assigned)) != len(assigned) or set(assigned) != all_participants:
        raise AssertionError("stratified folds must partition participants exactly once")
    for test in test_sets:
        if not len(test):
            raise AssertionError("stratified participant fold is empty")
        if len(set(test)) != len(test):
            raise AssertionError("participant appears twice in a stratified fold")
        yield test
