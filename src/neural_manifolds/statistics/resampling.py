"""Leakage-safe resampling where participants, not windows, are exchangeable."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np


def participant_folds(
    participant_ids: Sequence[str],
    *,
    n_splits: int,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    ids = np.asarray(participant_ids, dtype=str)
    unique = np.unique(ids)
    if n_splits < 2 or n_splits > unique.size:
        raise ValueError("n_splits must be between 2 and the participant count")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    for test_participants in np.array_split(shuffled, n_splits):
        test = np.isin(ids, test_participants)
        train_indices = np.flatnonzero(~test)
        test_indices = np.flatnonzero(test)
        if set(ids[train_indices]).intersection(ids[test_indices]):
            raise AssertionError("participant leakage detected")
        yield train_indices, test_indices


def participant_bootstrap(
    participant_ids: Sequence[str],
    *,
    repetitions: int,
    seed: int,
) -> Iterator[np.ndarray]:
    """Yield row indices for cluster bootstraps, preserving all rows per participant."""

    ids = np.asarray(participant_ids, dtype=str)
    unique = np.unique(ids)
    if unique.size < 2 or repetitions <= 0:
        raise ValueError("at least two participants and one repetition are required")
    rows = {participant: np.flatnonzero(ids == participant) for participant in unique}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        yield np.concatenate([rows[participant] for participant in sampled])


def permute_within_dataset(
    labels: Sequence[object],
    participant_ids: Sequence[str],
    dataset_ids: Sequence[str],
    *,
    seed: int,
) -> np.ndarray:
    """Permute participant-level labels within dataset and broadcast to all rows."""

    y = np.asarray(labels)
    participants = np.asarray(participant_ids, dtype=str)
    datasets = np.asarray(dataset_ids, dtype=str)
    if not (y.size == participants.size == datasets.size):
        raise ValueError("labels, participant_ids, and dataset_ids must align")
    rng = np.random.default_rng(seed)
    output = y.copy()
    for dataset in np.unique(datasets):
        mask = datasets == dataset
        unique = np.unique(participants[mask])
        participant_labels: dict[str, object] = {}
        for participant in unique:
            values = np.unique(y[mask & (participants == participant)])
            if values.size != 1:
                raise ValueError("each participant must have one label within a dataset")
            participant_labels[participant] = values[0]
        permuted = rng.permutation([participant_labels[participant] for participant in unique])
        mapping = dict(zip(unique, permuted, strict=True))
        output[mask] = [mapping[participant] for participant in participants[mask]]
    return output


def sample_match_conditions(
    values: np.ndarray,
    conditions: Sequence[str],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    """Draw the same number of rows per condition without replacement."""

    x = np.asarray(values)
    condition = np.asarray(conditions, dtype=str)
    if x.shape[0] != condition.size:
        raise ValueError("values and conditions must align")
    groups = {name: np.flatnonzero(condition == name) for name in np.unique(condition)}
    target = min(map(len, groups.values()))
    if target == 0:
        raise ValueError("every condition must contain observations")
    rng = np.random.default_rng(seed)
    output: list[dict[str, np.ndarray]] = []
    for _ in range(repetitions):
        output.append(
            {
                name: x[rng.choice(indices, size=target, replace=False)]
                for name, indices in groups.items()
            }
        )
    return output
