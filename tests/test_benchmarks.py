import numpy as np
import pytest

from neural_manifolds.benchmarks import (
    FrozenMicrostateModel,
    normalized_lempel_ziv,
    permutation_entropy,
    relative_band_power,
    spectral_exponent,
    weighted_phase_lag_index,
    weighted_symbolic_mutual_information,
)


def test_benchmarks_are_finite() -> None:
    rng = np.random.default_rng(18)
    times = np.arange(2000) / 200
    data = np.stack([np.sin(2 * np.pi * 10 * times), np.sin(2 * np.pi * 10 * times + 0.3)])
    data += rng.normal(scale=0.1, size=data.shape)
    powers = relative_band_power(data, 200)
    assert powers["alpha"] > powers["delta"]
    assert np.isfinite(spectral_exponent(data, 200))
    assert 0 <= permutation_entropy(data[0]) <= 1
    assert normalized_lempel_ziv(data) > 0
    assert weighted_phase_lag_index(data, 200).shape == (2, 2)


def test_wsmi_recovers_coupling_and_is_bitwise_deterministic() -> None:
    rng = np.random.default_rng(123)
    sfreq = 200.0
    first = rng.normal(size=5000)
    coupled = np.roll(first, 2) + rng.normal(scale=0.1, size=len(first))
    independent = rng.normal(size=len(first))
    data = np.stack([first, coupled, independent])

    observed = weighted_symbolic_mutual_information(data, sfreq)
    repeated = weighted_symbolic_mutual_information(data, sfreq)

    assert observed.value == repeated.value
    assert np.array_equal(observed.matrix, repeated.matrix)
    assert observed.matrix[0, 1] > 10 * max(
        abs(observed.matrix[0, 2]),
        abs(observed.matrix[1, 2]),
    )
    upper = observed.matrix[np.triu_indices(3, k=1)]
    assert observed.value == float(np.median(upper))
    audit = observed.audit(sfreq=sfreq)
    assert audit["order"] == 3
    assert audit["lag_samples"] == 6
    assert audit["lag_seconds"] == pytest.approx(0.03)
    assert audit["lowpass_hz"] == 10.0
    assert audit["minimum_symbol_samples"] == 180
    assert audit["excluded_pair_weights"] == [
        "identical_patterns",
        "sign_reversed_patterns",
    ]
    assert audit["channel_pair_aggregation"] == "median_upper_triangle"


def test_wsmi_excludes_shared_source_and_sign_reversal_and_requires_enough_symbols() -> None:
    rng = np.random.default_rng(321)
    source = rng.normal(size=2000)
    identical = weighted_symbolic_mutual_information(np.stack([source, source]), 200.0)
    sign_reversed = weighted_symbolic_mutual_information(np.stack([source, -source]), 200.0)
    assert identical.value == pytest.approx(0.0, abs=1e-15)
    assert sign_reversed.value == pytest.approx(0.0, abs=1e-15)

    with pytest.raises(ValueError, match=r"requires at least 180"):
        weighted_symbolic_mutual_information(rng.normal(size=(2, 191)), 200.0)


def _microstate_discovery_maps() -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(4)
    prototypes = np.asarray(
        [
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
        ]
    )
    prototypes -= prototypes.mean(axis=1, keepdims=True)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    participant_maps: dict[str, np.ndarray] = {}
    for participant in ("discovery-b", "discovery-a", "discovery-c"):
        labels = np.tile(np.arange(4), 30)
        participant_maps[participant] = prototypes[labels] + rng.normal(
            scale=0.03,
            size=(len(labels), 4),
        )
    return participant_maps, prototypes


def test_microstate_prototypes_are_discovery_only_frozen_and_deterministic() -> None:
    participant_maps, prototypes = _microstate_discovery_maps()
    first = FrozenMicrostateModel().fit(participant_maps)
    second = FrozenMicrostateModel().fit(dict(reversed(list(participant_maps.items()))))
    assert np.array_equal(first.prototypes_, second.prototypes_)
    assert first.prototype_sha256_ == second.prototype_sha256_
    assert first.audit()["discovery_participants"] == 3
    assert first.audit()["fit_label_fields"] == []
    assert first.audit()["refit_after_discovery"] is False

    rng = np.random.default_rng(8)
    labels = np.repeat(np.arange(4), 50)
    signal = (prototypes[labels] + rng.normal(scale=0.03, size=(len(labels), 4))).T
    outcomes = first.score(signal, 200.0)
    assert set(outcomes) == {
        "microstate_transition_entropy",
        "microstate_global_explained_variance",
        "microstate_median_duration_seconds",
    }
    assert all(np.isfinite(value) for value in outcomes.values())
    assert outcomes["microstate_global_explained_variance"] > 0.95
    with pytest.raises(RuntimeError, match="already frozen"):
        first.fit(participant_maps)
