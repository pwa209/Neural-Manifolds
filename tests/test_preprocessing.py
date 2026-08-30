from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from neural_manifolds.preprocessing import eeg
from neural_manifolds.preprocessing.eeg import (
    SensitivityBranchResult,
    auxiliary_channel_inventory,
    canonicalize_channel_name,
    detect_artifact_windows,
    detect_bad_channels,
    make_windows,
    preprocess_mne_raw,
    preprocess_mne_sensitivity_branches,
)
from neural_manifolds.preprocessing.tms import (
    interpolate_continuous_pulses,
    interpolate_pulse_interval,
)


def test_channel_name_normalisation() -> None:
    assert canonicalize_channel_name("EEG FP1-REF") == "Fp1"
    assert canonicalize_channel_name("T3") == "T7"
    assert canonicalize_channel_name("EEG FZ") == "Fz"
    assert canonicalize_channel_name("EEG C3:A2") == "C3"


def test_windows_are_deterministic() -> None:
    x = np.arange(2 * 1000).reshape(2, 1000)
    windows, starts = make_windows(x, 200, 2.0, 1.0)
    assert windows.shape == (4, 2, 400)
    assert starts.tolist() == [0, 200, 400, 600]


def test_flat_bad_channel_is_found() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(scale=1e-6, size=(8, 2000))
    x[3] = 0.0
    result = detect_bad_channels(x, 200)
    assert 3 in result.bad_indices


def test_extreme_window_is_rejected() -> None:
    rng = np.random.default_rng(7)
    windows = rng.normal(scale=1e-6, size=(30, 4, 400))
    windows[-1, 0, 200] = 0.5
    result = detect_artifact_windows(windows, 200)
    assert not result.keep[-1]


def test_auxiliary_inventory_reports_support_without_claiming_ica() -> None:
    raw = SimpleNamespace(
        ch_names=["Fz", "EOG-L", "ECG", "EMG"],
        get_channel_types=lambda: ["eeg", "eog", "ecg", "emg"],
    )
    audit = auxiliary_channel_inventory(raw)
    assert audit["channels"] == {"eog": ["EOG-L"], "ecg": ["ECG"], "emg": ["EMG"]}
    assert audit["ica_support_status"] == "available_eog_or_ecg_reference"
    assert audit["ica_status"] == "not_performed_policy_report_only_with_auxiliary_support"
    assert (
        audit["auxiliary_artifact_control_support_status"] == "available_eog_ecg_or_emg_reference"
    )
    assert (
        audit["auxiliary_artifact_control_status"]
        == "not_performed_policy_report_only_with_auxiliary_support"
    )
    assert audit["auxiliary_channels_used_for_cleaning"] is False


def test_sensitivity_availability_contract_cannot_fabricate_output() -> None:
    with pytest.raises(ValueError, match="requires a signal"):
        SensitivityBranchResult(
            raw=None,
            status="available",
            reason=None,
            metadata={},
        )


class _FakeMNERaw:
    def __init__(
        self,
        data: np.ndarray,
        names: list[str],
        channel_types: list[str],
        *,
        sfreq: float = 200.0,
    ) -> None:
        self._data = np.asarray(data, dtype=float)
        self.ch_names = list(names)
        self._channel_types = list(channel_types)
        self.info: dict[str, object] = {"sfreq": sfreq, "bads": []}

    def copy(self) -> _FakeMNERaw:
        return _FakeMNERaw(
            self._data.copy(),
            self.ch_names.copy(),
            self._channel_types.copy(),
            sfreq=float(self.info["sfreq"]),
        )

    def load_data(self) -> _FakeMNERaw:
        return self

    def get_channel_types(self) -> list[str]:
        return self._channel_types.copy()

    def rename_channels(self, mapping: dict[str, str], *, allow_duplicates: bool) -> None:
        del allow_duplicates
        self.ch_names = [mapping.get(name, name) for name in self.ch_names]

    def pick(self, names: list[str], *, ordered: bool) -> None:
        del ordered
        indices = [self.ch_names.index(name) for name in names]
        self._data = self._data[indices]
        self._channel_types = [self._channel_types[index] for index in indices]
        self.ch_names = list(names)

    def get_data(self) -> np.ndarray:
        return self._data

    def filter(self, *_args: object, **_kwargs: object) -> None:
        return None

    def notch_filter(self, *_args: object, **_kwargs: object) -> None:
        return None

    def resample(self, sampling_hz: float, **_kwargs: object) -> None:
        self.info["sfreq"] = sampling_hz

    def set_montage(self, *_args: object, **_kwargs: object) -> None:
        return None

    def interpolate_bads(self, *, reset_bads: bool, mode: str) -> None:
        del mode
        bads = list(self.info["bads"])
        good = [index for index, name in enumerate(self.ch_names) if name not in bads]
        replacement = np.mean(self._data[good], axis=0)
        for name in bads:
            self._data[self.ch_names.index(name)] = replacement
        if reset_bads:
            self.info["bads"] = []

    def set_eeg_reference(self, reference: str, *, projection: bool) -> None:
        assert reference == "average"
        assert projection is False
        self._data -= np.mean(self._data, axis=0, keepdims=True)


def test_primary_preprocessing_materialises_exact_ordered_19_channel_montage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "O1",
        "O2",
    ]
    observed = canonical[:-1]
    rng = np.random.default_rng(12)
    raw = _FakeMNERaw(
        rng.normal(size=(len(observed) + 1, 1000)),
        [*observed, "EOG-L"],
        [*["eeg"] * len(observed), "eog"],
    )

    def add_reference_channels(
        instance: _FakeMNERaw, names: list[str], *, copy: bool
    ) -> _FakeMNERaw:
        assert copy is False
        instance._data = np.vstack([instance._data, np.zeros((len(names), 1000))])
        instance.ch_names.extend(names)
        instance._channel_types.extend(["eeg"] * len(names))
        return instance

    fake_mne = SimpleNamespace(
        __version__="1.12.1-test-double",
        add_reference_channels=add_reference_channels,
        channels=SimpleNamespace(make_standard_montage=lambda _name: object()),
    )
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    monkeypatch.setattr(
        eeg,
        "detect_bad_channels",
        lambda *_args, **_kwargs: SimpleNamespace(bad_indices=np.empty(0, dtype=int)),
    )
    clean, provenance = preprocess_mne_raw(
        raw,
        canonical_channels=canonical,
        maximum_interpolation_fraction=0.15,
        require_complete_canonical=True,
    )
    assert clean.ch_names == canonical
    assert provenance["missing_canonical_channels"] == ["O2"]
    assert provenance["interpolated_channels"] == ["O2"]
    assert provenance["canonical_montage_complete"] is True
    assert provenance["reference"] == "average"
    assert provenance["auxiliary_channel_audit"]["channels"]["eog"] == ["EOG-L"]
    np.testing.assert_allclose(np.mean(clean.get_data(), axis=0), 0.0, atol=1e-12)


def test_missing_positions_make_csd_unavailable_without_losing_native_average(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "O1",
        "O2",
    ]
    rng = np.random.default_rng(21)
    raw = _FakeMNERaw(
        rng.normal(size=(len(canonical), 1000)),
        canonical,
        ["eeg"] * len(canonical),
    )
    fake_mne = SimpleNamespace(
        __version__="1.12.1-test-double",
        preprocessing=SimpleNamespace(
            compute_current_source_density=lambda _raw: pytest.fail(
                "CSD must not run without the configured position fraction"
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    monkeypatch.setattr(
        eeg,
        "detect_bad_channels",
        lambda *_args, **_kwargs: SimpleNamespace(bad_indices=np.empty(0, dtype=int)),
    )
    results = preprocess_mne_sensitivity_branches(
        raw,
        canonical_channels=canonical,
        target_sampling_hz=200.0,
        primary_highpass_hz=0.1,
        sleep_highpass_hz=0.3,
        lowpass_hz=75.0,
        notch_hz=50.0,
        maximum_interpolation_fraction=0.15,
        require_complete_canonical=True,
        native_montage_sensitivity=True,
        csd_sensitivity=True,
        csd_minimum_channels=15,
        csd_minimum_position_fraction=1.0,
        is_sleep_recording=False,
    )
    assert results["native_full_montage_average_reference"].status == "available"
    csd = results["native_full_montage_csd"]
    assert csd.status == "unavailable"
    assert csd.raw is None
    assert csd.reason is not None and csd.reason.startswith("insufficient_montage_positions")
    assert results["sleep_highpass_sensitivity"].status == "not_applicable"


def test_real_mne_primary_native_csd_and_sleep_smoke_when_extra_is_installed() -> None:
    mne = pytest.importorskip("mne")
    canonical = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "O1",
        "O2",
    ]
    sampling_hz = 200.0
    times = np.arange(int(60 * sampling_hz)) / sampling_hz
    rng = np.random.default_rng(91)
    data = np.stack(
        [
            1e-6 * np.sin(2 * np.pi * (7.0 + index / 19.0) * times + index / 7.0)
            + rng.normal(scale=0.05e-6, size=len(times))
            for index in range(len(canonical))
        ]
    )
    raw = mne.io.RawArray(
        data,
        mne.create_info(canonical, sampling_hz, ch_types="eeg"),
        verbose="ERROR",
    )
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
    primary, provenance = preprocess_mne_raw(
        raw,
        canonical_channels=canonical,
        target_sampling_hz=sampling_hz,
        highpass_hz=0.1,
        lowpass_hz=75.0,
        notch_hz=50.0,
        maximum_interpolation_fraction=0.15,
        require_complete_canonical=True,
    )
    assert primary.ch_names == canonical
    assert provenance["canonical_montage_complete"] is True
    results = preprocess_mne_sensitivity_branches(
        raw,
        canonical_channels=canonical,
        target_sampling_hz=sampling_hz,
        primary_highpass_hz=0.1,
        sleep_highpass_hz=0.3,
        lowpass_hz=75.0,
        notch_hz=50.0,
        maximum_interpolation_fraction=0.15,
        require_complete_canonical=True,
        native_montage_sensitivity=True,
        csd_sensitivity=True,
        csd_minimum_channels=15,
        csd_minimum_position_fraction=1.0,
        is_sleep_recording=True,
    )
    assert results["native_full_montage_average_reference"].status == "available"
    assert results["native_full_montage_csd"].status == "available"
    assert results["sleep_highpass_sensitivity"].status == "available"
    with pytest.raises(ValueError, match="requires a reason"):
        SensitivityBranchResult(
            raw=None,
            status="unavailable",
            reason=None,
            metadata={},
        )


def test_tms_interpolation_removes_pulse() -> None:
    times = np.linspace(-0.1, 0.1, 401)
    base = np.sin(2 * np.pi * 10 * times)
    epochs = np.tile(base, (3, 2, 1))
    epochs[:, :, (times >= -0.005) & (times <= 0.015)] = 100.0
    clean = interpolate_pulse_interval(epochs, times)
    assert np.max(np.abs(clean)) < 2.0


def test_continuous_tms_interpolation_uses_clean_bilateral_support() -> None:
    sampling_hz = 1000.0
    times = np.arange(2000) / sampling_hz
    base = np.sin(2 * np.pi * 10 * times)
    data = np.stack([base, 0.5 * base])
    corrupted = data.copy()
    corrupted[:, 995:1016] = 100.0
    clean = interpolate_continuous_pulses(corrupted, np.array([1000]), sampling_hz)
    assert np.max(np.abs(clean[:, 995:1016])) < 2.0
    np.testing.assert_array_equal(clean[:, :975], corrupted[:, :975])
    np.testing.assert_array_equal(clean[:, 1036:], corrupted[:, 1036:])


def test_continuous_tms_interpolation_rejects_overlapping_pulses() -> None:
    with pytest.raises(ValueError, match="overlap"):
        interpolate_continuous_pulses(np.zeros((2, 2000)), np.array([1000, 1005]), 1000.0)
