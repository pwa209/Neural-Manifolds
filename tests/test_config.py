from pathlib import Path

import pytest
from pydantic import ValidationError

from neural_manifolds.config import StudyConfig, config_sha256, load_study, load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_study_config_loads_and_is_stable() -> None:
    config = load_study(ROOT / "configs" / "study.yaml")
    assert config.status == "exploratory_non_preregistered"
    assert config.scientific_gates is False
    assert config.statistics.inference_unit == "participant"
    assert (
        config.representation.alignment_step_seconds
        == config.representation.alignment_window_seconds
    )
    assert min(config.metrics["alignment"]["lags_ms"]) >= 1000
    assert config.metrics["alignment"]["short_lag_status"].startswith("unavailable_")
    assert len(config.preprocessing.canonical_channels) == 19
    assert config.preprocessing.require_complete_harmonised_montage is True
    assert config.preprocessing.primary_reference == "average"
    assert config.preprocessing.csd_minimum_position_fraction == 1.0
    assert config.preprocessing.auxiliary_ica_policy == "report_support_not_performed"
    assert config.preprocessing.sleep_sensitivity_modalities == ["psg"]
    assert len(config_sha256(config)) == 64


def test_overlapping_latent_alignment_windows_are_rejected() -> None:
    payload = load_yaml(ROOT / "configs" / "study.yaml")
    payload["representation"]["alignment_step_seconds"] = 0.02
    with pytest.raises(ValidationError, match="non-overlapping windows"):
        StudyConfig.model_validate(payload)


def test_subwindow_latent_alignment_lags_are_rejected() -> None:
    payload = load_yaml(ROOT / "configs" / "study.yaml")
    payload["metrics"]["alignment"]["lags_ms"] = [200]
    with pytest.raises(ValidationError, match="at least one non-overlapping"):
        StudyConfig.model_validate(payload)


def test_preprocessing_sensitivity_contract_rejects_impossible_csd_channel_count() -> None:
    payload = load_yaml(ROOT / "configs" / "study.yaml")
    payload["preprocessing"]["csd_minimum_channels"] = 20
    with pytest.raises(ValidationError, match="exceeds canonical channel count"):
        StudyConfig.model_validate(payload)


def test_sleep_highpass_sensitivity_must_differ_in_the_configured_direction() -> None:
    payload = load_yaml(ROOT / "configs" / "study.yaml")
    payload["preprocessing"]["sleep_sensitivity_highpass_hz"] = 0.05
    with pytest.raises(ValidationError, match="above the primary high-pass"):
        StudyConfig.model_validate(payload)
