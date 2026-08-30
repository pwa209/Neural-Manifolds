from pathlib import Path

from neural_manifolds.config import config_sha256, load_study

ROOT = Path(__file__).resolve().parents[1]


def test_study_config_loads_and_is_stable() -> None:
    config = load_study(ROOT / "configs" / "study.yaml")
    assert config.status == "exploratory_non_preregistered"
    assert config.scientific_gates is False
    assert config.statistics.inference_unit == "participant"
    assert len(config_sha256(config)) == 64
