from pathlib import Path

from neural_manifolds.cli import validate_configs

ROOT = Path(__file__).resolve().parents[1]


def test_validate_configs_allows_unresolved_roots_locally() -> None:
    result = validate_configs(
        ROOT / "configs" / "study.yaml",
        ROOT / "configs" / "datasets.yaml",
        ROOT / "configs" / "server.yaml",
    )
    assert result["valid"] is True
    assert result["scientific_gates"] is False
    assert result["unresolved_server_roots"]
