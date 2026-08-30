from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neural_manifolds.figures import FigureInputError, load_figure_config
from neural_manifolds.figures.io import load_source_bundle


def test_all_six_contracts_are_complete_and_python_only() -> None:
    config = load_figure_config()
    assert config.backend == "python-matplotlib"
    assert set(config.figures) == {f"figure_{number}" for number in range(1, 7)}
    assert config.export["pdf_fonttype"] == 42
    assert config.export["tiff_dpi"] == 600
    assert config.integrity["participant_points_required"] is True
    for contract in config.figures.values():
        assert contract.core_conclusion
        assert contract.panel_map
        assert contract.archetype in {
            "quantitative grid",
            "schematic-led composite",
            "asymmetric mixed-modality figure",
        }
        assert contract.reviewer_risks


def test_source_tables_require_upstream_artifact_hash(tmp_path: Path) -> None:
    path = tmp_path / "profiles.csv"
    pd.DataFrame({"participant_id": ["p1"], "R": [0.1]}).to_csv(path, index=False)
    with pytest.raises(FigureInputError, match="source_artifact_sha256"):
        load_source_bundle("profiles", path)
