from __future__ import annotations

import pytest

from workflow.phases import PHASE_BY_NAME, PHASE_NAMES, select_phases


def test_phase_order_and_cli_mapping() -> None:
    assert PHASE_NAMES == (
        "audit",
        "acquire",
        "qc",
        "preprocess",
        "encode",
        "metrics",
        "models",
        "tms",
        "locked-clinical",
        "fmri",
        "figures",
    )
    assert PHASE_BY_NAME["locked-clinical"].cli_phase == "clinical"
    assert PHASE_BY_NAME["locked-clinical"].dependencies == ("tms",)
    assert PHASE_BY_NAME["fmri"].dependencies == ("locked-clinical",)
    assert set(PHASE_BY_NAME["figures"].dependencies) == {
        "models",
        "tms",
        "locked-clinical",
        "fmri",
    }


def test_phase_selection_is_contiguous() -> None:
    selected = select_phases(from_phase="encode", through_phase="tms")
    assert tuple(item.name for item in selected) == (
        "encode",
        "metrics",
        "models",
        "tms",
    )


def test_only_phase_cannot_be_combined_with_range() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        select_phases(only_phase="audit", through_phase="qc")


def test_no_phase_is_conditioned_on_scientific_results() -> None:
    # Dependencies are phase names only; the graph carries no score, P-value,
    # support, or journal gate.
    for phase in PHASE_BY_NAME.values():
        assert set(phase.dependencies) <= set(PHASE_NAMES)
        assert "result" not in phase.storage_class
        assert "support" not in phase.storage_class
