from __future__ import annotations

import json

import pytest

from neural_manifolds.stage_units import (
    CLINICAL_LOW_CHANNEL_BRANCH,
    CLINICAL_LOW_CHANNEL_MIN_CHANNELS,
    _clinical_property_scope,
    _spatial_regions,
)
from neural_manifolds.stages.clinical import _property_scope_from_metadata


def test_low_channel_psg_contract_is_explicitly_limited_not_imputed() -> None:
    regions = _spatial_regions(["F3", "F4", "C3", "C4", "O1", "O2"])
    assert regions == ["central", "frontal", "occipital"]
    scope = _clinical_property_scope(["F3", "F4", "C3", "C4", "O1", "O2"])
    assert set(scope) == {
        "repertoire",
        "metastability",
        "directionality",
        "alignment",
        "reachability",
    }
    assert all(
        "limited" in scope[axis] for axis in ("repertoire", "metastability", "directionality")
    )
    assert scope["alignment"] == "unavailable_requires_three_channels_in_each_of_two_modules"
    assert scope["reachability"].startswith("available_secondary_passive_only_limited")
    assert CLINICAL_LOW_CHANNEL_BRANCH == "clinical_low_channel_psg"
    assert CLINICAL_LOW_CHANNEL_MIN_CHANNELS == 2


def test_low_channel_alignment_requires_three_channels_per_module() -> None:
    unavailable = _clinical_property_scope(["F3", "F4", "C3", "C4"])
    assert unavailable["alignment"] == "unavailable_requires_three_channels_in_each_of_two_modules"
    available = _clinical_property_scope(["F3", "F4", "F7", "C3", "C4", "Cz"])
    assert available["alignment"].startswith("available_secondary_limited")


def test_low_channel_metric_scope_fails_closed_and_preserves_unavailable_axes() -> None:
    with pytest.raises(ValueError, match="lacks explicit property-scope"):
        _property_scope_from_metadata({"analysis_branch": CLINICAL_LOW_CHANNEL_BRANCH})
    declared = _clinical_property_scope(["F3", "F4", "C3", "C4", "O1", "O2"])
    observed = _property_scope_from_metadata(
        {
            "analysis_branch": CLINICAL_LOW_CHANNEL_BRANCH,
            "property_scope_json": json.dumps(declared),
        }
    )
    assert observed == declared
    assert observed["alignment"].startswith("unavailable")
