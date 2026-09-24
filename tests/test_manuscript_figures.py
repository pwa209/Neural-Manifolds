"""Figure source contracts, using explicit synthetic test fixtures only."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "manuscript_figures", Path(__file__).parents[1] / "scripts/render_manuscript_figures.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def null_fixture():
    rows = [
        dict(
            portfolio="core",
            track="sparse",
            representation="sensor",
            kind="label_permutation",
            replicate=i,
            summary={"models": [{"model": "shared_dynamics", "log_loss": i / 100}]},
        )
        for i in range(100)
    ]
    audit = dict(
        portfolio="core",
        analysis="sparse:sensor",
        kind="label_permutation",
        metric="absolute_dynamic_log_loss",
        observed=-0.1,
        p_lower=1 / 101,
    )
    return {"controls": rows, "review": {"null_controls": [audit]}}


def test_null_values_preserve_all_actual_replicates():
    values, audit = m.null_values(
        null_fixture(), "core", "sensor", "label_permutation", "absolute_dynamic_log_loss"
    )
    np.testing.assert_allclose(values, np.arange(100) / 100)
    assert audit["p_lower"] == 1 / 101


@pytest.mark.parametrize("change", ["missing", "duplicate", "mismatched_p"])
def test_null_values_reject_incomplete_or_inconsistent_sources(change):
    data = null_fixture()
    if change == "missing":
        data["controls"].pop()
    elif change == "duplicate":
        data["controls"][-1]["replicate"] = 0
    else:
        data["review"]["null_controls"][0]["p_lower"] = 0.5
    with pytest.raises(ValueError):
        m.null_values(data, "core", "sensor", "label_permutation", "absolute_dynamic_log_loss")


def test_interval_retains_endpoints_even_if_estimate_outside():
    fig, ax = m.plt.subplots()
    m.interval(ax, 0, 3, [1, 2], "black")
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [1, 2])
    np.testing.assert_allclose(ax.lines[1].get_xdata(), [3])
    m.plt.close(fig)


def test_duplicate_and_nonfinite_rows_are_not_silently_plotted():
    with pytest.raises(ValueError):
        m.unique([{"feature": "x"}, {"feature": "x"}], feature="x")
    with pytest.raises(ValueError):
        m.finite([1, np.nan])


def test_scale_label_preserves_cognitive_impairment_meaning():
    book = {"ASC11_COGNITION": {"Description": "11D-ASC Impaired Cognition and Control Subscale"}}
    assert m.scale_label("ASC11_COGNITION", book) == "Impaired cognition/control"


def test_participant_pairing_is_checked_against_aggregate():
    measurements = []
    for person, effect in [("synthetic-a", 1.0), ("synthetic-b", 3.0)]:
        for context, session, value in [
            ("rest", "01", 0.0),
            ("rest", "02", 1.0),
            ("movie", "01", 2.0),
            ("movie", "02", 3.0 + effect),
        ]:
            measurements.append(
                dict(
                    participant_id=person,
                    feature="repertoire",
                    context=context,
                    session=session,
                    value=value,
                )
            )
    original = dict(
        feature="repertoire",
        context_minus_rest="movie",
        participants=2,
        difference_in_change=2.0,
        interval_95=[1.0, 3.0],
    )
    audit = dict(
        feature="repertoire",
        context="movie",
        kind="context_interaction",
        participants=2,
        estimate=2.0,
        p_holm_all_boundary_tests=1.0,
    )
    data = {
        "boundary": {"measurements": measurements, "context_interactions": [original]},
        "review": {"boundary": {"contrasts": [audit]}},
    }
    np.testing.assert_allclose(m.participant_interaction(data, "repertoire", "movie"), [1.0, 3.0])
    data["boundary"]["measurements"][0]["value"] = 100
    with pytest.raises(ValueError):
        m.participant_interaction(data, "repertoire", "movie")
