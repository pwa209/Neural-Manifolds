import importlib.util
from pathlib import Path

import numpy as np
import pytest

from neural_manifolds.manifold.metastability import estimate_metastability

spec = importlib.util.spec_from_file_location(
    "review", Path(__file__).parents[1] / "scripts/review_completed_evidence.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_holm_and_null_direction():
    assert review.holm([0.04, 0.001, 0.03]) == pytest.approx([0.06, 0.003, 0.06])
    assert review.empirical_tail(-3, [-2, -1, 0]) == 0.25
    assert review.empirical_tail(1, [-2, -1, 0]) == 1
    with pytest.raises(ValueError):
        review.empirical_tail(0, [])


def test_nonfinite_is_explicit_not_zero():
    paths = []
    assert review.json_finite({"x": float("nan")}, missing=paths) == {"x": None}
    assert paths == ["/x"]


def test_precision_equal_studies_not_trials():
    rows = []
    for group, n in [("a", 2), ("b", 20), ("c", 3)]:
        for i in range(n):
            for model, prob in [("shared_dynamics", 0.8), ("base", 0.5)]:
                rows.append(
                    dict(
                        study_group=group,
                        participant_id="p",
                        unit_id=str(i),
                        model=model,
                        probability=prob,
                        experience=1,
                    )
                )
    r = review.precision(rows, "base")
    assert r["participants"] == 3
    assert r["independent_studies"] == 3
    assert r["mean_delta"] == pytest.approx(-0.4700036292)


def test_boundary_includes_all_contrasts():
    rows = []
    for person in range(5):
        for context in ["rest", "music"]:
            for session in ["01", "02"]:
                rows.append(
                    dict(
                        participant_id=str(person),
                        context=context,
                        session=session,
                        feature="x",
                        value=int(session) * (1 if context == "rest" else 2),
                    )
                )
    r = review.boundary_audit(rows, draws=99)
    assert r["family_size"] == 3
    assert next(x for x in r["contrasts"] if x["kind"] == "context_interaction")["estimate"] == 1


def test_recurrence_depends_on_segmentation():
    states = np.tile([0, 1, 0], 100)
    continuous = estimate_metastability(states).recurrence_probability
    blocks = estimate_metastability(
        states, segment_ids=np.repeat(np.arange(100), 3)
    ).recurrence_probability
    assert continuous > 0.98
    assert blocks == 0.5


def test_constant_reference_never_uses_target_labels():
    rows = []
    for group, labels in [("a", [0, 0]), ("b", [1, 1]), ("c", [0, 1])]:
        for i, label in enumerate(labels):
            rows.append(
                dict(
                    study_group=group,
                    participant_id=group,
                    unit_id=str(i),
                    model="shared_dynamics",
                    experience=label,
                    probability=0.5,
                )
            )
    result = review.prediction_diagnostics(rows)
    probs = {
        r["study_group"]: r["training_only_constant_probability"]
        for r in result["training_constant"]
    }
    assert probs == {"a": 0.75, "b": 0.25, "c": 0.5}
    assert result["participants_with_both_labels"] == 1


def test_public_scale_mapping_uses_dictionary_and_ids(monkeypatch, tmp_path):
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from review_subjective_mapping import read_public_scores

    columns = [f"ASC11_S{i}" for i in range(11)]
    book = {column: {"Description": "synthetic test scale"} for column in columns}
    tsv = (
        "\t".join(["participant_id", *columns])
        + "\n"
        + "\t".join(["sub-PC001", *["1"] * 11])
        + "\n"
    )

    def public_blob(command):
        file = command[-1]
        if file.endswith(".json"):
            return json.dumps(book).encode()
        if file.endswith(".tsv"):
            return tsv.encode()
        return b"Synthetic codebook README"

    monkeypatch.setattr("review_subjective_mapping.subprocess.check_output", public_blob)
    frame, scales, _, sources = read_public_scores(tmp_path, tmp_path)
    assert frame.participant_id.tolist() == ["PC001"]
    assert scales == columns
    assert len(sources) == 3
