import copy

from neural_manifolds.revised.high_density_extension import admitted_cohort, extension_policy


def test_rome_extension_keeps_independence_and_rejects_ambiguity_and_quality_notes():
    policy, extension = extension_policy()
    rows, files = [], []
    for index, (report, remarks) in enumerate(
        [("2", ""), ("0", ""), ("-2", ""), ("2", "derivations with noise/artifacts: F3")]
    ):
        rows.append(
            {
                "Filename": f"{index}.edf",
                "records_origin": "archive.zip::Records.csv",
                "Case ID": str(index),
                "Duration": "60",
                "Experience": report,
                "Last sleep stage": "2",
                "Subject healthy": "1",
                "Subject ID": str(index),
                "Treatment group": "0",
                "Remarks": remarks,
            }
        )
        channels = ["01" if c == "O1" else c for c in policy["high_density_channels"]]
        files.append(
            {"member": f"Data/PSG/{index}.edf", "duration_seconds": 60, "channels": channels}
        )
    original = copy.deepcopy(rows)
    result = admitted_cohort(
        {
            "release": "/synthetic",
            "completion_marker_sha256": "test",
            "records": rows,
            "files": files,
        },
        policy,
        extension,
    )
    assert rows == original
    assert len(result["units"]) == 2
    assert {u["experience"] for u in result["units"]} == {0, 1}
    assert all(u["tracks"] == ["high_density"] for u in result["units"])
    assert {u["study_group"] for u in result["units"]} == {"rome_sapienza"}
    assert len(result["exclusions"]) == 2
    assert any("content_recall" in value for value in result["limitations"])


def test_original_cohort_policy_is_not_mutated_on_disk():
    from pathlib import Path

    import yaml

    policy, extension = extension_policy()
    original = yaml.safe_load(Path(extension["base_policy"]).read_text())
    assert "dream_set_09" not in original["source_policy"]
    assert policy["high_density_channels"] == original["high_density_channels"]
    assert len(policy["high_density_channels"]) == 14


def test_phase_receipt_rejects_changed_artifact_and_identity(tmp_path):
    import json

    import pytest

    from neural_manifolds.provenance import sha256_file
    from neural_manifolds.revised.high_density_extension import verified_phase

    folder = tmp_path / "transfer" / "outputs"
    folder.mkdir(parents=True)
    artifact = folder / "transfer.json"
    artifact.write_text("{}")
    identity = {"source_digest": "abc", "scope": "supplementary"}
    receipt = {
        **identity,
        "phase": "transfer",
        "replicate": 0,
        "status": "executed",
        "artifact": str(artifact),
        "sha256": sha256_file(artifact),
    }
    (folder / "execution.json").write_text(json.dumps(receipt))
    assert verified_phase(tmp_path, "transfer", identity) == artifact
    with pytest.raises(ValueError, match="identity_mismatch"):
        verified_phase(tmp_path, "transfer", {**identity, "source_digest": "other"})
    artifact.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="artifact_changed"):
        verified_phase(tmp_path, "transfer", identity)
