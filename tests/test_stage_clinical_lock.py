from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neural_manifolds.config import load_study
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stages.clinical import _validate_lock, run_clinical_transfer


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _locked_metrics_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    state_root = tmp_path / "state"
    artifact_root = tmp_path / "metrics"
    artifact_root.mkdir()
    dictionary = artifact_root / "state-dictionary.joblib"
    estimator = artifact_root / "profile-estimator.joblib"
    dictionary.write_bytes(b"locked-state-dictionary\n")
    estimator.write_bytes(b"locked-profile-estimator\n")
    artifacts = [dictionary, estimator]

    marker_path = state_root / "phases" / "metrics" / "success.json"
    atomic_write_json(
        marker_path,
        {
            "schema_version": 1,
            "phase": "metrics",
            "status": "succeeded",
            "phase_hash": "metrics-phase-hash",
            "artifacts": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
                for path in artifacts
            ],
        },
    )
    basis = {
        "schema_version": 1,
        "kind": "technical_clinical_transfer_snapshot",
        "project_status": "exploratory_non_preregistered",
        "scientific_gate": False,
        "source_manifest_sha256": "a" * 64,
        "config_fingerprints": {"study": {"sha256": "b" * 64, "size": 1}},
        "healthy_success_markers": {"metrics": sha256_file(marker_path)},
        "healthy_validated_artifacts": {
            "metrics": {str(path): sha256_file(path) for path in artifacts}
        },
    }
    lock_path = state_root / "clinical_lock.json"
    atomic_write_json(
        lock_path,
        {
            **basis,
            "basis_sha256": _sha256_json(basis),
            "created_at": "2026-08-30T00:00:00Z",
            "notice": "Technical provenance only; not a preregistration.",
        },
    )
    return lock_path, marker_path, dictionary, estimator


def test_validate_lock_rejects_artifact_changed_after_success_marker(
    tmp_path: Path,
) -> None:
    lock, _marker, dictionary, _estimator = _locked_metrics_artifacts(tmp_path)
    assert _validate_lock(lock)["basis_sha256"]

    dictionary.write_bytes(b"tampered-state-dictionary-with-different-content\n")

    with pytest.raises(ValueError, match="healthy artifact changed after lock"):
        _validate_lock(lock)


def test_validate_lock_rejects_changed_success_marker(tmp_path: Path) -> None:
    lock, marker, _dictionary, _estimator = _locked_metrics_artifacts(tmp_path)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["unexpected_post_lock_edit"] = True
    atomic_write_json(marker, payload)

    with pytest.raises(ValueError, match="healthy success marker changed after lock"):
        _validate_lock(lock)


@pytest.mark.parametrize("substituted_object", ["dictionary", "estimator"])
def test_clinical_transfer_rejects_fitted_artifact_path_not_bound_to_metrics_receipt(
    tmp_path: Path,
    substituted_object: str,
) -> None:
    lock, _marker, dictionary, estimator = _locked_metrics_artifacts(tmp_path)
    substitute = tmp_path / f"substitute-{substituted_object}.joblib"
    original = dictionary if substituted_object == "dictionary" else estimator
    # Even a byte-identical copy is rejected: the lock binds both artifact path
    # and digest, preventing post-lock path substitution.
    substitute.write_bytes(original.read_bytes())
    dictionary_argument = substitute if substituted_object == "dictionary" else dictionary
    estimator_argument = substitute if substituted_object == "estimator" else estimator

    with pytest.raises(
        ValueError,
        match="frozen clinical object is not bound to the metrics receipt",
    ):
        run_clinical_transfer(
            encoding_manifest=tmp_path / "not-reached.parquet",
            state_dictionary_path=dictionary_argument,
            profile_estimator_path=estimator_argument,
            clinical_lock_path=lock,
            output_root=tmp_path / "not-created",
            study=load_study(Path("configs/study.yaml")),
        )

    assert not (tmp_path / "not-created").exists()
