from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from neural_manifolds.config import load_study
from neural_manifolds.stages import fmri_driver


def test_fmri_driver_composes_pinned_manifest_encoder_and_analysis(
    tmp_path: Path, monkeypatch: object
) -> None:
    models = tmp_path / "models.yaml"
    models.write_text(
        """schema_version: 1
models:
  brainlm:
    trainable: false
    checkpoint_variant: pinned
""",
        encoding="utf-8",
    )
    release = tmp_path / "release"
    release.mkdir()
    atlas = tmp_path / "atlas.nii.gz"
    coordinates = tmp_path / "coordinates.csv"
    manifest = tmp_path / "prepared.parquet"
    manifest_audit = tmp_path / "manifest-audit.json"
    for path in (atlas, coordinates, manifest, manifest_audit):
        path.write_text(path.name, encoding="utf-8")
    analysis = tuple(tmp_path / f"analysis-{index}.bin" for index in range(7))
    for path in analysis:
        path.write_text(path.name, encoding="utf-8")
    calls: dict[str, object] = {}

    def fake_prepare(**kwargs: object) -> SimpleNamespace:
        calls["prepare"] = kwargs
        return SimpleNamespace(
            manifest_path=manifest,
            audit_path=manifest_audit,
            atlas_path=atlas,
            coordinates_path=coordinates,
        )

    class FakeEncoder:
        @classmethod
        def from_environment(cls, spec: object, **kwargs: object) -> object:
            calls["encoder"] = (spec, kwargs)
            return cls()

    def fake_run(**kwargs: object) -> tuple[Path, ...]:
        calls["analysis"] = kwargs
        return analysis

    monkeypatch.setattr(fmri_driver, "prepare_ds006623_fmri_manifest", fake_prepare)  # type: ignore[attr-defined]
    monkeypatch.setattr(fmri_driver, "OfficialBrainLMEncoder", FakeEncoder)  # type: ignore[attr-defined]
    monkeypatch.setattr(fmri_driver, "run_fmri_triangulation", fake_run)  # type: ignore[attr-defined]

    result = fmri_driver.run_ds006623_brainlm_stage(
        release_root=release,
        models_path=models,
        output_root=tmp_path / "output",
        study=load_study(Path(__file__).parents[1] / "configs" / "study.yaml"),
        timing_index_origin=1,
        atlas_path=atlas,
        coordinates_path=coordinates,
        batch_size=2,
    )

    assert result == (manifest, manifest_audit, *analysis)
    assert calls["prepare"]["timing_index_origin"] == 1  # type: ignore[index]
    assert calls["encoder"][1]["batch_size"] == 2  # type: ignore[index]
    assert calls["analysis"]["manifest_path"] == manifest  # type: ignore[index]
