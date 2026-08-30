"""Synthetic fixtures for each audited native metadata mapping."""

from __future__ import annotations

import pandas as pd
import pytest

from neural_manifolds.adapters import (
    CogitateMEEGAdapter,
    DreamSerialAwakeningsAdapter,
    FigshareDoCRestingAdapter,
    MendeleyDoCPSGAdapter,
    PropofolFMRIAdapter,
    PropofolTMSEEGAdapter,
    PsiConnectAdapter,
    SchemaError,
    SomatosensoryReportTaskAdapter,
    TactileDetectionAdapter,
    UnresolvedMetadataError,
)


def test_ds005620_maps_native_id_and_blocks_missing_dream_labels() -> None:
    participants = pd.DataFrame(
        [
            {
                "participant_id": "sub-SD_1010",
                "age": 31,
                "sex": "M",
                "awakenings": 3,
                "TMS": "True",
                "tms_count": 2,
                "excluded": "False",
                "bad_after_preprocessing": "False",
            }
        ]
    )
    files = [
        "sub-1010/eeg/sub-1010_task-awake_acq-EC_eeg.vhdr",
        "sub-1010/eeg/sub-1010_task-sed_acq-tms_run-1_eeg.vhdr",
        "sub-1010/eeg/sub-1010_task-sed2_acq-rest_run-2_eeg.vhdr",
    ]
    units = PropofolTMSEEGAdapter().adapt(participants, files)
    assert {unit.condition for unit in units} == {
        "awake",
        "propofol_sedation",
        "pre_awakening_propofol",
    }
    assert {unit.participant_id for unit in units} == {"sub-1010"}
    sed2 = next(unit for unit in units if unit.condition == "pre_awakening_propofol")
    assert sed2.metadata_status == "unresolved"
    assert sed2.variables["dream_report"] is None
    with pytest.raises(UnresolvedMetadataError, match=r"no audited.*dream-report"):
        PropofolTMSEEGAdapter.require_dream_reports()


def _dream_row(*, case: str, experience: int) -> dict[str, object]:
    return {
        "Filename": f"s01_ep{case.split('_')[1]}.edf",
        "Case ID": case,
        "Subject ID": 1,
        "Experience": experience,
        "Treatment group": 0,
        "Duration": 300.0,
        "EEG sample rate": 256,
        "Number of EEG channels": 64,
        "Last sleep stage": 2,
        "Has EOG": 1,
        "Has EMG": 1,
        "Has ECG": 1,
        "Proportion artifacts": 0.05,
        "Time of awakening": "02:10:00",
        "Subject age": 24,
        "Subject sex": "F",
        "Subject healthy": 1,
        "Has more data": 1,
        "Remarks": "",
    }


def test_dream_experience_codes_create_episode_intervals() -> None:
    records = pd.DataFrame(
        [_dream_row(case="1_1", experience=2), _dream_row(case="1_2", experience=0)]
    )
    units = DreamSerialAwakeningsAdapter().adapt(records)
    assert [unit.condition for unit in units] == [
        "dream_experience_with_recall",
        "no_dream_experience",
    ]
    assert units[0].selector.kind == "interval_seconds"
    assert units[0].selector.stop_seconds == 300.0
    assert units[0].content == "recalled_experience"
    assert units[1].report_produced is False


def test_dream_unknown_experience_is_explicitly_unresolved() -> None:
    unit = DreamSerialAwakeningsAdapter().adapt(
        pd.DataFrame([_dream_row(case="1_1", experience=-4)])
    )[0]
    assert unit.metadata_status == "unresolved"
    assert unit.content is None


def _tactile_event(
    marker: str, *, onset: float, stimon: float | None = None, confidence: float | None = None
) -> dict[str, object]:
    return {
        "onset": onset,
        "duration": 0.0,
        "trial_type": marker,
        "event_value": 1,
        "event_sample": int(onset * 1000),
        "response_time": "n/a",
        "response_time2": "n/a",
        "stimamp": 1.25 if marker == "stim-adapt" else "n/a",
        "stimon": 0.05 if stimon is None else stimon,
        "sdt": "h",
        "confidence": 0.8 if confidence is None else confidence,
    }


def test_ds001785_builds_trial_level_event_selectors() -> None:
    participants = pd.DataFrame([{"participant_id": "sub-001", "age": 22, "sex": "F"}])
    path = "sub-001/ses-01/eeg/sub-001_ses-01_task-adapt_run-1_events.tsv"
    events = pd.DataFrame(
        [
            _tactile_event("stim-adapt", onset=1.0, stimon=0.05),
            _tactile_event("hit", onset=1.4, confidence=0.91),
            _tactile_event("conf", onset=1.5),
            _tactile_event("conf-resp", onset=1.6),
        ]
    )
    unit = TactileDetectionAdapter().adapt(participants, {path: events})[0]
    assert unit.condition == "tactile_detected"
    assert unit.selector.event_onset_seconds == pytest.approx(1.05)
    assert unit.selector.epoch_start_offset_seconds == -0.4
    assert unit.variables["confidence"] == pytest.approx(0.91)
    assert unit.source_file.endswith("_eeg.vhdr")


def test_native_tables_reject_unverified_columns() -> None:
    participants = pd.DataFrame(
        [{"participant_id": "sub-001", "age": 22, "sex": "F", "group": "case"}]
    )
    with pytest.raises(SchemaError, match="columns"):
        TactileDetectionAdapter().adapt(participants, {})


def test_osf_condition_parser_and_download_time_signal_assertion() -> None:
    adapter = SomatosensoryReportTaskAdapter()
    relevant = adapter.build_unit(
        participant_id="sub-01",
        source_file="sub-01/R/01_ST_CT/eeg_signal.mat",
        trial_index=3,
        signal_matrix_verified=True,
    )
    assert relevant.condition == "report_task_relevant"
    assert relevant.task_relevance == "relevant"
    assert relevant.selector.kind == "pre_epoched"
    assert relevant.selector.trial_index == 3
    no_report = adapter.build_unit(
        participant_id="sub-01",
        source_file="sub-01/NR/eeg_ST.mat",
        trial_index=0,
        signal_matrix_verified=True,
    )
    assert no_report.condition == "no_report"
    assert no_report.report_produced is False
    with pytest.raises(UnresolvedMetadataError, match="must be verified"):
        adapter.build_unit(
            participant_id="sub-01",
            source_file="sub-01/R/01_ST_CV/eeg_signal.mat",
            trial_index=0,
            signal_matrix_verified=False,
        )


def test_cogitate_blocks_paper_inferred_event_columns() -> None:
    with pytest.raises(UnresolvedMetadataError, match="account-gated"):
        CogitateMEEGAdapter.adapt(pd.DataFrame())


def test_psiconnect_maps_session_exposure_without_context_guessing() -> None:
    participants = pd.DataFrame(
        [{"participant_id": "sub-PC001", "age": 28, "dose_mg_per_kg": 0.32}]
    )
    files = [
        "sub-PC001/ses-01/eeg/sub-PC001_ses-01_task-series_eeg.vhdr",
        "sub-PC001/ses-02/eeg/sub-PC001_ses-02_task-series_eeg.vhdr",
    ]
    units = PsiConnectAdapter().adapt(participants, files)
    by_session = {unit.session_id: unit for unit in units}
    assert by_session["ses-01"].condition == "baseline_series"
    assert by_session["ses-01"].variables["dose_mg_per_kg"] == 0.0
    assert by_session["ses-02"].condition == "psilocybin_series"
    assert by_session["ses-02"].variables["dose_mg_per_kg"] == pytest.approx(0.32)
    with pytest.raises(UnresolvedMetadataError, match="start/end semantics"):
        PsiConnectAdapter.require_context_segments()


def test_figshare_doc_inventory_requires_triplets_and_withholds_labels() -> None:
    units = FigshareDoCRestingAdapter().adapt(["abc.dat", "abc.vhdr", "abc.vmrk"])
    assert len(units) == 1
    assert units[0].healthy_wake_reference is None
    assert units[0].clinical_holdout is True
    assert units[0].metadata_status == "unresolved"
    with pytest.raises(SchemaError, match="incomplete BrainVision triplet"):
        FigshareDoCRestingAdapter().adapt(["abc.dat", "abc.vhdr"])
    with pytest.raises(UnresolvedMetadataError, match="no audited stem-to"):
        FigshareDoCRestingAdapter.require_clinical_labels()


def test_mendeley_doc_filename_diagnosis_is_holdout_only() -> None:
    unit = MendeleyDoCPSGAdapter().adapt(["Patient_2_MCS+.edf"])[0]
    assert unit.participant_id == "patient-002"
    assert unit.condition == "minimally_conscious_plus"
    assert unit.clinical_holdout is True
    assert unit.variables["crs_r"] is None


def test_ds006623_segments_induction_at_lor() -> None:
    units = PropofolFMRIAdapter().adapt_run(
        participant_id="sub-01",
        task="imagery",
        run=2,
        source_file="sub-01/func/sub-01_task-imagery_run-2_bold.nii.gz",
        effect_site_concentration=[0.0, 0.1, 0.2, 0.5, 0.8, 1.0],
        lor_volume=3,
    )
    assert [unit.condition for unit in units] == [
        "responsive_induction",
        "behaviorally_unresponsive",
    ]
    assert [(unit.selector.volume_start, unit.selector.volume_stop) for unit in units] == [
        (0, 3),
        (3, 6),
    ]


def test_ds006623_missing_ror_remains_unresolved_and_zero_esc_is_wake() -> None:
    unresolved = PropofolFMRIAdapter().adapt_run(
        participant_id="sub-01",
        task="imagery",
        run=3,
        source_file="sub-01/func/sub-01_task-imagery_run-3_bold.nii.gz",
        effect_site_concentration=[1.0, 0.8, 0.5],
    )[0]
    assert unresolved.metadata_status == "unresolved"
    assert unresolved.variables["behavioral_responsiveness"] is None
    wake = PropofolFMRIAdapter().adapt_run(
        participant_id="sub-01",
        task="rest",
        run=1,
        source_file="sub-01/func/sub-01_task-rest_run-1_bold.nii.gz",
        effect_site_concentration=[0.0, 0.0, 0.0],
    )[0]
    assert wake.condition == "no_propofol_rest"
    assert wake.healthy_wake_reference is True
