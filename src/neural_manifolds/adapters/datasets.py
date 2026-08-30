"""Audited dataset-specific mappings to standardized analysis units.

The adapters deliberately stop at metadata boundaries that the immutable public
release does not document.  Callers must never infer those labels from filename
order, participant order, signal features, or downstream outcomes.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar

import numpy as np
import pandas as pd

from .models import (
    AnalysisUnit,
    SchemaError,
    SignalSelector,
    UnresolvedMetadataError,
    make_unit_id,
)
from .strict import (
    boolean,
    integer,
    match,
    normalize_relative_path,
    number,
    require_exact_columns,
    require_values,
    text,
)

DS005620_PARTICIPANT_COLUMNS = [
    "participant_id",
    "age",
    "sex",
    "awakenings",
    "TMS",
    "tms_count",
    "excluded",
    "bad_after_preprocessing",
]
DS005620_RECORDING = re.compile(
    r"(?P<participant>sub-\d+)/eeg/(?P=participant)_task-(?P<task>awake|sed|sed2)_"
    r"acq-(?P<acquisition>EC|EO|rest|tms)(?:_run-(?P<run>\d+))?_eeg\.vhdr"
)


class PropofolTMSEEGAdapter:
    """OpenNeuro ds005620 v1.0.0 recording and participant mapping."""

    dataset_id = "propofol_tms_eeg"

    def adapt(
        self, participants: pd.DataFrame, recording_files: Iterable[str]
    ) -> list[AnalysisUnit]:
        require_exact_columns(
            participants, DS005620_PARTICIPANT_COLUMNS, source="ds005620 participants.tsv"
        )
        participant_lookup: dict[str, pd.Series] = {}
        for _, row in participants.iterrows():
            native = text(row["participant_id"], field="participant_id")
            assert native is not None
            native_match = re.fullmatch(r"sub-SD_(\d+)", native)
            if native_match is None:
                raise SchemaError(f"undocumented ds005620 participant_id: {native!r}")
            standardized = f"sub-{native_match.group(1)}"
            if standardized in participant_lookup:
                raise SchemaError(
                    f"duplicate participant after ds005620 ID mapping: {standardized}"
                )
            boolean(row["excluded"], field=f"{native}.excluded")
            boolean(row["TMS"], field=f"{native}.TMS")
            integer(row["awakenings"], field=f"{native}.awakenings")
            integer(row["tms_count"], field=f"{native}.tms_count")
            participant_lookup[standardized] = row

        units: list[AnalysisUnit] = []
        conditions = {
            "awake": ("awake", True),
            "sed": ("propofol_sedation", False),
            "sed2": ("pre_awakening_propofol", False),
        }
        for raw_path in recording_files:
            path = normalize_relative_path(raw_path)
            parsed = match(DS005620_RECORDING, path, field="ds005620 recording path")
            participant = parsed.group("participant")
            if participant not in participant_lookup:
                raise SchemaError(
                    f"recording participant absent from participants.tsv: {participant}"
                )
            row = participant_lookup[participant]
            if boolean(row["excluded"], field=f"{participant}.excluded"):
                continue
            task = parsed.group("task")
            acquisition = parsed.group("acquisition")
            run = parsed.group("run")
            if task == "awake" and run is not None:
                raise SchemaError("ds005620 awake recordings must not carry a run entity")
            if task != "awake" and acquisition in {"EC", "EO"}:
                raise SchemaError("ds005620 EC/EO acquisitions are documented only for task-awake")
            if task == "sed2" and acquisition != "rest":
                raise SchemaError("ds005620 task-sed2 is documented only for acq-rest")
            condition, is_wake = conditions[task]
            units.append(
                AnalysisUnit(
                    dataset_id=self.dataset_id,
                    unit_id=make_unit_id(self.dataset_id, path),
                    participant_id=participant,
                    run_id=run,
                    source_file=path,
                    modality="tms-eeg" if acquisition == "tms" else "eeg",
                    selector=SignalSelector(kind="full_recording"),
                    condition=condition,
                    explanatory_target="conscious_level",
                    healthy_wake_reference=is_wake,
                    clinical_holdout=False,
                    task_relevance="not_applicable",
                    metadata_status="unresolved" if task == "sed2" else "verified",
                    variables={
                        "native_participant_id": str(row["participant_id"]),
                        "task": task,
                        "acquisition": acquisition,
                        "dream_report": None,
                    },
                )
            )
        return sorted(units, key=lambda unit: unit.unit_id)

    @staticmethod
    def require_dream_reports() -> None:
        raise UnresolvedMetadataError(
            "ds005620 v1.0.0 contains pre-awakening task-sed2 recordings but no audited "
            "participant/run dream-report table; experienced-content labels cannot be inferred"
        )


DREAM_COLUMNS = [
    "Filename",
    "Case ID",
    "Subject ID",
    "Experience",
    "Treatment group",
    "Duration",
    "EEG sample rate",
    "Number of EEG channels",
    "Last sleep stage",
    "Has EOG",
    "Has EMG",
    "Has ECG",
    "Proportion artifacts",
    "Time of awakening",
    "Subject age",
    "Subject sex",
    "Subject healthy",
    "Has more data",
    "Remarks",
]
DREAM_EXPERIENCE = {
    2: ("dream_experience_with_recall", True, "recalled_experience", True),
    1: ("dream_experience_without_recall", True, "experience_without_recall", True),
    0: ("no_dream_experience", False, "no_experience", True),
    -1: ("ambiguous_experience_with_or_without_recall", True, None, False),
    -2: ("ambiguous_no_experience_or_without_recall", None, None, False),
    -3: ("ambiguous_recalled_experience_or_none", None, None, False),
    -4: ("unknown_experience", None, None, False),
}
DREAM_FILE = re.compile(r"/?s(?P<subject>\d+)_ep(?P<episode>\d+)\.edf")


class DreamSerialAwakeningsAdapter:
    """DREAM/Tononi Records.csv mapping from Figshare article 23306054 v2."""

    dataset_id = "dream_tononi_serial_awakenings"

    def adapt(self, records: pd.DataFrame) -> list[AnalysisUnit]:
        require_exact_columns(records, DREAM_COLUMNS, source="DREAM Records.csv")
        units: list[AnalysisUnit] = []
        case_ids: set[str] = set()
        for _, row in records.iterrows():
            filename = text(row["Filename"], field="Filename")
            assert filename is not None
            file_match = match(DREAM_FILE, filename, field="DREAM Filename")
            subject_number = integer(row["Subject ID"], field="Subject ID", minimum=1)
            if int(file_match.group("subject")) != subject_number:
                raise SchemaError("DREAM Filename subject and Subject ID disagree")
            case_id = text(row["Case ID"], field="Case ID")
            assert case_id is not None
            expected_case = f"{subject_number}_{int(file_match.group('episode'))}"
            if case_id != expected_case:
                raise SchemaError(f"DREAM Case ID {case_id!r} does not match {expected_case!r}")
            if case_id in case_ids:
                raise SchemaError(f"duplicate DREAM Case ID: {case_id}")
            case_ids.add(case_id)
            experience = integer(row["Experience"], field="Experience", minimum=-4)
            if experience not in DREAM_EXPERIENCE:
                raise SchemaError(f"undocumented DREAM Experience code: {experience}")
            condition, report, content, contrast_eligible = DREAM_EXPERIENCE[experience]
            duration = number(row["Duration"], field="Duration", minimum=0.001)
            assert duration is not None
            sleep_stage = integer(row["Last sleep stage"], field="Last sleep stage")
            if sleep_stage not in {0, 1, 2, 3, 5}:
                raise SchemaError(f"undocumented DREAM sleep-stage code: {sleep_stage}")
            if integer(row["Subject healthy"], field="Subject healthy") != 1:
                raise SchemaError(
                    "Tononi Serial Awakenings release documents a healthy cohort only"
                )
            normalized_file = normalize_relative_path(filename.lstrip("/"))
            units.append(
                AnalysisUnit(
                    dataset_id=self.dataset_id,
                    unit_id=make_unit_id(self.dataset_id, case_id),
                    participant_id=f"sub-{subject_number:02d}",
                    run_id=case_id,
                    source_file=f"Data/PSG/{normalized_file}",
                    modality="psg",
                    selector=SignalSelector(
                        kind="interval_seconds", start_seconds=0.0, stop_seconds=duration
                    ),
                    condition=condition,
                    explanatory_target="experienced_content",
                    healthy_wake_reference=False,
                    clinical_holdout=False,
                    report_produced=report,
                    task_relevance="not_applicable",
                    content=content,
                    metadata_status="unresolved"
                    if experience == -4
                    else ("ambiguous" if not contrast_eligible else "verified"),
                    variables={
                        "experience_code": experience,
                        "contrast_eligible": contrast_eligible,
                        "last_sleep_stage_code": sleep_stage,
                        "sample_rate_hz": number(
                            row["EEG sample rate"], field="EEG sample rate", minimum=1
                        ),
                    },
                )
            )
        return units


TACTILE_PARTICIPANT_COLUMNS = ["participant_id", "age", "sex"]
TACTILE_EVENT_COLUMNS = [
    "onset",
    "duration",
    "trial_type",
    "event_value",
    "event_sample",
    "response_time",
    "response_time2",
    "stimamp",
    "stimon",
    "sdt",
    "confidence",
]
TACTILE_EVENT_PATH = re.compile(
    r"(?P<participant>sub-\d+)/(?P<session>ses-\d+)/eeg/"
    r"(?P=participant)_(?P=session)_task-adapt_run-(?P<run>\d+)_events\.tsv"
)


class TactileDetectionAdapter:
    """OpenNeuro ds001785 v1.1.1 adaptive-task trial mapping."""

    dataset_id = "tactile_detection"
    markers: ClassVar[set[str]] = {
        "stim-adapt",
        "hit",
        "miss",
        "cr",
        "fa",
        "conf",
        "conf-resp",
    }

    def adapt(
        self,
        participants: pd.DataFrame,
        events_by_file: Mapping[str, pd.DataFrame],
    ) -> list[AnalysisUnit]:
        require_exact_columns(
            participants, TACTILE_PARTICIPANT_COLUMNS, source="ds001785 participants.tsv"
        )
        participant_ids = {str(value).strip() for value in participants["participant_id"]}
        if len(participant_ids) != len(participants):
            raise SchemaError("duplicate ds001785 participant_id")
        units: list[AnalysisUnit] = []
        for raw_path, events in events_by_file.items():
            path = normalize_relative_path(raw_path)
            parsed = match(TACTILE_EVENT_PATH, path, field="ds001785 adaptive events path")
            participant = parsed.group("participant")
            if participant not in participant_ids:
                raise SchemaError(f"events participant absent from participants.tsv: {participant}")
            require_exact_columns(events, TACTILE_EVENT_COLUMNS, source=path)
            require_values(events["trial_type"], self.markers, field=f"{path}.trial_type")
            pending: pd.Series | None = None
            trial_index = 0
            for _, row in events.iterrows():
                marker = str(row["trial_type"]).strip()
                if marker == "stim-adapt":
                    if pending is not None:
                        raise SchemaError(f"{path} has a stimulus without a first-order response")
                    pending = row
                    continue
                if marker not in {"hit", "miss", "cr", "fa"}:
                    continue
                if pending is None:
                    raise SchemaError(f"{path} has a response without a preceding stimulus")
                stimulus_present = marker in {"hit", "miss"}
                detected = marker in {"hit", "fa"}
                condition = {
                    "hit": "tactile_detected",
                    "miss": "tactile_undetected",
                    "cr": "catch_correct_rejection",
                    "fa": "catch_false_alarm",
                }[marker]
                onset = number(pending["onset"], field="onset", minimum=0)
                stimon = number(pending["stimon"], field="stimon", minimum=0)
                confidence = number(row["confidence"], field="confidence", minimum=0, maximum=1)
                assert onset is not None and stimon is not None and confidence is not None
                source_file = path.removesuffix("_events.tsv") + "_eeg.vhdr"
                units.append(
                    AnalysisUnit(
                        dataset_id=self.dataset_id,
                        unit_id=make_unit_id(self.dataset_id, path, trial_index),
                        participant_id=participant,
                        session_id=parsed.group("session"),
                        run_id=parsed.group("run"),
                        source_file=source_file,
                        modality="eeg",
                        selector=SignalSelector(
                            kind="event_epoch",
                            event_onset_seconds=onset + stimon,
                            epoch_start_offset_seconds=-0.4,
                            epoch_stop_offset_seconds=0.8,
                        ),
                        condition=condition,
                        explanatory_target="experienced_content",
                        healthy_wake_reference=True,
                        clinical_holdout=False,
                        report_produced=True,
                        task_relevance="relevant",
                        content="detected" if detected else "not_detected",
                        variables={
                            "stimulus_present": stimulus_present,
                            "detected": detected,
                            "confidence": confidence,
                            "stimulus_amplitude": number(
                                pending["stimamp"], field="stimamp", minimum=0
                            ),
                            "first_order_response": marker,
                        },
                    )
                )
                trial_index += 1
                pending = None
            if pending is not None:
                raise SchemaError(f"{path} ends with an unmatched stimulus")
        return units


OSF_REPORT_FOLDER = re.compile(r"(?P<sequence>\d+)_(?P<intensity>ST|MT)_(?P<instruction>CV|CT)")


class SomatosensoryReportTaskAdapter:
    """Condition parser for OSF hqkym.

    The article verifies directory condition codes but does not name the signal
    matrix variable/file.  ``build_unit`` therefore requires a download-time QC
    assertion that the selected MAT file contains the channels x time x trials
    signal matrix, preventing electrode-coordinate MAT files from being encoded.
    """

    dataset_id = "somatosensory_report_task"

    @staticmethod
    def condition_from_path(source_file: str) -> tuple[str, str, str]:
        path = normalize_relative_path(source_file)
        parts = path.split("/")
        if "R" in parts:
            index = parts.index("R")
            if index + 1 >= len(parts):
                raise SchemaError("OSF R path is missing its sequence condition folder")
            parsed = match(OSF_REPORT_FOLDER, parts[index + 1], field="OSF R condition folder")
            instruction = parsed.group("instruction")
            condition = "report_task_relevant" if instruction == "CT" else "report_task_irrelevant"
            relevance = "relevant" if instruction == "CT" else "irrelevant"
            return condition, relevance, parsed.group("intensity")
        if "NR" in parts:
            tokens = {token for part in parts for token in re.split(r"[^A-Za-z0-9]+", part)}
            intensities = tokens.intersection({"ST", "MT"})
            if len(intensities) != 1:
                raise UnresolvedMetadataError(
                    "OSF NR signal filename/intensity token is not uniquely documented; inspect the "
                    "downloaded archive and preserve an audited inventory"
                )
            return "no_report", "not_applicable", intensities.pop()
        raise SchemaError("OSF signal path contains neither the documented NR nor R block")

    def build_unit(
        self,
        *,
        participant_id: str,
        source_file: str,
        trial_index: int,
        signal_matrix_verified: bool,
    ) -> AnalysisUnit:
        if not signal_matrix_verified:
            raise UnresolvedMetadataError(
                "OSF MAT signal variable/file must be verified after archive extraction"
            )
        condition, relevance, intensity = self.condition_from_path(source_file)
        return AnalysisUnit(
            dataset_id=self.dataset_id,
            unit_id=make_unit_id(self.dataset_id, source_file, trial_index),
            participant_id=participant_id,
            source_file=normalize_relative_path(source_file),
            modality="eeg",
            selector=SignalSelector(kind="pre_epoched", trial_index=trial_index),
            condition=condition,
            explanatory_target="report_task_relevance",
            healthy_wake_reference=True,
            clinical_holdout=False,
            report_produced=condition != "no_report",
            task_relevance=relevance,  # type: ignore[arg-type]
            content="suprathreshold_tactile",
            variables={"stimulation_intensity": intensity},
        )


class CogitateMEEGAdapter:
    """Intentional blocker until the account-gated BIDS event schema is acquired."""

    dataset_id = "cogitate_meeg"

    @staticmethod
    def adapt(*_: object, **__: object) -> list[AnalysisUnit]:
        raise UnresolvedMetadataError(
            "Cogitate BIDS event column names/levels are available only inside the account-gated "
            "bundle. Download it under the terms, snapshot *_events.json, then implement the exact "
            "column-level mapping; do not infer columns from the paper's factorial design"
        )


PSICONNECT_PARTICIPANT_COLUMNS = ["participant_id", "age", "dose_mg_per_kg"]
PSICONNECT_RECORDING = re.compile(
    r"(?P<participant>sub-PC\d+)/(?P<session>ses-0[12])/eeg/"
    r"(?P=participant)_(?P=session)_task-series_eeg\.vhdr"
)


class PsiConnectAdapter:
    """OpenNeuro ds006110 v1.2.1 pre/post-psilocybin EEG mapping."""

    dataset_id = "psiconnect"

    def adapt(
        self, participants: pd.DataFrame, recording_files: Iterable[str]
    ) -> list[AnalysisUnit]:
        require_exact_columns(
            participants, PSICONNECT_PARTICIPANT_COLUMNS, source="ds006110 participants.tsv"
        )
        lookup = {str(row["participant_id"]).strip(): row for _, row in participants.iterrows()}
        if len(lookup) != len(participants):
            raise SchemaError("duplicate PsiConnect participant_id")
        units: list[AnalysisUnit] = []
        for raw_path in recording_files:
            path = normalize_relative_path(raw_path)
            parsed = match(PSICONNECT_RECORDING, path, field="PsiConnect EEG path")
            participant = parsed.group("participant")
            if participant not in lookup:
                raise SchemaError(
                    f"PsiConnect recording participant absent from participants.tsv: {participant}"
                )
            session = parsed.group("session")
            dose = number(lookup[participant]["dose_mg_per_kg"], field="dose_mg_per_kg", minimum=0)
            units.append(
                AnalysisUnit(
                    dataset_id=self.dataset_id,
                    unit_id=make_unit_id(self.dataset_id, path),
                    participant_id=participant,
                    session_id=session,
                    source_file=path,
                    modality="eeg",
                    selector=SignalSelector(kind="full_recording"),
                    condition="baseline_series" if session == "ses-01" else "psilocybin_series",
                    explanatory_target="psychedelic_organisation",
                    healthy_wake_reference=session == "ses-01",
                    clinical_holdout=False,
                    task_relevance="not_applicable",
                    variables={
                        "psilocybin_exposure": session == "ses-02",
                        "dose_mg_per_kg": 0.0 if session == "ses-01" else dose,
                        "context_segments": None,
                    },
                )
            )
        return units

    @staticmethod
    def require_context_segments() -> None:
        raise UnresolvedMetadataError(
            "PsiConnect task-series event markers contain repeated near-adjacent boundary codes; "
            "their start/end semantics are not described in task-series_eeg.json and must be "
            "resolved from the official acquisition documentation before context cropping"
        )


class FigshareDoCRestingAdapter:
    """Signal inventory for Figshare 23552964 v4, with labels intentionally unresolved."""

    dataset_id = "doc_resting_eeg"

    def adapt(self, files: Iterable[str]) -> list[AnalysisUnit]:
        grouped: dict[str, set[str]] = defaultdict(set)
        path_by_stem: dict[str, str] = {}
        for raw_path in files:
            path = normalize_relative_path(raw_path)
            parsed = re.fullmatch(r"(?P<stem>[A-Za-z0-9]+)\.(?P<extension>dat|vhdr|vmrk)", path)
            if parsed is None:
                raise SchemaError(f"undocumented Figshare DoC filename: {path!r}")
            stem = parsed.group("stem")
            grouped[stem].add(parsed.group("extension"))
            path_by_stem[stem] = f"{stem}.vhdr"
        units: list[AnalysisUnit] = []
        for stem, extensions in sorted(grouped.items()):
            if extensions != {"dat", "vhdr", "vmrk"}:
                raise SchemaError(
                    f"incomplete BrainVision triplet for {stem}: {sorted(extensions)}"
                )
            units.append(
                AnalysisUnit(
                    dataset_id=self.dataset_id,
                    unit_id=make_unit_id(self.dataset_id, stem),
                    participant_id=f"opaque-{stem}",
                    source_file=path_by_stem[stem],
                    modality="eeg",
                    selector=SignalSelector(kind="full_recording"),
                    condition="unresolved_clinical_group",
                    explanatory_target="clinical_status",
                    healthy_wake_reference=None,
                    clinical_holdout=True,
                    task_relevance="not_applicable",
                    metadata_status="unresolved",
                    variables={"clinical_group": None, "crs_r": None},
                )
            )
        return units

    @staticmethod
    def require_clinical_labels() -> None:
        raise UnresolvedMetadataError(
            "Figshare 23552964 v4 publishes opaque BrainVision stems and cohort counts but no "
            "audited stem-to-healthy/DoC/diagnosis/CRS-R key; clinical labels cannot be inferred"
        )


MENDELEY_DOC_FILE = re.compile(r"Patient_(?P<participant>\d+)_(?P<diagnosis>VS|MCS\+|MCS-)\.edf")


class MendeleyDoCPSGAdapter:
    """Mendeley 6wx4n25h4v v1 diagnosis mapping documented in filenames."""

    dataset_id = "doc_polysomnography"
    diagnosis: ClassVar[dict[str, str]] = {
        "VS": "vegetative_state",
        "MCS-": "minimally_conscious_minus",
        "MCS+": "minimally_conscious_plus",
    }

    def adapt(self, files: Iterable[str]) -> list[AnalysisUnit]:
        units: list[AnalysisUnit] = []
        seen: set[int] = set()
        for raw_path in files:
            path = normalize_relative_path(raw_path)
            parsed = match(MENDELEY_DOC_FILE, path, field="Mendeley DoC PSG filename")
            participant = int(parsed.group("participant"))
            if participant in seen:
                raise SchemaError(f"duplicate Mendeley DoC patient number: {participant}")
            seen.add(participant)
            native = parsed.group("diagnosis")
            units.append(
                AnalysisUnit(
                    dataset_id=self.dataset_id,
                    unit_id=make_unit_id(self.dataset_id, participant),
                    participant_id=f"patient-{participant:03d}",
                    source_file=path,
                    modality="psg",
                    selector=SignalSelector(kind="full_recording"),
                    condition=self.diagnosis[native],
                    explanatory_target="clinical_status",
                    healthy_wake_reference=False,
                    clinical_holdout=True,
                    task_relevance="not_applicable",
                    variables={"diagnosis_native": native, "crs_r": None},
                )
            )
        return units


class PropofolFMRIAdapter:
    """OpenNeuro ds006623 v1.0.0 run/ESC/LOR/ROR mapping."""

    dataset_id = "propofol_fmri"

    def adapt_run(
        self,
        *,
        participant_id: str,
        task: str,
        run: int,
        source_file: str,
        effect_site_concentration: Sequence[float],
        lor_volume: int | None = None,
        ror_volume: int | None = None,
    ) -> list[AnalysisUnit]:
        if task not in {"rest", "imagery"}:
            raise SchemaError(f"undocumented ds006623 task: {task!r}")
        allowed_runs = {"rest": {1, 2}, "imagery": {1, 2, 3, 4}}
        if run not in allowed_runs[task]:
            raise SchemaError(f"undocumented ds006623 {task} run: {run}")
        path = normalize_relative_path(source_file)
        expected = re.compile(
            rf"{re.escape(participant_id)}/func/{re.escape(participant_id)}_task-{task}_run-{run}_bold\.nii\.gz"
        )
        match(expected, path, field="ds006623 BOLD path")
        concentration = np.asarray(effect_site_concentration, dtype=float)
        if (
            concentration.ndim != 1
            or concentration.size == 0
            or not np.all(np.isfinite(concentration))
        ):
            raise SchemaError("effect-site concentration must be a non-empty finite 1D vector")
        if np.any(concentration < 0):
            raise SchemaError("effect-site concentration cannot be negative")
        n_volumes = int(concentration.size)

        segments: list[tuple[int, int, str, str | None, str]]
        if task == "imagery" and run == 2:
            if lor_volume is None or not 0 < lor_volume < n_volumes:
                raise SchemaError("imagery run 2 requires an in-range LOR volume")
            segments = [
                (0, lor_volume, "responsive_induction", "responsive", "verified"),
                (lor_volume, n_volumes, "behaviorally_unresponsive", "unresponsive", "verified"),
            ]
        elif task == "imagery" and run == 3:
            if ror_volume is None:
                segments = [(0, n_volumes, "recovery_status_unresolved", None, "unresolved")]
            elif not 0 < ror_volume < n_volumes:
                raise SchemaError("ROR volume must lie strictly within imagery run 3")
            else:
                segments = [
                    (0, ror_volume, "behaviorally_unresponsive", "unresponsive", "verified"),
                    (ror_volume, n_volumes, "responsive_recovery", "responsive", "verified"),
                ]
        elif float(np.max(concentration)) == 0.0:
            segments = [(0, n_volumes, f"no_propofol_{task}", "responsive", "verified")]
        else:
            segments = [(0, n_volumes, f"propofol_{task}_run_{run}", None, "ambiguous")]

        units: list[AnalysisUnit] = []
        for start, stop, condition, responsiveness, status in segments:
            segment = concentration[start:stop]
            no_propofol = float(np.max(segment)) == 0.0
            units.append(
                AnalysisUnit(
                    dataset_id=self.dataset_id,
                    unit_id=make_unit_id(self.dataset_id, participant_id, task, run, start, stop),
                    participant_id=participant_id,
                    run_id=str(run),
                    source_file=path,
                    modality="fmri",
                    selector=SignalSelector(
                        kind="volume_interval", volume_start=start, volume_stop=stop
                    ),
                    condition=condition,
                    explanatory_target="conscious_level",
                    healthy_wake_reference=no_propofol,
                    clinical_holdout=False,
                    report_produced=None,
                    task_relevance="relevant" if task == "imagery" else "not_applicable",
                    content="mental_imagery_task" if task == "imagery" else None,
                    metadata_status=status,  # type: ignore[arg-type]
                    variables={
                        "task": task,
                        "effect_site_concentration_min": float(np.min(segment)),
                        "effect_site_concentration_mean": float(np.mean(segment)),
                        "effect_site_concentration_max": float(np.max(segment)),
                        "behavioral_responsiveness": responsiveness,
                        "lor_volume": lor_volume,
                        "ror_volume": ror_volume,
                    },
                )
            )
        return units


ADAPTERS = {
    "propofol_tms_eeg": PropofolTMSEEGAdapter,
    "dream_tononi_serial_awakenings": DreamSerialAwakeningsAdapter,
    "tactile_detection": TactileDetectionAdapter,
    "somatosensory_report_task": SomatosensoryReportTaskAdapter,
    "cogitate_meeg": CogitateMEEGAdapter,
    "psiconnect": PsiConnectAdapter,
    "doc_resting_eeg": FigshareDoCRestingAdapter,
    "doc_polysomnography": MendeleyDoCPSGAdapter,
    "propofol_fmri": PropofolFMRIAdapter,
}


def get_adapter(dataset_id: str) -> object:
    try:
        adapter = ADAPTERS[dataset_id]
    except KeyError as exc:
        raise SchemaError(f"no audited adapter for dataset: {dataset_id}") from exc
    return adapter()
