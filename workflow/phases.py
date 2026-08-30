"""Canonical phase graph and command mapping.

The graph encodes technical dependencies only.  It contains no result-based
scientific gate: a phase is complete when its process succeeds and its declared
artifacts validate, irrespective of whether the scientific result is positive,
negative, or inconclusive.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """One restartable workflow phase."""

    name: str
    cli_phase: str
    dependencies: tuple[str, ...]
    uses_gpu: bool
    storage_class: str
    purpose: str


PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec(
        "audit",
        "audit",
        (),
        False,
        "canonical+work",
        "Validate configuration, source releases, licences, metadata, and inventories.",
    ),
    PhaseSpec(
        "acquire",
        "acquire",
        ("audit",),
        False,
        "canonical",
        "Download immutable raw releases directly to the canonical NAS root.",
    ),
    PhaseSpec(
        "qc",
        "qc",
        ("acquire",),
        False,
        "work",
        "Build blind recording inventories, exclusions, and machine-readable QC reports.",
    ),
    PhaseSpec(
        "preprocess",
        "preprocess",
        ("qc",),
        False,
        "work",
        "Create harmonised and native-montage analysis windows on fast storage.",
    ),
    PhaseSpec(
        "encode",
        "encode",
        ("preprocess",),
        True,
        "work",
        "Run frozen, label-free LaBraM inference and representation controls.",
    ),
    PhaseSpec(
        "metrics",
        "metrics",
        ("encode",),
        True,
        "work",
        "Estimate the five manifold properties, nulls, benchmarks, and reliability curves.",
    ),
    PhaseSpec(
        "models",
        "models",
        ("metrics",),
        False,
        "work",
        "Fit participant-level contrasts, held-out transfer models, ablations, and calibration.",
    ),
    PhaseSpec(
        "tms",
        "tms",
        ("metrics", "models"),
        True,
        "work",
        "Validate passive reachability against direct TMS-evoked spread and recovery.",
    ),
    PhaseSpec(
        "locked-clinical",
        "clinical",
        ("tms",),
        True,
        "work",
        "Apply the technically frozen healthy pipeline to held-out DoC resources without retraining.",
    ),
    PhaseSpec(
        "fmri",
        "fmri",
        ("locked-clinical",),
        True,
        "work",
        "Run the secondary BrainLM propofol-fMRI triangulation.",
    ),
    PhaseSpec(
        "figures",
        "figures",
        ("models", "tms", "locked-clinical", "fmri"),
        False,
        "work",
        "Render the final healthy, perturbational, clinical, and fMRI-integrated figures.",
    ),
)

PHASE_BY_NAME = {phase.name: phase for phase in PHASES}
PHASE_NAMES = tuple(phase.name for phase in PHASES)


def select_phases(
    *,
    from_phase: str | None = None,
    through_phase: str | None = None,
    only_phase: str | None = None,
) -> tuple[PhaseSpec, ...]:
    """Select a contiguous phase range or one phase.

    Dependency completion is checked separately by the queue runner.  Selection
    never weakens those dependencies.
    """

    for value in (from_phase, through_phase, only_phase):
        if value is not None and value not in PHASE_BY_NAME:
            raise ValueError(f"unknown phase {value!r}; choose from {', '.join(PHASE_NAMES)}")
    if only_phase and (from_phase or through_phase):
        raise ValueError("--only-phase cannot be combined with --from-phase or --through-phase")
    if only_phase:
        return (PHASE_BY_NAME[only_phase],)

    start = PHASE_NAMES.index(from_phase) if from_phase else 0
    stop = PHASE_NAMES.index(through_phase) + 1 if through_phase else len(PHASES)
    if start >= stop:
        raise ValueError("--from-phase must not occur after --through-phase")
    return PHASES[start:stop]
