# LODE external validation: fixed analysis declaration

This is a prospective **analysis freeze within an exploratory, non-preregistered
study**, made after the original results and before inspecting LODE EEG outcomes.
It is not a preregistration or a direct transfer of the original trained model.
The locked machine-readable decisions are in
`configs/lode_external_validation.yaml`. Changing those decisions requires a new
release and an explicitly labelled sensitivity analysis, never overwriting this run.

## Question and independent unit

Can the sensor-trajectory *method* yield temporally specific and report-related
information in an independent home-sleep cohort? The independent unit for
held-out prediction and uncertainty is the participant. Repeated nights are
nested within participant. LODE is from an independent Lucca laboratory and is
not one of the Oslo/Turku/Wisconsin sparse-core groups or the Rome high-density
extension. Its home dry-headband design, bipolar derivations, and delayed morning
reports make transport harder and change the measurement domain.

## Admission and measurement

Use source-coded N2 (`2`) and explicit experience (`2`) versus no experience
(`0`); exclude experience-without-recall (`1`) from the binary test and report
its count. Require a unique case identifier, a participant identifier, a
matching EDF member, at least 20 pre-awakening seconds, and the two native
bipolar channels `Fp1-O1` and `F7-O1`. Both must be absent from the source's
one-based bad-channel list. This fixed pair was chosen from label-blind N2
channel-availability metadata before EEG-outcome analysis: 70/118 N2 records
have both available, compared with 53/118 for the initially contemplated
`Fp1-O1`/`F8-O2` pair. It retains frontal-to-occipital coverage, though the
two bipolar signals share O1. No interpolation or inferred C3. Filter,
resample, one-second artifact QC, minimum 10 clean seconds, and the fixed
sensor log-RMS trajectory follow the core measurement implementation except
that already-bipolar channels are **not** common-average-referenced. Source
quality remarks and report delays are reported; there is no outcome-conditioned
change to admission. The source applies a one-hour report-delay limit to
experience reports but not no-experience reports, an important residual
ascertainment asymmetry that cannot be repaired from these data.

The metadata screen, before EEG signal QC, leaves 65 clear-label N2 records
(43 experience, 22 no-experience) from 25 participants; only eight participants
have both labels. These counts are a feasibility warning, not an empirical
model result. They limit the exchangeability support of the label null.

## Held-out comparison

Use five deterministic, label-blind participant-disjoint outer folds. Within each
outer training set, all dynamics dictionaries, scaling, imputation, dimension
selection and regularization tuning are fitted inside inner folds. Prediction
families are the original spectral, nonlinear-scalar, conventional multivariate,
and shared dynamics models. The training-only constant uses the participant-
balanced experience rate of the outer training set, never the held-out labels.
The primary estimand is equal-participant mean log loss; model-minus-comparator
differences are negative when the model is better. Report Brier and AUROC as
secondary diagnostics, without upgrading them to new primary hypotheses.

## Nulls and interpretation

The two fixed primary one-sided tests each have 999 seeded nested refits:

1. Label permutation within participant and sleep stage: is the observed
   dynamics-minus-training-constant log loss unusually low? Only participants
   with both labels contribute to exchangeability; report their number after QC.
2. Segment-local temporal permutation of the *sensor representation*: is the
   observed absolute dynamics log loss unusually low? This is **not** an
   original-EEG spectral control or a test of a conserved consciousness marker.

Use plus-one Monte Carlo p-values and Holm adjustment across exactly these two
tests. Report all refits and failures; no significance stopping. The family is
separate from, and does not retroactively replace, the original 72-test audit.
The external dataset's selection was informed by the original findings, which
must be disclosed. A significant temporal test without label-specific or
absolute predictive utility supports only a measurement-specific temporal
effect. A non-significant test is not evidence of equivalence or no information.

## Stages and provenance

`preflight` verifies source hashes and records/ZIP linkage without reading EEG
signals; `measure` reads only qualifying EDF windows under Slurm; `assemble`
checks every QC receipt; `observed` produces held-out predictions; `null` runs
all fixed refits; `finalize` reports complete or incomplete inference with
source, config and code-release identities. Raw data and participant-level
outputs stay on personal Alliance storage. The Git repository stores only
code, this declaration, tests and non-identifying progress documentation.
