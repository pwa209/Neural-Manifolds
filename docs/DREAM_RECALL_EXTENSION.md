# Independent DREAM report-sensitivity extension

This is an exploratory extension selected after seeing the original study's results. It is **not** preregistered and is **not** an enlargement of the LODE strict experience-versus-no-experience validation. The DREAM source files are already present on `pwa209` personal Alliance scratch under `cognition-and-consciousness/fresh-20260916/data/raw/dream/registry_v6_2025-06-08`; the pipeline reads those existing archives in place and does not copy raw data to GitHub or project storage.

## Scientific question and limits

Test whether the same training-only, common-sensor trajectory method has held-out predictive information for subsequent N2 dream report or impression in *independent people and datasets*. Six open DREAM datasets are considered: Zhang & Wamsley (targeted NREM only), ChildrenDreaming, Rome young and older adults, Brazil morning reports, and Kumral et al. The sources span four laboratories, but the three Rome cohorts are one laboratory cluster. Source questions and timing differ. A `0` in several datasets means no recalled report and does **not** establish absent conscious experience. Children's positive category combines recall and the impression of dreaming. Kumral's end-of-recording may precede awakening by up to one minute. Neither these categories nor their null tests may be pooled with the strict LODE E-versus-NE p-values.

The metadata preflight fixes one eligible N2 awakening per participant before signal or model analysis, using the lexicographically first linked recording. It retains every source and QC exclusion in receipts. The EEG is limited to source-named F3, C3 and O1 (documented `01` and mastoid-reference aliases only), common-averaged across these three sensors, filtered 0.5–30 Hz, resampled to 200 Hz, and summarized over the final 20 seconds of the provided report-linked recording. The Zhang targeted NREM selection takes 10–30 seconds before the report marker to reduce documented marker-delay contamination. Signal QC is label-blind and requires at least ten clean seconds. The analysis is a method-level re-fit, not a zero-shot transfer of original coefficients.

Observed analysis leaves each dataset out in turn and nests all representation fitting and tuning within training datasets. It reports held-out log loss, Brier score and AUROC when estimable, with training-only constant, spectral, nonlinear-scalar and conventional multivariate comparators. Equal dataset/participant weighting prevents prolific cohorts from defining the pooled loss. Two fixed 199-refit null families test cohort-stratified label association and sensitivity to temporal ordering; both use the complete fitting procedure and Holm correction across the two tests. The small number of genuinely independent laboratories, report semantic differences, and wake/timing contamination risk remain limitations even if many participants pass QC. All outcomes, including null and unavailable results, are reported.

## Cluster workflow

1. `preflight`: audit registry source archives, linked CSV records, EDF headers and independent-person counts.
2. `measure[0-11]`: materialize one EDF at a time on scratch, extract a 20-second EEG window, run label-blind QC and save source-linked measurements.
3. `assemble`: audit exclusions and count the measured people and labels by dataset.
4. `observed`: perform leave-one-dataset-out nested prediction and comparators.
5. `null[0-79]`: 199 label and 199 temporal refits in checkpointed batches of five.
6. `finalize`: aggregate the two tests regardless of sign; incomplete refits are explicitly reported as incomplete.

Jobs use `afterany` dependencies so a failure still produces a visible downstream receipt or explicit missing-input error, not a silent stalled chain. The `preflight` source is personal scratch; all outputs are in a distinct personal-scratch validation directory. No source data, personal identifiers, or EEG arrays are committed.
