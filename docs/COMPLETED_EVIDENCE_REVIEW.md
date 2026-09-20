# Exploratory evidence-review methods

This is a post-results audit, not a preregistration, confirmatory report, or scientific gate. Original analyses, outputs, exclusions and contrary findings remain unchanged. Private results and operational identities are not part of this public document.

## Reproducible scripts

- `scripts/review_completed_evidence.py`: verifies source artifact hashes; inventories transfer, sensitivity and robustness analyses; compares observed statistics with all completed label, temporal and representation-phase null refits; audits conditional precision and boundary multiplicity.
- `scripts/review_subjective_mapping.py`: reads the pinned public, non-imputed ASC11 scores and codebook from the original dataset Git repository on the data host; uses exact participant identifiers; reports every planned scale–feature–context association.
- `scripts/review_recovery_geometry.py`: separates finite-window targets from long-recording references in a known-parameter simulation of recurrence, exit entropy and dwell dispersion. It is not biological calibration.

Run these scripts on the data host with private input/output paths. They require the existing qualified scientific Python environment. They do not need to download raw data to a workstation. Use a new output filename/directory on each run, retaining historical attempts. The evidence audit refuses to overwrite its output and validates JSON serialization before opening the destination.

## Null comparisons

Lower log loss is better. The observed absolute dynamics log loss is averaged equally across studies, each study weighting participants equally. Relative performance uses the same dynamics-minus-comparator loss statistic as the original analysis. Each observed statistic is compared with the same statistic from each complete nested null refit. One-sided Monte Carlo tails use `(1 + count(null <= observed)) / (B + 1)`.

All comparisons are retained, with an explicit broad Holm-adjusted audit family across portfolios, representations, null types and comparators. This family was chosen after seeing the original results and is not a retroactive preregistration. Report unadjusted tails alongside adjusted values. With 100 replicates, minimum attainable unadjusted p is 1/101; absence of an adjusted finding is not evidence of equivalence. Repeated correlated contrasts and the finite permutation budget make the broad audit conservative.

Label permutation assumes exchangeability within participant and sleep stage. Temporal permutation and phase randomization operate on the representation; they are not original-EEG spectral controls. Existing signal-level phase perturbations are inventoried separately. A model advantage can persist under a null and therefore cannot, on its own, establish specificity for the removed structure.

## Precision and comparison references

The original hierarchical study/participant bootstrap is conditional on fitted predictions. The review adds study-specific differences, delete-one-study averages and a small-sample Student-t interval across study mean differences. This last interval is a sensitivity calculation, not a definitive replacement: folds share training data, and only a few independent study groups are available. Neither approach propagates model-refitting uncertainty.

No observed post-hoc power is calculated. No equivalence margin is chosen from the results. More observations from the same cohorts cannot be counted as more independent studies. The additional constant predictor uses only the other represented studies' label rates, weighting studies and participants equally, and is not fitted to held-out labels.

## Boundary multiplicity

The audit retains all 36 session contrasts and 27 context-versus-rest interactions. Participant-level sign flipping uses 19,999 draws and Holm adjustment across all 63 tests. Symmetry under the null is assumed. Previously fitted feature transformations and cross-fitted standardization are held fixed; shared training dependence is not fully resampled. Fixed context order, eyes-open versus eyes-closed differences and absence of a placebo session remain interpretive limits.

## Subjective-scale mapping

Official scores are under `derivatives/phenotype/scored_not_imputed/ses-02/`, not the top-level `phenotype/` directory searched by the initial subset acquirer. The mapping uses the pinned public release, its README and scoring dictionary. The eleven original ASC11 subscales are retained, not the overlapping composites. This defines 11 scales × 9 features × 4 contexts = 396 exploratory comparisons.

The analysis relates within-person EEG session differences to between-person subjective ratings. It does not treat the ratings as simultaneous or context-specific reports. Missing values are not imputed; complete-pair sample sizes are reported individually. Spearman correlations use asymptotic p-values, 999 participant bootstrap draws for descriptive intervals, and Holm correction across all estimable associations. Those intervals are not simultaneous intervals. No new composite or scale is selected because of its observed association.

## Measurement-interval diagnosis

The original recurrence measure is the fraction of subsequent runs revisiting any state previously visited within the same segment. It depends on observation duration and segmentation. Short blocks reset this history, so a short segmented bootstrap and a long-record reference do not have interchangeable targets.

The added diagnostic uses a known stationary three-state Markov model, independent reference/calibration/test samples, lengths 12/20/120, two persistence settings and two missingness rates. Model-based finite-window error quantiles are compared with the old three-observation block procedure. Coverage has binomial intervals. The simulation tests a failure mechanism and a possible model-dependent remedy; it does not supply replacement EEG confidence intervals or validate all eight features. The prediction-level bootstrap intervals are a different procedure and must not be confused with these within-trajectory interval diagnostics.

## Reporting boundaries

Do not add sample sizes across overlapping portfolios. Do not compare low- versus high-density significance as though channel density were randomized: participant composition and report semantics also differ. Preserve unavailable sensitivity cells and tiny held-out samples. Graphical and manuscript review remain separate from successful execution.

Primary-source context: [PsiConnect descriptor](https://doi.org/10.1038/s41597-026-07312-1), [existing context-dynamics study](https://doi.org/10.1038/s41586-026-10910-z), and [DREAM database](https://doi.org/10.1038/s41467-025-61945-1). Context-sensitive psychedelic dynamics is prior work, not a new discovery attributable to this reanalysis.
