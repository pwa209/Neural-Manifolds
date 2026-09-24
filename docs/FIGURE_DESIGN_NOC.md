# Figure design: EEG markers of reported experience

Design version: 20 September 2026. Target audience: Neuroscience of Consciousness.
Status: manuscript figure specification. Separate draft artwork for Figures 2-5 is implemented by `scripts/render_manuscript_figures.py`; see `MANUSCRIPT_FIGURE_RENDERING.md`. Neither this plan nor draft artwork implies submission readiness.

This is an exploratory, non-preregistered study. Figure order is an explanatory order, not a claim about the chronology of discovery. This document contains the public design and methods; numerical findings and draft result-bearing captions remain in the private working directory. It introduces no scientific gates and changes no analyses or scheduled jobs.

## Editorial decision

Working title: **Separating predictive performance from temporal specificity in EEG markers of experience**.

The figures should distinguish four questions: Does a model improve on an appropriate comparator? Does its performance depend on the structure targeted by a control? Are observed changes specific to the operational experience measure? Are the underlying summaries measured on an appropriate time scale?

Do not draw a conserved consciousness manifold, an ordering of consciousness levels, a validated biomarker, or a causal mechanism as if established. Do not introduce a decorative brain, manifold embedding, radar chart, or composite consciousness score. The contribution is the empirical separation of these questions, including mixed and unresolved answers.

The existing `configs/figures.yaml` is a legacy operational specification, not the manuscript plan. In particular, its positive claims about state fingerprints, perturbational prediction, control survival and clinical transfer must not become manuscript titles without independent evidence. Leave it unchanged until the renderer and its source contracts are deliberately migrated together. This design supersedes its narrative for manuscript planning only.

## Figure-to-argument map

| Figure | Reader's question | Evidential job | Evidence IDs | Results/Discussion boundary |
|---|---|---|---|---|
| 1. Study structure and inferential questions | What was measured, held out and compared? | Orient the reader to operational reports, overlapping portfolios and independent units | E1, E2, E4, E7, E8 | Multiple datasets are not one pooled cohort; held-out prediction is not mechanism |
| 2. Prediction depends on the comparator | Is apparent improvement useful prediction? | Compare absolute loss, relative loss and discrimination against explicit references | E1, E2 | Comparator-specific advantage is not a general decoding claim |
| 3. Structure controls qualify prediction gains | Is the gain exceptional under the stated null manipulation? | Put observed statistics beside complete null-refit distributions | E2, E3, E8 | Failure to reject is not equivalence or proof of no temporal information |
| 4. Context effects and subjective associations answer different questions | Do context changes establish subjective specificity? | Show context interactions and the complete subjective screen together | E4, E5 | Global ratings, visual conditions and task order limit interpretation |
| 5. Observation history changes the measurement target | Does the uncertainty procedure cover the target actually being estimated? | Diagnose duration/segmentation dependence with explicitly synthetic evidence | E6 | A known-model diagnostic does not repair biological EEG inference |
| Main-text Table 1 | What do the other assays contribute? | Retain TMS, perception and report-task constraints | E7 | Parallel boundary tests, not confirmatory replications or causal proof |

Reading order: establish the design, inspect predictive performance, immediately examine its controls, then consider the independent context and measurement branches. Figures 2 and 3 must be discussed together; never promote Figure 2's favourable comparator result while relegating Figure 3 to the supplement.

## Common graphical language

- Proposed working width: 183 mm. This is a project design choice, not a verified journal requirement. Set physical dimensions before rendering; do not shrink dense artwork to fit.
- White background; sparse light-grey guides; no background gradients, shadows, 3-D effects or significance stars.
- Arial or equivalent sans-serif; aim for 8 pt body labels, 9 pt panel titles and 10 pt bold lower-case panel letters at final size. Avoid text smaller than 7 pt. Split crowded supplementary pages rather than shrinking labels.
- Representation identity: encoder blue `#0F4D92`/circle; sensor teal `#42949E`/square; time-frequency violet `#9A4D8E`/triangle. Use shapes as well as colour.
- Comparator identity: conventional medium grey, nonlinear Lempel-Ziv dark grey, training-only constant black diamond/dashed reference. Use direct labels or line style, not a new palette for every model.
- Core and high-density extension occupy separate labelled facets with identical scales where the estimand is the same. Never connect them as a paired density intervention.
- Show zero for difference plots, 0.5 for binary AUROC, 95% for nominal coverage. Log loss has no universal chance reference: use the actual training-only constant predictor.
- Negative dynamics-minus-comparator loss favours dynamics. Print that direction once per figure. Do not reverse axes between panels.
- Missing, unavailable and unestimable are distinct from zero: grey hatch plus a reason in the source table. Never use blank white cells to encode both zero and missingness.
- Keep observed/predicted evidence, null refits, schematic examples and simulated evidence explicitly labelled. Simulation conditions and null replicates are not biological participants.
- Every interval is defined locally. Prediction bootstrap intervals, study-t sensitivities, participant contrast intervals, correlation intervals and binomial coverage intervals are not interchangeable.
- Native participant identifiers stay on the data host. Display anonymous dots, study labels and counts only; do not place participant-level data in the public repository automatically.

## Figure 1 — Study structure and inferential questions

Working size: 183 x 115 mm. Three panels. No quantitative hypothesis-confirmation graphic.

```text
| a  Cohort x analysis-role matrix                       (full width) |
| b  Held-out evaluation and training-only references | c  Questions |
```

**1a — Cohorts and their roles.** Rows are actual dataset/cohort identities; columns are core transfer, high-density extension, context/ratings, TMS, tactile and report task. Filled cells mean inclusion, not success. Print participant-identifier and observation counts separately. Use explicit overlap connectors or a text bracket, not proportional Venn areas unless intersections have been verified. Do not sum portfolio sample sizes. Source: qualified cohort inventory and original evidence manifests; exact overlap and final branch-specific counts must be extracted before drawing.

**1b — What was held out.** A simple study-level train/held-out schematic distinguishes representation preparation, fitted predictor and evaluation. Place conventional, nonlinear scalar and training-only constant references alongside dynamics. Show preprocessing/tuning boundaries from the actual code, not an idealized leakage-free pipeline. Any unresolved encoder-pretraining overlap remains an explicit caveat. Independent inference concerns studies and participants, not the number of windows.

**1c — Four distinct questions.** A compact diagram links comparison, temporal controls, experience/context contrasts and measurement-target checks to their respective figures. Use connectors labelled 'evaluated by', not causal arrows or a pass/fail funnel. Operational label: reported experience, with dataset-specific report semantics noted. These are interpretive questions, not newly imposed scientific gates.

Caption must define the two overlapping portfolios, nesting, the meaning of filled cells and the separate status of synthetic diagnostics. Figure 1 must not imply every dataset supplies every type of evidence.

## Figure 2 — Prediction depends on the comparator

Working size: 183 x 160 mm. The widest panel is the absolute-performance comparison.

```text
| a  Absolute held-out log loss: core | extension          (full width) |
| b  Dynamics - conventional         | c  Dynamics - nonlinear LZ       |
| d  Held-out discrimination by study and representation  (full width) |
```

**2a — Absolute loss.** Dot-and-line plots place all dynamics representations and comparators on a common loss axis within each portfolio. Small dots are held-out study summaries; connect matching study summaries across models to expose heterogeneity. Large marks are equal-study means, with participants weighted equally inside each study. Include the actual training-only constant reference. No bars or invented uncertainty for that new reference. Display participant-level variation in Supplement S1, not as independent study replicates. If model evaluation sets differ, align verified matched sets or disclose separate sets; never quietly compare incompatible populations.

**2b — Conventional-model comparison.** Forest plot with three representations in each portfolio. Plot source dynamics-minus-conventional log-loss estimates and original 95% bootstrap endpoints conditional on fitted predictions. Add small study-specific points from the precision audit. A zero line and directional labels make the contrast readable. Fully refitted population-generalization intervals are not available; do not label these as such.

**2c — Nonlinear scalar comparison.** The identical forest geometry for dynamics minus nonlinear Lempel-Ziv. Use the same difference-axis limits as 2b, with no hidden truncation. The comparator should be adjacent to 2b because it changes the interpretation of the model ranking. Spectral/arousal comparators remain fully tabulated and plotted in S1/S2.

**2d — Discrimination.** Study-level AUROC dots for each representation, separated by portfolio, with a 0.5 reference. Include unavailable single-class cells explicitly. Do not infer AUROC from log loss or fabricate intervals absent from the source. Fixed y range 0-1. Probability calibration and distribution diagnostics belong in S1, labelled descriptive.

Sources: original `transfer.json` predictions/model summaries; evidence review `models`, `comparisons`, `comparisons[].precision.study_mean_deltas`, and `prediction_diagnostics[].training_constant`. The review output alone does not contain every participant prediction needed for plotting.

Main caption: define held-out study, participant and observation counts, the loss aggregation, comparator construction, bootstrap meaning and the descriptive status of the constant-reference comparison. State that the high-density extension is not a matched channel-density experiment. A one-line caption cross-reference directs readers immediately to Figure 3.

## Figure 3 — Structure controls qualify prediction gains

Working size: 183 x 180 mm. Two large comparison panels, each containing matched small multiples. No selection of only favourable or only unfavourable representations.

```text
| a  Absolute dynamics loss       | b  Dynamics - conventional loss |
|    core and extension          |    core and extension            |
|    all three representations   |    all three representations     |
| c  Complete family / resolution / exchangeability key (full width)|
```

**3a — Absolute performance against null refits.** For both portfolios and every representation, show label and temporal nulls as stacked one-dimensional rugs/strips of all completed replicate statistics. Add a solid black observed-statistic line. Use deterministic display offsets; they encode no variable. Prefer empirical marks over smoothed violin densities with only 100 replicates. Compare the observed statistic to the same statistic recomputed in each nested null refit.

**3b — Relative advantage against null refits.** Repeat exactly the same layout for dynamics-minus-conventional loss. Share a difference axis across these facets; absolute and relative panels have separately labelled axes. Display the exact plus-one one-sided raw tail and the full-family Holm-adjusted value beside each row. No p=0, p<.001 or star labels with 100 replicates.

**3c — What the controls do and do not test.** A compact annotation strip reports completed refit counts, full multiplicity family, minimum Monte Carlo resolution and effective participant label-permutation support. Name the exchangeability assumption. Representation temporal/phase manipulations are not original-EEG spectral controls. Put the full representation-phase family, alternative comparators and signal-level perturbations in S2; report any interpretation-changing exception in the main caption, not only the supplement.

Sources: source `controls.json` repeat-level summaries joined by portfolio, track, representation, null kind and seed/repeat; review `null_controls` supplies observed statistics and raw/adjusted tails. Crucially, the review JSON stores median/range but not the entire replicate distribution. Retrieve the original verified records; never manufacture a distribution from a range, median or p-value.

Main caption: explain the difference between testing absolute prediction and testing relative model advantage; identify all null operations and refitting; disclose post-results family definition and low resolution. A null tail is not a confidence interval, and a non-small tail does not prove equality.

## Figure 4 — Context effects and subjective associations answer different questions

Working size: 183 x approximately 215 mm. Three panel groups; split supplementary detail if 8 pt text cannot fit.

```text
| a  All context-minus-rest interactions: 9 feature facets, 3 contrasts |
| b  Movie-minus-rest participant differences: repertoire | LZ          |
| c  All subjective associations: four context-specific heatmaps        |
```

**4a — Complete context interaction set.** Nine small multiples in a 3 x 3 arrangement, one per measured feature, each with movie/rest, meditation/rest and music/rest rows in a fixed design order. Each point estimates `(session 02 - session 01)context - (session 02 - session 01)rest`. Use original descriptive participant-bootstrap 95% intervals matched by contrast key; report corrected sign-flip p-values from the review, adjusted across both session changes and interactions. The review also contains t intervals: do not silently substitute those for the original bootstrap intervals. Each feature has its own labelled native or verified standardized units, with consistent limits across its three contexts. Never force repertoire and LZ onto a shared effect scale. Put paired n next to each row. Label all 27 contrasts, not just the corrected findings.

**4b — Participant variation in the two highlighted movie/rest contrasts.** Two compact strip plots show every complete paired participant contrast, a zero line, mean and the same descriptive interval used in 4a. These are result-selected illustrative distributions; the complete interaction family is already visible above. A visible annotation states fixed context order, eyes-open/closed differences and no placebo. Do not draw an isolated context-to-consciousness causal arrow.

**4c — Complete subjective screen.** Four heatmaps, one per context; rows are the eleven official ASC11 subscales and columns the nine features, using identical order. Colour encodes Spearman rho on a fixed -1 to +1 diverging blue-neutral-orange scale. Name this panel-local colour meaning explicitly; it does not encode the representation colours used elsewhere. Show all 396 cells. An outline may indicate a Holm-adjusted finding only if supported by the source; non-significant cells remain coloured by effect size rather than erased. Missing cells are hatched. Exact n, raw p, adjusted p and intervals for every cell go in the accompanying source table and S4. Caption gives n range, non-imputation, multiplicity family and the distinction between global post-session scales and context-specific EEG. Do not choose the largest unadjusted correlation for a main-text scatterplot.

Sources: original `boundary.json` measurements and contrast intervals; review `boundary.contrasts` for corrected inference; `subjective_review.json` rows for scale mapping. Exact feature/unit and codebook labels are required; do not conflate five conceptual regime axes with the nine operational features in this branch.

Main caption: qualify these as context-associated changes, not novel discovery of context sensitivity or verified subjective specificity. Cite the original PsiConnect context-dynamics study when writing the manuscript; this is a distinct operational reanalysis, not an independent cohort replication.

## Figure 5 — Observation history changes the measurement target

Working size: 183 x 165 mm. Label the whole figure 'Measurement diagnostic: schematic and simulation'.

```text
| a  Same state sequence, continuous history versus segmented history |
| b  Coverage across every simulation setting: 3 feature facets       |
| c  Finite-window target versus long-record reference by length      |
```

**5a — Mechanism schematic.** Draw the same short categorical state sequence twice, with continuous history above and explicit segment boundaries below. Show where a repeated state counts as revisited under each segmentation rule. Mark it 'illustrative state sequence, not EEG'. Compute any printed recurrence values using the actual estimator; do not attach numerical values from a different example. No smooth manifold trajectory is needed.

**5b — Coverage by setting.** Three feature facets compare original block resampling against the long-record reference, original block resampling against the finite-window target, and known-model finite-window intervals. Show all 12 setting-specific coverage estimates per method, with a 95% nominal line and a common 0-100% scale. Use labelled neutral marker shapes for methods to avoid borrowing the biological representation colours. All 200-test-replicate binomial intervals are plotted in S6; a main-panel median is only a descriptive summary across settings, not a pooled coverage estimate. Do not smooth the twelve settings or treat them as biological replicates.

**5c — Target dependence.** Plot the actual finite-window targets and long-record references against lengths 12, 20 and 120, separately for each feature, persistence and missingness condition. Use only recorded source targets; no interpolation suggesting an estimated asymptotic curve. The finite-window target is a Monte Carlo estimate, and the long-record value is a simulated reference, not an exact population constant. Omit error bars if target Monte Carlo uncertainty was not retained. The recurrence facet receives the greatest width if necessary.

Sources: `recovery-geometry-v3.json` rows keyed by `axis`, `n`, `stay_probability` and `missing_fraction`; target fields `finite_window_target`/`long_record_reference`; coverage fields `block3_long_record`, `block3_finite_window`, `model_based_finite_window`, including `coverage_binomial95`. Original measurement diagnostics are retained in S6.

Main caption: known stationary three-state model; independent reference, calibration and test samples; exact simulation settings and replicate counts. State that this concerns within-trajectory measurement intervals, not the prediction-level intervals in Figure 2. No claim of repaired EEG calibration or validation of all features follows.

## Main-text Table 1 and supplementary programme

Table 1 columns: assay; operational contrast/outcome; independent sample size; estimate and interval (separate rows for different units); multiplicity; warranted conclusion; principal limitation. Include TMS incremental prediction, the very small paired awake/sedation subset, tactile confidence/perception and the separate report-task results. Counts of corrected tests are descriptive, not a common cross-assay effect size. No traffic-light success score.

| Supplement | Required content | Reason it remains visible |
|---|---|---|
| S1 | All prediction comparators, study/participant loss summaries, AUROC, probability distributions, descriptive calibration, study-t and delete-one-study sensitivities | Comparator fairness, heterogeneity and conditional precision |
| S2 | Complete null family, representation-phase and signal-level controls, all seeds/replicate statuses and exact raw/adjusted tails | No selective controls; distinct null operations remain distinct |
| S3 | All session changes and all context interactions with participant-level pairing and exclusions | Full boundary multiplicity and individual variation |
| S4 | All subjective correlations, intervals, complete-pair n, missingness and official scale mapping | No selected scales or apparent precision from omitted missingness |
| S5 | TMS, tactile and report-task forest/participant plots; complete-file exclusion audit; tiny paired sample display without fitted mechanistic lines | Preserve independent boundary evidence and inconclusive results |
| S6 | Every simulation cell and binomial interval; original measurement failures and unavailable axes; estimator definition | Separate diagnosis from biological validation |
| S7 | All sensitivity and ablation branches, exact evaluated study/participant/observation counts, unavailability reasons | Preserve opposite directions and tiny or non-transferable subsets |
| S8 | Cohort/report-label mapping, portfolio intersections, data lineage, preprocessing and hold-out provenance | Prevent false pooling, label equivalence and replication claims |

Supplement numbers denote expandable figure groups; use multiple pages where needed. Source tables are machine-readable and accompany each figure, subject to the project's data-sharing policy.

## Reproducible production handoff

1. Plot on the data host from original immutable artifacts and the reviewed outputs. No raw EEG or participant questionnaire download is required for figure design. Bring back rendered figures and authorized aggregate source tables only.
2. Create a separate manuscript figure build; do not overwrite historical figures or source analyses. Use the established Python/Matplotlib backend. A revised config must distinguish interval methods per panel rather than applying a single global bootstrap setting.
3. Before plotting, resolve cohort intersections, exact per-panel samples/units, comparison keys and original interval field names. These are source-extraction dependencies, not requests for a new favourable result. If unavailable, disclose them and omit the unsupported mark, never invent it.
4. Save a manifest for each panel: evidence ID, source path on host, SHA-256, JSON/table selector, filters, contrast, aggregation, interval definition, multiplicity family and output filename. Keep private paths out of public manifests.
5. Initial exports: editable-text SVG, embedded-font PDF and PNG preview. Add TIFF only if required for submission; no need for a raster version of every vector plot.
6. Verify numerical correspondence, full family/branch inclusion, sample counts, missingness, finite values and all axis/unit labels. Preserve disagreements or unavailable intervals explicitly.
7. Inspect actual PDF/SVG exports at final physical size and enlarged for clipping, legibility, colour accessibility, overlapping marks and caption agreement. A successful render is not visual QA.

Planned output names: `fig01_design`, `fig02_prediction`, `fig03_null_controls`, `fig04_context_subjective`, `fig05_measurement`, and `table01_boundary_evidence`.

No distributions, participant dots or intervals have been fabricated for this design. Execution and visual-inspection status belong to each separate figure build's manifest and QA record. Journal-specific technical compliance should be checked against current official instructions at the export/submission stage.
