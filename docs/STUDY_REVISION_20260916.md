# Neural dynamics of reported experience across independent settings

Exploratory research design, revised 16 September 2026. This revision supersedes the scientific claims and execution priorities of the August proposal. The project is not registered or preregistered. There are no scientific go/no-go gates; negative, mixed and inconclusive outcomes remain reportable. Technical checks establish that data and computations are interpretable, not that an expected effect exists.

## Question and contribution

Does neural dynamical organisation predict reported experience across independent settings beyond responsiveness, spectral changes and reporting demands, and does it predict responses to physical perturbation? We will compare competing explanations of these observations. Repertoire, metastability, transition directionality, cross-module alignment and passive reachability are candidate measurements, not five assumed necessary components of consciousness. A reduced profile is an admissible scientific outcome.

The frozen EEG encoder is a measurement instrument. Its coordinates do not establish a biological manifold, consciousness, or a mechanism. A common model supplies a common transformation, not automatically measurement invariance across montages, populations and recording systems. We will test transportability and compare sensor, time-frequency and linear representations. The title will retain question-level wording until empirical results establish the scope of any conservation claim.

## Literature boundary

The DREAM database already established that EEG features predict dream reports; our contribution must be cross-study explanatory comparison and independent perturbational prediction, not another within-cohort classifier [1]. PsiConnect has already supported a Nature paper linking low-dimensional trajectories, context and subjective experience [2]. We use it as a boundary test of the shared profile, not as a new discovery of psychedelic latent organisation. Cross-state AI and dynamical modelling of disorders of consciousness also precede this study [3]. We will compare relevant open implementations where input compatibility and participant provenance permit, and will not treat a different neural network as sufficient novelty.

## Open data policy and portfolio

Only data downloadable without individual approval, contributor contact or an account-mediated research access decision enter this execution plan. A public article or metadata page does not establish that the signal data are open. Record version, licence, file checksums, retrieval date and access evidence before analysis. Unknown reuse terms remain an explicit unresolved item; no implied relicensing or redistribution.

The primary portfolio consists of eligible open DREAM cohorts, ds005620 spontaneous/TMS EEG, ds001785 tactile detection and the OSF HQKYM somatosensory task. PsiConnect ds006110 is a secondary boundary test. Clinical PSG and propofol fMRI ds006623 are optional extensions whose absence does not prevent a core report. The opaque-label resting DoC resource is excluded from inferential analyses until a publicly available participant-label key is verified. Cogitate is excluded from the required portfolio because its current access route requires an account/approval. Zenodo 806176 is excluded because its API reports restricted access, despite citations describing repository availability.

The DREAM publication reports 20 datasets, 505 participants and 2,643 awakenings across access categories [1]. These are not this study's eligible sample sizes. Resolve the live public registry, archive its exact bytes on the cluster, enumerate directly downloadable constituent releases and document exclusions. Audit study/laboratory identifiers and reused participants; datasets sharing participants belong to the same outer group. Do not count overlapping Tononi subsets as independent replications.

## Main experience analysis

Use dream experience with recall versus no reported experience within N2 as the initial common contrast. Experience without recall is a separate categorical outcome and sensitivity analysis, not an ordinal midpoint. Analyse REM or other stages separately when both outcomes and sufficient independent participants are present. No sleep stage, missed tactile stimulus, anaesthetic dose or lack of movement is a ground-truth absence-of-consciousness label.

Use the final 20 seconds before each awakening when the source supports that interval. Preserve recording boundaries, discontinuities and artifact gaps. Repeated awakenings are nested within participant and study. Select channel intersections using acquisition metadata without experience labels. Run a common sparse-channel analysis across compatible studies and a separate high-density track; do not synthesize missing electrodes or require the old 15-channel montage for every DREAM cohort.

The primary estimand is the within-stage association between candidate dynamics and reported experience, followed by improvement in predictive log loss on an entirely held-out study relative to conventional alternatives. Balance contributions across participants and studies, report study-specific effects and heterogeneity, and bootstrap at the study/participant levels where the number of independent studies permits. With few laboratories, show each held-out result and avoid overprecise population-level claims. Report recall ambiguity and differential exclusions as measurement limitations.

All representation learning, scaling, dimension selection, state dictionaries, imputation and tuning remain inside the outer training studies. Participants must never cross folds, including aliases across dataset releases. Discovery, inner validation and outer evaluation roles are recorded explicitly. A supplementary subject-separated analysis within studies does not substitute for study transfer.

## Competing explanations

Compare five families on identical eligible observations and outer splits: spectral/arousal features; a conventional complexity scalar with nonlinear calibration; a capacity-matched conventional multivariate model; a shared dynamical profile; and dataset-specific dynamics. The dataset-specific model is evaluated on within-study held-out participants only; its performance is not misrepresented as zero-shot prediction of an unseen study. If a transferable population component is used, distinguish it from study-specific calibration.

Use the same inner-fold tuning budget and independent-unit weighting. Primary comparison: paired held-out log-loss difference. Secondary measures: AUROC, precision-recall performance, calibration and Brier score. Report uncertainty and failure cases, not only winning models. Study-specific calibration of the evaluation cohort is prohibited in zero-shot claims. Missing axes are recorded as unavailable, not zero.

Fit one-, two-, three- and higher-dimensional profile alternatives inside training partitions. Test whether extra axes improve external predictions; do not protect a five-axis theory by relabelling redundant components. Estimate directions and magnitudes of held-out condition changes as well as classification accuracy. A learned multivariate score is a prediction summary, never a probability that a person is conscious.

## Estimability and temporal resolution

Twenty seconds supplies too few independent transitions for unrestricted 6–20-state, 32-dimensional dynamics. Simulate observation lengths, noise, missing channels and artifact gaps actually present in the recordings. Measure bias, uncertainty coverage, null behaviour and recovery of known effects. Candidate simpler dynamics use training-selected dimensions and regularisation. Pool transition evidence hierarchically within studies and participants where justified, never by concatenating independent awakenings into a continuous trajectory. Report an unavailable estimate when its information requirements are not met; all other supported analyses continue.

Maintain two distinct measurement tracks. The slow frozen-encoder track measures second-scale organisation. A fast sensor/CSD or time-resolved linear track tests event timing with explicit temporal support and filter impulse-response checks. Predefine physiologically motivated early and later intervals without inspecting condition effects; quantify window/filter-induced smearing. Do not infer 20–200 ms interactions from one-second LaBraM patches. Avoid zero-phase leakage in prospective timing claims. For scalp alignment, use appropriate volume-conduction controls and describe predictive dependence rather than directed anatomical communication.

## Independent perturbational prediction

Link spontaneous EEG and TMS sessions using official participant, condition and session metadata. Fit passive predictors without TMS outcome access in the held-out participants. Compare condition plus conventional EEG features against the same baseline plus candidate dynamical measurements. Evaluate held-out improvement in squared prediction error and calibration for response spread, differentiation and recovery, with uncertainty clustered by participant.

Report within-condition associations and within-participant wake-to-sedation changes alongside pooled predictions. A pooled correlation induced solely by state separation is insufficient evidence for independent reachability validity. Include stimulation site/intensity, artifact burden and sensory contamination measures when available; state which confounds cannot be addressed without sham or control conditions. Passive reachability remains a model proxy even if it predicts TMS outcomes. Neither this association nor a drug-state contrast identifies the proposed geometry as a cause of consciousness.

## Perception and reporting specificity

For ds001785, compare detected and undetected physical stimuli, analyse catch trials and confidence separately, and distinguish pre-response from response-related intervals. Undetected stimuli do not imply global unconsciousness. Cross-transfer between tactile detection and dream reports is exploratory construct transfer, not evidence that both labels measure the same experience dimension.

HQKYM has sequential no-report and report blocks and unmatched arousal [4]. Report-versus-no-report estimates therefore include order, habituation and context. The within-report task-relevant versus task-irrelevant comparison is the more interpretable specificity contrast. Absence of explicit report does not prove identical experience. Account for physical intensity and timing, and do not claim that covariate adjustment removes perfect design confounding.

## Boundary tests and optional extensions

PsiConnect tests whether altered experience changes the joint profile in ways inconsistent with a monotonic high-complexity account. Reproduce official context mappings before segmenting music, film, meditation or rest. Analyse available subjective measures without using restricted demographic supplements. Because EEG and fMRI were collected at different post-dose times, treat agreement as convergent association rather than simultaneous cross-modal validation [2].

Clinical PSG asks whether estimable profile components associate with available diagnosis categories under substantial montage and clinical uncertainty. Restrict healthy comparisons to matched observed channels. Missing CRS-R, medication or diagnosis keys remain missing. A wake-like profile does not establish covert consciousness or justify diagnostic reclassification. fMRI tests modality-appropriate slow measures and never passive controllability. Both extensions are separate branches and cannot block publication of completed core analyses.

## Precision and robustness

Independent units are participants and studies, not windows, random seeds or repeated partitions. Simulate power/precision using the verified independent sample structure before interpreting results. Use participant-separated and study-separated uncertainty, matched artifact-clean samples and training-only choices. Repeat key comparisons with spectrum-preserving surrogates, temporal nulls and alternative representations. Distinguish unadjusted total associations from adjustments for plausible mediators; list estimands rather than automatically adding every available covariate.

Pretraining overlap must be audited by original cohort provenance, not repository publication date. Confirmed overlap is excluded from strict unseen-cohort claims; unresolved overlap remains separately labelled. An unsupervised encoder can still have encountered evaluation participants. Heterogeneity or failures to transfer are substantive results and do not terminate the workflow.

## Fresh Compute Canada execution

Use a new project namespace, fresh source snapshots, isolated environment, original-source downloads and new run identifiers. Do not import university-server raw files, model caches, checkpoints, fitted objects or results. Source code may be reused after verification. Authenticate through the user's visible SSH password/Duo console. Operational identity, absolute cluster paths and durable logs stay in ignored local records and cluster storage.

The initial deployment runs software qualification and a bounded open-data acquisition pass. Later empirical phases are commissioned only as their actual adapters and technical inputs are verified. A planned module is never reported as executed. Slurm runs CPU/GPU computation; internet-connected transfer/login facilities handle bounded acquisition. Record quotas before expanding storage. Keep retained provenance in project storage; scratch is temporary and subject to cluster retention policy.

The roadmap is: R0 source/environment qualification; R1 open registry and independent-cohort audit; R2 raw acquisition and integrity; R3 blind QC/channel and duration inventory; R4 estimator recovery at observed lengths; R5 frozen representations and metric extraction; R6 study-held-out experience/model comparisons; R7 independent TMS prediction; R8 perception/report and psychedelic boundary tests; R9 optional clinical/fMRI branches; R10 evidence synthesis and figures. These are technical dependencies, not result thresholds.

## Planned evidence displays

Figure 1 explains the competing accounts and independent evidence units. Figure 2 shows study-held-out dream prediction and uncertainty. Figure 3 shows dimension/axis necessity and conventional benchmarks. Figure 4 tests held-out TMS predictions beyond state. Figure 5 presents fast perception/report specificity with confounding boundaries. Figure 6 shows external boundary tests, heterogeneity and failed predictions. Optional clinical/fMRI material is included only to the extent its data support the stated claim.

## References

1. Wong et al. A dream EEG and mentation database. Nature Communications (2025). https://doi.org/10.1038/s41467-025-61945-1
2. Stoliker et al. Psychedelics align brain activity with context. Nature (2026). https://doi.org/10.1038/s41586-026-10910-z
3. Toker et al. Adversarial AI reveals mechanisms and treatments for disorders of consciousness. Nature Neuroscience (2026). https://doi.org/10.1038/s41593-026-02220-4
4. Del Vecchio et al. An HD-EEG Database to dissect Somatosensory Awareness from Task Relevance and Report. Scientific Data (2025). https://doi.org/10.1038/s41597-025-05970-1
