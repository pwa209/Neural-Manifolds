# Supplementary high-density transfer repair

The original 14-channel analysis had only two eligible independent laboratory
groups (Turku and Oslo). Oslo evening and morning samples remain one group.
Neither lowering the three-group nested-validation requirement nor treating
those samples as independent laboratories would fix that limitation.

An additional, separately identified exploratory analysis uses DREAM set 9,
**Multiple awakenings**, amendment 1 (approved 26 February 2026). Its supplied
`ExperimentalDescription.txt` explicitly maps 2 to reported experience, 0 to no
experience, and -2 to an ambiguous absence/experience-without-recall category.
Code -2 and other unsupported codes are excluded, not assigned negative labels.
The source describes healthy participants at Sapienza University of Rome,
pre-awakening recordings and a compatible observed scalp montage. Sampled EDF
headers document microvolts. The acquisition description identifies O1; the
corresponding EDF label `01` receives a source-specific alias, not interpolation.
All records with nonempty source quality remarks are conservatively excluded
before signal processing, in addition to the unchanged physical-unit QC.

**Construct boundary:** dream contents were not collected in this cohort. This
extension concerns reported experience, not verified content recall. It does not
retroactively convert the original strict analysis into a completed three-lab
result or claim that the report assays are identical. It is non-preregistered.
Rome, Oslo and Turku remain separate outer groups; unresolved Oslo aliases still
exclude within-laboratory participant-transfer claims. No scientific outcome
controls admission or scheduling.

Other inspected candidates are not silently substituted: set 4 contains extensive
upstream channel interpolation and a lucid-dream cueing protocol; set 10 is a
sleep-talking clinical sample; sets 11 and 12 primarily operationalize recall.
Their availability does not establish suitability for the original contrast.

## Implementation and reuse

`configs/high_density_extension.yaml` records the source decisions.
`revised/high_density_extension.py` preserves all original core outputs. It checks
the original producer receipts, source-code compatibility and file hashes before
reusing high-density measurements from sets 16–18. Only Rome requires new
measurement. New artifacts live under a distinct source-bound extension root.

The dependency chain is qualification → measurement → transfer → sensitivities
→ signal perturbations → robustness → 100 serially capped null replicates →
supplementary synthesis. All empirical work runs in Slurm. The supplementary
chain uses at most one job at a time alongside the existing controller's two-job
limit. Completion receipts distinguish execution from estimability; this repair
does not promise successful generalisation or a particular journal destination.

## Deployment status — 17 September 2026

Source and original producer receipts were verified on the cluster. The Rome
metadata audit admitted 343 recordings before signal QC, including 209 in N2;
these are recording counts, not independent participants. Qualification and all
seven downstream stages have been submitted with explicit success dependencies.
The null stage is a 100-element array capped at one simultaneous task. Signal-QC
survival and estimability remain to be established by these jobs. The original
two-laboratory limitation is preserved, not retrospectively declared solved.
