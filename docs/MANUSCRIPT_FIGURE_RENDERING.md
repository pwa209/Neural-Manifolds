# Reproducible manuscript figure drafts

**Historical 100-refit renderer.** The final 5,000-refit continuation supersedes Figure 3 and its legend from this draft. Use `scripts/render_null_5000_update.py` and `docs/NULL_5000_RESULTS_AND_FIGURES.md` for the current manuscript figures. Figures 2, 4 and 5 are unchanged. Do not present the 100-refit Figure 3 as the final result.

The exploratory, non-preregistered Figures 2-5 follow `FIGURE_DESIGN_NOC.md`, not the hypothesis-led legacy renderer. This build does not modify original analyses, existing figures or the study controller. Numerical results, rendered figures and aggregate source tables remain in private working/output directories, outside public Git tracking.

## Rendering on the data host

Use the qualified study environment, with the project source on `PYTHONPATH`:

```text
python scripts/render_manuscript_figures.py \
  --core /private/path/core/evidence.json \
  --hd /private/path/extension/evidence.json \
  --review-root /private/path/review \
  --output /private/path/new-figure-build
```

The renderer is intentionally pinned to the completed review identities in its `EXPECTED` mapping. Changing those identities requires explicit source review, not silently accepting newer files. The original synthesis hashes must match the review, and every selected transfer, control and boundary artifact is checked against its recorded SHA-256. Original files are read directly after verification, avoiding reliance on potentially divergent embedded copies.

The output directory must not exist. Historical drafts are preserved. Exports include editable-text SVG, embedded-font PDF, PNG previews, a combined four-page PDF, legends, aggregate CSV source tables and a provenance manifest. The renderer performs no model fitting or inferential resampling. Participant-level context differences are derived on the host solely to plot the original paired contrast and are checked against the original aggregate mean and sample size. No participant records are exported in source tables.

## Panel-specific source contracts

- Figure 2: original model summaries and reviewed conditional intervals; the training-only constant reference is descriptive. Do not silently add intervals or pool overlapping portfolios.
- Figure 3: original values from all 100 unique null refits per displayed family. Recomputed plotting tails must equal the reviewed tails. Full-family Holm values remain those of the review. Aggregate ranges/medians are never expanded into artificial distributions.
- Figure 4: original participant-bootstrap contrast intervals and separately matched corrected sign-flip tests. All 27 interactions and 396 subjective associations are retained. A main-panel statistical colour distinction is keyed; heatmap colour encodes effect magnitude, not significance.
- Figure 5: all recorded simulation settings, targets and coverage values. The schematic alone uses a labelled illustrative sequence, evaluated by the actual estimator. It is not empirical EEG evidence.

## Transfer and inspection

`scripts/receive_manuscript_bundle.py` can receive a generated ZIP encoded in an already-authenticated SSH session log. It checks the transfer hash, rejects unsafe paths and verifies every artifact against the renderer manifest before extraction. It does not open a new connection or handle passwords/OTP. Keep the source log and resulting artifacts private.

```text
python scripts/receive_manuscript_bundle.py \
  --log /private/path/session.log --version v2 \
  --destination /private/path/new-local-draft
```

Inspect all final PNGs and rasterized PDF pages. Canvas-bound checks are supplementary to visual inspection: they cannot detect every collision or scientific mislabelling. Inspect actual SVG text structure and embedded PDF fonts. Keep the renderer's source hash, the transfer receipt and a separate human-readable QA record beside the private drafts.

Tests in `tests/test_manuscript_figures.py` use explicit synthetic fixtures only and do not generate scientific artwork. They check complete null families, duplicate rejection, agreement with reviewed tails, interval endpoint preservation, nonfinite rejection and correct paired contrasts.

Draft completion does not establish fully refitted predictive uncertainty, biological EEG calibration or submission-specific journal compliance. Those remain separate scientific/editorial questions.
