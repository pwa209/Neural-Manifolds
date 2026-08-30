# University-server operations

This is an exploratory, non-preregistered project with no scientific gates. All
checks described here are technical integrity/provenance requirements.

These scripts target the verified SSH compute endpoint
`wangpeng@127.0.0.1:1022`, whose required remote identity is
`wangpeng@kemove-Rack-Server`. They never accept or store a password. Run them
inside an authenticated interactive SSH session (or through an SSH key/agent).

Every mutating command requires all three **user-confirmed, project-specific**
roots. The scripts deliberately refuse the broad storage parents and do not infer
a Neural Manifolds path from any other project:

- canonical/raw NAS: `/private_nas/wangpeng/neural-manifolds`;
- active work: `/data1/wangpeng/neural-manifolds-work`;
- restart markers/logs: `/data2/wangpeng/neural-manifolds-checkpoints`.

These exact roots are confirmed and recorded in `configs/server.yaml`. The safe
remote sequence is:

1. `preflight.sh` (read-only);
2. `bootstrap.sh --check-only`, then `bootstrap.sh --apply`;
3. push reviewed source to GitHub and select an exact commit;
4. `deploy_from_git.sh --check-only`, then `--apply`; if the private repository is
   not reachable from the server, use the explicit hash-verified archive transport
   described below instead—there is no automatic network-to-archive downgrade;
5. build a Python 3.11 environment with `bootstrap_runtime.sh` from the pinned
   requirements lock (SHA-256
   `e9a37e9acafaead6a6f77d966a9e0b4ef083acd73cf0dc78ac3dd193310ce39e`);
6. run `bootstrap_models.sh --stage core` to clone both exact source revisions and
   download/hash-verify only LaBraM; its generated `model_paths.env` is sourced by
   the launcher;
7. `launch_queue.sh --check-only`, inspect `--dry-run`, then use `--apply`;
8. immediately before the fMRI phase, run `bootstrap_models.sh --stage fmri` to
   materialise exact-revision, SHA-256-pinned BrainLM files under its
   CC-BY-NC-ND-4.0 restrictions;
9. copy `configs/fmri-inputs.template.yaml` to reviewed checkpoint or canonical
   metadata storage and pass the absolute copy as `--fmri-input-manifest` to each
   fMRI check/dry-run/apply launch;
10. inspect with `status.sh` and the reported tmux/log paths.

No deployment, queue launch, or dataset download has started for this project.
After bootstrap, launch phases in exactly this order:

```text
audit
acquire
qc
preprocess
encode
metrics
models
tms
locked-clinical
fmri
figures
```

The queue is resumable. Re-running the same run ID validates successful artifacts
and skips them; a failed or interrupted phase receives a new numbered attempt.
Changed code/configuration under the same run ID is rejected, so use a new run ID
instead of overwriting provenance. Acquisition receives the canonical
`raw` directory directly; active derivatives are written under the fast work root.

`bootstrap_runtime.sh` may leave a package copy in the dependency environment for
its build-time smoke checks, but launches never trust that copy. `launch_queue.sh`
sets `PYTHONPATH` exactly to `DEPLOYED_RELEASE/src:DEPLOYED_RELEASE`, verifies both
imported module paths, and runs the queue and CLI with the selected Python via
`-s -P -m`. The content-addressed deployed release therefore remains the source
authority even when several releases share one dependency lock/environment.

### Private-repository archive transport

The archive path is a transport fallback for an already reviewed, pushed exact
commit; it does not replace GitHub as the source of truth. On a trusted machine
that can read the private repository, verify the checkout and create an
unprefixed Git archive, then record its SHA-256 independently:

```bash
git rev-parse HEAD
git archive --format=tar --output Neural-Manifolds-<COMMIT>.tar <COMMIT>
sha256sum Neural-Manifolds-<COMMIT>.tar
```

Transfer that tar file to a server-local path with an authenticated, resumable
transport. Do not put a password or token in the command, archive, or repository.
Inside the authenticated server session, run:

```bash
bash scripts/remote/deploy_from_archive.sh \
  --canonical-root <CONFIRMED_CANONICAL_ROOT> \
  --work-root <CONFIRMED_WORK_ROOT> \
  --checkpoint-root <CONFIRMED_CHECKPOINT_ROOT> \
  --repository https://github.com/pwa209/Neural-Manifolds.git \
  --commit <EXACT_LOWERCASE_COMMIT> \
  --archive <SERVER_LOCAL_TAR> \
  --archive-sha256 <INDEPENDENT_SHA256> --check-only
# Review, then repeat with --dry-run and finally --apply.
```

The script verifies the archive hash and embedded Git commit, rejects unsafe
members and links, re-verifies an exact private staging copy, publishes a complete
source manifest/provenance record, and changes `source/current` only after the
content-addressed release validates.

The metrics phase includes 100 explicit pre-encoder EEG sensor-row permutation
repeats and the repeated equal-window/reliability sensitivity stage. The healthy
representation evaluation partition remains outside state-dictionary and profile
calibration fitting.

`locked-clinical` creates a technical hash snapshot of every healthy success marker
through TMS before held-out DoC transfer. Clinical releases may already exist on
canonical storage and in file inventories, but their signals are not parsed,
preprocessed, or encoded until after that lock is validated. It is explicitly
**not** a registration, preregistration, scientific result gate, or favourable-result
check.

The subsequent fMRI phase enforces the strict `ds006623` manifest, labels-after-
encoding boundary, participant partitions, and discovery-only calibration of the
fMRI-compatible R/M/D/A axes. It never reports passive-fMRI reachability. Figures
run last so the clinical and fMRI panels are genuinely late, integrated outputs.

The model bootstrap exports `NEURAL_MANIFOLDS_MODEL_MANIFEST`, LaBraM source and
checkpoint paths, BrainLM source, and—only after the fMRI bootstrap—the BrainLM
checkpoint directory. The queue rehashes these files before model-dependent phases.

The fMRI manifest is deliberately late-bound and is not part of the immutable base
run contract. `audit` and all unrelated phases therefore require no fMRI manifest.
For `fmri`, `--check-only` read-validates and hashes the strict YAML/JSON manifest,
both referenced assets, the 0/1 timing origin, and the fMRI model cache without
writing. `--apply` atomically binds the resolved record at
`CHECKPOINT_ROOT/queue/RUN_ID/late-inputs/fmri.json`; any later mismatch requires a
new run ID. The manifest itself belongs under `CHECKPOINT_ROOT/metadata` or
`CANONICAL_ROOT/metadata`, never in the raw data tree or as raw data in Git. Static
non-null `configs/server.yaml` `fmri_inputs` remain a legacy alternative, but they
cannot be combined with `--fmri-input-manifest`.

Remaining external blockers are concrete rather than scientific: the approved
UKB_424 atlas and ordered coordinates; the explicit
`ds006623` timing-index origin; Cogitate account access and native event schema;
official clinical label availability; and the possibility that the `ds006623`
git-annex special remote cannot serve required content. The BrainLM revision and
checkpoint hashes and the server requirements lock are already pinned; do not list
them as unresolved prerequisites.
