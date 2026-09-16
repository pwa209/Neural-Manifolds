# Fresh open-data study on Compute Canada

The [revised executable graph](REVISED_EXECUTION.md) supersedes implementation-gap
statements below. Refer to the latest top section of `STATUS.md` for deployment
evidence; planned dependencies, submitted jobs and completed analyses are distinct.

This roadmap implements the September revision. The previous university run is historical and is not resumed. Source code is reused after tests; raw data, environments, caches, fitted representations and results are created afresh. Public Git stores source, configuration and status only. Private cluster identity, paths and logs stay outside Git.

See [Continuous execution](CONTINUOUS_EXECUTION.md) for the controller, bounded retries and exact implemented-versus-pending boundary. The assistant monitor was deleted at the user's request. Source-specific report/timing adjudications are recorded in `configs/dream_adjudication.yaml`; raw numeric labels alone are not admission to a scientific contrast.

## Phase mapping

| Phase | Scientific or technical output | Execution and dependency |
|---|---|---|
| R0 qualification | Exact source manifest, environment freeze, test report, CUDA smoke receipt | Bootstrap on login; bounded Slurm test job |
| R1 source audit | Versioned DREAM registry, open-access screening, constituent metadata | `scripts/alliance/resolve_registry.py`; metadata only |
| R2 acquisition | Original-source files, checksums, per-dataset status | `scripts/alliance/acquire.py`; bounded internet-connected worker |
| R3 inventory and QC | Official report/stage map, study overlap groups, observed channels, clean durations | Per-source adapters audited against the downloaded records; no inferred experience labels |
| R4 estimability | Recovery/bias/coverage at actual lengths and missingness | CPU Slurm; supported estimator choices recorded irrespective of condition effects |
| R5 measurement | Slow encoder and independent fast-track measurements, conventional features | CPU preprocessing then GPU frozen encoding; training-only fitted objects |
| R6 experience transfer | Study-held-out predictions, competing-model losses, axis/dimension comparisons | `statistics/study_transfer.py`; requires audited common feature table |
| R7 perturbation | Participant-held-out incremental TMS predictions | `predict_tms`; requires verified spontaneous/TMS links and conventional covariates |
| R8 specificity | Perception/report and psychedelic boundary analyses | Independent branches; report block-order and context limitations |
| R9 extensions | Limited clinical and modality-appropriate fMRI results | Optional; missing technical inputs do not block core results |
| R10 synthesis | All outcomes, uncertainty, provenance and figures | Assemble completed core and available extension evidence |

## Connection and storage

The local ignored `work/alliance-deploy` folder holds the interactive SSH bridge and its command queue. The password and Duo OTP are entered directly into SSH's console. The bridge is temporary; submitted Slurm jobs and detached acquisition processes have independent lifetimes. A sent command is not evidence of success: require its completion marker and inspect the remote artifact or job record before retrying.

Use a dedicated personal subdirectory of verified project storage for raw files and retained provenance, and a dedicated scratch directory for transient derivatives and caches. A new release is named by the hash of every allowlisted source file. `scripts/alliance/verify_release.py` checks that source before installation or jobs. Operational values are injected from private deployment scripts, never committed.

Load the recorded Python, CUDA, Arrow and Git Annex modules on Rorqual. PyTorch uses the Alliance wheelhouse. The final environment is recorded with `pip freeze`, module inventory and `pip check`. The existing university dependency lock is not claimed to describe this new environment. Compute nodes must use prestaged packages and model files; no internet downloads are attempted in GPU jobs.

For code-only repairs, `runtime.sh` can reuse the newly built Alliance environment after comparing the dependency specification, constraints and module configuration hashes. It explicitly selects the current release's source and verifies the import path. Qualification records both source and environment identities; it never silently claims a reused environment was rebuilt.

## Acquisition contract

The initial source budget is 100 GiB per dataset and 180 GiB total, with a 100 GiB filesystem headroom check. Shared-project quotas are checked separately before launch; filesystem free space is not a substitute for group quota. Expand only after a fresh quota check. Concurrency is one downloader, using a process lock and individual dataset locks. Failed sources remain explicit in a partial summary while other admitted sources continue.

The resolver records all directly downloadable supported DREAM candidates, including cohorts not suitable for the primary N2 contrast. Downloading them is not admission to analysis. Stage/report availability, age groups, interventions, montage, duration and participant overlap must be mapped from official files. Unknown-provider records remain unresolved and are not silently treated as absent. Public direct downloads with unresolved reuse terms retain their licence warning.

The new registry is created once per discovery directory; resumes use identical bytes and hashes. No credentials, gated data, or previously downloaded university-server data are accepted. Complete raw releases are checksum validated and made read-only. Partial files and failures are recoverable. Storage estimates for OpenNeuro are computed from the pinned Git Annex inventory before raw content is fetched.

## Analysis implementation boundary

Legacy acquisition, preprocessing, frozen encoding, metrics and controls remain reusable code, not evidence that the revised analysis has run. New nested study-transfer and incremental TMS regression routines are software-tested using synthetic inputs. Multi-cohort adapter integration, observed-length simulations, a separate fast-timescale driver and full empirical orchestration require the R3 data audit. They are not automatically replaced by the earlier 11-phase queue.

The new design does not auto-submit the legacy clinical/fMRI-dependent final figure graph. Downstream Slurm jobs must reference the new source digest, registry, audited input receipts and explicit phase. Statistical significance, direction of effect, model ranking and journal preference never determine whether an analysis executes.
