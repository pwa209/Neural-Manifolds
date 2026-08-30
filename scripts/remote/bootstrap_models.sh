#!/usr/bin/env bash
# Clone pinned model sources and materialise only phase-authorised checkpoints.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  note "Usage: $0 --canonical-root PATH --work-root PATH --checkpoint-root PATH --repo-root PATH --python PATH --stage core|fmri (--check-only|--dry-run|--apply)"
}

canonical_root=""; work_root=""; checkpoint_root=""; repo_root=""
python_bin=""; stage=""; mode=""
while (($#)); do
  case "$1" in
    --canonical-root) canonical_root="${2:?missing value}"; shift 2 ;;
    --work-root) work_root="${2:?missing value}"; shift 2 ;;
    --checkpoint-root) checkpoint_root="${2:?missing value}"; shift 2 ;;
    --repo-root) repo_root="${2:?missing value}"; shift 2 ;;
    --python) python_bin="${2:?missing value}"; shift 2 ;;
    --stage) stage="${2:?missing value}"; shift 2 ;;
    --check-only) mode="check"; shift ;;
    --dry-run) mode="dry"; shift ;;
    --apply) mode="apply"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$canonical_root" && -n "$work_root" && -n "$checkpoint_root" && -n "$repo_root" ]] || die "all roots are required"
[[ -n "$python_bin" && -n "$stage" && -n "$mode" ]] || { usage; exit 2; }
validate_roots "$canonical_root" "$work_root" "$checkpoint_root"
validate_identity
require_bootstrapped_roots "$canonical_root" "$work_root" "$checkpoint_root"
require_command git
[[ -x "$python_bin" ]] || die "Python executable is not executable: $python_bin"
[[ "$stage" == "core" || "$stage" == "fmri" ]] || die "--stage must be core or fmri"
[[ -f "$repo_root/configs/models.yaml" ]] || die "models config missing from deployed release"
[[ -f "$repo_root/scripts/remote/model_cache.py" ]] || die "model cache helper missing from deployed release"
case "$repo_root" in
  "$work_root/source/releases/"*) ;;
  *) die "repo root must be a content-addressed release below $work_root/source/releases" ;;
esac

note "BrainLM checkpoint policy: download is permitted only with --stage fmri and exact Hugging Face commit/file SHA-256 pins. Usage is CC-BY-NC-ND-4.0; no commercial use or derivative redistribution."
"$python_bin" "$repo_root/scripts/remote/model_cache.py" \
  --models "$repo_root/configs/models.yaml" \
  --work-root "$work_root" \
  --git "$(command -v git)" \
  --stage "$stage" \
  --mode "$mode"
