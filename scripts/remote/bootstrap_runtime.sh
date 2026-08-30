#!/usr/bin/env bash
# Build a content-addressed Python environment from a fully pinned lock file.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  note "Usage: $0 --canonical-root PATH --work-root PATH --checkpoint-root PATH --repo-root PATH --python PATH --requirements-lock PATH --lock-sha256 HASH (--check-only|--dry-run|--apply)"
}

canonical_root=""; work_root=""; checkpoint_root=""; repo_root=""
python_bin=""; lock_file=""; lock_sha256=""; mode=""
while (($#)); do
  case "$1" in
    --canonical-root) canonical_root="${2:?missing value}"; shift 2 ;;
    --work-root) work_root="${2:?missing value}"; shift 2 ;;
    --checkpoint-root) checkpoint_root="${2:?missing value}"; shift 2 ;;
    --repo-root) repo_root="${2:?missing value}"; shift 2 ;;
    --python) python_bin="${2:?missing value}"; shift 2 ;;
    --requirements-lock) lock_file="${2:?missing value}"; shift 2 ;;
    --lock-sha256) lock_sha256="${2:?missing value}"; shift 2 ;;
    --check-only) mode="check"; shift ;;
    --dry-run) mode="dry"; shift ;;
    --apply) mode="apply"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$canonical_root" && -n "$work_root" && -n "$checkpoint_root" && -n "$repo_root" ]] || die "all roots are required"
[[ -n "$python_bin" && -n "$lock_file" && -n "$lock_sha256" && -n "$mode" ]] || { usage; exit 2; }
validate_roots "$canonical_root" "$work_root" "$checkpoint_root"
validate_identity
[[ "$lock_sha256" =~ ^[0-9a-f]{64}$ ]] || die "invalid --lock-sha256"
[[ -x "$python_bin" ]] || die "Python executable is not executable: $python_bin"
[[ -d "$repo_root" ]] || die "deployed repository is missing: $repo_root"
[[ -f "$lock_file" ]] || die "requirements lock is missing: $lock_file"
actual_lock_sha="$(sha256sum "$lock_file" | awk '{print $1}')"
[[ "$actual_lock_sha" == "$lock_sha256" ]] || die "requirements lock hash mismatch"
[[ "$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]] || \
  die "runtime requires Python 3.11"

environment="$work_root/envs/$lock_sha256"
if [[ "$mode" == "dry" ]]; then
  note "would_create_environment=$environment"
  exit 0
fi
require_bootstrapped_roots "$canonical_root" "$work_root" "$checkpoint_root"
if [[ "$mode" == "check" ]]; then
  note "check-only passed: interpreter, lock hash, repository, and storage"
  [[ -x "$environment/bin/python" ]] && note "environment_present=true" || note "environment_present=false"
  exit 0
fi

if [[ ! -x "$environment/bin/python" ]]; then
  stage="$work_root/envs/.${lock_sha256}.$$.tmp"
  "$python_bin" -m venv "$stage"
  "$stage/bin/python" -m pip install \
    --require-hashes --no-deps --only-binary=:all: -r "$lock_file"
  "$stage/bin/python" -m pip install --no-deps --no-build-isolation "$repo_root"
  "$stage/bin/python" -m pytest --collect-only -q "$repo_root/tests" >/dev/null
  printf '%s\n' "$lock_sha256" >"$stage/LOCK_SHA256"
  mv -- "$stage" "$environment"
fi
[[ "$(<"$environment/LOCK_SHA256")" == "$lock_sha256" ]] || die "environment lock receipt mismatch"
note "runtime_python=$environment/bin/python"
