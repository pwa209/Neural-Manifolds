#!/usr/bin/env bash
# Read queue and tmux status without changing remote state.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  note "Usage: $0 --canonical-root PATH --work-root PATH --checkpoint-root PATH --repo-root PATH --python PATH --run-id ID [--json]"
}

canonical_root=""; work_root=""; checkpoint_root=""; repo_root=""
python_bin=""; run_id=""; json_flag=""
while (($#)); do
  case "$1" in
    --canonical-root) canonical_root="${2:?missing value}"; shift 2 ;;
    --work-root) work_root="${2:?missing value}"; shift 2 ;;
    --checkpoint-root) checkpoint_root="${2:?missing value}"; shift 2 ;;
    --repo-root) repo_root="${2:?missing value}"; shift 2 ;;
    --python) python_bin="${2:?missing value}"; shift 2 ;;
    --run-id) run_id="${2:?missing value}"; shift 2 ;;
    --json) json_flag="--json"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$canonical_root" && -n "$work_root" && -n "$checkpoint_root" && -n "$repo_root" ]] || die "all roots are required"
[[ -n "$python_bin" && -n "$run_id" ]] || { usage; exit 2; }
validate_roots "$canonical_root" "$work_root" "$checkpoint_root"
validate_run_id "$run_id"
validate_identity
[[ -d "$repo_root" ]] || die "deployed repository is missing: $repo_root"
[[ -x "$python_bin" ]] || die "Python executable is not executable: $python_bin"
cd -- "$repo_root"

session="neural-manifolds-$run_id"
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "=$session" 2>/dev/null; then
  note "tmux=$session,running=true"
else
  note "tmux=$session,running=false"
fi
command=(
  "$python_bin" -m workflow.queue
  --repo-root "$repo_root"
  --canonical-root "$canonical_root"
  --work-root "$work_root"
  --checkpoint-root "$checkpoint_root"
  --run-id "$run_id"
  --status
)
[[ -n "$json_flag" ]] && command+=("$json_flag")
"${command[@]}"
