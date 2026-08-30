#!/usr/bin/env bash
# Check or launch the restartable phase queue in a named tmux session.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  note "Usage: $0 --canonical-root PATH --work-root PATH --checkpoint-root PATH --repo-root PATH --python PATH --run-id ID [--from-phase NAME|--through-phase NAME|--only-phase NAME] (--check-only|--dry-run|--apply)"
}

canonical_root=""; work_root=""; checkpoint_root=""; repo_root=""
python_bin=""; run_id=""; mode=""; from_phase=""; through_phase=""; only_phase=""
while (($#)); do
  case "$1" in
    --canonical-root) canonical_root="${2:?missing value}"; shift 2 ;;
    --work-root) work_root="${2:?missing value}"; shift 2 ;;
    --checkpoint-root) checkpoint_root="${2:?missing value}"; shift 2 ;;
    --repo-root) repo_root="${2:?missing value}"; shift 2 ;;
    --python) python_bin="${2:?missing value}"; shift 2 ;;
    --run-id) run_id="${2:?missing value}"; shift 2 ;;
    --from-phase) from_phase="${2:?missing value}"; shift 2 ;;
    --through-phase) through_phase="${2:?missing value}"; shift 2 ;;
    --only-phase) only_phase="${2:?missing value}"; shift 2 ;;
    --check-only) mode="check"; shift ;;
    --dry-run) mode="dry"; shift ;;
    --apply) mode="apply"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$canonical_root" && -n "$work_root" && -n "$checkpoint_root" && -n "$repo_root" ]] || die "all roots are required"
[[ -n "$python_bin" && -n "$run_id" && -n "$mode" ]] || { usage; exit 2; }
validate_roots "$canonical_root" "$work_root" "$checkpoint_root"
validate_run_id "$run_id"
validate_identity
[[ -x "$python_bin" ]] || die "Python executable is not executable: $python_bin"
cli_bin="$(dirname -- "$python_bin")/neural-manifolds"
[[ -x "$cli_bin" ]] || die "workflow CLI is not installed beside the selected Python: $cli_bin"
[[ -f "$repo_root/SOURCE_MANIFEST.sha256" ]] || die "deployed source manifest is missing"
[[ -z "$only_phase" || ( -z "$from_phase" && -z "$through_phase" ) ]] || \
  die "--only-phase cannot be combined with a range"
cd -- "$repo_root"
model_environment="$work_root/cache/models/model_paths.env"
if [[ -f "$model_environment" ]]; then
  # Generated exclusively by bootstrap_models.sh from a hash-verified manifest.
  # shellcheck disable=SC1090
  source "$model_environment"
fi

command=(
  "$python_bin" -m workflow.queue
  --repo-root "$repo_root"
  --canonical-root "$canonical_root"
  --work-root "$work_root"
  --checkpoint-root "$checkpoint_root"
  --run-id "$run_id"
  --cli "$cli_bin"
)
[[ -n "$from_phase" ]] && command+=(--from-phase "$from_phase")
[[ -n "$through_phase" ]] && command+=(--through-phase "$through_phase")
[[ -n "$only_phase" ]] && command+=(--only-phase "$only_phase")

if [[ "$mode" == "dry" ]]; then
  command+=(--dry-run)
  printf 'command=%s\n' "$(shell_join "${command[@]}")"
  "${command[@]}"
  exit 0
fi

require_bootstrapped_roots "$canonical_root" "$work_root" "$checkpoint_root"
if [[ "$mode" == "check" ]]; then
  command+=(--check-only)
  "${command[@]}"
  exit 0
fi

require_command tmux
session="neural-manifolds-$run_id"
if tmux has-session -t "=$session" 2>/dev/null; then
  note "queue_already_running=$session"
  tmux display-message -p -t "=$session" 'pane_pid=#{pane_pid}'
  exit 0
fi

state_root="$checkpoint_root/queue/$run_id"
install -d -m 2770 -- "$state_root"
tmux_log="$state_root/tmux.log"
command_string="$(shell_join "${command[@]}")"
printf -v quoted_log '%q' "$tmux_log"
tmux new-session -d -s "$session" "exec ${command_string} >>${quoted_log} 2>&1"
tmux has-session -t "=$session" || die "tmux session disappeared immediately; inspect $tmux_log"
pane_pid="$(tmux display-message -p -t "=$session" '#{pane_pid}')"
printf '{"schema_version":1,"session":"%s","pane_pid":%s,"log":"%s"}\n' \
  "$session" "$pane_pid" "$tmux_log" | atomic_write_from_stdin "$state_root/launch.json"
note "tmux_session=$session"
note "pane_pid=$pane_pid"
note "queue_log=$tmux_log"
note "state_root=$state_root"
