#!/usr/bin/env bash
# Read-only host/storage/runtime preflight.  Run on the remote server.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  note "Usage: $0 --canonical-root PATH --work-root PATH --checkpoint-root PATH"
}

canonical_root=""
work_root=""
checkpoint_root=""
while (($#)); do
  case "$1" in
    --canonical-root) canonical_root="${2:?missing value}"; shift 2 ;;
    --work-root) work_root="${2:?missing value}"; shift 2 ;;
    --checkpoint-root) checkpoint_root="${2:?missing value}"; shift 2 ;;
    --check-only) shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$canonical_root" && -n "$work_root" && -n "$checkpoint_root" ]] || {
  usage
  exit 2
}
validate_roots "$canonical_root" "$work_root" "$checkpoint_root"
validate_identity
require_parent_storage

note "identity=$(id -un)@$(hostname)"
note "os=$(awk -F= '$1 == "PRETTY_NAME" {gsub(/^"|"$/, "", $2); print $2}' /etc/os-release)"
note "canonical_parent=$CANONICAL_PARENT"
note "work_parent=$WORK_PARENT"
note "checkpoint_parent=$CHECKPOINT_PARENT"

for tool in git tmux rsync sha256sum python3 findmnt; do
  if command -v "$tool" >/dev/null 2>&1; then
    note "tool.$tool=$(command -v "$tool")"
  else
    note "tool.$tool=MISSING"
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
    | sed 's/^/gpu=/'
else
  note "gpu=MISSING_NVIDIA_SMI"
fi

for path in "$CANONICAL_PARENT" "$WORK_PARENT" "$CHECKPOINT_PARENT"; do
  df -Pk -- "$path" | awk -v target="$path" 'NR == 2 {print "disk=" target ",available_kib=" $4}'
  findmnt -n -o TARGET,SOURCE,FSTYPE -T "$path" | sed 's/^/mount=/'
done

for root in "$canonical_root" "$work_root" "$checkpoint_root"; do
  if [[ -d "$root" ]]; then
    note "project_root=$root,present=true"
  else
    note "project_root=$root,present=false"
  fi
done

if getent ahosts github.com >/dev/null 2>&1; then
  note "outbound_dns=ok"
else
  note "outbound_dns=failed"
fi

if command -v sbatch >/dev/null 2>&1 || command -v squeue >/dev/null 2>&1; then
  note "scheduler=slurm_detected_but_not_configured"
else
  note "scheduler=tmux (no Slurm commands detected)"
fi
