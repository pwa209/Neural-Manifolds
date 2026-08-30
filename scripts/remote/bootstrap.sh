#!/usr/bin/env bash
# Create only the explicitly approved project roots and fixed subdirectories.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  note "Usage: $0 --canonical-root PATH --work-root PATH --checkpoint-root PATH (--check-only|--dry-run|--apply)"
}

canonical_root=""
work_root=""
checkpoint_root=""
mode=""
while (($#)); do
  case "$1" in
    --canonical-root) canonical_root="${2:?missing value}"; shift 2 ;;
    --work-root) work_root="${2:?missing value}"; shift 2 ;;
    --checkpoint-root) checkpoint_root="${2:?missing value}"; shift 2 ;;
    --check-only) mode="check"; shift ;;
    --dry-run) mode="dry"; shift ;;
    --apply) mode="apply"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$canonical_root" && -n "$work_root" && -n "$checkpoint_root" && -n "$mode" ]] || {
  usage
  exit 2
}
validate_roots "$canonical_root" "$work_root" "$checkpoint_root"
validate_identity
require_parent_storage

directories=(
  "$canonical_root"
  "$canonical_root/raw"
  "$canonical_root/manifests"
  "$canonical_root/licences"
  "$canonical_root/snapshots"
  "$work_root"
  "$work_root/source/releases"
  "$work_root/source/staging"
  "$work_root/runs"
  "$work_root/cache"
  "$work_root/envs"
  "$work_root/scratch"
  "$checkpoint_root"
  "$checkpoint_root/queue"
  "$checkpoint_root/logs"
  "$checkpoint_root/snapshots"
)

if [[ "$mode" == "dry" ]]; then
  printf 'would_create=%s\n' "${directories[@]}"
  exit 0
fi

if [[ "$mode" == "check" ]]; then
  note "check-only passed: identity, explicit roots, and parent storage"
  for directory in "${directories[@]}"; do
    [[ -d "$directory" ]] && note "present=$directory" || note "would_create=$directory"
  done
  exit 0
fi

umask 0007
for directory in "${directories[@]}"; do
  install -d -m 2770 -- "$directory"
done

layout_receipt="$checkpoint_root/layout-v1.json"
if [[ -f "$layout_receipt" ]]; then
  grep -Fq "\"canonical_root\":\"$canonical_root\"" "$layout_receipt" || \
    die "existing layout receipt records different roots: $layout_receipt"
else
  printf '{"schema_version":1,"canonical_root":"%s","work_root":"%s","checkpoint_root":"%s","raw_root":"%s/raw"}\n' \
    "$canonical_root" "$work_root" "$checkpoint_root" "$canonical_root" \
    | atomic_write_from_stdin "$layout_receipt"
fi
note "bootstrap_complete=$layout_receipt"
