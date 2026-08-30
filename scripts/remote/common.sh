#!/usr/bin/env bash
# Shared safety primitives for the kemove research server.

set -euo pipefail
IFS=$'\n\t'

EXPECTED_HOSTNAME="kemove-Rack-Server"
EXPECTED_USER="wangpeng"
CANONICAL_PARENT="/private_nas/wangpeng"
WORK_PARENT="/data1/wangpeng"
CHECKPOINT_PARENT="/data2/wangpeng"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

note() {
  printf '%s\n' "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

validate_identity() {
  local actual_host actual_user
  actual_host="$(hostname)"
  actual_user="$(id -un)"
  [[ "$actual_host" == "$EXPECTED_HOSTNAME" ]] || \
    die "wrong host: expected $EXPECTED_HOSTNAME, got $actual_host"
  [[ "$actual_user" == "$EXPECTED_USER" ]] || \
    die "wrong user: expected $EXPECTED_USER, got $actual_user"
}

validate_run_id() {
  local run_id="$1"
  [[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{2,79}$ ]] || \
    die "unsafe run id: $run_id"
}

validate_project_root() {
  local kind="$1" value="$2" parent component
  case "$kind" in
    canonical) parent="$CANONICAL_PARENT" ;;
    work) parent="$WORK_PARENT" ;;
    checkpoint) parent="$CHECKPOINT_PARENT" ;;
    *) die "unknown root kind: $kind" ;;
  esac
  [[ "$value" == /* ]] || die "$kind root must be absolute"
  [[ "$value" != *".."* ]] || die "$kind root cannot contain '..'"
  [[ "$value" == "$parent/"* ]] || \
    die "$kind root must be one project-specific directory directly below $parent"
  component="${value#"$parent/"}"
  [[ "$component" != */* ]] || \
    die "$kind root must be one project-specific directory directly below $parent"
  [[ "$component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$ ]] || \
    die "unsafe project directory name in $kind root: $component"
  [[ "$value" != "$parent" ]] || die "refusing broad $kind root: $value"
}

validate_roots() {
  local canonical_root="$1" work_root="$2" checkpoint_root="$3"
  validate_project_root canonical "$canonical_root"
  validate_project_root work "$work_root"
  validate_project_root checkpoint "$checkpoint_root"
  [[ "$canonical_root" != "$work_root" ]] || die "roots must be distinct"
  [[ "$canonical_root" != "$checkpoint_root" ]] || die "roots must be distinct"
  [[ "$work_root" != "$checkpoint_root" ]] || die "roots must be distinct"
}

require_parent_storage() {
  local parent
  for parent in "$CANONICAL_PARENT" "$WORK_PARENT" "$CHECKPOINT_PARENT"; do
    [[ -d "$parent" ]] || die "required storage parent is unavailable: $parent"
    [[ -x "$parent" ]] || die "required storage parent is not searchable: $parent"
  done
}

require_bootstrapped_roots() {
  local root
  for root in "$1" "$2" "$3"; do
    [[ -d "$root" ]] || die "project root has not been bootstrapped: $root"
  done
}

atomic_write_from_stdin() {
  local target="$1" temporary
  temporary="${target}.tmp.$$"
  umask 0007
  command cat >"$temporary"
  command mv -fT -- "$temporary" "$target"
}

shell_join() {
  local quoted item output=""
  for item in "$@"; do
    printf -v quoted '%q' "$item"
    output+="${quoted} "
  done
  printf '%s' "$output"
}
