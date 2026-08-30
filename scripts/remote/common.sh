#!/usr/bin/env bash
# Shared safety primitives for an explicitly configured research server.

set -euo pipefail
IFS=$'\n\t'

EXPECTED_HOSTNAME=""
EXPECTED_USER=""
CANONICAL_PARENT=""
WORK_PARENT=""
CHECKPOINT_PARENT=""
CONFIGURED_CANONICAL_ROOT=""
CONFIGURED_WORK_ROOT=""
CONFIGURED_CHECKPOINT_ROOT=""
SERVER_CONFIG_CONTRACT_LOADED="false"

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

trim_yaml_scalar() {
  local value="$1"
  value="${value%%#*}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  if [[ "$value" =~ ^\"([^\"]*)\"$ ]]; then
    value="${BASH_REMATCH[1]}"
  elif [[ "$value" =~ ^\'([^\']*)\'$ ]]; then
    value="${BASH_REMATCH[1]}"
  fi
  REPLY="$value"
}

load_server_config_contract() {
  local config="$1" line section="" subsection="" raw key
  local expected_hostname="" expected_user=""
  local canonical_parent="" work_parent="" checkpoint_parent=""
  local configured_canonical_root="" configured_work_root="" configured_checkpoint_root=""
  [[ "$config" == /* && "$config" != *".."* ]] || \
    die "server config must be an unambiguous absolute path"
  [[ -f "$config" && ! -L "$config" ]] || \
    die "server config must be a regular non-symlink file: $config"

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    if [[ "$line" =~ ^[^[:space:]#] ]]; then
      subsection=""
      case "$line" in
        identity:) section="identity" ;;
        storage:) section="storage" ;;
        *) section="other" ;;
      esac
      continue
    fi
    if [[ "$section" == "storage" && "$line" =~ ^[[:space:]]{2}allowed_parent_mounts:[[:space:]]*$ ]]; then
      subsection="allowed_parent_mounts"
      continue
    fi
    if [[ "$subsection" == "allowed_parent_mounts" && "$line" =~ ^[[:space:]]{2}[^[:space:]] ]]; then
      subsection=""
    fi

    key=""
    raw=""
    if [[ "$section" == "identity" && "$line" =~ ^[[:space:]]{2}(expected_hostname|expected_user):[[:space:]]*(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      raw="${BASH_REMATCH[2]}"
    elif [[ "$section" == "storage" && -z "$subsection" && "$line" =~ ^[[:space:]]{2}(canonical_root|work_root|checkpoint_root):[[:space:]]*(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      raw="${BASH_REMATCH[2]}"
    elif [[ "$section" == "storage" && "$subsection" == "allowed_parent_mounts" && "$line" =~ ^[[:space:]]{4}(canonical|work|checkpoint):[[:space:]]*(.*)$ ]]; then
      key="${BASH_REMATCH[1]}_parent"
      raw="${BASH_REMATCH[2]}"
    else
      continue
    fi
    trim_yaml_scalar "$raw"
    [[ -n "$REPLY" ]] || die "server config leaves $key unresolved"
    case "$key" in
      expected_hostname)
        [[ -z "$expected_hostname" ]] || die "duplicate identity.expected_hostname"
        expected_hostname="$REPLY"
        ;;
      expected_user)
        [[ -z "$expected_user" ]] || die "duplicate identity.expected_user"
        expected_user="$REPLY"
        ;;
      canonical_root)
        [[ -z "$configured_canonical_root" ]] || die "duplicate storage.canonical_root"
        configured_canonical_root="$REPLY"
        ;;
      work_root)
        [[ -z "$configured_work_root" ]] || die "duplicate storage.work_root"
        configured_work_root="$REPLY"
        ;;
      checkpoint_root)
        [[ -z "$configured_checkpoint_root" ]] || die "duplicate storage.checkpoint_root"
        configured_checkpoint_root="$REPLY"
        ;;
      canonical_parent)
        [[ -z "$canonical_parent" ]] || die "duplicate canonical parent mount"
        canonical_parent="$REPLY"
        ;;
      work_parent)
        [[ -z "$work_parent" ]] || die "duplicate work parent mount"
        work_parent="$REPLY"
        ;;
      checkpoint_parent)
        [[ -z "$checkpoint_parent" ]] || die "duplicate checkpoint parent mount"
        checkpoint_parent="$REPLY"
        ;;
    esac
  done <"$config"

  [[ "$expected_hostname" =~ ^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$ ]] || \
    die "identity.expected_hostname is missing or unsafe"
  [[ "$expected_user" =~ ^[A-Za-z_][A-Za-z0-9._-]{0,63}$ ]] || \
    die "identity.expected_user is missing or unsafe"
  for raw in "$canonical_parent" "$work_parent" "$checkpoint_parent"; do
    [[ "$raw" == /* && "$raw" != "/" && "$raw" != *".."* && "$raw" != *[[:space:]]* ]] || \
      die "allowed parent mounts must be resolved, non-root absolute paths"
  done
  [[ "$canonical_parent" != "$work_parent" && "$canonical_parent" != "$checkpoint_parent" && "$work_parent" != "$checkpoint_parent" ]] || \
    die "canonical, work, and checkpoint parent mounts must be distinct"

  EXPECTED_HOSTNAME="$expected_hostname"
  EXPECTED_USER="$expected_user"
  CANONICAL_PARENT="$canonical_parent"
  WORK_PARENT="$work_parent"
  CHECKPOINT_PARENT="$checkpoint_parent"
  CONFIGURED_CANONICAL_ROOT="$configured_canonical_root"
  CONFIGURED_WORK_ROOT="$configured_work_root"
  CONFIGURED_CHECKPOINT_ROOT="$configured_checkpoint_root"
  SERVER_CONFIG_CONTRACT_LOADED="true"
  validate_project_root canonical "$CONFIGURED_CANONICAL_ROOT"
  validate_project_root work "$CONFIGURED_WORK_ROOT"
  validate_project_root checkpoint "$CONFIGURED_CHECKPOINT_ROOT"
}

ensure_server_config_contract() {
  local config repo_root
  [[ "$SERVER_CONFIG_CONTRACT_LOADED" == "true" ]] && return 0
  config="${NEURAL_MANIFOLDS_SERVER_CONFIG:-}"
  if [[ -z "$config" ]]; then
    [[ -n "${SCRIPT_DIR:-}" ]] || die "SCRIPT_DIR is required to locate the default server config"
    repo_root="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
    config="$repo_root/configs/server.yaml"
  fi
  load_server_config_contract "$config"
}

validate_identity() {
  local actual_host actual_user
  ensure_server_config_contract
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
  ensure_server_config_contract
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
  [[ "$canonical_root" == "$CONFIGURED_CANONICAL_ROOT" ]] || \
    die "canonical root does not match the selected server config"
  [[ "$work_root" == "$CONFIGURED_WORK_ROOT" ]] || \
    die "work root does not match the selected server config"
  [[ "$checkpoint_root" == "$CONFIGURED_CHECKPOINT_ROOT" ]] || \
    die "checkpoint root does not match the selected server config"
  [[ "$canonical_root" != "$work_root" ]] || die "roots must be distinct"
  [[ "$canonical_root" != "$checkpoint_root" ]] || die "roots must be distinct"
  [[ "$work_root" != "$checkpoint_root" ]] || die "roots must be distinct"
}

require_parent_storage() {
  local parent
  ensure_server_config_contract
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
