#!/usr/bin/env bash
# Content-addressed deployment from one exact GitHub commit.  Run on the server.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  note "Usage: $0 --canonical-root PATH --work-root PATH --checkpoint-root PATH --repository URL --commit SHA (--check-only|--dry-run|--apply)"
}

canonical_root=""
work_root=""
checkpoint_root=""
repository=""
commit=""
mode=""
while (($#)); do
  case "$1" in
    --canonical-root) canonical_root="${2:?missing value}"; shift 2 ;;
    --work-root) work_root="${2:?missing value}"; shift 2 ;;
    --checkpoint-root) checkpoint_root="${2:?missing value}"; shift 2 ;;
    --repository) repository="${2:?missing value}"; shift 2 ;;
    --commit) commit="${2:?missing value}"; shift 2 ;;
    --check-only) mode="check"; shift ;;
    --dry-run) mode="dry"; shift ;;
    --apply) mode="apply"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$canonical_root" && -n "$work_root" && -n "$checkpoint_root" ]] || die "all roots are required"
[[ -n "$repository" && -n "$commit" && -n "$mode" ]] || { usage; exit 2; }
validate_roots "$canonical_root" "$work_root" "$checkpoint_root"
validate_identity
require_command git
require_command sha256sum
[[ "$repository" == "https://github.com/pwa209/Neural-Manifolds.git" || \
   "$repository" == "https://github.com/pwa209/Neural-Manifolds" ]] || \
  die "repository must be the approved Neural-Manifolds GitHub repository"
[[ "$commit" =~ ^[0-9a-f]{40}$ || "$commit" =~ ^[0-9a-f]{64}$ ]] || \
  die "--commit must be an exact lowercase Git object id, not a branch or tag"

release="$work_root/source/releases/$commit"
if [[ "$mode" == "dry" ]]; then
  note "would_verify_repository=$repository"
  note "would_publish_release=$release"
  exit 0
fi

require_bootstrapped_roots "$canonical_root" "$work_root" "$checkpoint_root"
git ls-remote "$repository" | awk -v commit="$commit" '$1 == commit {found=1} END {exit !found}' || \
  die "exact commit is not currently advertised by an approved remote ref; use the verified archive transport"
if [[ "$mode" == "check" ]]; then
  note "check-only passed: identity, storage, repository reachability, advertised exact commit"
  note "release=$release"
  exit 0
fi

if [[ -d "$release" ]]; then
  actual="$(git -C "$release" rev-parse HEAD)"
  [[ "$actual" == "$commit" ]] || die "existing release has unexpected commit: $actual"
  [[ -s "$release/SOURCE_MANIFEST.sha256" ]] || die "existing release lacks source manifest"
else
  stage="$work_root/source/staging/${commit}.$(date -u +%Y%m%dT%H%M%SZ).$$"
  install -d -m 2770 -- "$stage"
  git -C "$stage" init --quiet
  git -C "$stage" remote add origin "$repository"
  git -C "$stage" fetch --quiet --depth 1 origin "$commit"
  git -C "$stage" checkout --quiet --detach FETCH_HEAD
  actual="$(git -C "$stage" rev-parse HEAD)"
  [[ "$actual" == "$commit" ]] || die "fetched commit mismatch: $actual"
  printf '{"schema_version":1,"repository":"%s","commit":"%s"}\n' "$repository" "$commit" \
    >"$stage/SOURCE_PROVENANCE.json.tmp"
  mv -fT -- "$stage/SOURCE_PROVENANCE.json.tmp" "$stage/SOURCE_PROVENANCE.json"
  (
    cd -- "$stage"
    {
      git ls-files -z
      printf '%s\0' SOURCE_PROVENANCE.json
    } | LC_ALL=C sort -z | xargs -0 sha256sum -- >SOURCE_MANIFEST.sha256.tmp
    mv -fT -- SOURCE_MANIFEST.sha256.tmp SOURCE_MANIFEST.sha256
    sha256sum --check --strict SOURCE_MANIFEST.sha256 >/dev/null
  )
  mv -- "$stage" "$release"
fi

link_tmp="$work_root/source/.current.$$.tmp"
ln -s -- "releases/$commit" "$link_tmp"
mv -fT -- "$link_tmp" "$work_root/source/current"
note "deployment_release=$release"
note "source_manifest=$release/SOURCE_MANIFEST.sha256"
note "active_source=$work_root/source/current"
