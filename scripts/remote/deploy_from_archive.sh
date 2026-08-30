#!/usr/bin/env bash
# Content-addressed deployment from one hash-verified, server-local Git archive.

set -euo pipefail
IFS=$'\n\t'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  note "Usage: $0 --canonical-root PATH --work-root PATH --checkpoint-root PATH --repository URL --commit SHA --archive FILE --archive-sha256 SHA256 (--check-only|--dry-run|--apply)"
}

set_mode() {
  [[ -z "$mode" ]] || die "choose exactly one of --check-only, --dry-run, or --apply"
  mode="$1"
}

validate_archive_members() {
  python3 - "$1" <<'PY'
from __future__ import annotations

import re
import sys
import tarfile
from pathlib import PurePosixPath

archive_path = sys.argv[1]
reserved = {
    "SOURCE_MANIFEST.sha256",
    "SOURCE_PROVENANCE.json",
}
required = {
    "configs/study.yaml",
    "pyproject.toml",
    "workflow/queue.py",
}
seen: set[str] = set()
maximum_members = 100_000
maximum_file_bytes = 1 * 1024**3
maximum_total_bytes = 4 * 1024**3

try:
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
except (OSError, tarfile.TarError) as error:
    raise SystemExit(f"archive is not a readable uncompressed tar payload: {error}")

if not members:
    raise SystemExit("archive has no members")
if len(members) > maximum_members:
    raise SystemExit(f"archive has too many members: {len(members)} > {maximum_members}")

total_bytes = 0

for member in members:
    raw_name = member.name
    if not raw_name or "\\" in raw_name:
        raise SystemExit(f"archive member has an unsafe or ambiguous name: {raw_name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_name):
        raise SystemExit(f"archive member contains a control character: {raw_name!r}")
    name = raw_name[:-1] if member.isdir() and raw_name.endswith("/") else raw_name
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or re.match(r"^[A-Za-z]:", name)
        or name in {"", "."}
        or ".." in path.parts
        or "." in path.parts
        or str(path) != name
    ):
        raise SystemExit(f"archive member is absolute, traversing, or non-canonical: {raw_name!r}")
    normalised = str(path)
    if normalised in seen:
        raise SystemExit(f"archive repeats a member path: {normalised!r}")
    seen.add(normalised)
    if normalised in reserved:
        raise SystemExit(f"archive collides with deployment metadata: {normalised}")
    if member.issym() or member.islnk():
        raise SystemExit(f"archive links are forbidden: {normalised!r}")
    if member.issparse():
        raise SystemExit(f"archive sparse members are forbidden: {normalised!r}")
    if not member.isfile() and not member.isdir():
        raise SystemExit(f"archive special members are forbidden: {normalised!r}")
    if member.isfile():
        if member.size < 0 or member.size > maximum_file_bytes:
            raise SystemExit(f"archive member is too large: {normalised!r}")
        total_bytes += member.size
        if total_bytes > maximum_total_bytes:
            raise SystemExit("archive uncompressed payload exceeds the 4 GiB safety limit")
    if member.mode & 0o7000:
        raise SystemExit(f"archive member has set-id or sticky mode bits: {normalised!r}")

missing = sorted(required.difference(seen))
if missing:
    raise SystemExit(
        "archive is not an unprefixed Neural-Manifolds Git payload; missing "
        + ", ".join(missing)
    )
PY
}

verify_extracted_payload() {
  python3 - "$1" "$2" <<'PY'
from __future__ import annotations

import hashlib
import os
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath

archive_path = sys.argv[1]
root = Path(sys.argv[2]).resolve(strict=True)
expected: dict[str, tuple[str, bool]] = {}
with tarfile.open(archive_path, mode="r:") as archive:
    for member in archive.getmembers():
        if not member.isfile():
            continue
        stream = archive.extractfile(member)
        if stream is None:
            raise SystemExit(f"cannot read archived regular file: {member.name!r}")
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        expected[str(PurePosixPath(member.name))] = (
            digest.hexdigest(),
            bool(member.mode & 0o111),
        )

actual: dict[str, tuple[str, bool]] = {}
for candidate in root.rglob("*"):
    relative = candidate.relative_to(root).as_posix()
    if candidate.is_symlink():
        raise SystemExit(f"extracted payload contains a link: {relative!r}")
    if candidate.is_dir():
        continue
    if not candidate.is_file():
        raise SystemExit(f"extracted payload contains a special file: {relative!r}")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual[relative] = (
        digest.hexdigest(),
        bool(candidate.stat().st_mode & stat.S_IXUSR),
    )

if actual != expected:
    missing = sorted(set(expected).difference(actual))
    extra = sorted(set(actual).difference(expected))
    changed = sorted(
        name for name in set(actual).intersection(expected) if actual[name] != expected[name]
    )
    raise SystemExit(
        "extracted payload differs from the validated Git archive "
        f"(missing={missing}, extra={extra}, changed={changed})"
    )
PY
}

write_provenance() {
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
from __future__ import annotations

import json
import os
import sys

repository, commit, archive_path, archive_sha256, destination = sys.argv[1:]
payload = {
    "schema_version": 1,
    "repository": repository,
    "commit": commit,
    "transport": {
        "type": "server_local_git_archive",
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
    },
}
with open(destination, "x", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
}

manifest_stream() {
  local root="$1"
  (
    cd -- "$root"
    find . -xdev -type f ! -path './SOURCE_MANIFEST.sha256' -printf '%P\0' |
      LC_ALL=C sort -z |
      xargs -0 -r sha256sum --
  )
}

verify_release() {
  local root="$1" provenance="$1/SOURCE_PROVENANCE.json"
  [[ -d "$root" && ! -L "$root" ]] || die "release is not a regular directory: $root"
  [[ -f "$provenance" && ! -L "$provenance" ]] || \
    die "existing release lacks regular SOURCE_PROVENANCE.json"
  [[ -f "$root/SOURCE_MANIFEST.sha256" && ! -L "$root/SOURCE_MANIFEST.sha256" ]] || \
    die "existing release lacks regular SOURCE_MANIFEST.sha256"
  python3 - "$root" "$repository" "$commit" "$expected_archive_sha256" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
repository, commit, expected_archive_sha256 = sys.argv[2:]
for candidate in root.rglob("*"):
    relative = candidate.relative_to(root).as_posix()
    if candidate.is_symlink() or (not candidate.is_dir() and not candidate.is_file()):
        raise SystemExit(f"release contains a link or special file: {relative!r}")
with (root / "SOURCE_PROVENANCE.json").open("r", encoding="utf-8") as stream:
    payload = json.load(stream)
transport = payload.get("transport")
if (
    payload.get("schema_version") != 1
    or payload.get("repository") != repository
    or payload.get("commit") != commit
    or not isinstance(transport, dict)
    or transport.get("type") != "server_local_git_archive"
    or transport.get("archive_sha256") != expected_archive_sha256
):
    raise SystemExit("existing release provenance differs from the requested archive deployment")
PY
  (
    cd -- "$root"
    sha256sum --check --strict SOURCE_MANIFEST.sha256 >/dev/null
  )
  manifest_stream "$root" | cmp -s -- - "$root/SOURCE_MANIFEST.sha256" || \
    die "existing release manifest does not exactly cover its regular files"
}

canonical_root=""
work_root=""
checkpoint_root=""
repository=""
commit=""
archive_input=""
expected_archive_sha256=""
mode=""
while (($#)); do
  case "$1" in
    --canonical-root) canonical_root="${2:?missing value}"; shift 2 ;;
    --work-root) work_root="${2:?missing value}"; shift 2 ;;
    --checkpoint-root) checkpoint_root="${2:?missing value}"; shift 2 ;;
    --repository) repository="${2:?missing value}"; shift 2 ;;
    --commit) commit="${2:?missing value}"; shift 2 ;;
    --archive) archive_input="${2:?missing value}"; shift 2 ;;
    --archive-sha256) expected_archive_sha256="${2:?missing value}"; shift 2 ;;
    --check-only) set_mode check; shift ;;
    --dry-run) set_mode dry; shift ;;
    --apply) set_mode apply; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$canonical_root" && -n "$work_root" && -n "$checkpoint_root" ]] || \
  die "all roots are required"
[[ -n "$repository" && -n "$commit" && -n "$archive_input" ]] || { usage; exit 2; }
[[ -n "$expected_archive_sha256" && -n "$mode" ]] || { usage; exit 2; }
validate_roots "$canonical_root" "$work_root" "$checkpoint_root"
validate_identity
for command_name in cmp date find git install ln mv python3 readlink rm sha256sum sort tar xargs; do
  require_command "$command_name"
done
[[ "$repository" == "https://github.com/pwa209/Neural-Manifolds.git" || \
   "$repository" == "https://github.com/pwa209/Neural-Manifolds" ]] || \
  die "repository must be the approved Neural-Manifolds GitHub repository"
[[ "$commit" =~ ^[0-9a-f]{40}$ || "$commit" =~ ^[0-9a-f]{64}$ ]] || \
  die "--commit must be an exact lowercase Git object id, not a branch or tag"
[[ "$expected_archive_sha256" =~ ^[0-9a-f]{64}$ ]] || \
  die "--archive-sha256 must be an exact lowercase SHA-256"
[[ -f "$archive_input" && ! -L "$archive_input" ]] || \
  die "--archive must name a server-local regular file, not a link"
archive="$(readlink -f -- "$archive_input")"
[[ -n "$archive" && -f "$archive" ]] || die "cannot resolve archive: $archive_input"
IFS=' ' read -r actual_archive_sha256 _ < <(sha256sum -- "$archive")
[[ "$actual_archive_sha256" == "$expected_archive_sha256" ]] || \
  die "archive SHA-256 mismatch: expected $expected_archive_sha256, got $actual_archive_sha256"
archive_commit="$(git get-tar-commit-id <"$archive" 2>/dev/null)" || \
  die "archive lacks Git commit metadata; create it with git archive --format=tar"
[[ "$archive_commit" == "$commit" ]] || \
  die "archive commit mismatch: expected $commit, got $archive_commit"
validate_archive_members "$archive"

release="$work_root/source/releases/$commit"
if [[ "$mode" == "dry" ]]; then
  note "validated_archive=$archive"
  note "validated_archive_sha256=$actual_archive_sha256"
  note "validated_archive_commit=$archive_commit"
  note "would_publish_release=$release"
  note "would_activate=$work_root/source/current"
  exit 0
fi

require_bootstrapped_roots "$canonical_root" "$work_root" "$checkpoint_root"
[[ -d "$work_root/source/releases" && -d "$work_root/source/staging" ]] || \
  die "source release/staging roots are missing; run bootstrap.sh --apply first"
if [[ -e "$release" || -L "$release" ]]; then
  verify_release "$release"
  release_status="existing_verified"
else
  release_status="absent"
fi
if [[ "$mode" == "check" ]]; then
  note "check-only passed: identity, storage, archive hash, member safety, exact commit"
  note "release=$release"
  note "release_status=$release_status"
  exit 0
fi

stage=""
archive_copy=""
provenance_tmp=""
manifest_tmp=""
link_tmp=""
cleanup_temporary_files() {
  local path
  for path in "$archive_copy" "$provenance_tmp" "$manifest_tmp"; do
    if [[ -n "$path" && "$path" == "$work_root/source/staging/${commit}."* ]]; then
      rm -f -- "$path"
    fi
  done
  if [[ -n "$link_tmp" && "$link_tmp" == "$work_root/source/.current."* ]]; then
    rm -f -- "$link_tmp"
  fi
  if [[ -n "$stage" && -d "$stage" ]]; then
    printf 'preserved_incomplete_staging=%s\n' "$stage" >&2
  fi
}
trap cleanup_temporary_files EXIT

if [[ "$release_status" == "absent" ]]; then
  stage="$work_root/source/staging/${commit}.$(date -u +%Y%m%dT%H%M%SZ).$$"
  archive_copy="${stage}.archive.tar"
  provenance_tmp="${stage}.provenance.tmp"
  manifest_tmp="${stage}.manifest.tmp"
  install -d -m 2770 -- "$stage"
  install -m 0440 -- "$archive" "$archive_copy"
  IFS=' ' read -r copied_archive_sha256 _ < <(sha256sum -- "$archive_copy")
  [[ "$copied_archive_sha256" == "$expected_archive_sha256" ]] || \
    die "staged archive copy failed SHA-256 validation"
  copied_commit="$(git get-tar-commit-id <"$archive_copy" 2>/dev/null)" || \
    die "staged archive copy lost Git commit metadata"
  [[ "$copied_commit" == "$commit" ]] || die "staged archive commit changed"
  validate_archive_members "$archive_copy"
  tar --extract --file "$archive_copy" --directory "$stage" \
    --no-same-owner --no-same-permissions --delay-directory-restore
  verify_extracted_payload "$archive_copy" "$stage"
  rm -f -- "$archive_copy"
  archive_copy=""
  write_provenance \
    "$repository" "$commit" "$archive" "$expected_archive_sha256" "$provenance_tmp"
  mv -fT -- "$provenance_tmp" "$stage/SOURCE_PROVENANCE.json"
  provenance_tmp=""
  manifest_stream "$stage" >"$manifest_tmp"
  (
    cd -- "$stage"
    sha256sum --check --strict "$manifest_tmp" >/dev/null
  )
  mv -fT -- "$manifest_tmp" "$stage/SOURCE_MANIFEST.sha256"
  manifest_tmp=""
  verify_release "$stage"
  if mv -T -- "$stage" "$release"; then
    stage=""
  elif [[ -d "$release" && ! -L "$release" ]]; then
    verify_release "$release"
    note "another deployment published the same verified release first"
  else
    die "could not atomically publish release: $release"
  fi
fi

current="$work_root/source/current"
[[ ! -e "$current" || -L "$current" ]] || \
  die "refusing to replace non-symlink source/current: $current"
link_tmp="$work_root/source/.current.$$.tmp"
[[ ! -e "$link_tmp" && ! -L "$link_tmp" ]] || die "temporary current link already exists"
ln -s -- "releases/$commit" "$link_tmp"
mv -fT -- "$link_tmp" "$current"
link_tmp=""
note "deployment_release=$release"
note "source_manifest=$release/SOURCE_MANIFEST.sha256"
note "source_provenance=$release/SOURCE_PROVENANCE.json"
note "active_source=$current"
