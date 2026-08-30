"""Repository-specific acquisition providers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .http import HttpClient, public_url, safe_filename
from .models import DatasetSpec


class ProviderError(RuntimeError):
    """An upstream provider could not be acquired safely."""


class AccessBlocked(ProviderError):
    """Upstream terms or authentication require user action."""


@dataclass(frozen=True)
class RemoteFile:
    relative_path: str
    download_url: str
    size: int | None
    hashes: dict[str, str]
    source_id: str | None = None

    def provenance(self) -> dict[str, Any]:
        value = asdict(self)
        value["download_url"] = public_url(self.download_url)
        return value


@dataclass(frozen=True)
class Discovery:
    files: tuple[RemoteFile, ...]
    metadata: dict[str, Any]

    @property
    def total_known_bytes(self) -> int | None:
        if any(item.size is None for item in self.files):
            return None
        return sum(item.size or 0 for item in self.files)


def safe_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/").lstrip("/")
    path = PurePosixPath(normalized)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ProviderError(f"unsafe remote path: {value!r}")
    if path.is_absolute() or "\x00" in normalized:
        raise ProviderError(f"unsafe remote path: {value!r}")
    return path.as_posix()


def _extract_hashes(value: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    nested = value.get("hashes")
    if isinstance(nested, dict):
        for key, digest in nested.items():
            if isinstance(digest, str) and key.lower() in {"md5", "sha1", "sha256"}:
                hashes[key.lower()] = digest
    for key in (
        "md5",
        "sha1",
        "sha256",
        "checksum",
        "sha256_hash",
        "md5_hash",
        "computed_md5",
        "supplied_md5",
    ):
        digest = value.get(key)
        if not isinstance(digest, str):
            continue
        algorithm = "sha256" if "sha256" in key else "md5" if "md5" in key else key
        if algorithm in {"md5", "sha1", "sha256"}:
            hashes[algorithm] = digest
    return hashes


class Provider:
    def __init__(self, dataset: DatasetSpec, client: HttpClient) -> None:
        self.dataset = dataset
        self.client = client

    def check(self) -> dict[str, Any]:
        raise NotImplementedError

    def materialize(self, staging: Path) -> dict[str, Any]:
        raise NotImplementedError


class RemoteFileProvider(Provider):
    def discover(self) -> Discovery:
        raise NotImplementedError

    def check(self) -> dict[str, Any]:
        discovery = self.discover()
        return {
            "status": "ready",
            "provider": self.dataset.source.provider,
            "file_count": len(discovery.files),
            "total_known_bytes": discovery.total_known_bytes,
            "metadata": discovery.metadata,
        }

    def materialize(self, staging: Path) -> dict[str, Any]:
        discovery = self.discover()
        if not discovery.files:
            raise ProviderError(f"provider returned no files for {self.dataset.id}")
        staging.mkdir(parents=True, exist_ok=True)
        total = discovery.total_known_bytes
        if total is not None:
            free = shutil.disk_usage(staging).free
            required = int(total * 1.02)
            if free < required:
                raise ProviderError(
                    f"insufficient free space for {self.dataset.id}: {free} < {required} bytes"
                )
        seen: set[str] = set()
        transfers: list[dict[str, Any]] = []
        for remote in discovery.files:
            relative = safe_relative_path(remote.relative_path)
            if relative in seen:
                raise ProviderError(f"duplicate remote path: {relative}")
            seen.add(relative)
            target = staging / Path(*PurePosixPath(relative).parts)
            transfer = self.client.download(
                remote.download_url,
                target,
                expected_size=remote.size,
                expected_hashes=remote.hashes,
            )
            transfer["relative_path"] = relative
            transfers.append(transfer)
        return {
            "provider": self.dataset.source.provider,
            "accession": self.dataset.source.accession,
            "version": self.dataset.source.version,
            "metadata": discovery.metadata,
            "remote_files": [item.provenance() for item in discovery.files],
            "transfers": transfers,
        }


def _normalise_license(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


class FigshareProvider(RemoteFileProvider):
    def discover(self) -> Discovery:
        api_url = self.dataset.source.api_url
        if api_url is None:
            raise ProviderError("Figshare API URL is missing")
        payload = self.client.get_json(api_url)
        if not isinstance(payload, dict):
            raise ProviderError("Figshare version API returned a non-object")
        returned_version = payload.get("version")
        if returned_version is not None and str(returned_version) != self.dataset.source.version:
            raise ProviderError(
                f"Figshare version mismatch: {returned_version} != {self.dataset.source.version}"
            )
        files = payload.get("files")
        if not isinstance(files, list):
            files = self.client.get_json(f"{api_url.rstrip('/')}/files")
        if not isinstance(files, list):
            raise ProviderError("Figshare version API did not return a file list")
        license_value = payload.get("license")
        license_name = ""
        if isinstance(license_value, dict):
            license_name = str(license_value.get("name") or license_value.get("url") or "")
        elif isinstance(license_value, str):
            license_name = license_value
        expected_license = self.dataset.license.spdx
        if (
            license_name
            and expected_license != "NOASSERTION"
            and _normalise_license(expected_license) not in _normalise_license(license_name)
            and _normalise_license(license_name) not in _normalise_license(expected_license)
        ):
            raise ProviderError(
                f"Figshare licence mismatch: registry={expected_license!r}, API={license_name!r}"
            )
        remote_files: list[RemoteFile] = []
        for item in files:
            if not isinstance(item, dict):
                raise ProviderError("malformed Figshare file entry")
            url = item.get("download_url")
            name = item.get("name")
            if not isinstance(url, str) or not isinstance(name, str):
                raise ProviderError("Figshare file is missing name or download_url")
            size = item.get("size")
            remote_files.append(
                RemoteFile(
                    relative_path=safe_filename(name),
                    download_url=url,
                    size=int(size) if isinstance(size, int | float) else None,
                    hashes=_extract_hashes(item),
                    source_id=str(item.get("id")) if item.get("id") is not None else None,
                )
            )
        metadata = {
            "api_url": public_url(api_url),
            "article_id": payload.get("id") or self.dataset.source.accession,
            "version": returned_version or self.dataset.source.version,
            "doi": payload.get("doi") or self.dataset.source.doi,
            "title": payload.get("title") or self.dataset.title,
            "license": license_name or expected_license,
            "published_date": payload.get("published_date"),
            "modified_date": payload.get("modified_date"),
        }
        return Discovery(files=tuple(remote_files), metadata=metadata)


def _mendeley_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "entries", "files", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ProviderError("Mendeley public API returned an unsupported file-list shape")


class MendeleyProvider(RemoteFileProvider):
    def discover(self) -> Discovery:
        api_url = self.dataset.source.api_url
        if api_url is None:
            raise ProviderError("Mendeley API URL is missing")
        version = self.dataset.source.version
        metadata_payload = self.client.get_json(api_url, params={"version": version})
        if not isinstance(metadata_payload, dict):
            raise ProviderError("Mendeley metadata API returned a non-object")
        if str(metadata_payload.get("version")) != version:
            raise ProviderError(
                f"Mendeley version mismatch: {metadata_payload.get('version')} != {version}"
            )
        licence = metadata_payload.get("data_licence")
        licence = licence if isinstance(licence, dict) else {}
        observed_licence = str(licence.get("short_name") or licence.get("full_name") or "")
        if observed_licence and not (
            _normalise_license(self.dataset.license.spdx) in _normalise_license(observed_licence)
            or _normalise_license(observed_licence) in _normalise_license(self.dataset.license.spdx)
        ):
            raise ProviderError(
                f"Mendeley licence mismatch: registry={self.dataset.license.spdx!r}, "
                f"API={observed_licence!r}"
            )
        folders_payload = self.client.get_json(f"{api_url.rstrip('/')}/folders/{version}")
        folders = _mendeley_entries(folders_payload)
        folders_by_id = {
            str(folder.get("id")): folder for folder in folders if folder.get("id") is not None
        }

        def folder_path(folder_id: str, trail: frozenset[str] = frozenset()) -> str:
            if folder_id in trail:
                raise ProviderError("Mendeley folder cycle detected")
            try:
                folder = folders_by_id[folder_id]
            except KeyError as error:
                raise ProviderError(f"unknown Mendeley folder id: {folder_id}") from error
            name_value = folder.get("name")
            if not isinstance(name_value, str):
                raise ProviderError("Mendeley folder is missing a name")
            name = safe_filename(name_value)
            parent = folder.get("parent_id")
            if parent:
                return safe_relative_path(f"{folder_path(str(parent), trail | {folder_id})}/{name}")
            return name

        locations = [("root", "")]
        locations.extend((folder_id, folder_path(folder_id)) for folder_id in folders_by_id)
        remote_files: list[RemoteFile] = []
        file_api_url = f"{api_url.rstrip('/')}/files"
        for folder_id, prefix in locations:
            start = 0
            while True:
                payload = self.client.get_json(
                    file_api_url,
                    params={
                        "folder_id": folder_id,
                        "version": version,
                        "$start": start,
                        "$limit": 100,
                    },
                )
                entries = _mendeley_entries(payload)
                for item in entries:
                    if str(item.get("status", "COMPLETED")).upper() != "COMPLETED":
                        raise ProviderError("Mendeley returned an incomplete file record")
                    name_value = item.get("name") or item.get("filename") or item.get("file_name")
                    if not isinstance(name_value, str):
                        raise ProviderError("Mendeley file is missing a name")
                    name = safe_filename(name_value)
                    details = item.get("content_details")
                    details = details if isinstance(details, dict) else {}
                    url = details.get("download_url") or item.get("download_url")
                    if not isinstance(url, str):
                        raise ProviderError(f"Mendeley file {name!r} is missing download_url")
                    size = item.get("size") or details.get("size")
                    hashes = _extract_hashes(item)
                    hashes.update(_extract_hashes(details))
                    remote_files.append(
                        RemoteFile(
                            relative_path=safe_relative_path(f"{prefix}/{name}"),
                            download_url=url,
                            size=int(size) if isinstance(size, int | float) else None,
                            hashes=hashes,
                            source_id=str(item.get("id")) if item.get("id") else None,
                        )
                    )
                if len(entries) < 100:
                    break
                start += len(entries)
        metadata = {
            "api_url": public_url(api_url),
            "accession": self.dataset.source.accession,
            "version": version,
            "doi": metadata_payload.get("doi", {}).get("id")
            if isinstance(metadata_payload.get("doi"), dict)
            else self.dataset.source.doi,
            "license": observed_licence or self.dataset.license.spdx,
            "published_date": metadata_payload.get("publish_date"),
            "folders_visited": len(locations),
        }
        return Discovery(files=tuple(remote_files), metadata=metadata)


def _next_link(payload: dict[str, Any]) -> str | None:
    links = payload.get("links")
    if not isinstance(links, dict):
        return None
    value = links.get("next")
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("href"), str):
        return value["href"]
    return None


def _related_files_url(item: dict[str, Any]) -> str | None:
    relationships = item.get("relationships")
    if not isinstance(relationships, dict):
        return None
    files = relationships.get("files")
    if not isinstance(files, dict):
        return None
    links = files.get("links")
    if not isinstance(links, dict):
        return None
    related = links.get("related")
    if isinstance(related, str):
        return related
    if isinstance(related, dict) and isinstance(related.get("href"), str):
        return related["href"]
    return None


class OsfProvider(RemoteFileProvider):
    def _pages(self, first_url: str) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        next_url: str | None = first_url
        seen: set[str] = set()
        while next_url:
            if next_url in seen:
                raise ProviderError("OSF pagination loop detected")
            seen.add(next_url)
            payload = self.client.get_json(next_url)
            if not isinstance(payload, dict):
                raise ProviderError("OSF API returned a non-object")
            pages.append(payload)
            next_url = _next_link(payload)
        return pages

    def discover(self) -> Discovery:
        api_url = self.dataset.source.api_url
        if api_url is None:
            raise ProviderError("OSF API URL is missing")
        node_metadata: dict[str, Any] = {}
        if self.dataset.source.metadata_url:
            node_payload = self.client.get_json(self.dataset.source.metadata_url)
            if not isinstance(node_payload, dict) or not isinstance(node_payload.get("data"), dict):
                raise ProviderError("OSF node metadata API returned an unsupported shape")
            node_data = node_payload["data"]
            if str(node_data.get("id")) != self.dataset.source.accession:
                raise ProviderError("OSF node id differs from the registry accession")
            attributes = node_data.get("attributes")
            attributes = attributes if isinstance(attributes, dict) else {}
            if attributes.get("public") is not True:
                raise AccessBlocked(f"OSF node {self.dataset.source.accession} is not public")
            node_metadata = {
                "title": attributes.get("title"),
                "date_created": attributes.get("date_created"),
                "date_modified": attributes.get("date_modified"),
                "registration": attributes.get("registration"),
                "node_license": attributes.get("node_license"),
            }
        provider_pages = self._pages(api_url)
        roots: list[str] = []
        for page in provider_pages:
            data = page.get("data")
            if not isinstance(data, list):
                raise ProviderError("OSF provider response is missing data")
            for provider in data:
                if isinstance(provider, dict):
                    related = _related_files_url(provider)
                    if related:
                        roots.append(related)
        if not roots:
            raise ProviderError("OSF project exposes no public storage providers")
        queue = list(roots)
        visited: set[str] = set()
        remote_files: list[RemoteFile] = []
        while queue:
            listing_url = queue.pop(0)
            if listing_url in visited:
                continue
            visited.add(listing_url)
            for page in self._pages(listing_url):
                entries = page.get("data")
                if not isinstance(entries, list):
                    raise ProviderError("OSF file response is missing data")
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    attributes = item.get("attributes")
                    attributes = attributes if isinstance(attributes, dict) else {}
                    kind = attributes.get("kind")
                    if kind == "folder":
                        related = _related_files_url(item)
                        if related:
                            queue.append(related)
                        continue
                    if kind != "file":
                        continue
                    links = item.get("links")
                    links = links if isinstance(links, dict) else {}
                    url = links.get("download")
                    if not isinstance(url, str):
                        raise ProviderError("OSF file is missing a download link")
                    path_value = attributes.get("materialized_path") or attributes.get("name")
                    if not isinstance(path_value, str):
                        raise ProviderError("OSF file is missing its materialized path")
                    extra = attributes.get("extra")
                    extra = extra if isinstance(extra, dict) else {}
                    size = attributes.get("size")
                    remote_files.append(
                        RemoteFile(
                            relative_path=safe_relative_path(path_value),
                            download_url=url,
                            size=int(size) if isinstance(size, int | float) else None,
                            hashes=_extract_hashes(extra),
                            source_id=str(item.get("id")) if item.get("id") else None,
                        )
                    )
        metadata = {
            "api_url": public_url(api_url),
            "project_id": self.dataset.source.accession,
            "version_label": self.dataset.source.version,
            "doi": self.dataset.source.doi,
            "mutable_upstream": True,
            "storage_roots": [public_url(url) for url in roots],
            "listings_visited": len(visited),
            "node": node_metadata,
            "license": self.dataset.license.spdx,
        }
        return Discovery(files=tuple(remote_files), metadata=metadata)


URL_IN_OUTPUT = re.compile(r"https?://[^\s'\"<>]+")


def _safe_output_tail(value: str) -> str:
    tail = value[-4000:]
    tail = URL_IN_OUTPUT.sub(lambda match: public_url(match.group(0)), tail)
    return re.sub(
        r"(?i)(authorization\s*[:=]\s*)(bearer|basic)\s+\S+",
        r"\1[REDACTED]",
        tail,
    )


class CommandRunner:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 3600,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        self.records.append(
            {
                "command": command,
                "cwd": str(cwd) if cwd else None,
                "returncode": result.returncode,
                "stdout_tail": _safe_output_tail(result.stdout),
                "stderr_tail": _safe_output_tail(result.stderr),
            }
        )
        if check and result.returncode != 0:
            raise ProviderError(
                f"command failed ({result.returncode}): {' '.join(command[:3])}; "
                f"stderr tail: {result.stderr[-1000:]}"
            )
        return result


class OpenNeuroProvider(Provider):
    REQUIRED_TOOLS = ("git", "git-annex", "datalad")

    def _require_tools(self) -> dict[str, str]:
        found: dict[str, str] = {}
        for tool in self.REQUIRED_TOOLS:
            path = shutil.which(tool)
            if path is None:
                raise ProviderError(f"{tool} is required for version-pinned OpenNeuro acquisition")
            found[tool] = path
        return found

    def check(self) -> dict[str, Any]:
        tools = self._require_tools()
        repository = self.dataset.source.repository_url
        if repository is None:
            raise ProviderError("OpenNeuro repository URL is missing")
        runner = CommandRunner()
        tag = f"refs/tags/{self.dataset.source.version}"
        result = runner.run(
            ["git", "ls-remote", "--exit-code", "--tags", repository, tag],
            timeout=120,
        )
        if not result.stdout.strip():
            raise ProviderError(f"OpenNeuro tag is not available: {tag}")
        observed_revision = result.stdout.split()[0]
        if observed_revision != self.dataset.source.revision:
            raise ProviderError(
                f"OpenNeuro tag revision differs from registry: "
                f"{observed_revision} != {self.dataset.source.revision}"
            )
        return {
            "status": "ready",
            "provider": "openneuro",
            "tools": tools,
            "tag": tag,
            "revision": observed_revision,
            "repository": repository,
            "commands": runner.records,
        }

    def materialize(self, staging: Path) -> dict[str, Any]:
        self._require_tools()
        repository = self.dataset.source.repository_url
        if repository is None:
            raise ProviderError("OpenNeuro repository URL is missing")
        runner = CommandRunner()
        if not (staging / ".git").is_dir():
            if staging.exists() and any(staging.iterdir()):
                raise ProviderError(f"non-DataLad content already exists in staging: {staging}")
            if staging.exists():
                staging.rmdir()
            runner.run(["datalad", "clone", repository, str(staging)], timeout=3600)
        origin = runner.run(
            ["git", "config", "--get", "remote.origin.url"], cwd=staging, timeout=60
        ).stdout.strip()
        if origin.rstrip("/") != repository.rstrip("/"):
            raise ProviderError(f"staging origin mismatch: {origin!r} != {repository!r}")
        runner.run(["git", "fetch", "--force", "--tags", "origin"], cwd=staging, timeout=600)
        tag = self.dataset.source.version
        runner.run(["git", "checkout", "--detach", tag], cwd=staging, timeout=300)
        head = runner.run(["git", "rev-parse", "HEAD"], cwd=staging, timeout=60).stdout.strip()
        tag_commit = runner.run(
            ["git", "rev-list", "-n", "1", tag], cwd=staging, timeout=60
        ).stdout.strip()
        if head != tag_commit:
            raise ProviderError(f"checked-out commit does not match OpenNeuro tag {tag}")
        if head != self.dataset.source.revision:
            raise ProviderError(
                f"OpenNeuro revision differs from registry: "
                f"{head} != {self.dataset.source.revision}"
            )
        runner.run(["datalad", "get", "-r", "."], cwd=staging, timeout=7 * 24 * 3600)
        missing = runner.run(
            ["git", "annex", "find", "--not", "--in=here"],
            cwd=staging,
            timeout=3600,
        ).stdout.strip()
        if missing:
            examples = missing.splitlines()[:10]
            raise ProviderError(f"annex content is still absent after datalad get: {examples}")
        return {
            "provider": "openneuro",
            "accession": self.dataset.source.accession,
            "version": self.dataset.source.version,
            "doi": self.dataset.source.doi,
            "repository": repository,
            "git_commit": head,
            "commands": runner.records,
        }


class GatedProvider(Provider):
    def check(self) -> dict[str, Any]:
        return {
            "status": "blocked",
            "provider": "gated",
            "access_mode": self.dataset.access.mode,
            "landing_url": self.dataset.source.landing_url,
            "terms_url": self.dataset.access.terms_url,
            "instructions": self.dataset.access.instructions,
        }

    def materialize(self, staging: Path) -> dict[str, Any]:
        del staging
        raise AccessBlocked(
            f"{self.dataset.id} requires account-mediated access. "
            f"{self.dataset.access.instructions}"
        )


def make_provider(dataset: DatasetSpec, client: HttpClient) -> Provider:
    providers: dict[str, type[Provider]] = {
        "openneuro": OpenNeuroProvider,
        "figshare": FigshareProvider,
        "mendeley": MendeleyProvider,
        "osf": OsfProvider,
        "gated": GatedProvider,
    }
    try:
        provider_type = providers[dataset.source.provider]
    except KeyError as error:
        raise ProviderError(f"unsupported provider: {dataset.source.provider}") from error
    return provider_type(dataset, client)
