"""Small, retrying HTTP client with range-resumable downloads."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests


class DownloadError(RuntimeError):
    """Raised after bounded retries or integrity validation failure."""


def public_url(url: str) -> str:
    """Strip signed query strings and fragments before logging provenance."""

    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def safe_filename(name: str) -> str:
    candidate = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if candidate in {"", ".", ".."} or "\x00" in candidate:
        raise DownloadError(f"unsafe remote filename: {name!r}")
    return candidate


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class HttpClient:
    def __init__(
        self,
        *,
        maximum_attempts: int = 3,
        connect_timeout_seconds: int = 20,
        read_timeout_seconds: int = 120,
        session: requests.Session | None = None,
    ) -> None:
        self.maximum_attempts = maximum_attempts
        self.timeout = (connect_timeout_seconds, read_timeout_seconds)
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "neural-manifolds-acquisition/0.1")

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        last_error: BaseException | None = None
        for attempt in range(1, self.maximum_attempts + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if attempt < self.maximum_attempts:
                    time.sleep(attempt)
        raise DownloadError(
            f"metadata request failed after {self.maximum_attempts} attempts: {public_url(url)}"
        ) from last_error

    def download(
        self,
        url: str,
        destination: str | Path,
        *,
        expected_size: int | None = None,
        expected_hashes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.part")
        hashes = {key.lower(): value.lower() for key, value in (expected_hashes or {}).items()}
        if target.is_file():
            self._verify(target, expected_size, hashes)
            return self._metadata(target, url, resumed=False)
        last_error: BaseException | None = None
        resumed = partial.is_file() and partial.stat().st_size > 0
        for attempt in range(1, self.maximum_attempts + 1):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with self.session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=self.timeout,
                    allow_redirects=True,
                ) as response:
                    if offset and response.status_code == 416 and expected_size == offset:
                        pass
                    else:
                        response.raise_for_status()
                        append = offset > 0 and response.status_code == 206
                        mode = "ab" if append else "wb"
                        with partial.open(mode) as stream:
                            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                                if chunk:
                                    stream.write(chunk)
                            stream.flush()
                            os.fsync(stream.fileno())
                self._verify(partial, expected_size, hashes)
                os.replace(partial, target)
                return self._metadata(target, url, resumed=resumed)
            except (OSError, requests.RequestException, DownloadError) as error:
                last_error = error
                if attempt < self.maximum_attempts:
                    time.sleep(attempt)
        raise DownloadError(
            f"download failed after {self.maximum_attempts} attempts: {public_url(url)}"
        ) from last_error

    @staticmethod
    def _verify(path: Path, expected_size: int | None, expected_hashes: dict[str, str]) -> None:
        if expected_size is not None and path.stat().st_size != expected_size:
            raise DownloadError(
                f"size mismatch for {path.name}: {path.stat().st_size} != {expected_size}"
            )
        for algorithm, expected in expected_hashes.items():
            if algorithm not in hashlib.algorithms_available:
                continue
            actual = _hash_file(path, algorithm)
            if actual.lower() != expected.lower():
                raise DownloadError(f"{algorithm} mismatch for {path.name}")

    @staticmethod
    def _metadata(path: Path, url: str, *, resumed: bool) -> dict[str, Any]:
        return {
            "path": path.name,
            "size": path.stat().st_size,
            "source_url": public_url(url),
            "resumed": resumed,
        }
