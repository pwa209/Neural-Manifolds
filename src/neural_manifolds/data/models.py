"""Strict models for the dataset registry."""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProviderName = Literal["openneuro", "figshare", "mendeley", "osf", "gated"]
AccessMode = Literal["open", "account_required", "manual"]
LicenseStatus = Literal["verified", "verify_at_acquisition", "unresolved"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegistryDefaults(StrictModel):
    immutable_raw_releases: bool = True
    checksum_algorithm: Literal["sha256"] = "sha256"
    maximum_attempts: int = Field(default=3, ge=1, le=10)
    connect_timeout_seconds: int = Field(default=20, ge=1)
    read_timeout_seconds: int = Field(default=120, ge=1)


class SourceSpec(StrictModel):
    provider: ProviderName
    accession: str = Field(min_length=1)
    version: str = Field(min_length=1)
    doi: str = Field(min_length=1)
    landing_url: str
    repository_url: str | None = None
    api_url: str | None = None
    metadata_url: str | None = None
    revision: str | None = None
    mutable_upstream: bool = False
    note: str | None = None

    @model_validator(mode="after")
    def validate_provider_fields(self) -> SourceSpec:
        if self.version.lower() in {"latest", "draft", "main", "master", "current"}:
            raise ValueError("source.version must identify a release, never a moving label")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.version):
            raise ValueError("source.version must be a portable path component")
        required = {
            "openneuro": self.repository_url,
            "figshare": self.api_url,
            "mendeley": self.api_url,
            "osf": self.api_url,
            "gated": self.api_url,
        }
        if not required[self.provider]:
            raise ValueError(f"{self.provider} source is missing its acquisition endpoint")
        if self.provider == "openneuro" and not re.fullmatch(r"[0-9a-f]{40}", self.revision or ""):
            raise ValueError("OpenNeuro sources require a pinned 40-character Git revision")
        for name, value in (
            ("landing_url", self.landing_url),
            ("repository_url", self.repository_url),
            ("api_url", self.api_url),
            ("metadata_url", self.metadata_url),
        ):
            if value and urlsplit(value).scheme != "https":
                raise ValueError(f"{name} must use HTTPS")
        if self.mutable_upstream and self.provider not in {"osf"}:
            raise ValueError("only explicitly audited mutable providers may be marked mutable")
        return self


class LicenseSpec(StrictModel):
    spdx: str = Field(min_length=1)
    status: LicenseStatus
    source_url: str
    note: str | None = None


class AccessSpec(StrictModel):
    mode: AccessMode
    terms_url: str
    instructions: str | None = None

    @model_validator(mode="after")
    def require_gated_instructions(self) -> AccessSpec:
        if self.mode != "open" and not self.instructions:
            raise ValueError("non-open access requires explicit instructions")
        return self


class ValidationSpec(StrictModel):
    required_paths: list[str] = Field(default_factory=list)
    required_globs: list[str] = Field(default_factory=list)
    minimum_files: int = Field(default=1, ge=1)
    minimum_bytes: int = Field(default=1, ge=0)


class DatasetSpec(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    title: str = Field(min_length=1)
    role: str = Field(min_length=1)
    modalities: list[str] = Field(min_length=1)
    source: SourceSpec
    license: LicenseSpec
    access: AccessSpec
    validation: ValidationSpec
    official_sources: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self) -> DatasetSpec:
        if self.source.provider == "gated" and self.access.mode == "open":
            raise ValueError("gated providers cannot be marked open")
        if self.source.provider != "gated" and self.access.mode != "open":
            raise ValueError("non-gated provider unexpectedly requires credentials")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be unique")
        for pattern in (*self.validation.required_paths, *self.validation.required_globs):
            normalized = pattern.replace("\\", "/")
            if (
                not normalized
                or normalized.startswith("/")
                or re.match(r"^[A-Za-z]:", normalized)
                or ".." in normalized.split("/")
            ):
                raise ValueError(f"unsafe validation path: {pattern}")
        for label, url in (
            ("license.source_url", self.license.source_url),
            ("access.terms_url", self.access.terms_url),
            *(("official_sources", url) for url in self.official_sources),
        ):
            if urlsplit(url).scheme != "https":
                raise ValueError(f"{label} must use HTTPS")
        return self


class DatasetRegistryModel(StrictModel):
    schema_version: int = Field(ge=1)
    registry_verified_on: str
    defaults: RegistryDefaults
    datasets: list[DatasetSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_registry(self) -> DatasetRegistryModel:
        identifiers = [dataset.id for dataset in self.datasets]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("dataset identifiers must be unique")
        accessions = [f"{d.source.provider}:{d.source.accession}" for d in self.datasets]
        if len(set(accessions)) != len(accessions):
            raise ValueError("provider accessions must be unique")
        return self


SECRET_KEY = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|credential)")
SECRET_VALUE = re.compile(
    r"(?i)(authorization:\s*(bearer|basic)|[?&](token|key|signature|x-amz-credential)=)"
)


def assert_no_embedded_secrets(value: object, path: str = "registry") -> None:
    """Reject credential-shaped keys and signed URLs from committed configuration."""

    if isinstance(value, dict):
        for key, child in value.items():
            if SECRET_KEY.search(str(key)):
                raise ValueError(f"credential-shaped key is forbidden at {path}.{key}")
            assert_no_embedded_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_embedded_secrets(child, f"{path}[{index}]")
    elif isinstance(value, str) and SECRET_VALUE.search(value):
        raise ValueError(f"credential-shaped value is forbidden at {path}")
