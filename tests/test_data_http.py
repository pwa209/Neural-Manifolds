import hashlib
from pathlib import Path
from typing import Any

from neural_manifolds.data.http import HttpClient, public_url


class FakeResponse:
    status_code = 206

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        assert chunk_size > 0
        return [b"def"]


class FakeSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return FakeResponse()


def test_range_download_resumes_and_strips_signed_query(tmp_path: Path) -> None:
    session = FakeSession()
    client = HttpClient(session=session)  # type: ignore[arg-type]
    target = tmp_path / "sample.bin"
    target.with_name("sample.bin.part").write_bytes(b"abc")
    digest = hashlib.sha256(b"abcdef").hexdigest()

    result = client.download(
        "https://files.example.test/sample.bin?X-Amz-Credential=secret",
        target,
        expected_size=6,
        expected_hashes={"sha256": digest},
    )

    assert target.read_bytes() == b"abcdef"
    assert session.calls[0]["headers"] == {"Range": "bytes=3-"}
    assert result["resumed"] is True
    assert result["source_url"] == "https://files.example.test/sample.bin"
    assert public_url("https://example.test/a?token=secret#fragment") == ("https://example.test/a")
