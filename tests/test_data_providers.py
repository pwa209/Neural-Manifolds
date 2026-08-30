from typing import Any

from neural_manifolds.data.models import DatasetSpec
from neural_manifolds.data.providers import MendeleyProvider


class MendeleyClient:
    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        if url.endswith("/folders/1"):
            return [{"id": "folder-1", "name": "records"}]
        if url.endswith("/files") and params and params["folder_id"] == "root":
            return []
        if url.endswith("/files"):
            return [
                {
                    "filename": "Patient_1_VS.edf",
                    "id": "file-1",
                    "folder_id": "folder-1",
                    "status": "COMPLETED",
                    "content_details": {
                        "size": 4,
                        "sha256_hash": "a" * 64,
                        "download_url": "https://data.example.test/file-1",
                    },
                }
            ]
        return {
            "id": "example",
            "version": 1,
            "doi": {"id": "10.17632/example.1"},
            "publish_date": "2020-01-01T00:00:00Z",
            "data_licence": {"short_name": "CC BY 4.0"},
        }


def test_mendeley_provider_enumerates_versioned_folders() -> None:
    dataset = DatasetSpec.model_validate(
        {
            "id": "example_data",
            "title": "Example",
            "role": "test",
            "modalities": ["eeg"],
            "source": {
                "provider": "mendeley",
                "accession": "example",
                "version": "1",
                "doi": "10.17632/example.1",
                "landing_url": "https://data.example.test/dataset",
                "api_url": "https://data.example.test/public-api/datasets/example",
            },
            "license": {
                "spdx": "CC-BY-4.0",
                "status": "verified",
                "source_url": "https://data.example.test/dataset",
            },
            "access": {"mode": "open", "terms_url": "https://data.example.test/terms"},
            "validation": {"minimum_files": 1, "minimum_bytes": 1},
            "official_sources": ["https://data.example.test/dataset"],
        }
    )
    provider = MendeleyProvider(dataset, MendeleyClient())  # type: ignore[arg-type]
    discovery = provider.discover()
    assert discovery.total_known_bytes == 4
    assert discovery.files[0].relative_path == "records/Patient_1_VS.edf"
    assert discovery.files[0].hashes == {"sha256": "a" * 64}
