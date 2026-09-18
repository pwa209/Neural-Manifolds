"""Validate EEGLAB payload extent before tactile epoch reads.

Only a short payload that matches its pinned annex size AND checksum is an
auditable source-data exclusion. Unverified truncation or another read error
remains fatal: a failed download must not silently remove a participant.
"""

import hashlib
import re
from pathlib import Path

from neural_manifolds.stage_processing import read_raw_recording


class IncompleteSourceRecordingError(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__("verified_source_eeglab_payload_shorter_than_header")


def open_tactile_recording(path: Path):
    raw = read_raw_recording(path)
    try:
        files = [Path(p) for p in raw.filenames]
        if len(files) != 1 or files[0].suffix.lower() != ".fdt":
            raise ValueError("expected_one_external_eeglab_float32_payload")
        payload = files[0]
        expected = int(raw.n_times) * len(raw.ch_names) * 4
        actual = payload.stat().st_size
        audit = {
            "declared_samples": int(raw.n_times),
            "stored_channels": len(raw.ch_names),
            "expected_payload_bytes": expected,
            "actual_payload_bytes": actual,
        }
        if actual == expected:
            return raw, {**audit, "status": "payload_extent_matches_header"}
        if actual > expected:
            raise ValueError("eeglab_payload_larger_than_header")
        # The acquisition's immutable annex key identifies publisher bytes.
        key = re.fullmatch(r"MD5E-s(\d+)--([0-9a-f]{32})\.fdt", payload.resolve().name)
        if key is None or int(key[1]) != actual:
            raise ValueError("short_eeglab_payload_not_verified_against_pinned_source")
        with payload.open("rb") as stream:
            digest = hashlib.file_digest(stream, "md5").hexdigest()
        if digest != key[2]:
            raise ValueError("short_eeglab_payload_checksum_mismatch")
        raise IncompleteSourceRecordingError(
            {
                **audit,
                "status": "excluded_incomplete_source_recording",
                "source_annex_key": payload.resolve().name,
                "source_checksum_verified": True,
                "source_md5": digest,
            }
        )
    except Exception:
        raw.close()
        raise
