import io
import json
import zlib

import pytest

from neural_manifolds.continuous.edf_compat import body_sha256, qc_clock_copy
from neural_manifolds.continuous.rar_archive import backend, parse_listing, safe_name
from neural_manifolds.continuous.signal_qc import _copy_bounded
from neural_manifolds.provenance import atomic_write_json, sha256_file


def test_invalid_clock_is_qc_only_and_raw_unchanged(tmp_path):
    source = tmp_path / "original.edf"
    fixed = bytearray(b" " * 256)
    fixed[176:184] = b"16.42.60"
    payload = bytes(fixed) + b"signal bytes and remaining header" * 100
    source.write_bytes(payload)
    original_hash = sha256_file(source)
    with qc_clock_copy(source, tmp_path / "scratch") as (copy, receipt):
        assert copy != source
        assert copy.read_bytes()[176:184] == b"00.00.00"
        assert body_sha256(copy) == body_sha256(source)
        assert receipt["source_sha256"] == original_hash
        assert receipt["absolute_clock_available"] is False
        assert receipt["all_other_bytes_verified_identical"] is True
    assert source.read_bytes() == payload
    assert not copy.exists()


def test_valid_clock_unchanged(tmp_path):
    source = tmp_path / "valid.edf"
    fixed = bytearray(b" " * 256)
    fixed[176:184] = b"16.42.59"
    source.write_bytes(fixed)
    with qc_clock_copy(source, tmp_path) as (path, receipt):
        assert path == source
        assert receipt is None


def test_rar_listing_rejects_unsafe_ambiguous_or_encrypted_members():
    good = "Path = Data/a.edf\nFolder = -\nSize = 9\nCRC = 12345678\nEncrypted = -"
    assert parse_listing(good) == [{"member": "Data/a.edf", "bytes": 9, "crc32": "12345678"}]
    assert len(parse_listing(good + "\nSymbolic Link = \n\n")) == 1
    for bad in [
        good + "\n\n" + good,
        good.replace("Data/a.edf", "../a.edf"),
        good.replace("Encrypted = -", "Encrypted = +"),
        good + "\nSymbolic Link = outside",
    ]:
        with pytest.raises(ValueError):
            parse_listing(bad)
    assert not safe_name("@listfile")
    assert not safe_name("Data/\x00.edf")
    assert not safe_name("C:\\outside")


def test_copy_bound_enforced_before_writing_extra_bytes():
    out = io.BytesIO()
    with pytest.raises(ValueError, match="exceeded"):
        _copy_bounded(io.BytesIO(b"123456"), out, 5)
    assert len(out.getvalue()) <= 5
    out = io.BytesIO()
    assert _copy_bounded(io.BytesIO(b"12345"), out, 5) == 5


def test_rar_payload_requires_independent_crc_match():
    payload = b"verified bytes"
    crc = f"{zlib.crc32(payload):08X}"
    assert _copy_bounded(io.BytesIO(payload), io.BytesIO(), len(payload), expected_crc=crc) == len(
        payload
    )
    with pytest.raises(ValueError, match="CRC32 mismatch"):
        _copy_bounded(io.BytesIO(payload), io.BytesIO(), len(payload), expected_crc="00000000")


def test_configured_decoder_is_checksum_bound(tmp_path, monkeypatch):
    binary = tmp_path / "decoder"
    binary.write_bytes(b"test decoder, not executed")
    monkeypatch.setenv("NEURAL_MANIFOLDS_7ZIP", str(binary))
    monkeypatch.setenv("NEURAL_MANIFOLDS_7ZIP_SHA256", sha256_file(binary))
    assert backend() == str(binary.resolve())
    monkeypatch.setenv("NEURAL_MANIFOLDS_7ZIP_SHA256", "wrong")
    with pytest.raises(ValueError, match="checksum"):
        backend()


def test_annotation_fallback_is_narrow_and_recorded(tmp_path, monkeypatch):
    from neural_manifolds.continuous import signal_qc

    release = tmp_path / "study" / "1"
    (release / ".acquisition").mkdir(parents=True)
    (release / ".acquisition/COMPLETE.json").write_text("{}")
    (release / "a.edf").write_bytes(b"synthetic")
    audit = tmp_path / "audit.json"
    atomic_write_json(
        audit,
        {
            "release": str(release),
            "completion_marker_sha256": sha256_file(release / ".acquisition/COMPLETE.json"),
            "files": [{"container": None, "member": "a.edf", "bytes": 9}],
            "issues": [],
        },
    )
    calls = []

    def inspect(row, *, study, **options):
        calls.append(options)
        if not options:
            # MNE 1.13 wraps UnicodeDecodeError in a plain Exception.
            raise Exception("Encountered invalid byte in at least one annotations channel.")
        assert options == {"reader_options": {"edf_annotation_encoding": "latin1"}}
        return dict(row), []

    monkeypatch.setattr(signal_qc, "_inspect_recording", inspect)
    monkeypatch.setenv("NM_SCRATCH_ROOT", str(tmp_path / "scratch"))
    result = signal_qc.run(audit, tmp_path / "out")
    assert len(calls) == 2
    assert len(result["recordings"]) == 1
    checkpoint = next((tmp_path / "out/recordings").glob("*.json"))
    assert (
        json.loads(checkpoint.read_text())["annotation_decoding"]["annotation_semantics_verified"]
        is False
    )
