from pathlib import Path

import pytest

from neural_manifolds.stage_processing import read_raw_recording


def test_unknown_recording_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        read_raw_recording(tmp_path / "recording.txt")
