"""Dataset-independent EEG and TMS-EEG preprocessing primitives."""

from .eeg import (
    ArtifactWindowResult,
    BadChannelResult,
    canonicalize_channel_name,
    detect_artifact_windows,
    detect_bad_channels,
    make_windows,
)

__all__ = [
    "ArtifactWindowResult",
    "BadChannelResult",
    "canonicalize_channel_name",
    "detect_artifact_windows",
    "detect_bad_channels",
    "make_windows",
]
