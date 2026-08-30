"""Frozen foundation-model adapters with an explicit label firewall."""

from .base import EncodedBatch, assert_label_firewall, validate_frozen_model
from .labram import OfficialLaBraMEncoder, channel_position_indices

__all__ = [
    "EncodedBatch",
    "OfficialLaBraMEncoder",
    "assert_label_firewall",
    "channel_position_indices",
    "validate_frozen_model",
]
