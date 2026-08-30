import pytest

from neural_manifolds.foundation.base import assert_label_firewall
from neural_manifolds.foundation.labram import channel_position_indices


def test_label_firewall_rejects_outcome_fields() -> None:
    with pytest.raises(ValueError, match="prohibited"):
        assert_label_firewall(["participant_id", "condition"])


def test_labram_channel_positions_include_cls() -> None:
    indices = channel_position_indices(["Fp1", "Cz", "O2"])
    assert indices[0] == 0
    assert len(indices) == 4
    assert len(set(indices)) == 4
