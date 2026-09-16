import pytest

from neural_manifolds.foundation.base import assert_label_firewall
from neural_manifolds.foundation.labram import channel_position_indices, pretrained_backbone_state


def test_label_firewall_rejects_outcome_fields() -> None:
    with pytest.raises(ValueError, match="prohibited"):
        assert_label_firewall(["participant_id", "condition"])


def test_labram_channel_positions_include_cls() -> None:
    indices = channel_position_indices(["Fp1", "Cz", "O2"])
    assert indices[0] == 0
    assert len(indices) == 4
    assert len(set(indices)) == 4


def test_pretrained_student_retains_backbone_norm_and_rejects_no_keys_silently():
    payload = {
        "model": {
            "student.norm.weight": 1,
            "student.mask_token": 2,
            "student.lm_head.weight": 3,
            "projection_head.0.weight": 4,
            "student.blocks.0.gamma_1": 5,
            "student.unexpected": 6,
        }
    }
    assert pretrained_backbone_state(payload) == {
        "norm.weight": 1,
        "blocks.0.gamma_1": 5,
        "unexpected": 6,
    }
