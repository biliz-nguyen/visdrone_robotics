from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect

from src.mentor_transfer_head import AdvantageGatedMentorDetect
from src.mentor_transfer_loss import (
    advantage_gate,
    mentor_transfer_enabled,
    object_normalized_weights,
)


def _nparams(module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_agtl_head_is_exact_h1_inference_budget():
    ch = (32, 64, 128)
    h1 = Detect(nc=10, reg_max=1, end2end=False, ch=ch)
    student = AdvantageGatedMentorDetect(
        nc=10,
        mentor_lambda=0.25,
        tiny_threshold=16.0,
        advantage_margin=0.05,
        min_teacher_iou=0.10,
        end2end=False,
        ch=ch,
    )
    assert student.reg_max == 1
    assert _nparams(student) == _nparams(h1)
    assert [m[-1].out_channels for m in student.cv2] == [4, 4, 4]
    assert [m[-1].out_channels for m in student.cv3] == [10, 10, 10]


def test_advantage_gate_selects_only_tiny_teacher_better_samples():
    student_iou = torch.tensor([0.20, 0.40, 0.20, 0.20, 0.20])
    teacher_iou = torch.tensor([0.40, 0.43, 0.50, 0.40, 0.05])
    min_side = torch.tensor([8.0, 8.0, 20.0, 15.0, 8.0])

    mask, advantage = advantage_gate(
        student_iou,
        teacher_iou,
        min_side,
        tiny_threshold=16.0,
        advantage_margin=0.05,
        min_teacher_iou=0.10,
    )
    assert mask.tolist() == [True, False, False, True, False]
    assert torch.allclose(advantage, torch.tensor([0.15, 0.0, 0.25, 0.15, 0.0]), atol=1e-7)


def test_object_normalized_weights_preserve_v2_object_budget_but_keep_all_candidates():
    # image0/gt0 has three eligible positives. The maximum-advantage reference
    # is idx0, whose raw v2 mentor weight is 0.20 * 1.00 = 0.20.
    candidate = torch.tensor([True, True, True, False, True, True])
    advantage = torch.tensor([0.20, 0.10, 0.05, 0.30, 0.40, 0.20])
    tal = torch.tensor([1.00, 0.50, 2.00, 1.00, 0.50, 1.00])
    batch_idx = torch.tensor([0, 0, 0, 0, 1, 1])
    gt_idx = torch.tensor([0, 0, 0, 1, 0, 0])

    weights = object_normalized_weights(candidate, advantage, tal, batch_idx, gt_idx)

    # object image0/gt0 raw weights: [0.20, 0.05, 0.10], sum=0.35.
    # v2 reference budget is 0.20, spread proportionally over all three.
    expected0 = torch.tensor([0.20, 0.05, 0.10]) / 0.35 * 0.20
    assert torch.allclose(weights[:3], expected0, atol=1e-7)
    assert torch.isclose(weights[:3].sum(), torch.tensor(0.20), atol=1e-7)

    # idx3 is not eligible and must remain zero.
    assert weights[3].item() == 0.0

    # object image1/gt0: max advantage is idx4, v2 budget=0.40*0.50=0.20.
    assert weights[4] > 0 and weights[5] > 0
    assert torch.isclose(weights[4:6].sum(), torch.tensor(0.20), atol=1e-7)


def test_object_normalized_weights_empty_candidates():
    candidate = torch.zeros(4, dtype=torch.bool)
    weights = object_normalized_weights(
        candidate,
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.ones(4),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
    )
    assert torch.count_nonzero(weights) == 0


def test_mentor_transfer_is_training_only():
    assert torch.is_grad_enabled()
    assert mentor_transfer_enabled(0.25)
    assert not mentor_transfer_enabled(0.0)
    with torch.no_grad():
        assert not mentor_transfer_enabled(0.25)


def test_head_metadata_validation():
    ch = (32, 64, 128)
    invalid = [
        {"mentor_lambda": -0.1},
        {"tiny_threshold": 0.0},
        {"advantage_margin": 1.0},
        {"min_teacher_iou": -0.1},
        {"min_teacher_iou": 1.1},
    ]
    for kwargs in invalid:
        try:
            AdvantageGatedMentorDetect(nc=10, ch=ch, **kwargs)
            raise AssertionError(f"invalid AGTL metadata should fail: {kwargs}")
        except ValueError:
            pass
