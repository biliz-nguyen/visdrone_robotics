from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect

from src.mentor_transfer_head import AdvantageGatedMentorDetect
from src.mentor_transfer_loss import (
    advantage_gate,
    mentor_transfer_enabled,
    object_balanced_gate,
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

    # 0: tiny and mentor is better by 0.20 -> eligible.
    # 1: improvement is only 0.03 -> rejected by advantage margin.
    # 2: mentor is better but GT is not tiny -> rejected.
    # 3: tiny and mentor is better by 0.20 -> eligible.
    # 4: teacher IoU below minimum-quality guard -> rejected.
    assert mask.tolist() == [True, False, False, True, False]
    assert torch.allclose(advantage, torch.tensor([0.15, 0.0, 0.25, 0.15, 0.0]), atol=1e-7)


def test_object_balanced_gate_keeps_one_best_positive_per_gt():
    # Seven positive locations from two images. Several locations map to the
    # same assigned GT, which is exactly the dense-TAL case AGTL v2 targets.
    candidate = torch.tensor([True, True, True, False, True, True, True])
    advantage = torch.tensor([0.10, 0.30, 0.20, 0.50, 0.40, 0.40, 0.05])
    batch_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    gt_idx = torch.tensor([0, 0, 1, 1, 0, 0, 1])

    selected = object_balanced_gate(candidate, advantage, batch_idx, gt_idx)

    # image0/gt0 -> idx1 wins (0.30 > 0.10)
    # image0/gt1 -> idx2 wins because idx3 is not eligible
    # image1/gt0 -> idx4 wins exact tie deterministically (first occurrence)
    # image1/gt1 -> idx6 is the only candidate
    assert selected.tolist() == [False, True, True, False, True, False, True]


def test_object_balanced_gate_empty_candidates():
    candidate = torch.zeros(4, dtype=torch.bool)
    selected = object_balanced_gate(
        candidate,
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
    )
    assert not selected.any()


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
