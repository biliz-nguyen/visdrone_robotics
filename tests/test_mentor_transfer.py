from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect

from src.mentor_transfer_head import AdvantageGatedMentorDetect
from src.mentor_transfer_loss import advantage_gate


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

    # 0: tiny and mentor is better by 0.20 -> selected.
    # 1: improvement is only 0.03 -> rejected by advantage margin.
    # 2: mentor is better but GT is not tiny -> rejected.
    # 3: tiny and mentor is better by 0.20 -> selected.
    # 4: teacher IoU below minimum-quality guard -> rejected.
    assert mask.tolist() == [True, False, False, True, False]
    assert torch.allclose(advantage, torch.tensor([0.15, 0.0, 0.25, 0.15, 0.0]), atol=1e-7)


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
