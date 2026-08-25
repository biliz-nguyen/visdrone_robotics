from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect

from src.shift_consistency_head import TinyShiftConsistencyDetect
from src.shift_consistency_loss import (
    inverse_shift_boxes,
    object_representatives,
    shift_equivariance_penalty,
    shift_image,
)


def _nparams(module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_shift_head_matches_h1_inference_budget():
    ch = (32, 64, 128)
    h1 = Detect(nc=10, reg_max=1, end2end=False, ch=ch)
    test = TinyShiftConsistencyDetect(
        nc=10,
        shift_lambda=0.25,
        tiny_threshold=16.0,
        max_shift_px=1,
        end2end=False,
        ch=ch,
    )
    assert test.reg_max == 1
    assert _nparams(test) == _nparams(h1)
    assert [m[-1].out_channels for m in test.cv2] == [4, 4, 4]
    assert [m[-1].out_channels for m in test.cv3] == [10, 10, 10]


def test_shift_image_cardinal_translation():
    x = torch.arange(9, dtype=torch.float32).view(1, 1, 3, 3)
    right = shift_image(x, 1, 0)
    assert right.shape == x.shape
    assert torch.equal(right[0, 0, :, 1:], x[0, 0, :, :-1])
    left = shift_image(x, -1, 0)
    assert torch.equal(left[0, 0, :, :-1], x[0, 0, :, 1:])


def test_object_representative_uses_best_tal_score_per_gt():
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 3.0, 3.0], [3.0, 3.0, 4.0, 4.0]]]
    )
    fg = torch.tensor([[True, True, True, False]])
    gt_idx = torch.tensor([[0, 0, 1, 0]])
    target_scores = torch.tensor(
        [[[0.1, 0.0], [0.6, 0.0], [0.0, 0.4], [0.0, 0.0]]]
    )
    reps, valid = object_representatives(boxes, fg, gt_idx, target_scores, max_gt=2)
    assert valid.tolist() == [True, True]
    assert torch.equal(reps[0], boxes[0, 1])
    assert torch.equal(reps[1], boxes[0, 2])


def test_shift_penalty_zero_for_equivariant_boxes():
    base = torch.tensor([[10.0, 12.0, 20.0, 24.0]])
    shifted = base + torch.tensor([1.0, 0.0, 1.0, 0.0])
    reliability = torch.ones(1)
    penalty = shift_equivariance_penalty(base, shifted, reliability, 1, 0)
    assert float(penalty) < 1e-6
    assert torch.equal(inverse_shift_boxes(shifted, 1, 0), base)


def test_shift_penalty_positive_for_inconsistent_boxes():
    base = torch.tensor([[10.0, 12.0, 20.0, 24.0]])
    shifted = torch.tensor([[13.0, 12.0, 23.0, 24.0]])
    reliability = torch.ones(1)
    penalty = shift_equivariance_penalty(base, shifted, reliability, 1, 0)
    assert float(penalty) > 0.0
