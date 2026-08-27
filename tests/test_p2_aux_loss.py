from __future__ import annotations

import torch

from src.p2_aux_loss import P2TinyAuxDetectionLoss


def _criterion_stub():
    obj = P2TinyAuxDetectionLoss.__new__(P2TinyAuxDetectionLoss)
    obj.nc = 10
    obj.stride = torch.tensor([4.0, 8.0, 16.0])
    obj.device = torch.device("cpu")
    obj.tiny_min_side = 16.0
    obj.focus_classes = (5, 6)
    return obj


def test_aux_selects_only_tiny_focus_centers_and_backprops():
    criterion = _criterion_stub()
    scores = torch.zeros(1, 10, 64, requires_grad=True)
    preds = {
        "scores": scores,
        "feats": [torch.zeros(1, 40, 8, 8)],
    }
    batch = {
        "batch_idx": torch.tensor([0, 0, 0]),
        "cls": torch.tensor([[5.0], [6.0], [0.0]]),
        # 8x8 P2 at stride 4 => 32x32 image. First two boxes have
        # min side <16 px; the third is non-focus and must be ignored.
        "bboxes": torch.tensor(
            [
                [0.25, 0.25, 0.20, 0.20],
                [0.75, 0.75, 0.10, 0.10],
                [0.50, 0.50, 0.10, 0.10],
            ]
        ),
    }

    loss, count = criterion._tiny_center_positive_loss(preds, batch)
    assert count == 2
    assert torch.isfinite(loss) and float(loss.detach()) > 0.0
    loss.backward()

    # Only two selected center/class logits receive auxiliary gradient.
    assert int(scores.grad.ne(0).sum()) == 2


def test_aux_is_zero_without_tiny_focus_targets():
    criterion = _criterion_stub()
    scores = torch.randn(1, 10, 64, requires_grad=True)
    preds = {
        "scores": scores,
        "feats": [torch.zeros(1, 40, 8, 8)],
    }
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[5.0], [0.0]]),
        # Focus box is 19.2 px wide/tall, so it is not tiny; the second
        # box is tiny but not pedestrian/people.
        "bboxes": torch.tensor(
            [
                [0.25, 0.25, 0.60, 0.60],
                [0.75, 0.75, 0.10, 0.10],
            ]
        ),
    }

    loss, count = criterion._tiny_center_positive_loss(preds, batch)
    assert count == 0
    assert float(loss.detach()) == 0.0


def test_aux_deduplicates_same_class_same_p2_cell():
    criterion = _criterion_stub()
    scores = torch.zeros(1, 10, 64, requires_grad=True)
    preds = {
        "scores": scores,
        "feats": [torch.zeros(1, 40, 8, 8)],
    }
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[5.0], [5.0]]),
        "bboxes": torch.tensor(
            [
                [0.251, 0.251, 0.10, 0.10],
                [0.260, 0.260, 0.10, 0.10],
            ]
        ),
    }

    _, count = criterion._tiny_center_positive_loss(preds, batch)
    assert count == 1
