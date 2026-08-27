from __future__ import annotations

import torch

from src.p2_quality_aux_loss import P2QualityAuxDetectionLoss


def _criterion_stub():
    obj = P2QualityAuxDetectionLoss.__new__(P2QualityAuxDetectionLoss)
    obj.nc = 10
    obj.stride = torch.tensor([4.0, 8.0, 16.0])
    obj.device = torch.device("cpu")
    obj.tiny_min_side = 16.0
    obj.focus_classes = (5, 6)
    obj.target_floor = 0.50
    obj.quality_gamma = 0.50
    obj.use_dfl = False
    return obj


def test_quality_mapping_is_tempered_and_monotonic():
    criterion = _criterion_stub()
    q = torch.tensor([0.0, 0.25, 1.0], requires_grad=True)
    target = criterion._quality_to_target(q)
    assert torch.allclose(target, torch.tensor([0.50, 0.75, 1.00]), atol=1e-7)
    assert not target.requires_grad


def test_quality_aux_uses_iou_soft_targets_and_detaches_regression():
    criterion = _criterion_stub()
    scores = torch.zeros(1, 10, 64, requires_grad=True)
    boxes = torch.ones(1, 4, 64, requires_grad=True)
    preds = {
        "scores": scores,
        "boxes": boxes,
        "feats": [torch.zeros(1, 40, 8, 8)],
    }
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[5.0], [6.0]]),
        # P2 image is 32x32. Centers 10px and 22px align exactly to P2
        # anchor centers. With box distances=1 feature unit, predicted boxes
        # are 8x8. First GT is 8x8 => IoU=1; second GT is 4x4 => IoU=.25.
        "bboxes": torch.tensor(
            [
                [10.0 / 32.0, 10.0 / 32.0, 8.0 / 32.0, 8.0 / 32.0],
                [22.0 / 32.0, 22.0 / 32.0, 4.0 / 32.0, 4.0 / 32.0],
            ]
        ),
    }

    loss, count, quality_mean, target_mean = criterion._tiny_center_quality_loss(preds, batch)
    assert count == 2
    assert torch.allclose(quality_mean, torch.tensor(0.625), atol=1e-5)
    assert torch.allclose(target_mean, torch.tensor(0.875), atol=1e-5)
    assert torch.isfinite(loss) and float(loss.detach()) > 0.0

    loss.backward()
    assert int(scores.grad.ne(0).sum()) == 2
    assert boxes.grad is None or int(boxes.grad.ne(0).sum()) == 0


def test_quality_aux_is_zero_without_tiny_focus_targets():
    criterion = _criterion_stub()
    scores = torch.randn(1, 10, 64, requires_grad=True)
    boxes = torch.ones(1, 4, 64, requires_grad=True)
    preds = {
        "scores": scores,
        "boxes": boxes,
        "feats": [torch.zeros(1, 40, 8, 8)],
    }
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[5.0], [0.0]]),
        "bboxes": torch.tensor(
            [
                [0.25, 0.25, 0.60, 0.60],
                [0.75, 0.75, 0.10, 0.10],
            ]
        ),
    }

    loss, count, quality_mean, target_mean = criterion._tiny_center_quality_loss(preds, batch)
    assert count == 0
    assert float(loss.detach()) == 0.0
    assert float(quality_mean) == 0.0
    assert float(target_mean) == 0.0


def test_quality_aux_deduplicates_cell_with_max_quality():
    criterion = _criterion_stub()
    scores = torch.zeros(1, 10, 64, requires_grad=True)
    boxes = torch.ones(1, 4, 64, requires_grad=True)
    preds = {
        "scores": scores,
        "boxes": boxes,
        "feats": [torch.zeros(1, 40, 8, 8)],
    }
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[5.0], [5.0]]),
        # Same class/cell. First matches the 8x8 prediction (IoU=1), second
        # is 4x4 (IoU=.25). Dedup must keep quality 1.0 for the shared logit.
        "bboxes": torch.tensor(
            [
                [10.0 / 32.0, 10.0 / 32.0, 8.0 / 32.0, 8.0 / 32.0],
                [10.0 / 32.0, 10.0 / 32.0, 4.0 / 32.0, 4.0 / 32.0],
            ]
        ),
    }

    _, count, quality_mean, target_mean = criterion._tiny_center_quality_loss(preds, batch)
    assert count == 1
    assert torch.allclose(quality_mean, torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(target_mean, torch.tensor(1.0), atol=1e-5)
