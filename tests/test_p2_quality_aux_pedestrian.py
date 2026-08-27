from __future__ import annotations

import torch

from src.p2_quality_aux_loss import P2QualityAuxDetectionLoss


def _criterion_stub():
    obj = P2QualityAuxDetectionLoss.__new__(P2QualityAuxDetectionLoss)
    obj.nc = 10
    obj.stride = torch.tensor([4.0])
    obj.device = torch.device("cpu")
    obj.tiny_min_side = 16.0
    obj.aux_weight = 0.10
    obj.focus_classes = (5,)
    obj.target_floor = 0.50
    obj.quality_gamma = 0.50
    obj.use_dfl = False
    return obj


def test_c10_pedestrian_only_ignores_people_target():
    criterion = _criterion_stub()
    scores = torch.zeros(1, 10, 64, requires_grad=True)
    boxes = torch.zeros(1, 4, 64, requires_grad=True)
    preds = {
        "scores": scores,
        "boxes": boxes,
        "feats": [torch.zeros(1, 40, 8, 8)],
    }
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[5.0], [6.0]]),
        "bboxes": torch.tensor(
            [
                [0.25, 0.25, 0.10, 0.10],
                [0.75, 0.75, 0.10, 0.10],
            ]
        ),
    }

    loss, count, quality_mean, target_mean = criterion._tiny_center_quality_loss(preds, batch)
    assert count == 1
    assert torch.isfinite(loss) and float(loss.detach()) > 0.0
    assert 0.0 <= float(quality_mean) <= 1.0
    assert 0.50 <= float(target_mean) <= 1.0

    loss.backward()
    assert scores.grad is not None
    # Exactly one pedestrian center/class logit receives the C10 auxiliary gradient.
    assert int(scores.grad.ne(0).sum()) == 1
    # Quality is detached, so the auxiliary term must not optimize regression.
    assert boxes.grad is None or int(boxes.grad.ne(0).sum()) == 0


def test_c10_quality_mapping_is_unchanged_from_c9():
    criterion = _criterion_stub()
    q = torch.tensor([0.0, 0.25, 1.0], requires_grad=True)
    target = criterion._quality_to_target(q)
    expected = torch.tensor([0.50, 0.75, 1.00])
    assert torch.allclose(target, expected, atol=1e-7)
    assert not target.requires_grad
