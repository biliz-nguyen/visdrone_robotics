from __future__ import annotations

import torch

from src.tiny_nwd_loss import blended_ciou_nwd_loss, nwd_similarity_xyxy, tiny_gate_xyxy


def test_nwd_identical_boxes_is_one():
    box = torch.tensor([[10.0, 20.0, 18.0, 30.0]])
    sim = nwd_similarity_xyxy(box, box, c=12.8)
    assert torch.allclose(sim, torch.ones_like(sim), atol=1e-6)


def test_non_tiny_recovers_stock_ciou_exactly():
    pred = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
    target = torch.tensor([[1.0, 1.0, 21.0, 21.0]])
    ciou = torch.tensor([[0.61]])
    out = blended_ciou_nwd_loss(ciou, pred, target, c=12.8, tiny_min_side=16.0, ciou_weight=0.75, nwd_weight=0.25)
    assert torch.allclose(out, 1.0 - ciou, atol=0.0, rtol=0.0)
    assert float(tiny_gate_xyxy(target, 16.0).item()) == 0.0


def test_tiny_box_blends_nwd_without_changing_selection_state():
    pred = torch.tensor([[0.0, 0.0, 8.0, 8.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 9.0, 8.0]])
    ciou = torch.tensor([[0.70]])
    out = blended_ciou_nwd_loss(ciou, pred, target, c=12.8, tiny_min_side=16.0, ciou_weight=0.75, nwd_weight=0.25)
    assert out.shape == ciou.shape
    assert float(tiny_gate_xyxy(target, 16.0).item()) == 0.5
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
