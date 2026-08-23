from __future__ import annotations

import torch

from src.paired_boundary_head import PairedBoundaryDetect


def test_boundary_transform_is_positive_and_shape_preserving():
    head = PairedBoundaryDetect(nc=10, ratio_limit=0.99, ch=(32, 64, 128))
    raw = torch.randn(2, 4, 11, 13)
    out = head._couple_boundaries(raw)
    assert out.shape == raw.shape
    assert torch.isfinite(out).all()
    assert (out > 0).all()


def test_symmetric_offset_zero_gives_equal_opposing_distances():
    head = PairedBoundaryDetect(nc=10, ratio_limit=0.99, ch=(32, 64, 128))
    raw = torch.zeros(1, 4, 3, 5)
    raw[:, 0] = 1.2
    raw[:, 2] = 0.7
    out = head._couple_boundaries(raw)
    left, top, right, bottom = out.chunk(4, dim=1)
    assert torch.allclose(left, right, atol=1e-7, rtol=0)
    assert torch.allclose(top, bottom, atol=1e-7, rtol=0)


def test_transform_can_represent_asymmetric_positive_distances():
    head = PairedBoundaryDetect(nc=10, ratio_limit=0.99, ch=(32, 64, 128))
    # Strong signed offsets must change opposite boundaries in opposite directions.
    raw_pos = torch.tensor([[[[1.0]], [[1.5]], [[1.0]], [[-1.0]]]])
    out = head._couple_boundaries(raw_pos)
    left, top, right, bottom = [x.item() for x in out.chunk(4, dim=1)]
    assert right > left > 0
    assert top > bottom > 0


def test_forward_budget_matches_direct_regression_head():
    head = PairedBoundaryDetect(nc=10, ratio_limit=0.99, ch=(32, 64, 128))
    head.train()
    feats = [
        torch.randn(2, 32, 40, 40),
        torch.randn(2, 64, 20, 20),
        torch.randn(2, 128, 10, 10),
    ]
    out = head.forward_head(feats, head.cv2, head.cv3)
    n = 40 * 40 + 20 * 20 + 10 * 10
    assert out["boxes"].shape == (2, 4, n)
    assert out["scores"].shape == (2, 10, n)


def test_transform_receives_gradients():
    head = PairedBoundaryDetect(nc=10, ratio_limit=0.99, ch=(32, 64, 128))
    raw = torch.randn(1, 4, 8, 8, requires_grad=True)
    out = head._couple_boundaries(raw)
    out.square().mean().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum().item() > 0
