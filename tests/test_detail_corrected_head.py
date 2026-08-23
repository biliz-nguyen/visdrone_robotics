from __future__ import annotations

import torch

from src.detail_corrected_head import DetailCorrectedDetect


def test_detail_branch_starts_as_exact_direct_regression():
    head = DetailCorrectedDetect(nc=10, max_correction_cell=0.5, ch=(32, 64, 128))
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

    p2_base = head.cv2[0](feats[0]).view(2, 4, -1)
    assert torch.allclose(out["boxes"][..., : 40 * 40], p2_base, atol=0, rtol=0)
    assert torch.count_nonzero(head.detail_proj.weight).item() == 0
    assert torch.count_nonzero(head.detail_proj.bias).item() == 0


def test_detail_correction_is_bounded_and_p2_only():
    head = DetailCorrectedDetect(nc=10, max_correction_cell=0.5, ch=(32, 64, 128))
    head.train()
    feats = [
        torch.randn(1, 32, 16, 16),
        torch.randn(1, 64, 8, 8),
        torch.randn(1, 128, 4, 4),
    ]
    with torch.no_grad():
        head.detail_proj.weight.fill_(0.1)
        head.detail_proj.bias.fill_(0.2)

    out = head.forward_head(feats, head.cv2, head.cv3)["boxes"]
    bases = torch.cat([head.cv2[i](feats[i]).view(1, 4, -1) for i in range(3)], dim=-1)
    diff = out - bases

    p2_n = 16 * 16
    assert diff[..., :p2_n].abs().max().item() <= 0.500001
    assert torch.count_nonzero(diff[..., :p2_n]).item() > 0
    assert torch.allclose(diff[..., p2_n:], torch.zeros_like(diff[..., p2_n:]), atol=0, rtol=0)


def test_detail_branch_receives_gradients():
    head = DetailCorrectedDetect(nc=10, max_correction_cell=0.5, ch=(32, 64, 128))
    head.train()
    feats = [
        torch.randn(1, 32, 12, 12, requires_grad=True),
        torch.randn(1, 64, 6, 6, requires_grad=True),
        torch.randn(1, 128, 3, 3, requires_grad=True),
    ]
    boxes = head.forward_head(feats, head.cv2, head.cv3)["boxes"]
    boxes.square().mean().backward()
    assert head.detail_proj.weight.grad is not None
    assert torch.isfinite(head.detail_proj.weight.grad).all()
    assert head.detail_proj.weight.grad.abs().sum().item() > 0
