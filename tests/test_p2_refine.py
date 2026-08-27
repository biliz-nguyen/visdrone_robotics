from __future__ import annotations

import torch

from src.p2_refine import P2Refine


def test_p2_refine_preserves_shape_and_budget():
    m = P2Refine(40, 40, alpha_init=0.10)
    x = torch.randn(2, 40, 32, 32)
    y = m(x)
    assert y.shape == x.shape
    assert sum(p.numel() for p in m.parameters()) == 2121


def test_p2_refine_zero_gate_is_exact_identity():
    m = P2Refine(40, 40, alpha_init=0.0).eval()
    x = torch.randn(1, 40, 16, 16)
    with torch.no_grad():
        y = m(x)
    assert torch.equal(y, x)


def test_p2_refine_has_trainable_residual_path():
    m = P2Refine(40, 40, alpha_init=0.10)
    x = torch.randn(1, 40, 16, 16, requires_grad=True)
    m(x).mean().backward()
    assert x.grad is not None
    assert m.alpha.grad is not None
    assert torch.isfinite(m.alpha.grad)
