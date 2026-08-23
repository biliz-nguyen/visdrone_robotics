import torch
import torch.nn.functional as F

from src.sprdown_v2 import SPRDownV2


def _explicit_phase_reference(x, weights):
    """Reference weighted sum of the four 2x2 phases per channel."""
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    phases = torch.stack((x00, x01, x10, x11), dim=2)
    w = weights.view(1, weights.shape[0], 4, 1, 1)
    return (phases * w).sum(dim=2)


def test_sprdown_v2_even_shape_and_grad():
    m = SPRDownV2(64, 128)
    x = torch.randn(2, 64, 40, 40, requires_grad=True)
    y = m(x)

    assert y.shape == (2, 128, 20, 20)

    y.mean().backward()
    assert m.phase_mix.weight.grad is not None
    assert torch.isfinite(m.phase_mix.weight.grad).all()


def test_sprdown_v2_odd_shape_is_safe():
    m = SPRDownV2(32, 48)
    x = torch.randn(1, 32, 41, 39)
    y = m(x)
    assert y.shape == (1, 48, 21, 20)


def test_uniform_initialization_matches_2x2_average():
    m = SPRDownV2(8, 8)
    x = torch.randn(2, 8, 20, 18)

    with torch.no_grad():
        y = m.phase_reassemble(x)
        ref = F.avg_pool2d(x, kernel_size=2, stride=2)

    assert torch.allclose(y, ref, atol=1e-6, rtol=1e-6)


def test_fused_kernel_matches_explicit_four_phase_reassembly():
    torch.manual_seed(7)
    m = SPRDownV2(6, 10)
    x = torch.randn(2, 6, 20, 18)

    with torch.no_grad():
        m.phase_mix.weight.copy_(torch.randn_like(m.phase_mix.weight))
        y = m.phase_reassemble(x)
        ref = _explicit_phase_reference(x, m.phase_mix.weight)

    assert torch.allclose(y, ref, atol=1e-6, rtol=1e-6)


def test_sprdown_v2_rejects_non_stride2():
    try:
        SPRDownV2(16, 32, s=1)
    except ValueError:
        return
    raise AssertionError("SPRDownV2 must reject s != 2")
