import torch

from src.custom_blocks import SPRDown


def test_sprdown_even_shape_and_grad():
    m = SPRDown(64, 128)
    x = torch.randn(2, 64, 40, 40, requires_grad=True)
    y = m(x)

    assert y.shape == (2, 128, 20, 20)

    y.mean().backward()
    assert m.phase_scale.grad is not None
    assert torch.isfinite(m.phase_scale.grad).all()


def test_sprdown_odd_shape_is_safe():
    m = SPRDown(32, 48)
    x = torch.randn(1, 32, 41, 39)
    y = m(x)

    # Right/bottom replicate padding followed by 2x phase decomposition.
    assert y.shape == (1, 48, 21, 20)


def test_phase_weights_are_normalized():
    m = SPRDown(16, 32)
    x = torch.randn(2, 16, 20, 20)
    phases = m._polyphase_split(m._pad_to_even(x))
    weights = m.phase_weights(phases)

    assert weights.shape == (2, 16, 4)
    assert torch.allclose(
        weights.sum(dim=2),
        torch.ones(2, 16),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sprdown_rejects_non_stride2():
    try:
        SPRDown(16, 32, s=1)
    except ValueError:
        return
    raise AssertionError("SPRDown must reject s != 2")
