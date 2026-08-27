from __future__ import annotations

from types import SimpleNamespace

import torch

from src.scale_velocity_loss import _normalize_hyp_for_loss, compute_velocity_weights, scale_group_from_min_side


def test_scale_groups_follow_locked_16_32_pixel_boundaries():
    x = torch.tensor([4.0, 15.999, 16.0, 24.0, 31.999, 32.0, 80.0])
    got = scale_group_from_min_side(x, 16.0, 32.0)
    expected = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    assert torch.equal(got, expected)


def test_equal_learning_progress_keeps_stock_weights():
    progress = torch.tensor([0.55, 0.55, 0.55])
    counts = torch.tensor([100.0, 80.0, 20.0])
    weights = compute_velocity_weights(progress, counts)
    assert torch.allclose(weights, torch.ones(3), atol=1e-7)


def test_slower_tiny_learning_is_upweighted_without_global_gradient_inflation():
    # Larger residual progress means less improvement from the first-epoch
    # reference. Tiny is deliberately lagging here.
    progress = torch.tensor([0.90, 0.50, 0.40])
    counts = torch.tensor([120.0, 120.0, 120.0])
    weights = compute_velocity_weights(progress, counts, alpha=0.50, weight_min=0.75, weight_max=1.25)

    assert float(weights[0]) > 1.0
    assert float(weights[0]) > float(weights[1]) > float(weights[2])
    assert bool((weights >= 0.75).all())
    assert bool((weights <= 1.25).all())

    weighted_mean = (weights * counts).sum() / counts.sum()
    assert abs(float(weighted_mean) - 1.0) < 0.02


def test_absent_group_is_left_neutral():
    progress = torch.tensor([0.80, 0.60, 0.30])
    counts = torch.tensor([20.0, 0.0, 10.0])
    weights = compute_velocity_weights(progress, counts)
    assert float(weights[1]) == 1.0


def test_hyp_container_supports_dict_and_namespace_without_mutating_input():
    hyp_dict = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
    normalized = _normalize_hyp_for_loss(hyp_dict)
    assert isinstance(normalized, SimpleNamespace)
    assert normalized.box == 7.5
    assert normalized.cls == 0.5
    assert normalized.dfl == 1.5
    assert hyp_dict == {"box": 7.5, "cls": 0.5, "dfl": 1.5}

    hyp_ns = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    assert _normalize_hyp_for_loss(hyp_ns) is hyp_ns
