from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect

from src.qoc_head import QualityOverconfidenceDetect
from src.qoc_loss import adaptive_qoc_margin, quality_overconfidence_penalty


def _nparams(module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_qoc_head_has_exact_h1_prediction_budget():
    ch = (32, 64, 128)
    h1 = Detect(nc=10, reg_max=1, end2end=False, ch=ch)
    qoc = QualityOverconfidenceDetect(
        nc=10,
        qoc_lambda=0.25,
        qoc_margin=0.05,
        qoc_tiny_threshold=16.0,
        qoc_tiny_margin_bonus=0.05,
        end2end=False,
        ch=ch,
    )
    assert qoc.reg_max == 1
    assert _nparams(qoc) == _nparams(h1)
    assert [m[-1].out_channels for m in qoc.cv2] == [4, 4, 4]
    assert [m[-1].out_channels for m in qoc.cv3] == [10, 10, 10]


def test_qoc_penalty_is_one_sided():
    confidence = torch.tensor([0.80, 0.40, 0.55], requires_grad=True)
    quality = torch.tensor([0.50, 0.70, 0.50])
    weight = torch.ones(3)
    loss = quality_overconfidence_penalty(confidence, quality, weight, margin=0.05, normalizer=3.0)
    expected = torch.tensor((0.25**2) / 3.0)
    assert torch.allclose(loss.detach(), expected, atol=1e-8, rtol=0)

    loss.backward()
    assert confidence.grad is not None
    assert confidence.grad[0].abs() > 0
    assert confidence.grad[1].item() == 0.0
    assert confidence.grad[2].item() == 0.0


def test_q3_midpoint_margin_is_exact_and_continuous():
    min_side = torch.tensor([0.0, 4.0, 8.0, 16.0, 32.0])
    margin = adaptive_qoc_margin(min_side, base_margin=0.05, tiny_threshold=16.0, tiny_margin_bonus=0.05)
    expected = torch.tensor([0.10, 0.0875, 0.075, 0.05, 0.05])
    assert torch.allclose(margin, expected, atol=1e-7, rtol=0)


def test_q3_penalty_lies_between_q1_and_q2_for_tiny_samples():
    confidence = torch.tensor([0.70, 0.70, 0.70, 0.70])
    quality = torch.tensor([0.50, 0.50, 0.50, 0.50])
    weight = torch.ones(4)
    min_side = torch.tensor([4.0, 8.0, 16.0, 32.0])

    q1 = quality_overconfidence_penalty(confidence, quality, weight, margin=0.05, normalizer=4.0)
    q2_margin = adaptive_qoc_margin(min_side, 0.05, 16.0, 0.10)
    q3_margin = adaptive_qoc_margin(min_side, 0.05, 16.0, 0.05)
    q2 = quality_overconfidence_penalty(confidence, quality, weight, margin=q2_margin, normalizer=4.0)
    q3 = quality_overconfidence_penalty(confidence, quality, weight, margin=q3_margin, normalizer=4.0)

    assert q2 <= q3 <= q1
    assert torch.allclose(q2_margin[2:], q3_margin[2:], atol=0, rtol=0)
    assert torch.allclose(q3_margin[2:], torch.tensor([0.05, 0.05]), atol=0, rtol=0)


def test_qoc_metadata_validation():
    ch = (32, 64, 128)
    invalid_kwargs = [
        {"qoc_lambda": -0.1},
        {"qoc_margin": 1.0},
        {"qoc_tiny_threshold": 0.0},
        {"qoc_tiny_margin_bonus": -0.1},
        {"qoc_margin": 0.95, "qoc_tiny_margin_bonus": 0.10},
    ]
    for kwargs in invalid_kwargs:
        try:
            QualityOverconfidenceDetect(nc=10, ch=ch, **kwargs)
            raise AssertionError(f"invalid metadata should fail: {kwargs}")
        except ValueError:
            pass
