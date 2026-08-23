from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect

from src.qoc_head import QualityOverconfidenceDetect
from src.qoc_loss import quality_overconfidence_penalty


def _nparams(module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_qoc_head_has_exact_h1_prediction_budget():
    ch = (32, 64, 128)
    h1 = Detect(nc=10, reg_max=1, end2end=False, ch=ch)
    qoc = QualityOverconfidenceDetect(
        nc=10,
        qoc_lambda=0.25,
        qoc_margin=0.05,
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

    # Only the first item is above quality+margin: 0.80 - 0.55 = 0.25.
    expected = torch.tensor((0.25**2) / 3.0)
    assert torch.allclose(loss.detach(), expected, atol=1e-8, rtol=0)

    loss.backward()
    assert confidence.grad is not None
    assert confidence.grad[0].abs() > 0
    assert confidence.grad[1].item() == 0.0
    assert confidence.grad[2].item() == 0.0


def test_qoc_metadata_validation():
    ch = (32, 64, 128)
    try:
        QualityOverconfidenceDetect(nc=10, qoc_lambda=-0.1, ch=ch)
        raise AssertionError("negative qoc_lambda should fail")
    except ValueError:
        pass

    try:
        QualityOverconfidenceDetect(nc=10, qoc_margin=1.0, ch=ch)
        raise AssertionError("qoc_margin >= 1 should fail")
    except ValueError:
        pass
