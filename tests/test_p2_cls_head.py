from __future__ import annotations

from pathlib import Path

import torch

from src.p2_refine import P2Refine


ROOT = Path(__file__).resolve().parents[1]


def test_refiner_budget_and_identity_gate():
    block = P2Refine(40, 40, alpha_init=0.0).eval()
    assert sum(p.numel() for p in block.parameters()) == 2121
    x = torch.randn(2, 40, 32, 32)
    with torch.no_grad():
        y = block(x)
    assert torch.equal(x, y)


def test_c6_source_isolates_refinement_to_classification():
    text = (ROOT / "src" / "p2_cls_head.py").read_text(encoding="utf-8")
    assert "box_head[i](x[i])" in text
    assert "cls_inputs = [self.p2_cls_refine(x[0]), x[1], x[2]]" in text
    assert '"feats": x' in text
    assert "reg_max=1" in text


def test_model_builder_keeps_stock_n2b_feature_indices_for_c6():
    text = (ROOT / "src" / "model_builder.py").read_text(encoding="utf-8")
    assert 'c6_p2_cls_refine' in text
    assert "_detect_line(cfg, [19, 22, 25])" in text
    assert "P2ClsDetect" in text
