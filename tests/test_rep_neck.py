from __future__ import annotations

import copy

import torch

from src.rep_neck import RepC3k2, RepConvUnit, switch_reparameterized_neck_to_deploy


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def test_repconv_unit_fuses_equivalently():
    torch.manual_seed(7)
    m = RepConvUnit(16).eval()
    x = torch.randn(2, 16, 11, 13)
    with torch.no_grad():
        y_train_graph = m(x)
    m.switch_to_deploy()
    with torch.no_grad():
        y_deploy = m(x)
    assert m.deploy is True
    assert _max_abs(y_train_graph, y_deploy) < 2e-5


def test_repc3k2_shape_and_deploy_equivalence():
    torch.manual_seed(11)
    m = RepC3k2(48, 32, n=2, shortcut=True).eval()
    x = torch.randn(1, 48, 20, 20)
    deploy = copy.deepcopy(m)
    switch_reparameterized_neck_to_deploy(deploy)
    with torch.no_grad():
        y0 = m(x)
        y1 = deploy(x)
    assert y0.shape == (1, 32, 20, 20)
    assert y1.shape == y0.shape
    assert _max_abs(y0, y1) < 3e-5


def test_deploy_removes_parallel_rep_branches():
    m = RepC3k2(32, 32, n=2, shortcut=True).eval()
    switch_reparameterized_neck_to_deploy(m)
    assert m.deploy is True
    for block in m.m:
        assert block.cv1.deploy is True
        assert hasattr(block.cv1, "reparam")
        assert not hasattr(block.cv1, "rbr_dense")
        assert not hasattr(block.cv1, "rbr_1x1")
        assert not hasattr(block.cv1, "rbr_identity")
