from __future__ import annotations

import copy

import torch

from src.rep_neck import RepC3k2, RepConvUnit, switch_reparameterized_neck_to_deploy
from src.ultralytics_patch import _insert_into_frozenset


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


def test_asymmetric_repconv_matches_c3k2_bottleneck_width_and_fuses():
    """C3k2(c3k=False) first bottleneck conv is c -> c/2, so N1 must match it."""
    torch.manual_seed(9)
    m = RepConvUnit(16, 8).eval()
    assert m.in_channels == 16
    assert m.out_channels == 8
    assert m.rbr_identity is None
    x = torch.randn(2, 16, 13, 15)
    with torch.no_grad():
        y0 = m(x)
    m.switch_to_deploy()
    with torch.no_grad():
        y1 = m(x)
    assert y0.shape == (2, 8, 13, 15)
    assert y1.shape == y0.shape
    assert m.reparam.in_channels == 16
    assert m.reparam.out_channels == 8
    assert _max_abs(y0, y1) < 2e-5


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

    # Stock Ultralytics C3k2(c3k=False) uses Bottleneck(self.c, self.c)
    # with Bottleneck default e=0.5. N1 must therefore use hidden -> hidden/2
    # on the first 3x3 before returning hidden channels on the second 3x3.
    for block in m.m:
        assert block.cv1.in_channels == m.hidden
        assert block.cv1.out_channels == max(1, m.hidden // 2)
        assert block.cv2.conv.in_channels == max(1, m.hidden // 2)
        assert block.cv2.conv.out_channels == m.hidden


def test_deploy_removes_parallel_rep_branches():
    m = RepC3k2(32, 32, n=2, shortcut=True).eval()
    switch_reparameterized_neck_to_deploy(m)
    assert m.deploy is True
    for block in m.m:
        assert block.cv1.deploy is True
        assert hasattr(block.cv1, "reparam")
        assert not hasattr(block.cv1, "rbr_dense")
        assert not hasattr(block.cv1, "rbr_1x1")
        # The internal C3k2 bottleneck is asymmetric, so there was no identity branch.
        assert block.cv1.in_channels != block.cv1.out_channels


def test_repeat_modules_patch_accepts_inline_comment_after_frozenset_open():
    # Mirrors the exact formatting used by the pinned Ultralytics tasks.py:
    # repeat_modules = frozenset(  # modules with 'repeat' arguments
    #     {
    #         C3k2,
    #     }
    # )
    text = """
repeat_modules = frozenset(  # modules with 'repeat' arguments
    {
        C3k2,
        C2f,
    }
)
"""
    patched = _insert_into_frozenset(text, "repeat_modules", ["RepC3k2"])
    assert "RepC3k2," in patched
    assert "C3k2," in patched
    assert "# modules with 'repeat' arguments" in patched
    # Idempotence matters because prepare_runtime() may patch more than once.
    patched_twice = _insert_into_frozenset(patched, "repeat_modules", ["RepC3k2"])
    assert patched_twice == patched
