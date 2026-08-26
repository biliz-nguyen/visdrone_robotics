from __future__ import annotations

from pathlib import Path

import pytest

from src.config import normalize_neck_channels
from src.model_builder import build_model_yaml


def _cfg(tmp_path: Path, p4: int = 384) -> dict:
    return {
        "generated_dir": str(tmp_path),
        "experiment_tag": f"realloc-{p4}-test",
        "backbone_down": "sprdown",
        "spr_placements": ["p4_p5"],
        "head_mode": "standard",
        "head_reg_bins": [],
        "neck_mode": "realloc",
        "neck_channels_nominal": {"p2": 160, "p3": 256, "p4": p4},
        "reg_max": 1,
        "attention": "none",
        "attention_cfg": {
            "eca_kernel": 3,
            "ca_reduction": 32,
            "ca_min_channels": 8,
            "rlca_alpha_init": 0.1,
        },
    }


def test_only_registered_reallocations_are_allowed():
    n2 = {"neck_mode": "realloc", "neck_channels_nominal": {"p2": 160, "p3": 256, "p4": 384}}
    n2b = {"neck_mode": "realloc", "neck_channels_nominal": {"p2": 160, "p3": 256, "p4": 416}}
    assert normalize_neck_channels(n2) == {"p2": 160, "p3": 256, "p4": 384}
    assert normalize_neck_channels(n2b) == {"p2": 160, "p3": 256, "p4": 416}

    # 448 is intentionally rejected: its measured preflight was 6.2450 GFLOPs,
    # above the fixed 1.03x-H1 deployment budget. Other unregistered widths are
    # rejected too, preventing a post-hoc width sweep.
    for bad_widths in (
        {"p2": 192, "p3": 256, "p4": 384},
        {"p2": 160, "p3": 320, "p4": 416},
        {"p2": 160, "p3": 256, "p4": 448},
    ):
        with pytest.raises(ValueError):
            normalize_neck_channels({"neck_mode": "realloc", "neck_channels_nominal": bad_widths})


def test_standard_neck_widths_remain_unchanged():
    assert normalize_neck_channels({"neck_mode": "standard"}) == {"p2": 128, "p3": 256, "p4": 512}
    assert normalize_neck_channels({"neck_mode": "rep"}) == {"p2": 128, "p3": 256, "p4": 512}


def _assert_common_yaml(text: str):
    assert text.count("C3k2, [") >= 9
    assert "RepC3k2" not in text
    assert text.count("C3k2, [256, false]") >= 2
    assert text.count("C3k2, [160, false]") == 1
    assert "[[19, 22, 25], 1, Detect, [nc]]" in text


def test_n2_yaml_moves_capacity_to_fine_scale(tmp_path: Path):
    text = build_model_yaml(_cfg(tmp_path, p4=384)).read_text(encoding="utf-8")
    _assert_common_yaml(text)
    assert text.count("C3k2, [384, false]") == 2
    head = text.split("head:", 1)[1]
    assert "Conv, [256, 3, 2]" in head
    assert "Conv, [384, 3, 2]" in head
    assert "Conv, [512, 3, 2]" not in head


def test_n2b_yaml_restores_budgeted_part_of_p4_capacity(tmp_path: Path):
    text = build_model_yaml(_cfg(tmp_path, p4=416)).read_text(encoding="utf-8")
    _assert_common_yaml(text)
    assert text.count("C3k2, [416, false]") == 2
    head = text.split("head:", 1)[1]
    assert "Conv, [256, 3, 2]" in head
    assert "Conv, [416, 3, 2]" in head
    assert "Conv, [448, 3, 2]" not in head
    assert "Conv, [512, 3, 2]" not in head
