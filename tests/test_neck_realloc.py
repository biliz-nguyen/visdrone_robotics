from __future__ import annotations

from pathlib import Path

import pytest

from src.config import normalize_neck_channels
from src.model_builder import build_model_yaml


def _cfg(tmp_path: Path) -> dict:
    return {
        "generated_dir": str(tmp_path),
        "experiment_tag": "n2-test",
        "backbone_down": "sprdown",
        "spr_placements": ["p4_p5"],
        "head_mode": "standard",
        "head_reg_bins": [],
        "neck_mode": "realloc",
        "neck_channels_nominal": {"p2": 160, "p3": 256, "p4": 384},
        "reg_max": 1,
        "attention": "none",
        "attention_cfg": {
            "eca_kernel": 3,
            "ca_reduction": 32,
            "ca_min_channels": 8,
            "rlca_alpha_init": 0.1,
        },
    }


def test_n2_reallocation_is_locked_before_training():
    cfg = {"neck_mode": "realloc", "neck_channels_nominal": {"p2": 160, "p3": 256, "p4": 384}}
    assert normalize_neck_channels(cfg) == {"p2": 160, "p3": 256, "p4": 384}

    bad = {"neck_mode": "realloc", "neck_channels_nominal": {"p2": 192, "p3": 256, "p4": 384}}
    with pytest.raises(ValueError):
        normalize_neck_channels(bad)


def test_standard_neck_widths_remain_unchanged():
    assert normalize_neck_channels({"neck_mode": "standard"}) == {"p2": 128, "p3": 256, "p4": 512}
    assert normalize_neck_channels({"neck_mode": "rep"}) == {"p2": 128, "p3": 256, "p4": 512}


def test_n2_yaml_moves_capacity_to_fine_scale(tmp_path: Path):
    path = build_model_yaml(_cfg(tmp_path))
    text = path.read_text(encoding="utf-8")

    # Five standard C3k2 fusion blocks, no attention and no RepC3k2.
    assert text.count("C3k2, [") >= 10  # backbone + five neck blocks
    assert "RepC3k2" not in text

    # N2 nominal widths: P2=160 (+25%), P3=256, P4=384 (-25%).
    assert text.count("C3k2, [384, false]") == 2
    assert text.count("C3k2, [256, false]") >= 2
    assert text.count("C3k2, [160, false]") == 1

    # Bottom-up downsampling follows the reallocated output widths.
    assert "Conv, [256, 3, 2]" in text
    assert "Conv, [384, 3, 2]" in text
    assert "Conv, [512, 3, 2]" not in text.split("head:", 1)[1]

    # Detection scales stay P2/P3/P4 with the same layer topology.
    assert "[[19, 22, 25], 1, Detect, [nc]]" in text
