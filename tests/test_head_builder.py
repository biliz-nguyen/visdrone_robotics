from pathlib import Path

import yaml

from src.model_builder import build_model_yaml


def base_cfg(tmp_path: Path) -> dict:
    return {
        "generated_dir": str(tmp_path),
        "experiment_tag": "head_test",
        "backbone_down": "sprdown",
        "spr_placements": ["p4_p5"],
        "reg_max": 16,
        "head_mode": "standard",
        "head_reg_bins": [],
        "attention": "none",
        "attention_cfg": {
            "eca_kernel": 3,
            "ca_reduction": 32,
            "ca_min_channels": 8,
            "rlca_alpha_init": 0.1,
        },
    }


def test_standard_direct_head_uses_detect(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["reg_max"] = 1
    p = build_model_yaml(cfg)
    d = yaml.safe_load(p.read_text())
    assert d["reg_max"] == 1
    assert d["head"][-1][2] == "Detect"


def test_stride_reg_head_uses_level_specific_bins(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["head_mode"] = "stride_reg"
    cfg["head_reg_bins"] = [16, 8, 4]
    p = build_model_yaml(cfg)
    d = yaml.safe_load(p.read_text())
    assert d["head"][-1][2] == "StrideRegDetect"
    assert d["head"][-1][3][1] == [16, 8, 4]


def test_aggressive_hybrid_bins_are_preserved(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["head_mode"] = "stride_reg"
    cfg["head_reg_bins"] = [16, 4, 1]
    p = build_model_yaml(cfg)
    d = yaml.safe_load(p.read_text())
    assert d["head"][-1][3][1] == [16, 4, 1]
