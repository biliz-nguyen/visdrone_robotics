from __future__ import annotations

from pathlib import Path

import yaml

from src.model_builder import build_model_yaml


def test_paired_boundary_builder_uses_s1_and_custom_head(tmp_path: Path):
    cfg = {
        "generated_dir": str(tmp_path),
        "experiment_tag": "pbr_test",
        "backbone_down": "sprdown",
        "spr_placements": ["p4_p5"],
        "reg_max": 1,
        "attention": "none",
        "attention_cfg": {
            "eca_kernel": 3,
            "ca_reduction": 32,
            "ca_min_channels": 8,
            "rlca_alpha_init": 0.10,
        },
        "head_mode": "paired_boundary",
        "paired_ratio_limit": 0.99,
    }
    path = build_model_yaml(cfg)
    data = yaml.safe_load(path.read_text())
    modules = [layer[2] for layer in data["backbone"]]
    assert modules[3] == "Conv"
    assert modules[5] == "Conv"
    assert modules[7] == "SPRDown"
    detect = data["head"][-1]
    assert detect[0] == [19, 22, 25]
    assert detect[2] == "PairedBoundaryDetect"
    assert detect[3] == ["nc", 0.99]
    assert data["reg_max"] == 1
