from __future__ import annotations

from pathlib import Path

import yaml

from src.model_builder import build_model_yaml


def test_dcr_builder_emits_custom_head(tmp_path: Path):
    cfg = {
        "generated_dir": str(tmp_path),
        "experiment_tag": "dcr_builder",
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
        "head_mode": "detail_corrected",
        "head_reg_bins": [],
        "detail_max_correction_cell": 0.5,
    }
    path = build_model_yaml(cfg)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["reg_max"] == 1
    assert data["backbone"][7][2] == "SPRDown"
    assert data["head"][-1][2] == "DetailCorrectedDetect"
    assert data["head"][-1][3] == ["nc", 0.5]
