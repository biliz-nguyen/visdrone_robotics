from __future__ import annotations

from pathlib import Path

import yaml

from src.model_builder import build_model_yaml


STAGE_INDEX = {"p2_p3": 3, "p3_p4": 5, "p4_p5": 7}


def _cfg(tmp_path: Path, placements: list[str]) -> dict:
    return {
        "generated_dir": str(tmp_path),
        "experiment_tag": "placement_test",
        "backbone_down": "sprdown" if placements else "conv",
        "spr_placements": placements,
        "reg_max": 16,
        "attention": "none",
        "attention_cfg": {
            "eca_kernel": 3,
            "ca_reduction": 32,
            "ca_min_channels": 8,
            "rlca_alpha_init": 0.10,
        },
    }


def _modules(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    return [layer[2] for layer in data["backbone"]]


def test_all_eight_placement_combinations_build_expected_layers(tmp_path):
    combinations = [
        [],
        ["p4_p5"],
        ["p3_p4"],
        ["p2_p3"],
        ["p3_p4", "p4_p5"],
        ["p2_p3", "p3_p4"],
        ["p2_p3", "p4_p5"],
        ["p2_p3", "p3_p4", "p4_p5"],
    ]

    for i, placements in enumerate(combinations):
        cfg = _cfg(tmp_path / str(i), placements)
        cfg["experiment_tag"] = f"placement_{i}"
        model_yaml = build_model_yaml(cfg)
        modules = _modules(model_yaml)
        enabled = set(placements)
        for stage, idx in STAGE_INDEX.items():
            assert modules[idx] == ("SPRDown" if stage in enabled else "Conv")


def test_placement_does_not_change_backbone_depth_or_head_indices(tmp_path):
    cfg = _cfg(tmp_path, ["p2_p3", "p3_p4", "p4_p5"])
    model_yaml = build_model_yaml(cfg)
    data = yaml.safe_load(model_yaml.read_text())

    assert len(data["backbone"]) == 11
    assert data["head"][-1][0] == [19, 22, 25]
    assert data["head"][-1][2] == "Detect"
