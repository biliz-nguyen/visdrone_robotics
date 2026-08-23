from __future__ import annotations

from pathlib import Path
import textwrap

from src.config import normalize_spr_placements


_STAGE_TO_CHANNELS = {
    "p2_p3": 256,
    "p3_p4": 512,
    "p4_p5": 1024,
}


def _downsample_line(cfg: dict, stage: str) -> str:
    """Build one backbone downsample line for a named stage."""
    c2 = _STAGE_TO_CHANNELS[stage]
    placements = set(normalize_spr_placements(cfg))

    if stage in placements:
        module = "SPRDown"
    elif cfg["backbone_down"] == "aconv" and stage == "p4_p5":
        # Legacy compatibility only; placement screening never uses AConv.
        module = "AConv"
    else:
        module = "Conv"

    return f"  - [-1, 1, {module}, [{c2}, 3, 2]]"


def build_model_yaml(cfg: dict) -> Path:
    generated = Path(cfg["generated_dir"])
    generated.mkdir(parents=True, exist_ok=True)
    model_yaml = generated / f"model_{cfg['experiment_tag']}.yaml"

    down_p2_p3 = _downsample_line(cfg, "p2_p3")
    down_p3_p4 = _downsample_line(cfg, "p3_p4")
    down_p4_p5 = _downsample_line(cfg, "p4_p5")

    attn = cfg["attention"]
    attention_cfg = cfg["attention_cfg"]

    if attn == "none":
        attention_line = None
    elif attn == "eca":
        attention_line = (
            "  - [-1, 1, ECA, "
            f"[128, {int(attention_cfg['eca_kernel'])}]]"
        )
    elif attn == "ca":
        attention_line = (
            "  - [-1, 1, CoordAtt, "
            f"[128, {int(attention_cfg['ca_reduction'])}, "
            f"{int(attention_cfg['ca_min_channels'])}]]"
        )
    elif attn == "rlca":
        attention_line = (
            "  - [-1, 1, ResidualLiteCA, "
            f"[128, {int(attention_cfg['ca_reduction'])}, "
            f"{int(attention_cfg['ca_min_channels'])}, "
            f"{float(attention_cfg['rlca_alpha_init'])}]]"
        )
    else:
        raise ValueError(attn)

    if attn == "none":
        head = """
head:
  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, false]]

  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, false]]

  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, false]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, false]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, false]]

  - [[19, 22, 25], 1, Detect, [nc]]
"""
    else:
        head = f"""
head:
  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, false]]

  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, false]]

  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, false]]

{attention_line}

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, false]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, false]]

  - [[20, 23, 26], 1, Detect, [nc]]
"""

    text = f"""
nc: 10
reg_max: {int(cfg['reg_max'])}

scale: n
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, false, 0.25]]
{down_p2_p3}
  - [-1, 2, C3k2, [512, false, 0.25]]
{down_p3_p4}
  - [-1, 2, C3k2, [512, true]]
{down_p4_p5}
  - [-1, 2, C3k2, [1024, true]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

{head}
"""

    model_yaml.write_text(
        textwrap.dedent(text).strip() + "\n",
        encoding="utf-8",
    )
    return model_yaml
