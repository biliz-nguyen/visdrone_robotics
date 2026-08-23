from __future__ import annotations

from pathlib import Path
import textwrap


def build_model_yaml(cfg: dict) -> Path:
    generated = Path(cfg["generated_dir"])
    generated.mkdir(parents=True, exist_ok=True)

    model_yaml = (
        generated
        / f"model_{cfg['experiment_tag']}.yaml"
    )

    if cfg["backbone_down"] == "aconv":
        downsample_line = (
            "  - [-1, 1, AConv, [1024, 3, 2]]"
        )
    elif cfg["backbone_down"] == "sprdown":
        downsample_line = (
            "  - [-1, 1, SPRDown, [1024, 3, 2]]"
        )
    elif cfg["backbone_down"] == "sprdown_v2":
        downsample_line = (
            "  - [-1, 1, SPRDownV2, [1024, 3, 2]]"
        )
    elif cfg["backbone_down"] == "conv":
        downsample_line = (
            "  - [-1, 1, Conv, [1024, 3, 2]]"
        )
    else:
        raise ValueError(cfg["backbone_down"])

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
            f"[128, "
            f"{int(attention_cfg['ca_reduction'])}, "
            f"{int(attention_cfg['ca_min_channels'])}]]"
        )

    elif attn == "rlca":
        attention_line = (
            "  - [-1, 1, ResidualLiteCA, "
            f"[128, "
            f"{int(attention_cfg['ca_reduction'])}, "
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
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, false, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, true]]
{downsample_line}
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
