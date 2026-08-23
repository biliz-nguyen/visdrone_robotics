from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "config" / "experiment.yaml"
DEFAULT_LOCAL = ROOT / "config" / "local.yaml"

CLASS_NAMES = [
    "awning-tricycle",
    "bicycle",
    "bus",
    "car",
    "motor",
    "pedestrian",
    "people",
    "tricycle",
    "truck",
    "van",
]

SPR_PLACEMENT_ORDER = ("p2_p3", "p3_p4", "p4_p5")
SPR_PLACEMENT_SET = set(SPR_PLACEMENT_ORDER)
HEAD_MODES = {"standard", "stride_reg", "quality_overconfidence"}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _resolve_local_path(value: str | None, default: Path) -> str:
    if value:
        return str(Path(value).expanduser().resolve())
    return str(default.resolve())


def normalize_spr_placements(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("spr_placements")
    if raw is None:
        return ["p4_p5"] if cfg.get("backbone_down") == "sprdown" else []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("spr_placements must be a list/tuple")
    unknown = set(raw) - SPR_PLACEMENT_SET
    if unknown:
        raise ValueError(f"Unknown SPR placements: {sorted(unknown)}")
    return [stage for stage in SPR_PLACEMENT_ORDER if stage in set(raw)]


def normalize_head_bins(cfg: dict[str, Any]) -> list[int]:
    raw = cfg.get("head_reg_bins", [])
    if not isinstance(raw, (list, tuple)):
        raise ValueError("head_reg_bins must be a list/tuple")
    return [int(x) for x in raw]


def load_config(
    experiment_path: str | Path = DEFAULT_EXPERIMENT,
    local_path: str | Path = DEFAULT_LOCAL,
) -> dict[str, Any]:
    experiment_path = Path(experiment_path).resolve()
    local_path = Path(local_path).resolve()

    if not local_path.exists():
        example = ROOT / "config" / "local.example.yaml"
        raise FileNotFoundError(
            f"\nMissing local config: {local_path}\n"
            f"Create it first:\n  cp {example} {local_path}\n"
            "Then set dataset_root."
        )

    exp = _read_yaml(experiment_path)
    local = _read_yaml(local_path)

    preset = exp.get("preset")
    presets = exp.get("presets", {})
    if preset not in presets:
        raise ValueError(f"Unknown preset={preset!r}. Available: {sorted(presets)}")

    resolved = deepcopy(exp)
    resolved.update(deepcopy(presets[preset]))
    resolved["preset"] = preset
    resolved["spr_placements"] = normalize_spr_placements(resolved)
    resolved["head_mode"] = resolved.get("head_mode", "standard")
    resolved["head_reg_bins"] = normalize_head_bins(resolved)
    resolved["study"] = resolved.get("study", "placement")

    resolved["dataset_root"] = str(Path(local["dataset_root"]).expanduser().resolve())
    resolved["dataset_format"] = local.get("dataset_format", "visdrone_official")
    resolved["train_images"] = local.get("train_images", "VisDrone2019-DET-train/images")
    resolved["val_images"] = local.get("val_images", "VisDrone2019-DET-val/images")
    resolved["test_images"] = local.get("test_images", "VisDrone2019-DET-test-dev/images")
    resolved["train_annotations"] = local.get("train_annotations", "VisDrone2019-DET-train/annotations")
    resolved["val_annotations"] = local.get("val_annotations", "VisDrone2019-DET-val/annotations")
    resolved["test_annotations"] = local.get("test_annotations", "VisDrone2019-DET-test-dev/annotations")
    resolved["test_image"] = local.get("test_image", "")

    resolved["project_root"] = str(ROOT)
    resolved["ultra_repo"] = str(ROOT / "third_party" / "ultralytics")
    resolved["generated_dir"] = _resolve_local_path(local.get("generated_dir"), ROOT / "generated")
    resolved["runs_dir"] = _resolve_local_path(local.get("runs_dir"), ROOT / "runs")
    resolved["state_dir"] = _resolve_local_path(local.get("state_dir"), ROOT / "state")
    resolved["outputs_dir"] = _resolve_local_path(local.get("outputs_dir"), ROOT / "outputs")

    _validate(resolved)
    resolved["experiment_tag"] = experiment_tag(resolved)
    return resolved


def _validate(cfg: dict[str, Any]) -> None:
    if cfg["backbone_down"] not in {"conv", "aconv", "sprdown"}:
        raise ValueError(cfg["backbone_down"])
    if cfg["loss_mode"] not in {"standard", "hybrid_nwd"}:
        raise ValueError(cfg["loss_mode"])
    if cfg["attention"] not in {"none", "eca", "ca", "rlca"}:
        raise ValueError(cfg["attention"])
    if int(cfg["reg_max"]) not in {1, 2, 4, 8, 16}:
        raise ValueError(cfg["reg_max"])
    if cfg.get("head_mode", "standard") not in HEAD_MODES:
        raise ValueError(cfg.get("head_mode"))
    if cfg.get("dataset_format") not in {"visdrone_official", "yolo"}:
        raise ValueError("dataset_format must be 'visdrone_official' or 'yolo'")
    if cfg.get("pretrained", False):
        raise ValueError("This project is locked to scratch training: pretrained=false")

    placements = normalize_spr_placements(cfg)
    if cfg["backbone_down"] == "sprdown" and not placements:
        raise ValueError("SPR-Down preset must enable at least one placement")
    if cfg["backbone_down"] == "conv" and placements:
        raise ValueError("Conv baseline cannot contain SPR placements")
    if cfg["backbone_down"] == "aconv" and placements:
        raise ValueError("AConv legacy mode cannot be mixed with SPR placements")

    study = cfg.get("study", "placement")
    head_mode = cfg.get("head_mode", "standard")

    if study == "placement":
        if placements:
            if cfg["loss_mode"] != "standard":
                raise ValueError("SPR placement screening must keep standard loss")
            if cfg["attention"] != "none":
                raise ValueError("SPR placement screening must keep attention disabled")
            if int(cfg["reg_max"]) != 16:
                raise ValueError("SPR placement screening must keep reg_max=16")
            if head_mode != "standard":
                raise ValueError("SPR placement screening must keep the standard Detect head")

    elif study == "head":
        if placements != ["p4_p5"]:
            raise ValueError("Head study is locked to the confirmed S1 SPR placement P4->P5")
        if cfg["loss_mode"] != "standard" or cfg["attention"] != "none":
            raise ValueError("Head study must keep standard loss and no attention")
        if head_mode == "stride_reg":
            bins = normalize_head_bins(cfg)
            if len(bins) != 3:
                raise ValueError("stride_reg head requires exactly three level bin counts")
            if any(x not in {1, 2, 4, 8, 16} for x in bins):
                raise ValueError("head_reg_bins values must be one of 1,2,4,8,16")
            if bins != sorted(bins, reverse=True):
                raise ValueError("head_reg_bins must be non-increasing from P2 to P4")
        elif head_mode == "quality_overconfidence":
            if int(cfg["reg_max"]) != 1:
                raise ValueError("QOC is locked to DFL-free reg_max=1")
            lam = float(cfg.get("qoc_lambda", 0.25))
            margin = float(cfg.get("qoc_margin", 0.05))
            tiny_threshold = float(cfg.get("qoc_tiny_threshold", 16.0))
            tiny_bonus = float(cfg.get("qoc_tiny_margin_bonus", 0.0))
            if lam < 0:
                raise ValueError("qoc_lambda must be non-negative")
            if not (0.0 <= margin < 1.0):
                raise ValueError("qoc_margin must be in [0,1)")
            if tiny_threshold <= 0:
                raise ValueError("qoc_tiny_threshold must be positive")
            if tiny_bonus < 0 or margin + tiny_bonus >= 1.0:
                raise ValueError("invalid qoc_tiny_margin_bonus")
        elif int(cfg["reg_max"]) != 1:
            raise ValueError("Standard-head variant in the head study is reserved for the DFL-free reg_max=1 control")
    else:
        raise ValueError(f"Unknown study={study!r}")

    t = cfg["train"]
    if int(t["batch"]) != int(t["nbs"]):
        print("INFO: batch != nbs. Ultralytics will use gradient accumulation so the nominal batch remains nbs.")


def _tag_float(x: float) -> str:
    return str(float(x)).replace(".", "p")


def experiment_tag(cfg: dict[str, Any]) -> str:
    loss_tag = "standard" if cfg["loss_mode"] == "standard" else "hybrid"
    placements = normalize_spr_placements(cfg)
    if placements:
        short = {"p2_p3": "p2p3", "p3_p4": "p3p4", "p4_p5": "p4p5"}
        arch_tag = "spr-" + "-".join(short[p] for p in placements)
    else:
        arch_tag = cfg["backbone_down"]

    if cfg.get("study", "placement") == "head":
        mode = cfg.get("head_mode")
        if mode == "stride_reg":
            bins = "-".join(str(x) for x in normalize_head_bins(cfg))
            head_tag = f"snr-{bins}"
        elif mode == "quality_overconfidence":
            lam = _tag_float(float(cfg.get("qoc_lambda", 0.25)))
            margin = _tag_float(float(cfg.get("qoc_margin", 0.05)))
            tiny_threshold = float(cfg.get("qoc_tiny_threshold", 16.0))
            tiny_bonus = float(cfg.get("qoc_tiny_margin_bonus", 0.0))
            if tiny_bonus > 0:
                head_tag = f"qoc2-l{lam}-m{margin}-t{_tag_float(tiny_threshold)}-b{_tag_float(tiny_bonus)}"
            else:
                head_tag = f"qoc-l{lam}-m{margin}"
        else:
            head_tag = "direct-r1"
        return (
            f"{arch_tag}_{head_tag}_{loss_tag}_attn-{cfg['attention']}_"
            f"{int(cfg['train']['epochs'])}e_seed{int(cfg['seed'])}"
        )

    return (
        f"{arch_tag}_reg{int(cfg['reg_max'])}_{loss_tag}_"
        f"attn-{cfg['attention']}_{int(cfg['train']['epochs'])}e_"
        f"seed{int(cfg['seed'])}"
    )
