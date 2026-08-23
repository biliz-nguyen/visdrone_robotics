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
    """Return canonical SPR-Down placement names.

    Legacy SPR configs without ``spr_placements`` are interpreted as the
    original Stage-1 P4->P5 placement so older experiments remain readable.
    """
    raw = cfg.get("spr_placements")
    if raw is None:
        return ["p4_p5"] if cfg.get("backbone_down") == "sprdown" else []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("spr_placements must be a list/tuple")
    unknown = set(raw) - SPR_PLACEMENT_SET
    if unknown:
        raise ValueError(f"Unknown SPR placements: {sorted(unknown)}")
    return [stage for stage in SPR_PLACEMENT_ORDER if stage in set(raw)]


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

    resolved["dataset_root"] = str(Path(local["dataset_root"]).expanduser().resolve())
    resolved["dataset_format"] = local.get("dataset_format", "visdrone_official")
    resolved["train_images"] = local.get("train_images", "VisDrone2019-DET-train/images")
    resolved["val_images"] = local.get("val_images", "VisDrone2019-DET-val/images")
    resolved["test_images"] = local.get("test_images", "VisDrone2019-DET-test-dev/images")
    resolved["train_annotations"] = local.get(
        "train_annotations", "VisDrone2019-DET-train/annotations"
    )
    resolved["val_annotations"] = local.get(
        "val_annotations", "VisDrone2019-DET-val/annotations"
    )
    resolved["test_annotations"] = local.get(
        "test_annotations", "VisDrone2019-DET-test-dev/annotations"
    )
    resolved["test_image"] = local.get("test_image", "")

    resolved["project_root"] = str(ROOT)
    resolved["ultra_repo"] = str(ROOT / "third_party" / "ultralytics")
    resolved["generated_dir"] = _resolve_local_path(
        local.get("generated_dir"), ROOT / "generated"
    )
    resolved["runs_dir"] = _resolve_local_path(local.get("runs_dir"), ROOT / "runs")
    resolved["state_dir"] = _resolve_local_path(local.get("state_dir"), ROOT / "state")
    resolved["outputs_dir"] = _resolve_local_path(
        local.get("outputs_dir"), ROOT / "outputs"
    )

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

    # Placement study contract: architecture is the only variable.
    if placements:
        if cfg["loss_mode"] != "standard":
            raise ValueError("SPR placement screening must keep standard loss")
        if cfg["attention"] != "none":
            raise ValueError("SPR placement screening must keep attention disabled")
        if int(cfg["reg_max"]) != 16:
            raise ValueError("SPR placement screening must keep reg_max=16")

    t = cfg["train"]
    if int(t["batch"]) != int(t["nbs"]):
        print(
            "INFO: batch != nbs. Ultralytics will use gradient accumulation "
            "so the nominal batch remains nbs."
        )


def experiment_tag(cfg: dict[str, Any]) -> str:
    if cfg["loss_mode"] == "standard":
        loss_tag = "standard"
    else:
        nwd = cfg["nwd"]
        loss_tag = (
            f"ciou{int(float(nwd['ciou_weight']) * 100)}_"
            f"nwd{int(float(nwd['nwd_weight']) * 100)}"
        )

    placements = normalize_spr_placements(cfg)
    if placements:
        short = {"p2_p3": "p2p3", "p3_p4": "p3p4", "p4_p5": "p4p5"}
        arch_tag = "spr-" + "-".join(short[p] for p in placements)
    else:
        arch_tag = cfg["backbone_down"]

    return (
        f"{arch_tag}_reg{int(cfg['reg_max'])}_{loss_tag}_"
        f"attn-{cfg['attention']}_{int(cfg['train']['epochs'])}e_"
        f"seed{int(cfg['seed'])}"
    )
