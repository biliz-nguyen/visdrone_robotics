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
HEAD_MODES = {"standard", "stride_reg"}
NECK_MODES = {"standard", "rep", "realloc"}
ASSIGNER_MODES = {"standard", "tiny_center_rank"}
N2_REALLOC_NOMINAL = {"p2": 160, "p3": 256, "p4": 384}
N2B_REALLOC_NOMINAL = {"p2": 160, "p3": 256, "p4": 416}
ALLOWED_REALLOC_NOMINALS = (N2_REALLOC_NOMINAL, N2B_REALLOC_NOMINAL)


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


def normalize_neck_channels(cfg: dict[str, Any]) -> dict[str, int]:
    """Return nominal YAML widths for P2/P3/P4 neck outputs.

    Reallocation screens are locked to two hypotheses only:
    N2  = 160/256/384 (effective 40/64/96 at scale n), and
    N2b = 160/256/416 (effective 40/64/104 at scale n).
    The original N2b P4=448 preflight was rejected before training because it
    exceeded the declared 1.03x-H1 FLOPs budget; 416 is the single
    budget-constrained recovery setting used for the actual N2b screen.
    """
    if cfg.get("neck_mode", "standard") != "realloc":
        return {"p2": 128, "p3": 256, "p4": 512}
    raw = cfg.get("neck_channels_nominal", N2_REALLOC_NOMINAL)
    if not isinstance(raw, dict):
        raise ValueError("neck_channels_nominal must be a mapping with p2/p3/p4")
    out = {k: int(raw[k]) for k in ("p2", "p3", "p4")}
    if out not in ALLOWED_REALLOC_NOMINALS:
        raise ValueError(
            "Reallocation study is locked to registered widths "
            f"{list(ALLOWED_REALLOC_NOMINALS)}, got {out}"
        )
    return out


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
    resolved["neck_mode"] = resolved.get("neck_mode", "standard")
    resolved["neck_channels_nominal"] = normalize_neck_channels(resolved)
    resolved["assigner_mode"] = resolved.get("assigner_mode", "standard")
    resolved["study"] = resolved.get("study", "placement")

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
    resolved["generated_dir"] = _resolve_local_path(local.get("generated_dir"), ROOT / "generated")
    resolved["runs_dir"] = _resolve_local_path(local.get("runs_dir"), ROOT / "runs")
    resolved["state_dir"] = _resolve_local_path(local.get("state_dir"), ROOT / "state")
    resolved["outputs_dir"] = _resolve_local_path(local.get("outputs_dir"), ROOT / "outputs")

    _validate(resolved)
    resolved["experiment_tag"] = experiment_tag(resolved)
    return resolved


def _validate_tiny_center_rank(cfg: dict[str, Any]) -> None:
    q = cfg.get("tiny_center_rank", {})
    tiny_min_side = float(q.get("tiny_min_side", -1))
    # The 16 px threshold is preregistered from the tiny-object definition used
    # in this project. C3-v2 has no extra ranking strength hyperparameter.
    if tiny_min_side != 16.0:
        raise ValueError("C3-v2 locks tiny_min_side=16.0 pixels")


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
    if cfg.get("neck_mode", "standard") not in NECK_MODES:
        raise ValueError(cfg.get("neck_mode"))
    if cfg.get("assigner_mode", "standard") not in ASSIGNER_MODES:
        raise ValueError(cfg.get("assigner_mode"))
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
    neck_mode = cfg.get("neck_mode", "standard")
    assigner_mode = cfg.get("assigner_mode", "standard")

    if study == "placement":
        if assigner_mode != "standard":
            raise ValueError("Placement study must keep standard TAL")
        if placements:
            if cfg["loss_mode"] != "standard":
                raise ValueError("SPR placement screening must keep standard loss")
            if cfg["attention"] != "none":
                raise ValueError("SPR placement screening must keep attention disabled")
            if int(cfg["reg_max"]) != 16:
                raise ValueError("SPR placement screening must keep reg_max=16")
            if head_mode != "standard":
                raise ValueError("SPR placement screening must keep the standard Detect head")
        if neck_mode != "standard":
            raise ValueError("Placement study must keep the standard neck")

    elif study == "head":
        if assigner_mode != "standard":
            raise ValueError("Head study must keep standard TAL")
        if placements != ["p4_p5"]:
            raise ValueError("Head study is locked to the confirmed S1 SPR placement P4->P5")
        if cfg["loss_mode"] != "standard" or cfg["attention"] != "none":
            raise ValueError("Head study must keep standard loss and no attention")
        if neck_mode != "standard":
            raise ValueError("Head study must keep the standard neck")
        if head_mode == "stride_reg":
            bins = normalize_head_bins(cfg)
            if len(bins) != 3:
                raise ValueError("stride_reg head requires exactly three level bin counts")
            if any(x not in {1, 2, 4, 8, 16} for x in bins):
                raise ValueError("head_reg_bins values must be one of 1,2,4,8,16")
            if bins != sorted(bins, reverse=True):
                raise ValueError("head_reg_bins must be non-increasing from P2 to P4")
        elif int(cfg["reg_max"]) != 1:
            raise ValueError(
                "Standard-head variant in the head study is reserved for the DFL-free reg_max=1 control"
            )

    elif study == "neck":
        if assigner_mode != "standard":
            raise ValueError("Neck study must keep standard TAL")
        if placements != ["p4_p5"]:
            raise ValueError("Neck study is locked to the confirmed S1 SPR placement P4->P5")
        if cfg["loss_mode"] != "standard" or cfg["attention"] != "none":
            raise ValueError("Neck study must keep standard loss and no attention")
        if head_mode != "standard" or int(cfg["reg_max"]) != 1:
            raise ValueError("Neck study is locked to H1 direct reg_max=1 head")
        if neck_mode not in NECK_MODES:
            raise ValueError(f"Unknown neck_mode={neck_mode!r}")
        if neck_mode == "realloc":
            normalize_neck_channels(cfg)

    elif study == "optimization":
        # C3 is evaluated only on frozen C1+C2. No architecture/head/augmentation
        # movement is allowed during this isolated assignment experiment.
        if placements != ["p4_p5"]:
            raise ValueError("C3 optimization is locked to S1 SPR P4->P5")
        if cfg["loss_mode"] != "standard" or cfg["attention"] != "none":
            raise ValueError("C3 optimization must keep standard loss and no attention")
        if head_mode != "standard" or int(cfg["reg_max"]) != 1:
            raise ValueError("C3 optimization is locked to the DFL-free reg_max=1 head")
        if neck_mode != "realloc" or normalize_neck_channels(cfg) != N2B_REALLOC_NOMINAL:
            raise ValueError("C3 optimization is locked to frozen N2b 160/256/416 neck")
        if assigner_mode != "tiny_center_rank":
            raise ValueError("C3-v2 requires assigner_mode=tiny_center_rank")
        _validate_tiny_center_rank(cfg)
    else:
        raise ValueError(f"Unknown study={study!r}")

    t = cfg["train"]
    if int(t["batch"]) != int(t["nbs"]):
        print(
            "INFO: batch != nbs. Ultralytics will use gradient accumulation "
            "so the nominal batch remains nbs."
        )


def experiment_tag(cfg: dict[str, Any]) -> str:
    loss_tag = "standard" if cfg["loss_mode"] == "standard" else "hybrid"
    placements = normalize_spr_placements(cfg)
    if placements:
        short = {"p2_p3": "p2p3", "p3_p4": "p3p4", "p4_p5": "p4p5"}
        arch_tag = "spr-" + "-".join(short[p] for p in placements)
    else:
        arch_tag = cfg["backbone_down"]

    study = cfg.get("study", "placement")
    if study == "head":
        if cfg.get("head_mode") == "stride_reg":
            bins = "-".join(str(x) for x in normalize_head_bins(cfg))
            head_tag = f"snr-{bins}"
        else:
            head_tag = "direct-r1"
        return (
            f"{arch_tag}_{head_tag}_{loss_tag}_attn-{cfg['attention']}_"
            f"{int(cfg['train']['epochs'])}e_seed{int(cfg['seed'])}"
        )

    if study in {"neck", "optimization"}:
        mode = cfg.get("neck_mode")
        if mode == "rep":
            neck_tag = "repneck"
        elif mode == "realloc":
            c = normalize_neck_channels(cfg)
            neck_tag = f"realloc-{c['p2']}-{c['p3']}-{c['p4']}"
        else:
            neck_tag = "stdneck"
        opt_tag = "_tcsr" if study == "optimization" else ""
        return (
            f"{arch_tag}_{neck_tag}_direct-r1{opt_tag}_{loss_tag}_attn-{cfg['attention']}_"
            f"{int(cfg['train']['epochs'])}e_seed{int(cfg['seed'])}"
        )

    return (
        f"{arch_tag}_reg{int(cfg['reg_max'])}_{loss_tag}_"
        f"attn-{cfg['attention']}_{int(cfg['train']['epochs'])}e_"
        f"seed{int(cfg['seed'])}"
    )
