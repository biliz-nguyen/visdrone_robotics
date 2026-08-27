from __future__ import annotations

from pathlib import Path
import json
import os
import random
import sys

import numpy as np
import torch

from .config import load_config
from .data import build_data_yaml
from .model_builder import build_model_yaml
from .ultralytics_patch import patch_ultralytics


def prepare_runtime(
    experiment_path=None,
    local_path=None,
):
    cfg = load_config(
        experiment_path=experiment_path
        if experiment_path is not None
        else Path(__file__).resolve().parents[1] / "config" / "experiment.yaml",
        local_path=local_path
        if local_path is not None
        else Path(__file__).resolve().parents[1] / "config" / "local.yaml",
    )

    for key in ["generated_dir", "runs_dir", "state_dir", "outputs_dir"]:
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)

    os.environ["PYTHONHASHSEED"] = str(cfg["seed"])
    os.environ["ULTRALYTICS_DISABLE_RAY"] = "1"
    os.environ["WANDB_DISABLED"] = "true"

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    c3_assigner_mode = cfg.get("c3_assigner_mode", "standard")
    if c3_assigner_mode == "tiny_quality":
        from .ultralytics_patch_taq import patch_ultralytics_taq

        patch_ultralytics_taq(cfg)
    elif c3_assigner_mode == "standard":
        patch_ultralytics(cfg)
    else:
        raise ValueError(f"Unsupported c3_assigner_mode={c3_assigner_mode!r}")

    data_yaml = build_data_yaml(cfg)
    model_yaml = build_model_yaml(cfg)

    # Guarantee patched local source takes precedence.
    repo = str(Path(cfg["ultra_repo"]).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    return cfg, data_yaml, model_yaml


def state_path(cfg: dict) -> Path:
    return Path(cfg["state_dir"]) / f"{cfg['experiment_tag']}.json"


def save_state(cfg: dict, payload: dict) -> Path:
    path = state_path(cfg)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    latest = Path(cfg["state_dir"]) / "latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_state(cfg: dict) -> dict:
    path = state_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved state for current experiment: {path}\n"
            "Train first or provide --weights."
        )
    return json.loads(path.read_text(encoding="utf-8"))
