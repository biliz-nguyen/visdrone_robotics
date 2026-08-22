#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES
from src.runtime import prepare_runtime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(weights)

    cfg, data_yaml, _ = prepare_runtime()
    t = cfg["train"]

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for screening evaluation")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=int(t["imgsz"]),
        batch=int(t["batch"]),
        device=0,
        workers=int(t["workers"]),
        conf=0.001,
        iou=0.70,
        max_det=int(t["max_det"]),
        plots=False,
        save_json=False,
        verbose=True,
    )

    box = metrics.box
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "map50": float(box.ap50[i]),
            "map50_95": float(box.maps[i]),
        }

    payload = {
        "weights": str(weights),
        "preset": cfg["preset"],
        "experiment_tag": cfg["experiment_tag"],
        "aggregate": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": per_class,
        "focus": {
            "pedestrian": per_class["pedestrian"],
            "people": per_class["people"],
        },
        "speed_ms": getattr(metrics, "speed", {}),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
