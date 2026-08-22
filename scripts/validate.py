#!/usr/bin/env python3

from pathlib import Path
import argparse
import gc
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import (
    prepare_runtime,
    load_state,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        default="",
        help="Optional explicit best.pt",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cfg, data_yaml, _ = prepare_runtime()
    t = cfg["train"]

    if args.weights:
        weights = Path(
            args.weights
        ).expanduser().resolve()
    else:
        state = load_state(cfg)
        weights = Path(
            state["best_pt"]
        )

    if not weights.exists():
        raise FileNotFoundError(weights)

    import torch
    from ultralytics import YOLO

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(str(weights))

    detect = model.model.model[-1]

    assert int(detect.reg_max) == int(
        cfg["reg_max"]
    ), (
        f"Checkpoint reg_max={detect.reg_max}, "
        f"config reg_max={cfg['reg_max']}"
    )

    device = 0 if torch.cuda.is_available() else "cpu"

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=int(t["imgsz"]),
        batch=int(t["batch"]),
        device=device,
        workers=int(t["workers"]),
        conf=0.001,
        iou=0.70,
        max_det=int(t["max_det"]),
        save_json=True,
        plots=True,
        project=cfg["runs_dir"],
        name=(
            "test_"
            + cfg["experiment_tag"]
        ),
        exist_ok=True,
        verbose=True,
    )

    summary = {
        "experiment_tag":
            cfg["experiment_tag"],
        "weights":
            str(weights),
        "reg_max":
            int(detect.reg_max),
        "attention":
            cfg["attention"],
        "loss_mode":
            cfg["loss_mode"],
        "precision":
            float(metrics.box.mp),
        "recall":
            float(metrics.box.mr),
        "map50":
            float(metrics.box.map50),
        "map50_95":
            float(metrics.box.map),
        "inference_ms":
            getattr(
                metrics,
                "speed",
                {},
            ).get(
                "inference",
                None,
            ),
    }

    out = (
        Path(cfg["outputs_dir"])
        / (
            "summary_"
            + cfg["experiment_tag"]
            + ".json"
        )
    )

    out.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("TEST SUMMARY")
    print("=" * 90)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("Saved:", out)
    print("=" * 90)


if __name__ == "__main__":
    main()
