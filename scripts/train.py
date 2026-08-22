#!/usr/bin/env python3

from pathlib import Path
import gc
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import (
    prepare_runtime,
    save_state,
)


def unique_run_name(cfg: dict) -> str:
    runs_dir = Path(cfg["runs_dir"])
    runs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = cfg["experiment_tag"]
    name = base
    i = 2

    while (runs_dir / name).exists():
        name = f"{base}_r{i}"
        i += 1

    return name


def main():
    cfg, data_yaml, model_yaml = prepare_runtime()

    import torch
    from ultralytics import YOLO

    t = cfg["train"]
    run_name = unique_run_name(cfg)

    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
    ):
        device = [0, 1]
    elif torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    print("=" * 90)
    print("START TRAIN")
    print("=" * 90)
    print("Experiment:", cfg["experiment_tag"])
    print("Run:", run_name)
    print("reg_max:", cfg["reg_max"])
    print("Attention:", cfg["attention"])
    print("Loss:", cfg["loss_mode"])
    print("Pretrained: False")
    print("Epochs:", t["epochs"])
    print(
        "Batch/NBS:",
        t["batch"],
        "/",
        t["nbs"],
    )
    print("Device:", device)
    print("=" * 90)

    model = YOLO(
        str(model_yaml)
    )

    # IMPORTANT:
    # no model.load("yolo11n.pt")

    model.train(
        data=str(data_yaml),
        imgsz=int(t["imgsz"]),
        epochs=int(t["epochs"]),
        batch=int(t["batch"]),
        workers=int(t["workers"]),
        nbs=int(t["nbs"]),
        device=device,
        amp=bool(t["amp"]),
        pretrained=False,

        optimizer=t["optimizer"],
        lr0=float(t["lr0"]),
        lrf=float(t["lrf"]),
        momentum=float(t["momentum"]),
        weight_decay=float(t["weight_decay"]),
        cos_lr=bool(t["cos_lr"]),
        warmup_epochs=float(
            t["warmup_epochs"]
        ),

        seed=int(cfg["seed"]),
        deterministic=bool(
            t["deterministic"]
        ),

        hsv_h=float(t["hsv_h"]),
        hsv_s=float(t["hsv_s"]),
        hsv_v=float(t["hsv_v"]),
        degrees=float(t["degrees"]),
        translate=float(t["translate"]),
        scale=float(t["scale"]),
        shear=float(t["shear"]),
        perspective=float(t["perspective"]),
        flipud=float(t["flipud"]),
        fliplr=float(t["fliplr"]),
        mosaic=float(t["mosaic"]),
        close_mosaic=int(
            t["close_mosaic"]
        ),
        mixup=float(t["mixup"]),
        copy_paste=float(
            t["copy_paste"]
        ),
        cutmix=float(t["cutmix"]),

        box=float(t["box"]),
        cls=float(t["cls"]),
        dfl=float(t["dfl"]),

        max_det=int(t["max_det"]),
        val=True,

        project=cfg["runs_dir"],
        name=run_name,
        exist_ok=False,

        save=True,
        save_period=int(
            t["save_period"]
        ),
        plots=True,
        patience=int(t["patience"]),
        verbose=True,
    )

    save_dir = Path(
        model.trainer.save_dir
    )

    best_pt = (
        save_dir
        / "weights"
        / "best.pt"
    )

    last_pt = (
        save_dir
        / "weights"
        / "last.pt"
    )

    assert best_pt.exists(), best_pt
    assert last_pt.exists(), last_pt

    payload = {
        "experiment_tag":
            cfg["experiment_tag"],
        "preset":
            cfg["preset"],
        "run_name":
            run_name,
        "save_dir":
            str(save_dir),
        "best_pt":
            str(best_pt),
        "last_pt":
            str(last_pt),
        "reg_max":
            int(cfg["reg_max"]),
        "attention":
            cfg["attention"],
        "loss_mode":
            cfg["loss_mode"],
    }

    state = save_state(
        cfg,
        payload,
    )

    print()
    print("✅ BEST:", best_pt)
    print("✅ LAST:", last_pt)
    print("✅ STATE:", state)

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
