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


def _install_c11_aux_schedule(model, cfg: dict) -> None:
    mode = cfg.get("c11_aux_schedule", "constant")
    if mode == "constant":
        return
    if mode != "cosine_to_zero":
        raise ValueError(f"Unsupported c11_aux_schedule={mode!r}")

    from src.aux_schedule import cosine_to_zero_weight

    base_weight = float(cfg.get("c11_aux_base_weight", 0.10))
    expected_focus = tuple(int(x) for x in cfg.get("c9_focus_classes", [5]))

    def on_train_epoch_start(trainer):
        criterion = getattr(trainer.model, "criterion", None)
        if criterion is None:
            criterion = trainer.model.init_criterion()
            trainer.model.criterion = criterion

        if criterion.__class__.__name__ != "P2QualityAuxDetectionLoss":
            raise RuntimeError(
                "C11 cosine schedule requires P2QualityAuxDetectionLoss, got "
                f"{criterion.__class__.__name__}"
            )
        if tuple(criterion.focus_classes) != expected_focus:
            raise RuntimeError(
                f"C11 focus mismatch: criterion={criterion.focus_classes}, expected={expected_focus}"
            )

        weight = cosine_to_zero_weight(base_weight, int(trainer.epoch), int(trainer.epochs))
        criterion.aux_weight = float(weight)
        criterion.c11_epoch = int(trainer.epoch)
        criterion.c11_epochs = int(trainer.epochs)
        criterion.c11_base_weight = base_weight
        print(
            f"C11_AUX_WEIGHT epoch={int(trainer.epoch) + 1}/{int(trainer.epochs)} "
            f"weight={weight:.8f}"
        )

    model.add_callback("on_train_epoch_start", on_train_epoch_start)


def _install_c12_scale_velocity(model, cfg: dict) -> None:
    mode = cfg.get("c12_scale_velocity_mode", "standard")
    if mode == "standard":
        return
    if mode != "tslve_cls":
        raise ValueError(f"Unsupported c12_scale_velocity_mode={mode!r}")
    if cfg.get("c11_aux_schedule", "constant") != "constant":
        raise ValueError("C12 cannot run together with C11 auxiliary scheduling")

    dynamics_path = Path(cfg["state_dir"]) / "c12_scale_dynamics.jsonl"
    dynamics_path.parent.mkdir(parents=True, exist_ok=True)
    if dynamics_path.exists():
        dynamics_path.unlink()

    def get_criterion(trainer):
        criterion = getattr(trainer.model, "criterion", None)
        if criterion is None:
            criterion = trainer.model.init_criterion()
            trainer.model.criterion = criterion
        if criterion.__class__.__name__ != "TemporalScaleVelocityDetectionLoss":
            raise RuntimeError(
                "C12 TSLVE requires TemporalScaleVelocityDetectionLoss, got "
                f"{criterion.__class__.__name__}"
            )
        return criterion

    def on_train_epoch_start(trainer):
        criterion = get_criterion(trainer)
        criterion.set_epoch(int(trainer.epoch))
        print(
            f"C12_TSLVE_EPOCH_START epoch={int(trainer.epoch) + 1}/{int(trainer.epochs)} "
            f"mode={'calibration' if int(trainer.epoch) == 0 else 'adaptive'}"
        )

    def on_train_epoch_end(trainer):
        criterion = get_criterion(trainer)
        summary = criterion.epoch_summary()
        summary["epoch"] = int(trainer.epoch) + 1
        summary["epochs"] = int(trainer.epochs)
        with dynamics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
        print("C12_TSLVE_EPOCH_SUMMARY", json.dumps(summary, sort_keys=True))

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)


def main():
    cfg, data_yaml, model_yaml = prepare_runtime()

    import torch
    from ultralytics import YOLO

    t = cfg["train"]
    run_name = unique_run_name(cfg)

    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for VisDrone training on this runner; "
            "refusing to fall back to CPU."
        )

    if torch.cuda.device_count() >= 2:
        device = [0, 1]
    else:
        device = 0

    print("GPU:", torch.cuda.get_device_name(0))

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
    print("C11 aux schedule:", cfg.get("c11_aux_schedule", "constant"))
    print("C12 scale velocity:", cfg.get("c12_scale_velocity_mode", "standard"))
    print("=" * 90)

    model = YOLO(
        str(model_yaml)
    )

    # IMPORTANT:
    # no model.load("yolo11n.pt")
    _install_c11_aux_schedule(model, cfg)
    _install_c12_scale_velocity(model, cfg)

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
        "c11_aux_schedule":
            cfg.get("c11_aux_schedule", "constant"),
        "c12_scale_velocity_mode":
            cfg.get("c12_scale_velocity_mode", "standard"),
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
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
