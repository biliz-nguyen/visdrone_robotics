#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPERIMENT = ROOT / "config" / "experiment.yaml"
LOCAL = ROOT / "config" / "local.yaml"
CONTROL_SUMMARY = ROOT / "reports" / "yoloedge27" / "stage3" / "head_snr_v1_5e" / "summary.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shift-lambda", type=float, default=0.25)
    p.add_argument("--tiny-threshold", type=float, default=16.0)
    p.add_argument("--max-shift-px", type=int, default=1)
    return p.parse_args()


def run_live(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def newest(root: Path, pattern: str) -> Path:
    xs = list(root.glob(pattern))
    if not xs:
        raise FileNotFoundError(f"No file matching {pattern} under {root}")
    return max(xs, key=lambda p: p.stat().st_mtime)


def parse_epoch_row(path: Path, epoch_number: int) -> dict[str, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    row = rows[epoch_number - 1]

    def metric(*names: str) -> float:
        for name in names:
            value = row.get(name)
            if value not in (None, ""):
                return float(value)
        raise KeyError(names)

    return {
        "precision": metric("metrics/precision(B)", "metrics/precision"),
        "recall": metric("metrics/recall(B)", "metrics/recall"),
        "map50": metric("metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": metric("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def metric_delta(a: dict, b: dict) -> dict[str, float]:
    return {
        k: float(a[k]) - float(b[k])
        for k in ("precision", "recall", "map50", "map50_95")
    }


def focus_delta(a: dict, b: dict) -> dict[str, dict[str, float]]:
    out = {}
    for cls in ("pedestrian", "people"):
        if cls in a and cls in b:
            out[cls] = metric_delta(a[cls], b[cls])
    return out


def patch_model_yaml(path: Path, args) -> None:
    text = path.read_text(encoding="utf-8")
    old = "  - [[19, 22, 25], 1, Detect, [nc]]"
    new = (
        "  - [[19, 22, 25], 1, TinyShiftConsistencyDetect, "
        f"[nc, {args.shift_lambda}, {args.tiny_threshold}, {args.max_shift_px}]]"
    )
    if old not in text:
        raise RuntimeError("Could not find direct-r1 Detect line in generated model YAML")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def assert_shift_student(student) -> None:
    detect = student.model.model[-1]
    if detect.__class__.__name__ != "TinyShiftConsistencyDetect":
        raise RuntimeError(f"Shift-consistency head replaced: {detect.__class__.__name__}")
    assert int(detect.reg_max) == 1
    assert [int(x) for x in detect.stride.tolist()] == [4, 8, 16]
    criterion = student.model.init_criterion()
    if criterion.__class__.__name__ != "TinyShiftConsistencyLoss":
        raise RuntimeError(f"Shift-consistency criterion replaced: {criterion.__class__.__name__}")
    del criterion


def model_complexity(model, imgsz: int) -> dict[str, float | int | None]:
    import torch

    params = sum(p.numel() for p in model.model.parameters())
    gflops = None
    try:
        from thop import profile

        model.model.cpu().eval()
        x = torch.randn(1, 3, imgsz, imgsz)
        with torch.no_grad():
            macs, _ = profile(model.model, inputs=(x,), verbose=False)
        gflops = float(macs * 2 / 1e9)
        del x
    except Exception as exc:
        print("THOP skipped:", exc)
    return {"params": int(params), "gflops": gflops}


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not CONTROL_SUMMARY.exists():
        raise FileNotFoundError(CONTROL_SUMMARY)
    prior = json.loads(CONTROL_SUMMARY.read_text(encoding="utf-8"))
    controls = {k: prior["variants"][k] for k in ("H0", "H1")}

    shutil.rmtree(screen_root, ignore_errors=True)
    for p in (screen_root / "runs", screen_root / "state", screen_root / "outputs", screen_root / "generated"):
        p.mkdir(parents=True, exist_ok=True)

    original_text = EXPERIMENT.read_text(encoding="utf-8")
    original = yaml.safe_load(original_text)
    cfg = json.loads(json.dumps(original))
    cfg["preset"] = "edge27_head_direct_r1"
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["batch"] = int(args.batch)
    cfg["train"]["nbs"] = int(args.nbs)
    cfg["train"]["workers"] = int(args.workers)
    cfg["train"]["save_period"] = 1
    cfg["train"]["patience"] = 0
    cfg["pretrained"] = False

    LOCAL.write_text(
        "\n".join(
            [
                f'dataset_root: "{dataset}"',
                'dataset_format: "visdrone_official"',
                'train_images: "VisDrone2019-DET-train/images"',
                'train_annotations: "VisDrone2019-DET-train/annotations"',
                'val_images: "VisDrone2019-DET-val/images"',
                'val_annotations: "VisDrone2019-DET-val/annotations"',
                'test_images: "VisDrone2019-DET-test-dev/images"',
                'test_annotations: "VisDrone2019-DET-test-dev/annotations"',
                'test_image: ""',
                f'runs_dir: "{screen_root / "runs"}"',
                f'state_dir: "{screen_root / "state"}"',
                f'outputs_dir: "{screen_root / "outputs"}"',
                f'generated_dir: "{screen_root / "generated"}"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'third_party' / 'ultralytics'}:{ROOT}"

    try:
        EXPERIMENT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        from src.runtime import prepare_runtime

        resolved, data_yaml, generated_model_yaml = prepare_runtime()
        patch_model_yaml(generated_model_yaml, args)
        shift_model_yaml = report_dir / "model_shiftcons_v1.yaml"
        shutil.copy2(generated_model_yaml, shift_model_yaml)

        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the shift-consistency screen")

        student = YOLO(str(shift_model_yaml))
        assert_shift_student(student)
        complexity = model_complexity(student, int(resolved["train"]["imgsz"]))
        h1_complexity = controls["H1"]["complexity"]
        if int(complexity["params"]) != int(h1_complexity["params"]):
            raise RuntimeError(f"Shift student params differ from H1: {complexity} vs {h1_complexity}")
        if complexity["gflops"] is not None and abs(float(complexity["gflops"]) - float(h1_complexity["gflops"])) > 1e-3:
            raise RuntimeError(f"Shift student GFLOPs differ from H1: {complexity} vs {h1_complexity}")

        del student
        gc.collect()
        torch.cuda.empty_cache()

        t = resolved["train"]
        run_name = (
            f"spr-p4p5_shiftcons-l{args.shift_lambda:g}-tiny{args.tiny_threshold:g}-"
            f"px{args.max_shift_px}_reg1_{args.epochs}e_seed{resolved['seed']}"
        )

        student = YOLO(str(shift_model_yaml))
        assert_shift_student(student)
        print("Shift-consistency pre-train head:", student.model.model[-1].__class__.__name__)
        print("Shift-consistency pre-train YAML:", shift_model_yaml)

        student.train(
            data=str(data_yaml),
            imgsz=int(t["imgsz"]),
            epochs=int(args.epochs),
            batch=int(args.batch),
            workers=int(args.workers),
            nbs=int(args.nbs),
            device=0,
            amp=bool(t["amp"]),
            pretrained=False,
            optimizer=t["optimizer"],
            lr0=float(t["lr0"]),
            lrf=float(t["lrf"]),
            momentum=float(t["momentum"]),
            weight_decay=float(t["weight_decay"]),
            cos_lr=bool(t["cos_lr"]),
            warmup_epochs=float(t["warmup_epochs"]),
            seed=int(resolved["seed"]),
            deterministic=bool(t["deterministic"]),
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
            close_mosaic=int(t["close_mosaic"]),
            mixup=float(t["mixup"]),
            copy_paste=float(t["copy_paste"]),
            cutmix=float(t["cutmix"]),
            box=float(t["box"]),
            cls=float(t["cls"]),
            dfl=float(t["dfl"]),
            max_det=int(t["max_det"]),
            val=True,
            project=str(screen_root / "runs"),
            name=run_name,
            exist_ok=False,
            save=True,
            save_period=1,
            plots=True,
            patience=0,
            verbose=True,
        )

        best = newest(screen_root, "runs/*/weights/best.pt")
        results = newest(screen_root, "runs/*/results.csv")
        train_args = newest(screen_root, "runs/*/args.yaml")

        eval_path = report_dir / "shiftcons_v1_5e_eval.json"
        run_live([sys.executable, "scripts/eval_screening.py", "--weights", str(best), "--output", str(eval_path)], env)

        onnx_ok = True
        onnx_error = None
        try:
            run_live([sys.executable, "scripts/export.py", "--weights", str(best), "--format", "onnx"], env)
        except Exception as exc:
            onnx_ok = False
            onnx_error = f"{type(exc).__name__}: {exc}"

        shutil.copy2(results, report_dir / "results.csv")
        shutil.copy2(train_args, report_dir / "args.yaml")
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))

        s1 = {
            "id": "S1",
            "description": "H1 DFL-free student + Tiny Shift Equivariance Consistency",
            "status": "complete",
            "complexity": complexity,
            "epoch5": parse_epoch_row(results, int(args.epochs)),
            "best_eval": eval_data["aggregate"],
            "focus_best_eval": eval_data.get("focus", {}),
            "speed_ms_best_eval": eval_data.get("speed_ms", {}),
            "onnx_export": {"ok": onnx_ok, "error": onnx_error},
            "local_best_pt": str(best),
        }
        for cid, control in controls.items():
            s1[f"delta_best_eval_vs_{cid.lower()}"] = metric_delta(s1["best_eval"], control["best_eval"])
            s1[f"delta_focus_vs_{cid.lower()}"] = focus_delta(s1["focus_best_eval"], control.get("focus_best_eval", {}))

        summary = {
            "purpose": "Tiny shift-equivariance local 5e mechanism screen; not final paper evidence.",
            "novelty_status": "working hypothesis only; translation equivariance and anti-aliasing are established topics, exact tiny object-level paired-box regularizer not claimed novel",
            "student": "H1: S1 SPR P4->P5 + stock-width DFL-free Detect reg_max=1",
            "mechanism": {
                "name": "Tiny Shift Equivariance Consistency (TSEC) v1",
                "shift_lambda": float(args.shift_lambda),
                "tiny_threshold_px": float(args.tiny_threshold),
                "shift_px": int(args.max_shift_px),
                "shift_set": ["(+1,0)", "(-1,0)", "(0,+1)", "(0,-1)"],
                "pairing": "same GT object across base and one-pixel shifted views; one highest-TAL-score representative per object per view",
                "reliability": "detached geometric mean of base/shift IoU-to-GT",
                "consistency": "CIoU(base box, inverse-shifted auxiliary-view box)",
                "base_assignment": "standard TAL unchanged",
                "base_losses": "standard BCE + CIoU + direct-regression unchanged",
                "training_cost": "one auxiliary forward per training batch; BN running statistics frozen for auxiliary pass",
                "inference_change": "none",
                "extra_prediction_channels": 0,
            },
            "protocol": {
                "epochs": int(args.epochs),
                "batch": int(args.batch),
                "nbs": int(args.nbs),
                "imgsz": int(t["imgsz"]),
                "seed": int(resolved["seed"]),
                "pretrained_student": False,
            },
            "controls": controls,
            "S1": s1,
            "decision_rule": "Keep only if S1 improves H1 aggregate and does not materially damage pedestrian/people; otherwise reject without parameter fishing.",
            "caution": "single local seed and 5 epochs; auxiliary-view training cost is higher even though deployment graph is identical to H1.",
        }
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (report_dir / "paths.txt").write_text(
            f"best_pt={best}\nresults_csv={results}\ncontrol_summary={CONTROL_SUMMARY}\nshift_model_yaml={shift_model_yaml}\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
    finally:
        EXPERIMENT.write_text(original_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
