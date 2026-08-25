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
    p.add_argument("--mentor", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mentor-lambda", type=float, default=0.25)
    p.add_argument("--tiny-threshold", type=float, default=16.0)
    p.add_argument("--advantage-margin", type=float, default=0.05)
    p.add_argument("--min-teacher-iou", type=float, default=0.10)
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
        "  - [[19, 22, 25], 1, AdvantageGatedMentorDetect, "
        f"[nc, {args.mentor_lambda}, {args.tiny_threshold}, "
        f"{args.advantage_margin}, {args.min_teacher_iou}]]"
    )
    if old not in text:
        raise RuntimeError("Could not find direct-r1 Detect line in generated model YAML")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


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
    mentor = Path(args.mentor).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not mentor.is_file():
        raise FileNotFoundError(f"Mentor checkpoint not found: {mentor}")
    if not CONTROL_SUMMARY.exists():
        raise FileNotFoundError(CONTROL_SUMMARY)

    prior = json.loads(CONTROL_SUMMARY.read_text(encoding="utf-8"))
    controls = {k: prior["variants"][k] for k in ("H0", "H1")}

    shutil.rmtree(screen_root, ignore_errors=True)
    for p in (
        screen_root / "runs",
        screen_root / "state",
        screen_root / "outputs",
        screen_root / "generated",
    ):
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
    env["YOLOEDGE27_MENTOR_PT"] = str(mentor)

    try:
        EXPERIMENT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        from src.runtime import prepare_runtime

        resolved, data_yaml, model_yaml = prepare_runtime()
        patch_model_yaml(model_yaml, args)
        shutil.copy2(model_yaml, report_dir / "model_agtl_v1.yaml")

        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the AGTL screen")

        student = YOLO(str(model_yaml))
        detect = student.model.model[-1]
        assert detect.__class__.__name__ == "AdvantageGatedMentorDetect"
        assert int(detect.reg_max) == 1
        assert [int(x) for x in detect.stride.tolist()] == [4, 8, 16]

        complexity = model_complexity(student, int(resolved["train"]["imgsz"]))
        h1_complexity = controls["H1"]["complexity"]
        if int(complexity["params"]) != int(h1_complexity["params"]):
            raise RuntimeError(f"AGTL student params differ from H1: {complexity} vs {h1_complexity}")
        if complexity["gflops"] is not None and abs(float(complexity["gflops"]) - float(h1_complexity["gflops"])) > 1e-3:
            raise RuntimeError(f"AGTL student GFLOPs differ from H1: {complexity} vs {h1_complexity}")

        student.model.cuda().eval()
        criterion = student.model.init_criterion()
        assert criterion.__class__.__name__ == "AdvantageGatedTinyLocalizationLoss"
        del criterion
        student.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # Record the actual frozen mentor's screening metrics before training.
        mentor_eval_path = report_dir / "mentor_eval.json"
        run_live(
            [sys.executable, "scripts/eval_screening.py", "--weights", str(mentor), "--output", str(mentor_eval_path)],
            env,
        )

        t = resolved["train"]
        run_name = (
            f"spr-p4p5_agtl-l{args.mentor_lambda:g}-t{args.tiny_threshold:g}-"
            f"a{args.advantage_margin:g}_reg1_{args.epochs}e_seed{resolved['seed']}"
        )

        # Re-create on CUDA after the CPU THOP pass.
        student = YOLO(str(model_yaml))
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

        eval_path = report_dir / "agtl_v1_5e_eval.json"
        run_live(
            [sys.executable, "scripts/eval_screening.py", "--weights", str(best), "--output", str(eval_path)],
            env,
        )

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
        mentor_eval = json.loads(mentor_eval_path.read_text(encoding="utf-8"))

        m1 = {
            "id": "M1",
            "description": "H1 DFL-free student + Advantage-Gated Tiny Localization Transfer",
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
            m1[f"delta_best_eval_vs_{cid.lower()}"] = metric_delta(m1["best_eval"], control["best_eval"])
            m1[f"delta_focus_vs_{cid.lower()}"] = focus_delta(
                m1["focus_best_eval"], control.get("focus_best_eval", {})
            )

        summary = {
            "purpose": "AGTL v1 local 5e mechanism screen; not final paper evidence.",
            "novelty_status": "working contribution hypothesis only; inspired by selective/task-oriented KD literature, no novelty claim yet",
            "student": "H1: S1 SPR P4->P5 + stock-width DFL-free Detect reg_max=1",
            "mentor": {
                "checkpoint": str(mentor),
                "frozen": True,
                "eval": mentor_eval,
            },
            "mechanism": {
                "name": "Advantage-Gated Tiny Localization Transfer (AGTL)",
                "mentor_lambda": float(args.mentor_lambda),
                "tiny_threshold_px": float(args.tiny_threshold),
                "advantage_margin": float(args.advantage_margin),
                "min_teacher_iou": float(args.min_teacher_iou),
                "selection": "student-assigned positive AND GT min-side<threshold AND mentor IoU > student IoU + margin",
                "transfer": "decoded mentor box -> student decoded box via CIoU, weighted by detached mentor advantage",
                "capacity_bridge": "mentor DFL distribution is decoded first; reg1 student never mimics mentor bins/features/logits",
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
            "M1": m1,
            "decision_rule": "Promote only if M1 improves H1 accuracy, especially pedestrian/people, at exactly H1 inference complexity.",
            "caution": "single local seed and 5 epochs; mentor-assisted training is a screen only.",
        }
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (report_dir / "paths.txt").write_text(
            f"mentor_pt={mentor}\nbest_pt={best}\nresults_csv={results}\ncontrol_summary={CONTROL_SUMMARY}\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
    finally:
        EXPERIMENT.write_text(original_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
