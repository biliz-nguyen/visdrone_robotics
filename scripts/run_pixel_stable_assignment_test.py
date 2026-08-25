#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
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
    p.add_argument("--tiny-threshold", type=float, default=16.0)
    p.add_argument("--perturb-px", type=float, default=1.0)
    return p.parse_args()


def run_capture(cmd: list[str], env: dict[str, str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc.stdout


def run_live(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def newest(root: Path, pattern: str) -> Path:
    xs = list(root.glob(pattern))
    if not xs:
        raise FileNotFoundError(f"No file matching {pattern} under {root}")
    return max(xs, key=lambda p: p.stat().st_mtime)


def parse_epoch_row(path: Path, epoch_number: int) -> dict[str, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) < epoch_number:
        raise RuntimeError(f"Need epoch {epoch_number} in {path}, got {len(rows)} rows")
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


def parse_complexity(text: str) -> dict[str, float | int | None]:
    pm = re.search(r"Params:\s*([0-9,]+)", text)
    gm = re.search(r"GFLOPs:\s*([0-9.]+)", text)
    return {
        "params": int(pm.group(1).replace(",", "")) if pm else None,
        "gflops": float(gm.group(1)) if gm else None,
    }


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(screen_root, ignore_errors=True)
    for p in (screen_root / "runs", screen_root / "state", screen_root / "outputs", screen_root / "generated"):
        p.mkdir(parents=True, exist_ok=True)

    if not CONTROL_SUMMARY.exists():
        raise FileNotFoundError(f"Existing H0/H1 controls not found: {CONTROL_SUMMARY}")
    prior = json.loads(CONTROL_SUMMARY.read_text(encoding="utf-8"))
    controls = {k: prior["variants"][k] for k in ("H0", "H1")}

    original_text = EXPERIMENT.read_text(encoding="utf-8")
    cfg = yaml.safe_load(original_text)
    cfg["preset"] = "edge27_head_direct_r1"
    cfg["assigner_mode"] = "pixel_stable"
    cfg["pixel_stable_assigner"] = {
        "tiny_min_side": float(args.tiny_threshold),
        "perturb_px": float(args.perturb_px),
    }
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

        sanity = run_capture([sys.executable, "scripts/sanity.py"], env)
        (report_dir / "sanity.txt").write_text(sanity, encoding="utf-8")
        complexity = parse_complexity(sanity)
        h1_complexity = controls["H1"]["complexity"]
        if complexity.get("params") != h1_complexity.get("params"):
            raise RuntimeError(f"Pixel-stable assigner changed params: {complexity} vs {h1_complexity}")
        if abs(float(complexity.get("gflops")) - float(h1_complexity.get("gflops"))) > 1e-4:
            raise RuntimeError(f"Pixel-stable assigner changed GFLOPs: {complexity} vs {h1_complexity}")

        # Strong preflight: instantiate the actual patched criterion and verify
        # that training will not silently fall back to stock TAL.
        probe_code = r'''
from src.runtime import prepare_runtime
cfg, data_yaml, model_yaml = prepare_runtime()
from ultralytics import YOLO
m = YOLO(str(model_yaml))
criterion = m.model.init_criterion()
name = criterion.assigner.__class__.__name__
print("Criterion assigner:", name)
print("tiny_min_side:", getattr(criterion.assigner, "tiny_min_side", None))
print("perturb_px:", getattr(criterion.assigner, "perturb_px", None))
assert name == "TinyPixelStableAssigner"
assert float(criterion.assigner.tiny_min_side) == float(cfg["pixel_stable_assigner"]["tiny_min_side"])
assert float(criterion.assigner.perturb_px) == float(cfg["pixel_stable_assigner"]["perturb_px"])
'''
        preflight = run_capture([sys.executable, "-c", probe_code], env)
        (report_dir / "assigner_preflight.txt").write_text(preflight, encoding="utf-8")

        run_live([sys.executable, "scripts/train.py"], env)
        best = newest(screen_root, "runs/*/weights/best.pt")
        results = newest(screen_root, "runs/*/results.csv")
        train_args = newest(screen_root, "runs/*/args.yaml")

        eval_path = report_dir / "pixel_stable_v1_5e_eval.json"
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

        p1 = {
            "id": "P1",
            "description": "H1 DFL-free student + Pixel-Stable Tiny Assignment",
            "status": "complete",
            "complexity": complexity,
            "epoch5": parse_epoch_row(results, int(args.epochs)),
            "best_eval": eval_data["aggregate"],
            "focus_best_eval": eval_data.get("focus", {}),
            "speed_ms_best_eval": eval_data.get("speed_ms", {}),
            "onnx_export": {"ok": onnx_ok, "error": onnx_error},
            "local_best_pt": str(best),
        }
        for control_id, control in controls.items():
            p1[f"delta_best_eval_vs_{control_id.lower()}"] = {
                key: float(p1["best_eval"][key]) - float(control["best_eval"][key])
                for key in ("precision", "recall", "map50", "map50_95")
            }
            p1[f"delta_focus_vs_{control_id.lower()}"] = {
                cls: {
                    key: float(p1["focus_best_eval"][cls][key]) - float(control["focus_best_eval"][cls][key])
                    for key in ("precision", "recall", "map50", "map50_95")
                }
                for cls in ("pedestrian", "people")
                if cls in p1["focus_best_eval"] and cls in control["focus_best_eval"]
            }

        summary = {
            "purpose": "Pixel-stable tiny assignment local 5e mechanism screen; not final paper evidence.",
            "novelty_status": "working hypothesis only; no novelty claim",
            "mechanism": {
                "name": "Pixel-Stable Tiny Assignment (PSTA) v1",
                "base": "H1: S1 SPR P4->P5 + stock-width direct reg_max=1",
                "candidate_region": "standard TAL unchanged",
                "topk_and_conflict_resolution": "standard TAL unchanged",
                "non_tiny_quality": "standard IoU exactly",
                "tiny_quality": "sqrt(nominal_IoU * mean(IoU under GT shifts left/right/up/down by 1 px))",
                "classification_term": "standard TAL cls_score^alpha unchanged",
                "beta": 6.0,
                "tiny_threshold_px": float(args.tiny_threshold),
                "perturb_px": float(args.perturb_px),
                "losses": "standard BCE + CIoU + direct-regression unchanged",
                "inference_change": "none",
                "extra_prediction_channels": 0,
                "training_cost": "four extra IoU evaluations only inside assignment; no auxiliary network forward",
            },
            "protocol": {
                "epochs": int(args.epochs),
                "batch": int(args.batch),
                "nbs": int(args.nbs),
                "imgsz": int(cfg["train"]["imgsz"]),
                "seed": int(cfg["seed"]),
                "pretrained_student": False,
            },
            "controls": controls,
            "P1": p1,
            "decision_rule": "Promote only if P1 improves H1 aggregate and does not materially damage pedestrian/people at exactly H1 inference complexity.",
            "caution": "single local seed, deterministic=false, 5 epochs only; final evidence requires longer controlled runs if promoted.",
        }
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (report_dir / "paths.txt").write_text(
            f"best_pt={best}\nresults_csv={results}\ncontrol_summary={CONTROL_SUMMARY}\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
    finally:
        EXPERIMENT.write_text(original_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
