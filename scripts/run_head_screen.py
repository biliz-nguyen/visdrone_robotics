#!/usr/bin/env python3
"""Run 5e head screening while reusing the existing S1 control.

H0 (S1 P4->P5, stock Detect, reg_max=16) is read from the already completed
placement screen and is never retrained here. Only H1-H3 are new runs.
"""

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
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "experiment.yaml"
LOCAL = ROOT / "config" / "local.yaml"

VARIANTS = [
    ("H1", "edge27_head_direct_r1", "stock Detect, global reg_max=1 / DFL-free control"),
    ("H2", "edge27_head_snr_16_8_4", "level-specific bins P2/P3/P4 = 16/8/4"),
    ("H3", "edge27_head_hybrid_16_4_1", "aggressive tiny-preserving bins P2/P3/P4 = 16/4/1"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def write_local(dataset: Path, variant_root: Path):
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
                f'runs_dir: "{variant_root / "runs"}"',
                f'state_dir: "{variant_root / "state"}"',
                f'outputs_dir: "{variant_root / "outputs"}"',
                f'generated_dir: "{variant_root / "generated"}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def set_preset(original: dict, preset: str, args):
    c = json.loads(json.dumps(original))
    c["preset"] = preset
    c["train"]["epochs"] = int(args.epochs)
    c["train"]["batch"] = int(args.batch)
    c["train"]["nbs"] = int(args.nbs)
    c["train"]["workers"] = int(args.workers)
    c["train"]["save_period"] = 1
    c["train"]["patience"] = 0
    c["pretrained"] = False
    EXPERIMENT.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")


def run_capture(cmd, env):
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout, end="")
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout)
    return p.stdout


def run_live(cmd, env):
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def newest(root: Path, pattern: str) -> Path:
    xs = list(root.glob(pattern))
    if not xs:
        raise FileNotFoundError(f"No file matching {pattern} under {root}")
    return max(xs, key=lambda p: p.stat().st_mtime)


def parse_epoch_row(path: Path, epoch_number: int):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    row = rows[epoch_number - 1]

    def get(*names):
        for n in names:
            if row.get(n) not in (None, ""):
                return float(row[n])
        raise KeyError(names)

    return {
        "precision": get("metrics/precision(B)", "metrics/precision"),
        "recall": get("metrics/recall(B)", "metrics/recall"),
        "map50": get("metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": get("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def parse_complexity(text: str):
    pm = re.search(r"Params:\s*([0-9,]+)", text)
    gm = re.search(r"GFLOPs:\s*([0-9.]+)", text)
    return {
        "params": int(pm.group(1).replace(",", "")) if pm else None,
        "gflops": float(gm.group(1)) if gm else None,
    }


def load_control():
    placement_summary = ROOT / "reports/yoloedge27/stage1/spr_placement_v1_5e/summary.json"
    control_results = ROOT / "reports/yoloedge27/stage1/spr_placement_v1_5e/s1/results.csv"
    control_eval = ROOT / "reports/yoloedge27/stage1/spr_placement_v1_5e/s1/best_eval.json"
    if not (placement_summary.exists() and control_results.exists() and control_eval.exists()):
        raise FileNotFoundError("Existing S1 5e control reports are required; refusing to retrain H0")
    ps = json.loads(placement_summary.read_text(encoding="utf-8"))["variants"]["S1"]
    ev = json.loads(control_eval.read_text(encoding="utf-8"))
    return {
        "id": "H0",
        "preset": "edge27_spr_p4p5",
        "description": "REUSED existing S1: SPR P4->P5 + stock Detect reg_max=16",
        "status": "reused_existing",
        "rerun": False,
        "complexity": ps.get("complexity", {}),
        "epoch5": parse_epoch_row(control_results, 5),
        "best_eval": ev["aggregate"],
        "focus_best_eval": ev.get("focus", {}),
        "speed_ms_best_eval": ev.get("speed_ms", {}),
        "source_results": str(control_results),
        "source_eval": str(control_eval),
    }


def try_export(best: Path, env: dict, variant_report: Path):
    try:
        text = run_capture([sys.executable, "scripts/export.py", "--weights", str(best), "--format", "onnx"], env)
        (variant_report / "onnx_export.txt").write_text(text, encoding="utf-8")
        return {"ok": True}
    except Exception as exc:
        (variant_report / "onnx_export_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    screen_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    original_text = EXPERIMENT.read_text(encoding="utf-8")
    original = yaml.safe_load(original_text)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'third_party' / 'ultralytics'}:{ROOT}"

    control = load_control()
    summary = {
        "purpose": "Local 5e head mechanism screening; final paper evidence requires later locked confirmation.",
        "novelty_status": "engineering hypothesis only; no novelty claim",
        "control_policy": "H0 is reused from the existing S1 5e placement run and is not retrained.",
        "protocol": {
            "epochs_new_variants": args.epochs,
            "batch": args.batch,
            "nbs": args.nbs,
            "workers": args.workers,
            "imgsz": int(original["train"]["imgsz"]),
            "optimizer": original["train"]["optimizer"],
            "lr0": float(original["train"]["lr0"]),
            "lrf": float(original["train"]["lrf"]),
            "weight_decay": float(original["train"]["weight_decay"]),
            "seed": int(original["seed"]),
            "pretrained": False,
            "backbone": "S1: one SPR-Down at P4->P5",
            "assigner": "standard TAL",
            "loss": "standard CIoU/BCE plus DFL or direct-reg term according to head bins",
            "attention": "none",
        },
        "variants": {"H0": control},
        "ranking_best_eval_map50_95": [],
        "failed_variants": 0,
        "caution": "deterministic=false and only 5 epochs; use only to shortlist a head for 50e confirmation.",
    }

    failures = 0
    try:
        for vid, preset, description in VARIANTS:
            print("\n" + "#" * 100)
            print(f"HEAD {vid}: {preset} | {description}")
            print("#" * 100)
            variant_root = screen_root / f"{vid.lower()}_{preset}"
            variant_report = report_dir / vid.lower()
            shutil.rmtree(variant_root, ignore_errors=True)
            shutil.rmtree(variant_report, ignore_errors=True)
            for p in (variant_root / "runs", variant_root / "state", variant_root / "outputs", variant_root / "generated", variant_report):
                p.mkdir(parents=True, exist_ok=True)

            record = {"id": vid, "preset": preset, "description": description, "status": "running", "rerun": True}
            summary["variants"][vid] = record
            (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

            try:
                write_local(dataset, variant_root)
                set_preset(original, preset, args)
                sanity = run_capture([sys.executable, "scripts/sanity.py"], env)
                (variant_report / "sanity.txt").write_text(sanity, encoding="utf-8")
                record["complexity"] = parse_complexity(sanity)

                run_live([sys.executable, "scripts/train.py"], env)
                best = newest(variant_root, "runs/*/weights/best.pt")
                results = newest(variant_root, "runs/*/results.csv")
                train_args = newest(variant_root, "runs/*/args.yaml")
                eval_path = variant_report / "best_eval.json"
                run_live([sys.executable, "scripts/eval_screening.py", "--weights", str(best), "--output", str(eval_path)], env)

                shutil.copy2(results, variant_report / "results.csv")
                shutil.copy2(train_args, variant_report / "args.yaml")
                ev = json.loads(eval_path.read_text(encoding="utf-8"))
                record.update(
                    {
                        "status": "complete",
                        "epoch5": parse_epoch_row(results, args.epochs),
                        "best_eval": ev["aggregate"],
                        "focus_best_eval": ev.get("focus", {}),
                        "speed_ms_best_eval": ev.get("speed_ms", {}),
                        "onnx_export": try_export(best, env, variant_report),
                        "local_best_pt": str(best),
                    }
                )
                (variant_report / "paths.txt").write_text(
                    f"id={vid}\npreset={preset}\nroot={variant_root}\nbest_pt={best}\n"
                    "note=local 5e head screening; checkpoint remains runner-local\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                failures += 1
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                (variant_report / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(traceback.format_exc())

            (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        complete = [v for v in summary["variants"].values() if v.get("status") in {"complete", "reused_existing"}]
        h0 = summary["variants"]["H0"]
        for v in complete:
            v["delta_best_eval_vs_h0"] = {
                k: float(v["best_eval"][k]) - float(h0["best_eval"][k])
                for k in ("precision", "recall", "map50", "map50_95")
            }
        summary["ranking_best_eval_map50_95"] = [
            v["id"] for v in sorted(complete, key=lambda x: float(x["best_eval"]["map50_95"]), reverse=True)
        ]
        summary["failed_variants"] = failures
        summary["completed_new_variants"] = sum(v.get("status") == "complete" for v in summary["variants"].values())
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        EXPERIMENT.write_text(original_text, encoding="utf-8")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
