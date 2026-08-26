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
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "experiment.yaml"
LOCAL = ROOT / "config" / "local.yaml"
H1_5E_SUMMARY = ROOT / "reports" / "yoloedge27" / "stage3" / "head_snr_v1_5e" / "summary.json"
N2B_50E_EVAL = ROOT / "reports" / "yoloedge27" / "stage8" / "neck_realloc_v2_50e" / "neck_realloc_v2_50e_eval.json"
N2B_50E_COMPLEXITY = ROOT / "reports" / "yoloedge27" / "stage8" / "neck_realloc_v2_50e" / "complexity_preflight.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def run_capture(cmd: list[str], env: dict[str, str]) -> str:
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout, end="")
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout)
    return p.stdout


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
        raise RuntimeError(f"Need epoch {epoch_number}, got {len(rows)} rows in {path}")
    row = rows[epoch_number - 1]

    def metric(*names: str) -> float:
        for name in names:
            v = row.get(name)
            if v not in (None, ""):
                return float(v)
        raise KeyError(names)

    return {
        "precision": metric("metrics/precision(B)", "metrics/precision"),
        "recall": metric("metrics/recall(B)", "metrics/recall"),
        "map50": metric("metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": metric("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def parse_complexity(sanity: str) -> dict:
    pm = re.search(r"Params:\s*([0-9,]+)", sanity)
    gm = re.search(r"GFLOPs:\s*([0-9.]+)", sanity)
    if not pm or not gm:
        raise RuntimeError("Unable to parse Params/GFLOPs from sanity output")
    return {"params": int(pm.group(1).replace(",", "")), "gflops": float(gm.group(1))}


def deltas(a: dict, b: dict) -> dict[str, float]:
    return {k: float(a[k]) - float(b[k]) for k in ("precision", "recall", "map50", "map50_95")}


def focus_deltas(a: dict, b: dict) -> dict:
    out = {}
    for cls in ("pedestrian", "people"):
        if cls in a and cls in b:
            out[cls] = deltas(a[cls], b[cls])
    return out


def write_local(dataset: Path, root: Path) -> None:
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
                f'runs_dir: "{root / "runs"}"',
                f'state_dir: "{root / "state"}"',
                f'outputs_dir: "{root / "outputs"}"',
                f'generated_dir: "{root / "generated"}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    shutil.rmtree(screen_root, ignore_errors=True)
    for p in (screen_root / "runs", screen_root / "state", screen_root / "outputs", screen_root / "generated", report_dir):
        p.mkdir(parents=True, exist_ok=True)

    if not H1_5E_SUMMARY.exists():
        raise FileNotFoundError(H1_5E_SUMMARY)
    if not (N2B_50E_EVAL.exists() and N2B_50E_COMPLEXITY.exists()):
        raise FileNotFoundError("N2b 50e reference report is required for the fair 50e comparison")

    prior = json.loads(H1_5E_SUMMARY.read_text(encoding="utf-8"))["variants"]["H1"]
    n2b_eval = json.loads(N2B_50E_EVAL.read_text(encoding="utf-8"))
    n2b_complexity = json.loads(N2B_50E_COMPLEXITY.read_text(encoding="utf-8"))

    original_text = EXPERIMENT.read_text(encoding="utf-8")
    cfg = yaml.safe_load(original_text)
    cfg["preset"] = "edge27_head_direct_r1"
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["batch"] = int(args.batch)
    cfg["train"]["nbs"] = int(args.nbs)
    cfg["train"]["workers"] = int(args.workers)
    cfg["train"]["save_period"] = 10
    cfg["train"]["patience"] = 0
    cfg["pretrained"] = False

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'third_party' / 'ultralytics'}:{ROOT}"

    try:
        write_local(dataset, screen_root)
        EXPERIMENT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        sanity = run_capture([sys.executable, "scripts/sanity.py"], env)
        (report_dir / "sanity.txt").write_text(sanity, encoding="utf-8")
        required = [
            "Preset: edge27_head_direct_r1",
            "Study: head",
            "SPR placements: ['p4_p5']",
            "Neck mode: standard",
            "RepC3k2 count: 0",
            "Head: Detect",
            "Regression bins P2/P3/P4: [1, 1, 1]",
            "Assigner: TaskAlignedAssigner",
            "Attention: none",
            "Loss: standard",
            "Pretrained: False",
        ]
        for marker in required:
            if marker not in sanity:
                raise RuntimeError(f"H1 50e sanity marker missing: {marker}")

        complexity = parse_complexity(sanity)
        expected = prior["complexity"]
        if int(complexity["params"]) != int(expected["params"]):
            raise RuntimeError(f"H1 params changed: {complexity['params']} vs prior {expected['params']}")
        if abs(float(complexity["gflops"]) - float(expected["gflops"])) > 1e-4:
            raise RuntimeError(f"H1 GFLOPs changed: {complexity['gflops']} vs prior {expected['gflops']}")
        (report_dir / "complexity_preflight.json").write_text(json.dumps(complexity, indent=2) + "\n", encoding="utf-8")

        run_live([sys.executable, "scripts/train.py"], env)
        best = newest(screen_root, "runs/*/weights/best.pt")
        results = newest(screen_root, "runs/*/results.csv")
        train_args = newest(screen_root, "runs/*/args.yaml")

        eval_path = report_dir / "h1_50e_eval.json"
        run_live([sys.executable, "scripts/eval_screening.py", "--weights", str(best), "--output", str(eval_path)], env)
        ev = json.loads(eval_path.read_text(encoding="utf-8"))

        onnx = {"ok": True, "error": None}
        try:
            text = run_capture([sys.executable, "scripts/export.py", "--weights", str(best), "--format", "onnx"], env)
            (report_dir / "onnx_export.txt").write_text(text, encoding="utf-8")
        except Exception as exc:
            onnx = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            (report_dir / "onnx_export_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")

        shutil.copy2(results, report_dir / "results.csv")
        shutil.copy2(train_args, report_dir / "args.yaml")

        h1 = {
            "id": "H1-50e",
            "preset": "edge27_head_direct_r1",
            "description": "S1 SPR P4->P5 + stock Detect reg_max=1 / DFL-free + standard neck",
            "complexity": complexity,
            "epoch50": parse_epoch_row(results, int(args.epochs)),
            "best_eval": ev["aggregate"],
            "focus_best_eval": ev.get("focus", {}),
            "speed_ms_best_eval": ev.get("speed_ms", {}),
            "onnx_export": onnx,
            "local_best_pt": str(best),
        }
        n2b = {
            "id": "N2b-50e",
            "complexity": n2b_complexity,
            "best_eval": n2b_eval["aggregate"],
            "focus_best_eval": n2b_eval.get("focus", {}),
            "speed_ms_best_eval": n2b_eval.get("speed_ms", {}),
            "source_eval": str(N2B_50E_EVAL),
        }

        comparison = {
            "delta_n2b_minus_h1": deltas(n2b["best_eval"], h1["best_eval"]),
            "delta_focus_n2b_minus_h1": focus_deltas(n2b["focus_best_eval"], h1["focus_best_eval"]),
            "params_delta_n2b_minus_h1": int(n2b_complexity["params"]) - int(complexity["params"]),
            "params_ratio_n2b_vs_h1": int(n2b_complexity["params"]) / int(complexity["params"]),
            "gflops_delta_n2b_minus_h1": float(n2b_complexity["gflops"]) - float(complexity["gflops"]),
            "gflops_ratio_n2b_vs_h1": float(n2b_complexity["gflops"]) / float(complexity["gflops"]),
        }

        summary = {
            "purpose": "Fair local 50e H1 control for the N2b neck-reallocation convergence comparison; not final paper evidence.",
            "protocol": {
                "epochs": int(args.epochs),
                "batch": int(args.batch),
                "nbs": int(args.nbs),
                "workers": int(args.workers),
                "imgsz": int(cfg["train"]["imgsz"]),
                "seed": int(cfg["seed"]),
                "pretrained": False,
                "optimizer": cfg["train"]["optimizer"],
                "lr0": float(cfg["train"]["lr0"]),
                "lrf": float(cfg["train"]["lrf"]),
                "weight_decay": float(cfg["train"]["weight_decay"]),
                "attention": "none",
                "assigner": "stock TaskAlignedAssigner",
                "loss": "standard",
                "test_dev_used": False,
            },
            "H1": h1,
            "N2b_reference": n2b,
            "comparison": comparison,
            "caution": "single local seed and deterministic=false; final paper evidence requires locked Kaggle confirmation and ideally multiple seeds.",
        }
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (report_dir / "paths.txt").write_text(
            f"best={best}\nresults={results}\neval={eval_path}\n",
            encoding="utf-8",
        )
        print("H1_50E_SUMMARY_JSON=" + json.dumps({
            "complexity": complexity,
            "best_eval": h1["best_eval"],
            "focus": h1["focus_best_eval"],
            "delta_n2b_minus_h1": comparison["delta_n2b_minus_h1"],
            "delta_focus_n2b_minus_h1": comparison["delta_focus_n2b_minus_h1"],
            "onnx": onnx,
        }, sort_keys=True))
        return 0
    finally:
        EXPERIMENT.write_text(original_text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
