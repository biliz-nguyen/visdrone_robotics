#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "experiment.yaml"
LOCAL = ROOT / "config" / "local.yaml"
CONTROL_SUMMARY = ROOT / "reports" / "yoloedge27" / "stage3" / "head_snr_v1_5e" / "summary.json"
N1_SUMMARY = ROOT / "reports" / "yoloedge27" / "stage5" / "neck_rep_v1_5e" / "summary.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-gflops-ratio", type=float, default=1.03)
    p.add_argument("--max-params-ratio", type=float, default=1.10)
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


def parse_marker(text: str, prefix: str) -> dict:
    values = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    if not values:
        raise RuntimeError(f"Missing marker {prefix!r}")
    return json.loads(values[-1])


def deltas(a: dict, b: dict) -> dict[str, float]:
    return {k: float(a[k]) - float(b[k]) for k in ("precision", "recall", "map50", "map50_95")}


def focus_deltas(a: dict, b: dict) -> dict:
    out = {}
    for cls in ("pedestrian", "people"):
        if cls in a and cls in b:
            out[cls] = deltas(a[cls], b[cls])
    return out


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(screen_root, ignore_errors=True)
    for p in (screen_root / "runs", screen_root / "state", screen_root / "outputs", screen_root / "generated"):
        p.mkdir(parents=True, exist_ok=True)

    prior = json.loads(CONTROL_SUMMARY.read_text(encoding="utf-8"))
    controls = {k: prior["variants"][k] for k in ("H0", "H1")}
    h1 = controls["H1"]
    n1_prior = json.loads(N1_SUMMARY.read_text(encoding="utf-8"))["N1"] if N1_SUMMARY.exists() else None

    original_text = EXPERIMENT.read_text(encoding="utf-8")
    cfg = yaml.safe_load(original_text)
    cfg["preset"] = "edge27_neck_realloc_v1"
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
        required = [
            "Neck mode: realloc",
            "RepC3k2 count: 0",
            "Neck effective channels P2/P3/P4: {'p2': 40, 'p3': 64, 'p4': 96}",
            "Assigner: TaskAlignedAssigner",
            "Regression bins P2/P3/P4: [1, 1, 1]",
        ]
        for marker in required:
            if marker not in sanity:
                raise RuntimeError(f"N2 sanity marker missing: {marker}")

        probe_code = r'''
import json
import torch
from src.runtime import prepare_runtime
cfg, _, model_yaml = prepare_runtime()
from ultralytics import YOLO
from thop import profile
m = YOLO(str(model_yaml)).model.float().cpu().eval()
criterion = m.init_criterion()
seq = m.model
assert criterion.assigner.__class__.__name__ == 'TaskAlignedAssigner'
assert seq[-1].__class__.__name__ == 'Detect'
assert int(seq[-1].reg_max) == 1
assert sum(x.__class__.__name__ == 'RepC3k2' for x in m.modules()) == 0
neck = {
    'p2': int(seq[19].cv2.conv.out_channels),
    'p3': int(seq[22].cv2.conv.out_channels),
    'p4': int(seq[25].cv2.conv.out_channels),
}
pan_down = {
    'p2_p3': int(seq[20].conv.out_channels),
    'p3_p4': int(seq[23].conv.out_channels),
}
assert neck == {'p2': 40, 'p3': 64, 'p4': 96}, neck
assert pan_down == {'p2_p3': 64, 'p3_p4': 96}, pan_down
params = sum(p.numel() for p in m.parameters())
x = torch.randn(1, 3, int(cfg['train']['imgsz']), int(cfg['train']['imgsz']))
with torch.no_grad():
    macs, _ = profile(m, inputs=(x,), verbose=False)
payload = {
    'params': params,
    'gflops': macs * 2 / 1e9,
    'neck_effective': neck,
    'pan_down_effective': pan_down,
    'assigner': criterion.assigner.__class__.__name__,
    'head': seq[-1].__class__.__name__,
    'reg_max': int(seq[-1].reg_max),
}
print('N2_PROBE_JSON=' + json.dumps(payload, sort_keys=True))
'''
        probe_text = run_capture([sys.executable, "-c", probe_code], env)
        probe = parse_marker(probe_text, "N2_PROBE_JSON=")
        (report_dir / "complexity_preflight.json").write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")

        h1_params = int(h1["complexity"]["params"])
        h1_gflops = float(h1["complexity"]["gflops"])
        if int(probe["params"]) > h1_params * float(args.max_params_ratio):
            raise RuntimeError(
                f"N2 params exceed budget: {probe['params']} vs H1 {h1_params} (limit {args.max_params_ratio:.3f}x)"
            )
        if float(probe["gflops"]) > h1_gflops * float(args.max_gflops_ratio):
            raise RuntimeError(
                f"N2 GFLOPs exceed budget: {probe['gflops']:.4f} vs H1 {h1_gflops:.4f} (limit {args.max_gflops_ratio:.3f}x)"
            )

        run_live([sys.executable, "scripts/train.py"], env)
        best = newest(screen_root, "runs/*/weights/best.pt")
        results = newest(screen_root, "runs/*/results.csv")
        train_args = newest(screen_root, "runs/*/args.yaml")

        eval_path = report_dir / "neck_realloc_v1_5e_eval.json"
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

        n2 = {
            "id": "N2",
            "description": "H1 DFL-free detector with fixed fine-scale-biased neck widths",
            "status": "complete",
            "allocation": {
                "h1_nominal": {"p2": 128, "p3": 256, "p4": 512},
                "n2_nominal": {"p2": 160, "p3": 256, "p4": 384},
                "relative": {"p2": "+25%", "p3": "0%", "p4": "-25%"},
                "effective": probe["neck_effective"],
                "pan_down_effective": probe["pan_down_effective"],
            },
            "complexity": {
                "params": int(probe["params"]),
                "gflops": float(probe["gflops"]),
                "params_ratio_vs_h1": int(probe["params"]) / h1_params,
                "gflops_ratio_vs_h1": float(probe["gflops"]) / h1_gflops,
            },
            "epoch5": parse_epoch_row(results, int(args.epochs)),
            "best_eval": eval_data["aggregate"],
            "focus_best_eval": eval_data.get("focus", {}),
            "speed_ms_best_eval": eval_data.get("speed_ms", {}),
            "onnx_export": {"ok": onnx_ok, "error": onnx_error},
            "local_best_pt": str(best),
        }
        for control_id, control in controls.items():
            n2[f"delta_best_eval_vs_{control_id.lower()}"] = deltas(n2["best_eval"], control["best_eval"])
            n2[f"delta_focus_vs_{control_id.lower()}"] = focus_deltas(n2["focus_best_eval"], control["focus_best_eval"])
        if n1_prior is not None:
            n2["delta_best_eval_vs_n1"] = deltas(n2["best_eval"], n1_prior["best_eval"])
            n2["delta_focus_vs_n1"] = focus_deltas(n2["focus_best_eval"], n1_prior["focus_best_eval"])

        focus_ok = True
        for cls in ("pedestrian", "people"):
            if cls in n2["focus_best_eval"] and cls in h1["focus_best_eval"]:
                if float(n2["focus_best_eval"][cls]["map50_95"]) < float(h1["focus_best_eval"][cls]["map50_95"]) - 0.001:
                    focus_ok = False
        promote = (
            float(n2["best_eval"]["map50_95"]) >= float(h1["best_eval"]["map50_95"])
            and float(n2["best_eval"]["map50"]) >= float(h1["best_eval"]["map50"]) - 0.001
            and focus_ok
            and float(probe["gflops"]) <= h1_gflops * float(args.max_gflops_ratio)
        )

        summary = {
            "purpose": "N2 fixed fine-scale neck capacity reallocation local 5e mechanism screen; not final paper evidence.",
            "novelty_status": "capacity-allocation hypothesis/control; no novelty claim from one width setting",
            "mechanism": {
                "name": "N2 Fine-Scale Capacity Reallocation",
                "base": "H1: S1 SPR P4->P5 + direct reg_max=1",
                "changed_scope": "PAN/FPN widths only; standard C3k2 retained",
                "hypothesis": "move a limited inference budget from coarse P4 to high-resolution P2 to favor tiny-object representation",
                "attention": "none",
                "assignment": "stock TaskAlignedAssigner",
                "loss": "standard",
                "reparameterization": "none",
            },
            "protocol": {
                "epochs": int(args.epochs),
                "batch": int(args.batch),
                "nbs": int(args.nbs),
                "imgsz": 640,
                "seed": 42,
                "pretrained": False,
                "max_gflops_ratio_vs_h1": float(args.max_gflops_ratio),
                "max_params_ratio_vs_h1": float(args.max_params_ratio),
                "widths_locked_before_training": True,
            },
            "controls": controls,
            "N1_reference": n1_prior,
            "N2": n2,
            "promotion": {
                "promote_to_longer_run": bool(promote),
                "rule": "mAP50-95 >= H1, mAP50 loss <=0.1 pp, pedestrian/people mAP50-95 each no worse by >0.1 pp, GFLOPs <=1.03x H1",
            },
            "caution": "single local seed, deterministic=false, 5 epochs only; no final-paper claim from this screen",
        }
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (report_dir / "paths.txt").write_text(
            f"best={best}\nresults={results}\neval={eval_path}\n",
            encoding="utf-8",
        )
        print("N2_SUMMARY_JSON=" + json.dumps({
            "params": n2["complexity"]["params"],
            "gflops": n2["complexity"]["gflops"],
            "best_eval": n2["best_eval"],
            "focus": n2["focus_best_eval"],
            "promote": promote,
        }, sort_keys=True))
        return 0
    finally:
        EXPERIMENT.write_text(original_text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
