#!/usr/bin/env python3
"""Run one paper-ready 50-epoch ablation phase in a fresh Python process.

The paper ablation is intentionally sequential so every row changes exactly one
factor:
  A0: Conv baseline + standard neck + reg_max=16
  A1: A0 + one SPR-Down at P4->P5
  A2: A1 + direct reg_max=1 head
  A3: A2 + N2b neck reallocation (final C1+C2 detector)

All phases use stock TAL, stock v8DetectionLoss, no attention, no AConv, no C12,
scratch training, and the same optimizer/augmentation protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "experiment.yaml"
LOCAL = ROOT / "config" / "local.yaml"

VARIANTS = {
    "a0_baseline": {
        "id": "A0",
        "label": "Baseline",
        "preset": "edge27_place_b0",
        "expected_down": "Conv",
        "expected_spr": 0,
        "expected_reg_max": 16,
        "expected_neck": {"p2": 32, "p3": 64, "p4": 128},
        "change": "Conv baseline; standard neck; standard reg_max=16",
    },
    "a1_spr": {
        "id": "A1",
        "label": "+SPR",
        "preset": "edge27_spr_p4p5",
        "expected_down": "SPRDown",
        "expected_spr": 1,
        "expected_reg_max": 16,
        "expected_neck": {"p2": 32, "p3": 64, "p4": 128},
        "change": "A0 + SPR-Down at P4->P5",
    },
    "a2_spr_r1": {
        "id": "A2",
        "label": "+Direct-R1",
        "preset": "edge27_head_direct_r1",
        "expected_down": "SPRDown",
        "expected_spr": 1,
        "expected_reg_max": 1,
        "expected_neck": {"p2": 32, "p3": 64, "p4": 128},
        "change": "A1 + direct reg_max=1 deployment head",
    },
    "a3_final_n2b": {
        "id": "A3",
        "label": "+Neck Reallocation (Final)",
        "preset": "edge27_neck_realloc_v2",
        "expected_down": "SPRDown",
        "expected_spr": 1,
        "expected_reg_max": 1,
        "expected_neck": {"p2": 40, "p3": 64, "p4": 104},
        "change": "A2 + N2b fine-scale neck reallocation",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


def write_local(dataset: Path, root: Path):
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


def run_live(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_capture(cmd: list[str], env: dict[str, str]) -> str:
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout, end="")
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout)
    return p.stdout


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

    def get(*names: str) -> float:
        for name in names:
            v = row.get(name)
            if v not in (None, ""):
                return float(v)
        raise KeyError(names)

    return {
        "precision": get("metrics/precision(B)", "metrics/precision"),
        "recall": get("metrics/recall(B)", "metrics/recall"),
        "map50": get("metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": get("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def parse_metric_trajectory(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = []
    for i, row in enumerate(rows, start=1):
        def get(name: str, fallback: str):
            v = row.get(name, row.get(fallback, ""))
            return float(v) if v not in (None, "") else None

        out.append(
            {
                "epoch": i,
                "precision": get("metrics/precision(B)", "metrics/precision"),
                "recall": get("metrics/recall(B)", "metrics/recall"),
                "map50": get("metrics/mAP50(B)", "metrics/mAP50"),
                "map50_95": get("metrics/mAP50-95(B)", "metrics/mAP50-95"),
            }
        )
    return out


def build_cfg(base_cfg: dict, spec: dict, args) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg["preset"] = spec["preset"]
    # Explicitly neutralize every later experimental mechanism.
    cfg["c3_assigner_mode"] = "standard"
    cfg["c4_loss_mode"] = "standard"
    cfg["c5_p2_refine"] = False
    cfg["c6_p2_cls_refine"] = False
    cfg["c7_p2_reg_refine"] = False
    cfg["c8_aux_mode"] = "standard"
    cfg["c9_aux_mode"] = "standard"
    cfg["c11_aux_schedule"] = "constant"
    cfg["c12_scale_velocity_mode"] = "standard"
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["batch"] = int(args.batch)
    cfg["train"]["nbs"] = int(args.nbs)
    cfg["train"]["workers"] = int(args.workers)
    cfg["train"]["save_period"] = 10
    cfg["train"]["patience"] = 0
    cfg["pretrained"] = False
    return cfg


def architecture_sanity(spec: dict, report_dir: Path) -> dict:
    from src.runtime import prepare_runtime

    cfg, data_yaml, model_yaml = prepare_runtime()

    import torch
    from thop import profile
    from ultralytics import YOLO

    yolo = YOLO(str(model_yaml))
    model = yolo.model
    seq = model.model
    detect = seq[-1]
    criterion = model.init_criterion()

    assert criterion.__class__.__name__ == "v8DetectionLoss", criterion.__class__.__name__
    assert criterion.assigner.__class__.__name__ == "TaskAlignedAssigner"
    assert detect.__class__.__name__ == "Detect", detect.__class__.__name__
    assert int(detect.reg_max) == int(spec["expected_reg_max"]), (detect.reg_max, spec)
    assert [int(x) for x in detect.stride.tolist()] == [4, 8, 16]
    assert list(detect.f) == [19, 22, 25]
    assert seq[7].__class__.__name__ == spec["expected_down"], (seq[7].__class__.__name__, spec)
    assert sum(m.__class__.__name__ == "SPRDown" for m in model.modules()) == int(spec["expected_spr"])
    assert sum(m.__class__.__name__ == "AConv" for m in model.modules()) == 0
    assert not any(m.__class__.__name__ in {"P2Refine", "P2ClsDetect", "P2RegDetect"} for m in model.modules())

    neck = {
        "p2": int(seq[19].cv2.conv.out_channels),
        "p3": int(seq[22].cv2.conv.out_channels),
        "p4": int(seq[25].cv2.conv.out_channels),
    }
    assert neck == spec["expected_neck"], (neck, spec["expected_neck"])

    params = sum(p.numel() for p in model.parameters())
    model.float().cpu().eval()
    x = torch.randn(1, 3, int(cfg["train"]["imgsz"]), int(cfg["train"]["imgsz"]))
    with torch.no_grad():
        macs, _ = profile(model, inputs=(x,), verbose=False)
    gflops = macs * 2 / 1e9

    payload = {
        "id": spec["id"],
        "label": spec["label"],
        "preset": spec["preset"],
        "criterion": criterion.__class__.__name__,
        "assigner": criterion.assigner.__class__.__name__,
        "head": detect.__class__.__name__,
        "reg_max": int(detect.reg_max),
        "strides": [int(x) for x in detect.stride.tolist()],
        "feature_indices": list(detect.f),
        "p4_p5_downsample": seq[7].__class__.__name__,
        "spr_count": sum(m.__class__.__name__ == "SPRDown" for m in model.modules()),
        "aconv_count": sum(m.__class__.__name__ == "AConv" for m in model.modules()),
        "neck_effective": neck,
        "params": int(params),
        "gflops": float(gflops),
        "data_yaml": str(data_yaml),
        "model_yaml": str(model_yaml),
        "test_dev_used": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"sanity_{spec['id'].lower()}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PAPER_ABLATION_SANITY_PASSED=" + json.dumps(payload, sort_keys=True))
    return {"params": int(params), "gflops": float(gflops)}


def try_export(best: Path, env: dict[str, str], report_dir: Path, variant_id: str) -> dict:
    try:
        text = run_capture([sys.executable, "scripts/export.py", "--weights", str(best), "--format", "onnx"], env)
        (report_dir / f"onnx_export_{variant_id.lower()}.txt").write_text(text, encoding="utf-8")
        return {"ok": True, "error": None}
    except Exception as exc:
        (report_dir / f"onnx_export_{variant_id.lower()}_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    args = parse_args()
    if args.epochs != 50:
        raise ValueError("Paper ablation is locked to exactly 50 epochs per variant")

    spec = VARIANTS[args.variant]
    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    phase_root = screen_root / args.variant
    report_dir.mkdir(parents=True, exist_ok=True)

    original_text = EXPERIMENT.read_text(encoding="utf-8")
    base_cfg = yaml.safe_load(original_text)
    cfg = build_cfg(base_cfg, spec, args)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'third_party' / 'ultralytics'}:{ROOT}"

    print("=" * 96)
    print("PAPER ABLATION 50E PHASE")
    print(f"variant={args.variant} id={spec['id']} label={spec['label']}")
    print(f"preset={spec['preset']}")
    print("fresh_process=true")
    print("stock_TAL=true stock_loss=true AConv=false C12=false pretrained=false")
    print("=" * 96)

    try:
        shutil.rmtree(phase_root, ignore_errors=True)
        for p in (phase_root / "runs", phase_root / "state", phase_root / "outputs", phase_root / "generated"):
            p.mkdir(parents=True, exist_ok=True)
        write_local(dataset, phase_root)
        EXPERIMENT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        complexity = architecture_sanity(spec, report_dir)
        if args.preflight_only:
            print(f"PAPER_ABLATION_PREFLIGHT_PASSED variant={args.variant}")
            return

        run_live([sys.executable, "scripts/train.py"], env)
        best = newest(phase_root, "runs/*/weights/best.pt")
        results = newest(phase_root, "runs/*/results.csv")
        train_args = newest(phase_root, "runs/*/args.yaml")

        eval_path = report_dir / f"{spec['id'].lower()}_best_eval.json"
        run_live([sys.executable, "scripts/eval_screening.py", "--weights", str(best), "--output", str(eval_path)], env)

        shutil.copy2(results, report_dir / f"{spec['id'].lower()}_results.csv")
        shutil.copy2(train_args, report_dir / f"{spec['id'].lower()}_args.yaml")
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        focus_mean = sum(ev["focus"][c]["map50_95"] for c in ("pedestrian", "people")) / 2.0

        result = {
            "status": "complete",
            "id": spec["id"],
            "label": spec["label"],
            "change": spec["change"],
            "preset": spec["preset"],
            "epochs": 50,
            "complexity": complexity,
            "epoch10": parse_epoch_row(results, 10),
            "epoch50": parse_epoch_row(results, 50),
            "metric_trajectory": parse_metric_trajectory(results),
            "best_eval": ev["aggregate"],
            "focus_best_eval": ev["focus"],
            "focus_map50_95_mean": float(focus_mean),
            "speed_ms_best_eval": ev.get("speed_ms", {}),
            "onnx_export": try_export(best, env, report_dir, spec["id"]),
            "local_best_pt": str(best),
        }
        if len(result["metric_trajectory"]) != 50:
            raise RuntimeError(f"Expected 50 metric rows, got {len(result['metric_trajectory'])}")

        out = report_dir / f"{args.variant}_result.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"PAPER_ABLATION_PHASE_COMPLETE variant={args.variant} result={out}")
    finally:
        EXPERIMENT.write_text(original_text, encoding="utf-8")
        if LOCAL.exists():
            LOCAL.unlink()


if __name__ == "__main__":
    main()
