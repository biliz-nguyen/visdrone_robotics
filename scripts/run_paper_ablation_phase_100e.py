#!/usr/bin/env python3
"""Run one paper-ready 100-epoch ablation phase in a fresh Python process.

This wrapper deliberately reuses the validated architecture sanity, VisDrone
conversion, standalone best.pt evaluation, ONNX export, and reporting helpers
from the 50e runner while keeping the 100e protocol isolated from historical
50e experiments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import yaml

import scripts.run_paper_ablation_phase_50e as base


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(base.VARIANTS), required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.epochs != 100:
        raise ValueError("Paper 100e ablation is locked to exactly 100 epochs per variant")

    spec = base.VARIANTS[args.variant]
    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (base.ROOT / args.report_dir).resolve()
    phase_root = screen_root / args.variant
    report_dir.mkdir(parents=True, exist_ok=True)

    original_text = base.EXPERIMENT.read_text(encoding="utf-8")
    base_cfg = yaml.safe_load(original_text)
    cfg = base.build_cfg(base_cfg, spec, args)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{base.ROOT / 'third_party' / 'ultralytics'}:{base.ROOT}"

    print("=" * 96)
    print("PAPER ABLATION 100E PHASE")
    print(f"variant={args.variant} id={spec['id']} label={spec['label']}")
    print(f"preset={spec['preset']}")
    print("fresh_process=true")
    print("stock_TAL=true stock_loss=true AConv=false C12=false pretrained=false")
    print("=" * 96)

    try:
        shutil.rmtree(phase_root, ignore_errors=True)
        for p in (phase_root / "runs", phase_root / "state", phase_root / "outputs", phase_root / "generated"):
            p.mkdir(parents=True, exist_ok=True)
        base.write_local(dataset, phase_root)
        base.EXPERIMENT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        complexity = base.architecture_sanity(spec, report_dir)
        if args.preflight_only:
            print(f"PAPER_ABLATION_100E_PREFLIGHT_PASSED variant={args.variant}")
            return

        base.run_live([sys.executable, "scripts/train.py"], env)
        best = base.newest(phase_root, "runs/*/weights/best.pt")
        results = base.newest(phase_root, "runs/*/results.csv")
        train_args = base.newest(phase_root, "runs/*/args.yaml")

        eval_path = report_dir / f"{spec['id'].lower()}_best_eval.json"
        base.run_live([sys.executable, "scripts/eval_screening.py", "--weights", str(best), "--output", str(eval_path)], env)

        shutil.copy2(results, report_dir / f"{spec['id'].lower()}_results.csv")
        shutil.copy2(train_args, report_dir / f"{spec['id'].lower()}_args.yaml")
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        focus_mean = sum(ev["focus"][c]["map50_95"] for c in ("pedestrian", "people")) / 2.0

        trajectory = base.parse_metric_trajectory(results)
        result = {
            "status": "complete",
            "id": spec["id"],
            "label": spec["label"],
            "change": spec["change"],
            "preset": spec["preset"],
            "epochs": 100,
            "complexity": complexity,
            "epoch10": base.parse_epoch_row(results, 10),
            "epoch50": base.parse_epoch_row(results, 50),
            "epoch100": base.parse_epoch_row(results, 100),
            "metric_trajectory": trajectory,
            "best_eval": ev["aggregate"],
            "focus_best_eval": ev["focus"],
            "focus_map50_95_mean": float(focus_mean),
            "speed_ms_best_eval": ev.get("speed_ms", {}),
            "onnx_export": base.try_export(best, env, report_dir, spec["id"]),
            "local_best_pt": str(best),
        }
        if len(trajectory) != 100:
            raise RuntimeError(f"Expected 100 metric rows, got {len(trajectory)}")

        out = report_dir / f"{args.variant}_result.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"PAPER_ABLATION_100E_PHASE_COMPLETE variant={args.variant} result={out}")
    finally:
        base.EXPERIMENT.write_text(original_text, encoding="utf-8")
        if base.LOCAL.exists():
            base.LOCAL.unlink()


if __name__ == "__main__":
    main()
