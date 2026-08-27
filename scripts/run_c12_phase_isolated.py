#!/usr/bin/env python3
"""Run one C12 paired-screen phase in a fresh Python process.

This deliberately isolates the N2b control and C12 candidate so patched
Ultralytics modules, model classes, and loss state cannot leak across phases.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_c12_tslve_n2b_10e as paired


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["control_n2b_10e", "candidate_c12_tslve_10e"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    if args.epochs != 10:
        raise ValueError("C12 isolated paired screen is locked to exactly 10 epochs")

    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    original_text = paired.EXPERIMENT.read_text(encoding="utf-8")
    base_cfg = yaml.safe_load(original_text)
    expect_c12 = args.phase == "candidate_c12_tslve_10e"
    mode = "tslve_cls" if expect_c12 else "standard"
    cfg = paired.common_config(base_cfg, args, mode)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'third_party' / 'ultralytics'}:{ROOT}"

    print("=" * 90)
    print("C12 ISOLATED PHASE")
    print("phase=", args.phase)
    print("pid=", os.getpid())
    print("mode=", mode)
    print("fresh_process=true")
    print("=" * 90)

    try:
        result = paired.run_phase(
            phase=args.phase,
            cfg=cfg,
            dataset=dataset,
            phase_root=screen_root / args.phase,
            report_dir=report_dir,
            env=env,
            expect_c12=expect_c12,
        )
        out = report_dir / f"{args.phase}_result.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"C12_ISOLATED_PHASE_COMPLETE phase={args.phase} result={out}")
    finally:
        paired.EXPERIMENT.write_text(original_text, encoding="utf-8")
        if paired.LOCAL.exists():
            paired.LOCAL.unlink()


if __name__ == "__main__":
    main()
