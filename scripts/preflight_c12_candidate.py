#!/usr/bin/env python3
"""Run the full C12 candidate architecture/loss sanity before any 10e training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_c12_tslve_n2b_10e as paired


def parse_args():
    p = argparse.ArgumentParser()
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
    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    original_text = paired.EXPERIMENT.read_text(encoding="utf-8")
    base_cfg = yaml.safe_load(original_text)
    cfg = paired.common_config(base_cfg, args, "tslve_cls")

    phase_root = screen_root / "preflight_candidate"
    phase_root.mkdir(parents=True, exist_ok=True)

    try:
        paired.write_local(dataset, phase_root)
        paired.EXPERIMENT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        complexity = paired.architecture_sanity(True, report_dir)
        print("C12_CANDIDATE_PREFLIGHT_PASSED")
        print("criterion=TemporalScaleVelocityDetectionLoss")
        print(f"params={complexity['params']}")
        print(f"gflops={complexity['gflops']:.5f}")
        print("synthetic_calibration_and_adaptive_loss=passed")
        print("candidate_train_allowed=true")
    finally:
        paired.EXPERIMENT.write_text(original_text, encoding="utf-8")
        if paired.LOCAL.exists():
            paired.LOCAL.unlink()


if __name__ == "__main__":
    main()
