#!/usr/bin/env python3
"""Run one C12 50e paired phase in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_c12_tslve_n2b_10e as base
from scripts.c12_50e_common import run_phase_50e


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["control_n2b_50e", "candidate_c12_tslve_50e"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.epochs != 50:
        raise ValueError("C12 paired final screen is locked to exactly 50 epochs")

    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    original_text = base.EXPERIMENT.read_text(encoding="utf-8")
    base_cfg = yaml.safe_load(original_text)
    expect_c12 = args.phase == "candidate_c12_tslve_50e"
    mode = "tslve_cls" if expect_c12 else "standard"
    cfg = base.common_config(base_cfg, args, mode)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'third_party' / 'ultralytics'}:{ROOT}"

    print("=" * 90)
    print("C12 ISOLATED 50E PHASE")
    print("phase=", args.phase)
    print("pid=", os.getpid())
    print("mode=", mode)
    print("epochs=50")
    print("fresh_process=true")
    print("=" * 90)

    try:
        phase_root = screen_root / args.phase
        base.write_local(dataset, phase_root)
        base.EXPERIMENT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        if args.preflight_only:
            complexity = base.architecture_sanity(expect_c12, report_dir)
            if not expect_c12:
                raise RuntimeError("50e preflight-only is intended for the C12 candidate")
            print("C12_50E_CANDIDATE_PREFLIGHT_PASSED")
            print(f"params={complexity['params']}")
            print(f"gflops={complexity['gflops']:.5f}")
            print("candidate_train_allowed=true")
            return

        result = run_phase_50e(
            phase=args.phase,
            cfg=cfg,
            dataset=dataset,
            phase_root=phase_root,
            report_dir=report_dir,
            env=env,
            expect_c12=expect_c12,
        )
        out = report_dir / f"{args.phase}_result.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"C12_50E_ISOLATED_PHASE_COMPLETE phase={args.phase} result={out}")
    finally:
        base.EXPERIMENT.write_text(original_text, encoding="utf-8")
        if base.LOCAL.exists():
            base.LOCAL.unlink()


if __name__ == "__main__":
    main()
