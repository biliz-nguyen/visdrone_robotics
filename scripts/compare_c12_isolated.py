#!/usr/bin/env python3
"""Compare isolated N2b 10e control and C12 10e candidate results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HIST_N2B_5E = ROOT / "reports/yoloedge27/stage7/neck_realloc_v2_5e/neck_realloc_v2_5e_eval.json"
EXPECTED_PARAMS = 1_454_374
EXPECTED_GFLOPS = 6.15552


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True)
    return p.parse_args()


def delta_dict(candidate: dict, control: dict):
    return {k: float(candidate[k]) - float(control[k]) for k in ("precision", "recall", "map50", "map50_95")}


def main():
    args = parse_args()
    report_dir = (ROOT / args.report_dir).resolve()
    control = json.loads((report_dir / "control_n2b_10e_result.json").read_text(encoding="utf-8"))
    candidate = json.loads((report_dir / "candidate_c12_tslve_10e_result.json").read_text(encoding="utf-8"))

    control_focus = control["focus_best_eval"]
    candidate_focus = candidate["focus_best_eval"]
    comparison = {
        "delta_candidate_minus_control": delta_dict(candidate["best_eval"], control["best_eval"]),
        "delta_focus_candidate_minus_control": {
            cls: delta_dict(candidate_focus[cls], control_focus[cls])
            for cls in ("pedestrian", "people")
        },
        "delta_focus_map50_95_mean": candidate["focus_map50_95_mean"] - control["focus_map50_95_mean"],
        "same_params": candidate["complexity"]["params"] == control["complexity"]["params"] == EXPECTED_PARAMS,
        "same_gflops": abs(candidate["complexity"]["gflops"] - control["complexity"]["gflops"]) <= 1e-4,
    }

    d = comparison["delta_candidate_minus_control"]
    df = comparison["delta_focus_candidate_minus_control"]
    promote = (
        d["map50_95"] >= 0.0
        and d["map50"] >= -0.001
        and comparison["delta_focus_map50_95_mean"] >= 0.0
        and df["pedestrian"]["map50_95"] >= -0.001
        and df["people"]["map50_95"] >= -0.001
        and comparison["same_params"]
        and comparison["same_gflops"]
        and candidate.get("onnx_export", {}).get("ok", False)
    )

    summary = {
        "purpose": "Paired 10e isolated-process test of Temporal Scale Learning-Velocity Equalization (TSLVE) against frozen N2b stock loss.",
        "mechanism": {
            "name": "Temporal Scale Learning-Velocity Equalization (TSLVE)",
            "scope": "training-only positive classification gradient redistribution inside stock TAL",
            "scale_groups_px_min_side": {"tiny": "<16", "small": "16-32", "regular": ">=32"},
            "calibration": "epoch 1 stock loss establishes per-scale confidence-difficulty reference",
            "difficulty_statistic": "BCE(target=1) of assigned true-class logit on stock TAL positives",
            "progress": "EMA_current_difficulty / epoch1_reference_difficulty",
            "ema_beta": 0.95,
            "velocity_alpha": 0.50,
            "weight_range": [0.75, 1.25],
            "normalization": "count-weighted mean positive weight approximately 1",
            "changed_gradient": "assigned true-class BCE only",
            "unchanged": ["all classes treated identically", "stock TAL", "stock box loss", "direct reg1", "architecture", "inference graph"],
            "deployment_overhead": "0 params / 0 GFLOPs / 0 nodes",
        },
        "protocol": {
            "paired_control": True,
            "process_isolation": True,
            "epochs_each": 10,
            "batch": 8,
            "nbs": 16,
            "workers": 4,
            "imgsz": 640,
            "seed": 42,
            "pretrained": False,
            "architecture": "frozen C1 SPR + C2 N2b + stock direct-reg1 Detect",
            "test_dev_used": False,
        },
        "control_10e": control,
        "candidate_10e": candidate,
        "comparison": comparison,
        "promotion": {
            "promote_beyond_10e": bool(promote),
            "rule": "paired 10e: overall mAP50-95 >= control; mAP50 loss <=0.1pp; focus mean >= control; each focus class no worse by >0.1pp; identical deploy complexity; ONNX pass",
        },
        "caution": "C12 novelty is treated as a candidate claim; exact priority was not established by the literature search.",
    }

    if HIST_N2B_5E.exists():
        hist = json.loads(HIST_N2B_5E.read_text(encoding="utf-8"))
        summary["historical_n2b_5e"] = {
            "aggregate": hist.get("aggregate"),
            "focus": hist.get("focus"),
            "note": "context only; not used for promotion because epochs differ",
        }

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (report_dir / "paths.txt").write_text(
        "\n".join(
            [
                f"control_best_pt={control['local_best_pt']}",
                f"candidate_best_pt={candidate['local_best_pt']}",
                "process_isolation=true",
                "checkpoint_note=runner-local only",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
