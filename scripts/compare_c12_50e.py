#!/usr/bin/env python3
"""Compare isolated N2b 50e control and C12 TSLVE 50e candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARAMS = 1_454_374
EXPECTED_GFLOPS = 6.15552
HIST_C12_10E = ROOT / "reports/yoloedge27/stage22/c12_tslve_n2b_v1_10e/summary.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True)
    return p.parse_args()


def delta_dict(candidate: dict, control: dict):
    return {k: float(candidate[k]) - float(control[k]) for k in ("precision", "recall", "map50", "map50_95")}


def temporal_analysis(rows: list[dict]) -> dict:
    adaptive = [r for r in rows if int(r["epoch"]) >= 2]
    if len(adaptive) != 49:
        raise RuntimeError(f"Expected 49 adaptive epochs, got {len(adaptive)}")

    tiny_highest = 0
    tiny_weight_gt1 = 0
    margins = []
    for r in adaptive:
        p = [float(x) for x in r["last_progress"]]
        w = [float(x) for x in r["mean_applied_weight"]]
        if p[0] >= max(p[1], p[2]):
            tiny_highest += 1
        if w[0] > 1.0:
            tiny_weight_gt1 += 1
        margins.append(p[0] - max(p[1], p[2]))

    late = adaptive[-10:]
    late_progress = [statistics.mean(float(r["last_progress"][g]) for r in late) for g in range(3)]
    late_weights = [statistics.mean(float(r["mean_applied_weight"][g]) for r in late) for g in range(3)]

    return {
        "adaptive_epochs": len(adaptive),
        "tiny_progress_highest_epochs": tiny_highest,
        "tiny_progress_highest_fraction": tiny_highest / len(adaptive),
        "tiny_mean_margin_over_next_scale": statistics.mean(margins),
        "tiny_mean_weight_gt1_epochs": tiny_weight_gt1,
        "tiny_mean_weight_gt1_fraction": tiny_weight_gt1 / len(adaptive),
        "late10_mean_progress_tiny_small_regular": late_progress,
        "late10_mean_weight_tiny_small_regular": late_weights,
        "interpretation_rule": "higher residual progress means slower relative learning than the epoch-1 reference",
    }


def main():
    args = parse_args()
    report_dir = (ROOT / args.report_dir).resolve()
    control = json.loads((report_dir / "control_n2b_50e_result.json").read_text(encoding="utf-8"))
    candidate = json.loads((report_dir / "candidate_c12_tslve_50e_result.json").read_text(encoding="utf-8"))

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
    retain = (
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
        "purpose": "Paired 50e isolated-process validation of C12 Temporal Scale Learning-Velocity Equalization (TSLVE) against frozen N2b.",
        "mechanism": {
            "name": "Temporal Scale Learning-Velocity Equalization (TSLVE)",
            "scale_groups_px_min_side": {"tiny": "<16", "small": "16-32", "regular": ">=32"},
            "calibration": "epoch 1 stock loss reference",
            "ema_beta": 0.95,
            "velocity_alpha": 0.50,
            "weight_range": [0.75, 1.25],
            "changed_gradient": "assigned true-class BCE only",
            "unchanged": ["all classes treated identically", "stock TAL", "stock box loss", "direct reg1", "architecture", "inference graph"],
            "deployment_overhead": "0 params / 0 GFLOPs / 0 nodes",
        },
        "protocol": {
            "paired_control": True,
            "process_isolation": True,
            "epochs_each": 50,
            "batch": 8,
            "nbs": 16,
            "workers": 4,
            "imgsz": 640,
            "seed": 42,
            "pretrained": False,
            "architecture": "frozen C1 SPR + C2 N2b + stock direct-reg1 Detect",
            "test_dev_used": False,
        },
        "control_50e": control,
        "candidate_50e": candidate,
        "comparison": comparison,
        "temporal_scale_analysis": temporal_analysis(candidate["scale_dynamics"]),
        "retention": {
            "retain_c12_after_50e": bool(retain),
            "rule": "paired 50e: overall mAP50-95 >= control; mAP50 loss <=0.1pp; focus mean >= control; each focus class no worse by >0.1pp; identical deploy complexity; ONNX pass",
        },
        "caution": "50e validates persistence under one seed; multi-seed validation is still required before a strong generalization or novelty claim.",
    }

    if HIST_C12_10E.exists():
        hist = json.loads(HIST_C12_10E.read_text(encoding="utf-8"))
        summary["historical_c12_10e"] = {
            "comparison": hist.get("comparison"),
            "promotion": hist.get("promotion"),
            "note": "context only; 50e retention uses the paired 50e control",
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
