#!/usr/bin/env python3
"""Aggregate the four 50e paper ablation phases into paper-ready JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["a0_baseline", "a1_spr", "a2_spr_r1", "a3_final_n2b"]
METRICS = ("precision", "recall", "map50", "map50_95")
FOCUS = ("pedestrian", "people")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True)
    return p.parse_args()


def delta(a: dict, b: dict) -> dict:
    return {k: float(a[k]) - float(b[k]) for k in METRICS}


def focus_delta(a: dict, b: dict) -> dict:
    return {cls: delta(a[cls], b[cls]) for cls in FOCUS}


def main():
    args = parse_args()
    report_dir = (ROOT / args.report_dir).resolve()
    results = {}
    for key in ORDER:
        path = report_dir / f"{key}_result.json"
        if not path.exists():
            raise FileNotFoundError(path)
        results[key] = json.loads(path.read_text(encoding="utf-8"))

    rows = [results[k] for k in ORDER]
    sequential = {}
    baseline = {}
    for i, key in enumerate(ORDER):
        r = results[key]
        baseline[key] = {
            "overall": delta(r["best_eval"], results[ORDER[0]]["best_eval"]),
            "focus": focus_delta(r["focus_best_eval"], results[ORDER[0]]["focus_best_eval"]),
            "focus_mean_map50_95": float(r["focus_map50_95_mean"]) - float(results[ORDER[0]]["focus_map50_95_mean"]),
            "params": int(r["complexity"]["params"]) - int(results[ORDER[0]]["complexity"]["params"]),
            "gflops": float(r["complexity"]["gflops"]) - float(results[ORDER[0]]["complexity"]["gflops"]),
        }
        if i > 0:
            prev = results[ORDER[i - 1]]
            sequential[key] = {
                "from": prev["id"],
                "to": r["id"],
                "overall": delta(r["best_eval"], prev["best_eval"]),
                "focus": focus_delta(r["focus_best_eval"], prev["focus_best_eval"]),
                "focus_mean_map50_95": float(r["focus_map50_95_mean"]) - float(prev["focus_map50_95_mean"]),
                "params": int(r["complexity"]["params"]) - int(prev["complexity"]["params"]),
                "gflops": float(r["complexity"]["gflops"]) - float(prev["complexity"]["gflops"]),
            }

    summary = {
        "purpose": "Paper-ready sequential 50e ablation of the lightweight VisDrone detector without AConv or C12.",
        "protocol": {
            "epochs_each": 50,
            "seed": 42,
            "pretrained": False,
            "imgsz": 640,
            "batch": 8,
            "nbs": 16,
            "workers": 4,
            "assigner": "stock TaskAlignedAssigner",
            "loss": "stock v8DetectionLoss",
            "attention": "none",
            "test_dev_used": False,
            "comparison_basis": "best validation checkpoint for each independently trained variant",
        },
        "design": [
            "A0 = Conv baseline + standard neck + reg_max16",
            "A1 = A0 + SPR-Down at P4->P5 (isolates C1)",
            "A2 = A1 + direct reg_max1 deployment head (explicit compression control)",
            "A3 = A2 + N2b neck reallocation (isolates C2; final C1+C2 detector)",
        ],
        "variants": {k: results[k] for k in ORDER},
        "sequential_deltas": sequential,
        "deltas_vs_a0": baseline,
        "paper_claim_scope": {
            "C1": "A1-A0",
            "direct_reg1_control": "A2-A1",
            "C2": "A3-A2",
            "final_vs_baseline": "A3-A0",
        },
        "all_onnx_pass": all(bool(results[k].get("onnx_export", {}).get("ok")) for k in ORDER),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = report_dir / "paper_ablation_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ID", "Variant", "Params", "GFLOPs", "Precision", "Recall", "mAP50", "mAP50-95",
            "Pedestrian_AP50-95", "People_AP50-95", "Focus_mean_AP50-95", "Inference_ms", "ONNX"
        ])
        for r in rows:
            w.writerow([
                r["id"], r["label"], r["complexity"]["params"], f"{r['complexity']['gflops']:.5f}",
                f"{r['best_eval']['precision']:.8f}", f"{r['best_eval']['recall']:.8f}",
                f"{r['best_eval']['map50']:.8f}", f"{r['best_eval']['map50_95']:.8f}",
                f"{r['focus_best_eval']['pedestrian']['map50_95']:.8f}",
                f"{r['focus_best_eval']['people']['map50_95']:.8f}",
                f"{r['focus_map50_95_mean']:.8f}",
                f"{float(r.get('speed_ms_best_eval', {}).get('inference', 0.0)):.6f}",
                "PASS" if r.get("onnx_export", {}).get("ok") else "FAIL",
            ])

    print("PAPER_ABLATION_COMPARISON_COMPLETE")
    print(json.dumps({
        "final_vs_baseline": baseline["a3_final_n2b"],
        "C1_A1_minus_A0": sequential["a1_spr"],
        "R1_A2_minus_A1": sequential["a2_spr_r1"],
        "C2_A3_minus_A2": sequential["a3_final_n2b"],
        "all_onnx_pass": summary["all_onnx_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
