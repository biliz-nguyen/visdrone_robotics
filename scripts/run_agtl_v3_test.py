#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
V2_SUMMARY = ROOT / "reports" / "yoloedge27" / "stage4" / "mentor_agtl_v2_5e" / "summary.json"


def metric_delta(a: dict, b: dict) -> dict[str, float]:
    return {
        k: float(a[k]) - float(b[k])
        for k in ("precision", "recall", "map50", "map50_95")
    }


def focus_delta(a: dict, b: dict) -> dict[str, dict[str, float]]:
    out = {}
    for cls in ("pedestrian", "people"):
        if cls in a and cls in b:
            out[cls] = metric_delta(a[cls], b[cls])
    return out


def parse_paths(argv: list[str]) -> tuple[Path, Path]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    args, _ = p.parse_known_args(argv)
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    return screen_root, report_dir


def main() -> int:
    argv = sys.argv[1:]
    screen_root, report_dir = parse_paths(argv)
    if not V2_SUMMARY.exists():
        raise FileNotFoundError(f"AGTL v2 control summary missing: {V2_SUMMARY}")

    # Reuse the validated v2 runner for dataset conversion, training, evaluation,
    # complexity checking and ONNX export. The actual mentor behaviour comes from
    # src/mentor_transfer_loss.py, which is AGTL v3 on this research branch.
    subprocess.run([sys.executable, "scripts/run_agtl_test.py", *argv], cwd=ROOT, check=True)

    summary_path = report_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prior_v2 = json.loads(V2_SUMMARY.read_text(encoding="utf-8"))

    current = summary.pop("M2")
    prior_m2 = prior_v2["M2"]
    controls = summary.setdefault("controls", {})
    controls["M2"] = prior_m2

    current["id"] = "M3"
    current["description"] = "H1 DFL-free student + Object-Normalized AGTL v3"
    current["delta_best_eval_vs_m2"] = metric_delta(current["best_eval"], prior_m2["best_eval"])
    current["delta_focus_vs_m2"] = focus_delta(
        current.get("focus_best_eval", {}), prior_m2.get("focus_best_eval", {})
    )

    # Rename the local run directory so checkpoint provenance does not retain the
    # v2 runner's historical label.
    best = Path(current["local_best_pt"])
    run_dir = best.parent.parent
    if run_dir.exists() and "agtl2-object-balanced" in run_dir.name:
        new_run_dir = run_dir.with_name(run_dir.name.replace("agtl2-object-balanced", "agtl3-object-normalized"))
        if not new_run_dir.exists():
            run_dir.rename(new_run_dir)
        current["local_best_pt"] = str(new_run_dir / "weights" / "best.pt")

    # Rename compact artifacts produced by the reused v2 runner.
    for old_name, new_name in (
        ("agtl_v2_5e_eval.json", "agtl_v3_5e_eval.json"),
        ("model_agtl_v2.yaml", "model_agtl_v3.yaml"),
    ):
        old = report_dir / old_name
        new = report_dir / new_name
        if old.exists():
            old.replace(new)

    summary["purpose"] = "AGTL v3 object-normalized local 5e mechanism screen; not final paper evidence."
    summary["novelty_status"] = "working contribution hypothesis only; no novelty claim yet"
    summary["mechanism"] = {
        "name": "Object-Normalized Advantage-Gated Tiny Localization Transfer (AGTL v3)",
        "mentor_lambda": 0.25,
        "tiny_threshold_px": 16.0,
        "advantage_margin": 0.05,
        "min_teacher_iou": 0.10,
        "candidate_selection": "same as v1/v2: student-assigned positive AND tiny GT AND mentor IoU > student IoU + margin",
        "object_normalization": "keep all eligible positives; raw weight=TAL_weight*teacher_advantage; per-object sum is normalized to the raw weight of that object's maximum-advantage v2 reference positive",
        "interpretation": "same object-level mentor budget as v2, but distributed across all useful positives instead of a hard top-1 location",
        "capacity_bridge": "mentor DFL distribution decoded first; reg1 student never mimics mentor bins/features/logits",
        "inference_change": "none",
        "extra_prediction_channels": 0,
    }
    summary["M3"] = current
    summary["decision_rule"] = (
        "Promote only if M3 improves H1 aggregate accuracy while recovering pedestrian/people relative to M2, "
        "at exactly H1 inference complexity."
    )
    summary["caution"] = "single local seed and 5 epochs; mentor-assisted training is a mechanism screen only."
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    paths = report_dir / "paths.txt"
    if paths.exists():
        text = paths.read_text(encoding="utf-8")
        text = text.replace("model_agtl_v2.yaml", "model_agtl_v3.yaml")
        text += f"v2_control_summary={V2_SUMMARY}\n"
        paths.write_text(text, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
