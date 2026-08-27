#!/usr/bin/env python3
"""Run C3b selection-preserving tiny-quality screening on frozen N2b.

TAQ-v1 changed the TAL ranking before positive selection and regressed the N2b
control. C3b keeps stock TAL selection exactly and applies a milder tiny-object
quality exponent only to the normalized target scores after selection.
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
CONTROL_EVAL = ROOT / "reports/yoloedge27/stage7/neck_realloc_v2_5e/neck_realloc_v2_5e_eval.json"
PRIOR_TAQ_V1 = ROOT / "reports/yoloedge27/stage12/c3_taq_n2b_v1_5e/summary.json"
EXPECTED_PARAMS = 1_454_374
EXPECTED_GFLOPS = 6.15552


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
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


def run_capture(cmd, env):
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout, end="")
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout)
    return p.stdout


def run_live(cmd, env):
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def newest(root: Path, pattern: str) -> Path:
    xs = list(root.glob(pattern))
    if not xs:
        raise FileNotFoundError(f"No file matching {pattern} under {root}")
    return max(xs, key=lambda p: p.stat().st_mtime)


def parse_epoch_row(path: Path, epoch_number: int):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    row = rows[epoch_number - 1]

    def get(*names):
        for n in names:
            if row.get(n) not in (None, ""):
                return float(row[n])
        raise KeyError(names)

    return {
        "precision": get("metrics/precision(B)", "metrics/precision"),
        "recall": get("metrics/recall(B)", "metrics/recall"),
        "map50": get("metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": get("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def delta_dict(candidate: dict, control: dict):
    return {k: float(candidate[k]) - float(control[k]) for k in ("precision", "recall", "map50", "map50_95")}


def focus_mean(focus: dict, metric: str) -> float:
    return sum(float(focus[c][metric]) for c in ("pedestrian", "people")) / 2.0


def try_export(best: Path, env: dict, report_dir: Path):
    try:
        text = run_capture([sys.executable, "scripts/export.py", "--weights", str(best), "--format", "onnx"], env)
        (report_dir / "onnx_export.txt").write_text(text, encoding="utf-8")
        return {"ok": True, "error": None}
    except Exception as exc:
        (report_dir / "onnx_export_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def architecture_sanity(report_dir: Path):
    from src.runtime import prepare_runtime

    cfg, data_yaml, model_yaml = prepare_runtime()
    import torch
    from ultralytics import YOLO

    model = YOLO(str(model_yaml))
    seq = model.model.model
    detect = seq[-1]
    criterion = model.model.init_criterion()
    assigner = criterion.assigner

    assert assigner.__class__.__name__ == "SelectionPreservingTinyQualityAssigner"
    assert float(assigner.tiny_min_side) == 16.0
    assert float(assigner.beta_floor) == 5.0
    assert int(detect.reg_max) == 1
    assert [int(x) for x in detect.stride.tolist()] == [4, 8, 16]
    assert seq[7].__class__.__name__ == "SPRDown"
    assert int(seq[19].cv2.conv.out_channels) == 40
    assert int(seq[22].cv2.conv.out_channels) == 64
    assert int(seq[25].cv2.conv.out_channels) == 104

    params = sum(p.numel() for p in model.model.parameters())
    assert params == EXPECTED_PARAMS, params

    model.model.cpu().eval()
    from thop import profile

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        macs, _ = profile(model.model, inputs=(x,), verbose=False)
    gflops = macs * 2 / 1e9
    assert abs(gflops - EXPECTED_GFLOPS) <= 1e-4, gflops

    text = "\n".join(
        [
            "C3b TAQ-SP N2b SANITY PASSED",
            f"data_yaml={data_yaml}",
            f"model_yaml={model_yaml}",
            f"assigner={assigner.__class__.__name__}",
            "selection=stock_TAL_exact",
            "quality_reweight=post_selection_only",
            "tiny_min_side=16.0",
            "beta_floor=5.0",
            "SPR=P4->P5",
            "neck_nominal=160/256/416",
            "head=Detect reg_max=1",
            f"params={params}",
            f"gflops={gflops:.5f}",
            "inference_graph_change=false",
        ]
    ) + "\n"
    print(text)
    (report_dir / "sanity.txt").write_text(text, encoding="utf-8")
    return {"params": params, "gflops": gflops}


def main():
    args = parse_args()
    if args.epochs != 5:
        raise ValueError("C3b TAQ-SP screening is locked to exactly 5 epochs")

    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    control = json.loads(CONTROL_EVAL.read_text(encoding="utf-8"))
    prior_v1 = json.loads(PRIOR_TAQ_V1.read_text(encoding="utf-8"))

    original_text = EXPERIMENT.read_text(encoding="utf-8")
    cfg = yaml.safe_load(original_text)
    cfg["preset"] = "edge27_neck_realloc_v2"
    cfg["c3_assigner_mode"] = "tiny_quality_sp"
    cfg["tiny_quality_sp"] = {"tiny_min_side": 16.0, "beta_floor": 5.0}
    cfg["train"]["epochs"] = 5
    cfg["train"]["batch"] = int(args.batch)
    cfg["train"]["nbs"] = int(args.nbs)
    cfg["train"]["workers"] = int(args.workers)
    cfg["train"]["save_period"] = 1
    cfg["train"]["patience"] = 0
    cfg["pretrained"] = False

    shutil.rmtree(screen_root, ignore_errors=True)
    shutil.rmtree(report_dir, ignore_errors=True)
    for p in (
        screen_root / "runs",
        screen_root / "state",
        screen_root / "outputs",
        screen_root / "generated",
        report_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'third_party' / 'ultralytics'}:{ROOT}"

    summary = {
        "purpose": "C3b TAQ-SP on frozen N2b; 5e screening only, not final paper evidence.",
        "mechanism": {
            "name": "Selection-Preserving Tiny-Adaptive Quality Reweighting (TAQ-SP)",
            "changed_scope": "post-selection TAL target-quality normalization only",
            "positive_selection": "stock TAL exact",
            "tiny_min_side": 16.0,
            "beta_floor": 5.0,
            "retuned_for_n2b": True,
            "inference_overhead": "none",
        },
        "protocol": {
            "epochs": 5,
            "batch": args.batch,
            "nbs": args.nbs,
            "workers": args.workers,
            "imgsz": 640,
            "seed": 42,
            "pretrained": False,
            "architecture": "frozen C1 SPR + C2 N2b + direct reg1",
            "test_dev_used": False,
        },
        "control": {
            "id": "N2b-5e-stock-TAL",
            "best_eval": control["aggregate"],
            "focus_best_eval": control["focus"],
            "focus_map50_95_mean": focus_mean(control["focus"], "map50_95"),
            "complexity": {"params": EXPECTED_PARAMS, "gflops": EXPECTED_GFLOPS},
        },
        "prior_taq_v1": {
            "best_eval": prior_v1["candidate"]["best_eval"],
            "focus_best_eval": prior_v1["candidate"]["focus_best_eval"],
            "promoted": prior_v1["promotion"]["promote_to_50e"],
        },
        "candidate": {"status": "running"},
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    try:
        write_local(dataset, screen_root)
        EXPERIMENT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        complexity = architecture_sanity(report_dir)
        run_live([sys.executable, "scripts/train.py"], env)

        best = newest(screen_root, "runs/*/weights/best.pt")
        results = newest(screen_root, "runs/*/results.csv")
        train_args = newest(screen_root, "runs/*/args.yaml")
        eval_path = report_dir / "c3_taq_sp_n2b_v2_eval.json"
        run_live([sys.executable, "scripts/eval_screening.py", "--weights", str(best), "--output", str(eval_path)], env)

        shutil.copy2(results, report_dir / "results.csv")
        shutil.copy2(train_args, report_dir / "args.yaml")
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        onnx = try_export(best, env, report_dir)

        candidate = {
            "status": "complete",
            "complexity": complexity,
            "epoch5": parse_epoch_row(results, 5),
            "best_eval": ev["aggregate"],
            "focus_best_eval": ev.get("focus", {}),
            "focus_map50_95_mean": focus_mean(ev["focus"], "map50_95"),
            "speed_ms_best_eval": ev.get("speed_ms", {}),
            "onnx_export": onnx,
            "local_best_pt": str(best),
        }
        summary["candidate"] = candidate
        summary["comparison"] = {
            "delta_candidate_minus_n2b": delta_dict(candidate["best_eval"], control["aggregate"]),
            "delta_focus_candidate_minus_n2b": {
                cls: delta_dict(candidate["focus_best_eval"][cls], control["focus"][cls])
                for cls in ("pedestrian", "people")
            },
            "delta_focus_map50_95_mean": candidate["focus_map50_95_mean"]
            - summary["control"]["focus_map50_95_mean"],
            "same_params": complexity["params"] == EXPECTED_PARAMS,
            "same_gflops": abs(complexity["gflops"] - EXPECTED_GFLOPS) <= 1e-4,
        }
        d = summary["comparison"]["delta_candidate_minus_n2b"]
        df = summary["comparison"]["delta_focus_candidate_minus_n2b"]
        promote = (
            d["map50_95"] >= 0.0
            and d["map50"] >= -0.001
            and df["pedestrian"]["map50_95"] >= -0.001
            and df["people"]["map50_95"] >= -0.001
            and summary["comparison"]["delta_focus_map50_95_mean"] >= 0.0
            and complexity["params"] == EXPECTED_PARAMS
            and abs(complexity["gflops"] - EXPECTED_GFLOPS) <= 1e-4
            and onnx["ok"]
        )
        summary["promotion"] = {
            "promote_to_50e": bool(promote),
            "rule": "overall mAP50-95 >= N2b; mAP50 loss <=0.1pp; focus mean mAP50-95 >= N2b; each pedestrian/people mAP50-95 no worse by >0.1pp; identical inference complexity; ONNX pass",
        }
        summary["caution"] = "single local seed, deterministic=false, 5 epochs only; promote only if the locked rule passes."
        (report_dir / "paths.txt").write_text(
            f"best_pt={best}\ncontrol_eval={CONTROL_EVAL}\nprior_taq_v1={PRIOR_TAQ_V1}\ncheckpoint_note=runner-local only\n",
            encoding="utf-8",
        )
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        summary["candidate"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        (report_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        EXPERIMENT.write_text(original_text, encoding="utf-8")


if __name__ == "__main__":
    main()
