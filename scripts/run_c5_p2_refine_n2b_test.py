#!/usr/bin/env python3
"""Run C5 head-only P2 refinement screening on frozen N2b direct-reg1."""

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
BASE_PARAMS = 1_454_374
BASE_GFLOPS = 6.15552
EXPECTED_REFINE_PARAMS = 2_121
EXPECTED_PARAMS = BASE_PARAMS + EXPECTED_REFINE_PARAMS
MAX_PARAMS = 1_500_000
MAX_GFLOPS = 6.30


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
    row = list(csv.DictReader(path.open(encoding="utf-8")))[epoch_number - 1]

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

    assert criterion.assigner.__class__.__name__ == "TaskAlignedAssigner"
    assert int(detect.reg_max) == 1
    assert [int(x) for x in detect.stride.tolist()] == [4, 8, 16]
    assert list(detect.f) == [26, 22, 25], detect.f
    assert seq[7].__class__.__name__ == "SPRDown"
    assert int(seq[19].cv2.conv.out_channels) == 40
    assert int(seq[22].cv2.conv.out_channels) == 64
    assert int(seq[25].cv2.conv.out_channels) == 104
    refine = seq[26]
    assert refine.__class__.__name__ == "P2Refine"
    assert int(refine.c1) == 40 and int(refine.c2) == 40
    assert abs(float(refine.alpha.detach()) - 0.10) <= 1e-7
    assert sum(p.numel() for p in refine.parameters()) == EXPECTED_REFINE_PARAMS

    params = sum(p.numel() for p in model.model.parameters())
    assert params == EXPECTED_PARAMS, params
    assert params <= MAX_PARAMS

    model.model.cpu().eval()
    from thop import profile

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        macs, _ = profile(model.model, inputs=(x,), verbose=False)
    gflops = macs * 2 / 1e9
    assert gflops > BASE_GFLOPS, gflops
    assert gflops <= MAX_GFLOPS, gflops

    text = "\n".join(
        [
            "C5 P2-REFINE N2b SANITY PASSED",
            f"data_yaml={data_yaml}",
            f"model_yaml={model_yaml}",
            f"assigner={criterion.assigner.__class__.__name__}",
            "SPR=P4->P5",
            "neck_nominal=160/256/416",
            "head=Detect reg_max=1",
            "p2_refine=DW3x3+PW1x1 residual alpha=0.10",
            "detect_inputs=refined-P2/stock-P3/stock-P4",
            f"params={params}",
            f"delta_params={params - BASE_PARAMS}",
            f"gflops={gflops:.5f}",
            f"delta_gflops={gflops - BASE_GFLOPS:.5f}",
            "inference_graph_change=true",
        ]
    ) + "\n"
    print(text)
    (report_dir / "sanity.txt").write_text(text, encoding="utf-8")
    return {"params": params, "gflops": gflops}


def main():
    args = parse_args()
    if args.epochs != 5:
        raise ValueError("C5 screening is locked to exactly 5 epochs")

    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    control = json.loads(CONTROL_EVAL.read_text(encoding="utf-8"))

    original_text = EXPERIMENT.read_text(encoding="utf-8")
    cfg = yaml.safe_load(original_text)
    cfg["preset"] = "edge27_neck_realloc_v2"
    cfg["c3_assigner_mode"] = "standard"
    cfg["c4_loss_mode"] = "standard"
    cfg["c5_p2_refine"] = True
    cfg["train"]["epochs"] = 5
    cfg["train"]["batch"] = int(args.batch)
    cfg["train"]["nbs"] = int(args.nbs)
    cfg["train"]["workers"] = int(args.workers)
    cfg["train"]["save_period"] = 1
    cfg["train"]["patience"] = 0
    cfg["pretrained"] = False

    shutil.rmtree(screen_root, ignore_errors=True)
    shutil.rmtree(report_dir, ignore_errors=True)
    for p in (screen_root / "runs", screen_root / "state", screen_root / "outputs", screen_root / "generated", report_dir):
        p.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'third_party' / 'ultralytics'}:{ROOT}"

    control_focus_mean = sum(control["focus"][c]["map50_95"] for c in ("pedestrian", "people")) / 2.0
    summary = {
        "purpose": "C5 head-only P2 refinement on frozen N2b; 5e screening only, not final paper evidence.",
        "mechanism": {
            "name": "P2 Residual Head Refinement (P2R)",
            "changed_scope": "P2 Detect input only",
            "path_isolation": "refined P2 is not fed back into PAN/P3/P4",
            "block": "DW3x3 + BN + SiLU + PW1x1 + BN + alpha residual",
            "alpha_init": 0.10,
            "assigner": "stock TAL exact",
            "loss": "stock",
            "reg_max": 1,
            "inference_overhead": "one lightweight stride-4 residual block",
        },
        "protocol": {
            "epochs": 5,
            "batch": args.batch,
            "nbs": args.nbs,
            "workers": args.workers,
            "imgsz": 640,
            "seed": 42,
            "pretrained": False,
            "architecture": "frozen C1 SPR + C2 N2b + direct reg1 + isolated P2R",
            "test_dev_used": False,
        },
        "budget": {"max_params": MAX_PARAMS, "max_gflops": MAX_GFLOPS},
        "control": {
            "id": "N2b-5e-stock-head",
            "best_eval": control["aggregate"],
            "focus_best_eval": control["focus"],
            "focus_map50_95_mean": control_focus_mean,
            "complexity": {"params": BASE_PARAMS, "gflops": BASE_GFLOPS},
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
        eval_path = report_dir / "c5_p2_refine_n2b_v1_eval.json"
        run_live([sys.executable, "scripts/eval_screening.py", "--weights", str(best), "--output", str(eval_path)], env)

        shutil.copy2(results, report_dir / "results.csv")
        shutil.copy2(train_args, report_dir / "args.yaml")
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        onnx = try_export(best, env, report_dir)
        focus_mean = sum(ev["focus"][c]["map50_95"] for c in ("pedestrian", "people")) / 2.0

        candidate = {
            "status": "complete",
            "complexity": complexity,
            "epoch5": parse_epoch_row(results, 5),
            "best_eval": ev["aggregate"],
            "focus_best_eval": ev.get("focus", {}),
            "focus_map50_95_mean": focus_mean,
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
            "delta_focus_map50_95_mean": focus_mean - control_focus_mean,
            "delta_params": complexity["params"] - BASE_PARAMS,
            "delta_gflops": complexity["gflops"] - BASE_GFLOPS,
            "within_param_budget": complexity["params"] <= MAX_PARAMS,
            "within_gflops_budget": complexity["gflops"] <= MAX_GFLOPS,
        }
        d = summary["comparison"]["delta_candidate_minus_n2b"]
        df = summary["comparison"]["delta_focus_candidate_minus_n2b"]
        promote = (
            d["map50_95"] >= 0.0
            and d["map50"] >= -0.001
            and summary["comparison"]["delta_focus_map50_95_mean"] >= 0.0
            and df["pedestrian"]["map50_95"] >= -0.001
            and df["people"]["map50_95"] >= -0.001
            and summary["comparison"]["within_param_budget"]
            and summary["comparison"]["within_gflops_budget"]
            and onnx["ok"]
        )
        summary["promotion"] = {
            "promote_to_50e": bool(promote),
            "rule": "overall mAP50-95 >= N2b; mAP50 loss <=0.1pp; focus mean >= N2b; each focus class no worse by >0.1pp; params<=1.50M; GFLOPs<=6.30; ONNX pass",
        }
        summary["caution"] = "single local seed, deterministic=false, 5 epochs only; promote only if locked rule passes."
        (report_dir / "paths.txt").write_text(
            f"best_pt={best}\ncontrol_eval={CONTROL_EVAL}\ncheckpoint_note=runner-local only\n",
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
