#!/usr/bin/env python3
"""Paired 10e screen: frozen N2b stock loss vs C12 TSLVE, both from scratch."""

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
HIST_N2B_5E = ROOT / "reports/yoloedge27/stage7/neck_realloc_v2_5e/neck_realloc_v2_5e_eval.json"
EXPECTED_PARAMS = 1_454_374
EXPECTED_GFLOPS = 6.15552


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


def run_live(cmd, env):
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_capture(cmd, env):
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout, end="")
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout)
    return p.stdout


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


def parse_metric_trajectory(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = []
    for i, row in enumerate(rows, start=1):
        def get(name, fallback):
            v = row.get(name, row.get(fallback, ""))
            return float(v) if v not in (None, "") else None

        out.append(
            {
                "epoch": i,
                "precision": get("metrics/precision(B)", "metrics/precision"),
                "recall": get("metrics/recall(B)", "metrics/recall"),
                "map50": get("metrics/mAP50(B)", "metrics/mAP50"),
                "map50_95": get("metrics/mAP50-95(B)", "metrics/mAP50-95"),
            }
        )
    return out


def delta_dict(candidate: dict, control: dict):
    return {k: float(candidate[k]) - float(control[k]) for k in ("precision", "recall", "map50", "map50_95")}


def common_config(base: dict, args, c12_mode: str) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["preset"] = "edge27_neck_realloc_v2"
    cfg["c3_assigner_mode"] = "standard"
    cfg["c4_loss_mode"] = "standard"
    cfg["c5_p2_refine"] = False
    cfg["c6_p2_cls_refine"] = False
    cfg["c7_p2_reg_refine"] = False
    cfg["c8_aux_mode"] = "standard"
    cfg["c9_aux_mode"] = "standard"
    cfg["c11_aux_schedule"] = "constant"
    cfg["c12_scale_velocity_mode"] = c12_mode
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["batch"] = int(args.batch)
    cfg["train"]["nbs"] = int(args.nbs)
    cfg["train"]["workers"] = int(args.workers)
    cfg["train"]["save_period"] = 5
    cfg["train"]["patience"] = 0
    cfg["pretrained"] = False
    return cfg


def architecture_sanity(expect_c12: bool, report_dir: Path):
    from src.runtime import prepare_runtime

    cfg, data_yaml, model_yaml = prepare_runtime()
    import torch
    from ultralytics import YOLO

    model = YOLO(str(model_yaml))
    seq = model.model.model
    detect = seq[-1]
    criterion = model.model.init_criterion()

    if expect_c12:
        assert criterion.__class__.__name__ == "TemporalScaleVelocityDetectionLoss", criterion.__class__.__name__
        assert abs(float(criterion.tiny_thr) - 16.0) <= 1e-9
        assert abs(float(criterion.small_thr) - 32.0) <= 1e-9
        assert abs(float(criterion.ema_beta) - 0.95) <= 1e-9
        assert abs(float(criterion.velocity_alpha) - 0.50) <= 1e-9
        assert abs(float(criterion.weight_min) - 0.75) <= 1e-9
        assert abs(float(criterion.weight_max) - 1.25) <= 1e-9
        criterion.set_epoch(0)
    else:
        assert criterion.__class__.__name__ == "v8DetectionLoss", criterion.__class__.__name__

    assert criterion.assigner.__class__.__name__ == "TaskAlignedAssigner"
    assert detect.__class__.__name__ == "Detect"
    assert int(detect.reg_max) == 1
    assert [int(x) for x in detect.stride.tolist()] == [4, 8, 16]
    assert list(detect.f) == [19, 22, 25]
    assert seq[7].__class__.__name__ == "SPRDown"
    assert int(seq[19].cv2.conv.out_channels) == 40
    assert int(seq[22].cv2.conv.out_channels) == 64
    assert int(seq[25].cv2.conv.out_channels) == 104
    assert not any(m.__class__.__name__ in {"P2Refine", "P2ClsDetect", "P2RegDetect"} for m in model.model.modules())

    params = sum(p.numel() for p in model.model.parameters())
    assert params == EXPECTED_PARAMS, params

    if expect_c12:
        # Two synthetic passes validate both calibration and adaptive epochs.
        model.model.train()
        x = torch.randn(2, 3, 64, 64)
        batch = {
            "batch_idx": torch.tensor([0, 0, 0, 1, 1, 1]),
            "cls": torch.tensor([[0.0], [5.0], [6.0], [1.0], [2.0], [3.0]]),
            "bboxes": torch.tensor(
                [
                    [0.20, 0.20, 0.10, 0.10],
                    [0.50, 0.20, 0.25, 0.25],
                    [0.75, 0.25, 0.55, 0.55],
                    [0.20, 0.70, 0.12, 0.12],
                    [0.50, 0.70, 0.30, 0.30],
                    [0.75, 0.70, 0.60, 0.60],
                ]
            ),
        }
        preds = model.model(x)
        loss0, _ = criterion.loss(preds, batch)
        assert torch.isfinite(loss0).all()
        criterion.set_epoch(1)
        preds = model.model(x)
        loss1, _ = criterion.loss(preds, batch)
        assert torch.isfinite(loss1).all()
        assert bool(torch.isfinite(criterion.reference).all())
        assert bool((criterion.last_weights >= 0.75).all())
        assert bool((criterion.last_weights <= 1.25).all())

    model.model.cpu().eval()
    from thop import profile

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        macs, _ = profile(model.model, inputs=(x,), verbose=False)
    gflops = macs * 2 / 1e9
    assert abs(gflops - EXPECTED_GFLOPS) <= 1e-4, gflops

    text = "\n".join(
        [
            f"C12 {'CANDIDATE' if expect_c12 else 'CONTROL'} SANITY PASSED",
            f"data_yaml={data_yaml}",
            f"model_yaml={model_yaml}",
            f"criterion={criterion.__class__.__name__}",
            f"assigner={criterion.assigner.__class__.__name__}",
            "architecture=C1 SPR + C2 N2b + stock Detect direct-reg1",
            "feature_indices=19,22,25",
            "strides=4,8,16",
            f"params={params}",
            f"gflops={gflops:.5f}",
            "test_dev_used=false",
        ]
    ) + "\n"
    print(text)
    (report_dir / ("sanity_candidate.txt" if expect_c12 else "sanity_control.txt")).write_text(text, encoding="utf-8")
    return {"params": params, "gflops": gflops}


def try_export(best: Path, env: dict, report_dir: Path):
    try:
        text = run_capture([sys.executable, "scripts/export.py", "--weights", str(best), "--format", "onnx"], env)
        (report_dir / "onnx_export.txt").write_text(text, encoding="utf-8")
        return {"ok": True, "error": None}
    except Exception as exc:
        (report_dir / "onnx_export_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def run_phase(
    *,
    phase: str,
    cfg: dict,
    dataset: Path,
    phase_root: Path,
    report_dir: Path,
    env: dict,
    expect_c12: bool,
):
    shutil.rmtree(phase_root, ignore_errors=True)
    for p in (phase_root / "runs", phase_root / "state", phase_root / "outputs", phase_root / "generated"):
        p.mkdir(parents=True, exist_ok=True)

    write_local(dataset, phase_root)
    EXPERIMENT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    complexity = architecture_sanity(expect_c12, report_dir)

    run_live([sys.executable, "scripts/train.py"], env)
    best = newest(phase_root, "runs/*/weights/best.pt")
    results = newest(phase_root, "runs/*/results.csv")
    train_args = newest(phase_root, "runs/*/args.yaml")
    eval_path = report_dir / f"{phase}_eval.json"
    run_live([sys.executable, "scripts/eval_screening.py", "--weights", str(best), "--output", str(eval_path)], env)

    shutil.copy2(results, report_dir / f"{phase}_results.csv")
    shutil.copy2(train_args, report_dir / f"{phase}_args.yaml")
    ev = json.loads(eval_path.read_text(encoding="utf-8"))
    focus_mean = sum(ev["focus"][c]["map50_95"] for c in ("pedestrian", "people")) / 2.0

    out = {
        "status": "complete",
        "complexity": complexity,
        "epoch10": parse_epoch_row(results, 10),
        "metric_trajectory": parse_metric_trajectory(results),
        "best_eval": ev["aggregate"],
        "focus_best_eval": ev["focus"],
        "focus_map50_95_mean": focus_mean,
        "speed_ms_best_eval": ev.get("speed_ms", {}),
        "local_best_pt": str(best),
    }

    if expect_c12:
        dynamics = phase_root / "state" / "c12_scale_dynamics.jsonl"
        if not dynamics.exists():
            raise FileNotFoundError(f"C12 dynamics log missing: {dynamics}")
        shutil.copy2(dynamics, report_dir / "c12_scale_dynamics.jsonl")
        rows = [json.loads(line) for line in dynamics.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 10:
            raise RuntimeError(f"Expected 10 C12 dynamics rows, got {len(rows)}")
        out["scale_dynamics"] = rows
        out["onnx_export"] = try_export(best, env, report_dir)

    return out


def main():
    args = parse_args()
    if args.epochs != 10:
        raise ValueError("C12 paired screen is locked to exactly 10 epochs")

    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    original_text = EXPERIMENT.read_text(encoding="utf-8")
    original_cfg = yaml.safe_load(original_text)

    shutil.rmtree(screen_root, ignore_errors=True)
    shutil.rmtree(report_dir, ignore_errors=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'third_party' / 'ultralytics'}:{ROOT}"

    summary = {
        "purpose": "Paired 10e test of Temporal Scale Learning-Velocity Equalization (TSLVE) against frozen N2b stock loss.",
        "mechanism": {
            "name": "Temporal Scale Learning-Velocity Equalization (TSLVE)",
            "scope": "training-only positive classification gradient redistribution inside stock TAL",
            "scale_groups_px_min_side": {"tiny": "<16", "small": "16-32", "regular": ">=32"},
            "calibration": "epoch 1 stock loss establishes per-scale confidence-difficulty reference",
            "difficulty_statistic": "BCE(target=1) of the assigned true-class logit on stock TAL positives",
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
            "epochs_each": 10,
            "batch": args.batch,
            "nbs": args.nbs,
            "workers": args.workers,
            "imgsz": 640,
            "seed": 42,
            "pretrained": False,
            "architecture": "frozen C1 SPR + C2 N2b + stock direct-reg1 Detect",
            "test_dev_used": False,
        },
        "control_10e": {"status": "pending"},
        "candidate_10e": {"status": "pending"},
    }
    if HIST_N2B_5E.exists():
        hist = json.loads(HIST_N2B_5E.read_text(encoding="utf-8"))
        summary["historical_n2b_5e"] = {
            "aggregate": hist.get("aggregate"),
            "focus": hist.get("focus"),
            "note": "context only; NOT used for C12 promotion because epochs differ",
        }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    try:
        control_cfg = common_config(original_cfg, args, "standard")
        candidate_cfg = common_config(original_cfg, args, "tslve_cls")

        control = run_phase(
            phase="control_n2b_10e",
            cfg=control_cfg,
            dataset=dataset,
            phase_root=screen_root / "control_n2b_10e",
            report_dir=report_dir,
            env=env,
            expect_c12=False,
        )
        summary["control_10e"] = control
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        candidate = run_phase(
            phase="candidate_c12_tslve_10e",
            cfg=candidate_cfg,
            dataset=dataset,
            phase_root=screen_root / "candidate_c12_tslve_10e",
            report_dir=report_dir,
            env=env,
            expect_c12=True,
        )
        summary["candidate_10e"] = candidate

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
        summary["comparison"] = comparison

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
            and candidate["onnx_export"]["ok"]
        )
        summary["promotion"] = {
            "promote_beyond_10e": bool(promote),
            "rule": "paired 10e: overall mAP50-95 >= control; mAP50 loss <=0.1pp; focus mean >= control; each focus class no worse by >0.1pp; identical deploy complexity; ONNX pass",
        }
        summary["caution"] = (
            "Novelty search found related dynamic scale weighting/curriculum and generic learning-progress balancing; "
            "no source found for this exact within-detector scale learning-velocity rule. This is a novel candidate, not an absolute priority claim."
        )
        (report_dir / "paths.txt").write_text(
            "\n".join(
                [
                    f"control_best_pt={control['local_best_pt']}",
                    f"candidate_best_pt={candidate['local_best_pt']}",
                    "comparison_basis=paired_10e_same_protocol",
                    "test_dev_used=false",
                    "checkpoint_note=runner-local only",
                ]
            ) + "\n",
            encoding="utf-8",
        )
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        (report_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        EXPERIMENT.write_text(original_text, encoding="utf-8")
        if LOCAL.exists():
            LOCAL.unlink()


if __name__ == "__main__":
    main()
