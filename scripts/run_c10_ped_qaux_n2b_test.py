#!/usr/bin/env python3
"""Run C10 pedestrian-only quality-aware P2 auxiliary supervision on frozen N2b."""

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
C9_SUMMARY = ROOT / "reports/yoloedge27/stage19/c9_p2_qaux_n2b_v1_5e/summary.json"
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

    assert criterion.__class__.__name__ == "P2QualityAuxDetectionLoss"
    assert criterion.assigner.__class__.__name__ == "TaskAlignedAssigner"
    assert detect.__class__.__name__ == "Detect"
    assert int(detect.reg_max) == 1
    assert [int(x) for x in detect.stride.tolist()] == [4, 8, 16]
    assert list(detect.f) == [19, 22, 25], detect.f
    assert seq[7].__class__.__name__ == "SPRDown"
    assert int(seq[19].cv2.conv.out_channels) == 40
    assert int(seq[22].cv2.conv.out_channels) == 64
    assert int(seq[25].cv2.conv.out_channels) == 104
    assert abs(float(criterion.tiny_min_side) - 16.0) <= 1e-9
    assert abs(float(criterion.aux_weight) - 0.10) <= 1e-9
    assert abs(float(criterion.target_floor) - 0.50) <= 1e-9
    assert abs(float(criterion.quality_gamma) - 0.50) <= 1e-9
    assert tuple(criterion.focus_classes) == (5,), criterion.focus_classes
    assert not any(m.__class__.__name__ in {"P2Refine", "P2ClsDetect", "P2RegDetect"} for m in model.model.modules())

    params = sum(p.numel() for p in model.model.parameters())
    assert params == EXPECTED_PARAMS, params

    # C10 isolation guard: pedestrian is supervised, people is stock-only.
    # Detached localization quality must not create box-head gradients.
    model.model.train()
    model.model.zero_grad(set_to_none=True)
    x_small = torch.randn(1, 3, 64, 64)
    preds = model.model(x_small)
    aux_batch = {
        "batch_idx": torch.tensor([0, 0, 0]),
        "cls": torch.tensor([[5.0], [6.0], [0.0]]),
        "bboxes": torch.tensor(
            [
                [0.30, 0.30, 0.08, 0.08],
                [0.70, 0.70, 0.10, 0.10],
                [0.50, 0.50, 0.08, 0.08],
            ]
        ),
    }
    aux_raw, positive_count, quality_mean, target_mean = criterion._tiny_center_quality_loss(preds, aux_batch)
    assert positive_count == 1, positive_count
    assert torch.isfinite(aux_raw) and float(aux_raw.detach()) > 0.0
    assert 0.0 <= float(quality_mean) <= 1.0
    assert 0.50 <= float(target_mean) <= 1.0
    aux_raw.backward()
    cls_param = next(detect.cv3[0].parameters())
    box_param = next(detect.cv2[0].parameters())
    assert cls_param.grad is not None and bool(cls_param.grad.abs().sum() > 0)
    assert box_param.grad is None or bool(box_param.grad.abs().sum() == 0)

    model.model.zero_grad(set_to_none=True)
    model.model.cpu().eval()
    from thop import profile

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        macs, _ = profile(model.model, inputs=(x,), verbose=False)
    gflops = macs * 2 / 1e9
    assert abs(gflops - EXPECTED_GFLOPS) <= 1e-4, gflops

    text = "\n".join(
        [
            "C10 PED-ONLY P2 QUALITY-AUX N2b SANITY PASSED",
            f"data_yaml={data_yaml}",
            f"model_yaml={model_yaml}",
            f"criterion={criterion.__class__.__name__}",
            f"assigner={criterion.assigner.__class__.__name__}",
            "SPR=P4->P5",
            "neck_nominal=160/256/416",
            "head=stock Detect reg_max=1",
            "aux=P2 tiny-center IoU-tempered BCE",
            "soft_target=0.5+0.5*sqrt(detached_IoU)",
            "focus_classes=pedestrian(5)_only",
            "people_auxiliary=false",
            "tiny_min_side=16px",
            "aux_weight=0.10",
            "aux_gradient=P2 classification only; box-head untouched",
            f"sanity_quality_mean={float(quality_mean):.6f}",
            f"sanity_target_mean={float(target_mean):.6f}",
            f"params={params}",
            f"gflops={gflops:.5f}",
            "deployment_overhead_params=0",
            "deployment_overhead_gflops=0",
            "inference_graph_change=false",
        ]
    ) + "\n"
    print(text)
    (report_dir / "sanity.txt").write_text(text, encoding="utf-8")
    return {"params": params, "gflops": gflops}


def main():
    args = parse_args()
    if args.epochs != 5:
        raise ValueError("C10 screening is locked to exactly 5 epochs")

    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    control = json.loads(CONTROL_EVAL.read_text(encoding="utf-8"))
    c9 = json.loads(C9_SUMMARY.read_text(encoding="utf-8")) if C9_SUMMARY.exists() else None

    original_text = EXPERIMENT.read_text(encoding="utf-8")
    cfg = yaml.safe_load(original_text)
    cfg["preset"] = "edge27_neck_realloc_v2"
    cfg["c3_assigner_mode"] = "standard"
    cfg["c4_loss_mode"] = "standard"
    cfg["c5_p2_refine"] = False
    cfg["c6_p2_cls_refine"] = False
    cfg["c7_p2_reg_refine"] = False
    cfg["c8_aux_mode"] = "standard"
    cfg["c9_aux_mode"] = "quality_center"
    cfg["c9_focus_classes"] = [5]
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

    control_focus_mean = sum(control["focus"][c]["map50_95"] for c in ("pedestrian", "people")) / 2.0
    summary = {
        "purpose": "C10 pedestrian-only quality-aware training-only P2 supervision on frozen N2b; 5e screen only.",
        "mechanism": {
            "name": "Pedestrian-Only P2 Quality-Tempered Auxiliary Supervision (Ped-P2-QTAS)",
            "changed_scope": "training loss only; existing P2 pedestrian classification logits",
            "target": "unique stride-4 center cell for tiny pedestrian targets only",
            "people_auxiliary": False,
            "quality": "detached IoU of current prediction at the same P2 center cell",
            "soft_target": "0.5 + 0.5 * sqrt(IoU_detached)",
            "target_floor": 0.50,
            "quality_gamma": 0.50,
            "tiny_min_side_px": 16.0,
            "focus_classes": ["pedestrian"],
            "aux_weight": 0.10,
            "aux_loss": "BCEWithLogits to quality-aware soft target",
            "regression_aux_gradient": False,
            "main_assigner": "stock TAL exact",
            "main_loss": "stock box/cls/direct-reg1 loss exact plus pedestrian-only auxiliary cls term",
            "inference_graph_change": False,
            "deployment_overhead": "0 params / 0 GFLOPs / 0 nodes",
        },
        "protocol": {
            "epochs": 5,
            "batch": args.batch,
            "nbs": args.nbs,
            "workers": args.workers,
            "imgsz": 640,
            "seed": 42,
            "pretrained": False,
            "architecture": "frozen C1 SPR + C2 N2b + stock direct-reg1 Detect",
            "test_dev_used": False,
        },
        "control": {
            "id": "N2b-5e-stock-loss",
            "best_eval": control["aggregate"],
            "focus_best_eval": control["focus"],
            "focus_map50_95_mean": control_focus_mean,
            "complexity": {"params": EXPECTED_PARAMS, "gflops": EXPECTED_GFLOPS},
        },
        "c9_reference": c9.get("candidate") if c9 else None,
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
        eval_path = report_dir / "c10_ped_qaux_n2b_v1_eval.json"
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
            "same_params": complexity["params"] == EXPECTED_PARAMS,
            "same_gflops": abs(complexity["gflops"] - EXPECTED_GFLOPS) <= 1e-4,
        }
        if c9 and c9.get("candidate", {}).get("status") == "complete":
            c9_candidate = c9["candidate"]
            summary["comparison_vs_c9"] = {
                "delta_candidate_minus_c9": delta_dict(candidate["best_eval"], c9_candidate["best_eval"]),
                "delta_focus_candidate_minus_c9": {
                    cls: delta_dict(candidate["focus_best_eval"][cls], c9_candidate["focus_best_eval"][cls])
                    for cls in ("pedestrian", "people")
                },
                "delta_focus_map50_95_mean": focus_mean - float(c9_candidate["focus_map50_95_mean"]),
            }

        d = summary["comparison"]["delta_candidate_minus_n2b"]
        df = summary["comparison"]["delta_focus_candidate_minus_n2b"]
        promote = (
            d["map50_95"] >= 0.0
            and d["map50"] >= -0.001
            and summary["comparison"]["delta_focus_map50_95_mean"] >= 0.0
            and df["pedestrian"]["map50_95"] >= -0.001
            and df["people"]["map50_95"] >= -0.001
            and summary["comparison"]["same_params"]
            and summary["comparison"]["same_gflops"]
            and onnx["ok"]
        )
        summary["promotion"] = {
            "promote_to_50e": bool(promote),
            "rule": "overall mAP50-95 >= N2b; mAP50 loss <=0.1pp; focus mean >= N2b; each focus class no worse by >0.1pp; identical deploy complexity; ONNX pass",
        }
        summary["caution"] = "single local seed, deterministic=false, 5 epochs only; C10 changes only C9 focus_classes from pedestrian+people to pedestrian-only."
        (report_dir / "paths.txt").write_text(
            f"best_pt={best}\ncontrol_eval={CONTROL_EVAL}\nc9_summary={C9_SUMMARY}\ncheckpoint_note=runner-local only\n",
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
