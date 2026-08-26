#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "experiment.yaml"
LOCAL = ROOT / "config" / "local.yaml"
CONTROL_SUMMARY = ROOT / "reports" / "yoloedge27" / "stage3" / "head_snr_v1_5e" / "summary.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-deploy-gflops-ratio", type=float, default=1.03)
    p.add_argument("--max-train-params-ratio", type=float, default=1.20)
    return p.parse_args()


def run_capture(cmd: list[str], env: dict[str, str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc.stdout


def run_live(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def newest(root: Path, pattern: str) -> Path:
    xs = list(root.glob(pattern))
    if not xs:
        raise FileNotFoundError(f"No file matching {pattern} under {root}")
    return max(xs, key=lambda p: p.stat().st_mtime)


def parse_epoch_row(path: Path, epoch_number: int) -> dict[str, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) < epoch_number:
        raise RuntimeError(f"Need epoch {epoch_number} in {path}, got {len(rows)} rows")
    row = rows[epoch_number - 1]

    def metric(*names: str) -> float:
        for name in names:
            value = row.get(name)
            if value not in (None, ""):
                return float(value)
        raise KeyError(names)

    return {
        "precision": metric("metrics/precision(B)", "metrics/precision"),
        "recall": metric("metrics/recall(B)", "metrics/recall"),
        "map50": metric("metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": metric("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def parse_marker(text: str, prefix: str) -> dict:
    matches = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    if not matches:
        raise RuntimeError(f"Missing marker {prefix!r} in subprocess output")
    return json.loads(matches[-1])


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(screen_root, ignore_errors=True)
    for p in (
        screen_root / "runs",
        screen_root / "state",
        screen_root / "outputs",
        screen_root / "generated",
        screen_root / "deploy",
    ):
        p.mkdir(parents=True, exist_ok=True)

    if not CONTROL_SUMMARY.exists():
        raise FileNotFoundError(f"Existing H0/H1 controls not found: {CONTROL_SUMMARY}")
    prior = json.loads(CONTROL_SUMMARY.read_text(encoding="utf-8"))
    controls = {k: prior["variants"][k] for k in ("H0", "H1")}
    h1 = controls["H1"]

    original_text = EXPERIMENT.read_text(encoding="utf-8")
    cfg = yaml.safe_load(original_text)
    cfg["preset"] = "edge27_neck_rep_v1"
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["batch"] = int(args.batch)
    cfg["train"]["nbs"] = int(args.nbs)
    cfg["train"]["workers"] = int(args.workers)
    cfg["train"]["save_period"] = 1
    cfg["train"]["patience"] = 0
    cfg["pretrained"] = False

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
                f'runs_dir: "{screen_root / "runs"}"',
                f'state_dir: "{screen_root / "state"}"',
                f'outputs_dir: "{screen_root / "outputs"}"',
                f'generated_dir: "{screen_root / "generated"}"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'third_party' / 'ultralytics'}:{ROOT}"

    try:
        EXPERIMENT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        sanity = run_capture([sys.executable, "scripts/sanity.py"], env)
        (report_dir / "sanity.txt").write_text(sanity, encoding="utf-8")
        if "Neck mode: rep" not in sanity or "RepC3k2 count: 5" not in sanity:
            raise RuntimeError("Sanity output did not confirm five RepC3k2 neck blocks")
        if "Assigner: TaskAlignedAssigner" not in sanity:
            raise RuntimeError("N1 must keep stock TAL assignment")

        # Strong pre-flight: profile both training and analytically fused graphs,
        # verify exact module counts and full-model numerical equivalence before
        # spending GPU time on the 5e screen.
        probe_code = r'''
import json
import torch
from src.runtime import prepare_runtime
cfg, _, model_yaml = prepare_runtime()
from ultralytics import YOLO
from ultralytics.nn.modules.visdrone_rep_neck import switch_reparameterized_neck_to_deploy
from thop import profile

def flatten(obj):
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(flatten(x))
        return out
    if isinstance(obj, dict):
        out = []
        for k in sorted(obj):
            out.extend(flatten(obj[k]))
        return out
    return []

def diff(a, b):
    aa, bb = flatten(a), flatten(b)
    assert len(aa) == len(bb) and len(aa) > 0
    out = 0.0
    for x, y in zip(aa, bb):
        assert x.shape == y.shape
        out = max(out, float((x.float() - y.float()).abs().max().item()))
    return out

m = YOLO(str(model_yaml)).model.float().cpu().eval()
criterion = m.init_criterion()
assert criterion.assigner.__class__.__name__ == 'TaskAlignedAssigner'
assert m.model[-1].__class__.__name__ == 'Detect'
assert int(m.model[-1].reg_max) == 1
rep_count = sum(x.__class__.__name__ == 'RepC3k2' for x in m.modules())
assert rep_count == 5
train_params = sum(p.numel() for p in m.parameters())
x = torch.randn(1, 3, int(cfg['train']['imgsz']), int(cfg['train']['imgsz']))
with torch.no_grad():
    y0 = m(x)
    train_macs, _ = profile(m, inputs=(x,), verbose=False)

m2 = YOLO(str(model_yaml)).model.float().cpu().eval()
m2.load_state_dict(m.state_dict(), strict=True)
switch_reparameterized_neck_to_deploy(m2)
flags = [bool(getattr(z, 'deploy', False)) for z in m2.modules() if z.__class__.__name__ == 'RepC3k2']
assert len(flags) == 5 and all(flags)
with torch.no_grad():
    y1 = m2(x)
    deploy_macs, _ = profile(m2, inputs=(x,), verbose=False)
max_diff = diff(y0, y1)
assert max_diff < 2e-4, max_diff
payload = {
    'rep_blocks': rep_count,
    'train_params': train_params,
    'deploy_params': sum(p.numel() for p in m2.parameters()),
    'train_gflops': train_macs * 2 / 1e9,
    'deploy_gflops': deploy_macs * 2 / 1e9,
    'max_abs_diff': max_diff,
    'assigner': criterion.assigner.__class__.__name__,
    'head': m.model[-1].__class__.__name__,
    'reg_max': int(m.model[-1].reg_max),
}
print('N1_PROBE_JSON=' + json.dumps(payload, sort_keys=True))
'''
        probe_text = run_capture([sys.executable, "-c", probe_code], env)
        probe = parse_marker(probe_text, "N1_PROBE_JSON=")
        (report_dir / "deploy_preflight.json").write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")

        h1_params = int(h1["complexity"]["params"])
        h1_gflops = float(h1["complexity"]["gflops"])
        if int(probe["train_params"]) > h1_params * float(args.max_train_params_ratio):
            raise RuntimeError(
                f"N1 training params exceed budget: {probe['train_params']} vs H1 {h1_params} "
                f"(limit {args.max_train_params_ratio:.3f}x)"
            )
        if float(probe["deploy_gflops"]) > h1_gflops * float(args.max_deploy_gflops_ratio):
            raise RuntimeError(
                f"N1 deployed GFLOPs exceed budget: {probe['deploy_gflops']:.4f} vs H1 {h1_gflops:.4f} "
                f"(limit {args.max_deploy_gflops_ratio:.3f}x)"
            )

        run_live([sys.executable, "scripts/train.py"], env)
        best = newest(screen_root, "runs/*/weights/best.pt")
        results = newest(screen_root, "runs/*/results.csv")
        train_args = newest(screen_root, "runs/*/args.yaml")

        deploy_pt = screen_root / "deploy" / "best_deploy.pt"
        fusion_text = run_capture(
            [
                sys.executable,
                "scripts/fuse_rep_neck_checkpoint.py",
                "--weights",
                str(best),
                "--output",
                str(deploy_pt),
                "--imgsz",
                str(int(cfg["train"]["imgsz"])),
                "--tol",
                "0.0005",
            ],
            env,
        )
        fusion = parse_marker(fusion_text, "N1_FUSION_JSON=")
        (report_dir / "fusion_equivalence.json").write_text(json.dumps(fusion, indent=2) + "\n", encoding="utf-8")

        eval_path = report_dir / "rep_neck_v1_5e_eval.json"
        run_live([sys.executable, "scripts/eval_screening.py", "--weights", str(deploy_pt), "--output", str(eval_path)], env)

        onnx_ok = True
        onnx_error = None
        try:
            run_live([sys.executable, "scripts/export.py", "--weights", str(deploy_pt), "--format", "onnx"], env)
        except Exception as exc:
            onnx_ok = False
            onnx_error = f"{type(exc).__name__}: {exc}"

        shutil.copy2(results, report_dir / "results.csv")
        shutil.copy2(train_args, report_dir / "args.yaml")
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))

        n1 = {
            "id": "N1",
            "description": "H1 DFL-free detector + five reparameterized PAN/FPN fusion blocks",
            "status": "complete",
            "complexity": {
                "train_params": int(probe["train_params"]),
                "deploy_params": int(probe["deploy_params"]),
                "train_gflops": float(probe["train_gflops"]),
                "deploy_gflops": float(probe["deploy_gflops"]),
                "h1_deploy_gflops_ratio": float(probe["deploy_gflops"]) / h1_gflops,
            },
            "fusion_preflight_max_abs_diff": float(probe["max_abs_diff"]),
            "fusion_checkpoint": fusion,
            "epoch5": parse_epoch_row(results, int(args.epochs)),
            "best_eval": eval_data["aggregate"],
            "focus_best_eval": eval_data.get("focus", {}),
            "speed_ms_best_eval": eval_data.get("speed_ms", {}),
            "onnx_export": {"ok": onnx_ok, "error": onnx_error},
            "local_training_best_pt": str(best),
            "local_deploy_best_pt": str(deploy_pt),
        }
        for control_id, control in controls.items():
            n1[f"delta_best_eval_vs_{control_id.lower()}"] = {
                key: float(n1["best_eval"][key]) - float(control["best_eval"][key])
                for key in ("precision", "recall", "map50", "map50_95")
            }
            n1[f"delta_focus_vs_{control_id.lower()}"] = {
                cls: {
                    key: float(n1["focus_best_eval"][cls][key]) - float(control["focus_best_eval"][cls][key])
                    for key in ("precision", "recall", "map50", "map50_95")
                }
                for cls in ("pedestrian", "people")
                if cls in n1["focus_best_eval"] and cls in control["focus_best_eval"]
            }

        summary = {
            "purpose": "N1 reparameterized neck local 5e mechanism screen; not final paper evidence.",
            "novelty_status": "established structural reparameterization control; no novelty claim for N1",
            "mechanism": {
                "name": "N1 Reparameterized Fusion Neck",
                "base": "H1: S1 SPR P4->P5 + direct reg_max=1",
                "changed_scope": "all five PAN/FPN C3k2 fusion blocks only",
                "training_branch": "3x3+BN, 1x1+BN and identity+BN inside first conv of each residual bottleneck",
                "deployment_branch": "analytically fused single 3x3 conv at each reparameterized location",
                "attention": "none",
                "assignment": "stock TaskAlignedAssigner",
                "loss": "standard BCE + CIoU direct-regression path",
                "inference_graph_extra_parallel_branches": 0,
            },
            "protocol": {
                "epochs": int(args.epochs),
                "batch": int(args.batch),
                "nbs": int(args.nbs),
                "imgsz": int(cfg["train"]["imgsz"]),
                "seed": int(cfg["seed"]),
                "pretrained": False,
                "max_deploy_gflops_ratio_vs_h1": float(args.max_deploy_gflops_ratio),
                "max_train_params_ratio_vs_h1": float(args.max_train_params_ratio),
            },
            "controls": controls,
            "N1": n1,
            "decision_rule": (
                "Promote only if deployed N1 is no worse than H1 on mAP50-95, mAP50 is not lower by more than 0.1 pp, "
                "pedestrian/people are not materially damaged, and deployed GFLOPs remain within 1.03x H1."
            ),
            "caution": "single local seed, deterministic=false, 5 epochs only; longer controlled runs require explicit approval.",
        }
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (report_dir / "paths.txt").write_text(
            f"training_best_pt={best}\ndeploy_best_pt={deploy_pt}\nresults_csv={results}\ncontrol_summary={CONTROL_SUMMARY}\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
    finally:
        EXPERIMENT.write_text(original_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
