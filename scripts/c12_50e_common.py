from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import yaml

import scripts.run_c12_tslve_n2b_10e as base


def run_phase_50e(
    *,
    phase: str,
    cfg: dict,
    dataset: Path,
    phase_root: Path,
    report_dir: Path,
    env: dict,
    expect_c12: bool,
):
    epochs = int(cfg["train"]["epochs"])
    if epochs != 50:
        raise ValueError(f"C12 50e phase helper requires epochs=50, got {epochs}")

    shutil.rmtree(phase_root, ignore_errors=True)
    for p in (phase_root / "runs", phase_root / "state", phase_root / "outputs", phase_root / "generated"):
        p.mkdir(parents=True, exist_ok=True)

    base.write_local(dataset, phase_root)
    base.EXPERIMENT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    complexity = base.architecture_sanity(expect_c12, report_dir)

    base.run_live([sys.executable, "scripts/train.py"], env)
    best = base.newest(phase_root, "runs/*/weights/best.pt")
    results = base.newest(phase_root, "runs/*/results.csv")
    train_args = base.newest(phase_root, "runs/*/args.yaml")
    eval_path = report_dir / f"{phase}_eval.json"
    base.run_live([sys.executable, "scripts/eval_screening.py", "--weights", str(best), "--output", str(eval_path)], env)

    shutil.copy2(results, report_dir / f"{phase}_results.csv")
    shutil.copy2(train_args, report_dir / f"{phase}_args.yaml")
    ev = json.loads(eval_path.read_text(encoding="utf-8"))
    focus_mean = sum(ev["focus"][c]["map50_95"] for c in ("pedestrian", "people")) / 2.0

    out = {
        "status": "complete",
        "epochs": epochs,
        "complexity": complexity,
        "epoch10": base.parse_epoch_row(results, 10),
        "epoch50": base.parse_epoch_row(results, 50),
        "metric_trajectory": base.parse_metric_trajectory(results),
        "best_eval": ev["aggregate"],
        "focus_best_eval": ev["focus"],
        "focus_map50_95_mean": focus_mean,
        "speed_ms_best_eval": ev.get("speed_ms", {}),
        "local_best_pt": str(best),
    }

    if len(out["metric_trajectory"]) != epochs:
        raise RuntimeError(f"Expected {epochs} metric rows, got {len(out['metric_trajectory'])}")

    if expect_c12:
        dynamics = phase_root / "state" / "c12_scale_dynamics.jsonl"
        if not dynamics.exists():
            raise FileNotFoundError(f"C12 dynamics log missing: {dynamics}")
        shutil.copy2(dynamics, report_dir / "c12_scale_dynamics.jsonl")
        rows = [json.loads(line) for line in dynamics.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != epochs:
            raise RuntimeError(f"Expected {epochs} C12 dynamics rows, got {len(rows)}")
        out["scale_dynamics"] = rows
        out["onnx_export"] = base.try_export(best, env, report_dir)

    return out
