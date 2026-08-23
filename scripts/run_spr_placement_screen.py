#!/usr/bin/env python3
"""Run the exhaustive 5-epoch SPR-Down backbone placement screen.

This is local mechanism screening only. It intentionally keeps standard TAL,
standard loss, reg_max=16, no attention, scratch training, and the same seed /
optimizer / augmentation protocol for every placement. Checkpoints stay on the
runner; only compact reports are copied into the repository report directory.
"""

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
import traceback

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "experiment.yaml"
LOCAL = ROOT / "config" / "local.yaml"

VARIANTS = [
    ("B0", "edge27_place_b0", []),
    ("S1", "edge27_spr_p4p5", ["p4_p5"]),
    ("S2", "edge27_spr_p3p4", ["p3_p4"]),
    ("S3", "edge27_spr_p2p3", ["p2_p3"]),
    ("S4", "edge27_spr_p3p4_p4p5", ["p3_p4", "p4_p5"]),
    ("S5", "edge27_spr_p2p3_p3p4", ["p2_p3", "p3_p4"]),
    ("S6", "edge27_spr_p2p3_p4p5", ["p2_p3", "p4_p5"]),
    ("S7", "edge27_spr_all", ["p2_p3", "p3_p4", "p4_p5"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--screen-root", required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def write_local(dataset: Path, variant_root: Path) -> None:
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
                f'runs_dir: "{variant_root / "runs"}"',
                f'state_dir: "{variant_root / "state"}"',
                f'outputs_dir: "{variant_root / "outputs"}"',
                f'generated_dir: "{variant_root / "generated"}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def set_preset(original: dict, preset: str, args: argparse.Namespace) -> None:
    c = json.loads(json.dumps(original))
    c["preset"] = preset
    c["train"]["epochs"] = int(args.epochs)
    c["train"]["batch"] = int(args.batch)
    c["train"]["nbs"] = int(args.nbs)
    c["train"]["workers"] = int(args.workers)
    c["train"]["save_period"] = 1
    c["train"]["patience"] = 0
    c["pretrained"] = False
    EXPERIMENT.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")


def run_capture(cmd: list[str], env: dict[str, str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
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


def parse_complexity(text: str) -> dict[str, float | int | None]:
    pm = re.search(r"Params:\s*([0-9,]+)", text)
    gm = re.search(r"GFLOPs:\s*([0-9.]+)", text)
    return {
        "params": int(pm.group(1).replace(",", "")) if pm else None,
        "gflops": float(gm.group(1)) if gm else None,
    }


def focus(eval_data: dict) -> dict:
    out = {}
    for cls in ("pedestrian", "people"):
        if cls in eval_data.get("per_class", {}):
            out[cls] = eval_data["per_class"][cls]
    return out


def write_summary(report_dir: Path, summary: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    screen_root = Path(args.screen_root).expanduser().resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    screen_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    original_text = EXPERIMENT.read_text(encoding="utf-8")
    original = yaml.safe_load(original_text)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'third_party' / 'ultralytics'}:{ROOT}"

    summary = {
        "purpose": "Exhaustive local 5e SPR-Down backbone placement screening; not final paper evidence.",
        "protocol": {
            "epochs": args.epochs,
            "batch": args.batch,
            "nbs": args.nbs,
            "workers": args.workers,
            "imgsz": int(original["train"]["imgsz"]),
            "optimizer": original["train"]["optimizer"],
            "lr0": float(original["train"]["lr0"]),
            "lrf": float(original["train"]["lrf"]),
            "weight_decay": float(original["train"]["weight_decay"]),
            "seed": int(original["seed"]),
            "pretrained": False,
            "assigner": "standard TAL",
            "loss": "standard",
            "reg_max": 16,
            "attention": "none",
        },
        "placement_space": {
            "p2_p3": "backbone layer 3, P2->P3",
            "p3_p4": "backbone layer 5, P3->P4",
            "p4_p5": "backbone layer 7, P4->P5",
            "coverage": "all 2^3 Conv/SPR combinations",
        },
        "variants": {},
        "ranking_best_eval_map50_95": [],
        "ranking_epoch5_map50_95": [],
        "caution": "deterministic=false and only 5 epochs; use solely to shortlist placements for 50e confirmation.",
    }

    failures = 0
    try:
        for variant_id, preset, placements in VARIANTS:
            print("\n" + "#" * 100)
            print(f"PLACEMENT {variant_id}: {preset} | {placements or ['all Conv']}")
            print("#" * 100)

            variant_root = screen_root / f"{variant_id.lower()}_{preset}"
            variant_report = report_dir / variant_id.lower()
            shutil.rmtree(variant_root, ignore_errors=True)
            shutil.rmtree(variant_report, ignore_errors=True)
            for p in (
                variant_root / "runs",
                variant_root / "state",
                variant_root / "outputs",
                variant_root / "generated",
                variant_report,
            ):
                p.mkdir(parents=True, exist_ok=True)

            record = {
                "id": variant_id,
                "preset": preset,
                "spr_placements": placements,
                "status": "running",
            }
            summary["variants"][variant_id] = record
            write_summary(report_dir, summary)

            try:
                write_local(dataset, variant_root)
                set_preset(original, preset, args)

                sanity_text = run_capture([sys.executable, "scripts/sanity.py"], env)
                (variant_report / "sanity.txt").write_text(sanity_text, encoding="utf-8")
                record["complexity"] = parse_complexity(sanity_text)

                run_live([sys.executable, "scripts/train.py"], env)

                best = newest(variant_root, "runs/*/weights/best.pt")
                results = newest(variant_root, "runs/*/results.csv")
                train_args = newest(variant_root, "runs/*/args.yaml")

                eval_path = variant_report / "best_eval.json"
                run_live(
                    [
                        sys.executable,
                        "scripts/eval_screening.py",
                        "--weights",
                        str(best),
                        "--output",
                        str(eval_path),
                    ],
                    env,
                )

                shutil.copy2(results, variant_report / "results.csv")
                shutil.copy2(train_args, variant_report / "args.yaml")
                eval_data = json.loads(eval_path.read_text(encoding="utf-8"))

                record.update(
                    {
                        "status": "complete",
                        "epoch5": parse_epoch_row(results, args.epochs),
                        "best_eval": eval_data.get("aggregate", {}),
                        "focus_best_eval": focus(eval_data),
                        "speed_ms_best_eval": eval_data.get("speed_ms", {}),
                        "local_best_pt": str(best),
                        "local_results_csv": str(results),
                    }
                )

                (variant_report / "paths.txt").write_text(
                    "\n".join(
                        [
                            f"id={variant_id}",
                            f"preset={preset}",
                            f"spr_placements={','.join(placements) if placements else 'none'}",
                            f"root={variant_root}",
                            f"best_pt={best}",
                            f"results_csv={results}",
                            "note=local 5e placement screening; checkpoints remain runner-local",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                failures += 1
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                (variant_report / "failure.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                print(traceback.format_exc())

            write_summary(report_dir, summary)

        complete = [v for v in summary["variants"].values() if v.get("status") == "complete"]
        summary["ranking_best_eval_map50_95"] = [
            v["id"]
            for v in sorted(
                complete,
                key=lambda x: float(x["best_eval"]["map50_95"]),
                reverse=True,
            )
        ]
        summary["ranking_epoch5_map50_95"] = [
            v["id"]
            for v in sorted(
                complete,
                key=lambda x: float(x["epoch5"]["map50_95"]),
                reverse=True,
            )
        ]

        if "B0" in summary["variants"] and summary["variants"]["B0"].get("status") == "complete":
            b0 = summary["variants"]["B0"]
            for v in complete:
                v["delta_best_eval_vs_b0"] = {
                    k: float(v["best_eval"][k]) - float(b0["best_eval"][k])
                    for k in ("precision", "recall", "map50", "map50_95")
                }
                v["delta_epoch5_vs_b0"] = {
                    k: float(v["epoch5"][k]) - float(b0["epoch5"][k])
                    for k in ("precision", "recall", "map50", "map50_95")
                }

        summary["completed_variants"] = len(complete)
        summary["failed_variants"] = failures
        write_summary(report_dir, summary)
        print(json.dumps(summary, indent=2))
    finally:
        EXPERIMENT.write_text(original_text, encoding="utf-8")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
