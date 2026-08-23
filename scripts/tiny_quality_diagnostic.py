#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--output", default="")
    return p.parse_args()


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main():
    args = parse_args()
    cfg = load_config()
    q = cfg["tiny_quality_assigner"]

    imgsz = int(cfg["train"]["imgsz"])
    threshold = float(q["tiny_min_side"])
    beta = 6.0
    floor = float(q["beta_floor"])

    cache = Path(cfg["dataset_root"]) / ".visdrone_yolo" / args.split
    label_dir = cache / "labels"
    image_dir = cache / "images"
    if not label_dir.exists() or not image_dir.exists():
        raise FileNotFoundError(f"Missing converted cache under {cache}")

    all_beta = []
    affected_beta = []
    focus_beta = []
    focus_affected_beta = []
    total = affected = focus_total = focus_affected = 0

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = image_dir / f"{label_path.stem}{ext}"
            if p.exists():
                image_path = p
                break
        if image_path is None:
            continue

        with Image.open(image_path) as im:
            w0, h0 = im.size
        r = min(imgsz / w0, imgsz / h0)

        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            bw = float(parts[3]) * w0 * r
            bh = float(parts[4]) * h0 * r
            min_side = min(bw, bh)
            ratio = min(max(min_side / threshold, 0.0), 1.0)
            beta_eff = floor + (beta - floor) * ratio

            total += 1
            all_beta.append(beta_eff)
            if min_side < threshold:
                affected += 1
                affected_beta.append(beta_eff)

            if cls in {5, 6}:
                focus_total += 1
                focus_beta.append(beta_eff)
                if min_side < threshold:
                    focus_affected += 1
                    focus_affected_beta.append(beta_eff)

    report = {
        "split": args.split,
        "imgsz": imgsz,
        "base_beta": beta,
        "beta_floor": floor,
        "tiny_min_side_px": threshold,
        "schedule": "beta_floor + (base_beta-beta_floor)*clamp(min_side/tiny_min_side,0,1)",
        "objects": total,
        "affected_objects": affected,
        "affected_fraction": float(affected / total) if total else 0.0,
        "beta_all": stats(all_beta),
        "beta_affected": stats(affected_beta),
        "pedestrian_people": {
            "objects": focus_total,
            "affected_objects": focus_affected,
            "affected_fraction": float(focus_affected / focus_total) if focus_total else 0.0,
            "beta_all": stats(focus_beta),
            "beta_affected": stats(focus_affected_beta),
        },
        "interpretation": (
            "TAQ keeps the standard TAL candidate region and changes only the localization exponent "
            "for GTs below the tiny threshold; non-tiny GTs remain beta=6 exactly."
        ),
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
