#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
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


def axis_count(lo: float, hi: float, stride: int, imgsz: int) -> int:
    """Count grid centers s/2+n*s that lie inside [lo, hi]."""
    first = math.ceil((lo - stride / 2) / stride)
    last = math.floor((hi - stride / 2) / stride)
    first = max(first, 0)
    last = min(last, imgsz // stride - 1)
    return max(0, last - first + 1)


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    args = parse_args()
    cfg = load_config()

    imgsz = int(cfg["train"]["imgsz"])
    strides = [4, 8, 16]
    tiny_px = float(cfg["tiny_assigner"]["tiny_min_side"])
    min_candidates = int(cfg["tiny_assigner"]["min_candidates"])

    cache = Path(cfg["dataset_root"]) / ".visdrone_yolo" / args.split
    label_dir = cache / "labels"
    image_dir = cache / "images"
    if not label_dir.exists() or not image_dir.exists():
        raise FileNotFoundError(
            f"Missing converted cache under {cache}. Run prepare_runtime/sanity first."
        )

    total = 0
    tiny_total = 0
    tiny_starved = 0
    tiny_zero = 0
    focus_tiny_total = 0
    focus_tiny_starved = 0
    focus_tiny_zero = 0
    tiny_counts = []
    focus_tiny_counts = []
    all_counts = []

    labels = sorted(label_dir.glob("*.txt"))
    for label_path in labels:
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = image_dir / f"{label_path.stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        with Image.open(image_path) as im:
            w0, h0 = im.size

        r = min(imgsz / w0, imgsz / h0)
        new_w = w0 * r
        new_h = h0 * r
        pad_x = (imgsz - new_w) * 0.5
        pad_y = (imgsz - new_h) * 0.5

        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])

            cx = xc * w0 * r + pad_x
            cy = yc * h0 * r + pad_y
            box_w = bw * w0 * r
            box_h = bh * h0 * r

            x1 = cx - box_w * 0.5
            y1 = cy - box_h * 0.5
            x2 = cx + box_w * 0.5
            y2 = cy + box_h * 0.5

            count = 0
            for s in strides:
                count += axis_count(x1, x2, s, imgsz) * axis_count(y1, y2, s, imgsz)

            total += 1
            all_counts.append(count)

            is_tiny = min(box_w, box_h) <= tiny_px
            if is_tiny:
                tiny_total += 1
                tiny_counts.append(count)
                if count < min_candidates:
                    tiny_starved += 1
                if count == 0:
                    tiny_zero += 1

                if cls in {5, 6}:  # pedestrian / people in project class order
                    focus_tiny_total += 1
                    focus_tiny_counts.append(count)
                    if count < min_candidates:
                        focus_tiny_starved += 1
                    if count == 0:
                        focus_tiny_zero += 1

    def frac(n, d):
        return float(n / d) if d else 0.0

    report = {
        "split": args.split,
        "imgsz": imgsz,
        "strides": strides,
        "tiny_min_side_px": tiny_px,
        "min_candidates": min_candidates,
        "objects": total,
        "all_candidate_counts": summarize(all_counts),
        "tiny": {
            "objects": tiny_total,
            "fraction_of_all": frac(tiny_total, total),
            "starved_lt_min_candidates": tiny_starved,
            "starved_fraction": frac(tiny_starved, tiny_total),
            "zero_candidate": tiny_zero,
            "zero_fraction": frac(tiny_zero, tiny_total),
            "candidate_counts": summarize(tiny_counts),
        },
        "pedestrian_people_tiny": {
            "objects": focus_tiny_total,
            "starved_lt_min_candidates": focus_tiny_starved,
            "starved_fraction": frac(focus_tiny_starved, focus_tiny_total),
            "zero_candidate": focus_tiny_zero,
            "zero_fraction": frac(focus_tiny_zero, focus_tiny_total),
            "candidate_counts": summarize(focus_tiny_counts),
        },
        "interpretation": (
            "TCR changes only tiny GTs with fewer than min_candidates inside-GT "
            "anchor centers; it recovers the nearest unused centers before standard TAL ranking."
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
