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


def axis_count(lo: float, hi: float, stride: int, imgsz: int, eps: float = 1e-9) -> int:
    """Count grid centers strictly inside (lo, hi), matching TAL's eps test."""
    # Centers are s/2 + n*s. TAL uses bbox_deltas.amin(...).gt_(eps),
    # so centers on the boundary are not valid.
    first = math.floor((lo + eps - stride / 2) / stride) + 1
    last = math.ceil((hi - eps - stride / 2) / stride) - 1
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
    smallest_stride = strides[0]
    # Ultralytics v8.4.56 TaskAlignedAssigner uses stride[1] as stride_val
    # when multiple strides are present.
    stride_val = strides[1] if len(strides) > 1 else strides[0]
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
    expanded_any = 0
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
            raw_w = bw * w0 * r
            raw_h = bh * h0 * r

            # Match Ultralytics v8.4.56 select_candidates_in_gts exactly:
            # each GT dimension smaller than stride[0] is replaced by stride_val.
            cand_w = float(stride_val) if raw_w < smallest_stride else raw_w
            cand_h = float(stride_val) if raw_h < smallest_stride else raw_h
            if cand_w != raw_w or cand_h != raw_h:
                expanded_any += 1

            x1 = cx - cand_w * 0.5
            y1 = cy - cand_h * 0.5
            x2 = cx + cand_w * 0.5
            y2 = cy + cand_h * 0.5

            count = 0
            for s in strides:
                count += axis_count(x1, x2, s, imgsz) * axis_count(y1, y2, s, imgsz)

            total += 1
            all_counts.append(count)

            # TCR's tiny decision uses the original GT dimensions, not the
            # internally expanded candidate region.
            is_tiny = min(raw_w, raw_h) <= tiny_px
            if is_tiny:
                tiny_total += 1
                tiny_counts.append(count)
                if count < min_candidates:
                    tiny_starved += 1
                if count == 0:
                    tiny_zero += 1

                if cls in {5, 6}:
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
        "diagnostic": "ultralytics_v8.4.56_TAL_exact_candidate_region",
        "imgsz": imgsz,
        "strides": strides,
        "smallest_stride": smallest_stride,
        "stride_val": stride_val,
        "tiny_min_side_px": tiny_px,
        "min_candidates": min_candidates,
        "objects": total,
        "objects_with_builtin_tiny_expansion": expanded_any,
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
            "Counts reproduce Ultralytics v8.4.56 built-in tiny-GT expansion before "
            "measuring whether TCR would add candidates."
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
