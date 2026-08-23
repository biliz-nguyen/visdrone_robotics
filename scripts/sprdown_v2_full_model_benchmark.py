#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_builder import build_model_yaml
from src.runtime import prepare_runtime


VARIANTS = [
    ("baseline_conv", "conv"),
    ("spr_v1", "sprdown"),
    ("spr_v2", "sprdown_v2"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def bench(model, x, device, warmup, iters, repeats):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.inference_mode():
                start.record()
                for _ in range(iters):
                    _ = model(x)
                end.record()
            torch.cuda.synchronize()
            samples.append(float(start.elapsed_time(end)) / iters)
        else:
            t0 = time.perf_counter_ns()
            with torch.inference_mode():
                for _ in range(iters):
                    _ = model(x)
            samples.append((time.perf_counter_ns() - t0) / 1e6 / iters)

    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main():
    args = parse_args()
    cfg, _, _ = prepare_runtime()

    from ultralytics import YOLO

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
    else:
        torch.set_num_threads(args.threads)
        device = torch.device("cpu")
        device_name = f"CPU ({torch.get_num_threads()} torch threads)"

    results = {}
    for label, downsample in VARIANTS:
        variant_cfg = deepcopy(cfg)
        variant_cfg["backbone_down"] = downsample
        variant_cfg["experiment_tag"] = f"fullbench_{label}"
        model_yaml = build_model_yaml(variant_cfg)

        yolo = YOLO(str(model_yaml))
        model = yolo.model.to(device).eval()
        params = sum(p.numel() for p in model.parameters())
        x = torch.randn(1, 3, int(cfg["train"]["imgsz"]), int(cfg["train"]["imgsz"]), device=device)

        with torch.inference_mode():
            _ = model(x)

        timing = bench(model, x, device, args.warmup, args.iters, args.repeats)
        timing["params"] = params
        timing["model_yaml"] = str(model_yaml)
        results[label] = timing

        del x, model, yolo
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base = results["baseline_conv"]["median_ms"]
    v1 = results["spr_v1"]["median_ms"]
    v2 = results["spr_v2"]["median_ms"]

    report = {
        "device": args.device,
        "device_name": device_name,
        "torch": torch.__version__,
        "imgsz": int(cfg["train"]["imgsz"]),
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "results": results,
        "ratios": {
            "v1_over_baseline": v1 / base,
            "v2_over_baseline": v2 / base,
            "v2_over_v1": v2 / v1,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"FULL MODEL BENCHMARK — {device_name}")
    for name, row in results.items():
        print(f"{name:14s} {row['median_ms']:.4f} ms  params={row['params']:,}")
    print("ratios:", report["ratios"])


if __name__ == "__main__":
    main()
