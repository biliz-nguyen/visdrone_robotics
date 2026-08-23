#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.custom_blocks import SPRDown
from src.sprdown_v2 import SPRDownV2


CASES = [
    {"name": "P4_to_P5", "c1": 128, "c2": 256, "h": 40, "w": 40},
    {"name": "P3_to_P4", "c1": 64, "c2": 128, "h": 80, "w": 80},
    {"name": "P2_to_P3", "c1": 32, "c2": 64, "h": 160, "w": 160},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output-json", type=Path)
    p.add_argument("--output-md", type=Path)
    return p.parse_args()


def make_conv(c1: int, c2: int):
    return nn.Sequential(
        nn.Conv2d(c1, c2, 3, 2, 1, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True),
    )


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _bench_cuda(m: nn.Module, x: torch.Tensor, warmup: int, iters: int, repeats: int):
    m.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = m(x)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            start.record()
            for _ in range(iters):
                _ = m(x)
            end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)) / iters)
    return samples


def _bench_cpu(m: nn.Module, x: torch.Tensor, warmup: int, iters: int, repeats: int):
    m.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = m(x)

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            for _ in range(iters):
                _ = m(x)
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        samples.append(dt_ms / iters)
    return samples


def bench(m: nn.Module, x: torch.Tensor, args):
    if x.device.type == "cuda":
        samples = _bench_cuda(m, x, args.warmup, args.iters, args.repeats)
    else:
        samples = _bench_cpu(m, x, args.warmup, args.iters, args.repeats)

    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def run_case(case, device: torch.device, args):
    c1, c2, h, w = case["c1"], case["c2"], case["h"], case["w"]
    x = torch.randn(1, c1, h, w, device=device)

    modules = {
        "conv": make_conv(c1, c2).to(device),
        "spr_v1": SPRDown(c1, c2).to(device),
        "spr_v2": SPRDownV2(c1, c2).to(device),
    }

    expected = (1, c2, (h + 1) // 2, (w + 1) // 2)
    with torch.inference_mode():
        for name, module in modules.items():
            y = module(x)
            if tuple(y.shape) != expected:
                raise RuntimeError(f"{case['name']} {name}: {tuple(y.shape)} != {expected}")
            if not torch.isfinite(y).all():
                raise RuntimeError(f"{case['name']} {name}: non-finite output")

    result = {**case, "expected_output": list(expected), "modules": {}}

    # Rotate the benchmark order between repeats at the case level by using a
    # fixed module order per case. Median across repeated timing windows is the
    # reported value; this is a screening benchmark, not a final Pi benchmark.
    for name, module in modules.items():
        timing = bench(module, x, args)
        timing["params"] = count_params(module)
        result["modules"][name] = timing

    conv_ms = result["modules"]["conv"]["median_ms"]
    v1_ms = result["modules"]["spr_v1"]["median_ms"]
    v2_ms = result["modules"]["spr_v2"]["median_ms"]

    result["ratios"] = {
        "v1_over_conv": v1_ms / conv_ms,
        "v2_over_conv": v2_ms / conv_ms,
        "v2_over_v1": v2_ms / v1_ms,
    }
    return result


def decide(results):
    # The current trained v1 replaced P4->P5, so this is the primary gate.
    primary = next(r for r in results if r["name"] == "P4_to_P5")
    ratio = primary["ratios"]["v2_over_v1"]
    if ratio <= 0.80:
        return {
            "status": "PASS",
            "reason": "SPR-Down v2 is at least 20% faster than v1 at the current P4->P5 site.",
            "primary_v2_over_v1": ratio,
        }
    if ratio < 1.0:
        return {
            "status": "BORDERLINE",
            "reason": "SPR-Down v2 is faster than v1, but the gain is below the 20% material-speedup gate.",
            "primary_v2_over_v1": ratio,
        }
    return {
        "status": "FAIL",
        "reason": "SPR-Down v2 does not reduce v1 latency at the current P4->P5 site.",
        "primary_v2_over_v1": ratio,
    }


def to_markdown(report):
    lines = [
        f"# SPR-Down v2 latency gate — {report['device']}",
        "",
        f"- PyTorch: `{report['torch_version']}`",
        f"- Device: `{report['device_name']}`",
        f"- Warmup/iters/repeats: `{report['warmup']}/{report['iters']}/{report['repeats']}`",
        f"- Verdict: **{report['verdict']['status']}** — {report['verdict']['reason']}",
        "",
        "| Site | Conv ms | SPR-v1 ms | SPR-v2 ms | v1/Conv | v2/Conv | v2/v1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["results"]:
        m = r["modules"]
        q = r["ratios"]
        lines.append(
            f"| {r['name']} | {m['conv']['median_ms']:.4f} | "
            f"{m['spr_v1']['median_ms']:.4f} | {m['spr_v2']['median_ms']:.4f} | "
            f"{q['v1_over_conv']:.3f} | {q['v2_over_conv']:.3f} | {q['v2_over_v1']:.3f} |"
        )
    lines.extend([
        "",
        "> This is a local micro-benchmark for screening. It is not a Raspberry Pi 5 deployment result and must not be reported as final edge latency.",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.iters <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("warmup/iters/repeats must be positive (warmup may be zero)")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
    else:
        torch.set_num_threads(args.threads)
        device = torch.device("cpu")
        device_name = f"CPU ({torch.get_num_threads()} torch threads)"

    results = [run_case(case, device, args) for case in CASES]
    report = {
        "device": args.device,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "torch_threads": torch.get_num_threads(),
        "results": results,
    }
    report["verdict"] = decide(results)

    text = to_markdown(report)
    print(text)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
