#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.custom_blocks import SPRDown
from src.sprdown_v2 import SPRDownV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--c1", type=int, default=128)
    p.add_argument("--c2", type=int, default=256)
    p.add_argument("--h", type=int, default=40)
    p.add_argument("--w", type=int, default=40)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def bench(m, x, iters):
    m.eval()
    with torch.no_grad():
        for _ in range(30):
            _ = m(x)

    if x.device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = m(x)

    if x.device.type == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    args = parse_args()
    device = torch.device(args.device)

    conv = nn.Sequential(
        nn.Conv2d(args.c1, args.c2, 3, 2, 1, bias=False),
        nn.BatchNorm2d(args.c2),
        nn.SiLU(inplace=True),
    ).to(device)
    v1 = SPRDown(args.c1, args.c2).to(device)
    v2 = SPRDownV2(args.c1, args.c2).to(device)

    x = torch.randn(1, args.c1, args.h, args.w, device=device)

    with torch.no_grad():
        y = v2(x)

    expected = (1, args.c2, (args.h + 1) // 2, (args.w + 1) // 2)
    assert tuple(y.shape) == expected

    print("SPR-DOWN V2 PROTOTYPE SMOKE PASSED")
    print("input:", tuple(x.shape))
    print("output:", tuple(y.shape))
    print()
    print("Params")
    print(f"  Conv3x3-s2 : {count_params(conv):,}")
    print(f"  SPR-Down v1: {count_params(v1):,}")
    print(f"  SPR-Down v2: {count_params(v2):,}")

    if args.iters > 0:
        conv_ms = bench(conv, x, args.iters)
        v1_ms = bench(v1, x, args.iters)
        v2_ms = bench(v2, x, args.iters)
        print()
        print(f"Latency on {device}")
        print(f"  Conv3x3-s2 : {conv_ms:.4f} ms")
        print(f"  SPR-Down v1: {v1_ms:.4f} ms  ({v1_ms / conv_ms:.3f}x conv)")
        print(f"  SPR-Down v2: {v2_ms:.4f} ms  ({v2_ms / conv_ms:.3f}x conv)")
        print(f"  v2 / v1    : {v2_ms / v1_ms:.3f}x")
        print()
        print("Target: v2 should materially reduce v1 runtime overhead before training.")


if __name__ == "__main__":
    main()
