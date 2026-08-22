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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--c1", type=int, default=128)
    p.add_argument("--c2", type=int, default=256)
    p.add_argument("--h", type=int, default=40)
    p.add_argument("--w", type=int, default=40)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def theoretical_macs(c1, c2, h, w):
    h2 = (h + 1) // 2
    w2 = (w + 1) // 2

    standard = h2 * w2 * 9 * c1 * c2
    spr_conv = h2 * w2 * (9 * c1 + c1 * c2)

    return standard, spr_conv


def bench(m, x, iters):
    m.eval()
    device = x.device.type

    with torch.no_grad():
        for _ in range(20):
            _ = m(x)

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = m(x)

    if device == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    args = parse_args()
    device = torch.device(args.device)

    spr = SPRDown(args.c1, args.c2).to(device)
    conv = nn.Sequential(
        nn.Conv2d(
            args.c1,
            args.c2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(args.c2),
        nn.SiLU(inplace=True),
    ).to(device)

    x = torch.randn(
        1,
        args.c1,
        args.h,
        args.w,
        device=device,
    )

    with torch.no_grad():
        y = spr(x)

    expected_h = (args.h + 1) // 2
    expected_w = (args.w + 1) // 2
    assert y.shape == (
        1,
        args.c2,
        expected_h,
        expected_w,
    )

    standard_macs, spr_macs = theoretical_macs(
        args.c1,
        args.c2,
        args.h,
        args.w,
    )

    print("SPR-DOWN V1 SMOKE PASSED")
    print("input:", tuple(x.shape))
    print("output:", tuple(y.shape))
    print()
    print("Params")
    print(f"  Conv3x3-s2 : {count_params(conv):,}")
    print(f"  SPR-Down   : {count_params(spr):,}")
    print()
    print("Conv MACs only (phase scorer/pooling excluded)")
    print(f"  Conv3x3-s2 : {standard_macs / 1e6:.4f} M")
    print(f"  SPR-Down   : {spr_macs / 1e6:.4f} M")
    print(f"  ratio      : {spr_macs / standard_macs:.4f}")

    if args.iters > 0:
        conv_ms = bench(conv, x, args.iters)
        spr_ms = bench(spr, x, args.iters)
        print()
        print(f"Latency on {device}")
        print(f"  Conv3x3-s2 : {conv_ms:.4f} ms")
        print(f"  SPR-Down   : {spr_ms:.4f} ms")
        print(f"  ratio      : {spr_ms / conv_ms:.4f}")
        print(
            "NOTE: runtime latency includes phase split/reassembly/softmax; "
            "theoretical Conv MACs do not."
        )


if __name__ == "__main__":
    main()
