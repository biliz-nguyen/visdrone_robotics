#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tol", type=float, default=2e-4)
    return p.parse_args()


def _flatten_tensors(obj):
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(_flatten_tensors(item))
        return out
    if isinstance(obj, dict):
        out = []
        for key in sorted(obj):
            out.extend(_flatten_tensors(obj[key]))
        return out
    return []


def _max_output_diff(a, b) -> float:
    ta = _flatten_tensors(a)
    tb = _flatten_tensors(b)
    if len(ta) != len(tb) or not ta:
        raise RuntimeError(f"Prediction tensor structure mismatch: {len(ta)} vs {len(tb)}")
    diff = 0.0
    for xa, xb in zip(ta, tb):
        if xa.shape != xb.shape:
            raise RuntimeError(f"Prediction shape mismatch: {xa.shape} vs {xb.shape}")
        diff = max(diff, float((xa.float() - xb.float()).abs().max().item()))
    return diff


def main() -> int:
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Runtime preparation guarantees the custom class exists at the exact
    # module path recorded inside Ultralytics checkpoints before torch.load.
    from src.runtime import prepare_runtime

    cfg, _, _ = prepare_runtime()
    from ultralytics import YOLO
    from ultralytics.nn.modules.visdrone_rep_neck import switch_reparameterized_neck_to_deploy

    before = YOLO(str(weights)).model.float().cpu().eval()
    after = YOLO(str(weights)).model.float().cpu().eval()
    rep_before = sum(m.__class__.__name__ == "RepC3k2" for m in before.modules())
    if rep_before != 5:
        raise RuntimeError(f"Expected 5 RepC3k2 blocks before fusion, got {rep_before}")

    switch_reparameterized_neck_to_deploy(after)
    deploy_flags = [bool(getattr(m, "deploy", False)) for m in after.modules() if m.__class__.__name__ == "RepC3k2"]
    if len(deploy_flags) != 5 or not all(deploy_flags):
        raise RuntimeError(f"RepC3k2 deploy conversion incomplete: {deploy_flags}")

    torch.manual_seed(int(cfg.get("seed", 42)))
    x = torch.randn(1, 3, int(args.imgsz), int(args.imgsz))
    with torch.no_grad():
        y0 = before(x)
        y1 = after(x)
    max_diff = _max_output_diff(y0, y1)
    if max_diff > float(args.tol):
        raise RuntimeError(f"Rep-neck fusion changed outputs: max_abs_diff={max_diff} > {args.tol}")

    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    converted = []
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected Ultralytics checkpoint dict, got {type(ckpt)}")
    for key in ("model", "ema"):
        obj = ckpt.get(key)
        if isinstance(obj, torch.nn.Module):
            obj.float().cpu().eval()
            switch_reparameterized_neck_to_deploy(obj)
            converted.append(key)
    if not converted:
        raise RuntimeError("Checkpoint has no model/ema module to reparameterize")
    torch.save(ckpt, str(output))

    # Reload the emitted checkpoint to catch pickling/module-path errors now,
    # before evaluation/export consumes GPU time.
    reloaded = YOLO(str(output)).model.float().cpu().eval()
    flags = [bool(getattr(m, "deploy", False)) for m in reloaded.modules() if m.__class__.__name__ == "RepC3k2"]
    if len(flags) != 5 or not all(flags):
        raise RuntimeError(f"Reloaded deploy checkpoint is not fully fused: {flags}")
    with torch.no_grad():
        y2 = reloaded(x)
    reload_diff = _max_output_diff(y1, y2)
    if reload_diff > float(args.tol):
        raise RuntimeError(f"Reloaded deploy checkpoint changed outputs: {reload_diff}")

    payload = {
        "input_weights": str(weights),
        "output_weights": str(output),
        "converted_keys": converted,
        "rep_blocks": len(flags),
        "max_abs_diff_pre_post_fusion": max_diff,
        "max_abs_diff_after_reload": reload_diff,
        "tolerance": float(args.tol),
    }
    print("N1_FUSION_JSON=" + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
