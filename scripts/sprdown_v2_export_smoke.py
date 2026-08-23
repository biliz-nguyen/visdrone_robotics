#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import prepare_runtime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg, _, model_yaml = prepare_runtime()

    from ultralytics import YOLO

    if args.weights:
        source = Path(args.weights).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        model = YOLO(str(source))
    else:
        model = YOLO(str(model_yaml))

    exported = Path(
        model.export(
            format="onnx",
            imgsz=int(cfg["export"]["imgsz"]),
            batch=1,
            device="cpu",
            dynamic=False,
            simplify=False,
            opset=int(cfg["export"]["onnx_opset"]),
        )
    )

    import onnx
    checked = onnx.load(str(exported))
    onnx.checker.check_model(checked)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / exported.name
    if exported.resolve() != target.resolve():
        target.write_bytes(exported.read_bytes())

    print("ONNX_EXPORT_OK", target)
    print("nodes", len(checked.graph.node))
    print("inputs", [x.name for x in checked.graph.input])
    print("outputs", [x.name for x in checked.graph.output])


if __name__ == "__main__":
    main()
