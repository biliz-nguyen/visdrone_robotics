#!/usr/bin/env python3

from pathlib import Path
import argparse
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import (
    prepare_runtime,
    load_state,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--weights",
        default="",
    )

    p.add_argument(
        "--format",
        nargs="+",
        default=["ncnn"],
        choices=["ncnn", "onnx"],
    )

    return p.parse_args()


def main():
    args = parse_args()
    cfg, _, _ = prepare_runtime()

    if args.weights:
        weights = Path(
            args.weights
        ).expanduser().resolve()
    else:
        weights = Path(
            load_state(cfg)["best_pt"]
        )

    if not weights.exists():
        raise FileNotFoundError(weights)

    from ultralytics import YOLO

    model = YOLO(str(weights))

    export_cfg = cfg["export"]

    export_dir = (
        Path(cfg["outputs_dir"])
        / "exports"
        / cfg["experiment_tag"]
    )

    export_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {}

    for fmt in args.format:
        kwargs = {
            "format": fmt,
            "imgsz": int(
                export_cfg["imgsz"]
            ),
            "batch": int(
                export_cfg["batch"]
            ),
            "device": "cpu",
            "half": bool(
                export_cfg["half"]
            ),
            "int8": bool(
                export_cfg["int8"]
            ),
            "dynamic": bool(
                export_cfg["dynamic"]
            ),
        }

        if fmt == "onnx":
            kwargs["simplify"] = bool(
                export_cfg[
                    "onnx_simplify"
                ]
            )
            kwargs["opset"] = int(
                export_cfg[
                    "onnx_opset"
                ]
            )

        exported = Path(
            model.export(**kwargs)
        )

        target = (
            export_dir
            / exported.name
        )

        if exported.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(
                exported,
                target,
            )
        else:
            shutil.copy2(
                exported,
                target,
            )

        results[fmt] = str(target)

        print(
            f"✅ {fmt.upper()}: {target}"
        )

    print()
    print("EXPORT DONE")
    for fmt, path in results.items():
        print(f"{fmt}: {path}")


if __name__ == "__main__":
    main()
