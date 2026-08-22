#!/usr/bin/env python3

from pathlib import Path
import argparse
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
        "--source",
        default="",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
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

    if args.source:
        source = Path(
            args.source
        ).expanduser().resolve()
    elif cfg.get("test_image"):
        source = Path(
            cfg["test_image"]
        ).expanduser().resolve()
    else:
        test_dir = (
            Path(cfg["dataset_root"])
            / cfg["test_images"]
        )

        candidates = []

        for ext in [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.JPG",
            "*.PNG",
        ]:
            candidates.extend(
                test_dir.glob(ext)
            )

        if not candidates:
            raise FileNotFoundError(
                f"No test image in {test_dir}"
            )

        source = sorted(candidates)[0]

    if not weights.exists():
        raise FileNotFoundError(weights)

    if not source.exists():
        raise FileNotFoundError(source)

    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))

    device = 0 if torch.cuda.is_available() else "cpu"

    results = model.predict(
        source=str(source),
        imgsz=int(
            cfg["train"]["imgsz"]
        ),
        conf=float(args.conf),
        iou=0.70,
        max_det=int(
            cfg["train"]["max_det"]
        ),
        device=device,
        save=True,
        project=str(
            Path(cfg["outputs_dir"])
            / "predict"
        ),
        name=cfg["experiment_tag"],
        exist_ok=True,
        verbose=True,
    )

    print("Source:", source)
    print(
        "Detections:",
        len(results[0].boxes),
    )
    print(
        "Saved under:",
        Path(cfg["outputs_dir"])
        / "predict"
        / cfg["experiment_tag"],
    )


if __name__ == "__main__":
    main()
