#!/usr/bin/env python3

from pathlib import Path
import gc
import inspect
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import prepare_runtime


def main():
    cfg, data_yaml, model_yaml = prepare_runtime()

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.loss import BboxLoss

    print("=" * 90)
    print("SANITY CHECK")
    print("=" * 90)
    print("Ultralytics:", ultralytics.__version__)
    print("Source:", ultralytics.__file__)
    print("Data YAML:", data_yaml)
    print("Model YAML:", model_yaml)

    expected_repo = Path(cfg["ultra_repo"]).resolve()

    assert (
        Path(ultralytics.__file__)
        .resolve()
        .is_relative_to(expected_repo)
    ), "Wrong Ultralytics source imported"

    bbox_source = inspect.getsource(
        BboxLoss
    )

    if cfg["loss_mode"] == "hybrid_nwd":
        assert "_nwd_similarity" in bbox_source
        assert "loss_nwd" in bbox_source
    else:
        assert "_nwd_similarity" not in bbox_source
        assert "loss_nwd" not in bbox_source

    model = YOLO(str(model_yaml))
    detect = model.model.model[-1]

    reg_max = int(cfg["reg_max"])
    assert int(detect.reg_max) == reg_max

    expected_no = int(
        detect.nc
        + 4 * reg_max
    )
    assert int(detect.no) == expected_no

    strides = [
        int(x)
        for x in detect.stride.tolist()
    ]
    assert strides == [4, 8, 16]

    aconvs = [
        m
        for m in model.model.modules()
        if m.__class__.__name__ == "AConv"
    ]

    sprdowns = [
        m
        for m in model.model.modules()
        if m.__class__.__name__ == "SPRDown"
    ]

    attn_names = {
        "ECA",
        "CoordAtt",
        "ResidualLiteCA",
    }

    attn_modules = [
        m
        for m in model.model.modules()
        if m.__class__.__name__
        in attn_names
    ]

    if cfg["backbone_down"] == "aconv":
        assert len(aconvs) == 1
        assert len(sprdowns) == 0
    elif cfg["backbone_down"] == "sprdown":
        assert len(sprdowns) == 1
        assert len(aconvs) == 0
    else:
        assert len(aconvs) == 0
        assert len(sprdowns) == 0

    if cfg["attention"] == "none":
        assert len(attn_modules) == 0
    else:
        expected_class = {
            "eca": "ECA",
            "ca": "CoordAtt",
            "rlca": "ResidualLiteCA",
        }[
            cfg["attention"]
        ]

        assert len(attn_modules) == 1
        assert (
            attn_modules[0]
            .__class__
            .__name__
        ) == expected_class

    # Q1 Stage-1 contract: SPR-Down must be isolated from legacy blocks.
    if cfg["backbone_down"] == "sprdown":
        assert cfg["loss_mode"] == "standard", (
            "SPR-Down v1 screening must use standard localization loss "
            "to isolate the architectural contribution."
        )
        assert int(cfg["reg_max"]) == 16, (
            "SPR-Down v1 screening must keep reg_max=16."
        )
        assert cfg["attention"] == "none", (
            "SPR-Down v1 screening must not use attention."
        )

    params = sum(
        p.numel()
        for p in model.model.parameters()
    )

    model.model.eval()

    dummy = torch.randn(
        1,
        3,
        int(cfg["train"]["imgsz"]),
        int(cfg["train"]["imgsz"]),
    )

    with torch.no_grad():
        _ = model.model(dummy)

    del dummy

    gflops = None

    try:
        from thop import profile

        model.model.cpu().eval()

        x = torch.randn(
            1,
            3,
            int(cfg["train"]["imgsz"]),
            int(cfg["train"]["imgsz"]),
        )

        with torch.no_grad():
            macs, _ = profile(
                model.model,
                inputs=(x,),
                verbose=False,
            )

        gflops = (
            macs
            * 2
            / 1e9
        )

        del x

    except Exception as e:
        print("THOP skipped:", e)

    print()
    print("=" * 90)
    print("SANITY CHECK PASSED")
    print("=" * 90)
    print("Preset:", cfg["preset"])
    print(
        "Experiment:",
        cfg["experiment_tag"],
    )
    print(
        "Backbone downsample:",
        cfg["backbone_down"],
    )
    print(
        "REAL reg_max:",
        detect.reg_max,
    )
    print(
        "REAL Detect.no:",
        detect.no,
    )
    print("Strides:", strides)
    print("SPRDown count:", len(sprdowns))
    print("AConv count:", len(aconvs))
    print(
        "Attention:",
        (
            attn_modules[0]
            .__class__
            .__name__
            if attn_modules
            else "none"
        ),
    )
    print("Loss:", cfg["loss_mode"])
    print(
        f"Params: {params:,} "
        f"({params/1e6:.4f} M)"
    )

    if gflops is not None:
        print(f"GFLOPs: {gflops:.4f}")

    print("Pretrained: False")
    print("=" * 90)

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
