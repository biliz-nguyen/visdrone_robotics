#!/usr/bin/env python3

from pathlib import Path
import gc
import inspect
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import normalize_spr_placements
from src.runtime import prepare_runtime


STAGE_LAYER_INDEX = {
    "p2_p3": 3,
    "p3_p4": 5,
    "p4_p5": 7,
}


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
    assert Path(ultralytics.__file__).resolve().is_relative_to(expected_repo), (
        "Wrong Ultralytics source imported"
    )

    bbox_source = inspect.getsource(BboxLoss)
    if cfg["loss_mode"] == "hybrid_nwd":
        assert "_nwd_similarity" in bbox_source
        assert "loss_nwd" in bbox_source
    else:
        assert "_nwd_similarity" not in bbox_source
        assert "loss_nwd" not in bbox_source

    model = YOLO(str(model_yaml))
    seq = model.model.model
    detect = seq[-1]

    reg_max = int(cfg["reg_max"])
    assert int(detect.reg_max) == reg_max
    assert int(detect.no) == int(detect.nc + 4 * reg_max)

    strides = [int(x) for x in detect.stride.tolist()]
    assert strides == [4, 8, 16]

    placements = normalize_spr_placements(cfg)
    placement_set = set(placements)

    # Verify exact layer-by-layer placement, not just total module count.
    stage_modules = {}
    for stage, idx in STAGE_LAYER_INDEX.items():
        name = seq[idx].__class__.__name__
        stage_modules[stage] = name
        expected = "SPRDown" if stage in placement_set else "Conv"
        assert name == expected, f"{stage}: got {name}, expected {expected}"

    sprdowns = [m for m in model.model.modules() if m.__class__.__name__ == "SPRDown"]
    aconvs = [m for m in model.model.modules() if m.__class__.__name__ == "AConv"]
    assert len(sprdowns) == len(placements)
    assert len(aconvs) == 0

    attn_names = {"ECA", "CoordAtt", "ResidualLiteCA"}
    attn_modules = [
        m for m in model.model.modules() if m.__class__.__name__ in attn_names
    ]
    assert cfg["attention"] == "none"
    assert not attn_modules

    # Placement ablation contract: architecture is the only experimental variable.
    assert cfg["loss_mode"] == "standard"
    assert int(cfg["reg_max"]) == 16
    assert cfg["attention"] == "none"
    assert cfg.get("pretrained", False) is False

    params = sum(p.numel() for p in model.model.parameters())
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
            macs, _ = profile(model.model, inputs=(x,), verbose=False)
        gflops = macs * 2 / 1e9
        del x
    except Exception as e:
        print("THOP skipped:", e)

    print()
    print("=" * 90)
    print("SANITY CHECK PASSED")
    print("=" * 90)
    print("Preset:", cfg["preset"])
    print("Experiment:", cfg["experiment_tag"])
    print("SPR placements:", placements)
    print("Stage modules:", stage_modules)
    print("REAL reg_max:", detect.reg_max)
    print("REAL Detect.no:", detect.no)
    print("Strides:", strides)
    print("SPRDown count:", len(sprdowns))
    print("Attention: none")
    print("Loss: standard")
    print(f"Params: {params:,} ({params/1e6:.4f} M)")
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
