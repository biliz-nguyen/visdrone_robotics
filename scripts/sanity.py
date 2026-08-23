#!/usr/bin/env python3

from pathlib import Path
import gc
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import normalize_head_bins, normalize_spr_placements
from src.runtime import prepare_runtime


STAGE_LAYER_INDEX = {"p2_p3": 3, "p3_p4": 5, "p4_p5": 7}


def main():
    cfg, data_yaml, model_yaml = prepare_runtime()

    import torch
    import ultralytics
    from ultralytics import YOLO

    print("=" * 90)
    print("SANITY CHECK")
    print("=" * 90)
    print("Ultralytics:", ultralytics.__version__)
    print("Source:", ultralytics.__file__)
    print("Data YAML:", data_yaml)
    print("Model YAML:", model_yaml)

    expected_repo = Path(cfg["ultra_repo"]).resolve()
    assert Path(ultralytics.__file__).resolve().is_relative_to(expected_repo)

    model = YOLO(str(model_yaml))
    seq = model.model.model
    detect = seq[-1]
    strides = [int(x) for x in detect.stride.tolist()]
    assert strides == [4, 8, 16]

    placements = normalize_spr_placements(cfg)
    placement_set = set(placements)
    stage_modules = {}
    for stage, idx in STAGE_LAYER_INDEX.items():
        name = seq[idx].__class__.__name__
        stage_modules[stage] = name
        expected = "SPRDown" if stage in placement_set else "Conv"
        assert name == expected, f"{stage}: got {name}, expected {expected}"

    sprdowns = [m for m in model.model.modules() if m.__class__.__name__ == "SPRDown"]
    assert len(sprdowns) == len(placements)

    head_mode = cfg.get("head_mode", "standard")
    if head_mode == "stride_reg":
        assert detect.__class__.__name__ == "StrideRegDetect"
        bins = normalize_head_bins(cfg)
        assert list(detect.reg_bins) == bins
        assert [m[-1].out_channels for m in detect.cv2] == [4 * x for x in bins]
    elif head_mode == "quality_overconfidence":
        assert detect.__class__.__name__ == "QualityOverconfidenceDetect"
        assert int(detect.reg_max) == 1
        assert abs(float(detect.qoc_lambda) - float(cfg["qoc_lambda"])) < 1e-12
        assert abs(float(detect.qoc_margin) - float(cfg["qoc_margin"])) < 1e-12
        assert abs(float(detect.qoc_tiny_threshold) - float(cfg.get("qoc_tiny_threshold", 16.0))) < 1e-12
        assert abs(float(detect.qoc_tiny_margin_bonus) - float(cfg.get("qoc_tiny_margin_bonus", 0.0))) < 1e-12
        bins = [1, 1, 1]
    else:
        bins = [int(detect.reg_max)] * len(strides)
        assert detect.__class__.__name__ == "Detect"
        assert int(detect.reg_max) == int(cfg["reg_max"])
        assert int(detect.no) == int(detect.nc + 4 * int(cfg["reg_max"]))

    if cfg.get("study") == "head":
        assert placements == ["p4_p5"]
        assert cfg["loss_mode"] == "standard"
        assert cfg["attention"] == "none"
        assert cfg.get("pretrained", False) is False

    criterion = model.model.init_criterion()
    assigner_name = criterion.assigner.__class__.__name__
    assert assigner_name == "TaskAlignedAssigner"
    if head_mode == "stride_reg":
        assert criterion.__class__.__name__ == "StrideRegDetectionLoss"
        assert list(criterion.reg_bins) == bins
    elif head_mode == "quality_overconfidence":
        assert criterion.__class__.__name__ == "QualityOverconfidenceLoss"
        assert abs(float(criterion.qoc_lambda) - float(cfg["qoc_lambda"])) < 1e-12
        assert abs(float(criterion.qoc_margin) - float(cfg["qoc_margin"])) < 1e-12
        assert abs(float(criterion.qoc_tiny_threshold) - float(cfg.get("qoc_tiny_threshold", 16.0))) < 1e-12
        assert abs(float(criterion.qoc_tiny_margin_bonus) - float(cfg.get("qoc_tiny_margin_bonus", 0.0))) < 1e-12

    params = sum(p.numel() for p in model.model.parameters())
    model.model.eval()
    dummy = torch.randn(1, 3, int(cfg["train"]["imgsz"]), int(cfg["train"]["imgsz"]))
    with torch.no_grad():
        pred = model.model(dummy)
    del dummy

    gflops = None
    try:
        from thop import profile
        model.model.cpu().eval()
        x = torch.randn(1, 3, int(cfg["train"]["imgsz"]), int(cfg["train"]["imgsz"]))
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
    print("Study:", cfg.get("study"))
    print("SPR placements:", placements)
    print("Stage modules:", stage_modules)
    print("Head:", detect.__class__.__name__)
    print("Head mode:", head_mode)
    print("Regression bins P2/P3/P4:", bins)
    if head_mode == "quality_overconfidence":
        print("QOC lambda:", criterion.qoc_lambda)
        print("QOC base margin:", criterion.qoc_margin)
        print("QOC tiny threshold px:", criterion.qoc_tiny_threshold)
        print("QOC tiny margin bonus:", criterion.qoc_tiny_margin_bonus)
    print("Assigner:", assigner_name)
    print("Strides:", strides)
    print("SPRDown count:", len(sprdowns))
    print("Attention:", cfg["attention"])
    print("Loss:", cfg["loss_mode"])
    print(f"Params: {params:,} ({params/1e6:.4f} M)")
    if gflops is not None:
        print(f"GFLOPs: {gflops:.4f}")
    print("Pretrained: False")
    print("=" * 90)

    del criterion, model, pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
