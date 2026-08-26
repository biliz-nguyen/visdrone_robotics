#!/usr/bin/env python3

from pathlib import Path
import gc
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import normalize_head_bins, normalize_neck_channels, normalize_spr_placements
from src.runtime import prepare_runtime


STAGE_LAYER_INDEX = {"p2_p3": 3, "p3_p4": 5, "p4_p5": 7}


def _check_realloc_shapes(seq, cfg):
    nominal = normalize_neck_channels(cfg)
    expected_map = {
        (160, 256, 384): ({"p2": 40, "p3": 64, "p4": 96}, {"p2_p3": 64, "p3_p4": 96}),
        (160, 256, 416): ({"p2": 40, "p3": 64, "p4": 104}, {"p2_p3": 64, "p3_p4": 104}),
    }
    key = (nominal["p2"], nominal["p3"], nominal["p4"])
    if key not in expected_map:
        raise AssertionError(f"Unregistered realloc widths: {nominal}")
    expected_neck, expected_pan = expected_map[key]
    neck_effective = {
        "p2": int(seq[19].cv2.conv.out_channels),
        "p3": int(seq[22].cv2.conv.out_channels),
        "p4": int(seq[25].cv2.conv.out_channels),
    }
    assert neck_effective == expected_neck, (neck_effective, expected_neck)
    pan_down_effective = {
        "p2_p3": int(seq[20].conv.out_channels),
        "p3_p4": int(seq[23].conv.out_channels),
    }
    assert pan_down_effective == expected_pan, (pan_down_effective, expected_pan)
    return nominal, neck_effective, pan_down_effective


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
    else:
        bins = [int(detect.reg_max)] * len(strides)
        assert detect.__class__.__name__ == "Detect"
        assert int(detect.reg_max) == int(cfg["reg_max"])
        assert int(detect.no) == int(detect.nc + 4 * int(cfg["reg_max"]))

    study = cfg.get("study")
    neck_mode = cfg.get("neck_mode", "standard")
    rep_neck_blocks = [m for m in model.model.modules() if m.__class__.__name__ == "RepC3k2"]
    neck_nominal = None
    neck_effective = None
    pan_down_effective = None

    if study == "head":
        assert placements == ["p4_p5"]
        assert cfg["loss_mode"] == "standard"
        assert cfg["attention"] == "none"
        assert neck_mode == "standard"
        assert cfg.get("pretrained", False) is False

    if study == "neck":
        assert placements == ["p4_p5"]
        assert cfg["loss_mode"] == "standard"
        assert cfg["attention"] == "none"
        assert head_mode == "standard"
        assert int(cfg["reg_max"]) == 1
        assert cfg.get("pretrained", False) is False
        if neck_mode == "rep":
            assert len(rep_neck_blocks) == 5
        else:
            assert len(rep_neck_blocks) == 0
        if neck_mode == "realloc":
            neck_nominal, neck_effective, pan_down_effective = _check_realloc_shapes(seq, cfg)

    if study == "optimization":
        assert placements == ["p4_p5"]
        assert cfg["loss_mode"] == "standard"
        assert cfg["attention"] == "none"
        assert head_mode == "standard"
        assert int(cfg["reg_max"]) == 1
        assert neck_mode == "realloc"
        assert cfg.get("pretrained", False) is False
        assert len(rep_neck_blocks) == 0
        neck_nominal, neck_effective, pan_down_effective = _check_realloc_shapes(seq, cfg)
        assert neck_nominal == {"p2": 160, "p3": 256, "p4": 416}, neck_nominal

    criterion = model.model.init_criterion()
    assigner_name = criterion.assigner.__class__.__name__
    assigner_mode = cfg.get("assigner_mode", "standard")
    expected_assigners = {
        "standard": "TaskAlignedAssigner",
        "tiny_center_rank": "TinyCenterStableRankAssigner",
    }
    if assigner_mode not in expected_assigners:
        raise AssertionError(f"Unsupported sanity assigner_mode={assigner_mode!r}")
    expected_assigner = expected_assigners[assigner_mode]
    assert assigner_name == expected_assigner, (assigner_name, expected_assigner)

    if study == "optimization":
        assert assigner_mode == "tiny_center_rank"
        assert float(cfg["tiny_center_rank"]["tiny_min_side"]) == 16.0

    if head_mode == "stride_reg":
        assert criterion.__class__.__name__ == "StrideRegDetectionLoss"
        assert list(criterion.reg_bins) == bins

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
    print("Study:", study)
    print("SPR placements:", placements)
    print("Stage modules:", stage_modules)
    print("Neck mode:", neck_mode)
    print("RepC3k2 count:", len(rep_neck_blocks))
    if neck_effective is not None:
        print("Neck nominal channels P2/P3/P4:", neck_nominal)
        print("Neck effective channels P2/P3/P4:", neck_effective)
        print("PAN downsample effective channels:", pan_down_effective)
    print("Head:", detect.__class__.__name__)
    print("Head mode:", head_mode)
    print("Regression bins P2/P3/P4:", bins)
    print("Assigner mode:", assigner_mode)
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
