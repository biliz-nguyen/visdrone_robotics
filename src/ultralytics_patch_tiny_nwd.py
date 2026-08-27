from __future__ import annotations

from pathlib import Path
import shutil

from .ultralytics_patch import patch_ultralytics as patch_base_ultralytics


def patch_ultralytics_tiny_nwd(cfg: dict) -> None:
    """Apply frozen YOLOEdge27 patch, then add training-only tiny-gated NWD."""
    patch_base_ultralytics(cfg)

    repo = Path(cfg["ultra_repo"])
    loss_py = repo / "ultralytics" / "utils" / "loss.py"
    utils_dir = repo / "ultralytics" / "utils"
    project = Path(cfg["project_root"])

    src = project / "src" / "tiny_nwd_loss.py"
    dst = utils_dir / "visdrone_tiny_nwd_loss.py"
    shutil.copy2(src, dst)

    q = cfg.get("tiny_nwd", cfg.get("nwd", {}))
    c = float(q.get("c", 12.8))
    tiny_min_side = float(q.get("tiny_min_side", 16.0))
    ciou_weight = float(q.get("ciou_weight", 0.75))
    nwd_weight = float(q.get("nwd_weight", 0.25))
    if c != 12.8 or tiny_min_side != 16.0 or ciou_weight != 0.75 or nwd_weight != 0.25:
        raise ValueError("C4 screen is locked to c=12.8, tiny_min_side=16, CIoU/NWD=0.75/0.25")

    text = loss_py.read_text(encoding="utf-8")
    import_line = "from ultralytics.utils.visdrone_tiny_nwd_loss import blended_ciou_nwd_loss"
    anchor = "from .tal import bbox2dist, rbox2dist\n"
    if import_line not in text:
        if anchor not in text:
            raise RuntimeError("Cannot find loss.py import anchor for C4")
        text = text.replace(anchor, anchor + import_line + "\n", 1)

    old = """        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
"""
    new = f"""        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        stride_fg = stride.view(1, -1, 1).expand(target_bboxes.shape[0], -1, -1)[fg_mask]
        pred_boxes_px = pred_bboxes[fg_mask] * stride_fg
        target_boxes_px = target_bboxes[fg_mask] * stride_fg
        loc_loss = blended_ciou_nwd_loss(
            iou,
            pred_boxes_px,
            target_boxes_px,
            c={c},
            tiny_min_side={tiny_min_side},
            ciou_weight={ciou_weight},
            nwd_weight={nwd_weight},
        )
        loss_iou = (loc_loss * weight).sum() / target_scores_sum
"""
    if old not in text:
        raise RuntimeError("Cannot find stock BboxLoss CIoU block for C4 patch")
    text = text.replace(old, new, 1)
    loss_py.write_text(text, encoding="utf-8")

    compile(dst.read_text(encoding="utf-8"), str(dst), "exec")
    compile(loss_py.read_text(encoding="utf-8"), str(loss_py), "exec")
