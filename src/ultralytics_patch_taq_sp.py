from __future__ import annotations

from pathlib import Path
import shutil

from .ultralytics_patch import patch_ultralytics as patch_base_ultralytics


def patch_ultralytics_taq_sp(cfg: dict) -> None:
    """Apply YOLOEdge27 patches, then install selection-preserving TAQ for training."""
    patch_base_ultralytics(cfg)

    repo = Path(cfg["ultra_repo"])
    loss_py = repo / "ultralytics" / "utils" / "loss.py"
    utils_dir = repo / "ultralytics" / "utils"
    project = Path(cfg["project_root"])

    src = project / "src" / "tiny_quality_assigner_sp.py"
    dst = utils_dir / "visdrone_tiny_quality_assigner_sp.py"
    shutil.copy2(src, dst)

    text = loss_py.read_text(encoding="utf-8")
    import_line = (
        "from ultralytics.utils.visdrone_tiny_quality_assigner_sp import "
        "SelectionPreservingTinyQualityAssigner"
    )
    anchor = "from ultralytics.utils.torch_utils import autocast\n"
    if import_line not in text:
        if anchor not in text:
            raise RuntimeError("Cannot find loss.py import anchor for TAQ-SP")
        text = text.replace(anchor, anchor + import_line + "\n", 1)

    old = """        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )"""
    q = cfg.get("tiny_quality_sp", {})
    tiny_min_side = float(q.get("tiny_min_side", 16.0))
    beta_floor = float(q.get("beta_floor", 5.0))
    if tiny_min_side != 16.0 or beta_floor != 5.0:
        raise ValueError("C3 TAQ-SP N2b screen is locked to tiny_min_side=16 and beta_floor=5")

    new = f"""        self.assigner = SelectionPreservingTinyQualityAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
            tiny_min_side={tiny_min_side},
            beta_floor={beta_floor},
        )"""
    if old not in text:
        raise RuntimeError("Cannot find TaskAlignedAssigner constructor for TAQ-SP patch")
    text = text.replace(old, new, 1)
    loss_py.write_text(text, encoding="utf-8")

    compile(dst.read_text(encoding="utf-8"), str(dst), "exec")
    compile(loss_py.read_text(encoding="utf-8"), str(loss_py), "exec")
