from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import textwrap


def ensure_repo(cfg: dict) -> Path:
    repo = Path(cfg["ultra_repo"])
    tag = cfg["ultralytics_tag"]

    if not (repo / ".git").exists():
        raise FileNotFoundError(
            f"Ultralytics repo not prepared: {repo}\n"
            "Run: bash setup_local.sh"
        )

    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--force", tag],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "-C", str(repo), "reset", "--hard", tag],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    return repo


def patch_ultralytics(cfg: dict) -> None:
    repo = ensure_repo(cfg)

    tasks_py = repo / "ultralytics" / "nn" / "tasks.py"
    loss_py = repo / "ultralytics" / "utils" / "loss.py"
    module_dir = repo / "ultralytics" / "nn" / "modules"
    custom_dst = module_dir / "visdrone_custom_blocks.py"
    custom_src = Path(cfg["project_root"]) / "src" / "custom_blocks.py"
    spr_v2_dst = module_dir / "visdrone_sprdown_v2.py"
    spr_v2_src = Path(cfg["project_root"]) / "src" / "sprdown_v2.py"

    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "checkout",
            "--",
            "ultralytics/nn/tasks.py",
            "ultralytics/utils/loss.py",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    shutil.copy2(custom_src, custom_dst)
    shutil.copy2(spr_v2_src, spr_v2_dst)

    _patch_tasks(tasks_py)

    if cfg["loss_mode"] == "hybrid_nwd":
        _patch_loss(loss_py, cfg)

    for path in [tasks_py, loss_py, custom_dst, spr_v2_dst]:
        compile(
            path.read_text(encoding="utf-8"),
            str(path),
            "exec",
        )


def _patch_tasks(tasks_py: Path) -> None:
    text = tasks_py.read_text(encoding="utf-8")

    import_lines = [
        (
            "from ultralytics.nn.modules.visdrone_custom_blocks "
            "import SPRDown, AConv, ECA, CoordAtt, ResidualLiteCA"
        ),
        (
            "from ultralytics.nn.modules.visdrone_sprdown_v2 "
            "import SPRDownV2"
        ),
    ]

    idx = text.find("class BaseModel")
    if idx == -1:
        raise RuntimeError("Cannot find class BaseModel in tasks.py")

    missing_imports = [line for line in import_lines if line not in text]
    if missing_imports:
        text = (
            text[:idx]
            + "\n".join(missing_imports)
            + "\n\n"
            + text[idx:]
        )

    base_match = re.search(
        r"base_modules\s*=\s*frozenset\(\s*\{"
        r"(?P<body>.*?)"
        r"\}\s*\)",
        text,
        flags=re.S,
    )

    if base_match is None:
        raise RuntimeError("Cannot find base_modules in tasks.py")

    body = base_match.group("body")
    needed = [
        "SPRDown",
        "SPRDownV2",
        "AConv",
        "ECA",
        "CoordAtt",
        "ResidualLiteCA",
    ]
    missing = [
        name
        for name in needed
        if not re.search(rf"\b{name}\b", body)
    ]

    if missing:
        insertion = "".join(
            f"\n            {name},"
            for name in missing
        )
        text = (
            text[:base_match.start("body")]
            + insertion
            + body
            + text[base_match.end("body"):]
        )

    tasks_py.write_text(text, encoding="utf-8")


def _patch_loss(loss_py: Path, cfg: dict) -> None:
    nwd_cfg = cfg["nwd"]
    nwd_c = float(nwd_cfg["c"])
    ciou_weight = float(nwd_cfg["ciou_weight"])
    nwd_weight = float(nwd_cfg["nwd_weight"])

    loss_text = loss_py.read_text(encoding="utf-8")
    bbox_start = loss_text.find("class BboxLoss(nn.Module):")

    if bbox_start == -1:
        raise RuntimeError("Cannot find BboxLoss in loss.py")

    next_class_match = re.search(
        r"\nclass\s+[A-Za-z_][A-Za-z0-9_]*"
        r"(?:\([^)]*\))?:",
        loss_text[bbox_start + len("class BboxLoss(nn.Module):"):],
    )

    if next_class_match is None:
        raise RuntimeError("Cannot find class after BboxLoss")

    bbox_end = (
        bbox_start
        + len("class BboxLoss(nn.Module):")
        + next_class_match.start()
        + 1
    )

    replacement = f"""
class BboxLoss(nn.Module):
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.nwd_constant = {nwd_c}

    @staticmethod
    def _xyxy_to_nwd_vector(boxes: torch.Tensor) -> torch.Tensor:
        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp_min(0.0)
        h = (y2 - y1).clamp_min(0.0)
        return torch.stack((cx, cy, w * 0.5, h * 0.5), dim=-1)

    def _nwd_similarity(self, pred_boxes, target_boxes):
        p = self._xyxy_to_nwd_vector(pred_boxes.float())
        t = self._xyxy_to_nwd_vector(target_boxes.float())
        distance = torch.sqrt(
            (p - t).pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-9)
        )
        return torch.exp(-distance / self.nwd_constant)

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        batch_size = pred_bboxes.shape[0]
        stride_map = stride.view(1, -1, 1).expand(batch_size, -1, -1)
        fg_stride = stride_map[fg_mask]
        pred_px = pred_bboxes[fg_mask] * fg_stride
        target_px = target_bboxes[fg_mask] * fg_stride
        nwd = self._nwd_similarity(pred_px, target_px)
        loss_nwd = (((1.0 - nwd) * weight.float()).sum() / target_scores_sum)

        iou = bbox_iou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            xywh=False,
            CIoU=True,
        )
        loss_ciou = (((1.0 - iou) * weight).sum() / target_scores_sum)
        loss_box = {ciou_weight} * loss_ciou + {nwd_weight} * loss_nwd

        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points,
                target_bboxes,
                self.dfl_loss.reg_max - 1,
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(
                    pred_dist[fg_mask],
                    target_ltrb[fg_mask],
                    reduction="none",
                )
                .mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_box, loss_dfl
"""

    loss_py.write_text(
        loss_text[:bbox_start]
        + textwrap.dedent(replacement).strip()
        + "\n\n"
        + loss_text[bbox_end:],
        encoding="utf-8",
    )
