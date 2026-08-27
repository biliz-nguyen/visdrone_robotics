from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.visdrone_p2_refine import P2Refine


class P2ClsDetect(Detect):
    """Detect head with refinement isolated to P2 classification only.

    Regression (cv2) consumes the original P2/P3/P4 features exactly as stock
    Detect. Classification (cv3) consumes a lightly refined P2 feature while
    P3/P4 remain stock. The feature tensors returned in ``preds['feats']`` are
    also the original features so anchor generation and regression targets are
    unchanged.
    """

    def __init__(self, nc: int = 80, end2end: bool = False, ch: tuple = (), alpha_init: float = 0.10):
        super().__init__(nc=nc, reg_max=1, end2end=end2end, ch=ch)
        if len(ch) != 3:
            raise ValueError(f"P2ClsDetect expects exactly P2/P3/P4 channels, got {ch}")
        self.p2_cls_refine = P2Refine(int(ch[0]), int(ch[0]), alpha_init=float(alpha_init))

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]

        # Box branch remains byte-for-byte equivalent in topology to stock
        # Detect: no refinement is applied before cv2 at any pyramid level.
        boxes = torch.cat(
            [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )

        # Only stride-4/P2 classification sees the residual refinement.
        cls_inputs = [self.p2_cls_refine(x[0]), x[1], x[2]]
        scores = torch.cat(
            [cls_head[i](cls_inputs[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": x}
