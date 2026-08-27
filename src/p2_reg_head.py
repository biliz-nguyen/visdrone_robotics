from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.visdrone_p2_refine import P2Refine


class P2RegDetect(Detect):
    """Detect head with refinement isolated to P2 regression only.

    Regression (cv2) consumes a lightly refined P2 feature while P3/P4 stay
    stock. Classification (cv3) consumes the original P2/P3/P4 features
    exactly as stock Detect. ``preds['feats']`` also returns the original
    features so anchor generation and target geometry remain unchanged.
    """

    def __init__(
        self,
        nc: int = 80,
        end2end: bool = False,
        ch: tuple = (),
        alpha_init: float = 0.10,
    ):
        super().__init__(nc=nc, reg_max=1, end2end=end2end, ch=ch)
        if len(ch) != 3:
            raise ValueError(f"P2RegDetect expects exactly P2/P3/P4 channels, got {ch}")
        self.p2_reg_refine = P2Refine(int(ch[0]), int(ch[0]), alpha_init=float(alpha_init))

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]

        # Only stride-4/P2 regression sees the residual refinement.
        box_inputs = [self.p2_reg_refine(x[0]), x[1], x[2]]
        boxes = torch.cat(
            [box_head[i](box_inputs[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )

        # Classification remains stock at every pyramid level.
        scores = torch.cat(
            [cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": x}
