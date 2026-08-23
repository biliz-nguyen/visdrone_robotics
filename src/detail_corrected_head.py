from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect


class DetailCorrectedDetect(Detect):
    """DFL-free Detect head with a bounded P2-only detail correction.

    All pyramid levels use the stock direct-regression (reg_max=1) tower.
    Only the finest P2 level receives an additional high-frequency residual
    correction derived from x - AvgPool3x3(x). The correction is bounded to a
    fraction of one feature cell, keeping the mechanism local and inexpensive.
    Classification is unchanged from the parent Detect head.
    """

    def __init__(
        self,
        nc: int = 80,
        max_correction_cell: float = 0.5,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ):
        if end2end:
            raise ValueError("DetailCorrectedDetect v1 supports one-to-many detection only")
        if len(ch) != 3:
            raise ValueError(f"Expected 3 detection levels (P2/P3/P4), got {len(ch)}")
        if max_correction_cell <= 0:
            raise ValueError("max_correction_cell must be positive")

        # Build the efficient stock DFL-free head first. This preserves the
        # exact H1 regression/classification towers and standard v8 loss.
        super().__init__(nc=nc, reg_max=1, end2end=False, ch=ch)

        self.max_correction_cell = float(max_correction_cell)
        self.detail_proj = nn.Conv2d(ch[0], 4, kernel_size=1, stride=1, padding=0, bias=True)

        # Start exactly from the H1 direct-regression behavior. The detail
        # branch learns only if gradients show that local P2 correction helps.
        nn.init.zeros_(self.detail_proj.weight)
        nn.init.zeros_(self.detail_proj.bias)

    def _p2_detail_correction(self, x: torch.Tensor) -> torch.Tensor:
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - local_mean
        delta = self.detail_proj(high_freq)
        return self.max_correction_cell * torch.tanh(delta)

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}

        bs = x[0].shape[0]
        box_levels = []
        score_levels = []
        for i in range(self.nl):
            box = box_head[i](x[i])
            if i == 0:
                box = box + self._p2_detail_correction(x[i])
            box_levels.append(box.view(bs, 4, -1))
            score_levels.append(cls_head[i](x[i]).view(bs, self.nc, -1))

        return {
            "boxes": torch.cat(box_levels, dim=-1),
            "scores": torch.cat(score_levels, dim=-1),
            "feats": x,
        }
