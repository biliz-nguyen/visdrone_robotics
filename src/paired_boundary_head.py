from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect


class PairedBoundaryDetect(Detect):
    """DFL-free Detect head with coupled opposing-boundary parameterization.

    The stock direct-regression tower still emits four scalars per location, but
    they are interpreted as horizontal/vertical half-extent and signed center
    shift rather than four independent l/t/r/b distances. The mapping is a
    smooth bijection for positive distances, which matches TAL positives whose
    anchor point lies inside the ground-truth box.

    Raw channels per location: [sx, ox, sy, oy]
        hx = softplus(sx)
        cx = ratio_limit * tanh(ox) * hx
        hy = softplus(sy)
        cy = ratio_limit * tanh(oy) * hy
        l = hx - cx; r = hx + cx
        t = hy - cy; b = hy + cy

    Classification, assignment, CIoU, and direct-regression losses remain the
    stock Ultralytics ones. No extra prediction channels are introduced.
    """

    def __init__(
        self,
        nc: int = 80,
        ratio_limit: float = 0.99,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ):
        if end2end:
            raise ValueError("PairedBoundaryDetect v1 supports one-to-many detection only")
        if len(ch) != 3:
            raise ValueError(f"Expected 3 detection levels (P2/P3/P4), got {len(ch)}")
        if not (0.0 < ratio_limit < 1.0):
            raise ValueError("ratio_limit must be in (0, 1)")

        super().__init__(nc=nc, reg_max=1, end2end=False, ch=ch)
        self.ratio_limit = float(ratio_limit)

    def _couple_boundaries(self, raw: torch.Tensor) -> torch.Tensor:
        sx, ox, sy, oy = raw.chunk(4, dim=1)
        hx = F.softplus(sx) + 1e-4
        hy = F.softplus(sy) + 1e-4
        cx = self.ratio_limit * torch.tanh(ox) * hx
        cy = self.ratio_limit * torch.tanh(oy) * hy
        left = hx - cx
        right = hx + cx
        top = hy - cy
        bottom = hy + cy
        return torch.cat((left, top, right, bottom), dim=1)

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}

        bs = x[0].shape[0]
        boxes = []
        scores = []
        for i in range(self.nl):
            raw = box_head[i](x[i])
            coupled = self._couple_boundaries(raw)
            boxes.append(coupled.view(bs, 4, -1))
            scores.append(cls_head[i](x[i]).view(bs, self.nc, -1))

        return {
            "boxes": torch.cat(boxes, dim=-1),
            "scores": torch.cat(scores, dim=-1),
            "feats": x,
        }

    def bias_init(self):
        """Symmetric initial boxes: positive extent logits, zero center shift."""
        for i, (box_head, cls_head) in enumerate(zip(self.cv2, self.cv3)):
            bias = box_head[-1].bias.data.view(4)
            bias[0] = 1.0
            bias[1] = 0.0
            bias[2] = 1.0
            bias[3] = 0.0
            cls_head[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )
