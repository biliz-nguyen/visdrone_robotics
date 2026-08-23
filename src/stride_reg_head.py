from __future__ import annotations

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors


class StrideRegDetect(Detect):
    """Detect head with level-specific regression distribution sizes.

    The classification branch is unchanged from Ultralytics Detect. Only the
    regression output width varies by pyramid level. This is an engineering
    prototype for testing whether fine-resolution levels need richer DFL
    distributions than coarser levels.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_bins: list[int] | tuple[int, ...] = (16, 8, 4),
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ):
        nn.Module.__init__(self)
        if end2end:
            raise ValueError("StrideRegDetect v1 supports one-to-many detection only")
        if len(reg_bins) != len(ch):
            raise ValueError(f"reg_bins length {len(reg_bins)} != detection levels {len(ch)}")
        if not ch:
            raise ValueError("ch must contain detection-level channels")

        bins = tuple(int(x) for x in reg_bins)
        if any(x < 1 for x in bins):
            raise ValueError("Every regression bin count must be >= 1")

        self.nc = int(nc)
        self.nl = len(ch)
        self.reg_bins = bins
        self.reg_max = max(bins)  # compatibility/introspection only
        self.no = self.nc + 4 * self.reg_max  # maximum per-level width, not a uniform contract
        self.stride = torch.zeros(self.nl)
        self._end2end = False
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)
        self.shape = None

        c2 = max(16, ch[0] // 4, 4 * self.reg_max)
        c3 = max(ch[0], min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(c, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * r, 1),
            )
            for c, r in zip(ch, self.reg_bins)
        )
        self.cv3 = (
            nn.ModuleList(
                nn.Sequential(
                    Conv(c, c3, 3),
                    Conv(c3, c3, 3),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for c in ch
            )
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(c, c, 3), Conv(c, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for c in ch
            )
        )

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        boxes = [
            box_head[i](x[i]).view(bs, 4 * self.reg_bins[i], -1)
            for i in range(self.nl)
        ]
        scores = torch.cat(
            [cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": x}

    @staticmethod
    def _decode_distribution(box_logits: torch.Tensor, bins: int) -> torch.Tensor:
        bs, _, n = box_logits.shape
        if bins == 1:
            return box_logits.view(bs, 4, n)
        logits = box_logits.view(bs, 4, bins, n)
        probs = logits.softmax(2)
        proj = torch.arange(bins, device=box_logits.device, dtype=box_logits.dtype).view(1, 1, bins, 1)
        return (probs * proj).sum(2)

    def _get_decode_boxes(self, x):
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5)
            )
            self.shape = shape

        decoded = []
        start = 0
        for feat, box_logits, bins in zip(x["feats"], x["boxes"], self.reg_bins):
            n = int(feat.shape[-2] * feat.shape[-1])
            end = start + n
            dist = self._decode_distribution(box_logits, bins)
            dbox = self.decode_bboxes(
                dist,
                self.anchors[:, start:end].unsqueeze(0),
            ) * self.strides[:, start:end]
            decoded.append(dbox)
            start = end
        return torch.cat(decoded, dim=-1)

    def bias_init(self):
        for i, (box_head, cls_head) in enumerate(zip(self.cv2, self.cv3)):
            box_head[-1].bias.data[:] = 2.0
            cls_head[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )
