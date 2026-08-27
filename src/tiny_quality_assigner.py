"""Tiny-adaptive quality alignment for frozen N2b.

This training-only assigner keeps the standard TAL candidate region, top-k
mechanism, conflict resolution, target boxes, and inference graph. It changes
only the localization-quality exponent for GT boxes below 16 px minimum side.

The setting is intentionally copied from the previously positive TAQ-v1
screen (tiny_min_side=16, beta_floor=4) rather than retuned on N2b.
"""

from __future__ import annotations

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


class TinyAdaptiveQualityAssigner(TaskAlignedAssigner):
    """Task-Aligned Assigner with a size-adaptive IoU exponent for tiny GTs."""

    def __init__(
        self,
        *args,
        tiny_min_side: float = 16.0,
        beta_floor: float = 4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tiny_min_side = float(tiny_min_side)
        self.beta_floor = float(beta_floor)
        if self.tiny_min_side <= 0:
            raise ValueError("tiny_min_side must be > 0")
        if not (0.0 < self.beta_floor <= float(self.beta)):
            raise ValueError("beta_floor must be in (0, beta]")

    def effective_beta(self, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """Return per-GT localization exponent with exact TAL recovery at >=16 px."""
        wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp_min(0.0)
        min_side = wh.amin(dim=-1, keepdim=True)
        ratio = (min_side / self.tiny_min_side).clamp(0.0, 1.0)
        base_beta = torch.as_tensor(float(self.beta), dtype=gt_bboxes.dtype, device=gt_bboxes.device)
        floor = torch.as_tensor(self.beta_floor, dtype=gt_bboxes.dtype, device=gt_bboxes.device)
        return floor + (base_beta - floor) * ratio

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()

        overlaps = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device
        )
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device
        )

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1).long().cpu()
        ind = ind.to(pd_scores.device)

        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        beta_eff = self.effective_beta(gt_bboxes).to(overlaps.dtype)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(beta_eff)
        return align_metric, overlaps
