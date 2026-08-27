"""Selection-preserving tiny quality reweighting for frozen N2b.

TAQ-v1 changed the TAL alignment metric before top-k selection and reduced
pedestrian recall on N2b. This variant keeps stock TAL for candidate ranking,
positive selection, and conflict resolution. Only the normalized quality
weight applied to already-selected tiny positives is softened.

The mechanism is training-only and does not modify the inference graph.
"""

from __future__ import annotations

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


class SelectionPreservingTinyQualityAssigner(TaskAlignedAssigner):
    """TAL with stock positive selection and size-adaptive target reweighting."""

    def __init__(
        self,
        *args,
        tiny_min_side: float = 16.0,
        beta_floor: float = 5.0,
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
        """Return a mild per-GT quality exponent; recover stock beta at >= threshold."""
        wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp_min(0.0)
        min_side = wh.amin(dim=-1, keepdim=True)
        ratio = (min_side / self.tiny_min_side).clamp(0.0, 1.0)
        base_beta = torch.as_tensor(float(self.beta), dtype=gt_bboxes.dtype, device=gt_bboxes.device)
        floor = torch.as_tensor(self.beta_floor, dtype=gt_bboxes.dtype, device=gt_bboxes.device)
        return floor + (base_beta - floor) * ratio

    def _adaptive_align_metric(
        self,
        stock_align_metric: torch.Tensor,
        overlaps: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """Reweight stock alignment after selection without changing selected positives."""
        beta_eff = self.effective_beta(gt_bboxes).to(overlaps.dtype)
        delta_beta = beta_eff - float(self.beta)
        delta_beta = delta_beta.expand_as(overlaps)

        factor = torch.ones_like(overlaps)
        valid = overlaps > 0
        factor[valid] = overlaps[valid].pow(delta_beta[valid])
        return stock_align_metric * factor

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        # Stock TAL decides every positive. This is the key difference from TAQ-v1.
        mask_pos, stock_align_metric, overlaps = super().get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, stock_align_metric
        )
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # Only quality normalization is adapted for already-selected tiny positives.
        align_metric = self._adaptive_align_metric(stock_align_metric, overlaps, gt_bboxes)
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        ).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
