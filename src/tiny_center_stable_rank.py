"""Tiny Center-Stable Ranking (TCSR) for Task-Aligned Assignment.

Working C3-v2 research hypothesis for YOLOEdge27; no novelty claim yet.

C3-v1 TSC increased tiny-object supervision amplitude after assignment. Its
5-epoch screen improved recall but reduced precision and aggregate AP. TCSR
changes a different part of training: only the ranking used to choose TAL
positives for tiny GTs.

For a candidate anchor a inside GT g, define FCOS-style center prior

    c(a,g) = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))

and a size-dependent weight

    lambda(g) = clamp(1 - min_side(g) / tau, 0, 1).

The ranking metric becomes

    A_rank = A_TAL * c(a,g) ** lambda(g)

with tau=16 px locked for v2. Therefore GTs with min_side >= 16 px are exactly
standard TAL. For tiny GTs, central anchors are preferred smoothly as object
size shrinks. Importantly, conflict resolution, target boxes, and TAL's final
soft-target normalization still use the original IoU and original TAL alignment
metric. The change is restricted to top-k positive membership.

There is no inference-time module, parameter, or FLOP overhead.
"""

from __future__ import annotations

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


class TinyCenterStableRankAssigner(TaskAlignedAssigner):
    """Standard TAL with tiny-aware center-stable top-k ranking only."""

    def __init__(self, *args, tiny_min_side: float = 16.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiny_min_side = float(tiny_min_side)
        if self.tiny_min_side <= 0:
            raise ValueError("tiny_min_side must be > 0")

    def tiny_weight(self, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """Return lambda in [0,1] with shape (B,G,1); non-tiny GTs get zero."""
        wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp_min(0.0)
        min_side = wh.amin(dim=-1, keepdim=True)
        return (1.0 - min_side / self.tiny_min_side).clamp(0.0, 1.0)

    def center_prior(
        self,
        anc_points: torch.Tensor,
        gt_bboxes: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """FCOS-style centerness for each GT/anchor pair.

        Args:
            anc_points: (A,2) anchor centers in image pixels.
            gt_bboxes: (B,G,4) xyxy boxes in image pixels.
            valid_mask: (B,G,A) valid anchor-in-GT mask.
        """
        anc = anc_points.view(1, 1, -1, 2)
        gt_lt = gt_bboxes[..., None, 0:2]
        gt_rb = gt_bboxes[..., None, 2:4]
        lt = (anc - gt_lt).clamp_min(0.0)
        rb = (gt_rb - anc).clamp_min(0.0)

        l, t = lt.unbind(-1)
        r, b = rb.unbind(-1)
        lr = torch.minimum(l, r) / torch.maximum(l, r).clamp_min(self.eps)
        tb = torch.minimum(t, b) / torch.maximum(t, b).clamp_min(self.eps)
        prior = (lr * tb).clamp(0.0, 1.0).sqrt()
        return torch.where(valid_mask.bool(), prior, torch.zeros_like(prior))

    def rank_alignment(
        self,
        standard_align: torch.Tensor,
        anc_points: torch.Tensor,
        gt_bboxes: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the tiny-only center prior to TAL ranking.

        For non-tiny GTs lambda=0 and this returns standard_align exactly.
        """
        lam = self.tiny_weight(gt_bboxes).to(device=standard_align.device, dtype=standard_align.dtype)
        prior = self.center_prior(anc_points, gt_bboxes, valid_mask).to(dtype=standard_align.dtype)
        factor = torch.where(
            lam > 0,
            prior.clamp_min(self.eps).pow(lam),
            torch.ones_like(prior),
        )
        return standard_align * factor

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Run TAL with TCSR used only for top-k membership selection."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        valid = mask_in_gts * mask_gt

        # Standard TAL geometry/classification quantities are preserved.
        standard_align, overlaps = super().get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, valid
        )
        rank_align = self.rank_alignment(standard_align, anc_points, gt_bboxes, valid)

        mask_topk = self.select_topk_candidates(
            rank_align,
            topk_mask=mask_gt.expand(-1, -1, self.topk).bool(),
        )
        mask_pos = mask_topk * mask_in_gts * mask_gt

        # Keep standard TAL conflict resolution (original IoU/alignment), so the
        # only changed decision is which candidates enter the top-k set.
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, standard_align
        )
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # Keep standard TAL target-score normalization exactly, but over the
        # TCSR-selected positive set. This avoids the broad amplitude boost that
        # hurt precision in C3-v1 TSC.
        selected_align = standard_align * mask_pos
        pos_align_metrics = selected_align.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            selected_align * pos_overlaps / (pos_align_metrics + self.eps)
        ).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
