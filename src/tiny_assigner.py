"""Tiny-aware candidate recovery for Task-Aligned Assignment.

Research prototype for YOLOEdge27 Stage 2.

Motivation:
Tiny objects can contain very few (or zero) anchor centers even with a P2 head.
Standard Task-Aligned Assignment (TAL) only ranks anchors whose centers fall
inside the ground-truth box, so these objects can receive sparse supervision.

This module keeps standard TAL unchanged for non-tiny objects. For tiny ground
truths only, if the inside-GT candidate count is below ``min_candidates``, it
adds the nearest unused anchor centers to the candidate pool before TAL ranks
candidates with its normal classification/localization alignment metric.
Targets remain the original boxes; only the coarse candidate pool is repaired.

This is an engineering hypothesis and not a novelty claim.
"""

from __future__ import annotations

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


class TinyCandidateRecoveryAssigner(TaskAlignedAssigner):
    """Task-Aligned Assigner with candidate recovery for tiny ground truths."""

    def __init__(
        self,
        *args,
        tiny_min_side: float = 16.0,
        min_candidates: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tiny_min_side = float(tiny_min_side)
        self.min_candidates = int(min_candidates)
        if self.tiny_min_side <= 0:
            raise ValueError("tiny_min_side must be > 0")
        if self.min_candidates < 1:
            raise ValueError("min_candidates must be >= 1")

    def recover_candidate_mask(
        self,
        anc_points: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Return inside-GT candidates plus nearest-center recovery for tiny GTs.

        Args:
            anc_points: Anchor centers in image pixels, shape ``(A, 2)``.
            gt_bboxes: Ground-truth boxes in image pixels, shape ``(B, G, 4)``.
            mask_gt: Valid-GT mask, shape ``(B, G, 1)``.
        """
        base = self.select_candidates_in_gts(
            anc_points,
            gt_bboxes,
            mask_gt,
        ).bool()

        if gt_bboxes.numel() == 0 or anc_points.numel() == 0:
            return base

        valid = mask_gt.squeeze(-1).bool()
        wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp_min(0)
        tiny = (wh.amin(dim=-1) <= self.tiny_min_side) & valid

        inside_count = base.sum(dim=-1)
        need = (self.min_candidates - inside_count).clamp(
            min=0,
            max=self.min_candidates,
        )
        recover = tiny & (need > 0)

        if not bool(recover.any()):
            return base

        centers = (gt_bboxes[..., 0:2] + gt_bboxes[..., 2:4]) * 0.5
        delta = centers.unsqueeze(-2) - anc_points.view(1, 1, -1, 2)
        dist2 = delta.square().sum(dim=-1)

        # Existing inside-GT candidates should not consume recovery slots.
        dist2 = dist2.masked_fill(base, float("inf"))

        k = min(self.min_candidates, int(anc_points.shape[0]))
        nearest = torch.topk(
            dist2,
            k=k,
            dim=-1,
            largest=False,
        ).indices

        supplement = torch.zeros_like(base)
        for rank in range(k):
            active = recover & (need > rank)
            if not bool(active.any()):
                continue
            idx = nearest[..., rank : rank + 1]
            supplement.scatter_(-1, idx, active.unsqueeze(-1))

        return base | supplement

    def get_pos_mask(
        self,
        pd_scores,
        pd_bboxes,
        gt_labels,
        gt_bboxes,
        anc_points,
        mask_gt,
    ):
        """Build the TAL positive mask from the recovered candidate pool."""
        candidate_mask = self.recover_candidate_mask(
            anc_points,
            gt_bboxes,
            mask_gt,
        )

        align_metric, overlaps = self.get_box_metrics(
            pd_scores,
            pd_bboxes,
            gt_labels,
            gt_bboxes,
            candidate_mask * mask_gt,
        )

        mask_topk = self.select_topk_candidates(
            align_metric,
            topk_mask=mask_gt.expand(-1, -1, self.topk).bool(),
        )
        mask_pos = mask_topk * candidate_mask * mask_gt

        return mask_pos, align_metric, overlaps
