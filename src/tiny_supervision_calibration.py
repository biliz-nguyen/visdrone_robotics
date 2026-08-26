"""Tiny-Supervision Calibration (TSC) for Task-Aligned Assignment.

Working C3 research hypothesis for YOLOEdge27; no novelty claim yet.

Motivation
----------
Ultralytics TAL first selects positives, then scales their soft classification
scores by a GT-level quality amplitude derived from the best positive overlap.
For very small boxes that amplitude can be brittle because a small localization
error causes a comparatively large IoU change. C3 deliberately leaves TAL's
candidate region, top-k ranking, conflict resolution, target boxes and network
architecture unchanged. It changes only the post-assignment supervision
amplitude for tiny GTs.

For GT minimum side s and tiny threshold tau:

    r = clamp(s / tau, 0, 1)
    gamma = gamma_floor + (1 - gamma_floor) * r
    q_cal = q ** gamma

where q is TAL's original per-GT positive-overlap amplitude. Thus non-tiny GTs
(s >= tau) are exactly standard TAL (gamma=1). Tiny GTs receive a bounded,
monotonic calibration q <= q_cal <= sqrt(q) for the locked v1 gamma_floor=0.5.
There is no inference-time change or extra parameter/FLOP cost.
"""

from __future__ import annotations

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


class TinySupervisionCalibratedAssigner(TaskAlignedAssigner):
    """Standard TAL assignment with tiny-aware soft-target amplitude calibration."""

    def __init__(
        self,
        *args,
        tiny_min_side: float = 16.0,
        gamma_floor: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tiny_min_side = float(tiny_min_side)
        self.gamma_floor = float(gamma_floor)
        if self.tiny_min_side <= 0:
            raise ValueError("tiny_min_side must be > 0")
        if not (0.0 < self.gamma_floor <= 1.0):
            raise ValueError("gamma_floor must be in (0, 1]")

    def quality_exponent(self, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """Return per-GT calibration exponent with shape ``(B, G, 1)``."""
        wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp_min(0.0)
        min_side = wh.amin(dim=-1, keepdim=True)
        ratio = (min_side / self.tiny_min_side).clamp(0.0, 1.0)
        floor = gt_bboxes.new_tensor(self.gamma_floor)
        return floor + (1.0 - floor) * ratio

    def calibrate_positive_quality(
        self,
        pos_overlaps: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Calibrate TAL's GT-level positive quality without changing positives.

        Invalid/padded GT slots are kept unchanged. For valid non-tiny GTs the
        result is bit-for-bit the same mathematical expression as standard TAL
        because the exponent equals 1.
        """
        q = pos_overlaps.clamp(0.0, 1.0)
        gamma = self.quality_exponent(gt_bboxes).to(dtype=q.dtype, device=q.device)
        calibrated = q.pow(gamma)
        valid = mask_gt.bool()
        return torch.where(valid, calibrated, q)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Run standard TAL, changing only its final soft-target amplitude."""
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )

        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # Standard TAL normalization up to the GT-level quality amplitude.
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)

        # C3/TSC: calibrate only tiny-GT supervision amplitude. Candidate
        # ranking, selected positives, conflicts and target boxes are unchanged.
        calibrated_overlaps = self.calibrate_positive_quality(pos_overlaps, gt_bboxes, mask_gt)
        norm_align_metric = (
            align_metric * calibrated_overlaps / (pos_align_metrics + self.eps)
        ).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
