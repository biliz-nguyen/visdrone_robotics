"""Pixel-stable tiny-object task alignment for YOLOEdge27.

Working research hypothesis only; no novelty claim.

Standard TAL ranks candidates with cls_score**alpha * IoU**beta. For tiny
objects, candidate ordering can be brittle because a one-pixel localization
perturbation changes IoU strongly. This assigner keeps TAL's candidate region,
top-k logic, conflict resolution, targets, and loss unchanged. It changes only
the tiny-GT quality used inside the alignment metric.

For each tiny GT/candidate pair, v1 evaluates IoU at the annotated box and at
four cardinal one-pixel perturbations of that GT. The tiny quality is

    q_stable = sqrt(q0 * mean(q_left, q_right, q_up, q_down)).

Non-tiny GTs use q0 exactly. Returning the original IoU tensor separately keeps
TAL's downstream target-quality normalization standard; only candidate ranking
for tiny GTs sees q_stable.
"""

from __future__ import annotations

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


def cardinal_shifted_boxes(boxes: torch.Tensor, perturb_px: float) -> tuple[torch.Tensor, ...]:
    """Return xyxy boxes shifted left/right/up/down by ``perturb_px`` pixels."""
    p = float(perturb_px)
    if p <= 0:
        raise ValueError("perturb_px must be > 0")
    offsets = ((-p, 0.0), (p, 0.0), (0.0, -p), (0.0, p))
    out = []
    for dx, dy in offsets:
        offset = boxes.new_tensor([dx, dy, dx, dy])
        out.append(boxes + offset)
    return tuple(out)


def pixel_stable_quality(standard_iou: torch.Tensor, shifted_ious: torch.Tensor) -> torch.Tensor:
    """Geometric blend of nominal IoU and mean one-pixel-perturbed IoU."""
    if shifted_ious.ndim != standard_iou.ndim + 1:
        raise ValueError("shifted_ious must add exactly one cardinal dimension")
    q0 = standard_iou.clamp(0.0, 1.0)
    qs = shifted_ious.clamp(0.0, 1.0).mean(dim=0)
    return torch.sqrt((q0 * qs).clamp_min(0.0))


class TinyPixelStableAssigner(TaskAlignedAssigner):
    """Task-Aligned Assigner with pixel-stable ranking for tiny GT boxes."""

    def __init__(self, *args, tiny_min_side: float = 16.0, perturb_px: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiny_min_side = float(tiny_min_side)
        self.perturb_px = float(perturb_px)
        if self.tiny_min_side <= 0:
            raise ValueError("tiny_min_side must be > 0")
        if self.perturb_px <= 0:
            raise ValueError("perturb_px must be > 0")

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute TAL metrics, replacing only tiny-GT ranking quality."""
        na = pd_bboxes.shape[-2]
        pair_mask = mask_gt.bool()

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
        bbox_scores[pair_mask] = pd_scores[ind[0], :, ind[1]][pair_mask]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[pair_mask]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[pair_mask]
        nominal = self.iou_calculation(gt_boxes, pd_boxes).reshape(-1).clamp_min_(0.0)
        overlaps[pair_mask] = nominal

        ranking_quality = overlaps.clone()
        wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp_min(0.0)
        tiny_gt = wh.amin(dim=-1, keepdim=True) < self.tiny_min_side
        tiny_pairs = pair_mask & tiny_gt.expand_as(pair_mask)

        if tiny_pairs.any():
            tiny_pd = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[tiny_pairs]
            tiny_gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[tiny_pairs]
            tiny_nominal = overlaps[tiny_pairs]
            shifted = []
            for shifted_gt in cardinal_shifted_boxes(tiny_gt_boxes, self.perturb_px):
                q = self.iou_calculation(shifted_gt, tiny_pd).reshape(-1).clamp_min_(0.0)
                shifted.append(q)
            shifted_ious = torch.stack(shifted, dim=0)
            ranking_quality[tiny_pairs] = pixel_stable_quality(tiny_nominal, shifted_ious)

        align_metric = bbox_scores.pow(self.alpha) * ranking_quality.pow(self.beta)
        return align_metric, overlaps
