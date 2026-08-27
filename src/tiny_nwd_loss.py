"""Tiny-gated NWD localization helper for direct-reg1 training.

The inference graph is untouched. For non-tiny assigned boxes the loss is
exactly the stock CIoU loss. For tiny boxes, a smooth size gate blends a small
Normalized Wasserstein Distance (NWD) term into CIoU to reduce sensitivity to
few-pixel localization errors.
"""

from __future__ import annotations

import torch


def nwd_similarity_xyxy(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, c: float = 12.8) -> torch.Tensor:
    """Return NWD similarity for aligned xyxy boxes in pixel coordinates."""
    if c <= 0:
        raise ValueError("c must be > 0")

    pred_cxcy = (pred_boxes[..., 0:2] + pred_boxes[..., 2:4]) * 0.5
    target_cxcy = (target_boxes[..., 0:2] + target_boxes[..., 2:4]) * 0.5
    pred_wh = (pred_boxes[..., 2:4] - pred_boxes[..., 0:2]).clamp_min(0.0)
    target_wh = (target_boxes[..., 2:4] - target_boxes[..., 0:2]).clamp_min(0.0)

    center_term = (pred_cxcy - target_cxcy).pow(2).sum(dim=-1, keepdim=True)
    size_term = ((pred_wh - target_wh) * 0.5).pow(2).sum(dim=-1, keepdim=True)
    wasserstein = torch.sqrt((center_term + size_term).clamp_min(1e-12))
    return torch.exp(-wasserstein / float(c))


def tiny_gate_xyxy(target_boxes: torch.Tensor, tiny_min_side: float = 16.0) -> torch.Tensor:
    """Smooth gate: 1 for vanishing boxes, linearly decays to 0 at tiny_min_side."""
    if tiny_min_side <= 0:
        raise ValueError("tiny_min_side must be > 0")
    wh = (target_boxes[..., 2:4] - target_boxes[..., 0:2]).clamp_min(0.0)
    min_side = wh.amin(dim=-1, keepdim=True)
    return (1.0 - min_side / float(tiny_min_side)).clamp(0.0, 1.0)


def blended_ciou_nwd_loss(
    ciou: torch.Tensor,
    pred_boxes_px: torch.Tensor,
    target_boxes_px: torch.Tensor,
    *,
    c: float = 12.8,
    tiny_min_side: float = 16.0,
    ciou_weight: float = 0.75,
    nwd_weight: float = 0.25,
) -> torch.Tensor:
    """Blend NWD into CIoU only for tiny targets; recover stock CIoU otherwise."""
    if ciou_weight < 0 or nwd_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if abs((ciou_weight + nwd_weight) - 1.0) > 1e-8:
        raise ValueError("ciou_weight + nwd_weight must equal 1")

    ciou_loss = 1.0 - ciou
    nwd_loss = 1.0 - nwd_similarity_xyxy(pred_boxes_px, target_boxes_px, c=c)
    gate = tiny_gate_xyxy(target_boxes_px, tiny_min_side=tiny_min_side).to(ciou_loss.dtype)
    tiny_blend = float(ciou_weight) * ciou_loss + float(nwd_weight) * nwd_loss
    return ciou_loss + gate * (tiny_blend - ciou_loss)
