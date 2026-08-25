from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors


def shift_image(img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate an image tensor by integer pixels using replicate padding."""
    if img.ndim != 4:
        raise ValueError("shift_image expects BCHW input")
    h, w = int(img.shape[-2]), int(img.shape[-1])
    left, right = max(dx, 0), max(-dx, 0)
    top, bottom = max(dy, 0), max(-dy, 0)
    padded = F.pad(img, (left, right, top, bottom), mode="replicate")
    x0, y0 = max(-dx, 0), max(-dy, 0)
    return padded[..., y0 : y0 + h, x0 : x0 + w]


def inverse_shift_boxes(boxes: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Map xyxy boxes from a shifted image back to the original coordinates."""
    offset = boxes.new_tensor([dx, dy, dx, dy])
    return boxes - offset


def object_representatives(
    pred_bboxes_px: torch.Tensor,
    fg_mask: torch.Tensor,
    target_gt_idx: torch.Tensor,
    target_scores: torch.Tensor,
    max_gt: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one deterministic high-quality positive prediction per GT object.

    The representative is the assigned positive with maximum detached TAL target
    score for each (image, local-GT-index) pair. This avoids anchor-to-anchor
    matching across the translated view: consistency is enforced at object level.
    """
    batch_size = int(pred_bboxes_px.shape[0])
    num_keys = batch_size * int(max_gt)
    reps = pred_bboxes_px.new_zeros((num_keys, 4))
    valid = torch.zeros(num_keys, dtype=torch.bool, device=pred_bboxes_px.device)

    pos_loc = torch.nonzero(fg_mask, as_tuple=False)
    if pos_loc.numel() == 0 or max_gt <= 0:
        return reps, valid

    pos_boxes = pred_bboxes_px[fg_mask]
    pos_gt = target_gt_idx[fg_mask].long()
    keys = pos_loc[:, 0].long() * int(max_gt) + pos_gt
    scores = target_scores.sum(-1)[fg_mask].detach()

    # Rank by target score descending, then stably group by object key. The first
    # element in every key group is therefore its highest-score representative.
    by_score = torch.argsort(scores, descending=True, stable=True)
    ranked = by_score
    by_key = torch.argsort(keys[ranked], stable=True)
    ranked = ranked[by_key]
    ranked_keys = keys[ranked]

    first = torch.ones_like(ranked_keys, dtype=torch.bool)
    if ranked_keys.numel() > 1:
        first[1:] = ranked_keys[1:] != ranked_keys[:-1]
    selected = ranked[first]
    selected_keys = keys[selected]

    reps[selected_keys] = pos_boxes[selected]
    valid[selected_keys] = True
    return reps, valid


def shift_equivariance_penalty(
    original_boxes: torch.Tensor,
    shifted_boxes: torch.Tensor,
    reliability: torch.Tensor,
    dx: int,
    dy: int,
) -> torch.Tensor:
    """CIoU consistency after mapping shifted-view boxes back to the base view."""
    if original_boxes.numel() == 0:
        return original_boxes.new_zeros(())
    shifted_back = inverse_shift_boxes(shifted_boxes, dx, dy)
    ciou = bbox_iou(original_boxes, shifted_back, xywh=False, CIoU=True).reshape(-1)
    return ((1.0 - ciou) * reliability).mean()


@contextmanager
def _freeze_bn_running_stats(model: torch.nn.Module):
    """Prevent the auxiliary shifted forward from updating BN running stats."""
    states: list[tuple[torch.nn.Module, bool]] = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            states.append((module, bool(module.training)))
            module.training = False
    try:
        yield
    finally:
        for module, training in states:
            module.training = training


class TinyShiftConsistencyLoss(v8DetectionLoss):
    """Stock H1 loss plus training-only tiny-object translation consistency.

    Motivation: tiny detections are unusually sensitive to one-pixel input
    translations because downsampling changes the sampling phase. For each
    training batch, v1 draws one cardinal one-pixel translation, performs one
    auxiliary forward pass, matches the two views by GT object identity, and
    requires their decoded boxes to agree after undoing the known translation.

    Only tiny GTs (< tiny_threshold on the shorter side) that receive a positive
    assignment in both views contribute. The consistency weight is the detached
    geometric mean of each view's IoU to its GT, so two equally bad predictions
    are not strongly encouraged merely for being mutually consistent.

    Validation and inference are unchanged from H1; the extra forward exists only
    while gradients are enabled during training.
    """

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        head = model.model[-1]
        if self.reg_max != 1:
            raise ValueError("TinyShiftConsistency v1 is locked to reg_max=1")
        self.shift_lambda = float(head.shift_lambda)
        self.tiny_threshold = float(head.tiny_threshold)
        self.max_shift_px = int(head.max_shift_px)
        self.student_model = model

    @staticmethod
    def _raw_predictions(output: Any) -> dict[str, torch.Tensor]:
        raw = output[1] if isinstance(output, tuple) else output
        if not isinstance(raw, dict) or "boxes" not in raw or "scores" not in raw or "feats" not in raw:
            raise TypeError("Shifted forward did not return a compatible raw detection dictionary")
        return raw

    def _assign_view(
        self,
        preds: dict[str, torch.Tensor],
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        return pred_distri, pred_scores, pred_bboxes, target_bboxes, target_scores, fg_mask, target_gt_idx

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        image_h, image_w = float(imgsz[0].item()), float(imgsz[1].item())

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        shift_term = torch.zeros((), device=self.device, dtype=dtype)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

            if self.shift_lambda > 0.0 and torch.is_grad_enabled():
                # One cardinal one-pixel phase change per batch. Sampling uses the
                # seeded Torch RNG, so the screening run is reproducible up to the
                # existing deterministic=False protocol.
                shifts = ((1, 0), (-1, 0), (0, 1), (0, -1))
                shift_id = int(torch.randint(0, len(shifts), (1,), device=self.device).item())
                dx, dy = shifts[shift_id]
                dx *= self.max_shift_px
                dy *= self.max_shift_px

                shifted_img = shift_image(batch["img"].to(self.device, non_blocking=True), dx, dy)
                with _freeze_bn_running_stats(self.student_model):
                    shifted_preds = self._raw_predictions(self.student_model(shifted_img))

                shifted_anchor, shifted_stride = make_anchors(shifted_preds["feats"], self.stride, 0.5)
                shifted_distri = shifted_preds["boxes"].permute(0, 2, 1).contiguous()
                shifted_scores = shifted_preds["scores"].permute(0, 2, 1).contiguous()
                shifted_pred_bboxes = self.bbox_decode(shifted_anchor, shifted_distri)

                shifted_gt = gt_bboxes.clone()
                shifted_gt[..., (0, 2)] += float(dx)
                shifted_gt[..., (1, 3)] += float(dy)

                boundary_ok = (
                    (shifted_gt[..., 0] >= 0.0)
                    & (shifted_gt[..., 1] >= 0.0)
                    & (shifted_gt[..., 2] <= image_w)
                    & (shifted_gt[..., 3] <= image_h)
                )
                shifted_mask_gt = mask_gt & boundary_ok.unsqueeze(-1)
                shifted_gt[..., (0, 2)] = shifted_gt[..., (0, 2)].clamp(0.0, image_w)
                shifted_gt[..., (1, 3)] = shifted_gt[..., (1, 3)].clamp(0.0, image_h)

                _, shifted_target_bboxes, shifted_target_scores, shifted_fg, shifted_gt_idx = self.assigner(
                    shifted_scores.detach().sigmoid(),
                    (shifted_pred_bboxes.detach() * shifted_stride).type(shifted_gt.dtype),
                    shifted_anchor * shifted_stride,
                    gt_labels,
                    shifted_gt,
                    shifted_mask_gt,
                )

                max_gt = int(gt_bboxes.shape[1])
                original_px = pred_bboxes * stride_tensor
                shifted_px = shifted_pred_bboxes * shifted_stride
                orig_rep, orig_valid = object_representatives(
                    original_px, fg_mask, target_gt_idx, target_scores, max_gt
                )
                shift_rep, shift_valid = object_representatives(
                    shifted_px, shifted_fg, shifted_gt_idx, shifted_target_scores, max_gt
                )

                if max_gt > 0:
                    gt_flat = gt_bboxes.reshape(-1, 4)
                    shifted_gt_flat = shifted_gt.reshape(-1, 4)
                    gt_valid = mask_gt.squeeze(-1).reshape(-1)
                    boundary_flat = boundary_ok.reshape(-1)
                    widths = (gt_flat[:, 2] - gt_flat[:, 0]).clamp_min(0.0)
                    heights = (gt_flat[:, 3] - gt_flat[:, 1]).clamp_min(0.0)
                    tiny = torch.minimum(widths, heights) < self.tiny_threshold
                    matched = orig_valid & shift_valid & gt_valid & boundary_flat & tiny

                    if matched.any():
                        orig_boxes = orig_rep[matched]
                        shifted_boxes = shift_rep[matched]
                        orig_q = bbox_iou(
                            orig_boxes.detach(), gt_flat[matched], xywh=False, CIoU=False
                        ).reshape(-1).clamp_(0.0, 1.0)
                        shift_q = bbox_iou(
                            shifted_boxes.detach(), shifted_gt_flat[matched], xywh=False, CIoU=False
                        ).reshape(-1).clamp_(0.0, 1.0)
                        reliability = torch.sqrt((orig_q * shift_q).clamp_min(0.0)).detach()
                        shift_term = shift_equivariance_penalty(
                            orig_boxes, shifted_boxes, reliability, dx, dy
                        )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[0] = loss[0] + self.shift_lambda * shift_term

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )
