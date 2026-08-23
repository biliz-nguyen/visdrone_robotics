from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors


def adaptive_qoc_margin(
    min_side_px: torch.Tensor,
    base_margin: float,
    tiny_threshold: float,
    tiny_margin_bonus: float,
) -> torch.Tensor:
    """Continuous tiny tolerance; objects >= threshold exactly recover v1 margin."""
    ratio = (min_side_px / float(tiny_threshold)).clamp(0.0, 1.0)
    return float(base_margin) + float(tiny_margin_bonus) * (1.0 - ratio)


def quality_overconfidence_penalty(
    confidence: torch.Tensor,
    quality: torch.Tensor,
    weight: torch.Tensor,
    margin: torch.Tensor | float,
    normalizer: torch.Tensor | float,
) -> torch.Tensor:
    """One-sided penalty: only confidence that outruns localization quality is suppressed."""
    overconfidence = F.relu(confidence - (quality + margin))
    return (overconfidence.square() * weight).sum() / normalizer


class QualityOverconfidenceLoss(v8DetectionLoss):
    """Stock YOLO detection loss plus tiny-aware one-sided QOC calibration.

    Standard TAL soft labels, classification BCE, CIoU and direct-regression
    losses remain unchanged. QOC only acts on assigned positives and only sends
    gradient to the classification logits. Localization IoU is detached.

    v2 changes one principle from v1: localization tolerance depends on the
    assigned GT min-side in image pixels. Tiny objects get a larger allowed
    confidence-vs-IoU gap because a few pixels can sharply reduce IoU. Objects
    at or above the threshold use the exact v1 margin.
    """

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        head = model.model[-1]
        self.qoc_lambda = float(head.qoc_lambda)
        self.qoc_margin = float(head.qoc_margin)
        self.qoc_tiny_threshold = float(head.qoc_tiny_threshold)
        self.qoc_tiny_margin_bonus = float(head.qoc_tiny_margin_bonus)
        if self.reg_max != 1:
            raise ValueError("QOC is locked to the DFL-free reg_max=1 control")

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)  # box, cls(+QOC), direct-reg
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

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

        # Exact stock classification loss.
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.sum():
            # Exact stock box + direct-regression loss.
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

            target_bboxes_feat = target_bboxes / stride_tensor
            quality = bbox_iou(
                pred_bboxes[fg_mask].detach(),
                target_bboxes_feat[fg_mask],
                xywh=False,
                CIoU=False,
            ).reshape(-1).clamp_(0.0, 1.0)

            pos_targets = target_scores[fg_mask]
            assigned_cls = pos_targets.argmax(dim=-1)
            pos_logits = pred_scores[fg_mask].gather(1, assigned_cls[:, None]).squeeze(1)
            confidence = pos_logits.sigmoid()
            pos_weight = pos_targets.sum(-1).detach()

            # target_bboxes is in image pixels at this point. The min-side is
            # therefore independent of pyramid level and directly interpretable.
            pos_boxes_px = target_bboxes[fg_mask]
            width_px = (pos_boxes_px[:, 2] - pos_boxes_px[:, 0]).clamp_min(0.0)
            height_px = (pos_boxes_px[:, 3] - pos_boxes_px[:, 1]).clamp_min(0.0)
            min_side_px = torch.minimum(width_px, height_px).detach()
            margin = adaptive_qoc_margin(
                min_side_px,
                base_margin=self.qoc_margin,
                tiny_threshold=self.qoc_tiny_threshold,
                tiny_margin_bonus=self.qoc_tiny_margin_bonus,
            )

            qoc = quality_overconfidence_penalty(
                confidence,
                quality,
                pos_weight,
                margin=margin,
                normalizer=target_scores_sum,
            )
            loss[1] = loss[1] + self.qoc_lambda * qoc

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )
