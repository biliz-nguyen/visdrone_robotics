from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import make_anchors


class P2QualityAuxDetectionLoss(v8DetectionLoss):
    """Stock detection loss plus training-only quality-aware P2 supervision.

    The model/inference graph is unchanged. Existing P2 classification logits
    receive a soft target only for tiny pedestrian/people center cells. The
    target is tempered by the detached IoU of the prediction at the same P2
    cell so poorly localized boxes are not forced toward confidence 1.0.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tiny_min_side: float = 16.0,
        aux_weight: float = 0.10,
        focus_classes: tuple[int, ...] = (5, 6),
        target_floor: float = 0.50,
        quality_gamma: float = 0.50,
    ):
        super().__init__(model)
        if float(tiny_min_side) <= 0:
            raise ValueError("tiny_min_side must be > 0")
        if not (0.0 < float(aux_weight) <= 1.0):
            raise ValueError("aux_weight must be in (0, 1]")
        if not focus_classes:
            raise ValueError("focus_classes cannot be empty")
        if not (0.0 <= float(target_floor) < 1.0):
            raise ValueError("target_floor must be in [0, 1)")
        if float(quality_gamma) <= 0:
            raise ValueError("quality_gamma must be > 0")

        self.tiny_min_side = float(tiny_min_side)
        self.aux_weight = float(aux_weight)
        self.focus_classes = tuple(int(x) for x in focus_classes)
        self.target_floor = float(target_floor)
        self.quality_gamma = float(quality_gamma)

        self.last_aux_raw = torch.tensor(0.0, device=self.device)
        self.last_aux_scaled = torch.tensor(0.0, device=self.device)
        self.last_aux_positive_count = 0
        self.last_quality_mean = torch.tensor(0.0, device=self.device)
        self.last_target_mean = torch.tensor(0.0, device=self.device)

    @staticmethod
    def _aligned_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Aligned IoU for xyxy boxes of shape [N, 4]."""
        lt = torch.maximum(box1[:, :2], box2[:, :2])
        rb = torch.minimum(box1[:, 2:], box2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
        area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
        return inter / (area1 + area2 - inter + 1e-9)

    def _quality_to_target(self, quality: torch.Tensor) -> torch.Tensor:
        """Map detached IoU quality to a non-suppressive soft target."""
        q = quality.detach().clamp(0.0, 1.0).pow(self.quality_gamma)
        return self.target_floor + (1.0 - self.target_floor) * q

    def _tiny_center_quality_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Return quality-aware BCE for unique tiny focus centers at P2."""
        feats = preds["feats"]
        if len(feats) < 1:
            raise RuntimeError("P2 quality auxiliary loss requires at least one feature level")

        p2_h, p2_w = int(feats[0].shape[-2]), int(feats[0].shape[-1])
        p2_count = p2_h * p2_w
        scores = preds["scores"]
        if scores.ndim != 3 or scores.shape[2] < p2_count:
            raise RuntimeError(f"Unexpected score shape for P2 quality auxiliary loss: {tuple(scores.shape)}")

        p2_scores = scores[:, :, :p2_count].reshape(scores.shape[0], self.nc, p2_h, p2_w)
        bboxes = batch["bboxes"].to(device=self.device, dtype=p2_scores.dtype)
        cls = batch["cls"].view(-1).to(device=self.device, dtype=torch.long)
        batch_idx = batch["batch_idx"].view(-1).to(device=self.device, dtype=torch.long)

        zero = p2_scores.sum() * 0.0
        if bboxes.numel() == 0:
            return zero, 0, zero.detach(), zero.detach()
        if bboxes.shape[-1] != 4:
            raise RuntimeError(f"Expected normalized xywh targets, got shape={tuple(bboxes.shape)}")

        img_h = float(p2_h) * float(self.stride[0])
        img_w = float(p2_w) * float(self.stride[0])
        min_side_px = torch.minimum(bboxes[:, 2] * img_w, bboxes[:, 3] * img_h)

        focus_mask = torch.zeros_like(cls, dtype=torch.bool)
        for class_id in self.focus_classes:
            focus_mask |= cls.eq(class_id)
        tiny_mask = focus_mask & min_side_px.lt(self.tiny_min_side)
        if not bool(tiny_mask.any()):
            return zero, 0, zero.detach(), zero.detach()

        tiny_boxes = bboxes[tiny_mask]
        tiny_cls = cls[tiny_mask]
        tiny_bi = batch_idx[tiny_mask].clamp_(0, scores.shape[0] - 1)

        centers = tiny_boxes[:, :2]
        gx = torch.floor(centers[:, 0] * p2_w).to(torch.long).clamp_(0, p2_w - 1)
        gy = torch.floor(centers[:, 1] * p2_h).to(torch.long).clamp_(0, p2_h - 1)
        flat = gy * p2_w + gx

        # Decode current boxes and measure localization quality at the same P2
        # center cell. Quality is detached before it becomes a classification
        # target, so this auxiliary term cannot optimize regression directly.
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        pred_boxes_px = self.bbox_decode(anchor_points, pred_distri) * stride_tensor
        selected_pred = pred_boxes_px[tiny_bi, flat]

        gt_xywh_px = tiny_boxes.clone()
        gt_xywh_px[:, 0] *= img_w
        gt_xywh_px[:, 2] *= img_w
        gt_xywh_px[:, 1] *= img_h
        gt_xywh_px[:, 3] *= img_h
        selected_gt = xywh2xyxy(gt_xywh_px)
        quality = self._aligned_iou(selected_pred, selected_gt).detach().clamp(0.0, 1.0)

        positive_index = torch.stack((tiny_bi, tiny_cls, gy, gx), dim=1)
        unique_index, inverse = torch.unique(positive_index, dim=0, return_inverse=True)

        # If crowded same-class objects collide in one stride-4 cell, keep the
        # best localization quality for that single classification logit.
        unique_quality = torch.zeros(unique_index.shape[0], device=self.device, dtype=quality.dtype)
        unique_quality.scatter_reduce_(0, inverse, quality, reduce="amax", include_self=True)
        soft_target = self._quality_to_target(unique_quality)

        logits = p2_scores[
            unique_index[:, 0],
            unique_index[:, 1],
            unique_index[:, 2],
            unique_index[:, 3],
        ]
        aux = F.binary_cross_entropy_with_logits(logits, soft_target.to(logits.dtype), reduction="mean")
        return aux, int(unique_index.shape[0]), unique_quality.mean(), soft_target.mean()

    def loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the quality-aware auxiliary term into the classification slot."""
        batch_size = preds["boxes"].shape[0]
        loss, _ = self.get_assigned_targets_and_loss(preds, batch)[1:]

        aux_raw, positive_count, quality_mean, target_mean = self._tiny_center_quality_loss(preds, batch)
        aux_scaled = aux_raw * float(self.hyp.cls) * self.aux_weight
        loss[1] = loss[1] + aux_scaled

        self.last_aux_raw = aux_raw.detach()
        self.last_aux_scaled = aux_scaled.detach()
        self.last_aux_positive_count = int(positive_count)
        self.last_quality_mean = quality_mean.detach()
        self.last_target_mean = target_mean.detach()
        return loss * batch_size, loss.detach()
