from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import DFLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors


class StrideRegDetectionLoss(v8DetectionLoss):
    """YOLO detection loss for a head with level-specific regression bins."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.reg_bins = tuple(int(x) for x in m.reg_bins)
        self.device = device

        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.dfl_losses = [DFLoss(r).to(device) if r > 1 else None for r in self.reg_bins]
        self.proj = [torch.arange(r, dtype=torch.float, device=device) for r in self.reg_bins]

    def _decode_level(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, bins: int, level: int) -> torch.Tensor:
        if bins > 1:
            b, a, c = pred_dist.shape
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj[level].type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _level_slices(self, feats: list[torch.Tensor]) -> list[tuple[int, int]]:
        out = []
        start = 0
        for feat in feats:
            n = int(feat.shape[-2] * feat.shape[-1])
            out.append((start, start + n))
            start += n
        return out

    def get_assigned_targets_and_loss(self, preds: dict[str, Any], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl/direct-reg
        pred_dists = [x.permute(0, 2, 1).contiguous() for x in preds["boxes"]]
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        slices = self._level_slices(preds["feats"])

        decoded_levels = []
        for i, ((start, end), pred_i, bins) in enumerate(zip(slices, pred_dists, self.reg_bins)):
            decoded_levels.append(
                self._decode_level(anchor_points[start:end], pred_i, bins, i)
            )
        pred_bboxes = torch.cat(decoded_levels, dim=1)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

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

        if fg_mask.sum():
            target_bboxes_feat = target_bboxes / stride_tensor
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            iou = bbox_iou(
                pred_bboxes[fg_mask],
                target_bboxes_feat[fg_mask],
                xywh=False,
                CIoU=True,
            )
            loss[0] = ((1.0 - iou) * weight).sum() / target_scores_sum

            dfl_total = torch.zeros((), device=self.device, dtype=pred_scores.dtype)
            for i, ((start, end), pred_i, bins) in enumerate(zip(slices, pred_dists, self.reg_bins)):
                fg_i = fg_mask[:, start:end]
                if not fg_i.any():
                    continue

                weight_i = target_scores[:, start:end].sum(-1)[fg_i].unsqueeze(-1)
                anchor_i = anchor_points[start:end]
                stride_i = stride_tensor[start:end].view(1, -1, 1)
                target_i = target_bboxes[:, start:end] / stride_i

                if bins > 1:
                    target_ltrb = bbox2dist(anchor_i, target_i, bins - 1)
                    level_loss = (
                        self.dfl_losses[i](
                            pred_i[fg_i].view(-1, bins),
                            target_ltrb[fg_i],
                        )
                        * weight_i
                    ).sum()
                else:
                    target_ltrb = bbox2dist(anchor_i, target_i)
                    target_px = target_ltrb * stride_i
                    target_px[..., 0::2] /= imgsz[1]
                    target_px[..., 1::2] /= imgsz[0]

                    pred_direct = pred_i.view(batch_size, end - start, 4) * stride_i
                    pred_direct[..., 0::2] /= imgsz[1]
                    pred_direct[..., 1::2] /= imgsz[0]
                    level_loss = (
                        F.l1_loss(
                            pred_direct[fg_i],
                            target_px[fg_i],
                            reduction="none",
                        )
                        .mean(-1, keepdim=True)
                        .mul(weight_i)
                        .sum()
                    )

                dfl_total = dfl_total + level_loss

            loss[2] = dfl_total / target_scores_sum

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )

    def loss(self, preds: dict[str, Any], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = preds["boxes"][0].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach
