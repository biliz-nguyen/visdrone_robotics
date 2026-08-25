from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import dist2bbox, make_anchors


def advantage_gate(
    student_iou: torch.Tensor,
    teacher_iou: torch.Tensor,
    min_side_px: torch.Tensor,
    tiny_threshold: float,
    advantage_margin: float,
    min_teacher_iou: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the tiny-object mentor candidate gate and detached teacher advantage."""
    advantage = (teacher_iou - student_iou - float(advantage_margin)).clamp_min(0.0)
    mask = (
        (min_side_px < float(tiny_threshold))
        & (teacher_iou >= float(min_teacher_iou))
        & (advantage > 0)
    )
    return mask, advantage.detach()


def object_normalized_weights(
    candidate_gate: torch.Tensor,
    advantage: torch.Tensor,
    tal_weight: torch.Tensor,
    positive_batch_idx: torch.Tensor,
    positive_gt_idx: torch.Tensor,
) -> torch.Tensor:
    """Distribute one object-level mentor budget across all eligible positives.

    AGTL v3 keeps every eligible positive from v1, but removes the dependence of
    total mentor pressure on the number of TAL positives assigned to an object.

    For each (image, assigned-GT) object:
      * raw per-positive weight = TAL weight * teacher advantage;
      * choose the v2 reference location with maximum teacher advantage;
      * define the object's total mentor budget as that reference raw weight;
      * redistribute that budget over all eligible positives in proportion to
        their raw weights.

    Therefore the sum of returned weights for one object equals the mentor
    weight that AGTL v2 would have assigned to its single selected location,
    while the localization signal is still spread over all useful positives.
    """
    tensors = (candidate_gate, advantage, tal_weight, positive_batch_idx, positive_gt_idx)
    if any(t.ndim != 1 for t in tensors):
        raise ValueError("object-normalized AGTL inputs must be 1-D")
    n = candidate_gate.numel()
    if any(t.numel() != n for t in tensors[1:]):
        raise ValueError("object-normalized AGTL inputs must have equal lengths")

    weights = torch.zeros_like(advantage)
    eligible_idx = torch.nonzero(candidate_gate, as_tuple=False).flatten()
    if eligible_idx.numel() == 0:
        return weights

    batch_idx = positive_batch_idx.to(device=advantage.device, dtype=torch.long)
    gt_idx = positive_gt_idx.to(device=advantage.device, dtype=torch.long)
    if torch.any(batch_idx < 0) or torch.any(gt_idx < 0):
        raise ValueError("positive batch/GT indices must be non-negative")

    gt_stride = int(gt_idx.max().item()) + 1
    object_key = batch_idx * max(gt_stride, 1) + gt_idx
    eligible_key = object_key[eligible_idx]
    raw = (tal_weight.detach() * advantage.detach()).clamp_min(0.0)

    # Process unique object ids. The number of objects per training image is much
    # smaller than the number of dense positive locations, and this loop keeps
    # the reference rule explicit and deterministic.
    for key in torch.unique(eligible_key):
        local = eligible_idx[eligible_key == key]
        local_adv = advantage[local]
        # torch.argmax returns the first maximum, matching v2's deterministic
        # tie behaviour for its maximum-advantage reference location.
        ref = local[torch.argmax(local_adv)]
        object_budget = raw[ref]
        raw_sum = raw[local].sum()
        if raw_sum > 0:
            weights[local] = raw[local] / raw_sum * object_budget

    return weights.detach()


def mentor_transfer_enabled(mentor_lambda: float) -> bool:
    """Run the frozen mentor only during gradient-enabled training."""
    return float(mentor_lambda) > 0.0 and torch.is_grad_enabled()


class AdvantageGatedTinyLocalizationLoss(v8DetectionLoss):
    """Stock H1 loss plus object-normalized tiny localization mentor transfer.

    AGTL v3 keeps the original capacity bridge: a richer mentor regression
    representation (e.g. DFL16) is decoded to continuous boxes before transfer,
    so the reg_max=1 student never mimics mentor bins, features, or logits.

    Relative to the earlier variants:
      * v1 transfers every eligible positive independently;
      * v2 keeps only one maximum-advantage positive per object;
      * v3 keeps all eligible positives but normalizes their combined mentor
        budget per object to the single-location budget used by v2.

    The mentor is training-only and is never part of the exported student graph.
    """

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        head = model.model[-1]
        if self.reg_max != 1:
            raise ValueError("AGTL v3 is locked to the DFL-free reg_max=1 student")

        self.mentor_lambda = float(head.mentor_lambda)
        self.tiny_threshold = float(head.tiny_threshold)
        self.advantage_margin = float(head.advantage_margin)
        self.min_teacher_iou = float(head.min_teacher_iou)

        mentor_path = os.environ.get("YOLOEDGE27_MENTOR_PT", "").strip()
        if not mentor_path:
            raise RuntimeError("YOLOEDGE27_MENTOR_PT must point to a frozen mentor checkpoint")
        self.mentor_path = str(Path(mentor_path).expanduser().resolve())
        if not Path(self.mentor_path).is_file():
            raise FileNotFoundError(self.mentor_path)

        self._mentor_model: torch.nn.Module | None = None
        self._mentor_reg_max: int | None = None
        self._mentor_proj: torch.Tensor | None = None

    def _ensure_mentor(self, device: torch.device) -> torch.nn.Module:
        if self._mentor_model is None:
            from ultralytics import YOLO

            wrapper = YOLO(self.mentor_path)
            mentor = wrapper.model.to(device).float().eval()
            for p in mentor.parameters():
                p.requires_grad_(False)

            mentor_head = mentor.model[-1]
            if mentor_head.__class__.__name__ not in {"Detect", "AdvantageGatedMentorDetect"}:
                raise TypeError(f"Unsupported mentor head: {mentor_head.__class__.__name__}")
            if int(mentor_head.nc) != int(self.nc):
                raise ValueError(f"Mentor classes {mentor_head.nc} != student classes {self.nc}")
            if [int(x) for x in mentor_head.stride.tolist()] != [int(x) for x in self.stride.tolist()]:
                raise ValueError("Mentor/student strides must match for anchor-aligned AGTL")

            self._mentor_model = mentor
            self._mentor_reg_max = int(mentor_head.reg_max)
            self._mentor_proj = torch.arange(self._mentor_reg_max, dtype=torch.float, device=device)
            print("AGTL v3 mentor loaded:", self.mentor_path, "reg_max=", self._mentor_reg_max)
        return self._mentor_model

    @staticmethod
    def _raw_predictions(output: Any) -> dict[str, torch.Tensor]:
        raw = output[1] if isinstance(output, tuple) else output
        if not isinstance(raw, dict) or "boxes" not in raw or "scores" not in raw or "feats" not in raw:
            raise TypeError("Mentor forward did not return a compatible raw detection dictionary")
        return raw

    def _decode_mentor(
        self,
        mentor_preds: dict[str, torch.Tensor],
        anchor_points: torch.Tensor,
    ) -> torch.Tensor:
        pred_dist = mentor_preds["boxes"].permute(0, 2, 1).contiguous()
        reg_max = int(self._mentor_reg_max)
        if reg_max > 1:
            b, a, c = pred_dist.shape
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self._mentor_proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

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

        mentor_term = torch.zeros((), device=self.device, dtype=dtype)
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

            if mentor_transfer_enabled(self.mentor_lambda):
                mentor = self._ensure_mentor(self.device)
                mentor_dtype = next(mentor.parameters()).dtype
                mentor_img = batch["img"].to(self.device, dtype=mentor_dtype, non_blocking=True)
                with torch.no_grad():
                    mentor_out = mentor(mentor_img)
                    mentor_preds = self._raw_predictions(mentor_out)

                student_shapes = [(int(f.shape[-2]), int(f.shape[-1])) for f in preds["feats"]]
                mentor_shapes = [(int(f.shape[-2]), int(f.shape[-1])) for f in mentor_preds["feats"]]
                if mentor_shapes != student_shapes:
                    raise ValueError(f"Mentor/student feature grids differ: {mentor_shapes} vs {student_shapes}")

                mentor_bboxes = self._decode_mentor(mentor_preds, anchor_points).detach()
                target_bboxes_feat = target_bboxes / stride_tensor

                student_pos = pred_bboxes[fg_mask]
                mentor_pos = mentor_bboxes[fg_mask]
                target_pos = target_bboxes_feat[fg_mask]

                student_iou = bbox_iou(
                    student_pos.detach(), target_pos, xywh=False, CIoU=False
                ).reshape(-1).clamp_(0.0, 1.0)
                teacher_iou = bbox_iou(
                    mentor_pos, target_pos, xywh=False, CIoU=False
                ).reshape(-1).clamp_(0.0, 1.0)

                target_px = target_bboxes[fg_mask]
                width_px = (target_px[:, 2] - target_px[:, 0]).clamp_min(0.0)
                height_px = (target_px[:, 3] - target_px[:, 1]).clamp_min(0.0)
                min_side_px = torch.minimum(width_px, height_px).detach()

                candidate_gate, advantage = advantage_gate(
                    student_iou,
                    teacher_iou,
                    min_side_px,
                    tiny_threshold=self.tiny_threshold,
                    advantage_margin=self.advantage_margin,
                    min_teacher_iou=self.min_teacher_iou,
                )

                positive_locations = torch.nonzero(fg_mask, as_tuple=False)
                positive_batch_idx = positive_locations[:, 0]
                positive_gt_idx = target_gt_idx[fg_mask]
                tal_weight_all = target_scores.sum(-1)[fg_mask].detach()
                mentor_weight = object_normalized_weights(
                    candidate_gate,
                    advantage,
                    tal_weight_all,
                    positive_batch_idx,
                    positive_gt_idx,
                )
                gate = mentor_weight > 0

                if gate.any():
                    mentor_ciou = bbox_iou(
                        student_pos[gate],
                        mentor_pos[gate],
                        xywh=False,
                        CIoU=True,
                    ).reshape(-1)
                    mentor_term = (
                        (1.0 - mentor_ciou) * mentor_weight[gate]
                    ).sum() / target_scores_sum

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[0] = loss[0] + self.mentor_lambda * mentor_term

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )
