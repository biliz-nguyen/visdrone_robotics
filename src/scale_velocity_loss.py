from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


def _normalize_hyp_for_loss(hyp: Any) -> Any:
    """Return a loss-local hyperparameter object with attribute access.

    Depending on how pinned Ultralytics is re-imported in the paired C12 runner,
    ``model.args`` can reach ``v8DetectionLoss`` either as an attribute-style
    namespace or as a plain dict. Stock loss code expects ``hyp.box/cls/dfl``.
    Convert only the criterion-local reference when needed; do not mutate
    ``model.args`` itself.
    """
    if isinstance(hyp, dict):
        return SimpleNamespace(**hyp)
    return hyp


def scale_group_from_min_side(min_side_px: torch.Tensor, tiny_thr: float = 16.0, small_thr: float = 32.0) -> torch.Tensor:
    """Map GT minimum side length to tiny/small/regular groups 0/1/2."""
    if not (0.0 < float(tiny_thr) < float(small_thr)):
        raise ValueError("Require 0 < tiny_thr < small_thr")
    group = torch.full_like(min_side_px, 2, dtype=torch.long)
    group[min_side_px < float(small_thr)] = 1
    group[min_side_px < float(tiny_thr)] = 0
    return group


def compute_velocity_weights(
    progress: torch.Tensor,
    counts: torch.Tensor,
    alpha: float = 0.50,
    weight_min: float = 0.75,
    weight_max: float = 1.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert scale-wise residual learning progress to mean-preserving weights.

    progress is current difficulty / first-epoch reference difficulty. A larger
    value means that scale group is learning more slowly. Equal progress gives
    exactly equal weights. The count-weighted mean is kept near one so C12
    redistributes positive classification gradient instead of globally scaling it.
    """
    progress = progress.detach().float().clamp_min(eps)
    counts = counts.detach().float().clamp_min(0.0)
    if progress.numel() != 3 or counts.numel() != 3:
        raise ValueError("C12 expects exactly three scale groups")
    if float(alpha) <= 0:
        raise ValueError("alpha must be > 0")
    if not (0 < float(weight_min) <= 1.0 <= float(weight_max)):
        raise ValueError("Require 0 < weight_min <= 1 <= weight_max")

    present = counts > 0
    out = torch.ones_like(progress)
    if int(present.sum()) <= 1:
        return out

    log_center = progress[present].log().mean()
    raw = torch.exp((progress.log() - log_center) * float(alpha))
    raw = raw.clamp(float(weight_min), float(weight_max))

    mean_w = (raw[present] * counts[present]).sum() / counts[present].sum().clamp_min(1.0)
    raw = (raw / mean_w.clamp_min(eps)).clamp(float(weight_min), float(weight_max))
    out[present] = raw[present]
    return out


class TemporalScaleVelocityDetectionLoss(v8DetectionLoss):
    """Stock TAL loss with training-only scale learning-velocity equalization.

    C12 treats object scale as three within-task learning streams. Epoch 1 is a
    calibration epoch with stock loss. From epoch 2 onward, an EMA tracks the
    assigned-class confidence difficulty of tiny (<16 px), small (16-32 px),
    and regular (>=32 px) GTs. A group that retains more of its first-epoch
    difficulty receives a temporary positive-class gradient boost; faster groups
    are reduced. No class identity, assigner, box loss, model module, or inference
    graph is changed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tiny_thr: float = 16.0,
        small_thr: float = 32.0,
        ema_beta: float = 0.95,
        velocity_alpha: float = 0.50,
        weight_min: float = 0.75,
        weight_max: float = 1.25,
    ):
        super().__init__(model)
        # Paired control -> candidate re-imports can leave model.args as a dict.
        # Normalize only this criterion's reference so stock-style hyp.box/cls/dfl
        # access remains valid without mutating the detector configuration.
        self.hyp = _normalize_hyp_for_loss(self.hyp)

        if not (0.0 < tiny_thr < small_thr):
            raise ValueError("Require 0 < tiny_thr < small_thr")
        if not (0.0 <= ema_beta < 1.0):
            raise ValueError("ema_beta must be in [0,1)")

        self.tiny_thr = float(tiny_thr)
        self.small_thr = float(small_thr)
        self.ema_beta = float(ema_beta)
        self.velocity_alpha = float(velocity_alpha)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)

        self.current_epoch = -1
        self.reference = torch.full((3,), float("nan"), device=self.device)
        self.ema_difficulty = torch.full((3,), float("nan"), device=self.device)
        self.calib_sum = torch.zeros(3, device=self.device)
        self.calib_count = torch.zeros(3, device=self.device)
        self.epoch_difficulty_sum = torch.zeros(3, device=self.device)
        self.epoch_count = torch.zeros(3, device=self.device)
        self.epoch_weight_sum = torch.zeros(3, device=self.device)
        self.last_progress = torch.ones(3, device=self.device)
        self.last_weights = torch.ones(3, device=self.device)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        if epoch == self.current_epoch:
            return

        # Epoch 0 is pure stock-loss calibration. Freeze its group difficulty as
        # the temporal reference exactly when epoch 1 begins.
        if self.current_epoch == 0 and epoch >= 1:
            valid = self.calib_count > 0
            ref = torch.ones(3, device=self.device)
            ref[valid] = self.calib_sum[valid] / self.calib_count[valid]
            ref = ref.clamp_min(1e-6)
            self.reference = ref.detach()
            self.ema_difficulty = ref.detach().clone()

        self.current_epoch = epoch
        self.epoch_difficulty_sum.zero_()
        self.epoch_count.zero_()
        self.epoch_weight_sum.zero_()

    def epoch_summary(self) -> dict[str, Any]:
        count = self.epoch_count.detach().cpu()
        diff = torch.zeros(3)
        w = torch.ones(3)
        valid = count > 0
        diff[valid] = self.epoch_difficulty_sum.detach().cpu()[valid] / count[valid]
        w[valid] = self.epoch_weight_sum.detach().cpu()[valid] / count[valid]
        return {
            "epoch_index": int(self.current_epoch),
            "group_names": ["tiny_lt16", "small_16_32", "regular_ge32"],
            "counts": [float(x) for x in count.tolist()],
            "mean_confidence_difficulty": [float(x) for x in diff.tolist()],
            "mean_applied_weight": [float(x) for x in w.tolist()],
            "reference": [float(x) for x in torch.nan_to_num(self.reference.detach().cpu(), nan=0.0).tolist()],
            "ema_difficulty": [float(x) for x in torch.nan_to_num(self.ema_difficulty.detach().cpu(), nan=0.0).tolist()],
            "last_progress": [float(x) for x in self.last_progress.detach().cpu().tolist()],
            "last_weights": [float(x) for x in self.last_weights.detach().cpu().tolist()],
        }

    def _update_and_get_group_weights(self, group: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
        counts = torch.stack([(group == g).sum() for g in range(3)]).to(dtype=torch.float32, device=self.device)
        means = torch.zeros(3, device=self.device)
        for g in range(3):
            mask = group == g
            if bool(mask.any()):
                means[g] = difficulty[mask].mean()

        if self.current_epoch <= 0 or not bool(torch.isfinite(self.reference).all()):
            for g in range(3):
                if counts[g] > 0:
                    self.calib_sum[g] += difficulty[group == g].sum()
                    self.calib_count[g] += counts[g]
            weights = torch.ones(3, device=self.device)
            progress = torch.ones(3, device=self.device)
        else:
            for g in range(3):
                if counts[g] > 0:
                    if not bool(torch.isfinite(self.ema_difficulty[g])):
                        self.ema_difficulty[g] = means[g].detach()
                    else:
                        self.ema_difficulty[g] = (
                            self.ema_beta * self.ema_difficulty[g]
                            + (1.0 - self.ema_beta) * means[g].detach()
                        )
            progress = (self.ema_difficulty / self.reference.clamp_min(1e-6)).clamp_min(1e-6)
            weights = compute_velocity_weights(
                progress,
                counts,
                alpha=self.velocity_alpha,
                weight_min=self.weight_min,
                weight_max=self.weight_max,
            ).to(self.device)

        self.last_progress = progress.detach()
        self.last_weights = weights.detach()
        for g in range(3):
            mask = group == g
            if bool(mask.any()):
                self.epoch_difficulty_sum[g] += difficulty[mask].sum()
                self.epoch_count[g] += mask.sum()
                self.epoch_weight_sum[g] += weights[g] * mask.sum()
        return weights

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
        cls_sum = bce_loss.sum()

        if fg_mask.sum():
            # Resolve each TAL-positive anchor back to its assigned GT and true class.
            batch_grid = torch.arange(batch_size, device=self.device)[:, None].expand_as(fg_mask)
            pos_bi = batch_grid[fg_mask]
            pos_gt_idx = target_gt_idx[fg_mask].long()
            pos_gt_boxes = gt_bboxes[pos_bi, pos_gt_idx]
            pos_gt_cls = gt_labels[pos_bi, pos_gt_idx, 0].long().clamp_(0, self.nc - 1)

            wh = (pos_gt_boxes[:, 2:] - pos_gt_boxes[:, :2]).clamp_min(0.0)
            min_side = torch.minimum(wh[:, 0], wh[:, 1])
            group = scale_group_from_min_side(min_side, self.tiny_thr, self.small_thr)

            pos_logits = pred_scores[fg_mask].gather(1, pos_gt_cls[:, None]).squeeze(1)
            confidence_difficulty = F.softplus(-pos_logits.detach())  # BCE(target=1), detached statistic only
            group_weights = self._update_and_get_group_weights(group, confidence_difficulty)
            sample_weights = group_weights[group].to(dtype)

            # Modify only the stock BCE term of the assigned true class. All
            # other class logits, negatives, TAL targets, box and direct-reg1
            # objectives remain exactly stock.
            pos_stock_bce = bce_loss[fg_mask].gather(1, pos_gt_cls[:, None]).squeeze(1)
            cls_sum = cls_sum + ((sample_weights - 1.0) * pos_stock_bce).sum()

        loss[1] = cls_sum / target_scores_sum

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

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )
