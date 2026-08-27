from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss


class P2TinyAuxDetectionLoss(v8DetectionLoss):
    """Stock detection loss plus training-only P2 tiny-center positive supervision.

    The model graph is unchanged. The auxiliary term reuses the existing raw
    P2 classification logits and only adds a mild positive BCE term at the
    center cell of tiny pedestrian/people targets. No auxiliary module or
    parameter exists in the model, so inference/export remain identical to the
    frozen N2b Detect graph.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tiny_min_side: float = 16.0,
        aux_weight: float = 0.10,
        focus_classes: tuple[int, ...] = (5, 6),
    ):
        super().__init__(model)
        if float(tiny_min_side) <= 0:
            raise ValueError("tiny_min_side must be > 0")
        if not (0.0 < float(aux_weight) <= 1.0):
            raise ValueError("aux_weight must be in (0, 1]")
        if not focus_classes:
            raise ValueError("focus_classes cannot be empty")
        self.tiny_min_side = float(tiny_min_side)
        self.aux_weight = float(aux_weight)
        self.focus_classes = tuple(int(x) for x in focus_classes)
        self.last_aux_raw = torch.tensor(0.0, device=self.device)
        self.last_aux_scaled = torch.tensor(0.0, device=self.device)
        self.last_aux_positive_count = 0

    def _tiny_center_positive_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        """Return positive-only BCE on unique tiny focus centers at P2."""
        feats = preds["feats"]
        if len(feats) < 1:
            raise RuntimeError("P2 auxiliary loss requires at least one feature level")

        p2_h, p2_w = int(feats[0].shape[-2]), int(feats[0].shape[-1])
        p2_count = p2_h * p2_w
        scores = preds["scores"]
        if scores.ndim != 3 or scores.shape[2] < p2_count:
            raise RuntimeError(f"Unexpected score shape for P2 auxiliary loss: {tuple(scores.shape)}")

        # Detect concatenates levels in P2/P3/P4 order. Only the first H*W
        # anchors are used by the auxiliary term.
        p2_scores = scores[:, :, :p2_count].reshape(scores.shape[0], self.nc, p2_h, p2_w)

        bboxes = batch["bboxes"].to(device=self.device, dtype=p2_scores.dtype)
        cls = batch["cls"].view(-1).to(device=self.device, dtype=torch.long)
        batch_idx = batch["batch_idx"].view(-1).to(device=self.device, dtype=torch.long)
        if bboxes.numel() == 0:
            return p2_scores.sum() * 0.0, 0

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
            return p2_scores.sum() * 0.0, 0

        centers = bboxes[tiny_mask, :2]
        gx = torch.floor(centers[:, 0] * p2_w).to(torch.long).clamp_(0, p2_w - 1)
        gy = torch.floor(centers[:, 1] * p2_h).to(torch.long).clamp_(0, p2_h - 1)
        bi = batch_idx[tiny_mask].clamp_(0, scores.shape[0] - 1)
        ci = cls[tiny_mask].clamp_(0, self.nc - 1)

        # Dense VisDrone scenes can place several same-class centers in one
        # stride-4 cell. Deduplicate them so a crowded cell is not overweighted.
        positive_index = torch.stack((bi, ci, gy, gx), dim=1)
        positive_index = torch.unique(positive_index, dim=0)
        logits = p2_scores[
            positive_index[:, 0],
            positive_index[:, 1],
            positive_index[:, 2],
            positive_index[:, 3],
        ]

        # Positive-only BCE directly encourages recall at tiny focus centers.
        # No negative auxiliary term is added, so stock TAL/BCE remains the
        # sole source of background suppression.
        aux = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="mean")
        return aux, int(positive_index.shape[0])

    def loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the auxiliary term into the classification loss slot."""
        batch_size = preds["boxes"].shape[0]
        loss, _ = self.get_assigned_targets_and_loss(preds, batch)[1:]

        aux_raw, positive_count = self._tiny_center_positive_loss(preds, batch)
        aux_scaled = aux_raw * float(self.hyp.cls) * self.aux_weight
        loss[1] = loss[1] + aux_scaled

        self.last_aux_raw = aux_raw.detach()
        self.last_aux_scaled = aux_scaled.detach()
        self.last_aux_positive_count = int(positive_count)
        return loss * batch_size, loss.detach()
