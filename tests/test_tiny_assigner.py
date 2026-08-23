from __future__ import annotations

import torch

from src.tiny_assigner import TinyCandidateRecoveryAssigner


def _assigner():
    return TinyCandidateRecoveryAssigner(
        topk=10,
        num_classes=2,
        alpha=0.5,
        beta=6.0,
        stride=[4, 8, 16],
        tiny_min_side=16.0,
        min_candidates=4,
    )


def _grid(step=4, size=24):
    coords = torch.arange(step / 2, size, step)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def test_tiny_gt_recovers_minimum_candidates():
    assigner = _assigner()
    anchors = _grid()

    # This 2x2 box lies between 4-pixel grid centers and has no center inside.
    gt = torch.tensor([[[7.0, 7.0, 9.0, 9.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)

    base = assigner.select_candidates_in_gts(anchors, gt, mask).bool()
    recovered = assigner.recover_candidate_mask(anchors, gt, mask)

    assert int(base.sum()) == 0
    assert int(recovered.sum()) == 4


def test_non_tiny_gt_is_unchanged():
    assigner = _assigner()
    anchors = _grid(size=40)
    gt = torch.tensor([[[4.0, 4.0, 28.0, 28.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)

    base = assigner.select_candidates_in_gts(anchors, gt, mask).bool()
    recovered = assigner.recover_candidate_mask(anchors, gt, mask)

    assert torch.equal(base, recovered)


def test_invalid_gt_is_never_recovered():
    assigner = _assigner()
    anchors = _grid()
    gt = torch.tensor([[[7.0, 7.0, 9.0, 9.0]]])
    mask = torch.zeros((1, 1, 1), dtype=torch.bool)

    recovered = assigner.recover_candidate_mask(anchors, gt, mask)
    assert int(recovered.sum()) == 0


def test_existing_candidates_do_not_consume_recovery_slots():
    assigner = _assigner()
    anchors = _grid()
    # Small box containing exactly one grid center.
    gt = torch.tensor([[[5.0, 5.0, 7.0, 7.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)

    recovered = assigner.recover_candidate_mask(anchors, gt, mask)
    assert int(recovered.sum()) == 4


def test_get_pos_mask_shapes_are_standard_tal_compatible():
    assigner = _assigner()
    anchors = _grid(size=24)
    a = anchors.shape[0]

    gt = torch.tensor([[[7.0, 7.0, 9.0, 9.0]]])
    labels = torch.tensor([[[1.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)
    scores = torch.full((1, a, 2), 0.5)

    # Predicted boxes centered on anchors with a modest 8x8 extent.
    half = 4.0
    boxes = torch.cat((anchors - half, anchors + half), dim=-1).unsqueeze(0)

    mask_pos, align_metric, overlaps = assigner.get_pos_mask(
        scores,
        boxes,
        labels,
        gt,
        anchors,
        mask,
    )

    assert mask_pos.shape == (1, 1, a)
    assert align_metric.shape == (1, 1, a)
    assert overlaps.shape == (1, 1, a)
    assert torch.isfinite(align_metric).all()
    assert torch.isfinite(overlaps).all()
