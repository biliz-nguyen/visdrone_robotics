from __future__ import annotations

import torch

from src.tiny_assigner import TinyCandidateRecoveryAssigner


def _assigner(min_candidates: int = 4):
    return TinyCandidateRecoveryAssigner(
        topk=10,
        num_classes=2,
        alpha=0.5,
        beta=6.0,
        stride=[4, 8, 16],
        tiny_min_side=16.0,
        min_candidates=min_candidates,
    )


def _grid(step=4, size=24):
    coords = torch.arange(step / 2, size, step)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def test_standard_tal_already_expands_extreme_tiny_gt():
    """Ultralytics v8.4.56 already expands sub-stride GTs before center selection."""
    assigner = _assigner()
    anchors = _grid()
    gt = torch.tensor([[[7.0, 7.0, 9.0, 9.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)

    base = assigner.select_candidates_in_gts(anchors, gt, mask).bool()
    recovered = assigner.recover_candidate_mask(anchors, gt, mask)

    # v8.4.56 expands width/height < stride[0] to stride_val (=8 here),
    # yielding four P2 centers for this 2x2 box.
    assert int(base.sum()) == 4
    assert torch.equal(base, recovered)


def test_recovery_can_supplement_when_requested_threshold_is_higher():
    """Exercise recovery without assuming the default TAL is candidate-starved."""
    assigner = _assigner(min_candidates=6)
    anchors = _grid()
    gt = torch.tensor([[[7.0, 7.0, 9.0, 9.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)

    base = assigner.select_candidates_in_gts(anchors, gt, mask).bool()
    recovered = assigner.recover_candidate_mask(anchors, gt, mask)

    assert int(base.sum()) == 4
    assert int(recovered.sum()) == 6


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


def test_get_pos_mask_shapes_are_standard_tal_compatible():
    assigner = _assigner()
    anchors = _grid(size=24)
    a = anchors.shape[0]

    gt = torch.tensor([[[7.0, 7.0, 9.0, 9.0]]])
    labels = torch.tensor([[[1.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)
    scores = torch.full((1, a, 2), 0.5)

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
