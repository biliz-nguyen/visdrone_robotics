from __future__ import annotations

import torch

from ultralytics.utils.tal import TaskAlignedAssigner

from src.tiny_center_stable_rank import TinyCenterStableRankAssigner


def _assigner(cls, topk=1):
    extra = {"tiny_min_side": 16.0} if cls is TinyCenterStableRankAssigner else {}
    return cls(
        topk=topk,
        num_classes=1,
        alpha=0.5,
        beta=6.0,
        stride=[4, 8, 16],
        topk2=topk,
        **extra,
    )


def test_tiny_weight_is_monotonic_and_zero_at_threshold():
    tcsr = _assigner(TinyCenterStableRankAssigner)
    boxes = torch.tensor(
        [[[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 32.0, 32.0]]]
    )
    lam = tcsr.tiny_weight(boxes).flatten()
    expected = torch.tensor([0.75, 0.50, 0.0, 0.0])
    assert torch.allclose(lam, expected, atol=1e-7)
    assert torch.all(lam[:-1] >= lam[1:])


def test_center_prior_prefers_central_anchor():
    tcsr = _assigner(TinyCenterStableRankAssigner)
    gt = torch.tensor([[[0.0, 0.0, 8.0, 8.0]]])
    anchors = torch.tensor([[1.0, 4.0], [4.0, 4.0]])
    valid = torch.ones((1, 1, 2))
    prior = tcsr.center_prior(anchors, gt, valid)
    assert prior.shape == (1, 1, 2)
    assert prior[0, 0, 1] > prior[0, 0, 0]
    assert torch.allclose(prior[0, 0, 1], torch.tensor(1.0), atol=1e-7)


def test_non_tiny_assignment_is_exactly_standard_tal():
    std = _assigner(TaskAlignedAssigner)
    tcsr = _assigner(TinyCenterStableRankAssigner)

    pd_scores = torch.tensor([[[0.95], [0.80]]], dtype=torch.float32)
    pd_bboxes = torch.tensor(
        [[[0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 16.0, 16.0]]], dtype=torch.float32
    )
    anc_points = torch.tensor([[2.0, 8.0], [8.0, 8.0]], dtype=torch.float32)
    gt_labels = torch.zeros((1, 1, 1), dtype=torch.long)
    gt_bboxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]], dtype=torch.float32)
    mask_gt = torch.ones((1, 1, 1), dtype=torch.float32)

    out_std = std(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    out_tcsr = tcsr(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    for a, b in zip(out_std, out_tcsr):
        assert torch.equal(a, b)


def test_tiny_ranking_can_replace_boundary_positive_with_center_positive():
    """TCSR changes top-k membership, not post-assignment amplitude."""
    std = _assigner(TaskAlignedAssigner, topk=1)
    tcsr = _assigner(TinyCenterStableRankAssigner, topk=1)

    # Both predictions have perfect IoU. Standard TAL prefers anchor 0 because
    # it has the higher class score; TCSR should prefer the central anchor 1.
    pd_scores = torch.tensor([[[0.99], [0.80]]], dtype=torch.float32)
    pd_bboxes = torch.tensor(
        [[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0]]], dtype=torch.float32
    )
    anc_points = torch.tensor([[1.0, 4.0], [4.0, 4.0]], dtype=torch.float32)
    gt_labels = torch.zeros((1, 1, 1), dtype=torch.long)
    gt_bboxes = torch.tensor([[[0.0, 0.0, 8.0, 8.0]]], dtype=torch.float32)
    mask_gt = torch.ones((1, 1, 1), dtype=torch.float32)

    out_std = std(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    out_tcsr = tcsr(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    fg_std = out_std[3]
    fg_tcsr = out_tcsr[3]
    assert fg_std.tolist() == [[True, False]]
    assert fg_tcsr.tolist() == [[False, True]]

    # The assigned target box/label remain the same GT; only which anchor is
    # selected changes.
    assert torch.equal(out_std[0][fg_std], out_tcsr[0][fg_tcsr])
    assert torch.equal(out_std[1][fg_std], out_tcsr[1][fg_tcsr])
