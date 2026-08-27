from __future__ import annotations

import torch

from ultralytics.utils.tal import TaskAlignedAssigner

from src.tiny_quality_assigner import TinyAdaptiveQualityAssigner


def _make_assigners():
    kwargs = dict(
        topk=3,
        num_classes=2,
        alpha=0.5,
        beta=6.0,
        stride=[4, 8, 16],
        topk2=3,
    )
    return (
        TaskAlignedAssigner(**kwargs),
        TinyAdaptiveQualityAssigner(**kwargs, tiny_min_side=16.0, beta_floor=4.0),
    )


def test_effective_beta_schedule_is_locked():
    _, taq = _make_assigners()
    boxes = torch.tensor(
        [[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 32.0, 20.0]]]
    )
    beta = taq.effective_beta(boxes).squeeze(-1)
    assert torch.allclose(beta, torch.tensor([[5.0, 6.0, 6.0]]), atol=1e-7)


def test_non_tiny_forward_matches_standard_tal_exactly():
    tal, taq = _make_assigners()
    pd_scores = torch.tensor(
        [[[0.80, 0.20], [0.65, 0.30], [0.55, 0.40], [0.45, 0.50]]], dtype=torch.float32
    )
    pd_bboxes = torch.tensor(
        [[[0.0, 0.0, 20.0, 20.0], [1.0, 1.0, 21.0, 21.0], [2.0, 2.0, 22.0, 22.0], [3.0, 3.0, 23.0, 23.0]]],
        dtype=torch.float32,
    )
    anc_points = torch.tensor([[4.0, 4.0], [8.0, 8.0], [12.0, 12.0], [16.0, 16.0]])
    gt_labels = torch.tensor([[[0.0]]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 20.0, 20.0]]])
    mask_gt = torch.tensor([[[1.0]]])

    out_std = tal(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    out_taq = taq(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    for a, b in zip(out_std, out_taq):
        if a.dtype == torch.bool:
            assert torch.equal(a, b)
        else:
            assert torch.allclose(a, b, atol=0.0, rtol=0.0)


def test_tiny_quality_changes_training_ranking_without_new_parameters():
    _, taq = _make_assigners()
    assert sum(p.numel() for p in taq.parameters()) == 0
    tiny = torch.tensor([[[0.0, 0.0, 8.0, 8.0]]])
    assert float(taq.effective_beta(tiny).item()) == 5.0
