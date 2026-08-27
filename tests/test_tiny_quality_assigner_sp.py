from __future__ import annotations

import torch

from ultralytics.utils.tal import TaskAlignedAssigner

from src.tiny_quality_assigner_sp import SelectionPreservingTinyQualityAssigner


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
        SelectionPreservingTinyQualityAssigner(
            **kwargs,
            tiny_min_side=16.0,
            beta_floor=5.0,
        ),
    )


def test_effective_beta_schedule_is_mild_and_locked():
    _, taq = _make_assigners()
    boxes = torch.tensor(
        [[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 32.0, 20.0]]]
    )
    beta = taq.effective_beta(boxes).squeeze(-1)
    assert torch.allclose(beta, torch.tensor([[5.5, 6.0, 6.0]]), atol=1e-7)


def _forward_inputs(gt_side: float):
    pd_scores = torch.tensor(
        [[[0.82, 0.18], [0.72, 0.20], [0.62, 0.24], [0.52, 0.28]]],
        dtype=torch.float32,
    )
    pd_bboxes = torch.tensor(
        [[[0.0, 0.0, gt_side, gt_side],
          [0.5, 0.5, gt_side + 0.5, gt_side + 0.5],
          [1.0, 1.0, gt_side - 0.5, gt_side - 0.5],
          [1.5, 1.5, gt_side, gt_side]]],
        dtype=torch.float32,
    )
    quarter = gt_side / 4.0
    anc_points = torch.tensor(
        [[quarter, quarter], [2 * quarter, 2 * quarter], [3 * quarter, 3 * quarter], [3.5 * quarter, 3.5 * quarter]],
        dtype=torch.float32,
    )
    gt_labels = torch.tensor([[[0.0]]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, gt_side, gt_side]]])
    mask_gt = torch.tensor([[[1.0]]])
    return pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt


def test_non_tiny_forward_matches_standard_tal():
    tal, taq = _make_assigners()
    inputs = _forward_inputs(20.0)
    out_std = tal(*inputs)
    out_taq = taq(*inputs)
    for a, b in zip(out_std, out_taq):
        if a.dtype == torch.bool:
            assert torch.equal(a, b)
        else:
            assert torch.allclose(a, b, atol=1e-7, rtol=0.0)


def test_tiny_positive_selection_is_preserved_exactly():
    tal, taq = _make_assigners()
    inputs = _forward_inputs(8.0)
    out_std = tal(*inputs)
    out_taq = taq(*inputs)

    # Labels, boxes, foreground mask, and GT assignment must remain stock TAL.
    assert torch.equal(out_std[0], out_taq[0])
    assert torch.allclose(out_std[1], out_taq[1], atol=0.0, rtol=0.0)
    assert torch.equal(out_std[3], out_taq[3])
    assert torch.equal(out_std[4], out_taq[4])
    assert int(out_std[3].sum().item()) >= 2

    # Only soft target quality is allowed to change for tiny positives.
    assert not torch.allclose(out_std[2], out_taq[2], atol=1e-7, rtol=0.0)


def test_training_only_assigner_has_no_parameters():
    _, taq = _make_assigners()
    assert sum(p.numel() for p in taq.parameters()) == 0
