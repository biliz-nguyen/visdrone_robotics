from __future__ import annotations

import torch

from src.pixel_stable_assigner import (
    TinyPixelStableAssigner,
    cardinal_shifted_boxes,
    pixel_stable_quality,
)


def test_cardinal_shifted_boxes_are_exact_one_pixel_translations():
    box = torch.tensor([[10.0, 20.0, 14.0, 24.0]])
    left, right, up, down = cardinal_shifted_boxes(box, 1.0)
    assert torch.equal(left, torch.tensor([[9.0, 20.0, 13.0, 24.0]]))
    assert torch.equal(right, torch.tensor([[11.0, 20.0, 15.0, 24.0]]))
    assert torch.equal(up, torch.tensor([[10.0, 19.0, 14.0, 23.0]]))
    assert torch.equal(down, torch.tensor([[10.0, 21.0, 14.0, 25.0]]))


def test_pixel_stable_quality_identity_and_penalty():
    q0 = torch.tensor([0.8, 0.5])
    identical = torch.stack([q0, q0, q0, q0], dim=0)
    assert torch.allclose(pixel_stable_quality(q0, identical), q0)

    fragile = torch.tensor([[0.2], [0.3], [0.2], [0.3]])
    robust = pixel_stable_quality(torch.tensor([0.8]), fragile)
    assert 0.0 < float(robust) < 0.8


def test_assigner_handles_float_pair_mask_and_preserves_non_tiny_quality():
    assigner = TinyPixelStableAssigner(
        topk=2,
        num_classes=2,
        alpha=0.5,
        beta=6.0,
        stride=[4, 8, 16],
        tiny_min_side=16.0,
        perturb_px=1.0,
    )
    # get_box_metrics uses these values set by TaskAlignedAssigner.forward.
    assigner.bs = 1
    assigner.n_max_boxes = 2

    pd_scores = torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]])
    pd_bboxes = torch.tensor(
        [[[0.0, 0.0, 4.0, 4.0], [0.5, 0.0, 4.5, 4.0], [0.0, 0.0, 20.0, 20.0]]]
    )
    gt_labels = torch.tensor([[[0.0], [1.0]]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 20.0, 20.0]]])
    # Deliberately float: protects against the dtype class of bug seen in TSEC.
    pair_mask = torch.ones((1, 2, 3), dtype=torch.float32)

    align, overlaps = assigner.get_box_metrics(
        pd_scores, pd_bboxes, gt_labels, gt_bboxes, pair_mask
    )
    assert align.shape == (1, 2, 3)
    assert overlaps.shape == (1, 2, 3)
    assert torch.isfinite(align).all()
    assert torch.isfinite(overlaps).all()

    # Exact-match non-tiny candidate keeps nominal IoU=1; returned overlaps are
    # always standard TAL IoUs even when tiny ranking uses stable quality.
    assert torch.allclose(overlaps[0, 1, 2], torch.tensor(1.0), atol=1e-6)


def test_invalid_hyperparameters_fail_early():
    for kwargs in ({"tiny_min_side": 0.0}, {"perturb_px": 0.0}):
        try:
            TinyPixelStableAssigner(topk=2, num_classes=2, **kwargs)
            raise AssertionError(f"invalid config should fail: {kwargs}")
        except ValueError:
            pass
