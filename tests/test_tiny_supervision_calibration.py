from __future__ import annotations

# C3-v1 is preregistered to tau=16 px and gamma_floor=0.5; tests must not be
# relaxed after looking at validation results.
import torch

from ultralytics.utils.tal import TaskAlignedAssigner

from src.tiny_supervision_calibration import TinySupervisionCalibratedAssigner


def _assigner(cls):
    return cls(
        topk=2,
        num_classes=1,
        alpha=0.5,
        beta=6.0,
        stride=[4, 8, 16],
        topk2=2,
        **({"tiny_min_side": 16.0, "gamma_floor": 0.5} if cls is TinySupervisionCalibratedAssigner else {}),
    )


def test_quality_exponent_is_monotonic_and_standard_at_threshold():
    tsc = _assigner(TinySupervisionCalibratedAssigner)
    boxes = torch.tensor(
        [[[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 32.0, 32.0]]]
    )
    gamma = tsc.quality_exponent(boxes).flatten()
    expected = torch.tensor([0.625, 0.75, 1.0, 1.0])
    assert torch.allclose(gamma, expected, atol=1e-7)
    assert torch.all(gamma[1:] >= gamma[:-1])


def test_calibration_boosts_only_valid_tiny_quality():
    tsc = _assigner(TinySupervisionCalibratedAssigner)
    q = torch.tensor([[[0.25], [0.25], [0.25], [0.25]]])
    boxes = torch.tensor(
        [[[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 4.0, 4.0]]]
    )
    mask = torch.tensor([[[1.0], [1.0], [1.0], [0.0]]])
    calibrated = tsc.calibrate_positive_quality(q, boxes, mask)

    assert calibrated[0, 0, 0] > q[0, 0, 0]
    assert calibrated[0, 1, 0] > q[0, 1, 0]
    assert torch.equal(calibrated[0, 2], q[0, 2])  # >=16 px is exactly standard TAL
    assert torch.equal(calibrated[0, 3], q[0, 3])  # padded GT remains untouched


def test_zero_and_one_quality_are_fixed_points():
    tsc = _assigner(TinySupervisionCalibratedAssigner)
    q = torch.tensor([[[0.0], [1.0]]])
    boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0]]])
    mask = torch.ones((1, 2, 1))
    calibrated = tsc.calibrate_positive_quality(q, boxes, mask)
    assert torch.equal(calibrated, q)


def test_tsc_keeps_standard_tal_positive_assignment_identical():
    """C3-v1 must change supervision amplitude, not candidate membership."""
    std = _assigner(TaskAlignedAssigner)
    tsc = _assigner(TinySupervisionCalibratedAssigner)

    pd_scores = torch.tensor([[[0.90], [0.82], [0.70], [0.55]]], dtype=torch.float32)
    pd_bboxes = torch.tensor(
        [[[0.0, 0.0, 3.4, 3.4], [0.3, 0.0, 3.8, 3.7], [0.0, 0.4, 3.5, 3.9], [0.5, 0.5, 3.5, 3.5]]],
        dtype=torch.float32,
    )
    anc_points = torch.tensor([[1.0, 1.0], [3.0, 1.0], [1.0, 3.0], [3.0, 3.0]], dtype=torch.float32)
    gt_labels = torch.zeros((1, 1, 1), dtype=torch.long)
    gt_bboxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0]]], dtype=torch.float32)
    mask_gt = torch.ones((1, 1, 1), dtype=torch.float32)

    out_std = std(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    out_tsc = tsc(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    # labels, boxes, foreground mask and GT assignment must be identical.
    assert torch.equal(out_tsc[0], out_std[0])
    assert torch.equal(out_tsc[1], out_std[1])
    assert torch.equal(out_tsc[3], out_std[3])
    assert torch.equal(out_tsc[4], out_std[4])

    # TSC may only increase tiny positive soft-target amplitude; negatives stay zero.
    assert torch.all(out_tsc[2] >= out_std[2] - 1e-7)
    assert torch.any(out_tsc[2] > out_std[2] + 1e-7)
