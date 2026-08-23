from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ULTRA_REPO = ROOT / "third_party" / "ultralytics"
if not ULTRA_REPO.exists():
    raise RuntimeError(
        f"Pinned Ultralytics checkout is missing: {ULTRA_REPO}. Run setup_local.sh first."
    )
sys.path.insert(0, str(ULTRA_REPO))

import torch

from ultralytics.utils.tal import TaskAlignedAssigner
from src.tiny_quality_assigner import TinyAdaptiveQualityAssigner


def _taq():
    return TinyAdaptiveQualityAssigner(
        topk=10,
        num_classes=2,
        alpha=0.5,
        beta=6.0,
        stride=[4, 8, 16],
        tiny_min_side=16.0,
        beta_floor=4.0,
    )


def _tal():
    return TaskAlignedAssigner(
        topk=10,
        num_classes=2,
        alpha=0.5,
        beta=6.0,
        stride=[4, 8, 16],
    )


def _grid(step=4, size=32):
    coords = torch.arange(step / 2, size, step)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def test_effective_beta_schedule():
    assigner = _taq()
    boxes = torch.tensor(
        [[
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 8.0, 8.0],
            [0.0, 0.0, 16.0, 16.0],
            [0.0, 0.0, 32.0, 20.0],
        ]]
    )
    beta = assigner.effective_beta(boxes).squeeze(-1)

    assert torch.allclose(beta[0, 0], torch.tensor(4.125))
    assert torch.allclose(beta[0, 1], torch.tensor(5.0))
    assert torch.allclose(beta[0, 2], torch.tensor(6.0))
    assert torch.allclose(beta[0, 3], torch.tensor(6.0))


def test_non_tiny_metric_is_exact_standard_tal():
    taq = _taq()
    tal = _tal()
    anchors = _grid()
    a = anchors.shape[0]

    gt = torch.tensor([[[4.0, 4.0, 28.0, 28.0]]])
    labels = torch.tensor([[[1.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)
    scores = torch.linspace(0.15, 0.85, a * 2).view(1, a, 2)
    half = 5.0
    boxes = torch.cat((anchors - half, anchors + half), dim=-1).unsqueeze(0)

    out_taq = taq(scores, boxes, anchors, labels, gt, mask)
    out_tal = tal(scores, boxes, anchors, labels, gt, mask)

    for a_taq, a_tal in zip(out_taq, out_tal):
        if a_taq.dtype == torch.bool:
            assert torch.equal(a_taq, a_tal)
        else:
            assert torch.allclose(a_taq, a_tal, atol=1e-7, rtol=1e-6)


def test_candidate_region_is_unchanged_for_tiny_gt():
    taq = _taq()
    tal = _tal()
    anchors = _grid(size=24)
    gt = torch.tensor([[[7.0, 7.0, 11.0, 11.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)

    c_taq = taq.select_candidates_in_gts(anchors, gt, mask)
    c_tal = tal.select_candidates_in_gts(anchors, gt, mask)
    assert torch.equal(c_taq, c_tal)


def test_tiny_localization_penalty_is_softened():
    taq = _taq()
    tal = _tal()

    # Directly compare the same score and overlap under beta_eff < 6.
    overlap = torch.tensor(0.5)
    score = torch.tensor(0.5)
    beta_eff = taq.effective_beta(
        torch.tensor([[[0.0, 0.0, 8.0, 8.0]]])
    ).squeeze()

    metric_taq = score.pow(taq.alpha) * overlap.pow(beta_eff)
    metric_tal = score.pow(tal.alpha) * overlap.pow(tal.beta)
    assert beta_eff < tal.beta
    assert metric_taq > metric_tal


def test_full_forward_is_finite_and_shape_compatible():
    assigner = _taq()
    anchors = _grid(size=24)
    a = anchors.shape[0]

    gt = torch.tensor([[[7.0, 7.0, 11.0, 11.0]]])
    labels = torch.tensor([[[1.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)
    scores = torch.full((1, a, 2), 0.5)
    half = 4.0
    boxes = torch.cat((anchors - half, anchors + half), dim=-1).unsqueeze(0)

    target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
        scores,
        boxes,
        anchors,
        labels,
        gt,
        mask,
    )

    assert target_labels.shape == (1, a)
    assert target_bboxes.shape == (1, a, 4)
    assert target_scores.shape == (1, a, 2)
    assert fg_mask.shape == (1, a)
    assert target_gt_idx.shape == (1, a)
    assert torch.isfinite(target_bboxes).all()
    assert torch.isfinite(target_scores).all()
