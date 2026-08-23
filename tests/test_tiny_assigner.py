from __future__ import annotations

from pathlib import Path
import sys

# Unit tests must import the repository-pinned Ultralytics checkout, not any
# unrelated package that may already be installed in the runner environment.
ROOT = Path(__file__).resolve().parents[1]
ULTRA_REPO = ROOT / "third_party" / "ultralytics"
if not ULTRA_REPO.exists():
    raise RuntimeError(
        f"Pinned Ultralytics checkout is missing: {ULTRA_REPO}. Run setup_local.sh first."
    )
sys.path.insert(0, str(ULTRA_REPO))

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


def test_pinned_ultralytics_import_is_available():
    import ultralytics
    from ultralytics.utils.tal import TaskAlignedAssigner

    assert TaskAlignedAssigner is not None
    assert Path(ultralytics.__file__).resolve().is_relative_to(ULTRA_REPO.resolve())


def test_ultralytics_builtin_tiny_expansion_is_preserved():
    assigner = _assigner()
    anchors = _grid()

    # Ultralytics v8.4.56 expands a GT smaller than the smallest stride
    # to stride_val (=8 for strides [4, 8, 16]) before center filtering.
    gt = torch.tensor([[[7.0, 7.0, 9.0, 9.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)

    base = assigner.select_candidates_in_gts(anchors, gt, mask).bool()
    recovered = assigner.recover_candidate_mask(anchors, gt, mask)

    assert int(base.sum()) == 4
    assert torch.equal(base, recovered)


def test_recovery_can_extend_beyond_builtin_minimum():
    assigner = _assigner(min_candidates=6)
    anchors = _grid()
    gt = torch.tensor([[[7.0, 7.0, 9.0, 9.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)

    base = assigner.select_candidates_in_gts(anchors, gt, mask).bool()
    recovered = assigner.recover_candidate_mask(anchors, gt, mask)

    assert int(base.sum()) == 4
    assert int(recovered.sum()) == 6
    assert torch.all(recovered[base])


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
    assigner = _assigner(min_candidates=6)
    anchors = _grid()
    gt = torch.tensor([[[5.0, 5.0, 7.0, 7.0]]])
    mask = torch.ones((1, 1, 1), dtype=torch.bool)

    base = assigner.select_candidates_in_gts(anchors, gt, mask).bool()
    recovered = assigner.recover_candidate_mask(anchors, gt, mask)

    assert int(base.sum()) >= 1
    assert int(recovered.sum()) == 6
    assert torch.all(recovered[base])


def test_full_forward_is_standard_tal_compatible():
    """Call the public forward path so TaskAlignedAssigner initializes bs/G state."""
    assigner = _assigner(min_candidates=6)
    anchors = _grid(size=24)
    a = anchors.shape[0]

    gt = torch.tensor([[[7.0, 7.0, 9.0, 9.0]]])
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
