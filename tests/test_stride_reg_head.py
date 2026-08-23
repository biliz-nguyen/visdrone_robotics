import torch

from src.stride_reg_head import StrideRegDetect


def make_feats(requires_grad=False):
    return [
        torch.randn(2, 32, 20, 20, requires_grad=requires_grad),
        torch.randn(2, 64, 10, 10, requires_grad=requires_grad),
        torch.randn(2, 128, 5, 5, requires_grad=requires_grad),
    ]


def test_training_outputs_keep_per_level_regression_widths():
    head = StrideRegDetect(nc=10, reg_bins=(16, 8, 4), ch=(32, 64, 128))
    head.train()
    out = head(make_feats())
    assert [x.shape[1] for x in out["boxes"]] == [64, 32, 16]
    assert out["scores"].shape == (2, 10, 525)


def test_inference_decodes_all_levels():
    head = StrideRegDetect(nc=10, reg_bins=(16, 8, 4), ch=(32, 64, 128))
    head.stride[:] = torch.tensor([4.0, 8.0, 16.0])
    head.eval()
    y, raw = head(make_feats())
    assert y.shape == (2, 14, 525)
    assert torch.isfinite(y).all()
    assert len(raw["boxes"]) == 3


def test_hybrid_direct_level_backpropagates():
    head = StrideRegDetect(nc=10, reg_bins=(16, 4, 1), ch=(32, 64, 128))
    head.train()
    feats = make_feats(requires_grad=True)
    out = head(feats)
    loss = sum(x.square().mean() for x in out["boxes"]) + out["scores"].square().mean()
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in feats)
