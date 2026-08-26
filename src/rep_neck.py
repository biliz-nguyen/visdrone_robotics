from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Conv-BN-activation helper with explicit deploy-time BN fusion."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, act: bool = True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
        self.deploy = False

    def forward(self, x):
        if self.deploy:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        fused = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
        ).to(device=conv.weight.device, dtype=conv.weight.dtype)
        w = conv.weight
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        std = torch.sqrt(var + bn.eps)
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        fused.weight.data.copy_(w * scale)
        fused.bias.data.copy_(beta - mean * gamma / std)
        return fused

    def switch_to_deploy(self):
        if self.deploy:
            return self
        self.conv = self._fuse_conv_bn(self.conv, self.bn)
        del self.bn
        self.deploy = True
        return self


class RepConvUnit(nn.Module):
    """RepVGG-style 3x3/1x1/identity training block fused to one 3x3 conv."""

    def __init__(self, channels: int, act: bool = True):
        super().__init__()
        c = int(channels)
        self.channels = c
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
        )
        self.rbr_identity = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
        self.deploy = False

    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam(x))
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x))

    @staticmethod
    def _fuse_branch(branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            conv, bn = branch[0], branch[1]
            kernel = conv.weight
        else:
            bn = branch
            c = bn.num_features
            kernel = torch.zeros(
                (c, c, 3, 3),
                device=bn.weight.device,
                dtype=bn.weight.dtype,
            )
            idx = torch.arange(c, device=bn.weight.device)
            kernel[idx, idx, 1, 1] = 1.0

        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        std = torch.sqrt(var + bn.eps)
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * scale, beta - mean * gamma / std

    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_branch(self.rbr_dense)
        k1, b1 = self._fuse_branch(self.rbr_1x1)
        kid, bid = self._fuse_branch(self.rbr_identity)

        if isinstance(k1, torch.Tensor):
            k1 = torch.nn.functional.pad(k1, [1, 1, 1, 1])
        return k3 + k1 + kid, b3 + b1 + bid

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam = nn.Conv2d(
            self.channels,
            self.channels,
            3,
            1,
            1,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        self.reparam.weight.data.copy_(kernel)
        self.reparam.bias.data.copy_(bias)
        del self.rbr_dense
        del self.rbr_1x1
        del self.rbr_identity
        self.deploy = True
        return self


class RepBottleneck(nn.Module):
    """YOLO-style residual bottleneck with a reparameterized first 3x3 conv."""

    def __init__(self, channels: int, shortcut: bool = True):
        super().__init__()
        c = int(channels)
        self.cv1 = RepConvUnit(c)
        self.cv2 = ConvBNAct(c, c, 3, 1, act=True)
        self.add = bool(shortcut)

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

    def switch_to_deploy(self):
        self.cv1.switch_to_deploy()
        self.cv2.switch_to_deploy()
        return self


class RepC3k2(nn.Module):
    """C2f/C3k2-like fusion block with train-rich, deploy-simple bottlenecks.

    The block is used only in the PAN/FPN neck. During training each internal
    RepConvUnit has 3x3, 1x1 and identity branches. At deployment those branches
    are analytically fused to a single 3x3 convolution, so the deployed graph
    retains one spatial conv at that location rather than three parallel paths.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 2,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        if int(g) != 1:
            raise ValueError("RepC3k2 v1 supports g=1 only")
        if int(n) < 1:
            raise ValueError("RepC3k2 requires n>=1")
        hidden = max(1, int(c2 * float(e)))
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.n = int(n)
        self.hidden = hidden
        self.cv1 = ConvBNAct(self.c1, 2 * hidden, 1, 1, act=True)
        self.m = nn.ModuleList(RepBottleneck(hidden, shortcut=shortcut) for _ in range(self.n))
        self.cv2 = ConvBNAct((2 + self.n) * hidden, self.c2, 1, 1, act=True)
        self.deploy = False

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for block in self.m:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def switch_to_deploy(self):
        if self.deploy:
            return self
        self.cv1.switch_to_deploy()
        for block in self.m:
            block.switch_to_deploy()
        self.cv2.switch_to_deploy()
        self.deploy = True
        return self


def switch_reparameterized_neck_to_deploy(model: nn.Module) -> nn.Module:
    """Fuse every RepC3k2 in-place and return the model."""
    blocks = [m for m in model.modules() if isinstance(m, RepC3k2)]
    for block in blocks:
        block.switch_to_deploy()
    return model


def deployed_copy(model: nn.Module) -> nn.Module:
    """Deep-copy then fuse, useful for equivalence/complexity preflight."""
    out = copy.deepcopy(model).eval()
    return switch_reparameterized_neck_to_deploy(out)
