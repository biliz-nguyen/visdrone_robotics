"""SPR-Down v2 edge-latency prototype.

The v1 module uses input-conditioned polyphase scoring (split + global
statistics + softmax + weighted reassembly).  That preserves the intended
phase-aware behavior, but the extra tensor operations can dominate latency.

SPR-Down v2 tests a deployment-oriented approximation.  A learned per-channel
2x2 stride-2 depthwise kernel performs the four-phase reassembly directly:

    even-pad -> DWConv 2x2 s2 (phase reassembly)
             -> DWConv 3x3 s1 (spatial mixing)
             -> PWConv 1x1 (channel projection)

A depthwise 2x2 stride-2 convolution is algebraically equivalent to exposing
the four 2x2 sampling phases and applying one static learned 4->1 linear
reassembly kernel per input channel.  Using the native convolution avoids an
explicit pixel_unshuffle tensor plus grouped 1x1 convolution in the deployment
path.  The phase kernel is initialized to 0.25, so the initial reassembly is
exactly 2x2 average pooling before end-to-end learning.

This file is a research/engineering prototype, not a novelty claim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPRDownV2(nn.Module):
    """Fused static-phase reassembly downsampling for edge deployment."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
        super().__init__()

        if int(s) != 2:
            raise ValueError(f"SPRDownV2 is stride-2 only, got s={s}")
        if int(k) != 3:
            raise ValueError(f"SPRDownV2 expects k=3, got k={k}")
        if int(g) != 1:
            raise ValueError(f"SPRDownV2 expects g=1, got g={g}")
        if int(d) != 1:
            raise ValueError(f"SPRDownV2 expects d=1, got d={d}")

        self.c1 = int(c1)
        self.c2 = int(c2)
        self.k = int(k)
        self.s = int(s)

        # Fused four-phase reassembly. For each input channel this kernel has
        # four coefficients corresponding to the 2x2 sampling phases.
        self.phase_mix = nn.Conv2d(
            self.c1,
            self.c1,
            kernel_size=2,
            stride=2,
            padding=0,
            groups=self.c1,
            bias=False,
        )
        nn.init.constant_(self.phase_mix.weight, 0.25)

        self.spatial_mix = nn.Sequential(
            nn.Conv2d(
                self.c1,
                self.c1,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.c1,
                bias=False,
            ),
            nn.BatchNorm2d(self.c1),
            nn.SiLU(inplace=True) if act is True else nn.Identity(),
        )

        activation = (
            nn.SiLU(inplace=True)
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

        self.channel_mix = nn.Sequential(
            nn.Conv2d(
                self.c1,
                self.c2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.c2),
            activation,
        )

    @staticmethod
    def _pad_to_even(x: torch.Tensor) -> torch.Tensor:
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x

    def phase_reassemble(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_to_even(x)
        return self.phase_mix(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(
                f"SPRDownV2 expected {self.c1} channels, got {x.shape[1]}"
            )
        y = self.phase_reassemble(x)
        y = self.spatial_mix(y)
        y = self.channel_mix(y)
        return y

    def extra_repr(self) -> str:
        return f"c1={self.c1}, c2={self.c2}, stride=2, phase_kernel=2x2-dw"
