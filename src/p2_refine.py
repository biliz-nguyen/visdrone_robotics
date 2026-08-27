from __future__ import annotations

import torch
import torch.nn as nn


class P2Refine(nn.Module):
    """Tiny-object head refinement applied only to the P2 Detect input.

    The block is intentionally close to identity at initialization. A single
    depthwise 3x3 captures local spatial evidence at stride 4, a pointwise 1x1
    remixes channels, and a learnable scalar residual gate controls how much of
    the refinement enters the Detect head. It does not feed back into the PAN
    path, so P3/P4 neck features remain identical to frozen N2b.
    """

    def __init__(self, c1: int, c2: int, alpha_init: float = 0.10):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c2)
        if self.c1 != self.c2:
            raise ValueError(f"P2Refine requires c1==c2, got {self.c1}->{self.c2}")
        if not (0.0 <= float(alpha_init) <= 1.0):
            raise ValueError("alpha_init must be in [0, 1]")

        self.dw = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, 3, 1, 1, groups=self.c1, bias=False),
            nn.BatchNorm2d(self.c1),
            nn.SiLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(self.c1, self.c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.c2),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(f"P2Refine expected {self.c1} channels, got {x.shape[1]}")
        return x + self.alpha * self.pw(self.dw(x))

    def extra_repr(self) -> str:
        return f"channels={self.c1}, alpha={float(self.alpha.detach()):.4f}"
