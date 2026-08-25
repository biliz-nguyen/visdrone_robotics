from __future__ import annotations

from ultralytics.nn.modules.head import Detect


class TinyShiftConsistencyDetect(Detect):
    """DFL-free Detect head carrying training-only shift-consistency metadata.

    Inference is intentionally identical to the H1 direct-regression head
    (reg_max=1). During training, the custom criterion adds a one-pixel
    translation-equivariance regularizer for tiny objects. No extra prediction
    channels or inference-time modules are introduced.
    """

    def __init__(
        self,
        nc: int = 80,
        shift_lambda: float = 0.25,
        tiny_threshold: float = 16.0,
        max_shift_px: int = 1,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ):
        if end2end:
            raise ValueError("TinyShiftConsistencyDetect v1 supports one-to-many detection only")
        if shift_lambda < 0:
            raise ValueError("shift_lambda must be non-negative")
        if tiny_threshold <= 0:
            raise ValueError("tiny_threshold must be positive")
        if int(max_shift_px) != 1:
            raise ValueError("v1 is intentionally locked to a one-pixel shift")

        super().__init__(nc=nc, reg_max=1, end2end=False, ch=ch)
        self.shift_lambda = float(shift_lambda)
        self.tiny_threshold = float(tiny_threshold)
        self.max_shift_px = int(max_shift_px)
