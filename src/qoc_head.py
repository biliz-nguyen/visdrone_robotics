from __future__ import annotations

from ultralytics.nn.modules.head import Detect


class QualityOverconfidenceDetect(Detect):
    """DFL-free Detect head carrying training-only QOC calibration metadata.

    Inference is intentionally identical to the stock direct-regression Detect
    head (reg_max=1). The only change is the criterion selected during training.
    QOC v2 makes the overconfidence tolerance size-aware: assigned boxes whose
    original pixel min-side is below a threshold receive a larger margin, while
    normal objects recover the exact v1 margin. No extra prediction channels or
    runtime branches are introduced.
    """

    def __init__(
        self,
        nc: int = 80,
        qoc_lambda: float = 0.25,
        qoc_margin: float = 0.05,
        qoc_tiny_threshold: float = 16.0,
        qoc_tiny_margin_bonus: float = 0.10,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ):
        if end2end:
            raise ValueError("QualityOverconfidenceDetect supports one-to-many detection only")
        if qoc_lambda < 0:
            raise ValueError("qoc_lambda must be non-negative")
        if not (0.0 <= qoc_margin < 1.0):
            raise ValueError("qoc_margin must be in [0, 1)")
        if qoc_tiny_threshold <= 0:
            raise ValueError("qoc_tiny_threshold must be positive")
        if qoc_tiny_margin_bonus < 0:
            raise ValueError("qoc_tiny_margin_bonus must be non-negative")
        if qoc_margin + qoc_tiny_margin_bonus >= 1.0:
            raise ValueError("maximum QOC margin must stay below 1")

        super().__init__(nc=nc, reg_max=1, end2end=False, ch=ch)
        self.qoc_lambda = float(qoc_lambda)
        self.qoc_margin = float(qoc_margin)
        self.qoc_tiny_threshold = float(qoc_tiny_threshold)
        self.qoc_tiny_margin_bonus = float(qoc_tiny_margin_bonus)
