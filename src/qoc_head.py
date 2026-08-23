from __future__ import annotations

from ultralytics.nn.modules.head import Detect


class QualityOverconfidenceDetect(Detect):
    """DFL-free Detect head carrying training-only QOC calibration metadata.

    Inference is intentionally identical to the stock direct-regression Detect
    head (reg_max=1). The only change is the criterion selected during training:
    a one-sided localization-overconfidence regularizer is added to the stock
    YOLO detection loss. No extra prediction channels or runtime branches are
    introduced.
    """

    def __init__(
        self,
        nc: int = 80,
        qoc_lambda: float = 0.25,
        qoc_margin: float = 0.05,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ):
        if end2end:
            raise ValueError("QualityOverconfidenceDetect v1 supports one-to-many detection only")
        if qoc_lambda < 0:
            raise ValueError("qoc_lambda must be non-negative")
        if not (0.0 <= qoc_margin < 1.0):
            raise ValueError("qoc_margin must be in [0, 1)")

        super().__init__(nc=nc, reg_max=1, end2end=False, ch=ch)
        self.qoc_lambda = float(qoc_lambda)
        self.qoc_margin = float(qoc_margin)
