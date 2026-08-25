from __future__ import annotations

from ultralytics.nn.modules.head import Detect


class AdvantageGatedMentorDetect(Detect):
    """DFL-free student head carrying training-only AGTL metadata.

    Inference is identical to the H1 direct-regression control (reg_max=1).
    A frozen mentor is used only by the training criterion. No extra prediction
    channels, feature adapters, or runtime branches are added to the student.
    """

    def __init__(
        self,
        nc: int = 80,
        mentor_lambda: float = 0.25,
        tiny_threshold: float = 16.0,
        advantage_margin: float = 0.05,
        min_teacher_iou: float = 0.10,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ):
        if end2end:
            raise ValueError("AGTL v1 supports one-to-many detection only")
        if mentor_lambda < 0:
            raise ValueError("mentor_lambda must be non-negative")
        if tiny_threshold <= 0:
            raise ValueError("tiny_threshold must be positive")
        if not (0.0 <= advantage_margin < 1.0):
            raise ValueError("advantage_margin must be in [0,1)")
        if not (0.0 <= min_teacher_iou <= 1.0):
            raise ValueError("min_teacher_iou must be in [0,1]")

        super().__init__(nc=nc, reg_max=1, end2end=False, ch=ch)
        self.mentor_lambda = float(mentor_lambda)
        self.tiny_threshold = float(tiny_threshold)
        self.advantage_margin = float(advantage_margin)
        self.min_teacher_iou = float(min_teacher_iou)
