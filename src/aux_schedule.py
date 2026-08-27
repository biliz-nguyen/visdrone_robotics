from __future__ import annotations

import math


def cosine_to_zero_weight(base_weight: float, epoch: int, epochs: int) -> float:
    """Decay an auxiliary loss weight from base_weight to exactly zero.

    Epoch 0 uses the full base weight. The final epoch uses zero, restoring
    the stock objective late in training while preserving early tiny-object
    guidance. The mapping automatically scales to any total epoch count.
    """
    base = float(base_weight)
    e = int(epoch)
    total = int(epochs)
    if base < 0.0:
        raise ValueError("base_weight must be >= 0")
    if total <= 0:
        raise ValueError("epochs must be > 0")
    if not 0 <= e < total:
        raise ValueError(f"epoch must be in [0, {total - 1}], got {e}")
    if total == 1:
        return 0.0
    progress = e / float(total - 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))
