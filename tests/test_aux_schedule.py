from __future__ import annotations

import math

import pytest

from src.aux_schedule import cosine_to_zero_weight


def test_c11_five_epoch_schedule_is_locked():
    got = [cosine_to_zero_weight(0.10, e, 5) for e in range(5)]
    expected = [
        0.10,
        0.10 * 0.5 * (1.0 + math.cos(math.pi * 0.25)),
        0.05,
        0.10 * 0.5 * (1.0 + math.cos(math.pi * 0.75)),
        0.0,
    ]
    assert got == pytest.approx(expected, abs=1e-12)
    assert got[0] == pytest.approx(0.10)
    assert got[-1] == pytest.approx(0.0, abs=1e-15)
    assert all(a >= b for a, b in zip(got, got[1:]))


def test_c11_schedule_scales_to_longer_training():
    weights = [cosine_to_zero_weight(0.10, e, 50) for e in range(50)]
    assert weights[0] == pytest.approx(0.10)
    assert weights[-1] == pytest.approx(0.0, abs=1e-15)
    assert all(0.0 <= x <= 0.10 for x in weights)
    assert all(a >= b for a, b in zip(weights, weights[1:]))


def test_c11_schedule_rejects_invalid_epoch():
    with pytest.raises(ValueError):
        cosine_to_zero_weight(0.10, 5, 5)
