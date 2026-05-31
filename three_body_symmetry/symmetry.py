"""Cyclic Fourier constraints for equal-mass choreographies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FourierMode:
    """One complex planar Fourier mode."""

    harmonic: int
    cosine: complex
    sine: complex


def allowed_choreography_harmonics(max_harmonic: int) -> list[int]:
    """Harmonics enforcing both C3 center-of-mass and half-turn symmetry."""

    if max_harmonic < 1:
        return []
    return [n for n in range(1, max_harmonic + 1) if n % 2 == 1 and n % 3 != 0]


def evaluate_curve(modes: Iterable[FourierMode], times: np.ndarray, period: float) -> np.ndarray:
    """Evaluate a complex-valued choreography curve q(t)."""

    times = np.asarray(times, dtype=float)
    omega = 2.0 * np.pi / period
    values = np.zeros(times.shape, dtype=complex)
    for mode in modes:
        n = mode.harmonic
        values += mode.cosine * np.cos(n * omega * times)
        values += mode.sine * np.sin(n * omega * times)
    return values


def choreography_positions(
    modes: Iterable[FourierMode],
    time: float,
    period: float,
) -> np.ndarray:
    """Project one curve into the three bodies with one-third-period shifts."""

    shifted_times = time + period * np.arange(3, dtype=float) / 3.0
    values = evaluate_curve(modes, shifted_times, period)
    return np.column_stack([values.real, values.imag])

