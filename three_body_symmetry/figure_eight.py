"""Figure-eight choreography seed data and scale family helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import pack_state


@dataclass(frozen=True)
class SymmetricParameters:
    """Parameters for the standard zero-momentum symmetric three-body section."""

    x: float
    y: float
    vx: float
    vy: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], dtype=float)


# Moore-Chenciner figure-eight initial condition, truncated to common precision.
FIGURE_EIGHT_PARAMETERS = SymmetricParameters(
    x=0.97000436,
    y=-0.24308753,
    vx=0.4662036850,
    vy=0.4323657300,
)
FIGURE_EIGHT_PERIOD = 6.32591398


def symmetric_state(parameters: SymmetricParameters | np.ndarray) -> np.ndarray:
    """Build a center-of-mass-zero state from `(x, y, vx, vy)`."""

    if isinstance(parameters, SymmetricParameters):
        x, y, vx, vy = parameters.x, parameters.y, parameters.vx, parameters.vy
    else:
        x, y, vx, vy = np.asarray(parameters, dtype=float)

    positions = np.array(
        [
            [x, y],
            [-x, -y],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    velocities = np.array(
        [
            [vx, vy],
            [vx, vy],
            [-2.0 * vx, -2.0 * vy],
        ],
        dtype=float,
    )
    return pack_state(positions, velocities)


def figure_eight_state() -> np.ndarray:
    return symmetric_state(FIGURE_EIGHT_PARAMETERS)


def scaled_figure_eight(scale: float) -> tuple[np.ndarray, float]:
    """Return a scaled figure-eight state and its Kepler-scaled period."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    state = figure_eight_state().copy()
    state[:6] *= scale
    state[6:] *= scale ** -0.5
    return state, FIGURE_EIGHT_PERIOD * scale**1.5

