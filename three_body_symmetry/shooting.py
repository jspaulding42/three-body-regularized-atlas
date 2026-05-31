"""Symmetry-reduced periodic shooting for the figure-eight family."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .dynamics import integrate, max_periodic_error
from .figure_eight import FIGURE_EIGHT_PERIOD, FIGURE_EIGHT_PARAMETERS, symmetric_state


@dataclass(frozen=True)
class ShootingResult:
    parameters: np.ndarray
    period: float
    residual_norm: float
    cost: float
    nfev: int
    success: bool
    message: str

    @property
    def state(self) -> np.ndarray:
        return symmetric_state(self.parameters)


def residual(variable: np.ndarray, *, rtol: float = 3e-10, atol: float = 1e-12) -> np.ndarray:
    """Full-period closing residual for `(x, y, vx, vy, T)`."""

    variable = np.asarray(variable, dtype=float)
    if variable.shape != (5,):
        raise ValueError("variable must be [x, y, vx, vy, period]")
    period = variable[4]
    if period <= 0.0:
        return np.full(12, 1e6)
    state0 = symmetric_state(variable[:4])
    return integrate(state0, period, rtol=rtol, atol=atol).final_state - state0


def refine_figure_eight(
    seed: np.ndarray | None = None,
    *,
    max_nfev: int = 30,
    rtol: float = 3e-10,
    atol: float = 1e-12,
) -> ShootingResult:
    """Refine a rough symmetric seed into a periodic figure-eight orbit."""

    if seed is None:
        seed = np.concatenate([FIGURE_EIGHT_PARAMETERS.as_vector(), [FIGURE_EIGHT_PERIOD]])
    seed = np.asarray(seed, dtype=float)
    result = least_squares(
        lambda variable: residual(variable, rtol=rtol, atol=atol),
        seed,
        bounds=(
            np.array([1e-6, -np.inf, -np.inf, -np.inf, 1e-6]),
            np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
        ),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=max_nfev,
    )
    residual_norm = float(np.linalg.norm(result.fun, ord=np.inf))
    return ShootingResult(
        parameters=result.x[:4],
        period=float(result.x[4]),
        residual_norm=residual_norm,
        cost=float(result.cost),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


def main() -> None:
    seed = np.array([0.97, -0.243, 0.466, 0.432, 6.326], dtype=float)
    result = refine_figure_eight(seed)
    print("success:", result.success)
    print("parameters:", result.parameters)
    print("period:", result.period)
    print("residual infinity norm:", result.residual_norm)
    print("verification error:", max_periodic_error(result.state, result.period, rtol=1e-11, atol=1e-13))


if __name__ == "__main__":
    main()

