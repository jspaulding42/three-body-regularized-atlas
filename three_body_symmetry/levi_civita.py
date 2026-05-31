"""Planar Levi-Civita binary-collision chart.

The chart is the local collision model needed by a Sundman-style global
construction. For an isolated planar binary with relative coordinate `q`, write
`q = z^2` in complex notation and use `dt/ds = |z|^2`. At fixed Kepler energy
`h`, the singular relative equation `q_ddot = -mu q / |q|^3` lifts to the
regular linear equation `z'' = (h / 2) z`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .series import scalar_series_product


Array = np.ndarray


@dataclass(frozen=True)
class LeviCivitaChart:
    z: Array
    z_velocity: Array
    physical_time: Array
    energy: float

    @property
    def order(self) -> int:
        return int(self.z.shape[0] - 1)

    def z_at(self, s_value: float) -> Array:
        return _evaluate(self.z, s_value)

    def z_velocity_at(self, s_value: float) -> Array:
        return _evaluate(self.z_velocity, s_value)

    def physical_time_at(self, s_value: float) -> float:
        return float(_evaluate(self.physical_time[:, None], s_value)[0])

    def relative_position_at(self, s_value: float) -> Array:
        return lc_square(self.z_at(s_value))

    def relative_velocity_at(self, s_value: float) -> Array:
        return lc_velocity(self.z_at(s_value), self.z_velocity_at(s_value))

    def relative_state_at(self, s_value: float) -> Array:
        return np.concatenate([self.relative_position_at(s_value), self.relative_velocity_at(s_value)])


def _evaluate(coefficients: Array, value: float) -> Array:
    out = np.zeros(coefficients.shape[1:], dtype=float)
    for coefficient in coefficients[::-1]:
        out = out * value + coefficient
    return out


def lc_square(z: Array) -> Array:
    x, y = np.asarray(z, dtype=float)
    return np.array([x * x - y * y, 2.0 * x * y], dtype=float)


def lc_matrix(z: Array) -> Array:
    x, y = np.asarray(z, dtype=float)
    return np.array([[2.0 * x, -2.0 * y], [2.0 * y, 2.0 * x]], dtype=float)


def lc_velocity(z: Array, z_velocity: Array) -> Array:
    rho = float(np.dot(z, z))
    if rho == 0.0:
        raise ValueError("physical velocity is singular at binary collision")
    return lc_matrix(z) @ np.asarray(z_velocity, dtype=float) / rho


def physical_to_levi_civita(relative_position: Array, relative_velocity: Array, mu: float) -> tuple[Array, Array, float]:
    """Lift a non-collision relative Kepler state into Levi-Civita variables."""

    if mu <= 0.0:
        raise ValueError("mu must be positive")
    q = np.asarray(relative_position, dtype=float)
    v = np.asarray(relative_velocity, dtype=float)
    radius = float(np.linalg.norm(q))
    if radius == 0.0:
        raise ValueError("cannot lift a collision state from physical velocity")
    root = np.sqrt(q[0] + 1j * q[1])
    z = np.array([root.real, root.imag], dtype=float)
    z_velocity = 0.25 * lc_matrix(z).T @ v
    energy = 0.5 * float(np.dot(v, v)) - mu / radius
    return z, z_velocity, energy


def levi_civita_mu(z: Array, z_velocity: Array, energy: float) -> float:
    """Recover the Kepler parameter from the LC energy constraint."""

    rho = float(np.dot(z, z))
    return float(2.0 * np.dot(z_velocity, z_velocity) - energy * rho)


def projected_acceleration_from_lift(z: Array, z_velocity: Array, energy: float) -> Array:
    """Project the regularized equation back to physical acceleration."""

    z = np.asarray(z, dtype=float)
    z_velocity = np.asarray(z_velocity, dtype=float)
    rho = float(np.dot(z, z))
    if rho == 0.0:
        raise ValueError("physical acceleration is singular at binary collision")
    z_acceleration = 0.5 * energy * z
    matrix = lc_matrix(z)
    matrix_prime = lc_matrix(z_velocity)
    q_prime = matrix @ z_velocity
    rho_prime = 2.0 * float(np.dot(z, z_velocity))
    v_prime = (matrix_prime @ z_velocity + matrix @ z_acceleration) / rho
    v_prime -= q_prime * rho_prime / rho**2
    return v_prime / rho


def construct_levi_civita_chart(
    z0: Array,
    z_velocity0: Array,
    energy: float,
    *,
    order: int,
) -> LeviCivitaChart:
    """Construct a Taylor chart for the regular LC oscillator."""

    if order < 1:
        raise ValueError("order must be at least 1")
    z0 = np.asarray(z0, dtype=float)
    z_velocity0 = np.asarray(z_velocity0, dtype=float)
    if z0.shape != (2,) or z_velocity0.shape != (2,):
        raise ValueError("z0 and z_velocity0 must have shape (2,)")
    z = np.zeros((order + 1, 2), dtype=float)
    z_velocity = np.zeros_like(z)
    physical_time = np.zeros(order + 1, dtype=float)
    z[0] = z0
    z_velocity[0] = z_velocity0

    for n in range(order):
        z[n + 1] = z_velocity[n] / (n + 1)
        z_velocity[n + 1] = (0.5 * energy * z[n]) / (n + 1)
        rho = scalar_series_product(z[: n + 1, 0], z[: n + 1, 0], n)
        rho += scalar_series_product(z[: n + 1, 1], z[: n + 1, 1], n)
        physical_time[n + 1] = rho[n] / (n + 1)

    return LeviCivitaChart(z=z, z_velocity=z_velocity, physical_time=physical_time, energy=float(energy))


def integrate_relative_kepler(relative_position: Array, relative_velocity: Array, mu: float, time: float) -> Array:
    """Reference physical-time integrator for the isolated relative Kepler problem."""

    if mu <= 0.0:
        raise ValueError("mu must be positive")
    q0 = np.asarray(relative_position, dtype=float)
    v0 = np.asarray(relative_velocity, dtype=float)
    state0 = np.concatenate([q0, v0])

    def rhs(_time: float, state: Array) -> Array:
        q = state[:2]
        v = state[2:]
        radius = np.linalg.norm(q)
        if radius == 0.0:
            raise FloatingPointError("collision in physical Kepler integrator")
        acceleration = -mu * q / radius**3
        return np.concatenate([v, acceleration])

    solution = solve_ivp(rhs, (0.0, time), state0, method="DOP853", rtol=1e-12, atol=1e-14)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1]

