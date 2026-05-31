"""Newtonian three-body dynamics and verification utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


Array = np.ndarray


@dataclass(frozen=True)
class OrbitSample:
    """Integrated orbit samples."""

    times: Array
    states: Array

    @property
    def final_state(self) -> Array:
        return self.states[-1]


def pack_state(positions: Array, velocities: Array) -> Array:
    """Pack `(3, dimension)` positions and velocities into one state vector."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("positions must have shape (3, dimension)")
    if velocities.shape != positions.shape:
        raise ValueError("positions and velocities must have matching shapes")
    return np.concatenate([positions.reshape(-1), velocities.reshape(-1)])


def split_state(state: Array) -> tuple[Array, Array]:
    """Return `(positions, velocities)` from a packed three-body state vector."""

    state = np.asarray(state, dtype=float)
    if state.ndim != 1 or state.shape[0] % 6 != 0:
        raise ValueError("state must be a flat vector of length 6 * dimension")
    dimension = state.shape[0] // 6
    coordinate_count = 3 * dimension
    return (
        state[:coordinate_count].reshape(3, dimension),
        state[coordinate_count:].reshape(3, dimension),
    )


def accelerations(
    positions: Array,
    masses: Array | None = None,
    gravitational_constant: float = 1.0,
) -> Array:
    """Compute Newtonian accelerations for three bodies in any dimension."""

    positions = np.asarray(positions, dtype=float)
    masses = np.ones(3) if masses is None else np.asarray(masses, dtype=float)
    if positions.ndim != 2 or positions.shape[0] != 3 or masses.shape != (3,):
        raise ValueError("expected positions shape (3, dimension) and masses shape (3,)")

    acc = np.zeros_like(positions)
    for i in range(3):
        for j in range(i + 1, 3):
            delta = positions[j] - positions[i]
            distance = np.linalg.norm(delta)
            if distance == 0.0:
                raise FloatingPointError("collision singularity in acceleration")
            direction_over_r2 = delta / distance**3
            acc[i] += gravitational_constant * masses[j] * direction_over_r2
            acc[j] -= gravitational_constant * masses[i] * direction_over_r2
    return acc


def rhs(
    _time: float,
    state: Array,
    masses: Array | None = None,
    gravitational_constant: float = 1.0,
) -> Array:
    """First-order ODE for the Newtonian three-body problem."""

    positions, velocities = split_state(state)
    acc = accelerations(positions, masses, gravitational_constant)
    return np.concatenate([velocities.reshape(-1), acc.reshape(-1)])


def integrate(
    initial_state: Array,
    t_final: float,
    *,
    masses: Array | None = None,
    gravitational_constant: float = 1.0,
    samples: int | None = None,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> OrbitSample:
    """Integrate an orbit to any finite target time using DOP853."""

    initial_state = np.asarray(initial_state, dtype=float)
    if samples is not None and samples < 1:
        raise ValueError("samples must be at least one")
    if t_final == 0.0:
        times = np.array([0.0]) if samples is None else np.zeros(samples)
        states = np.repeat(initial_state[None, :], len(times), axis=0)
        return OrbitSample(times=times, states=states)
    t_eval = None if samples is None else np.linspace(0.0, t_final, samples)
    solution = solve_ivp(
        rhs,
        (0.0, t_final),
        initial_state,
        args=(masses, gravitational_constant),
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    states = solution.y.T
    return OrbitSample(times=solution.t, states=states)


def center_of_mass(state: Array, masses: Array | None = None) -> Array:
    positions, _velocities = split_state(state)
    masses = np.ones(3) if masses is None else np.asarray(masses, dtype=float)
    return np.average(positions, axis=0, weights=masses)


def linear_momentum(state: Array, masses: Array | None = None) -> Array:
    _positions, velocities = split_state(state)
    masses = np.ones(3) if masses is None else np.asarray(masses, dtype=float)
    return np.sum(masses[:, None] * velocities, axis=0)


def angular_momentum_components(state: Array, masses: Array | None = None) -> Array:
    """Return angular-momentum bivector components in coordinate-pair order.

    For planar states this returns one `(xy)` component. For spatial states it
    returns `(xy, xz, yz)`, corresponding to `(L_z, -L_y, L_x)` in the usual
    axial-vector convention.
    """

    positions, velocities = split_state(state)
    if positions.shape[1] < 2:
        raise ValueError("angular_momentum_components requires dimension at least two")
    masses = np.ones(3) if masses is None else np.asarray(masses, dtype=float)
    if masses.shape != (3,):
        raise ValueError("masses must have shape (3,)")
    momenta = masses[:, None] * velocities
    components = []
    for first_axis in range(positions.shape[1]):
        for second_axis in range(first_axis + 1, positions.shape[1]):
            components.append(
                float(
                    np.sum(
                        positions[:, first_axis] * momenta[:, second_axis]
                        - positions[:, second_axis] * momenta[:, first_axis]
                    )
                )
            )
    return np.asarray(components, dtype=float)


def angular_momentum_z(state: Array, masses: Array | None = None) -> float:
    return float(angular_momentum_components(state, masses)[0])


def energy(
    state: Array,
    masses: Array | None = None,
    gravitational_constant: float = 1.0,
) -> float:
    positions, velocities = split_state(state)
    masses = np.ones(3) if masses is None else np.asarray(masses, dtype=float)
    kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)
    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            distance = np.linalg.norm(positions[i] - positions[j])
            potential -= gravitational_constant * masses[i] * masses[j] / distance
    return float(kinetic + potential)


def max_periodic_error(initial_state: Array, period: float, **integrate_kwargs: float) -> float:
    """Integrate one period and return the infinity norm of the closing error."""

    final_state = integrate(initial_state, period, **integrate_kwargs).final_state
    return float(np.linalg.norm(final_state - initial_state, ord=np.inf))
