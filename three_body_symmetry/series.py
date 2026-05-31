"""Local Taylor-series construction for the general non-collision three-body problem.

The series code is deliberately more general than the figure-eight harness: it
allows arbitrary positive masses and arbitrary spatial dimension. It constructs
the analytic local solution by lifting positions into a differential algebra of
truncated power series, doing the inverse-distance operations there, and then
projecting the coefficients back to body coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .intervals import (
    FloatInterval,
    interval_array_contains_point,
    interval_array_series_eval,
    interval_series_add,
    interval_series_power,
    interval_series_product,
    zero_interval,
)


Array = np.ndarray


@dataclass(frozen=True)
class TaylorSolution:
    """Power-series coefficients for positions and velocities."""

    position: Array
    velocity: Array
    masses: Array

    @property
    def order(self) -> int:
        return int(self.position.shape[0] - 1)

    @property
    def body_count(self) -> int:
        return int(self.position.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.position.shape[2])

    def positions_at(self, time: float) -> Array:
        return _evaluate(self.position, time)

    def velocities_at(self, time: float) -> Array:
        return _evaluate(self.velocity, time)

    def state_at(self, time: float) -> Array:
        return np.concatenate([self.positions_at(time).reshape(-1), self.velocities_at(time).reshape(-1)])


@dataclass(frozen=True)
class IntervalTaylorSolution:
    """Interval enclosures for Taylor coefficients of positions and velocities."""

    position: Array
    velocity: Array
    masses: Array

    @property
    def order(self) -> int:
        return int(self.position.shape[0] - 1)

    @property
    def body_count(self) -> int:
        return int(self.position.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.position.shape[2])

    @property
    def position_lower(self) -> Array:
        return _interval_lower(self.position)

    @property
    def position_upper(self) -> Array:
        return _interval_upper(self.position)

    @property
    def velocity_lower(self) -> Array:
        return _interval_lower(self.velocity)

    @property
    def velocity_upper(self) -> Array:
        return _interval_upper(self.velocity)

    def contains_point_solution(self, solution: TaylorSolution) -> bool:
        if solution.position.shape != self.position.shape or solution.velocity.shape != self.velocity.shape:
            return False
        return bool(
            np.all(self.position_lower <= solution.position)
            and np.all(solution.position <= self.position_upper)
            and np.all(self.velocity_lower <= solution.velocity)
            and np.all(solution.velocity <= self.velocity_upper)
        )

    def positions_at(self, time: float) -> Array:
        return interval_array_series_eval(self.position, FloatInterval.point(time))

    def velocities_at(self, time: float) -> Array:
        return interval_array_series_eval(self.velocity, FloatInterval.point(time))

    def state_at(self, time: float) -> Array:
        return np.concatenate([self.positions_at(time).reshape(-1), self.velocities_at(time).reshape(-1)])

    def state_contains(self, state: Array, time: float) -> bool:
        return interval_array_contains_point(self.state_at(time), np.asarray(state, dtype=float))


def _validate_initial_data(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)

    if positions.ndim != 2:
        raise ValueError("positions must have shape (body_count, dimension)")
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    if positions.shape[0] != 3:
        raise ValueError("this module constructs three-body solutions")
    if masses.shape != (3,):
        raise ValueError("masses must have shape (3,)")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive")
    for i in range(3):
        for j in range(i + 1, 3):
            if np.linalg.norm(positions[i] - positions[j]) == 0.0:
                raise ValueError("initial data must be collision-free")
    return positions, velocities, masses


def _as_interval(value: object) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


def _validate_interval_initial_data(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=object)
    velocities = np.asarray(velocities, dtype=object)
    masses = np.asarray(masses, dtype=float)

    if positions.ndim != 2:
        raise ValueError("positions must have shape (body_count, dimension)")
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    if positions.shape[0] != 3:
        raise ValueError("this module constructs three-body solutions")
    if masses.shape != (3,):
        raise ValueError("masses must have shape (3,)")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive")

    interval_positions = np.empty(positions.shape, dtype=object)
    interval_velocities = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        interval_positions[index] = _as_interval(positions[index])
        interval_velocities[index] = _as_interval(velocities[index])
    return interval_positions, interval_velocities, masses


def _evaluate(coefficients: Array, time: float) -> Array:
    value = np.zeros(coefficients.shape[1:], dtype=float)
    for coefficient in coefficients[::-1]:
        value = value * time + coefficient
    return value


def _interval_zeros(shape: tuple[int, ...]) -> Array:
    out = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        out[index] = zero_interval()
    return out


def _interval_array_from_points(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    out = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        out[index] = FloatInterval.point(values[index])
    return out


def _construct_interval_taylor_from_arrays(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
) -> IntervalTaylorSolution:
    body_count, dimension = positions.shape
    q = _interval_zeros((order + 1, body_count, dimension))
    v = _interval_zeros((order + 1, body_count, dimension))
    q[0] = positions
    v[0] = velocities
    for n in range(order):
        acc = acceleration_interval_coefficients(q, masses, n)
        scale = 1.0 / (n + 1)
        for index in np.ndindex((body_count, dimension)):
            q[(n + 1, *index)] = v[(n, *index)].scale(scale)
            v[(n + 1, *index)] = acc[(n, *index)].scale(scale)
    return IntervalTaylorSolution(position=q, velocity=v, masses=masses)


def _interval_lower(values: Array) -> Array:
    out = np.empty(values.shape, dtype=float)
    for index in np.ndindex(values.shape):
        out[index] = values[index].lower
    return out


def _interval_upper(values: Array) -> Array:
    out = np.empty(values.shape, dtype=float)
    for index in np.ndindex(values.shape):
        out[index] = values[index].upper
    return out


def _product(left: Array, right: Array, max_degree: int) -> Array:
    out = np.zeros(max_degree + 1, dtype=float)
    for n in range(max_degree + 1):
        out[n] = sum(left[k] * right[n - k] for k in range(n + 1))
    return out


def scalar_series_product(left: Array, right: Array, max_degree: int) -> Array:
    """Return the truncated product of two scalar power series."""

    return _product(left, right, max_degree)


def _series_power(base: Array, exponent: float, max_degree: int) -> Array:
    """Return coefficients of `base ** exponent` through `max_degree`.

    The recurrence follows from `base * y' = exponent * base' * y`.
    """

    base = np.asarray(base[: max_degree + 1], dtype=float)
    if base[0] <= 0.0:
        raise ValueError("series power requires a positive constant term")
    out = np.zeros(max_degree + 1, dtype=float)
    out[0] = base[0] ** exponent
    for n in range(1, max_degree + 1):
        right = exponent * sum(i * base[i] * out[n - i] for i in range(1, n + 1))
        left_known = sum(base[i] * (n - i) * out[n - i] for i in range(1, n))
        out[n] = (right - left_known) / (n * base[0])
    return out


def scalar_series_power(base: Array, exponent: float, max_degree: int) -> Array:
    """Return coefficients of a scalar power series raised to `exponent`."""

    return _series_power(base, exponent, max_degree)


def acceleration_coefficients(position: Array, masses: Array, max_degree: int) -> Array:
    """Compute acceleration-series coefficients from position coefficients."""

    position = np.asarray(position, dtype=float)
    masses = np.asarray(masses, dtype=float)
    body_count, dimension = position.shape[1:]
    acc = np.zeros((max_degree + 1, body_count, dimension), dtype=float)

    for i in range(body_count):
        for j in range(i + 1, body_count):
            delta = position[: max_degree + 1, j] - position[: max_degree + 1, i]
            distance_squared = np.zeros(max_degree + 1, dtype=float)
            for axis in range(dimension):
                distance_squared += _product(delta[:, axis], delta[:, axis], max_degree)
            inverse_cube = _series_power(distance_squared, -1.5, max_degree)
            pair = np.zeros((max_degree + 1, dimension), dtype=float)
            for n in range(max_degree + 1):
                for k in range(n + 1):
                    pair[n] += delta[k] * inverse_cube[n - k]
            acc[:, i] += masses[j] * pair
            acc[:, j] -= masses[i] * pair
    return acc


def acceleration_interval_coefficients(position: Array, masses: Array, max_degree: int) -> Array:
    """Compute interval acceleration-series coefficients from interval positions."""

    position = np.asarray(position, dtype=object)
    masses = np.asarray(masses, dtype=float)
    body_count, dimension = position.shape[1:]
    acc = _interval_zeros((max_degree + 1, body_count, dimension))

    for i in range(body_count):
        for j in range(i + 1, body_count):
            delta_by_axis = []
            distance_squared = tuple(zero_interval() for _ in range(max_degree + 1))
            for axis in range(dimension):
                delta = tuple(
                    position[n, j, axis] - position[n, i, axis]
                    for n in range(max_degree + 1)
                )
                delta_by_axis.append(delta)
                distance_squared = interval_series_add(
                    distance_squared,
                    interval_series_product(delta, delta, max_degree),
                )
            inverse_cube = interval_series_power(distance_squared, -1.5, max_degree)
            for n in range(max_degree + 1):
                for axis in range(dimension):
                    pair = zero_interval()
                    delta = delta_by_axis[axis]
                    for k in range(n + 1):
                        pair = pair + delta[k] * inverse_cube[n - k]
                    acc[n, i, axis] = acc[n, i, axis] + pair.scale(masses[j])
                    acc[n, j, axis] = acc[n, j, axis] - pair.scale(masses[i])
    return acc


def construct_taylor_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
) -> TaylorSolution:
    """Construct the local analytic solution through a fixed Taylor order."""

    if order < 1:
        raise ValueError("order must be at least 1")
    positions, velocities, masses = _validate_initial_data(positions, velocities, masses)
    body_count, dimension = positions.shape

    q = np.zeros((order + 1, body_count, dimension), dtype=float)
    v = np.zeros_like(q)
    q[0] = positions
    v[0] = velocities
    for n in range(order):
        acc = acceleration_coefficients(q, masses, n)
        q[n + 1] = v[n] / (n + 1)
        v[n + 1] = acc[n] / (n + 1)
    return TaylorSolution(position=q, velocity=v, masses=masses)


def construct_interval_taylor_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
) -> IntervalTaylorSolution:
    """Construct outward-rounded interval Taylor coefficients for ordinary charts."""

    if order < 1:
        raise ValueError("order must be at least 1")
    positions, velocities, masses = _validate_initial_data(positions, velocities, masses)
    return _construct_interval_taylor_from_arrays(
        _interval_array_from_points(positions),
        _interval_array_from_points(velocities),
        masses,
        order=order,
    )


def construct_interval_taylor_solution_from_intervals(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
) -> IntervalTaylorSolution:
    """Construct ordinary interval Taylor coefficients from interval initial data."""

    if order < 1:
        raise ValueError("order must be at least 1")
    positions, velocities, masses = _validate_interval_initial_data(positions, velocities, masses)
    return _construct_interval_taylor_from_arrays(positions, velocities, masses, order=order)


def integrate_reference(
    positions: Array,
    velocities: Array,
    masses: Array,
    time: float,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> Array:
    """Numerically integrate arbitrary-dimension three-body data for comparison."""

    positions, velocities, masses = _validate_initial_data(positions, velocities, masses)
    dimension = positions.shape[1]
    state0 = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])

    def rhs(_time: float, state: Array) -> Array:
        q = state[: 3 * dimension].reshape(3, dimension)
        v = state[3 * dimension :].reshape(3, dimension)
        acc = np.zeros_like(q)
        for i in range(3):
            for j in range(i + 1, 3):
                delta = q[j] - q[i]
                distance = np.linalg.norm(delta)
                force_shape = delta / distance**3
                acc[i] += masses[j] * force_shape
                acc[j] -= masses[i] * force_shape
        return np.concatenate([v.reshape(-1), acc.reshape(-1)])

    solution = solve_ivp(rhs, (0.0, time), state0, method="DOP853", rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1]


def center_of_mass_coefficients(solution: TaylorSolution) -> Array:
    total_mass = float(np.sum(solution.masses))
    return np.sum(solution.position * solution.masses[None, :, None], axis=1) / total_mass


def linear_momentum_coefficients(solution: TaylorSolution) -> Array:
    return np.sum(solution.velocity * solution.masses[None, :, None], axis=1)
