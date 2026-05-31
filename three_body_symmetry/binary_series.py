"""Taylor series for the regularized planar binary-collision chart."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .binary_chart import (
    IntervalRegularizedBinaryCollisionChartState,
    RegularizedBinaryCollisionChartDerivative,
    RegularizedBinaryCollisionChartState,
    regularized_binary_collision_chart_rhs,
)
from .intervals import FloatInterval, interval_series_add, interval_series_power, interval_series_product, zero_interval
from .intervals import interval_array_contains_point, interval_array_series_eval
from .series import scalar_series_power, scalar_series_product


Array = np.ndarray


@dataclass(frozen=True)
class RegularizedBinaryTaylorSolution:
    masses: Array
    pair: tuple[int, int]
    z: Array
    z_velocity: Array
    pair_energy: Array
    binary_center: Array
    binary_center_velocity: Array
    third_offset: Array
    third_offset_velocity: Array
    physical_time: Array

    @property
    def order(self) -> int:
        return int(self.z.shape[0] - 1)

    def state_at(self, s_value: float) -> RegularizedBinaryCollisionChartState:
        return RegularizedBinaryCollisionChartState(
            masses=self.masses,
            pair=self.pair,
            z=_evaluate(self.z, s_value),
            z_velocity=_evaluate(self.z_velocity, s_value),
            pair_energy=float(_evaluate(self.pair_energy[:, None], s_value)[0]),
            binary_center=_evaluate(self.binary_center, s_value),
            binary_center_velocity=_evaluate(self.binary_center_velocity, s_value),
            third_offset=_evaluate(self.third_offset, s_value),
            third_offset_velocity=_evaluate(self.third_offset_velocity, s_value),
        )

    def physical_time_at(self, s_value: float) -> float:
        return float(_evaluate(self.physical_time[:, None], s_value)[0])

    def vector_at(self, s_value: float) -> Array:
        return pack_regularized_state(self.state_at(s_value), self.physical_time_at(s_value))


@dataclass(frozen=True)
class IntervalRegularizedBinaryTaylorSolution:
    masses: Array
    pair: tuple[int, int]
    z: Array
    z_velocity: Array
    pair_energy: Array
    binary_center: Array
    binary_center_velocity: Array
    third_offset: Array
    third_offset_velocity: Array
    physical_time: Array

    @property
    def order(self) -> int:
        return int(self.z.shape[0] - 1)

    def contains_point_solution(self, solution: RegularizedBinaryTaylorSolution) -> bool:
        checks = [
            (self.z, solution.z),
            (self.z_velocity, solution.z_velocity),
            (self.pair_energy, solution.pair_energy),
            (self.binary_center, solution.binary_center),
            (self.binary_center_velocity, solution.binary_center_velocity),
            (self.third_offset, solution.third_offset),
            (self.third_offset_velocity, solution.third_offset_velocity),
            (self.physical_time, solution.physical_time),
        ]
        return all(_interval_array_contains(interval, point) for interval, point in checks)

    def z_at(self, s_value: float) -> Array:
        return interval_array_series_eval(self.z, FloatInterval.point(s_value))

    def z_velocity_at(self, s_value: float) -> Array:
        return interval_array_series_eval(self.z_velocity, FloatInterval.point(s_value))

    def pair_energy_at(self, s_value: float) -> FloatInterval:
        return interval_array_series_eval(self.pair_energy[:, None], FloatInterval.point(s_value))[0]

    def binary_center_at(self, s_value: float) -> Array:
        return interval_array_series_eval(self.binary_center, FloatInterval.point(s_value))

    def binary_center_velocity_at(self, s_value: float) -> Array:
        return interval_array_series_eval(self.binary_center_velocity, FloatInterval.point(s_value))

    def third_offset_at(self, s_value: float) -> Array:
        return interval_array_series_eval(self.third_offset, FloatInterval.point(s_value))

    def third_offset_velocity_at(self, s_value: float) -> Array:
        return interval_array_series_eval(self.third_offset_velocity, FloatInterval.point(s_value))

    def physical_time_at(self, s_value: float) -> FloatInterval:
        return interval_array_series_eval(self.physical_time[:, None], FloatInterval.point(s_value))[0]

    def vector_at(self, s_value: float) -> Array:
        return np.concatenate(
            [
                self.z_at(s_value),
                self.z_velocity_at(s_value),
                np.array([self.pair_energy_at(s_value)], dtype=object),
                self.binary_center_at(s_value),
                self.binary_center_velocity_at(s_value),
                self.third_offset_at(s_value),
                self.third_offset_velocity_at(s_value),
                np.array([self.physical_time_at(s_value)], dtype=object),
            ]
        )

    def vector_contains(self, vector: Array, s_value: float) -> bool:
        return interval_array_contains_point(self.vector_at(s_value), np.asarray(vector, dtype=float))


def _evaluate(coefficients: Array, value: float) -> Array:
    out = np.zeros(coefficients.shape[1:], dtype=float)
    for coefficient in coefficients[::-1]:
        out = out * value + coefficient
    return out


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


def _as_interval(value: object) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


def _interval_array_contains(interval: Array, point: Array) -> bool:
    if interval.shape != point.shape:
        return False
    for index in np.ndindex(interval.shape):
        value = interval[index]
        if value.lower > point[index] or point[index] > value.upper:
            return False
    return True


def _vector_product(scalar: Array, vector: Array, max_degree: int) -> Array:
    out = np.zeros_like(vector[: max_degree + 1])
    for n in range(max_degree + 1):
        for k in range(n + 1):
            out[n] += scalar[k] * vector[n - k]
    return out


def _vector_product_interval(scalar: tuple[FloatInterval, ...], vector: Array, max_degree: int) -> Array:
    out = _interval_zeros(vector[: max_degree + 1].shape)
    for n in range(max_degree + 1):
        for axis in range(vector.shape[1]):
            value = zero_interval()
            for k in range(n + 1):
                value = value + scalar[k] * _as_interval(vector[n - k, axis])
            out[n, axis] = value
    return out


def _vector_dot(left: Array, right: Array, max_degree: int) -> Array:
    out = np.zeros(max_degree + 1, dtype=float)
    for axis in range(left.shape[1]):
        out += scalar_series_product(left[:, axis], right[:, axis], max_degree)
    return out


def _vector_dot_interval(left: Array, right: Array, max_degree: int) -> tuple[FloatInterval, ...]:
    out = tuple(zero_interval() for _ in range(max_degree + 1))
    for axis in range(left.shape[1]):
        left_axis = tuple(_as_interval(left[n, axis]) for n in range(max_degree + 1))
        right_axis = tuple(_as_interval(right[n, axis]) for n in range(max_degree + 1))
        out = interval_series_add(out, interval_series_product(left_axis, right_axis, max_degree))
    return out


def _lc_square_coefficients(z: Array, max_degree: int) -> Array:
    x = z[: max_degree + 1, 0]
    y = z[: max_degree + 1, 1]
    return np.column_stack(
        [
            scalar_series_product(x, x, max_degree) - scalar_series_product(y, y, max_degree),
            2.0 * scalar_series_product(x, y, max_degree),
        ]
    )


def _lc_square_coefficients_interval(z: Array, max_degree: int) -> Array:
    x = tuple(_as_interval(z[n, 0]) for n in range(max_degree + 1))
    y = tuple(_as_interval(z[n, 1]) for n in range(max_degree + 1))
    xx = interval_series_product(x, x, max_degree)
    yy = interval_series_product(y, y, max_degree)
    xy = interval_series_product(x, y, max_degree)
    out = _interval_zeros((max_degree + 1, 2))
    for n in range(max_degree + 1):
        out[n, 0] = xx[n] - yy[n]
        out[n, 1] = xy[n].scale(2.0)
    return out


def rho_coefficients(z: Array, max_degree: int) -> Array:
    x = z[: max_degree + 1, 0]
    y = z[: max_degree + 1, 1]
    return scalar_series_product(x, x, max_degree) + scalar_series_product(y, y, max_degree)


def rho_interval_coefficients(z: Array, max_degree: int) -> tuple[FloatInterval, ...]:
    x = tuple(_as_interval(z[n, 0]) for n in range(max_degree + 1))
    y = tuple(_as_interval(z[n, 1]) for n in range(max_degree + 1))
    return interval_series_add(
        interval_series_product(x, x, max_degree),
        interval_series_product(y, y, max_degree),
    )


def lc_jacobian_times_coefficients(z: Array, vector: Array, max_degree: int) -> Array:
    """Coefficients of `A(z) vector` where `A` is Jacobian of `z^2`."""

    x = z[: max_degree + 1, 0]
    y = z[: max_degree + 1, 1]
    vx = vector[: max_degree + 1, 0]
    vy = vector[: max_degree + 1, 1]
    return np.column_stack(
        [
            2.0 * scalar_series_product(x, vx, max_degree)
            - 2.0 * scalar_series_product(y, vy, max_degree),
            2.0 * scalar_series_product(y, vx, max_degree)
            + 2.0 * scalar_series_product(x, vy, max_degree),
        ]
    )


def lc_jacobian_times_interval_coefficients(z: Array, vector: Array, max_degree: int) -> Array:
    x = tuple(_as_interval(z[n, 0]) for n in range(max_degree + 1))
    y = tuple(_as_interval(z[n, 1]) for n in range(max_degree + 1))
    vx = tuple(_as_interval(vector[n, 0]) for n in range(max_degree + 1))
    vy = tuple(_as_interval(vector[n, 1]) for n in range(max_degree + 1))
    xvx = interval_series_product(x, vx, max_degree)
    yvy = interval_series_product(y, vy, max_degree)
    yvx = interval_series_product(y, vx, max_degree)
    xvy = interval_series_product(x, vy, max_degree)
    out = _interval_zeros((max_degree + 1, 2))
    for n in range(max_degree + 1):
        out[n, 0] = (xvx[n] - yvy[n]).scale(2.0)
        out[n, 1] = (yvx[n] + xvy[n]).scale(2.0)
    return out


def lc_jacobian_transpose_times_coefficients(z: Array, vector: Array, max_degree: int) -> Array:
    """Coefficients of `A(z)^T vector` where `A` is Jacobian of `z^2`."""

    x = z[: max_degree + 1, 0]
    y = z[: max_degree + 1, 1]
    vx = vector[: max_degree + 1, 0]
    vy = vector[: max_degree + 1, 1]
    return np.column_stack(
        [
            2.0 * scalar_series_product(x, vx, max_degree)
            + 2.0 * scalar_series_product(y, vy, max_degree),
            -2.0 * scalar_series_product(y, vx, max_degree)
            + 2.0 * scalar_series_product(x, vy, max_degree),
        ]
    )


def lc_jacobian_transpose_times_interval_coefficients(z: Array, vector: Array, max_degree: int) -> Array:
    x = tuple(_as_interval(z[n, 0]) for n in range(max_degree + 1))
    y = tuple(_as_interval(z[n, 1]) for n in range(max_degree + 1))
    vx = tuple(_as_interval(vector[n, 0]) for n in range(max_degree + 1))
    vy = tuple(_as_interval(vector[n, 1]) for n in range(max_degree + 1))
    xvx = interval_series_product(x, vx, max_degree)
    yvy = interval_series_product(y, vy, max_degree)
    yvx = interval_series_product(y, vx, max_degree)
    xvy = interval_series_product(x, vy, max_degree)
    out = _interval_zeros((max_degree + 1, 2))
    for n in range(max_degree + 1):
        out[n, 0] = (xvx[n] + yvy[n]).scale(2.0)
        out[n, 1] = (xvy[n] - yvx[n]).scale(2.0)
    return out


def _inverse_square_field_coefficients(vector: Array, max_degree: int) -> Array:
    distance_squared = np.zeros(max_degree + 1, dtype=float)
    for axis in range(2):
        distance_squared += scalar_series_product(vector[:, axis], vector[:, axis], max_degree)
    inverse_cube = scalar_series_power(distance_squared, -1.5, max_degree)
    return _vector_product(inverse_cube, vector, max_degree)


def _inverse_square_field_interval_coefficients(vector: Array, max_degree: int) -> Array:
    distance_squared = tuple(zero_interval() for _ in range(max_degree + 1))
    for axis in range(2):
        axis_series = tuple(_as_interval(vector[n, axis]) for n in range(max_degree + 1))
        distance_squared = interval_series_add(
            distance_squared,
            interval_series_product(axis_series, axis_series, max_degree),
        )
    inverse_cube = interval_series_power(distance_squared, -1.5, max_degree)
    return _vector_product_interval(inverse_cube, vector, max_degree)


def analytic_field_coefficients(
    solution: RegularizedBinaryTaylorSolution,
    max_degree: int,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return `(Rdd, ydd, p, rho, q_prime)` coefficient arrays."""

    first, second = solution.pair
    third = ({0, 1, 2} - set(solution.pair)).pop()
    masses = solution.masses
    pair_mass = masses[first] + masses[second]
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    relative_position = _lc_square_coefficients(solution.z, max_degree)
    from_first = solution.third_offset[: max_degree + 1] + alpha * relative_position
    from_second = solution.third_offset[: max_degree + 1] - beta * relative_position
    field_first = _inverse_square_field_coefficients(from_first, max_degree)
    field_second = _inverse_square_field_coefficients(from_second, max_degree)
    binary_center_acceleration = (
        masses[third] / pair_mass * (masses[first] * field_first + masses[second] * field_second)
    )
    third_acceleration = -masses[first] * field_first - masses[second] * field_second
    third_offset_acceleration = third_acceleration - binary_center_acceleration
    relative_perturbation = masses[third] * (field_second - field_first)
    rho = rho_coefficients(solution.z, max_degree)
    q_prime = lc_jacobian_times_coefficients(solution.z, solution.z_velocity, max_degree)
    return binary_center_acceleration, third_offset_acceleration, relative_perturbation, rho, q_prime


def _scaled_vector_sum_interval(
    first: Array,
    first_scale: float,
    second: Array,
    second_scale: float,
    max_degree: int,
) -> Array:
    out = _interval_zeros((max_degree + 1, 2))
    for n in range(max_degree + 1):
        for axis in range(2):
            out[n, axis] = _as_interval(first[n, axis]).scale(first_scale) + _as_interval(second[n, axis]).scale(second_scale)
    return out


def _vector_difference_interval(left: Array, right: Array, max_degree: int) -> Array:
    out = _interval_zeros((max_degree + 1, 2))
    for n in range(max_degree + 1):
        for axis in range(2):
            out[n, axis] = _as_interval(left[n, axis]) - _as_interval(right[n, axis])
    return out


def analytic_field_interval_coefficients(
    solution: IntervalRegularizedBinaryTaylorSolution,
    max_degree: int,
) -> tuple[Array, Array, Array, tuple[FloatInterval, ...], Array]:
    """Interval enclosures for `(Rdd, ydd, p, rho, q_prime)` coefficients."""

    first, second = solution.pair
    third = ({0, 1, 2} - set(solution.pair)).pop()
    masses = solution.masses
    pair_mass = masses[first] + masses[second]
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    relative_position = _lc_square_coefficients_interval(solution.z, max_degree)
    from_first = _scaled_vector_sum_interval(solution.third_offset, 1.0, relative_position, alpha, max_degree)
    from_second = _scaled_vector_sum_interval(solution.third_offset, 1.0, relative_position, -beta, max_degree)
    field_first = _inverse_square_field_interval_coefficients(from_first, max_degree)
    field_second = _inverse_square_field_interval_coefficients(from_second, max_degree)
    binary_center_acceleration = _scaled_vector_sum_interval(
        field_first,
        masses[third] * masses[first] / pair_mass,
        field_second,
        masses[third] * masses[second] / pair_mass,
        max_degree,
    )
    third_acceleration = _scaled_vector_sum_interval(
        field_first,
        -masses[first],
        field_second,
        -masses[second],
        max_degree,
    )
    third_offset_acceleration = _vector_difference_interval(
        third_acceleration,
        binary_center_acceleration,
        max_degree,
    )
    relative_perturbation = _scaled_vector_sum_interval(
        field_second,
        masses[third],
        field_first,
        -masses[third],
        max_degree,
    )
    rho = rho_interval_coefficients(solution.z, max_degree)
    q_prime = lc_jacobian_times_interval_coefficients(solution.z, solution.z_velocity, max_degree)
    return binary_center_acceleration, third_offset_acceleration, relative_perturbation, rho, q_prime


def regularized_rhs_coefficients(
    solution: RegularizedBinaryTaylorSolution,
    max_degree: int,
) -> RegularizedBinaryCollisionChartDerivative:
    """Compute coefficient arrays for the regularized binary RHS through `max_degree`."""

    center_acc, third_offset_acc, perturbation, rho, q_prime = analytic_field_coefficients(solution, max_degree)
    z_linear = _vector_product(solution.pair_energy[: max_degree + 1], solution.z, max_degree)
    perturbing_z = _vector_product(
        rho,
        lc_jacobian_transpose_times_coefficients(solution.z, perturbation, max_degree),
        max_degree,
    )
    return RegularizedBinaryCollisionChartDerivative(
        z=solution.z_velocity[: max_degree + 1],
        z_velocity=0.5 * z_linear + 0.25 * perturbing_z,
        pair_energy=_vector_dot(q_prime, perturbation, max_degree),
        binary_center=_vector_product(rho, solution.binary_center_velocity, max_degree),
        binary_center_velocity=_vector_product(rho, center_acc, max_degree),
        third_offset=_vector_product(rho, solution.third_offset_velocity, max_degree),
        third_offset_velocity=_vector_product(rho, third_offset_acc, max_degree),
        physical_time=rho,
    )


def regularized_rhs_interval_coefficients(
    solution: IntervalRegularizedBinaryTaylorSolution,
    max_degree: int,
) -> RegularizedBinaryCollisionChartDerivative:
    """Compute interval coefficient enclosures for the regularized binary RHS."""

    center_acc, third_offset_acc, perturbation, rho, q_prime = analytic_field_interval_coefficients(solution, max_degree)
    pair_energy = tuple(_as_interval(solution.pair_energy[n]) for n in range(max_degree + 1))
    z_linear = _vector_product_interval(pair_energy, solution.z, max_degree)
    perturbing_z = _vector_product_interval(
        rho,
        lc_jacobian_transpose_times_interval_coefficients(solution.z, perturbation, max_degree),
        max_degree,
    )
    z_velocity = _scaled_vector_sum_interval(z_linear, 0.5, perturbing_z, 0.25, max_degree)
    return RegularizedBinaryCollisionChartDerivative(
        z=solution.z_velocity[: max_degree + 1],
        z_velocity=z_velocity,
        pair_energy=_vector_dot_interval(q_prime, perturbation, max_degree),
        binary_center=_vector_product_interval(rho, solution.binary_center_velocity, max_degree),
        binary_center_velocity=_vector_product_interval(rho, center_acc, max_degree),
        third_offset=_vector_product_interval(rho, solution.third_offset_velocity, max_degree),
        third_offset_velocity=_vector_product_interval(rho, third_offset_acc, max_degree),
        physical_time=rho,
    )


def construct_regularized_binary_taylor_solution(
    initial_state: RegularizedBinaryCollisionChartState,
    *,
    order: int,
) -> RegularizedBinaryTaylorSolution:
    """Construct a local Taylor chart for the regularized binary-collision RHS."""

    if order < 1:
        raise ValueError("order must be at least 1")
    z = np.zeros((order + 1, 2), dtype=float)
    z_velocity = np.zeros_like(z)
    pair_energy = np.zeros(order + 1, dtype=float)
    binary_center = np.zeros_like(z)
    binary_center_velocity = np.zeros_like(z)
    third_offset = np.zeros_like(z)
    third_offset_velocity = np.zeros_like(z)
    physical_time = np.zeros(order + 1, dtype=float)
    z[0] = initial_state.z
    z_velocity[0] = initial_state.z_velocity
    pair_energy[0] = initial_state.pair_energy
    binary_center[0] = initial_state.binary_center
    binary_center_velocity[0] = initial_state.binary_center_velocity
    third_offset[0] = initial_state.third_offset
    third_offset_velocity[0] = initial_state.third_offset_velocity

    solution = RegularizedBinaryTaylorSolution(
        masses=np.asarray(initial_state.masses, dtype=float),
        pair=initial_state.pair,
        z=z,
        z_velocity=z_velocity,
        pair_energy=pair_energy,
        binary_center=binary_center,
        binary_center_velocity=binary_center_velocity,
        third_offset=third_offset,
        third_offset_velocity=third_offset_velocity,
        physical_time=physical_time,
    )
    for n in range(order):
        rhs = regularized_rhs_coefficients(solution, n)
        z[n + 1] = rhs.z[n] / (n + 1)
        z_velocity[n + 1] = rhs.z_velocity[n] / (n + 1)
        pair_energy[n + 1] = rhs.pair_energy[n] / (n + 1)
        binary_center[n + 1] = rhs.binary_center[n] / (n + 1)
        binary_center_velocity[n + 1] = rhs.binary_center_velocity[n] / (n + 1)
        third_offset[n + 1] = rhs.third_offset[n] / (n + 1)
        third_offset_velocity[n + 1] = rhs.third_offset_velocity[n] / (n + 1)
        physical_time[n + 1] = rhs.physical_time[n] / (n + 1)
    return solution


def construct_interval_regularized_binary_taylor_solution(
    initial_state: RegularizedBinaryCollisionChartState,
    *,
    order: int,
) -> IntervalRegularizedBinaryTaylorSolution:
    """Construct outward-rounded interval Taylor coefficients for a binary chart."""

    return construct_interval_regularized_binary_taylor_solution_from_intervals(
        IntervalRegularizedBinaryCollisionChartState(
            masses=np.asarray(initial_state.masses, dtype=float),
            pair=initial_state.pair,
            z=_interval_array_from_points(initial_state.z),
            z_velocity=_interval_array_from_points(initial_state.z_velocity),
            pair_energy=FloatInterval.point(initial_state.pair_energy),
            binary_center=_interval_array_from_points(initial_state.binary_center),
            binary_center_velocity=_interval_array_from_points(initial_state.binary_center_velocity),
            third_offset=_interval_array_from_points(initial_state.third_offset),
            third_offset_velocity=_interval_array_from_points(initial_state.third_offset_velocity),
            branch_certificate=None,
        ),
        order=order,
    )


def construct_interval_regularized_binary_taylor_solution_from_intervals(
    initial_state: IntervalRegularizedBinaryCollisionChartState,
    *,
    order: int,
) -> IntervalRegularizedBinaryTaylorSolution:
    """Construct interval Taylor coefficients from interval binary chart data."""

    if order < 1:
        raise ValueError("order must be at least 1")
    z = _interval_zeros((order + 1, 2))
    z_velocity = _interval_zeros((order + 1, 2))
    pair_energy = _interval_zeros((order + 1,))
    binary_center = _interval_zeros((order + 1, 2))
    binary_center_velocity = _interval_zeros((order + 1, 2))
    third_offset = _interval_zeros((order + 1, 2))
    third_offset_velocity = _interval_zeros((order + 1, 2))
    physical_time = _interval_zeros((order + 1,))
    z[0] = np.asarray(initial_state.z, dtype=object)
    z_velocity[0] = np.asarray(initial_state.z_velocity, dtype=object)
    pair_energy[0] = initial_state.pair_energy
    binary_center[0] = np.asarray(initial_state.binary_center, dtype=object)
    binary_center_velocity[0] = np.asarray(initial_state.binary_center_velocity, dtype=object)
    third_offset[0] = np.asarray(initial_state.third_offset, dtype=object)
    third_offset_velocity[0] = np.asarray(initial_state.third_offset_velocity, dtype=object)

    solution = IntervalRegularizedBinaryTaylorSolution(
        masses=np.asarray(initial_state.masses, dtype=float),
        pair=initial_state.pair,
        z=z,
        z_velocity=z_velocity,
        pair_energy=pair_energy,
        binary_center=binary_center,
        binary_center_velocity=binary_center_velocity,
        third_offset=third_offset,
        third_offset_velocity=third_offset_velocity,
        physical_time=physical_time,
    )
    for n in range(order):
        rhs = regularized_rhs_interval_coefficients(solution, n)
        scale = 1.0 / (n + 1)
        for axis in range(2):
            z[n + 1, axis] = _as_interval(rhs.z[n, axis]).scale(scale)
            z_velocity[n + 1, axis] = _as_interval(rhs.z_velocity[n, axis]).scale(scale)
            binary_center[n + 1, axis] = _as_interval(rhs.binary_center[n, axis]).scale(scale)
            binary_center_velocity[n + 1, axis] = _as_interval(rhs.binary_center_velocity[n, axis]).scale(scale)
            third_offset[n + 1, axis] = _as_interval(rhs.third_offset[n, axis]).scale(scale)
            third_offset_velocity[n + 1, axis] = _as_interval(rhs.third_offset_velocity[n, axis]).scale(scale)
        pair_energy[n + 1] = _as_interval(rhs.pair_energy[n]).scale(scale)
        physical_time[n + 1] = _as_interval(rhs.physical_time[n]).scale(scale)
    return solution


def pair_energy_constraint_coefficients(
    solution: RegularizedBinaryTaylorSolution,
    max_degree: int,
) -> Array:
    first, second = solution.pair
    pair_mass = solution.masses[first] + solution.masses[second]
    speed_square = np.zeros(max_degree + 1, dtype=float)
    for axis in range(2):
        speed_square += scalar_series_product(
            solution.z_velocity[: max_degree + 1, axis],
            solution.z_velocity[: max_degree + 1, axis],
            max_degree,
        )
    rho = rho_coefficients(solution.z, max_degree)
    energy_term = scalar_series_product(rho, solution.pair_energy[: max_degree + 1], max_degree)
    constraint = 2.0 * speed_square - energy_term
    constraint[0] -= pair_mass
    return constraint


def pack_regularized_state(state: RegularizedBinaryCollisionChartState, physical_time: float = 0.0) -> Array:
    return np.concatenate(
        [
            state.z,
            state.z_velocity,
            np.array([state.pair_energy]),
            state.binary_center,
            state.binary_center_velocity,
            state.third_offset,
            state.third_offset_velocity,
            np.array([physical_time]),
        ]
    )


def unpack_regularized_state(
    vector: Array,
    masses: Array,
    pair: tuple[int, int],
) -> tuple[RegularizedBinaryCollisionChartState, float]:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (14,):
        raise ValueError("regularized binary state vector must have length 14")
    state = RegularizedBinaryCollisionChartState(
        masses=np.asarray(masses, dtype=float),
        pair=pair,
        z=vector[0:2],
        z_velocity=vector[2:4],
        pair_energy=float(vector[4]),
        binary_center=vector[5:7],
        binary_center_velocity=vector[7:9],
        third_offset=vector[9:11],
        third_offset_velocity=vector[11:13],
    )
    return state, float(vector[13])


def integrate_regularized_binary_reference(
    initial_state: RegularizedBinaryCollisionChartState,
    s_final: float,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> Array:
    """Numerically integrate the regularized binary chart in `s` time."""

    vector0 = pack_regularized_state(initial_state)

    def rhs(_s: float, vector: Array) -> Array:
        state, _physical_time = unpack_regularized_state(vector, initial_state.masses, initial_state.pair)
        derivative = regularized_binary_collision_chart_rhs(state)
        return np.concatenate(
            [
                derivative.z,
                derivative.z_velocity,
                np.array([derivative.pair_energy]),
                derivative.binary_center,
                derivative.binary_center_velocity,
                derivative.third_offset,
                derivative.third_offset_velocity,
                np.array([derivative.physical_time]),
            ]
        )

    solution = solve_ivp(rhs, (0.0, s_final), vector0, method="DOP853", rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1]
