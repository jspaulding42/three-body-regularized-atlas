"""Taylor series for the spatial KS regularized binary chart."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .ks_binary_chart import (
    KSBranchCertificate,
    IntervalKSStateChart,
    SpatialKSBinaryChartDerivative,
    SpatialKSBinaryChartState,
    ks_interval_state_atlas,
    ks_interval_state_chart,
    physical_to_ks_on_branch,
    regularized_ks_binary_chart_rhs,
    spatial_to_ks_binary_chart,
)
from .global_invariants import (
    AngularMomentumConservationCertificate,
    CenterOfMassMotionCertificate,
    EnergyConservationCertificate,
    LinearMomentumConservationCertificate,
)
from .intervals import (
    FloatInterval,
    interval_array_as_tuples,
    interval_array_contains_point,
    interval_array_series_eval,
    interval_contains_zero,
    interval_polyder,
    interval_polynomial_eval,
    interval_sign,
    interval_series_add,
    interval_series_power,
    interval_series_product,
    zero_interval,
)
from .series import scalar_series_power, scalar_series_product
from .series import IntervalTaylorSolution


Array = np.ndarray


@dataclass(frozen=True)
class SpatialKSBinaryTaylorSolution:
    """Power-series coefficients for a spatial KS binary chart."""

    masses: Array
    pair: tuple[int, int]
    u: Array
    u_velocity: Array
    pair_energy: Array
    binary_center: Array
    binary_center_velocity: Array
    third_offset: Array
    third_offset_velocity: Array
    physical_time: Array

    @property
    def order(self) -> int:
        return int(self.u.shape[0] - 1)

    def state_at(self, s_value: float) -> SpatialKSBinaryChartState:
        return SpatialKSBinaryChartState(
            masses=self.masses,
            pair=self.pair,
            u=_evaluate(self.u, s_value),
            u_velocity=_evaluate(self.u_velocity, s_value),
            pair_energy=float(_evaluate(self.pair_energy[:, None], s_value)[0]),
            binary_center=_evaluate(self.binary_center, s_value),
            binary_center_velocity=_evaluate(self.binary_center_velocity, s_value),
            third_offset=_evaluate(self.third_offset, s_value),
            third_offset_velocity=_evaluate(self.third_offset_velocity, s_value),
        )

    def physical_time_at(self, s_value: float) -> float:
        return float(_evaluate(self.physical_time[:, None], s_value)[0])

    def vector_at(self, s_value: float) -> Array:
        return pack_spatial_ks_binary_state(self.state_at(s_value), self.physical_time_at(s_value))


@dataclass(frozen=True)
class IntervalSpatialKSBinaryChartState:
    """Interval initial data for a spatial KS separated-binary chart."""

    masses: Array
    pair: tuple[int, int]
    u: Array
    u_velocity: Array
    pair_energy: FloatInterval
    binary_center: Array
    binary_center_velocity: Array
    third_offset: Array
    third_offset_velocity: Array
    branch_certificate: KSBranchCertificate | None = None

    @property
    def certified(self) -> bool:
        return bool(self.branch_certificate is None or self.branch_certificate.certified)

    def contains_point_state(self, state: SpatialKSBinaryChartState) -> bool:
        return bool(
            tuple(self.pair) == tuple(state.pair)
            and np.allclose(np.asarray(self.masses, dtype=float), np.asarray(state.masses, dtype=float))
            and _interval_array_contains(self.u, state.u)
            and _interval_array_contains(self.u_velocity, state.u_velocity)
            and self.pair_energy.lower <= state.pair_energy <= self.pair_energy.upper
            and _interval_array_contains(self.binary_center, state.binary_center)
            and _interval_array_contains(self.binary_center_velocity, state.binary_center_velocity)
            and _interval_array_contains(self.third_offset, state.third_offset)
            and _interval_array_contains(self.third_offset_velocity, state.third_offset_velocity)
        )

    def contains_physical_point(self, positions: Array, velocities: Array) -> bool:
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        if positions.shape != (3, 3) or velocities.shape != (3, 3):
            return False
        if self.branch_certificate is None:
            try:
                point_state = spatial_to_ks_binary_chart(
                    positions,
                    velocities,
                    self.masses,
                    pair=self.pair,
                )
            except ValueError:
                return False
            return self.contains_point_state(point_state)
        if not self.branch_certificate.certified:
            return False
        try:
            first, second = self.pair
            third = _third_index_for_pair(self.pair)
            masses = np.asarray(self.masses, dtype=float)
            pair_mass = float(masses[first] + masses[second])
            relative_position = positions[second] - positions[first]
            relative_velocity = velocities[second] - velocities[first]
            u, u_velocity, pair_energy = physical_to_ks_on_branch(
                relative_position,
                relative_velocity,
                pair_mass,
                branch=self.branch_certificate.branch,
            )
            binary_center = (
                masses[first] * positions[first] + masses[second] * positions[second]
            ) / pair_mass
            binary_center_velocity = (
                masses[first] * velocities[first] + masses[second] * velocities[second]
            ) / pair_mass
            point_state = SpatialKSBinaryChartState(
                masses=masses,
                pair=tuple(self.pair),
                u=u,
                u_velocity=u_velocity,
                pair_energy=pair_energy,
                binary_center=binary_center,
                binary_center_velocity=binary_center_velocity,
                third_offset=positions[third] - binary_center,
                third_offset_velocity=velocities[third] - binary_center_velocity,
            )
        except ValueError:
            return False
        return self.contains_point_state(point_state)


@dataclass(frozen=True)
class IntervalSpatialKSBinaryTaylorSolution:
    """Outward-rounded interval Taylor coefficients for a spatial KS chart."""

    masses: Array
    pair: tuple[int, int]
    u: Array
    u_velocity: Array
    pair_energy: Array
    binary_center: Array
    binary_center_velocity: Array
    third_offset: Array
    third_offset_velocity: Array
    physical_time: Array

    @property
    def order(self) -> int:
        return int(self.u.shape[0] - 1)

    def contains_point_solution(self, solution: SpatialKSBinaryTaylorSolution) -> bool:
        checks = [
            (self.u, solution.u),
            (self.u_velocity, solution.u_velocity),
            (self.pair_energy, solution.pair_energy),
            (self.binary_center, solution.binary_center),
            (self.binary_center_velocity, solution.binary_center_velocity),
            (self.third_offset, solution.third_offset),
            (self.third_offset_velocity, solution.third_offset_velocity),
            (self.physical_time, solution.physical_time),
        ]
        return all(_interval_array_contains(interval, point) for interval, point in checks)

    def u_at(self, s_value: float) -> Array:
        return interval_array_series_eval(self.u, FloatInterval.point(s_value))

    def u_velocity_at(self, s_value: float) -> Array:
        return interval_array_series_eval(self.u_velocity, FloatInterval.point(s_value))

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
                self.u_at(s_value),
                self.u_velocity_at(s_value),
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


@dataclass(frozen=True)
class SpatialKSBinaryEquationResidualCertificate:
    """Coefficient residual certificate for the spatial KS Taylor equations."""

    coefficient_count: int
    residual_coefficients: tuple[FloatInterval, ...]
    pair: tuple[int, int]

    @property
    def certified(self) -> bool:
        return bool(
            self.coefficient_count > 0
            and self.residual_coefficients
            and all(interval_contains_zero(coefficient) for coefficient in self.residual_coefficients)
        )

    @property
    def max_residual_radius(self) -> float:
        return float(
            max(
                max(abs(coefficient.lower), abs(coefficient.upper))
                for coefficient in self.residual_coefficients
            )
        )


@dataclass(frozen=True)
class SpatialKSBinaryConstraintCertificate:
    """Coefficient certificate for an algebraic KS chart constraint."""

    constraint_id: str
    coefficient_count: int
    coefficients: tuple[FloatInterval, ...]
    pair: tuple[int, int]
    coefficient_source: str = "spatial_ks_binary_interval_series"

    @property
    def certified(self) -> bool:
        return bool(
            self.coefficient_count >= 0
            and self.coefficients
            and all(interval_contains_zero(coefficient) for coefficient in self.coefficients)
        )

    @property
    def max_radius(self) -> float:
        return float(
            max(
                max(abs(coefficient.lower), abs(coefficient.upper))
                for coefficient in self.coefficients
            )
        )


@dataclass(frozen=True)
class SpatialKSBinaryPhysicalProjectionCertificate:
    """Certified ordinary physical state box projected from a KS endpoint."""

    masses: Array
    pair: tuple[int, int]
    s_value: float
    physical_time: FloatInterval
    rho: FloatInterval
    state_interval: tuple[tuple[float, float], ...]
    projection_domain: str
    missing_obligations: tuple[str, ...] = ()
    coefficient_source: str = "spatial_ks_binary_interval_endpoint"

    @property
    def certified(self) -> bool:
        return bool(
            self.projection_domain == "rho_positive_interval"
            and self.rho.lower > 0.0
            and len(self.state_interval) == 18
            and not self.missing_obligations
            and all(
                np.isfinite(lower) and np.isfinite(upper) and lower <= upper
                for lower, upper in self.state_interval
            )
        )

    def contains_physical_point(self, positions: Array, velocities: Array) -> bool:
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        if positions.shape != (3, 3) or velocities.shape != (3, 3):
            return False
        if not self.certified:
            return False
        point = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
        return all(
            lower <= value <= upper
            for (lower, upper), value in zip(self.state_interval, point, strict=True)
        )


@dataclass(frozen=True)
class SpatialKSRhoExitEventCertificate:
    """Interval certificate for the earliest increasing ``rho=exit_rho`` event."""

    exit_rho: float
    root: float | None
    search_interval: tuple[float, float]
    root_interval: tuple[float, float] | None
    value_intervals: tuple[tuple[float, float], tuple[float, float]] | None
    derivative_interval: tuple[float, float] | None
    pre_event_interval: tuple[float, float] | None
    coefficient_intervals: tuple[tuple[float, float], ...]
    pre_event_subintervals: tuple[tuple[float, float], ...] | None = None
    coefficient_source: str = "spatial_ks_binary_interval_taylor"

    @property
    def interval_is_isolated(self) -> bool:
        if self.value_intervals is None or self.derivative_interval is None:
            return False
        return bool(
            interval_sign(FloatInterval(*self.value_intervals[0])) < 0
            and interval_sign(FloatInterval(*self.value_intervals[1])) > 0
            and interval_sign(FloatInterval(*self.derivative_interval)) > 0
        )

    @property
    def excludes_earlier_roots(self) -> bool:
        if self.pre_event_subintervals is not None:
            return bool(
                self.pre_event_subintervals
                and all(
                    interval_sign(FloatInterval(*value_interval)) < 0
                    for value_interval in self.pre_event_subintervals
                )
            )
        if self.pre_event_interval is None:
            return False
        return interval_sign(FloatInterval(*self.pre_event_interval)) < 0

    @property
    def certified(self) -> bool:
        return bool(
            self.exit_rho > 0.0
            and self.root is not None
            and self.root_interval is not None
            and self.search_interval[0] <= self.root <= self.search_interval[1]
            and self.root_interval[0] <= self.root <= self.root_interval[1]
            and self.interval_is_isolated
            and self.excludes_earlier_roots
        )


@dataclass(frozen=True)
class SpatialKSEntryEventCertificate:
    """Interval certificate for the earliest ordinary-to-KS entry event."""

    pair: tuple[int, int]
    enter_distance: float
    root: float | None
    search_interval: tuple[float, float]
    root_interval: tuple[float, float] | None
    value_intervals: tuple[tuple[float, float], tuple[float, float]] | None
    derivative_interval: tuple[float, float] | None
    pre_event_interval: tuple[float, float] | None
    coefficient_intervals: tuple[tuple[float, float], ...]
    pre_event_subintervals: tuple[tuple[float, float], ...] | None = None
    coefficient_source: str = "spatial_ordinary_interval_taylor"

    @property
    def interval_is_isolated(self) -> bool:
        if self.value_intervals is None or self.derivative_interval is None:
            return False
        return bool(
            interval_sign(FloatInterval(*self.value_intervals[0])) > 0
            and interval_sign(FloatInterval(*self.value_intervals[1])) < 0
            and interval_sign(FloatInterval(*self.derivative_interval)) < 0
        )

    @property
    def excludes_earlier_roots(self) -> bool:
        if self.pre_event_subintervals is not None:
            return bool(
                self.pre_event_subintervals
                and all(
                    interval_sign(FloatInterval(*value_interval)) > 0
                    for value_interval in self.pre_event_subintervals
                )
            )
        if self.pre_event_interval is None:
            return False
        return interval_sign(FloatInterval(*self.pre_event_interval)) > 0

    @property
    def certified(self) -> bool:
        return bool(
            self.enter_distance > 0.0
            and self.root is not None
            and self.root_interval is not None
            and self.search_interval[0] <= self.root <= self.search_interval[1]
            and self.root_interval[0] <= self.root <= self.root_interval[1]
            and self.interval_is_isolated
            and self.excludes_earlier_roots
        )


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


def _interval_square(value: object) -> FloatInterval:
    interval = _as_interval(value)
    if interval.lower <= 0.0 <= interval.upper:
        radius = max(abs(interval.lower), abs(interval.upper))
        return FloatInterval(0.0, float(np.nextafter(radius * radius, np.inf)))
    return interval * interval


def _interval_array_contains(interval: Array, point: Array) -> bool:
    interval = np.asarray(interval, dtype=object)
    point = np.asarray(point, dtype=float)
    if interval.shape != point.shape:
        return False
    for index in np.ndindex(interval.shape):
        value = _as_interval(interval[index])
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


def rho_coefficients(u: Array, max_degree: int) -> Array:
    """Coefficients of ``|u|^2``."""

    rho = np.zeros(max_degree + 1, dtype=float)
    for axis in range(4):
        rho += scalar_series_product(
            u[: max_degree + 1, axis],
            u[: max_degree + 1, axis],
            max_degree,
        )
    return rho


def rho_interval_coefficients(u: Array, max_degree: int) -> tuple[FloatInterval, ...]:
    """Interval coefficients of ``|u|^2``."""

    rho = tuple(zero_interval() for _ in range(max_degree + 1))
    for axis in range(4):
        axis_series = tuple(_as_interval(u[n, axis]) for n in range(max_degree + 1))
        rho = interval_series_add(rho, interval_series_product(axis_series, axis_series, max_degree))
    return rho


def ks_project_coefficients(u: Array, max_degree: int) -> Array:
    """Coefficients of the KS projection ``K(u)``."""

    a = u[: max_degree + 1, 0]
    b = u[: max_degree + 1, 1]
    c = u[: max_degree + 1, 2]
    d = u[: max_degree + 1, 3]
    aa = scalar_series_product(a, a, max_degree)
    bb = scalar_series_product(b, b, max_degree)
    cc = scalar_series_product(c, c, max_degree)
    dd = scalar_series_product(d, d, max_degree)
    ab = scalar_series_product(a, b, max_degree)
    cd = scalar_series_product(c, d, max_degree)
    ac = scalar_series_product(a, c, max_degree)
    bd = scalar_series_product(b, d, max_degree)
    return np.column_stack(
        [
            aa - bb - cc + dd,
            2.0 * (ab - cd),
            2.0 * (ac + bd),
        ]
    )


def ks_project_interval_coefficients(u: Array, max_degree: int) -> Array:
    """Interval coefficients of the KS projection ``K(u)``."""

    a = tuple(_as_interval(u[n, 0]) for n in range(max_degree + 1))
    b = tuple(_as_interval(u[n, 1]) for n in range(max_degree + 1))
    c = tuple(_as_interval(u[n, 2]) for n in range(max_degree + 1))
    d = tuple(_as_interval(u[n, 3]) for n in range(max_degree + 1))
    aa = interval_series_product(a, a, max_degree)
    bb = interval_series_product(b, b, max_degree)
    cc = interval_series_product(c, c, max_degree)
    dd = interval_series_product(d, d, max_degree)
    ab = interval_series_product(a, b, max_degree)
    cd = interval_series_product(c, d, max_degree)
    ac = interval_series_product(a, c, max_degree)
    bd = interval_series_product(b, d, max_degree)
    out = _interval_zeros((max_degree + 1, 3))
    for n in range(max_degree + 1):
        out[n, 0] = aa[n] - bb[n] - cc[n] + dd[n]
        out[n, 1] = (ab[n] - cd[n]).scale(2.0)
        out[n, 2] = (ac[n] + bd[n]).scale(2.0)
    return out


def _ks_project_interval_value(u: Array) -> Array:
    """Interval value of the KS projection ``K(u)``."""

    a, b, c, d = (_as_interval(u[axis]) for axis in range(4))
    return np.array(
        [
            _interval_square(a) - _interval_square(b) - _interval_square(c) + _interval_square(d),
            (a * b - c * d).scale(2.0),
            (a * c + b * d).scale(2.0),
        ],
        dtype=object,
    )


def ks_jacobian_times_coefficients(u: Array, vector: Array, max_degree: int) -> Array:
    """Coefficients of ``DK(u) vector``."""

    a = u[: max_degree + 1, 0]
    b = u[: max_degree + 1, 1]
    c = u[: max_degree + 1, 2]
    d = u[: max_degree + 1, 3]
    va = vector[: max_degree + 1, 0]
    vb = vector[: max_degree + 1, 1]
    vc = vector[: max_degree + 1, 2]
    vd = vector[: max_degree + 1, 3]
    return np.column_stack(
        [
            2.0 * (
                scalar_series_product(a, va, max_degree)
                - scalar_series_product(b, vb, max_degree)
                - scalar_series_product(c, vc, max_degree)
                + scalar_series_product(d, vd, max_degree)
            ),
            2.0 * (
                scalar_series_product(b, va, max_degree)
                + scalar_series_product(a, vb, max_degree)
                - scalar_series_product(d, vc, max_degree)
                - scalar_series_product(c, vd, max_degree)
            ),
            2.0 * (
                scalar_series_product(c, va, max_degree)
                + scalar_series_product(d, vb, max_degree)
                + scalar_series_product(a, vc, max_degree)
                + scalar_series_product(b, vd, max_degree)
            ),
        ]
    )


def ks_jacobian_times_interval_coefficients(u: Array, vector: Array, max_degree: int) -> Array:
    """Interval coefficients of ``DK(u) vector``."""

    a = tuple(_as_interval(u[n, 0]) for n in range(max_degree + 1))
    b = tuple(_as_interval(u[n, 1]) for n in range(max_degree + 1))
    c = tuple(_as_interval(u[n, 2]) for n in range(max_degree + 1))
    d = tuple(_as_interval(u[n, 3]) for n in range(max_degree + 1))
    va = tuple(_as_interval(vector[n, 0]) for n in range(max_degree + 1))
    vb = tuple(_as_interval(vector[n, 1]) for n in range(max_degree + 1))
    vc = tuple(_as_interval(vector[n, 2]) for n in range(max_degree + 1))
    vd = tuple(_as_interval(vector[n, 3]) for n in range(max_degree + 1))
    ava = interval_series_product(a, va, max_degree)
    bvb = interval_series_product(b, vb, max_degree)
    cvc = interval_series_product(c, vc, max_degree)
    dvd = interval_series_product(d, vd, max_degree)
    bva = interval_series_product(b, va, max_degree)
    avb = interval_series_product(a, vb, max_degree)
    dvc = interval_series_product(d, vc, max_degree)
    cvd = interval_series_product(c, vd, max_degree)
    cva = interval_series_product(c, va, max_degree)
    dvb = interval_series_product(d, vb, max_degree)
    avc = interval_series_product(a, vc, max_degree)
    bvd = interval_series_product(b, vd, max_degree)
    out = _interval_zeros((max_degree + 1, 3))
    for n in range(max_degree + 1):
        out[n, 0] = (ava[n] - bvb[n] - cvc[n] + dvd[n]).scale(2.0)
        out[n, 1] = (bva[n] + avb[n] - dvc[n] - cvd[n]).scale(2.0)
        out[n, 2] = (cva[n] + dvb[n] + avc[n] + bvd[n]).scale(2.0)
    return out


def _ks_jacobian_times_interval_value(u: Array, vector: Array) -> Array:
    """Interval value of ``DK(u) vector``."""

    a, b, c, d = (_as_interval(u[axis]) for axis in range(4))
    va, vb, vc, vd = (_as_interval(vector[axis]) for axis in range(4))
    return np.array(
        [
            (a * va - b * vb - c * vc + d * vd).scale(2.0),
            (b * va + a * vb - d * vc - c * vd).scale(2.0),
            (c * va + d * vb + a * vc + b * vd).scale(2.0),
        ],
        dtype=object,
    )


def ks_jacobian_transpose_times_coefficients(u: Array, vector: Array, max_degree: int) -> Array:
    """Coefficients of ``DK(u)^T vector``."""

    a = u[: max_degree + 1, 0]
    b = u[: max_degree + 1, 1]
    c = u[: max_degree + 1, 2]
    d = u[: max_degree + 1, 3]
    vx = vector[: max_degree + 1, 0]
    vy = vector[: max_degree + 1, 1]
    vz = vector[: max_degree + 1, 2]
    return np.column_stack(
        [
            2.0 * (
                scalar_series_product(a, vx, max_degree)
                + scalar_series_product(b, vy, max_degree)
                + scalar_series_product(c, vz, max_degree)
            ),
            2.0 * (
                -scalar_series_product(b, vx, max_degree)
                + scalar_series_product(a, vy, max_degree)
                + scalar_series_product(d, vz, max_degree)
            ),
            2.0 * (
                -scalar_series_product(c, vx, max_degree)
                - scalar_series_product(d, vy, max_degree)
                + scalar_series_product(a, vz, max_degree)
            ),
            2.0 * (
                scalar_series_product(d, vx, max_degree)
                - scalar_series_product(c, vy, max_degree)
                + scalar_series_product(b, vz, max_degree)
            ),
        ]
    )


def ks_jacobian_transpose_times_interval_coefficients(u: Array, vector: Array, max_degree: int) -> Array:
    """Interval coefficients of ``DK(u)^T vector``."""

    a = tuple(_as_interval(u[n, 0]) for n in range(max_degree + 1))
    b = tuple(_as_interval(u[n, 1]) for n in range(max_degree + 1))
    c = tuple(_as_interval(u[n, 2]) for n in range(max_degree + 1))
    d = tuple(_as_interval(u[n, 3]) for n in range(max_degree + 1))
    vx = tuple(_as_interval(vector[n, 0]) for n in range(max_degree + 1))
    vy = tuple(_as_interval(vector[n, 1]) for n in range(max_degree + 1))
    vz = tuple(_as_interval(vector[n, 2]) for n in range(max_degree + 1))
    avx = interval_series_product(a, vx, max_degree)
    bvy = interval_series_product(b, vy, max_degree)
    cvz = interval_series_product(c, vz, max_degree)
    bvx = interval_series_product(b, vx, max_degree)
    avy = interval_series_product(a, vy, max_degree)
    dvz = interval_series_product(d, vz, max_degree)
    cvx = interval_series_product(c, vx, max_degree)
    dvy = interval_series_product(d, vy, max_degree)
    avz = interval_series_product(a, vz, max_degree)
    dvx = interval_series_product(d, vx, max_degree)
    cvy = interval_series_product(c, vy, max_degree)
    bvz = interval_series_product(b, vz, max_degree)
    out = _interval_zeros((max_degree + 1, 4))
    for n in range(max_degree + 1):
        out[n, 0] = (avx[n] + bvy[n] + cvz[n]).scale(2.0)
        out[n, 1] = (avy[n] + dvz[n] - bvx[n]).scale(2.0)
        out[n, 2] = (avz[n] - cvx[n] - dvy[n]).scale(2.0)
        out[n, 3] = (dvx[n] - cvy[n] + bvz[n]).scale(2.0)
    return out


def _inverse_square_field_coefficients(vector: Array, max_degree: int) -> Array:
    distance_squared = np.zeros(max_degree + 1, dtype=float)
    for axis in range(3):
        distance_squared += scalar_series_product(
            vector[: max_degree + 1, axis],
            vector[: max_degree + 1, axis],
            max_degree,
        )
    inverse_cube = scalar_series_power(distance_squared, -1.5, max_degree)
    return _vector_product(inverse_cube, vector, max_degree)


def _inverse_square_field_interval_coefficients(vector: Array, max_degree: int) -> Array:
    distance_squared = tuple(zero_interval() for _ in range(max_degree + 1))
    for axis in range(3):
        axis_series = tuple(_as_interval(vector[n, axis]) for n in range(max_degree + 1))
        distance_squared = interval_series_add(
            distance_squared,
            interval_series_product(axis_series, axis_series, max_degree),
        )
    inverse_cube = interval_series_power(distance_squared, -1.5, max_degree)
    return _vector_product_interval(inverse_cube, vector, max_degree)


def _inverse_norm_interval_coefficients(vector: Array, max_degree: int) -> tuple[FloatInterval, ...]:
    distance_squared = tuple(zero_interval() for _ in range(max_degree + 1))
    for axis in range(3):
        axis_series = tuple(_as_interval(vector[n, axis]) for n in range(max_degree + 1))
        distance_squared = interval_series_add(
            distance_squared,
            interval_series_product(axis_series, axis_series, max_degree),
        )
    return interval_series_power(distance_squared, -0.5, max_degree)


def analytic_field_coefficients(
    solution: SpatialKSBinaryTaylorSolution,
    max_degree: int,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return ``(Rdd, ydd, p, rho, q_prime)`` coefficient arrays."""

    first, second = solution.pair
    third = ({0, 1, 2} - set(solution.pair)).pop()
    masses = solution.masses
    pair_mass = masses[first] + masses[second]
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    relative_position = ks_project_coefficients(solution.u, max_degree)
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
    rho = rho_coefficients(solution.u, max_degree)
    q_prime = ks_jacobian_times_coefficients(solution.u, solution.u_velocity, max_degree)
    return binary_center_acceleration, third_offset_acceleration, relative_perturbation, rho, q_prime


def _scaled_vector_sum_interval(
    first: Array,
    first_scale: float,
    second: Array,
    second_scale: float,
    max_degree: int,
) -> Array:
    out = _interval_zeros(first[: max_degree + 1].shape)
    for n in range(max_degree + 1):
        for axis in range(first.shape[1]):
            out[n, axis] = _as_interval(first[n, axis]).scale(first_scale) + _as_interval(
                second[n, axis]
            ).scale(second_scale)
    return out


def _vector_difference_interval(left: Array, right: Array, max_degree: int) -> Array:
    out = _interval_zeros(left[: max_degree + 1].shape)
    for n in range(max_degree + 1):
        for axis in range(left.shape[1]):
            out[n, axis] = _as_interval(left[n, axis]) - _as_interval(right[n, axis])
    return out


def analytic_field_interval_coefficients(
    solution: IntervalSpatialKSBinaryTaylorSolution,
    max_degree: int,
) -> tuple[Array, Array, Array, tuple[FloatInterval, ...], Array]:
    """Interval enclosures for ``(Rdd, ydd, p, rho, q_prime)`` coefficients."""

    first, second = solution.pair
    third = ({0, 1, 2} - set(solution.pair)).pop()
    masses = solution.masses
    pair_mass = masses[first] + masses[second]
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    relative_position = ks_project_interval_coefficients(solution.u, max_degree)
    from_first = _scaled_vector_sum_interval(
        solution.third_offset,
        1.0,
        relative_position,
        alpha,
        max_degree,
    )
    from_second = _scaled_vector_sum_interval(
        solution.third_offset,
        1.0,
        relative_position,
        -beta,
        max_degree,
    )
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
    rho = rho_interval_coefficients(solution.u, max_degree)
    q_prime = ks_jacobian_times_interval_coefficients(solution.u, solution.u_velocity, max_degree)
    return binary_center_acceleration, third_offset_acceleration, relative_perturbation, rho, q_prime


def regularized_rhs_coefficients(
    solution: SpatialKSBinaryTaylorSolution,
    max_degree: int,
) -> SpatialKSBinaryChartDerivative:
    """Compute coefficient arrays for the spatial KS binary RHS."""

    center_acc, third_acc, relative_perturbation, rho, q_prime = analytic_field_coefficients(
        solution,
        max_degree,
    )
    pair_energy_u = _vector_product(solution.pair_energy[: max_degree + 1], solution.u, max_degree)
    perturbation = ks_jacobian_transpose_times_coefficients(
        solution.u,
        relative_perturbation,
        max_degree,
    )
    u_acceleration = 0.5 * pair_energy_u + 0.25 * _vector_product(rho, perturbation, max_degree)
    return SpatialKSBinaryChartDerivative(
        u=solution.u_velocity[: max_degree + 1].copy(),
        u_velocity=u_acceleration,
        pair_energy=_vector_dot(q_prime, relative_perturbation, max_degree),
        binary_center=_vector_product(rho, solution.binary_center_velocity, max_degree),
        binary_center_velocity=_vector_product(rho, center_acc, max_degree),
        third_offset=_vector_product(rho, solution.third_offset_velocity, max_degree),
        third_offset_velocity=_vector_product(rho, third_acc, max_degree),
        physical_time=rho,
    )


def regularized_rhs_interval_coefficients(
    solution: IntervalSpatialKSBinaryTaylorSolution,
    max_degree: int,
) -> SpatialKSBinaryChartDerivative:
    """Compute interval coefficient enclosures for the spatial KS binary RHS."""

    center_acc, third_acc, relative_perturbation, rho, q_prime = analytic_field_interval_coefficients(
        solution,
        max_degree,
    )
    pair_energy = tuple(_as_interval(solution.pair_energy[n]) for n in range(max_degree + 1))
    pair_energy_u = _vector_product_interval(pair_energy, solution.u, max_degree)
    perturbation = ks_jacobian_transpose_times_interval_coefficients(
        solution.u,
        relative_perturbation,
        max_degree,
    )
    u_acceleration = _scaled_vector_sum_interval(
        pair_energy_u,
        0.5,
        _vector_product_interval(rho, perturbation, max_degree),
        0.25,
        max_degree,
    )
    return SpatialKSBinaryChartDerivative(
        u=solution.u_velocity[: max_degree + 1],
        u_velocity=u_acceleration,
        pair_energy=_vector_dot_interval(q_prime, relative_perturbation, max_degree),
        binary_center=_vector_product_interval(rho, solution.binary_center_velocity, max_degree),
        binary_center_velocity=_vector_product_interval(rho, center_acc, max_degree),
        third_offset=_vector_product_interval(rho, solution.third_offset_velocity, max_degree),
        third_offset_velocity=_vector_product_interval(rho, third_acc, max_degree),
        physical_time=rho,
    )


def construct_spatial_ks_binary_taylor_solution(
    initial_state: SpatialKSBinaryChartState,
    *,
    order: int,
) -> SpatialKSBinaryTaylorSolution:
    """Construct a local Taylor chart for the spatial KS binary RHS."""

    if order < 1:
        raise ValueError("order must be at least 1")
    u = np.zeros((order + 1, 4), dtype=float)
    u_velocity = np.zeros_like(u)
    pair_energy = np.zeros(order + 1, dtype=float)
    binary_center = np.zeros((order + 1, 3), dtype=float)
    binary_center_velocity = np.zeros_like(binary_center)
    third_offset = np.zeros_like(binary_center)
    third_offset_velocity = np.zeros_like(binary_center)
    physical_time = np.zeros(order + 1, dtype=float)
    u[0] = initial_state.u
    u_velocity[0] = initial_state.u_velocity
    pair_energy[0] = initial_state.pair_energy
    binary_center[0] = initial_state.binary_center
    binary_center_velocity[0] = initial_state.binary_center_velocity
    third_offset[0] = initial_state.third_offset
    third_offset_velocity[0] = initial_state.third_offset_velocity

    solution = SpatialKSBinaryTaylorSolution(
        masses=np.asarray(initial_state.masses, dtype=float),
        pair=initial_state.pair,
        u=u,
        u_velocity=u_velocity,
        pair_energy=pair_energy,
        binary_center=binary_center,
        binary_center_velocity=binary_center_velocity,
        third_offset=third_offset,
        third_offset_velocity=third_offset_velocity,
        physical_time=physical_time,
    )
    for n in range(order):
        rhs = regularized_rhs_coefficients(solution, n)
        scale = 1.0 / (n + 1)
        u[n + 1] = rhs.u[n] * scale
        u_velocity[n + 1] = rhs.u_velocity[n] * scale
        pair_energy[n + 1] = rhs.pair_energy[n] * scale
        binary_center[n + 1] = rhs.binary_center[n] * scale
        binary_center_velocity[n + 1] = rhs.binary_center_velocity[n] * scale
        third_offset[n + 1] = rhs.third_offset[n] * scale
        third_offset_velocity[n + 1] = rhs.third_offset_velocity[n] * scale
        physical_time[n + 1] = rhs.physical_time[n] * scale
    return solution


def interval_spatial_ks_binary_chart_state_from_point(
    initial_state: SpatialKSBinaryChartState,
) -> IntervalSpatialKSBinaryChartState:
    """Embed a point spatial KS chart state as outward-rounded intervals."""

    return IntervalSpatialKSBinaryChartState(
        masses=np.asarray(initial_state.masses, dtype=float),
        pair=initial_state.pair,
        u=_interval_array_from_points(initial_state.u),
        u_velocity=_interval_array_from_points(initial_state.u_velocity),
        pair_energy=FloatInterval.point(initial_state.pair_energy),
        binary_center=_interval_array_from_points(initial_state.binary_center),
        binary_center_velocity=_interval_array_from_points(initial_state.binary_center_velocity),
        third_offset=_interval_array_from_points(initial_state.third_offset),
        third_offset_velocity=_interval_array_from_points(initial_state.third_offset_velocity),
    )


def construct_interval_spatial_ks_binary_taylor_solution(
    initial_state: SpatialKSBinaryChartState,
    *,
    order: int,
) -> IntervalSpatialKSBinaryTaylorSolution:
    """Construct interval Taylor coefficients for a point spatial KS chart state."""

    return construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        interval_spatial_ks_binary_chart_state_from_point(initial_state),
        order=order,
    )


def construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
    initial_state: IntervalSpatialKSBinaryChartState,
    *,
    order: int,
) -> IntervalSpatialKSBinaryTaylorSolution:
    """Construct interval Taylor coefficients from interval spatial KS chart data."""

    if order < 1:
        raise ValueError("order must be at least 1")
    u = _interval_zeros((order + 1, 4))
    u_velocity = _interval_zeros((order + 1, 4))
    pair_energy = _interval_zeros((order + 1,))
    binary_center = _interval_zeros((order + 1, 3))
    binary_center_velocity = _interval_zeros((order + 1, 3))
    third_offset = _interval_zeros((order + 1, 3))
    third_offset_velocity = _interval_zeros((order + 1, 3))
    physical_time = _interval_zeros((order + 1,))
    u[0] = np.asarray(initial_state.u, dtype=object)
    u_velocity[0] = np.asarray(initial_state.u_velocity, dtype=object)
    pair_energy[0] = initial_state.pair_energy
    binary_center[0] = np.asarray(initial_state.binary_center, dtype=object)
    binary_center_velocity[0] = np.asarray(initial_state.binary_center_velocity, dtype=object)
    third_offset[0] = np.asarray(initial_state.third_offset, dtype=object)
    third_offset_velocity[0] = np.asarray(initial_state.third_offset_velocity, dtype=object)

    solution = IntervalSpatialKSBinaryTaylorSolution(
        masses=np.asarray(initial_state.masses, dtype=float),
        pair=initial_state.pair,
        u=u,
        u_velocity=u_velocity,
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
        for axis in range(4):
            u[n + 1, axis] = _as_interval(rhs.u[n, axis]).scale(scale)
            u_velocity[n + 1, axis] = _as_interval(rhs.u_velocity[n, axis]).scale(scale)
        for axis in range(3):
            binary_center[n + 1, axis] = _as_interval(rhs.binary_center[n, axis]).scale(scale)
            binary_center_velocity[n + 1, axis] = _as_interval(
                rhs.binary_center_velocity[n, axis]
            ).scale(scale)
            third_offset[n + 1, axis] = _as_interval(rhs.third_offset[n, axis]).scale(scale)
            third_offset_velocity[n + 1, axis] = _as_interval(
                rhs.third_offset_velocity[n, axis]
            ).scale(scale)
        pair_energy[n + 1] = _as_interval(rhs.pair_energy[n]).scale(scale)
        physical_time[n + 1] = _as_interval(rhs.physical_time[n]).scale(scale)
    return solution


def certify_spatial_ks_binary_rho_exit_event(
    solution: IntervalSpatialKSBinaryTaylorSolution,
    *,
    exit_rho: float,
    s_upper: float,
    coefficient_count: int | None = None,
) -> SpatialKSRhoExitEventCertificate:
    """Certify the earliest increasing local KS exit event ``rho=exit_rho``."""

    exit_rho = float(exit_rho)
    s_upper = float(s_upper)
    if exit_rho <= 0.0:
        raise ValueError("exit_rho must be positive")
    if s_upper <= 0.0:
        raise ValueError("s_upper must be positive")
    if coefficient_count is None:
        coefficient_count = solution.order
    coefficient_count = int(coefficient_count)
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > solution.order:
        raise ValueError("coefficient_count cannot exceed chart order")

    coefficients = list(rho_interval_coefficients(solution.u, coefficient_count))
    coefficients[0] = coefficients[0] - FloatInterval.point(exit_rho)
    coefficient_intervals = tuple(coefficient.as_tuple() for coefficient in coefficients)
    point_coefficients = _midpoint_coefficients(tuple(coefficients))
    root = _first_increasing_root(point_coefficients, s_upper)
    if root is None:
        return SpatialKSRhoExitEventCertificate(
            exit_rho=exit_rho,
            root=None,
            search_interval=(0.0, s_upper),
            root_interval=None,
            value_intervals=None,
            derivative_interval=None,
            pre_event_interval=None,
            coefficient_intervals=coefficient_intervals,
        )
    return _certify_spatial_ks_rho_exit_root(
        tuple(coefficients),
        exit_rho=exit_rho,
        root=root,
        s_upper=s_upper,
    )


def certify_spatial_ordinary_ks_entry_event(
    solution: IntervalTaylorSolution,
    *,
    pair: tuple[int, int],
    enter_distance: float,
    time_upper: float,
    coefficient_count: int | None = None,
) -> SpatialKSEntryEventCertificate:
    """Certify the earliest decreasing ordinary event ``|q_j-q_i|=enter_distance``."""

    enter_distance = float(enter_distance)
    time_upper = float(time_upper)
    pair = tuple(pair)
    _third_index_for_pair(pair)
    if enter_distance <= 0.0:
        raise ValueError("enter_distance must be positive")
    if time_upper <= 0.0:
        raise ValueError("time_upper must be positive")
    if coefficient_count is None:
        coefficient_count = solution.order
    coefficient_count = int(coefficient_count)
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > solution.order:
        raise ValueError("coefficient_count cannot exceed chart order")
    if solution.position.shape[1:] != (3, 3):
        raise ValueError("spatial ordinary entry expects a three-body 3D Taylor solution")

    coefficients = list(
        _ordinary_pair_distance_squared_interval_coefficients(
            solution.position,
            pair,
            coefficient_count,
        )
    )
    threshold = FloatInterval.point(enter_distance) * FloatInterval.point(enter_distance)
    coefficients[0] = coefficients[0] - threshold
    coefficient_intervals = tuple(coefficient.as_tuple() for coefficient in coefficients)
    point_coefficients = _midpoint_coefficients(tuple(coefficients))
    root = _first_directional_root(point_coefficients, time_upper, direction="decreasing")
    if root is None:
        return SpatialKSEntryEventCertificate(
            pair=pair,
            enter_distance=enter_distance,
            root=None,
            search_interval=(0.0, time_upper),
            root_interval=None,
            value_intervals=None,
            derivative_interval=None,
            pre_event_interval=None,
            coefficient_intervals=coefficient_intervals,
        )
    return _certify_spatial_ordinary_ks_entry_root(
        tuple(coefficients),
        pair=pair,
        enter_distance=enter_distance,
        root=root,
        time_upper=time_upper,
    )


def certify_spatial_ks_competing_binary_entry_event(
    solution: IntervalSpatialKSBinaryTaylorSolution,
    *,
    pair: tuple[int, int],
    enter_distance: float,
    s_upper: float,
    coefficient_count: int | None = None,
) -> SpatialKSEntryEventCertificate:
    """Certify a competing binary entry event inside a selected KS chart."""

    enter_distance = float(enter_distance)
    s_upper = float(s_upper)
    pair = tuple(pair)
    if enter_distance <= 0.0:
        raise ValueError("enter_distance must be positive")
    if s_upper <= 0.0:
        raise ValueError("s_upper must be positive")
    if coefficient_count is None:
        coefficient_count = solution.order
    coefficient_count = int(coefficient_count)
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > solution.order:
        raise ValueError("coefficient_count cannot exceed chart order")

    coefficients = list(
        _spatial_ks_competing_pair_distance_squared_interval_coefficients(
            solution,
            pair,
            coefficient_count,
        )
    )
    threshold = FloatInterval.point(enter_distance) * FloatInterval.point(enter_distance)
    coefficients[0] = coefficients[0] - threshold
    coefficient_intervals = tuple(coefficient.as_tuple() for coefficient in coefficients)
    point_coefficients = _midpoint_coefficients(tuple(coefficients))
    root = _first_directional_root(point_coefficients, s_upper, direction="decreasing")
    if root is None:
        return SpatialKSEntryEventCertificate(
            pair=pair,
            enter_distance=enter_distance,
            root=None,
            search_interval=(0.0, s_upper),
            root_interval=None,
            value_intervals=None,
            derivative_interval=None,
            pre_event_interval=None,
            coefficient_intervals=coefficient_intervals,
            coefficient_source="spatial_ks_binary_interval_taylor",
        )
    return _certify_spatial_ordinary_ks_entry_root(
        tuple(coefficients),
        pair=pair,
        enter_distance=enter_distance,
        root=root,
        time_upper=s_upper,
        coefficient_source="spatial_ks_binary_interval_taylor",
    )


def spatial_ordinary_entry_event_to_ks_chart_state(
    solution: IntervalTaylorSolution,
    certificate: SpatialKSEntryEventCertificate,
    *,
    branch: str,
) -> IntervalSpatialKSBinaryChartState:
    """Lift a certified ordinary entry event interval into one KS branch chart."""

    if not certificate.certified or certificate.root_interval is None:
        raise ValueError("entry certificate must be certified before lifting to KS")
    time = FloatInterval(*certificate.root_interval)
    positions = interval_array_series_eval(solution.position, time)
    velocities = interval_array_series_eval(solution.velocity, time)
    state_interval = interval_array_as_tuples(
        np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
    )
    return spatial_interval_to_ks_binary_chart_state(
        state_interval,
        solution.masses,
        pair=certificate.pair,
        branch=branch,
    )


def spatial_ordinary_entry_event_to_ks_chart_state_atlas(
    solution: IntervalTaylorSolution,
    certificate: SpatialKSEntryEventCertificate,
) -> tuple[IntervalSpatialKSBinaryChartState, ...]:
    """Lift a certified ordinary entry event interval into all certified KS branches."""

    if not certificate.certified or certificate.root_interval is None:
        raise ValueError("entry certificate must be certified before lifting to KS")
    time = FloatInterval(*certificate.root_interval)
    positions = interval_array_series_eval(solution.position, time)
    velocities = interval_array_series_eval(solution.velocity, time)
    state_interval = interval_array_as_tuples(
        np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
    )
    return spatial_interval_to_ks_binary_chart_state_atlas(
        state_interval,
        solution.masses,
        pair=certificate.pair,
    )


def spatial_ks_competing_entry_event_to_ks_chart_state(
    solution: IntervalSpatialKSBinaryTaylorSolution,
    certificate: SpatialKSEntryEventCertificate,
    *,
    branch: str,
) -> IntervalSpatialKSBinaryChartState:
    """Lift a certified competing-pair KS event into one next KS branch."""

    if not certificate.certified or certificate.root_interval is None:
        raise ValueError("competing entry certificate must be certified before lifting to KS")
    if certificate.coefficient_source != "spatial_ks_binary_interval_taylor":
        raise ValueError("entry certificate must come from a spatial KS binary chart")
    projection = _project_spatial_ks_binary_taylor_interval_to_physical(
        solution,
        FloatInterval(*certificate.root_interval),
    )
    if not projection.certified:
        raise ValueError("competing KS entry projection did not certify rho-positive state")
    return spatial_interval_to_ks_binary_chart_state(
        projection.state_interval,
        solution.masses,
        pair=certificate.pair,
        branch=branch,
    )


def spatial_ks_competing_entry_event_to_ks_chart_state_atlas(
    solution: IntervalSpatialKSBinaryTaylorSolution,
    certificate: SpatialKSEntryEventCertificate,
) -> tuple[IntervalSpatialKSBinaryChartState, ...]:
    """Lift a certified competing-pair KS event into all next KS branches."""

    if not certificate.certified or certificate.root_interval is None:
        raise ValueError("competing entry certificate must be certified before lifting to KS")
    if certificate.coefficient_source != "spatial_ks_binary_interval_taylor":
        raise ValueError("entry certificate must come from a spatial KS binary chart")
    projection = _project_spatial_ks_binary_taylor_interval_to_physical(
        solution,
        FloatInterval(*certificate.root_interval),
    )
    if not projection.certified:
        raise ValueError("competing KS entry projection did not certify rho-positive state")
    return spatial_interval_to_ks_binary_chart_state_atlas(
        projection.state_interval,
        solution.masses,
        pair=certificate.pair,
    )


def project_spatial_ks_binary_interval_chart_state_to_physical(
    state: IntervalSpatialKSBinaryChartState,
    *,
    s_value: float = 0.0,
    physical_time: FloatInterval | None = None,
) -> SpatialKSBinaryPhysicalProjectionCertificate:
    """Project an interval KS chart state to an ordinary physical state box.

    The physical velocity formula divides by ``rho=|u|^2``.  The certificate is
    therefore proof-certified only on endpoints whose interval ``rho`` is
    strictly positive.
    """

    if physical_time is None:
        physical_time = FloatInterval.point(0.0)
    rho = _interval_rho_value(state.u)
    if not state.certified:
        return _spatial_ks_projection_failure(
            masses=state.masses,
            pair=state.pair,
            s_value=s_value,
            physical_time=physical_time,
            rho=rho,
            missing_obligations=("certified_ks_branch",),
        )
    return _project_spatial_ks_binary_interval_values_to_physical(
        masses=state.masses,
        pair=state.pair,
        u=state.u,
        u_velocity=state.u_velocity,
        binary_center=state.binary_center,
        binary_center_velocity=state.binary_center_velocity,
        third_offset=state.third_offset,
        third_offset_velocity=state.third_offset_velocity,
        s_value=s_value,
        physical_time=physical_time,
    )


def project_spatial_ks_binary_taylor_endpoint_to_physical(
    solution: IntervalSpatialKSBinaryTaylorSolution,
    s_value: float,
) -> SpatialKSBinaryPhysicalProjectionCertificate:
    """Evaluate an interval KS Taylor chart and project a rho-positive endpoint."""

    s_value = float(s_value)
    return _project_spatial_ks_binary_interval_values_to_physical(
        masses=solution.masses,
        pair=solution.pair,
        u=solution.u_at(s_value),
        u_velocity=solution.u_velocity_at(s_value),
        binary_center=solution.binary_center_at(s_value),
        binary_center_velocity=solution.binary_center_velocity_at(s_value),
        third_offset=solution.third_offset_at(s_value),
        third_offset_velocity=solution.third_offset_velocity_at(s_value),
        s_value=s_value,
        physical_time=solution.physical_time_at(s_value),
    )


def _project_spatial_ks_binary_taylor_interval_to_physical(
    solution: IntervalSpatialKSBinaryTaylorSolution,
    s_interval: FloatInterval,
) -> SpatialKSBinaryPhysicalProjectionCertificate:
    s_interval = _as_interval(s_interval)
    return _project_spatial_ks_binary_interval_values_to_physical(
        masses=solution.masses,
        pair=solution.pair,
        u=interval_array_series_eval(solution.u, s_interval),
        u_velocity=interval_array_series_eval(solution.u_velocity, s_interval),
        binary_center=interval_array_series_eval(solution.binary_center, s_interval),
        binary_center_velocity=interval_array_series_eval(
            solution.binary_center_velocity,
            s_interval,
        ),
        third_offset=interval_array_series_eval(solution.third_offset, s_interval),
        third_offset_velocity=interval_array_series_eval(
            solution.third_offset_velocity,
            s_interval,
        ),
        s_value=0.5 * (s_interval.lower + s_interval.upper),
        physical_time=interval_array_series_eval(
            solution.physical_time[:, None],
            s_interval,
        )[0],
    )


def _midpoint_coefficients(coefficients: tuple[FloatInterval, ...]) -> Array:
    return np.asarray(
        [0.5 * (coefficient.lower + coefficient.upper) for coefficient in coefficients],
        dtype=float,
    )


def _first_increasing_root(coefficients: Array, s_upper: float) -> float | None:
    return _first_directional_root(coefficients, s_upper, direction="increasing")


def _first_directional_root(
    coefficients: Array,
    upper: float,
    *,
    direction: str,
) -> float | None:
    if direction not in {"increasing", "decreasing"}:
        raise ValueError("direction must be increasing or decreasing")
    roots = np.polynomial.polynomial.polyroots(coefficients)
    derivative = np.polynomial.polynomial.polyder(coefficients)
    candidates = []
    for root in roots:
        if abs(root.imag) > 1e-9 * max(1.0, abs(root.real)):
            continue
        value = float(root.real)
        if value <= 16.0 * np.finfo(float).eps or value > upper:
            continue
        slope = float(np.polynomial.polynomial.polyval(value, derivative)) if derivative.size else 0.0
        if (direction == "increasing" and slope > 0.0) or (
            direction == "decreasing" and slope < 0.0
        ):
            candidates.append(value)
    return min(candidates) if candidates else None


def _certify_spatial_ks_rho_exit_root(
    coefficients: tuple[FloatInterval, ...],
    *,
    exit_rho: float,
    root: float,
    s_upper: float,
) -> SpatialKSRhoExitEventCertificate:
    derivative = interval_polyder(coefficients)
    span = float(s_upper)
    radius = max(
        1e-10 * max(1.0, abs(root)),
        1e-10 * span,
        64.0 * np.finfo(float).eps,
    )
    for _attempt in range(80):
        left = max(0.0, float(root - radius))
        right = min(s_upper, float(root + radius))
        if left >= right:
            radius *= 2.0
            continue
        left_value = interval_polynomial_eval(coefficients, FloatInterval.point(left))
        right_value = interval_polynomial_eval(coefficients, FloatInterval.point(right))
        derivative_value = interval_polynomial_eval(derivative, FloatInterval(left, right))
        pre_event_value = interval_polynomial_eval(coefficients, FloatInterval(0.0, left))
        pre_event_subintervals = _signed_polynomial_range_evidence(
            coefficients,
            0.0,
            left,
            sign=-1,
        )
        certificate = SpatialKSRhoExitEventCertificate(
            exit_rho=exit_rho,
            root=float(root),
            search_interval=(0.0, s_upper),
            root_interval=(left, right),
            value_intervals=(left_value.as_tuple(), right_value.as_tuple()),
            derivative_interval=derivative_value.as_tuple(),
            pre_event_interval=pre_event_value.as_tuple(),
            coefficient_intervals=tuple(coefficient.as_tuple() for coefficient in coefficients),
            pre_event_subintervals=pre_event_subintervals,
        )
        if certificate.certified:
            return certificate
        radius *= 2.0
    return SpatialKSRhoExitEventCertificate(
        exit_rho=exit_rho,
        root=float(root),
        search_interval=(0.0, s_upper),
        root_interval=None,
        value_intervals=None,
        derivative_interval=None,
        pre_event_interval=None,
        coefficient_intervals=tuple(coefficient.as_tuple() for coefficient in coefficients),
    )


def _certify_spatial_ordinary_ks_entry_root(
    coefficients: tuple[FloatInterval, ...],
    *,
    pair: tuple[int, int],
    enter_distance: float,
    root: float,
    time_upper: float,
    coefficient_source: str = "spatial_ordinary_interval_taylor",
) -> SpatialKSEntryEventCertificate:
    derivative = interval_polyder(coefficients)
    span = float(time_upper)
    radius = max(
        1e-10 * max(1.0, abs(root)),
        1e-10 * span,
        64.0 * np.finfo(float).eps,
    )
    for _attempt in range(80):
        left = max(0.0, float(root - radius))
        right = min(time_upper, float(root + radius))
        if left >= right:
            radius *= 2.0
            continue
        left_value = interval_polynomial_eval(coefficients, FloatInterval.point(left))
        right_value = interval_polynomial_eval(coefficients, FloatInterval.point(right))
        derivative_value = interval_polynomial_eval(derivative, FloatInterval(left, right))
        pre_event_value = interval_polynomial_eval(coefficients, FloatInterval(0.0, left))
        pre_event_subintervals = _signed_polynomial_range_evidence(
            coefficients,
            0.0,
            left,
            sign=1,
        )
        certificate = SpatialKSEntryEventCertificate(
            pair=pair,
            enter_distance=enter_distance,
            root=float(root),
            search_interval=(0.0, time_upper),
            root_interval=(left, right),
            value_intervals=(left_value.as_tuple(), right_value.as_tuple()),
            derivative_interval=derivative_value.as_tuple(),
            pre_event_interval=pre_event_value.as_tuple(),
            coefficient_intervals=tuple(coefficient.as_tuple() for coefficient in coefficients),
            pre_event_subintervals=pre_event_subintervals,
            coefficient_source=coefficient_source,
        )
        if certificate.certified:
            return certificate
        radius *= 2.0
    return SpatialKSEntryEventCertificate(
        pair=pair,
        enter_distance=enter_distance,
        root=float(root),
        search_interval=(0.0, time_upper),
        root_interval=None,
        value_intervals=None,
        derivative_interval=None,
        pre_event_interval=None,
        coefficient_intervals=tuple(coefficient.as_tuple() for coefficient in coefficients),
        coefficient_source=coefficient_source,
    )


def _signed_polynomial_range_evidence(
    coefficients: tuple[FloatInterval, ...],
    lower: float,
    upper: float,
    *,
    sign: int,
    max_subintervals: int = 512,
) -> tuple[tuple[float, float], ...] | None:
    if sign == 0:
        raise ValueError("sign must be nonzero")
    if lower < 0.0 or upper < lower:
        raise ValueError("invalid signed range")
    if lower == upper:
        value = interval_polynomial_eval(coefficients, FloatInterval.point(lower))
        return (value.as_tuple(),) if interval_sign(value) == sign else None

    pieces = 1
    while pieces <= max_subintervals:
        values = []
        certified = True
        for index in range(pieces):
            piece_lower = lower + (upper - lower) * index / pieces
            piece_upper = lower + (upper - lower) * (index + 1) / pieces
            value = interval_polynomial_eval(coefficients, FloatInterval(piece_lower, piece_upper))
            if interval_sign(value) != sign:
                certified = False
                break
            values.append(value.as_tuple())
        if certified:
            return tuple(values)
        pieces *= 2
    return None


def spatial_interval_to_ks_binary_chart_state(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    *,
    pair: tuple[int, int] = (0, 1),
    branch: str,
) -> IntervalSpatialKSBinaryChartState:
    """Lift a spatial interval state box into one KS binary branch chart."""

    masses = np.asarray(masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    positions, velocities, relative_position, relative_velocity = _spatial_interval_components(
        state_interval,
        pair,
    )
    pair_mass = float(masses[pair[0]] + masses[pair[1]])
    ks_state = ks_interval_state_chart(
        relative_position,
        relative_velocity,
        pair_mass,
        branch=branch,
    )
    return _spatial_interval_ks_binary_chart_from_components(
        masses=masses,
        pair=pair,
        positions=positions,
        velocities=velocities,
        ks_state=ks_state,
    )


def spatial_interval_to_ks_binary_chart_state_atlas(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    *,
    pair: tuple[int, int] = (0, 1),
) -> tuple[IntervalSpatialKSBinaryChartState, ...]:
    """Return certified KS branch boxes covering a spatial interval lift."""

    masses = np.asarray(masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    positions, velocities, relative_position, relative_velocity = _spatial_interval_components(
        state_interval,
        pair,
    )
    pair_mass = float(masses[pair[0]] + masses[pair[1]])
    return tuple(
        _spatial_interval_ks_binary_chart_from_components(
            masses=masses,
            pair=pair,
            positions=positions,
            velocities=velocities,
            ks_state=ks_state,
        )
        for ks_state in ks_interval_state_atlas(relative_position, relative_velocity, pair_mass)
    )


def _interval_rho_value(u: Array) -> FloatInterval:
    rho = zero_interval()
    for axis in range(4):
        rho = rho + _interval_square(u[axis])
    return rho


def _ordinary_pair_distance_squared_interval_coefficients(
    position: Array,
    pair: tuple[int, int],
    max_degree: int,
) -> tuple[FloatInterval, ...]:
    first, second = pair
    out = tuple(zero_interval() for _degree in range(max_degree + 1))
    for axis in range(3):
        delta = tuple(
            _as_interval(position[degree, second, axis])
            - _as_interval(position[degree, first, axis])
            for degree in range(max_degree + 1)
        )
        out = interval_series_add(out, interval_series_product(delta, delta, max_degree))
    return out


def _spatial_ks_competing_pair_distance_squared_interval_coefficients(
    solution: IntervalSpatialKSBinaryTaylorSolution,
    pair: tuple[int, int],
    max_degree: int,
) -> tuple[FloatInterval, ...]:
    selected_first, selected_second = tuple(solution.pair)
    selected_third = _third_index_for_pair(tuple(solution.pair))
    pair = tuple(pair)
    _third_index_for_pair(pair)
    if set(pair) == set(solution.pair):
        raise ValueError("competing KS entry pair cannot be the selected KS pair")
    if selected_third not in pair or not (
        selected_first in pair or selected_second in pair
    ):
        raise ValueError("competing KS entry pair must contain the third body")
    if max_degree < 0:
        raise ValueError("max_degree cannot be negative")
    if max_degree > solution.order:
        raise ValueError("max_degree cannot exceed chart order")

    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[selected_first] + masses[selected_second])
    alpha = float(masses[selected_second] / pair_mass)
    beta = float(masses[selected_first] / pair_mass)
    relative_position = ks_project_interval_coefficients(solution.u, max_degree)
    if selected_first in pair:
        vector = _scaled_vector_sum_interval(
            solution.third_offset,
            1.0,
            relative_position,
            alpha,
            max_degree,
        )
    else:
        vector = _scaled_vector_sum_interval(
            solution.third_offset,
            1.0,
            relative_position,
            -beta,
            max_degree,
        )
    out = tuple(zero_interval() for _degree in range(max_degree + 1))
    for axis in range(3):
        axis_series = tuple(_as_interval(vector[n, axis]) for n in range(max_degree + 1))
        out = interval_series_add(out, interval_series_product(axis_series, axis_series, max_degree))
    return out


def _spatial_ks_projection_failure(
    *,
    masses: Array,
    pair: tuple[int, int],
    s_value: float,
    physical_time: FloatInterval,
    rho: FloatInterval,
    missing_obligations: tuple[str, ...],
) -> SpatialKSBinaryPhysicalProjectionCertificate:
    return SpatialKSBinaryPhysicalProjectionCertificate(
        masses=np.asarray(masses, dtype=float),
        pair=tuple(pair),
        s_value=float(s_value),
        physical_time=physical_time,
        rho=rho,
        state_interval=(),
        projection_domain="rho_not_positive_interval",
        missing_obligations=missing_obligations,
    )


def _project_spatial_ks_binary_interval_values_to_physical(
    *,
    masses: Array,
    pair: tuple[int, int],
    u: Array,
    u_velocity: Array,
    binary_center: Array,
    binary_center_velocity: Array,
    third_offset: Array,
    third_offset_velocity: Array,
    s_value: float,
    physical_time: FloatInterval,
) -> SpatialKSBinaryPhysicalProjectionCertificate:
    masses = np.asarray(masses, dtype=float)
    pair = tuple(pair)
    first, second = pair
    third = _third_index_for_pair(pair)
    pair_mass = float(masses[first] + masses[second])
    rho = _interval_rho_value(u)
    if rho.lower <= 0.0:
        return _spatial_ks_projection_failure(
            masses=masses,
            pair=pair,
            s_value=s_value,
            physical_time=physical_time,
            rho=rho,
            missing_obligations=("rho_positive_interval",),
        )

    relative_position = _ks_project_interval_value(u)
    relative_velocity_prime = _ks_jacobian_times_interval_value(u, u_velocity)
    relative_velocity = np.array(
        [relative_velocity_prime[axis] / rho for axis in range(3)],
        dtype=object,
    )
    positions = np.empty((3, 3), dtype=object)
    velocities = np.empty((3, 3), dtype=object)
    first_weight = float(masses[second] / pair_mass)
    second_weight = float(masses[first] / pair_mass)
    for axis in range(3):
        center = _as_interval(binary_center[axis])
        center_velocity = _as_interval(binary_center_velocity[axis])
        positions[first, axis] = center - relative_position[axis].scale(first_weight)
        positions[second, axis] = center + relative_position[axis].scale(second_weight)
        positions[third, axis] = center + _as_interval(third_offset[axis])
        velocities[first, axis] = center_velocity - relative_velocity[axis].scale(first_weight)
        velocities[second, axis] = center_velocity + relative_velocity[axis].scale(second_weight)
        velocities[third, axis] = center_velocity + _as_interval(third_offset_velocity[axis])

    state_interval = interval_array_as_tuples(
        np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
    )
    return SpatialKSBinaryPhysicalProjectionCertificate(
        masses=masses,
        pair=pair,
        s_value=float(s_value),
        physical_time=physical_time,
        rho=rho,
        state_interval=state_interval,
        projection_domain="rho_positive_interval",
    )


def _spatial_interval_components(
    state_interval: tuple[tuple[float, float], ...],
    pair: tuple[int, int],
) -> tuple[Array, Array, Array, Array]:
    if len(state_interval) != 18:
        raise ValueError("spatial interval state must have length 18")
    _third_index_for_pair(pair)
    values = np.array(
        [FloatInterval(float(lower), float(upper)) for lower, upper in state_interval],
        dtype=object,
    )
    positions = values[:9].reshape(3, 3)
    velocities = values[9:].reshape(3, 3)
    first, second = pair
    relative_position = np.array(
        [positions[second, axis] - positions[first, axis] for axis in range(3)],
        dtype=object,
    )
    relative_velocity = np.array(
        [velocities[second, axis] - velocities[first, axis] for axis in range(3)],
        dtype=object,
    )
    return positions, velocities, relative_position, relative_velocity


def _spatial_interval_ks_binary_chart_from_components(
    *,
    masses: Array,
    pair: tuple[int, int],
    positions: Array,
    velocities: Array,
    ks_state: IntervalKSStateChart,
) -> IntervalSpatialKSBinaryChartState:
    first, second = pair
    third = _third_index_for_pair(pair)
    pair_mass = float(masses[first] + masses[second])
    binary_center = _interval_weighted_pair_average(
        positions[first],
        positions[second],
        float(masses[first]),
        float(masses[second]),
        pair_mass,
    )
    binary_center_velocity = _interval_weighted_pair_average(
        velocities[first],
        velocities[second],
        float(masses[first]),
        float(masses[second]),
        pair_mass,
    )
    third_offset = np.array(
        [positions[third, axis] - binary_center[axis] for axis in range(3)],
        dtype=object,
    )
    third_offset_velocity = np.array(
        [velocities[third, axis] - binary_center_velocity[axis] for axis in range(3)],
        dtype=object,
    )
    return IntervalSpatialKSBinaryChartState(
        masses=np.asarray(masses, dtype=float),
        pair=tuple(pair),
        u=ks_state.u,
        u_velocity=ks_state.u_velocity,
        pair_energy=ks_state.energy,
        binary_center=binary_center,
        binary_center_velocity=binary_center_velocity,
        third_offset=third_offset,
        third_offset_velocity=third_offset_velocity,
        branch_certificate=ks_state.branch_certificate,
    )


def _interval_weighted_pair_average(
    first: Array,
    second: Array,
    first_mass: float,
    second_mass: float,
    pair_mass: float,
) -> Array:
    out = np.empty(3, dtype=object)
    for axis in range(3):
        out[axis] = (
            _as_interval(first[axis]).scale(first_mass)
            + _as_interval(second[axis]).scale(second_mass)
        ).scale(1.0 / pair_mass)
    return out


def certify_spatial_ks_binary_interval_taylor_equations(
    series: IntervalSpatialKSBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> SpatialKSBinaryEquationResidualCertificate:
    """Certify the interval KS recurrence coefficient-by-coefficient."""

    if coefficient_count is None:
        coefficient_count = series.order
    coefficient_count = int(coefficient_count)
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > series.order:
        raise ValueError("coefficient_count cannot exceed chart order")
    rhs = regularized_rhs_interval_coefficients(series, coefficient_count - 1)
    residuals: list[FloatInterval] = []
    for degree in range(coefficient_count):
        scale = degree + 1
        for axis in range(4):
            residuals.append(series.u[degree + 1, axis].scale(scale) - rhs.u[degree, axis])
            residuals.append(
                series.u_velocity[degree + 1, axis].scale(scale)
                - rhs.u_velocity[degree, axis]
            )
        for axis in range(3):
            residuals.append(
                series.binary_center[degree + 1, axis].scale(scale)
                - rhs.binary_center[degree, axis]
            )
            residuals.append(
                series.binary_center_velocity[degree + 1, axis].scale(scale)
                - rhs.binary_center_velocity[degree, axis]
            )
            residuals.append(
                series.third_offset[degree + 1, axis].scale(scale)
                - rhs.third_offset[degree, axis]
            )
            residuals.append(
                series.third_offset_velocity[degree + 1, axis].scale(scale)
                - rhs.third_offset_velocity[degree, axis]
            )
        residuals.append(series.pair_energy[degree + 1].scale(scale) - rhs.pair_energy[degree])
        residuals.append(series.physical_time[degree + 1].scale(scale) - rhs.physical_time[degree])
    return SpatialKSBinaryEquationResidualCertificate(
        coefficient_count=coefficient_count,
        residual_coefficients=tuple(residuals),
        pair=tuple(series.pair),
    )


def ks_horizontal_constraint_coefficients(
    solution: SpatialKSBinaryTaylorSolution,
    max_degree: int,
) -> Array:
    """Coefficients of the KS horizontal-gauge constraint."""

    a = solution.u[: max_degree + 1, 0]
    b = solution.u[: max_degree + 1, 1]
    c = solution.u[: max_degree + 1, 2]
    d = solution.u[: max_degree + 1, 3]
    da = solution.u_velocity[: max_degree + 1, 0]
    db = solution.u_velocity[: max_degree + 1, 1]
    dc = solution.u_velocity[: max_degree + 1, 2]
    dd = solution.u_velocity[: max_degree + 1, 3]
    return (
        -scalar_series_product(d, da, max_degree)
        + scalar_series_product(c, db, max_degree)
        - scalar_series_product(b, dc, max_degree)
        + scalar_series_product(a, dd, max_degree)
    )


def ks_horizontal_constraint_interval_coefficients(
    solution: IntervalSpatialKSBinaryTaylorSolution,
    max_degree: int,
) -> tuple[FloatInterval, ...]:
    """Interval coefficients of the KS horizontal-gauge constraint."""

    a = _series_axis(solution.u, 0, max_degree)
    b = _series_axis(solution.u, 1, max_degree)
    c = _series_axis(solution.u, 2, max_degree)
    d = _series_axis(solution.u, 3, max_degree)
    da = _series_axis(solution.u_velocity, 0, max_degree)
    db = _series_axis(solution.u_velocity, 1, max_degree)
    dc = _series_axis(solution.u_velocity, 2, max_degree)
    dd = _series_axis(solution.u_velocity, 3, max_degree)
    return interval_series_add(
        interval_series_add(
            _scale_interval_series(interval_series_product(d, da, max_degree), -1.0),
            interval_series_product(c, db, max_degree),
        ),
        interval_series_add(
            _scale_interval_series(interval_series_product(b, dc, max_degree), -1.0),
            interval_series_product(a, dd, max_degree),
        ),
    )


def ks_pair_energy_constraint_interval_coefficients(
    solution: IntervalSpatialKSBinaryTaylorSolution,
    max_degree: int,
) -> tuple[FloatInterval, ...]:
    """Interval coefficients of ``2|u'|^2 - M - |u|^2 h``."""

    first, second = solution.pair
    pair_mass = solution.masses[first] + solution.masses[second]
    speed_square = tuple(zero_interval() for _ in range(max_degree + 1))
    for axis in range(4):
        axis_series = _series_axis(solution.u_velocity, axis, max_degree)
        speed_square = interval_series_add(
            speed_square,
            interval_series_product(axis_series, axis_series, max_degree),
        )
    rho = rho_interval_coefficients(solution.u, max_degree)
    pair_energy = tuple(_as_interval(solution.pair_energy[n]) for n in range(max_degree + 1))
    energy_term = interval_series_product(rho, pair_energy, max_degree)
    constraint = [
        speed_square[degree].scale(2.0) - energy_term[degree]
        for degree in range(max_degree + 1)
    ]
    constraint[0] = constraint[0] - FloatInterval.point(pair_mass)
    return tuple(constraint)


def certify_spatial_ks_binary_horizontal_constraint(
    series: IntervalSpatialKSBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> SpatialKSBinaryConstraintCertificate:
    """Certify that the KS horizontal-gauge constraint vanishes."""

    coefficient_count = _spatial_ks_constraint_coefficient_count(series, coefficient_count)
    return SpatialKSBinaryConstraintCertificate(
        constraint_id="ks_horizontal_constraint",
        coefficient_count=coefficient_count,
        coefficients=ks_horizontal_constraint_interval_coefficients(series, coefficient_count),
        pair=tuple(series.pair),
    )


def certify_spatial_ks_binary_pair_energy_constraint(
    series: IntervalSpatialKSBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> SpatialKSBinaryConstraintCertificate:
    """Certify the finite KS pair-energy algebraic constraint."""

    coefficient_count = _spatial_ks_constraint_coefficient_count(series, coefficient_count)
    return SpatialKSBinaryConstraintCertificate(
        constraint_id="ks_pair_energy_constraint",
        coefficient_count=coefficient_count,
        coefficients=ks_pair_energy_constraint_interval_coefficients(series, coefficient_count),
        pair=tuple(series.pair),
    )


def certify_spatial_ks_binary_center_of_mass_motion(
    series: IntervalSpatialKSBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> CenterOfMassMotionCertificate:
    """Certify inertial center-of-mass motion from finite spatial KS variables."""

    coefficient_count = _spatial_ks_invariant_coefficient_count(series, coefficient_count)
    total_mass = float(np.sum(series.masses))
    third_mass = float(series.masses[_third_index_for_pair(series.pair)])
    position_moment, momentum = _spatial_ks_mass_moment_and_momentum(
        series,
        coefficient_count,
        total_mass=total_mass,
        third_mass=third_mass,
    )
    physical_time = tuple(_as_interval(series.physical_time[n]) for n in range(coefficient_count + 1))
    initial_offset = tuple(
        position_moment[0][axis] - momentum[0][axis] * physical_time[0]
        for axis in range(3)
    )
    residuals = []
    for degree in range(coefficient_count + 1):
        components = []
        for axis in range(3):
            component = position_moment[degree][axis] - momentum[0][axis] * physical_time[degree]
            if degree == 0:
                component = component - initial_offset[axis]
            components.append(component)
        residuals.append(tuple(components))
    return CenterOfMassMotionCertificate(
        coefficient_count=coefficient_count,
        residual_coefficients=tuple(residuals),
        coefficient_source="spatial_ks_binary_interval_series",
    )


def certify_spatial_ks_binary_linear_momentum_conservation(
    series: IntervalSpatialKSBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> LinearMomentumConservationCertificate:
    """Certify total linear momentum in finite spatial KS variables."""

    coefficient_count = _spatial_ks_invariant_coefficient_count(series, coefficient_count)
    total_mass = float(np.sum(series.masses))
    third_mass = float(series.masses[_third_index_for_pair(series.pair)])
    _position_moment, momentum = _spatial_ks_mass_moment_and_momentum(
        series,
        coefficient_count,
        total_mass=total_mass,
        third_mass=third_mass,
    )
    return LinearMomentumConservationCertificate(
        coefficient_count=coefficient_count,
        linear_momentum_coefficients=momentum,
        coefficient_source="spatial_ks_binary_interval_series",
    )


def certify_spatial_ks_binary_centered_angular_momentum_conservation(
    series: IntervalSpatialKSBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> AngularMomentumConservationCertificate:
    """Certify centered angular momentum using the finite KS pair form.

    The Jacobi form is ``mu_y y x y_dot + mu_pair q x q_dot``.  On the
    horizontal KS lift, ``q x q_dot`` is the bilinear finite expression used by
    ``_ks_relative_angular_momentum_series``, so no singular physical velocity
    is evaluated at binary collision.
    """

    coefficient_count = _spatial_ks_invariant_coefficient_count(series, coefficient_count)
    first, second = series.pair
    third = _third_index_for_pair(series.pair)
    masses = np.asarray(series.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    total_mass = float(np.sum(masses))
    third_mass = float(masses[third])
    reduced_pair_mass = float(masses[first] * masses[second] / pair_mass)
    reduced_third_mass = float(pair_mass * third_mass / total_mass)
    y_cross = _axial_to_bivector_components(
        _interval_cross_series(
            series.third_offset,
            series.third_offset_velocity,
            coefficient_count,
        )
    )
    relative_cross = _axial_to_bivector_components(
        _ks_relative_angular_momentum_series(series, coefficient_count)
    )
    coefficients = []
    for degree in range(coefficient_count + 1):
        components = []
        for axis in range(3):
            components.append(
                y_cross[degree][axis].scale(reduced_third_mass)
                + relative_cross[degree][axis].scale(reduced_pair_mass)
            )
        coefficients.append(tuple(components))
    return AngularMomentumConservationCertificate(
        coefficient_count=coefficient_count,
        angular_momentum_coefficients=tuple(coefficients),
        coefficient_source="spatial_ks_binary_interval_series",
    )


def certify_spatial_ks_binary_total_energy_conservation(
    series: IntervalSpatialKSBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> EnergyConservationCertificate:
    """Certify finite Newtonian energy through a spatial KS binary chart."""

    coefficient_count = _spatial_ks_invariant_coefficient_count(series, coefficient_count)
    first, second = series.pair
    third = _third_index_for_pair(series.pair)
    masses = np.asarray(series.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    total_mass = float(np.sum(masses))
    third_mass = float(masses[third])
    reduced_pair_mass = float(masses[first] * masses[second] / pair_mass)
    reduced_third_mass = float(pair_mass * third_mass / total_mass)

    center_velocity = _interval_vector_series_scaled_sum(
        series.binary_center_velocity,
        1.0,
        series.third_offset_velocity,
        third_mass / total_mass,
        coefficient_count,
    )
    center_kinetic = _interval_dot_series(center_velocity, center_velocity, coefficient_count)
    third_kinetic = _interval_dot_series(
        series.third_offset_velocity,
        series.third_offset_velocity,
        coefficient_count,
    )
    energy = _scale_interval_series(center_kinetic, 0.5 * total_mass)
    energy = interval_series_add(
        energy,
        _scale_interval_series(third_kinetic, 0.5 * reduced_third_mass),
    )
    pair_energy = tuple(_as_interval(series.pair_energy[n]) for n in range(coefficient_count + 1))
    energy = interval_series_add(
        energy,
        _scale_interval_series(pair_energy, reduced_pair_mass),
    )

    relative_position = ks_project_interval_coefficients(series.u, coefficient_count)
    alpha = float(masses[second] / pair_mass)
    beta = float(masses[first] / pair_mass)
    from_first = _scaled_vector_sum_interval(
        series.third_offset,
        1.0,
        relative_position,
        alpha,
        coefficient_count,
    )
    from_second = _scaled_vector_sum_interval(
        series.third_offset,
        1.0,
        relative_position,
        -beta,
        coefficient_count,
    )
    energy = interval_series_add(
        energy,
        _scale_interval_series(
            _inverse_norm_interval_coefficients(from_first, coefficient_count),
            -float(masses[first] * third_mass),
        ),
    )
    energy = interval_series_add(
        energy,
        _scale_interval_series(
            _inverse_norm_interval_coefficients(from_second, coefficient_count),
            -float(masses[second] * third_mass),
        ),
    )
    return EnergyConservationCertificate(
        coefficient_count=coefficient_count,
        energy_coefficients=energy,
        coefficient_source="spatial_ks_binary_interval_series",
    )


def _spatial_ks_invariant_coefficient_count(
    series: IntervalSpatialKSBinaryTaylorSolution,
    coefficient_count: int | None,
) -> int:
    if coefficient_count is None:
        coefficient_count = series.order
    coefficient_count = int(coefficient_count)
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > series.order:
        raise ValueError("coefficient_count cannot exceed chart order")
    return coefficient_count


def _spatial_ks_constraint_coefficient_count(
    series: IntervalSpatialKSBinaryTaylorSolution,
    coefficient_count: int | None,
) -> int:
    if coefficient_count is None:
        coefficient_count = series.order
    coefficient_count = int(coefficient_count)
    if coefficient_count < 0:
        raise ValueError("coefficient_count cannot be negative")
    if coefficient_count > series.order:
        raise ValueError("coefficient_count cannot exceed chart order")
    return coefficient_count


def _third_index_for_pair(pair: tuple[int, int]) -> int:
    remaining = {0, 1, 2} - set(pair)
    if len(remaining) != 1:
        raise ValueError("pair must contain two distinct body indices from {0, 1, 2}")
    return remaining.pop()


def _spatial_ks_mass_moment_and_momentum(
    series: IntervalSpatialKSBinaryTaylorSolution,
    max_degree: int,
    *,
    total_mass: float,
    third_mass: float,
) -> tuple[tuple[tuple[FloatInterval, ...], ...], tuple[tuple[FloatInterval, ...], ...]]:
    position_moment = []
    momentum = []
    for degree in range(max_degree + 1):
        position_components = []
        momentum_components = []
        for axis in range(3):
            position_components.append(
                _as_interval(series.binary_center[degree, axis]).scale(total_mass)
                + _as_interval(series.third_offset[degree, axis]).scale(third_mass)
            )
            momentum_components.append(
                _as_interval(series.binary_center_velocity[degree, axis]).scale(total_mass)
                + _as_interval(series.third_offset_velocity[degree, axis]).scale(third_mass)
            )
        position_moment.append(tuple(position_components))
        momentum.append(tuple(momentum_components))
    return tuple(position_moment), tuple(momentum)


def _series_axis(values: Array, axis: int, max_degree: int) -> tuple[FloatInterval, ...]:
    return tuple(_as_interval(values[degree, axis]) for degree in range(max_degree + 1))


def _interval_dot_series(left: Array, right: Array, max_degree: int) -> tuple[FloatInterval, ...]:
    out = tuple(zero_interval() for _ in range(max_degree + 1))
    for axis in range(left.shape[1]):
        out = interval_series_add(
            out,
            interval_series_product(
                _series_axis(left, axis, max_degree),
                _series_axis(right, axis, max_degree),
                max_degree,
            ),
        )
    return out


def _interval_cross_series(left: Array, right: Array, max_degree: int) -> tuple[tuple[FloatInterval, ...], ...]:
    lx = _series_axis(left, 0, max_degree)
    ly = _series_axis(left, 1, max_degree)
    lz = _series_axis(left, 2, max_degree)
    rx = _series_axis(right, 0, max_degree)
    ry = _series_axis(right, 1, max_degree)
    rz = _series_axis(right, 2, max_degree)
    x = interval_series_add(
        interval_series_product(ly, rz, max_degree),
        _scale_interval_series(interval_series_product(lz, ry, max_degree), -1.0),
    )
    y = interval_series_add(
        interval_series_product(lz, rx, max_degree),
        _scale_interval_series(interval_series_product(lx, rz, max_degree), -1.0),
    )
    z = interval_series_add(
        interval_series_product(lx, ry, max_degree),
        _scale_interval_series(interval_series_product(ly, rx, max_degree), -1.0),
    )
    return tuple((x[degree], y[degree], z[degree]) for degree in range(max_degree + 1))


def _axial_to_bivector_components(
    axial_coefficients: tuple[tuple[FloatInterval, ...], ...],
) -> tuple[tuple[FloatInterval, ...], ...]:
    """Convert axial ``(Lx,Ly,Lz)`` to global-invariant ``(xy,xz,yz)`` order."""

    return tuple(
        (coefficient[2], coefficient[1].scale(-1.0), coefficient[0])
        for coefficient in axial_coefficients
    )


def _ks_relative_angular_momentum_series(
    series: IntervalSpatialKSBinaryTaylorSolution,
    max_degree: int,
) -> tuple[tuple[FloatInterval, ...], ...]:
    """Return interval coefficients of ``K(u) x dK(u)/dt`` on horizontal lifts."""

    a = _series_axis(series.u, 0, max_degree)
    b = _series_axis(series.u, 1, max_degree)
    c = _series_axis(series.u, 2, max_degree)
    d = _series_axis(series.u, 3, max_degree)
    da = _series_axis(series.u_velocity, 0, max_degree)
    db = _series_axis(series.u_velocity, 1, max_degree)
    dc = _series_axis(series.u_velocity, 2, max_degree)
    dd = _series_axis(series.u_velocity, 3, max_degree)
    bdc = interval_series_product(b, dc, max_degree)
    cdb = interval_series_product(c, db, max_degree)
    cda = interval_series_product(c, da, max_degree)
    ddb = interval_series_product(d, db, max_degree)
    adc = interval_series_product(a, dc, max_degree)
    bdd = interval_series_product(b, dd, max_degree)
    bda = interval_series_product(b, da, max_degree)
    adb = interval_series_product(a, db, max_degree)
    ddc = interval_series_product(d, dc, max_degree)
    cdd = interval_series_product(c, dd, max_degree)
    coefficients = []
    for degree in range(max_degree + 1):
        x = (bdc[degree] - cdb[degree]).scale(4.0)
        y = (cda[degree] - ddb[degree] - adc[degree] + bdd[degree]).scale(2.0)
        z = (adb[degree] - bda[degree] - ddc[degree] + cdd[degree]).scale(2.0)
        coefficients.append((x, y, z))
    return tuple(coefficients)


def _scale_interval_series(
    coefficients: tuple[FloatInterval, ...],
    factor: float,
) -> tuple[FloatInterval, ...]:
    return tuple(coefficient.scale(float(factor)) for coefficient in coefficients)


def _interval_vector_series_scaled_sum(
    first: Array,
    first_scale: float,
    second: Array,
    second_scale: float,
    max_degree: int,
) -> Array:
    out = _interval_zeros((max_degree + 1, first.shape[1]))
    for degree in range(max_degree + 1):
        for axis in range(first.shape[1]):
            out[degree, axis] = _as_interval(first[degree, axis]).scale(first_scale) + _as_interval(
                second[degree, axis]
            ).scale(second_scale)
    return out


def ks_pair_energy_constraint_coefficients(
    solution: SpatialKSBinaryTaylorSolution,
    max_degree: int,
) -> Array:
    first, second = solution.pair
    pair_mass = solution.masses[first] + solution.masses[second]
    speed_square = np.zeros(max_degree + 1, dtype=float)
    for axis in range(4):
        speed_square += scalar_series_product(
            solution.u_velocity[: max_degree + 1, axis],
            solution.u_velocity[: max_degree + 1, axis],
            max_degree,
        )
    rho = rho_coefficients(solution.u, max_degree)
    energy_term = scalar_series_product(rho, solution.pair_energy[: max_degree + 1], max_degree)
    constraint = 2.0 * speed_square - energy_term
    constraint[0] -= pair_mass
    return constraint


def pack_spatial_ks_binary_state(
    state: SpatialKSBinaryChartState,
    physical_time: float = 0.0,
) -> Array:
    return np.concatenate(
        [
            state.u,
            state.u_velocity,
            np.array([state.pair_energy]),
            state.binary_center,
            state.binary_center_velocity,
            state.third_offset,
            state.third_offset_velocity,
            np.array([physical_time]),
        ]
    )


def unpack_spatial_ks_binary_state(
    vector: Array,
    masses: Array,
    pair: tuple[int, int],
) -> tuple[SpatialKSBinaryChartState, float]:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (22,):
        raise ValueError("spatial KS binary state vector must have length 22")
    state = SpatialKSBinaryChartState(
        masses=np.asarray(masses, dtype=float),
        pair=pair,
        u=vector[0:4],
        u_velocity=vector[4:8],
        pair_energy=float(vector[8]),
        binary_center=vector[9:12],
        binary_center_velocity=vector[12:15],
        third_offset=vector[15:18],
        third_offset_velocity=vector[18:21],
    )
    return state, float(vector[21])


def integrate_spatial_ks_binary_reference(
    initial_state: SpatialKSBinaryChartState,
    s_final: float,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-14,
) -> Array:
    """Numerically integrate the regularized spatial KS binary chart."""

    vector0 = pack_spatial_ks_binary_state(initial_state)

    def rhs(_s: float, vector: Array) -> Array:
        state, _physical_time = unpack_spatial_ks_binary_state(
            vector,
            initial_state.masses,
            initial_state.pair,
        )
        derivative = regularized_ks_binary_chart_rhs(state)
        return np.concatenate(
            [
                derivative.u,
                derivative.u_velocity,
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
