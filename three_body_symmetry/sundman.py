"""Sundman-time lifted series for the general three-body problem.

This module moves one step closer to Sundman's global construction. It does not
complete Sundman's proof, but it implements the central regularizing lift: use a
fictitious independent variable `s` whose physical-time rate is the product of
pairwise distances. Away from collision this is equivalent to Newtonian time;
near binary collision, physical time advances slowly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .global_invariants import (
    AngularMomentumConservationCertificate,
    CenterOfMassMotionCertificate,
    EnergyConservationCertificate,
    LinearMomentumConservationCertificate,
    TripleCollisionExclusionCertificate,
    certify_interval_center_of_mass_motion,
    certify_interval_centered_angular_momentum_conservation,
    certify_interval_linear_momentum_conservation,
    certify_interval_total_energy_conservation,
    certify_nonzero_angular_momentum_excludes_triple_collision,
)
from .intervals import (
    FloatInterval,
    interval_array_contains_point,
    interval_array_series_eval,
    interval_polynomial_eval,
    interval_series_add,
    interval_series_power,
    interval_series_product,
    zero_interval,
)
from .series import (
    acceleration_coefficients,
    acceleration_interval_coefficients,
    integrate_reference,
    scalar_series_power,
    scalar_series_product,
)


Array = np.ndarray


@dataclass(frozen=True)
class SundmanTaylorSolution:
    """Local Taylor chart in Sundman time `s`."""

    position: Array
    velocity: Array
    physical_time: Array
    masses: Array
    distance_power: float

    @property
    def order(self) -> int:
        return int(self.position.shape[0] - 1)

    @property
    def body_count(self) -> int:
        return int(self.position.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.position.shape[2])

    def positions_at_s(self, s_value: float) -> Array:
        return _evaluate(self.position, s_value)

    def velocities_at_s(self, s_value: float) -> Array:
        return _evaluate(self.velocity, s_value)

    def physical_time_at_s(self, s_value: float) -> float:
        return float(_evaluate(self.physical_time[:, None], s_value)[0])

    def state_at_s(self, s_value: float) -> Array:
        return np.concatenate([self.positions_at_s(s_value).reshape(-1), self.velocities_at_s(s_value).reshape(-1)])


@dataclass(frozen=True)
class IntervalSundmanTaylorSolution:
    """Outward-rounded interval coefficients for a Sundman-time Taylor chart."""

    position: Array
    velocity: Array
    physical_time: Array
    masses: Array
    distance_power: float

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

    @property
    def physical_time_lower(self) -> Array:
        return _interval_lower(self.physical_time)

    @property
    def physical_time_upper(self) -> Array:
        return _interval_upper(self.physical_time)

    def contains_point_solution(self, solution: SundmanTaylorSolution) -> bool:
        if (
            solution.position.shape != self.position.shape
            or solution.velocity.shape != self.velocity.shape
            or solution.physical_time.shape != self.physical_time.shape
        ):
            return False
        return bool(
            np.all(self.position_lower <= solution.position)
            and np.all(solution.position <= self.position_upper)
            and np.all(self.velocity_lower <= solution.velocity)
            and np.all(solution.velocity <= self.velocity_upper)
            and np.all(self.physical_time_lower <= solution.physical_time)
            and np.all(solution.physical_time <= self.physical_time_upper)
        )

    def positions_at_s(self, s_value: float) -> Array:
        return interval_array_series_eval(self.position, FloatInterval.point(s_value))

    def positions_over_s_interval(self, s_interval: FloatInterval) -> Array:
        return interval_array_series_eval(self.position, s_interval)

    def velocities_at_s(self, s_value: float) -> Array:
        return interval_array_series_eval(self.velocity, FloatInterval.point(s_value))

    def velocities_over_s_interval(self, s_interval: FloatInterval) -> Array:
        return interval_array_series_eval(self.velocity, s_interval)

    def physical_time_at_s(self, s_value: float) -> FloatInterval:
        return interval_polynomial_eval(self.physical_time, FloatInterval.point(s_value))

    def physical_time_over_s_interval(self, s_interval: FloatInterval) -> FloatInterval:
        return interval_polynomial_eval(self.physical_time, s_interval)

    def state_at_s(self, s_value: float) -> Array:
        return np.concatenate([self.positions_at_s(s_value).reshape(-1), self.velocities_at_s(s_value).reshape(-1)])

    def state_over_s_interval(self, s_interval: FloatInterval) -> Array:
        return np.concatenate(
            [
                self.positions_over_s_interval(s_interval).reshape(-1),
                self.velocities_over_s_interval(s_interval).reshape(-1),
            ]
        )

    def state_contains(self, state: Array, s_value: float) -> bool:
        return interval_array_contains_point(self.state_at_s(s_value), np.asarray(state, dtype=float))

    def state_over_interval_contains(self, state: Array, s_interval: FloatInterval) -> bool:
        return interval_array_contains_point(self.state_over_s_interval(s_interval), np.asarray(state, dtype=float))


@dataclass(frozen=True)
class SundmanStep:
    start_time: float
    s_step: float
    physical_step: float
    series: SundmanTaylorSolution
    min_pair_distance: float
    truncation_certificate: object | None = None

    @property
    def end_time(self) -> float:
        return self.start_time + self.physical_step


@dataclass(frozen=True)
class SundmanContinuedSolution:
    masses: Array
    times: Array
    states: Array
    steps: tuple[SundmanStep, ...]

    @property
    def final_state(self) -> Array:
        return self.states[-1]

    @property
    def local_tail_bound(self) -> float:
        """Sum guarded numerical tail estimates over Sundman steps."""

        total = 0.0
        for step in self.steps:
            if step.truncation_certificate is None:
                continue
            total += step.truncation_certificate.tail_bound
        return float(total)

    @property
    def max_step_tail_bound(self) -> float:
        bounds = [
            step.truncation_certificate.tail_bound
            for step in self.steps
            if step.truncation_certificate is not None
        ]
        return float(max(bounds)) if bounds else 0.0

    @property
    def certified_step_count(self) -> int:
        return sum(step.truncation_certificate is not None for step in self.steps)


@dataclass(frozen=True)
class IntervalSundmanStep:
    start_s: float
    s_step: float
    start_time_interval: FloatInterval
    physical_step_interval: FloatInterval
    factor_interval: FloatInterval
    series: IntervalSundmanTaylorSolution
    start_state_interval: Array
    end_state_interval: Array
    equation_residual_certificate: SundmanEquationResidualCertificate
    angular_momentum_certificate: AngularMomentumConservationCertificate
    energy_certificate: EnergyConservationCertificate
    linear_momentum_certificate: LinearMomentumConservationCertificate
    center_of_mass_certificate: CenterOfMassMotionCertificate
    truncation_certificate: object | None = None

    @property
    def end_s(self) -> float:
        return self.start_s + self.s_step

    @property
    def end_time_interval(self) -> FloatInterval:
        return self.start_time_interval + self.physical_step_interval

    @property
    def time_monotone_certified(self) -> bool:
        return self.factor_interval.lower > 0.0

    @property
    def angular_momentum_certified(self) -> bool:
        return self.angular_momentum_certificate.certified

    @property
    def energy_certified(self) -> bool:
        return self.energy_certificate.certified

    @property
    def linear_momentum_certified(self) -> bool:
        return self.linear_momentum_certificate.certified

    @property
    def center_of_mass_certified(self) -> bool:
        return self.center_of_mass_certificate.certified

    def start_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.start_state_interval, np.asarray(state, dtype=float))

    def end_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.end_state_interval, np.asarray(state, dtype=float))


@dataclass(frozen=True)
class IntervalSundmanContinuedSolution:
    masses: Array
    s_values: Array
    time_intervals: tuple[FloatInterval, ...]
    state_intervals: tuple[Array, ...]
    steps: tuple[IntervalSundmanStep, ...]
    triple_collision_exclusion_certificate: TripleCollisionExclusionCertificate | None = None

    @property
    def final_time_interval(self) -> FloatInterval:
        return self.time_intervals[-1]

    @property
    def final_state_interval(self) -> Array:
        return self.state_intervals[-1]

    def final_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.final_state_interval, np.asarray(state, dtype=float))

    @property
    def certified(self) -> bool:
        return bool(
            all(step.time_monotone_certified for step in self.steps)
            and all(step.equation_residual_certificate.certified for step in self.steps)
            and all(step.angular_momentum_certificate.certified for step in self.steps)
            and all(step.energy_certificate.certified for step in self.steps)
            and all(step.linear_momentum_certificate.certified for step in self.steps)
            and all(step.center_of_mass_certificate.certified for step in self.steps)
        )

    @property
    def local_tail_bound(self) -> float:
        total = 0.0
        for step in self.steps:
            if step.truncation_certificate is None:
                continue
            total += step.truncation_certificate.tail_bound
        return float(total)

    @property
    def max_step_tail_bound(self) -> float:
        bounds = [
            step.truncation_certificate.tail_bound
            for step in self.steps
            if step.truncation_certificate is not None
        ]
        return float(max(bounds)) if bounds else 0.0

    @property
    def certified_step_count(self) -> int:
        return sum(step.truncation_certificate is not None for step in self.steps)

    @property
    def equation_residual_certified_step_count(self) -> int:
        return sum(step.equation_residual_certificate.certified for step in self.steps)

    @property
    def angular_momentum_certified_step_count(self) -> int:
        return sum(step.angular_momentum_certificate.certified for step in self.steps)

    @property
    def energy_certified_step_count(self) -> int:
        return sum(step.energy_certificate.certified for step in self.steps)

    @property
    def linear_momentum_certified_step_count(self) -> int:
        return sum(step.linear_momentum_certificate.certified for step in self.steps)

    @property
    def center_of_mass_certified_step_count(self) -> int:
        return sum(step.center_of_mass_certificate.certified for step in self.steps)

    @property
    def tail_certified(self) -> bool:
        return bool(self.steps and all(step.truncation_certificate is not None for step in self.steps))

    @property
    def chain_certified(self) -> bool:
        return _interval_sundman_chain_certified(
            self.s_values,
            self.time_intervals,
            self.state_intervals,
            self.steps,
        )

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and self.tail_certified and self.chain_certified)

    @property
    def triple_collision_excluded(self) -> bool:
        return bool(
            self.triple_collision_exclusion_certificate is not None
            and self.triple_collision_exclusion_certificate.certified
        )

    @property
    def triple_collision_status(self) -> str:
        if self.triple_collision_exclusion_certificate is None:
            return "missing"
        return self.triple_collision_exclusion_certificate.status

    @property
    def triple_collision_exclusion_reason(self) -> str | None:
        if self.triple_collision_exclusion_certificate is None:
            return None
        return self.triple_collision_exclusion_certificate.reason

    @property
    def triple_collision_undecided(self) -> bool:
        return self.triple_collision_status == "undecided"


@dataclass(frozen=True)
class SundmanPhysicalTimeTargetCertificate:
    target_time: float
    s_interval: FloatInterval
    time_at_lower: FloatInterval
    time_at_upper: FloatInterval
    factor_interval: FloatInterval
    bisections: int

    @property
    def certified(self) -> bool:
        return bool(
            self.factor_interval.lower > 0.0
            and self.time_at_lower.upper <= self.target_time <= self.time_at_upper.lower
        )

    @property
    def width(self) -> float:
        return self.s_interval.upper - self.s_interval.lower


@dataclass(frozen=True)
class SundmanEquationResidualCertificate:
    coefficient_count: int
    position_residual: Array
    velocity_residual: Array
    physical_time_residual: tuple[FloatInterval, ...]
    factor_coefficient_source: str = "sundman_interval"
    acceleration_coefficient_source: str = "newtonian_interval"

    @property
    def certified(self) -> bool:
        return bool(
            _interval_array_contains_zero(self.position_residual)
            and _interval_array_contains_zero(self.velocity_residual)
            and all(_interval_contains_zero(value) for value in self.physical_time_residual)
        )

    @property
    def max_residual_radius(self) -> float:
        radii = []
        for values in (self.position_residual, self.velocity_residual):
            for index in np.ndindex(values.shape):
                value = _as_interval(values[index])
                radii.append(max(abs(value.lower), abs(value.upper)))
        radii.extend(max(abs(value.lower), abs(value.upper)) for value in self.physical_time_residual)
        return float(max(radii, default=0.0))


@dataclass(frozen=True)
class SundmanGlobalPhysicalTimeTargetCertificate:
    target_time: float
    chart_start_s: float
    local_s_interval: FloatInterval
    global_s_interval: FloatInterval
    start_time_interval: FloatInterval
    local_time_at_lower: FloatInterval
    local_time_at_upper: FloatInterval
    global_time_at_lower: FloatInterval
    global_time_at_upper: FloatInterval
    factor_interval: FloatInterval
    bisections: int

    @property
    def certified(self) -> bool:
        return bool(
            self.factor_interval.lower > 0.0
            and self.global_time_at_lower.upper <= self.target_time <= self.global_time_at_upper.lower
        )

    @property
    def width(self) -> float:
        return self.local_s_interval.upper - self.local_s_interval.lower


@dataclass(frozen=True)
class IntervalSundmanTimeTargetSolution:
    masses: Array
    target_time: float
    s_values: Array
    time_intervals: tuple[FloatInterval, ...]
    state_intervals: tuple[Array, ...]
    steps: tuple[IntervalSundmanStep, ...]
    target_start_state_interval: Array
    target_certificate: SundmanGlobalPhysicalTimeTargetCertificate
    target_state_interval: Array
    target_equation_residual_certificate: SundmanEquationResidualCertificate
    target_angular_momentum_certificate: AngularMomentumConservationCertificate
    target_energy_certificate: EnergyConservationCertificate
    target_linear_momentum_certificate: LinearMomentumConservationCertificate
    target_center_of_mass_certificate: CenterOfMassMotionCertificate
    target_truncation_certificate: object | None = None
    distance_power: float = 1.0
    triple_collision_exclusion_certificate: TripleCollisionExclusionCertificate | None = None

    @property
    def certified(self) -> bool:
        return bool(
            self.target_certificate.certified
            and self.target_equation_residual_certificate.certified
            and self.target_angular_momentum_certificate.certified
            and self.target_energy_certificate.certified
            and self.target_linear_momentum_certificate.certified
            and self.target_center_of_mass_certificate.certified
            and all(step.time_monotone_certified for step in self.steps)
            and all(step.equation_residual_certificate.certified for step in self.steps)
            and all(step.angular_momentum_certificate.certified for step in self.steps)
            and all(step.energy_certificate.certified for step in self.steps)
            and all(step.linear_momentum_certificate.certified for step in self.steps)
            and all(step.center_of_mass_certificate.certified for step in self.steps)
        )

    @property
    def target_s_interval(self) -> FloatInterval:
        return self.target_certificate.global_s_interval

    def target_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))

    @property
    def local_tail_bound(self) -> float:
        total = 0.0
        for step in self.steps:
            if step.truncation_certificate is not None:
                total += step.truncation_certificate.tail_bound
        if self.target_truncation_certificate is not None:
            total += self.target_truncation_certificate.tail_bound
        return float(total)

    @property
    def max_step_tail_bound(self) -> float:
        bounds = [
            step.truncation_certificate.tail_bound
            for step in self.steps
            if step.truncation_certificate is not None
        ]
        if self.target_truncation_certificate is not None:
            bounds.append(self.target_truncation_certificate.tail_bound)
        return float(max(bounds)) if bounds else 0.0

    @property
    def certified_step_count(self) -> int:
        return sum(step.truncation_certificate is not None for step in self.steps) + int(
            self.target_truncation_certificate is not None
        )

    @property
    def equation_residual_certified_step_count(self) -> int:
        return sum(step.equation_residual_certificate.certified for step in self.steps) + int(
            self.target_equation_residual_certificate.certified
        )

    @property
    def angular_momentum_certified_step_count(self) -> int:
        return sum(step.angular_momentum_certificate.certified for step in self.steps) + int(
            self.target_angular_momentum_certificate.certified
        )

    @property
    def energy_certified_step_count(self) -> int:
        return sum(step.energy_certificate.certified for step in self.steps) + int(
            self.target_energy_certificate.certified
        )

    @property
    def linear_momentum_certified_step_count(self) -> int:
        return sum(step.linear_momentum_certificate.certified for step in self.steps) + int(
            self.target_linear_momentum_certificate.certified
        )

    @property
    def center_of_mass_certified_step_count(self) -> int:
        return sum(step.center_of_mass_certificate.certified for step in self.steps) + int(
            self.target_center_of_mass_certificate.certified
        )

    @property
    def tail_certified(self) -> bool:
        return bool(
            self.target_truncation_certificate is not None
            and all(step.truncation_certificate is not None for step in self.steps)
        )

    @property
    def chain_certified(self) -> bool:
        return _interval_sundman_time_target_chain_certified(self)

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and self.tail_certified and self.chain_certified)

    @property
    def triple_collision_excluded(self) -> bool:
        return bool(
            self.triple_collision_exclusion_certificate is not None
            and self.triple_collision_exclusion_certificate.certified
        )

    @property
    def triple_collision_status(self) -> str:
        if self.triple_collision_exclusion_certificate is None:
            return "missing"
        return self.triple_collision_exclusion_certificate.status

    @property
    def triple_collision_exclusion_reason(self) -> str | None:
        if self.triple_collision_exclusion_certificate is None:
            return None
        return self.triple_collision_exclusion_certificate.reason

    @property
    def triple_collision_undecided(self) -> bool:
        return self.triple_collision_status == "undecided"

    def set_propagated_interval_enclosure(
        self,
        *,
        retained_order: int | None = None,
        target_bisections: int = 60,
    ):
        from .error_budget import propagate_sundman_target_set_enclosure

        return propagate_sundman_target_set_enclosure(
            self,
            retained_order=retained_order,
            target_bisections=target_bisections,
        )


def _sundman_float_boundaries_match(left: float, right: float) -> bool:
    left = float(left)
    right = float(right)
    if not (np.isfinite(left) and np.isfinite(right)):
        return False
    tolerance = 1.0e-10 * max(1.0, abs(left), abs(right))
    return bool(abs(left - right) <= tolerance)


def _sundman_intervals_match(left: object, right: object) -> bool:
    return bool(_as_interval(left).as_tuple() == _as_interval(right).as_tuple())


def _sundman_interval_arrays_match(left: Array, right: Array) -> bool:
    left_array = np.asarray(left, dtype=object)
    right_array = np.asarray(right, dtype=object)
    if left_array.shape != right_array.shape:
        return False
    return bool(
        all(
            _sundman_intervals_match(left_array[index], right_array[index])
            for index in np.ndindex(left_array.shape)
        )
    )


def _interval_sundman_chain_certified(
    s_values: Array,
    time_intervals: tuple[FloatInterval, ...],
    state_intervals: tuple[Array, ...],
    steps: tuple[IntervalSundmanStep, ...],
) -> bool:
    step_count = len(steps)
    s_values = np.asarray(s_values, dtype=float)
    if (
        s_values.shape != (step_count + 1,)
        or len(time_intervals) != step_count + 1
        or len(state_intervals) != step_count + 1
    ):
        return False
    if step_count == 0:
        return True
    if not np.all(np.isfinite(s_values)):
        return False
    for index, step in enumerate(steps):
        if not (
            _sundman_float_boundaries_match(s_values[index], step.start_s)
            and _sundman_float_boundaries_match(s_values[index + 1], step.end_s)
            and _sundman_intervals_match(time_intervals[index], step.start_time_interval)
            and _sundman_intervals_match(
                _as_interval(time_intervals[index]) + step.physical_step_interval,
                time_intervals[index + 1],
            )
            and _sundman_interval_arrays_match(state_intervals[index], step.start_state_interval)
            and _sundman_interval_arrays_match(state_intervals[index + 1], step.end_state_interval)
        ):
            return False
    return True


def _interval_sundman_time_target_chain_certified(
    solution: IntervalSundmanTimeTargetSolution,
) -> bool:
    if not _interval_sundman_chain_certified(
        solution.s_values,
        solution.time_intervals,
        solution.state_intervals,
        solution.steps,
    ):
        return False
    if len(solution.s_values) == 0 or not solution.time_intervals or not solution.state_intervals:
        return False
    return bool(
        _sundman_float_boundaries_match(
            solution.s_values[-1],
            solution.target_certificate.chart_start_s,
        )
        and _sundman_interval_arrays_match(
            solution.state_intervals[-1],
            solution.target_start_state_interval,
        )
        and _sundman_intervals_match(
            solution.time_intervals[-1],
            solution.target_certificate.start_time_interval,
        )
    )


def _evaluate(coefficients: Array, value: float) -> Array:
    out = np.zeros(coefficients.shape[1:], dtype=float)
    for coefficient in coefficients[::-1]:
        out = out * value + coefficient
    return out


def pairwise_distance_product(positions: Array, *, distance_power: float = 1.0) -> float:
    positions = np.asarray(positions, dtype=float)
    product = 1.0
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            product *= np.linalg.norm(positions[i] - positions[j]) ** distance_power
    return float(product)


def sundman_factor_coefficients(position: Array, max_degree: int, *, distance_power: float = 1.0) -> Array:
    """Series coefficients for product over pair distances to `distance_power`."""

    position = np.asarray(position, dtype=float)
    _degree_count, body_count, dimension = position.shape
    factor = np.zeros(max_degree + 1, dtype=float)
    factor[0] = 1.0
    for i in range(body_count):
        for j in range(i + 1, body_count):
            delta = position[: max_degree + 1, j] - position[: max_degree + 1, i]
            distance_squared = np.zeros(max_degree + 1, dtype=float)
            for axis in range(dimension):
                distance_squared += scalar_series_product(delta[:, axis], delta[:, axis], max_degree)
            pair_factor = scalar_series_power(distance_squared, distance_power / 2.0, max_degree)
            factor = scalar_series_product(factor, pair_factor, max_degree)
    return factor


def sundman_factor_interval_coefficients(
    position: Array,
    max_degree: int,
    *,
    distance_power: float = 1.0,
) -> tuple[FloatInterval, ...]:
    """Interval coefficients enclosing the Sundman distance product."""

    if max_degree < 0:
        raise ValueError("max_degree cannot be negative")
    if distance_power <= 0.0:
        raise ValueError("distance_power must be positive")
    position = np.asarray(position, dtype=object)
    _degree_count, body_count, dimension = position.shape
    factor = tuple([FloatInterval.point(1.0)] + [zero_interval() for _ in range(max_degree)])
    for i in range(body_count):
        for j in range(i + 1, body_count):
            distance_squared = tuple(zero_interval() for _ in range(max_degree + 1))
            for axis in range(dimension):
                delta = tuple(
                    _as_interval(position[n, j, axis]) - _as_interval(position[n, i, axis])
                    for n in range(max_degree + 1)
                )
                squared = list(interval_series_product(delta, delta, max_degree))
                squared[0] = _interval_square(delta[0])
                distance_squared = interval_series_add(distance_squared, tuple(squared))
            pair_factor = interval_series_power(distance_squared, distance_power / 2.0, max_degree)
            factor = interval_series_product(factor, pair_factor, max_degree)
    return factor


def sundman_factor_interval_over_s_interval(
    chart: IntervalSundmanTaylorSolution,
    s_interval: FloatInterval,
) -> FloatInterval:
    """Evaluate an interval enclosure of `dt/ds` over an `s` interval."""

    factor = sundman_factor_interval_coefficients(
        chart.position,
        chart.order,
        distance_power=chart.distance_power,
    )
    return interval_polynomial_eval(factor, s_interval)


def _series_vector_product(scalar: Array, vector: Array, max_degree: int) -> Array:
    out = np.zeros_like(vector[: max_degree + 1])
    for n in range(max_degree + 1):
        for k in range(n + 1):
            out[n] += scalar[k] * vector[n - k]
    return out


def series_vector_product(scalar: Array, vector: Array, max_degree: int) -> Array:
    """Return the product of a scalar series and vector-valued series."""

    return _series_vector_product(scalar, vector, max_degree)


def _interval_series_vector_product(scalar: tuple[FloatInterval, ...], vector: Array, max_degree: int) -> Array:
    out = _interval_zeros(vector[: max_degree + 1].shape)
    for n in range(max_degree + 1):
        for k in range(n + 1):
            for index in np.ndindex(vector.shape[1:]):
                out[(n, *index)] = out[(n, *index)] + scalar[k] * _as_interval(vector[(n - k, *index)])
    return out


def _validate(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    if positions.shape[0] != 3 or masses.shape != (3,):
        raise ValueError("expected three bodies and three masses")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive")
    if pairwise_distance_product(positions) == 0.0:
        raise ValueError("initial data must be collision-free")
    return positions, velocities, masses


def _as_interval(value: object) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


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


def _inflate_interval(value: FloatInterval, radius: float) -> FloatInterval:
    if radius < 0.0:
        raise ValueError("radius cannot be negative")
    return FloatInterval(
        float(np.nextafter(value.lower - radius, -np.inf)),
        float(np.nextafter(value.upper + radius, np.inf)),
    )


def _inflate_interval_array(values: Array, radius: float) -> Array:
    values = np.asarray(values, dtype=object)
    out = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        out[index] = _inflate_interval(_as_interval(values[index]), radius)
    return out


def _interval_state(positions: Array, velocities: Array) -> Array:
    return np.concatenate([np.asarray(positions, dtype=object).reshape(-1), np.asarray(velocities, dtype=object).reshape(-1)])


def _interval_lower(values: Array) -> Array:
    values = np.asarray(values, dtype=object)
    out = np.empty(values.shape, dtype=float)
    for index in np.ndindex(values.shape):
        out[index] = _as_interval(values[index]).lower
    return out


def _interval_upper(values: Array) -> Array:
    values = np.asarray(values, dtype=object)
    out = np.empty(values.shape, dtype=float)
    for index in np.ndindex(values.shape):
        out[index] = _as_interval(values[index]).upper
    return out


def _interval_contains_zero(value: FloatInterval) -> bool:
    return value.lower <= 0.0 <= value.upper


def _interval_array_contains_zero(values: Array) -> bool:
    values = np.asarray(values, dtype=object)
    for index in np.ndindex(values.shape):
        if not _interval_contains_zero(_as_interval(values[index])):
            return False
    return True


def _interval_square(value: FloatInterval) -> FloatInterval:
    squares = (value.lower * value.lower, value.upper * value.upper)
    upper = float(np.nextafter(max(squares), np.inf))
    if value.lower <= 0.0 <= value.upper:
        lower = 0.0
    else:
        lower = max(0.0, float(np.nextafter(min(squares), -np.inf)))
    return FloatInterval(lower, upper)


def _validate_interval_initial_data(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=object)
    velocities = np.asarray(velocities, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 2:
        raise ValueError("positions must have shape (body_count, dimension)")
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    if positions.shape[0] != 3 or masses.shape != (3,):
        raise ValueError("expected three bodies and three masses")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive")

    interval_positions = np.empty(positions.shape, dtype=object)
    interval_velocities = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        interval_positions[index] = _as_interval(positions[index])
        interval_velocities[index] = _as_interval(velocities[index])

    for i in range(3):
        for j in range(i + 1, 3):
            distance_squared = FloatInterval.point(0.0)
            for axis in range(positions.shape[1]):
                distance_squared = distance_squared + _interval_square(
                    interval_positions[j, axis] - interval_positions[i, axis]
                )
            if distance_squared.lower <= 0.0:
                raise ValueError("interval initial data must certify non-collision")
    return interval_positions, interval_velocities, masses


def construct_sundman_taylor_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
    distance_power: float = 1.0,
) -> SundmanTaylorSolution:
    """Construct a local series in fictitious Sundman time."""

    if order < 1:
        raise ValueError("order must be at least 1")
    if distance_power <= 0.0:
        raise ValueError("distance_power must be positive")
    positions, velocities, masses = _validate(positions, velocities, masses)
    body_count, dimension = positions.shape

    q = np.zeros((order + 1, body_count, dimension), dtype=float)
    v = np.zeros_like(q)
    physical_time = np.zeros(order + 1, dtype=float)
    q[0] = positions
    v[0] = velocities

    for n in range(order):
        factor = sundman_factor_coefficients(q, n, distance_power=distance_power)
        acceleration = acceleration_coefficients(q, masses, n)
        q_rhs = _series_vector_product(factor, v, n)
        v_rhs = _series_vector_product(factor, acceleration, n)
        q[n + 1] = q_rhs[n] / (n + 1)
        v[n + 1] = v_rhs[n] / (n + 1)
        physical_time[n + 1] = factor[n] / (n + 1)

    return SundmanTaylorSolution(
        position=q,
        velocity=v,
        physical_time=physical_time,
        masses=masses,
        distance_power=distance_power,
    )


def _construct_interval_sundman_taylor_from_arrays(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
    distance_power: float,
) -> IntervalSundmanTaylorSolution:
    body_count, dimension = positions.shape
    q = _interval_zeros((order + 1, body_count, dimension))
    v = _interval_zeros((order + 1, body_count, dimension))
    physical_time = _interval_zeros((order + 1,))
    q[0] = positions
    v[0] = velocities

    for n in range(order):
        factor = sundman_factor_interval_coefficients(q, n, distance_power=distance_power)
        acceleration = acceleration_interval_coefficients(q, masses, n)
        q_rhs = _interval_series_vector_product(factor, v, n)
        v_rhs = _interval_series_vector_product(factor, acceleration, n)
        scale = 1.0 / (n + 1)
        for index in np.ndindex((body_count, dimension)):
            q[(n + 1, *index)] = q_rhs[(n, *index)].scale(scale)
            v[(n + 1, *index)] = v_rhs[(n, *index)].scale(scale)
        physical_time[n + 1] = factor[n].scale(scale)

    return IntervalSundmanTaylorSolution(
        position=q,
        velocity=v,
        physical_time=physical_time,
        masses=masses,
        distance_power=distance_power,
    )


def construct_interval_sundman_taylor_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
    distance_power: float = 1.0,
) -> IntervalSundmanTaylorSolution:
    """Construct interval Sundman-time coefficients from point initial data."""

    if order < 1:
        raise ValueError("order must be at least 1")
    if distance_power <= 0.0:
        raise ValueError("distance_power must be positive")
    positions, velocities, masses = _validate(positions, velocities, masses)
    return _construct_interval_sundman_taylor_from_arrays(
        _interval_array_from_points(positions),
        _interval_array_from_points(velocities),
        masses,
        order=order,
        distance_power=distance_power,
    )


def construct_interval_sundman_taylor_solution_from_intervals(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
    distance_power: float = 1.0,
) -> IntervalSundmanTaylorSolution:
    """Construct interval Sundman-time coefficients from interval initial data."""

    if order < 1:
        raise ValueError("order must be at least 1")
    if distance_power <= 0.0:
        raise ValueError("distance_power must be positive")
    positions, velocities, masses = _validate_interval_initial_data(positions, velocities, masses)
    return _construct_interval_sundman_taylor_from_arrays(
        positions,
        velocities,
        masses,
        order=order,
        distance_power=distance_power,
    )


def certify_interval_sundman_equations(
    chart: IntervalSundmanTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> SundmanEquationResidualCertificate:
    """Certify the interval Sundman coefficients satisfy the lifted ODE.

    The residual arrays enclose, coefficient by coefficient,
    ``dq/ds - g(q) v``, ``dv/ds - g(q) a(q)``, and ``dt/ds - g(q)``.
    The certificate is proof-grade when every residual interval contains zero.
    """

    max_count = chart.order
    if coefficient_count is None:
        coefficient_count = max_count
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > max_count:
        raise ValueError("coefficient_count cannot exceed chart.order")

    max_degree = coefficient_count - 1
    factor = sundman_factor_interval_coefficients(
        chart.position,
        max_degree,
        distance_power=chart.distance_power,
    )
    acceleration = acceleration_interval_coefficients(chart.position, chart.masses, max_degree)
    position_rhs = _interval_series_vector_product(factor, chart.velocity, max_degree)
    velocity_rhs = _interval_series_vector_product(factor, acceleration, max_degree)

    position_residual = _interval_zeros(position_rhs.shape)
    velocity_residual = _interval_zeros(velocity_rhs.shape)
    physical_time_residual = []
    for degree in range(coefficient_count):
        scale = float(degree + 1)
        for index in np.ndindex(chart.position.shape[1:]):
            position_residual[(degree, *index)] = _as_interval(chart.position[(degree + 1, *index)]).scale(
                scale
            ) - _as_interval(position_rhs[(degree, *index)])
            velocity_residual[(degree, *index)] = _as_interval(chart.velocity[(degree + 1, *index)]).scale(
                scale
            ) - _as_interval(velocity_rhs[(degree, *index)])
        physical_time_residual.append(
            _as_interval(chart.physical_time[degree + 1]).scale(scale) - factor[degree]
        )

    return SundmanEquationResidualCertificate(
        coefficient_count=coefficient_count,
        position_residual=position_residual,
        velocity_residual=velocity_residual,
        physical_time_residual=tuple(physical_time_residual),
    )


def _certified_interval_sundman_equation_residual(
    series: IntervalSundmanTaylorSolution,
) -> SundmanEquationResidualCertificate:
    certificate = certify_interval_sundman_equations(series)
    if not certificate.certified:
        raise RuntimeError("Sundman interval equation residual is not certified")
    return certificate


def _sundman_tail_certificate(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    retained_order: int,
    guard_order: int | None,
    step_size: float,
    distance_power: float,
    certificate_mode: str = "guarded",
):
    if certificate_mode == "cauchy":
        from .tail_bounds import sundman_cauchy_majorant_tail_certificate

        return sundman_cauchy_majorant_tail_certificate(
            positions,
            velocities,
            masses,
            retained_order=retained_order,
            step_size=step_size,
            distance_power=distance_power,
        )
    if guard_order is None:
        return None
    if certificate_mode != "guarded":
        raise ValueError("tail_certificate_mode must be 'guarded' or 'cauchy'")
    from .tail_bounds import sundman_taylor_tail_certificate

    return sundman_taylor_tail_certificate(
        positions,
        velocities,
        masses,
        retained_order=retained_order,
        guard_order=guard_order,
        step_size=step_size,
        distance_power=distance_power,
    )


def continue_sundman_to_s(
    positions: Array,
    velocities: Array,
    masses: Array,
    s_final: float,
    *,
    order: int = 16,
    max_s_step: float = 0.03,
    distance_power: float = 1.0,
    tail_guard_order: int | None = None,
    tail_certificate_mode: str = "guarded",
) -> SundmanContinuedSolution:
    """Chain point Sundman-time charts to a target fictitious time `s`."""

    if max_s_step <= 0.0:
        raise ValueError("max_s_step must be positive")
    if tail_certificate_mode not in {"guarded", "cauchy"}:
        raise ValueError("tail_certificate_mode must be 'guarded' or 'cauchy'")
    positions, velocities, masses = _validate(positions, velocities, masses)
    current_positions = positions.copy()
    current_velocities = velocities.copy()
    current_s = 0.0
    current_time = 0.0
    times = [current_time]
    states = [np.concatenate([current_positions.reshape(-1), current_velocities.reshape(-1)])]
    steps: list[SundmanStep] = []

    while abs(s_final - current_s) > 10.0 * np.finfo(float).eps:
        remaining = s_final - current_s
        direction = np.sign(remaining)
        s_step = float(direction * min(abs(remaining), max_s_step))
        if tail_certificate_mode == "cauchy":
            cauchy_trial_certificate = _sundman_tail_certificate(
                current_positions,
                current_velocities,
                masses,
                retained_order=order,
                guard_order=tail_guard_order,
                step_size=0.0,
                distance_power=distance_power,
                certificate_mode=tail_certificate_mode,
            )
            s_step = float(direction * min(abs(s_step), 0.5 * cauchy_trial_certificate.s_radius))
        series = construct_sundman_taylor_solution(
            current_positions,
            current_velocities,
            masses,
            order=order,
            distance_power=distance_power,
        )
        physical_step = series.physical_time_at_s(s_step)
        truncation_certificate = _sundman_tail_certificate(
            current_positions,
            current_velocities,
            masses,
            retained_order=order,
            guard_order=tail_guard_order,
            step_size=s_step,
            distance_power=distance_power,
            certificate_mode=tail_certificate_mode,
        )
        step = SundmanStep(
            start_time=current_time,
            s_step=s_step,
            physical_step=physical_step,
            series=series,
            min_pair_distance=_min_pair_distance(current_positions),
            truncation_certificate=truncation_certificate,
        )
        current_positions = series.positions_at_s(s_step)
        current_velocities = series.velocities_at_s(s_step)
        current_s += s_step
        current_time = step.end_time
        steps.append(step)
        times.append(current_time)
        states.append(np.concatenate([current_positions.reshape(-1), current_velocities.reshape(-1)]))

    return SundmanContinuedSolution(
        masses=masses,
        times=np.array(times, dtype=float),
        states=np.vstack(states),
        steps=tuple(steps),
    )


def continue_interval_sundman_to_s(
    positions: Array,
    velocities: Array,
    masses: Array,
    s_final: float,
    *,
    order: int = 16,
    max_s_step: float = 0.03,
    distance_power: float = 1.0,
    tail_certificate_mode: str = "none",
) -> IntervalSundmanContinuedSolution:
    """Propagate interval Sundman-time charts to a target fictitious time `s`."""

    if max_s_step <= 0.0:
        raise ValueError("max_s_step must be positive")
    if tail_certificate_mode not in {"none", "cauchy"}:
        raise ValueError("tail_certificate_mode must be 'none' or 'cauchy'")
    current_positions, current_velocities, masses = _validate_interval_initial_data(positions, velocities, masses)
    triple_collision_exclusion_certificate = certify_nonzero_angular_momentum_excludes_triple_collision(
        current_positions,
        current_velocities,
        masses,
    )
    current_s = 0.0
    current_time_interval = FloatInterval.point(0.0)
    s_values = [current_s]
    time_intervals = [current_time_interval]
    state_intervals = [_interval_state(current_positions, current_velocities)]
    steps: list[IntervalSundmanStep] = []

    while abs(s_final - current_s) > 10.0 * np.finfo(float).eps:
        remaining = s_final - current_s
        direction = np.sign(remaining)
        s_step = float(direction * min(abs(remaining), max_s_step))
        if tail_certificate_mode == "cauchy":
            from .tail_bounds import sundman_interval_cauchy_majorant_tail_certificate

            cauchy_trial_certificate = sundman_interval_cauchy_majorant_tail_certificate(
                current_positions,
                current_velocities,
                masses,
                retained_order=order,
                step_size=0.0,
                distance_power=distance_power,
            )
            s_step = float(direction * min(abs(s_step), 0.5 * cauchy_trial_certificate.s_radius))
        local_bounds = FloatInterval(0.0, s_step) if s_step > 0.0 else FloatInterval(s_step, 0.0)
        start_state_interval = _interval_state(current_positions, current_velocities)
        series = construct_interval_sundman_taylor_solution_from_intervals(
            current_positions,
            current_velocities,
            masses,
            order=order,
            distance_power=distance_power,
        )
        equation_residual_certificate = _certified_interval_sundman_equation_residual(series)
        angular_momentum_certificate = certify_interval_centered_angular_momentum_conservation(
            series.position,
            series.velocity,
            series.masses,
            coefficient_count=order,
        )
        if not angular_momentum_certificate.certified:
            raise RuntimeError("Sundman centered angular momentum conservation is not certified")
        energy_certificate = certify_interval_total_energy_conservation(
            series.position,
            series.velocity,
            series.masses,
            coefficient_count=order,
        )
        if not energy_certificate.certified:
            raise RuntimeError("Sundman total-energy conservation is not certified")
        linear_momentum_certificate = certify_interval_linear_momentum_conservation(
            series.velocity,
            series.masses,
            coefficient_count=order,
        )
        if not linear_momentum_certificate.certified:
            raise RuntimeError("Sundman linear momentum conservation is not certified")
        center_of_mass_certificate = certify_interval_center_of_mass_motion(
            series.position,
            series.velocity,
            series.physical_time,
            series.masses,
            coefficient_count=order,
        )
        if not center_of_mass_certificate.certified:
            raise RuntimeError("Sundman center-of-mass motion is not certified")
        factor_interval = sundman_factor_interval_over_s_interval(series, local_bounds)
        if factor_interval.lower <= 0.0:
            raise RuntimeError("Sundman time monotonicity is not certified on interval step")
        truncated_next_positions = series.positions_at_s(s_step)
        truncated_next_velocities = series.velocities_at_s(s_step)
        physical_step_interval = series.physical_time_at_s(s_step)
        truncation_certificate = None
        if tail_certificate_mode == "cauchy":
            from .tail_bounds import sundman_interval_cauchy_majorant_tail_certificate

            truncation_certificate = sundman_interval_cauchy_majorant_tail_certificate(
                current_positions,
                current_velocities,
                masses,
                retained_order=order,
                step_size=s_step,
                distance_power=distance_power,
            )
            tail_bound = truncation_certificate.tail_bound
            next_positions = _inflate_interval_array(truncated_next_positions, tail_bound)
            next_velocities = _inflate_interval_array(truncated_next_velocities, tail_bound)
            physical_step_interval = _inflate_interval(physical_step_interval, tail_bound)
        else:
            next_positions = truncated_next_positions
            next_velocities = truncated_next_velocities
        end_state_interval = _interval_state(next_positions, next_velocities)
        step = IntervalSundmanStep(
            start_s=current_s,
            s_step=s_step,
            start_time_interval=current_time_interval,
            physical_step_interval=physical_step_interval,
            factor_interval=factor_interval,
            series=series,
            start_state_interval=start_state_interval,
            end_state_interval=end_state_interval,
            equation_residual_certificate=equation_residual_certificate,
            angular_momentum_certificate=angular_momentum_certificate,
            energy_certificate=energy_certificate,
            linear_momentum_certificate=linear_momentum_certificate,
            center_of_mass_certificate=center_of_mass_certificate,
            truncation_certificate=truncation_certificate,
        )
        current_positions = next_positions
        current_velocities = next_velocities
        current_s = step.end_s
        current_time_interval = step.end_time_interval
        steps.append(step)
        s_values.append(current_s)
        time_intervals.append(current_time_interval)
        state_intervals.append(end_state_interval)

    return IntervalSundmanContinuedSolution(
        masses=masses,
        s_values=np.array(s_values, dtype=float),
        time_intervals=tuple(time_intervals),
        state_intervals=tuple(state_intervals),
        steps=tuple(steps),
        triple_collision_exclusion_certificate=triple_collision_exclusion_certificate,
    )


def certify_sundman_physical_time_target(
    chart: IntervalSundmanTaylorSolution,
    target_time: float,
    s_bounds: FloatInterval,
    *,
    max_bisections: int = 60,
) -> SundmanPhysicalTimeTargetCertificate:
    """Bracket a physical-time target inside one interval Sundman chart.

    The certificate is proof-grade only when `certified` is true: `dt/ds` is
    strictly positive on the returned interval, and interval endpoint times
    force every represented trajectory to cross `target_time` inside it.
    """

    target_time = float(target_time)
    if max_bisections < 0:
        raise ValueError("max_bisections cannot be negative")
    if s_bounds.lower >= s_bounds.upper:
        raise ValueError("s_bounds must have positive width")

    low = s_bounds.lower
    high = s_bounds.upper

    def make_certificate(bisections: int) -> SundmanPhysicalTimeTargetCertificate:
        interval = FloatInterval(low, high)
        return SundmanPhysicalTimeTargetCertificate(
            target_time=target_time,
            s_interval=interval,
            time_at_lower=chart.physical_time_at_s(low),
            time_at_upper=chart.physical_time_at_s(high),
            factor_interval=sundman_factor_interval_over_s_interval(chart, interval),
            bisections=bisections,
        )

    certificate = make_certificate(0)
    if not certificate.certified:
        return certificate

    for bisection in range(1, max_bisections + 1):
        midpoint = (low + high) / 2.0
        midpoint_time = chart.physical_time_at_s(midpoint)
        if midpoint_time.upper <= target_time:
            low = midpoint
        elif midpoint_time.lower >= target_time:
            high = midpoint
        else:
            break

        next_certificate = make_certificate(bisection)
        if not next_certificate.certified:
            break
        certificate = next_certificate

    return certificate


def _certify_global_sundman_physical_time_target(
    chart: IntervalSundmanTaylorSolution,
    start_time_interval: FloatInterval,
    target_time: float,
    local_s_bounds: FloatInterval,
    *,
    chart_start_s: float,
    max_bisections: int,
) -> SundmanGlobalPhysicalTimeTargetCertificate:
    target_time = float(target_time)
    if local_s_bounds.lower >= local_s_bounds.upper:
        raise ValueError("local_s_bounds must have positive width")
    if max_bisections < 0:
        raise ValueError("max_bisections cannot be negative")

    low = local_s_bounds.lower
    high = local_s_bounds.upper

    def make_certificate(bisections: int) -> SundmanGlobalPhysicalTimeTargetCertificate:
        local_interval = FloatInterval(low, high)
        local_lower = chart.physical_time_at_s(low)
        local_upper = chart.physical_time_at_s(high)
        global_lower = start_time_interval + local_lower
        global_upper = start_time_interval + local_upper
        return SundmanGlobalPhysicalTimeTargetCertificate(
            target_time=target_time,
            chart_start_s=float(chart_start_s),
            local_s_interval=local_interval,
            global_s_interval=FloatInterval(chart_start_s + low, chart_start_s + high),
            start_time_interval=start_time_interval,
            local_time_at_lower=local_lower,
            local_time_at_upper=local_upper,
            global_time_at_lower=global_lower,
            global_time_at_upper=global_upper,
            factor_interval=sundman_factor_interval_over_s_interval(chart, local_interval),
            bisections=bisections,
        )

    certificate = make_certificate(0)
    if not certificate.certified:
        return certificate

    for bisection in range(1, max_bisections + 1):
        midpoint = (low + high) / 2.0
        midpoint_time = start_time_interval + chart.physical_time_at_s(midpoint)
        if midpoint_time.upper <= target_time:
            low = midpoint
        elif midpoint_time.lower >= target_time:
            high = midpoint
        else:
            break

        next_certificate = make_certificate(bisection)
        if not next_certificate.certified:
            break
        certificate = next_certificate

    return certificate


def certify_global_sundman_physical_time_target(
    chart: IntervalSundmanTaylorSolution,
    start_time_interval: FloatInterval,
    target_time: float,
    local_s_bounds: FloatInterval,
    *,
    chart_start_s: float = 0.0,
    max_bisections: int = 60,
) -> SundmanGlobalPhysicalTimeTargetCertificate:
    """Bracket a physical-time target inside a chart with interval start time."""

    return _certify_global_sundman_physical_time_target(
        chart,
        start_time_interval,
        target_time,
        local_s_bounds,
        chart_start_s=chart_start_s,
        max_bisections=max_bisections,
    )


def continue_interval_sundman_to_time(
    positions: Array,
    velocities: Array,
    masses: Array,
    target_time: float,
    *,
    order: int = 16,
    max_s_step: float = 0.03,
    distance_power: float = 1.0,
    max_steps: int = 10000,
    target_bisections: int = 60,
    step_shrink_bisections: int = 60,
    tail_certificate_mode: str = "none",
) -> IntervalSundmanTimeTargetSolution:
    """Propagate interval Sundman charts until a physical-time target is certified."""

    target_time = float(target_time)
    if max_s_step <= 0.0:
        raise ValueError("max_s_step must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if target_bisections < 0:
        raise ValueError("target_bisections cannot be negative")
    if step_shrink_bisections < 0:
        raise ValueError("step_shrink_bisections cannot be negative")
    if tail_certificate_mode not in {"none", "cauchy"}:
        raise ValueError("tail_certificate_mode must be 'none' or 'cauchy'")
    if target_time == 0.0:
        raise ValueError("target_time must be nonzero")

    current_positions, current_velocities, masses = _validate_interval_initial_data(positions, velocities, masses)
    triple_collision_exclusion_certificate = certify_nonzero_angular_momentum_excludes_triple_collision(
        current_positions,
        current_velocities,
        masses,
    )
    direction = float(np.sign(target_time))
    current_s = 0.0
    current_time_interval = FloatInterval.point(0.0)
    s_values = [current_s]
    time_intervals = [current_time_interval]
    state_intervals = [_interval_state(current_positions, current_velocities)]
    steps: list[IntervalSundmanStep] = []

    for _step_index in range(max_steps):
        local_step = direction * max_s_step
        if tail_certificate_mode == "cauchy":
            from .tail_bounds import sundman_interval_cauchy_majorant_tail_certificate

            cauchy_trial_certificate = sundman_interval_cauchy_majorant_tail_certificate(
                current_positions,
                current_velocities,
                masses,
                retained_order=order,
                step_size=0.0,
                distance_power=distance_power,
            )
            local_step = float(direction * min(abs(local_step), 0.5 * cauchy_trial_certificate.s_radius))
        local_bounds = FloatInterval(0.0, local_step) if local_step > 0.0 else FloatInterval(local_step, 0.0)
        start_state_interval = _interval_state(current_positions, current_velocities)
        series = construct_interval_sundman_taylor_solution_from_intervals(
            current_positions,
            current_velocities,
            masses,
            order=order,
            distance_power=distance_power,
        )
        equation_residual_certificate = _certified_interval_sundman_equation_residual(series)
        angular_momentum_certificate = certify_interval_centered_angular_momentum_conservation(
            series.position,
            series.velocity,
            series.masses,
            coefficient_count=order,
        )
        if not angular_momentum_certificate.certified:
            raise RuntimeError("Sundman centered angular momentum conservation is not certified")
        energy_certificate = certify_interval_total_energy_conservation(
            series.position,
            series.velocity,
            series.masses,
            coefficient_count=order,
        )
        if not energy_certificate.certified:
            raise RuntimeError("Sundman total-energy conservation is not certified")
        linear_momentum_certificate = certify_interval_linear_momentum_conservation(
            series.velocity,
            series.masses,
            coefficient_count=order,
        )
        if not linear_momentum_certificate.certified:
            raise RuntimeError("Sundman linear momentum conservation is not certified")
        center_of_mass_certificate = certify_interval_center_of_mass_motion(
            series.position,
            series.velocity,
            series.physical_time,
            series.masses,
            coefficient_count=order,
        )
        if not center_of_mass_certificate.certified:
            raise RuntimeError("Sundman center-of-mass motion is not certified")

        for _shrink_index in range(step_shrink_bisections + 1):
            local_bounds = (
                FloatInterval(0.0, local_step)
                if local_step > 0.0
                else FloatInterval(local_step, 0.0)
            )
            target_certificate = _certify_global_sundman_physical_time_target(
                series,
                current_time_interval,
                target_time,
                local_bounds,
                chart_start_s=current_s,
                max_bisections=target_bisections,
            )
            if target_certificate.certified:
                target_equation_residual_certificate = equation_residual_certificate
                target_angular_momentum_certificate = angular_momentum_certificate
                target_energy_certificate = energy_certificate
                target_linear_momentum_certificate = linear_momentum_certificate
                target_center_of_mass_certificate = center_of_mass_certificate
                target_truncation_certificate = None
                target_state_interval = series.state_over_s_interval(target_certificate.local_s_interval)
                if tail_certificate_mode == "cauchy":
                    from .tail_bounds import sundman_interval_cauchy_majorant_tail_certificate

                    target_radius_step = max(
                        abs(target_certificate.local_s_interval.lower),
                        abs(target_certificate.local_s_interval.upper),
                    )
                    target_truncation_certificate = sundman_interval_cauchy_majorant_tail_certificate(
                        current_positions,
                        current_velocities,
                        masses,
                        retained_order=order,
                        step_size=target_radius_step,
                        distance_power=distance_power,
                    )
                    target_state_interval = _inflate_interval_array(
                        target_state_interval,
                        target_truncation_certificate.tail_bound,
                    )
                return IntervalSundmanTimeTargetSolution(
                    masses=masses,
                    target_time=target_time,
                    s_values=np.array(s_values, dtype=float),
                    time_intervals=tuple(time_intervals),
                    state_intervals=tuple(state_intervals),
                    steps=tuple(steps),
                    target_start_state_interval=start_state_interval,
                    target_certificate=target_certificate,
                    target_state_interval=target_state_interval,
                    target_equation_residual_certificate=target_equation_residual_certificate,
                    target_angular_momentum_certificate=target_angular_momentum_certificate,
                    target_energy_certificate=target_energy_certificate,
                    target_linear_momentum_certificate=target_linear_momentum_certificate,
                    target_center_of_mass_certificate=target_center_of_mass_certificate,
                    target_truncation_certificate=target_truncation_certificate,
                    distance_power=distance_power,
                    triple_collision_exclusion_certificate=triple_collision_exclusion_certificate,
                )

            factor_interval = sundman_factor_interval_over_s_interval(series, local_bounds)
            if factor_interval.lower <= 0.0:
                raise RuntimeError("Sundman time monotonicity is not certified on interval step")
            truncated_next_positions = series.positions_at_s(local_step)
            truncated_next_velocities = series.velocities_at_s(local_step)
            physical_step_interval = series.physical_time_at_s(local_step)
            truncation_certificate = None
            if tail_certificate_mode == "cauchy":
                from .tail_bounds import sundman_interval_cauchy_majorant_tail_certificate

                truncation_certificate = sundman_interval_cauchy_majorant_tail_certificate(
                    current_positions,
                    current_velocities,
                    masses,
                    retained_order=order,
                    step_size=local_step,
                    distance_power=distance_power,
                )
                tail_bound = truncation_certificate.tail_bound
                next_positions = _inflate_interval_array(truncated_next_positions, tail_bound)
                next_velocities = _inflate_interval_array(truncated_next_velocities, tail_bound)
                physical_step_interval = _inflate_interval(physical_step_interval, tail_bound)
            else:
                next_positions = truncated_next_positions
                next_velocities = truncated_next_velocities
            end_time_interval = current_time_interval + physical_step_interval
            target_inside_uncertified_step = (
                (direction > 0.0 and target_time <= end_time_interval.upper)
                or (direction < 0.0 and target_time >= end_time_interval.lower)
            )
            if not target_inside_uncertified_step:
                break
            local_step *= 0.5
        else:
            raise RuntimeError(
                "target physical time remained inside an uncertified interval step "
                "after adaptive shortening"
            )

        end_state_interval = _interval_state(next_positions, next_velocities)
        step = IntervalSundmanStep(
            start_s=current_s,
            s_step=local_step,
            start_time_interval=current_time_interval,
            physical_step_interval=physical_step_interval,
            factor_interval=factor_interval,
            series=series,
            start_state_interval=start_state_interval,
            end_state_interval=end_state_interval,
            equation_residual_certificate=equation_residual_certificate,
            angular_momentum_certificate=angular_momentum_certificate,
            energy_certificate=energy_certificate,
            linear_momentum_certificate=linear_momentum_certificate,
            center_of_mass_certificate=center_of_mass_certificate,
            truncation_certificate=truncation_certificate,
        )
        current_positions = next_positions
        current_velocities = next_velocities
        current_s = step.end_s
        current_time_interval = end_time_interval
        steps.append(step)
        s_values.append(current_s)
        time_intervals.append(current_time_interval)
        state_intervals.append(end_state_interval)

    raise RuntimeError("target physical time was not reached before max_steps")


def continue_sundman_to_time(
    positions: Array,
    velocities: Array,
    masses: Array,
    t_final: float,
    *,
    order: int = 16,
    max_s_step: float = 0.03,
    distance_power: float = 1.0,
    tail_guard_order: int | None = None,
    tail_certificate_mode: str = "guarded",
) -> SundmanContinuedSolution:
    """Chain Sundman-time Taylor charts until a target physical time is reached."""

    if max_s_step <= 0.0:
        raise ValueError("max_s_step must be positive")
    if tail_certificate_mode not in {"guarded", "cauchy"}:
        raise ValueError("tail_certificate_mode must be 'guarded' or 'cauchy'")
    positions, velocities, masses = _validate(positions, velocities, masses)
    current_positions = positions.copy()
    current_velocities = velocities.copy()
    current_time = 0.0
    times = [current_time]
    states = [np.concatenate([current_positions.reshape(-1), current_velocities.reshape(-1)])]
    steps: list[SundmanStep] = []

    while abs(t_final - current_time) > 10.0 * np.finfo(float).eps:
        remaining = t_final - current_time
        direction = np.sign(remaining)
        trial_s_step = float(direction * max_s_step)
        if tail_certificate_mode == "cauchy":
            cauchy_trial_certificate = _sundman_tail_certificate(
                current_positions,
                current_velocities,
                masses,
                retained_order=order,
                guard_order=tail_guard_order,
                step_size=0.0,
                distance_power=distance_power,
                certificate_mode=tail_certificate_mode,
            )
            trial_s_step = float(direction * min(abs(trial_s_step), 0.5 * cauchy_trial_certificate.s_radius))
        series = construct_sundman_taylor_solution(
            current_positions,
            current_velocities,
            masses,
            order=order,
            distance_power=distance_power,
        )
        trial_physical_step = series.physical_time_at_s(trial_s_step)

        if abs(trial_physical_step) > abs(remaining):
            low, high = (0.0, trial_s_step) if trial_s_step > 0.0 else (trial_s_step, 0.0)

            def root(s_value: float) -> float:
                return series.physical_time_at_s(s_value) - remaining

            s_step = float(brentq(root, low, high, xtol=1e-15, rtol=1e-15))
            physical_step = remaining
        else:
            s_step = trial_s_step
            physical_step = trial_physical_step

        truncation_certificate = _sundman_tail_certificate(
            current_positions,
            current_velocities,
            masses,
            retained_order=order,
            guard_order=tail_guard_order,
            step_size=s_step,
            distance_power=distance_power,
            certificate_mode=tail_certificate_mode,
        )
        step = SundmanStep(
            start_time=current_time,
            s_step=s_step,
            physical_step=physical_step,
            series=series,
            min_pair_distance=_min_pair_distance(current_positions),
            truncation_certificate=truncation_certificate,
        )
        current_positions = series.positions_at_s(s_step)
        current_velocities = series.velocities_at_s(s_step)
        current_time = step.end_time
        steps.append(step)
        times.append(current_time)
        states.append(np.concatenate([current_positions.reshape(-1), current_velocities.reshape(-1)]))

    return SundmanContinuedSolution(
        masses=masses,
        times=np.array(times, dtype=float),
        states=np.vstack(states),
        steps=tuple(steps),
    )


def _min_pair_distance(positions: Array) -> float:
    distances = []
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            distances.append(np.linalg.norm(positions[i] - positions[j]))
    return float(np.min(distances))


def reference_state_at_sundman_chart(
    chart: SundmanTaylorSolution,
    s_value: float,
) -> Array:
    """Integrate in Newtonian time to the physical time represented by a chart."""

    physical_time = chart.physical_time_at_s(s_value)
    return integrate_reference(chart.position[0], chart.velocity[0], chart.masses, physical_time)
