"""Taylor-tail certificates.

These certificates are guarded majorants: compute extra coefficients beyond the
retained order, check that the guard terms decrease geometrically at the chosen
step, and use that observed ratio to bound the omitted tail.  Point
coefficients give numerical certificates.  Interval coefficients strengthen the
same check by enclosing each computed guard term before the ratio test.
Ordinary planar, Sundman-time, and regularized-binary Cauchy majorants can also
certify interval initial-state boxes without sampling representative points;
ordinary Cauchy certificates additionally aggregate interval-state unions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .binary_chart import IntervalRegularizedBinaryCollisionChartState, RegularizedBinaryCollisionChartState
from .binary_series import (
    IntervalRegularizedBinaryTaylorSolution,
    RegularizedBinaryTaylorSolution,
    construct_interval_regularized_binary_taylor_solution,
)
from .intervals import FloatInterval, interval_array_series_eval
from .ks_binary_chart import SpatialKSBinaryChartState
from .ks_binary_series import (
    IntervalSpatialKSBinaryChartState,
    IntervalSpatialKSBinaryTaylorSolution,
    SpatialKSBinaryTaylorSolution,
    certify_spatial_ks_binary_horizontal_constraint,
    certify_spatial_ks_binary_interval_taylor_equations,
    certify_spatial_ks_binary_pair_energy_constraint,
    construct_interval_spatial_ks_binary_taylor_solution,
    construct_interval_spatial_ks_binary_taylor_solution_from_intervals,
)
from .series import (
    IntervalTaylorSolution,
    TaylorSolution,
    construct_interval_taylor_solution,
)
from .sundman import (
    IntervalSundmanTaylorSolution,
    SundmanTaylorSolution,
    construct_interval_sundman_taylor_solution,
)


Array = np.ndarray


@dataclass(frozen=True)
class TailBoundCertificate:
    retained_order: int
    computed_order: int
    step_size: float
    first_omitted_term: float
    observed_tail: float
    ratio_bound: float
    tail_bound: float
    coefficient_source: str = "point"

    @property
    def guard_terms(self) -> int:
        return self.computed_order - self.retained_order

    @property
    def is_geometric(self) -> bool:
        return self.guard_terms >= 2 and np.isfinite(self.ratio_bound) and self.ratio_bound < 1.0

    @property
    def is_nontrivial(self) -> bool:
        return self.is_geometric and self.tail_bound >= self.observed_tail >= 0.0

    @property
    def uses_interval_coefficients(self) -> bool:
        return self.coefficient_source == "interval"


@dataclass(frozen=True)
class SpatialKSSegmentPropagationStep:
    """One certified re-expanded KS interval substep."""

    index: int
    start_state: IntervalSpatialKSBinaryChartState
    parameter_interval: FloatInterval
    physical_time_delta: FloatInterval
    tail_certificate: TailBoundCertificate
    end_state: IntervalSpatialKSBinaryChartState
    equation_residual_certificate: object | None = None
    horizontal_constraint_certificate: object | None = None
    pair_energy_constraint_certificate: object | None = None

    @property
    def step_size(self) -> float:
        return float(self.parameter_interval.upper - self.parameter_interval.lower)

    @property
    def equation_residual_certified(self) -> bool:
        return bool(
            self.equation_residual_certificate is not None
            and getattr(self.equation_residual_certificate, "certified", False)
        )

    @property
    def projection_constraints_certified(self) -> bool:
        return bool(
            self.horizontal_constraint_certificate is not None
            and self.pair_energy_constraint_certificate is not None
            and getattr(self.horizontal_constraint_certificate, "certified", False)
            and getattr(self.pair_energy_constraint_certificate, "certified", False)
        )

    @property
    def chart_evidence_certified(self) -> bool:
        return bool(
            self.equation_residual_certified
            and self.projection_constraints_certified
            and self.tail_certificate.is_nontrivial
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.index >= 0
            and self.start_state.certified
            and self.end_state.certified
            and self.parameter_interval.lower == 0.0
            and self.parameter_interval.upper > 0.0
            and np.isfinite(self.parameter_interval.upper)
            and self.physical_time_delta.lower <= self.physical_time_delta.upper
            and np.isfinite(self.physical_time_delta.lower)
            and np.isfinite(self.physical_time_delta.upper)
            and self.tail_certificate.is_nontrivial
            and self.equation_residual_certified
            and self.projection_constraints_certified
        )


@dataclass(frozen=True)
class SpatialKSSegmentedTailCertificate:
    retained_order: int
    guard_order: int
    requested_step_size: float
    segment_step_sizes: tuple[float, ...]
    segment_certificates: tuple[TailBoundCertificate, ...]
    final_state: IntervalSpatialKSBinaryChartState | None
    final_physical_time_delta: FloatInterval | None
    segment_propagation_steps: tuple[SpatialKSSegmentPropagationStep, ...] = ()

    @property
    def segment_count(self) -> int:
        return len(self.segment_certificates)

    @property
    def local_tail_bound(self) -> float:
        total = 0.0
        for certificate in self.segment_certificates:
            total = _round_up(total + float(certificate.tail_bound))
        return float(total)

    @property
    def max_step_tail_bound(self) -> float:
        return float(
            max((certificate.tail_bound for certificate in self.segment_certificates), default=np.inf)
        )

    @property
    def coefficient_source(self) -> str:
        return "spatial_ks_interval_segmented_guarded"

    @property
    def segment_handoffs_certified(self) -> bool:
        steps = self.segment_propagation_steps
        return bool(
            steps
            and all(step.certified for step in steps)
            and all(
                left.end_state is right.start_state
                for left, right in zip(steps, steps[1:])
            )
        )

    @property
    def final_state_matches_last_segment(self) -> bool:
        return bool(
            self.segment_propagation_steps
            and self.final_state is self.segment_propagation_steps[-1].end_state
        )

    @property
    def physical_time_accumulation_certified(self) -> bool:
        if self.final_physical_time_delta is None or not self.segment_propagation_steps:
            return False
        total = FloatInterval.point(0.0)
        for step in self.segment_propagation_steps:
            total = total + step.physical_time_delta
        return bool(
            total.lower == self.final_physical_time_delta.lower
            and total.upper == self.final_physical_time_delta.upper
        )

    @property
    def propagation_chain_certified(self) -> bool:
        return bool(
            len(self.segment_propagation_steps) == len(self.segment_certificates)
            and len(self.segment_step_sizes) == len(self.segment_propagation_steps)
            and self.segment_handoffs_certified
            and self.final_state_matches_last_segment
            and self.physical_time_accumulation_certified
            and all(
                step.tail_certificate is certificate
                for step, certificate in zip(
                    self.segment_propagation_steps,
                    self.segment_certificates,
                )
            )
            and all(
                step.step_size == step_size
                for step, step_size in zip(
                    self.segment_propagation_steps,
                    self.segment_step_sizes,
                )
            )
        )

    @property
    def is_nontrivial(self) -> bool:
        return bool(
            self.segment_certificates
            and self.final_state is not None
            and self.final_physical_time_delta is not None
            and len(self.segment_step_sizes) == len(self.segment_certificates)
            and all(certificate.is_nontrivial for certificate in self.segment_certificates)
            and np.isfinite(self.local_tail_bound)
            and np.isfinite(self.max_step_tail_bound)
            and self.local_tail_bound >= 0.0
            and self.max_step_tail_bound >= 0.0
            and self.final_physical_time_delta.lower <= self.final_physical_time_delta.upper
            and self.propagation_chain_certified
        )


@dataclass(frozen=True)
class OrdinaryCauchyMajorantCertificate:
    retained_order: int
    step_size: float
    time_radius: float
    position_radius: float
    velocity_radius: float
    min_pair_distance: float
    lower_squared_distance: float
    acceleration_bound: float
    state_sup_bound: float
    tail_bound: float

    @property
    def ratio_bound(self) -> float:
        return abs(self.step_size) / self.time_radius

    @property
    def coefficient_source(self) -> str:
        return "cauchy_majorant"

    @property
    def is_nontrivial(self) -> bool:
        return (
            np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
            and self.time_radius > 0.0
            and self.ratio_bound < 1.0
        )


@dataclass(frozen=True)
class OrdinaryIntervalCauchyMajorantCertificate:
    retained_order: int
    step_size: float
    time_radius: float
    position_radius: float
    velocity_radius: float
    min_pair_distance_lower_bound: float
    max_pair_distance_upper_bound: float
    lower_squared_distance: float
    acceleration_bound: float
    state_sup_bound: float
    tail_bound: float
    state_interval: tuple[tuple[float, float], ...]

    @property
    def ratio_bound(self) -> float:
        return abs(self.step_size) / self.time_radius

    @property
    def coefficient_source(self) -> str:
        return "interval_cauchy_majorant"

    @property
    def uses_interval_initial_state(self) -> bool:
        return True

    @property
    def is_nontrivial(self) -> bool:
        return (
            np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
            and self.time_radius > 0.0
            and self.ratio_bound < 1.0
        )


@dataclass(frozen=True)
class OrdinaryIntervalUnionCauchyMajorantCertificate:
    retained_order: int
    step_size: float
    member_certificates: tuple[OrdinaryIntervalCauchyMajorantCertificate, ...]

    @property
    def member_count(self) -> int:
        return len(self.member_certificates)

    @property
    def time_radius(self) -> float:
        return float(min(certificate.time_radius for certificate in self.member_certificates))

    @property
    def position_radius(self) -> float:
        return float(min(certificate.position_radius for certificate in self.member_certificates))

    @property
    def velocity_radius(self) -> float:
        return float(min(certificate.velocity_radius for certificate in self.member_certificates))

    @property
    def min_pair_distance_lower_bound(self) -> float:
        return float(min(certificate.min_pair_distance_lower_bound for certificate in self.member_certificates))

    @property
    def lower_squared_distance(self) -> float:
        return float(min(certificate.lower_squared_distance for certificate in self.member_certificates))

    @property
    def acceleration_bound(self) -> float:
        return float(max(certificate.acceleration_bound for certificate in self.member_certificates))

    @property
    def state_sup_bound(self) -> float:
        return float(max(certificate.state_sup_bound for certificate in self.member_certificates))

    @property
    def tail_bound(self) -> float:
        return float(max(certificate.tail_bound for certificate in self.member_certificates))

    @property
    def ratio_bound(self) -> float:
        return float(max(certificate.ratio_bound for certificate in self.member_certificates))

    @property
    def coefficient_source(self) -> str:
        return "interval_cauchy_majorant_union"

    @property
    def uses_interval_initial_state(self) -> bool:
        return True

    @property
    def is_nontrivial(self) -> bool:
        return bool(self.member_certificates) and all(
            certificate.is_nontrivial for certificate in self.member_certificates
        )


@dataclass(frozen=True)
class SundmanCauchyMajorantCertificate:
    retained_order: int
    step_size: float
    s_radius: float
    position_radius: float
    velocity_radius: float
    min_pair_distance: float
    lower_squared_distance: float
    upper_squared_distance: float
    acceleration_bound: float
    sundman_factor_bound: float
    state_sup_bound: float
    tail_bound: float
    distance_power: float

    @property
    def ratio_bound(self) -> float:
        return abs(self.step_size) / self.s_radius

    @property
    def coefficient_source(self) -> str:
        return "cauchy_majorant"

    @property
    def is_nontrivial(self) -> bool:
        return (
            np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
            and self.s_radius > 0.0
            and self.ratio_bound < 1.0
        )


@dataclass(frozen=True)
class SundmanIntervalCauchyMajorantCertificate:
    retained_order: int
    step_size: float
    s_radius: float
    position_radius: float
    velocity_radius: float
    min_pair_distance_lower_bound: float
    max_pair_distance_upper_bound: float
    lower_squared_distance: float
    upper_squared_distance: float
    acceleration_bound: float
    sundman_factor_bound: float
    state_sup_bound: float
    tail_bound: float
    distance_power: float

    @property
    def ratio_bound(self) -> float:
        return abs(self.step_size) / self.s_radius

    @property
    def coefficient_source(self) -> str:
        return "sundman_interval_cauchy_majorant"

    @property
    def uses_interval_initial_state(self) -> bool:
        return True

    @property
    def is_nontrivial(self) -> bool:
        return (
            np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
            and self.s_radius > 0.0
            and self.ratio_bound < 1.0
        )


@dataclass(frozen=True)
class RegularizedBinaryCauchyMajorantCertificate:
    retained_order: int
    step_size: float
    s_radius: float
    z_radius: float
    z_velocity_radius: float
    pair_energy_radius: float
    binary_center_radius: float
    binary_center_velocity_radius: float
    third_offset_radius: float
    third_offset_velocity_radius: float
    selected_pair_rho_bound: float
    third_distance_lower_bound: float
    perturbation_bound: float
    center_acceleration_bound: float
    third_offset_acceleration_bound: float
    state_sup_bound: float
    tail_bound: float

    @property
    def ratio_bound(self) -> float:
        return abs(self.step_size) / self.s_radius

    @property
    def coefficient_source(self) -> str:
        return "cauchy_majorant"

    @property
    def is_nontrivial(self) -> bool:
        return (
            np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
            and self.s_radius > 0.0
            and self.ratio_bound < 1.0
        )


@dataclass(frozen=True)
class RegularizedBinaryIntervalCauchyMajorantCertificate:
    retained_order: int
    step_size: float
    s_radius: float
    z_radius: float
    z_velocity_radius: float
    pair_energy_radius: float
    binary_center_radius: float
    binary_center_velocity_radius: float
    third_offset_radius: float
    third_offset_velocity_radius: float
    selected_pair_rho_bound: float
    third_distance_lower_bound: float
    perturbation_bound: float
    center_acceleration_bound: float
    third_offset_acceleration_bound: float
    state_sup_bound: float
    tail_bound: float

    @property
    def ratio_bound(self) -> float:
        return abs(self.step_size) / self.s_radius

    @property
    def coefficient_source(self) -> str:
        return "regularized_interval_cauchy_majorant"

    @property
    def uses_interval_initial_state(self) -> bool:
        return True

    @property
    def is_nontrivial(self) -> bool:
        return (
            np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
            and self.s_radius > 0.0
            and self.ratio_bound < 1.0
        )


@dataclass(frozen=True)
class RegularizedBinaryIntervalAtlasCauchyMajorantCertificate:
    retained_order: int
    step_size: float
    member_certificates: tuple[RegularizedBinaryIntervalCauchyMajorantCertificate, ...]

    @property
    def member_count(self) -> int:
        return len(self.member_certificates)

    @property
    def s_radius(self) -> float:
        return float(min(certificate.s_radius for certificate in self.member_certificates))

    @property
    def tail_bound(self) -> float:
        return float(max(certificate.tail_bound for certificate in self.member_certificates))

    @property
    def ratio_bound(self) -> float:
        return float(max(certificate.ratio_bound for certificate in self.member_certificates))

    @property
    def selected_pair_rho_bound(self) -> float:
        return float(max(certificate.selected_pair_rho_bound for certificate in self.member_certificates))

    @property
    def third_distance_lower_bound(self) -> float:
        return float(min(certificate.third_distance_lower_bound for certificate in self.member_certificates))

    @property
    def perturbation_bound(self) -> float:
        return float(max(certificate.perturbation_bound for certificate in self.member_certificates))

    @property
    def center_acceleration_bound(self) -> float:
        return float(max(certificate.center_acceleration_bound for certificate in self.member_certificates))

    @property
    def third_offset_acceleration_bound(self) -> float:
        return float(max(certificate.third_offset_acceleration_bound for certificate in self.member_certificates))

    @property
    def state_sup_bound(self) -> float:
        return float(max(certificate.state_sup_bound for certificate in self.member_certificates))

    @property
    def coefficient_source(self) -> str:
        return "regularized_interval_cauchy_majorant_atlas"

    @property
    def uses_interval_initial_state(self) -> bool:
        return True

    @property
    def is_nontrivial(self) -> bool:
        return bool(self.member_certificates) and all(
            certificate.is_nontrivial for certificate in self.member_certificates
        )


def _as_arrays(arrays: Iterable[Array]) -> list[Array]:
    normalized = [np.asarray(array, dtype=float) for array in arrays]
    if not normalized:
        raise ValueError("at least one coefficient array is required")
    degree_count = normalized[0].shape[0]
    if any(array.shape[0] != degree_count for array in normalized):
        raise ValueError("all coefficient arrays must have the same leading degree dimension")
    return normalized


def _as_interval(value: object) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


def _as_interval_arrays(arrays: Iterable[Array]) -> list[Array]:
    normalized = [np.asarray(array, dtype=object) for array in arrays]
    if not normalized:
        raise ValueError("at least one coefficient array is required")
    degree_count = normalized[0].shape[0]
    if any(array.shape[0] != degree_count for array in normalized):
        raise ValueError("all coefficient arrays must have the same leading degree dimension")

    out = []
    for array in normalized:
        interval_array = np.empty(array.shape, dtype=object)
        for index in np.ndindex(array.shape):
            interval_array[index] = _as_interval(array[index])
        out.append(interval_array)
    return out


def _round_up(value: float) -> float:
    return float(np.nextafter(float(value), np.inf))


def _round_down(value: float) -> float:
    return float(np.nextafter(float(value), -np.inf))


def _inflate_interval(value: FloatInterval, radius: float) -> FloatInterval:
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("inflation radius must be finite and nonnegative")
    return FloatInterval(
        _round_down(value.lower - radius),
        _round_up(value.upper + radius),
    )


def _inflate_interval_array(values: Array, radius: float) -> Array:
    values = np.asarray(values, dtype=object)
    out = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        out[index] = _inflate_interval(_as_interval(values[index]), radius)
    return out


def _interval_abs_bound(value: FloatInterval) -> float:
    return _round_up(max(abs(value.lower), abs(value.upper)))


def _validate_ordinary_data(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 2:
        raise ValueError("positions must have shape (body_count, dimension)")
    if positions.shape != velocities.shape or positions.shape[0] != 3:
        raise ValueError("positions and velocities must have matching three-body shapes")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    for i in range(3):
        for j in range(i + 1, 3):
            if np.linalg.norm(positions[j] - positions[i]) == 0.0:
                raise ValueError("initial data must be collision-free")
    return positions, velocities, masses


def _validate_planar_interval_state(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
) -> tuple[Array, Array, Array]:
    masses = np.asarray(masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    if len(state_interval) not in {12, 18}:
        raise ValueError("ordinary interval state must have length 12 or 18")
    dimension = len(state_interval) // 6

    normalized = []
    for lower, upper in state_interval:
        lower = float(lower)
        upper = float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("interval endpoints must be finite")
        if lower > upper:
            raise ValueError("interval lower endpoint cannot exceed upper endpoint")
        normalized.append(FloatInterval(lower, upper))
    values = np.array(normalized, dtype=object)
    coordinate_count = 3 * dimension
    return (
        values[:coordinate_count].reshape(3, dimension),
        values[coordinate_count:].reshape(3, dimension),
        masses,
    )


def _validate_interval_initial_arrays(
    positions: Array,
    velocities: Array,
    masses: Array,
) -> tuple[Array, Array, Array]:
    masses = np.asarray(masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    positions = np.asarray(positions, dtype=object)
    velocities = np.asarray(velocities, dtype=object)
    if positions.ndim != 2:
        raise ValueError("positions must have shape (body_count, dimension)")
    if positions.shape != velocities.shape or positions.shape[0] != 3:
        raise ValueError("positions and velocities must have matching three-body shapes")

    position_intervals = np.empty(positions.shape, dtype=object)
    velocity_intervals = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        position_intervals[index] = _as_interval(positions[index])
        velocity_intervals[index] = _as_interval(velocities[index])
    if min(bounds[0] for _pair, bounds in _interval_pairwise_distance_bounds_from_positions(position_intervals)) <= 0.0:
        raise ValueError("interval initial data must certify non-collision")
    return position_intervals, velocity_intervals, masses


def _interval_square_bounds(value: FloatInterval) -> tuple[float, float]:
    squares = (value.lower * value.lower, value.upper * value.upper)
    upper = _round_up(max(squares))
    if value.lower <= 0.0 <= value.upper:
        lower = 0.0
    else:
        lower = max(0.0, float(np.nextafter(min(squares), -np.inf)))
    return lower, upper


def _interval_pairwise_distance_bounds_from_positions(
    positions: Array,
) -> tuple[tuple[tuple[int, int], tuple[float, float]], ...]:
    positions = np.asarray(positions, dtype=object)
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("positions must have shape (3, dimension)")
    out = []
    for pair in ((0, 1), (0, 2), (1, 2)):
        lower_squared = 0.0
        upper_squared = 0.0
        for axis in range(positions.shape[1]):
            axis_lower, axis_upper = _interval_square_bounds(
                positions[pair[0], axis] - positions[pair[1], axis]
            )
            lower_squared = max(0.0, float(np.nextafter(lower_squared + axis_lower, -np.inf)))
            upper_squared = _round_up(upper_squared + axis_upper)
        lower_distance = (
            0.0
            if lower_squared <= 0.0
            else float(np.nextafter(np.sqrt(lower_squared), -np.inf))
        )
        upper_distance = _round_up(np.sqrt(max(upper_squared, 0.0)))
        out.append((pair, (lower_distance, upper_distance)))
    return tuple(out)


def _interval_vector_norm_upper_bound(vector: Array) -> float:
    upper_squared = 0.0
    for value in vector.reshape(-1):
        _lower, upper = _interval_square_bounds(value)
        upper_squared = _round_up(upper_squared + upper)
    return _round_up(np.sqrt(max(upper_squared, 0.0)))


def _interval_vector_norm_bounds(vector: Array) -> tuple[float, float]:
    lower_squared = 0.0
    upper_squared = 0.0
    for value in np.asarray(vector, dtype=object).reshape(-1):
        lower, upper = _interval_square_bounds(_as_interval(value))
        lower_squared = max(0.0, float(np.nextafter(lower_squared + lower, -np.inf)))
        upper_squared = _round_up(upper_squared + upper)
    lower = 0.0 if lower_squared <= 0.0 else float(np.nextafter(np.sqrt(lower_squared), -np.inf))
    upper = _round_up(np.sqrt(max(upper_squared, 0.0)))
    return lower, upper


def _interval_component_abs_bound(values: Array) -> float:
    bound = 0.0
    for value in values.reshape(-1):
        bound = max(bound, abs(value.lower), abs(value.upper))
    return _round_up(bound)


def _ordinary_interval_ball_bounds(
    positions: Array,
    masses: Array,
    position_radius: float,
) -> tuple[float, float, float, float, float]:
    if position_radius <= 0.0:
        raise ValueError("position_radius must be positive")

    distance_bounds = _interval_pairwise_distance_bounds_from_positions(positions)
    min_pair_distance_lower_bound = min(bounds[0] for _pair, bounds in distance_bounds)
    max_pair_distance_upper_bound = max(bounds[1] for _pair, bounds in distance_bounds)
    upper_squared_distance = 0.0
    if min_pair_distance_lower_bound <= 0.0:
        raise ValueError("interval state does not certify non-collision")

    lower_squared_distance = np.inf
    body_bounds = np.zeros(3, dtype=float)
    for (i, j), (distance_lower, distance_upper) in distance_bounds:
        pair_lower_squared = distance_lower**2 - 4.0 * distance_upper * position_radius - 4.0 * position_radius**2
        if pair_lower_squared <= 0.0:
            raise ValueError("position ball does not certify separation from complex collision")
        pair_upper_squared = distance_upper**2 + 4.0 * distance_upper * position_radius + 4.0 * position_radius**2
        lower_squared_distance = min(lower_squared_distance, pair_lower_squared)
        upper_squared_distance = max(upper_squared_distance, pair_upper_squared)
        numerator_bound = distance_upper + 2.0 * position_radius
        inverse_cube_bound = pair_lower_squared ** -1.5
        body_bounds[i] += masses[j] * numerator_bound * inverse_cube_bound
        body_bounds[j] += masses[i] * numerator_bound * inverse_cube_bound
    return (
        float(min_pair_distance_lower_bound),
        float(max_pair_distance_upper_bound),
        float(lower_squared_distance),
        float(upper_squared_distance),
        float(np.max(body_bounds)),
    )


def _default_interval_position_radius(positions: Array) -> float:
    distance_bounds = _interval_pairwise_distance_bounds_from_positions(positions)
    min_pair_distance_lower_bound = min(bounds[0] for _pair, bounds in distance_bounds)
    max_pair_distance_upper_bound = max(bounds[1] for _pair, bounds in distance_bounds)
    if min_pair_distance_lower_bound <= 0.0:
        raise ValueError("interval state does not certify non-collision")

    radius = min_pair_distance_lower_bound / 10.0
    while radius > np.finfo(float).tiny:
        if min_pair_distance_lower_bound**2 - 4.0 * max_pair_distance_upper_bound * radius - 4.0 * radius**2 > 0.0:
            return float(radius)
        radius *= 0.5
    raise ValueError("interval state is too wide to choose a certifying position radius")


def _ordinary_ball_acceleration_bound(
    positions: Array,
    masses: Array,
    position_radius: float,
) -> tuple[float, float, float]:
    if position_radius <= 0.0:
        raise ValueError("position_radius must be positive")

    min_pair_distance = np.inf
    lower_squared_distance = np.inf
    body_bounds = np.zeros(3, dtype=float)
    for i in range(3):
        for j in range(i + 1, 3):
            pair_distance = float(np.linalg.norm(positions[j] - positions[i]))
            min_pair_distance = min(min_pair_distance, pair_distance)
            pair_lower_squared = pair_distance**2 - 4.0 * pair_distance * position_radius - 4.0 * position_radius**2
            if pair_lower_squared <= 0.0:
                raise ValueError("position ball does not certify separation from complex collision")
            lower_squared_distance = min(lower_squared_distance, pair_lower_squared)
            numerator_bound = pair_distance + 2.0 * position_radius
            inverse_cube_bound = pair_lower_squared ** -1.5
            body_bounds[i] += masses[j] * numerator_bound * inverse_cube_bound
            body_bounds[j] += masses[i] * numerator_bound * inverse_cube_bound
    return float(min_pair_distance), float(lower_squared_distance), float(np.max(body_bounds))


def _ordinary_ball_bounds(
    positions: Array,
    masses: Array,
    position_radius: float,
) -> tuple[float, float, float, float]:
    if position_radius <= 0.0:
        raise ValueError("position_radius must be positive")

    min_pair_distance = np.inf
    lower_squared_distance = np.inf
    upper_squared_distance = 0.0
    body_bounds = np.zeros(3, dtype=float)
    for i in range(3):
        for j in range(i + 1, 3):
            pair_distance = float(np.linalg.norm(positions[j] - positions[i]))
            min_pair_distance = min(min_pair_distance, pair_distance)
            pair_lower_squared = pair_distance**2 - 4.0 * pair_distance * position_radius - 4.0 * position_radius**2
            if pair_lower_squared <= 0.0:
                raise ValueError("position ball does not certify separation from complex collision")
            pair_upper_squared = pair_distance**2 + 4.0 * pair_distance * position_radius + 4.0 * position_radius**2
            lower_squared_distance = min(lower_squared_distance, pair_lower_squared)
            upper_squared_distance = max(upper_squared_distance, pair_upper_squared)
            numerator_bound = pair_distance + 2.0 * position_radius
            inverse_cube_bound = pair_lower_squared ** -1.5
            body_bounds[i] += masses[j] * numerator_bound * inverse_cube_bound
            body_bounds[j] += masses[i] * numerator_bound * inverse_cube_bound
    return (
        float(min_pair_distance),
        float(lower_squared_distance),
        float(upper_squared_distance),
        float(np.max(body_bounds)),
    )


def _lc_square_point(z: Array) -> Array:
    z = np.asarray(z, dtype=float)
    return np.array([z[0] * z[0] - z[1] * z[1], 2.0 * z[0] * z[1]], dtype=float)


def _radius_over_bound(radius: float, bound: float) -> float:
    if radius <= 0.0:
        raise ValueError("majorant radii must be positive")
    if bound <= 0.0:
        return float("inf")
    return float(radius / bound)


def _regularized_binary_third_body_bounds(
    initial_state: RegularizedBinaryCollisionChartState,
    *,
    z_radius: float,
    third_offset_radius: float,
) -> tuple[float, float, float, float]:
    masses = np.asarray(initial_state.masses, dtype=float)
    first, second = initial_state.pair
    third = initial_state.third_index
    pair_mass = masses[first] + masses[second]
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    relative_position = _lc_square_point(initial_state.z)
    z_norm = float(np.linalg.norm(initial_state.z))
    relative_variation_bound = 2.0 * z_norm * z_radius + z_radius**2

    from_first = np.asarray(initial_state.third_offset, dtype=float) + alpha * relative_position
    from_second = np.asarray(initial_state.third_offset, dtype=float) - beta * relative_position
    lower_first = float(np.linalg.norm(from_first)) - third_offset_radius - alpha * relative_variation_bound
    lower_second = float(np.linalg.norm(from_second)) - third_offset_radius - beta * relative_variation_bound
    if lower_first <= 0.0 or lower_second <= 0.0:
        raise ValueError("majorant ball does not certify third-body separation in the binary chart")

    upper_first = float(np.linalg.norm(from_first)) + third_offset_radius + alpha * relative_variation_bound
    upper_second = float(np.linalg.norm(from_second)) + third_offset_radius + beta * relative_variation_bound
    field_first = upper_first / lower_first**3
    field_second = upper_second / lower_second**3
    center_acceleration_bound = (
        masses[third] / pair_mass * (masses[first] * field_first + masses[second] * field_second)
    )
    third_acceleration_bound = masses[first] * field_first + masses[second] * field_second
    third_offset_acceleration_bound = third_acceleration_bound + center_acceleration_bound
    perturbation_bound = masses[third] * (field_first + field_second)
    return (
        float(min(lower_first, lower_second)),
        float(perturbation_bound),
        float(center_acceleration_bound),
        float(third_offset_acceleration_bound),
    )


def _interval_lc_square_bounds(z: Array) -> Array:
    x = _as_interval(z[0])
    y = _as_interval(z[1])
    return np.array(
        [
            x * x - y * y,
            (x * y).scale(2.0),
        ],
        dtype=object,
    )


def _regularized_binary_interval_third_body_bounds(
    initial_state: IntervalRegularizedBinaryCollisionChartState,
    *,
    z_radius: float,
    third_offset_radius: float,
) -> tuple[float, float, float, float]:
    masses = np.asarray(initial_state.masses, dtype=float)
    first, second = initial_state.pair
    third = initial_state.third_index
    pair_mass = masses[first] + masses[second]
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    relative_position = _interval_lc_square_bounds(initial_state.z)
    z_norm_upper = _interval_vector_norm_upper_bound(initial_state.z)
    relative_variation_bound = 2.0 * z_norm_upper * z_radius + z_radius**2

    from_first = np.asarray(initial_state.third_offset, dtype=object) + np.array(
        [component.scale(alpha) for component in relative_position],
        dtype=object,
    )
    from_second = np.asarray(initial_state.third_offset, dtype=object) - np.array(
        [component.scale(beta) for component in relative_position],
        dtype=object,
    )
    first_lower, first_upper = _interval_vector_norm_bounds(from_first)
    second_lower, second_upper = _interval_vector_norm_bounds(from_second)
    lower_first = first_lower - third_offset_radius - alpha * relative_variation_bound
    lower_second = second_lower - third_offset_radius - beta * relative_variation_bound
    if lower_first <= 0.0 or lower_second <= 0.0:
        raise ValueError("majorant interval ball does not certify third-body separation in the binary chart")

    upper_first = first_upper + third_offset_radius + alpha * relative_variation_bound
    upper_second = second_upper + third_offset_radius + beta * relative_variation_bound
    field_first = upper_first / lower_first**3
    field_second = upper_second / lower_second**3
    center_acceleration_bound = (
        masses[third] / pair_mass * (masses[first] * field_first + masses[second] * field_second)
    )
    third_acceleration_bound = masses[first] * field_first + masses[second] * field_second
    third_offset_acceleration_bound = third_acceleration_bound + center_acceleration_bound
    perturbation_bound = masses[third] * (field_first + field_second)
    return (
        float(min(lower_first, lower_second)),
        float(perturbation_bound),
        float(center_acceleration_bound),
        float(third_offset_acceleration_bound),
    )


def coefficient_term_norms(arrays: Iterable[Array], step_size: float) -> Array:
    """Return infinity norms of each Taylor term evaluated at `step_size`."""

    normalized = _as_arrays(arrays)
    terms = np.zeros(normalized[0].shape[0], dtype=float)
    magnitude = abs(step_size)
    for degree in range(normalized[0].shape[0]):
        scale = magnitude**degree
        terms[degree] = max(np.linalg.norm(array[degree].reshape(-1), ord=np.inf) * scale for array in normalized)
    return terms


def interval_coefficient_term_bounds(arrays: Iterable[Array], step_size: float) -> Array:
    """Return interval upper bounds for each evaluated Taylor term."""

    normalized = _as_interval_arrays(arrays)
    terms = np.zeros(normalized[0].shape[0], dtype=float)
    magnitude = abs(step_size)
    for degree in range(normalized[0].shape[0]):
        scale = _round_up(magnitude**degree)
        term_bound = 0.0
        for array in normalized:
            component_bounds = [
                _round_up(_interval_abs_bound(_as_interval(value)) * scale)
                for value in array[degree].reshape(-1)
            ]
            if component_bounds:
                term_bound = max(term_bound, max(component_bounds))
        terms[degree] = _round_up(term_bound)
    return terms


def guarded_tail_certificate(
    arrays: Iterable[Array],
    retained_order: int,
    step_size: float,
) -> TailBoundCertificate:
    """Build a numerical geometric tail certificate from computed guard terms."""

    normalized = _as_arrays(arrays)
    computed_order = normalized[0].shape[0] - 1
    if retained_order < 0 or retained_order >= computed_order:
        raise ValueError("retained_order must leave at least one guard coefficient")

    terms = coefficient_term_norms(normalized, step_size)
    omitted = terms[retained_order + 1 :]
    first = float(omitted[0])
    observed_tail = float(np.sum(omitted))
    if first == 0.0 and np.all(omitted == 0.0):
        ratio_bound = 0.0
        tail_bound = 0.0
    else:
        ratios = []
        for left, right in zip(omitted[:-1], omitted[1:]):
            if left > 0.0:
                ratios.append(float(right / left))
            elif right > 0.0:
                ratios.append(np.inf)
        ratio_bound = float(max(ratios)) if ratios else np.inf
        if np.isfinite(ratio_bound) and ratio_bound < 1.0:
            tail_bound = max(observed_tail, first / (1.0 - ratio_bound))
        else:
            tail_bound = np.inf

    return TailBoundCertificate(
        retained_order=retained_order,
        computed_order=computed_order,
        step_size=float(step_size),
        first_omitted_term=first,
        observed_tail=observed_tail,
        ratio_bound=ratio_bound,
        tail_bound=float(tail_bound),
    )


def interval_guarded_tail_certificate(
    arrays: Iterable[Array],
    retained_order: int,
    step_size: float,
) -> TailBoundCertificate:
    """Build a guarded certificate from interval-enclosed coefficients."""

    normalized = _as_interval_arrays(arrays)
    computed_order = normalized[0].shape[0] - 1
    if retained_order < 0 or retained_order >= computed_order:
        raise ValueError("retained_order must leave at least one guard coefficient")

    terms = interval_coefficient_term_bounds(normalized, step_size)
    omitted = terms[retained_order + 1 :]
    first = float(omitted[0])
    observed_tail = 0.0
    for term in omitted:
        observed_tail = _round_up(observed_tail + float(term))
    if first == 0.0 and np.all(omitted == 0.0):
        ratio_bound = 0.0
        tail_bound = 0.0
    else:
        ratios = []
        for left, right in zip(omitted[:-1], omitted[1:]):
            if left > 0.0:
                ratios.append(_round_up(float(right / left)))
            elif right > 0.0:
                ratios.append(np.inf)
        ratio_bound = float(max(ratios)) if ratios else np.inf
        if np.isfinite(ratio_bound) and ratio_bound < 1.0:
            tail_bound = max(observed_tail, _round_up(first / (1.0 - ratio_bound)))
        else:
            tail_bound = np.inf

    return TailBoundCertificate(
        retained_order=retained_order,
        computed_order=computed_order,
        step_size=float(step_size),
        first_omitted_term=first,
        observed_tail=observed_tail,
        ratio_bound=ratio_bound,
        tail_bound=float(tail_bound),
        coefficient_source="interval",
    )


def ordinary_cauchy_majorant_tail_certificate(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    retained_order: int,
    step_size: float,
    position_radius: float | None = None,
    velocity_radius: float | None = None,
    time_radius: float | None = None,
) -> OrdinaryCauchyMajorantCertificate:
    """Certify an a priori Cauchy tail bound for an ordinary Taylor chart.

    The certificate uses a simple complex-time self-map.  If each body remains
    within `position_radius` of its initial position, the lower bound on
    `|r_ij dot r_ij|` keeps the inverse-cube force analytic.  The chosen time
    disk must map the position and velocity balls into themselves; Cauchy's
    estimate then gives a geometric bound for every omitted state coefficient.
    """

    if retained_order < 0:
        raise ValueError("retained_order cannot be negative")
    positions, velocities, masses = _validate_ordinary_data(positions, velocities, masses)
    min_pair_distance = min(
        float(np.linalg.norm(positions[j] - positions[i]))
        for i in range(3)
        for j in range(i + 1, 3)
    )
    if position_radius is None:
        position_radius = min_pair_distance / 10.0
    position_radius = float(position_radius)

    min_pair_distance, lower_squared_distance, acceleration_bound = _ordinary_ball_acceleration_bound(
        positions,
        masses,
        position_radius,
    )
    max_velocity = float(max(np.linalg.norm(velocity) for velocity in velocities))
    if velocity_radius is None:
        velocity_radius = float(np.sqrt(max(acceleration_bound * position_radius, np.finfo(float).tiny)))
    velocity_radius = float(velocity_radius)
    if velocity_radius <= 0.0:
        raise ValueError("velocity_radius must be positive")

    position_time_radius = position_radius / (max_velocity + velocity_radius)
    velocity_time_radius = velocity_radius / acceleration_bound
    certified_time_radius = min(position_time_radius, velocity_time_radius)
    if time_radius is None:
        time_radius = float(np.nextafter(certified_time_radius, 0.0))
    time_radius = float(time_radius)
    if time_radius <= 0.0:
        raise ValueError("time_radius must be positive")

    position_image_radius = time_radius * (max_velocity + velocity_radius)
    velocity_image_radius = time_radius * acceleration_bound
    if position_image_radius > position_radius or velocity_image_radius > velocity_radius:
        raise ValueError("time_radius does not map the majorant ball into itself")
    ratio = abs(float(step_size)) / time_radius
    if ratio >= 1.0:
        raise ValueError("step_size must lie inside the certified time radius")

    position_sup = float(np.max(np.abs(positions)) + position_radius)
    velocity_sup = float(np.max(np.abs(velocities)) + velocity_radius)
    state_sup_bound = max(position_sup, velocity_sup)
    tail_bound = _round_up(state_sup_bound * ratio ** (retained_order + 1) / (1.0 - ratio))

    return OrdinaryCauchyMajorantCertificate(
        retained_order=int(retained_order),
        step_size=float(step_size),
        time_radius=float(time_radius),
        position_radius=float(position_radius),
        velocity_radius=float(velocity_radius),
        min_pair_distance=float(min_pair_distance),
        lower_squared_distance=float(lower_squared_distance),
        acceleration_bound=float(acceleration_bound),
        state_sup_bound=float(state_sup_bound),
        tail_bound=float(tail_bound),
    )


def ordinary_interval_cauchy_majorant_tail_certificate(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    *,
    retained_order: int,
    step_size: float,
    position_radius: float | None = None,
    velocity_radius: float | None = None,
    time_radius: float | None = None,
) -> OrdinaryIntervalCauchyMajorantCertificate:
    """Certify a Cauchy tail bound for every ordinary chart in an interval box.

    The box encloses possible initial positions and velocities, in either
    planar or spatial coordinates.  The certificate chooses one complex-time
    majorant ball that works for every point state in the box, using interval
    lower and upper pair-distance bounds in place of point distances.
    """

    if retained_order < 0:
        raise ValueError("retained_order cannot be negative")
    positions, velocities, masses = _validate_planar_interval_state(state_interval, masses)
    state_interval = tuple((float(lower), float(upper)) for lower, upper in state_interval)
    if position_radius is None:
        position_radius = _default_interval_position_radius(positions)
    position_radius = float(position_radius)

    (
        min_pair_distance_lower_bound,
        max_pair_distance_upper_bound,
        lower_squared_distance,
        _upper_squared_distance,
        acceleration_bound,
    ) = _ordinary_interval_ball_bounds(
        positions,
        masses,
        position_radius,
    )
    max_velocity = max(_interval_vector_norm_upper_bound(velocities[body]) for body in range(3))
    if velocity_radius is None:
        velocity_radius = float(np.sqrt(max(acceleration_bound * position_radius, np.finfo(float).tiny)))
    velocity_radius = float(velocity_radius)
    if velocity_radius <= 0.0:
        raise ValueError("velocity_radius must be positive")

    position_time_radius = position_radius / (max_velocity + velocity_radius)
    velocity_time_radius = velocity_radius / acceleration_bound
    certified_time_radius = min(position_time_radius, velocity_time_radius)
    if time_radius is None:
        time_radius = float(np.nextafter(certified_time_radius, 0.0))
    time_radius = float(time_radius)
    if time_radius <= 0.0:
        raise ValueError("time_radius must be positive")

    position_image_radius = time_radius * (max_velocity + velocity_radius)
    velocity_image_radius = time_radius * acceleration_bound
    if position_image_radius > position_radius or velocity_image_radius > velocity_radius:
        raise ValueError("time_radius does not map the interval majorant ball into itself")
    ratio = abs(float(step_size)) / time_radius
    if ratio >= 1.0:
        raise ValueError("step_size must lie inside the certified time radius")

    position_sup = _interval_component_abs_bound(positions) + position_radius
    velocity_sup = _interval_component_abs_bound(velocities) + velocity_radius
    state_sup_bound = max(position_sup, velocity_sup)
    tail_bound = _round_up(state_sup_bound * ratio ** (retained_order + 1) / (1.0 - ratio))

    return OrdinaryIntervalCauchyMajorantCertificate(
        retained_order=int(retained_order),
        step_size=float(step_size),
        time_radius=float(time_radius),
        position_radius=float(position_radius),
        velocity_radius=float(velocity_radius),
        min_pair_distance_lower_bound=float(min_pair_distance_lower_bound),
        max_pair_distance_upper_bound=float(max_pair_distance_upper_bound),
        lower_squared_distance=float(lower_squared_distance),
        acceleration_bound=float(acceleration_bound),
        state_sup_bound=float(state_sup_bound),
        tail_bound=float(tail_bound),
        state_interval=state_interval,
    )


def ordinary_interval_union_cauchy_majorant_tail_certificate(
    state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
    masses: Array,
    *,
    retained_order: int,
    step_size: float,
    position_radius: float | None = None,
    velocity_radius: float | None = None,
    time_radius: float | None = None,
) -> OrdinaryIntervalUnionCauchyMajorantCertificate:
    """Certify an ordinary Cauchy tail bound over an interval-state union."""

    if not state_interval_union:
        raise ValueError("state_interval_union cannot be empty")
    member_certificates = tuple(
        ordinary_interval_cauchy_majorant_tail_certificate(
            state_interval,
            masses,
            retained_order=retained_order,
            step_size=step_size,
            position_radius=position_radius,
            velocity_radius=velocity_radius,
            time_radius=time_radius,
        )
        for state_interval in state_interval_union
    )
    return OrdinaryIntervalUnionCauchyMajorantCertificate(
        retained_order=int(retained_order),
        step_size=float(step_size),
        member_certificates=member_certificates,
    )


def sundman_cauchy_majorant_tail_certificate(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    retained_order: int,
    step_size: float,
    distance_power: float = 1.0,
    position_radius: float | None = None,
    velocity_radius: float | None = None,
    s_radius: float | None = None,
) -> SundmanCauchyMajorantCertificate:
    """Certify an a priori Cauchy tail bound for a Sundman-time chart."""

    if retained_order < 0:
        raise ValueError("retained_order cannot be negative")
    if distance_power <= 0.0:
        raise ValueError("distance_power must be positive")
    positions, velocities, masses = _validate_ordinary_data(positions, velocities, masses)
    min_pair_distance = min(
        float(np.linalg.norm(positions[j] - positions[i]))
        for i in range(3)
        for j in range(i + 1, 3)
    )
    if position_radius is None:
        position_radius = min_pair_distance / 10.0
    position_radius = float(position_radius)

    min_pair_distance, lower_squared_distance, upper_squared_distance, acceleration_bound = _ordinary_ball_bounds(
        positions,
        masses,
        position_radius,
    )
    sundman_factor_bound = float(upper_squared_distance ** (1.5 * distance_power))
    max_velocity = float(max(np.linalg.norm(velocity) for velocity in velocities))
    if velocity_radius is None:
        velocity_radius = float(np.sqrt(max(acceleration_bound * position_radius, np.finfo(float).tiny)))
    velocity_radius = float(velocity_radius)
    if velocity_radius <= 0.0:
        raise ValueError("velocity_radius must be positive")

    position_s_radius = position_radius / (sundman_factor_bound * (max_velocity + velocity_radius))
    velocity_s_radius = velocity_radius / (sundman_factor_bound * acceleration_bound)
    certified_s_radius = min(position_s_radius, velocity_s_radius)
    if s_radius is None:
        s_radius = float(np.nextafter(certified_s_radius, 0.0))
    s_radius = float(s_radius)
    if s_radius <= 0.0:
        raise ValueError("s_radius must be positive")

    position_image_radius = s_radius * sundman_factor_bound * (max_velocity + velocity_radius)
    velocity_image_radius = s_radius * sundman_factor_bound * acceleration_bound
    if position_image_radius > position_radius or velocity_image_radius > velocity_radius:
        raise ValueError("s_radius does not map the Sundman majorant ball into itself")
    ratio = abs(float(step_size)) / s_radius
    if ratio >= 1.0:
        raise ValueError("step_size must lie inside the certified s radius")

    position_sup = float(np.max(np.abs(positions)) + position_radius)
    velocity_sup = float(np.max(np.abs(velocities)) + velocity_radius)
    physical_time_sup = s_radius * sundman_factor_bound
    state_sup_bound = max(position_sup, velocity_sup, physical_time_sup)
    tail_bound = _round_up(state_sup_bound * ratio ** (retained_order + 1) / (1.0 - ratio))

    return SundmanCauchyMajorantCertificate(
        retained_order=int(retained_order),
        step_size=float(step_size),
        s_radius=float(s_radius),
        position_radius=float(position_radius),
        velocity_radius=float(velocity_radius),
        min_pair_distance=float(min_pair_distance),
        lower_squared_distance=float(lower_squared_distance),
        upper_squared_distance=float(upper_squared_distance),
        acceleration_bound=float(acceleration_bound),
        sundman_factor_bound=float(sundman_factor_bound),
        state_sup_bound=float(state_sup_bound),
        tail_bound=float(tail_bound),
        distance_power=float(distance_power),
    )


def sundman_interval_cauchy_majorant_tail_certificate(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    retained_order: int,
    step_size: float,
    distance_power: float = 1.0,
    position_radius: float | None = None,
    velocity_radius: float | None = None,
    s_radius: float | None = None,
) -> SundmanIntervalCauchyMajorantCertificate:
    """Certify a Sundman-time Cauchy tail bound for an interval initial box."""

    if retained_order < 0:
        raise ValueError("retained_order cannot be negative")
    if distance_power <= 0.0:
        raise ValueError("distance_power must be positive")
    positions, velocities, masses = _validate_interval_initial_arrays(positions, velocities, masses)
    if position_radius is None:
        position_radius = _default_interval_position_radius(positions)
    position_radius = float(position_radius)

    (
        min_pair_distance_lower_bound,
        max_pair_distance_upper_bound,
        lower_squared_distance,
        upper_squared_distance,
        acceleration_bound,
    ) = _ordinary_interval_ball_bounds(
        positions,
        masses,
        position_radius,
    )
    sundman_factor_bound = float(upper_squared_distance ** (1.5 * distance_power))
    max_velocity = max(_interval_vector_norm_upper_bound(velocities[body]) for body in range(3))
    if velocity_radius is None:
        velocity_radius = float(np.sqrt(max(acceleration_bound * position_radius, np.finfo(float).tiny)))
    velocity_radius = float(velocity_radius)
    if velocity_radius <= 0.0:
        raise ValueError("velocity_radius must be positive")

    position_s_radius = position_radius / (sundman_factor_bound * (max_velocity + velocity_radius))
    velocity_s_radius = velocity_radius / (sundman_factor_bound * acceleration_bound)
    certified_s_radius = min(position_s_radius, velocity_s_radius)
    if s_radius is None:
        s_radius = float(np.nextafter(certified_s_radius, 0.0))
    s_radius = float(s_radius)
    if s_radius <= 0.0:
        raise ValueError("s_radius must be positive")

    position_image_radius = s_radius * sundman_factor_bound * (max_velocity + velocity_radius)
    velocity_image_radius = s_radius * sundman_factor_bound * acceleration_bound
    if position_image_radius > position_radius or velocity_image_radius > velocity_radius:
        raise ValueError("s_radius does not map the interval Sundman majorant ball into itself")
    ratio = abs(float(step_size)) / s_radius
    if ratio >= 1.0:
        raise ValueError("step_size must lie inside the certified s radius")

    position_sup = _interval_component_abs_bound(positions) + position_radius
    velocity_sup = _interval_component_abs_bound(velocities) + velocity_radius
    physical_time_sup = s_radius * sundman_factor_bound
    state_sup_bound = max(position_sup, velocity_sup, physical_time_sup)
    tail_bound = _round_up(state_sup_bound * ratio ** (retained_order + 1) / (1.0 - ratio))

    return SundmanIntervalCauchyMajorantCertificate(
        retained_order=int(retained_order),
        step_size=float(step_size),
        s_radius=float(s_radius),
        position_radius=float(position_radius),
        velocity_radius=float(velocity_radius),
        min_pair_distance_lower_bound=float(min_pair_distance_lower_bound),
        max_pair_distance_upper_bound=float(max_pair_distance_upper_bound),
        lower_squared_distance=float(lower_squared_distance),
        upper_squared_distance=float(upper_squared_distance),
        acceleration_bound=float(acceleration_bound),
        sundman_factor_bound=float(sundman_factor_bound),
        state_sup_bound=float(state_sup_bound),
        tail_bound=float(tail_bound),
        distance_power=float(distance_power),
    )


def regularized_binary_cauchy_majorant_tail_certificate(
    initial_state: RegularizedBinaryCollisionChartState,
    *,
    retained_order: int,
    step_size: float,
    z_radius: float | None = None,
    z_velocity_radius: float | None = None,
    pair_energy_radius: float | None = None,
    binary_center_radius: float | None = None,
    binary_center_velocity_radius: float | None = None,
    third_offset_radius: float | None = None,
    third_offset_velocity_radius: float | None = None,
    s_radius: float | None = None,
) -> RegularizedBinaryCauchyMajorantCertificate:
    """Certify an a priori Cauchy tail bound for a regularized binary chart."""

    if retained_order < 0:
        raise ValueError("retained_order cannot be negative")
    masses = np.asarray(initial_state.masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")

    z_norm = float(np.linalg.norm(initial_state.z))
    z_velocity_norm = float(np.linalg.norm(initial_state.z_velocity))
    center_norm = float(np.linalg.norm(initial_state.binary_center))
    center_velocity_norm = float(np.linalg.norm(initial_state.binary_center_velocity))
    third_offset_norm = float(np.linalg.norm(initial_state.third_offset))
    third_offset_velocity_norm = float(np.linalg.norm(initial_state.third_offset_velocity))
    if z_radius is None:
        z_radius = 0.05 * max(1.0, z_norm)
    if z_velocity_radius is None:
        z_velocity_radius = 0.05 * max(1.0, z_velocity_norm)
    if pair_energy_radius is None:
        pair_energy_radius = 0.05 * max(1.0, abs(float(initial_state.pair_energy)))
    if binary_center_radius is None:
        binary_center_radius = 0.05 * max(1.0, center_norm)
    if binary_center_velocity_radius is None:
        binary_center_velocity_radius = 0.05 * max(1.0, center_velocity_norm)
    if third_offset_radius is None:
        third_offset_radius = 0.05 * max(1.0, third_offset_norm)
    if third_offset_velocity_radius is None:
        third_offset_velocity_radius = 0.05 * max(1.0, third_offset_velocity_norm)

    z_radius = float(z_radius)
    z_velocity_radius = float(z_velocity_radius)
    pair_energy_radius = float(pair_energy_radius)
    binary_center_radius = float(binary_center_radius)
    binary_center_velocity_radius = float(binary_center_velocity_radius)
    third_offset_radius = float(third_offset_radius)
    third_offset_velocity_radius = float(third_offset_velocity_radius)
    radii = (
        z_radius,
        z_velocity_radius,
        pair_energy_radius,
        binary_center_radius,
        binary_center_velocity_radius,
        third_offset_radius,
        third_offset_velocity_radius,
    )
    if any(radius <= 0.0 for radius in radii):
        raise ValueError("majorant radii must be positive")

    third_distance_lower, perturbation_bound, center_acceleration_bound, third_offset_acceleration_bound = (
        _regularized_binary_third_body_bounds(
            initial_state,
            z_radius=z_radius,
            third_offset_radius=third_offset_radius,
        )
    )
    z_bound = z_norm + z_radius
    z_velocity_bound = z_velocity_norm + z_velocity_radius
    pair_energy_bound = abs(float(initial_state.pair_energy)) + pair_energy_radius
    center_velocity_bound = center_velocity_norm + binary_center_velocity_radius
    third_offset_velocity_bound = third_offset_velocity_norm + third_offset_velocity_radius
    rho_bound = z_bound**2

    z_rhs_bound = z_velocity_bound
    z_velocity_rhs_bound = (
        0.5 * pair_energy_bound * z_bound
        + 0.5 * rho_bound * z_bound * perturbation_bound
    )
    pair_energy_rhs_bound = 2.0 * z_bound * z_velocity_bound * perturbation_bound
    binary_center_rhs_bound = rho_bound * center_velocity_bound
    binary_center_velocity_rhs_bound = rho_bound * center_acceleration_bound
    third_offset_rhs_bound = rho_bound * third_offset_velocity_bound
    third_offset_velocity_rhs_bound = rho_bound * third_offset_acceleration_bound

    certified_s_radius = min(
        _radius_over_bound(z_radius, z_rhs_bound),
        _radius_over_bound(z_velocity_radius, z_velocity_rhs_bound),
        _radius_over_bound(pair_energy_radius, pair_energy_rhs_bound),
        _radius_over_bound(binary_center_radius, binary_center_rhs_bound),
        _radius_over_bound(binary_center_velocity_radius, binary_center_velocity_rhs_bound),
        _radius_over_bound(third_offset_radius, third_offset_rhs_bound),
        _radius_over_bound(third_offset_velocity_radius, third_offset_velocity_rhs_bound),
    )
    if not np.isfinite(certified_s_radius):
        certified_s_radius = 1.0
    if s_radius is None:
        s_radius = float(np.nextafter(certified_s_radius, 0.0))
    s_radius = float(s_radius)
    if s_radius <= 0.0:
        raise ValueError("s_radius must be positive")

    if (
        s_radius * z_rhs_bound > z_radius
        or s_radius * z_velocity_rhs_bound > z_velocity_radius
        or s_radius * pair_energy_rhs_bound > pair_energy_radius
        or s_radius * binary_center_rhs_bound > binary_center_radius
        or s_radius * binary_center_velocity_rhs_bound > binary_center_velocity_radius
        or s_radius * third_offset_rhs_bound > third_offset_radius
        or s_radius * third_offset_velocity_rhs_bound > third_offset_velocity_radius
    ):
        raise ValueError("s_radius does not map the regularized binary majorant ball into itself")
    ratio = abs(float(step_size)) / s_radius
    if ratio >= 1.0:
        raise ValueError("step_size must lie inside the certified s radius")

    state_sup_bound = max(
        z_bound,
        z_velocity_bound,
        pair_energy_bound,
        center_norm + binary_center_radius,
        center_velocity_bound,
        third_offset_norm + third_offset_radius,
        third_offset_velocity_bound,
        s_radius * rho_bound,
    )
    tail_bound = _round_up(state_sup_bound * ratio ** (retained_order + 1) / (1.0 - ratio))

    return RegularizedBinaryCauchyMajorantCertificate(
        retained_order=int(retained_order),
        step_size=float(step_size),
        s_radius=float(s_radius),
        z_radius=float(z_radius),
        z_velocity_radius=float(z_velocity_radius),
        pair_energy_radius=float(pair_energy_radius),
        binary_center_radius=float(binary_center_radius),
        binary_center_velocity_radius=float(binary_center_velocity_radius),
        third_offset_radius=float(third_offset_radius),
        third_offset_velocity_radius=float(third_offset_velocity_radius),
        selected_pair_rho_bound=float(rho_bound),
        third_distance_lower_bound=float(third_distance_lower),
        perturbation_bound=float(perturbation_bound),
        center_acceleration_bound=float(center_acceleration_bound),
        third_offset_acceleration_bound=float(third_offset_acceleration_bound),
        state_sup_bound=float(state_sup_bound),
        tail_bound=float(tail_bound),
    )


def regularized_binary_interval_cauchy_majorant_tail_certificate(
    initial_state: IntervalRegularizedBinaryCollisionChartState,
    *,
    retained_order: int,
    step_size: float,
    z_radius: float | None = None,
    z_velocity_radius: float | None = None,
    pair_energy_radius: float | None = None,
    binary_center_radius: float | None = None,
    binary_center_velocity_radius: float | None = None,
    third_offset_radius: float | None = None,
    third_offset_velocity_radius: float | None = None,
    s_radius: float | None = None,
) -> RegularizedBinaryIntervalCauchyMajorantCertificate:
    """Certify a Cauchy tail bound for a regularized binary interval chart."""

    if retained_order < 0:
        raise ValueError("retained_order cannot be negative")
    masses = np.asarray(initial_state.masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")

    z_norm = _interval_vector_norm_upper_bound(initial_state.z)
    z_velocity_norm = _interval_vector_norm_upper_bound(initial_state.z_velocity)
    center_norm = _interval_vector_norm_upper_bound(initial_state.binary_center)
    center_velocity_norm = _interval_vector_norm_upper_bound(initial_state.binary_center_velocity)
    third_offset_norm = _interval_vector_norm_upper_bound(initial_state.third_offset)
    third_offset_velocity_norm = _interval_vector_norm_upper_bound(initial_state.third_offset_velocity)
    pair_energy_norm = _interval_abs_bound(initial_state.pair_energy)
    if z_radius is None:
        z_radius = 0.05 * max(1.0, z_norm)
    if z_velocity_radius is None:
        z_velocity_radius = 0.05 * max(1.0, z_velocity_norm)
    if pair_energy_radius is None:
        pair_energy_radius = 0.05 * max(1.0, pair_energy_norm)
    if binary_center_radius is None:
        binary_center_radius = 0.05 * max(1.0, center_norm)
    if binary_center_velocity_radius is None:
        binary_center_velocity_radius = 0.05 * max(1.0, center_velocity_norm)
    if third_offset_radius is None:
        third_offset_radius = 0.05 * max(1.0, third_offset_norm)
    if third_offset_velocity_radius is None:
        third_offset_velocity_radius = 0.05 * max(1.0, third_offset_velocity_norm)

    z_radius = float(z_radius)
    z_velocity_radius = float(z_velocity_radius)
    pair_energy_radius = float(pair_energy_radius)
    binary_center_radius = float(binary_center_radius)
    binary_center_velocity_radius = float(binary_center_velocity_radius)
    third_offset_radius = float(third_offset_radius)
    third_offset_velocity_radius = float(third_offset_velocity_radius)
    radii = (
        z_radius,
        z_velocity_radius,
        pair_energy_radius,
        binary_center_radius,
        binary_center_velocity_radius,
        third_offset_radius,
        third_offset_velocity_radius,
    )
    if any(radius <= 0.0 for radius in radii):
        raise ValueError("majorant radii must be positive")

    third_distance_lower, perturbation_bound, center_acceleration_bound, third_offset_acceleration_bound = (
        _regularized_binary_interval_third_body_bounds(
            initial_state,
            z_radius=z_radius,
            third_offset_radius=third_offset_radius,
        )
    )
    z_bound = z_norm + z_radius
    z_velocity_bound = z_velocity_norm + z_velocity_radius
    pair_energy_bound = pair_energy_norm + pair_energy_radius
    center_velocity_bound = center_velocity_norm + binary_center_velocity_radius
    third_offset_velocity_bound = third_offset_velocity_norm + third_offset_velocity_radius
    rho_bound = z_bound**2

    z_rhs_bound = z_velocity_bound
    z_velocity_rhs_bound = (
        0.5 * pair_energy_bound * z_bound
        + 0.5 * rho_bound * z_bound * perturbation_bound
    )
    pair_energy_rhs_bound = 2.0 * z_bound * z_velocity_bound * perturbation_bound
    binary_center_rhs_bound = rho_bound * center_velocity_bound
    binary_center_velocity_rhs_bound = rho_bound * center_acceleration_bound
    third_offset_rhs_bound = rho_bound * third_offset_velocity_bound
    third_offset_velocity_rhs_bound = rho_bound * third_offset_acceleration_bound

    certified_s_radius = min(
        _radius_over_bound(z_radius, z_rhs_bound),
        _radius_over_bound(z_velocity_radius, z_velocity_rhs_bound),
        _radius_over_bound(pair_energy_radius, pair_energy_rhs_bound),
        _radius_over_bound(binary_center_radius, binary_center_rhs_bound),
        _radius_over_bound(binary_center_velocity_radius, binary_center_velocity_rhs_bound),
        _radius_over_bound(third_offset_radius, third_offset_rhs_bound),
        _radius_over_bound(third_offset_velocity_radius, third_offset_velocity_rhs_bound),
    )
    if not np.isfinite(certified_s_radius):
        certified_s_radius = 1.0
    if s_radius is None:
        s_radius = float(np.nextafter(certified_s_radius, 0.0))
    s_radius = float(s_radius)
    if s_radius <= 0.0:
        raise ValueError("s_radius must be positive")

    if (
        s_radius * z_rhs_bound > z_radius
        or s_radius * z_velocity_rhs_bound > z_velocity_radius
        or s_radius * pair_energy_rhs_bound > pair_energy_radius
        or s_radius * binary_center_rhs_bound > binary_center_radius
        or s_radius * binary_center_velocity_rhs_bound > binary_center_velocity_radius
        or s_radius * third_offset_rhs_bound > third_offset_radius
        or s_radius * third_offset_velocity_rhs_bound > third_offset_velocity_radius
    ):
        raise ValueError("s_radius does not map the regularized binary interval majorant ball into itself")
    ratio = abs(float(step_size)) / s_radius
    if ratio >= 1.0:
        raise ValueError("step_size must lie inside the certified s radius")

    state_sup_bound = max(
        z_bound,
        z_velocity_bound,
        pair_energy_bound,
        center_norm + binary_center_radius,
        center_velocity_bound,
        third_offset_norm + third_offset_radius,
        third_offset_velocity_bound,
        s_radius * rho_bound,
    )
    tail_bound = _round_up(state_sup_bound * ratio ** (retained_order + 1) / (1.0 - ratio))

    return RegularizedBinaryIntervalCauchyMajorantCertificate(
        retained_order=int(retained_order),
        step_size=float(step_size),
        s_radius=float(s_radius),
        z_radius=float(z_radius),
        z_velocity_radius=float(z_velocity_radius),
        pair_energy_radius=float(pair_energy_radius),
        binary_center_radius=float(binary_center_radius),
        binary_center_velocity_radius=float(binary_center_velocity_radius),
        third_offset_radius=float(third_offset_radius),
        third_offset_velocity_radius=float(third_offset_velocity_radius),
        selected_pair_rho_bound=float(rho_bound),
        third_distance_lower_bound=float(third_distance_lower),
        perturbation_bound=float(perturbation_bound),
        center_acceleration_bound=float(center_acceleration_bound),
        third_offset_acceleration_bound=float(third_offset_acceleration_bound),
        state_sup_bound=float(state_sup_bound),
        tail_bound=float(tail_bound),
    )


def regularized_binary_interval_atlas_cauchy_majorant_tail_certificate(
    atlas_initial_states: tuple[IntervalRegularizedBinaryCollisionChartState, ...],
    *,
    retained_order: int,
    step_size: float,
    z_radius: float | None = None,
    z_velocity_radius: float | None = None,
    pair_energy_radius: float | None = None,
    binary_center_radius: float | None = None,
    binary_center_velocity_radius: float | None = None,
    third_offset_radius: float | None = None,
    third_offset_velocity_radius: float | None = None,
    s_radius: float | None = None,
) -> RegularizedBinaryIntervalAtlasCauchyMajorantCertificate:
    """Aggregate regularized-binary Cauchy certificates over an interval atlas."""

    if not atlas_initial_states:
        raise ValueError("atlas_initial_states cannot be empty")
    member_certificates = tuple(
        regularized_binary_interval_cauchy_majorant_tail_certificate(
            initial_state,
            retained_order=retained_order,
            step_size=step_size,
            z_radius=z_radius,
            z_velocity_radius=z_velocity_radius,
            pair_energy_radius=pair_energy_radius,
            binary_center_radius=binary_center_radius,
            binary_center_velocity_radius=binary_center_velocity_radius,
            third_offset_radius=third_offset_radius,
            third_offset_velocity_radius=third_offset_velocity_radius,
            s_radius=s_radius,
        )
        for initial_state in atlas_initial_states
    )
    return RegularizedBinaryIntervalAtlasCauchyMajorantCertificate(
        retained_order=int(retained_order),
        step_size=float(step_size),
        member_certificates=member_certificates,
    )


def evaluate_coefficients(arrays: Iterable[Array], step_size: float, *, max_order: int | None = None) -> Array:
    """Evaluate coefficient arrays and return one flattened vector."""

    normalized = _as_arrays(arrays)
    order = normalized[0].shape[0] - 1 if max_order is None else max_order
    if order < 0 or order >= normalized[0].shape[0]:
        raise ValueError("max_order is outside the coefficient range")
    values = []
    for array in normalized:
        value = np.zeros(array.shape[1:], dtype=float)
        for coefficient in array[: order + 1][::-1]:
            value = value * step_size + coefficient
        values.append(value.reshape(-1))
    return np.concatenate(values)


def ordinary_solution_arrays(solution: TaylorSolution) -> list[Array]:
    return [solution.position, solution.velocity]


def ordinary_interval_solution_arrays(solution: IntervalTaylorSolution) -> list[Array]:
    return [solution.position, solution.velocity]


def regularized_binary_solution_arrays(solution: RegularizedBinaryTaylorSolution) -> list[Array]:
    return [
        solution.z,
        solution.z_velocity,
        solution.pair_energy[:, None],
        solution.binary_center,
        solution.binary_center_velocity,
        solution.third_offset,
        solution.third_offset_velocity,
        solution.physical_time[:, None],
    ]


def regularized_binary_interval_solution_arrays(solution: IntervalRegularizedBinaryTaylorSolution) -> list[Array]:
    return [
        solution.z,
        solution.z_velocity,
        solution.pair_energy[:, None],
        solution.binary_center,
        solution.binary_center_velocity,
        solution.third_offset,
        solution.third_offset_velocity,
        solution.physical_time[:, None],
    ]


def spatial_ks_binary_solution_arrays(solution: SpatialKSBinaryTaylorSolution) -> list[Array]:
    return [
        solution.u,
        solution.u_velocity,
        solution.pair_energy[:, None],
        solution.binary_center,
        solution.binary_center_velocity,
        solution.third_offset,
        solution.third_offset_velocity,
        solution.physical_time[:, None],
    ]


def spatial_ks_binary_interval_solution_arrays(
    solution: IntervalSpatialKSBinaryTaylorSolution,
) -> list[Array]:
    return [
        solution.u,
        solution.u_velocity,
        solution.pair_energy[:, None],
        solution.binary_center,
        solution.binary_center_velocity,
        solution.third_offset,
        solution.third_offset_velocity,
        solution.physical_time[:, None],
    ]


def sundman_solution_arrays(solution: SundmanTaylorSolution) -> list[Array]:
    return [
        solution.position,
        solution.velocity,
        solution.physical_time[:, None],
    ]


def sundman_interval_solution_arrays(solution: IntervalSundmanTaylorSolution) -> list[Array]:
    return [
        solution.position,
        solution.velocity,
        solution.physical_time[:, None],
    ]


def ordinary_taylor_tail_certificate(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    retained_order: int,
    guard_order: int,
    step_size: float,
) -> TailBoundCertificate:
    if guard_order < 1:
        raise ValueError("guard_order must be positive")
    solution = construct_interval_taylor_solution(positions, velocities, masses, order=retained_order + guard_order)
    return interval_guarded_tail_certificate(ordinary_interval_solution_arrays(solution), retained_order, step_size)


def regularized_binary_tail_certificate(
    initial_state: RegularizedBinaryCollisionChartState,
    *,
    retained_order: int,
    guard_order: int,
    step_size: float,
) -> TailBoundCertificate:
    if guard_order < 1:
        raise ValueError("guard_order must be positive")
    solution = construct_interval_regularized_binary_taylor_solution(initial_state, order=retained_order + guard_order)
    return interval_guarded_tail_certificate(
        regularized_binary_interval_solution_arrays(solution),
        retained_order,
        step_size,
    )


def spatial_ks_binary_tail_certificate(
    initial_state: SpatialKSBinaryChartState,
    *,
    retained_order: int,
    guard_order: int,
    step_size: float,
) -> TailBoundCertificate:
    if guard_order < 1:
        raise ValueError("guard_order must be positive")
    solution = construct_interval_spatial_ks_binary_taylor_solution(
        initial_state,
        order=retained_order + guard_order,
    )
    return interval_guarded_tail_certificate(
        spatial_ks_binary_interval_solution_arrays(solution),
        retained_order,
        step_size,
    )


def spatial_ks_binary_interval_tail_certificate(
    initial_state: IntervalSpatialKSBinaryChartState,
    *,
    retained_order: int,
    guard_order: int,
    step_size: float,
) -> TailBoundCertificate:
    if guard_order < 1:
        raise ValueError("guard_order must be positive")
    solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        initial_state,
        order=retained_order + guard_order,
    )
    return interval_guarded_tail_certificate(
        spatial_ks_binary_interval_solution_arrays(solution),
        retained_order,
        step_size,
    )


def spatial_ks_binary_interval_segmented_tail_certificate(
    initial_state: IntervalSpatialKSBinaryChartState,
    *,
    retained_order: int,
    guard_order: int,
    step_size: float,
    segment_count: int,
) -> SpatialKSSegmentedTailCertificate:
    """Certify a KS interval tail by re-expanding on smaller substeps.

    Each substep is a full interval Taylor construction from the inflated
    endpoint of the previous certified substep.  This avoids treating a
    small-step guarded estimate as evidence for the original large chart.
    """

    if guard_order < 1:
        raise ValueError("guard_order must be positive")
    retained_order = int(retained_order)
    segment_count = int(segment_count)
    step_size = float(step_size)
    if retained_order < 0:
        raise ValueError("retained_order cannot be negative")
    if segment_count < 1:
        raise ValueError("segment_count must be positive")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be positive and finite")

    segment_step = step_size / segment_count
    state = initial_state
    certificates: list[TailBoundCertificate] = []
    steps: list[float] = []
    propagation_steps: list[SpatialKSSegmentPropagationStep] = []
    physical_time_delta = FloatInterval.point(0.0)
    for index in range(segment_count):
        start_state = state
        solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
            start_state,
            order=retained_order + guard_order,
        )
        equation_residual = certify_spatial_ks_binary_interval_taylor_equations(
            solution,
            coefficient_count=retained_order,
        )
        horizontal_constraint = certify_spatial_ks_binary_horizontal_constraint(
            solution,
            coefficient_count=retained_order,
        )
        pair_energy_constraint = certify_spatial_ks_binary_pair_energy_constraint(
            solution,
            coefficient_count=retained_order,
        )
        certificate = interval_guarded_tail_certificate(
            spatial_ks_binary_interval_solution_arrays(solution),
            retained_order,
            segment_step,
        )
        certificates.append(certificate)
        steps.append(segment_step)
        if not certificate.is_nontrivial:
            return SpatialKSSegmentedTailCertificate(
                retained_order=retained_order,
                guard_order=guard_order,
                requested_step_size=step_size,
                segment_step_sizes=tuple(steps),
                segment_certificates=tuple(certificates),
                final_state=None,
                final_physical_time_delta=None,
                segment_propagation_steps=tuple(propagation_steps),
            )
        evaluation_time = FloatInterval.point(segment_step)
        tail_bound = certificate.tail_bound
        step_physical_time_delta = _inflate_interval(
            interval_array_series_eval(solution.physical_time[:, None], evaluation_time)[0],
            tail_bound,
        )
        physical_time_delta = physical_time_delta + step_physical_time_delta
        state = IntervalSpatialKSBinaryChartState(
            masses=start_state.masses,
            pair=start_state.pair,
            u=_inflate_interval_array(
                interval_array_series_eval(solution.u, evaluation_time),
                tail_bound,
            ),
            u_velocity=_inflate_interval_array(
                interval_array_series_eval(solution.u_velocity, evaluation_time),
                tail_bound,
            ),
            pair_energy=_inflate_interval(
                interval_array_series_eval(solution.pair_energy[:, None], evaluation_time)[0],
                tail_bound,
            ),
            binary_center=_inflate_interval_array(
                interval_array_series_eval(solution.binary_center, evaluation_time),
                tail_bound,
            ),
            binary_center_velocity=_inflate_interval_array(
                interval_array_series_eval(solution.binary_center_velocity, evaluation_time),
                tail_bound,
            ),
            third_offset=_inflate_interval_array(
                interval_array_series_eval(solution.third_offset, evaluation_time),
                tail_bound,
            ),
            third_offset_velocity=_inflate_interval_array(
                interval_array_series_eval(solution.third_offset_velocity, evaluation_time),
                tail_bound,
            ),
            branch_certificate=None,
        )
        propagation_steps.append(
            SpatialKSSegmentPropagationStep(
                index=index,
                start_state=start_state,
                parameter_interval=FloatInterval(0.0, segment_step),
                physical_time_delta=step_physical_time_delta,
                tail_certificate=certificate,
                end_state=state,
                equation_residual_certificate=equation_residual,
                horizontal_constraint_certificate=horizontal_constraint,
                pair_energy_constraint_certificate=pair_energy_constraint,
            )
        )

    return SpatialKSSegmentedTailCertificate(
        retained_order=retained_order,
        guard_order=guard_order,
        requested_step_size=step_size,
        segment_step_sizes=tuple(steps),
        segment_certificates=tuple(certificates),
        final_state=state,
        final_physical_time_delta=physical_time_delta,
        segment_propagation_steps=tuple(propagation_steps),
    )


def sundman_taylor_tail_certificate(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    retained_order: int,
    guard_order: int,
    step_size: float,
    distance_power: float = 1.0,
) -> TailBoundCertificate:
    if guard_order < 1:
        raise ValueError("guard_order must be positive")
    solution = construct_interval_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=retained_order + guard_order,
        distance_power=distance_power,
    )
    return interval_guarded_tail_certificate(sundman_interval_solution_arrays(solution), retained_order, step_size)
