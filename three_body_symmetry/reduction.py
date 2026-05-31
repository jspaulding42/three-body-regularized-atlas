"""Center-of-mass-frame symmetry reduction and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .intervals import FloatInterval, interval_array_contains_point, zero_interval
from .series import TaylorSolution
from .sundman import SundmanTaylorSolution


Array = np.ndarray


@dataclass(frozen=True)
class CenterOfMassReductionCertificate:
    """Certificate for the inertial center-of-mass-frame lift."""

    center_position: Array
    center_velocity: Array
    reduced_position_moment: Array
    reduced_linear_momentum: Array
    initial_position_reconstruction_residual: Array
    initial_velocity_reconstruction_residual: Array
    tolerance: float = 1e-12

    @property
    def max_residual(self) -> float:
        residuals = (
            np.linalg.norm(self.reduced_position_moment, ord=np.inf),
            np.linalg.norm(self.reduced_linear_momentum, ord=np.inf),
            np.linalg.norm(self.initial_position_reconstruction_residual, ord=np.inf),
            np.linalg.norm(self.initial_velocity_reconstruction_residual, ord=np.inf),
        )
        return float(max(residuals))

    @property
    def certified(self) -> bool:
        return bool(self.max_residual <= self.tolerance)


@dataclass(frozen=True)
class CenterOfMassFrameData:
    """Initial data lifted to an inertial center-of-mass frame."""

    positions: Array
    velocities: Array
    masses: Array
    center_position: Array
    center_velocity: Array
    certificate: CenterOfMassReductionCertificate


@dataclass(frozen=True)
class IntervalCenterOfMassReductionCertificate:
    """Interval certificate for the inertial center-of-mass-frame lift."""

    center_position: Array
    center_velocity: Array
    reduced_position_moment: tuple[FloatInterval, ...]
    reduced_linear_momentum: tuple[FloatInterval, ...]
    initial_position_reconstruction_residual: Array
    initial_velocity_reconstruction_residual: Array

    @property
    def certified(self) -> bool:
        return bool(
            all(_interval_contains_zero(value) for value in self.reduced_position_moment)
            and all(_interval_contains_zero(value) for value in self.reduced_linear_momentum)
            and _interval_array_contains_zero(self.initial_position_reconstruction_residual)
            and _interval_array_contains_zero(self.initial_velocity_reconstruction_residual)
        )


@dataclass(frozen=True)
class IntervalCenterOfMassFrameData:
    """Interval initial data lifted to an inertial center-of-mass frame."""

    positions: Array
    velocities: Array
    masses: Array
    center_position: Array
    center_velocity: Array
    certificate: IntervalCenterOfMassReductionCertificate

    def contains_point_reduction(self, point_reduction: CenterOfMassFrameData) -> bool:
        return bool(
            interval_array_contains_point(self.positions, point_reduction.positions)
            and interval_array_contains_point(self.velocities, point_reduction.velocities)
            and interval_array_contains_point(self.center_position, point_reduction.center_position)
            and interval_array_contains_point(self.center_velocity, point_reduction.center_velocity)
        )


def reduce_to_center_of_mass_frame(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    tolerance: float = 1e-12,
) -> CenterOfMassFrameData:
    """Translate arbitrary initial data into the inertial COM frame."""

    positions, velocities, masses = _validate_initial_data(positions, velocities, masses)
    total_mass = float(np.sum(masses))
    center_position = np.sum(masses[:, None] * positions, axis=0) / total_mass
    center_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    reduced_positions = positions - center_position
    reduced_velocities = velocities - center_velocity
    reconstructed_positions, reconstructed_velocities = reconstruct_from_center_of_mass_frame(
        reduced_positions,
        reduced_velocities,
        center_position,
        center_velocity,
        time=0.0,
    )
    certificate = CenterOfMassReductionCertificate(
        center_position=center_position,
        center_velocity=center_velocity,
        reduced_position_moment=np.sum(masses[:, None] * reduced_positions, axis=0),
        reduced_linear_momentum=np.sum(masses[:, None] * reduced_velocities, axis=0),
        initial_position_reconstruction_residual=reconstructed_positions - positions,
        initial_velocity_reconstruction_residual=reconstructed_velocities - velocities,
        tolerance=float(tolerance),
    )
    return CenterOfMassFrameData(
        positions=reduced_positions,
        velocities=reduced_velocities,
        masses=masses,
        center_position=center_position,
        center_velocity=center_velocity,
        certificate=certificate,
    )


def reduce_interval_to_center_of_mass_frame(
    positions: Array,
    velocities: Array,
    masses: Array,
) -> IntervalCenterOfMassFrameData:
    """Translate an interval initial-data box into an interval COM frame."""

    positions, velocities, masses = _validate_interval_initial_data(positions, velocities, masses)
    total_mass = float(np.sum(masses))
    center_position = _interval_weighted_average(positions, masses, total_mass)
    center_velocity = _interval_weighted_average(velocities, masses, total_mass)
    reduced_positions = _subtract_interval_vector(positions, center_position)
    reduced_velocities = _subtract_interval_vector(velocities, center_velocity)
    reconstructed_positions, reconstructed_velocities = reconstruct_interval_from_center_of_mass_frame(
        reduced_positions,
        reduced_velocities,
        center_position,
        center_velocity,
        time=FloatInterval.point(0.0),
    )
    certificate = IntervalCenterOfMassReductionCertificate(
        center_position=center_position,
        center_velocity=center_velocity,
        reduced_position_moment=_interval_mass_moment(reduced_positions, masses),
        reduced_linear_momentum=_interval_mass_moment(reduced_velocities, masses),
        initial_position_reconstruction_residual=_subtract_interval_arrays(reconstructed_positions, positions),
        initial_velocity_reconstruction_residual=_subtract_interval_arrays(reconstructed_velocities, velocities),
    )
    return IntervalCenterOfMassFrameData(
        positions=reduced_positions,
        velocities=reduced_velocities,
        masses=masses,
        center_position=center_position,
        center_velocity=center_velocity,
        certificate=certificate,
    )


def reconstruct_from_center_of_mass_frame(
    reduced_positions: Array,
    reduced_velocities: Array,
    center_position: Array,
    center_velocity: Array,
    *,
    time: float,
) -> tuple[Array, Array]:
    """Project a COM-frame state back to physical coordinates at time ``t``."""

    reduced_positions = np.asarray(reduced_positions, dtype=float)
    reduced_velocities = np.asarray(reduced_velocities, dtype=float)
    center_position = np.asarray(center_position, dtype=float)
    center_velocity = np.asarray(center_velocity, dtype=float)
    if reduced_positions.shape != reduced_velocities.shape or reduced_positions.ndim != 2:
        raise ValueError("reduced positions and velocities must have matching shape (body_count, dimension)")
    if center_position.shape != (reduced_positions.shape[1],) or center_velocity.shape != center_position.shape:
        raise ValueError("center vectors must match the state dimension")
    shift = center_position + float(time) * center_velocity
    return reduced_positions + shift, reduced_velocities + center_velocity


def reconstruct_interval_from_center_of_mass_frame(
    reduced_positions: Array,
    reduced_velocities: Array,
    center_position: Array,
    center_velocity: Array,
    *,
    time: FloatInterval | float,
) -> tuple[Array, Array]:
    """Project an interval COM-frame state box back to physical coordinates."""

    reduced_positions = np.asarray(reduced_positions, dtype=object)
    reduced_velocities = np.asarray(reduced_velocities, dtype=object)
    center_position = np.asarray(center_position, dtype=object)
    center_velocity = np.asarray(center_velocity, dtype=object)
    time_interval = _as_interval(time)
    if reduced_positions.shape != reduced_velocities.shape or reduced_positions.ndim != 2:
        raise ValueError("reduced positions and velocities must have matching shape (body_count, dimension)")
    if center_position.shape != (reduced_positions.shape[1],) or center_velocity.shape != center_position.shape:
        raise ValueError("center vectors must match the state dimension")
    positions = np.empty(reduced_positions.shape, dtype=object)
    velocities = np.empty(reduced_velocities.shape, dtype=object)
    for body in range(reduced_positions.shape[0]):
        for axis in range(reduced_positions.shape[1]):
            shift = _as_interval(center_position[axis]) + _as_interval(center_velocity[axis]) * time_interval
            positions[body, axis] = _as_interval(reduced_positions[body, axis]) + shift
            velocities[body, axis] = _as_interval(reduced_velocities[body, axis]) + _as_interval(center_velocity[axis])
    return positions, velocities


def reconstruct_taylor_solution_from_center_of_mass(
    reduced_solution: TaylorSolution,
    center_position: Array,
    center_velocity: Array,
) -> TaylorSolution:
    """Project an ordinary COM-frame Taylor solution back to physical space."""

    center_position, center_velocity = _validate_center_vectors(
        center_position,
        center_velocity,
        reduced_solution.dimension,
    )
    position = np.array(reduced_solution.position, dtype=float, copy=True)
    velocity = np.array(reduced_solution.velocity, dtype=float, copy=True)
    position[0] += center_position
    position[1] += center_velocity
    velocity[0] += center_velocity
    return TaylorSolution(position=position, velocity=velocity, masses=reduced_solution.masses)


def reconstruct_sundman_solution_from_center_of_mass(
    reduced_solution: SundmanTaylorSolution,
    center_position: Array,
    center_velocity: Array,
) -> SundmanTaylorSolution:
    """Project a COM-frame Sundman chart back using its physical-time series."""

    center_position, center_velocity = _validate_center_vectors(
        center_position,
        center_velocity,
        reduced_solution.dimension,
    )
    position = np.array(reduced_solution.position, dtype=float, copy=True)
    velocity = np.array(reduced_solution.velocity, dtype=float, copy=True)
    for degree, physical_time_coefficient in enumerate(reduced_solution.physical_time):
        position[degree] += center_velocity * physical_time_coefficient
    position[0] += center_position
    velocity[0] += center_velocity
    return SundmanTaylorSolution(
        position=position,
        velocity=velocity,
        physical_time=np.array(reduced_solution.physical_time, dtype=float, copy=True),
        masses=reduced_solution.masses,
        distance_power=reduced_solution.distance_power,
    )


def _validate_initial_data(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.shape != velocities.shape or positions.ndim != 2:
        raise ValueError("positions and velocities must have matching shape (body_count, dimension)")
    if positions.shape[0] != 3:
        raise ValueError("expected three bodies")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    return positions, velocities, masses


def _validate_interval_initial_data(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=object)
    velocities = np.asarray(velocities, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if positions.shape != velocities.shape or positions.ndim != 2:
        raise ValueError("positions and velocities must have matching shape (body_count, dimension)")
    if positions.shape[0] != 3:
        raise ValueError("expected three bodies")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    interval_positions = np.empty(positions.shape, dtype=object)
    interval_velocities = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        interval_positions[index] = _as_interval(positions[index])
        interval_velocities[index] = _as_interval(velocities[index])
    return interval_positions, interval_velocities, masses


def _validate_center_vectors(center_position: Array, center_velocity: Array, dimension: int) -> tuple[Array, Array]:
    center_position = np.asarray(center_position, dtype=float)
    center_velocity = np.asarray(center_velocity, dtype=float)
    if center_position.shape != (dimension,) or center_velocity.shape != (dimension,):
        raise ValueError("center vectors must match the solution dimension")
    return center_position, center_velocity


def _as_interval(value: object) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


def _interval_contains_zero(value: FloatInterval) -> bool:
    return value.lower <= 0.0 <= value.upper


def _interval_array_contains_zero(values: Array) -> bool:
    values = np.asarray(values, dtype=object)
    for index in np.ndindex(values.shape):
        if not _interval_contains_zero(_as_interval(values[index])):
            return False
    return True


def _interval_weighted_average(values: Array, masses: Array, total_mass: float) -> Array:
    out = np.empty(values.shape[1:], dtype=object)
    for axis in np.ndindex(out.shape):
        total = zero_interval()
        for body in range(3):
            total = total + _as_interval(values[(body, *axis)]).scale(float(masses[body]))
        out[axis] = total.scale(1.0 / total_mass)
    return out


def _subtract_interval_vector(values: Array, vector: Array) -> Array:
    out = np.empty(values.shape, dtype=object)
    for body in range(values.shape[0]):
        for axis in range(values.shape[1]):
            out[body, axis] = _as_interval(values[body, axis]) - _as_interval(vector[axis])
    return out


def _subtract_interval_arrays(left: Array, right: Array) -> Array:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape:
        raise ValueError("interval arrays must have matching shapes")
    out = np.empty(left.shape, dtype=object)
    for index in np.ndindex(left.shape):
        out[index] = _as_interval(left[index]) - _as_interval(right[index])
    return out


def _interval_mass_moment(values: Array, masses: Array) -> tuple[FloatInterval, ...]:
    values = np.asarray(values, dtype=object)
    moments = []
    for axis in range(values.shape[1]):
        moment = zero_interval()
        for body in range(3):
            moment = moment + _as_interval(values[body, axis]).scale(float(masses[body]))
        moments.append(moment)
    return tuple(moments)
