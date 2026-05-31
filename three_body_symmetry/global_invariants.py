"""Global invariant certificates for three-body continuation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .intervals import (
    FloatInterval,
    interval_series_add,
    interval_series_power,
    interval_series_product,
    zero_interval,
)
from .series import scalar_series_power, scalar_series_product


Array = np.ndarray


@dataclass(frozen=True)
class AngularMomentumConservationCertificate:
    """Certificate that a Taylor chart preserves centered angular momentum."""

    coefficient_count: int
    angular_momentum_coefficients: tuple[tuple[FloatInterval, ...], ...]
    coefficient_source: str = "interval_series"

    @property
    def nonconstant_coefficients(self) -> tuple[tuple[FloatInterval, ...], ...]:
        return self.angular_momentum_coefficients[1:]

    @property
    def certified(self) -> bool:
        return all(
            _interval_contains_zero(component)
            for coefficient in self.nonconstant_coefficients
            for component in coefficient
        )

    @property
    def max_nonconstant_radius(self) -> float:
        radii = [
            max(abs(component.lower), abs(component.upper))
            for coefficient in self.nonconstant_coefficients
            for component in coefficient
        ]
        return float(max(radii, default=0.0))


@dataclass(frozen=True)
class EnergyConservationCertificate:
    """Certificate that a Taylor chart preserves Newtonian total energy."""

    coefficient_count: int
    energy_coefficients: tuple[FloatInterval, ...]
    coefficient_source: str = "interval_series"

    @property
    def nonconstant_coefficients(self) -> tuple[FloatInterval, ...]:
        return self.energy_coefficients[1:]

    @property
    def certified(self) -> bool:
        return all(_interval_contains_zero(coefficient) for coefficient in self.nonconstant_coefficients)

    @property
    def max_nonconstant_radius(self) -> float:
        radii = [
            max(abs(coefficient.lower), abs(coefficient.upper))
            for coefficient in self.nonconstant_coefficients
        ]
        return float(max(radii, default=0.0))


@dataclass(frozen=True)
class LinearMomentumConservationCertificate:
    """Certificate that a Taylor chart preserves total linear momentum."""

    coefficient_count: int
    linear_momentum_coefficients: tuple[tuple[FloatInterval, ...], ...]
    coefficient_source: str = "interval_series"

    @property
    def nonconstant_coefficients(self) -> tuple[tuple[FloatInterval, ...], ...]:
        return self.linear_momentum_coefficients[1:]

    @property
    def certified(self) -> bool:
        return all(
            _interval_contains_zero(component)
            for coefficient in self.nonconstant_coefficients
            for component in coefficient
        )

    @property
    def max_nonconstant_radius(self) -> float:
        radii = [
            max(abs(component.lower), abs(component.upper))
            for coefficient in self.nonconstant_coefficients
            for component in coefficient
        ]
        return float(max(radii, default=0.0))


@dataclass(frozen=True)
class CenterOfMassMotionCertificate:
    """Certificate that a Taylor chart preserves inertial center-of-mass motion."""

    coefficient_count: int
    residual_coefficients: tuple[tuple[FloatInterval, ...], ...]
    coefficient_source: str = "interval_series"

    @property
    def certified(self) -> bool:
        return all(
            _interval_contains_zero(component)
            for coefficient in self.residual_coefficients
            for component in coefficient
        )

    @property
    def max_residual_radius(self) -> float:
        radii = [
            max(abs(component.lower), abs(component.upper))
            for coefficient in self.residual_coefficients
            for component in coefficient
        ]
        return float(max(radii, default=0.0))


@dataclass(frozen=True)
class TripleCollisionExclusionCertificate:
    """Certificate that an initial-data set cannot reach total collision.

    The certificate is intentionally one-sided: it only certifies exclusion
    when interval arithmetic proves the translation-reduced angular momentum
    norm is strictly positive. Zero angular momentum remains undecided here.
    """

    angular_momentum_components: tuple[FloatInterval, ...]
    angular_momentum_norm_squared: FloatInterval
    reason: str

    @property
    def angular_momentum_norm_squared_lower_bound(self) -> float:
        return float(self.angular_momentum_norm_squared.lower)

    @property
    def certified(self) -> bool:
        return self.angular_momentum_norm_squared.lower > 0.0

    @property
    def status(self) -> str:
        return "excluded" if self.certified else "undecided"

    @property
    def undecided(self) -> bool:
        return not self.certified


def centered_angular_momentum_components(
    positions: Array,
    velocities: Array,
    masses: Array,
) -> Array:
    """Return translation-reduced angular-momentum bivector components.

    For planar inputs this has one component, equal to the usual scalar
    angular momentum about the center of mass. For 3D inputs the components are
    `(xy, xz, yz)`.
    """

    positions, velocities, masses = _validate_point_initial_data(positions, velocities, masses)
    total_mass = float(np.sum(masses))
    center = np.sum(masses[:, None] * positions, axis=0) / total_mass
    center_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    centered_positions = positions - center
    centered_velocities = velocities - center_velocity
    return _point_angular_momentum_components(centered_positions, centered_velocities, masses)


def interval_centered_angular_momentum_components(
    positions: Array,
    velocities: Array,
    masses: Array,
) -> tuple[FloatInterval, ...]:
    """Return interval-enclosed translation-reduced angular momentum."""

    positions, velocities, masses = _validate_interval_initial_data(positions, velocities, masses)
    total_mass = float(np.sum(masses))
    center = _interval_weighted_average(positions, masses, total_mass)
    center_velocity = _interval_weighted_average(velocities, masses, total_mass)
    centered_positions = _subtract_interval_vector(positions, center)
    centered_velocities = _subtract_interval_vector(velocities, center_velocity)
    return _interval_angular_momentum_components(centered_positions, centered_velocities, masses)


def certify_nonzero_angular_momentum_excludes_triple_collision(
    positions: Array,
    velocities: Array,
    masses: Array,
) -> TripleCollisionExclusionCertificate:
    """Certify Sundman's nonzero-angular-momentum triple-collision obstruction.

    A total collision has zero translation-reduced angular momentum. Since the
    Newtonian three-body flow conserves angular momentum, any initial-data set
    whose centered angular-momentum interval excludes zero cannot reach a
    triple collision. Binary collisions are not excluded by this certificate.
    """

    components = interval_centered_angular_momentum_components(positions, velocities, masses)
    norm_squared = _interval_norm_squared(components)
    reason = (
        "centered angular momentum is bounded away from zero"
        if norm_squared.lower > 0.0
        else "centered angular momentum interval contains zero"
    )
    return TripleCollisionExclusionCertificate(
        angular_momentum_components=components,
        angular_momentum_norm_squared=norm_squared,
        reason=reason,
    )


def centered_angular_momentum_series_coefficients(
    position: Array,
    velocity: Array,
    masses: Array,
    max_degree: int,
) -> Array:
    """Return point Taylor coefficients of centered angular momentum."""

    position, velocity, masses = _validate_point_series(position, velocity, masses, max_degree)
    centered_position, centered_velocity = _center_point_series(position, velocity, masses, max_degree)
    return _point_angular_momentum_series_coefficients(
        centered_position,
        centered_velocity,
        masses,
        max_degree,
    )


def interval_centered_angular_momentum_series_coefficients(
    position: Array,
    velocity: Array,
    masses: Array,
    max_degree: int,
) -> tuple[tuple[FloatInterval, ...], ...]:
    """Return interval Taylor coefficients enclosing centered angular momentum."""

    position, velocity, masses = _validate_interval_series(position, velocity, masses, max_degree)
    centered_position, centered_velocity = _center_interval_series(position, velocity, masses, max_degree)
    return _interval_angular_momentum_series_coefficients(
        centered_position,
        centered_velocity,
        masses,
        max_degree,
    )


def certify_interval_centered_angular_momentum_conservation(
    position: Array,
    velocity: Array,
    masses: Array,
    *,
    coefficient_count: int | None = None,
) -> AngularMomentumConservationCertificate:
    """Certify all nonconstant centered-angular-momentum coefficients vanish.

    The input arrays may contain point floats or ``FloatInterval`` values. The
    certificate is proof-grade when every nonconstant coefficient interval
    contains zero.
    """

    position = np.asarray(position, dtype=object)
    max_degree = int(position.shape[0] - 1)
    if coefficient_count is None:
        coefficient_count = max_degree
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > max_degree:
        raise ValueError("coefficient_count cannot exceed chart order")
    coefficients = interval_centered_angular_momentum_series_coefficients(
        position,
        velocity,
        masses,
        coefficient_count,
    )
    return AngularMomentumConservationCertificate(
        coefficient_count=int(coefficient_count),
        angular_momentum_coefficients=coefficients,
    )


def linear_momentum_series_coefficients(
    velocity: Array,
    masses: Array,
    max_degree: int,
) -> Array:
    """Return point Taylor coefficients of total linear momentum."""

    velocity = np.asarray(velocity, dtype=float)
    masses = np.asarray(masses, dtype=float)
    _validate_momentum_series_shape(velocity, masses, max_degree)
    return _point_linear_momentum_series_coefficients(velocity, masses, max_degree)


def interval_linear_momentum_series_coefficients(
    velocity: Array,
    masses: Array,
    max_degree: int,
) -> tuple[tuple[FloatInterval, ...], ...]:
    """Return interval Taylor coefficients enclosing total linear momentum."""

    velocity = np.asarray(velocity, dtype=object)
    masses = np.asarray(masses, dtype=float)
    _validate_momentum_series_shape(velocity, masses, max_degree)
    return _interval_linear_momentum_series_coefficients(velocity, masses, max_degree)


def certify_interval_linear_momentum_conservation(
    velocity: Array,
    masses: Array,
    *,
    coefficient_count: int | None = None,
) -> LinearMomentumConservationCertificate:
    """Certify all nonconstant total-linear-momentum coefficients vanish."""

    velocity = np.asarray(velocity, dtype=object)
    max_degree = int(velocity.shape[0] - 1)
    if coefficient_count is None:
        coefficient_count = max_degree
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > max_degree:
        raise ValueError("coefficient_count cannot exceed chart order")
    coefficients = interval_linear_momentum_series_coefficients(
        velocity,
        masses,
        coefficient_count,
    )
    return LinearMomentumConservationCertificate(
        coefficient_count=int(coefficient_count),
        linear_momentum_coefficients=coefficients,
    )


def center_of_mass_motion_residual_coefficients(
    position: Array,
    velocity: Array,
    physical_time: Array,
    masses: Array,
    max_degree: int,
) -> Array:
    """Return point residuals for inertial center-of-mass motion.

    The residual is
    ``sum_i m_i q_i(s) - P(0) t(s) - (sum_i m_i q_i(0) - P(0) t(0))``.
    It vanishes iff the projected center of mass moves with constant physical
    velocity, even though Sundman time itself is nonlinear.
    """

    position, velocity, masses = _validate_point_series(position, velocity, masses, max_degree)
    physical_time = _validate_point_scalar_series(physical_time, max_degree)
    return _point_center_of_mass_motion_residual_coefficients(
        position,
        velocity,
        physical_time,
        masses,
        max_degree,
    )


def interval_center_of_mass_motion_residual_coefficients(
    position: Array,
    velocity: Array,
    physical_time: Array,
    masses: Array,
    max_degree: int,
) -> tuple[tuple[FloatInterval, ...], ...]:
    """Return interval residuals for inertial center-of-mass motion."""

    position, velocity, masses = _validate_interval_series(position, velocity, masses, max_degree)
    physical_time = _validate_interval_scalar_series(physical_time, max_degree)
    return _interval_center_of_mass_motion_residual_coefficients(
        position,
        velocity,
        physical_time,
        masses,
        max_degree,
    )


def certify_interval_center_of_mass_motion(
    position: Array,
    velocity: Array,
    physical_time: Array,
    masses: Array,
    *,
    coefficient_count: int | None = None,
) -> CenterOfMassMotionCertificate:
    """Certify the center-of-mass affine law against the physical-time series."""

    position = np.asarray(position, dtype=object)
    max_degree = int(position.shape[0] - 1)
    if coefficient_count is None:
        coefficient_count = max_degree
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > max_degree:
        raise ValueError("coefficient_count cannot exceed chart order")
    residual = interval_center_of_mass_motion_residual_coefficients(
        position,
        velocity,
        physical_time,
        masses,
        coefficient_count,
    )
    return CenterOfMassMotionCertificate(
        coefficient_count=int(coefficient_count),
        residual_coefficients=residual,
    )


def total_energy_series_coefficients(
    position: Array,
    velocity: Array,
    masses: Array,
    max_degree: int,
) -> Array:
    """Return point Taylor coefficients of Newtonian total energy."""

    position, velocity, masses = _validate_point_series(position, velocity, masses, max_degree)
    return _point_energy_series_coefficients(position, velocity, masses, max_degree)


def interval_total_energy_series_coefficients(
    position: Array,
    velocity: Array,
    masses: Array,
    max_degree: int,
) -> tuple[FloatInterval, ...]:
    """Return interval Taylor coefficients enclosing Newtonian total energy."""

    position, velocity, masses = _validate_interval_series(position, velocity, masses, max_degree)
    return _interval_energy_series_coefficients(position, velocity, masses, max_degree)


def certify_interval_total_energy_conservation(
    position: Array,
    velocity: Array,
    masses: Array,
    *,
    coefficient_count: int | None = None,
) -> EnergyConservationCertificate:
    """Certify all nonconstant Newtonian-energy coefficients vanish."""

    position = np.asarray(position, dtype=object)
    max_degree = int(position.shape[0] - 1)
    if coefficient_count is None:
        coefficient_count = max_degree
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > max_degree:
        raise ValueError("coefficient_count cannot exceed chart order")
    coefficients = interval_total_energy_series_coefficients(
        position,
        velocity,
        masses,
        coefficient_count,
    )
    return EnergyConservationCertificate(
        coefficient_count=int(coefficient_count),
        energy_coefficients=coefficients,
    )


def _validate_point_initial_data(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    if positions.ndim != 2 or positions.shape[0] != 3 or positions.shape[1] < 2:
        raise ValueError("expected three bodies in dimension at least two")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    return positions, velocities, masses


def _validate_interval_initial_data(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=object)
    velocities = np.asarray(velocities, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    if positions.ndim != 2 or positions.shape[0] != 3 or positions.shape[1] < 2:
        raise ValueError("expected three bodies in dimension at least two")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")

    interval_positions = np.empty(positions.shape, dtype=object)
    interval_velocities = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        interval_positions[index] = _as_interval(positions[index])
        interval_velocities[index] = _as_interval(velocities[index])
    return interval_positions, interval_velocities, masses


def _validate_point_series(
    position: Array,
    velocity: Array,
    masses: Array,
    max_degree: int,
) -> tuple[Array, Array, Array]:
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if max_degree < 0:
        raise ValueError("max_degree cannot be negative")
    if position.shape != velocity.shape:
        raise ValueError("position and velocity series must have matching shapes")
    if position.ndim != 3 or position.shape[0] <= max_degree or position.shape[1] != 3 or position.shape[2] < 2:
        raise ValueError("expected series shape (degree, 3, dimension>=2)")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    return position, velocity, masses


def _validate_interval_series(
    position: Array,
    velocity: Array,
    masses: Array,
    max_degree: int,
) -> tuple[Array, Array, Array]:
    position = np.asarray(position, dtype=object)
    velocity = np.asarray(velocity, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if max_degree < 0:
        raise ValueError("max_degree cannot be negative")
    if position.shape != velocity.shape:
        raise ValueError("position and velocity series must have matching shapes")
    if position.ndim != 3 or position.shape[0] <= max_degree or position.shape[1] != 3 or position.shape[2] < 2:
        raise ValueError("expected series shape (degree, 3, dimension>=2)")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")

    interval_position = np.empty(position[: max_degree + 1].shape, dtype=object)
    interval_velocity = np.empty(velocity[: max_degree + 1].shape, dtype=object)
    for index in np.ndindex(interval_position.shape):
        interval_position[index] = _as_interval(position[index])
        interval_velocity[index] = _as_interval(velocity[index])
    return interval_position, interval_velocity, masses


def _validate_momentum_series_shape(velocity: Array, masses: Array, max_degree: int) -> None:
    if max_degree < 0:
        raise ValueError("max_degree cannot be negative")
    if velocity.ndim != 3 or velocity.shape[0] <= max_degree or velocity.shape[1] != 3 or velocity.shape[2] < 1:
        raise ValueError("expected velocity series shape (degree, 3, dimension>=1)")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")


def _validate_point_scalar_series(values: Array, max_degree: int) -> Array:
    values = np.asarray(values, dtype=float)
    if max_degree < 0:
        raise ValueError("max_degree cannot be negative")
    if values.ndim != 1 or values.shape[0] <= max_degree:
        raise ValueError("expected scalar series shape (degree,)")
    return values[: max_degree + 1]


def _validate_interval_scalar_series(values: Array, max_degree: int) -> tuple[FloatInterval, ...]:
    values = np.asarray(values, dtype=object)
    if max_degree < 0:
        raise ValueError("max_degree cannot be negative")
    if values.ndim != 1 or values.shape[0] <= max_degree:
        raise ValueError("expected scalar series shape (degree,)")
    return tuple(_as_interval(values[index]) for index in range(max_degree + 1))


def _as_interval(value: object) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


def _interval_weighted_average(values: Array, masses: Array, total_mass: float) -> Array:
    out = np.empty(values.shape[1:], dtype=object)
    for axis in np.ndindex(out.shape):
        total = FloatInterval.point(0.0)
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


def _center_point_series(position: Array, velocity: Array, masses: Array, max_degree: int) -> tuple[Array, Array]:
    total_mass = float(np.sum(masses))
    centered_position = np.empty(position[: max_degree + 1].shape, dtype=float)
    centered_velocity = np.empty(velocity[: max_degree + 1].shape, dtype=float)
    for degree in range(max_degree + 1):
        center = np.sum(masses[:, None] * position[degree], axis=0) / total_mass
        center_velocity = np.sum(masses[:, None] * velocity[degree], axis=0) / total_mass
        centered_position[degree] = position[degree] - center
        centered_velocity[degree] = velocity[degree] - center_velocity
    return centered_position, centered_velocity


def _center_interval_series(position: Array, velocity: Array, masses: Array, max_degree: int) -> tuple[Array, Array]:
    total_mass = float(np.sum(masses))
    centered_position = np.empty(position[: max_degree + 1].shape, dtype=object)
    centered_velocity = np.empty(velocity[: max_degree + 1].shape, dtype=object)
    for degree in range(max_degree + 1):
        center = _interval_weighted_average(position[degree], masses, total_mass)
        center_velocity = _interval_weighted_average(velocity[degree], masses, total_mass)
        centered_position[degree] = _subtract_interval_vector(position[degree], center)
        centered_velocity[degree] = _subtract_interval_vector(velocity[degree], center_velocity)
    return centered_position, centered_velocity


def _component_pairs(dimension: int) -> tuple[tuple[int, int], ...]:
    return tuple((first, second) for first in range(dimension) for second in range(first + 1, dimension))


def _point_angular_momentum_components(positions: Array, velocities: Array, masses: Array) -> Array:
    components = []
    for first_axis, second_axis in _component_pairs(positions.shape[1]):
        value = 0.0
        for body in range(3):
            value += masses[body] * (
                positions[body, first_axis] * velocities[body, second_axis]
                - positions[body, second_axis] * velocities[body, first_axis]
            )
        components.append(float(value))
    return np.asarray(components, dtype=float)


def _point_angular_momentum_series_coefficients(position: Array, velocity: Array, masses: Array, max_degree: int) -> Array:
    components = np.zeros((max_degree + 1, len(_component_pairs(position.shape[2]))), dtype=float)
    for degree in range(max_degree + 1):
        for component_index, (first_axis, second_axis) in enumerate(_component_pairs(position.shape[2])):
            value = 0.0
            for body in range(3):
                for left_degree in range(degree + 1):
                    right_degree = degree - left_degree
                    value += masses[body] * (
                        position[left_degree, body, first_axis] * velocity[right_degree, body, second_axis]
                        - position[left_degree, body, second_axis] * velocity[right_degree, body, first_axis]
                    )
            components[degree, component_index] = float(value)
    return components


def _interval_angular_momentum_components(positions: Array, velocities: Array, masses: Array) -> tuple[FloatInterval, ...]:
    components = []
    for first_axis, second_axis in _component_pairs(positions.shape[1]):
        value = FloatInterval.point(0.0)
        for body in range(3):
            wedge = (
                _as_interval(positions[body, first_axis]) * _as_interval(velocities[body, second_axis])
                - _as_interval(positions[body, second_axis]) * _as_interval(velocities[body, first_axis])
            )
            value = value + wedge.scale(float(masses[body]))
        components.append(value)
    return tuple(components)


def _interval_angular_momentum_series_coefficients(
    position: Array,
    velocity: Array,
    masses: Array,
    max_degree: int,
) -> tuple[tuple[FloatInterval, ...], ...]:
    coefficients = []
    for degree in range(max_degree + 1):
        components = []
        for first_axis, second_axis in _component_pairs(position.shape[2]):
            value = FloatInterval.point(0.0)
            for body in range(3):
                for left_degree in range(degree + 1):
                    right_degree = degree - left_degree
                    wedge = (
                        _as_interval(position[left_degree, body, first_axis])
                        * _as_interval(velocity[right_degree, body, second_axis])
                        - _as_interval(position[left_degree, body, second_axis])
                        * _as_interval(velocity[right_degree, body, first_axis])
                    )
                    value = value + wedge.scale(float(masses[body]))
            components.append(value)
        coefficients.append(tuple(components))
    return tuple(coefficients)


def _point_linear_momentum_series_coefficients(velocity: Array, masses: Array, max_degree: int) -> Array:
    coefficients = np.zeros((max_degree + 1, velocity.shape[2]), dtype=float)
    for degree in range(max_degree + 1):
        coefficients[degree] = np.sum(masses[:, None] * velocity[degree], axis=0)
    return coefficients


def _interval_linear_momentum_series_coefficients(
    velocity: Array,
    masses: Array,
    max_degree: int,
) -> tuple[tuple[FloatInterval, ...], ...]:
    coefficients = []
    for degree in range(max_degree + 1):
        components = []
        for axis in range(velocity.shape[2]):
            value = FloatInterval.point(0.0)
            for body in range(3):
                value = value + _as_interval(velocity[degree, body, axis]).scale(float(masses[body]))
            components.append(value)
        coefficients.append(tuple(components))
    return tuple(coefficients)


def _point_center_of_mass_motion_residual_coefficients(
    position: Array,
    velocity: Array,
    physical_time: Array,
    masses: Array,
    max_degree: int,
) -> Array:
    body_position_moments = np.zeros((max_degree + 1, position.shape[2]), dtype=float)
    for degree in range(max_degree + 1):
        body_position_moments[degree] = np.sum(masses[:, None] * position[degree], axis=0)
    initial_linear_momentum = np.sum(masses[:, None] * velocity[0], axis=0)
    initial_affine_offset = body_position_moments[0] - initial_linear_momentum * physical_time[0]

    residual = np.zeros_like(body_position_moments)
    for degree in range(max_degree + 1):
        residual[degree] = body_position_moments[degree] - initial_linear_momentum * physical_time[degree]
        if degree == 0:
            residual[degree] -= initial_affine_offset
    return residual


def _interval_center_of_mass_motion_residual_coefficients(
    position: Array,
    velocity: Array,
    physical_time: tuple[FloatInterval, ...],
    masses: Array,
    max_degree: int,
) -> tuple[tuple[FloatInterval, ...], ...]:
    position_moment_coefficients = []
    for degree in range(max_degree + 1):
        components = []
        for axis in range(position.shape[2]):
            value = FloatInterval.point(0.0)
            for body in range(3):
                value = value + _as_interval(position[degree, body, axis]).scale(float(masses[body]))
            components.append(value)
        position_moment_coefficients.append(tuple(components))

    initial_linear_momentum = []
    for axis in range(position.shape[2]):
        value = FloatInterval.point(0.0)
        for body in range(3):
            value = value + _as_interval(velocity[0, body, axis]).scale(float(masses[body]))
        initial_linear_momentum.append(value)
    initial_linear_momentum = tuple(initial_linear_momentum)

    initial_affine_offset = tuple(
        position_moment_coefficients[0][axis] - initial_linear_momentum[axis] * physical_time[0]
        for axis in range(position.shape[2])
    )

    residual = []
    for degree in range(max_degree + 1):
        components = []
        for axis in range(position.shape[2]):
            component = position_moment_coefficients[degree][axis] - (
                initial_linear_momentum[axis] * physical_time[degree]
            )
            if degree == 0:
                component = component - initial_affine_offset[axis]
            components.append(component)
        residual.append(tuple(components))
    return tuple(residual)


def _point_energy_series_coefficients(position: Array, velocity: Array, masses: Array, max_degree: int) -> Array:
    coefficients = np.zeros(max_degree + 1, dtype=float)
    body_count, dimension = position.shape[1:]

    for body in range(body_count):
        for axis in range(dimension):
            coefficients += 0.5 * masses[body] * scalar_series_product(
                velocity[:, body, axis],
                velocity[:, body, axis],
                max_degree,
            )

    for first_body in range(body_count):
        for second_body in range(first_body + 1, body_count):
            distance_squared = np.zeros(max_degree + 1, dtype=float)
            delta = position[: max_degree + 1, second_body] - position[: max_degree + 1, first_body]
            for axis in range(dimension):
                distance_squared += scalar_series_product(delta[:, axis], delta[:, axis], max_degree)
            inverse_distance = scalar_series_power(distance_squared, -0.5, max_degree)
            coefficients -= masses[first_body] * masses[second_body] * inverse_distance

    return coefficients


def _interval_energy_series_coefficients(
    position: Array,
    velocity: Array,
    masses: Array,
    max_degree: int,
) -> tuple[FloatInterval, ...]:
    coefficients = tuple(zero_interval() for _ in range(max_degree + 1))
    body_count, dimension = position.shape[1:]

    for body in range(body_count):
        for axis in range(dimension):
            velocity_axis = tuple(_as_interval(velocity[degree, body, axis]) for degree in range(max_degree + 1))
            kinetic = interval_series_product(velocity_axis, velocity_axis, max_degree)
            coefficients = interval_series_add(
                coefficients,
                _scale_interval_series(kinetic, 0.5 * masses[body]),
            )

    for first_body in range(body_count):
        for second_body in range(first_body + 1, body_count):
            distance_squared = tuple(zero_interval() for _ in range(max_degree + 1))
            for axis in range(dimension):
                delta = tuple(
                    _as_interval(position[degree, second_body, axis])
                    - _as_interval(position[degree, first_body, axis])
                    for degree in range(max_degree + 1)
                )
                distance_squared = interval_series_add(
                    distance_squared,
                    interval_series_product(delta, delta, max_degree),
                )
            inverse_distance = interval_series_power(distance_squared, -0.5, max_degree)
            coefficients = interval_series_add(
                coefficients,
                _scale_interval_series(inverse_distance, -masses[first_body] * masses[second_body]),
            )

    return coefficients


def _scale_interval_series(coefficients: tuple[FloatInterval, ...], factor: float) -> tuple[FloatInterval, ...]:
    return tuple(coefficient.scale(float(factor)) for coefficient in coefficients)


def _interval_contains_zero(value: FloatInterval) -> bool:
    return value.lower <= 0.0 <= value.upper


def _interval_square(value: FloatInterval) -> FloatInterval:
    squares = (value.lower * value.lower, value.upper * value.upper)
    upper = float(np.nextafter(max(squares), np.inf))
    if value.lower <= 0.0 <= value.upper:
        lower = 0.0
    else:
        lower = max(0.0, float(np.nextafter(min(squares), -np.inf)))
    return FloatInterval(lower, upper)


def _interval_norm_squared(values: tuple[FloatInterval, ...]) -> FloatInterval:
    total = FloatInterval.point(0.0)
    for value in values:
        total = total + _interval_square(value)
    return FloatInterval(max(0.0, total.lower), max(0.0, total.upper))
