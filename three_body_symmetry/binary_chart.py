"""Planar three-body coordinates around one possible binary collision.

This module embeds the Levi-Civita relative coordinate into the full planar
three-body problem. A selected pair is represented by its binary center of mass
and a regularized relative coordinate `r = z^2`; the remaining body is stored as
an offset from that binary center. The chart is local to a pair collision with a
separated third body.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .dynamics import accelerations
from .intervals import FloatInterval, directed_nonnegative_sqrt_endpoint
from .levi_civita import lc_matrix, lc_square, lc_velocity


Array = np.ndarray


@dataclass(frozen=True)
class BinaryCollisionChartState:
    masses: Array
    pair: tuple[int, int]
    z: Array
    z_velocity: Array
    binary_center: Array
    binary_center_velocity: Array
    third_offset: Array
    third_offset_velocity: Array

    @property
    def third_index(self) -> int:
        return _third_index(self.pair)

    @property
    def rho(self) -> float:
        return float(np.dot(self.z, self.z))


@dataclass(frozen=True)
class BinaryCollisionChartDerivative:
    z: Array
    z_velocity: Array
    binary_center: Array
    binary_center_velocity: Array
    third_offset: Array
    third_offset_velocity: Array
    physical_time: float


@dataclass(frozen=True)
class RegularizedBinaryCollisionChartState:
    masses: Array
    pair: tuple[int, int]
    z: Array
    z_velocity: Array
    pair_energy: float
    binary_center: Array
    binary_center_velocity: Array
    third_offset: Array
    third_offset_velocity: Array

    @property
    def third_index(self) -> int:
        return _third_index(self.pair)

    @property
    def rho(self) -> float:
        return float(np.dot(self.z, self.z))


@dataclass(frozen=True)
class LeviCivitaBranchCertificate:
    branch: str
    certified: bool
    reason: str


@dataclass(frozen=True)
class IntervalRegularizedBinaryCollisionChartState:
    masses: Array
    pair: tuple[int, int]
    z: Array
    z_velocity: Array
    pair_energy: FloatInterval
    binary_center: Array
    binary_center_velocity: Array
    third_offset: Array
    third_offset_velocity: Array
    branch_certificate: LeviCivitaBranchCertificate | None = None

    @property
    def third_index(self) -> int:
        return _third_index(self.pair)

    def contains_point(self, state: RegularizedBinaryCollisionChartState) -> bool:
        if tuple(state.pair) != tuple(self.pair):
            return False
        checks = [
            (self.z, state.z),
            (self.z_velocity, state.z_velocity),
            (self.binary_center, state.binary_center),
            (self.binary_center_velocity, state.binary_center_velocity),
            (self.third_offset, state.third_offset),
            (self.third_offset_velocity, state.third_offset_velocity),
        ]
        if not (self.pair_energy.lower <= state.pair_energy <= self.pair_energy.upper):
            return False
        return all(_interval_array_contains(intervals, values) for intervals, values in checks)


@dataclass(frozen=True)
class RegularizedBinaryCollisionChartDerivative:
    z: Array
    z_velocity: Array
    pair_energy: float
    binary_center: Array
    binary_center_velocity: Array
    third_offset: Array
    third_offset_velocity: Array
    physical_time: float


def _third_index(pair: tuple[int, int]) -> int:
    remaining = ({0, 1, 2} - set(pair))
    if len(remaining) != 1 or pair[0] == pair[1]:
        raise ValueError("pair must contain two distinct body indices from {0, 1, 2}")
    return remaining.pop()


def _validate_planar(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.shape != (3, 2) or velocities.shape != (3, 2):
        raise ValueError("positions and velocities must have shape (3, 2)")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    return positions, velocities, masses


def _as_interval(value: object) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


def _coerce_interval(value: object) -> FloatInterval:
    if isinstance(value, FloatInterval):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return FloatInterval(float(value[0]), float(value[1]))
    return FloatInterval.point(float(value))


def _coerce_interval_vector(values: Array, *, name: str) -> Array:
    values = np.asarray(values, dtype=object)
    if values.shape == (2, 2):
        return np.array(
            [FloatInterval(float(values[index, 0]), float(values[index, 1])) for index in range(2)],
            dtype=object,
        )
    if values.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    return np.array([_coerce_interval(value) for value in values], dtype=object)


def _interval_array_contains(intervals: Array, values: Array) -> bool:
    intervals = np.asarray(intervals, dtype=object)
    values = np.asarray(values, dtype=float)
    if intervals.shape != values.shape:
        return False
    for index in np.ndindex(intervals.shape):
        interval = _as_interval(intervals[index])
        if interval.lower > values[index] or values[index] > interval.upper:
            return False
    return True


def _interval_square_bounds(value: FloatInterval) -> FloatInterval:
    squares = (value.lower * value.lower, value.upper * value.upper)
    if value.lower <= 0.0 <= value.upper:
        lower = 0.0
    else:
        lower = max(0.0, float(np.nextafter(min(squares), -np.inf)))
    return FloatInterval(lower, float(np.nextafter(max(squares), np.inf)))


def _interval_norm(vector: Array) -> FloatInterval:
    total = FloatInterval.point(0.0)
    for value in np.asarray(vector, dtype=object).reshape(-1):
        total = total + _interval_square_bounds(_as_interval(value))
    return FloatInterval(
        0.0
        if total.lower <= 0.0
        else directed_nonnegative_sqrt_endpoint(total.lower, upward=False),
        directed_nonnegative_sqrt_endpoint(max(total.upper, 0.0), upward=True),
    )


def _interval_sqrt_nonnegative(value: FloatInterval) -> FloatInterval:
    if value.upper < 0.0:
        raise ValueError("cannot take square root of a negative interval")
    lower = max(0.0, value.lower)
    return FloatInterval(
        directed_nonnegative_sqrt_endpoint(lower, upward=False)
        if lower > 0.0
        else 0.0,
        directed_nonnegative_sqrt_endpoint(max(value.upper, 0.0), upward=True),
    )


def _principal_sqrt_interval(relative_position: Array) -> Array:
    x = _as_interval(relative_position[0])
    y = _as_interval(relative_position[1])
    radius = _interval_norm(relative_position)
    real_square = (radius + x).scale(0.5)
    imag_square = (radius - x).scale(0.5)
    real_part = _interval_sqrt_nonnegative(FloatInterval(max(0.0, real_square.lower), real_square.upper))
    imag_magnitude = _interval_sqrt_nonnegative(FloatInterval(max(0.0, imag_square.lower), imag_square.upper))
    if y.lower >= 0.0:
        imag_part = imag_magnitude
    elif y.upper <= 0.0:
        imag_part = FloatInterval(-imag_magnitude.upper, -imag_magnitude.lower)
    else:
        magnitude = max(abs(imag_magnitude.lower), abs(imag_magnitude.upper))
        imag_part = FloatInterval(-magnitude, magnitude)
    return np.array([real_part, imag_part], dtype=object)


def levi_civita_branch_certificate(relative_position: Array) -> LeviCivitaBranchCertificate:
    """Return a sufficient branch certificate for the principal square root."""

    x = _as_interval(relative_position[0])
    y = _as_interval(relative_position[1])
    if y.lower >= 0.0:
        return LeviCivitaBranchCertificate(
            branch="principal_upper_half",
            certified=True,
            reason="relative-position interval lies in the closed upper half-plane",
        )
    if y.upper <= 0.0:
        return LeviCivitaBranchCertificate(
            branch="principal_lower_half",
            certified=True,
            reason="relative-position interval lies in the closed lower half-plane",
        )
    if x.lower > 0.0:
        return LeviCivitaBranchCertificate(
            branch="principal_right_half",
            certified=True,
            reason="relative-position interval lies strictly in the right half-plane",
        )
    return LeviCivitaBranchCertificate(
        branch="principal_branch_cut_overlap",
        certified=False,
        reason="relative-position interval may overlap the principal square-root branch cut",
    )


def _interval_lc_transpose_times(z: Array, vector: Array) -> Array:
    x, y = z
    vx, vy = vector
    return np.array(
        [
            (x * vx + y * vy).scale(2.0),
            (x * vy - y * vx).scale(2.0),
        ],
        dtype=object,
    )


def _exact_endpoint_interval_difference(
    left: FloatInterval,
    right: FloatInterval,
) -> FloatInterval:
    """Tight directed difference of binary-float interval endpoints.

    The generic interval operation deliberately expands every subtraction by
    one adjacent float.  Exact endpoint arithmetic avoids turning a proved
    closed-half-plane boundary at zero into an artificial sign crossing.
    """

    lower_q = Fraction.from_float(left.lower) - Fraction.from_float(right.upper)
    upper_q = Fraction.from_float(left.upper) - Fraction.from_float(right.lower)
    lower = float(lower_q)
    upper = float(upper_q)
    if Fraction.from_float(lower) > lower_q:
        lower = float(np.nextafter(lower, -np.inf))
    if Fraction.from_float(upper) < upper_q:
        upper = float(np.nextafter(upper, np.inf))
    return FloatInterval(lower, upper)


def _exact_fraction_interval(value: Fraction) -> FloatInterval:
    """Tight binary-float interval enclosing one exact rational coefficient."""

    candidate = float(value)
    candidate_q = Fraction.from_float(candidate)
    lower = candidate
    upper = candidate
    if candidate_q > value:
        lower = float(np.nextafter(candidate, -np.inf))
    elif candidate_q < value:
        upper = float(np.nextafter(candidate, np.inf))
    return FloatInterval(lower, upper)


def _interval_planar_components(
    state_interval: tuple[tuple[float, float], ...],
    pair: tuple[int, int],
) -> tuple[Array, Array, Array, Array]:
    if len(state_interval) != 12:
        raise ValueError("planar interval state must have length 12")
    first, second = pair
    values = np.array([FloatInterval(lower, upper) for lower, upper in state_interval], dtype=object)
    positions = values[:6].reshape(3, 2)
    velocities = values[6:].reshape(3, 2)
    relative_position = np.array(
        [
            _exact_endpoint_interval_difference(
                positions[second, axis], positions[first, axis]
            )
            for axis in range(2)
        ],
        dtype=object,
    )
    relative_velocity = np.array(
        [
            _exact_endpoint_interval_difference(
                velocities[second, axis], velocities[first, axis]
            )
            for axis in range(2)
        ],
        dtype=object,
    )
    return positions, velocities, relative_position, relative_velocity


def _regularized_interval_chart_from_components(
    masses: Array,
    pair: tuple[int, int],
    positions: Array,
    velocities: Array,
    relative_position: Array,
    relative_velocity: Array,
    branch_certificate: LeviCivitaBranchCertificate,
) -> IntervalRegularizedBinaryCollisionChartState:
    first, second = pair
    third = _third_index(pair)
    first_mass_q = Fraction.from_float(float(masses[first]))
    second_mass_q = Fraction.from_float(float(masses[second]))
    pair_mass_q = first_mass_q + second_mass_q
    pair_mass = _exact_fraction_interval(pair_mass_q)
    first_mass_ratio = _exact_fraction_interval(first_mass_q / pair_mass_q)
    second_mass_ratio = _exact_fraction_interval(second_mass_q / pair_mass_q)
    radius = _interval_norm(relative_position)
    if radius.lower <= 0.0:
        raise ValueError("selected pair interval contains binary collision")

    z = _principal_sqrt_interval(relative_position)
    z_velocity = np.array(
        [component.scale(0.25) for component in _interval_lc_transpose_times(z, relative_velocity)],
        dtype=object,
    )

    binary_center = np.empty(2, dtype=object)
    binary_center_velocity = np.empty(2, dtype=object)
    third_offset = np.empty(2, dtype=object)
    third_offset_velocity = np.empty(2, dtype=object)
    for axis in range(2):
        binary_center[axis] = (
            positions[first, axis] * first_mass_ratio
            + positions[second, axis] * second_mass_ratio
        )
        binary_center_velocity[axis] = (
            velocities[first, axis] * first_mass_ratio
            + velocities[second, axis] * second_mass_ratio
        )
        third_offset[axis] = positions[third, axis] - binary_center[axis]
        third_offset_velocity[axis] = velocities[third, axis] - binary_center_velocity[axis]

    speed_square = FloatInterval.point(0.0)
    for axis in range(2):
        speed_square = speed_square + _interval_square_bounds(relative_velocity[axis])
    pair_energy = speed_square.scale(0.5) - radius.reciprocal() * pair_mass

    return IntervalRegularizedBinaryCollisionChartState(
        masses=masses,
        pair=pair,
        z=z,
        z_velocity=z_velocity,
        pair_energy=pair_energy,
        binary_center=binary_center,
        binary_center_velocity=binary_center_velocity,
        third_offset=third_offset,
        third_offset_velocity=third_offset_velocity,
        branch_certificate=branch_certificate,
    )


def exact_binary_collision_interval_chart(
    masses: Array,
    *,
    pair: tuple[int, int] = (0, 1),
    z_velocity: Array,
    pair_energy: FloatInterval | tuple[float, float] | float,
    binary_center: Array,
    binary_center_velocity: Array,
    third_offset: Array,
    third_offset_velocity: Array,
) -> IntervalRegularizedBinaryCollisionChartState:
    """Build certified interval chart data at exact selected-pair collision.

    Finite planar velocities do not define a regularized lift at exact binary
    collision. This constructor therefore accepts already-lifted interval data
    and certifies the two conditions needed for a local regularized chart:
    the third body is separated from the binary center, and the collision
    constraint ``2 |z'|^2 = m_i + m_j`` is interval-satisfied.
    """

    masses = np.asarray(masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    first, second = pair
    _third_index(pair)
    pair_mass = masses[first] + masses[second]
    z = np.array([FloatInterval.point(0.0), FloatInterval.point(0.0)], dtype=object)
    z_velocity = _coerce_interval_vector(z_velocity, name="z_velocity")
    binary_center = _coerce_interval_vector(binary_center, name="binary_center")
    binary_center_velocity = _coerce_interval_vector(
        binary_center_velocity,
        name="binary_center_velocity",
    )
    third_offset = _coerce_interval_vector(third_offset, name="third_offset")
    third_offset_velocity = _coerce_interval_vector(
        third_offset_velocity,
        name="third_offset_velocity",
    )
    pair_energy = _coerce_interval(pair_energy)

    if _interval_norm(third_offset).lower <= 0.0:
        raise ValueError("third body must be separated from the binary collision center")

    speed_square = FloatInterval.point(0.0)
    for axis in range(2):
        speed_square = speed_square + _interval_square_bounds(z_velocity[axis])
    collision_constraint = speed_square.scale(2.0) - FloatInterval.point(pair_mass)
    if not (collision_constraint.lower <= 0.0 <= collision_constraint.upper):
        raise ValueError("exact binary collision interval must satisfy 2|z_velocity|^2 = pair mass")

    return IntervalRegularizedBinaryCollisionChartState(
        masses=masses,
        pair=pair,
        z=z,
        z_velocity=z_velocity,
        pair_energy=pair_energy,
        binary_center=binary_center,
        binary_center_velocity=binary_center_velocity,
        third_offset=third_offset,
        third_offset_velocity=third_offset_velocity,
        branch_certificate=LeviCivitaBranchCertificate(
            branch="exact_binary_collision",
            certified=True,
            reason="lifted interval data start at exact binary collision with separated third body",
        ),
    )


def planar_to_binary_collision_chart(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    pair: tuple[int, int] = (0, 1),
) -> BinaryCollisionChartState:
    """Lift a non-collision planar state into a pair-specific binary chart."""

    positions, velocities, masses = _validate_planar(positions, velocities, masses)
    first, second = pair
    third = _third_index(pair)
    pair_mass = masses[first] + masses[second]
    relative_position = positions[second] - positions[first]
    relative_velocity = velocities[second] - velocities[first]
    radius = np.linalg.norm(relative_position)
    if radius == 0.0:
        raise ValueError("cannot lift physical velocities at exact binary collision")

    root = np.sqrt(relative_position[0] + 1j * relative_position[1])
    z = np.array([root.real, root.imag], dtype=float)
    z_velocity = 0.25 * lc_matrix(z).T @ relative_velocity
    rho = float(np.dot(z, z))
    binary_center = (masses[first] * positions[first] + masses[second] * positions[second]) / pair_mass
    binary_center_dot = (masses[first] * velocities[first] + masses[second] * velocities[second]) / pair_mass
    third_offset = positions[third] - binary_center
    third_offset_dot = velocities[third] - binary_center_dot

    return BinaryCollisionChartState(
        masses=masses,
        pair=pair,
        z=z,
        z_velocity=z_velocity,
        binary_center=binary_center,
        binary_center_velocity=rho * binary_center_dot,
        third_offset=third_offset,
        third_offset_velocity=rho * third_offset_dot,
    )


def planar_to_regularized_binary_collision_chart(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    pair: tuple[int, int] = (0, 1),
) -> RegularizedBinaryCollisionChartState:
    """Lift a non-collision planar state into the exact-collision-ready chart."""

    positions, velocities, masses = _validate_planar(positions, velocities, masses)
    first, second = pair
    _third = _third_index(pair)
    pair_mass = masses[first] + masses[second]
    relative_position = positions[second] - positions[first]
    relative_velocity = velocities[second] - velocities[first]
    radius = np.linalg.norm(relative_position)
    if radius == 0.0:
        raise ValueError("cannot lift physical velocities at exact binary collision")

    root = np.sqrt(relative_position[0] + 1j * relative_position[1])
    z = np.array([root.real, root.imag], dtype=float)
    z_velocity = 0.25 * lc_matrix(z).T @ relative_velocity
    binary_center = (masses[first] * positions[first] + masses[second] * positions[second]) / pair_mass
    binary_center_velocity = (masses[first] * velocities[first] + masses[second] * velocities[second]) / pair_mass
    third = _third_index(pair)
    third_offset = positions[third] - binary_center
    third_offset_velocity = velocities[third] - binary_center_velocity
    pair_energy = 0.5 * float(np.dot(relative_velocity, relative_velocity)) - pair_mass / radius

    return RegularizedBinaryCollisionChartState(
        masses=masses,
        pair=pair,
        z=z,
        z_velocity=z_velocity,
        pair_energy=pair_energy,
        binary_center=binary_center,
        binary_center_velocity=binary_center_velocity,
        third_offset=third_offset,
        third_offset_velocity=third_offset_velocity,
    )


def planar_interval_to_regularized_binary_collision_chart(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    *,
    pair: tuple[int, int] = (0, 1),
) -> IntervalRegularizedBinaryCollisionChartState:
    """Lift a planar interval state into an interval regularized binary chart.

    This covers noncollision interval boxes whose selected pair has a positive
    lower distance bound. The complex square-root branch is enclosed by a box
    around the principal Levi-Civita root.
    """

    masses = np.asarray(masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    _third_index(pair)
    positions, velocities, relative_position, relative_velocity = _interval_planar_components(
        state_interval,
        pair,
    )
    branch_certificate = levi_civita_branch_certificate(relative_position)
    return _regularized_interval_chart_from_components(
        masses=masses,
        pair=pair,
        positions=positions,
        velocities=velocities,
        relative_position=relative_position,
        relative_velocity=relative_velocity,
        branch_certificate=branch_certificate,
    )


def planar_interval_to_regularized_binary_collision_chart_atlas(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    *,
    pair: tuple[int, int] = (0, 1),
) -> tuple[IntervalRegularizedBinaryCollisionChartState, ...]:
    """Return certified LC branch boxes covering a planar interval lift when possible."""

    masses = np.asarray(masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    _third_index(pair)
    positions, velocities, relative_position, relative_velocity = _interval_planar_components(
        state_interval,
        pair,
    )
    base_certificate = levi_civita_branch_certificate(relative_position)
    if base_certificate.certified:
        return (
            _regularized_interval_chart_from_components(
                masses,
                pair,
                positions,
                velocities,
                relative_position,
                relative_velocity,
                base_certificate,
            ),
        )

    x_interval = _as_interval(relative_position[0])
    y_interval = _as_interval(relative_position[1])
    if not (x_interval.lower <= 0.0 and y_interval.lower <= 0.0 <= y_interval.upper):
        return (
            _regularized_interval_chart_from_components(
                masses,
                pair,
                positions,
                velocities,
                relative_position,
                relative_velocity,
                base_certificate,
            ),
        )

    charts = []
    if y_interval.upper >= 0.0:
        upper_position = np.array(
            [x_interval, FloatInterval(max(0.0, y_interval.lower), y_interval.upper)],
            dtype=object,
        )
        charts.append(
            _regularized_interval_chart_from_components(
                masses,
                pair,
                positions,
                velocities,
                upper_position,
                relative_velocity,
                LeviCivitaBranchCertificate(
                    branch="atlas_upper_closed_half",
                    certified=True,
                    reason="atlas split covers the upper closed half-plane of the LC square-root chart",
                ),
            )
        )
    if y_interval.lower <= 0.0:
        lower_position = np.array(
            [x_interval, FloatInterval(y_interval.lower, min(0.0, y_interval.upper))],
            dtype=object,
        )
        charts.append(
            _regularized_interval_chart_from_components(
                masses,
                pair,
                positions,
                velocities,
                lower_position,
                relative_velocity,
                LeviCivitaBranchCertificate(
                    branch="atlas_lower_closed_half",
                    certified=True,
                    reason="atlas split covers the lower closed half-plane of the LC square-root chart",
                ),
            )
        )
    return tuple(charts)


def regularized_binary_collision_chart_to_planar(
    state: RegularizedBinaryCollisionChartState,
) -> tuple[Array, Array]:
    """Project the regularized chart to physical planar coordinates for `rho > 0`."""

    rho = state.rho
    if rho == 0.0:
        raise ValueError("physical velocities are singular at exact binary collision")
    first, second = state.pair
    third = state.third_index
    masses = np.asarray(state.masses, dtype=float)
    pair_mass = masses[first] + masses[second]
    relative_position = lc_square(state.z)
    relative_velocity = lc_velocity(state.z, state.z_velocity)

    positions = np.zeros((3, 2), dtype=float)
    velocities = np.zeros_like(positions)
    positions[first] = state.binary_center - (masses[second] / pair_mass) * relative_position
    positions[second] = state.binary_center + (masses[first] / pair_mass) * relative_position
    positions[third] = state.binary_center + state.third_offset
    velocities[first] = state.binary_center_velocity - (masses[second] / pair_mass) * relative_velocity
    velocities[second] = state.binary_center_velocity + (masses[first] / pair_mass) * relative_velocity
    velocities[third] = state.binary_center_velocity + state.third_offset_velocity
    return positions, velocities


def binary_collision_chart_to_planar(state: BinaryCollisionChartState) -> tuple[Array, Array]:
    """Project binary-chart coordinates back to physical positions and velocities."""

    rho = state.rho
    if rho == 0.0:
        raise ValueError("physical velocities are singular at exact binary collision")
    first, second = state.pair
    third = state.third_index
    masses = np.asarray(state.masses, dtype=float)
    pair_mass = masses[first] + masses[second]
    relative_position = lc_square(state.z)
    relative_velocity = lc_velocity(state.z, state.z_velocity)
    binary_center_dot = state.binary_center_velocity / rho
    third_offset_dot = state.third_offset_velocity / rho

    positions = np.zeros((3, 2), dtype=float)
    velocities = np.zeros_like(positions)
    positions[first] = state.binary_center - (masses[second] / pair_mass) * relative_position
    positions[second] = state.binary_center + (masses[first] / pair_mass) * relative_position
    positions[third] = state.binary_center + state.third_offset
    velocities[first] = binary_center_dot - (masses[second] / pair_mass) * relative_velocity
    velocities[second] = binary_center_dot + (masses[first] / pair_mass) * relative_velocity
    velocities[third] = binary_center_dot + third_offset_dot
    return positions, velocities


def _inverse_square_field(vector: Array) -> Array:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("field singularity in binary-collision chart")
    return np.asarray(vector, dtype=float) / norm**3


def analytic_third_body_fields(
    state: RegularizedBinaryCollisionChartState,
) -> tuple[Array, Array]:
    """Return the third-body force-shape fields from the two binary bodies."""

    first, second = state.pair
    masses = state.masses
    pair_mass = masses[first] + masses[second]
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    relative_position = lc_square(state.z)
    from_first_to_third = state.third_offset + alpha * relative_position
    from_second_to_third = state.third_offset - beta * relative_position
    return _inverse_square_field(from_first_to_third), _inverse_square_field(from_second_to_third)


def analytic_coordinate_accelerations(
    state: RegularizedBinaryCollisionChartState,
) -> tuple[Array, Array, Array]:
    """Analytic accelerations `(R_ddot, y_ddot, perturbing r_ddot)` near pair collision."""

    first, second = state.pair
    third = state.third_index
    masses = state.masses
    pair_mass = masses[first] + masses[second]
    field_first, field_second = analytic_third_body_fields(state)
    binary_center_acceleration = (
        masses[third] / pair_mass * (masses[first] * field_first + masses[second] * field_second)
    )
    third_acceleration = -masses[first] * field_first - masses[second] * field_second
    third_offset_acceleration = third_acceleration - binary_center_acceleration
    relative_perturbation = masses[third] * (field_second - field_first)
    return binary_center_acceleration, third_offset_acceleration, relative_perturbation


def coordinate_accelerations_from_planar(state: BinaryCollisionChartState) -> tuple[Array, Array, Array]:
    """Return `(R_ddot, y_ddot, r_ddot)` from projected Newtonian accelerations."""

    positions, _velocities = binary_collision_chart_to_planar(state)
    physical_acceleration = accelerations(positions, state.masses)
    first, second = state.pair
    third = state.third_index
    masses = state.masses
    pair_mass = masses[first] + masses[second]
    binary_center_acceleration = (
        masses[first] * physical_acceleration[first] + masses[second] * physical_acceleration[second]
    ) / pair_mass
    third_offset_acceleration = physical_acceleration[third] - binary_center_acceleration
    relative_acceleration = physical_acceleration[second] - physical_acceleration[first]
    return binary_center_acceleration, third_offset_acceleration, relative_acceleration


def regularized_z_acceleration(
    state: RegularizedBinaryCollisionChartState,
) -> Array:
    """Regular Levi-Civita acceleration including analytic third-body perturbation."""

    rho = state.rho
    matrix = lc_matrix(state.z)
    _center_acceleration, _third_acceleration, relative_perturbation = analytic_coordinate_accelerations(state)
    return 0.5 * state.pair_energy * state.z + 0.25 * rho * matrix.T @ relative_perturbation


def z_acceleration_from_relative_acceleration(z: Array, z_velocity: Array, relative_acceleration: Array) -> Array:
    """Convert physical relative acceleration into Levi-Civita `s`-acceleration."""

    z = np.asarray(z, dtype=float)
    z_velocity = np.asarray(z_velocity, dtype=float)
    relative_acceleration = np.asarray(relative_acceleration, dtype=float)
    rho = float(np.dot(z, z))
    if rho == 0.0:
        raise ValueError("this formula projects away from exact binary collision")
    matrix = lc_matrix(z)
    matrix_prime = lc_matrix(z_velocity)
    q_prime = matrix @ z_velocity
    rho_prime = 2.0 * float(np.dot(z, z_velocity))
    right_hand_side = rho**2 * relative_acceleration + q_prime * rho_prime / rho - matrix_prime @ z_velocity
    return matrix.T @ right_hand_side / (4.0 * rho)


def relative_acceleration_from_z_acceleration(z: Array, z_velocity: Array, z_acceleration: Array) -> Array:
    """Project a Levi-Civita `s`-acceleration to physical relative acceleration."""

    z = np.asarray(z, dtype=float)
    z_velocity = np.asarray(z_velocity, dtype=float)
    z_acceleration = np.asarray(z_acceleration, dtype=float)
    rho = float(np.dot(z, z))
    if rho == 0.0:
        raise ValueError("physical acceleration is singular at exact binary collision")
    matrix = lc_matrix(z)
    matrix_prime = lc_matrix(z_velocity)
    q_prime = matrix @ z_velocity
    rho_prime = 2.0 * float(np.dot(z, z_velocity))
    velocity_prime = (matrix_prime @ z_velocity + matrix @ z_acceleration) / rho
    velocity_prime -= q_prime * rho_prime / rho**2
    return velocity_prime / rho


def binary_collision_chart_rhs(state: BinaryCollisionChartState) -> BinaryCollisionChartDerivative:
    """Right-hand side of the pair-regularized chart in `s` time."""

    rho = state.rho
    if rho == 0.0:
        raise ValueError("full perturbed binary chart RHS is only implemented for rho > 0")
    binary_center_acceleration, third_offset_acceleration, relative_acceleration = coordinate_accelerations_from_planar(
        state
    )
    rho_prime = 2.0 * float(np.dot(state.z, state.z_velocity))
    z_acceleration = z_acceleration_from_relative_acceleration(
        state.z,
        state.z_velocity,
        relative_acceleration,
    )
    binary_center_acceleration_s = (
        (rho_prime / rho) * state.binary_center_velocity + rho**2 * binary_center_acceleration
    )
    third_offset_acceleration_s = (
        (rho_prime / rho) * state.third_offset_velocity + rho**2 * third_offset_acceleration
    )
    return BinaryCollisionChartDerivative(
        z=state.z_velocity,
        z_velocity=z_acceleration,
        binary_center=state.binary_center_velocity,
        binary_center_velocity=binary_center_acceleration_s,
        third_offset=state.third_offset_velocity,
        third_offset_velocity=third_offset_acceleration_s,
        physical_time=rho,
    )


def regularized_binary_collision_chart_rhs(
    state: RegularizedBinaryCollisionChartState,
) -> RegularizedBinaryCollisionChartDerivative:
    """Analytic `s`-time RHS for a separated-third-body binary-collision chart."""

    rho = state.rho
    binary_center_acceleration, third_offset_acceleration, relative_perturbation = analytic_coordinate_accelerations(
        state
    )
    matrix = lc_matrix(state.z)
    z_acceleration = 0.5 * state.pair_energy * state.z + 0.25 * rho * matrix.T @ relative_perturbation
    pair_energy_derivative = float((matrix @ state.z_velocity).dot(relative_perturbation))
    return RegularizedBinaryCollisionChartDerivative(
        z=state.z_velocity,
        z_velocity=z_acceleration,
        pair_energy=pair_energy_derivative,
        binary_center=rho * state.binary_center_velocity,
        binary_center_velocity=rho * binary_center_acceleration,
        third_offset=rho * state.third_offset_velocity,
        third_offset_velocity=rho * third_offset_acceleration,
        physical_time=rho,
    )


def planar_accelerations_from_chart_rhs(
    state: BinaryCollisionChartState,
    derivative: BinaryCollisionChartDerivative,
) -> Array:
    """Project an `s`-time chart derivative back to physical body accelerations."""

    rho = state.rho
    if rho == 0.0:
        raise ValueError("physical acceleration is singular at exact binary collision")
    first, second = state.pair
    third = state.third_index
    masses = state.masses
    pair_mass = masses[first] + masses[second]
    rho_prime = 2.0 * float(np.dot(state.z, state.z_velocity))
    relative_acceleration = relative_acceleration_from_z_acceleration(
        state.z,
        state.z_velocity,
        derivative.z_velocity,
    )
    binary_center_acceleration = (
        derivative.binary_center_velocity - (rho_prime / rho) * state.binary_center_velocity
    ) / rho**2
    third_offset_acceleration = (
        derivative.third_offset_velocity - (rho_prime / rho) * state.third_offset_velocity
    ) / rho**2
    physical_acceleration = np.zeros((3, 2), dtype=float)
    physical_acceleration[first] = binary_center_acceleration - (masses[second] / pair_mass) * relative_acceleration
    physical_acceleration[second] = binary_center_acceleration + (masses[first] / pair_mass) * relative_acceleration
    physical_acceleration[third] = binary_center_acceleration + third_offset_acceleration
    return physical_acceleration


def planar_accelerations_from_regularized_chart_rhs(
    state: RegularizedBinaryCollisionChartState,
    derivative: RegularizedBinaryCollisionChartDerivative,
) -> Array:
    """Project regularized chart derivatives to physical accelerations for `rho > 0`."""

    rho = state.rho
    if rho == 0.0:
        raise ValueError("physical acceleration is singular at exact binary collision")
    first, second = state.pair
    third = state.third_index
    masses = state.masses
    pair_mass = masses[first] + masses[second]
    relative_acceleration = relative_acceleration_from_z_acceleration(
        state.z,
        state.z_velocity,
        derivative.z_velocity,
    )
    binary_center_acceleration = derivative.binary_center_velocity / rho
    third_offset_acceleration = derivative.third_offset_velocity / rho
    physical_acceleration = np.zeros((3, 2), dtype=float)
    physical_acceleration[first] = binary_center_acceleration - (masses[second] / pair_mass) * relative_acceleration
    physical_acceleration[second] = binary_center_acceleration + (masses[first] / pair_mass) * relative_acceleration
    physical_acceleration[third] = binary_center_acceleration + third_offset_acceleration
    return physical_acceleration


def pair_energy_constraint(state: RegularizedBinaryCollisionChartState) -> float:
    """Return `2|z'|^2 - M - rho*h`, which is preserved by the regularized RHS."""

    first, second = state.pair
    pair_mass = state.masses[first] + state.masses[second]
    return float(2.0 * np.dot(state.z_velocity, state.z_velocity) - pair_mass - state.rho * state.pair_energy)


def pair_energy_constraint_derivative(
    state: RegularizedBinaryCollisionChartState,
    derivative: RegularizedBinaryCollisionChartDerivative,
) -> float:
    rho_prime = 2.0 * float(np.dot(state.z, state.z_velocity))
    return float(
        4.0 * np.dot(state.z_velocity, derivative.z_velocity)
        - rho_prime * state.pair_energy
        - state.rho * derivative.pair_energy
    )
