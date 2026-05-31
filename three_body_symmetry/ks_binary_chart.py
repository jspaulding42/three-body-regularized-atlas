"""Kustaanheimo-Stiefel chart for spatial binary regularization.

This is the isolated spatial binary analogue of the planar Levi-Civita chart.
For a relative coordinate ``q`` in R^3, write ``q = K(u)`` with ``u`` in R^4 and
``|q| = |u|^2``. With the same time scaling ``dt/ds = |u|^2`` used by the planar
chart, the Kepler binary equation lifts to ``u'' = (h / 2) u`` on the horizontal
KS gauge. This module deliberately certifies only the local pair model; the full
three-body spatial binary handoff still needs a branch/gauge atlas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .intervals import FloatInterval, interval_array_contains_point
from .series import scalar_series_product


Array = np.ndarray


@dataclass(frozen=True)
class KustaanheimoStiefelDerivative:
    """Regularized isolated-binary KS right-hand side."""

    u: Array
    u_velocity: Array
    physical_time: float


@dataclass(frozen=True)
class SpatialKSBinaryChartState:
    """Point state for one spatial KS-regularized three-body binary chart."""

    masses: Array
    pair: tuple[int, int]
    u: Array
    u_velocity: Array
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
        return float(np.dot(self.u, self.u))


@dataclass(frozen=True)
class SpatialKSBinaryChartDerivative:
    """Regularized spatial KS three-body binary RHS."""

    u: Array
    u_velocity: Array
    pair_energy: float
    binary_center: Array
    binary_center_velocity: Array
    third_offset: Array
    third_offset_velocity: Array
    physical_time: float


@dataclass(frozen=True)
class KSBranchCertificate:
    """Sufficient certificate for one algebraic KS position branch."""

    branch: str
    certified: bool
    reason: str


@dataclass(frozen=True)
class IntervalKSPositionChart:
    """Interval box for one gauge-fixed KS position branch."""

    relative_position: Array
    u: Array
    branch_certificate: KSBranchCertificate

    @property
    def certified(self) -> bool:
        return bool(self.branch_certificate.certified)

    def contains_point(self, relative_position: Array) -> bool:
        relative_position = np.asarray(relative_position, dtype=float)
        if relative_position.shape != (3,):
            return False
        if not interval_array_contains_point(self.relative_position, relative_position):
            return False
        try:
            point_lift = ks_lift_position_on_branch(
                relative_position,
                branch=self.branch_certificate.branch,
            )
        except ValueError:
            return False
        return interval_array_contains_point(self.u, point_lift)


@dataclass(frozen=True)
class IntervalKSStateChart:
    """Interval KS lift of a spatial relative position/velocity box."""

    relative_position: Array
    relative_velocity: Array
    u: Array
    u_velocity: Array
    energy: FloatInterval
    horizontal_constraint: FloatInterval
    mu: float
    branch_certificate: KSBranchCertificate

    @property
    def horizontal_constraint_contains_zero(self) -> bool:
        return bool(self.horizontal_constraint.lower <= 0.0 <= self.horizontal_constraint.upper)

    @property
    def certified(self) -> bool:
        return bool(
            self.branch_certificate.certified
            and self.mu > 0.0
            and self.horizontal_constraint_contains_zero
        )

    def contains_point(self, relative_position: Array, relative_velocity: Array, mu: float) -> bool:
        relative_position = np.asarray(relative_position, dtype=float)
        relative_velocity = np.asarray(relative_velocity, dtype=float)
        if relative_position.shape != (3,) or relative_velocity.shape != (3,):
            return False
        if abs(float(mu) - self.mu) > 0.0:
            return False
        if not interval_array_contains_point(self.relative_position, relative_position):
            return False
        if not interval_array_contains_point(self.relative_velocity, relative_velocity):
            return False
        try:
            point_u, point_u_velocity, point_energy = physical_to_ks_on_branch(
                relative_position,
                relative_velocity,
                mu,
                branch=self.branch_certificate.branch,
            )
        except ValueError:
            return False
        return bool(
            interval_array_contains_point(self.u, point_u)
            and interval_array_contains_point(self.u_velocity, point_u_velocity)
            and self.energy.lower <= point_energy <= self.energy.upper
        )


@dataclass(frozen=True)
class KustaanheimoStiefelChart:
    """Taylor chart for the regularized isolated spatial binary."""

    u: Array
    u_velocity: Array
    physical_time: Array
    energy: float

    @property
    def order(self) -> int:
        return int(self.u.shape[0] - 1)

    def u_at(self, s_value: float) -> Array:
        return _evaluate(self.u, s_value)

    def u_velocity_at(self, s_value: float) -> Array:
        return _evaluate(self.u_velocity, s_value)

    def physical_time_at(self, s_value: float) -> float:
        return float(_evaluate(self.physical_time[:, None], s_value)[0])

    def relative_position_at(self, s_value: float) -> Array:
        return ks_project(self.u_at(s_value))

    def relative_velocity_at(self, s_value: float) -> Array:
        return ks_velocity(self.u_at(s_value), self.u_velocity_at(s_value))

    def relative_state_at(self, s_value: float) -> Array:
        return np.concatenate(
            [self.relative_position_at(s_value), self.relative_velocity_at(s_value)]
        )


def _evaluate(coefficients: Array, value: float) -> Array:
    out = np.zeros(coefficients.shape[1:], dtype=float)
    for coefficient in coefficients[::-1]:
        out = out * value + coefficient
    return out


def ks_project(u: Array) -> Array:
    """Project a KS coordinate ``u`` in R^4 to a relative vector in R^3."""

    a, b, c, d = np.asarray(u, dtype=float)
    return np.array(
        [
            a * a - b * b - c * c + d * d,
            2.0 * (a * b - c * d),
            2.0 * (a * c + b * d),
        ],
        dtype=float,
    )


def ks_matrix(u: Array) -> Array:
    """Jacobian of ``ks_project``."""

    a, b, c, d = np.asarray(u, dtype=float)
    return np.array(
        [
            [2.0 * a, -2.0 * b, -2.0 * c, 2.0 * d],
            [2.0 * b, 2.0 * a, -2.0 * d, -2.0 * c],
            [2.0 * c, 2.0 * d, 2.0 * a, 2.0 * b],
        ],
        dtype=float,
    )


def ks_vertical_generator(u: Array) -> Array:
    """Infinitesimal generator of the KS gauge fiber through ``u``."""

    a, b, c, d = np.asarray(u, dtype=float)
    return np.array([-d, c, -b, a], dtype=float)


def ks_gauge_rotate(u: Array, angle: float) -> Array:
    """Rotate along the KS gauge fiber while preserving ``ks_project(u)``."""

    u = np.asarray(u, dtype=float)
    angle = float(angle)
    return np.cos(angle) * u + np.sin(angle) * ks_vertical_generator(u)


def ks_velocity(u: Array, u_velocity: Array) -> Array:
    """Project the regularized velocity to physical relative velocity."""

    u = np.asarray(u, dtype=float)
    u_velocity = np.asarray(u_velocity, dtype=float)
    rho = float(np.dot(u, u))
    if rho == 0.0:
        raise ValueError("physical velocity is singular at binary collision")
    return ks_matrix(u) @ u_velocity / rho


def ks_horizontal_constraint(u: Array, u_velocity: Array) -> float:
    """Return the KS fiber momentum; horizontal lifts have value zero."""

    return float(np.dot(ks_vertical_generator(u), np.asarray(u_velocity, dtype=float)))


def spatial_to_ks_binary_chart(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    pair: tuple[int, int],
) -> SpatialKSBinaryChartState:
    """Lift a noncollision spatial three-body state into one KS binary chart."""

    positions, velocities, masses = _validate_spatial_three_body(positions, velocities, masses)
    first, second = pair
    third = _third_index(pair)
    pair_mass = float(masses[first] + masses[second])
    relative_position = positions[second] - positions[first]
    relative_velocity = velocities[second] - velocities[first]
    u, u_velocity, pair_energy = physical_to_ks(relative_position, relative_velocity, pair_mass)
    binary_center = (
        masses[first] * positions[first] + masses[second] * positions[second]
    ) / pair_mass
    binary_center_velocity = (
        masses[first] * velocities[first] + masses[second] * velocities[second]
    ) / pair_mass
    return SpatialKSBinaryChartState(
        masses=masses,
        pair=(int(first), int(second)),
        u=u,
        u_velocity=u_velocity,
        pair_energy=float(pair_energy),
        binary_center=binary_center,
        binary_center_velocity=binary_center_velocity,
        third_offset=positions[third] - binary_center,
        third_offset_velocity=velocities[third] - binary_center_velocity,
    )


def ks_binary_chart_to_spatial(state: SpatialKSBinaryChartState) -> tuple[Array, Array]:
    """Project a spatial KS binary chart back to inertial positions/velocities."""

    rho = state.rho
    if rho == 0.0:
        raise ValueError("physical velocities are singular at exact binary collision")
    first, second = state.pair
    third = state.third_index
    masses = np.asarray(state.masses, dtype=float)
    pair_mass = masses[first] + masses[second]
    relative_position = ks_project(state.u)
    relative_velocity = ks_velocity(state.u, state.u_velocity)
    positions = np.zeros((3, 3), dtype=float)
    velocities = np.zeros((3, 3), dtype=float)
    positions[first] = state.binary_center - (masses[second] / pair_mass) * relative_position
    positions[second] = state.binary_center + (masses[first] / pair_mass) * relative_position
    positions[third] = state.binary_center + state.third_offset
    velocities[first] = state.binary_center_velocity - (masses[second] / pair_mass) * relative_velocity
    velocities[second] = state.binary_center_velocity + (masses[first] / pair_mass) * relative_velocity
    velocities[third] = state.binary_center_velocity + state.third_offset_velocity
    return positions, velocities


def ks_analytic_coordinate_accelerations(
    state: SpatialKSBinaryChartState,
) -> tuple[Array, Array, Array]:
    """Analytic `(R_ddot, y_ddot, perturbing q_ddot)` near binary collision."""

    first, second = state.pair
    third = state.third_index
    masses = np.asarray(state.masses, dtype=float)
    pair_mass = masses[first] + masses[second]
    field_first, field_second = ks_analytic_third_body_fields(state)
    binary_center_acceleration = (
        masses[third] / pair_mass * (masses[first] * field_first + masses[second] * field_second)
    )
    third_acceleration = -masses[first] * field_first - masses[second] * field_second
    third_offset_acceleration = third_acceleration - binary_center_acceleration
    relative_perturbation = masses[third] * (field_second - field_first)
    return binary_center_acceleration, third_offset_acceleration, relative_perturbation


def ks_analytic_third_body_fields(
    state: SpatialKSBinaryChartState,
) -> tuple[Array, Array]:
    """Return third-body inverse-square fields from the two binary bodies."""

    first, second = state.pair
    masses = np.asarray(state.masses, dtype=float)
    pair_mass = masses[first] + masses[second]
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    relative_position = ks_project(state.u)
    from_first_to_third = state.third_offset + alpha * relative_position
    from_second_to_third = state.third_offset - beta * relative_position
    return _inverse_square_field(from_first_to_third), _inverse_square_field(from_second_to_third)


def regularized_ks_binary_u_acceleration(state: SpatialKSBinaryChartState) -> Array:
    """Regular KS acceleration including analytic third-body perturbation."""

    rho = state.rho
    _center_acc, _third_acc, relative_perturbation = ks_analytic_coordinate_accelerations(state)
    return (
        0.5 * float(state.pair_energy) * np.asarray(state.u, dtype=float)
        + 0.25 * rho * ks_matrix(state.u).T @ relative_perturbation
    )


def regularized_ks_binary_chart_rhs(
    state: SpatialKSBinaryChartState,
) -> SpatialKSBinaryChartDerivative:
    """Analytic `s`-time RHS for a separated-third-body spatial binary chart."""

    rho = state.rho
    center_acceleration, third_offset_acceleration, relative_perturbation = (
        ks_analytic_coordinate_accelerations(state)
    )
    u_acceleration = regularized_ks_binary_u_acceleration(state)
    pair_energy_derivative = float((ks_matrix(state.u) @ state.u_velocity).dot(relative_perturbation))
    return SpatialKSBinaryChartDerivative(
        u=np.asarray(state.u_velocity, dtype=float),
        u_velocity=u_acceleration,
        pair_energy=pair_energy_derivative,
        binary_center=rho * np.asarray(state.binary_center_velocity, dtype=float),
        binary_center_velocity=rho * center_acceleration,
        third_offset=rho * np.asarray(state.third_offset_velocity, dtype=float),
        third_offset_velocity=rho * third_offset_acceleration,
        physical_time=rho,
    )


def relative_acceleration_from_ks_acceleration(
    u: Array,
    u_velocity: Array,
    u_acceleration: Array,
) -> Array:
    """Project a KS `s`-acceleration to physical relative acceleration."""

    u = np.asarray(u, dtype=float)
    u_velocity = np.asarray(u_velocity, dtype=float)
    u_acceleration = np.asarray(u_acceleration, dtype=float)
    rho = float(np.dot(u, u))
    if rho == 0.0:
        raise ValueError("physical acceleration is singular at exact binary collision")
    q_prime = ks_matrix(u) @ u_velocity
    q_second = ks_matrix(u_velocity) @ u_velocity + ks_matrix(u) @ u_acceleration
    rho_prime = 2.0 * float(np.dot(u, u_velocity))
    return q_second / rho**2 - q_prime * rho_prime / rho**3


def spatial_accelerations_from_ks_binary_rhs(
    state: SpatialKSBinaryChartState,
    derivative: SpatialKSBinaryChartDerivative,
) -> Array:
    """Project regularized spatial KS derivatives to physical accelerations."""

    rho = state.rho
    if rho == 0.0:
        raise ValueError("physical acceleration is singular at exact binary collision")
    first, second = state.pair
    third = state.third_index
    masses = np.asarray(state.masses, dtype=float)
    pair_mass = masses[first] + masses[second]
    relative_acceleration = relative_acceleration_from_ks_acceleration(
        state.u,
        state.u_velocity,
        derivative.u_velocity,
    )
    binary_center_acceleration = derivative.binary_center_velocity / rho
    third_offset_acceleration = derivative.third_offset_velocity / rho
    physical_acceleration = np.zeros((3, 3), dtype=float)
    physical_acceleration[first] = (
        binary_center_acceleration - (masses[second] / pair_mass) * relative_acceleration
    )
    physical_acceleration[second] = (
        binary_center_acceleration + (masses[first] / pair_mass) * relative_acceleration
    )
    physical_acceleration[third] = binary_center_acceleration + third_offset_acceleration
    return physical_acceleration


def ks_pair_energy_constraint(state: SpatialKSBinaryChartState) -> float:
    """Return `2|u'|^2 - M - |u|^2 h` for the KS binary pair."""

    first, second = state.pair
    pair_mass = float(state.masses[first] + state.masses[second])
    return float(
        2.0 * np.dot(state.u_velocity, state.u_velocity)
        - pair_mass
        - state.rho * state.pair_energy
    )


def ks_pair_energy_constraint_derivative(
    state: SpatialKSBinaryChartState,
    derivative: SpatialKSBinaryChartDerivative,
) -> float:
    """Derivative of the KS pair-energy constraint along the regularized RHS."""

    rho_prime = 2.0 * float(np.dot(state.u, state.u_velocity))
    return float(
        4.0 * np.dot(state.u_velocity, derivative.u_velocity)
        - rho_prime * state.pair_energy
        - state.rho * derivative.pair_energy
    )


def physical_to_ks(
    relative_position: Array,
    relative_velocity: Array,
    mu: float,
) -> tuple[Array, Array, float]:
    """Lift a non-collision spatial Kepler state into horizontal KS variables."""

    if mu <= 0.0:
        raise ValueError("mu must be positive")
    q = np.asarray(relative_position, dtype=float)
    v = np.asarray(relative_velocity, dtype=float)
    if q.shape != (3,) or v.shape != (3,):
        raise ValueError("relative_position and relative_velocity must have shape (3,)")
    radius = float(np.linalg.norm(q))
    if radius == 0.0:
        raise ValueError("cannot lift a collision state from physical velocity")
    u = ks_lift_position(q)
    u_velocity = 0.25 * ks_matrix(u).T @ v
    energy = 0.5 * float(np.dot(v, v)) - float(mu) / radius
    return u, u_velocity, energy


def physical_to_ks_on_branch(
    relative_position: Array,
    relative_velocity: Array,
    mu: float,
    *,
    branch: str,
) -> tuple[Array, Array, float]:
    """Lift a non-collision spatial Kepler state on a chosen KS branch."""

    if mu <= 0.0:
        raise ValueError("mu must be positive")
    q = np.asarray(relative_position, dtype=float)
    v = np.asarray(relative_velocity, dtype=float)
    if q.shape != (3,) or v.shape != (3,):
        raise ValueError("relative_position and relative_velocity must have shape (3,)")
    radius = float(np.linalg.norm(q))
    if radius == 0.0:
        raise ValueError("cannot lift a collision state from physical velocity")
    u = ks_lift_position_on_branch(q, branch=branch)
    u_velocity = 0.25 * ks_matrix(u).T @ v
    energy = 0.5 * float(np.dot(v, v)) - float(mu) / radius
    return u, u_velocity, energy


def ks_lift_position(relative_position: Array) -> Array:
    """Return one explicit KS lift for a nonzero relative position."""

    q = np.asarray(relative_position, dtype=float)
    if q.shape != (3,):
        raise ValueError("relative_position must have shape (3,)")
    radius = float(np.linalg.norm(q))
    if radius == 0.0:
        raise ValueError("cannot lift the collision point from physical position")
    if radius + q[0] > 1.0e-14 * max(1.0, radius):
        return ks_lift_position_on_branch(q, branch="positive_x")
    return ks_lift_position_on_branch(q, branch="negative_x")


def ks_lift_position_on_branch(relative_position: Array, *, branch: str) -> Array:
    """Return the gauge-fixed KS lift on one named algebraic branch."""

    q = np.asarray(relative_position, dtype=float)
    if q.shape != (3,):
        raise ValueError("relative_position must have shape (3,)")
    radius = float(np.linalg.norm(q))
    if radius == 0.0:
        raise ValueError("cannot lift the collision point from physical position")
    x, y, z = q
    if branch == "positive_x":
        if radius + x <= 0.0:
            raise ValueError("positive_x KS branch is singular on the negative x-axis")
        a = float(np.sqrt(0.5 * (radius + x)))
        b = y / (2.0 * a)
        c = z / (2.0 * a)
        d = 0.0
    elif branch == "negative_x":
        if radius - x <= 0.0:
            raise ValueError("negative_x KS branch is singular on the positive x-axis")
        b = float(np.sqrt(0.5 * (radius - x)))
        a = y / (2.0 * b)
        c = 0.0
        d = z / (2.0 * b)
    else:
        raise ValueError("branch must be 'positive_x' or 'negative_x'")
    return np.array([a, b, c, d], dtype=float)


def ks_interval_position_chart(
    relative_position: Array,
    *,
    branch: str,
) -> IntervalKSPositionChart:
    """Lift a spatial relative-position interval into one KS gauge chart."""

    relative_position = _coerce_interval_vector(relative_position, name="relative_position")
    radius = _interval_norm(relative_position)
    x, y, z = relative_position
    if branch == "positive_x":
        denominator = (radius + x).scale(0.5)
        if denominator.lower <= 0.0:
            return IntervalKSPositionChart(
                relative_position=relative_position,
                u=_uncertified_interval_lift(),
                branch_certificate=KSBranchCertificate(
                    branch=branch,
                    certified=False,
                    reason="relative-position interval may meet the positive_x branch singularity",
                ),
            )
        a = _interval_sqrt_nonnegative(denominator)
        b = y / a.scale(2.0)
        c = z / a.scale(2.0)
        u = np.array([a, b, c, FloatInterval.point(0.0)], dtype=object)
    elif branch == "negative_x":
        denominator = (radius - x).scale(0.5)
        if denominator.lower <= 0.0:
            return IntervalKSPositionChart(
                relative_position=relative_position,
                u=_uncertified_interval_lift(),
                branch_certificate=KSBranchCertificate(
                    branch=branch,
                    certified=False,
                    reason="relative-position interval may meet the negative_x branch singularity",
                ),
            )
        b = _interval_sqrt_nonnegative(denominator)
        a = y / b.scale(2.0)
        c = FloatInterval.point(0.0)
        d = z / b.scale(2.0)
        u = np.array([a, b, c, d], dtype=object)
    else:
        raise ValueError("branch must be 'positive_x' or 'negative_x'")
    return IntervalKSPositionChart(
        relative_position=relative_position,
        u=u,
        branch_certificate=KSBranchCertificate(
            branch=branch,
            certified=True,
            reason=f"relative-position interval stays inside the {branch} KS chart",
        ),
    )


def ks_interval_position_atlas(relative_position: Array) -> tuple[IntervalKSPositionChart, ...]:
    """Return all certified gauge-fixed KS position charts for an interval box."""

    charts = tuple(
        chart
        for chart in (
            ks_interval_position_chart(relative_position, branch="positive_x"),
            ks_interval_position_chart(relative_position, branch="negative_x"),
        )
        if chart.certified
    )
    return charts


def ks_interval_state_chart(
    relative_position: Array,
    relative_velocity: Array,
    mu: float,
    *,
    branch: str,
) -> IntervalKSStateChart:
    """Lift spatial relative position/velocity intervals into one KS chart."""

    if mu <= 0.0:
        raise ValueError("mu must be positive")
    position_chart = ks_interval_position_chart(relative_position, branch=branch)
    relative_velocity = _coerce_interval_vector(relative_velocity, name="relative_velocity")
    if not position_chart.certified:
        return IntervalKSStateChart(
            relative_position=position_chart.relative_position,
            relative_velocity=relative_velocity,
            u=position_chart.u,
            u_velocity=_uncertified_interval_lift(),
            energy=FloatInterval.point(0.0),
            horizontal_constraint=FloatInterval.point(0.0),
            mu=float(mu),
            branch_certificate=position_chart.branch_certificate,
        )
    radius = _interval_norm(position_chart.relative_position)
    if radius.lower <= 0.0:
        branch_certificate = KSBranchCertificate(
            branch=branch,
            certified=False,
            reason="relative-position interval may contain binary collision",
        )
        return IntervalKSStateChart(
            relative_position=position_chart.relative_position,
            relative_velocity=relative_velocity,
            u=position_chart.u,
            u_velocity=_uncertified_interval_lift(),
            energy=FloatInterval.point(0.0),
            horizontal_constraint=FloatInterval.point(0.0),
            mu=float(mu),
            branch_certificate=branch_certificate,
        )
    u_velocity = _interval_ks_transpose_times_velocity(position_chart.u, relative_velocity)
    kinetic = _interval_norm_squared(relative_velocity).scale(0.5)
    potential = FloatInterval.point(float(mu)) / radius
    energy = kinetic - potential
    horizontal_constraint = _interval_dot(
        _interval_vertical_generator(position_chart.u),
        u_velocity,
    )
    return IntervalKSStateChart(
        relative_position=position_chart.relative_position,
        relative_velocity=relative_velocity,
        u=position_chart.u,
        u_velocity=u_velocity,
        energy=energy,
        horizontal_constraint=horizontal_constraint,
        mu=float(mu),
        branch_certificate=position_chart.branch_certificate,
    )


def ks_interval_state_atlas(
    relative_position: Array,
    relative_velocity: Array,
    mu: float,
) -> tuple[IntervalKSStateChart, ...]:
    """Return all certified KS interval state charts for a relative state box."""

    return tuple(
        chart
        for chart in (
            ks_interval_state_chart(relative_position, relative_velocity, mu, branch="positive_x"),
            ks_interval_state_chart(relative_position, relative_velocity, mu, branch="negative_x"),
        )
        if chart.certified
    )


def ks_mu(u: Array, u_velocity: Array, energy: float) -> float:
    """Recover the Kepler parameter from the KS energy constraint."""

    u = np.asarray(u, dtype=float)
    u_velocity = np.asarray(u_velocity, dtype=float)
    rho = float(np.dot(u, u))
    return float(2.0 * np.dot(u_velocity, u_velocity) - float(energy) * rho)


def ks_regularized_rhs(
    u: Array,
    u_velocity: Array,
    energy: float,
) -> KustaanheimoStiefelDerivative:
    """Return the regularized isolated-binary KS vector field."""

    u = np.asarray(u, dtype=float)
    u_velocity = np.asarray(u_velocity, dtype=float)
    if u.shape != (4,) or u_velocity.shape != (4,):
        raise ValueError("u and u_velocity must have shape (4,)")
    return KustaanheimoStiefelDerivative(
        u=u_velocity,
        u_velocity=0.5 * float(energy) * u,
        physical_time=float(np.dot(u, u)),
    )


def projected_acceleration_from_ks_lift(
    u: Array,
    u_velocity: Array,
    energy: float,
) -> Array:
    """Project the regularized KS equation back to physical acceleration."""

    u = np.asarray(u, dtype=float)
    u_velocity = np.asarray(u_velocity, dtype=float)
    rho = float(np.dot(u, u))
    if rho == 0.0:
        raise ValueError("physical acceleration is singular at binary collision")
    derivative = ks_regularized_rhs(u, u_velocity, energy)
    q_prime = ks_matrix(u) @ u_velocity
    q_second = ks_matrix(u_velocity) @ u_velocity + ks_matrix(u) @ derivative.u_velocity
    rho_prime = 2.0 * float(np.dot(u, u_velocity))
    return q_second / rho**2 - q_prime * rho_prime / rho**3


def construct_ks_chart(
    u0: Array,
    u_velocity0: Array,
    energy: float,
    *,
    order: int,
) -> KustaanheimoStiefelChart:
    """Construct a Taylor chart for the regular KS oscillator."""

    if order < 1:
        raise ValueError("order must be at least 1")
    u0 = np.asarray(u0, dtype=float)
    u_velocity0 = np.asarray(u_velocity0, dtype=float)
    if u0.shape != (4,) or u_velocity0.shape != (4,):
        raise ValueError("u0 and u_velocity0 must have shape (4,)")
    u = np.zeros((order + 1, 4), dtype=float)
    u_velocity = np.zeros_like(u)
    physical_time = np.zeros(order + 1, dtype=float)
    u[0] = u0
    u_velocity[0] = u_velocity0

    for n in range(order):
        u[n + 1] = u_velocity[n] / (n + 1)
        u_velocity[n + 1] = (0.5 * float(energy) * u[n]) / (n + 1)
        rho = np.zeros(n + 1, dtype=float)
        for axis in range(4):
            rho += scalar_series_product(u[: n + 1, axis], u[: n + 1, axis], n)
        physical_time[n + 1] = rho[n] / (n + 1)

    return KustaanheimoStiefelChart(
        u=u,
        u_velocity=u_velocity,
        physical_time=physical_time,
        energy=float(energy),
    )


def integrate_relative_kepler_spatial(
    relative_position: Array,
    relative_velocity: Array,
    mu: float,
    time: float,
) -> Array:
    """Reference physical-time integrator for the spatial relative Kepler problem."""

    if mu <= 0.0:
        raise ValueError("mu must be positive")
    q0 = np.asarray(relative_position, dtype=float)
    v0 = np.asarray(relative_velocity, dtype=float)
    if q0.shape != (3,) or v0.shape != (3,):
        raise ValueError("relative_position and relative_velocity must have shape (3,)")
    state0 = np.concatenate([q0, v0])

    def rhs(_time: float, state: Array) -> Array:
        q = state[:3]
        v = state[3:]
        radius = np.linalg.norm(q)
        if radius == 0.0:
            raise FloatingPointError("collision in physical Kepler integrator")
        acceleration = -float(mu) * q / radius**3
        return np.concatenate([v, acceleration])

    solution = solve_ivp(rhs, (0.0, time), state0, method="DOP853", rtol=1e-12, atol=1e-14)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1]


def _third_index(pair: tuple[int, int]) -> int:
    remaining = ({0, 1, 2} - set(pair))
    if len(remaining) != 1 or pair[0] == pair[1]:
        raise ValueError("pair must contain two distinct body indices from {0, 1, 2}")
    return remaining.pop()


def _validate_spatial_three_body(
    positions: Array,
    velocities: Array,
    masses: Array,
) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.shape != (3, 3) or velocities.shape != (3, 3):
        raise ValueError("positions and velocities must have shape (3, 3)")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    return positions, velocities, masses


def _inverse_square_field(vector: Array) -> Array:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("field singularity in KS binary chart")
    return vector / norm**3


def _coerce_interval(value: object) -> FloatInterval:
    if isinstance(value, FloatInterval):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return FloatInterval(float(value[0]), float(value[1]))
    return FloatInterval.point(float(value))


def _coerce_interval_vector(values: Array, *, name: str) -> Array:
    values = np.asarray(values, dtype=object)
    if values.shape == (3, 2):
        return np.array(
            [
                FloatInterval(float(values[index, 0]), float(values[index, 1]))
                for index in range(3)
            ],
            dtype=object,
        )
    if values.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return np.array([_coerce_interval(value) for value in values], dtype=object)


def _interval_square_bounds(value: FloatInterval) -> FloatInterval:
    squares = (value.lower * value.lower, value.upper * value.upper)
    if value.lower <= 0.0 <= value.upper:
        lower = 0.0
    else:
        lower = max(0.0, float(np.nextafter(min(squares), -np.inf)))
    return FloatInterval(lower, float(np.nextafter(max(squares), np.inf)))


def _interval_norm_squared(vector: Array) -> FloatInterval:
    total = FloatInterval.point(0.0)
    for value in np.asarray(vector, dtype=object).reshape(-1):
        total = total + _interval_square_bounds(_coerce_interval(value))
    return total


def _interval_norm(vector: Array) -> FloatInterval:
    total = _interval_norm_squared(vector)
    return FloatInterval(
        0.0 if total.lower <= 0.0 else float(np.nextafter(np.sqrt(total.lower), -np.inf)),
        float(np.nextafter(np.sqrt(max(total.upper, 0.0)), np.inf)),
    )


def _interval_sqrt_nonnegative(value: FloatInterval) -> FloatInterval:
    if value.upper < 0.0:
        raise ValueError("cannot take square root of a negative interval")
    lower = max(0.0, value.lower)
    return FloatInterval(
        float(np.nextafter(np.sqrt(lower), -np.inf)) if lower > 0.0 else 0.0,
        float(np.nextafter(np.sqrt(max(value.upper, 0.0)), np.inf)),
    )


def _uncertified_interval_lift() -> Array:
    return np.array([FloatInterval.point(0.0) for _axis in range(4)], dtype=object)


def _interval_ks_transpose_times_velocity(u: Array, velocity: Array) -> Array:
    a, b, c, d = np.asarray(u, dtype=object)
    vx, vy, vz = np.asarray(velocity, dtype=object)
    return np.array(
        [
            (a * vx + b * vy + c * vz).scale(0.5),
            (b.scale(-1.0) * vx + a * vy + d * vz).scale(0.5),
            (c.scale(-1.0) * vx + d.scale(-1.0) * vy + a * vz).scale(0.5),
            (d * vx + c.scale(-1.0) * vy + b * vz).scale(0.5),
        ],
        dtype=object,
    )


def _interval_vertical_generator(u: Array) -> Array:
    a, b, c, d = np.asarray(u, dtype=object)
    return np.array([d.scale(-1.0), c, b.scale(-1.0), a], dtype=object)


def _interval_dot(left: Array, right: Array) -> FloatInterval:
    total = FloatInterval.point(0.0)
    for left_value, right_value in zip(np.asarray(left, dtype=object), np.asarray(right, dtype=object)):
        total = total + _coerce_interval(left_value) * _coerce_interval(right_value)
    return total
