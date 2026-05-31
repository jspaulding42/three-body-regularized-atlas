"""Hybrid continuation using ordinary and regularized binary Taylor charts."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

from .binary_chart import (
    IntervalRegularizedBinaryCollisionChartState,
    RegularizedBinaryCollisionChartState,
    planar_interval_to_regularized_binary_collision_chart_atlas,
    planar_interval_to_regularized_binary_collision_chart,
    planar_to_regularized_binary_collision_chart,
    regularized_binary_collision_chart_to_planar,
)
from .binary_series import (
    IntervalRegularizedBinaryTaylorSolution,
    construct_interval_regularized_binary_taylor_solution,
    construct_interval_regularized_binary_taylor_solution_from_intervals,
    construct_regularized_binary_taylor_solution,
    regularized_rhs_interval_coefficients,
)
from .continuation import truncation_indicator
from .error_budget import (
    LohnerOrdinaryPropagatedEnclosure,
    OrdinarySetPropagatedEnclosure,
    PropagatedErrorBudget,
    PropagatedIntervalEnclosure,
    propagate_lohner_ordinary_set_enclosures,
    propagate_ordinary_set_enclosures,
    propagate_error_budget,
    propagate_interval_enclosures,
)
from .global_invariants import (
    AngularMomentumConservationCertificate,
    CenterOfMassMotionCertificate,
    EnergyConservationCertificate,
    LinearMomentumConservationCertificate,
    certify_interval_center_of_mass_motion,
    certify_interval_centered_angular_momentum_conservation,
    certify_interval_linear_momentum_conservation,
    certify_interval_total_energy_conservation,
)
from .intervals import (
    FloatInterval,
    interval_contains_zero,
    interval_array_as_tuples,
    interval_array_series_eval,
    interval_coefficients_as_tuples,
    interval_polynomial_eval,
    interval_polyder,
    interval_series_add,
    interval_series_power,
    interval_series_product,
    interval_sign,
    subtract_interval_from_constant_term,
    zero_interval,
)
from .series import (
    IntervalTaylorSolution,
    acceleration_interval_coefficients,
    construct_interval_taylor_solution,
    construct_interval_taylor_solution_from_intervals,
    construct_taylor_solution,
)
from .tail_bounds import (
    TailBoundCertificate,
    ordinary_interval_union_cauchy_majorant_tail_certificate,
    ordinary_taylor_tail_certificate,
    regularized_binary_cauchy_majorant_tail_certificate,
    regularized_binary_interval_atlas_cauchy_majorant_tail_certificate,
    regularized_binary_interval_cauchy_majorant_tail_certificate,
    regularized_binary_tail_certificate,
)


Array = np.ndarray


@dataclass(frozen=True)
class OrdinaryTaylorEquationResidualCertificate:
    """Coefficient residual check for an ordinary interval Taylor chart."""

    coefficient_count: int
    position_residual_coefficients: tuple[tuple[tuple[FloatInterval, ...], ...], ...]
    velocity_residual_coefficients: tuple[tuple[tuple[FloatInterval, ...], ...], ...]
    coefficient_source: str = "ordinary_interval_taylor"

    @property
    def certified(self) -> bool:
        return bool(
            self.coefficient_count > 0
            and all(
                interval_contains_zero(coefficient)
                for residuals in (
                    self.position_residual_coefficients,
                    self.velocity_residual_coefficients,
                )
                for degree in residuals
                for body in degree
                for coefficient in body
            )
        )

    @property
    def max_residual_radius(self) -> float:
        radii = [
            max(abs(coefficient.lower), abs(coefficient.upper))
            for residuals in (
                self.position_residual_coefficients,
                self.velocity_residual_coefficients,
            )
            for degree in residuals
            for body in degree
            for coefficient in body
        ]
        return float(max(radii, default=0.0))


@dataclass(frozen=True)
class PlanarLeviCivitaProjectionDomainCertificate:
    """Domain proof for projecting a planar Levi-Civita binary chart.

    The lifted equations are analytic through a binary collision, but the
    physical velocity projection divides by ``rho = |z|^2``.  This certificate
    distinguishes a closed rho-positive projection interval from an explicitly
    punctured interval that starts at a regularized binary collision.
    """

    pair: tuple[int, int]
    parameter_interval: tuple[float, float]
    endpoint_parameter_interval: tuple[float, float]
    rho_interval: tuple[float, float]
    endpoint_rho_interval: tuple[float, float]
    factored_rho_interval: tuple[float, float] | None
    competing_pair_squared_distance_intervals: tuple[tuple[tuple[int, int], tuple[float, float]], ...]
    branch_or_atlas_certified: bool
    starts_at_binary_collision: bool
    initial_collision_selector_certified: bool
    missing_obligations: tuple[str, ...]

    @property
    def rho_positive_on_closed_interval(self) -> bool:
        return bool(self.rho_interval[0] > 0.0)

    @property
    def punctured_initial_collision_certified(self) -> bool:
        return bool(
            self.starts_at_binary_collision
            and self.initial_collision_selector_certified
            and self.factored_rho_interval is not None
            and self.factored_rho_interval[0] > 0.0
            and self.endpoint_rho_interval[0] > 0.0
        )

    @property
    def endpoint_projection_certified(self) -> bool:
        return bool(self.endpoint_rho_interval[0] > 0.0)

    @property
    def third_body_separation_certified(self) -> bool:
        return bool(
            self.competing_pair_squared_distance_intervals
            and all(
                squared_distance[0] > 0.0
                for _pair, squared_distance in self.competing_pair_squared_distance_intervals
            )
        )

    @property
    def min_competing_pair_squared_distance_lower_bound(self) -> float:
        if not self.competing_pair_squared_distance_intervals:
            return 0.0
        return float(
            min(
                squared_distance[0]
                for _pair, squared_distance in self.competing_pair_squared_distance_intervals
            )
        )

    @property
    def projection_domain(self) -> str:
        if self.rho_positive_on_closed_interval:
            return "rho_positive_interval"
        if self.punctured_initial_collision_certified:
            return "punctured_initial_collision_rho_positive_interval"
        return "rho_not_certified_positive_interval"

    @property
    def certified(self) -> bool:
        return not self.missing_obligations


@dataclass(frozen=True)
class RegularizedBinaryEquationResidualCertificate:
    """Coefficient residual and projection certificate for a Levi-Civita chart."""

    coefficient_count: int
    residual_coefficients: tuple[FloatInterval, ...]
    pair: tuple[int, int]
    coefficient_source: str = "regularized_interval_taylor"
    projection_domain_certificate: PlanarLeviCivitaProjectionDomainCertificate | None = None

    @property
    def projection_domain(self) -> str:
        if self.projection_domain_certificate is None:
            return "missing_projection_domain_certificate"
        return self.projection_domain_certificate.projection_domain

    @property
    def lifted_equations_certified(self) -> bool:
        return bool(
            self.coefficient_count > 0
            and self.residual_coefficients
            and all(interval_contains_zero(coefficient) for coefficient in self.residual_coefficients)
        )

    @property
    def projection_certified(self) -> bool:
        return bool(
            self.lifted_equations_certified
            and self.projection_domain_certificate is not None
            and self.projection_domain_certificate.certified
        )

    @property
    def certified(self) -> bool:
        return self.lifted_equations_certified

    @property
    def max_residual_radius(self) -> float:
        radii = [
            max(abs(coefficient.lower), abs(coefficient.upper))
            for coefficient in self.residual_coefficients
        ]
        return float(max(radii, default=0.0))


@dataclass(frozen=True)
class IntervalUnionCertificate:
    """Certificate that an obligation holds on every member of a split interval union."""

    member_certificates: tuple[object, ...]
    coefficient_source: str

    @property
    def member_count(self) -> int:
        return len(self.member_certificates)

    @property
    def certified(self) -> bool:
        return bool(
            self.member_certificates
            and all(getattr(certificate, "certified", False) for certificate in self.member_certificates)
        )

    @property
    def projection_certified(self) -> bool:
        return bool(
            self.member_certificates
            and all(
                getattr(certificate, "projection_certified", getattr(certificate, "certified", False))
                for certificate in self.member_certificates
            )
        )

    @property
    def max_residual_radius(self) -> float:
        radii = [
            float(getattr(certificate, "max_residual_radius"))
            for certificate in self.member_certificates
            if hasattr(certificate, "max_residual_radius")
        ]
        return float(max(radii, default=0.0))

    @property
    def max_nonconstant_radius(self) -> float:
        radii = [
            float(getattr(certificate, "max_nonconstant_radius"))
            for certificate in self.member_certificates
            if hasattr(certificate, "max_nonconstant_radius")
        ]
        return float(max(radii, default=0.0))


@dataclass(frozen=True)
class PolynomialRootIntervalEnclosure:
    interval: tuple[float, float]
    direction: str
    value_intervals: tuple[tuple[float, float], tuple[float, float]] | None = None
    derivative_interval: tuple[float, float] | None = None
    pre_event_range: tuple[float, float] | None = None
    pre_event_value_intervals: tuple[tuple[float, float], ...] = ()
    coefficient_intervals: tuple[tuple[float, float], ...] | None = None
    coefficient_source: str | None = None

    @property
    def width(self) -> float:
        return float(self.interval[1] - self.interval[0])

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.interval[0] + self.interval[1])

    @property
    def entry_sign(self) -> int:
        if self.direction == "decreasing":
            return 1
        if self.direction == "increasing":
            return -1
        return 0

    @property
    def exit_sign(self) -> int:
        return -self.entry_sign

    @property
    def interval_derivative_sign(self) -> int:
        if self.derivative_interval is None:
            return 0
        return interval_sign(FloatInterval(*self.derivative_interval))

    @property
    def interval_is_isolated(self) -> bool:
        if self.value_intervals is None or self.derivative_interval is None:
            return False
        left_sign = interval_sign(FloatInterval(*self.value_intervals[0]))
        right_sign = interval_sign(FloatInterval(*self.value_intervals[1]))
        derivative_sign = self.interval_derivative_sign
        if self.entry_sign == 0:
            return False
        derivative_ok = (
            (self.direction == "decreasing" and derivative_sign < 0)
            or (self.direction == "increasing" and derivative_sign > 0)
        )
        return left_sign == self.entry_sign and right_sign == self.exit_sign and derivative_ok

    @property
    def excludes_earlier_roots(self) -> bool:
        if self.entry_sign == 0 or not self.pre_event_value_intervals:
            return False
        return all(
            interval_sign(FloatInterval(*value_interval)) == self.entry_sign
            for value_interval in self.pre_event_value_intervals
        )

    @property
    def certifies_earliest_root(self) -> bool:
        return self.interval_is_isolated and self.excludes_earlier_roots


@dataclass(frozen=True)
class PolynomialRootCertificate:
    root: float
    bracket: tuple[float, float]
    values: tuple[float, float]
    derivative_sign: int
    derivative_roots_in_bracket: int
    value_intervals: tuple[tuple[float, float], tuple[float, float]] | None = None
    derivative_interval: tuple[float, float] | None = None
    coefficient_intervals: tuple[tuple[float, float], ...] | None = None
    coefficient_source: str | None = None
    root_enclosure: PolynomialRootIntervalEnclosure | None = None

    @property
    def is_isolated(self) -> bool:
        left, right = self.values
        return left * right < 0.0 and self.derivative_roots_in_bracket == 0 and self.derivative_sign != 0

    @property
    def interval_derivative_sign(self) -> int:
        if self.derivative_interval is None:
            return 0
        return interval_sign(FloatInterval(*self.derivative_interval))

    @property
    def interval_is_isolated(self) -> bool:
        if self.value_intervals is None or self.derivative_interval is None:
            return False
        left_sign = interval_sign(FloatInterval(*self.value_intervals[0]))
        right_sign = interval_sign(FloatInterval(*self.value_intervals[1]))
        derivative_sign = self.interval_derivative_sign
        return (
            left_sign * right_sign < 0
            and derivative_sign == self.derivative_sign
            and derivative_sign != 0
            and self.derivative_roots_in_bracket == 0
        )

    @property
    def earliest_interval_is_certified(self) -> bool:
        return self.root_enclosure is not None and self.root_enclosure.certifies_earliest_root


@dataclass(frozen=True)
class PolynomialRootUnionCertificate:
    root: float
    pair: tuple[int, int]
    member_certificates: tuple[PolynomialRootCertificate, ...]
    member_root_enclosures: tuple[PolynomialRootIntervalEnclosure | None, ...] = ()

    @property
    def member_count(self) -> int:
        return len(self.member_certificates)

    @property
    def interval_is_isolated(self) -> bool:
        return bool(self.member_certificates) and all(
            certificate.interval_is_isolated for certificate in self.member_certificates
        )

    @property
    def has_member_root_enclosures(self) -> bool:
        return (
            len(self.member_root_enclosures) == len(self.member_certificates)
            and bool(self.member_root_enclosures)
            and all(enclosure is not None for enclosure in self.member_root_enclosures)
        )

    @property
    def event_time_interval(self) -> tuple[float, float] | None:
        if not self.has_member_root_enclosures:
            return None
        enclosures = [enclosure for enclosure in self.member_root_enclosures if enclosure is not None]
        return (
            float(min(enclosure.interval[0] for enclosure in enclosures)),
            float(max(enclosure.interval[1] for enclosure in enclosures)),
        )

    @property
    def earliest_interval_is_certified(self) -> bool:
        return self.has_member_root_enclosures and all(
            enclosure is not None and enclosure.certifies_earliest_root
            for enclosure in self.member_root_enclosures
        )


@dataclass(frozen=True)
class HybridStep:
    chart: str
    start_time: float
    physical_step: float
    parameter_step: float
    min_pair_distance: float
    pair: tuple[int, int] | None = None
    event: str | None = None
    event_certificate: PolynomialRootCertificate | None = None
    event_union_certificate: PolynomialRootUnionCertificate | None = None
    event_time_interval: tuple[float, float] | None = None
    truncation_certificate: TailBoundCertificate | None = None
    start_state_interval: tuple[tuple[float, float], ...] | None = None
    end_state_interval: tuple[tuple[float, float], ...] | None = None
    start_state_interval_union: tuple[tuple[tuple[float, float], ...], ...] | None = None
    end_state_interval_union: tuple[tuple[tuple[float, float], ...], ...] | None = None
    interval_min_pair_distance: float | None = None
    interval_max_speed: float | None = None
    interval_max_acceleration: float | None = None
    interval_chart: str | None = None
    interval_chart_pair: tuple[int, int] | None = None
    interval_chart_certified: bool = False
    binary_interval_lift_certified: bool = False
    binary_interval_lift_reason: str | None = None
    binary_lc_branch: str | None = None
    binary_lc_branch_certified: bool = False
    binary_lc_atlas_chart_count: int = 0
    binary_lc_atlas_certified: bool = False
    binary_lc_atlas_propagated: bool = False
    start_regularized_interval_state: IntervalRegularizedBinaryCollisionChartState | None = None
    start_regularized_interval_state_union: tuple[IntervalRegularizedBinaryCollisionChartState, ...] = ()
    residual_certificate: OrdinaryTaylorEquationResidualCertificate | RegularizedBinaryEquationResidualCertificate | None = None
    center_of_mass_certificate: CenterOfMassMotionCertificate | None = None
    linear_momentum_certificate: LinearMomentumConservationCertificate | None = None
    angular_momentum_certificate: AngularMomentumConservationCertificate | None = None
    energy_certificate: EnergyConservationCertificate | None = None

    @property
    def residual_certified(self) -> bool:
        return bool(self.residual_certificate is not None and self.residual_certificate.certified)

    @property
    def projection_certified(self) -> bool:
        if self.chart == "ordinary":
            return self.residual_certified
        if self.chart == "binary":
            return bool(
                self.residual_certificate is not None
                and getattr(self.residual_certificate, "projection_certified", False)
            )
        return False

    @property
    def invariants_certified(self) -> bool:
        certificates = (
            self.center_of_mass_certificate,
            self.linear_momentum_certificate,
            self.angular_momentum_certificate,
            self.energy_certificate,
        )
        return bool(all(certificate is not None and certificate.certified for certificate in certificates))

    @property
    def end_time(self) -> float:
        return self.start_time + self.physical_step

    def end_state_interval_contains(self, state: Array) -> bool:
        if self.end_state_interval is None:
            return False
        state = np.asarray(state, dtype=float).reshape(-1)
        if len(self.end_state_interval) != state.shape[0]:
            return False
        for value, (lower, upper) in zip(state, self.end_state_interval):
            if lower > value or value > upper:
                return False
        return True

    def start_state_interval_contains(self, state: Array) -> bool:
        if self.start_state_interval is None:
            return False
        state = np.asarray(state, dtype=float).reshape(-1)
        if len(self.start_state_interval) != state.shape[0]:
            return False
        for value, (lower, upper) in zip(state, self.start_state_interval):
            if lower > value or value > upper:
                return False
        return True

    @property
    def event_certified(self) -> bool:
        if self.event is None:
            return True
        if self.event == "enter_binary":
            union_certified = (
                self.event_union_certificate is not None
                and self.event_union_certificate.earliest_interval_is_certified
            )
            point_certified = (
                self.event_certificate is not None
                and self.event_certificate.earliest_interval_is_certified
            )
            return bool(self.event_time_interval is not None and (union_certified or point_certified))
        if self.event == "exit_binary":
            return bool(
                self.event_certificate is not None
                and self.event_certificate.earliest_interval_is_certified
            )
        return False

    @property
    def proof_certified(self) -> bool:
        if not self.interval_chart_certified or self.truncation_certificate is None:
            return False
        coefficient_source = getattr(self.truncation_certificate, "coefficient_source", "")
        if "interval" not in coefficient_source or "cauchy_majorant" not in coefficient_source:
            return False
        if not self.event_certified:
            return False
        if not self.projection_certified or not self.invariants_certified:
            return False
        if self.chart == "binary":
            certified_binary_chart = self.binary_interval_lift_certified and (
                self.binary_lc_branch_certified
                or (self.binary_lc_atlas_certified and self.binary_lc_atlas_propagated)
            )
            return bool(certified_binary_chart)
        return self.chart == "ordinary"


@dataclass(frozen=True)
class HybridContinuedSolution:
    masses: Array
    times: Array
    states: Array
    steps: tuple[HybridStep, ...]

    @property
    def final_state(self) -> Array:
        return self.states[-1]

    def final_positions_velocities(self) -> tuple[Array, Array]:
        return unpack_planar_state(self.final_state)

    @property
    def local_tail_bound(self) -> float:
        """Sum of step-local Taylor tail bounds.

        This is a local ledger of guarded numerical tail estimates, not a
        rigorous propagated global error bound.
        """

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
    def interval_chart_certified_step_count(self) -> int:
        return sum(step.interval_chart_certified for step in self.steps)

    @property
    def proof_certified_step_count(self) -> int:
        return sum(step.proof_certified for step in self.steps)

    @property
    def proof_certified(self) -> bool:
        return bool(self.steps and all(step.proof_certified for step in self.steps))

    def propagated_error_budget(self) -> PropagatedErrorBudget:
        """Return a numerical Gronwall propagation of local tail bounds."""

        return propagate_error_budget(self.masses, self.steps)

    def propagated_interval_enclosure(self) -> PropagatedIntervalEnclosure:
        """Return endpoint interval boxes inflated by propagated tail bounds."""

        return propagate_interval_enclosures(self.masses, self.steps)

    def ordinary_set_propagated_interval_enclosure(
        self,
        *,
        retained_order: int | None = None,
        ordinary_substeps: int = 1,
    ) -> OrdinarySetPropagatedEnclosure:
        """Rebuild ordinary interval charts from each incoming endpoint box."""

        return propagate_ordinary_set_enclosures(
            self.masses,
            self.steps,
            retained_order=retained_order,
            ordinary_substeps=ordinary_substeps,
        )

    def lohner_ordinary_set_propagated_interval_enclosure(
        self,
        *,
        retained_order: int | None = None,
    ) -> LohnerOrdinaryPropagatedEnclosure:
        """Rebuild ordinary charts with Lohner-shaped incoming sets.

        This path supports ordinary charts, including certified ordinary
        event-time intervals. Binary, Sundman, and KS charts are intentionally
        rejected until they have their own Lohner chart maps.
        """

        return propagate_lohner_ordinary_set_enclosures(
            self.masses,
            self.steps,
            retained_order=retained_order,
        )


@dataclass(frozen=True)
class IntervalChartDecision:
    chart: str
    pair: tuple[int, int] | None
    certified: bool
    pair_distance_bounds: tuple[tuple[tuple[int, int], tuple[float, float]], ...]
    reason: str


def pack_planar_state(positions: Array, velocities: Array) -> Array:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if positions.shape != (3, 2) or velocities.shape != (3, 2):
        raise ValueError("positions and velocities must both have shape (3, 2)")
    return np.concatenate([positions.reshape(-1), velocities.reshape(-1)])


def _pack_interval_planar_state(positions: Array, velocities: Array) -> tuple[tuple[float, float], ...]:
    return interval_array_as_tuples(np.concatenate([positions.reshape(-1), velocities.reshape(-1)]))


def _collision_position_summary(
    state: RegularizedBinaryCollisionChartState,
) -> tuple[Array, Array]:
    """Return planar positions and center velocities at an exact binary collision."""

    if state.rho != 0.0:
        return regularized_binary_collision_chart_to_planar(state)
    first, second = state.pair
    third = state.third_index
    positions = np.empty((3, 2), dtype=float)
    velocities = np.empty((3, 2), dtype=float)
    positions[first] = state.binary_center
    positions[second] = state.binary_center
    positions[third] = state.binary_center + state.third_offset
    velocities[first] = state.binary_center_velocity
    velocities[second] = state.binary_center_velocity
    velocities[third] = state.binary_center_velocity + state.third_offset_velocity
    return positions, velocities


def _unpack_interval_planar_state(state_interval: tuple[tuple[float, float], ...]) -> tuple[Array, Array]:
    if len(state_interval) != 12:
        raise ValueError("planar interval state must have length 12")
    values = np.array([FloatInterval(lower, upper) for lower, upper in state_interval], dtype=object)
    return values[:6].reshape(3, 2), values[6:].reshape(3, 2)


def certify_ordinary_interval_taylor_equations(
    series: IntervalTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> OrdinaryTaylorEquationResidualCertificate:
    """Certify ``q'=v`` and ``v'=a(q)`` coefficient residuals for a chart."""

    if coefficient_count is None:
        coefficient_count = series.order
    coefficient_count = int(coefficient_count)
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > series.order:
        raise ValueError("coefficient_count cannot exceed chart order")
    acceleration = acceleration_interval_coefficients(
        series.position,
        series.masses,
        coefficient_count - 1,
    )
    position_residuals = []
    velocity_residuals = []
    for degree in range(coefficient_count):
        position_degree = []
        velocity_degree = []
        for body in range(series.body_count):
            position_body = []
            velocity_body = []
            for axis in range(series.dimension):
                derivative_position = series.position[degree + 1, body, axis].scale(
                    degree + 1
                )
                derivative_velocity = series.velocity[degree + 1, body, axis].scale(
                    degree + 1
                )
                position_body.append(derivative_position - series.velocity[degree, body, axis])
                velocity_body.append(derivative_velocity - acceleration[degree, body, axis])
            position_degree.append(tuple(position_body))
            velocity_degree.append(tuple(velocity_body))
        position_residuals.append(tuple(position_degree))
        velocity_residuals.append(tuple(velocity_degree))
    return OrdinaryTaylorEquationResidualCertificate(
        coefficient_count=coefficient_count,
        position_residual_coefficients=tuple(position_residuals),
        velocity_residual_coefficients=tuple(velocity_residuals),
    )


def certify_regularized_binary_interval_taylor_equations(
    series: IntervalRegularizedBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
    projection_domain_certificate: PlanarLeviCivitaProjectionDomainCertificate | None = None,
) -> RegularizedBinaryEquationResidualCertificate:
    """Certify the interval Levi-Civita recurrence coefficient-by-coefficient."""

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
        for axis in range(2):
            residuals.append(series.z[degree + 1, axis].scale(scale) - rhs.z[degree, axis])
            residuals.append(
                series.z_velocity[degree + 1, axis].scale(scale)
                - rhs.z_velocity[degree, axis]
            )
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
    return RegularizedBinaryEquationResidualCertificate(
        coefficient_count=coefficient_count,
        residual_coefficients=tuple(residuals),
        pair=tuple(series.pair),
        projection_domain_certificate=projection_domain_certificate,
    )


def certify_regularized_binary_projection_domain(
    series: IntervalRegularizedBinaryTaylorSolution,
    *,
    parameter_interval: tuple[float, float],
    endpoint_parameter_interval: tuple[float, float] | None = None,
    branch_or_atlas_certified: bool,
    starts_at_binary_collision: bool = False,
    initial_collision_selector_certified: bool = False,
) -> PlanarLeviCivitaProjectionDomainCertificate:
    """Certify that LC projection is used only where ``rho=|z|^2`` permits it."""

    lower, upper = (float(parameter_interval[0]), float(parameter_interval[1]))
    if lower < 0.0 or upper < lower:
        raise ValueError("parameter_interval must be a forward nonnegative interval")
    if endpoint_parameter_interval is None:
        endpoint_parameter_interval = (upper, upper)
    endpoint_lower, endpoint_upper = (
        float(endpoint_parameter_interval[0]),
        float(endpoint_parameter_interval[1]),
    )
    if endpoint_lower < 0.0 or endpoint_upper < endpoint_lower:
        raise ValueError("endpoint_parameter_interval must be a forward nonnegative interval")
    rho_coefficients = _squared_norm_polynomial_intervals(series.z)
    rho_interval = interval_polynomial_eval(rho_coefficients, FloatInterval(lower, upper))
    endpoint_rho_interval = interval_polynomial_eval(
        rho_coefficients,
        FloatInterval(endpoint_lower, endpoint_upper),
    )
    factored_rho_interval = None
    if starts_at_binary_collision:
        z_factor = np.asarray(series.z[1:, :], dtype=object)
        if z_factor.shape[0] > 0:
            factored_coefficients = _squared_norm_polynomial_intervals(z_factor)
            factored_rho_interval = interval_polynomial_eval(
                factored_coefficients,
                FloatInterval(0.0, upper),
            )
    competing_pair_squared_distances = _regularized_binary_competing_pair_squared_distance_intervals(
        series,
        FloatInterval(lower, upper),
    )

    missing: list[str] = []
    if not branch_or_atlas_certified:
        missing.append("lc_branch_or_atlas_not_certified")
    closed_rho_positive = rho_interval.lower > 0.0
    punctured_collision_positive = bool(
        starts_at_binary_collision
        and initial_collision_selector_certified
        and factored_rho_interval is not None
        and factored_rho_interval.lower > 0.0
        and endpoint_rho_interval.lower > 0.0
    )
    if not (closed_rho_positive or punctured_collision_positive):
        missing.append("projection_rho_interval_contains_zero")
    if endpoint_rho_interval.lower <= 0.0:
        missing.append("endpoint_projection_rho_not_positive")
    if starts_at_binary_collision and not initial_collision_selector_certified:
        missing.append("initial_collision_selector_not_certified")
    if not competing_pair_squared_distances or any(
        squared_distance[0] <= 0.0
        for _pair, squared_distance in competing_pair_squared_distances
    ):
        missing.append("third_body_separation_not_certified")

    return PlanarLeviCivitaProjectionDomainCertificate(
        pair=tuple(series.pair),
        parameter_interval=(lower, upper),
        endpoint_parameter_interval=(endpoint_lower, endpoint_upper),
        rho_interval=rho_interval.as_tuple(),
        endpoint_rho_interval=endpoint_rho_interval.as_tuple(),
        factored_rho_interval=(
            factored_rho_interval.as_tuple()
            if factored_rho_interval is not None
            else None
        ),
        competing_pair_squared_distance_intervals=competing_pair_squared_distances,
        branch_or_atlas_certified=bool(branch_or_atlas_certified),
        starts_at_binary_collision=bool(starts_at_binary_collision),
        initial_collision_selector_certified=bool(initial_collision_selector_certified),
        missing_obligations=tuple(missing),
    )


def _regularized_binary_competing_pair_squared_distance_intervals(
    series: IntervalRegularizedBinaryTaylorSolution,
    parameter_interval: FloatInterval,
) -> tuple[tuple[tuple[int, int], tuple[float, float]], ...]:
    first, second = tuple(series.pair)
    third_candidates = {0, 1, 2} - {first, second}
    if len(third_candidates) != 1:
        raise ValueError("binary pair must select two distinct bodies from {0, 1, 2}")
    third = third_candidates.pop()
    masses = np.asarray(series.masses, dtype=float).reshape(-1)
    pair_mass = float(masses[first] + masses[second])
    if pair_mass <= 0.0:
        raise ValueError("selected binary pair must have positive total mass")

    relative_position = _lc_relative_position_interval_coefficients(series.z)
    max_degree = max(relative_position.shape[0], series.third_offset.shape[0]) - 1
    first_third_delta = _regularized_binary_delta_coefficients(
        relative_position,
        series.third_offset,
        relative_factor=-float(masses[second] / pair_mass),
        max_degree=max_degree,
    )
    second_third_delta = _regularized_binary_delta_coefficients(
        relative_position,
        series.third_offset,
        relative_factor=float(masses[first] / pair_mass),
        max_degree=max_degree,
    )
    return (
        ((min(first, third), max(first, third)), _squared_norm_interval_over_parameter(first_third_delta, parameter_interval)),
        ((min(second, third), max(second, third)), _squared_norm_interval_over_parameter(second_third_delta, parameter_interval)),
    )


def _lc_relative_position_interval_coefficients(z_coefficients: Array) -> Array:
    z_coefficients = np.asarray(z_coefficients, dtype=object)
    if z_coefficients.ndim != 2 or z_coefficients.shape[1] != 2:
        raise ValueError("Levi-Civita z coefficients must have shape (order + 1, 2)")
    max_degree = z_coefficients.shape[0] - 1
    x = tuple(_as_interval(z_coefficients[n, 0]) for n in range(max_degree + 1))
    y = tuple(_as_interval(z_coefficients[n, 1]) for n in range(max_degree + 1))
    x_square = interval_series_product(x, x, 2 * max_degree)
    y_square = interval_series_product(y, y, 2 * max_degree)
    xy = interval_series_product(x, y, 2 * max_degree)
    relative_position = np.empty((2 * max_degree + 1, 2), dtype=object)
    for degree in range(2 * max_degree + 1):
        relative_position[degree, 0] = x_square[degree] - y_square[degree]
        relative_position[degree, 1] = xy[degree].scale(2.0)
    return relative_position


def _regularized_binary_delta_coefficients(
    relative_position: Array,
    third_offset: Array,
    *,
    relative_factor: float,
    max_degree: int,
) -> tuple[tuple[FloatInterval, ...], ...]:
    deltas = []
    for axis in range(2):
        coefficients = []
        for degree in range(max_degree + 1):
            relative_value = (
                _as_interval(relative_position[degree, axis])
                if degree < relative_position.shape[0]
                else zero_interval()
            )
            third_value = (
                _as_interval(third_offset[degree, axis])
                if degree < third_offset.shape[0]
                else zero_interval()
            )
            coefficients.append(relative_value.scale(relative_factor) - third_value)
        deltas.append(tuple(coefficients))
    return tuple(deltas)


def _squared_norm_interval_over_parameter(
    component_coefficients: tuple[tuple[FloatInterval, ...], ...],
    parameter_interval: FloatInterval,
) -> tuple[float, float]:
    if not component_coefficients:
        raise ValueError("component coefficients cannot be empty")
    max_degree = max(len(component) for component in component_coefficients) - 1
    squared = tuple(FloatInterval.point(0.0) for _ in range(2 * max_degree + 1))
    for component in component_coefficients:
        padded = tuple(
            component[degree] if degree < len(component) else zero_interval()
            for degree in range(max_degree + 1)
        )
        squared = interval_series_add(
            squared,
            interval_series_product(padded, padded, 2 * max_degree),
        )
    return interval_polynomial_eval(squared, parameter_interval).as_tuple()


def certify_regularized_binary_center_of_mass_motion(
    series: IntervalRegularizedBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> CenterOfMassMotionCertificate:
    """Certify inertial center-of-mass motion from regularized binary variables.

    In binary coordinates, the total mass moment is
    ``M_total R + m_3 y`` and the total momentum is
    ``M_total R_dot + m_3 y_dot``.  These are finite regularized quantities
    even at exact binary collision.
    """

    coefficient_count = _regularized_binary_invariant_coefficient_count(
        series,
        coefficient_count,
    )
    total_mass = float(np.sum(series.masses))
    third_mass = float(series.masses[_third_index_for_pair(series.pair)])
    position_moment, momentum = _regularized_binary_mass_moment_and_momentum(
        series,
        coefficient_count,
        total_mass=total_mass,
        third_mass=third_mass,
    )
    physical_time = tuple(_as_interval(series.physical_time[n]) for n in range(coefficient_count + 1))
    initial_offset = tuple(
        position_moment[0][axis] - momentum[0][axis] * physical_time[0]
        for axis in range(2)
    )
    residuals = []
    for degree in range(coefficient_count + 1):
        components = []
        for axis in range(2):
            component = position_moment[degree][axis] - momentum[0][axis] * physical_time[degree]
            if degree == 0:
                component = component - initial_offset[axis]
            components.append(component)
        residuals.append(tuple(components))
    return CenterOfMassMotionCertificate(
        coefficient_count=coefficient_count,
        residual_coefficients=tuple(residuals),
        coefficient_source="regularized_binary_interval_series",
    )


def certify_regularized_binary_linear_momentum_conservation(
    series: IntervalRegularizedBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> LinearMomentumConservationCertificate:
    """Certify total linear momentum from regularized binary variables."""

    coefficient_count = _regularized_binary_invariant_coefficient_count(
        series,
        coefficient_count,
    )
    total_mass = float(np.sum(series.masses))
    third_mass = float(series.masses[_third_index_for_pair(series.pair)])
    _position_moment, momentum = _regularized_binary_mass_moment_and_momentum(
        series,
        coefficient_count,
        total_mass=total_mass,
        third_mass=third_mass,
    )
    return LinearMomentumConservationCertificate(
        coefficient_count=coefficient_count,
        linear_momentum_coefficients=momentum,
        coefficient_source="regularized_binary_interval_series",
    )


def certify_regularized_binary_centered_angular_momentum_conservation(
    series: IntervalRegularizedBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> AngularMomentumConservationCertificate:
    """Certify centered angular momentum in finite Levi-Civita coordinates.

    With pair mass ``M``, third mass ``m_3``, total mass ``T`` and reduced pair
    mass ``mu``, the centered planar angular momentum is
    ``(M m_3 / T) y wedge y_dot + 2 mu z wedge z_dot``.  The second term is the
    finite Levi-Civita form of ``mu r wedge r_dot``.
    """

    coefficient_count = _regularized_binary_invariant_coefficient_count(
        series,
        coefficient_count,
    )
    first, second = series.pair
    third = _third_index_for_pair(series.pair)
    pair_mass = float(series.masses[first] + series.masses[second])
    total_mass = float(np.sum(series.masses))
    third_mass = float(series.masses[third])
    reduced_pair_mass = float(series.masses[first] * series.masses[second] / pair_mass)
    reduced_third_mass = float(pair_mass * third_mass / total_mass)
    y_wedge = _interval_wedge_series(
        series.third_offset,
        series.third_offset_velocity,
        coefficient_count,
    )
    z_wedge = _interval_wedge_series(
        series.z,
        series.z_velocity,
        coefficient_count,
    )
    coefficients = []
    for degree in range(coefficient_count + 1):
        component = y_wedge[degree].scale(reduced_third_mass) + z_wedge[degree].scale(
            2.0 * reduced_pair_mass
        )
        coefficients.append((component,))
    return AngularMomentumConservationCertificate(
        coefficient_count=coefficient_count,
        angular_momentum_coefficients=tuple(coefficients),
        coefficient_source="regularized_binary_interval_series",
    )


def certify_regularized_binary_total_energy_conservation(
    series: IntervalRegularizedBinaryTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> EnergyConservationCertificate:
    """Certify finite Newtonian energy through a regularized binary chart."""

    coefficient_count = _regularized_binary_invariant_coefficient_count(
        series,
        coefficient_count,
    )
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

    relative_position = _regularized_binary_relative_position_coefficients(
        series,
        coefficient_count,
    )
    alpha = float(masses[second] / pair_mass)
    beta = float(masses[first] / pair_mass)
    from_first = _interval_vector_series_scaled_sum(
        series.third_offset,
        1.0,
        relative_position,
        alpha,
        coefficient_count,
    )
    from_second = _interval_vector_series_scaled_sum(
        series.third_offset,
        1.0,
        relative_position,
        -beta,
        coefficient_count,
    )
    energy = interval_series_add(
        energy,
        _scale_interval_series(
            _interval_inverse_norm_series(from_first, coefficient_count),
            -float(masses[first] * third_mass),
        ),
    )
    energy = interval_series_add(
        energy,
        _scale_interval_series(
            _interval_inverse_norm_series(from_second, coefficient_count),
            -float(masses[second] * third_mass),
        ),
    )
    return EnergyConservationCertificate(
        coefficient_count=coefficient_count,
        energy_coefficients=energy,
        coefficient_source="regularized_binary_interval_series",
    )


def _regularized_binary_invariant_coefficient_count(
    series: IntervalRegularizedBinaryTaylorSolution,
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


def _third_index_for_pair(pair: tuple[int, int]) -> int:
    remaining = ({0, 1, 2} - set(pair))
    if len(remaining) != 1:
        raise ValueError("pair must contain two distinct body indices from {0, 1, 2}")
    return remaining.pop()


def _regularized_binary_mass_moment_and_momentum(
    series: IntervalRegularizedBinaryTaylorSolution,
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
        for axis in range(2):
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


def _interval_wedge_series(left: Array, right: Array, max_degree: int) -> tuple[FloatInterval, ...]:
    left_x = _series_axis(left, 0, max_degree)
    left_y = _series_axis(left, 1, max_degree)
    right_x = _series_axis(right, 0, max_degree)
    right_y = _series_axis(right, 1, max_degree)
    left_cross = interval_series_product(left_x, right_y, max_degree)
    right_cross = interval_series_product(left_y, right_x, max_degree)
    return tuple(left_cross[degree] - right_cross[degree] for degree in range(max_degree + 1))


def _interval_dot_series(left: Array, right: Array, max_degree: int) -> tuple[FloatInterval, ...]:
    out = tuple(FloatInterval.point(0.0) for _ in range(max_degree + 1))
    for axis in range(2):
        out = interval_series_add(
            out,
            interval_series_product(
                _series_axis(left, axis, max_degree),
                _series_axis(right, axis, max_degree),
                max_degree,
            ),
        )
    return out


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
    out = np.empty((max_degree + 1, 2), dtype=object)
    for degree in range(max_degree + 1):
        for axis in range(2):
            out[degree, axis] = _as_interval(first[degree, axis]).scale(first_scale) + _as_interval(
                second[degree, axis]
            ).scale(second_scale)
    return out


def _regularized_binary_relative_position_coefficients(
    series: IntervalRegularizedBinaryTaylorSolution,
    max_degree: int,
) -> Array:
    x = _series_axis(series.z, 0, max_degree)
    y = _series_axis(series.z, 1, max_degree)
    xx = interval_series_product(x, x, max_degree)
    yy = interval_series_product(y, y, max_degree)
    xy = interval_series_product(x, y, max_degree)
    out = np.empty((max_degree + 1, 2), dtype=object)
    for degree in range(max_degree + 1):
        out[degree, 0] = xx[degree] - yy[degree]
        out[degree, 1] = xy[degree].scale(2.0)
    return out


def _interval_inverse_norm_series(
    vector: Array,
    max_degree: int,
) -> tuple[FloatInterval, ...]:
    distance_squared = tuple(FloatInterval.point(0.0) for _ in range(max_degree + 1))
    for axis in range(2):
        axis_series = _series_axis(vector, axis, max_degree)
        distance_squared = interval_series_add(
            distance_squared,
            interval_series_product(axis_series, axis_series, max_degree),
        )
    return interval_series_power(distance_squared, -0.5, max_degree)


def _ordinary_interval_taylor_solution_from_state_interval(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    *,
    order: int,
) -> IntervalTaylorSolution:
    positions, velocities = _unpack_interval_planar_state(state_interval)
    return construct_interval_taylor_solution_from_intervals(
        positions,
        velocities,
        masses,
        order=order,
    )


def _ordinary_interval_union_certificates(
    state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
    masses: Array,
    *,
    order: int,
) -> tuple[
    IntervalUnionCertificate,
    IntervalUnionCertificate,
    IntervalUnionCertificate,
    IntervalUnionCertificate,
    IntervalUnionCertificate,
]:
    if not state_interval_union:
        raise ValueError("state_interval_union cannot be empty")
    physical_time_series = _ordinary_physical_time_series(order)
    residual_certificates = []
    center_certificates = []
    linear_certificates = []
    angular_certificates = []
    energy_certificates = []
    for state_interval in state_interval_union:
        ordinary_interval_series = _ordinary_interval_taylor_solution_from_state_interval(
            state_interval,
            masses,
            order=order,
        )
        residual_certificates.append(
            certify_ordinary_interval_taylor_equations(
                ordinary_interval_series,
                coefficient_count=order,
            )
        )
        center_certificates.append(
            certify_interval_center_of_mass_motion(
                ordinary_interval_series.position,
                ordinary_interval_series.velocity,
                physical_time_series,
                masses,
                coefficient_count=order,
            )
        )
        linear_certificates.append(
            certify_interval_linear_momentum_conservation(
                ordinary_interval_series.velocity,
                masses,
                coefficient_count=order,
            )
        )
        angular_certificates.append(
            certify_interval_centered_angular_momentum_conservation(
                ordinary_interval_series.position,
                ordinary_interval_series.velocity,
                masses,
                coefficient_count=order,
            )
        )
        energy_certificates.append(
            certify_interval_total_energy_conservation(
                ordinary_interval_series.position,
                ordinary_interval_series.velocity,
                masses,
                coefficient_count=order,
            )
        )
    return (
        IntervalUnionCertificate(tuple(residual_certificates), "ordinary_interval_taylor_union"),
        IntervalUnionCertificate(tuple(center_certificates), "ordinary_interval_taylor_union"),
        IntervalUnionCertificate(tuple(linear_certificates), "ordinary_interval_taylor_union"),
        IntervalUnionCertificate(tuple(angular_certificates), "ordinary_interval_taylor_union"),
        IntervalUnionCertificate(tuple(energy_certificates), "ordinary_interval_taylor_union"),
    )


def _ordinary_physical_time_series(order: int) -> tuple[FloatInterval, ...]:
    if order < 1:
        raise ValueError("order must be at least one")
    return (
        FloatInterval.point(0.0),
        FloatInterval.point(1.0),
        *(FloatInterval.point(0.0) for _ in range(order - 1)),
    )


def _state_interval_contains_point(
    state_interval: tuple[tuple[float, float], ...],
    state: Array,
) -> bool:
    state = np.asarray(state, dtype=float).reshape(-1)
    if len(state_interval) != state.shape[0]:
        return False
    return all(lower <= value <= upper for value, (lower, upper) in zip(state, state_interval))


def _normalize_planar_state_interval(
    state_interval: tuple[tuple[float, float], ...],
    *,
    name: str,
) -> tuple[tuple[float, float], ...]:
    if len(state_interval) != 12:
        raise ValueError(f"{name} must describe a planar 12-coordinate state")
    out = []
    for lower, upper in state_interval:
        lower_value = float(lower)
        upper_value = float(upper)
        if not np.isfinite(lower_value) or not np.isfinite(upper_value):
            raise ValueError(f"{name} bounds must be finite")
        if lower_value > upper_value:
            raise ValueError(f"{name} lower bounds must not exceed upper bounds")
        out.append((lower_value, upper_value))
    return tuple(out)


def _normalize_planar_state_interval_union(
    state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
    *,
    name: str,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not state_interval_union:
        raise ValueError(f"{name} cannot be empty")
    return tuple(
        _normalize_planar_state_interval(state_interval, name=f"{name} member")
        for state_interval in state_interval_union
    )


def _state_interval_union_contains_point(
    state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
    state: Array,
) -> bool:
    return any(_state_interval_contains_point(state_interval, state) for state_interval in state_interval_union)


def _local_interval_step_size(interval: tuple[float, float]) -> float:
    return float(max(abs(interval[0]), abs(interval[1])))


def _interval_square_bounds(value: FloatInterval) -> tuple[float, float]:
    squares = (value.lower * value.lower, value.upper * value.upper)
    upper = float(np.nextafter(max(squares), np.inf))
    if value.lower <= 0.0 <= value.upper:
        lower = 0.0
    else:
        lower = float(np.nextafter(min(squares), -np.inf))
        lower = max(0.0, lower)
    return lower, upper


def interval_pairwise_distance_bounds(
    state_interval: tuple[tuple[float, float], ...],
) -> tuple[tuple[tuple[int, int], tuple[float, float]], ...]:
    """Conservative Euclidean distance bounds for all planar body pairs."""

    positions, _velocities = _unpack_interval_planar_state(state_interval)
    out = []
    for pair in ((0, 1), (0, 2), (1, 2)):
        lower_squared = 0.0
        upper_squared = 0.0
        for axis in range(2):
            axis_lower, axis_upper = _interval_square_bounds(
                positions[pair[0], axis] - positions[pair[1], axis]
            )
            lower_squared = max(0.0, float(np.nextafter(lower_squared + axis_lower, -np.inf)))
            upper_squared = float(np.nextafter(upper_squared + axis_upper, np.inf))

        lower_distance = (
            0.0
            if lower_squared <= 0.0
            else float(np.nextafter(np.sqrt(lower_squared), -np.inf))
        )
        upper_distance = float(np.nextafter(np.sqrt(max(upper_squared, 0.0)), np.inf))
        out.append((pair, (lower_distance, upper_distance)))
    return tuple(out)


def interval_min_pair_distance_lower_bound(state_interval: tuple[tuple[float, float], ...]) -> float:
    """Return a lower bound for the minimum pair distance in a planar interval state."""

    return float(min(bounds[0] for _pair, bounds in interval_pairwise_distance_bounds(state_interval)))


def interval_body_speed_upper_bounds(state_interval: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    """Return upper bounds for each body's speed over a planar interval state."""

    _positions, velocities = _unpack_interval_planar_state(state_interval)
    out = []
    for body in range(3):
        upper_squared = 0.0
        for axis in range(2):
            _axis_lower, axis_upper = _interval_square_bounds(velocities[body, axis])
            upper_squared = float(np.nextafter(upper_squared + axis_upper, np.inf))
        out.append(float(np.nextafter(np.sqrt(max(upper_squared, 0.0)), np.inf)))
    return tuple(out)


def interval_max_speed_upper_bound(state_interval: tuple[tuple[float, float], ...]) -> float:
    """Return an upper bound for the maximum body speed over a planar interval state."""

    return float(max(interval_body_speed_upper_bounds(state_interval)))


def interval_body_acceleration_upper_bounds(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
) -> tuple[float, ...]:
    """Return triangle-inequality acceleration upper bounds for each body."""

    masses = np.asarray(masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    distance_bounds = dict(interval_pairwise_distance_bounds(state_interval))
    out = []
    for body in range(3):
        bound = 0.0
        for other in range(3):
            if body == other:
                continue
            pair = (body, other) if body < other else (other, body)
            lower_distance = distance_bounds[pair][0]
            if lower_distance <= 0.0:
                bound = float("inf")
                break
            bound = float(
                np.nextafter(bound + masses[other] / (lower_distance * lower_distance), np.inf)
            )
        out.append(bound)
    return tuple(out)


def interval_max_acceleration_upper_bound(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
) -> float:
    """Return an upper bound for the maximum Newtonian acceleration norm."""

    return float(max(interval_body_acceleration_upper_bounds(state_interval, masses)))


def choose_interval_ordinary_step_size(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    remaining_time: float,
    *,
    max_step: float,
    safety: float = 0.08,
) -> float:
    """Choose an ordinary chart size using interval distance, speed, and acceleration bounds."""

    if remaining_time == 0.0:
        return 0.0
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    if safety <= 0.0:
        raise ValueError("safety must be positive")

    min_distance = interval_min_pair_distance_lower_bound(state_interval)
    max_speed = interval_max_speed_upper_bound(state_interval)
    max_acceleration = interval_max_acceleration_upper_bound(state_interval, masses)
    speed_limited = min_distance / max(max_speed, 1e-15)
    acceleration_limited = np.sqrt(min_distance / max(max_acceleration, 1e-15))
    candidate = min(safety * min(speed_limited, acceleration_limited), max_step)
    magnitude = min(abs(remaining_time), max(candidate, np.finfo(float).eps))
    return float(np.copysign(magnitude, remaining_time))


def choose_interval_union_ordinary_step_size(
    state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
    masses: Array,
    remaining_time: float,
    *,
    max_step: float,
    safety: float = 0.08,
) -> float:
    """Choose an ordinary chart size from split interval members without hulling."""

    if not state_interval_union:
        raise ValueError("state_interval_union cannot be empty")
    member_steps = tuple(
        abs(
            choose_interval_ordinary_step_size(
                state_interval,
                masses,
                remaining_time,
                max_step=max_step,
                safety=safety,
            )
        )
        for state_interval in state_interval_union
    )
    magnitude = min(abs(remaining_time), min(member_steps))
    return float(np.copysign(magnitude, remaining_time))


def unpack_planar_state(state: Array) -> tuple[Array, Array]:
    state = np.asarray(state, dtype=float)
    if state.shape != (12,):
        raise ValueError("planar state must have length 12")
    return state[:6].reshape(3, 2), state[6:].reshape(3, 2)


def closest_pair(positions: Array) -> tuple[tuple[int, int], float]:
    positions = np.asarray(positions, dtype=float)
    best_pair = (0, 1)
    best_distance = np.inf
    for i in range(3):
        for j in range(i + 1, 3):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if distance < best_distance:
                best_pair = (i, j)
                best_distance = distance
    return best_pair, best_distance


def choose_chart(positions: Array, *, binary_distance_threshold: float) -> tuple[str, tuple[int, int] | None, float]:
    """Return the chart type and pair for a planar state."""

    if binary_distance_threshold <= 0.0:
        raise ValueError("binary_distance_threshold must be positive")
    pair, distance = closest_pair(positions)
    if distance <= binary_distance_threshold:
        return "binary", pair, distance
    return "ordinary", None, distance


def choose_interval_chart(
    state_interval: tuple[tuple[float, float], ...],
    *,
    binary_distance_threshold: float,
) -> IntervalChartDecision:
    """Choose a chart only when the interval state certifies the threshold side."""

    if binary_distance_threshold <= 0.0:
        raise ValueError("binary_distance_threshold must be positive")

    distance_bounds = interval_pairwise_distance_bounds(state_interval)
    if all(lower > binary_distance_threshold for _pair, (lower, _upper) in distance_bounds):
        return IntervalChartDecision(
            chart="ordinary",
            pair=None,
            certified=True,
            pair_distance_bounds=distance_bounds,
            reason="all interval pair distances are above the binary threshold",
        )

    binary_candidates = [
        (pair, bounds)
        for pair, bounds in distance_bounds
        if bounds[1] <= binary_distance_threshold
    ]
    if len(binary_candidates) == 1:
        candidate_pair, _candidate_bounds = binary_candidates[0]
        separated_others = all(
            pair == candidate_pair or bounds[0] > binary_distance_threshold
            for pair, bounds in distance_bounds
        )
        if separated_others:
            return IntervalChartDecision(
                chart="binary",
                pair=candidate_pair,
                certified=True,
                pair_distance_bounds=distance_bounds,
                reason="one interval pair distance is below the binary threshold and the third body is separated",
            )

    return IntervalChartDecision(
        chart="ambiguous",
        pair=None,
        certified=False,
        pair_distance_bounds=distance_bounds,
        reason="interval pair-distance bounds overlap the chart threshold",
    )


def choose_transition_interval_chart(
    state_interval: tuple[tuple[float, float], ...],
    previous_step: HybridStep | None,
    *,
    binary_distance_threshold: float,
) -> IntervalChartDecision:
    """Choose a chart using certified transition events before raw state bounds."""

    distance_decision = choose_interval_chart(
        state_interval,
        binary_distance_threshold=binary_distance_threshold,
    )
    if previous_step is None or distance_decision.certified:
        return distance_decision

    if (
        previous_step.event == "enter_binary"
        and previous_step.pair is not None
        and previous_step.event_certificate is not None
        and previous_step.event_certificate.interval_is_isolated
    ):
        return IntervalChartDecision(
            chart="binary",
            pair=previous_step.pair,
            certified=True,
            pair_distance_bounds=distance_decision.pair_distance_bounds,
            reason="certified ordinary binary-entry event selects the next binary chart",
        )

    return distance_decision


def choose_interval_chart_union(
    state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
    *,
    binary_distance_threshold: float,
) -> IntervalChartDecision:
    """Choose a chart from a union of interval states when all members agree."""

    if not state_interval_union:
        raise ValueError("state interval union cannot be empty")
    decisions = tuple(
        choose_interval_chart(
            state_interval,
            binary_distance_threshold=binary_distance_threshold,
        )
        for state_interval in state_interval_union
    )
    pair_distance_bounds = interval_pairwise_distance_bounds(_hull_state_intervals(state_interval_union))
    if all(decision.certified and decision.chart == "ordinary" for decision in decisions):
        return IntervalChartDecision(
            chart="ordinary",
            pair=None,
            certified=True,
            pair_distance_bounds=pair_distance_bounds,
            reason="all interval-state union members certify ordinary chart",
        )
    binary_pairs = {
        decision.pair
        for decision in decisions
        if decision.certified and decision.chart == "binary" and decision.pair is not None
    }
    if len(binary_pairs) == 1 and all(
        decision.certified and decision.chart == "binary" and decision.pair in binary_pairs
        for decision in decisions
    ):
        pair = next(iter(binary_pairs))
        return IntervalChartDecision(
            chart="binary",
            pair=pair,
            certified=True,
            pair_distance_bounds=pair_distance_bounds,
            reason="all interval-state union members certify the same binary chart",
        )
    return IntervalChartDecision(
        chart="ambiguous",
        pair=None,
        certified=False,
        pair_distance_bounds=pair_distance_bounds,
        reason="interval-state union members do not certify a common chart",
    )


def choose_transition_interval_chart_union(
    state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
    previous_step: HybridStep | None,
    *,
    binary_distance_threshold: float,
) -> IntervalChartDecision:
    """Choose a transition chart from a union before falling back to event evidence."""

    union_decision = choose_interval_chart_union(
        state_interval_union,
        binary_distance_threshold=binary_distance_threshold,
    )
    if previous_step is None or union_decision.certified:
        return union_decision
    transition_decision = choose_transition_interval_chart(
        _hull_state_intervals(state_interval_union),
        previous_step,
        binary_distance_threshold=binary_distance_threshold,
    )
    if transition_decision.certified:
        return transition_decision
    if (
        previous_step.event == "enter_binary"
        and previous_step.pair is not None
        and previous_step.event_union_certificate is not None
        and previous_step.event_union_certificate.pair == previous_step.pair
        and previous_step.event_union_certificate.interval_is_isolated
    ):
        return IntervalChartDecision(
            chart="binary",
            pair=previous_step.pair,
            certified=True,
            pair_distance_bounds=union_decision.pair_distance_bounds,
            reason="certified union ordinary binary-entry event selects the next binary chart",
        )
    return union_decision


def _pair_distance(positions: Array, pair: tuple[int, int]) -> float:
    return float(np.linalg.norm(positions[pair[0]] - positions[pair[1]]))


def polynomial_real_roots_in_interval(coefficients: Array, lower: float, upper: float, *, tol: float = 1e-9) -> list[float]:
    """Return real roots of a scalar polynomial in a closed interval."""

    coefficients = np.asarray(coefficients, dtype=float)
    while coefficients.shape[0] > 1 and abs(coefficients[-1]) < tol:
        coefficients = coefficients[:-1]
    if coefficients.shape[0] <= 1:
        return []
    roots = np.polynomial.polynomial.polyroots(coefficients)
    real_roots = []
    for root in roots:
        if abs(root.imag) <= tol * max(1.0, abs(root.real)):
            value = float(root.real)
            if lower - tol <= value <= upper + tol:
                real_roots.append(float(np.clip(value, lower, upper)))
    return sorted(real_roots)


def _as_interval(value: object) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


def _interval_root_evidence(
    coefficients: Array,
    derivative: Array,
    root: float,
    bracket: tuple[float, float],
    *,
    direction: str,
    tol: float,
    coefficient_intervals: tuple[FloatInterval, ...] | None = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Shrink a root bracket until interval endpoint/derivative signs are clear.

    With coefficient intervals, this checks the whole enclosed polynomial
    family. It is a step toward proof-grade event isolation, not a formal
    statement that the upstream Taylor coefficients have been interval-proved.
    """

    left, right = bracket
    best = (left, right)
    checked_coefficients = coefficient_intervals if coefficient_intervals is not None else coefficients
    checked_derivative = interval_polyder(coefficient_intervals) if coefficient_intervals is not None else derivative
    best_left = interval_polynomial_eval(checked_coefficients, FloatInterval.point(left))
    best_right = interval_polynomial_eval(checked_coefficients, FloatInterval.point(right))
    best_derivative = (
        interval_polynomial_eval(checked_derivative, FloatInterval(left, right))
        if len(checked_derivative)
        else FloatInterval.point(0.0)
    )
    for _ in range(30):
        left_value = interval_polynomial_eval(checked_coefficients, FloatInterval.point(left))
        right_value = interval_polynomial_eval(checked_coefficients, FloatInterval.point(right))
        derivative_value = (
            interval_polynomial_eval(checked_derivative, FloatInterval(left, right))
            if len(checked_derivative)
            else FloatInterval.point(0.0)
        )
        left_sign = interval_sign(left_value)
        right_sign = interval_sign(right_value)
        derivative_sign = interval_sign(derivative_value)
        endpoint_ok = left_sign * right_sign < 0
        derivative_ok = (
            (direction == "decreasing" and derivative_sign < 0)
            or (direction == "increasing" and derivative_sign > 0)
        )
        best = (left, right)
        best_left = left_value
        best_right = right_value
        best_derivative = derivative_value
        if endpoint_ok and derivative_ok:
            break
        left_width = root - left
        right_width = right - root
        if min(left_width, right_width) <= max(100.0 * tol, 16.0 * np.finfo(float).eps):
            break
        left = root - 0.5 * left_width
        right = root + 0.5 * right_width

    return best, best_left.as_tuple(), best_right.as_tuple(), best_derivative.as_tuple()


def _direction_signs(direction: str) -> tuple[int, int, int]:
    if direction == "decreasing":
        return 1, -1, -1
    if direction == "increasing":
        return -1, 1, 1
    raise ValueError("direction must be 'decreasing' or 'increasing'")


def _signed_polynomial_range_evidence(
    coefficient_intervals: tuple[FloatInterval, ...],
    lower: float,
    upper: float,
    *,
    sign: int,
    max_subintervals: int = 128,
) -> tuple[tuple[float, float], ...] | None:
    if sign == 0:
        raise ValueError("sign must be nonzero")
    if upper < lower:
        raise ValueError("range upper endpoint cannot be below lower endpoint")
    if lower == upper:
        value = interval_polynomial_eval(coefficient_intervals, FloatInterval.point(lower))
        return (value.as_tuple(),) if interval_sign(value) == sign else None

    pieces = 1
    while pieces <= max_subintervals:
        value_intervals = []
        ok = True
        for index in range(pieces):
            piece_lower = lower + (upper - lower) * index / pieces
            piece_upper = lower + (upper - lower) * (index + 1) / pieces
            value = interval_polynomial_eval(
                coefficient_intervals,
                FloatInterval(piece_lower, piece_upper),
            )
            if interval_sign(value) != sign:
                ok = False
                break
            value_intervals.append(value.as_tuple())
        if ok:
            return tuple(value_intervals)
        pieces *= 2
    return None


def _enclose_polynomial_root_interval(
    root_seed: float,
    lower: float,
    upper: float,
    *,
    direction: str,
    coefficient_intervals: tuple[FloatInterval, ...],
    coefficient_source: str | None = None,
    initial_radius: float | None = None,
    max_expansions: int = 60,
    pre_event_subintervals: int = 128,
) -> PolynomialRootIntervalEnclosure | None:
    """Return an interval-time enclosure proving a unique earliest crossing.

    The seed may come from a point polynomial, but the returned certificate is
    accepted only from interval-coefficient signs: all admissible polynomials
    have the pre-event sign before the interval, opposite endpoint signs across
    it, and a consistent derivative sign inside it.
    """

    entry_sign, exit_sign, derivative_sign = _direction_signs(direction)
    if lower < 0.0 or upper <= lower:
        raise ValueError("invalid root search range")
    if not (lower <= root_seed <= upper):
        return None

    derivative_intervals = interval_polyder(coefficient_intervals)
    span = upper - lower
    radius = initial_radius
    if radius is None:
        radius = max(1e-12 * max(1.0, abs(root_seed)), 1e-12 * span, 16.0 * np.finfo(float).eps)

    for _ in range(max_expansions):
        left = max(lower, root_seed - radius)
        right = min(upper, root_seed + radius)
        if left < right:
            left_value = interval_polynomial_eval(coefficient_intervals, FloatInterval.point(left))
            right_value = interval_polynomial_eval(coefficient_intervals, FloatInterval.point(right))
            if interval_sign(left_value) == entry_sign and interval_sign(right_value) == exit_sign:
                derivative_value = (
                    interval_polynomial_eval(derivative_intervals, FloatInterval(left, right))
                    if derivative_intervals
                    else FloatInterval.point(0.0)
                )
                if interval_sign(derivative_value) == derivative_sign:
                    pre_event_values = _signed_polynomial_range_evidence(
                        coefficient_intervals,
                        lower,
                        left,
                        sign=entry_sign,
                        max_subintervals=pre_event_subintervals,
                    )
                    if pre_event_values is not None:
                        return PolynomialRootIntervalEnclosure(
                            interval=(float(left), float(right)),
                            direction=direction,
                            value_intervals=(left_value.as_tuple(), right_value.as_tuple()),
                            derivative_interval=derivative_value.as_tuple(),
                            pre_event_range=(float(lower), float(left)),
                            pre_event_value_intervals=pre_event_values,
                            coefficient_intervals=interval_coefficients_as_tuples(coefficient_intervals),
                            coefficient_source=coefficient_source,
                        )
        if left == lower and right == upper:
            break
        radius *= 2.0
    return None


def first_interval_polynomial_crossing(
    coefficient_intervals: tuple[FloatInterval, ...],
    lower: float,
    upper: float,
    *,
    direction: str,
    coefficient_source: str | None = None,
    initial_subintervals: int = 64,
    max_subintervals: int = 4096,
    pre_event_subintervals: int = 128,
    max_width: float | None = None,
) -> PolynomialRootIntervalEnclosure | None:
    """Find the first certified crossing using only interval polynomial signs."""

    if lower < 0.0 or upper <= lower:
        raise ValueError("invalid crossing search range")
    if initial_subintervals <= 0 or max_subintervals < initial_subintervals:
        raise ValueError("invalid subdivision limits")
    if max_width is not None and max_width <= 0.0:
        raise ValueError("max_width must be positive")
    coefficient_intervals = tuple(coefficient_intervals)
    entry_sign, exit_sign, derivative_sign = _direction_signs(direction)
    derivative_intervals = interval_polyder(coefficient_intervals)
    pieces = initial_subintervals
    best_enclosure = None
    while pieces <= max_subintervals:
        span = upper - lower
        grids = (
            [lower + span * index / pieces for index in range(pieces + 1)],
            [lower]
            + [lower + span * (index + 0.5) / pieces for index in range(pieces)]
            + [upper],
        )
        for points in grids:
            for left, right in zip(points, points[1:]):
                left_value = interval_polynomial_eval(coefficient_intervals, FloatInterval.point(left))
                if interval_sign(left_value) != entry_sign:
                    continue
                right_value = interval_polynomial_eval(coefficient_intervals, FloatInterval.point(right))
                if interval_sign(right_value) != exit_sign:
                    continue
                derivative_value = (
                    interval_polynomial_eval(derivative_intervals, FloatInterval(left, right))
                    if derivative_intervals
                    else FloatInterval.point(0.0)
                )
                if interval_sign(derivative_value) != derivative_sign:
                    continue
                pre_event_values = _signed_polynomial_range_evidence(
                    coefficient_intervals,
                    lower,
                    left,
                    sign=entry_sign,
                    max_subintervals=pre_event_subintervals,
                )
                if pre_event_values is None:
                    continue
                enclosure = PolynomialRootIntervalEnclosure(
                    interval=(float(left), float(right)),
                    direction=direction,
                    value_intervals=(left_value.as_tuple(), right_value.as_tuple()),
                    derivative_interval=derivative_value.as_tuple(),
                    pre_event_range=(float(lower), float(left)),
                    pre_event_value_intervals=pre_event_values,
                    coefficient_intervals=interval_coefficients_as_tuples(coefficient_intervals),
                    coefficient_source=coefficient_source,
                )
                if max_width is None or enclosure.width <= max_width:
                    return enclosure
                if best_enclosure is None or enclosure.width < best_enclosure.width:
                    best_enclosure = enclosure
        pieces *= 2
    return best_enclosure


def certify_polynomial_root(
    coefficients: Array,
    root: float,
    lower: float,
    upper: float,
    *,
    direction: str,
    coefficient_intervals: tuple[FloatInterval, ...] | None = None,
    coefficient_source: str | None = None,
    tol: float = 1e-9,
) -> PolynomialRootCertificate:
    """Build a local isolation certificate for a simple polynomial root."""

    if direction not in {"decreasing", "increasing"}:
        raise ValueError("direction must be 'decreasing' or 'increasing'")
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficient_intervals is not None and len(coefficient_intervals) != coefficients.shape[0]:
        raise ValueError("coefficient interval count must match coefficient count")
    derivative = np.polynomial.polynomial.polyder(coefficients)
    polynomial_roots = polynomial_real_roots_in_interval(coefficients, lower, upper, tol=tol)
    derivative_roots = polynomial_real_roots_in_interval(derivative, lower, upper, tol=tol) if derivative.size else []
    neighbors = [lower, upper]
    neighbors.extend(candidate for candidate in polynomial_roots if abs(candidate - root) > 100.0 * tol)
    neighbors.extend(derivative_roots)
    left_limit = max(candidate for candidate in neighbors if candidate < root)
    right_limit = min(candidate for candidate in neighbors if candidate > root)
    margin = min(root - left_limit, right_limit - root) * 0.25
    if margin <= 0.0:
        margin = max(100.0 * tol, np.finfo(float).eps)
    bracket = (max(lower, root - margin), min(upper, root + margin))
    bracket, left_interval, right_interval, derivative_interval = _interval_root_evidence(
        coefficients,
        derivative,
        root,
        bracket,
        direction=direction,
        tol=tol,
        coefficient_intervals=coefficient_intervals,
    )
    values = (
        float(np.polynomial.polynomial.polyval(bracket[0], coefficients)),
        float(np.polynomial.polynomial.polyval(bracket[1], coefficients)),
    )
    derivative_at_root = float(np.polynomial.polynomial.polyval(root, derivative)) if derivative.size else 0.0
    derivative_sign = int(np.sign(derivative_at_root))
    derivative_roots_in_bracket = sum(bracket[0] < candidate < bracket[1] for candidate in derivative_roots)
    if direction == "decreasing" and derivative_sign > 0:
        derivative_sign = 0
    elif direction == "increasing" and derivative_sign < 0:
        derivative_sign = 0
    root_enclosure = (
        _enclose_polynomial_root_interval(
            root,
            lower,
            upper,
            direction=direction,
            coefficient_intervals=coefficient_intervals,
            coefficient_source=coefficient_source,
        )
        if coefficient_intervals is not None
        else None
    )
    return PolynomialRootCertificate(
        root=float(root),
        bracket=(float(bracket[0]), float(bracket[1])),
        values=values,
        derivative_sign=derivative_sign,
        derivative_roots_in_bracket=int(derivative_roots_in_bracket),
        value_intervals=(left_interval, right_interval),
        derivative_interval=derivative_interval,
        coefficient_intervals=(
            interval_coefficients_as_tuples(coefficient_intervals)
            if coefficient_intervals is not None
            else None
        ),
        coefficient_source=coefficient_source,
        root_enclosure=root_enclosure,
    )


def _threshold_root_from_polynomial(
    polynomial: Array,
    trial_step: float,
    *,
    direction: str,
    coefficient_intervals: tuple[FloatInterval, ...] | None = None,
    coefficient_source: str | None = None,
) -> PolynomialRootCertificate | None:
    if direction not in {"decreasing", "increasing"}:
        raise ValueError("direction must be 'decreasing' or 'increasing'")
    derivative = np.polynomial.polynomial.polyder(polynomial)
    candidates = []
    for root in polynomial_real_roots_in_interval(polynomial, 0.0, trial_step):
        if root <= 10.0 * np.finfo(float).eps:
            continue
        slope = float(np.polynomial.polynomial.polyval(root, derivative)) if derivative.size else 0.0
        if direction == "decreasing" and slope < 0.0:
            candidates.append(
                certify_polynomial_root(
                    polynomial,
                    root,
                    0.0,
                    trial_step,
                    direction=direction,
                    coefficient_intervals=coefficient_intervals,
                    coefficient_source=coefficient_source,
                )
            )
        elif direction == "increasing" and slope > 0.0:
            candidates.append(
                certify_polynomial_root(
                    polynomial,
                    root,
                    0.0,
                    trial_step,
                    direction=direction,
                    coefficient_intervals=coefficient_intervals,
                    coefficient_source=coefficient_source,
                )
            )
    if not candidates:
        return None
    return min(candidates, key=lambda certificate: certificate.root)


def _pair_distance_squared_polynomial(position_coefficients: Array, pair: tuple[int, int]) -> Array:
    max_degree = position_coefficients.shape[0] - 1
    delta = position_coefficients[:, pair[0], :] - position_coefficients[:, pair[1], :]
    out = np.zeros(2 * max_degree + 1, dtype=float)
    for axis in range(delta.shape[1]):
        out += np.convolve(delta[:, axis], delta[:, axis])
    return out


def _pair_distance_squared_polynomial_intervals(
    position_coefficients: Array,
    pair: tuple[int, int],
) -> tuple[FloatInterval, ...]:
    max_degree = position_coefficients.shape[0] - 1
    dimension = position_coefficients.shape[2]
    out = tuple(FloatInterval.point(0.0) for _ in range(2 * max_degree + 1))
    for axis in range(dimension):
        delta = tuple(
            _as_interval(position_coefficients[n, pair[0], axis])
            - _as_interval(position_coefficients[n, pair[1], axis])
            for n in range(max_degree + 1)
        )
        out = interval_series_add(out, interval_series_product(delta, delta, 2 * max_degree))
    return out


def _squared_norm_polynomial(vector_coefficients: Array) -> Array:
    max_degree = vector_coefficients.shape[0] - 1
    out = np.zeros(2 * max_degree + 1, dtype=float)
    for axis in range(vector_coefficients.shape[1]):
        out += np.convolve(vector_coefficients[:, axis], vector_coefficients[:, axis])
    return out


def _squared_norm_polynomial_intervals(vector_coefficients: Array) -> tuple[FloatInterval, ...]:
    max_degree = vector_coefficients.shape[0] - 1
    out = tuple(FloatInterval.point(0.0) for _ in range(2 * max_degree + 1))
    for axis in range(vector_coefficients.shape[1]):
        axis_coefficients = tuple(
            _as_interval(vector_coefficients[n, axis])
            for n in range(max_degree + 1)
        )
        out = interval_series_add(
            out,
            interval_series_product(axis_coefficients, axis_coefficients, 2 * max_degree),
        )
    return out


def _ordinary_end_state_interval(
    positions: Array,
    velocities: Array,
    masses: Array,
    physical_step: float,
    *,
    order: int,
    start_state_interval: tuple[tuple[float, float], ...] | None = None,
) -> tuple[tuple[float, float], ...]:
    if start_state_interval is None:
        series = construct_interval_taylor_solution(positions, velocities, masses, order=order)
    else:
        interval_positions, interval_velocities = _unpack_interval_planar_state(start_state_interval)
        series = construct_interval_taylor_solution_from_intervals(
            interval_positions,
            interval_velocities,
            masses,
            order=order,
        )
    return _pack_interval_planar_state(series.positions_at(physical_step), series.velocities_at(physical_step))


def _ordinary_end_state_interval_over_time_interval(
    positions: Array,
    velocities: Array,
    masses: Array,
    time_interval: tuple[float, float],
    *,
    order: int,
    start_state_interval: tuple[tuple[float, float], ...] | None = None,
) -> tuple[tuple[float, float], ...]:
    if time_interval[0] < 0.0 or time_interval[1] < time_interval[0]:
        raise ValueError("invalid ordinary time interval")
    if start_state_interval is None:
        series = construct_interval_taylor_solution(positions, velocities, masses, order=order)
    else:
        interval_positions, interval_velocities = _unpack_interval_planar_state(start_state_interval)
        series = construct_interval_taylor_solution_from_intervals(
            interval_positions,
            interval_velocities,
            masses,
            order=order,
        )
    time = FloatInterval(*time_interval)
    return _pack_interval_planar_state(
        interval_array_series_eval(series.position, time),
        interval_array_series_eval(series.velocity, time),
    )


def _ordinary_end_state_interval_union(
    positions: Array,
    velocities: Array,
    masses: Array,
    physical_step: float,
    *,
    order: int,
    start_state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Propagate a union of ordinary interval states member-by-member."""

    if not start_state_interval_union:
        raise ValueError("ordinary interval-state union cannot be empty")
    return tuple(
        _ordinary_end_state_interval(
            positions,
            velocities,
            masses,
            physical_step,
            order=order,
            start_state_interval=start_state_interval,
        )
        for start_state_interval in start_state_interval_union
    )


def _ordinary_end_state_interval_union_over_time_intervals(
    positions: Array,
    velocities: Array,
    masses: Array,
    time_intervals: tuple[tuple[float, float], ...],
    *,
    order: int,
    start_state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Propagate each ordinary interval-state union member over its own time interval."""

    if not start_state_interval_union:
        raise ValueError("ordinary interval-state union cannot be empty")
    if len(time_intervals) != len(start_state_interval_union):
        raise ValueError("time interval count must match interval-state union")
    return tuple(
        _ordinary_end_state_interval_over_time_interval(
            positions,
            velocities,
            masses,
            time_interval,
            order=order,
            start_state_interval=start_state_interval,
        )
        for time_interval, start_state_interval in zip(time_intervals, start_state_interval_union)
    )


def _lc_square_interval(z: Array) -> Array:
    x, y = z
    x_square = FloatInterval(*_interval_square_bounds(x))
    y_square = FloatInterval(*_interval_square_bounds(y))
    return np.array([x_square - y_square, (x * y).scale(2.0)], dtype=object)


def _lc_velocity_interval(z: Array, z_velocity: Array) -> Array:
    x, y = z
    vx, vy = z_velocity
    rho = FloatInterval(*_interval_square_bounds(x)) + FloatInterval(*_interval_square_bounds(y))
    return np.array(
        [
            ((x * vx).scale(2.0) - (y * vy).scale(2.0)) / rho,
            ((y * vx).scale(2.0) + (x * vy).scale(2.0)) / rho,
        ],
        dtype=object,
    )


def _regularized_binary_end_state_interval(
    initial_state,
    s_step: float,
    *,
    order: int,
    interval_initial_state=None,
) -> tuple[tuple[float, float], ...]:
    if interval_initial_state is None:
        solution = construct_interval_regularized_binary_taylor_solution(initial_state, order=order)
    else:
        solution = construct_interval_regularized_binary_taylor_solution_from_intervals(
            interval_initial_state,
            order=order,
        )
    z = solution.z_at(s_step)
    z_velocity = solution.z_velocity_at(s_step)
    relative_position = _lc_square_interval(z)
    relative_velocity = _lc_velocity_interval(z, z_velocity)
    binary_center = solution.binary_center_at(s_step)
    binary_center_velocity = solution.binary_center_velocity_at(s_step)
    third_offset = solution.third_offset_at(s_step)
    third_offset_velocity = solution.third_offset_velocity_at(s_step)

    first, second = initial_state.pair
    third = initial_state.third_index
    masses = np.asarray(initial_state.masses, dtype=float)
    pair_mass = masses[first] + masses[second]

    positions = np.empty((3, 2), dtype=object)
    velocities = np.empty((3, 2), dtype=object)
    for axis in range(2):
        positions[first, axis] = binary_center[axis] - relative_position[axis].scale(masses[second] / pair_mass)
        positions[second, axis] = binary_center[axis] + relative_position[axis].scale(masses[first] / pair_mass)
        positions[third, axis] = binary_center[axis] + third_offset[axis]
        velocities[first, axis] = binary_center_velocity[axis] - relative_velocity[axis].scale(masses[second] / pair_mass)
        velocities[second, axis] = binary_center_velocity[axis] + relative_velocity[axis].scale(masses[first] / pair_mass)
        velocities[third, axis] = binary_center_velocity[axis] + third_offset_velocity[axis]

    return _pack_interval_planar_state(positions, velocities)


def _hull_state_intervals(
    states: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[float, float], ...]:
    if not states:
        raise ValueError("cannot hull an empty collection of interval states")
    length = len(states[0])
    if any(len(state) != length for state in states):
        raise ValueError("all interval states must have the same length")
    out = []
    for index in range(length):
        lower = min(state[index][0] for state in states)
        upper = max(state[index][1] for state in states)
        out.append((float(np.nextafter(lower, -np.inf)), float(np.nextafter(upper, np.inf))))
    return tuple(out)


def _regularized_binary_atlas_end_state_intervals(
    initial_state,
    atlas_initial_states,
    s_step: float,
    *,
    order: int,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Propagate each interval binary atlas chart to a projected endpoint interval."""

    return tuple(
        _regularized_binary_end_state_interval(
            initial_state,
            s_step,
            order=order,
            interval_initial_state=atlas_initial_state,
        )
        for atlas_initial_state in atlas_initial_states
    )


def _regularized_binary_atlas_for_state_interval_union(
    state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
    masses: Array,
    *,
    pair: tuple[int, int],
):
    """Lift every planar interval-state union member into binary atlas charts."""

    if not state_interval_union:
        raise ValueError("state interval union cannot be empty")
    charts = []
    for state_interval in state_interval_union:
        charts.extend(
            planar_interval_to_regularized_binary_collision_chart_atlas(
                state_interval,
                masses,
                pair=pair,
            )
        )
    return tuple(charts)


def _regularized_binary_atlas_end_state_interval(
    initial_state,
    atlas_initial_states,
    s_step: float,
    *,
    order: int,
) -> tuple[tuple[float, float], ...]:
    """Propagate each interval binary atlas chart and hull projected endpoints."""

    return _hull_state_intervals(
        _regularized_binary_atlas_end_state_intervals(
            initial_state,
            atlas_initial_states,
            s_step,
            order=order,
        )
    )


def _first_distance_crossing_from_position_series(
    position_coefficients: Array,
    pairs: tuple[tuple[int, int], ...],
    trial_step: float,
    threshold: float,
    *,
    direction: str,
    position_coefficient_intervals: Array | None = None,
) -> tuple[float, tuple[int, int], PolynomialRootCertificate] | None:
    if trial_step <= 0.0:
        raise ValueError("trial_step must be positive")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    roots: list[tuple[float, tuple[int, int]]] = []
    for pair in pairs:
        polynomial = _pair_distance_squared_polynomial(position_coefficients, pair)
        polynomial[0] -= threshold**2
        interval_positions = (
            position_coefficients
            if position_coefficient_intervals is None
            else position_coefficient_intervals
        )
        coefficient_intervals = subtract_interval_from_constant_term(
            _pair_distance_squared_polynomial_intervals(interval_positions, pair),
            FloatInterval.point(threshold) * FloatInterval.point(threshold),
        )
        certificate = _threshold_root_from_polynomial(
            polynomial,
            trial_step,
            direction=direction,
            coefficient_intervals=coefficient_intervals,
            coefficient_source=(
                "point_taylor"
                if position_coefficient_intervals is None
                else "ordinary_interval_taylor"
            ),
        )
        if certificate is not None:
            roots.append((certificate.root, pair, certificate))

    if not roots:
        return None
    return min(roots, key=lambda item: item[0])


def _first_distance_crossing_from_interval_position_series(
    position_coefficient_intervals: Array,
    pairs: tuple[tuple[int, int], ...],
    trial_step: float,
    threshold: float,
    *,
    direction: str,
    coefficient_source: str,
) -> tuple[tuple[int, int], PolynomialRootIntervalEnclosure] | None:
    if trial_step <= 0.0:
        raise ValueError("trial_step must be positive")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    crossings: list[tuple[float, tuple[int, int], PolynomialRootIntervalEnclosure]] = []
    threshold_square = FloatInterval.point(threshold) * FloatInterval.point(threshold)
    for pair in pairs:
        coefficient_intervals = subtract_interval_from_constant_term(
            _pair_distance_squared_polynomial_intervals(position_coefficient_intervals, pair),
            threshold_square,
        )
        enclosure = first_interval_polynomial_crossing(
            coefficient_intervals,
            0.0,
            trial_step,
            direction=direction,
            coefficient_source=coefficient_source,
            initial_subintervals=16,
            max_subintervals=2048,
            max_width=max(1e-8, trial_step / 2048.0),
        )
        if enclosure is not None:
            crossings.append((enclosure.interval[0], pair, enclosure))

    if not crossings:
        return None
    _left, pair, enclosure = min(crossings, key=lambda item: item[0])
    return pair, enclosure


def ordinary_taylor_step(
    positions: Array,
    velocities: Array,
    masses: Array,
    physical_step: float,
    *,
    order: int,
) -> tuple[Array, Array, float]:
    series = construct_taylor_solution(positions, velocities, masses, order=order)
    return series.positions_at(physical_step), series.velocities_at(physical_step), truncation_indicator(series, physical_step)


def event_limited_ordinary_taylor_step(
    positions: Array,
    velocities: Array,
    masses: Array,
    trial_physical_step: float,
    *,
    order: int,
    binary_enter_distance: float,
    start_state_interval: tuple[tuple[float, float], ...] | None = None,
) -> tuple[Array, Array, float, float, tuple[int, int] | None, PolynomialRootCertificate | None]:
    """Take an ordinary Taylor step, stopping at first binary-entry event."""

    series = construct_taylor_solution(positions, velocities, masses, order=order)
    if start_state_interval is None:
        interval_series = construct_interval_taylor_solution(positions, velocities, masses, order=order)
    else:
        interval_positions, interval_velocities = _unpack_interval_planar_state(start_state_interval)
        interval_series = construct_interval_taylor_solution_from_intervals(
            interval_positions,
            interval_velocities,
            masses,
            order=order,
        )
    crossing = _first_distance_crossing_from_position_series(
        series.position,
        ((0, 1), (0, 2), (1, 2)),
        trial_physical_step,
        binary_enter_distance,
        direction="decreasing",
        position_coefficient_intervals=interval_series.position,
    )
    if crossing is None:
        physical_step = trial_physical_step
        event_pair = None
        certificate = None
    else:
        physical_step, event_pair, certificate = crossing
    return (
        series.positions_at(physical_step),
        series.velocities_at(physical_step),
        physical_step,
        truncation_indicator(series, physical_step),
        event_pair,
        certificate,
    )


def _ordinary_binary_entry_event_certificate_for_interval(
    position_coefficients: Array,
    masses: Array,
    *,
    event_root: float,
    event_pair: tuple[int, int],
    trial_physical_step: float,
    binary_enter_distance: float,
    order: int,
    state_interval: tuple[tuple[float, float], ...],
    coefficient_source: str,
    root_enclosure: PolynomialRootIntervalEnclosure | None = None,
) -> PolynomialRootCertificate:
    interval_positions, interval_velocities = _unpack_interval_planar_state(state_interval)
    interval_series = construct_interval_taylor_solution_from_intervals(
        interval_positions,
        interval_velocities,
        masses,
        order=order,
    )
    threshold_square = FloatInterval.point(binary_enter_distance) * FloatInterval.point(binary_enter_distance)
    polynomial = _pair_distance_squared_polynomial(position_coefficients, event_pair)
    polynomial[0] -= binary_enter_distance**2
    coefficient_intervals = subtract_interval_from_constant_term(
        _pair_distance_squared_polynomial_intervals(interval_series.position, event_pair),
        threshold_square,
    )
    certificate = certify_polynomial_root(
        polynomial,
        event_root,
        0.0,
        trial_physical_step,
        direction="decreasing",
        coefficient_intervals=coefficient_intervals,
        coefficient_source=coefficient_source,
    )
    if root_enclosure is not None:
        certificate = replace(certificate, root_enclosure=root_enclosure)
    return certificate


def certify_ordinary_binary_entry_event_union(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    trial_physical_step: float,
    event_root: float,
    event_pair: tuple[int, int],
    binary_enter_distance: float,
    order: int,
    start_state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
    member_root_enclosures: tuple[PolynomialRootIntervalEnclosure | None, ...] | None = None,
) -> PolynomialRootUnionCertificate:
    """Certify the same ordinary binary-entry root over each interval union member."""

    if not start_state_interval_union:
        raise ValueError("start_state_interval_union cannot be empty")
    if member_root_enclosures is not None and len(member_root_enclosures) != len(start_state_interval_union):
        raise ValueError("member_root_enclosures must match start_state_interval_union")
    series = construct_taylor_solution(positions, velocities, masses, order=order)
    member_certificates = []
    root_enclosures = member_root_enclosures
    if root_enclosures is None:
        root_enclosures = tuple(None for _state_interval in start_state_interval_union)
    for state_interval, root_enclosure in zip(start_state_interval_union, root_enclosures):
        member_certificates.append(
            _ordinary_binary_entry_event_certificate_for_interval(
                series.position,
                masses,
                event_root=event_root,
                event_pair=event_pair,
                trial_physical_step=trial_physical_step,
                binary_enter_distance=binary_enter_distance,
                order=order,
                state_interval=state_interval,
                coefficient_source="ordinary_interval_taylor_union_member",
                root_enclosure=root_enclosure,
            )
        )
    return PolynomialRootUnionCertificate(
        root=float(event_root),
        pair=event_pair,
        member_certificates=tuple(member_certificates),
        member_root_enclosures=tuple(certificate.root_enclosure for certificate in member_certificates),
    )


def first_ordinary_binary_entry_event_union(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    trial_physical_step: float,
    binary_enter_distance: float,
    order: int,
    start_state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
    root_tolerance: float = 1e-10,
) -> tuple[float, tuple[int, int], PolynomialRootUnionCertificate] | None:
    """Find a common first ordinary binary-entry event across interval union members."""

    if not start_state_interval_union:
        raise ValueError("start_state_interval_union cannot be empty")
    point_series = construct_taylor_solution(positions, velocities, masses, order=order)
    crossings = []
    for state_interval in start_state_interval_union:
        interval_positions, interval_velocities = _unpack_interval_planar_state(state_interval)
        interval_series = construct_interval_taylor_solution_from_intervals(
            interval_positions,
            interval_velocities,
            masses,
            order=order,
        )
        crossing = _first_distance_crossing_from_interval_position_series(
            interval_series.position,
            ((0, 1), (0, 2), (1, 2)),
            trial_physical_step,
            binary_enter_distance,
            direction="decreasing",
            coefficient_source="ordinary_interval_taylor_union_member_interval_search",
        )
        if crossing is None:
            return None
        crossings.append(crossing)

    pairs = {pair for pair, _enclosure in crossings}
    if len(pairs) != 1:
        return None
    pair = next(iter(pairs))
    enclosures = tuple(enclosure for _pair, enclosure in crossings)
    common_lower = max(enclosure.interval[0] for enclosure in enclosures)
    common_upper = min(enclosure.interval[1] for enclosure in enclosures)
    if common_lower > common_upper:
        return None
    polynomial = _pair_distance_squared_polynomial(point_series.position, pair)
    polynomial[0] -= binary_enter_distance**2
    point_certificate = _threshold_root_from_polynomial(
        polynomial,
        trial_physical_step,
        direction="decreasing",
    )
    if point_certificate is None:
        return None
    root = point_certificate.root
    if not (
        common_lower - root_tolerance <= root <= common_upper + root_tolerance
    ):
        return None
    union_certificate = certify_ordinary_binary_entry_event_union(
        positions,
        velocities,
        masses,
        trial_physical_step=trial_physical_step,
        event_root=root,
        event_pair=pair,
        binary_enter_distance=binary_enter_distance,
        order=order,
        start_state_interval_union=start_state_interval_union,
        member_root_enclosures=enclosures,
    )
    if not union_certificate.interval_is_isolated or not union_certificate.earliest_interval_is_certified:
        return None
    return root, pair, union_certificate


def regularized_binary_taylor_step(
    positions: Array,
    velocities: Array,
    masses: Array,
    pair: tuple[int, int],
    physical_remaining: float,
    *,
    order: int,
    max_s_step: float,
    binary_exit_distance: float | None = None,
) -> tuple[Array, Array, float, float, PolynomialRootCertificate | None]:
    """Advance by a binary chart and return positions, velocities, s step, physical step."""

    if physical_remaining <= 0.0:
        raise ValueError("regularized binary step currently advances forward in physical time")
    if max_s_step <= 0.0:
        raise ValueError("max_s_step must be positive")
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=pair)
    series = construct_regularized_binary_taylor_solution(initial, order=order)
    trial_s = max_s_step
    exit_certificate = None
    if binary_exit_distance is not None:
        if binary_exit_distance <= 0.0:
            raise ValueError("binary_exit_distance must be positive")
        rho_polynomial = _squared_norm_polynomial(series.z)
        rho_polynomial[0] -= binary_exit_distance
        interval_series = construct_interval_regularized_binary_taylor_solution(initial, order=order)
        rho_coefficient_intervals = subtract_interval_from_constant_term(
            _squared_norm_polynomial_intervals(interval_series.z),
            FloatInterval.point(binary_exit_distance),
        )
        exit_certificate = _threshold_root_from_polynomial(
            rho_polynomial,
            trial_s,
            direction="increasing",
            coefficient_intervals=rho_coefficient_intervals,
            coefficient_source="regularized_interval_taylor",
        )
        if exit_certificate is not None:
            trial_s = exit_certificate.root

    trial_physical = series.physical_time_at(trial_s)
    if trial_physical <= 0.0:
        raise RuntimeError("binary chart did not advance physical time")

    if trial_physical > physical_remaining:
        def root(s_value: float) -> float:
            return series.physical_time_at(s_value) - physical_remaining

        s_step = float(brentq(root, 0.0, trial_s, xtol=1e-15, rtol=1e-15))
        physical_step = physical_remaining
        exit_certificate = None
    else:
        s_step = trial_s
        physical_step = trial_physical

    end_state = series.state_at(s_step)
    end_positions, end_velocities = regularized_binary_collision_chart_to_planar(end_state)
    return end_positions, end_velocities, s_step, physical_step, exit_certificate


def continue_hybrid_from_regularized_binary_collision(
    initial_state: RegularizedBinaryCollisionChartState,
    initial_interval_state: IntervalRegularizedBinaryCollisionChartState,
    t_final: float,
    *,
    ordinary_order: int = 16,
    binary_order: int = 16,
    max_time_step: float = 0.02,
    max_binary_s_step: float = 0.02,
    binary_distance_threshold: float = 0.08,
    binary_exit_distance: float | None = None,
    safety: float = 0.08,
    max_steps: int = 10000,
    tail_guard_order: int = 0,
    tail_certificate_mode: str = "guarded",
    require_interval_chart_certification: bool = False,
) -> HybridContinuedSolution:
    """Continue from an exact binary collision represented in regularized variables.

    The ordinary planar API intentionally rejects exact binary collision because
    finite physical pair velocities are not defined there. This entry point
    treats the lifted regularized state as authoritative for the first binary
    chart, then hands the positive-``s`` projected endpoint to the usual hybrid
    continuation if more physical time remains.
    """

    if t_final <= 0.0:
        raise ValueError("regularized binary collision continuation requires positive final time")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if max_binary_s_step <= 0.0:
        raise ValueError("max_binary_s_step must be positive")
    if tail_guard_order < 0:
        raise ValueError("tail_guard_order cannot be negative")
    if tail_certificate_mode not in {"guarded", "cauchy"}:
        raise ValueError("tail_certificate_mode must be 'guarded' or 'cauchy'")

    masses = np.asarray(initial_state.masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    if initial_state.rho != 0.0:
        raise ValueError("initial_state must be an exact binary collision with z = 0")
    if initial_interval_state.pair != initial_state.pair:
        raise ValueError("initial interval state pair must match initial_state pair")
    if not np.allclose(np.asarray(initial_interval_state.masses, dtype=float), masses, rtol=0.0, atol=0.0):
        raise ValueError("initial interval state masses must match initial_state masses")
    if not initial_interval_state.contains_point(initial_state):
        raise ValueError("initial interval state must contain initial_state")
    branch_certificate = initial_interval_state.branch_certificate
    branch_certified = branch_certificate is not None and branch_certificate.certified
    if require_interval_chart_certification and not branch_certified:
        raise RuntimeError("initial regularized binary interval state is not branch-certified")
    if binary_exit_distance is None:
        binary_exit_distance = 1.25 * binary_distance_threshold
    if binary_exit_distance <= binary_distance_threshold:
        raise ValueError("binary_exit_distance must be greater than binary_distance_threshold")

    local_max_binary_s_step = float(max_binary_s_step)
    if tail_certificate_mode == "cauchy":
        cauchy_trial_certificate = regularized_binary_interval_cauchy_majorant_tail_certificate(
            initial_interval_state,
            retained_order=binary_order,
            step_size=0.0,
        )
        local_max_binary_s_step = min(local_max_binary_s_step, 0.5 * cauchy_trial_certificate.s_radius)

    series = construct_regularized_binary_taylor_solution(initial_state, order=binary_order)
    interval_series = construct_interval_regularized_binary_taylor_solution_from_intervals(
        initial_interval_state,
        order=binary_order,
    )
    residual_certificate = certify_regularized_binary_interval_taylor_equations(
        interval_series,
        coefficient_count=binary_order,
    )
    center_of_mass_certificate = certify_regularized_binary_center_of_mass_motion(
        interval_series,
        coefficient_count=binary_order,
    )
    linear_momentum_certificate = certify_regularized_binary_linear_momentum_conservation(
        interval_series,
        coefficient_count=binary_order,
    )
    angular_momentum_certificate = certify_regularized_binary_centered_angular_momentum_conservation(
        interval_series,
        coefficient_count=binary_order,
    )
    energy_certificate = certify_regularized_binary_total_energy_conservation(
        interval_series,
        coefficient_count=binary_order,
    )
    trial_s = local_max_binary_s_step
    event_certificate = None
    rho_polynomial = _squared_norm_polynomial(series.z)
    rho_polynomial[0] -= binary_exit_distance
    rho_coefficient_intervals = subtract_interval_from_constant_term(
        _squared_norm_polynomial_intervals(interval_series.z),
        FloatInterval.point(binary_exit_distance),
    )
    exit_certificate = _threshold_root_from_polynomial(
        rho_polynomial,
        trial_s,
        direction="increasing",
        coefficient_intervals=rho_coefficient_intervals,
        coefficient_source="regularized_interval_taylor",
    )
    if exit_certificate is not None:
        trial_s = exit_certificate.root
        event_certificate = exit_certificate

    trial_physical = series.physical_time_at(trial_s)
    if trial_physical <= 0.0:
        raise RuntimeError("regularized binary chart did not advance physical time")
    if trial_physical > t_final:
        def root(s_value: float) -> float:
            return series.physical_time_at(s_value) - t_final

        parameter_step = float(brentq(root, 0.0, trial_s, xtol=1e-15, rtol=1e-15))
        physical_step = float(t_final)
        event_certificate = None
        step_event = None
    else:
        parameter_step = float(trial_s)
        physical_step = float(trial_physical)
        step_event = "exit_binary" if event_certificate is not None else None

    end_state = series.state_at(parameter_step)
    end_positions, end_velocities = regularized_binary_collision_chart_to_planar(end_state)
    start_positions, start_velocities = _collision_position_summary(initial_state)
    start_state = pack_planar_state(start_positions, start_velocities)
    start_state_interval = _pack_interval_planar_state(start_positions, start_velocities)
    end_state_interval = _regularized_binary_end_state_interval(
        initial_state,
        parameter_step,
        order=binary_order,
        interval_initial_state=initial_interval_state,
    )
    if tail_certificate_mode == "cauchy":
        cauchy_step_size = parameter_step
        if event_certificate is not None and event_certificate.root_enclosure is not None:
            cauchy_step_size = _local_interval_step_size(event_certificate.root_enclosure.interval)
        truncation_certificate = regularized_binary_interval_cauchy_majorant_tail_certificate(
            initial_interval_state,
            retained_order=binary_order,
            step_size=cauchy_step_size,
        )
    elif tail_guard_order:
        truncation_certificate = regularized_binary_tail_certificate(
            initial_state,
            retained_order=binary_order,
            guard_order=tail_guard_order,
            step_size=parameter_step,
        )
    else:
        truncation_certificate = None

    endpoint_parameter_interval = (
        event_certificate.root_enclosure.interval
        if event_certificate is not None and event_certificate.root_enclosure is not None
        else (parameter_step, parameter_step)
    )
    projection_domain_certificate = certify_regularized_binary_projection_domain(
        interval_series,
        parameter_interval=(0.0, endpoint_parameter_interval[1]),
        endpoint_parameter_interval=endpoint_parameter_interval,
        branch_or_atlas_certified=branch_certified,
        starts_at_binary_collision=True,
        initial_collision_selector_certified=branch_certified,
    )
    residual_certificate = replace(
        residual_certificate,
        projection_domain_certificate=projection_domain_certificate,
    )

    first_step = HybridStep(
        chart="binary",
        pair=initial_state.pair,
        event=step_event,
        event_certificate=event_certificate,
        truncation_certificate=truncation_certificate,
        start_state_interval=start_state_interval,
        end_state_interval=end_state_interval,
        start_state_interval_union=(start_state_interval,),
        end_state_interval_union=(end_state_interval,),
        start_time=0.0,
        physical_step=physical_step,
        parameter_step=parameter_step,
        min_pair_distance=0.0,
        interval_min_pair_distance=0.0,
        interval_max_speed=float("inf"),
        interval_max_acceleration=float("inf"),
        interval_chart="binary",
        interval_chart_pair=initial_state.pair,
        interval_chart_certified=branch_certified,
        binary_interval_lift_certified=initial_interval_state.contains_point(initial_state),
        binary_interval_lift_reason=(
            "lifted exact binary collision interval start"
            if branch_certified
            else "lifted exact binary collision interval start is not branch-certified"
        ),
        binary_lc_branch=branch_certificate.branch if branch_certificate is not None else None,
        binary_lc_branch_certified=branch_certified,
        start_regularized_interval_state=initial_interval_state,
        residual_certificate=residual_certificate,
        center_of_mass_certificate=center_of_mass_certificate,
        linear_momentum_certificate=linear_momentum_certificate,
        angular_momentum_certificate=angular_momentum_certificate,
        energy_certificate=energy_certificate,
    )
    times = [0.0, physical_step]
    states = [start_state, pack_planar_state(end_positions, end_velocities)]
    steps: tuple[HybridStep, ...] = (first_step,)

    remaining = t_final - physical_step
    if remaining > 10.0 * np.finfo(float).eps:
        if max_steps <= 1:
            raise RuntimeError("hybrid continuation exceeded max_steps")
        rest = continue_hybrid(
            end_positions,
            end_velocities,
            masses,
            remaining,
            ordinary_order=ordinary_order,
            binary_order=binary_order,
            max_time_step=max_time_step,
            max_binary_s_step=max_binary_s_step,
            binary_distance_threshold=binary_distance_threshold,
            binary_exit_distance=binary_exit_distance,
            safety=safety,
            max_steps=max_steps - 1,
            tail_guard_order=tail_guard_order,
            tail_certificate_mode=tail_certificate_mode,
            initial_state_interval=end_state_interval,
            require_interval_chart_certification=require_interval_chart_certification,
        )
        times.extend(float(physical_step + value) for value in rest.times[1:])
        states.extend(rest.states[1:])
        steps = steps + tuple(replace(step, start_time=step.start_time + physical_step) for step in rest.steps)

    return HybridContinuedSolution(
        masses=masses,
        times=np.array(times, dtype=float),
        states=np.vstack(states),
        steps=steps,
    )


def continue_hybrid(
    positions: Array,
    velocities: Array,
    masses: Array,
    t_final: float,
    *,
    ordinary_order: int = 16,
    binary_order: int = 16,
    max_time_step: float = 0.02,
    max_binary_s_step: float = 0.02,
    binary_distance_threshold: float = 0.08,
    binary_exit_distance: float | None = None,
    safety: float = 0.08,
    max_steps: int = 10000,
    tail_guard_order: int = 0,
    tail_certificate_mode: str = "guarded",
    initial_state_interval: tuple[tuple[float, float], ...] | None = None,
    initial_state_interval_union: tuple[tuple[tuple[float, float], ...], ...] | None = None,
    require_interval_chart_certification: bool = False,
) -> HybridContinuedSolution:
    """Continue a planar three-body state using ordinary and binary charts."""

    if t_final < 0.0:
        raise ValueError("hybrid continuation currently advances forward in time")
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.shape != (3, 2) or velocities.shape != (3, 2):
        raise ValueError("hybrid continuation is planar and expects shape (3, 2)")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if tail_guard_order < 0:
        raise ValueError("tail_guard_order cannot be negative")
    if tail_certificate_mode not in {"guarded", "cauchy"}:
        raise ValueError("tail_certificate_mode must be 'guarded' or 'cauchy'")
    if binary_exit_distance is None:
        binary_exit_distance = 1.25 * binary_distance_threshold
    if binary_exit_distance <= binary_distance_threshold:
        raise ValueError("binary_exit_distance must be greater than binary_distance_threshold")

    current_positions = positions.copy()
    current_velocities = velocities.copy()
    current_time = 0.0
    times = [current_time]
    states = [pack_planar_state(current_positions, current_velocities)]
    point_state = states[-1]
    if initial_state_interval is not None and initial_state_interval_union is not None:
        raise ValueError("provide initial_state_interval or initial_state_interval_union, not both")
    if initial_state_interval_union is not None:
        current_state_interval_union = _normalize_planar_state_interval_union(
            initial_state_interval_union,
            name="initial_state_interval_union",
        )
        if not _state_interval_union_contains_point(current_state_interval_union, point_state):
            raise ValueError("initial_state_interval_union must contain the initial point state")
        current_state_interval = _hull_state_intervals(current_state_interval_union)
    elif initial_state_interval is not None:
        current_state_interval = _normalize_planar_state_interval(
            initial_state_interval,
            name="initial_state_interval",
        )
        if not _state_interval_contains_point(current_state_interval, point_state):
            raise ValueError("initial_state_interval must contain the initial point state")
        current_state_interval_union = (current_state_interval,)
    else:
        current_state_interval = _pack_interval_planar_state(current_positions, current_velocities)
        current_state_interval_union = (current_state_interval,)
    steps: list[HybridStep] = []

    while t_final - current_time > 10.0 * np.finfo(float).eps:
        if len(steps) >= max_steps:
            raise RuntimeError("hybrid continuation exceeded max_steps")
        remaining = t_final - current_time
        start_positions = current_positions.copy()
        start_velocities = current_velocities.copy()
        current_point_state = pack_planar_state(start_positions, start_velocities)
        start_state_interval = current_state_interval
        start_state_interval_union = current_state_interval_union
        interval_min_distance = min(
            interval_min_pair_distance_lower_bound(state_interval)
            for state_interval in start_state_interval_union
        )
        interval_max_speed = max(
            interval_max_speed_upper_bound(state_interval)
            for state_interval in start_state_interval_union
        )
        interval_max_acceleration = max(
            interval_max_acceleration_upper_bound(state_interval, masses)
            for state_interval in start_state_interval_union
        )
        binary_interval_lift_certified = False
        binary_interval_lift_reason = None
        binary_lc_branch = None
        binary_lc_branch_certified = False
        binary_lc_atlas_chart_count = 0
        binary_lc_atlas_certified = False
        binary_lc_atlas_propagated = False
        residual_certificate = None
        center_of_mass_certificate = None
        linear_momentum_certificate = None
        angular_momentum_certificate = None
        energy_certificate = None
        event_union_certificate = None
        event_time_interval = None
        point_chart, point_pair, min_distance = choose_chart(
            current_positions,
            binary_distance_threshold=binary_distance_threshold,
        )
        interval_chart_decision = choose_transition_interval_chart_union(
            start_state_interval_union,
            steps[-1] if steps else None,
            binary_distance_threshold=binary_distance_threshold,
        )
        if require_interval_chart_certification and not interval_chart_decision.certified:
            raise RuntimeError(
                "interval chart decision is not certified: "
                f"{interval_chart_decision.reason}"
            )
        if interval_chart_decision.certified:
            chart = interval_chart_decision.chart
            pair = interval_chart_decision.pair
        else:
            chart = point_chart
            pair = point_pair
        if chart == "ordinary":
            point_member_state_interval = next(
                (
                    state_interval
                    for state_interval in start_state_interval_union
                    if _state_interval_contains_point(state_interval, current_point_state)
                ),
                start_state_interval,
            )
            trial_physical_step = choose_interval_union_ordinary_step_size(
                start_state_interval_union,
                masses,
                remaining,
                max_step=max_time_step,
                safety=safety,
            )
            if tail_certificate_mode == "cauchy":
                cauchy_trial_certificate = ordinary_interval_union_cauchy_majorant_tail_certificate(
                    start_state_interval_union,
                    masses,
                    retained_order=ordinary_order,
                    step_size=0.0,
                )
                trial_physical_step = min(trial_physical_step, 0.5 * cauchy_trial_certificate.time_radius)
            union_crossing = first_ordinary_binary_entry_event_union(
                start_positions,
                start_velocities,
                masses,
                trial_physical_step=trial_physical_step,
                binary_enter_distance=binary_distance_threshold,
                order=ordinary_order,
                start_state_interval_union=start_state_interval_union,
            )
            if union_crossing is None:
                (
                    current_positions,
                    current_velocities,
                    physical_step,
                    indicator,
                    event_pair,
                    event_certificate,
                ) = event_limited_ordinary_taylor_step(
                    current_positions,
                    current_velocities,
                    masses,
                    trial_physical_step,
                    order=ordinary_order,
                    binary_enter_distance=binary_distance_threshold,
                    start_state_interval=point_member_state_interval,
                )
                if event_pair is not None and event_certificate is not None:
                    event_union_certificate = certify_ordinary_binary_entry_event_union(
                        start_positions,
                        start_velocities,
                        masses,
                        trial_physical_step=trial_physical_step,
                        event_root=physical_step,
                        event_pair=event_pair,
                        binary_enter_distance=binary_distance_threshold,
                        order=ordinary_order,
                        start_state_interval_union=start_state_interval_union,
                    )
            else:
                physical_step, event_pair, event_union_certificate = union_crossing
                series = construct_taylor_solution(
                    start_positions,
                    start_velocities,
                    masses,
                    order=ordinary_order,
                )
                current_positions = series.positions_at(physical_step)
                current_velocities = series.velocities_at(physical_step)
                indicator = truncation_indicator(series, physical_step)
                event_certificate = _ordinary_binary_entry_event_certificate_for_interval(
                    series.position,
                    masses,
                    event_root=physical_step,
                    event_pair=event_pair,
                    trial_physical_step=trial_physical_step,
                    binary_enter_distance=binary_distance_threshold,
                    order=ordinary_order,
                    state_interval=start_state_interval,
                    coefficient_source="ordinary_interval_taylor",
                )
            parameter_step = physical_step
            step_chart = "ordinary"
            step_event = "enter_binary" if event_pair is not None else None
            step_pair = event_pair
            event_time_intervals = None
            if event_union_certificate is not None:
                event_time_interval = event_union_certificate.event_time_interval
                if event_union_certificate.has_member_root_enclosures:
                    event_time_intervals = tuple(
                        enclosure.interval
                        for enclosure in event_union_certificate.member_root_enclosures
                        if enclosure is not None
                    )
            if event_time_intervals is not None:
                end_state_interval_union = _ordinary_end_state_interval_union_over_time_intervals(
                    start_positions,
                    start_velocities,
                    masses,
                    event_time_intervals,
                    order=ordinary_order,
                    start_state_interval_union=start_state_interval_union,
                )
            else:
                end_state_interval_union = _ordinary_end_state_interval_union(
                    start_positions,
                    start_velocities,
                    masses,
                    physical_step,
                    order=ordinary_order,
                    start_state_interval_union=start_state_interval_union,
                )
            end_state_interval = _hull_state_intervals(end_state_interval_union)
            if tail_certificate_mode == "cauchy":
                cauchy_step_size = (
                    _local_interval_step_size(event_time_interval)
                    if event_time_interval is not None
                    else physical_step
                )
                truncation_certificate = ordinary_interval_union_cauchy_majorant_tail_certificate(
                    start_state_interval_union,
                    masses,
                    retained_order=ordinary_order,
                    step_size=cauchy_step_size,
                )
            elif tail_guard_order:
                truncation_certificate = ordinary_taylor_tail_certificate(
                    start_positions,
                    start_velocities,
                    masses,
                    retained_order=ordinary_order,
                    guard_order=tail_guard_order,
                    step_size=physical_step,
                )
            else:
                truncation_certificate = None
            try:
                if len(start_state_interval_union) == 1:
                    ordinary_interval_series = _ordinary_interval_taylor_solution_from_state_interval(
                        start_state_interval,
                        masses,
                        order=ordinary_order,
                    )
                    residual_certificate = certify_ordinary_interval_taylor_equations(
                        ordinary_interval_series,
                        coefficient_count=ordinary_order,
                    )
                    physical_time_series = _ordinary_physical_time_series(ordinary_order)
                    center_of_mass_certificate = certify_interval_center_of_mass_motion(
                        ordinary_interval_series.position,
                        ordinary_interval_series.velocity,
                        physical_time_series,
                        masses,
                        coefficient_count=ordinary_order,
                    )
                    linear_momentum_certificate = certify_interval_linear_momentum_conservation(
                        ordinary_interval_series.velocity,
                        masses,
                        coefficient_count=ordinary_order,
                    )
                    angular_momentum_certificate = certify_interval_centered_angular_momentum_conservation(
                        ordinary_interval_series.position,
                        ordinary_interval_series.velocity,
                        masses,
                        coefficient_count=ordinary_order,
                    )
                    energy_certificate = certify_interval_total_energy_conservation(
                        ordinary_interval_series.position,
                        ordinary_interval_series.velocity,
                        masses,
                        coefficient_count=ordinary_order,
                    )
                else:
                    (
                        residual_certificate,
                        center_of_mass_certificate,
                        linear_momentum_certificate,
                        angular_momentum_certificate,
                        energy_certificate,
                    ) = _ordinary_interval_union_certificates(
                        start_state_interval_union,
                        masses,
                        order=ordinary_order,
                    )
            except ValueError:
                residual_certificate = None
                center_of_mass_certificate = None
                linear_momentum_certificate = None
                angular_momentum_certificate = None
                energy_certificate = None
        else:
            assert pair is not None
            interval_binary_atlas = ()
            initial_binary_state = planar_to_regularized_binary_collision_chart(
                start_positions,
                start_velocities,
                masses,
                pair=pair,
            )
            interval_binary_state = None
            try:
                interval_binary_state = planar_interval_to_regularized_binary_collision_chart(
                    start_state_interval,
                    masses,
                    pair=pair,
                )
                binary_interval_lift_certified = interval_binary_state.contains_point(initial_binary_state)
                binary_interval_lift_reason = (
                    "planar interval state lifted into regularized binary chart"
                    if binary_interval_lift_certified
                    else "interval binary lift did not contain point lift"
                )
                if interval_binary_state.branch_certificate is not None:
                    binary_lc_branch = interval_binary_state.branch_certificate.branch
                    binary_lc_branch_certified = interval_binary_state.branch_certificate.certified
            except ValueError as error:
                interval_binary_state = None
                binary_interval_lift_reason = str(error)
            try:
                interval_binary_atlas = _regularized_binary_atlas_for_state_interval_union(
                    start_state_interval_union,
                    masses,
                    pair=pair,
                )
                binary_lc_atlas_chart_count = len(interval_binary_atlas)
                binary_lc_atlas_certified = all(
                    chart.branch_certificate is not None and chart.branch_certificate.certified
                    for chart in interval_binary_atlas
                )
                if not binary_interval_lift_certified:
                    binary_interval_lift_certified = any(
                        chart.contains_point(initial_binary_state)
                        for chart in interval_binary_atlas
                    )
                if binary_lc_atlas_certified and not binary_lc_branch_certified:
                    binary_lc_branch_certified = True
                if binary_interval_lift_certified and (
                    binary_interval_lift_reason is None or interval_binary_state is None
                ):
                    binary_interval_lift_reason = "interval-state union lifted into regularized binary atlas"
            except ValueError as error:
                if binary_interval_lift_reason is None:
                    binary_interval_lift_reason = str(error)
            local_max_binary_s_step = max_binary_s_step
            if tail_certificate_mode == "cauchy":
                direct_interval_cauchy_ok = (
                    interval_binary_state is not None
                    and interval_binary_state.contains_point(initial_binary_state)
                    and interval_binary_state.branch_certificate is not None
                    and interval_binary_state.branch_certificate.certified
                )
                atlas_cauchy_ok = (
                    binary_lc_atlas_certified
                    and bool(interval_binary_atlas)
                    and any(chart.contains_point(initial_binary_state) for chart in interval_binary_atlas)
                )
                if direct_interval_cauchy_ok:
                    cauchy_trial_certificate = regularized_binary_interval_cauchy_majorant_tail_certificate(
                        interval_binary_state,
                        retained_order=binary_order,
                        step_size=0.0,
                    )
                elif atlas_cauchy_ok:
                    cauchy_trial_certificate = regularized_binary_interval_atlas_cauchy_majorant_tail_certificate(
                        interval_binary_atlas,
                        retained_order=binary_order,
                        step_size=0.0,
                    )
                else:
                    cauchy_trial_certificate = regularized_binary_cauchy_majorant_tail_certificate(
                        initial_binary_state,
                        retained_order=binary_order,
                        step_size=0.0,
                    )
                local_max_binary_s_step = min(local_max_binary_s_step, 0.5 * cauchy_trial_certificate.s_radius)
            (
                current_positions,
                current_velocities,
                parameter_step,
                physical_step,
                event_certificate,
            ) = regularized_binary_taylor_step(
                current_positions,
                current_velocities,
                masses,
                pair,
                remaining,
                order=binary_order,
                max_s_step=local_max_binary_s_step,
                binary_exit_distance=binary_exit_distance,
            )
            indicator = 0.0
            step_chart = "binary"
            exit_distance = _pair_distance(current_positions, pair)
            step_event = "exit_binary" if exit_distance >= binary_exit_distance * (1.0 - 1e-12) else None
            step_pair = pair
            binary_interval_endpoint_fallback = False
            if binary_lc_atlas_certified and interval_binary_atlas:
                try:
                    end_state_interval_union = _regularized_binary_atlas_end_state_intervals(
                        initial_binary_state,
                        interval_binary_atlas,
                        parameter_step,
                        order=binary_order,
                    )
                    end_state_interval = _hull_state_intervals(end_state_interval_union)
                    binary_lc_atlas_propagated = True
                except ValueError as error:
                    binary_lc_atlas_propagated = False
                    binary_interval_endpoint_fallback = True
                    binary_interval_lift_reason = f"interval binary atlas propagation failed: {error}"
            else:
                binary_interval_endpoint_fallback = True
            if binary_interval_endpoint_fallback:
                try:
                    end_state_interval = _regularized_binary_end_state_interval(
                        initial_binary_state,
                        parameter_step,
                        order=binary_order,
                        interval_initial_state=interval_binary_state,
                    )
                except ValueError as error:
                    binary_interval_lift_reason = f"interval binary endpoint propagation failed: {error}"
                    end_state_interval = _pack_interval_planar_state(current_positions, current_velocities)
                end_state_interval_union = (end_state_interval,)
            if tail_certificate_mode == "cauchy":
                direct_interval_cauchy_ok = (
                    interval_binary_state is not None
                    and interval_binary_state.contains_point(initial_binary_state)
                    and interval_binary_state.branch_certificate is not None
                    and interval_binary_state.branch_certificate.certified
                )
                atlas_cauchy_ok = (
                    binary_lc_atlas_certified
                    and bool(interval_binary_atlas)
                    and any(chart.contains_point(initial_binary_state) for chart in interval_binary_atlas)
                )
                cauchy_step_size = parameter_step
                if event_certificate is not None and event_certificate.root_enclosure is not None:
                    cauchy_step_size = _local_interval_step_size(event_certificate.root_enclosure.interval)
                if direct_interval_cauchy_ok:
                    truncation_certificate = regularized_binary_interval_cauchy_majorant_tail_certificate(
                        interval_binary_state,
                        retained_order=binary_order,
                        step_size=cauchy_step_size,
                    )
                elif atlas_cauchy_ok:
                    truncation_certificate = regularized_binary_interval_atlas_cauchy_majorant_tail_certificate(
                        interval_binary_atlas,
                        retained_order=binary_order,
                        step_size=cauchy_step_size,
                    )
                else:
                    truncation_certificate = regularized_binary_cauchy_majorant_tail_certificate(
                        initial_binary_state,
                        retained_order=binary_order,
                        step_size=cauchy_step_size,
                    )
            elif tail_guard_order:
                truncation_certificate = regularized_binary_tail_certificate(
                    initial_binary_state,
                    retained_order=binary_order,
                    guard_order=tail_guard_order,
                    step_size=parameter_step,
                )
            else:
                truncation_certificate = None
            try:
                direct_binary_certified = bool(
                    interval_binary_state is not None
                    and binary_interval_lift_certified
                    and interval_binary_state.branch_certificate is not None
                    and interval_binary_state.branch_certificate.certified
                )
                atlas_binary_certified = bool(
                    not direct_binary_certified
                    and binary_lc_atlas_certified
                    and bool(interval_binary_atlas)
                )
                endpoint_parameter_interval = (
                    event_certificate.root_enclosure.interval
                    if event_certificate is not None
                    and event_certificate.root_enclosure is not None
                    else (parameter_step, parameter_step)
                )
                if direct_binary_certified:
                    binary_interval_series = construct_interval_regularized_binary_taylor_solution_from_intervals(
                        interval_binary_state,
                        order=binary_order,
                    )
                    projection_domain_certificate = certify_regularized_binary_projection_domain(
                        binary_interval_series,
                        parameter_interval=(0.0, endpoint_parameter_interval[1]),
                        endpoint_parameter_interval=endpoint_parameter_interval,
                        branch_or_atlas_certified=True,
                        starts_at_binary_collision=bool(initial_binary_state.rho == 0.0),
                        initial_collision_selector_certified=bool(initial_binary_state.rho == 0.0),
                    )
                    residual_certificate = certify_regularized_binary_interval_taylor_equations(
                        binary_interval_series,
                        coefficient_count=binary_order,
                        projection_domain_certificate=projection_domain_certificate,
                    )
                    center_of_mass_certificate = certify_regularized_binary_center_of_mass_motion(
                        binary_interval_series,
                        coefficient_count=binary_order,
                    )
                    linear_momentum_certificate = certify_regularized_binary_linear_momentum_conservation(
                        binary_interval_series,
                        coefficient_count=binary_order,
                    )
                    angular_momentum_certificate = certify_regularized_binary_centered_angular_momentum_conservation(
                        binary_interval_series,
                        coefficient_count=binary_order,
                    )
                    energy_certificate = certify_regularized_binary_total_energy_conservation(
                        binary_interval_series,
                        coefficient_count=binary_order,
                    )
                elif atlas_binary_certified:
                    residual_certificates = []
                    center_certificates = []
                    linear_certificates = []
                    angular_certificates = []
                    energy_certificates = []
                    for atlas_state in interval_binary_atlas:
                        binary_interval_series = (
                            construct_interval_regularized_binary_taylor_solution_from_intervals(
                                atlas_state,
                                order=binary_order,
                            )
                        )
                        branch_certificate = atlas_state.branch_certificate
                        projection_domain_certificate = certify_regularized_binary_projection_domain(
                            binary_interval_series,
                            parameter_interval=(0.0, endpoint_parameter_interval[1]),
                            endpoint_parameter_interval=endpoint_parameter_interval,
                            branch_or_atlas_certified=bool(
                                branch_certificate is not None
                                and branch_certificate.certified
                            ),
                            starts_at_binary_collision=bool(initial_binary_state.rho == 0.0),
                            initial_collision_selector_certified=bool(
                                initial_binary_state.rho == 0.0
                                and branch_certificate is not None
                                and branch_certificate.certified
                            ),
                        )
                        residual_certificates.append(
                            certify_regularized_binary_interval_taylor_equations(
                                binary_interval_series,
                                coefficient_count=binary_order,
                                projection_domain_certificate=projection_domain_certificate,
                            )
                        )
                        center_certificates.append(
                            certify_regularized_binary_center_of_mass_motion(
                                binary_interval_series,
                                coefficient_count=binary_order,
                            )
                        )
                        linear_certificates.append(
                            certify_regularized_binary_linear_momentum_conservation(
                                binary_interval_series,
                                coefficient_count=binary_order,
                            )
                        )
                        angular_certificates.append(
                            certify_regularized_binary_centered_angular_momentum_conservation(
                                binary_interval_series,
                                coefficient_count=binary_order,
                            )
                        )
                        energy_certificates.append(
                            certify_regularized_binary_total_energy_conservation(
                                binary_interval_series,
                                coefficient_count=binary_order,
                            )
                        )
                    residual_certificate = IntervalUnionCertificate(
                        tuple(residual_certificates),
                        "regularized_binary_interval_taylor_atlas",
                    )
                    center_of_mass_certificate = IntervalUnionCertificate(
                        tuple(center_certificates),
                        "regularized_binary_interval_taylor_atlas",
                    )
                    linear_momentum_certificate = IntervalUnionCertificate(
                        tuple(linear_certificates),
                        "regularized_binary_interval_taylor_atlas",
                    )
                    angular_momentum_certificate = IntervalUnionCertificate(
                        tuple(angular_certificates),
                        "regularized_binary_interval_taylor_atlas",
                    )
                    energy_certificate = IntervalUnionCertificate(
                        tuple(energy_certificates),
                        "regularized_binary_interval_taylor_atlas",
                    )
                else:
                    binary_interval_series = construct_interval_regularized_binary_taylor_solution(
                        initial_binary_state,
                        order=binary_order,
                    )
                    projection_domain_certificate = certify_regularized_binary_projection_domain(
                        binary_interval_series,
                        parameter_interval=(0.0, endpoint_parameter_interval[1]),
                        endpoint_parameter_interval=endpoint_parameter_interval,
                        branch_or_atlas_certified=bool(
                            binary_interval_lift_certified
                            and (
                                binary_lc_branch_certified
                                or binary_lc_atlas_certified
                            )
                        ),
                        starts_at_binary_collision=bool(initial_binary_state.rho == 0.0),
                        initial_collision_selector_certified=bool(
                            initial_binary_state.rho == 0.0
                            and binary_interval_lift_certified
                            and (
                                binary_lc_branch_certified
                                or binary_lc_atlas_certified
                            )
                        ),
                    )
                    residual_certificate = certify_regularized_binary_interval_taylor_equations(
                        binary_interval_series,
                        coefficient_count=binary_order,
                        projection_domain_certificate=projection_domain_certificate,
                    )
                    center_of_mass_certificate = certify_regularized_binary_center_of_mass_motion(
                        binary_interval_series,
                        coefficient_count=binary_order,
                    )
                    linear_momentum_certificate = certify_regularized_binary_linear_momentum_conservation(
                        binary_interval_series,
                        coefficient_count=binary_order,
                    )
                    angular_momentum_certificate = certify_regularized_binary_centered_angular_momentum_conservation(
                        binary_interval_series,
                        coefficient_count=binary_order,
                    )
                    energy_certificate = certify_regularized_binary_total_energy_conservation(
                        binary_interval_series,
                        coefficient_count=binary_order,
                    )
            except ValueError:
                residual_certificate = None
                center_of_mass_certificate = None
                linear_momentum_certificate = None
                angular_momentum_certificate = None
                energy_certificate = None

        step = HybridStep(
            chart=step_chart,
            pair=step_pair,
            event=step_event,
            event_certificate=event_certificate,
            event_union_certificate=event_union_certificate,
            event_time_interval=event_time_interval,
            truncation_certificate=truncation_certificate,
            start_state_interval=start_state_interval,
            end_state_interval=end_state_interval,
            start_state_interval_union=start_state_interval_union,
            end_state_interval_union=end_state_interval_union,
            start_time=current_time,
            physical_step=float(physical_step),
            parameter_step=float(parameter_step),
            min_pair_distance=float(min_distance),
            interval_min_pair_distance=float(interval_min_distance),
            interval_max_speed=float(interval_max_speed),
            interval_max_acceleration=float(interval_max_acceleration),
            interval_chart=interval_chart_decision.chart,
            interval_chart_pair=interval_chart_decision.pair,
            interval_chart_certified=interval_chart_decision.certified,
            binary_interval_lift_certified=binary_interval_lift_certified,
            binary_interval_lift_reason=binary_interval_lift_reason,
            binary_lc_branch=binary_lc_branch,
            binary_lc_branch_certified=binary_lc_branch_certified,
            binary_lc_atlas_chart_count=binary_lc_atlas_chart_count,
            binary_lc_atlas_certified=binary_lc_atlas_certified,
            binary_lc_atlas_propagated=binary_lc_atlas_propagated,
            residual_certificate=residual_certificate,
            center_of_mass_certificate=center_of_mass_certificate,
            linear_momentum_certificate=linear_momentum_certificate,
            angular_momentum_certificate=angular_momentum_certificate,
            energy_certificate=energy_certificate,
        )
        current_time = step.end_time
        steps.append(step)
        current_state_interval = end_state_interval
        current_state_interval_union = end_state_interval_union
        times.append(current_time)
        states.append(pack_planar_state(current_positions, current_velocities))

    return HybridContinuedSolution(
        masses=masses,
        times=np.array(times, dtype=float),
        states=np.vstack(states),
        steps=tuple(steps),
    )
