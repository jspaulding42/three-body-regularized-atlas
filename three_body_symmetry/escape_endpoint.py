"""Executable endpoint recurrences for log-subtracted escape charts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .continuation import (
    UniformCollisionFreeTaylorRecurrenceCertificate,
    certify_uniformly_collision_free_taylor_recurrence,
)
from .dynamics import accelerations
from .validated_atlas import ValidatedAtlasSolution
from .zero_angular_entry import (
    FiniteJetDerivedIdentitySelectorEntryCertificate,
    FiniteJetIdentitySelectorEntryCertificate,
)


Array = np.ndarray


@dataclass(frozen=True)
class HomotheticEscapeEndpointData:
    """Radial constants for a positive-energy homothetic escape branch."""

    gravitational_parameter: float
    initial_radius: float
    initial_speed: float
    energy: float
    asymptotic_speed: float
    beta: float
    time_shift: float
    forced_log_coefficient: float

    @property
    def certified(self) -> bool:
        return bool(
            self.gravitational_parameter > 0.0
            and self.initial_radius > 0.0
            and self.energy > 0.0
            and self.asymptotic_speed > 0.0
            and np.isfinite(self.time_shift)
        )


@dataclass(frozen=True)
class HomotheticEscapeDyadicRecurrence:
    """All-future dyadic shell tail bound for the homothetic escape endpoint.

    The supplied `cauchy_majorant` is the Cauchy bound for the log-subtracted
    scalar endpoint function, or for the requested derivative order. The
    constructor then derives the whole dyadic recurrence from the start time,
    polydisc radii, and retained total degree.
    """

    endpoint: HomotheticEscapeEndpointData
    start_time: float
    tau_radius: float
    rho_radius: float
    cauchy_majorant: float
    retained_degree: int
    derivative_order: int
    theta0: float
    shell_ratio: float
    tail_exponent: int
    majorant_certificate: object | None = None

    @property
    def certified(self) -> bool:
        return bool(
            self.endpoint.certified
            and self.start_time > 2.0
            and self.tau_radius > 0.0
            and self.rho_radius > 0.0
            and self.cauchy_majorant > 0.0
            and self.tail_exponent > 0
            and self.theta0 < 1.0
            and self.shell_ratio < 1.0
            and (
                self.majorant_certificate is None
                or _homothetic_escape_majorant_certificate_certified(
                    self.majorant_certificate,
                    self.endpoint,
                )
            )
        )

    @property
    def first_shell_tail_bound(self) -> float:
        return float(
            self.cauchy_majorant
            * self.theta0**self.tail_exponent
            / (1.0 - self.theta0)
        )

    @property
    def shell_tail_ratio(self) -> float:
        return float(self.shell_ratio**self.tail_exponent)

    @property
    def all_future_tail_bound(self) -> float:
        return float(self.first_shell_tail_bound / (1.0 - self.shell_tail_ratio))

    def theta(self, shell_index: int) -> float:
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        tau_n = 2.0 ** (-shell_index) / self.start_time
        log_n = np.log(self.start_time) + shell_index * np.log(2.0)
        return float(max(tau_n / self.tau_radius, tau_n * log_n / self.rho_radius))

    def shell_tail_bound(self, shell_index: int) -> float:
        theta_n = self.theta(shell_index)
        return float(
            self.cauchy_majorant
            * theta_n**self.tail_exponent
            / (1.0 - theta_n)
        )

    def geometric_shell_tail_bound(self, shell_index: int) -> float:
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        return float(self.first_shell_tail_bound * self.shell_tail_ratio**shell_index)

    def future_tail_from_shell(self, shell_index: int) -> float:
        return float(
            self.geometric_shell_tail_bound(shell_index)
            / (1.0 - self.shell_tail_ratio)
        )


@dataclass(frozen=True)
class TimeReversedHomotheticEscapeRecurrenceCertificate:
    """Past escape endpoint obtained by exact time reversal.

    The positive-energy homothetic branch has a finite collision time
    ``T = endpoint.time_shift`` in the physical time normalization where the
    supplied expanding state has radius one at ``t=0``.  If the future endpoint
    recurrence controls the expanding branch for ``t >= S``, then the identity
    selector continuation ``q(T+tau^3)=tau^2 u(tau^2)Q`` gives the incoming
    branch by reflection about ``T``.  Thus the same tail constants control
    ``t <= 2T-S``.
    """

    future_recurrence: HomotheticEscapeDyadicRecurrence
    compact_time_rate: float
    source: str = "time_reversed_homothetic_escape_endpoint_recurrence"

    @property
    def collision_time(self) -> float:
        return float(self.future_recurrence.endpoint.time_shift)

    @property
    def future_handoff_time(self) -> float:
        return float(self.future_recurrence.start_time)

    @property
    def past_handoff_time(self) -> float:
        return float(2.0 * self.collision_time - self.future_handoff_time)

    @property
    def past_handoff_compact_parameter(self) -> float:
        return float(np.tanh(self.compact_time_rate * self.past_handoff_time))

    @property
    def future_handoff_compact_parameter(self) -> float:
        return float(np.tanh(self.compact_time_rate * self.future_handoff_time))

    @property
    def all_past_tail_bound(self) -> float:
        return float(self.future_recurrence.all_future_tail_bound)

    @property
    def shell_tail_ratio(self) -> float:
        return float(self.future_recurrence.shell_tail_ratio)

    @property
    def recurrence_closes(self) -> bool:
        return self.certified

    @property
    def certified(self) -> bool:
        return bool(
            self.future_recurrence.certified
            and np.isfinite(self.compact_time_rate)
            and self.compact_time_rate > 0.0
            and np.isfinite(self.collision_time)
            and self.past_handoff_time < self.collision_time
            and self.collision_time < self.future_handoff_time
            and -1.0 < self.past_handoff_compact_parameter < 1.0
            and -1.0 < self.future_handoff_compact_parameter < 1.0
            and np.isfinite(self.all_past_tail_bound)
            and self.all_past_tail_bound >= 0.0
        )

    def shell_tail_bound(self, shell_index: int) -> float:
        return self.future_recurrence.shell_tail_bound(shell_index)

    def geometric_shell_tail_bound(self, shell_index: int) -> float:
        return self.future_recurrence.geometric_shell_tail_bound(shell_index)

    def past_tail_from_shell(self, shell_index: int) -> float:
        return self.future_recurrence.future_tail_from_shell(shell_index)


@dataclass(frozen=True)
class PositiveEnergyHomotheticAllRealGluingCertificate:
    """Glue homothetic past/future endpoints to the local collision chart."""

    branch_certificate: object
    projection_invariant_certificate: object
    future_recurrence: HomotheticEscapeDyadicRecurrence
    past_recurrence: TimeReversedHomotheticEscapeRecurrenceCertificate
    total_collision_atlas: object
    selector_entry: object
    middle_recurrence: object
    collision_tau: float
    collision_time: float
    future_collision_handoff_time: float
    past_collision_handoff_time: float
    future_endpoint_handoff_time: float
    past_endpoint_handoff_time: float
    scale_consistency_residual: float
    energy_consistency_residual: float
    collision_time_residual: float
    scalar_lower_bound: float
    scalar_velocity_lower_bound: float
    min_scaled_shape_pair_distance: float
    middle_pair_distance_lower_bound: float
    middle_position_upper_bound: float
    middle_speed_upper_bound: float
    past_middle_interval_length: float
    future_middle_interval_length: float
    middle_step_size: float
    future_middle_chart_count_bound: int
    past_middle_chart_count_bound: int
    middle_transition_count_bound: int
    per_middle_chart_tail_bound: float
    past_middle_tail_budget: float
    future_middle_tail_budget: float
    total_middle_tail_budget: float
    endpoint_tail_budget: float
    total_collision_tail_budget: float
    all_real_tail_budget_bound: float
    max_newton_residual_bound: float
    invariant_residual_bound: float
    tolerance: float
    source: str = "positive_energy_homothetic_all_real_gluing"

    @property
    def endpoint_recurrences_certified(self) -> bool:
        return bool(
            isinstance(self.future_recurrence, HomotheticEscapeDyadicRecurrence)
            and isinstance(
                self.past_recurrence,
                TimeReversedHomotheticEscapeRecurrenceCertificate,
            )
            and self.future_recurrence.certified is True
            and self.past_recurrence.certified is True
            and self.past_recurrence.future_recurrence is self.future_recurrence
        )

    @property
    def source_inputs_certified(self) -> bool:
        return bool(
            isinstance(self.branch_certificate, HomotheticEscapeBranchCertificate)
            and self.branch_certificate.certified is True
            and isinstance(
                self.projection_invariant_certificate,
                HomotheticEscapeProjectionInvariantCertificate,
            )
            and self.projection_invariant_certificate.certified is True
        )

    @property
    def collision_selector_certified(self) -> bool:
        return bool(
            isinstance(self.total_collision_atlas, ValidatedAtlasSolution)
            and self.total_collision_atlas.proof_certified is True
            and _finite_jet_selector_entry_certified(self.selector_entry)
        )

    @property
    def middle_recurrence_certified(self) -> bool:
        return bool(
            isinstance(
                self.middle_recurrence,
                UniformCollisionFreeTaylorRecurrenceCertificate,
            )
            and self.middle_recurrence.certified is True
            and self.middle_recurrence.proof_certified is True
            and self.middle_recurrence.tail_budget_certified is True
            and self.middle_recurrence.newton_residual_certified is True
        )

    @property
    def handoff_order_certified(self) -> bool:
        return bool(
            self.past_endpoint_handoff_time < self.past_collision_handoff_time
            and self.past_collision_handoff_time < self.collision_time
            and self.collision_time < self.future_collision_handoff_time
            and self.future_collision_handoff_time < self.future_endpoint_handoff_time
        )

    @property
    def domain_certified(self) -> bool:
        return bool(
            self.collision_tau > 0.0
            and self.scalar_lower_bound > 0.0
            and self.scalar_velocity_lower_bound > 0.0
            and self.min_scaled_shape_pair_distance > 0.0
            and self.middle_pair_distance_lower_bound > 0.0
            and self.middle_position_upper_bound >= 0.0
            and self.middle_speed_upper_bound >= 0.0
            and self.past_middle_interval_length > 0.0
            and self.future_middle_interval_length > 0.0
            and self.middle_step_size > 0.0
            and self.future_middle_chart_count_bound >= 1
            and self.past_middle_chart_count_bound >= 1
        )

    @property
    def tail_budget_certified(self) -> bool:
        return bool(
            np.isfinite(self.per_middle_chart_tail_bound)
            and np.isfinite(self.past_middle_tail_budget)
            and np.isfinite(self.future_middle_tail_budget)
            and np.isfinite(self.total_middle_tail_budget)
            and np.isfinite(self.endpoint_tail_budget)
            and np.isfinite(self.total_collision_tail_budget)
            and np.isfinite(self.all_real_tail_budget_bound)
            and self.per_middle_chart_tail_bound >= 0.0
            and self.past_middle_tail_budget >= 0.0
            and self.future_middle_tail_budget >= 0.0
            and self.total_middle_tail_budget >= 0.0
            and self.endpoint_tail_budget >= 0.0
            and self.total_collision_tail_budget >= 0.0
            and self.all_real_tail_budget_bound >= (
                self.endpoint_tail_budget
                + self.total_collision_tail_budget
                + self.total_middle_tail_budget
            )
        )

    @property
    def transition_budget_certified(self) -> bool:
        return bool(
            self.middle_transition_count_bound
            >= self.past_middle_chart_count_bound
            + self.future_middle_chart_count_bound
            + 2
        )

    @property
    def consistency_certified(self) -> bool:
        return bool(
            self.scale_consistency_residual <= self.tolerance
            and self.energy_consistency_residual <= self.tolerance
            and self.collision_time_residual <= self.tolerance
        )

    @property
    def residuals_certified(self) -> bool:
        return bool(
            self.max_newton_residual_bound <= self.tolerance
            and self.invariant_residual_bound <= self.tolerance
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.source_inputs_certified
            and self.endpoint_recurrences_certified
            and self.collision_selector_certified
            and self.middle_recurrence_certified
            and self.handoff_order_certified
            and self.domain_certified
            and self.tail_budget_certified
            and self.transition_budget_certified
            and self.consistency_certified
            and self.residuals_certified
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def recurrence_closes(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        obligations: list[str] = []
        if not self.source_inputs_certified:
            obligations.append("homothetic_source_inputs")
        if not self.endpoint_recurrences_certified:
            obligations.append("homothetic_endpoint_recurrences")
        if not self.collision_selector_certified:
            obligations.append("homothetic_total_collision_selector_chart")
        if not self.middle_recurrence_certified:
            obligations.append("homothetic_finite_middle_recurrence")
        if not self.handoff_order_certified:
            obligations.append("homothetic_handoff_order")
        if not self.domain_certified:
            obligations.append("homothetic_middle_collision_free_domain")
        if not self.tail_budget_certified:
            obligations.append("homothetic_all_real_tail_budget")
        if not self.transition_budget_certified:
            obligations.append("homothetic_all_real_transition_budget")
        if not self.consistency_certified:
            obligations.append("homothetic_branch_consistency")
        if not self.residuals_certified:
            obligations.append("homothetic_projection_residuals")
        return tuple(obligations)


@dataclass(frozen=True)
class HomotheticEscapeImplicitCauchyMajorantCertificate:
    """Cauchy majorant derived from the scalar implicit endpoint equation."""

    endpoint: HomotheticEscapeEndpointData
    tau_radius: float
    rho_radius: float
    x_radius: float
    singularity_margin: float
    log_argument_lower_bound: float
    log_argument_upper_bound: float
    log_factor_bound: float
    sqrt_remainder_bound: float
    implicit_remainder_bound: float
    rouche_linear_bound: float
    cauchy_majorant: float
    source: str = "homothetic_escape_implicit_function_rouche_majorant"

    @property
    def rouche_margin(self) -> float:
        return float(self.rouche_linear_bound - self.implicit_remainder_bound)

    @property
    def certified(self) -> bool:
        return bool(
            self.endpoint.certified
            and np.isfinite(self.tau_radius)
            and self.tau_radius > 0.0
            and np.isfinite(self.rho_radius)
            and self.rho_radius > 0.0
            and np.isfinite(self.x_radius)
            and self.x_radius > 0.0
            and self.singularity_margin > 0.0
            and self.log_argument_lower_bound > 0.0
            and self.log_argument_upper_bound >= self.log_argument_lower_bound
            and self.sqrt_remainder_bound >= 0.0
            and self.implicit_remainder_bound >= 0.0
            and self.rouche_margin > 0.0
            and self.cauchy_majorant > 0.0
        )


def _homothetic_escape_majorant_certificate_certified(
    majorant_certificate: object,
    endpoint: HomotheticEscapeEndpointData,
) -> bool:
    return bool(
        isinstance(
            majorant_certificate,
            HomotheticEscapeImplicitCauchyMajorantCertificate,
        )
        and majorant_certificate.certified is True
        and majorant_certificate.endpoint == endpoint
    )


def _finite_jet_selector_entry_certified(selector_entry: object) -> bool:
    if isinstance(selector_entry, FiniteJetIdentitySelectorEntryCertificate):
        return selector_entry.certified is True
    if isinstance(selector_entry, FiniteJetDerivedIdentitySelectorEntryCertificate):
        return selector_entry.identity_selector_certified is True
    return False


@dataclass(frozen=True)
class HomotheticEscapeBranchCertificate:
    """Full initial-data certificate for a positive-energy homothetic escape."""

    masses: tuple[float, ...]
    dimension: int
    gravitational_parameter: float
    initial_radius: float
    initial_speed: float
    energy: float
    center_of_mass_position_norm: float
    center_of_mass_velocity_norm: float
    homothetic_velocity_residual: float
    central_configuration_residual: float
    tolerance: float
    endpoint: HomotheticEscapeEndpointData | None

    @property
    def positive_energy_certified(self) -> bool:
        return bool(self.energy > 0.0 and self.endpoint is not None and self.endpoint.certified)

    @property
    def certified(self) -> bool:
        scale = max(1.0, abs(self.gravitational_parameter), abs(self.initial_speed))
        return bool(
            self.masses
            and all(mass > 0.0 for mass in self.masses)
            and self.dimension >= 2
            and self.gravitational_parameter > 0.0
            and self.initial_radius == 1.0
            and self.initial_speed > 0.0
            and self.positive_energy_certified
            and self.homothetic_velocity_residual <= self.tolerance * scale
            and self.central_configuration_residual <= self.tolerance * scale
        )


@dataclass(frozen=True)
class HomotheticEscapeProjectionInvariantCertificate:
    """Projected Newton and invariant ledger for a homothetic escape branch."""

    branch_certificate: HomotheticEscapeBranchCertificate
    masses: tuple[float, ...]
    dimension: int
    inertia: float
    centered_linear_momentum_norm: float
    centered_angular_momentum_norm: float
    normalized_energy_residual: float
    projected_newton_residual_bound: float
    future_radial_monotonic: bool
    tolerance: float
    source: str = "homothetic_escape_projection_invariant_constructor"

    @property
    def scale(self) -> float:
        return float(
            max(
                1.0,
                abs(self.branch_certificate.gravitational_parameter),
                abs(self.branch_certificate.initial_speed),
                abs(self.branch_certificate.energy),
            )
        )

    @property
    def newton_residual_certified(self) -> bool:
        return bool(
            self.projected_newton_residual_bound <= self.tolerance * self.scale
        )

    @property
    def invariant_ledger_certified(self) -> bool:
        return bool(
            self.centered_linear_momentum_norm <= self.tolerance * self.scale
            and self.centered_angular_momentum_norm <= self.tolerance * self.scale
            and self.normalized_energy_residual <= self.tolerance * self.scale
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.branch_certificate.certified
            and self.masses
            and all(mass > 0.0 for mass in self.masses)
            and self.dimension >= 2
            and self.inertia > 0.0
            and self.future_radial_monotonic
            and self.newton_residual_certified
            and self.invariant_ledger_certified
        )


def certify_homothetic_escape_branch_from_initial_data(
    masses: Array,
    positions: Array,
    velocities: Array,
    *,
    tolerance: float = 1.0e-10,
) -> HomotheticEscapeBranchCertificate:
    """Derive the positive-energy homothetic escape branch from full data."""

    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    tolerance = float(tolerance)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    if (
        positions.shape != velocities.shape
        or positions.ndim != 2
        or positions.shape[0] != 3
        or positions.shape[1] < 2
    ):
        raise ValueError("positions and velocities must have shape (3, dimension>=2)")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    total_mass = float(np.sum(masses))
    center = np.sum(masses[:, None] * positions, axis=0) / total_mass
    center_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    centered_positions = positions - center
    centered_velocities = velocities - center_velocity
    inertia = float(np.sum(masses[:, None] * centered_positions**2))
    if inertia <= 0.0:
        raise ValueError("centered positions must be nonzero")
    speed = float(
        np.sum(masses[:, None] * centered_positions * centered_velocities)
        / inertia
    )
    velocity_residual = float(
        np.linalg.norm(centered_velocities - speed * centered_positions, ord=np.inf)
    )
    acceleration = accelerations(centered_positions, masses)
    gravitational_parameter = float(
        -np.sum(masses[:, None] * centered_positions * acceleration) / inertia
    )
    central_residual = float(
        np.linalg.norm(acceleration + gravitational_parameter * centered_positions, ord=np.inf)
    )
    energy = 0.5 * speed**2 - gravitational_parameter
    endpoint = None
    if gravitational_parameter > 0.0 and speed > 0.0 and energy > 0.0:
        endpoint = construct_homothetic_escape_endpoint_data(
            gravitational_parameter=gravitational_parameter,
            initial_radius=1.0,
            initial_speed=speed,
        )
    return HomotheticEscapeBranchCertificate(
        masses=tuple(float(mass) for mass in masses),
        dimension=int(positions.shape[1]),
        gravitational_parameter=gravitational_parameter,
        initial_radius=1.0,
        initial_speed=speed,
        energy=float(energy),
        center_of_mass_position_norm=float(
            np.linalg.norm(
                np.sum(masses[:, None] * centered_positions, axis=0),
                ord=np.inf,
            )
        ),
        center_of_mass_velocity_norm=float(
            np.linalg.norm(
                np.sum(masses[:, None] * centered_velocities, axis=0),
                ord=np.inf,
            )
        ),
        homothetic_velocity_residual=velocity_residual,
        central_configuration_residual=central_residual,
        tolerance=tolerance,
        endpoint=endpoint,
    )


def certify_homothetic_escape_projection_invariants(
    masses: Array,
    positions: Array,
    velocities: Array,
    branch_certificate: HomotheticEscapeBranchCertificate,
    *,
    tolerance: float = 1.0e-10,
) -> HomotheticEscapeProjectionInvariantCertificate:
    """Certify that the homothetic endpoint projects to Newtonian motion.

    For a centered homothetic branch ``q_i(t)=R(t)Q_i`` with
    ``A(Q)=-mu Q`` and ``R''=-mu/R^2``, homogeneity gives
    ``A(RQ)=R^-2 A(Q)``.  Hence the projected Newton residual on the expanding
    future branch is bounded by the central-configuration residual at ``R=1``.
    """

    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    tolerance = float(tolerance)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    if (
        positions.shape != velocities.shape
        or positions.ndim != 2
        or positions.shape[0] != 3
    ):
        raise ValueError("positions and velocities must have shape (3, dimension)")
    if positions.shape[1] < 2:
        raise ValueError("homothetic escape projection requires dimension at least two")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    total_mass = float(np.sum(masses))
    center = np.sum(masses[:, None] * positions, axis=0) / total_mass
    center_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    centered_positions = positions - center
    centered_velocities = velocities - center_velocity
    inertia = float(np.sum(masses[:, None] * centered_positions**2))
    potential = _positive_newtonian_potential(centered_positions, masses)
    actual_energy = (
        0.5 * float(np.sum(masses[:, None] * centered_velocities**2))
        - potential
    )
    normalized_energy = actual_energy / inertia if inertia > 0.0 else float("inf")
    certificate = HomotheticEscapeProjectionInvariantCertificate(
        branch_certificate=branch_certificate,
        masses=tuple(float(mass) for mass in masses),
        dimension=int(positions.shape[1]),
        inertia=inertia,
        centered_linear_momentum_norm=float(
            np.linalg.norm(
                np.sum(masses[:, None] * centered_velocities, axis=0),
                ord=np.inf,
            )
        ),
        centered_angular_momentum_norm=_weighted_angular_momentum_norm(
            masses,
            centered_positions,
            centered_velocities,
        ),
        normalized_energy_residual=float(
            abs(normalized_energy - float(branch_certificate.energy))
        ),
        projected_newton_residual_bound=float(
            branch_certificate.central_configuration_residual
        ),
        future_radial_monotonic=bool(
            branch_certificate.positive_energy_certified
            and branch_certificate.initial_speed > 0.0
        ),
        tolerance=tolerance,
    )
    return certificate


def construct_homothetic_escape_endpoint_data(
    *,
    gravitational_parameter: float,
    initial_radius: float,
    initial_speed: float,
) -> HomotheticEscapeEndpointData:
    """Derive the endpoint constants from finite radial initial data."""

    gravitational_parameter = float(gravitational_parameter)
    initial_radius = float(initial_radius)
    initial_speed = float(initial_speed)
    if gravitational_parameter <= 0.0:
        raise ValueError("gravitational_parameter must be positive")
    if initial_radius <= 0.0:
        raise ValueError("initial_radius must be positive")
    energy = 0.5 * initial_speed**2 - gravitational_parameter / initial_radius
    if energy <= 0.0:
        raise ValueError("homothetic escape endpoint requires positive energy")
    asymptotic_speed = float(np.sqrt(2.0 * energy))
    beta = 2.0 * gravitational_parameter / asymptotic_speed**2
    time_shift = -homothetic_escape_time_integral(
        initial_radius,
        gravitational_parameter,
        asymptotic_speed,
    )
    forced_log_coefficient = -gravitational_parameter / asymptotic_speed**2
    return HomotheticEscapeEndpointData(
        gravitational_parameter=gravitational_parameter,
        initial_radius=initial_radius,
        initial_speed=initial_speed,
        energy=float(energy),
        asymptotic_speed=asymptotic_speed,
        beta=float(beta),
        time_shift=float(time_shift),
        forced_log_coefficient=float(forced_log_coefficient),
    )


def construct_homothetic_escape_dyadic_recurrence(
    endpoint: HomotheticEscapeEndpointData,
    *,
    start_time: float,
    tau_radius: float,
    rho_radius: float,
    cauchy_majorant: float,
    retained_degree: int,
    derivative_order: int = 0,
    majorant_certificate: object | None = None,
) -> HomotheticEscapeDyadicRecurrence:
    """Construct the all-future dyadic Cauchy tail recurrence."""

    if majorant_certificate is not None and not _homothetic_escape_majorant_certificate_certified(
        majorant_certificate,
        endpoint,
    ):
        raise ValueError(
            "majorant_certificate must be a certified homothetic escape "
            "implicit Cauchy majorant derived for the supplied endpoint"
        )
    start_time = float(start_time)
    tau_radius = float(tau_radius)
    rho_radius = float(rho_radius)
    cauchy_majorant = float(cauchy_majorant)
    retained_degree = int(retained_degree)
    derivative_order = int(derivative_order)
    if start_time <= 2.0:
        raise ValueError("start_time must exceed 2 for the dyadic log ratio")
    if tau_radius <= 0.0 or rho_radius <= 0.0:
        raise ValueError("Cauchy radii must be positive")
    if cauchy_majorant <= 0.0:
        raise ValueError("cauchy_majorant must be positive")
    if derivative_order < 0:
        raise ValueError("derivative_order must be nonnegative")
    tail_exponent = retained_degree + 1 - derivative_order
    if tail_exponent <= 0:
        raise ValueError("retained_degree must leave a positive tail exponent")

    log_start = np.log(start_time)
    theta0 = max(
        1.0 / (start_time * tau_radius),
        log_start / (start_time * rho_radius),
    )
    shell_ratio = 0.5 * (1.0 + np.log(2.0) / log_start)
    recurrence = HomotheticEscapeDyadicRecurrence(
        endpoint=endpoint,
        start_time=start_time,
        tau_radius=tau_radius,
        rho_radius=rho_radius,
        cauchy_majorant=cauchy_majorant,
        retained_degree=retained_degree,
        derivative_order=derivative_order,
        theta0=float(theta0),
        shell_ratio=float(shell_ratio),
        tail_exponent=tail_exponent,
        majorant_certificate=majorant_certificate,
    )
    if not recurrence.certified:
        raise ValueError("homothetic escape dyadic recurrence is not certified")
    return recurrence


def derive_time_reversed_homothetic_escape_recurrence(
    future_recurrence: HomotheticEscapeDyadicRecurrence,
    *,
    compact_time_rate: float,
) -> TimeReversedHomotheticEscapeRecurrenceCertificate:
    """Derive the incoming homothetic escape endpoint by time reversal."""

    if not isinstance(future_recurrence, HomotheticEscapeDyadicRecurrence):
        raise TypeError("future_recurrence must be a HomotheticEscapeDyadicRecurrence")
    if not future_recurrence.certified:
        raise ValueError("future_recurrence must be certified")
    certificate = TimeReversedHomotheticEscapeRecurrenceCertificate(
        future_recurrence=future_recurrence,
        compact_time_rate=float(compact_time_rate),
    )
    if not certificate.certified:
        raise ValueError("time-reversed homothetic escape recurrence did not certify")
    return certificate


def certify_positive_energy_homothetic_all_real_gluing(
    *,
    masses: Array,
    positions: Array,
    branch_certificate: HomotheticEscapeBranchCertificate,
    projection_invariant_certificate: HomotheticEscapeProjectionInvariantCertificate,
    future_recurrence: HomotheticEscapeDyadicRecurrence,
    past_recurrence: TimeReversedHomotheticEscapeRecurrenceCertificate,
    total_collision_atlas: object,
    selector_entry: object,
    compact_time_rate: float,
    middle_retained_order: int = 12,
    tolerance: float = 1.0e-6,
) -> PositiveEnergyHomotheticAllRealGluingCertificate:
    """Certify the finite homothetic middle between endpoint and collision charts.

    This is a scoped analytic gluing theorem, not an arbitrary-regime
    classifier.  It uses the homothetic radial energy identity to prove that
    the two compact middle intervals stay uniformly collision-free, then feeds
    those derived uniform bounds into the ordinary Taylor recurrence
    constructor.
    """

    masses = np.asarray(masses, dtype=float).reshape(-1)
    positions = np.asarray(positions, dtype=float)
    compact_time_rate = float(compact_time_rate)
    tolerance = float(tolerance)
    middle_retained_order = int(middle_retained_order)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    if positions.ndim != 2 or positions.shape[0] != 3 or positions.shape[1] < 2:
        raise ValueError("positions must have shape (3, dimension>=2)")
    if not future_recurrence.certified:
        raise ValueError("future_recurrence must be certified")
    if not past_recurrence.certified:
        raise ValueError("past_recurrence must be certified")
    if past_recurrence.future_recurrence is not future_recurrence:
        raise ValueError("past_recurrence must be derived from future_recurrence")
    if getattr(total_collision_atlas, "proof_certified", False) is not True:
        raise ValueError("total_collision_atlas must be proof certified")
    if not _finite_jet_selector_entry_certified(selector_entry):
        raise ValueError("selector_entry must be a certified finite-jet selector entry")
    if not np.isfinite(compact_time_rate) or compact_time_rate <= 0.0:
        raise ValueError("compact_time_rate must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    total_mass = float(np.sum(masses))
    centered_positions = positions - np.sum(masses[:, None] * positions, axis=0) / total_mass
    endpoint = future_recurrence.endpoint
    scale_factor = ((9.0 / 2.0) * branch_certificate.gravitational_parameter) ** (1.0 / 3.0)
    expected_quadratic = scale_factor * centered_positions
    collision_branch = getattr(getattr(total_collision_atlas, "evaluation", None), "branch", None)
    collision_certificate = getattr(
        getattr(total_collision_atlas, "evaluation", None),
        "certificate",
        None,
    )
    if collision_branch is None or collision_certificate is None:
        raise ValueError("total_collision_atlas must expose its homothetic branch certificate")
    quadratic = np.asarray(collision_branch.quadratic_coefficient, dtype=float)
    if quadratic.shape != expected_quadratic.shape:
        raise ValueError("collision branch shape does not match homothetic input data")
    scale_consistency_residual = float(
        np.linalg.norm(quadratic - expected_quadratic, ord=np.inf)
    )
    expected_energy_per_inertia = float(branch_certificate.energy / scale_factor**2)
    energy_consistency_residual = float(
        abs(float(collision_branch.energy_per_inertia) - expected_energy_per_inertia)
    )
    collision_time = float(endpoint.time_shift)
    collision_time_residual = float(
        abs(float(collision_certificate.event_time) - collision_time)
    )

    start_tau = abs(float(collision_certificate.start_tau))
    target_tau = abs(float(collision_certificate.target_tau))
    collision_tau = min(start_tau, target_tau)
    if collision_tau <= 0.0:
        raise ValueError("total collision chart must straddle tau=0")
    scalar_tail = float(getattr(collision_certificate, "scalar_tail_bound", 0.0))
    derivative_tail = float(getattr(collision_certificate, "derivative_tail_bound", 0.0))
    scalar_lower_bound = float(collision_branch.radial_u(collision_tau) - scalar_tail)
    z = collision_tau**2
    derivative_combination = float(
        sum(
            (degree + 1.0) * coefficient * z**degree
            for degree, coefficient in enumerate(collision_branch.coefficients)
        )
    )
    scalar_velocity_lower_bound = float(
        (2.0 / (3.0 * collision_tau))
        * (derivative_combination - derivative_tail)
    )
    radial_lower_bound = float(collision_tau**2 * scalar_lower_bound)

    pair_distances = [
        float(np.linalg.norm(quadratic[i] - quadratic[j]))
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    min_scaled_shape_pair_distance = float(min(pair_distances, default=0.0))
    middle_pair_distance_lower_bound = float(
        radial_lower_bound * min_scaled_shape_pair_distance
    )
    future_handoff_radius = (
        homothetic_escape_radius_at_time(
            future_recurrence.start_time,
            endpoint,
        )
        / scale_factor
    )
    shape_sup = float(np.linalg.norm(quadratic, ord=np.inf))
    middle_position_upper_bound = float(future_handoff_radius * shape_sup)
    radial_speed_upper = float(
        np.sqrt(
            max(
                0.0,
                2.0
                * (
                    float(collision_branch.energy_per_inertia)
                    + (2.0 / 9.0) / radial_lower_bound
                ),
            )
        )
    )
    middle_speed_upper_bound = float(radial_speed_upper * shape_sup)
    middle_recurrence = certify_uniformly_collision_free_taylor_recurrence(
        masses,
        pair_distance_lower_bound=middle_pair_distance_lower_bound,
        position_upper_bound=middle_position_upper_bound,
        speed_upper_bound=middle_speed_upper_bound,
        retained_order=middle_retained_order,
    )

    future_collision_handoff_time = float(collision_time + collision_tau**3)
    past_collision_handoff_time = float(collision_time - collision_tau**3)
    future_endpoint_handoff_time = float(future_recurrence.start_time)
    past_endpoint_handoff_time = float(past_recurrence.past_handoff_time)
    future_middle_interval_length = float(
        future_endpoint_handoff_time - future_collision_handoff_time
    )
    past_middle_interval_length = float(
        past_collision_handoff_time - past_endpoint_handoff_time
    )
    step_size = float(getattr(middle_recurrence, "step_size", float("nan")))
    if step_size > 0.0:
        future_middle_chart_count = int(
            np.ceil(future_middle_interval_length / step_size)
        )
        past_middle_chart_count = int(
            np.ceil(past_middle_interval_length / step_size)
        )
    else:
        future_middle_chart_count = 0
        past_middle_chart_count = 0
    per_middle_chart_tail_bound = float(
        getattr(middle_recurrence, "tail_bound", float("inf"))
    )
    future_middle_tail_budget = float(
        future_middle_chart_count * per_middle_chart_tail_bound
    )
    past_middle_tail_budget = float(
        past_middle_chart_count * per_middle_chart_tail_bound
    )
    total_middle_tail_budget = float(
        future_middle_tail_budget + past_middle_tail_budget
    )
    endpoint_tail_budget = float(
        future_recurrence.all_future_tail_bound + past_recurrence.all_past_tail_bound
    )
    total_collision_tail_budget = float(
        getattr(getattr(total_collision_atlas, "tail_budget", None), "local_tail_bound", float("inf"))
    )
    all_real_tail_budget_bound = float(
        endpoint_tail_budget + total_collision_tail_budget + total_middle_tail_budget
    )
    middle_transition_count = (
        max(0, past_middle_chart_count - 1)
        + max(0, future_middle_chart_count - 1)
        + 4
    )
    max_newton_residual_bound = float(
        max(
            float(getattr(projection_invariant_certificate, "projected_newton_residual_bound", np.inf)),
            float(getattr(collision_certificate, "max_newton_residual", np.inf)),
        )
    )
    invariant_residual_bound = float(
        max(
            float(getattr(projection_invariant_certificate, "centered_linear_momentum_norm", np.inf)),
            float(getattr(projection_invariant_certificate, "centered_angular_momentum_norm", np.inf)),
            float(getattr(projection_invariant_certificate, "normalized_energy_residual", np.inf)),
            float(getattr(collision_certificate, "max_center_of_mass", np.inf)),
            float(getattr(collision_certificate, "max_linear_momentum", np.inf)),
            float(getattr(collision_certificate, "max_angular_momentum", np.inf)),
            float(getattr(collision_certificate, "max_energy_error", np.inf)),
        )
    )
    certificate = PositiveEnergyHomotheticAllRealGluingCertificate(
        branch_certificate=branch_certificate,
        projection_invariant_certificate=projection_invariant_certificate,
        future_recurrence=future_recurrence,
        past_recurrence=past_recurrence,
        total_collision_atlas=total_collision_atlas,
        selector_entry=selector_entry,
        middle_recurrence=middle_recurrence,
        collision_tau=collision_tau,
        collision_time=collision_time,
        future_collision_handoff_time=future_collision_handoff_time,
        past_collision_handoff_time=past_collision_handoff_time,
        future_endpoint_handoff_time=future_endpoint_handoff_time,
        past_endpoint_handoff_time=past_endpoint_handoff_time,
        scale_consistency_residual=scale_consistency_residual,
        energy_consistency_residual=energy_consistency_residual,
        collision_time_residual=collision_time_residual,
        scalar_lower_bound=scalar_lower_bound,
        scalar_velocity_lower_bound=scalar_velocity_lower_bound,
        min_scaled_shape_pair_distance=min_scaled_shape_pair_distance,
        middle_pair_distance_lower_bound=middle_pair_distance_lower_bound,
        middle_position_upper_bound=middle_position_upper_bound,
        middle_speed_upper_bound=middle_speed_upper_bound,
        past_middle_interval_length=past_middle_interval_length,
        future_middle_interval_length=future_middle_interval_length,
        middle_step_size=step_size,
        future_middle_chart_count_bound=future_middle_chart_count,
        past_middle_chart_count_bound=past_middle_chart_count,
        middle_transition_count_bound=middle_transition_count,
        per_middle_chart_tail_bound=per_middle_chart_tail_bound,
        past_middle_tail_budget=past_middle_tail_budget,
        future_middle_tail_budget=future_middle_tail_budget,
        total_middle_tail_budget=total_middle_tail_budget,
        endpoint_tail_budget=endpoint_tail_budget,
        total_collision_tail_budget=total_collision_tail_budget,
        all_real_tail_budget_bound=all_real_tail_budget_bound,
        max_newton_residual_bound=max_newton_residual_bound,
        invariant_residual_bound=invariant_residual_bound,
        tolerance=tolerance,
    )
    if not certificate.certified:
        raise ValueError("positive-energy homothetic all-real gluing did not certify")
    return certificate


def derive_homothetic_escape_implicit_cauchy_majorant(
    endpoint: HomotheticEscapeEndpointData,
    *,
    tau_radius: float,
    rho_radius: float,
    x_radius: float,
) -> HomotheticEscapeImplicitCauchyMajorantCertificate:
    """Derive a scalar endpoint Cauchy majorant from the implicit equation.

    The proof uses Rouche's theorem on the circle ``|x-c|=x_radius`` for the
    analytic endpoint equation in ``(x,tau,rho)``.  The returned majorant bounds
    the log-subtracted scalar endpoint function
    ``Y=Phi(tau,rho)+(mu/c^2)rho`` on the certified polydisc.
    """

    if not endpoint.certified:
        raise ValueError("endpoint must be certified")
    tau_radius = float(tau_radius)
    rho_radius = float(rho_radius)
    x_radius = float(x_radius)
    if tau_radius <= 0.0 or rho_radius <= 0.0 or x_radius <= 0.0:
        raise ValueError("tau_radius, rho_radius, and x_radius must be positive")
    c = float(endpoint.asymptotic_speed)
    beta = float(endpoint.beta)
    if not (0.0 < x_radius < c):
        raise ValueError("x_radius must lie in (0, asymptotic_speed)")
    singularity_margin = c - x_radius - beta * tau_radius
    if singularity_margin <= 0.0:
        raise ValueError("polydisc reaches the square-root/log singularity barrier")

    sqrt_remainder_bound = (
        beta * tau_radius * (c + x_radius) / (c * (c - x_radius))
    )
    log_argument_lower_bound = np.sqrt(c - x_radius) + np.sqrt(singularity_margin)
    log_argument_upper_bound = np.sqrt(c + x_radius) + np.sqrt(
        c + x_radius + beta * tau_radius
    )
    log_factor_bound = (
        max(
            abs(np.log(log_argument_lower_bound)),
            abs(np.log(log_argument_upper_bound)),
        )
        + 0.5 * np.pi
        + 0.5 * abs(np.log(beta))
    )
    implicit_remainder_bound = (
        tau_radius * abs(float(endpoint.time_shift))
        + sqrt_remainder_bound
        + beta * tau_radius * log_factor_bound / c
        + beta * rho_radius / (2.0 * c)
    )
    rouche_linear_bound = x_radius / c
    cauchy_majorant = c + x_radius + 0.5 * beta * rho_radius
    certificate = HomotheticEscapeImplicitCauchyMajorantCertificate(
        endpoint=endpoint,
        tau_radius=tau_radius,
        rho_radius=rho_radius,
        x_radius=x_radius,
        singularity_margin=float(singularity_margin),
        log_argument_lower_bound=float(log_argument_lower_bound),
        log_argument_upper_bound=float(log_argument_upper_bound),
        log_factor_bound=float(log_factor_bound),
        sqrt_remainder_bound=float(sqrt_remainder_bound),
        implicit_remainder_bound=float(implicit_remainder_bound),
        rouche_linear_bound=float(rouche_linear_bound),
        cauchy_majorant=float(cauchy_majorant),
    )
    if not certificate.certified:
        raise ValueError("homothetic escape implicit Cauchy majorant is not certified")
    return certificate


def homothetic_escape_time_integral(
    radius: float,
    gravitational_parameter: float,
    asymptotic_speed: float,
) -> float:
    """Antiderivative for the positive-energy homothetic radial equation."""

    radius = float(radius)
    gravitational_parameter = float(gravitational_parameter)
    asymptotic_speed = float(asymptotic_speed)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if gravitational_parameter <= 0.0:
        raise ValueError("gravitational_parameter must be positive")
    if asymptotic_speed <= 0.0:
        raise ValueError("asymptotic_speed must be positive")
    beta = 2.0 * gravitational_parameter / asymptotic_speed**2
    return float(
        (
            np.sqrt(radius * (radius + beta))
            - beta * np.arcsinh(np.sqrt(radius / beta))
        )
        / asymptotic_speed
    )


def homothetic_escape_radius_at_time(
    physical_time: float,
    endpoint: HomotheticEscapeEndpointData,
    *,
    iterations: int = 96,
) -> float:
    """Invert the radial antiderivative on the expanding branch."""

    physical_time = float(physical_time)
    if physical_time <= 0.0:
        raise ValueError("physical_time must be positive")
    low_radius = endpoint.initial_radius
    high_radius = max(2.0 * low_radius, 2.0 * endpoint.asymptotic_speed * physical_time)
    while (
        homothetic_escape_time_integral(
            high_radius,
            endpoint.gravitational_parameter,
            endpoint.asymptotic_speed,
        )
        + endpoint.time_shift
        < physical_time
    ):
        high_radius *= 2.0

    for _iteration in range(iterations):
        mid_radius = 0.5 * (low_radius + high_radius)
        mid_time = (
            homothetic_escape_time_integral(
                mid_radius,
                endpoint.gravitational_parameter,
                endpoint.asymptotic_speed,
            )
            + endpoint.time_shift
        )
        if mid_time < physical_time:
            low_radius = mid_radius
        else:
            high_radius = mid_radius
    return float(0.5 * (low_radius + high_radius))


def homothetic_escape_endpoint_implicit_residual(
    scaled_radius: float,
    inverse_time: float,
    log_variable: float,
    endpoint: HomotheticEscapeEndpointData,
) -> float:
    """Evaluate the analytic implicit endpoint equation H(x,tau,rho)."""

    scaled_radius = float(scaled_radius)
    inverse_time = float(inverse_time)
    log_variable = float(log_variable)
    if scaled_radius <= 0.0:
        raise ValueError("scaled_radius must be positive")
    beta = endpoint.beta
    c = endpoint.asymptotic_speed
    analytic_part = (
        np.sqrt(scaled_radius * (scaled_radius + beta * inverse_time)) / c
        - (beta * inverse_time / c)
        * (
            np.log(
                np.sqrt(scaled_radius)
                + np.sqrt(scaled_radius + beta * inverse_time)
            )
            - 0.5 * np.log(beta)
        )
    )
    return float(
        inverse_time * endpoint.time_shift
        + analytic_part
        + (beta / (2.0 * c)) * log_variable
        - 1.0
    )


@dataclass(frozen=True)
class ScatteringEndpointConstants:
    """Fixed-point constants for a distinct-velocity scattering endpoint."""

    masses: Array
    asymptotic_velocities: Array
    offsets: Array
    log_vector: Array
    start_time: float
    velocity_gap: float
    lipschitz_constant: float
    pair_error_scale: float
    body_error_scale: float
    initial_position_increment: float
    contraction_factor: float
    ball_radius: float
    separation_loss: float
    tube_loss: float
    initial_velocity_increment: float
    velocity_transfer: float
    shell_ratio: float

    @property
    def log_start(self) -> float:
        return float(np.log(self.start_time))

    @property
    def certified(self) -> bool:
        return bool(
            self.velocity_gap > 0.0
            and self.lipschitz_constant > 0.0
            and self.start_time > np.exp(1.0)
            and self.contraction_factor < 1.0
            and self.separation_loss <= 0.5 * self.velocity_gap
            and self.tube_loss <= 0.125 * self.velocity_gap
            and self.shell_ratio < 1.0
        )

    @property
    def full_velocity_correction_weight(self) -> float:
        return float(
            self.initial_velocity_increment
            + self.velocity_transfer
            * self.initial_position_increment
            / (1.0 - self.contraction_factor)
        )

    def value_weight(self, retained_corrections: int) -> float:
        retained_corrections = int(retained_corrections)
        if retained_corrections < 0:
            raise ValueError("retained_corrections must be nonnegative")
        return float(
            self.initial_position_increment
            * self.contraction_factor**retained_corrections
            / (1.0 - self.contraction_factor)
        )

    def first_jet_weight(self, retained_corrections: int) -> float:
        retained_corrections = int(retained_corrections)
        if retained_corrections <= 0:
            raise ValueError("retained_corrections must be positive for first-jet tails")
        return float(
            self.value_weight(retained_corrections)
            + self.velocity_transfer
            * self.initial_position_increment
            * self.contraction_factor ** (retained_corrections - 1)
            / (1.0 - self.contraction_factor)
        )

    def residual_weight(self, retained_corrections: int) -> float:
        retained_corrections = int(retained_corrections)
        if retained_corrections <= 0:
            raise ValueError("retained_corrections must be positive for residual tails")
        return float(
            self.lipschitz_constant
            * self.initial_position_increment
            * self.contraction_factor ** (retained_corrections - 1)
        )


@dataclass(frozen=True)
class ScatteringEndpointDyadicRecurrence:
    """Dyadic all-future endpoint recurrence for a scattering fixed point."""

    constants: ScatteringEndpointConstants
    retained_corrections: int

    @property
    def certified(self) -> bool:
        return bool(self.constants.certified and self.retained_corrections > 0)

    @property
    def component_weights(self) -> dict[str, float]:
        residual_weight = self.constants.residual_weight(self.retained_corrections)
        return {
            "value": self.constants.value_weight(self.retained_corrections),
            "first_jet": self.constants.first_jet_weight(self.retained_corrections),
            "residual": residual_weight,
            "physical_residual": residual_weight,
        }

    @property
    def component_exponents(self) -> dict[str, int]:
        return {"value": 2, "first_jet": 1, "residual": 2, "physical_residual": 4}

    @property
    def component_ratios(self) -> dict[str, float]:
        log_ratio = 1.0 + np.log(2.0) / self.constants.log_start
        return {
            name: float(np.nextafter(2.0 ** (-exponent) * log_ratio**2, np.inf))
            for name, exponent in self.component_exponents.items()
        }

    def shell_left_tau(self, shell_index: int) -> float:
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        return float((1.0 / self.constants.start_time) * 2.0 ** (-shell_index))

    def shell_tail_bound(self, component: str, shell_index: int) -> float:
        if component not in self.component_weights:
            raise ValueError(f"unknown scattering endpoint component: {component}")
        tau = self.shell_left_tau(shell_index)
        exponent = self.component_exponents[component]
        return float(
            self.component_weights[component]
            * tau**exponent
            * np.log(1.0 / tau) ** 2
        )

    def geometric_shell_tail_bound(self, component: str, shell_index: int) -> float:
        if component not in self.component_weights:
            raise ValueError(f"unknown scattering endpoint component: {component}")
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        return float(
            self.shell_tail_bound(component, 0)
            * self.component_ratios[component] ** shell_index
        )

    def all_future_tail_bound(self, component: str) -> float:
        if component not in self.component_weights:
            raise ValueError(f"unknown scattering endpoint component: {component}")
        return float(
            self.shell_tail_bound(component, 0)
            / (1.0 - self.component_ratios[component])
        )

    def physical_residual_shell_bound(self, shell_index: int) -> float:
        return self.shell_tail_bound("physical_residual", shell_index)


def construct_scattering_endpoint_constants(
    masses: Array,
    asymptotic_velocities: Array,
    offsets: Array,
    *,
    start_time: float,
    require_certified: bool = True,
) -> ScatteringEndpointConstants:
    """Derive fixed-point constants for a prescribed scattering endpoint."""

    masses = np.asarray(masses, dtype=float)
    asymptotic_velocities = np.asarray(asymptotic_velocities, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    start_time = float(start_time)
    if masses.ndim != 1 or asymptotic_velocities.ndim != 2 or offsets.ndim != 2:
        raise ValueError("masses, asymptotic_velocities, and offsets must be arrays")
    if asymptotic_velocities.shape != offsets.shape:
        raise ValueError("asymptotic_velocities and offsets must have matching shape")
    if asymptotic_velocities.shape[0] != masses.shape[0]:
        raise ValueError("body count mismatch")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive")
    if start_time <= np.exp(1.0):
        raise ValueError("start_time must exceed e")

    velocity_gap = min(
        np.linalg.norm(asymptotic_velocities[j] - asymptotic_velocities[i])
        for i in range(asymptotic_velocities.shape[0])
        for j in range(i + 1, asymptotic_velocities.shape[0])
    )
    if velocity_gap <= 0.0:
        raise ValueError("asymptotic velocities must be distinct")
    log_vector = accelerations(asymptotic_velocities, masses)
    log_start = np.log(start_time)
    lipschitz_constant = 256.0 * float(np.sum(masses)) / velocity_gap**3
    pair_error_scale = max(
        np.linalg.norm(offsets[j] - offsets[i]) / log_start
        + np.linalg.norm(log_vector[j] - log_vector[i])
        for i in range(asymptotic_velocities.shape[0])
        for j in range(i + 1, asymptotic_velocities.shape[0])
    )
    body_error_scale = max(np.linalg.norm(offset) for offset in offsets) / log_start + max(
        np.linalg.norm(row) for row in log_vector
    )
    position_initial_factor = 1.0 / (2.0 * log_start) + 3.0 / (4.0 * log_start**2)
    position_contraction_factor = (
        1.0 / 6.0 + 5.0 / (18.0 * log_start) + 19.0 / (108.0 * log_start**2)
    ) / start_time
    velocity_initial_factor = 1.0 / (2.0 * log_start) + 1.0 / (4.0 * log_start**2)
    velocity_transfer_factor = (
        1.0 / 3.0 + 2.0 / (9.0 * log_start) + 2.0 / (27.0 * log_start**2)
    ) / start_time
    initial_position_increment = lipschitz_constant * body_error_scale * position_initial_factor
    contraction_factor = lipschitz_constant * position_contraction_factor
    ball_radius = 2.0 * initial_position_increment
    separation_loss = pair_error_scale * log_start / start_time
    tube_loss = ball_radius * log_start**2 / start_time**2
    initial_velocity_increment = lipschitz_constant * body_error_scale * velocity_initial_factor
    velocity_transfer = lipschitz_constant * velocity_transfer_factor
    shell_ratio = 0.5 * (1.0 + np.log(2.0) / log_start) ** 2
    constants = ScatteringEndpointConstants(
        masses=masses,
        asymptotic_velocities=asymptotic_velocities,
        offsets=offsets,
        log_vector=log_vector,
        start_time=start_time,
        velocity_gap=float(velocity_gap),
        lipschitz_constant=float(lipschitz_constant),
        pair_error_scale=float(pair_error_scale),
        body_error_scale=float(body_error_scale),
        initial_position_increment=float(initial_position_increment),
        contraction_factor=float(contraction_factor),
        ball_radius=float(ball_radius),
        separation_loss=float(separation_loss),
        tube_loss=float(tube_loss),
        initial_velocity_increment=float(initial_velocity_increment),
        velocity_transfer=float(velocity_transfer),
        shell_ratio=float(shell_ratio),
    )
    if require_certified and not constants.certified:
        raise ValueError("scattering endpoint constants are not certified")
    return constants


def find_scattering_endpoint_tail_start(
    masses: Array,
    asymptotic_velocities: Array,
    offsets: Array,
    *,
    initial_start_time: float | None = None,
    max_doublings: int = 24,
) -> ScatteringEndpointConstants:
    """Find an effective start time satisfying the scattering tail inequalities."""

    start_time = float(initial_start_time) if initial_start_time is not None else float(np.exp(2.0))
    if start_time <= 0.0:
        raise ValueError("initial_start_time must be positive")
    for _iteration in range(int(max_doublings) + 1):
        constants = construct_scattering_endpoint_constants(
            masses,
            asymptotic_velocities,
            offsets,
            start_time=start_time,
            require_certified=False,
        )
        if constants.certified:
            return constants
        start_time *= 2.0
    raise RuntimeError("effective scattering tail-start search did not terminate")


def construct_scattering_endpoint_dyadic_recurrence(
    constants: ScatteringEndpointConstants,
    *,
    retained_corrections: int,
) -> ScatteringEndpointDyadicRecurrence:
    """Construct the dyadic value, first-jet, and residual shell recurrence."""

    recurrence = ScatteringEndpointDyadicRecurrence(
        constants=constants,
        retained_corrections=int(retained_corrections),
    )
    if not recurrence.certified:
        raise ValueError("scattering endpoint dyadic recurrence is not certified")
    return recurrence


@dataclass(frozen=True)
class ScatteringMiddleBudgets:
    """Finite middle verification budgets for a two-ended scattering atlas."""

    value: float
    first_jet: float
    residual: float
    physical_residual: float

    @property
    def certified(self) -> bool:
        return bool(
            np.isfinite(self.value)
            and np.isfinite(self.first_jet)
            and np.isfinite(self.residual)
            and np.isfinite(self.physical_residual)
            and self.value >= 0.0
            and self.first_jet >= 0.0
            and self.residual >= 0.0
            and self.physical_residual >= 0.0
        )

    def for_component(self, component: str) -> float:
        if component == "value":
            return float(self.value)
        if component == "first_jet":
            return float(self.first_jet)
        if component == "residual":
            return float(self.residual)
        if component == "physical_residual":
            return float(self.physical_residual)
        raise ValueError(f"unknown scattering atlas budget component: {component}")


@dataclass(frozen=True)
class TwoEndedScatteringAtlasRecurrence:
    """All-real budget from past endpoint, finite middle, and future endpoint."""

    past: ScatteringEndpointDyadicRecurrence
    future: ScatteringEndpointDyadicRecurrence
    middle: ScatteringMiddleBudgets
    compact_time_rate: float

    @property
    def components(self) -> tuple[str, ...]:
        return ("value", "first_jet", "residual", "physical_residual")

    @property
    def certified(self) -> bool:
        return bool(
            self.past.certified
            and self.future.certified
            and self.middle.certified
            and self.compact_time_rate > 0.0
            and -1.0 < self.past_handoff_compact_parameter < 0.0
            and 0.0 < self.future_handoff_compact_parameter < 1.0
            and all(self.endpoint_budget(component) < np.inf for component in self.components)
            and all(self.total_budget(component) < np.inf for component in self.components)
        )

    @property
    def past_handoff_compact_parameter(self) -> float:
        return float(np.tanh(-self.compact_time_rate * self.past.constants.start_time))

    @property
    def future_handoff_compact_parameter(self) -> float:
        return float(np.tanh(self.compact_time_rate * self.future.constants.start_time))

    def endpoint_budget(self, component: str) -> float:
        return float(
            self.past.all_future_tail_bound(component)
            + self.future.all_future_tail_bound(component)
        )

    def total_budget(self, component: str) -> float:
        return float(self.middle.for_component(component) + self.endpoint_budget(component))

    def endpoint_shell_bound(self, side: str, component: str, shell_index: int) -> float:
        if side == "past":
            return self.past.shell_tail_bound(component, shell_index)
        if side == "future":
            return self.future.shell_tail_bound(component, shell_index)
        raise ValueError("side must be 'past' or 'future'")


@dataclass(frozen=True)
class TwoEndedScatteringInvariantMatchCertificate:
    """Asymptotic invariant compatibility for prescribed scattering ends."""

    past_momentum: tuple[float, ...]
    future_momentum: tuple[float, ...]
    past_center_offset: tuple[float, ...]
    future_center_offset: tuple[float, ...]
    past_angular_momentum: float
    future_angular_momentum: float
    past_energy: float
    future_energy: float
    past_log_momentum_drift_norm: float
    future_log_momentum_drift_norm: float
    past_log_angular_drift: float
    future_log_angular_drift: float
    tolerance: float

    @property
    def momentum_matches(self) -> bool:
        return _max_abs_difference(self.past_momentum, self.future_momentum) <= self.tolerance

    @property
    def center_offset_matches(self) -> bool:
        return _max_abs_difference(self.past_center_offset, self.future_center_offset) <= self.tolerance

    @property
    def angular_momentum_matches(self) -> bool:
        return abs(self.past_angular_momentum - self.future_angular_momentum) <= self.tolerance

    @property
    def energy_matches(self) -> bool:
        return abs(self.past_energy - self.future_energy) <= self.tolerance

    @property
    def log_drift_cancels(self) -> bool:
        return bool(
            self.past_log_momentum_drift_norm <= self.tolerance
            and self.future_log_momentum_drift_norm <= self.tolerance
            and abs(self.past_log_angular_drift) <= self.tolerance
            and abs(self.future_log_angular_drift) <= self.tolerance
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.tolerance > 0.0
            and self.momentum_matches
            and self.center_offset_matches
            and self.angular_momentum_matches
            and self.energy_matches
            and self.log_drift_cancels
        )


def certify_two_ended_scattering_invariant_match(
    atlas: TwoEndedScatteringAtlasRecurrence,
    *,
    tolerance: float = 1.0e-10,
) -> TwoEndedScatteringInvariantMatchCertificate:
    """Certify matching classical invariants for a two-ended scattering atlas."""

    if not atlas.certified:
        raise ValueError("two-ended scattering atlas recurrence must be certified")
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    masses = np.asarray(atlas.future.constants.masses, dtype=float)
    past_masses = np.asarray(atlas.past.constants.masses, dtype=float)
    if not np.allclose(masses, past_masses, rtol=0.0, atol=tolerance):
        raise ValueError("past and future scattering endpoints must use matching masses")
    past_velocities = -np.asarray(atlas.past.constants.asymptotic_velocities, dtype=float)
    past_offsets = np.asarray(atlas.past.constants.offsets, dtype=float)
    future_velocities = np.asarray(atlas.future.constants.asymptotic_velocities, dtype=float)
    future_offsets = np.asarray(atlas.future.constants.offsets, dtype=float)
    past = _scattering_endpoint_invariants(masses, past_velocities, past_offsets)
    future = _scattering_endpoint_invariants(masses, future_velocities, future_offsets)
    past_log_vector = accelerations(past_velocities, masses)
    future_log_vector = accelerations(future_velocities, masses)
    certificate = TwoEndedScatteringInvariantMatchCertificate(
        past_momentum=tuple(float(value) for value in past["momentum"]),
        future_momentum=tuple(float(value) for value in future["momentum"]),
        past_center_offset=tuple(float(value) for value in past["center_offset"]),
        future_center_offset=tuple(float(value) for value in future["center_offset"]),
        past_angular_momentum=float(past["angular_momentum"]),
        future_angular_momentum=float(future["angular_momentum"]),
        past_energy=float(past["energy"]),
        future_energy=float(future["energy"]),
        past_log_momentum_drift_norm=float(np.linalg.norm(np.sum(masses[:, None] * past_log_vector, axis=0), ord=np.inf)),
        future_log_momentum_drift_norm=float(np.linalg.norm(np.sum(masses[:, None] * future_log_vector, axis=0), ord=np.inf)),
        past_log_angular_drift=float(_weighted_planar_wedge_sum(masses, past_velocities, past_log_vector)),
        future_log_angular_drift=float(_weighted_planar_wedge_sum(masses, future_velocities, future_log_vector)),
        tolerance=tolerance,
    )
    return certificate


def coerce_scattering_middle_budgets(
    middle_budgets: ScatteringMiddleBudgets | dict[str, float],
) -> ScatteringMiddleBudgets:
    if isinstance(middle_budgets, ScatteringMiddleBudgets):
        return middle_budgets
    return ScatteringMiddleBudgets(
        value=float(middle_budgets["value"]),
        first_jet=float(middle_budgets["first_jet"]),
        residual=float(middle_budgets["residual"]),
        physical_residual=float(middle_budgets.get("physical_residual", middle_budgets["residual"])),
    )


def construct_two_ended_scattering_atlas_recurrence(
    masses: Array,
    past_physical_velocities: Array,
    past_offsets: Array,
    future_physical_velocities: Array,
    future_offsets: Array,
    *,
    retained_corrections: int,
    middle_budgets: ScatteringMiddleBudgets | dict[str, float],
    compact_time_rate: float = 1.0e-8,
    initial_start_time: float | None = None,
    max_doublings: int = 24,
) -> TwoEndedScatteringAtlasRecurrence:
    """Compose two endpoint recurrences with finite middle budgets."""

    past_constants = find_scattering_endpoint_tail_start(
        masses,
        -np.asarray(past_physical_velocities, dtype=float),
        past_offsets,
        initial_start_time=initial_start_time,
        max_doublings=max_doublings,
    )
    future_constants = find_scattering_endpoint_tail_start(
        masses,
        future_physical_velocities,
        future_offsets,
        initial_start_time=initial_start_time,
        max_doublings=max_doublings,
    )
    recurrence = TwoEndedScatteringAtlasRecurrence(
        past=construct_scattering_endpoint_dyadic_recurrence(
            past_constants,
            retained_corrections=retained_corrections,
        ),
        future=construct_scattering_endpoint_dyadic_recurrence(
            future_constants,
            retained_corrections=retained_corrections,
        ),
        middle=coerce_scattering_middle_budgets(middle_budgets),
        compact_time_rate=float(compact_time_rate),
    )
    if not recurrence.certified:
        raise ValueError("two-ended scattering atlas recurrence is not certified")
    return recurrence


def _scattering_endpoint_invariants(
    masses: Array,
    velocities: Array,
    offsets: Array,
) -> dict[str, Array | float]:
    return {
        "momentum": np.sum(masses[:, None] * velocities, axis=0),
        "center_offset": np.sum(masses[:, None] * offsets, axis=0),
        "angular_momentum": _weighted_planar_wedge_sum(masses, offsets, velocities),
        "energy": 0.5 * float(np.sum(masses[:, None] * velocities**2)),
    }


def _positive_newtonian_potential(positions: Array, masses: Array) -> float:
    potential = 0.0
    for first in range(positions.shape[0]):
        for second in range(first + 1, positions.shape[0]):
            distance = float(np.linalg.norm(positions[first] - positions[second]))
            if distance == 0.0:
                return float("inf")
            potential += float(masses[first] * masses[second] / distance)
    return potential


def _weighted_angular_momentum_norm(
    masses: Array,
    positions: Array,
    velocities: Array,
) -> float:
    components = []
    for first_axis in range(positions.shape[1]):
        for second_axis in range(first_axis + 1, positions.shape[1]):
            components.append(
                sum(
                    masses[index]
                    * (
                        positions[index, first_axis] * velocities[index, second_axis]
                        - positions[index, second_axis] * velocities[index, first_axis]
                    )
                    for index in range(positions.shape[0])
                )
            )
    return float(max((abs(component) for component in components), default=0.0))


def _weighted_planar_wedge_sum(masses: Array, left: Array, right: Array) -> float:
    if left.shape[1] != 2 or right.shape[1] != 2:
        raise ValueError("scattering invariant matching currently expects planar endpoint data")
    return float(
        sum(
            masses[index]
            * (left[index, 0] * right[index, 1] - left[index, 1] * right[index, 0])
            for index in range(left.shape[0])
        )
    )


def _max_abs_difference(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        return float("inf")
    return float(max((abs(a - b) for a, b in zip(left, right)), default=0.0))
