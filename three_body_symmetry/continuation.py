"""Taylor-chart continuation for general three-body initial data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .global_invariants import (
    certify_interval_center_of_mass_motion,
    certify_interval_centered_angular_momentum_conservation,
    certify_interval_linear_momentum_conservation,
    certify_interval_total_energy_conservation,
)
from .series import (
    TaylorSolution,
    acceleration_coefficients,
    construct_interval_taylor_solution_from_intervals,
    construct_taylor_solution,
)


Array = np.ndarray


@dataclass(frozen=True)
class TaylorStep:
    """One local Taylor chart used during continuation."""

    start_time: float
    step_size: float
    series: TaylorSolution
    truncation_indicator: float
    min_pair_distance: float

    @property
    def end_time(self) -> float:
        return self.start_time + self.step_size


@dataclass(frozen=True)
class ContinuedSolution:
    """Projected states produced by successive Taylor charts."""

    masses: Array
    times: Array
    states: Array
    steps: tuple[TaylorStep, ...]

    @property
    def final_state(self) -> Array:
        return self.states[-1]

    @property
    def position_shape(self) -> tuple[int, int]:
        body_count = int(self.masses.shape[0])
        dimension = int(self.states.shape[1] // (2 * body_count))
        return body_count, dimension

    def final_positions_velocities(self) -> tuple[Array, Array]:
        body_count, dimension = self.position_shape
        state = self.final_state
        positions = state[: body_count * dimension].reshape(body_count, dimension)
        velocities = state[body_count * dimension :].reshape(body_count, dimension)
        return positions, velocities


@dataclass(frozen=True)
class UniformCollisionFreeTaylorRecurrenceCertificate:
    """All-future ordinary Taylor recurrence under explicit uniform bounds.

    The hypotheses are intentionally visible: this certificate proves the
    fixed-step recurrence and one-chart tail formula once a future branch is
    known to stay inside the supplied separation, position, and speed bounds.
    It does not classify arbitrary initial data into that regime.
    """

    masses: tuple[float, ...]
    pair_distance_lower_bound: float
    position_upper_bound: float
    speed_upper_bound: float
    retained_order: int
    step_size: float
    position_radius: float
    velocity_radius: float
    lower_pair_distance_on_tube: float
    acceleration_bound: float
    time_radius: float
    step_ratio: float
    state_sup_bound: float
    tail_bound: float
    hypothesis_scope: str = "requires_global_uniform_bounds_on_the_actual_centered_branch"
    chart_family: str = "ordinary_taylor"

    @property
    def positive_masses_certified(self) -> bool:
        return bool(self.masses and all(np.isfinite(mass) and mass > 0.0 for mass in self.masses))

    @property
    def domain_certified(self) -> bool:
        return bool(
            self.pair_distance_lower_bound > 0.0
            and self.position_upper_bound >= 0.0
            and self.speed_upper_bound >= 0.0
            and self.position_radius > 0.0
            and self.velocity_radius > 0.0
            and self.lower_pair_distance_on_tube > 0.0
        )

    @property
    def recurrence_closes(self) -> bool:
        return bool(
            self.positive_masses_certified
            and self.domain_certified
            and self.retained_order >= 0
            and self.acceleration_bound > 0.0
            and self.time_radius > 0.0
            and self.step_size > 0.0
            and 0.0 < self.step_ratio < 1.0
            and np.isfinite(self.state_sup_bound)
            and self.state_sup_bound >= 0.0
            and np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
        )

    @property
    def newton_residual_certified(self) -> bool:
        return self.recurrence_closes

    @property
    def projection_certified(self) -> bool:
        return self.recurrence_closes

    @property
    def transition_certified(self) -> bool:
        return self.recurrence_closes

    @property
    def tail_budget_certified(self) -> bool:
        return self.recurrence_closes

    @property
    def invariant_domain_certified(self) -> bool:
        return self.recurrence_closes

    @property
    def certified(self) -> bool:
        return self.recurrence_closes

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        obligations: list[str] = []
        if not self.positive_masses_certified:
            obligations.append("positive_masses")
        if not self.domain_certified:
            obligations.append("uniform_collision_free_domain")
        if self.retained_order < 0:
            obligations.append("nonnegative_retained_order")
        if not (self.acceleration_bound > 0.0 and self.time_radius > 0.0):
            obligations.append("positive_cauchy_radius")
        if not (self.step_size > 0.0 and 0.0 < self.step_ratio < 1.0):
            obligations.append("fixed_step_inside_cauchy_radius")
        if not (np.isfinite(self.tail_bound) and self.tail_bound >= 0.0):
            obligations.append("finite_uniform_tail_bound")
        return tuple(obligations)

    def chart_tail_bound(self, retained_order: int | None = None) -> float:
        """Return the uniform omitted-tail bound for a retained Taylor order."""

        order = self.retained_order if retained_order is None else int(retained_order)
        if order < 0:
            raise ValueError("retained_order cannot be negative")
        if not (0.0 < self.step_ratio < 1.0):
            raise ValueError("step ratio must lie in (0, 1)")
        return float(
            self.state_sup_bound
            * self.step_ratio ** (order + 1)
            / (1.0 - self.step_ratio)
        )


@dataclass(frozen=True)
class UniformCollisionFreeOrdinaryChartLedgerCertificate:
    """Initial ordinary-chart ledgers for the uniform recurrence constructor."""

    residual_certificate: object
    center_of_mass_certificate: object
    linear_momentum_certificate: object
    angular_momentum_certificate: object
    energy_certificate: object
    recurrence_certificate: UniformCollisionFreeTaylorRecurrenceCertificate
    chart_family: str = "ordinary_taylor_physical_coordinates"

    @property
    def newton_residual_certified(self) -> bool:
        return bool(getattr(self.residual_certificate, "certified", False))

    @property
    def projection_certified(self) -> bool:
        return bool(
            self.chart_family == "ordinary_taylor_physical_coordinates"
            and self.newton_residual_certified
        )

    @property
    def invariants_certified(self) -> bool:
        return bool(
            getattr(self.center_of_mass_certificate, "certified", False)
            and getattr(self.linear_momentum_certificate, "certified", False)
            and getattr(self.angular_momentum_certificate, "certified", False)
            and getattr(self.energy_certificate, "certified", False)
        )

    @property
    def transition_certified(self) -> bool:
        return bool(self.recurrence_certificate.recurrence_closes)

    @property
    def tail_budget_certified(self) -> bool:
        return bool(
            self.recurrence_certificate.tail_budget_certified
            and np.isfinite(self.recurrence_certificate.tail_bound)
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.newton_residual_certified
            and self.projection_certified
            and self.invariants_certified
            and self.transition_certified
            and self.tail_budget_certified
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.newton_residual_certified:
            missing.append("ordinary_newton_residual")
        if not self.projection_certified:
            missing.append("ordinary_physical_projection")
        if not getattr(self.center_of_mass_certificate, "certified", False):
            missing.append("center_of_mass_motion")
        if not getattr(self.linear_momentum_certificate, "certified", False):
            missing.append("linear_momentum_conservation")
        if not getattr(self.angular_momentum_certificate, "certified", False):
            missing.append("centered_angular_momentum_conservation")
        if not getattr(self.energy_certificate, "certified", False):
            missing.append("total_energy_conservation")
        if not self.transition_certified:
            missing.append("fixed_step_transition_inside_cauchy_radius")
        if not self.tail_budget_certified:
            missing.append("uniform_tail_budget")
        return tuple(missing)


def pairwise_distances(positions: Array) -> Array:
    positions = np.asarray(positions, dtype=float)
    distances = []
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            distances.append(np.linalg.norm(positions[i] - positions[j]))
    return np.array(distances, dtype=float)


def _validate_three_body_initial_data(
    positions: Array,
    velocities: Array,
    masses: Array,
) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 2:
        raise ValueError("positions must have shape (3, dimension)")
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    if positions.shape[0] != 3:
        raise ValueError("this continuation is specialized to three bodies")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(velocities))
        and np.all(np.isfinite(masses))
    ):
        raise ValueError("initial data must be finite")
    if np.min(pairwise_distances(positions)) <= 0.0:
        raise ValueError("initial data must be collision-free")
    return positions, velocities, masses


def certify_uniformly_collision_free_taylor_recurrence(
    masses: Array,
    *,
    pair_distance_lower_bound: float,
    position_upper_bound: float,
    speed_upper_bound: float,
    retained_order: int,
    step_size: float | None = None,
    position_radius: float | None = None,
    velocity_radius: float | None = None,
    safety: float = 0.5,
) -> UniformCollisionFreeTaylorRecurrenceCertificate:
    """Derive the fixed-step all-future ordinary Taylor tail recurrence.

    This is the executable form of the uniformly collision-free recurrence:
    if every future centered state satisfies the supplied bounds, one Cauchy
    radius and one fixed step work for every ordinary Taylor chart.
    """

    masses = np.asarray(masses, dtype=float)
    if masses.ndim != 1 or masses.shape[0] != 3:
        raise ValueError("three positive masses are required")
    pair_distance_lower_bound = float(pair_distance_lower_bound)
    position_upper_bound = float(position_upper_bound)
    speed_upper_bound = float(speed_upper_bound)
    retained_order = int(retained_order)
    safety = float(safety)
    if position_radius is None:
        position_radius = pair_distance_lower_bound / 8.0
    position_radius = float(position_radius)
    lower_pair_distance_on_tube = pair_distance_lower_bound - 2.0 * position_radius
    if lower_pair_distance_on_tube > 0.0 and np.all(np.isfinite(masses)):
        acceleration_bound = float(np.sum(masses) / lower_pair_distance_on_tube**2)
    else:
        acceleration_bound = float("nan")
    if velocity_radius is None:
        if np.isfinite(acceleration_bound) and position_radius > 0.0:
            velocity_radius = max(1.0, float(np.sqrt(acceleration_bound * position_radius)))
        else:
            velocity_radius = float("nan")
    velocity_radius = float(velocity_radius)
    if (
        position_radius > 0.0
        and velocity_radius > 0.0
        and acceleration_bound > 0.0
        and speed_upper_bound >= 0.0
    ):
        time_radius = min(
            position_radius / (speed_upper_bound + velocity_radius),
            velocity_radius / acceleration_bound,
        )
    else:
        time_radius = float("nan")
    if step_size is None:
        step_size = safety * time_radius
    step_size = float(step_size)
    step_ratio = step_size / time_radius if time_radius > 0.0 else float("nan")
    state_sup_bound = max(
        position_upper_bound + position_radius,
        speed_upper_bound + velocity_radius,
    )
    if retained_order >= 0 and 0.0 < step_ratio < 1.0:
        tail_bound = float(
            np.nextafter(
                state_sup_bound
                * step_ratio ** (retained_order + 1)
                / (1.0 - step_ratio),
                np.inf,
            )
        )
    else:
        tail_bound = float("inf")
    return UniformCollisionFreeTaylorRecurrenceCertificate(
        masses=tuple(float(mass) for mass in masses),
        pair_distance_lower_bound=pair_distance_lower_bound,
        position_upper_bound=position_upper_bound,
        speed_upper_bound=speed_upper_bound,
        retained_order=retained_order,
        step_size=step_size,
        position_radius=position_radius,
        velocity_radius=velocity_radius,
        lower_pair_distance_on_tube=lower_pair_distance_on_tube,
        acceleration_bound=acceleration_bound,
        time_radius=float(time_radius),
        step_ratio=float(step_ratio),
        state_sup_bound=float(state_sup_bound),
        tail_bound=tail_bound,
    )


def certify_uniform_collision_free_ordinary_chart_ledgers(
    positions: Array,
    velocities: Array,
    masses: Array,
    recurrence_certificate: UniformCollisionFreeTaylorRecurrenceCertificate,
) -> UniformCollisionFreeOrdinaryChartLedgerCertificate:
    """Certify ordinary residual, invariant, projection, transition, and tail ledgers."""

    from .hybrid import certify_ordinary_interval_taylor_equations

    if not isinstance(recurrence_certificate, UniformCollisionFreeTaylorRecurrenceCertificate):
        raise TypeError("recurrence_certificate must be constructor-derived")
    positions, velocities, masses = _validate_three_body_initial_data(
        positions,
        velocities,
        masses,
    )
    order = int(recurrence_certificate.retained_order)
    if order < 1:
        raise ValueError("retained_order must be positive to certify chart ledgers")
    interval_solution = construct_interval_taylor_solution_from_intervals(
        positions,
        velocities,
        masses,
        order=order,
    )
    physical_time = np.empty(order + 1, dtype=object)
    for degree in range(order + 1):
        physical_time[degree] = 0.0
    physical_time[1] = 1.0
    residual = certify_ordinary_interval_taylor_equations(
        interval_solution,
        coefficient_count=order,
    )
    center = certify_interval_center_of_mass_motion(
        interval_solution.position,
        interval_solution.velocity,
        physical_time,
        masses,
        coefficient_count=order,
    )
    momentum = certify_interval_linear_momentum_conservation(
        interval_solution.velocity,
        masses,
        coefficient_count=order,
    )
    angular = certify_interval_centered_angular_momentum_conservation(
        interval_solution.position,
        interval_solution.velocity,
        masses,
        coefficient_count=order,
    )
    energy = certify_interval_total_energy_conservation(
        interval_solution.position,
        interval_solution.velocity,
        masses,
        coefficient_count=order,
    )
    return UniformCollisionFreeOrdinaryChartLedgerCertificate(
        residual_certificate=residual,
        center_of_mass_certificate=center,
        linear_momentum_certificate=momentum,
        angular_momentum_certificate=angular,
        energy_certificate=energy,
        recurrence_certificate=recurrence_certificate,
    )


def choose_step_size(
    positions: Array,
    velocities: Array,
    masses: Array,
    remaining_time: float,
    *,
    max_step: float,
    safety: float = 0.08,
    distance_margin: float | None = None,
) -> float:
    """Choose a conservative chart size from collision margin and local speed."""

    if remaining_time == 0.0:
        return 0.0
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    if safety <= 0.0:
        raise ValueError("safety must be positive")

    if distance_margin is None:
        min_distance = float(np.min(pairwise_distances(positions)))
    else:
        min_distance = float(distance_margin)
        if min_distance < 0.0:
            raise ValueError("distance_margin cannot be negative")
    max_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
    acceleration = acceleration_coefficients(
        np.asarray(positions, dtype=float)[None, :, :],
        np.asarray(masses, dtype=float),
        0,
    )[0]
    max_acceleration = float(np.max(np.linalg.norm(acceleration, axis=1)))

    speed_limited = min_distance / max(max_speed, 1e-15)
    acceleration_limited = np.sqrt(min_distance / max(max_acceleration, 1e-15))
    candidate = min(safety * min(speed_limited, acceleration_limited), max_step)
    magnitude = min(abs(remaining_time), max(candidate, np.finfo(float).eps))
    return float(np.copysign(magnitude, remaining_time))


def truncation_indicator(series: TaylorSolution, step_size: float) -> float:
    """A computable local truncation proxy from the highest retained term."""

    power = abs(step_size) ** series.order
    position_tail = np.linalg.norm(series.position[-1].reshape(-1), ord=np.inf) * power
    velocity_tail = np.linalg.norm(series.velocity[-1].reshape(-1), ord=np.inf) * power
    return float(max(position_tail, velocity_tail))


def continue_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    t_final: float,
    *,
    order: int = 18,
    max_step: float = 0.02,
    safety: float = 0.08,
) -> ContinuedSolution:
    """Continue a general non-collision solution by chaining Taylor charts."""

    if order < 2:
        raise ValueError("order must be at least 2")
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    body_count, dimension = positions.shape
    if body_count != 3:
        raise ValueError("this continuation is specialized to three bodies")

    current_time = 0.0
    current_positions = positions.copy()
    current_velocities = velocities.copy()
    times = [current_time]
    states = [np.concatenate([current_positions.reshape(-1), current_velocities.reshape(-1)])]
    steps: list[TaylorStep] = []

    while abs(t_final - current_time) > 10.0 * np.finfo(float).eps:
        remaining = t_final - current_time
        step_size = choose_step_size(
            current_positions,
            current_velocities,
            masses,
            remaining,
            max_step=max_step,
            safety=safety,
        )
        series = construct_taylor_solution(current_positions, current_velocities, masses, order=order)
        indicator = truncation_indicator(series, step_size)
        step = TaylorStep(
            start_time=current_time,
            step_size=step_size,
            series=series,
            truncation_indicator=indicator,
            min_pair_distance=float(np.min(pairwise_distances(current_positions))),
        )
        current_positions = series.positions_at(step_size)
        current_velocities = series.velocities_at(step_size)
        current_time = step.end_time
        steps.append(step)
        times.append(current_time)
        states.append(np.concatenate([current_positions.reshape(-1), current_velocities.reshape(-1)]))

    return ContinuedSolution(
        masses=masses,
        times=np.array(times, dtype=float),
        states=np.vstack(states),
        steps=tuple(steps),
    )
