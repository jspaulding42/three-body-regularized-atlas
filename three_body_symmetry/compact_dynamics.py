"""Local three-body Taylor charts in compactified physical time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compact_time import compact_parameter_from_physical_time, physical_time_from_compact_parameter
from .error_budget import (
    ErrorBudgetStep,
    PropagatedErrorBudget,
    PropagatedIntervalEnclosure,
    PropagatedIntervalStep,
    inflate_state_interval,
    newtonian_planar_lipschitz_bound,
)
from .series import acceleration_coefficients, integrate_reference
from .tail_bounds import TailBoundCertificate, guarded_tail_certificate


Array = np.ndarray


@dataclass(frozen=True)
class CompactifiedTaylorSolution:
    """Taylor chart for the Newtonian state as a function of compact time ``u``."""

    position: Array
    velocity: Array
    physical_time: Array
    masses: Array
    time_rate: float
    center: float = 0.0

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
    def analytic_radius(self) -> float:
        return float(1.0 - abs(self.center))

    @property
    def initial_physical_time(self) -> float:
        return float(self.physical_time[0])

    def positions_at_u(self, compact_parameter: float) -> Array:
        return _evaluate(self.position, self.delta_from_compact_parameter(compact_parameter))

    def velocities_at_u(self, compact_parameter: float) -> Array:
        return _evaluate(self.velocity, self.delta_from_compact_parameter(compact_parameter))

    def physical_time_at_u(self, compact_parameter: float) -> float:
        return float(_evaluate(self.physical_time[:, None], self.delta_from_compact_parameter(compact_parameter))[0])

    def physical_time_delta_at_u(self, compact_parameter: float) -> float:
        return float(self.physical_time_at_u(compact_parameter) - self.initial_physical_time)

    def state_at_u(self, compact_parameter: float) -> Array:
        return np.concatenate(
            [
                self.positions_at_u(compact_parameter).reshape(-1),
                self.velocities_at_u(compact_parameter).reshape(-1),
            ]
        )

    def delta_from_compact_parameter(self, compact_parameter: float) -> float:
        compact_parameter = float(compact_parameter)
        if not -1.0 < compact_parameter < 1.0:
            raise ValueError("compact_parameter must lie strictly between -1 and 1")
        delta = compact_parameter - self.center
        if abs(delta) >= self.analytic_radius:
            raise ValueError("compact_parameter must lie inside this chart's compact-time convergence disk")
        return float(delta)


@dataclass(frozen=True)
class CompactifiedDynamicsResidualCertificate:
    """Coefficient residual certificate for compact-time Newtonian dynamics."""

    coefficient_count: int
    position_residual: Array
    velocity_residual: Array
    physical_time_residual: Array
    tolerance: float = 1e-11

    @property
    def max_residual(self) -> float:
        return float(
            max(
                np.max(np.abs(self.position_residual)),
                np.max(np.abs(self.velocity_residual)),
                np.max(np.abs(self.physical_time_residual)),
            )
        )

    @property
    def certified(self) -> bool:
        return bool(self.max_residual <= self.tolerance)


@dataclass(frozen=True)
class CompactifiedAtlasStep:
    """One compact-time Taylor chart used in an atlas continuation."""

    start_compact_parameter: float
    end_compact_parameter: float
    chart: CompactifiedTaylorSolution
    residual_certificate: CompactifiedDynamicsResidualCertificate
    truncation_indicator: float
    tail_certificate: TailBoundCertificate | None = None

    @property
    def compact_step(self) -> float:
        return float(self.end_compact_parameter - self.start_compact_parameter)

    @property
    def physical_time_step(self) -> float:
        return self.chart.physical_time_delta_at_u(self.end_compact_parameter)

    @property
    def certified(self) -> bool:
        return bool(
            self.residual_certificate.certified
            and self.chart.center == self.start_compact_parameter
            and abs(self.compact_step) < self.chart.analytic_radius
        )

    @property
    def tail_certified(self) -> bool:
        return bool(self.tail_certificate is not None and self.tail_certificate.is_nontrivial)

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and self.tail_certified)


@dataclass(frozen=True)
class CompactifiedAtlasSolution:
    """Projected states produced by compact-time Taylor atlas continuation."""

    masses: Array
    time_rate: float
    compact_parameters: Array
    states: Array
    steps: tuple[CompactifiedAtlasStep, ...]

    @property
    def final_state(self) -> Array:
        return self.states[-1]

    @property
    def certified(self) -> bool:
        return bool(all(step.certified for step in self.steps))

    @property
    def tail_certified(self) -> bool:
        return bool(self.steps and all(step.tail_certified for step in self.steps))

    @property
    def chain_certified(self) -> bool:
        return _compactified_atlas_chain_certified(
            self.time_rate,
            self.masses,
            self.compact_parameters,
            self.states,
            self.steps,
        )

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and self.tail_certified and self.chain_certified)

    @property
    def total_physical_time_delta(self) -> float:
        start = self.compact_parameters[0]
        end = self.compact_parameters[-1]
        return float(
            physical_time_from_compact_parameter(end, rate=self.time_rate)
            - physical_time_from_compact_parameter(start, rate=self.time_rate)
        )

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

    def propagated_error_budget(self) -> PropagatedErrorBudget:
        return propagate_compact_atlas_error_budget(self)

    def propagated_interval_enclosure(self) -> PropagatedIntervalEnclosure:
        return propagate_compact_atlas_interval_enclosure(self)


def construct_compactified_taylor_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
    time_rate: float = 1.0,
    center: float = 0.0,
) -> CompactifiedTaylorSolution:
    """Construct a local compact-time Taylor chart around ``u = center``.

    The compact-time inverse is ``t(u) = atanh(u) / time_rate``. The
    position and velocity inputs are the physical state at ``u = center``.
    The chart solves the lifted equations in ``delta = u - center``:
    ``dq/du = v dt/du`` and ``dv/du = a(q) dt/du``.
    """

    if order < 1:
        raise ValueError("order must be at least 1")
    positions, velocities, masses = _validate_initial_data(positions, velocities, masses)
    time_rate = _validate_time_rate(time_rate)
    center = _validate_center(center)
    body_count, dimension = positions.shape

    q = np.zeros((order + 1, body_count, dimension), dtype=float)
    v = np.zeros_like(q)
    physical_time = np.zeros(order + 1, dtype=float)
    q[0] = positions
    v[0] = velocities
    physical_time[0] = physical_time_from_compact_parameter(center, rate=time_rate)
    factor = compact_time_factor_coefficients(order, center=center, time_rate=time_rate)

    for degree in range(order):
        acceleration = acceleration_coefficients(q, masses, degree)
        q_rhs = _series_vector_product(factor[: degree + 1], v, degree)
        v_rhs = _series_vector_product(factor[: degree + 1], acceleration, degree)
        scale = 1.0 / float(degree + 1)
        q[degree + 1] = q_rhs[degree] * scale
        v[degree + 1] = v_rhs[degree] * scale
        physical_time[degree + 1] = factor[degree] * scale

    return CompactifiedTaylorSolution(
        position=q,
        velocity=v,
        physical_time=physical_time,
        masses=masses,
        time_rate=time_rate,
        center=center,
    )


def continue_compactified_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    target_compact_parameter: float,
    *,
    initial_compact_parameter: float = 0.0,
    order: int = 16,
    time_rate: float = 1.0,
    max_compact_step: float = 0.04,
    radius_fraction: float = 0.25,
    residual_tolerance: float = 1e-8,
    guard_order: int = 0,
) -> CompactifiedAtlasSolution:
    """Continue a non-collision state by chaining compact-time Taylor charts."""

    if order < 1:
        raise ValueError("order must be at least 1")
    time_rate = _validate_time_rate(time_rate)
    current_u = _validate_center(initial_compact_parameter)
    target_u = _validate_center(target_compact_parameter)
    if max_compact_step <= 0.0:
        raise ValueError("max_compact_step must be positive")
    if not 0.0 < radius_fraction < 1.0:
        raise ValueError("radius_fraction must lie strictly between 0 and 1")
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive")
    if guard_order < 0:
        raise ValueError("guard_order cannot be negative")
    current_positions, current_velocities, masses = _validate_initial_data(positions, velocities, masses)

    compact_parameters = [current_u]
    states = [np.concatenate([current_positions.reshape(-1), current_velocities.reshape(-1)])]
    steps: list[CompactifiedAtlasStep] = []

    while abs(target_u - current_u) > 10.0 * np.finfo(float).eps:
        compact_step = choose_compact_step_size(
            current_u,
            target_u,
            max_compact_step=max_compact_step,
            radius_fraction=radius_fraction,
        )
        next_u = current_u + compact_step
        computed_order = order + guard_order
        computed_chart = construct_compactified_taylor_solution(
            current_positions,
            current_velocities,
            masses,
            order=computed_order,
            time_rate=time_rate,
            center=current_u,
        )
        chart = _truncate_compactified_chart(computed_chart, order)
        residual = certify_compactified_dynamics_equations(chart, tolerance=residual_tolerance)
        indicator = compact_truncation_indicator(chart, next_u)
        tail_certificate = (
            compact_guarded_tail_certificate(computed_chart, retained_order=order, compact_parameter=next_u)
            if guard_order > 0
            else None
        )
        next_state = chart.state_at_u(next_u)
        body_count, dimension = current_positions.shape
        current_positions = next_state[: body_count * dimension].reshape(body_count, dimension)
        current_velocities = next_state[body_count * dimension :].reshape(body_count, dimension)
        steps.append(
            CompactifiedAtlasStep(
                start_compact_parameter=current_u,
                end_compact_parameter=next_u,
                chart=chart,
                residual_certificate=residual,
                truncation_indicator=indicator,
                tail_certificate=tail_certificate,
            )
        )
        current_u = next_u
        compact_parameters.append(current_u)
        states.append(next_state)

    return CompactifiedAtlasSolution(
        masses=masses,
        time_rate=time_rate,
        compact_parameters=np.array(compact_parameters, dtype=float),
        states=np.vstack(states),
        steps=tuple(steps),
    )


def continue_compactified_solution_to_time(
    positions: Array,
    velocities: Array,
    masses: Array,
    target_physical_time: float,
    *,
    order: int = 16,
    time_rate: float = 1.0,
    max_compact_step: float = 0.04,
    radius_fraction: float = 0.25,
    residual_tolerance: float = 1e-8,
    guard_order: int = 0,
) -> CompactifiedAtlasSolution:
    """Continue an initial-value problem from ``t = 0`` to a physical target time.

    The wrapper is the target-time evaluator for the compact atlas: it maps the
    requested physical time to ``u = tanh(time_rate * t)`` and then chains the
    existing proof-carrying compact-time charts from ``u = 0`` to that target.
    """

    target_physical_time = float(target_physical_time)
    if not np.isfinite(target_physical_time):
        raise ValueError("target_physical_time must be finite")
    time_rate = _validate_time_rate(time_rate)
    target_compact_parameter = compact_parameter_from_physical_time(target_physical_time, rate=time_rate)
    if not -1.0 < target_compact_parameter < 1.0:
        raise ValueError("target_physical_time is too large for a finite floating compact-time target")

    return continue_compactified_solution(
        positions,
        velocities,
        masses,
        target_compact_parameter,
        initial_compact_parameter=0.0,
        order=order,
        time_rate=time_rate,
        max_compact_step=max_compact_step,
        radius_fraction=radius_fraction,
        residual_tolerance=residual_tolerance,
        guard_order=guard_order,
    )


def compact_guarded_tail_certificate(
    chart: CompactifiedTaylorSolution,
    *,
    retained_order: int,
    compact_parameter: float,
) -> TailBoundCertificate:
    """Build a guard-term tail certificate for a compact-time chart evaluation."""

    delta = chart.delta_from_compact_parameter(compact_parameter)
    arrays = _compact_solution_arrays(chart)
    certificate = guarded_tail_certificate(arrays, retained_order=retained_order, step_size=delta)
    scale = max(1.0, *(float(np.max(np.abs(array))) for array in arrays))
    roundoff_slack = 64.0 * np.finfo(float).eps * scale
    return TailBoundCertificate(
        retained_order=certificate.retained_order,
        computed_order=certificate.computed_order,
        step_size=certificate.step_size,
        first_omitted_term=certificate.first_omitted_term,
        observed_tail=certificate.observed_tail,
        ratio_bound=certificate.ratio_bound,
        tail_bound=float(np.nextafter(certificate.tail_bound + roundoff_slack, np.inf)),
        coefficient_source=certificate.coefficient_source,
    )


def propagate_compact_atlas_error_budget(atlas: CompactifiedAtlasSolution) -> PropagatedErrorBudget:
    """Propagate compact-atlas local tail bounds through a Gronwall ledger."""

    outgoing = 0.0
    budget_steps: list[ErrorBudgetStep] = []
    for step in atlas.steps:
        local_tail = 0.0 if step.tail_certificate is None else step.tail_certificate.tail_bound
        min_distance = min_pair_distance(step.chart.position[0])
        lipschitz = newtonian_planar_lipschitz_bound(atlas.masses, min_distance)
        incoming = outgoing
        growth = np.exp(min(lipschitz * abs(step.physical_time_step), 700.0))
        outgoing = growth * (incoming + local_tail)
        budget_steps.append(
            ErrorBudgetStep(
                chart="compact_time",
                start_time=step.chart.initial_physical_time,
                physical_step=step.physical_time_step,
                local_tail_bound=float(local_tail),
                lipschitz_bound=float(lipschitz),
                incoming_bound=float(incoming),
                outgoing_bound=float(outgoing),
            )
        )
    return PropagatedErrorBudget(steps=tuple(budget_steps))


def propagate_compact_atlas_interval_enclosure(atlas: CompactifiedAtlasSolution) -> PropagatedIntervalEnclosure:
    """Materialize compact-atlas propagated tail bounds as coordinate boxes."""

    outgoing = 0.0
    interval_steps: list[PropagatedIntervalStep] = []
    for index, step in enumerate(atlas.steps):
        local_tail = 0.0 if step.tail_certificate is None else step.tail_certificate.tail_bound
        min_distance = min_pair_distance(step.chart.position[0])
        lipschitz = newtonian_planar_lipschitz_bound(atlas.masses, min_distance)
        incoming = outgoing
        growth = np.exp(min(lipschitz * abs(step.physical_time_step), 700.0))
        outgoing = growth * (incoming + local_tail)
        base_start = _point_state_interval(atlas.states[index])
        base_end = _point_state_interval(atlas.states[index + 1])
        interval_steps.append(
            PropagatedIntervalStep(
                chart="compact_time",
                start_time=step.chart.initial_physical_time,
                physical_step=step.physical_time_step,
                local_tail_bound=float(local_tail),
                lipschitz_bound=float(lipschitz),
                incoming_radius=float(incoming),
                outgoing_radius=float(outgoing),
                base_start_state_interval=base_start,
                base_end_state_interval=base_end,
                start_state_interval=inflate_state_interval(base_start, incoming),
                end_state_interval=inflate_state_interval(base_end, outgoing),
            )
        )
    return PropagatedIntervalEnclosure(steps=tuple(interval_steps))


def choose_compact_step_size(
    current_compact_parameter: float,
    target_compact_parameter: float,
    *,
    max_compact_step: float,
    radius_fraction: float = 0.25,
) -> float:
    """Choose a compact-parameter step that stays inside the local analytic disk."""

    current = _validate_center(current_compact_parameter)
    target = _validate_center(target_compact_parameter)
    if max_compact_step <= 0.0:
        raise ValueError("max_compact_step must be positive")
    if not 0.0 < radius_fraction < 1.0:
        raise ValueError("radius_fraction must lie strictly between 0 and 1")
    remaining = target - current
    if remaining == 0.0:
        return 0.0
    radius_limited = radius_fraction * (1.0 - abs(current))
    magnitude = min(abs(remaining), max_compact_step, radius_limited)
    return float(np.copysign(max(magnitude, np.finfo(float).eps), remaining))


def compact_truncation_indicator(chart: CompactifiedTaylorSolution, compact_parameter: float) -> float:
    """A local truncation proxy from the highest retained compact-time term."""

    delta = chart.delta_from_compact_parameter(compact_parameter)
    power = abs(delta) ** chart.order
    position_tail = np.linalg.norm(chart.position[-1].reshape(-1), ord=np.inf) * power
    velocity_tail = np.linalg.norm(chart.velocity[-1].reshape(-1), ord=np.inf) * power
    physical_time_tail = abs(chart.physical_time[-1]) * power
    return float(max(position_tail, velocity_tail, physical_time_tail))


def min_pair_distance(positions: Array) -> float:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("positions must have shape (3, dimension)")
    return float(
        min(
            np.linalg.norm(positions[first] - positions[second])
            for first in range(3)
            for second in range(first + 1, 3)
        )
    )


def _compact_solution_arrays(chart: CompactifiedTaylorSolution) -> list[Array]:
    return [chart.position, chart.velocity, chart.physical_time[:, None]]


def _point_state_interval(state: Array) -> tuple[tuple[float, float], ...]:
    values = np.asarray(state, dtype=float).reshape(-1)
    return tuple(
        (
            float(np.nextafter(value, -np.inf)),
            float(np.nextafter(value, np.inf)),
        )
        for value in values
    )


def _truncate_compactified_chart(chart: CompactifiedTaylorSolution, order: int) -> CompactifiedTaylorSolution:
    if order < 0 or order > chart.order:
        raise ValueError("order must be inside the chart coefficient range")
    return CompactifiedTaylorSolution(
        position=chart.position[: order + 1].copy(),
        velocity=chart.velocity[: order + 1].copy(),
        physical_time=chart.physical_time[: order + 1].copy(),
        masses=chart.masses.copy(),
        time_rate=chart.time_rate,
        center=chart.center,
    )


def _compactified_float_boundaries_match(left: float, right: float) -> bool:
    left = float(left)
    right = float(right)
    if not (np.isfinite(left) and np.isfinite(right)):
        return False
    tolerance = 1.0e-10 * max(1.0, abs(left), abs(right))
    return bool(abs(left - right) <= tolerance)


def _compactified_float_arrays_match(left: Array, right: Array) -> bool:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        return False
    if left_array.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(left_array))), float(np.max(np.abs(right_array))))
    return bool(
        np.all(np.isfinite(left_array))
        and np.all(np.isfinite(right_array))
        and np.all(np.abs(left_array - right_array) <= 1.0e-10 * scale)
    )


def _compactified_atlas_chain_certified(
    time_rate: float,
    masses: Array,
    compact_parameters: Array,
    states: Array,
    steps: tuple[CompactifiedAtlasStep, ...],
) -> bool:
    step_count = len(steps)
    compact_parameters = np.asarray(compact_parameters, dtype=float)
    states = np.asarray(states, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if (
        compact_parameters.shape != (step_count + 1,)
        or states.ndim != 2
        or states.shape[0] != step_count + 1
    ):
        return False
    if not (
        np.isfinite(float(time_rate))
        and np.all(np.isfinite(compact_parameters))
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(masses))
    ):
        return False
    if step_count == 0:
        return True
    for index, step in enumerate(steps):
        if not (
            step.chart.time_rate == float(time_rate)
            and np.array_equal(step.chart.masses, masses)
            and _compactified_float_boundaries_match(
                compact_parameters[index],
                step.start_compact_parameter,
            )
            and _compactified_float_boundaries_match(
                compact_parameters[index + 1],
                step.end_compact_parameter,
            )
            and _compactified_float_boundaries_match(
                step.chart.initial_physical_time,
                physical_time_from_compact_parameter(step.start_compact_parameter, rate=time_rate),
            )
            and _compactified_float_arrays_match(
                states[index],
                step.chart.state_at_u(step.start_compact_parameter),
            )
            and _compactified_float_arrays_match(
                states[index + 1],
                step.chart.state_at_u(step.end_compact_parameter),
            )
        ):
            return False
    return True


def compact_time_factor_coefficients(order: int, *, center: float = 0.0, time_rate: float = 1.0) -> Array:
    """Return coefficients of ``dt/delta`` around ``u = center + delta``."""

    if order < 0:
        raise ValueError("order cannot be negative")
    time_rate = _validate_time_rate(time_rate)
    center = _validate_center(center)
    coefficients = np.zeros(order + 1, dtype=float)
    p0 = 1.0 - center * center
    p1 = -2.0 * center
    p2 = -1.0
    coefficients[0] = 1.0 / (time_rate * p0)
    for degree in range(1, order + 1):
        previous_1 = coefficients[degree - 1]
        previous_2 = coefficients[degree - 2] if degree - 2 >= 0 else 0.0
        coefficients[degree] = -(p1 * previous_1 + p2 * previous_2) / p0
    return coefficients


def certify_compactified_dynamics_equations(
    chart: CompactifiedTaylorSolution,
    *,
    coefficient_count: int | None = None,
    tolerance: float = 1e-11,
) -> CompactifiedDynamicsResidualCertificate:
    """Certify coefficient residuals for compact-time Newtonian dynamics."""

    if coefficient_count is None:
        coefficient_count = chart.order
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > chart.order:
        raise ValueError("coefficient_count cannot exceed chart.order")

    max_degree = coefficient_count - 1
    factor = compact_time_factor_coefficients(max_degree, center=chart.center, time_rate=chart.time_rate)
    acceleration = acceleration_coefficients(chart.position, chart.masses, max_degree)
    position_rhs = _series_vector_product(factor, chart.velocity, max_degree)
    velocity_rhs = _series_vector_product(factor, acceleration, max_degree)

    position_residual = np.zeros_like(position_rhs)
    velocity_residual = np.zeros_like(velocity_rhs)
    physical_time_residual = np.zeros(coefficient_count, dtype=float)
    for degree in range(coefficient_count):
        scale = float(degree + 1)
        position_residual[degree] = scale * chart.position[degree + 1] - position_rhs[degree]
        velocity_residual[degree] = scale * chart.velocity[degree + 1] - velocity_rhs[degree]
        physical_time_residual[degree] = scale * chart.physical_time[degree + 1] - factor[degree]

    return CompactifiedDynamicsResidualCertificate(
        coefficient_count=coefficient_count,
        position_residual=position_residual,
        velocity_residual=velocity_residual,
        physical_time_residual=physical_time_residual,
        tolerance=float(tolerance),
    )


def reference_state_at_compact_time(chart: CompactifiedTaylorSolution, compact_parameter: float) -> Array:
    """Numerically integrate from the chart center to ``compact_parameter``."""

    physical_time_delta = chart.physical_time_delta_at_u(compact_parameter)
    return integrate_reference(chart.position[0], chart.velocity[0], chart.masses, physical_time_delta)


def _series_vector_product(scalar: Array, vector: Array, max_degree: int) -> Array:
    out = np.zeros_like(vector[: max_degree + 1])
    for degree in range(max_degree + 1):
        for index in range(degree + 1):
            out[degree] += scalar[index] * vector[degree - index]
    return out


def _evaluate(coefficients: Array, value: float) -> Array:
    out = np.zeros(coefficients.shape[1:], dtype=float)
    for coefficient in coefficients[::-1]:
        out = out * float(value) + coefficient
    return out


def _validate_time_rate(time_rate: float) -> float:
    time_rate = float(time_rate)
    if not np.isfinite(time_rate) or time_rate <= 0.0:
        raise ValueError("time_rate must be a positive finite value")
    return time_rate


def _validate_center(center: float) -> float:
    center = float(center)
    if not np.isfinite(center) or not -1.0 < center < 1.0:
        raise ValueError("center must lie strictly between -1 and 1")
    return center


def _validate_initial_data(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 2 or positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shape (body_count, dimension)")
    if positions.shape[0] != 3 or masses.shape != (3,):
        raise ValueError("expected three bodies and three masses")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive")
    for first in range(3):
        for second in range(first + 1, 3):
            if np.linalg.norm(positions[first] - positions[second]) == 0.0:
                raise ValueError("initial data must be collision-free")
    return positions, velocities, masses
