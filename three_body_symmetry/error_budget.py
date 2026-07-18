"""Numerical propagated error-budget estimates for hybrid continuation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binary_chart import (
    IntervalRegularizedBinaryCollisionChartState,
    planar_interval_to_regularized_binary_collision_chart,
    planar_interval_to_regularized_binary_collision_chart_atlas,
)
from .binary_series import construct_interval_regularized_binary_taylor_solution_from_intervals
from .intervals import FloatInterval, interval_array_as_tuples, interval_array_series_eval, zero_interval
from .series import construct_interval_taylor_solution_from_intervals, construct_taylor_solution
from .tail_bounds import (
    ordinary_interval_cauchy_majorant_tail_certificate,
    regularized_binary_interval_atlas_cauchy_majorant_tail_certificate,
    regularized_binary_interval_cauchy_majorant_tail_certificate,
    sundman_interval_cauchy_majorant_tail_certificate,
)


Array = np.ndarray


@dataclass(frozen=True)
class ErrorBudgetStep:
    chart: str
    start_time: float
    physical_step: float
    local_tail_bound: float
    lipschitz_bound: float
    incoming_bound: float
    outgoing_bound: float


@dataclass(frozen=True)
class PropagatedErrorBudget:
    steps: tuple[ErrorBudgetStep, ...]

    @property
    def final_bound(self) -> float:
        return self.steps[-1].outgoing_bound if self.steps else 0.0

    @property
    def local_tail_bound(self) -> float:
        return float(sum(step.local_tail_bound for step in self.steps))

    @property
    def max_lipschitz_bound(self) -> float:
        return float(max((step.lipschitz_bound for step in self.steps), default=0.0))


@dataclass(frozen=True)
class PropagatedIntervalStep:
    chart: str
    start_time: float
    physical_step: float
    local_tail_bound: float
    lipschitz_bound: float
    incoming_radius: float
    outgoing_radius: float
    base_start_state_interval: tuple[tuple[float, float], ...]
    base_end_state_interval: tuple[tuple[float, float], ...]
    start_state_interval: tuple[tuple[float, float], ...]
    end_state_interval: tuple[tuple[float, float], ...]

    def start_state_contains(self, state: Array) -> bool:
        return interval_contains_state(self.start_state_interval, state)

    def end_state_contains(self, state: Array) -> bool:
        return interval_contains_state(self.end_state_interval, state)


@dataclass(frozen=True)
class PropagatedIntervalEnclosure:
    steps: tuple[PropagatedIntervalStep, ...]

    @property
    def final_state_interval(self) -> tuple[tuple[float, float], ...] | None:
        return self.steps[-1].end_state_interval if self.steps else None

    @property
    def final_radius(self) -> float:
        return self.steps[-1].outgoing_radius if self.steps else 0.0

    @property
    def local_tail_bound(self) -> float:
        return float(sum(step.local_tail_bound for step in self.steps))

    @property
    def max_lipschitz_bound(self) -> float:
        return float(max((step.lipschitz_bound for step in self.steps), default=0.0))

    def final_state_contains(self, state: Array) -> bool:
        if self.final_state_interval is None:
            return False
        return interval_contains_state(self.final_state_interval, state)


@dataclass(frozen=True)
class OrdinarySetPropagationStep:
    start_time: float
    physical_step: float
    retained_order: int
    start_state_interval: tuple[tuple[float, float], ...]
    truncated_end_state_interval: tuple[tuple[float, float], ...]
    end_state_interval: tuple[tuple[float, float], ...]
    tail_bound: float
    tail_ratio_bound: float
    tail_coefficient_source: str
    start_state_interval_union: tuple[tuple[tuple[float, float], ...], ...] = ()
    truncated_end_state_interval_union: tuple[tuple[tuple[float, float], ...], ...] = ()
    end_state_interval_union: tuple[tuple[tuple[float, float], ...], ...] = ()
    event: str | None = None
    event_time_interval: tuple[float, float] | None = None
    event_time_interval_union: tuple[tuple[float, float], ...] = ()
    event_parameter_interval: tuple[float, float] | None = None
    chart: str = "ordinary"
    pair: tuple[int, int] | None = None
    parameter_step: float | None = None
    binary_atlas_chart_count: int = 0
    ordinary_substeps: int = 1

    @property
    def union_member_count(self) -> int:
        return len(self.end_state_interval_union)

    @property
    def certified(self) -> bool:
        finite_parameter = (
            self.parameter_step is None
            or np.isfinite(float(self.parameter_step))
        )
        event_interval_certified = True
        if self.event_time_interval is not None:
            event_interval_certified = _interval_tuple_certified(self.event_time_interval)
        if self.event_parameter_interval is not None:
            event_interval_certified = (
                event_interval_certified
                and _interval_tuple_certified(self.event_parameter_interval)
            )
        return bool(
            _state_interval_certified(self.start_state_interval)
            and _state_interval_certified(self.truncated_end_state_interval)
            and _state_interval_certified(self.end_state_interval)
            and self.start_state_interval_union
            and self.truncated_end_state_interval_union
            and self.end_state_interval_union
            and all(_state_interval_certified(member) for member in self.start_state_interval_union)
            and all(
                _state_interval_certified(member)
                for member in self.truncated_end_state_interval_union
            )
            and all(_state_interval_certified(member) for member in self.end_state_interval_union)
            and all(
                state_interval_subset(member, self.start_state_interval)
                for member in self.start_state_interval_union
            )
            and all(
                state_interval_subset(member, self.truncated_end_state_interval)
                for member in self.truncated_end_state_interval_union
            )
            and all(
                state_interval_subset(member, self.end_state_interval)
                for member in self.end_state_interval_union
            )
            and np.isfinite(self.start_time)
            and np.isfinite(self.physical_step)
            and self.retained_order > 0
            and np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
            and np.isfinite(self.tail_ratio_bound)
            and self.tail_ratio_bound < 1.0
            and self.tail_coefficient_source
            and self.chart in {"ordinary", "binary"}
            and finite_parameter
            and event_interval_certified
            and self.binary_atlas_chart_count >= 0
            and self.ordinary_substeps > 0
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    def start_state_contains(self, state: Array) -> bool:
        return interval_contains_state(self.start_state_interval, state)

    def truncated_end_state_contains(self, state: Array) -> bool:
        return interval_contains_state(self.truncated_end_state_interval, state)

    def end_state_contains(self, state: Array) -> bool:
        return interval_contains_state(self.end_state_interval, state)


@dataclass(frozen=True)
class OrdinarySetPropagatedEnclosure:
    steps: tuple[OrdinarySetPropagationStep, ...]

    @property
    def final_state_interval(self) -> tuple[tuple[float, float], ...] | None:
        return self.steps[-1].end_state_interval if self.steps else None

    @property
    def local_tail_bound(self) -> float:
        return float(sum(step.tail_bound for step in self.steps))

    @property
    def max_step_tail_bound(self) -> float:
        return float(max((step.tail_bound for step in self.steps), default=0.0))

    @property
    def certified_step_count(self) -> int:
        return sum(step.certified for step in self.steps)

    @property
    def chain_certified(self) -> bool:
        return bool(
            self.steps
            and all(
                _ordinary_set_propagation_steps_chain(left, right)
                for left, right in zip(self.steps, self.steps[1:])
            )
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.steps
            and all(step.proof_certified for step in self.steps)
            and self.chain_certified
        )

    def final_state_contains(self, state: Array) -> bool:
        if self.final_state_interval is None:
            return False
        return interval_contains_state(self.final_state_interval, state)


@dataclass(frozen=True)
class LohnerStateSet:
    """Center-plus-linear-shape set with an interval remainder.

    It represents

    ``center + linear_shape @ xi + remainder``, where each component of
    ``xi`` lies in ``[-1, 1]`` and each remainder coordinate lies in its listed
    interval.  Affine maps can act on this form without first hulling the set
    into an interval box, preserving correlations needed by later proof
    constructors.
    """

    center: Array
    linear_shape: Array
    remainder_box: tuple[tuple[float, float], ...]
    source: str = "lohner_state_set"

    @property
    def dimension(self) -> int:
        return int(self.center.shape[0])

    @property
    def shape_dimension(self) -> int:
        return int(self.linear_shape.shape[1])

    @property
    def certified(self) -> bool:
        return bool(
            self.center.ndim == 1
            and self.linear_shape.ndim == 2
            and self.linear_shape.shape[0] == self.center.shape[0]
            and len(self.remainder_box) == self.center.shape[0]
            and np.all(np.isfinite(self.center))
            and np.all(np.isfinite(self.linear_shape))
            and all(
                np.isfinite(lower) and np.isfinite(upper) and lower <= upper
                for lower, upper in self.remainder_box
            )
        )

    @property
    def interval_hull(self) -> tuple[tuple[float, float], ...]:
        if not self.certified:
            raise ValueError("cannot hull an uncertified Lohner state set")
        shape_radii = np.sum(np.abs(self.linear_shape), axis=1)
        hull = []
        for index, radius in enumerate(shape_radii):
            lower, upper = self.remainder_box[index]
            hull.append(
                (
                    float(np.nextafter(self.center[index] - radius + lower, -np.inf)),
                    float(np.nextafter(self.center[index] + radius + upper, np.inf)),
                )
            )
        return tuple(hull)

    @property
    def max_width(self) -> float:
        return max((upper - lower for lower, upper in self.interval_hull), default=0.0)

    def contains_state(self, state: Array) -> bool:
        return interval_contains_state(self.interval_hull, state)

    def linear_observable_interval(self, weights: Array) -> tuple[float, float]:
        """Enclose a linear observable over the represented set."""

        if not self.certified:
            raise ValueError("cannot evaluate an uncertified Lohner state set")
        weights = np.asarray(weights, dtype=float)
        if weights.shape != self.center.shape:
            raise ValueError("weights must match the state dimension")
        center_value = float(weights @ self.center)
        shape_radius = float(np.sum(np.abs(weights @ self.linear_shape)))
        remainder = FloatInterval.point(0.0)
        for weight, (lower, upper) in zip(weights, self.remainder_box, strict=True):
            remainder = remainder + FloatInterval(float(lower), float(upper)).scale(float(weight))
        return (
            float(np.nextafter(center_value - shape_radius + remainder.lower, -np.inf)),
            float(np.nextafter(center_value + shape_radius + remainder.upper, np.inf)),
        )


@dataclass(frozen=True)
class LohnerAffineMapCertificate:
    """Exact affine image of a Lohner state set."""

    source_set: LohnerStateSet
    target_set: LohnerStateSet
    matrix_shape: tuple[int, int]
    exact_affine_map_certified: bool
    source: str = "lohner_affine_map"

    @property
    def certified(self) -> bool:
        return bool(
            self.exact_affine_map_certified
            and self.source_set.certified
            and self.target_set.certified
        )


@dataclass(frozen=True)
class LohnerOrdinaryTaylorStepCertificate:
    """Lohner enclosure for one planar ordinary Taylor chart step."""

    source_set: LohnerStateSet
    normalized_source_set: LohnerStateSet
    truncated_target_set: LohnerStateSet
    target_set: LohnerStateSet
    kinematic_affine_map: LohnerAffineMapCertificate
    retained_order: int
    physical_step: float
    cauchy_tail_certificate: object
    nonlinear_remainder_certified: bool
    tail_certified: bool
    source: str = "lohner_ordinary_taylor_step"
    time_interval: tuple[float, float] | None = None

    @property
    def certified(self) -> bool:
        return bool(
            self.source_set.certified
            and self.normalized_source_set.certified
            and self.truncated_target_set.certified
            and self.target_set.certified
            and self.kinematic_affine_map.certified
            and self.nonlinear_remainder_certified
            and self.tail_certified
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = []
        if not self.source_set.certified:
            missing.append("source_lohner_state_set")
        if not self.normalized_source_set.certified:
            missing.append("normalized_source_lohner_state_set")
        if not self.kinematic_affine_map.certified:
            missing.append("ordinary_kinematic_affine_map")
        if not self.nonlinear_remainder_certified:
            missing.append("ordinary_taylor_nonlinear_remainder")
        if not self.tail_certified:
            missing.append("ordinary_cauchy_tail")
        if not self.target_set.certified:
            missing.append("target_lohner_state_set")
        return tuple(missing)


@dataclass(frozen=True)
class LohnerOrdinaryPropagationStep:
    start_time: float
    physical_step: float
    retained_order: int
    start_state_set: LohnerStateSet
    truncated_end_state_set: LohnerStateSet
    end_state_set: LohnerStateSet
    step_certificate: LohnerOrdinaryTaylorStepCertificate
    event: str | None = None
    event_time_interval: tuple[float, float] | None = None

    @property
    def start_state_interval(self) -> tuple[tuple[float, float], ...]:
        return self.start_state_set.interval_hull

    @property
    def truncated_end_state_interval(self) -> tuple[tuple[float, float], ...]:
        return self.truncated_end_state_set.interval_hull

    @property
    def end_state_interval(self) -> tuple[tuple[float, float], ...]:
        return self.end_state_set.interval_hull

    @property
    def tail_bound(self) -> float:
        return float(self.step_certificate.cauchy_tail_certificate.tail_bound)

    @property
    def tail_ratio_bound(self) -> float:
        return float(self.step_certificate.cauchy_tail_certificate.ratio_bound)

    @property
    def certified(self) -> bool:
        return bool(self.step_certificate.certified)

    @property
    def proof_certified(self) -> bool:
        return self.certified

    def start_state_contains(self, state: Array) -> bool:
        return self.start_state_set.contains_state(state)

    def end_state_contains(self, state: Array) -> bool:
        return self.end_state_set.contains_state(state)


@dataclass(frozen=True)
class LohnerOrdinaryPropagatedEnclosure:
    steps: tuple[LohnerOrdinaryPropagationStep, ...]

    @property
    def final_state_set(self) -> LohnerStateSet | None:
        return self.steps[-1].end_state_set if self.steps else None

    @property
    def final_state_interval(self) -> tuple[tuple[float, float], ...] | None:
        return None if self.final_state_set is None else self.final_state_set.interval_hull

    @property
    def local_tail_bound(self) -> float:
        return float(sum(step.tail_bound for step in self.steps))

    @property
    def max_step_tail_bound(self) -> float:
        return float(max((step.tail_bound for step in self.steps), default=0.0))

    @property
    def certified_step_count(self) -> int:
        return len(self.steps)

    @property
    def chain_certified(self) -> bool:
        return bool(
            self.steps
            and all(
                _lohner_ordinary_propagation_steps_chain(left, right)
                for left, right in zip(self.steps, self.steps[1:])
            )
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.steps
            and all(step.proof_certified for step in self.steps)
            and self.chain_certified
        )

    def final_state_contains(self, state: Array) -> bool:
        if self.final_state_set is None:
            return False
        return self.final_state_set.contains_state(state)


def lohner_state_from_interval_box(
    state_interval: tuple[tuple[float, float], ...],
    *,
    source: str = "interval_box_diagonal_lohner_lift",
) -> LohnerStateSet:
    """Lift an interval box exactly into diagonal Lohner form."""

    if not state_interval:
        raise ValueError("state_interval cannot be empty")
    center = []
    radii = []
    for lower, upper in state_interval:
        lower = float(lower)
        upper = float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("state intervals must be finite and ordered")
        center.append(0.5 * (lower + upper))
        radii.append(0.5 * (upper - lower))
    linear_shape = np.diag(np.asarray(radii, dtype=float))
    remainder_box = tuple((0.0, 0.0) for _ in center)
    state_set = LohnerStateSet(
        center=np.asarray(center, dtype=float),
        linear_shape=linear_shape,
        remainder_box=remainder_box,
        source=source,
    )
    if not state_set.certified:
        raise ValueError("failed to construct certified Lohner state set")
    return state_set


def propagate_lohner_affine_map(
    state_set: LohnerStateSet,
    matrix: Array,
    bias: Array | None = None,
    *,
    source: str = "lohner_affine_map",
) -> LohnerAffineMapCertificate:
    """Apply an affine map exactly to a Lohner state set."""

    if not isinstance(state_set, LohnerStateSet) or not state_set.certified:
        raise ValueError("state_set must be a certified LohnerStateSet")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != state_set.dimension:
        raise ValueError("matrix must have shape (target_dimension, state_dimension)")
    if bias is None:
        bias_array = np.zeros(matrix.shape[0], dtype=float)
    else:
        bias_array = np.asarray(bias, dtype=float)
    if bias_array.shape != (matrix.shape[0],):
        raise ValueError("bias must match target dimension")
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(bias_array))):
        raise ValueError("matrix and bias must be finite")

    remainder_midpoints = np.array(
        [0.5 * (lower + upper) for lower, upper in state_set.remainder_box],
        dtype=float,
    )
    remainder_radii = np.array(
        [0.5 * (upper - lower) for lower, upper in state_set.remainder_box],
        dtype=float,
    )
    nonzero_remainder = np.nonzero(remainder_radii > 0.0)[0]
    remainder_shape = (
        matrix[:, nonzero_remainder] * remainder_radii[nonzero_remainder][None, :]
        if nonzero_remainder.size
        else np.zeros((matrix.shape[0], 0), dtype=float)
    )
    target_linear_shape = np.concatenate(
        (matrix @ state_set.linear_shape, remainder_shape),
        axis=1,
    )
    target = LohnerStateSet(
        center=matrix @ (state_set.center + remainder_midpoints) + bias_array,
        linear_shape=target_linear_shape,
        remainder_box=tuple((0.0, 0.0) for _ in range(matrix.shape[0])),
        source=source,
    )
    return LohnerAffineMapCertificate(
        source_set=state_set,
        target_set=target,
        matrix_shape=tuple(int(value) for value in matrix.shape),
        exact_affine_map_certified=target.certified,
        source=source,
    )


def center_of_mass_reduction_lohner_map(masses: Array, dimension: int) -> Array:
    """Return the affine matrix for position/velocity COM-frame reduction."""

    masses = np.asarray(masses, dtype=float)
    dimension = int(dimension)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    total_mass = float(np.sum(masses))
    body_matrix = np.eye(3) - np.ones((3, 1)) @ (masses[None, :] / total_mass)
    coordinate_matrix = np.kron(body_matrix, np.eye(dimension))
    return np.block(
        [
            [coordinate_matrix, np.zeros_like(coordinate_matrix)],
            [np.zeros_like(coordinate_matrix), coordinate_matrix],
        ]
    )


def reduce_lohner_state_to_center_of_mass_frame(
    state_set: LohnerStateSet,
    masses: Array,
    *,
    dimension: int,
) -> LohnerAffineMapCertificate:
    """Apply exact center-of-mass-frame reduction to a Lohner state set."""

    matrix = center_of_mass_reduction_lohner_map(masses, dimension)
    if state_set.dimension != matrix.shape[1]:
        raise ValueError("state_set dimension does not match masses/dimension")
    return propagate_lohner_affine_map(
        state_set,
        matrix,
        source="center_of_mass_reduction_lohner_affine_map",
    )


def propagate_lohner_ordinary_taylor_step(
    state_set: LohnerStateSet,
    masses: Array,
    physical_step: float,
    *,
    retained_order: int,
) -> LohnerOrdinaryTaylorStepCertificate:
    """Propagate a planar ordinary chart over a Lohner-shaped initial set.

    The affine kinematic part ``q -> q + h v, v -> v`` acts on the Lohner shape
    directly.  The nonlinear Taylor coefficients are bounded by interval
    recurrence over the source hull and stored only in the remainder.  This is
    the first shape-aware ordinary propagation layer; it is intentionally still
    local and planar.
    """

    if not isinstance(state_set, LohnerStateSet) or not state_set.certified:
        raise ValueError("state_set must be a certified LohnerStateSet")
    if state_set.dimension != 12:
        raise ValueError("ordinary Lohner Taylor propagation currently supports planar three-body states")
    retained_order = int(retained_order)
    if retained_order < 1:
        raise ValueError("retained_order must be at least one")
    physical_step = float(physical_step)
    if not np.isfinite(physical_step):
        raise ValueError("physical_step must be finite")
    return _propagate_lohner_ordinary_taylor_time_interval(
        state_set,
        masses,
        (physical_step, physical_step),
        retained_order=retained_order,
        source="lohner_ordinary_taylor_step",
    )


def propagate_lohner_ordinary_taylor_time_interval_step(
    state_set: LohnerStateSet,
    masses: Array,
    time_interval: tuple[float, float],
    *,
    retained_order: int,
) -> LohnerOrdinaryTaylorStepCertificate:
    """Propagate a planar ordinary chart over a certified physical-time interval."""

    return _propagate_lohner_ordinary_taylor_time_interval(
        state_set,
        masses,
        time_interval,
        retained_order=retained_order,
        source="lohner_ordinary_taylor_time_interval_step",
    )


def _propagate_lohner_ordinary_taylor_time_interval(
    state_set: LohnerStateSet,
    masses: Array,
    time_interval: tuple[float, float],
    *,
    retained_order: int,
    source: str,
) -> LohnerOrdinaryTaylorStepCertificate:
    if not isinstance(state_set, LohnerStateSet) or not state_set.certified:
        raise ValueError("state_set must be a certified LohnerStateSet")
    if state_set.dimension != 12:
        raise ValueError("ordinary Lohner Taylor propagation currently supports planar three-body states")
    retained_order = int(retained_order)
    if retained_order < 1:
        raise ValueError("retained_order must be at least one")
    time_interval = (float(time_interval[0]), float(time_interval[1]))
    if not (
        np.isfinite(time_interval[0])
        and np.isfinite(time_interval[1])
        and time_interval[0] <= time_interval[1]
    ):
        raise ValueError("time_interval must be finite and ordered")
    physical_step = 0.5 * (time_interval[0] + time_interval[1])
    normalized_source = propagate_lohner_affine_map(
        state_set,
        np.eye(state_set.dimension),
        source="lohner_source_remainder_to_shape",
    ).target_set
    start_interval = normalized_source.interval_hull
    positions, velocities = _unpack_planar_point_state(normalized_source.center)
    center_solution = construct_taylor_solution(
        positions,
        velocities,
        masses,
        order=retained_order,
    )
    center_endpoint = center_solution.state_at(physical_step)
    interval_positions, interval_velocities = _unpack_planar_interval_state(start_interval)
    interval_solution = construct_interval_taylor_solution_from_intervals(
        interval_positions,
        interval_velocities,
        masses,
        order=retained_order,
    )
    kinematic_matrix = _ordinary_kinematic_state_matrix(physical_step)
    kinematic_certificate = propagate_lohner_affine_map(
        normalized_source,
        kinematic_matrix,
        bias=center_endpoint - kinematic_matrix @ normalized_source.center,
        source="ordinary_taylor_kinematic_lohner_map",
    )
    nonlinear_remainder = _ordinary_nonlinear_remainder_box_over_time_interval(
        interval_solution,
        center_solution,
        time_interval,
        center_time=physical_step,
        tail_bound=0.0,
    )
    truncated_target = LohnerStateSet(
        center=center_endpoint,
        linear_shape=kinematic_certificate.target_set.linear_shape,
        remainder_box=nonlinear_remainder,
        source="lohner_ordinary_taylor_truncated_target",
    )
    tail_certificate = ordinary_interval_cauchy_majorant_tail_certificate(
        start_interval,
        masses,
        retained_order=retained_order,
        step_size=_local_interval_step_size(time_interval),
    )
    target = LohnerStateSet(
        center=center_endpoint,
        linear_shape=kinematic_certificate.target_set.linear_shape,
        remainder_box=_inflate_remainder_box(nonlinear_remainder, tail_certificate.tail_bound),
        source="lohner_ordinary_taylor_target",
    )
    return LohnerOrdinaryTaylorStepCertificate(
        source_set=state_set,
        normalized_source_set=normalized_source,
        truncated_target_set=truncated_target,
        target_set=target,
        kinematic_affine_map=kinematic_certificate,
        retained_order=retained_order,
        physical_step=physical_step,
        cauchy_tail_certificate=tail_certificate,
        nonlinear_remainder_certified=truncated_target.certified,
        tail_certified=bool(getattr(tail_certificate, "is_nontrivial", False)),
        source=source,
        time_interval=time_interval,
    )


def propagate_lohner_ordinary_set_enclosures(
    masses: Array,
    steps: tuple,
    *,
    retained_order: int | None = None,
    initial_state_set: LohnerStateSet | None = None,
) -> LohnerOrdinaryPropagatedEnclosure:
    """Propagate an ordinary-only step chain with Lohner-shaped sets.

    This is deliberately narrower than ``propagate_ordinary_set_enclosures``:
    it supports only non-event ordinary charts.  Binary, Sundman, KS, and
    event-localized transitions must grow their own Lohner maps before they can
    enter this chain without falling back to interval hulling.
    """

    if not steps:
        return LohnerOrdinaryPropagatedEnclosure(steps=())
    current: LohnerStateSet | None = initial_state_set
    propagated_steps: list[LohnerOrdinaryPropagationStep] = []
    for step in steps:
        if getattr(step, "chart", None) != "ordinary":
            raise ValueError("Lohner ordinary propagation only supports ordinary chart steps")
        if current is None:
            if getattr(step, "start_state_interval", None) is None:
                raise ValueError("ordinary steps must carry start_state_interval")
            current = lohner_state_from_interval_box(
                step.start_state_interval,
                source="ordinary_step_interval_box_to_lohner_chain_start",
            )
        order = _step_retained_order(step, retained_order)
        event = getattr(step, "event", None)
        event_time_interval = None
        if event is None:
            step_certificate = propagate_lohner_ordinary_taylor_step(
                current,
                masses,
                float(step.physical_step),
                retained_order=order,
            )
        else:
            event_time_interval = getattr(step, "event_time_interval", None)
            if event_time_interval is None:
                raise ValueError("event-localized Lohner ordinary steps require event_time_interval")
            step_certificate = propagate_lohner_ordinary_taylor_time_interval_step(
                current,
                masses,
                event_time_interval,
                retained_order=order,
            )
        propagated_step = LohnerOrdinaryPropagationStep(
            start_time=float(getattr(step, "start_time", 0.0)),
            physical_step=float(step_certificate.physical_step),
            retained_order=order,
            start_state_set=current,
            truncated_end_state_set=step_certificate.truncated_target_set,
            end_state_set=step_certificate.target_set,
            step_certificate=step_certificate,
            event=event,
            event_time_interval=event_time_interval,
        )
        propagated_steps.append(propagated_step)
        current = step_certificate.target_set
    return LohnerOrdinaryPropagatedEnclosure(steps=tuple(propagated_steps))


@dataclass(frozen=True)
class SundmanSetPropagationStep:
    kind: str
    start_s: float
    local_s_interval: tuple[float, float]
    global_s_interval: tuple[float, float]
    retained_order: int
    start_time_interval: tuple[float, float]
    end_time_interval: tuple[float, float]
    start_state_interval: tuple[tuple[float, float], ...]
    truncated_end_state_interval: tuple[tuple[float, float], ...]
    end_state_interval: tuple[tuple[float, float], ...]
    tail_bound: float
    tail_ratio_bound: float
    tail_coefficient_source: str
    equation_residual_certified: bool
    time_monotone_certified: bool
    target_certified: bool = False

    @property
    def certified(self) -> bool:
        return bool(
            self.kind in {"full", "target"}
            and np.isfinite(self.start_s)
            and _interval_tuple_certified(self.local_s_interval)
            and _interval_tuple_certified(self.global_s_interval)
            and _interval_tuple_certified(self.start_time_interval)
            and _interval_tuple_certified(self.end_time_interval)
            and _state_interval_certified(self.start_state_interval)
            and _state_interval_certified(self.truncated_end_state_interval)
            and _state_interval_certified(self.end_state_interval)
            and self.retained_order > 0
            and np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
            and np.isfinite(self.tail_ratio_bound)
            and self.tail_ratio_bound < 1.0
            and self.tail_coefficient_source
            and self.equation_residual_certified
            and self.time_monotone_certified
            and (self.kind != "target" or self.target_certified)
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    def start_state_contains(self, state: Array) -> bool:
        return interval_contains_state(self.start_state_interval, state)

    def truncated_end_state_contains(self, state: Array) -> bool:
        return interval_contains_state(self.truncated_end_state_interval, state)

    def end_state_contains(self, state: Array) -> bool:
        return interval_contains_state(self.end_state_interval, state)


@dataclass(frozen=True)
class SundmanSetPropagatedEnclosure:
    target_time: float
    steps: tuple[SundmanSetPropagationStep, ...]

    @property
    def final_state_interval(self) -> tuple[tuple[float, float], ...] | None:
        return self.steps[-1].end_state_interval if self.steps else None

    @property
    def local_tail_bound(self) -> float:
        return float(sum(step.tail_bound for step in self.steps))

    @property
    def max_step_tail_bound(self) -> float:
        return float(max((step.tail_bound for step in self.steps), default=0.0))

    @property
    def certified_step_count(self) -> int:
        return sum(step.certified for step in self.steps)

    @property
    def chain_certified(self) -> bool:
        return bool(
            self.steps
            and all(
                _sundman_set_propagation_steps_chain(left, right)
                for left, right in zip(self.steps, self.steps[1:])
            )
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.steps
            and self.steps[-1].kind == "target"
            and all(step.proof_certified for step in self.steps)
            and self.chain_certified
        )

    def final_state_contains(self, state: Array) -> bool:
        if self.final_state_interval is None:
            return False
        return interval_contains_state(self.final_state_interval, state)


def inflate_state_interval(
    state_interval: tuple[tuple[float, float], ...],
    radius: float,
) -> tuple[tuple[float, float], ...]:
    """Inflate every state coordinate interval by a scalar radius."""

    if radius < 0.0:
        raise ValueError("radius cannot be negative")
    inflated = []
    for lower, upper in state_interval:
        inflated.append(
            (
                float(np.nextafter(float(lower) - radius, -np.inf)),
                float(np.nextafter(float(upper) + radius, np.inf)),
            )
        )
    return tuple(inflated)


def interval_contains_state(state_interval: tuple[tuple[float, float], ...], state: Array) -> bool:
    state = np.asarray(state, dtype=float).reshape(-1)
    if len(state_interval) != state.shape[0]:
        return False
    for value, (lower, upper) in zip(state, state_interval):
        if lower > value or value > upper:
            return False
    return True


def _interval_tuple_certified(interval: tuple[float, float]) -> bool:
    if len(interval) != 2:
        return False
    lower, upper = interval
    return bool(np.isfinite(lower) and np.isfinite(upper) and lower <= upper)


def _state_interval_certified(state_interval: tuple[tuple[float, float], ...]) -> bool:
    return bool(
        state_interval
        and all(_interval_tuple_certified(interval) for interval in state_interval)
    )


def state_interval_subset(
    inner: tuple[tuple[float, float], ...],
    outer: tuple[tuple[float, float], ...],
) -> bool:
    if len(inner) != len(outer):
        return False
    for (inner_lower, inner_upper), (outer_lower, outer_upper) in zip(inner, outer):
        if inner_lower < outer_lower or inner_upper > outer_upper:
            return False
    return True


def _state_intervals_match(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
) -> bool:
    return bool(state_interval_subset(first, second) and state_interval_subset(second, first))


def _state_interval_unions_match(
    first: tuple[tuple[tuple[float, float], ...], ...],
    second: tuple[tuple[tuple[float, float], ...], ...],
) -> bool:
    return bool(
        len(first) == len(second)
        and all(
            _state_intervals_match(first_member, second_member)
            for first_member, second_member in zip(first, second)
        )
    )


def _time_boundaries_match(left_start: float, left_step: float, right_start: float) -> bool:
    left_end = float(left_start) + float(left_step)
    right_start = float(right_start)
    if not (np.isfinite(left_end) and np.isfinite(right_start)):
        return False
    tolerance = 1.0e-10 * max(1.0, abs(left_end), abs(right_start))
    return bool(abs(left_end - right_start) <= tolerance)


def _ordinary_set_propagation_steps_chain(
    left: OrdinarySetPropagationStep,
    right: OrdinarySetPropagationStep,
) -> bool:
    return bool(
        _time_boundaries_match(left.start_time, left.physical_step, right.start_time)
        and _state_intervals_match(left.end_state_interval, right.start_state_interval)
        and _state_interval_unions_match(
            left.end_state_interval_union,
            right.start_state_interval_union,
        )
    )


def _lohner_state_sets_match(left: LohnerStateSet, right: LohnerStateSet) -> bool:
    return bool(
        left.certified
        and right.certified
        and np.array_equal(left.center, right.center)
        and np.array_equal(left.linear_shape, right.linear_shape)
        and left.remainder_box == right.remainder_box
    )


def _lohner_ordinary_propagation_steps_chain(
    left: LohnerOrdinaryPropagationStep,
    right: LohnerOrdinaryPropagationStep,
) -> bool:
    return bool(
        _time_boundaries_match(left.start_time, left.physical_step, right.start_time)
        and _lohner_state_sets_match(left.end_state_set, right.start_state_set)
        and right.step_certificate.source_set is right.start_state_set
    )


def _sundman_interval_endpoint_offset(interval: tuple[float, float]) -> float:
    lower, upper = float(interval[0]), float(interval[1])
    if abs(lower) > abs(upper):
        return lower
    return upper


def _sundman_parameter_boundaries_match(
    left: SundmanSetPropagationStep,
    right: SundmanSetPropagationStep,
) -> bool:
    expected_start = float(left.start_s) + _sundman_interval_endpoint_offset(left.local_s_interval)
    actual_start = float(right.start_s)
    if not (np.isfinite(expected_start) and np.isfinite(actual_start)):
        return False
    tolerance = 1.0e-10 * max(1.0, abs(expected_start), abs(actual_start))
    return bool(
        abs(expected_start - actual_start) <= tolerance
        and left.global_s_interval[0]
        <= min(float(left.start_s), expected_start)
        <= left.global_s_interval[1]
        and left.global_s_interval[0]
        <= max(float(left.start_s), expected_start)
        <= left.global_s_interval[1]
    )


def _sundman_set_propagation_steps_chain(
    left: SundmanSetPropagationStep,
    right: SundmanSetPropagationStep,
) -> bool:
    return bool(
        _sundman_parameter_boundaries_match(left, right)
        and _state_intervals_match(left.end_state_interval, right.start_state_interval)
        and _interval_tuple_certified(left.end_time_interval)
        and _interval_tuple_certified(right.start_time_interval)
        and left.end_time_interval == right.start_time_interval
    )


def _unpack_planar_interval_state(
    state_interval: tuple[tuple[float, float], ...],
) -> tuple[Array, Array]:
    if len(state_interval) != 12:
        raise ValueError("planar interval state must have length 12")
    values = np.array([FloatInterval(lower, upper) for lower, upper in state_interval], dtype=object)
    return values[:6].reshape(3, 2), values[6:].reshape(3, 2)


def _pack_planar_interval_state(positions: Array, velocities: Array) -> tuple[tuple[float, float], ...]:
    return interval_array_as_tuples(np.concatenate([positions.reshape(-1), velocities.reshape(-1)]))


def _unpack_planar_point_state(state: Array) -> tuple[Array, Array]:
    state = np.asarray(state, dtype=float).reshape(-1)
    if state.shape != (12,):
        raise ValueError("planar point state must have length 12")
    return state[:6].reshape(3, 2), state[6:].reshape(3, 2)


def _ordinary_kinematic_state_matrix(physical_step: float) -> Array:
    identity = np.eye(6)
    zeros = np.zeros_like(identity)
    return np.block(
        [
            [identity, float(physical_step) * identity],
            [zeros, identity],
        ]
    )


def _ordinary_nonlinear_remainder_box_over_time_interval(
    interval_solution: object,
    center_solution: object,
    time_interval: tuple[float, float],
    *,
    center_time: float,
    tail_bound: float,
) -> tuple[tuple[float, float], ...]:
    """Bound ordinary Taylor nonlinearity relative to midpoint kinematics."""

    time = FloatInterval(float(time_interval[0]), float(time_interval[1]))
    center_time = float(center_time)
    delta_time = time - FloatInterval.point(center_time)
    tail = FloatInterval(
        float(np.nextafter(-float(tail_bound), -np.inf)),
        float(np.nextafter(float(tail_bound), np.inf)),
    )
    time_powers = [FloatInterval.point(1.0)]
    for _degree in range(1, interval_solution.order + 1):
        time_powers.append(time_powers[-1] * time)

    remainder = []
    for body in range(3):
        for axis in range(2):
            interval_value = interval_solution.velocity[0, body, axis] * delta_time
            center_value = 0.0
            for degree in range(2, interval_solution.order + 1):
                interval_value = interval_value + (
                    interval_solution.position[degree, body, axis] * time_powers[degree]
                )
                center_value += float(center_solution.position[degree, body, axis]) * center_time**degree
            remainder.append((interval_value - FloatInterval.point(center_value) + tail).as_tuple())
    for body in range(3):
        for axis in range(2):
            interval_value = zero_interval()
            center_value = 0.0
            for degree in range(1, interval_solution.order + 1):
                interval_value = interval_value + (
                    interval_solution.velocity[degree, body, axis] * time_powers[degree]
                )
                center_value += float(center_solution.velocity[degree, body, axis]) * center_time**degree
            remainder.append((interval_value - FloatInterval.point(center_value) + tail).as_tuple())
    return tuple(remainder)


def _inflate_remainder_box(
    remainder_box: tuple[tuple[float, float], ...],
    radius: float,
) -> tuple[tuple[float, float], ...]:
    if radius < 0.0:
        raise ValueError("radius cannot be negative")
    return tuple(
        (
            float(np.nextafter(lower - radius, -np.inf)),
            float(np.nextafter(upper + radius, np.inf)),
        )
        for lower, upper in remainder_box
    )


def _interval_square_bounds(value: FloatInterval) -> FloatInterval:
    squares = (value.lower * value.lower, value.upper * value.upper)
    if value.lower <= 0.0 <= value.upper:
        lower = 0.0
    else:
        lower = max(0.0, float(np.nextafter(min(squares), -np.inf)))
    return FloatInterval(lower, float(np.nextafter(max(squares), np.inf)))


def _lc_square_interval(z: Array) -> Array:
    x, y = z
    x_square = _interval_square_bounds(x)
    y_square = _interval_square_bounds(y)
    return np.array([x_square - y_square, (x * y).scale(2.0)], dtype=object)


def _lc_velocity_interval(z: Array, z_velocity: Array) -> Array:
    x, y = z
    vx, vy = z_velocity
    rho = _interval_square_bounds(x) + _interval_square_bounds(y)
    return np.array(
        [
            ((x * vx).scale(2.0) - (y * vy).scale(2.0)) / rho,
            ((y * vx).scale(2.0) + (x * vy).scale(2.0)) / rho,
        ],
        dtype=object,
    )


def _inflate_interval_array(values: Array, radius: float) -> Array:
    if radius < 0.0:
        raise ValueError("radius cannot be negative")
    values = np.asarray(values, dtype=object)
    out = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        value = values[index]
        interval = value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))
        out[index] = FloatInterval(
            float(np.nextafter(interval.lower - radius, -np.inf)),
            float(np.nextafter(interval.upper + radius, np.inf)),
        )
    return out


def _inflate_float_interval(value: FloatInterval, radius: float) -> FloatInterval:
    if radius < 0.0:
        raise ValueError("radius cannot be negative")
    return FloatInterval(
        float(np.nextafter(value.lower - radius, -np.inf)),
        float(np.nextafter(value.upper + radius, np.inf)),
    )


def _float_interval_as_tuple(value: FloatInterval) -> tuple[float, float]:
    return value.as_tuple()


def _target_time_tail_bracket_certified(
    *,
    lower_time: FloatInterval,
    upper_time: FloatInterval,
    target_time: float,
    tail_bound: float,
) -> bool:
    inflated_lower_time = _inflate_float_interval(lower_time, tail_bound)
    inflated_upper_time = _inflate_float_interval(upper_time, tail_bound)
    return bool(inflated_lower_time.upper <= target_time <= inflated_upper_time.lower)


def _project_regularized_binary_interval_to_planar(
    initial_state: IntervalRegularizedBinaryCollisionChartState,
    *,
    z: Array,
    z_velocity: Array,
    binary_center: Array,
    binary_center_velocity: Array,
    third_offset: Array,
    third_offset_velocity: Array,
) -> tuple[tuple[float, float], ...]:
    relative_position = _lc_square_interval(z)
    relative_velocity = _lc_velocity_interval(z, z_velocity)

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
        velocities[first, axis] = binary_center_velocity[axis] - relative_velocity[axis].scale(
            masses[second] / pair_mass
        )
        velocities[second, axis] = binary_center_velocity[axis] + relative_velocity[axis].scale(
            masses[first] / pair_mass
        )
        velocities[third, axis] = binary_center_velocity[axis] + third_offset_velocity[axis]

    return _pack_planar_interval_state(positions, velocities)


def _regularized_binary_interval_taylor_endpoint_pair(
    initial_state: IntervalRegularizedBinaryCollisionChartState,
    parameter_step: float | tuple[float, float],
    *,
    retained_order: int,
    tail_bound: float,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    solution = construct_interval_regularized_binary_taylor_solution_from_intervals(
        initial_state,
        order=retained_order,
    )
    parameter = (
        FloatInterval(*parameter_step)
        if isinstance(parameter_step, tuple)
        else FloatInterval.point(parameter_step)
    )
    z = interval_array_series_eval(solution.z, parameter)
    z_velocity = interval_array_series_eval(solution.z_velocity, parameter)
    binary_center = interval_array_series_eval(solution.binary_center, parameter)
    binary_center_velocity = interval_array_series_eval(solution.binary_center_velocity, parameter)
    third_offset = interval_array_series_eval(solution.third_offset, parameter)
    third_offset_velocity = interval_array_series_eval(solution.third_offset_velocity, parameter)
    truncated_end_interval = _project_regularized_binary_interval_to_planar(
        initial_state,
        z=z,
        z_velocity=z_velocity,
        binary_center=binary_center,
        binary_center_velocity=binary_center_velocity,
        third_offset=third_offset,
        third_offset_velocity=third_offset_velocity,
    )
    end_interval = _project_regularized_binary_interval_to_planar(
        initial_state,
        z=_inflate_interval_array(z, tail_bound),
        z_velocity=_inflate_interval_array(z_velocity, tail_bound),
        binary_center=_inflate_interval_array(binary_center, tail_bound),
        binary_center_velocity=_inflate_interval_array(binary_center_velocity, tail_bound),
        third_offset=_inflate_interval_array(third_offset, tail_bound),
        third_offset_velocity=_inflate_interval_array(third_offset_velocity, tail_bound),
    )
    return truncated_end_interval, end_interval


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


def _ordinary_interval_taylor_endpoint(
    start_state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    physical_step: float,
    *,
    retained_order: int,
) -> tuple[tuple[float, float], ...]:
    interval_positions, interval_velocities = _unpack_planar_interval_state(start_state_interval)
    solution = construct_interval_taylor_solution_from_intervals(
        interval_positions,
        interval_velocities,
        masses,
        order=retained_order,
    )
    return _pack_planar_interval_state(
        solution.positions_at(physical_step),
        solution.velocities_at(physical_step),
    )


def _ordinary_interval_taylor_endpoint_over_time_interval(
    start_state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    time_interval: tuple[float, float],
    *,
    retained_order: int,
) -> tuple[tuple[float, float], ...]:
    if time_interval[0] < 0.0 or time_interval[1] < time_interval[0]:
        raise ValueError("invalid ordinary time interval")
    interval_positions, interval_velocities = _unpack_planar_interval_state(start_state_interval)
    solution = construct_interval_taylor_solution_from_intervals(
        interval_positions,
        interval_velocities,
        masses,
        order=retained_order,
    )
    time = FloatInterval(*time_interval)
    return _pack_planar_interval_state(
        interval_array_series_eval(solution.position, time),
        interval_array_series_eval(solution.velocity, time),
    )


def _ordinary_interval_taylor_endpoint_with_substeps(
    start_state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    physical_step: float,
    *,
    retained_order: int,
    substeps: int,
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
    tuple[object, ...],
]:
    if substeps <= 0:
        raise ValueError("ordinary_substeps must be positive")
    if substeps == 1:
        truncated_end_interval = _ordinary_interval_taylor_endpoint(
            start_state_interval,
            masses,
            physical_step,
            retained_order=retained_order,
        )
        tail_certificate = ordinary_interval_cauchy_majorant_tail_certificate(
            start_state_interval,
            masses,
            retained_order=retained_order,
            step_size=physical_step,
        )
        end_state_interval = inflate_state_interval(
            truncated_end_interval,
            tail_certificate.tail_bound,
        )
        return truncated_end_interval, end_state_interval, (tail_certificate,)

    current_interval = start_state_interval
    substep = physical_step / substeps
    certificates = []
    truncated_end_interval: tuple[tuple[float, float], ...] | None = None
    for _index in range(substeps):
        truncated_end_interval = _ordinary_interval_taylor_endpoint(
            current_interval,
            masses,
            substep,
            retained_order=retained_order,
        )
        tail_certificate = ordinary_interval_cauchy_majorant_tail_certificate(
            current_interval,
            masses,
            retained_order=retained_order,
            step_size=substep,
        )
        certificates.append(tail_certificate)
        current_interval = inflate_state_interval(
            truncated_end_interval,
            tail_certificate.tail_bound,
        )
    if truncated_end_interval is None:
        raise ValueError("ordinary_substeps must be positive")
    return truncated_end_interval, current_interval, tuple(certificates)


def _local_interval_step_size(interval: tuple[float, float]) -> float:
    return float(max(abs(interval[0]), abs(interval[1])))


def _step_retained_order(step: object, retained_order: int | None) -> int:
    if retained_order is not None:
        if retained_order <= 0:
            raise ValueError("retained_order must be positive")
        return retained_order
    certificate = getattr(step, "truncation_certificate", None)
    if certificate is None or not hasattr(certificate, "retained_order"):
        raise ValueError("retained_order is required when a step has no truncation certificate")
    return int(certificate.retained_order)


def _direct_regularized_interval_state_union(
    step: object,
    masses: Array,
) -> tuple[IntervalRegularizedBinaryCollisionChartState, ...]:
    single_state = getattr(step, "start_regularized_interval_state", None)
    state_union = getattr(step, "start_regularized_interval_state_union", None)
    if state_union == ():
        state_union = None
    if single_state is not None and state_union is not None:
        raise ValueError("provide either start_regularized_interval_state or start_regularized_interval_state_union")
    if state_union is None:
        if single_state is None:
            return ()
        states = (single_state,)
    else:
        states = tuple(state_union)
        if not states:
            raise ValueError("start_regularized_interval_state_union cannot be empty")

    masses = np.asarray(masses, dtype=float)
    for state in states:
        if not isinstance(state, IntervalRegularizedBinaryCollisionChartState):
            raise TypeError("direct binary interval starts must be IntervalRegularizedBinaryCollisionChartState")
        if state.pair != step.pair:
            raise ValueError("direct binary interval start pair must match the propagation step pair")
        if not np.allclose(np.asarray(state.masses, dtype=float), masses, rtol=0.0, atol=0.0):
            raise ValueError("direct binary interval start masses must match propagation masses")
        certificate = state.branch_certificate
        if certificate is None or not certificate.certified:
            raise ValueError("direct binary interval starts require a certified regularized branch")
    return states


def newtonian_planar_lipschitz_bound(masses: Array, min_pair_distance: float) -> float:
    """Crude state-space Lipschitz bound for planar Newtonian three-body flow.

    For acceleration derivatives, each pair contributes at most `2*m/r^3` in
    operator norm. The constant below is intentionally conservative and uses
    only the minimum pair distance on the step.
    """

    masses = np.asarray(masses, dtype=float)
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    if min_pair_distance <= 0.0:
        return float("inf")
    acceleration_part = 4.0 * float(np.sum(masses)) / min_pair_distance**3
    return float(max(1.0, acceleration_part))


def propagate_error_budget(masses: Array, steps: tuple) -> PropagatedErrorBudget:
    """Propagate local Taylor-tail bounds through a numerical Gronwall ledger."""

    outgoing = 0.0
    budget_steps: list[ErrorBudgetStep] = []
    for step in steps:
        local_tail = 0.0 if step.truncation_certificate is None else step.truncation_certificate.tail_bound
        lipschitz = newtonian_planar_lipschitz_bound(masses, step.min_pair_distance)
        incoming = outgoing
        growth = np.exp(min(lipschitz * abs(step.physical_step), 700.0))
        outgoing = growth * (incoming + local_tail)
        budget_steps.append(
            ErrorBudgetStep(
                chart=step.chart,
                start_time=step.start_time,
                physical_step=step.physical_step,
                local_tail_bound=float(local_tail),
                lipschitz_bound=float(lipschitz),
                incoming_bound=float(incoming),
                outgoing_bound=float(outgoing),
            )
        )
    return PropagatedErrorBudget(steps=tuple(budget_steps))


def propagate_interval_enclosures(masses: Array, steps: tuple) -> PropagatedIntervalEnclosure:
    """Return endpoint interval boxes inflated by the propagated tail budget.

    Each hybrid step already records a chart-produced interval enclosure for
    the truncated endpoint. This layer materializes the scalar propagated tail
    budget as coordinate-wise endpoint boxes, so consumers can verify set
    containment directly instead of comparing only scalar norms.
    """

    outgoing = 0.0
    interval_steps: list[PropagatedIntervalStep] = []
    for step in steps:
        if step.start_state_interval is None or step.end_state_interval is None:
            raise ValueError("all steps must carry start and end state intervals")
        local_tail = 0.0 if step.truncation_certificate is None else step.truncation_certificate.tail_bound
        lipschitz = newtonian_planar_lipschitz_bound(masses, step.min_pair_distance)
        incoming = outgoing
        growth = np.exp(min(lipschitz * abs(step.physical_step), 700.0))
        outgoing = growth * (incoming + local_tail)
        interval_steps.append(
            PropagatedIntervalStep(
                chart=step.chart,
                start_time=step.start_time,
                physical_step=step.physical_step,
                local_tail_bound=float(local_tail),
                lipschitz_bound=float(lipschitz),
                incoming_radius=float(incoming),
                outgoing_radius=float(outgoing),
                base_start_state_interval=tuple(step.start_state_interval),
                base_end_state_interval=tuple(step.end_state_interval),
                start_state_interval=inflate_state_interval(step.start_state_interval, incoming),
                end_state_interval=inflate_state_interval(step.end_state_interval, outgoing),
            )
        )
    return PropagatedIntervalEnclosure(steps=tuple(interval_steps))


def propagate_ordinary_set_enclosures(
    masses: Array,
    steps: tuple,
    *,
    retained_order: int | None = None,
    ordinary_substeps: int = 1,
) -> OrdinarySetPropagatedEnclosure:
    """Propagate endpoint boxes from each incoming box.

    This is stricter than scalar budget inflation: supported steps rebuild an
    interval Taylor chart from the previous propagated endpoint interval, then
    recertify the local Cauchy tail on that incoming interval box. Non-event
    ordinary steps may be subdivided into smaller certified substeps to avoid
    losing near-collision separation to one large Cauchy tail. Certified
    ordinary event steps are propagated over their recorded event-time
    interval when the incoming box is covered by the event certificate's start
    box. Regularized-binary steps lift each incoming union member separately
    and use certified Levi-Civita branches or atlas splits for each member,
    or consume certified already-lifted interval regularized states when a
    step starts at a binary collision that has no finite planar velocity lift.
    """

    set_steps: list[OrdinarySetPropagationStep] = []
    current_interval: tuple[tuple[float, float], ...] | None = None
    current_interval_union: tuple[tuple[tuple[float, float], ...], ...] | None = None
    ordinary_substeps = int(ordinary_substeps)
    if ordinary_substeps <= 0:
        raise ValueError("ordinary_substeps must be positive")
    for step in steps:
        if step.start_state_interval is None:
            raise ValueError("steps must carry start state intervals")
        order = _step_retained_order(step, retained_order)
        if current_interval_union is None:
            step_union = getattr(step, "start_state_interval_union", None)
            start_interval_union = tuple(step_union) if step_union else (step.start_state_interval,)
        else:
            start_interval_union = current_interval_union
        if len(start_interval_union) == 1:
            start_interval = step.start_state_interval if current_interval is None else current_interval
        else:
            start_interval = _hull_state_intervals(start_interval_union)

        step_ordinary_substeps = 1
        if step.chart == "ordinary":
            event_time_interval = getattr(step, "event_time_interval", None)
            event_time_interval_union: tuple[tuple[float, float], ...] = ()
            event_parameter_interval = None
            if step.event is not None:
                if event_time_interval is None:
                    raise ValueError("ordinary event steps must carry a certified event-time interval")
                if not all(
                    state_interval_subset(member, step.start_state_interval)
                    for member in start_interval_union
                ):
                    raise ValueError("ordinary event certificate does not cover the propagated incoming union")
                event_union_certificate = getattr(step, "event_union_certificate", None)
                member_root_enclosures = getattr(event_union_certificate, "member_root_enclosures", ())
                if (
                    len(member_root_enclosures) == len(start_interval_union)
                    and member_root_enclosures
                    and all(
                        enclosure is not None and enclosure.certifies_earliest_root
                        for enclosure in member_root_enclosures
                    )
                ):
                    event_time_interval_union = tuple(
                        enclosure.interval for enclosure in member_root_enclosures
                    )
                else:
                    event_time_interval_union = tuple(
                        event_time_interval for _member in start_interval_union
                    )
                cauchy_step_sizes = tuple(
                    _local_interval_step_size(member_event_interval)
                    for member_event_interval in event_time_interval_union
                )
                truncated_end_interval_union = tuple(
                    _ordinary_interval_taylor_endpoint_over_time_interval(
                        member,
                        masses,
                        member_event_interval,
                        retained_order=order,
                    )
                    for member, member_event_interval in zip(
                        start_interval_union,
                        event_time_interval_union,
                    )
                )
                member_certificate_groups = tuple(
                    (
                        ordinary_interval_cauchy_majorant_tail_certificate(
                            member,
                            masses,
                            retained_order=order,
                            step_size=cauchy_step_size,
                        ),
                    )
                    for member, cauchy_step_size in zip(start_interval_union, cauchy_step_sizes)
                )
                end_state_interval_union = tuple(
                    inflate_state_interval(truncated_end_interval, certificates[0].tail_bound)
                    for truncated_end_interval, certificates in zip(
                        truncated_end_interval_union,
                        member_certificate_groups,
                    )
                )
            else:
                step_ordinary_substeps = ordinary_substeps
                member_results = tuple(
                    _ordinary_interval_taylor_endpoint_with_substeps(
                        member,
                        masses,
                        step.physical_step,
                        retained_order=order,
                        substeps=step_ordinary_substeps,
                    )
                    for member in start_interval_union
                )
                truncated_end_interval_union = tuple(result[0] for result in member_results)
                end_state_interval_union = tuple(result[1] for result in member_results)
                member_certificate_groups = tuple(result[2] for result in member_results)
            member_tail_certificates = tuple(
                certificate
                for certificates in member_certificate_groups
                for certificate in certificates
            )
            truncated_end_interval = _hull_state_intervals(truncated_end_interval_union)
            end_interval = _hull_state_intervals(end_state_interval_union)
            member_tail_bounds = tuple(
                sum(certificate.tail_bound for certificate in certificates)
                for certificates in member_certificate_groups
            )
            tail_ratio_bound = max(certificate.ratio_bound for certificate in member_tail_certificates)
            source_set = {certificate.coefficient_source for certificate in member_tail_certificates}
            tail_bound = float(max(member_tail_bounds))
            if step_ordinary_substeps == 1:
                tail_coefficient_source = (
                    next(iter(source_set))
                    if len(source_set) == 1
                    else "interval_cauchy_majorant_union"
                )
            else:
                tail_coefficient_source = (
                    "interval_cauchy_majorant_substeps"
                    if len(start_interval_union) == 1
                    else "interval_cauchy_majorant_union_substeps"
                )
            pair = getattr(step, "pair", None)
            parameter_step = getattr(step, "parameter_step", step.physical_step)
            binary_atlas_chart_count = 0
        elif step.chart == "binary":
            if step.pair is None:
                raise ValueError("binary steps must carry a selected pair")
            parameter_step = getattr(step, "parameter_step", None)
            if parameter_step is None:
                raise ValueError("binary steps must carry a regularized parameter step")
            direct_regularized_states = _direct_regularized_interval_state_union(step, masses)
            if direct_regularized_states and len(direct_regularized_states) != len(start_interval_union):
                raise ValueError(
                    "direct binary interval start union must match the propagated start interval union"
                )
            event_parameter_interval = None
            endpoint_parameter: float | tuple[float, float] = float(parameter_step)
            cauchy_step_size = float(parameter_step)
            if step.event is not None:
                if step.event != "exit_binary":
                    raise ValueError(f"binary set propagation does not support event {step.event!r}")
                event_certificate = getattr(step, "event_certificate", None)
                root_enclosure = getattr(event_certificate, "root_enclosure", None)
                if root_enclosure is None or not root_enclosure.certifies_earliest_root:
                    raise ValueError("binary exit set propagation requires a certified root enclosure")
                event_parameter_interval = root_enclosure.interval
                if not (event_parameter_interval[0] <= parameter_step <= event_parameter_interval[1]):
                    raise ValueError("binary event parameter is outside the certified root enclosure")
                endpoint_parameter = event_parameter_interval
                cauchy_step_size = _local_interval_step_size(event_parameter_interval)

            member_results = []
            for member_index, member_start_interval in enumerate(start_interval_union):
                if direct_regularized_states:
                    interval_binary_state = direct_regularized_states[member_index]
                    member_tail_certificate = regularized_binary_interval_cauchy_majorant_tail_certificate(
                        interval_binary_state,
                        retained_order=order,
                        step_size=cauchy_step_size,
                    )
                    member_truncated, member_end = _regularized_binary_interval_taylor_endpoint_pair(
                        interval_binary_state,
                        endpoint_parameter,
                        retained_order=order,
                        tail_bound=member_tail_certificate.tail_bound,
                    )
                    member_results.append((member_tail_certificate, (member_truncated,), (member_end,), 0))
                    continue

                interval_binary_state = planar_interval_to_regularized_binary_collision_chart(
                    member_start_interval,
                    masses,
                    pair=step.pair,
                )
                branch_certificate = interval_binary_state.branch_certificate
                direct_branch_certified = branch_certificate is not None and branch_certificate.certified
                if direct_branch_certified:
                    member_tail_certificate = regularized_binary_interval_cauchy_majorant_tail_certificate(
                        interval_binary_state,
                        retained_order=order,
                        step_size=cauchy_step_size,
                    )
                    member_truncated, member_end = _regularized_binary_interval_taylor_endpoint_pair(
                        interval_binary_state,
                        endpoint_parameter,
                        retained_order=order,
                        tail_bound=member_tail_certificate.tail_bound,
                    )
                    member_results.append((member_tail_certificate, (member_truncated,), (member_end,), 0))
                    continue

                atlas_initial_states = planar_interval_to_regularized_binary_collision_chart_atlas(
                    member_start_interval,
                    masses,
                    pair=step.pair,
                )
                if not (
                    atlas_initial_states
                    and all(
                        chart.branch_certificate is not None and chart.branch_certificate.certified
                        for chart in atlas_initial_states
                    )
                ):
                    raise ValueError("binary set propagation requires a certified Levi-Civita branch or atlas")
                member_tail_certificate = regularized_binary_interval_atlas_cauchy_majorant_tail_certificate(
                    atlas_initial_states,
                    retained_order=order,
                    step_size=cauchy_step_size,
                )
                member_intervals = tuple(
                    _regularized_binary_interval_taylor_endpoint_pair(
                        atlas_initial_state,
                        endpoint_parameter,
                        retained_order=order,
                        tail_bound=member_tail_certificate.tail_bound,
                    )
                    for atlas_initial_state in atlas_initial_states
                )
                member_results.append(
                    (
                        member_tail_certificate,
                        tuple(member[0] for member in member_intervals),
                        tuple(member[1] for member in member_intervals),
                        len(atlas_initial_states),
                    )
                )

            member_tail_certificates = tuple(result[0] for result in member_results)
            truncated_end_interval_union = tuple(
                interval for result in member_results for interval in result[1]
            )
            end_state_interval_union = tuple(
                interval for result in member_results for interval in result[2]
            )
            truncated_end_interval = _hull_state_intervals(truncated_end_interval_union)
            end_interval = _hull_state_intervals(end_state_interval_union)
            tail_certificate = max(member_tail_certificates, key=lambda certificate: certificate.tail_bound)
            tail_bound = float(tail_certificate.tail_bound)
            tail_ratio_bound = max(certificate.ratio_bound for certificate in member_tail_certificates)
            source_set = {certificate.coefficient_source for certificate in member_tail_certificates}
            tail_coefficient_source = (
                next(iter(source_set))
                if len(start_interval_union) == 1 and len(source_set) == 1
                else "regularized_interval_cauchy_majorant_union"
            )
            binary_atlas_chart_count = sum(result[3] for result in member_results)
            event_time_interval = None
            event_time_interval_union = ()
            pair = step.pair
        else:
            raise ValueError(f"set propagation does not support chart {step.chart!r}")

        set_steps.append(
            OrdinarySetPropagationStep(
                start_time=step.start_time,
                physical_step=step.physical_step,
                retained_order=order,
                start_state_interval=start_interval,
                truncated_end_state_interval=truncated_end_interval,
                end_state_interval=end_interval,
                tail_bound=float(tail_bound),
                tail_ratio_bound=float(tail_ratio_bound),
                tail_coefficient_source=tail_coefficient_source,
                start_state_interval_union=start_interval_union,
                truncated_end_state_interval_union=truncated_end_interval_union,
                end_state_interval_union=end_state_interval_union,
                event=step.event,
                event_time_interval=event_time_interval,
                event_time_interval_union=event_time_interval_union,
                event_parameter_interval=event_parameter_interval,
                chart=step.chart,
                pair=pair,
                parameter_step=float(parameter_step),
                binary_atlas_chart_count=binary_atlas_chart_count,
                ordinary_substeps=step_ordinary_substeps,
            )
        )
        current_interval = end_interval
        current_interval_union = end_state_interval_union
    return OrdinarySetPropagatedEnclosure(steps=tuple(set_steps))


def propagate_sundman_target_set_enclosure(
    target_solution: object,
    *,
    retained_order: int | None = None,
    target_bisections: int = 60,
) -> SundmanSetPropagatedEnclosure:
    """Rebuild a Sundman physical-time target proof from incoming boxes.

    The input is an ``IntervalSundmanTimeTargetSolution`` produced in Cauchy
    tail-certificate mode. Each full Sundman step is reconstructed from the
    previously propagated endpoint interval, then the final target-containing
    chart is reconstructed and recertified with a Cauchy-tail-aware target
    bracket.
    """

    from .sundman import (
        certify_global_sundman_physical_time_target,
        certify_interval_sundman_equations,
        construct_interval_sundman_taylor_solution_from_intervals,
        sundman_factor_interval_over_s_interval,
    )

    masses = np.asarray(target_solution.masses, dtype=float)
    distance_power = float(getattr(target_solution, "distance_power", 1.0))
    set_steps: list[SundmanSetPropagationStep] = []
    current_interval: tuple[tuple[float, float], ...] | None = None
    current_time = target_solution.time_intervals[0]
    current_s = float(target_solution.s_values[0])

    for recorded_step in target_solution.steps:
        if recorded_step.truncation_certificate is None:
            raise ValueError("Sundman set propagation requires Cauchy tail certificates")
        order = _step_retained_order(recorded_step, retained_order)
        start_interval = (
            interval_array_as_tuples(recorded_step.start_state_interval)
            if current_interval is None
            else current_interval
        )
        positions, velocities = _unpack_planar_interval_state(start_interval)
        chart = construct_interval_sundman_taylor_solution_from_intervals(
            positions,
            velocities,
            masses,
            order=order,
            distance_power=distance_power,
        )
        residual = certify_interval_sundman_equations(chart, coefficient_count=order)
        local_s_interval = (
            FloatInterval(0.0, recorded_step.s_step)
            if recorded_step.s_step > 0.0
            else FloatInterval(recorded_step.s_step, 0.0)
        )
        factor_interval = sundman_factor_interval_over_s_interval(chart, local_s_interval)
        truncated_state = chart.state_at_s(recorded_step.s_step)
        tail_certificate = sundman_interval_cauchy_majorant_tail_certificate(
            positions,
            velocities,
            masses,
            retained_order=order,
            step_size=recorded_step.s_step,
            distance_power=distance_power,
        )
        end_state = _inflate_interval_array(truncated_state, tail_certificate.tail_bound)
        physical_step = _inflate_float_interval(
            chart.physical_time_at_s(recorded_step.s_step),
            tail_certificate.tail_bound,
        )
        end_time = current_time + physical_step
        global_s_interval = FloatInterval(
            current_s + local_s_interval.lower,
            current_s + local_s_interval.upper,
        )
        end_interval = interval_array_as_tuples(end_state)
        set_steps.append(
            SundmanSetPropagationStep(
                kind="full",
                start_s=current_s,
                local_s_interval=_float_interval_as_tuple(local_s_interval),
                global_s_interval=_float_interval_as_tuple(global_s_interval),
                retained_order=order,
                start_time_interval=_float_interval_as_tuple(current_time),
                end_time_interval=_float_interval_as_tuple(end_time),
                start_state_interval=start_interval,
                truncated_end_state_interval=interval_array_as_tuples(truncated_state),
                end_state_interval=end_interval,
                tail_bound=float(tail_certificate.tail_bound),
                tail_ratio_bound=float(tail_certificate.ratio_bound),
                tail_coefficient_source=tail_certificate.coefficient_source,
                equation_residual_certified=residual.certified,
                time_monotone_certified=factor_interval.lower > 0.0,
            )
        )
        current_interval = end_interval
        current_time = end_time
        current_s += float(recorded_step.s_step)

    target_tail_certificate = target_solution.target_truncation_certificate
    if target_tail_certificate is None:
        raise ValueError("Sundman target set propagation requires a target Cauchy certificate")
    target_order = retained_order if retained_order is not None else int(target_tail_certificate.retained_order)
    if target_order <= 0:
        raise ValueError("retained_order must be positive")
    start_interval = (
        interval_array_as_tuples(target_solution.target_start_state_interval)
        if current_interval is None
        else current_interval
    )
    positions, velocities = _unpack_planar_interval_state(start_interval)
    chart = construct_interval_sundman_taylor_solution_from_intervals(
        positions,
        velocities,
        masses,
        order=target_order,
        distance_power=distance_power,
    )
    residual = certify_interval_sundman_equations(chart, coefficient_count=target_order)

    recorded_target_interval = target_solution.target_certificate.local_s_interval
    target_certificate = certify_global_sundman_physical_time_target(
        chart,
        current_time,
        target_solution.target_time,
        recorded_target_interval,
        chart_start_s=current_s,
        max_bisections=target_bisections,
    )
    target_step_size = max(
        abs(target_certificate.local_s_interval.lower),
        abs(target_certificate.local_s_interval.upper),
    )
    target_tail_certificate = sundman_interval_cauchy_majorant_tail_certificate(
        positions,
        velocities,
        masses,
        retained_order=target_order,
        step_size=target_step_size,
        distance_power=distance_power,
    )
    target_tail_bracket_certified = _target_time_tail_bracket_certified(
        lower_time=target_certificate.global_time_at_lower,
        upper_time=target_certificate.global_time_at_upper,
        target_time=float(target_solution.target_time),
        tail_bound=target_tail_certificate.tail_bound,
    )
    if not target_tail_bracket_certified:
        if float(target_solution.target_time) > current_time.upper:
            target_bounds = FloatInterval(0.0, recorded_target_interval.upper)
        elif float(target_solution.target_time) < current_time.lower:
            target_bounds = FloatInterval(recorded_target_interval.lower, 0.0)
        else:
            target_bounds = recorded_target_interval
        target_certificate = certify_global_sundman_physical_time_target(
            chart,
            current_time,
            target_solution.target_time,
            target_bounds,
            chart_start_s=current_s,
            max_bisections=0,
        )
        target_step_size = max(
            abs(target_certificate.local_s_interval.lower),
            abs(target_certificate.local_s_interval.upper),
        )
        target_tail_certificate = sundman_interval_cauchy_majorant_tail_certificate(
            positions,
            velocities,
            masses,
            retained_order=target_order,
            step_size=target_step_size,
            distance_power=distance_power,
        )
        target_tail_bracket_certified = _target_time_tail_bracket_certified(
            lower_time=target_certificate.global_time_at_lower,
            upper_time=target_certificate.global_time_at_upper,
            target_time=float(target_solution.target_time),
            tail_bound=target_tail_certificate.tail_bound,
        )

    truncated_target_state = chart.state_over_s_interval(target_certificate.local_s_interval)
    target_state = _inflate_interval_array(truncated_target_state, target_tail_certificate.tail_bound)
    target_end_time = FloatInterval(
        float(np.nextafter(target_certificate.global_time_at_lower.lower, -np.inf)),
        float(np.nextafter(target_certificate.global_time_at_upper.upper, np.inf)),
    )
    target_global_s_interval = target_certificate.global_s_interval
    target_factor_interval = sundman_factor_interval_over_s_interval(
        chart,
        target_certificate.local_s_interval,
    )
    set_steps.append(
        SundmanSetPropagationStep(
            kind="target",
            start_s=current_s,
            local_s_interval=_float_interval_as_tuple(target_certificate.local_s_interval),
            global_s_interval=_float_interval_as_tuple(target_global_s_interval),
            retained_order=target_order,
            start_time_interval=_float_interval_as_tuple(current_time),
            end_time_interval=_float_interval_as_tuple(target_end_time),
            start_state_interval=start_interval,
            truncated_end_state_interval=interval_array_as_tuples(truncated_target_state),
            end_state_interval=interval_array_as_tuples(target_state),
            tail_bound=float(target_tail_certificate.tail_bound),
            tail_ratio_bound=float(target_tail_certificate.ratio_bound),
            tail_coefficient_source=target_tail_certificate.coefficient_source,
            equation_residual_certified=residual.certified,
            time_monotone_certified=target_factor_interval.lower > 0.0,
            target_certified=bool(target_certificate.certified and target_tail_bracket_certified),
        )
    )
    return SundmanSetPropagatedEnclosure(
        target_time=float(target_solution.target_time),
        steps=tuple(set_steps),
    )
