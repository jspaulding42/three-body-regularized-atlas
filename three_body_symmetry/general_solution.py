"""Composed certified evaluator for general noncollision three-body data.

This is not a finite closed-form solution. It is the executable
lift/construct/project/verify pipeline that the broader proof would have to
globalize: reduce to the inertial center-of-mass frame, construct a certified
interval Sundman-time target chart there, then project the enclosure back to
the original inertial coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .compact_time import (
    CompactifiedTimeTargetCertificate,
    CompactifiedTimeTaylorEvaluationCertificate,
    certify_compactified_time_target,
    certify_compactified_time_taylor_evaluation,
)
from .compact_sundman import (
    IntervalCompactifiedSundmanTimeTargetSolution,
    continue_interval_compactified_sundman_to_time,
)
from .hybrid import continue_hybrid
from .intervals import FloatInterval, interval_array_contains_point
from .ks_binary_series import (
    certify_spatial_ordinary_ks_entry_event,
    construct_interval_spatial_ks_binary_taylor_solution_from_intervals,
    spatial_interval_to_ks_binary_chart_state,
    spatial_ks_competing_entry_event_to_ks_chart_state_atlas,
    spatial_ordinary_entry_event_to_ks_chart_state,
)
from .reduction import (
    CenterOfMassFrameData,
    IntervalCenterOfMassFrameData,
    reconstruct_interval_from_center_of_mass_frame,
    reduce_interval_to_center_of_mass_frame,
    reduce_to_center_of_mass_frame,
)
from .sundman import IntervalSundmanTimeTargetSolution, continue_interval_sundman_to_time
from .series import construct_interval_taylor_solution_from_intervals
from .validated_atlas import (
    FiniteTimeChartSelectorAttempt,
    FiniteTimeChartSelectorTrace,
    ProofLedger,
    ProofLedgerEntry,
    ValidatedAtlasSolution,
    finite_time_selector_trace_binding_token,
    certify_next_finite_time_event_set,
    partition_ks_state_by_competing_event_order,
    certify_simultaneous_close_pair_partition,
    time_reverse_validated_atlas_solution,
    validated_atlas_from_hybrid_solution,
    validated_atlas_from_spatial_close_pair_branch_partition,
    validated_atlas_from_spatial_ks_event_order_partition,
    validated_atlas_from_spatial_ks_competing_binary_handoff_to_atlas,
    validated_atlas_from_spatial_ks_competing_binary_handoff,
    validated_atlas_from_spatial_initial_ks_handoff,
    validated_atlas_from_spatial_ordinary_ks_competing_binary_handoff,
    validated_atlas_from_spatial_ordinary_ks_handoff,
    validated_atlas_from_spatial_ordinary_ks_suffix_atlas,
    validated_atlas_from_unrestricted_evaluation,
)


Array = np.ndarray

_COMPACTIFIED_SUNDMAN_AUTO_FALLBACK_MESSAGES = (
    "target physical time remained inside an uncertified compactified-Sundman step",
    "compactified Sundman tail bound stayed nonfinite",
    "compactified Sundman target tail bound stayed nonfinite",
    "compact_bounds must have positive width",
    "interval endpoints must be finite",
)
_SPATIAL_KS_AUTO_ENTER_FRACTION = 9.0 / 16.0
_SPATIAL_KS_AUTO_EXIT_MULTIPLIER = 2.0
_SPATIAL_KS_AUTO_ENTRY_TIME_SAFETY = 1.5
_SPATIAL_KS_AUTO_S_UPPER = 0.1
_SPATIAL_KS_AUTO_EXIT_SEARCH_FACTOR = 3.0
_SPATIAL_KS_ORDINARY_HANDOFF_EXIT_RETRY_FACTOR = 1.5
_SPATIAL_KS_DEFAULT_COMPETING_EVENT_BUDGETS = (3, 7)


class FiniteTimeChartSelectorError(RuntimeError):
    """Raised when a required finite-time chart route blocks fallback."""

    def __init__(
        self,
        blocking_attempts: tuple[FiniteTimeChartSelectorAttempt, ...],
    ):
        self.blocking_attempts = tuple(blocking_attempts)
        super().__init__(_format_finite_time_selector_failure(self.blocking_attempts))

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        for attempt in self.blocking_attempts:
            for obligation in attempt.missing_obligations:
                obligation_id = str(obligation)
                if obligation_id.isidentifier():
                    missing.append(obligation_id)
                elif ":" in obligation_id:
                    prefix = obligation_id.split(":", 1)[0]
                    if prefix.isidentifier():
                        missing.append(prefix)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class FiniteTimeKSLoopProgressStep:
    """One constructor-derived event decision inside a spatial-KS loop."""

    index: int
    active_pair: tuple[int, int]
    remaining_time_interval: FloatInterval
    event_set_certificate: object
    decision: str
    event_id: str | None = None
    event_pair: tuple[int, int] | None = None
    event_time_interval: FloatInterval | None = None
    next_branch: str | None = None
    event_order_partition_certificate: object | None = None
    missing_obligations: tuple[str, ...] = ()

    @property
    def certified(self) -> bool:
        return bool(
            self.index >= 0
            and self.decision
            and self.remaining_time_interval.lower <= self.remaining_time_interval.upper
            and np.isfinite(self.remaining_time_interval.lower)
            and np.isfinite(self.remaining_time_interval.upper)
            and not self.missing_obligations
            and (
                self.decision == "target_before_all_events"
                or (
                    self.decision == "unique_competing_event"
                    and self.event_id
                    and self.event_pair is not None
                    and self.event_time_interval is not None
                    and self.next_branch
                )
            )
        )

    @property
    def reaches_target(self) -> bool:
        return self.decision == "target_before_all_events"


@dataclass(frozen=True)
class FiniteTimeKSLoopProgressCertificate:
    """Finite spatial-KS event-loop evidence for one target-time request."""

    initial_pair: tuple[int, int]
    target_time_after_start: FloatInterval
    max_competing_events: int
    steps: tuple[FiniteTimeKSLoopProgressStep, ...]
    missing_obligations: tuple[str, ...]

    @property
    def target_reached(self) -> bool:
        return bool(self.steps and self.steps[-1].reaches_target)

    @property
    def event_count(self) -> int:
        return sum(step.decision == "unique_competing_event" for step in self.steps)

    @property
    def required_recursive_repeat_budget(self) -> int:
        return max(0, self.event_count - 1)

    @property
    def certified(self) -> bool:
        return bool(
            self.target_reached
            and not self.missing_obligations
            and all(step.certified for step in self.steps)
        )


class FiniteTimeKSLoopBlocked(RuntimeError):
    """Raised when the spatial-KS event loop reaches a typed stopping obligation."""

    def __init__(self, certificate: FiniteTimeKSLoopProgressCertificate):
        self.certificate = certificate
        super().__init__(
            "spatial KS finite-time loop blocked: "
            + ", ".join(certificate.missing_obligations)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return self.certificate.missing_obligations


@dataclass(frozen=True)
class ReducedSundmanEvaluation:
    """Certified target-time enclosure produced by the composed lift."""

    target_time: float
    point_reduction: CenterOfMassFrameData
    interval_reduction: IntervalCenterOfMassFrameData
    reduced_target: IntervalSundmanTimeTargetSolution | None
    target_position_interval: Array
    target_velocity_interval: Array

    @property
    def reduction_certified(self) -> bool:
        return bool(
            self.point_reduction.certificate.certified
            and self.interval_reduction.certificate.certified
            and self.interval_reduction.contains_point_reduction(self.point_reduction)
        )

    @property
    def target_time_certified(self) -> bool:
        return bool(self.reduced_target is None or self.reduced_target.target_certificate.certified)

    @property
    def dynamics_certified(self) -> bool:
        return bool(self.reduced_target is None or self.reduced_target.certified)

    @property
    def tail_certified(self) -> bool:
        return bool(self.reduced_target is None or self.reduced_target.tail_certified)

    @property
    def proof_certified(self) -> bool:
        return bool(self.reduction_certified and self.dynamics_certified and self.tail_certified)

    @property
    def target_state_interval(self) -> Array:
        return np.concatenate(
            [
                np.asarray(self.target_position_interval, dtype=object).reshape(-1),
                np.asarray(self.target_velocity_interval, dtype=object).reshape(-1),
            ]
        )

    @property
    def local_tail_bound(self) -> float:
        if self.reduced_target is None:
            return 0.0
        return self.reduced_target.local_tail_bound

    @property
    def max_step_tail_bound(self) -> float:
        if self.reduced_target is None:
            return 0.0
        return self.reduced_target.max_step_tail_bound

    @property
    def triple_collision_excluded(self) -> bool:
        return bool(self.reduced_target is not None and self.reduced_target.triple_collision_excluded)

    @property
    def triple_collision_status(self) -> str:
        if self.reduced_target is None:
            return "not_applicable"
        return self.reduced_target.triple_collision_status

    @property
    def triple_collision_exclusion_reason(self) -> str | None:
        if self.reduced_target is None:
            return None
        return self.reduced_target.triple_collision_exclusion_reason

    @property
    def triple_collision_undecided(self) -> bool:
        return self.triple_collision_status == "undecided"

    def target_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))


@dataclass(frozen=True)
class CompactifiedReducedSundmanEvaluation:
    """Reduced Sundman evaluation addressed by bounded compact time."""

    compactification_certificate: CompactifiedTimeTargetCertificate
    time_taylor_certificate: CompactifiedTimeTaylorEvaluationCertificate
    evaluation: ReducedSundmanEvaluation

    @property
    def target_time(self) -> float:
        return self.compactification_certificate.physical_time

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.compactification_certificate.certified
            and self.time_taylor_certificate.certified
            and self.evaluation.proof_certified
        )

    @property
    def target_state_interval(self) -> Array:
        return self.evaluation.target_state_interval

    def target_state_contains(self, state: Array) -> bool:
        return self.evaluation.target_state_contains(state)


@dataclass(frozen=True)
class ReducedCompactifiedSundmanEvaluation:
    """Certified target-time enclosure using bounded compactified Sundman charts."""

    target_time: float
    point_reduction: CenterOfMassFrameData
    interval_reduction: IntervalCenterOfMassFrameData
    reduced_target: IntervalCompactifiedSundmanTimeTargetSolution | None
    target_position_interval: Array
    target_velocity_interval: Array

    @property
    def reduction_certified(self) -> bool:
        return bool(
            self.point_reduction.certificate.certified
            and self.interval_reduction.certificate.certified
            and self.interval_reduction.contains_point_reduction(self.point_reduction)
        )

    @property
    def target_time_certified(self) -> bool:
        return bool(self.reduced_target is None or self.reduced_target.target_certificate.certified)

    @property
    def dynamics_certified(self) -> bool:
        return bool(self.reduced_target is None or self.reduced_target.certified)

    @property
    def tail_certified(self) -> bool:
        return bool(self.reduced_target is None or self.reduced_target.tail_certified)

    @property
    def proof_certified(self) -> bool:
        return bool(self.reduction_certified and self.dynamics_certified and self.tail_certified)

    @property
    def target_compact_parameter_interval(self) -> FloatInterval | None:
        if self.reduced_target is None:
            return None
        return self.reduced_target.target_compact_parameter_interval

    @property
    def target_state_interval(self) -> Array:
        return np.concatenate(
            [
                np.asarray(self.target_position_interval, dtype=object).reshape(-1),
                np.asarray(self.target_velocity_interval, dtype=object).reshape(-1),
            ]
        )

    @property
    def local_tail_bound(self) -> float:
        if self.reduced_target is None:
            return 0.0
        return self.reduced_target.local_tail_bound

    @property
    def max_step_tail_bound(self) -> float:
        if self.reduced_target is None:
            return 0.0
        return self.reduced_target.max_step_tail_bound

    @property
    def triple_collision_excluded(self) -> bool:
        return bool(self.reduced_target is not None and self.reduced_target.triple_collision_excluded)

    @property
    def triple_collision_status(self) -> str:
        if self.reduced_target is None:
            return "not_applicable"
        return self.reduced_target.triple_collision_status

    @property
    def triple_collision_exclusion_reason(self) -> str | None:
        if self.reduced_target is None:
            return None
        return self.reduced_target.triple_collision_exclusion_reason

    @property
    def triple_collision_undecided(self) -> bool:
        return self.triple_collision_status == "undecided"

    def target_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))


@dataclass(frozen=True)
class CompactTimeReducedCompactifiedSundmanEvaluation:
    """Bounded physical-time target evaluated through bounded Sundman charts."""

    compactification_certificate: CompactifiedTimeTargetCertificate
    time_taylor_certificate: CompactifiedTimeTaylorEvaluationCertificate
    evaluation: ReducedCompactifiedSundmanEvaluation

    @property
    def target_time(self) -> float:
        return self.compactification_certificate.physical_time

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.compactification_certificate.certified
            and self.time_taylor_certificate.certified
            and self.evaluation.proof_certified
        )

    @property
    def target_state_interval(self) -> Array:
        return self.evaluation.target_state_interval

    def target_state_contains(self, state: Array) -> bool:
        return self.evaluation.target_state_contains(state)


def evaluate_reduced_sundman_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    target_time: float,
    *,
    initial_radius: float = 0.0,
    order: int = 12,
    max_s_step: float = 0.03,
    distance_power: float = 1.0,
    max_steps: int = 10000,
    target_bisections: int = 60,
    step_shrink_bisections: int = 60,
    tail_certificate_mode: str = "cauchy",
) -> ReducedSundmanEvaluation:
    """Evaluate a certified target-time enclosure in inertial coordinates.

    The point state is first translated into a center-of-mass frame. A small
    interval box around the original initial state is translated by the same
    operation using interval arithmetic. The target-time solve is then
    performed only in the reduced frame, and the resulting enclosure is
    reconstructed with the affine inertial center-of-mass motion.
    """

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    target_time = float(target_time)
    initial_radius = float(initial_radius)
    if initial_radius < 0.0:
        raise ValueError("initial_radius cannot be negative")

    point_reduction = reduce_to_center_of_mass_frame(positions, velocities, masses)
    interval_reduction = reduce_interval_to_center_of_mass_frame(
        _interval_box_around(positions, initial_radius),
        _interval_box_around(velocities, initial_radius),
        masses,
    )

    if target_time == 0.0:
        target_positions, target_velocities = reconstruct_interval_from_center_of_mass_frame(
            interval_reduction.positions,
            interval_reduction.velocities,
            interval_reduction.center_position,
            interval_reduction.center_velocity,
            time=FloatInterval.point(0.0),
        )
        return ReducedSundmanEvaluation(
            target_time=target_time,
            point_reduction=point_reduction,
            interval_reduction=interval_reduction,
            reduced_target=None,
            target_position_interval=target_positions,
            target_velocity_interval=target_velocities,
        )

    reduced_target = continue_interval_sundman_to_time(
        interval_reduction.positions,
        interval_reduction.velocities,
        interval_reduction.masses,
        target_time,
        order=order,
        max_s_step=max_s_step,
        distance_power=distance_power,
        max_steps=max_steps,
        target_bisections=target_bisections,
        step_shrink_bisections=step_shrink_bisections,
        tail_certificate_mode=tail_certificate_mode,
    )
    reduced_positions, reduced_velocities = _split_state_interval(
        reduced_target.target_state_interval,
        body_count=positions.shape[0],
        dimension=positions.shape[1],
    )
    target_positions, target_velocities = reconstruct_interval_from_center_of_mass_frame(
        reduced_positions,
        reduced_velocities,
        interval_reduction.center_position,
        interval_reduction.center_velocity,
        time=FloatInterval.point(target_time),
    )
    return ReducedSundmanEvaluation(
        target_time=target_time,
        point_reduction=point_reduction,
        interval_reduction=interval_reduction,
        reduced_target=reduced_target,
        target_position_interval=target_positions,
        target_velocity_interval=target_velocities,
    )


def evaluate_compactified_reduced_sundman_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    compact_parameter: float,
    *,
    time_rate: float = 1.0,
    initial_radius: float = 0.0,
    order: int = 12,
    max_s_step: float = 0.03,
    distance_power: float = 1.0,
    max_steps: int = 10000,
    target_bisections: int = 60,
    step_shrink_bisections: int = 60,
    tail_certificate_mode: str = "cauchy",
    time_taylor_order: int = 17,
) -> CompactifiedReducedSundmanEvaluation:
    """Evaluate the composed solution at a bounded compact time parameter."""

    compactification_certificate = certify_compactified_time_target(
        compact_parameter,
        rate=time_rate,
    )
    time_taylor_certificate = certify_compactified_time_taylor_evaluation(
        compact_parameter,
        rate=time_rate,
        order=time_taylor_order,
    )
    evaluation = evaluate_reduced_sundman_solution(
        positions,
        velocities,
        masses,
        compactification_certificate.physical_time,
        initial_radius=initial_radius,
        order=order,
        max_s_step=max_s_step,
        distance_power=distance_power,
        max_steps=max_steps,
        target_bisections=target_bisections,
        step_shrink_bisections=step_shrink_bisections,
        tail_certificate_mode=tail_certificate_mode,
    )
    return CompactifiedReducedSundmanEvaluation(
        compactification_certificate=compactification_certificate,
        time_taylor_certificate=time_taylor_certificate,
        evaluation=evaluation,
    )


def evaluate_reduced_compactified_sundman_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    target_time: float,
    *,
    initial_radius: float = 0.0,
    order: int = 10,
    sundman_rate: float = 1.0,
    max_compact_step: float = 0.03,
    radius_fraction: float = 0.2,
    distance_power: float = 1.0,
    max_steps: int = 10000,
    target_bisections: int = 60,
    step_shrink_bisections: int = 60,
    guard_order: int = 6,
) -> ReducedCompactifiedSundmanEvaluation:
    """Evaluate a target-time enclosure through compactified Sundman charts."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    target_time = float(target_time)
    initial_radius = float(initial_radius)
    if initial_radius < 0.0:
        raise ValueError("initial_radius cannot be negative")

    point_reduction = reduce_to_center_of_mass_frame(positions, velocities, masses)
    interval_reduction = reduce_interval_to_center_of_mass_frame(
        _interval_box_around(positions, initial_radius),
        _interval_box_around(velocities, initial_radius),
        masses,
    )

    if target_time == 0.0:
        target_positions, target_velocities = reconstruct_interval_from_center_of_mass_frame(
            interval_reduction.positions,
            interval_reduction.velocities,
            interval_reduction.center_position,
            interval_reduction.center_velocity,
            time=FloatInterval.point(0.0),
        )
        return ReducedCompactifiedSundmanEvaluation(
            target_time=target_time,
            point_reduction=point_reduction,
            interval_reduction=interval_reduction,
            reduced_target=None,
            target_position_interval=target_positions,
            target_velocity_interval=target_velocities,
        )

    reduced_target = continue_interval_compactified_sundman_to_time(
        interval_reduction.positions,
        interval_reduction.velocities,
        interval_reduction.masses,
        target_time,
        order=order,
        sundman_rate=sundman_rate,
        distance_power=distance_power,
        max_compact_step=max_compact_step,
        radius_fraction=radius_fraction,
        max_steps=max_steps,
        target_bisections=target_bisections,
        step_shrink_bisections=step_shrink_bisections,
        guard_order=guard_order,
    )
    reduced_positions, reduced_velocities = _split_state_interval(
        reduced_target.target_state_interval,
        body_count=positions.shape[0],
        dimension=positions.shape[1],
    )
    target_positions, target_velocities = reconstruct_interval_from_center_of_mass_frame(
        reduced_positions,
        reduced_velocities,
        interval_reduction.center_position,
        interval_reduction.center_velocity,
        time=FloatInterval.point(target_time),
    )
    return ReducedCompactifiedSundmanEvaluation(
        target_time=target_time,
        point_reduction=point_reduction,
        interval_reduction=interval_reduction,
        reduced_target=reduced_target,
        target_position_interval=target_positions,
        target_velocity_interval=target_velocities,
    )


def evaluate_compact_time_reduced_compactified_sundman_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    compact_parameter: float,
    *,
    time_rate: float = 1.0,
    initial_radius: float = 0.0,
    order: int = 10,
    sundman_rate: float = 1.0,
    max_compact_step: float = 0.03,
    radius_fraction: float = 0.2,
    distance_power: float = 1.0,
    max_steps: int = 10000,
    target_bisections: int = 60,
    step_shrink_bisections: int = 60,
    guard_order: int = 6,
    time_taylor_order: int = 17,
) -> CompactTimeReducedCompactifiedSundmanEvaluation:
    """Evaluate a bounded physical-time target using compactified Sundman charts."""

    compactification_certificate = certify_compactified_time_target(
        compact_parameter,
        rate=time_rate,
    )
    time_taylor_certificate = certify_compactified_time_taylor_evaluation(
        compact_parameter,
        rate=time_rate,
        order=time_taylor_order,
    )
    evaluation = evaluate_reduced_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        compactification_certificate.physical_time,
        initial_radius=initial_radius,
        order=order,
        sundman_rate=sundman_rate,
        max_compact_step=max_compact_step,
        radius_fraction=radius_fraction,
        distance_power=distance_power,
        max_steps=max_steps,
        target_bisections=target_bisections,
        step_shrink_bisections=step_shrink_bisections,
        guard_order=guard_order,
    )
    return CompactTimeReducedCompactifiedSundmanEvaluation(
        compactification_certificate=compactification_certificate,
        time_taylor_certificate=time_taylor_certificate,
        evaluation=evaluation,
    )


def evaluate_planar_validated_atlas_solution(
    masses: Array,
    positions: Array,
    velocities: Array,
    target_time: float,
    *,
    initial_radius: float = 0.0,
    ordinary_order: int = 10,
    binary_order: int = 10,
    max_time_step: float = 0.03,
    max_binary_s_step: float = 0.03,
    binary_distance_threshold: float = 0.08,
    binary_exit_distance: float | None = None,
    safety: float = 0.08,
    max_steps: int = 10000,
    tail_guard_order: int = 0,
    tail_certificate_mode: str = "cauchy",
    initial_state_interval_union: tuple[tuple[tuple[float, float], ...], ...] | None = None,
) -> ValidatedAtlasSolution:
    """Evaluate a positive-time planar target through the hybrid atlas pipeline."""

    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    target_time = float(target_time)
    initial_radius = float(initial_radius)
    if positions.shape != (3, 2) or velocities.shape != (3, 2):
        raise ValueError("planar validated atlas requires positions and velocities with shape (3, 2)")
    if target_time == 0.0:
        raise ValueError("planar hybrid validated atlas expects a nonzero target time")
    if initial_radius < 0.0:
        raise ValueError("initial_radius cannot be negative")
    if initial_state_interval_union is not None and initial_radius != 0.0:
        raise ValueError("initial_radius must be zero when initial_state_interval_union is supplied")
    if target_time < 0.0:
        reversed_atlas = evaluate_planar_validated_atlas_solution(
            masses,
            positions,
            -velocities,
            -target_time,
            initial_radius=initial_radius,
            ordinary_order=ordinary_order,
            binary_order=binary_order,
            max_time_step=max_time_step,
            max_binary_s_step=max_binary_s_step,
            binary_distance_threshold=binary_distance_threshold,
            binary_exit_distance=binary_exit_distance,
            safety=safety,
            max_steps=max_steps,
            tail_guard_order=tail_guard_order,
            tail_certificate_mode=tail_certificate_mode,
            initial_state_interval_union=_time_reverse_planar_state_interval_union(
                initial_state_interval_union
            )
            if initial_state_interval_union is not None
            else None,
        )
        return time_reverse_validated_atlas_solution(
            reversed_atlas,
            source="evaluate_unrestricted_solution_time_reversal",
        )

    initial_state_interval = None
    if initial_state_interval_union is None:
        initial_state_interval = _planar_state_interval_around(
            positions,
            velocities,
            initial_radius,
        )
    hybrid_solution = continue_hybrid(
        positions,
        velocities,
        masses,
        target_time,
        ordinary_order=ordinary_order,
        binary_order=binary_order,
        max_time_step=max_time_step,
        max_binary_s_step=max_binary_s_step,
        binary_distance_threshold=binary_distance_threshold,
        binary_exit_distance=binary_exit_distance,
        safety=safety,
        max_steps=max_steps,
        tail_guard_order=tail_guard_order,
        tail_certificate_mode=tail_certificate_mode,
        initial_state_interval=initial_state_interval,
        initial_state_interval_union=initial_state_interval_union,
        require_interval_chart_certification=True,
    )
    return validated_atlas_from_hybrid_solution(
        hybrid_solution,
        target_time=target_time,
        source="evaluate_unrestricted_solution",
    )


def evaluate_spatial_ks_validated_atlas_solution(
    masses: Array,
    positions: Array,
    velocities: Array,
    target_time: float,
    *,
    pair: tuple[int, int],
    enter_distance: float,
    entry_time_upper: float,
    branch: str = "positive_x",
    exit_rho: float,
    s_upper: float,
    initial_radius: float = 0.0,
    order: int = 18,
    guard_order: int = 6,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    competing_pair_min_distance_required: float = 0.0,
) -> ValidatedAtlasSolution:
    """Evaluate an explicitly selected local spatial KS handoff.

    This is not an automatic spatial binary classifier.  The caller supplies a
    selected pair and event thresholds; the constructor then derives the entry
    event, KS branch lift, rho-exit event, local collision-domain certificate,
    and target-time enclosure from interval data.
    """

    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    target_time = float(target_time)
    initial_radius = float(initial_radius)
    if positions.shape != (3, 3) or velocities.shape != (3, 3):
        raise ValueError("spatial KS validated atlas requires positions and velocities with shape (3, 3)")
    if target_time < 0.0:
        reversed_atlas = evaluate_spatial_ks_validated_atlas_solution(
            masses,
            positions,
            -velocities,
            -target_time,
            pair=pair,
            enter_distance=enter_distance,
            entry_time_upper=entry_time_upper,
            branch=branch,
            exit_rho=exit_rho,
            s_upper=s_upper,
            initial_radius=initial_radius,
            order=order,
            guard_order=guard_order,
            ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
            ordinary_handoff_max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
            ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
            ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
            ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
            competing_pair_min_distance_required=competing_pair_min_distance_required,
        )
        return time_reverse_validated_atlas_solution(
            reversed_atlas,
            source="evaluate_unrestricted_solution_time_reversal",
        )
    if target_time == 0.0:
        raise ValueError("spatial KS validated atlas currently advances positive target times")
    if initial_radius < 0.0:
        raise ValueError("initial_radius cannot be negative")

    state_interval = _spatial_state_interval_around(positions, velocities, initial_radius)
    best_atlas = None
    last_error = None
    for candidate_exit_rho in _spatial_ks_ordinary_handoff_exit_rho_candidates(
        exit_rho,
        ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
    ):
        candidate_s_upper = max(
            float(s_upper),
            _SPATIAL_KS_AUTO_EXIT_SEARCH_FACTOR * float(np.sqrt(candidate_exit_rho)),
        )
        try:
            candidate_atlas = validated_atlas_from_spatial_ordinary_ks_handoff(
                state_interval,
                masses,
                pair=tuple(pair),
                enter_distance=float(enter_distance),
                entry_time_upper=float(entry_time_upper),
                branch=str(branch),
                exit_rho=float(candidate_exit_rho),
                s_upper=candidate_s_upper,
                target_time=target_time,
                retained_order=int(order),
                guard_order=int(guard_order),
                ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
                ordinary_handoff_max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
                ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
                ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
                ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
                competing_pair_min_distance_required=competing_pair_min_distance_required,
                source="evaluate_unrestricted_solution",
            )
        except (RuntimeError, ValueError) as error:
            last_error = error
            continue
        if _spatial_ordinary_after_ks_handoff_certified(candidate_atlas):
            return candidate_atlas
        if best_atlas is None:
            best_atlas = candidate_atlas
        if candidate_atlas.proof_certified:
            best_atlas = candidate_atlas
    if best_atlas is not None:
        return best_atlas
    if last_error is not None:
        raise last_error
    raise ValueError("spatial KS validated atlas could not construct any exit candidate")


def evaluate_unrestricted_solution(
    masses: Array,
    positions: Array,
    velocities: Array,
    target_time: float,
    *,
    method: str = "auto",
    initial_radius: float = 0.0,
    order: int = 10,
    max_s_step: float = 0.03,
    sundman_rate: float = 1.0,
    max_compact_step: float = 0.03,
    radius_fraction: float = 0.2,
    distance_power: float = 1.0,
    max_steps: int = 10000,
    target_bisections: int = 60,
    step_shrink_bisections: int = 60,
    guard_order: int = 6,
    tail_certificate_mode: str = "cauchy",
    binary_distance_threshold: float = 0.08,
    binary_exit_distance: float | None = None,
    max_binary_s_step: float | None = None,
    hybrid_safety: float = 0.08,
    spatial_binary_auto: bool = True,
    spatial_binary_pair: tuple[int, int] | None = None,
    spatial_binary_enter_distance: float | None = None,
    spatial_binary_entry_time_upper: float | None = None,
    spatial_binary_branch: str = "positive_x",
    spatial_binary_exit_rho: float | None = None,
    spatial_binary_s_upper: float | None = None,
    spatial_binary_ordinary_handoff_min_pair_distance_required: float | None = None,
    spatial_binary_ordinary_handoff_max_acceleration_bound: float = np.inf,
    spatial_binary_ordinary_handoff_min_cauchy_radius: float = 0.0,
    spatial_binary_ordinary_handoff_max_residual_bound: float = np.inf,
    spatial_binary_ordinary_handoff_max_tail_bound: float = np.inf,
    spatial_binary_max_competing_events: int | None = None,
    planar_initial_state_interval_union: tuple[tuple[tuple[float, float], ...], ...] | None = None,
) -> ReducedCompactifiedSundmanEvaluation | ReducedSundmanEvaluation | ValidatedAtlasSolution:
    """Evaluate a certified finite-time enclosure for arbitrary noncollision data.

    This is the Phase A finite-time entry point for the unrestricted problem. It
    does not claim a global closed-form theorem; it composes the existing
    center-of-mass reduction, Sundman or compactified-Sundman construction,
    inertial projection, and interval verification for one finite physical
    target time.
    """

    if method not in {"auto", "compactified_sundman", "sundman", "validated_atlas"}:
        raise ValueError(
            "method must be 'auto', 'compactified_sundman', 'sundman', or 'validated_atlas'"
        )
    if method != "validated_atlas" and planar_initial_state_interval_union is not None:
        raise ValueError("planar_initial_state_interval_union is only supported by method='validated_atlas'")
    if (
        method == "validated_atlas"
        and planar_initial_state_interval_union is not None
        and not _can_use_planar_hybrid_atlas(positions, velocities, target_time)
    ):
        raise ValueError("planar_initial_state_interval_union requires planar data and nonzero target time")

    if method == "validated_atlas":
        selector_attempts: list[FiniteTimeChartSelectorAttempt] = []
        spatial_competing_event_budgets = _spatial_competing_event_budgets(
            spatial_binary_max_competing_events
        )
        effective_ordinary_handoff_min_pair_distance_required = (
            _effective_spatial_ordinary_handoff_min_pair_distance_required(
                spatial_binary_ordinary_handoff_min_pair_distance_required,
                binary_distance_threshold=binary_distance_threshold,
            )
        )
        effective_competing_pair_min_distance_required = (
            _effective_spatial_competing_pair_min_distance_required(
                binary_distance_threshold
            )
        )
        if _spatial_ks_options_supplied(
            spatial_binary_pair=spatial_binary_pair,
            spatial_binary_enter_distance=spatial_binary_enter_distance,
            spatial_binary_entry_time_upper=spatial_binary_entry_time_upper,
            spatial_binary_exit_rho=spatial_binary_exit_rho,
            spatial_binary_s_upper=spatial_binary_s_upper,
        ):
            if (
                spatial_binary_pair is None
                or spatial_binary_enter_distance is None
                or spatial_binary_entry_time_upper is None
                or spatial_binary_exit_rho is None
                or spatial_binary_s_upper is None
            ):
                raise ValueError(
                    "spatial KS validated atlas requires spatial_binary_pair, "
                    "spatial_binary_enter_distance, spatial_binary_entry_time_upper, "
                    "spatial_binary_exit_rho, and spatial_binary_s_upper"
                )
            explicit_error: Exception | None = None
            try:
                explicit_solution = evaluate_spatial_ks_validated_atlas_solution(
                    masses,
                    positions,
                    velocities,
                    target_time,
                    pair=spatial_binary_pair,
                    enter_distance=spatial_binary_enter_distance,
                    entry_time_upper=spatial_binary_entry_time_upper,
                    branch=spatial_binary_branch,
                    exit_rho=spatial_binary_exit_rho,
                    s_upper=spatial_binary_s_upper,
                    initial_radius=initial_radius,
                    order=order,
                    guard_order=guard_order,
                    ordinary_handoff_min_pair_distance_required=(
                        effective_ordinary_handoff_min_pair_distance_required
                    ),
                    ordinary_handoff_max_acceleration_bound=(
                        spatial_binary_ordinary_handoff_max_acceleration_bound
                    ),
                    ordinary_handoff_min_cauchy_radius=(
                        spatial_binary_ordinary_handoff_min_cauchy_radius
                    ),
                    ordinary_handoff_max_residual_bound=(
                        spatial_binary_ordinary_handoff_max_residual_bound
                    ),
                    ordinary_handoff_max_tail_bound=(
                        spatial_binary_ordinary_handoff_max_tail_bound
                    ),
                    competing_pair_min_distance_required=(
                        effective_competing_pair_min_distance_required
                    ),
                )
            except (RuntimeError, ValueError) as error:
                explicit_solution = None
                explicit_error = error
            if explicit_solution is None or not explicit_solution.proof_certified:
                competing_explicit_solution = (
                    _try_with_spatial_competing_event_budget_retries(
                        lambda max_competing_events: _try_evaluate_auto_spatial_ordinary_ks_competing_handoff(
                            masses,
                            positions,
                            velocities,
                            target_time,
                            state_interval=_spatial_state_interval_around(
                                positions,
                                velocities,
                                initial_radius,
                            ),
                            selected_pair=spatial_binary_pair,
                            selected_enter_distance=spatial_binary_enter_distance,
                            selected_entry_time_upper=spatial_binary_entry_time_upper,
                            selected_branch=spatial_binary_branch,
                            binary_distance_threshold=binary_distance_threshold,
                            s_upper=spatial_binary_s_upper,
                            order=order,
                            guard_order=guard_order,
                            ordinary_handoff_min_pair_distance_required=(
                                effective_ordinary_handoff_min_pair_distance_required
                            ),
                            ordinary_handoff_max_acceleration_bound=(
                                spatial_binary_ordinary_handoff_max_acceleration_bound
                            ),
                            ordinary_handoff_min_cauchy_radius=(
                                spatial_binary_ordinary_handoff_min_cauchy_radius
                            ),
                            ordinary_handoff_max_residual_bound=(
                                spatial_binary_ordinary_handoff_max_residual_bound
                            ),
                            ordinary_handoff_max_tail_bound=(
                                spatial_binary_ordinary_handoff_max_tail_bound
                            ),
                            max_competing_events=max_competing_events,
                        ),
                        spatial_competing_event_budgets,
                    )
                )
                if competing_explicit_solution is not None:
                    explicit_solution = competing_explicit_solution
                elif explicit_error is not None:
                    raise explicit_error
                assert explicit_solution is not None
            selected_route_id = _spatial_ks_validated_selector_route_id(
                explicit_solution,
                default_route_id="explicit_spatial_ks",
            )
            return _attach_finite_time_selector_trace(
                explicit_solution,
                selected_route_id=selected_route_id,
                attempts=(
                    FiniteTimeChartSelectorAttempt(
                        route_id=selected_route_id,
                        attempted=True,
                        selected=True,
                        certified=explicit_solution.proof_certified,
                        reason=_spatial_ks_selector_reason(
                            selected_route_id,
                            explicit=True,
                        ),
                        missing_obligations=_validated_atlas_missing_obligations(
                            explicit_solution
                        ),
                    ),
                ),
            )
        if spatial_binary_auto:
            try:
                spatial_ks_solution = _try_with_spatial_competing_event_budget_retries(
                    lambda max_competing_events: _try_evaluate_auto_spatial_ks_validated_atlas(
                        masses,
                        positions,
                        velocities,
                        target_time,
                        initial_radius=initial_radius,
                        order=order,
                        guard_order=guard_order,
                        binary_distance_threshold=binary_distance_threshold,
                        s_upper=max(
                            _SPATIAL_KS_AUTO_S_UPPER,
                            float(max_binary_s_step)
                            if max_binary_s_step is not None
                            else float(max_s_step),
                        ),
                        ordinary_handoff_min_pair_distance_required=(
                            effective_ordinary_handoff_min_pair_distance_required
                        ),
                        ordinary_handoff_max_acceleration_bound=(
                            spatial_binary_ordinary_handoff_max_acceleration_bound
                        ),
                        ordinary_handoff_min_cauchy_radius=(
                            spatial_binary_ordinary_handoff_min_cauchy_radius
                        ),
                        ordinary_handoff_max_residual_bound=(
                            spatial_binary_ordinary_handoff_max_residual_bound
                        ),
                        ordinary_handoff_max_tail_bound=(
                            spatial_binary_ordinary_handoff_max_tail_bound
                        ),
                        competing_pair_min_distance_required=(
                            effective_competing_pair_min_distance_required
                        ),
                        max_competing_events=max_competing_events,
                    ),
                    spatial_competing_event_budgets,
                )
            except FiniteTimeKSLoopBlocked as spatial_error:
                spatial_ks_solution = None
                selector_attempts.append(
                    FiniteTimeChartSelectorAttempt(
                        route_id="auto_spatial_ks",
                        attempted=True,
                        selected=False,
                        certified=False,
                        reason="automatic spatial KS loop reached a typed stopping obligation",
                        missing_obligations=spatial_error.missing_obligations,
                        blocks_fallback_certification=True,
                    )
                )
            except (RuntimeError, ValueError) as spatial_error:
                spatial_ks_solution = None
                selector_attempts.append(
                    FiniteTimeChartSelectorAttempt(
                        route_id="auto_spatial_ks",
                        attempted=True,
                        selected=False,
                        certified=False,
                        reason="automatic spatial KS constructor failed",
                        missing_obligations=(str(spatial_error),),
                        blocks_fallback_certification=True,
                    )
                )
            if spatial_ks_solution is not None:
                spatial_missing = _validated_atlas_missing_obligations(
                    spatial_ks_solution
                )
                if spatial_ks_solution.proof_certified:
                    selected_route_id = _spatial_ks_validated_selector_route_id(
                        spatial_ks_solution,
                        default_route_id="auto_spatial_ks",
                    )
                    selector_attempts.append(
                        FiniteTimeChartSelectorAttempt(
                            route_id=selected_route_id,
                            attempted=True,
                            selected=True,
                            certified=True,
                            reason=_spatial_ks_selector_reason(
                                selected_route_id,
                                explicit=False,
                            ),
                            missing_obligations=spatial_missing,
                        )
                    )
                    return _attach_finite_time_selector_trace(
                        spatial_ks_solution,
                        selected_route_id=selected_route_id,
                        attempts=tuple(selector_attempts),
                    )
                selector_attempts.append(
                    FiniteTimeChartSelectorAttempt(
                        route_id="auto_spatial_ks",
                        attempted=True,
                        selected=False,
                        certified=False,
                        reason="automatic spatial KS constructor returned an uncertified atlas",
                        missing_obligations=spatial_missing,
                        blocks_fallback_certification=True,
                    )
                )
            if not any(attempt.route_id == "auto_spatial_ks" for attempt in selector_attempts):
                selector_attempts.append(
                    FiniteTimeChartSelectorAttempt(
                        route_id="auto_spatial_ks",
                        attempted=False,
                        selected=False,
                        certified=False,
                        reason="no certified close spatial binary candidate",
                    )
                )
        else:
            selector_attempts.append(
                FiniteTimeChartSelectorAttempt(
                    route_id="auto_spatial_ks",
                    attempted=False,
                    selected=False,
                    certified=False,
                    reason="automatic spatial KS selector disabled",
                )
            )
        close_binary_blocker = _spatial_close_binary_fallback_blocker(
            positions,
            velocities,
            masses,
            target_time,
            initial_radius=initial_radius,
            binary_distance_threshold=binary_distance_threshold,
        )
        if close_binary_blocker is not None and not any(
            attempt.route_id == "auto_spatial_ks" and attempt.selected
            for attempt in selector_attempts
        ):
            if spatial_binary_auto:
                branch_union_solution, branch_union_missing = (
                    _try_evaluate_spatial_close_pair_branch_union(
                        masses,
                        positions,
                        velocities,
                        target_time,
                        branch_partition=getattr(close_binary_blocker, "branch_partition", None),
                        order=order,
                        guard_order=guard_order,
                        s_endpoint=(
                            float(max_binary_s_step)
                            if max_binary_s_step is not None
                            else float(max_s_step)
                        ),
                        ordinary_handoff_min_pair_distance_required=(
                            effective_ordinary_handoff_min_pair_distance_required
                        ),
                        ordinary_handoff_max_acceleration_bound=(
                            spatial_binary_ordinary_handoff_max_acceleration_bound
                        ),
                        ordinary_handoff_min_cauchy_radius=(
                            spatial_binary_ordinary_handoff_min_cauchy_radius
                        ),
                        ordinary_handoff_max_residual_bound=(
                            spatial_binary_ordinary_handoff_max_residual_bound
                        ),
                        ordinary_handoff_max_tail_bound=(
                            spatial_binary_ordinary_handoff_max_tail_bound
                        ),
                        max_competing_events=spatial_binary_max_competing_events,
                    )
                )
                if branch_union_solution is not None and branch_union_solution.proof_certified:
                    selector_attempts.append(
                        FiniteTimeChartSelectorAttempt(
                            route_id="spatial_branch_union",
                            attempted=True,
                            selected=True,
                            certified=True,
                            reason=(
                                "certified close-pair partition leaves were consumed "
                                "by spatial KS member charts"
                            ),
                            branch_partition=getattr(close_binary_blocker, "branch_partition", None),
                        )
                    )
                    return _attach_finite_time_selector_trace(
                        branch_union_solution,
                        selected_route_id="spatial_branch_union",
                        attempts=tuple(selector_attempts),
                    )
                if branch_union_missing:
                    selector_attempts.append(
                        FiniteTimeChartSelectorAttempt(
                            route_id="spatial_branch_union",
                            attempted=True,
                            selected=False,
                            certified=False,
                            reason=(
                                "certified close-pair partition leaves did not "
                                "produce a proof-certified spatial KS branch union"
                            ),
                            missing_obligations=branch_union_missing,
                            blocks_fallback_certification=True,
                            branch_partition=getattr(close_binary_blocker, "branch_partition", None),
                        )
                    )
            selector_attempts.append(close_binary_blocker)
        blocking_failures = _blocking_selector_failures(selector_attempts)
        if blocking_failures:
            raise FiniteTimeChartSelectorError(blocking_failures)
        if _can_use_planar_hybrid_atlas(positions, velocities, target_time):
            try:
                planar_solution = evaluate_planar_validated_atlas_solution(
                    masses,
                    positions,
                    velocities,
                    target_time,
                    initial_radius=initial_radius,
                    ordinary_order=order,
                    binary_order=order,
                    max_time_step=max_compact_step,
                    max_binary_s_step=max_s_step if max_binary_s_step is None else max_binary_s_step,
                    binary_distance_threshold=binary_distance_threshold,
                    binary_exit_distance=binary_exit_distance,
                    safety=hybrid_safety,
                    max_steps=max_steps,
                    tail_guard_order=guard_order if tail_certificate_mode == "guarded" else 0,
                    tail_certificate_mode=tail_certificate_mode,
                    initial_state_interval_union=planar_initial_state_interval_union,
                )
                selector_attempts.append(
                    FiniteTimeChartSelectorAttempt(
                        route_id="planar_hybrid",
                        attempted=True,
                        selected=True,
                        certified=planar_solution.proof_certified,
                        reason="planar data routed through ordinary/Levi-Civita hybrid atlas",
                        missing_obligations=_validated_atlas_missing_obligations(
                            planar_solution
                        ),
                    )
                )
                return _attach_finite_time_selector_trace(
                    planar_solution,
                    selected_route_id="planar_hybrid",
                    attempts=tuple(selector_attempts),
                )
            except (RuntimeError, ValueError) as planar_error:
                selector_attempts.append(
                    FiniteTimeChartSelectorAttempt(
                        route_id="planar_hybrid",
                        attempted=True,
                        selected=False,
                        certified=False,
                        reason="planar hybrid constructor failed",
                        missing_obligations=(str(planar_error),),
                        blocks_fallback_certification=True,
                    )
                )
        else:
            selector_attempts.append(
                FiniteTimeChartSelectorAttempt(
                    route_id="planar_hybrid",
                    attempted=False,
                    selected=False,
                    certified=False,
                    reason="state is not planar or target time is zero",
                )
            )
        blocking_failures = _blocking_selector_failures(selector_attempts)
        if blocking_failures:
            raise FiniteTimeChartSelectorError(blocking_failures)
        evaluation = evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            target_time,
            method="auto",
            initial_radius=initial_radius,
            order=order,
            max_s_step=max_s_step,
            sundman_rate=sundman_rate,
            max_compact_step=max_compact_step,
            radius_fraction=radius_fraction,
            distance_power=distance_power,
            max_steps=max_steps,
            target_bisections=target_bisections,
            step_shrink_bisections=step_shrink_bisections,
            guard_order=guard_order,
            tail_certificate_mode=tail_certificate_mode,
            binary_distance_threshold=binary_distance_threshold,
            binary_exit_distance=binary_exit_distance,
            max_binary_s_step=max_binary_s_step,
            hybrid_safety=hybrid_safety,
            spatial_binary_auto=spatial_binary_auto,
            spatial_binary_pair=spatial_binary_pair,
            spatial_binary_enter_distance=spatial_binary_enter_distance,
            spatial_binary_entry_time_upper=spatial_binary_entry_time_upper,
            spatial_binary_branch=spatial_binary_branch,
            spatial_binary_exit_rho=spatial_binary_exit_rho,
            spatial_binary_s_upper=spatial_binary_s_upper,
            spatial_binary_ordinary_handoff_min_pair_distance_required=(
                effective_ordinary_handoff_min_pair_distance_required
            ),
            spatial_binary_ordinary_handoff_max_acceleration_bound=(
                spatial_binary_ordinary_handoff_max_acceleration_bound
            ),
            spatial_binary_ordinary_handoff_min_cauchy_radius=(
                spatial_binary_ordinary_handoff_min_cauchy_radius
            ),
            spatial_binary_ordinary_handoff_max_residual_bound=(
                spatial_binary_ordinary_handoff_max_residual_bound
            ),
            spatial_binary_ordinary_handoff_max_tail_bound=(
                spatial_binary_ordinary_handoff_max_tail_bound
            ),
            spatial_binary_max_competing_events=spatial_binary_max_competing_events,
        )
        fallback_solution = validated_atlas_from_unrestricted_evaluation(
            evaluation,
            masses=masses,
            target_time=target_time,
        )
        fallback_route = _fallback_validated_atlas_route_id(evaluation)
        selector_attempts.append(
            FiniteTimeChartSelectorAttempt(
                route_id=fallback_route,
                attempted=True,
                selected=True,
                certified=fallback_solution.proof_certified,
                reason="regularized finite-time selector fell back to reduced Sundman/compact-Sundman atlas",
                missing_obligations=_validated_atlas_missing_obligations(
                    fallback_solution
                ),
            )
        )
        return _attach_finite_time_selector_trace(
            fallback_solution,
            selected_route_id=fallback_route,
            attempts=tuple(selector_attempts),
        )

    if method in {"auto", "compactified_sundman"}:
        try:
            return evaluate_reduced_compactified_sundman_solution(
                positions,
                velocities,
                masses,
                target_time,
                initial_radius=initial_radius,
                order=order,
                sundman_rate=sundman_rate,
                max_compact_step=max_compact_step,
                radius_fraction=radius_fraction,
                distance_power=distance_power,
                max_steps=max_steps,
                target_bisections=target_bisections,
                step_shrink_bisections=step_shrink_bisections,
                guard_order=guard_order,
            )
        except (RuntimeError, ValueError) as compactified_error:
            if method == "compactified_sundman" or not _should_fallback_from_compactified_error(
                compactified_error
            ):
                raise

    if method in {"auto", "sundman"}:
        return evaluate_reduced_sundman_solution(
            positions,
            velocities,
            masses,
            target_time,
            initial_radius=initial_radius,
            order=order,
            distance_power=distance_power,
            max_steps=max_steps,
            max_s_step=max_s_step,
            target_bisections=target_bisections,
            step_shrink_bisections=step_shrink_bisections,
            tail_certificate_mode=tail_certificate_mode,
        )

    raise AssertionError("unreachable method dispatch")


def _attach_finite_time_selector_trace(
    atlas: ValidatedAtlasSolution,
    *,
    selected_route_id: str,
    attempts: tuple[FiniteTimeChartSelectorAttempt, ...],
) -> ValidatedAtlasSolution:
    trace = FiniteTimeChartSelectorTrace(
        selected_route_id=selected_route_id,
        attempts=attempts,
        atlas_binding_token=finite_time_selector_trace_binding_token(atlas),
    )
    ledger = ProofLedger(
        entries=(
            *atlas.proof_ledger.entries,
            ProofLedgerEntry(
                "finite_time_chart_selector",
                trace.certified,
                "evaluate_unrestricted_solution",
                detail=(
                    f"selected {selected_route_id}; considered "
                    + ", ".join(attempt.route_id for attempt in attempts)
                ),
            ),
        )
    )
    return replace(atlas, proof_ledger=ledger, selector_trace=trace)


def _validated_atlas_missing_obligations(
    atlas: ValidatedAtlasSolution,
) -> tuple[str, ...]:
    missing = list(
        getattr(
            atlas,
            "missing_certification_obligations",
            getattr(atlas.proof_ledger, "missing_required_obligations", ()),
        )
    )
    for entry in getattr(atlas.proof_ledger, "entries", ()):
        if (
            str(getattr(entry, "name", "")) == "spatial_collision_policy_scope"
            and not bool(getattr(entry, "certified", False))
            and str(getattr(entry, "detail", ""))
        ):
            missing.extend(
                part.strip()
                for part in str(getattr(entry, "detail", "")).split(";")
                if part.strip()
            )
    return tuple(dict.fromkeys(missing))


def _spatial_competing_event_budgets(requested: int | None) -> tuple[int, ...]:
    """Finite KS loop budgets used by the public validated-atlas selector.

    Passing an explicit value keeps the old strict behavior.  The default path
    performs a bounded constructor retry, which removes failures caused only by
    an undersized engineering budget without claiming arbitrary-input loop
    termination.
    """

    if requested is not None:
        return (max(0, int(requested)),)
    return _SPATIAL_KS_DEFAULT_COMPETING_EVENT_BUDGETS


def _ks_loop_blocked_only_by_repeat_budget(
    error: FiniteTimeKSLoopBlocked,
) -> bool:
    return set(error.missing_obligations) == {"finite_time_atlas_loop_repeat_budget"}


def _try_with_spatial_competing_event_budget_retries(
    builder,
    budgets: tuple[int, ...],
):
    """Retry a spatial-KS constructor only for pure repeat-budget exhaustion."""

    last_repeat_budget_error: FiniteTimeKSLoopBlocked | None = None
    for max_competing_events in budgets:
        try:
            return builder(max_competing_events)
        except FiniteTimeKSLoopBlocked as error:
            if not _ks_loop_blocked_only_by_repeat_budget(error):
                raise
            last_repeat_budget_error = error
    if last_repeat_budget_error is not None:
        raise last_repeat_budget_error
    return None


def _blocking_selector_failures(
    attempts: list[FiniteTimeChartSelectorAttempt] | tuple[FiniteTimeChartSelectorAttempt, ...],
) -> tuple[FiniteTimeChartSelectorAttempt, ...]:
    return tuple(
        attempt
        for attempt in attempts
        if attempt.blocks_fallback_certification and not attempt.certified
    )


def _format_finite_time_selector_failure(
    blocking_failures: tuple[FiniteTimeChartSelectorAttempt, ...],
) -> str:
    return (
        "finite-time chart selector stopped after required regularized route failed: "
        + "; ".join(
            f"{attempt.route_id}: {', '.join(attempt.missing_obligations) or attempt.reason}"
            for attempt in blocking_failures
        )
    )


def _effective_spatial_ordinary_handoff_min_pair_distance_required(
    requested: float | None,
    *,
    binary_distance_threshold: float,
) -> float:
    if requested is not None:
        return float(requested)
    threshold = float(binary_distance_threshold)
    if np.isfinite(threshold) and threshold > 0.0:
        return float(_SPATIAL_KS_AUTO_EXIT_MULTIPLIER * threshold)
    return 0.0


def _effective_spatial_competing_pair_min_distance_required(
    binary_distance_threshold: float,
) -> float:
    threshold = float(binary_distance_threshold)
    if np.isfinite(threshold) and threshold > 0.0:
        return threshold
    return 0.0


def _spatial_ks_ordinary_handoff_exit_rho_candidates(
    exit_rho: float,
    *,
    ordinary_handoff_min_pair_distance_required: float,
) -> tuple[float, ...]:
    """Candidate selected-binary distances for leaving KS into ordinary charts."""

    base = float(exit_rho)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("exit_rho must be positive and finite")
    candidates = [base]
    handoff_floor = float(ordinary_handoff_min_pair_distance_required)
    if np.isfinite(handoff_floor) and handoff_floor > base:
        candidates.append(float(handoff_floor * _SPATIAL_KS_ORDINARY_HANDOFF_EXIT_RETRY_FACTOR))
    deduped: list[float] = []
    for candidate in candidates:
        if not np.isfinite(candidate) or candidate <= 0.0:
            continue
        if not any(np.isclose(candidate, seen, rtol=1.0e-12, atol=1.0e-15) for seen in deduped):
            deduped.append(candidate)
    return tuple(deduped)


def _spatial_ordinary_after_ks_handoff_certified(atlas: object) -> bool:
    charts = tuple(getattr(atlas, "charts", ()))
    if not any(chart.chart_type == "spatial_ordinary_taylor_after_ks" for chart in charts):
        return False
    evaluation = getattr(atlas, "evaluation", None)
    ks_evaluation = getattr(evaluation, "ks_evaluation", evaluation)
    handoff = getattr(ks_evaluation, "ordinary_handoff_admissibility", None)
    return bool(
        getattr(atlas, "proof_certified", False)
        and handoff is not None
        and getattr(handoff, "certified", False)
    )


def _fallback_validated_atlas_route_id(evaluation: object) -> str:
    name = evaluation.__class__.__name__
    if name == "ReducedCompactifiedSundmanEvaluation":
        return "compactified_sundman"
    if name == "ReducedSundmanEvaluation":
        return "sundman"
    return "unrestricted_reduced_evaluation"


def _should_fallback_from_compactified_error(error: Exception) -> bool:
    message = str(error)
    return any(fragment in message for fragment in _COMPACTIFIED_SUNDMAN_AUTO_FALLBACK_MESSAGES)


def _can_use_planar_hybrid_atlas(positions: Array, velocities: Array, target_time: float) -> bool:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    return bool(positions.shape == (3, 2) and velocities.shape == (3, 2) and float(target_time) != 0.0)


def _try_evaluate_auto_spatial_ks_validated_atlas(
    masses: Array,
    positions: Array,
    velocities: Array,
    target_time: float,
    *,
    initial_radius: float,
    order: int,
    guard_order: int,
    binary_distance_threshold: float,
    s_upper: float,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    competing_pair_min_distance_required: float = 0.0,
    max_competing_events: int = 3,
) -> tuple[ValidatedAtlasSolution | None, tuple[str, ...]]:
    target_time = float(target_time)
    if target_time < 0.0:
        forward_atlas = _try_evaluate_auto_spatial_ks_validated_atlas(
            masses,
            positions,
            -np.asarray(velocities, dtype=float),
            -target_time,
            initial_radius=initial_radius,
            order=order,
            guard_order=guard_order,
            binary_distance_threshold=binary_distance_threshold,
            s_upper=s_upper,
            ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
            ordinary_handoff_max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
            ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
            ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
            ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
            competing_pair_min_distance_required=competing_pair_min_distance_required,
            max_competing_events=max_competing_events,
        )
        if forward_atlas is None:
            return None
        return time_reverse_validated_atlas_solution(
            forward_atlas,
            source="evaluate_unrestricted_solution_time_reversal",
        )
    initial_selector = _derive_initial_spatial_ks_selector(
        positions,
        velocities,
        binary_distance_threshold=binary_distance_threshold,
    )
    if initial_selector is not None:
        pair, branch, exit_rho = initial_selector
        auto_s_upper = max(
            float(s_upper),
            _SPATIAL_KS_AUTO_EXIT_SEARCH_FACTOR * float(np.sqrt(exit_rho)),
        )
        state_interval = _spatial_state_interval_around(positions, velocities, initial_radius)
        initial_atlas = validated_atlas_from_spatial_initial_ks_handoff(
            state_interval,
            masses,
            pair=pair,
            branch=branch,
            exit_rho=exit_rho,
            s_upper=auto_s_upper,
            target_time=target_time,
            retained_order=int(order),
            guard_order=int(guard_order),
            ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
            ordinary_handoff_max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
            ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
            ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
            ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
            competing_pair_min_distance_required=competing_pair_min_distance_required,
            source="evaluate_unrestricted_solution",
        )
        if initial_atlas.proof_certified:
            return initial_atlas
        competing_atlas = _try_evaluate_auto_initial_spatial_ks_competing_handoff(
            masses,
            positions,
            velocities,
            target_time,
            state_interval=state_interval,
            selected_pair=pair,
            selected_branch=branch,
            binary_distance_threshold=binary_distance_threshold,
            s_upper=auto_s_upper,
            order=order,
            guard_order=guard_order,
            ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
            ordinary_handoff_max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
            ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
            ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
            ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
            max_competing_events=max_competing_events,
        )
        if competing_atlas is not None:
            return competing_atlas
        return initial_atlas
    selector = _derive_auto_spatial_ks_selector(
        positions,
        velocities,
        target_time,
        binary_distance_threshold=binary_distance_threshold,
    )
    if selector is None:
        return None
    pair, branch, enter_distance, entry_time_upper, exit_rho = selector
    auto_s_upper = max(
        float(s_upper),
        _SPATIAL_KS_AUTO_EXIT_SEARCH_FACTOR * float(np.sqrt(max(exit_rho, enter_distance))),
    )
    ordinary_ks_error: Exception | None = None
    try:
        ordinary_ks_atlas = evaluate_spatial_ks_validated_atlas_solution(
            masses,
            positions,
            velocities,
            target_time,
            pair=pair,
            enter_distance=enter_distance,
            entry_time_upper=entry_time_upper,
            branch=branch,
            exit_rho=exit_rho,
            s_upper=auto_s_upper,
            initial_radius=initial_radius,
            order=order,
            guard_order=guard_order,
            ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
            ordinary_handoff_max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
            ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
            ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
            ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
            competing_pair_min_distance_required=competing_pair_min_distance_required,
        )
        if ordinary_ks_atlas.proof_certified:
            return ordinary_ks_atlas
    except (RuntimeError, ValueError) as error:
        ordinary_ks_atlas = None
        ordinary_ks_error = error
    competing_atlas = _try_evaluate_auto_spatial_ordinary_ks_competing_handoff(
        masses,
        positions,
        velocities,
        target_time,
        state_interval=_spatial_state_interval_around(positions, velocities, initial_radius),
        selected_pair=pair,
        selected_enter_distance=enter_distance,
        selected_entry_time_upper=entry_time_upper,
        selected_branch=branch,
        binary_distance_threshold=binary_distance_threshold,
        s_upper=auto_s_upper,
        order=order,
        guard_order=guard_order,
        ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
        ordinary_handoff_max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
        ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
        ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
        ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
        max_competing_events=max_competing_events,
    )
    if competing_atlas is not None:
        return competing_atlas
    if ordinary_ks_error is not None:
        raise ordinary_ks_error
    return ordinary_ks_atlas


def _try_evaluate_spatial_close_pair_branch_union(
    masses: Array,
    positions: Array,
    velocities: Array,
    target_time: float,
    *,
    branch_partition: object | None,
    order: int,
    guard_order: int,
    s_endpoint: float,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    max_competing_events: int | None = 3,
) -> tuple[ValidatedAtlasSolution | None, tuple[str, ...]]:
    """Try consuming a certified close-pair state partition as KS branch union."""

    target_time = float(target_time)
    if target_time <= 0.0:
        return None, ()
    if branch_partition is None or not getattr(branch_partition, "certified", False):
        return None, ()
    competing_event_budgets = _spatial_competing_event_budgets(max_competing_events)

    def member_atlas_builder(**kwargs):
        branch = kwargs["branch"]
        label = str(kwargs["branch_label"])
        target_interval = kwargs["target_time_interval"]
        interval_target_time = float(getattr(target_interval, "upper", target_time))
        return _try_with_spatial_competing_event_budget_retries(
            lambda retry_max_competing_events: _try_evaluate_auto_initial_spatial_ks_competing_handoff(
                masses,
                positions,
                velocities,
                interval_target_time,
                state_interval=branch.state_interval,
                selected_pair=branch.selected_pair,
                selected_branch=label,
                binary_distance_threshold=float(
                    getattr(branch_partition, "binary_distance_threshold", 0.0)
                ),
                s_upper=s_endpoint,
                order=order,
                guard_order=guard_order,
                ordinary_handoff_min_pair_distance_required=(
                    ordinary_handoff_min_pair_distance_required
                ),
                ordinary_handoff_max_acceleration_bound=(
                    ordinary_handoff_max_acceleration_bound
                ),
                ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
                ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
                ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
                max_competing_events=retry_max_competing_events,
            ),
            competing_event_budgets,
        )

    try:
        atlas = validated_atlas_from_spatial_close_pair_branch_partition(
            branch_partition,
            masses,
            target_time=target_time,
            s_endpoint=s_endpoint,
            retained_order=order,
            guard_order=guard_order,
            ordinary_handoff_min_pair_distance_required=(
                ordinary_handoff_min_pair_distance_required
            ),
            ordinary_handoff_max_acceleration_bound=(
                ordinary_handoff_max_acceleration_bound
            ),
            ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
            ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
            ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
            competing_pair_min_distance_required=float(
                getattr(branch_partition, "binary_distance_threshold", 0.0)
            ),
            member_atlas_builder=member_atlas_builder,
            source="evaluate_unrestricted_solution:spatial_branch_union",
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        return None, _close_pair_branch_union_failure_obligations(exc)
    if not atlas.proof_certified:
        return atlas, _validated_atlas_missing_obligations(atlas)
    return atlas, ()


def _close_pair_branch_union_failure_obligations(
    exc: BaseException,
) -> tuple[str, ...]:
    """Extract proof obligations surfaced by a failed close-pair branch union."""

    text = str(exc)
    candidates = (
        "finite_time_branch_union_consumption",
        "spatial_ks_target_time_containment",
        "spatial_ks_target_time_endpoint_bracketing",
        "ks_competing_close_binary_requires_next_regularized_chart_or_split",
        "simultaneous_close_pair_partitioning",
        "spatial_triple_close_cluster_requires_cluster_blowup_after_nonzero_angular_exclusion",
        "spatial_triple_close_cluster_requires_cluster_blowup_or_total_collision_selector",
        "finite_time_atlas_loop_repeat_budget",
        "ordinary_handoff_admissibility",
        "spatial_collision_policy_scope",
        "collision_policy",
    )
    missing = [candidate for candidate in candidates if candidate in text]
    if not missing:
        missing.append("finite_time_branch_union_consumption")
    return tuple(dict.fromkeys(missing))


def _selector_obligation_ids(obligations: tuple[str, ...]) -> tuple[str, ...]:
    """Keep typed selector obligations stable while preserving colon details."""

    missing: list[str] = []
    for obligation in obligations:
        text = str(obligation)
        if text.isidentifier():
            missing.append(text)
            continue
        if ":" in text:
            prefix = text.split(":", 1)[0]
            if prefix.isidentifier():
                missing.append(prefix)
                continue
        if text:
            missing.append(text)
    return tuple(dict.fromkeys(missing))


def _finite_time_event_set_missing_obligations(event_set: object) -> tuple[str, ...]:
    missing = list(_selector_obligation_ids(tuple(getattr(event_set, "missing_obligations", ()))))
    if getattr(event_set, "multiple_possible_first_events_requiring_split", False):
        missing.append("finite_time_event_order_requires_split")
    ambiguous_partition = getattr(event_set, "ambiguous_event_order_partition", None)
    if ambiguous_partition is not None:
        missing.extend(
            _selector_obligation_ids(
                tuple(getattr(ambiguous_partition, "missing_obligations", ()))
            )
        )
    first_event = getattr(event_set, "first_event", None)
    if first_event is not None and not getattr(event_set, "unique_first_event_certified", False):
        missing.append("target_event_order_requires_split")
    if not missing:
        missing.append("finite_time_event_order_not_certified")
    return tuple(dict.fromkeys(missing))


def _event_order_branch_union_failure_obligations(exc: BaseException) -> tuple[str, ...]:
    """Extract proof obligations surfaced by a failed branch-union constructor."""

    text = str(exc)
    candidates = (
        "spatial_ks_target_time_containment",
        "spatial_ks_target_time_endpoint_bracketing",
        "ks_competing_close_binary_requires_next_regularized_chart_or_split",
        "simultaneous_close_pair_partitioning",
        "spatial_triple_close_cluster_requires_cluster_blowup_after_nonzero_angular_exclusion",
        "spatial_triple_close_cluster_requires_cluster_blowup_or_total_collision_selector",
        "spatial_collision_policy_scope",
        "collision_policy",
    )
    missing = [candidate for candidate in candidates if candidate in text]
    if not missing:
        missing.append("finite_time_event_order_branch_union_consumption_failed")
    return tuple(dict.fromkeys(missing))


def _attach_spatial_ks_loop_progress(
    atlas: ValidatedAtlasSolution,
    loop_progress: FiniteTimeKSLoopProgressCertificate,
    *,
    source: str,
) -> ValidatedAtlasSolution:
    if not loop_progress.steps:
        return atlas
    loop_progress_certified = bool(
        loop_progress.certified
        or _atlas_certifies_spatial_ks_event_order_loop_consumption(atlas)
    )
    proof_entries = (
        *atlas.proof_ledger.entries,
        ProofLedgerEntry(
            "finite_time_atlas_loop_progress",
            loop_progress_certified,
            source,
            detail=(
                f"events={loop_progress.event_count}; "
                f"max_competing_events={loop_progress.max_competing_events}; "
                f"event_order_branch_union_consumed={loop_progress_certified and not loop_progress.certified}"
            ),
        ),
    )
    try:
        evaluation = replace(
            atlas.evaluation,
            loop_progress_certificate=loop_progress,
        )
    except TypeError:
        evaluation = atlas.evaluation
    return replace(
        atlas,
        proof_ledger=ProofLedger(entries=proof_entries),
        evaluation=evaluation,
    )


def _atlas_certifies_spatial_ks_event_order_loop_consumption(
    atlas: ValidatedAtlasSolution,
) -> bool:
    """Recognize a resolved event-order split after the raw loop stopped."""

    chart_types = {str(getattr(chart, "chart_type", "")) for chart in atlas.charts}
    if "spatial_ks_event_order_branch_union" not in chart_types:
        return False
    certified_entries = {
        str(getattr(entry, "name", ""))
        for entry in getattr(atlas.proof_ledger, "entries", ())
        if bool(getattr(entry, "certified", False))
    }
    if not {
        "ks_event_order_partition",
        "finite_time_event_order_branch_union_consumption",
    } <= certified_entries:
        return False
    prefix_required = bool(
        "spatial_ks_binary" in chart_types
        or "spatial_ordinary_taylor_before_ks" in chart_types
    )
    return bool(
        not prefix_required
        or "spatial_ks_prefix_event_order_branch_union_transition" in certified_entries
    )


def _certify_spatial_ks_competing_loop_progress(
    initial_ks_state,
    *,
    target_time_after_ks_start: float,
    binary_distance_threshold: float,
    s_upper: float,
    order: int,
    guard_order: int,
    max_competing_events: int,
) -> FiniteTimeKSLoopProgressCertificate:
    """Derive a finite KS competing-event loop from interval event sets."""

    target_interval = FloatInterval.point(float(target_time_after_ks_start))
    max_competing_events = int(max_competing_events)
    if max_competing_events < 0:
        max_competing_events = 0
    threshold = float(binary_distance_threshold)
    retained_order = int(order)
    computed_order = retained_order + int(guard_order)
    steps: list[FiniteTimeKSLoopProgressStep] = []
    missing: list[str] = []
    current_state = initial_ks_state
    remaining = target_interval
    if target_interval.lower < 0.0:
        missing.append("finite_time_target_time_nonnegative")
    if threshold <= 0.0:
        missing.append("spatial_ks_competing_enter_distance_positive")
    if retained_order < 1 or computed_order < retained_order:
        missing.append("finite_time_event_retained_order_positive")
    if missing:
        return FiniteTimeKSLoopProgressCertificate(
            initial_pair=tuple(getattr(initial_ks_state, "pair", ())),
            target_time_after_start=target_interval,
            max_competing_events=max_competing_events,
            steps=(),
            missing_obligations=tuple(dict.fromkeys(missing)),
        )

    for index in range(max_competing_events + 1):
        try:
            ks_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
                current_state,
                order=computed_order,
            )
            event_set = certify_next_finite_time_event_set(
                ks_solution,
                target_time_after_start_interval=remaining,
                selected_pair=tuple(getattr(current_state, "pair", ())),
                competing_enter_distance=threshold,
                competing_s_upper=float(s_upper),
                retained_order=retained_order,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            step_missing = ("finite_time_event_set_not_constructed",)
            steps.append(
                FiniteTimeKSLoopProgressStep(
                    index=index,
                    active_pair=tuple(getattr(current_state, "pair", ())),
                    remaining_time_interval=remaining,
                    event_set_certificate=None,
                    decision="event_set_not_constructed",
                    missing_obligations=step_missing,
                )
            )
            missing.extend(step_missing)
            missing.append(str(exc))
            break

        if event_set.target_before_all_events:
            steps.append(
                FiniteTimeKSLoopProgressStep(
                    index=index,
                    active_pair=tuple(getattr(current_state, "pair", ())),
                    remaining_time_interval=remaining,
                    event_set_certificate=event_set,
                    decision="target_before_all_events",
                )
            )
            break

        first_event = event_set.first_event
        if (
            not event_set.unique_first_event_certified
            or first_event is None
            or first_event.event_type != "spatial_ks_competing_binary_entry"
            or first_event.pair is None
            or first_event.certificate is None
        ):
            step_missing = _finite_time_event_set_missing_obligations(event_set)
            event_order_partition = None
            if getattr(event_set, "multiple_possible_first_events_requiring_split", False):
                try:
                    event_order_partition = partition_ks_state_by_competing_event_order(
                        current_state,
                        target_time_after_start_interval=remaining,
                        selected_pair=tuple(getattr(current_state, "pair", ())),
                        competing_enter_distance=threshold,
                        competing_s_upper=float(s_upper),
                        retained_order=retained_order,
                        guard_order=int(guard_order),
                        max_depth=2,
                        max_branches=32,
                    )
                except (RuntimeError, ValueError, TypeError):
                    event_order_partition = None
                if event_order_partition is not None:
                    step_missing = tuple(
                        obligation
                        for obligation in step_missing
                        if obligation != "state_space_event_order_partition_not_constructed"
                    )
                    if getattr(event_order_partition, "certified", False):
                        step_missing = (
                            *step_missing,
                            "finite_time_event_order_partition_requires_branch_union_constructor",
                        )
                    else:
                        step_missing = (
                            *step_missing,
                            *tuple(
                                getattr(
                                    event_order_partition,
                                    "missing_obligations",
                                    (),
                                )
                            ),
                        )
                    step_missing = tuple(dict.fromkeys(step_missing))
            steps.append(
                FiniteTimeKSLoopProgressStep(
                    index=index,
                    active_pair=tuple(getattr(current_state, "pair", ())),
                    remaining_time_interval=remaining,
                    event_set_certificate=event_set,
                    decision="event_order_blocked",
                    event_id=getattr(first_event, "event_id", None),
                    event_pair=getattr(first_event, "pair", None),
                    event_time_interval=getattr(first_event, "physical_time_interval", None),
                    event_order_partition_certificate=event_order_partition,
                    missing_obligations=step_missing,
                )
            )
            missing.extend(step_missing)
            break

        if index >= max_competing_events:
            step_missing = ("finite_time_atlas_loop_repeat_budget",)
            steps.append(
                FiniteTimeKSLoopProgressStep(
                    index=index,
                    active_pair=tuple(getattr(current_state, "pair", ())),
                    remaining_time_interval=remaining,
                    event_set_certificate=event_set,
                    decision="repeat_budget_exhausted",
                    event_id=first_event.event_id,
                    event_pair=first_event.pair,
                    event_time_interval=first_event.physical_time_interval,
                    missing_obligations=step_missing,
                )
            )
            missing.extend(step_missing)
            break

        try:
            branch_states = spatial_ks_competing_entry_event_to_ks_chart_state_atlas(
                ks_solution,
                first_event.certificate,
            )
        except (RuntimeError, ValueError, TypeError):
            branch_states = ()
        next_state = None
        next_branch = None
        for branch_state in branch_states:
            branch_certificate = getattr(branch_state, "branch_certificate", None)
            branch = str(getattr(branch_certificate, "branch", ""))
            if branch and getattr(branch_state, "certified", False):
                next_state = branch_state
                next_branch = branch
                break
        if next_state is None or next_branch is None:
            step_missing = ("spatial_ks_to_ks_branch_lift",)
            steps.append(
                FiniteTimeKSLoopProgressStep(
                    index=index,
                    active_pair=tuple(getattr(current_state, "pair", ())),
                    remaining_time_interval=remaining,
                    event_set_certificate=event_set,
                    decision="branch_lift_blocked",
                    event_id=first_event.event_id,
                    event_pair=first_event.pair,
                    event_time_interval=first_event.physical_time_interval,
                    missing_obligations=step_missing,
                )
            )
            missing.extend(step_missing)
            break

        steps.append(
            FiniteTimeKSLoopProgressStep(
                index=index,
                active_pair=tuple(getattr(current_state, "pair", ())),
                remaining_time_interval=remaining,
                event_set_certificate=event_set,
                decision="unique_competing_event",
                event_id=first_event.event_id,
                event_pair=first_event.pair,
                event_time_interval=first_event.physical_time_interval,
                next_branch=next_branch,
            )
        )
        remaining = remaining - first_event.physical_time_interval
        if remaining.lower < 0.0:
            remaining = FloatInterval(0.0, remaining.upper)
        current_state = next_state
    else:
        missing.append("finite_time_atlas_loop_repeat_budget")

    if not steps or not steps[-1].reaches_target:
        if not missing:
            missing.append("finite_time_atlas_loop_target_not_reached")
    return FiniteTimeKSLoopProgressCertificate(
        initial_pair=tuple(getattr(initial_ks_state, "pair", ())),
        target_time_after_start=target_interval,
        max_competing_events=max_competing_events,
        steps=tuple(steps),
        missing_obligations=tuple(dict.fromkeys(missing)),
    )


def _try_evaluate_spatial_ks_event_order_branch_union(
    initial_ks_state,
    loop_progress: FiniteTimeKSLoopProgressCertificate,
    *,
    order: int,
    guard_order: int,
    binary_distance_threshold: float,
    s_upper: float,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
) -> ValidatedAtlasSolution | None:
    """Consume a certified event-order partition exposed by the KS loop."""

    branch_union_failures: list[str] = []
    for step_index, step in enumerate(loop_progress.steps):
        partition = getattr(step, "event_order_partition_certificate", None)
        if partition is None or not getattr(partition, "certified", False):
            continue
        step_index = int(getattr(step, "index", step_index))
        if step_index < 0:
            continue
        prefix_steps = tuple(loop_progress.steps[:step_index])
        if any(
            getattr(prefix_step, "decision", "") != "unique_competing_event"
            for prefix_step in prefix_steps
        ):
            continue
        remaining_repeats = int(loop_progress.max_competing_events) - step_index - 1
        if remaining_repeats < 0:
            continue
        try:
            suffix_atlas = validated_atlas_from_spatial_ks_event_order_partition(
                partition,
                retained_order=order,
                guard_order=guard_order,
                ordinary_handoff_min_pair_distance_required=(
                    ordinary_handoff_min_pair_distance_required
                ),
                ordinary_handoff_max_acceleration_bound=(
                    ordinary_handoff_max_acceleration_bound
                ),
                ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
                ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
                ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
                competing_pair_min_distance_required=binary_distance_threshold,
                max_competing_repeats=remaining_repeats,
                source="evaluate_unrestricted_solution:ks_event_order_branch_union",
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            branch_union_failures.extend(
                _event_order_branch_union_failure_obligations(exc)
            )
            continue
        if not prefix_steps:
            return suffix_atlas
        try:
            prefix_states = _derive_spatial_ks_loop_prefix_states(
                initial_ks_state,
                prefix_steps,
                order=order,
                guard_order=guard_order,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            branch_union_failures.extend(
                _event_order_branch_union_failure_obligations(exc)
            )
            continue
        atlas = suffix_atlas
        for prefix_state, prefix_step in reversed(
            tuple(zip(prefix_states, prefix_steps, strict=True))
        ):
            if (
                prefix_step.event_pair is None
                or prefix_step.next_branch is None
            ):
                atlas = None
                break
            try:
                atlas = validated_atlas_from_spatial_ks_competing_binary_handoff_to_atlas(
                    prefix_state,
                    next_atlas=atlas,
                    competing_pair=prefix_step.event_pair,
                    enter_distance=binary_distance_threshold,
                    entry_s_upper=float(s_upper),
                    branch=prefix_step.next_branch,
                    target_time_after_ks_start_interval=prefix_step.remaining_time_interval,
                    retained_order=order,
                    guard_order=guard_order,
                    ordinary_handoff_min_pair_distance_required=(
                        ordinary_handoff_min_pair_distance_required
                    ),
                    ordinary_handoff_max_acceleration_bound=(
                        ordinary_handoff_max_acceleration_bound
                    ),
                    ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
                    ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
                    ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
                    competing_pair_min_distance_required=binary_distance_threshold,
                    source=(
                        "evaluate_unrestricted_solution:"
                        "ks_prefix_event_order_branch_union"
                    ),
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                branch_union_failures.extend(
                    _event_order_branch_union_failure_obligations(exc)
                )
                atlas = None
                break
        if atlas is not None and atlas.proof_certified:
            return atlas
    if branch_union_failures:
        raise FiniteTimeKSLoopBlocked(
            replace(
                loop_progress,
                missing_obligations=tuple(
                    dict.fromkeys(
                        loop_progress.missing_obligations
                        + tuple(branch_union_failures)
                    )
                ),
            )
        )
    return None


def _derive_spatial_ks_loop_prefix_states(
    initial_ks_state,
    prefix_steps: tuple[FiniteTimeKSLoopProgressStep, ...],
    *,
    order: int,
    guard_order: int,
) -> tuple[object, ...]:
    """Reconstruct the KS states at the start of each certified prefix step."""

    states: list[object] = []
    current_state = initial_ks_state
    computed_order = int(order) + int(guard_order)
    for step in prefix_steps:
        if (
            step.decision != "unique_competing_event"
            or step.event_set_certificate is None
            or step.next_branch is None
        ):
            raise ValueError("prefix step is not a certified unique competing event")
        states.append(current_state)
        first_event = getattr(step.event_set_certificate, "first_event", None)
        if (
            first_event is None
            or first_event.certificate is None
            or getattr(first_event, "pair", None) != step.event_pair
        ):
            raise ValueError("prefix event certificate does not match loop step")
        ks_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
            current_state,
            order=computed_order,
        )
        branch_states = spatial_ks_competing_entry_event_to_ks_chart_state_atlas(
            ks_solution,
            first_event.certificate,
        )
        next_state = None
        for branch_state in branch_states:
            branch_certificate = getattr(branch_state, "branch_certificate", None)
            branch = str(getattr(branch_certificate, "branch", ""))
            if branch == step.next_branch and getattr(branch_state, "certified", False):
                next_state = branch_state
                break
        if next_state is None:
            raise ValueError("prefix branch lift did not reproduce the loop state")
        current_state = next_state
    return tuple(states)


def _try_evaluate_auto_initial_spatial_ks_competing_handoff(
    masses: Array,
    positions: Array,
    velocities: Array,
    target_time: float,
    *,
    state_interval: tuple[tuple[float, float], ...],
    selected_pair: tuple[int, int],
    selected_branch: str,
    binary_distance_threshold: float,
    s_upper: float,
    order: int,
    guard_order: int,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    max_competing_events: int = 3,
) -> ValidatedAtlasSolution | None:
    """Try a finite constructor-derived KS-to-KS competing-binary loop."""

    target_time = float(target_time)
    if target_time <= 0.0:
        return None
    initial_ks_state = spatial_interval_to_ks_binary_chart_state(
        state_interval,
        masses,
        pair=selected_pair,
        branch=selected_branch,
    )
    if not initial_ks_state.certified:
        return None
    loop_progress = _certify_spatial_ks_competing_loop_progress(
        initial_ks_state,
        target_time_after_ks_start=target_time,
        binary_distance_threshold=binary_distance_threshold,
        s_upper=s_upper,
        order=order,
        guard_order=guard_order,
        max_competing_events=max_competing_events,
    )
    if not loop_progress.certified:
        branch_union = _try_evaluate_spatial_ks_event_order_branch_union(
            initial_ks_state,
            loop_progress,
            order=order,
            guard_order=guard_order,
            binary_distance_threshold=binary_distance_threshold,
            s_upper=s_upper,
            ordinary_handoff_min_pair_distance_required=(
                ordinary_handoff_min_pair_distance_required
            ),
            ordinary_handoff_max_acceleration_bound=(
                ordinary_handoff_max_acceleration_bound
            ),
            ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
            ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
            ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
        )
        if branch_union is not None and branch_union.proof_certified:
            return branch_union
        raise FiniteTimeKSLoopBlocked(loop_progress)
    if loop_progress.event_count == 0:
        return None
    candidates = _derive_certified_spatial_ks_competing_selectors_from_ks_state(
        initial_ks_state,
        target_time_after_ks_start=target_time,
        binary_distance_threshold=binary_distance_threshold,
        s_upper=s_upper,
        order=order,
        guard_order=guard_order,
    )
    if not candidates:
        return None
    first_uncertified_atlas: ValidatedAtlasSolution | None = None
    for (
        _entry_time_estimate,
        competing_pair,
        branch,
        enter_distance,
        entry_s_upper,
        next_s_endpoint,
    ) in candidates:
        try:
            atlas = validated_atlas_from_spatial_ks_competing_binary_handoff(
                initial_ks_state,
                competing_pair=competing_pair,
                enter_distance=enter_distance,
                entry_s_upper=entry_s_upper,
                branch=branch,
                next_s_endpoint=next_s_endpoint,
                target_time_after_ks_start_interval=FloatInterval.point(target_time),
                retained_order=order,
                guard_order=guard_order,
                ordinary_handoff_min_pair_distance_required=(
                    ordinary_handoff_min_pair_distance_required
                ),
                ordinary_handoff_max_acceleration_bound=(
                    ordinary_handoff_max_acceleration_bound
                ),
                ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
                ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
                ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
                competing_pair_min_distance_required=binary_distance_threshold,
                max_competing_repeats=loop_progress.required_recursive_repeat_budget,
                source="evaluate_unrestricted_solution",
            )
        except (RuntimeError, ValueError):
            continue
        atlas = _attach_spatial_ks_loop_progress(
            atlas,
            loop_progress,
            source="evaluate_unrestricted_solution",
        )
        if atlas.proof_certified:
            return atlas
        if first_uncertified_atlas is None:
            first_uncertified_atlas = atlas
    return first_uncertified_atlas


def _try_evaluate_auto_spatial_ordinary_ks_competing_handoff(
    masses: Array,
    positions: Array,
    velocities: Array,
    target_time: float,
    *,
    state_interval: tuple[tuple[float, float], ...],
    selected_pair: tuple[int, int],
    selected_enter_distance: float,
    selected_entry_time_upper: float,
    selected_branch: str,
    binary_distance_threshold: float,
    s_upper: float,
    order: int,
    guard_order: int,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    max_competing_events: int = 3,
) -> ValidatedAtlasSolution | None:
    """Try an ordinary-entry KS chart followed by a finite competing-KS loop."""

    target_time = float(target_time)
    if target_time <= 0.0:
        return None
    entry_data = _derive_certified_spatial_ordinary_entry_ks_state(
        state_interval,
        masses,
        selected_pair=selected_pair,
        enter_distance=selected_enter_distance,
        entry_time_upper=selected_entry_time_upper,
        branch=selected_branch,
        order=order,
        guard_order=guard_order,
    )
    if entry_data is None:
        return None
    entry_ks_state, entry_time_estimate = entry_data
    loop_progress = _certify_spatial_ks_competing_loop_progress(
        entry_ks_state,
        target_time_after_ks_start=target_time - entry_time_estimate,
        binary_distance_threshold=binary_distance_threshold,
        s_upper=s_upper,
        order=order,
        guard_order=guard_order,
        max_competing_events=max_competing_events,
    )
    if not loop_progress.certified:
        branch_union = _try_evaluate_spatial_ks_event_order_branch_union(
            entry_ks_state,
            loop_progress,
            order=order,
            guard_order=guard_order,
            binary_distance_threshold=binary_distance_threshold,
            s_upper=s_upper,
            ordinary_handoff_min_pair_distance_required=(
                ordinary_handoff_min_pair_distance_required
            ),
            ordinary_handoff_max_acceleration_bound=(
                ordinary_handoff_max_acceleration_bound
            ),
            ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
            ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
            ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
        )
        if branch_union is not None and branch_union.proof_certified:
            try:
                atlas = validated_atlas_from_spatial_ordinary_ks_suffix_atlas(
                    state_interval,
                    masses,
                    pair=selected_pair,
                    enter_distance=selected_enter_distance,
                    entry_time_upper=selected_entry_time_upper,
                    branch=selected_branch,
                    ks_suffix_atlas=branch_union,
                    target_time=target_time,
                    retained_order=order,
                    guard_order=guard_order,
                    source="evaluate_unrestricted_solution",
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                raise FiniteTimeKSLoopBlocked(
                    replace(
                        loop_progress,
                        missing_obligations=tuple(
                            dict.fromkeys(
                                loop_progress.missing_obligations
                                + _event_order_branch_union_failure_obligations(exc)
                            )
                        ),
                    )
                ) from exc
            atlas = _attach_spatial_ks_loop_progress(
                atlas,
                loop_progress,
                source="evaluate_unrestricted_solution",
            )
            if atlas.proof_certified:
                return atlas
            raise FiniteTimeKSLoopBlocked(
                replace(
                    loop_progress,
                    missing_obligations=tuple(
                        dict.fromkeys(
                            loop_progress.missing_obligations
                            + _validated_atlas_missing_obligations(atlas)
                        )
                    ),
                )
            )
        raise FiniteTimeKSLoopBlocked(loop_progress)
    if loop_progress.event_count == 0:
        return None
    candidates = _derive_certified_spatial_ks_competing_selectors_from_ks_state(
        entry_ks_state,
        target_time_after_ks_start=target_time - entry_time_estimate,
        binary_distance_threshold=binary_distance_threshold,
        s_upper=s_upper,
        order=order,
        guard_order=guard_order,
    )
    if not candidates:
        return None
    first_uncertified_atlas: ValidatedAtlasSolution | None = None
    for (
        _entry_time_estimate,
        competing_pair,
        branch,
        enter_distance,
        entry_s_upper,
        next_s_endpoint,
    ) in candidates:
        try:
            atlas = validated_atlas_from_spatial_ordinary_ks_competing_binary_handoff(
                state_interval,
                masses,
                pair=selected_pair,
                enter_distance=selected_enter_distance,
                entry_time_upper=selected_entry_time_upper,
                branch=selected_branch,
                competing_pair=competing_pair,
                competing_enter_distance=enter_distance,
                competing_entry_s_upper=entry_s_upper,
                competing_branch=branch,
                next_s_endpoint=next_s_endpoint,
                target_time=target_time,
                retained_order=order,
                guard_order=guard_order,
                ordinary_handoff_min_pair_distance_required=(
                    ordinary_handoff_min_pair_distance_required
                ),
                ordinary_handoff_max_acceleration_bound=(
                    ordinary_handoff_max_acceleration_bound
                ),
                ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
                ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
                ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
                competing_pair_min_distance_required=binary_distance_threshold,
                max_competing_repeats=loop_progress.required_recursive_repeat_budget,
                source="evaluate_unrestricted_solution",
            )
        except (RuntimeError, ValueError):
            continue
        atlas = _attach_spatial_ks_loop_progress(
            atlas,
            loop_progress,
            source="evaluate_unrestricted_solution",
        )
        if atlas.proof_certified:
            return atlas
        if first_uncertified_atlas is None:
            first_uncertified_atlas = atlas
    return first_uncertified_atlas


def _derive_certified_spatial_ordinary_entry_ks_state(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    *,
    selected_pair: tuple[int, int],
    enter_distance: float,
    entry_time_upper: float,
    branch: str,
    order: int,
    guard_order: int,
):
    """Lift the selected ordinary-entry event before deriving competing events."""

    retained_order = int(order)
    computed_order = retained_order + int(guard_order)
    try:
        ordinary_positions, ordinary_velocities = _spatial_interval_state_arrays(
            state_interval
        )
        ordinary_solution = construct_interval_taylor_solution_from_intervals(
            ordinary_positions,
            ordinary_velocities,
            np.asarray(masses, dtype=float),
            order=computed_order,
        )
        entry_certificate = certify_spatial_ordinary_ks_entry_event(
            ordinary_solution,
            pair=tuple(selected_pair),
            enter_distance=float(enter_distance),
            time_upper=float(entry_time_upper),
            coefficient_count=retained_order,
        )
        if not entry_certificate.certified:
            return None
        entry_time_estimate = 0.5 * (
            entry_certificate.root_interval[0] + entry_certificate.root_interval[1]
        )
        entry_ks_state = spatial_ordinary_entry_event_to_ks_chart_state(
            ordinary_solution,
            entry_certificate,
            branch=str(branch),
        )
    except (RuntimeError, ValueError, TypeError):
        return None
    if not entry_ks_state.certified:
        return None
    return entry_ks_state, entry_time_estimate


def _derive_certified_spatial_ks_competing_selectors_from_ks_state(
    initial_ks_state,
    *,
    target_time_after_ks_start: float,
    binary_distance_threshold: float,
    s_upper: float,
    order: int,
    guard_order: int,
) -> tuple[tuple[float, tuple[int, int], str, float, float, float], ...]:
    """Derive competing-pair candidates from certified KS event isolation."""

    target_time_after_ks_start = float(target_time_after_ks_start)
    threshold = float(binary_distance_threshold)
    s_upper = float(s_upper)
    if target_time_after_ks_start <= 0.0 or threshold <= 0.0 or s_upper <= 0.0:
        return ()
    retained_order = int(order)
    computed_order = retained_order + int(guard_order)
    if retained_order < 1 or computed_order < retained_order:
        return ()
    try:
        ks_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
            initial_ks_state,
            order=computed_order,
        )
    except (RuntimeError, ValueError, TypeError):
        return ()

    selected_pair = tuple(getattr(initial_ks_state, "pair", ()))
    event_set = certify_next_finite_time_event_set(
        ks_solution,
        target_time_after_start_interval=FloatInterval.point(target_time_after_ks_start),
        selected_pair=selected_pair,
        competing_enter_distance=threshold,
        competing_s_upper=s_upper,
        retained_order=retained_order,
    )
    first_event = event_set.first_event
    if (
        event_set.target_before_all_events
        or not event_set.unique_first_event_certified
        or first_event is None
        or first_event.event_type != "spatial_ks_competing_binary_entry"
        or first_event.pair is None
        or first_event.certificate is None
    ):
        return ()
    try:
        branch_states = spatial_ks_competing_entry_event_to_ks_chart_state_atlas(
            ks_solution,
            first_event.certificate,
        )
    except (RuntimeError, ValueError, TypeError):
        return ()
    entry_time_lower = float(first_event.physical_time_interval.lower)
    if not np.isfinite(entry_time_lower):
        return ()
    candidates: list[tuple[float, tuple[int, int], str, float, float, float]] = []
    for branch_state in branch_states:
        branch_certificate = getattr(branch_state, "branch_certificate", None)
        branch = str(getattr(branch_certificate, "branch", ""))
        if not branch or not getattr(branch_state, "certified", False):
            continue
        next_s_endpoint = _auto_spatial_ks_competing_next_s_endpoint(
            target_time=target_time_after_ks_start,
            entry_time_estimate=entry_time_lower,
            enter_distance=threshold,
            s_upper=s_upper,
        )
        candidates.append(
            (
                entry_time_lower,
                first_event.pair,
                branch,
                threshold,
                s_upper,
                next_s_endpoint,
            )
        )
    return tuple(candidates)


def _spatial_interval_state_arrays(
    state_interval: tuple[tuple[float, float], ...],
) -> tuple[Array, Array]:
    if len(state_interval) != 18:
        raise ValueError("spatial interval state must have length 18")
    intervals = np.array(
        [FloatInterval(float(lower), float(upper)) for lower, upper in state_interval],
        dtype=object,
    )
    return intervals[:9].reshape(3, 3), intervals[9:].reshape(3, 3)


def _auto_spatial_ks_competing_next_s_endpoint(
    *,
    target_time: float,
    entry_time_estimate: float,
    enter_distance: float,
    s_upper: float,
) -> float:
    target_gap = max(0.0, float(target_time) - float(entry_time_estimate))
    enter_distance = max(float(enter_distance), 1.0e-15)
    endpoint = max(1.0e-4, 4.0 * target_gap / enter_distance)
    return float(min(max(float(s_upper), 1.0e-4), endpoint))


def _derive_initial_spatial_ks_selector(
    positions: Array,
    velocities: Array,
    *,
    binary_distance_threshold: float,
) -> tuple[tuple[int, int], str, float] | None:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    binary_distance_threshold = float(binary_distance_threshold)
    if positions.shape != (3, 3) or velocities.shape != (3, 3):
        return None
    if binary_distance_threshold <= 0.0:
        return None

    candidates: list[tuple[float, tuple[int, int], str, float]] = []
    for pair in _three_body_pairs():
        first, second = pair
        third = ({0, 1, 2} - set(pair)).pop()
        relative_position = positions[second] - positions[first]
        distance = float(np.linalg.norm(relative_position))
        if (
            not np.isfinite(distance)
            or distance <= 0.0
            or distance > binary_distance_threshold
        ):
            continue
        third_distance = min(
            float(np.linalg.norm(positions[third] - positions[first])),
            float(np.linalg.norm(positions[third] - positions[second])),
        )
        if third_distance <= 4.0 * distance:
            continue
        branch = "positive_x" if relative_position[0] >= 0.0 else "negative_x"
        exit_rho = _SPATIAL_KS_AUTO_EXIT_MULTIPLIER * binary_distance_threshold
        candidates.append((distance, pair, branch, exit_rho))
    if not candidates:
        return None
    _distance, pair, branch, exit_rho = min(candidates, key=lambda candidate: candidate[0])
    return pair, branch, exit_rho


def _derive_auto_spatial_ks_selector(
    positions: Array,
    velocities: Array,
    target_time: float,
    *,
    binary_distance_threshold: float,
) -> tuple[tuple[int, int], str, float, float, float] | None:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    target_time = float(target_time)
    binary_distance_threshold = float(binary_distance_threshold)
    if positions.shape != (3, 3) or velocities.shape != (3, 3):
        return None
    if target_time <= 0.0 or binary_distance_threshold <= 0.0:
        return None

    candidates: list[tuple[float, float, tuple[int, int], str, float, float, float]] = []
    for pair in _three_body_pairs():
        first, second = pair
        third = ({0, 1, 2} - set(pair)).pop()
        relative_position = positions[second] - positions[first]
        relative_velocity = velocities[second] - velocities[first]
        distance = float(np.linalg.norm(relative_position))
        if not np.isfinite(distance) or distance <= 0.0:
            continue
        third_distance = min(
            float(np.linalg.norm(positions[third] - positions[first])),
            float(np.linalg.norm(positions[third] - positions[second])),
        )
        if third_distance <= 4.0 * distance:
            continue
        closing_speed = -float(np.dot(relative_position, relative_velocity)) / distance
        if not np.isfinite(closing_speed) or closing_speed <= 0.0:
            continue
        enter_distance = (
            _SPATIAL_KS_AUTO_ENTER_FRACTION * distance
            if distance <= binary_distance_threshold
            else binary_distance_threshold
        )
        if (
            not np.isfinite(enter_distance)
            or enter_distance <= 0.0
            or enter_distance >= distance
        ):
            continue
        entry_time_upper = (
            _SPATIAL_KS_AUTO_ENTRY_TIME_SAFETY
            * (distance - enter_distance)
            / closing_speed
        )
        if not np.isfinite(entry_time_upper) or entry_time_upper <= 0.0:
            continue
        if entry_time_upper >= target_time:
            continue
        branch = "positive_x" if relative_position[0] >= 0.0 else "negative_x"
        exit_rho = _SPATIAL_KS_AUTO_EXIT_MULTIPLIER * enter_distance
        candidates.append(
            (
                entry_time_upper,
                distance,
                pair,
                branch,
                enter_distance,
                entry_time_upper,
                exit_rho,
            )
        )
    if not candidates:
        return None
    _entry_time, _distance, pair, branch, enter_distance, entry_time_upper, exit_rho = min(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    return pair, branch, enter_distance, entry_time_upper, exit_rho


def _spatial_close_binary_fallback_blocker(
    positions: Array,
    velocities: Array,
    masses: Array,
    target_time: float,
    *,
    initial_radius: float = 0.0,
    binary_distance_threshold: float,
) -> FiniteTimeChartSelectorAttempt | None:
    """Detect close 3D binary approaches that need regularization or splitting."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    target_time = float(target_time)
    initial_radius = float(initial_radius)
    binary_distance_threshold = float(binary_distance_threshold)
    if positions.shape != (3, 3) or velocities.shape != (3, 3):
        return None
    if masses.shape != (3,) or np.any(masses <= 0.0):
        return None
    if target_time == 0.0 or binary_distance_threshold <= 0.0 or initial_radius < 0.0:
        return None
    time_direction = 1.0 if target_time > 0.0 else -1.0
    target_horizon = abs(target_time)

    blockers: list[tuple[float, tuple[int, int], tuple[str, ...], str]] = []
    for pair in _three_body_pairs():
        first, second = pair
        third = ({0, 1, 2} - set(pair)).pop()
        relative_position = positions[second] - positions[first]
        relative_velocity = time_direction * (velocities[second] - velocities[first])
        distance = float(np.linalg.norm(relative_position))
        interval_lower, interval_upper = _spatial_pair_distance_interval_bounds(
            positions,
            pair,
            initial_radius=initial_radius,
        )
        if (
            initial_radius > 0.0
            and np.isfinite(interval_lower)
            and interval_lower <= binary_distance_threshold
        ):
            third_lower = min(
                _spatial_pair_distance_interval_bounds(
                    positions,
                    (third, first),
                    initial_radius=initial_radius,
                )[0],
                _spatial_pair_distance_interval_bounds(
                    positions,
                    (third, second),
                    initial_radius=initial_radius,
                )[0],
            )
            missing = [
                "spatial_interval_close_binary_requires_regularized_chart_or_split",
            ]
            if (
                not np.isfinite(third_lower)
                or not np.isfinite(interval_upper)
                or third_lower <= 4.0 * interval_upper
            ):
                missing.append("spatial_interval_close_pair_not_separated_from_third_body")
            blockers.append(
                (
                    interval_lower,
                    pair,
                    tuple(missing),
                    (
                        f"pair={pair}; interval_distance=[{interval_lower:.6g}, "
                        f"{interval_upper:.6g}]; target_time={target_time:.6g}; "
                        f"initial_radius={initial_radius:.6g}; "
                        f"third_distance_lower={third_lower:.6g}"
                    ),
                )
            )
            continue
        if not np.isfinite(distance) or distance <= 0.0:
            continue
        if distance <= binary_distance_threshold:
            third_distance = min(
                float(np.linalg.norm(positions[third] - positions[first])),
                float(np.linalg.norm(positions[third] - positions[second])),
            )
            missing = ["spatial_close_binary_requires_regularized_chart_or_split"]
            if not np.isfinite(third_distance) or third_distance <= 4.0 * distance:
                missing.append("spatial_close_pair_not_separated_from_third_body")
            blockers.append(
                (
                    distance,
                    pair,
                    tuple(missing),
                    (
                        f"pair={pair}; distance={distance:.6g}; "
                        f"target_time={target_time:.6g}; "
                        f"third_distance={third_distance:.6g}"
                    ),
                )
            )
            continue
        closing_speed = -float(np.dot(relative_position, relative_velocity)) / distance
        if not np.isfinite(closing_speed) or closing_speed <= 0.0:
            continue
        enter_distance = (
            _SPATIAL_KS_AUTO_ENTER_FRACTION * distance
            if distance <= binary_distance_threshold
            else binary_distance_threshold
        )
        if (
            not np.isfinite(enter_distance)
            or enter_distance <= 0.0
            or enter_distance >= distance
        ):
            continue
        entry_time_upper = (
            _SPATIAL_KS_AUTO_ENTRY_TIME_SAFETY
            * (distance - enter_distance)
            / closing_speed
        )
        if not np.isfinite(entry_time_upper) or entry_time_upper <= 0.0:
            continue
        if entry_time_upper >= target_horizon:
            continue
        third_distance = min(
            float(np.linalg.norm(positions[third] - positions[first])),
            float(np.linalg.norm(positions[third] - positions[second])),
        )
        missing = [
            "spatial_close_binary_requires_regularized_chart_or_split"
            if distance <= binary_distance_threshold
            else "spatial_future_close_binary_requires_regularized_entry_chart_or_split"
        ]
        if not np.isfinite(third_distance) or third_distance <= 4.0 * distance:
            missing.append("spatial_close_pair_not_separated_from_third_body")
        blockers.append(
            (
                distance,
                pair,
                tuple(missing),
                (
                    f"pair={pair}; distance={distance:.6g}; "
                    f"entry_time_upper={entry_time_upper:.6g}; "
                    f"target_time={target_time:.6g}; third_distance={third_distance:.6g}"
                ),
            )
        )
    if not blockers:
        return None
    _distance, pair, missing, detail = min(blockers, key=lambda item: item[0])
    branch_partition = None
    partition_obligations: tuple[str, ...] = ()
    try:
        branch_partition = certify_simultaneous_close_pair_partition(
            _spatial_state_interval_around(positions, velocities, initial_radius),
            binary_distance_threshold=binary_distance_threshold,
            max_depth=6,
            masses=masses,
        )
    except (RuntimeError, ValueError, TypeError):
        branch_partition = None
    if branch_partition is not None:
        if branch_partition.certified:
            partition_obligations = ("finite_time_branch_union_consumption",)
        else:
            partition_obligations = branch_partition.missing_obligations
    return FiniteTimeChartSelectorAttempt(
        route_id="spatial_close_binary_guard",
        attempted=True,
        selected=False,
        certified=False,
        reason=(
            "spatial binary reaches the regularized-entry threshold before "
            "the requested target without a certified KS route"
        ),
        missing_obligations=(
            *missing,
            *partition_obligations,
            detail,
        ),
        blocks_fallback_certification=True,
        branch_partition=branch_partition,
    )


def _spatial_pair_distance_interval_bounds(
    positions: Array,
    pair: tuple[int, int],
    *,
    initial_radius: float,
) -> tuple[float, float]:
    positions = np.asarray(positions, dtype=float)
    initial_radius = float(initial_radius)
    first, second = pair
    relative_center = positions[second] - positions[first]
    component_radius = 2.0 * max(0.0, initial_radius)
    lower_components = np.maximum(0.0, np.abs(relative_center) - component_radius)
    upper_components = np.abs(relative_center) + component_radius
    return (
        float(np.linalg.norm(lower_components)),
        float(np.linalg.norm(upper_components)),
    )


def _three_body_pairs() -> tuple[tuple[int, int], ...]:
    return ((0, 1), (0, 2), (1, 2))


def _spatial_ks_options_supplied(
    *,
    spatial_binary_pair: tuple[int, int] | None,
    spatial_binary_enter_distance: float | None,
    spatial_binary_entry_time_upper: float | None,
    spatial_binary_exit_rho: float | None,
    spatial_binary_s_upper: float | None,
) -> bool:
    return any(
        option is not None
        for option in (
            spatial_binary_pair,
            spatial_binary_enter_distance,
            spatial_binary_entry_time_upper,
            spatial_binary_exit_rho,
            spatial_binary_s_upper,
        )
    )


def _spatial_ks_validated_selector_route_id(
    solution: ValidatedAtlasSolution,
    *,
    default_route_id: str,
) -> str:
    chart_types = {
        str(getattr(chart, "chart_type", ""))
        for chart in getattr(solution, "charts", ())
    }
    if chart_types <= {"spatial_ks_event_order_branch_union"}:
        return "spatial_ks_event_order_branch_union"
    if (
        "spatial_ks_event_order_branch_union" in chart_types
        and chart_types
        <= {
            "spatial_ordinary_taylor_before_ks",
            "spatial_ks_binary",
            "spatial_ks_event_order_branch_union",
        }
    ):
        return "spatial_ks_prefix_event_order_branch_union"
    if chart_types <= {"spatial_branch_union"}:
        return "spatial_branch_union"
    if "spatial_branch_union" in chart_types and "spatial_ks_binary" in chart_types:
        return "spatial_ks_prefix_close_pair_branch_union"
    return default_route_id


def _spatial_ks_selector_reason(route_id: str, *, explicit: bool) -> str:
    if route_id == "spatial_ks_event_order_branch_union":
        return "automatic spatial KS event-order partition selected a branch-union atlas"
    if route_id == "spatial_ks_prefix_event_order_branch_union":
        return "automatic spatial KS prefix composed into an event-order branch-union atlas"
    if route_id == "spatial_ks_prefix_close_pair_branch_union":
        return "spatial KS prefix composed into a close-pair branch-union atlas"
    if route_id == "spatial_branch_union":
        return "certified close-pair partition leaves were consumed by spatial KS member charts"
    if explicit:
        return "explicit spatial KS selector parameters supplied"
    return "automatic spatial pair selected a KS chart"


def _planar_state_interval_around(
    positions: Array,
    velocities: Array,
    radius: float,
) -> tuple[tuple[float, float], ...]:
    position_box = _interval_box_around(positions, radius)
    velocity_box = _interval_box_around(velocities, radius)
    return tuple(
        interval.as_tuple()
        for interval in np.concatenate(
            [
                np.asarray(position_box, dtype=object).reshape(-1),
                np.asarray(velocity_box, dtype=object).reshape(-1),
            ]
        )
    )


def _time_reverse_planar_state_interval_union(
    state_interval_union: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not state_interval_union:
        raise ValueError("state_interval_union cannot be empty")
    reversed_union = []
    for state_interval in state_interval_union:
        if len(state_interval) != 12:
            raise ValueError("planar state interval union members must have length 12")
        reversed_state = []
        for index, (lower, upper) in enumerate(state_interval):
            lower_value = float(lower)
            upper_value = float(upper)
            if index < 6:
                reversed_state.append((lower_value, upper_value))
            else:
                reversed_state.append((-upper_value, -lower_value))
        reversed_union.append(tuple(reversed_state))
    return tuple(reversed_union)


def _spatial_state_interval_around(
    positions: Array,
    velocities: Array,
    radius: float,
) -> tuple[tuple[float, float], ...]:
    position_box = _interval_box_around(positions, radius)
    velocity_box = _interval_box_around(velocities, radius)
    return tuple(
        interval.as_tuple()
        for interval in np.concatenate(
            [
                np.asarray(position_box, dtype=object).reshape(-1),
                np.asarray(velocity_box, dtype=object).reshape(-1),
            ]
        )
    )


def _interval_box_around(values: Array, radius: float) -> Array:
    values = np.asarray(values, dtype=float)
    out = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        out[index] = FloatInterval(
            float(np.nextafter(values[index] - radius, -np.inf)),
            float(np.nextafter(values[index] + radius, np.inf)),
        )
    return out


def _split_state_interval(state: Array, *, body_count: int, dimension: int) -> tuple[Array, Array]:
    state = np.asarray(state, dtype=object)
    coordinate_count = body_count * dimension
    expected_count = 2 * coordinate_count
    if state.shape != (expected_count,):
        raise ValueError("state interval does not match the requested body count and dimension")
    positions = np.array(state[:coordinate_count], dtype=object).reshape((body_count, dimension))
    velocities = np.array(state[coordinate_count:], dtype=object).reshape((body_count, dimension))
    return positions, velocities
