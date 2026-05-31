"""Shared finite-time validated atlas view.

The classes here are intentionally a thin integration surface. They do not add
new mathematical certificates; they collect the certificates already produced by
the finite-time evaluator into one object that later ordinary, binary,
compact-time, scattering, and total-collision charts can feed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import numpy as np
from scipy.optimize import brentq

from .dynamics import (
    angular_momentum_components,
    center_of_mass,
    energy,
    linear_momentum,
)
from .binary_chart import (
    IntervalRegularizedBinaryCollisionChartState,
    RegularizedBinaryCollisionChartState,
    planar_to_regularized_binary_collision_chart,
    regularized_binary_collision_chart_to_planar,
)
from .binary_series import construct_regularized_binary_taylor_solution
from .global_invariants import (
    certify_nonzero_angular_momentum_excludes_triple_collision,
    certify_interval_center_of_mass_motion,
    certify_interval_centered_angular_momentum_conservation,
    certify_interval_linear_momentum_conservation,
    certify_interval_total_energy_conservation,
)
from .hybrid import (
    _regularized_binary_atlas_end_state_interval,
    _regularized_binary_end_state_interval,
    certify_ordinary_interval_taylor_equations,
)
from .intervals import FloatInterval, interval_array_contains_point, interval_array_series_eval
from .ks_binary_series import (
    IntervalSpatialKSBinaryChartState,
    SpatialKSRhoExitEventCertificate,
    SpatialKSEntryEventCertificate,
    SpatialKSBinaryPhysicalProjectionCertificate,
    certify_spatial_ordinary_ks_entry_event,
    certify_spatial_ks_binary_center_of_mass_motion,
    certify_spatial_ks_binary_centered_angular_momentum_conservation,
    certify_spatial_ks_competing_binary_entry_event,
    certify_spatial_ks_binary_horizontal_constraint,
    certify_spatial_ks_binary_interval_taylor_equations,
    certify_spatial_ks_binary_linear_momentum_conservation,
    certify_spatial_ks_binary_pair_energy_constraint,
    certify_spatial_ks_binary_rho_exit_event,
    certify_spatial_ks_binary_total_energy_conservation,
    construct_interval_spatial_ks_binary_taylor_solution_from_intervals,
    interval_spatial_ks_binary_chart_state_from_point,
    project_spatial_ks_binary_interval_chart_state_to_physical,
    project_spatial_ks_binary_taylor_endpoint_to_physical,
    spatial_interval_to_ks_binary_chart_state,
    spatial_interval_to_ks_binary_chart_state_atlas,
    spatial_ks_competing_entry_event_to_ks_chart_state,
    spatial_ks_competing_entry_event_to_ks_chart_state_atlas,
    spatial_ordinary_entry_event_to_ks_chart_state,
)
from .obstructions import (
    construct_interval_jacobi_cluster_coordinates,
    construct_jacobi_cluster_coordinates,
)
from .reduction import reconstruct_interval_from_center_of_mass_frame
from .series import (
    IntervalTaylorSolution,
    construct_interval_taylor_solution_from_intervals,
    construct_taylor_solution,
)
from .tail_bounds import (
    interval_guarded_tail_certificate,
    ordinary_interval_cauchy_majorant_tail_certificate,
    ordinary_interval_solution_arrays,
    spatial_ks_binary_interval_segmented_tail_certificate,
    spatial_ks_binary_interval_tail_certificate,
)
from .triple_collision import (
    HomotheticTotalCollisionBranch,
    HomotheticTotalCollisionScalarMajorantCertificate,
    certify_homothetic_total_collision_scalar_majorant,
)


Array = np.ndarray

_SPATIAL_TRIPLE_CLOSE_ADJACENCY_FACTOR = 1.05


@dataclass(frozen=True)
class ProofLedgerEntry:
    """One required proof obligation derived from a constructor."""

    name: str
    certified: bool
    source: str
    required: bool = True
    detail: str = ""


@dataclass(frozen=True)
class ProofLedger:
    """Constructor-derived proof obligations for a validated atlas solution."""

    entries: tuple[ProofLedgerEntry, ...]

    @property
    def well_formed(self) -> bool:
        keys = tuple(
            (
                str(getattr(entry, "name", "")),
                str(getattr(entry, "source", "")),
            )
            for entry in self.entries
        )
        return bool(
            keys
            and all(name and source for name, source in keys)
            and len(set(keys)) == len(keys)
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.well_formed
            and all(entry.certified for entry in self.entries if entry.required)
        )

    @property
    def missing_required_obligations(self) -> tuple[str, ...]:
        return tuple(
            entry.name
            for entry in self.entries
            if entry.required and not entry.certified
        )


@dataclass(frozen=True)
class ValidatedChart:
    """One chart already certified by a local constructor."""

    chart_id: str
    chart_type: str
    source: str
    parameter_name: str
    parameter_interval: FloatInterval | None
    physical_time_interval: FloatInterval | None
    dynamics_certified: bool
    residual_certified: bool
    projection_certified: bool
    invariants_certified: bool
    tail_certified: bool
    tail_bound: float = 0.0

    @property
    def certified(self) -> bool:
        return bool(
            self.chart_id
            and self.chart_type
            and self.source
            and self.parameter_name
            and _float_interval_finite_nonempty(self.parameter_interval)
            and _float_interval_finite_nonempty(self.physical_time_interval)
            and self.dynamics_certified
            and self.residual_certified
            and self.projection_certified
            and self.invariants_certified
            and self.tail_certified
            and np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
        )


@dataclass(frozen=True)
class ValidatedTransition:
    """Continuity handoff between adjacent validated charts."""

    source_chart_id: str
    target_chart_id: str
    transition_type: str
    certified: bool
    source: str


@dataclass(frozen=True)
class GlobalInvariantLedger:
    """Invariant coverage extracted from local and target certificates."""

    center_of_mass_certified: bool
    linear_momentum_certified: bool
    angular_momentum_certified: bool
    energy_certified: bool
    certified_chart_count: int
    expected_chart_count: int

    @property
    def certified(self) -> bool:
        return bool(
            self.expected_chart_count > 0
            and self.certified_chart_count >= self.expected_chart_count
            and self.center_of_mass_certified
            and self.linear_momentum_certified
            and self.angular_momentum_certified
            and self.energy_certified
        )


@dataclass(frozen=True)
class TailBudgetLedger:
    """Tail budget inherited from the finite-time evaluator."""

    local_tail_bound: float
    max_step_tail_bound: float
    certified: bool

    @property
    def finite(self) -> bool:
        return bool(np.isfinite(self.local_tail_bound) and np.isfinite(self.max_step_tail_bound))

    @property
    def admissible(self) -> bool:
        return bool(
            self.finite
            and self.local_tail_bound >= 0.0
            and self.max_step_tail_bound >= 0.0
        )


@dataclass(frozen=True)
class NewtonResidualLedger:
    """Newton-equation residual coverage across charts and target chart."""

    certified: bool
    certified_chart_count: int
    expected_chart_count: int

    @property
    def coverage_certified(self) -> bool:
        return bool(
            self.certified
            and self.expected_chart_count > 0
            and self.certified_chart_count >= self.expected_chart_count
        )


@dataclass(frozen=True)
class CollisionPolicyWitness:
    """Finite-time collision policy exposed by the evaluator."""

    policy_id: str
    binary_policy: str
    total_collision_policy: str
    triple_collision_status: str
    triple_collision_reason: str | None
    certified: bool

    @property
    def well_formed(self) -> bool:
        return not self.missing_obligations

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.policy_id:
            missing.append("collision_policy:policy_id")
        if not self.binary_policy:
            missing.append("collision_policy:binary_policy")
        if not self.total_collision_policy:
            missing.append("collision_policy:total_collision_policy")
        if not self.triple_collision_status:
            missing.append("collision_policy:triple_collision_status")
        if self.certified and not self.triple_collision_reason:
            missing.append("collision_policy:triple_collision_reason")
        return tuple(missing)


@dataclass(frozen=True)
class SpatialLocalCollisionPolicyCertificate:
    """Local collision-domain check for an ordinary/KS/ordinary handoff.

    The selected binary pair is allowed to pass through the KS chart.  Every
    ordinary chart must stay away from all pair collisions, and the two
    competing third-body separations must stay positive throughout the KS
    parameter interval.
    """

    pair: tuple[int, int]
    ordinary_entry_squared_distance_lowers: tuple[tuple[tuple[int, int], float], ...]
    ks_competing_squared_distance_lowers: tuple[tuple[tuple[int, int], float], ...]
    ordinary_post_squared_distance_lowers: tuple[tuple[tuple[int, int], float], ...]
    competing_pair_min_distance_required: float
    missing_obligations: tuple[str, ...]

    @property
    def certified(self) -> bool:
        return not self.missing_obligations

    @property
    def min_squared_distance_lower_bound(self) -> float:
        lowers = tuple(
            lower
            for _pair, lower in (
                self.ordinary_entry_squared_distance_lowers
                + self.ks_competing_squared_distance_lowers
                + self.ordinary_post_squared_distance_lowers
            )
        )
        if not lowers:
            return 0.0
        return float(min(lowers))


@dataclass(frozen=True)
class FiniteTimeChartSelectorAttempt:
    """One constructor route considered by the finite-time chart selector."""

    route_id: str
    attempted: bool
    selected: bool
    certified: bool
    reason: str
    missing_obligations: tuple[str, ...] = ()
    blocks_fallback_certification: bool = False
    branch_partition: object | None = None

    @property
    def well_formed(self) -> bool:
        return bool(
            self.route_id
            and self.reason
            and (not self.selected or self.attempted)
            and (not self.certified or not self.missing_obligations)
            and all(str(obligation) for obligation in self.missing_obligations)
            and (
                self.branch_partition is None
                or bool(getattr(self.branch_partition, "well_formed", False))
            )
        )


@dataclass(frozen=True)
class FiniteTimeChartSelectorTrace:
    """Route provenance for ``evaluate_unrestricted_solution(method="validated_atlas")``."""

    selected_route_id: str
    attempts: tuple[FiniteTimeChartSelectorAttempt, ...]
    atlas_binding_token: str | None = None

    @property
    def selected_attempt(self) -> FiniteTimeChartSelectorAttempt | None:
        for attempt in self.attempts:
            if attempt.selected:
                return attempt
        return None

    @property
    def well_formed(self) -> bool:
        route_ids = tuple(str(getattr(attempt, "route_id", "")) for attempt in self.attempts)
        return bool(
            self.selected_route_id
            and route_ids
            and all(getattr(attempt, "well_formed", False) for attempt in self.attempts)
            and len(set(route_ids)) == len(route_ids)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.selected_route_id:
            missing.append("selector_trace_selected_route_id")
        if not self.attempts:
            missing.append("selector_trace_attempts")
        route_ids = tuple(str(getattr(attempt, "route_id", "")) for attempt in self.attempts)
        if len(set(route_ids)) != len(route_ids):
            missing.append("selector_trace_route_ids_unique")
        for index, attempt in enumerate(self.attempts):
            if not getattr(attempt, "well_formed", False):
                route_id = str(getattr(attempt, "route_id", "")) or str(index)
                missing.append(f"selector_trace_attempt_well_formed:{route_id}")
        selected_attempts = tuple(attempt for attempt in self.attempts if attempt.selected)
        if len(selected_attempts) != 1:
            missing.append("selector_trace_single_selected_route")
            return tuple(dict.fromkeys(missing))
        selected = selected_attempts[0]
        if selected.route_id != self.selected_route_id:
            missing.append("selector_trace_selected_route_match")
        if not selected.attempted:
            missing.append("selector_trace_selected_attempted")
        if not selected.certified:
            missing.append("selector_trace_selected_attempt_certified")
            missing.extend(selected.missing_obligations)
        for attempt in self.attempts:
            if attempt.blocks_fallback_certification and not attempt.certified:
                missing.append(f"selector_trace_blocking_attempt_certified:{attempt.route_id}")
                missing.extend(attempt.missing_obligations)
        return tuple(dict.fromkeys(missing))

    @property
    def certified(self) -> bool:
        selected_attempts = tuple(attempt for attempt in self.attempts if attempt.selected)
        selected = selected_attempts[0] if len(selected_attempts) == 1 else None
        return bool(
            self.well_formed
            and selected is not None
            and selected.route_id == self.selected_route_id
            and selected.attempted
            and selected.selected
            and selected.certified
            and all(
                attempt.certified
                for attempt in self.attempts
                if attempt.blocks_fallback_certification
            )
        )


@dataclass(frozen=True)
class FiniteTimeEventCandidate:
    """One certified finite-time event candidate in physical time."""

    event_id: str
    event_type: str
    physical_time_interval: FloatInterval
    parameter_interval: FloatInterval | None = None
    pair: tuple[int, int] | None = None
    certificate: object | None = None
    certified: bool = True
    missing_obligations: tuple[str, ...] = ()

    @property
    def well_formed(self) -> bool:
        return bool(
            self.event_id
            and self.event_type
            and _float_interval_finite_nonempty(self.physical_time_interval)
            and (
                self.parameter_interval is None
                or _float_interval_finite_nonempty(self.parameter_interval)
            )
            and (not self.certified or not self.missing_obligations)
            and all(str(obligation) for obligation in self.missing_obligations)
        )


@dataclass(frozen=True)
class AmbiguousEventOrderPartitionLeaf:
    """One event-order branch that a future state splitter must realize."""

    leaf_id: str
    assumed_first_event_id: str
    event_type: str
    physical_time_interval: FloatInterval
    parameter_interval: FloatInterval | None
    pair: tuple[int, int] | None
    competing_first_event_ids: tuple[str, ...]
    source_event: FiniteTimeEventCandidate
    certified: bool
    missing_obligations: tuple[str, ...] = ()

    @property
    def well_formed(self) -> bool:
        return bool(
            self.leaf_id
            and self.assumed_first_event_id
            and self.event_type
            and _float_interval_finite_nonempty(self.physical_time_interval)
            and (
                self.parameter_interval is None
                or _float_interval_finite_nonempty(self.parameter_interval)
            )
            and self.source_event.well_formed
            and self.source_event.event_id == self.assumed_first_event_id
            and all(str(event_id) for event_id in self.competing_first_event_ids)
            and (not self.certified or not self.missing_obligations)
        )


@dataclass(frozen=True)
class AmbiguousEventOrderSplitCertificate:
    """Constructor-derived finite-time event-order branch alternatives.

    This is not yet a state-space split.  It proves that the listed event
    alternatives are exactly the unresolved first-event leaves that a later
    branch-union state partition must consume.
    """

    target_time_interval: FloatInterval
    ambiguous_events: tuple[FiniteTimeEventCandidate, ...]
    branch_leaves: tuple[AmbiguousEventOrderPartitionLeaf, ...]
    event_alternative_cover_certified: bool
    state_partition_certified: bool
    missing_obligations: tuple[str, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.event_alternative_cover_certified
            and self.state_partition_certified
            and not self.missing_obligations
        )

    @property
    def well_formed(self) -> bool:
        ambiguous_ids = tuple(event.event_id for event in self.ambiguous_events)
        leaf_ids = tuple(leaf.assumed_first_event_id for leaf in self.branch_leaves)
        return bool(
            _float_interval_finite_nonempty(self.target_time_interval)
            and len(ambiguous_ids) >= 2
            and len(set(ambiguous_ids)) == len(ambiguous_ids)
            and set(leaf_ids) == set(ambiguous_ids)
            and all(event.well_formed for event in self.ambiguous_events)
            and all(leaf.well_formed for leaf in self.branch_leaves)
            and all(str(obligation) for obligation in self.missing_obligations)
        )


@dataclass(frozen=True)
class KSEventOrderPartitionLeaf:
    """One KS state-box leaf with a constructor-derived event-order decision."""

    branch_id: str
    state_interval: tuple[tuple[float, float], ...]
    ks_state: IntervalSpatialKSBinaryChartState
    event_set_certificate: object | None
    decision: str
    depth: int
    first_event_id: str | None = None
    first_event_pair: tuple[int, int] | None = None
    first_event_time_interval: FloatInterval | None = None
    certified: bool = False
    missing_obligations: tuple[str, ...] = ()

    @property
    def well_formed(self) -> bool:
        return bool(
            self.branch_id
            and _state_interval_tuple_finite_nonempty(self.state_interval, expected_length=21)
            and getattr(self.ks_state, "certified", False)
            and self.decision
            and self.depth >= 0
            and (
                self.first_event_time_interval is None
                or _float_interval_finite_nonempty(self.first_event_time_interval)
            )
            and (not self.certified or not self.missing_obligations)
            and all(str(obligation) for obligation in self.missing_obligations)
        )


@dataclass(frozen=True)
class KSEventOrderSplitCertificate:
    """Bounded bisection certificate for ambiguous spatial-KS event order."""

    original_state_interval: tuple[tuple[float, float], ...]
    target_time_interval: FloatInterval
    selected_pair: tuple[int, int]
    competing_enter_distance: float
    competing_s_upper: float
    retained_order: int
    computed_order: int
    leaves: tuple[KSEventOrderPartitionLeaf, ...]
    split_count: int
    max_depth: int
    recursive_bisection_cover_certified: bool
    leaf_decisions_certified: bool
    missing_obligations: tuple[str, ...]

    @property
    def certified_leaf_count(self) -> int:
        return sum(leaf.certified for leaf in self.leaves)

    @property
    def ambiguous_leaf_count(self) -> int:
        return sum(not leaf.certified for leaf in self.leaves)

    @property
    def certified(self) -> bool:
        return bool(
            self.recursive_bisection_cover_certified
            and self.leaf_decisions_certified
            and not self.missing_obligations
        )

    @property
    def well_formed(self) -> bool:
        return bool(
            _state_interval_tuple_finite_nonempty(
                self.original_state_interval,
                expected_length=21,
            )
            and _float_interval_finite_nonempty(self.target_time_interval)
            and self.selected_pair in _three_body_pairs()
            and self.competing_enter_distance > 0.0
            and self.competing_s_upper > 0.0
            and self.retained_order >= 1
            and self.computed_order >= self.retained_order
            and self.max_depth >= 0
            and self.leaves
            and all(leaf.well_formed for leaf in self.leaves)
            and all(str(obligation) for obligation in self.missing_obligations)
        )


@dataclass(frozen=True)
class FiniteTimeEventSetCertificate:
    """Interval ordering certificate for the next target/event decision."""

    target_time_interval: FloatInterval
    target_candidate: FiniteTimeEventCandidate
    event_candidates: tuple[FiniteTimeEventCandidate, ...]
    first_event: FiniteTimeEventCandidate | None
    target_before_all_events: bool
    unique_first_event_certified: bool
    multiple_possible_first_events_requiring_split: bool
    missing_obligations: tuple[str, ...]
    ambiguous_event_order_partition: AmbiguousEventOrderSplitCertificate | None = None

    @property
    def no_event_before_target_certified(self) -> bool:
        return self.target_before_all_events

    @property
    def certified(self) -> bool:
        return bool(
            not self.missing_obligations
            and (self.target_before_all_events or self.unique_first_event_certified)
        )

    @property
    def well_formed(self) -> bool:
        event_ids = tuple(candidate.event_id for candidate in self.event_candidates)
        return bool(
            _float_interval_finite_nonempty(self.target_time_interval)
            and self.target_candidate.well_formed
            and all(candidate.well_formed for candidate in self.event_candidates)
            and len(set(event_ids)) == len(event_ids)
            and (not self.first_event or self.first_event.event_id in event_ids)
            and (
                self.ambiguous_event_order_partition is None
                or self.ambiguous_event_order_partition.well_formed
            )
            and all(str(obligation) for obligation in self.missing_obligations)
        )


@dataclass(frozen=True)
class SimultaneousClosePairPartitionBranch:
    """One spatial state-box branch after close-pair ambiguity splitting."""

    branch_id: str
    state_interval: tuple[tuple[float, float], ...]
    pair_distance_bounds: tuple[tuple[tuple[int, int], tuple[float, float]], ...]
    possible_close_pairs: tuple[tuple[int, int], ...]
    certified_close_pairs: tuple[tuple[int, int], ...]
    selected_pair: tuple[int, int] | None
    leaf_type: str
    depth: int
    certified: bool
    missing_obligations: tuple[str, ...] = ()
    triple_collision_exclusion_certificate: object | None = None
    jacobi_cluster_coordinate_certificate: object | None = None
    jacobi_cluster_coordinate_certificates: tuple[object, ...] = ()
    interval_jacobi_cluster_coordinate_certificate: object | None = None
    interval_jacobi_cluster_coordinate_certificates: tuple[object, ...] = ()

    @property
    def triple_collision_excluded(self) -> bool:
        return bool(
            self.triple_collision_exclusion_certificate is not None
            and getattr(self.triple_collision_exclusion_certificate, "certified", False)
        )

    @property
    def jacobi_cluster_coordinates_certified(self) -> bool:
        return bool(
            self.jacobi_cluster_coordinate_certificates
            and all(
                getattr(certificate, "certified", False)
                for certificate in self.jacobi_cluster_coordinate_certificates
            )
        )

    @property
    def interval_jacobi_cluster_coordinates_certified(self) -> bool:
        return bool(
            self.interval_jacobi_cluster_coordinate_certificates
            and all(
                getattr(certificate, "certified", False)
                for certificate in self.interval_jacobi_cluster_coordinate_certificates
            )
        )

    @property
    def well_formed(self) -> bool:
        return bool(
            self.branch_id
            and _state_interval_tuple_finite_nonempty(self.state_interval, expected_length=18)
            and self.pair_distance_bounds
            and all(
                pair in _three_body_pairs()
                and np.isfinite(bounds[0])
                and np.isfinite(bounds[1])
                and 0.0 <= bounds[0] <= bounds[1]
                for pair, bounds in self.pair_distance_bounds
            )
            and all(pair in _three_body_pairs() for pair in self.possible_close_pairs)
            and all(pair in _three_body_pairs() for pair in self.certified_close_pairs)
            and (self.selected_pair is None or self.selected_pair in _three_body_pairs())
            and self.leaf_type
            and self.depth >= 0
            and (not self.certified or not self.missing_obligations)
            and all(str(obligation) for obligation in self.missing_obligations)
        )


@dataclass(frozen=True)
class SimultaneousClosePairSplitCertificate:
    """Axis-aligned branch partition for spatial simultaneous close-pair boxes."""

    original_state_interval: tuple[tuple[float, float], ...]
    binary_distance_threshold: float
    branches: tuple[SimultaneousClosePairPartitionBranch, ...]
    split_count: int
    max_depth: int
    recursive_bisection_cover_certified: bool
    branch_cover_certified: bool
    missing_obligations: tuple[str, ...]

    @property
    def certified_branch_count(self) -> int:
        return sum(1 for branch in self.branches if branch.certified)

    @property
    def ambiguous_branch_count(self) -> int:
        return sum(1 for branch in self.branches if not branch.certified)

    @property
    def state_interval_union(self) -> tuple[tuple[tuple[float, float], ...], ...]:
        return tuple(branch.state_interval for branch in self.branches)

    @property
    def certified(self) -> bool:
        return bool(
            self.recursive_bisection_cover_certified
            and self.branch_cover_certified
            and self.branches
            and all(branch.certified for branch in self.branches)
            and not self.missing_obligations
        )

    @property
    def well_formed(self) -> bool:
        branch_ids = tuple(branch.branch_id for branch in self.branches)
        return bool(
            _state_interval_tuple_finite_nonempty(
                self.original_state_interval,
                expected_length=18,
            )
            and self.binary_distance_threshold > 0.0
            and self.branches
            and len(set(branch_ids)) == len(branch_ids)
            and all(branch.well_formed for branch in self.branches)
            and self.split_count >= 0
            and self.max_depth >= 0
            and all(str(obligation) for obligation in self.missing_obligations)
        )


@dataclass(frozen=True)
class OrdinaryHandoffAdmissibilityCertificate:
    """Proof gate for leaving a spatial KS chart for ordinary coordinates.

    A rho-positive KS projection is only a coordinate-domain certificate.  This
    certificate additionally proves that the ordinary chart is numerically and
    analytically admissible over the requested post-handoff time interval.
    """

    retained_order: int
    requested_post_time_interval: FloatInterval
    min_pair_distance_required: float
    max_acceleration_bound: float
    min_cauchy_radius: float
    max_residual_bound: float
    max_tail_bound: float
    min_pair_distance_lower_bound: float
    lower_squared_distance: float
    acceleration_bound: float
    cauchy_radius: float
    residual_bound: float
    tail_bound: float
    feasible_retained_order: bool
    endpoint_projection_certified: bool
    ordinary_initial_matches_projection: bool
    ordinary_chart_evidence_required: bool
    residual_certified: bool
    tail_certified: bool
    cauchy_majorant_certified: bool
    missing_obligations: tuple[str, ...]

    @property
    def certified(self) -> bool:
        return not self.missing_obligations


@dataclass(frozen=True)
class ValidatedAtlasSolution:
    """Common proof-pipeline object for finite-time validated evaluations."""

    masses: tuple[float, ...]
    target_time: float
    initial_state_interval: Array | None
    charts: tuple[ValidatedChart, ...]
    transitions: tuple[ValidatedTransition, ...]
    invariants: GlobalInvariantLedger
    tail_budget: TailBudgetLedger
    residual_budget: NewtonResidualLedger
    collision_policy: CollisionPolicyWitness
    proof_ledger: ProofLedger
    evaluation: object
    initial_state_interval_union: tuple[Array, ...] | None = None
    selector_trace: FiniteTimeChartSelectorTrace | None = None

    @property
    def chart_count(self) -> int:
        return len(self.charts)

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    @property
    def target_state_interval(self) -> Array:
        return self.evaluation.target_state_interval

    @property
    def target_state_interval_union(self) -> tuple[Array, ...] | None:
        return getattr(self.evaluation, "target_state_interval_union", None)

    @property
    def target_time_certified(self) -> bool:
        return _target_time_in_final_chart_domain(self.target_time, self.charts)

    @property
    def physical_time_chain_progress_certified(self) -> bool:
        return _physical_time_chain_progress_certified(
            self.target_time,
            self.charts,
            require_zero_start=self.selector_trace is not None,
        )

    @property
    def spatial_ks_ordinary_handoff_structure_certified(self) -> bool:
        has_ordinary_after_ks = any(
            chart.chart_type == "spatial_ordinary_taylor_after_ks"
            for chart in self.charts
        )
        if not has_ordinary_after_ks:
            return True
        return _evaluation_contains_certified_ordinary_handoff_admissibility(
            self.evaluation
        )

    @property
    def mass_domain_certified(self) -> bool:
        return _masses_positive_finite(self.masses)

    @property
    def initial_state_interval_certified(self) -> bool:
        if not _interval_array_finite_nonempty(self.initial_state_interval):
            return False
        initial_state = _initial_state_point_from_evaluation(self.evaluation)
        return bool(
            initial_state is None
            or interval_array_contains_point(self.initial_state_interval, initial_state)
        )

    @property
    def initial_state_interval_union_certified(self) -> bool:
        has_hybrid_union_obligation = _proof_ledger_has_entry(
            self.proof_ledger,
            "hybrid_initial_state_union_domain",
        )
        if self.initial_state_interval_union is None:
            return not has_hybrid_union_obligation
        initial_state = _initial_state_point_from_evaluation(self.evaluation)
        expected_union = _expected_initial_state_interval_union_from_evaluation(
            self.evaluation,
        )
        return bool(
            _interval_array_union_finite_nonempty(self.initial_state_interval_union)
            and initial_state is not None
            and _interval_array_union_contains_point(self.initial_state_interval_union, initial_state)
            and (
                not has_hybrid_union_obligation
                or (
                    expected_union is not None
                    and _interval_array_unions_equal(
                        self.initial_state_interval_union,
                        expected_union,
                    )
                )
            )
        )

    @property
    def target_state_interval_certified(self) -> bool:
        try:
            target_interval = self.target_state_interval
        except (AttributeError, TypeError, ValueError):
            return False
        if not _interval_array_finite_nonempty(target_interval):
            return False
        target_state = _target_state_point_from_evaluation(self.evaluation)
        return bool(
            target_state is None
            or interval_array_contains_point(target_interval, target_state)
        )

    @property
    def target_state_interval_union_certified(self) -> bool:
        has_hybrid_union_obligation = _proof_ledger_has_entry(
            self.proof_ledger,
            "hybrid_target_state_union_domain",
        )
        if self.target_state_interval_union is None:
            return not has_hybrid_union_obligation
        target_state = getattr(self.evaluation, "final_state", None)
        expected_union = _expected_target_state_interval_union_from_evaluation(
            self.evaluation,
        )
        return bool(
            _interval_array_union_finite_nonempty(self.target_state_interval_union)
            and target_state is not None
            and _interval_array_union_contains_point(self.target_state_interval_union, target_state)
            and _interval_array_union_subsets(self.target_state_interval_union, self.target_state_interval)
            and (
                not has_hybrid_union_obligation
                or (
                    expected_union is not None
                    and _interval_array_unions_equal(
                        self.target_state_interval_union,
                        expected_union,
                    )
                )
            )
        )

    @property
    def invariant_coverage_matches_charts(self) -> bool:
        return bool(
            self.invariants.certified
            and self.invariants.expected_chart_count == self.chart_count
            and self.invariants.certified_chart_count >= self.chart_count
        )

    @property
    def residual_coverage_matches_charts(self) -> bool:
        return bool(
            self.residual_budget.coverage_certified
            and self.residual_budget.expected_chart_count == self.chart_count
            and self.residual_budget.certified_chart_count >= self.chart_count
        )

    @property
    def proof_ledger_covers_solution(self) -> bool:
        return _proof_ledger_covers_validated_solution(
            self.proof_ledger,
            self.charts,
            self.transitions,
            self.selector_trace,
        )

    @property
    def proof_ledger_missing_coverage_obligations(self) -> tuple[str, ...]:
        return _proof_ledger_missing_coverage_obligations(
            self.proof_ledger,
            self.charts,
            self.transitions,
            self.selector_trace,
        )

    @property
    def missing_certification_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.proof_ledger.well_formed:
            missing.append("proof_ledger_well_formed")
        missing.extend(self.proof_ledger.missing_required_obligations)
        missing.extend(self.proof_ledger_missing_coverage_obligations)
        if self.selector_trace is not None:
            missing.extend(self.selector_trace.missing_obligations)
        if not self.mass_domain_certified:
            missing.append("mass_domain")
        if not self.initial_state_interval_certified:
            missing.append("initial_state_interval")
        if not self.initial_state_interval_union_certified:
            missing.append("initial_state_interval_union")
        if not self.target_state_interval_certified:
            missing.append("target_state_interval")
        if not self.target_state_interval_union_certified:
            missing.append("target_state_interval_union")
        if not self.invariant_coverage_matches_charts:
            missing.append("invariant_coverage_matches_charts")
        if not (self.tail_budget.certified and self.tail_budget.admissible):
            missing.append("tail_budget")
        if not self.residual_coverage_matches_charts:
            missing.append("residual_coverage_matches_charts")
        missing.extend(self.collision_policy.missing_obligations)
        if not self.collision_policy.certified:
            missing.append("collision_policy")
        if not self.target_time_certified:
            missing.append("target_time_domain")
        if not self.physical_time_chain_progress_certified:
            missing.append("physical_time_chain_progress")
        if not self.spatial_ks_ordinary_handoff_structure_certified:
            missing.append("spatial_ks_ordinary_handoff_admissibility_structure")
        if not _selector_trace_matches_validated_atlas(
            self.selector_trace,
            self,
        ):
            missing.append("selector_trace_matches_validated_atlas")
        if not _transition_ledger_connects_charts(self.charts, self.transitions):
            missing.append("transition_ledger_connects_charts")
        if not all(chart.certified for chart in self.charts):
            missing.append("chart_certification")
        if not all(transition.certified for transition in self.transitions):
            missing.append("transition_certification")
        return tuple(dict.fromkeys(missing))

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.proof_ledger.certified
            and self.proof_ledger_covers_solution
            and self.mass_domain_certified
            and self.initial_state_interval_certified
            and self.initial_state_interval_union_certified
            and self.target_state_interval_certified
            and self.target_state_interval_union_certified
            and self.invariant_coverage_matches_charts
            and self.tail_budget.certified
            and self.tail_budget.admissible
            and self.residual_coverage_matches_charts
            and self.collision_policy.certified
            and self.collision_policy.well_formed
            and self.target_time_certified
            and self.physical_time_chain_progress_certified
            and self.spatial_ks_ordinary_handoff_structure_certified
            and _selector_trace_matches_validated_atlas(
                self.selector_trace,
                self,
            )
            and _transition_ledger_connects_charts(self.charts, self.transitions)
            and all(chart.certified for chart in self.charts)
            and all(transition.certified for transition in self.transitions)
        )

    def target_state_contains(self, state: Array) -> bool:
        if hasattr(self.evaluation, "target_state_contains"):
            return bool(self.evaluation.target_state_contains(state))
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))


def _evaluation_contains_certified_ordinary_handoff_admissibility(
    evaluation: object,
    *,
    _visited: set[int] | None = None,
) -> bool:
    """Find the constructor-produced KS-to-ordinary handoff certificate."""

    if evaluation is None:
        return False
    if _visited is None:
        _visited = set()
    object_id = id(evaluation)
    if object_id in _visited:
        return False
    _visited.add(object_id)

    handoff = getattr(evaluation, "ordinary_handoff_admissibility", None)
    if handoff is not None and bool(getattr(handoff, "certified", False)):
        return True

    for attribute in (
        "ks_evaluation",
        "ks_competing_evaluation",
        "next_ks_evaluation",
        "branch_union_evaluation",
    ):
        nested = getattr(evaluation, attribute, None)
        if _evaluation_contains_certified_ordinary_handoff_admissibility(
            nested,
            _visited=_visited,
        ):
            return True

    for atlas in getattr(evaluation, "branch_atlases", ()) or ():
        if _evaluation_contains_certified_ordinary_handoff_admissibility(
            getattr(atlas, "evaluation", None),
            _visited=_visited,
        ):
            return True

    return False


def _transition_ledger_connects_charts(
    charts: tuple[ValidatedChart, ...],
    transitions: tuple[ValidatedTransition, ...],
) -> bool:
    if not charts:
        return False
    if not _chart_ids_unique(charts):
        return False
    if len(transitions) != max(0, len(charts) - 1):
        return False
    return all(
        transition.source_chart_id == left.chart_id
        and transition.target_chart_id == right.chart_id
        and transition.transition_type
        and transition.source
        and _float_intervals_touch_or_overlap(
            left.physical_time_interval,
            right.physical_time_interval,
        )
        for transition, left, right in zip(transitions, charts, charts[1:])
    )


def _chart_ids_unique(charts: tuple[ValidatedChart, ...]) -> bool:
    chart_ids = tuple(str(getattr(chart, "chart_id", "")) for chart in charts)
    return bool(
        chart_ids
        and all(chart_id for chart_id in chart_ids)
        and len(set(chart_ids)) == len(chart_ids)
    )


def _float_intervals_touch_or_overlap(left: object, right: object) -> bool:
    if not (
        _float_interval_finite_nonempty(left)
        and _float_interval_finite_nonempty(right)
    ):
        return False
    left_lower, left_upper = _float_interval_bounds(left)
    right_lower, right_upper = _float_interval_bounds(right)
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(left_lower),
        abs(left_upper),
        abs(right_lower),
        abs(right_upper),
    )
    return bool(
        left_upper >= right_lower - tolerance
        and right_upper >= left_lower - tolerance
    )


def finite_time_selector_trace_binding_token(atlas: object) -> str:
    """Stable token binding a finite-time selector trace to one atlas surface."""

    signature = (
        "finite_time_selector_trace_v1",
        _selector_signature_floats(getattr(atlas, "masses", ())),
        _selector_signature_float(getattr(atlas, "target_time", float("nan"))),
        _selector_interval_array_signature(
            getattr(atlas, "initial_state_interval", None),
        ),
        tuple(
            (
                str(getattr(chart, "chart_id", "")),
                str(getattr(chart, "chart_type", "")),
                str(getattr(chart, "source", "")),
                _selector_float_interval_signature(
                    getattr(chart, "parameter_interval", None),
                ),
                _selector_float_interval_signature(
                    getattr(chart, "physical_time_interval", None),
                ),
                _selector_signature_float(getattr(chart, "tail_bound", 0.0)),
            )
            for chart in getattr(atlas, "charts", ())
        ),
        tuple(
            (
                str(getattr(transition, "source_chart_id", "")),
                str(getattr(transition, "target_chart_id", "")),
                str(getattr(transition, "transition_type", "")),
                bool(getattr(transition, "certified", False)),
                str(getattr(transition, "source", "")),
            )
            for transition in getattr(atlas, "transitions", ())
        ),
        tuple(
            (
                str(getattr(entry, "name", "")),
                bool(getattr(entry, "certified", False)),
                str(getattr(entry, "source", "")),
                bool(getattr(entry, "required", True)),
            )
            for entry in getattr(getattr(atlas, "proof_ledger", None), "entries", ())
            if str(getattr(entry, "name", "")) != "finite_time_chart_selector"
        ),
    )
    return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()


def _selector_signature_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return format(number, ".17g")


def _selector_signature_floats(values: object) -> tuple[str, ...]:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return ()
    return tuple(_selector_signature_float(value) for value in array)


def _selector_float_interval_signature(interval: object) -> tuple[str, str]:
    if interval is None:
        return ("missing", "missing")
    try:
        lower, upper = _float_interval_bounds(interval)
    except (AttributeError, TypeError, ValueError):
        return ("invalid", "invalid")
    return (_selector_signature_float(lower), _selector_signature_float(upper))


def _selector_interval_array_signature(intervals: object) -> tuple[tuple[str, str], ...]:
    if intervals is None:
        return ()
    try:
        interval_array = np.asarray(intervals, dtype=object).reshape(-1)
    except (TypeError, ValueError):
        return (("invalid", "invalid"),)
    return tuple(_selector_float_interval_signature(interval) for interval in interval_array)


def _selector_trace_matches_validated_atlas(
    selector_trace: FiniteTimeChartSelectorTrace | None,
    atlas: ValidatedAtlasSolution,
) -> bool:
    charts = atlas.charts
    proof_ledger = atlas.proof_ledger
    has_selector_ledger_entry = _proof_ledger_has_entry(
        proof_ledger,
        "finite_time_chart_selector",
    )
    if selector_trace is None:
        return not has_selector_ledger_entry
    if not has_selector_ledger_entry:
        return False
    if not selector_trace.certified:
        return False
    if selector_trace.atlas_binding_token != finite_time_selector_trace_binding_token(
        atlas,
    ):
        return False
    selected = selector_trace.selected_route_id
    chart_types = {chart.chart_type for chart in charts}
    if selected in {"auto_spatial_ks", "explicit_spatial_ks"}:
        allowed_spatial_ks_types = {
            "spatial_ordinary_taylor_before_ks",
            "spatial_ks_binary",
            "spatial_ordinary_taylor_after_ks",
        }
        return bool(
            charts
            and "spatial_ks_binary" in chart_types
            and chart_types <= allowed_spatial_ks_types
        )
    if selected == "spatial_ks_event_order_branch_union":
        return bool(charts and chart_types <= {"spatial_ks_event_order_branch_union"})
    if selected == "spatial_ks_prefix_event_order_branch_union":
        return bool(
            charts
            and "spatial_ks_event_order_branch_union" in chart_types
            and (
                "spatial_ks_binary" in chart_types
                or "spatial_ordinary_taylor_before_ks" in chart_types
            )
            and chart_types <= {
                "spatial_ordinary_taylor_before_ks",
                "spatial_ks_binary",
                "spatial_ks_event_order_branch_union",
            }
        )
    if selected == "spatial_ks_prefix_close_pair_branch_union":
        return bool(
            charts
            and "spatial_ks_binary" in chart_types
            and "spatial_branch_union" in chart_types
            and chart_types <= {
                "spatial_ordinary_taylor_before_ks",
                "spatial_ks_binary",
                "spatial_branch_union",
            }
        )
    if selected == "spatial_branch_union":
        return bool(charts and chart_types <= {"spatial_branch_union"})
    if selected == "planar_hybrid":
        return bool(
            charts
            and chart_types <= {
                "planar_ordinary_taylor",
                "planar_levi_civita_binary",
            }
        )
    if selected == "compactified_sundman":
        return bool(
            charts
            and chart_types <= {
                "compactified_sundman",
                "compactified_sundman_target",
            }
        )
    if selected == "sundman":
        return bool(charts and chart_types <= {"sundman", "sundman_target"})
    if selected == "unrestricted_reduced_evaluation":
        return bool(charts)
    return False


def _proof_ledger_covers_validated_solution(
    proof_ledger: ProofLedger,
    charts: tuple[ValidatedChart, ...],
    transitions: tuple[ValidatedTransition, ...],
    selector_trace: FiniteTimeChartSelectorTrace | None,
) -> bool:
    required_groups = _required_proof_ledger_groups_for_solution(
        charts,
        transitions,
        selector_trace,
    )
    return all(
        any(
            _proof_ledger_has_certified_entry(proof_ledger, name)
            for name in group
        )
        for group in required_groups
    )


def _proof_ledger_missing_coverage_obligations(
    proof_ledger: ProofLedger,
    charts: tuple[ValidatedChart, ...],
    transitions: tuple[ValidatedTransition, ...],
    selector_trace: FiniteTimeChartSelectorTrace | None,
) -> tuple[str, ...]:
    return tuple(
        _proof_ledger_coverage_obligation_name(group)
        for group in _required_proof_ledger_groups_for_solution(
            charts,
            transitions,
            selector_trace,
        )
        if not any(
            _proof_ledger_has_certified_entry(proof_ledger, name)
            for name in group
        )
    )


def _proof_ledger_coverage_obligation_name(group: tuple[str, ...]) -> str:
    if len(group) == 1:
        return f"proof_ledger_coverage:{group[0]}"
    return "proof_ledger_coverage:any_of(" + "|".join(group) + ")"


def _required_proof_ledger_groups_for_solution(
    charts: tuple[ValidatedChart, ...],
    transitions: tuple[ValidatedTransition, ...],
    selector_trace: FiniteTimeChartSelectorTrace | None,
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = [
        ("mass_domain",),
        (
            "initial_state_domain",
            "hybrid_initial_state_domain",
            "spatial_ks_branch_domain",
        ),
        ("target_time_domain",),
        (
            "target_time",
            "target_containment",
            "hybrid_requested_target_time",
            "hybrid_target_containment",
            "finite_time_physical_targeting",
            "spatial_ks_target_time_projection",
        ),
        (
            "projection_ledger",
            "center_of_mass_reduction",
            "spatial_ordinary_to_ks_branch_lift",
            "spatial_ks_to_ks_branch_lift",
            "spatial_ks_projection_constraints",
            "spatial_ks_rho_positive_endpoint_projection",
            "spatial_ks_target_time_projection",
        ),
        (
            "newton_residuals",
            "spatial_ordinary_entry_residuals",
            "spatial_ks_equation_residuals",
        ),
        (
            "invariant_ledger",
            "spatial_ordinary_ks_and_post_invariants",
            "spatial_ks_and_ordinary_invariants",
        ),
        (
            "tail_budget",
            "hybrid_cauchy_tail_budget",
            "local_tail_budget",
            "exact_parabolic_tail",
            "homothetic_scalar_cauchy_tail",
        ),
        ("collision_policy", "spatial_collision_policy_scope"),
    ]
    chart_types = {
        str(getattr(chart, "chart_type", ""))
        for chart in charts
    }
    transition_types = {
        str(getattr(transition, "transition_type", ""))
        for transition in transitions
    }
    if selector_trace is not None:
        groups.append(("finite_time_chart_selector",))
    if any(
        chart_type in {
            "sundman",
            "sundman_target",
            "compactified_sundman",
            "compactified_sundman_target",
            "initial_identity",
        }
        for chart_type in chart_types
    ):
        groups.append(("chart_dynamics",))
    if any(
        chart_type in {"planar_ordinary_taylor", "planar_levi_civita_binary"}
        for chart_type in chart_types
    ):
        groups.extend(
            [
                ("hybrid_interval_chart_choices",),
                ("hybrid_event_isolation",),
                ("hybrid_cauchy_tail_budget",),
                ("hybrid_target_containment",),
            ]
        )
        if transitions:
            groups.append(("hybrid_chart_transitions",))
    if "spatial_ordinary_taylor_before_ks" in chart_types:
        groups.extend(
            [
                ("spatial_ordinary_ks_entry_event_isolation",),
                ("spatial_ordinary_to_ks_branch_lift",),
                ("spatial_ordinary_entry_residuals",),
                ("ordinary_to_ks_transition",),
            ]
        )
    ks_projection_group = (
        "spatial_ks_rho_positive_endpoint_projection",
        "spatial_ks_target_time_projection",
    )
    ks_exit_or_target_group = (
        "spatial_ks_exit_event_isolation",
        "spatial_ks_competing_entry_event_isolation",
        "spatial_ks_target_inside_regularized_chart",
        "ordinary_handoff_admissibility",
    )
    ks_target_or_handoff_group = (
        "spatial_ks_target_inside_regularized_chart",
        "ordinary_handoff_admissibility",
    )
    if "spatial_ks_event_order_branch_union" in chart_types:
        ks_projection_group = (
            *ks_projection_group,
            "spatial_ks_prefix_event_order_branch_union_transition",
        )
        ks_exit_or_target_group = (
            *ks_exit_or_target_group,
            "spatial_ks_prefix_event_order_branch_union_transition",
        )
        ks_target_or_handoff_group = (
            *ks_target_or_handoff_group,
            "spatial_ks_prefix_event_order_branch_union_transition",
        )
    if "spatial_branch_union" in chart_types and "spatial_ks_binary" in chart_types:
        ks_projection_group = (
            *ks_projection_group,
            "spatial_ks_prefix_close_pair_branch_union_transition",
        )
        ks_exit_or_target_group = (
            *ks_exit_or_target_group,
            "spatial_ks_prefix_close_pair_branch_union_transition",
        )
        ks_target_or_handoff_group = (
            *ks_target_or_handoff_group,
            "spatial_ks_prefix_close_pair_branch_union_transition",
        )
    if "spatial_ks_binary" in chart_types:
        groups.extend(
            [
                ("spatial_ks_equation_residuals",),
                ("spatial_ks_projection_constraints",),
                ks_projection_group,
                ks_exit_or_target_group,
                ks_target_or_handoff_group,
            ]
        )
    if "spatial_ordinary_taylor_after_ks" in chart_types:
        groups.extend(
            [
                ("ordinary_handoff_admissibility",),
                ("ordinary_post_handoff_residuals",),
                ("rho_positive_handoff_transition",),
            ]
        )
    if "spatial_branch_union" in chart_types:
        groups.extend(
            [
                ("simultaneous_close_pair_partition",),
                ("finite_time_branch_union_consumption",),
            ]
        )
    if "spatial_ks_event_order_branch_union" in chart_types:
        groups.extend(
            [
                ("ks_event_order_partition",),
                ("finite_time_event_order_branch_union_consumption",),
            ]
        )
    if "finite_jet_identity_selector_total_collision" in chart_types:
        groups.extend(
            [
                ("projection_ledger",),
                ("target_containment",),
                (
                    "parabolic_homothetic_central_configuration",
                    "homothetic_central_configuration",
                ),
                ("exact_parabolic_tail", "homothetic_scalar_cauchy_tail"),
            ]
        )
    if any("ordinary_to_ks" in transition_type for transition_type in transition_types):
        groups.append(("ordinary_to_ks_transition",))
    if any("ks_to_ks_competing" in transition_type for transition_type in transition_types):
        groups.append(("spatial_ks_to_ks_transition",))
    if any("rho_positive" in transition_type for transition_type in transition_types):
        groups.append(("rho_positive_handoff_transition",))
    return tuple(groups)


def _proof_ledger_has_entry(proof_ledger: ProofLedger, name: str) -> bool:
    return any(
        str(getattr(entry, "name", "")) == name
        for entry in getattr(proof_ledger, "entries", ())
    )


def _proof_ledger_has_certified_entry(proof_ledger: ProofLedger, name: str) -> bool:
    return any(
        str(getattr(entry, "name", "")) == name
        and bool(getattr(entry, "certified", False))
        for entry in getattr(proof_ledger, "entries", ())
    )


@dataclass(frozen=True)
class HybridValidatedEvaluation:
    """Evaluation facade exposing a hybrid continuation through atlas APIs."""

    hybrid_solution: object
    target_state_interval: Array
    lohner_enclosure: object | None = None
    set_enclosure: object | None = None
    target_interval_source: str = "hybrid_step_interval"
    initial_state_interval_union: tuple[Array, ...] | None = None
    target_state_interval_union: tuple[Array, ...] | None = None
    target_reference_state: Array | None = None
    target_step_index: int | None = None

    @property
    def final_state(self) -> Array:
        if self.target_reference_state is not None:
            return np.asarray(self.target_reference_state, dtype=float).reshape(-1)
        return self.hybrid_solution.final_state

    def target_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))

    def lohner_ordinary_set_propagated_interval_enclosure(
        self,
        *,
        retained_order: int | None = None,
    ) -> object:
        """Expose the hybrid solution's ordinary/event Lohner propagation path."""

        if retained_order is None and self.lohner_enclosure is not None:
            return self.lohner_enclosure
        return self.hybrid_solution.lohner_ordinary_set_propagated_interval_enclosure(
            retained_order=retained_order,
        )

    def ordinary_set_propagated_interval_enclosure(
        self,
        *,
        retained_order: int | None = None,
        ordinary_substeps: int = 1,
    ) -> object:
        """Expose the hybrid solution's ordinary/binary set-propagation path."""

        if (
            retained_order is None
            and int(ordinary_substeps) == 1
            and self.set_enclosure is not None
        ):
            return self.set_enclosure
        return self.hybrid_solution.ordinary_set_propagated_interval_enclosure(
            retained_order=retained_order,
            ordinary_substeps=ordinary_substeps,
        )


@dataclass(frozen=True)
class HybridTargetEnclosure:
    """Target enclosure selected from a finite hybrid chart prefix."""

    target_state_interval: Array
    target_state_interval_union: tuple[Array, ...] | None
    target_reference_state: Array
    target_interval_source: str
    step_index: int | None
    target_time_in_chart: bool
    certified: bool
    detail: str


@dataclass(frozen=True)
class SpatialKSValidatedEvaluation:
    """Evaluation facade for a local spatial KS-to-ordinary handoff."""

    ks_solution: object
    ordinary_solution: IntervalTaylorSolution | None
    endpoint_projection: SpatialKSBinaryPhysicalProjectionCertificate
    exit_event_certificate: SpatialKSRhoExitEventCertificate | None
    target_state_interval: Array
    target_time_interval: FloatInterval
    ordinary_handoff_admissibility: OrdinaryHandoffAdmissibilityCertificate | None = None
    ks_tail_certificate: object | None = None

    def target_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))


@dataclass(frozen=True)
class SpatialKSTargetTailEvaluation:
    endpoint_projection: SpatialKSBinaryPhysicalProjectionCertificate
    target_state_interval: Array
    tail_certificate: object
    tail_bound: float
    tail_certified: bool
    proof_detail: str


@dataclass(frozen=True)
class SpatialKSSegmentTailEvaluation:
    tail_certificate: object
    tail_bound: float
    tail_certified: bool
    final_state: IntervalSpatialKSBinaryChartState | None
    physical_time_delta: FloatInterval | None
    proof_detail: str


@dataclass(frozen=True)
class SpatialKSCompetingHandoffEvaluation:
    """Evaluation facade for a local spatial KS-to-competing-KS handoff."""

    first_ks_solution: object
    initial_projection: SpatialKSBinaryPhysicalProjectionCertificate
    competing_entry_event_certificate: SpatialKSEntryEventCertificate
    next_ks_state: IntervalSpatialKSBinaryChartState
    next_ks_evaluation: SpatialKSValidatedEvaluation
    target_state_interval: Array
    target_time_interval: FloatInterval
    first_collision_policy_certificate: SpatialLocalCollisionPolicyCertificate | None = None
    next_collision_policy_certificate: SpatialLocalCollisionPolicyCertificate | None = None
    first_ks_tail_certificate: object | None = None
    loop_progress_certificate: object | None = None

    @property
    def endpoint_projection(self) -> SpatialKSBinaryPhysicalProjectionCertificate:
        return self.next_ks_evaluation.endpoint_projection

    def target_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))


@dataclass(frozen=True)
class SpatialKSPrefixEventOrderBranchUnionEvaluation:
    """Evaluation facade for a KS prefix followed by an event-order branch union."""

    first_ks_solution: object
    initial_projection: SpatialKSBinaryPhysicalProjectionCertificate
    competing_entry_event_certificate: SpatialKSEntryEventCertificate
    next_ks_state: IntervalSpatialKSBinaryChartState
    branch_union_evaluation: SpatialBranchUnionValidatedEvaluation
    target_state_interval: Array
    target_state_interval_union: tuple[Array, ...]
    target_time_interval: FloatInterval
    first_collision_policy_certificate: SpatialLocalCollisionPolicyCertificate | None = None
    first_ks_tail_certificate: object | None = None
    loop_progress_certificate: object | None = None

    @property
    def next_ks_evaluation(self) -> SpatialBranchUnionValidatedEvaluation:
        return self.branch_union_evaluation

    @property
    def initial_state(self) -> Array:
        return _interval_midpoint_state(
            _flat_float_interval_array(self.initial_projection.state_interval)
        )

    @property
    def final_state(self) -> Array:
        if self.target_state_interval_union:
            return _interval_midpoint_state(self.target_state_interval_union[0])
        return _interval_midpoint_state(self.target_state_interval)

    def target_state_contains(self, state: Array) -> bool:
        state = np.asarray(state, dtype=float).reshape(-1)
        return bool(
            _interval_array_union_contains_point(self.target_state_interval_union, state)
            or interval_array_contains_point(self.target_state_interval, state)
        )


@dataclass(frozen=True)
class SpatialOrdinaryKSHandoffEvaluation:
    """Evaluation facade for an ordinary-to-KS-to-ordinary local handoff."""

    ordinary_entry_solution: IntervalTaylorSolution
    entry_event_certificate: SpatialKSEntryEventCertificate
    entry_ks_state: IntervalSpatialKSBinaryChartState
    entry_projection: SpatialKSBinaryPhysicalProjectionCertificate
    ks_evaluation: SpatialKSValidatedEvaluation
    target_state_interval: Array
    target_time_interval: FloatInterval
    collision_policy_certificate: SpatialLocalCollisionPolicyCertificate | None = None

    @property
    def initial_state(self) -> Array:
        return np.concatenate(
            [
                _interval_midpoint_array(self.ordinary_entry_solution.position[0]).reshape(-1),
                _interval_midpoint_array(self.ordinary_entry_solution.velocity[0]).reshape(-1),
            ]
        )

    @property
    def exit_event_certificate(self) -> SpatialKSRhoExitEventCertificate | None:
        return self.ks_evaluation.exit_event_certificate

    @property
    def endpoint_projection(self) -> SpatialKSBinaryPhysicalProjectionCertificate:
        return self.ks_evaluation.endpoint_projection

    def target_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))


@dataclass(frozen=True)
class SpatialOrdinaryKSCompetingHandoffEvaluation:
    """Evaluation facade for an ordinary-to-KS-to-competing-KS handoff."""

    ordinary_entry_solution: IntervalTaylorSolution
    entry_event_certificate: SpatialKSEntryEventCertificate
    entry_ks_state: IntervalSpatialKSBinaryChartState
    entry_projection: SpatialKSBinaryPhysicalProjectionCertificate
    ks_competing_evaluation: SpatialKSCompetingHandoffEvaluation
    target_state_interval: Array
    target_time_interval: FloatInterval
    collision_policy_certificate: SpatialLocalCollisionPolicyCertificate | None = None
    loop_progress_certificate: object | None = None

    @property
    def endpoint_projection(self) -> SpatialKSBinaryPhysicalProjectionCertificate:
        return self.ks_competing_evaluation.endpoint_projection

    def target_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))


@dataclass(frozen=True)
class SpatialOrdinaryKSSuffixAtlasEvaluation:
    """Evaluation facade for an ordinary entry chart prepended to a KS suffix."""

    ordinary_entry_solution: IntervalTaylorSolution
    entry_event_certificate: SpatialKSEntryEventCertificate
    entry_ks_state: IntervalSpatialKSBinaryChartState
    entry_projection: SpatialKSBinaryPhysicalProjectionCertificate
    ks_suffix_evaluation: object
    target_state_interval: Array
    target_time_interval: FloatInterval
    target_state_interval_union: tuple[Array, ...] | None = None
    collision_policy_certificate: SpatialLocalCollisionPolicyCertificate | None = None
    loop_progress_certificate: object | None = None

    @property
    def branch_union_evaluation(self) -> object | None:
        if hasattr(self.ks_suffix_evaluation, "branch_partition"):
            return self.ks_suffix_evaluation
        return getattr(self.ks_suffix_evaluation, "branch_union_evaluation", None)

    @property
    def final_state(self) -> Array:
        if self.target_state_interval_union:
            return _interval_midpoint_state(self.target_state_interval_union[0])
        return _interval_midpoint_state(self.target_state_interval)

    def target_state_contains(self, state: Array) -> bool:
        state = np.asarray(state, dtype=float).reshape(-1)
        return bool(
            (
                self.target_state_interval_union is not None
                and _interval_array_union_contains_point(
                    self.target_state_interval_union,
                    state,
                )
            )
            or interval_array_contains_point(self.target_state_interval, state)
        )


@dataclass(frozen=True)
class SpatialBranchUnionValidatedEvaluation:
    """Evaluation facade for a finite spatial close-pair branch union."""

    branch_partition: SimultaneousClosePairSplitCertificate
    branch_atlases: tuple[ValidatedAtlasSolution, ...]
    initial_state_interval: Array
    initial_state_interval_union: tuple[Array, ...]
    target_state_interval: Array
    target_state_interval_union: tuple[Array, ...]
    initial_state: Array
    final_state: Array
    target_time_interval: FloatInterval | None = None

    def target_state_contains(self, state: Array) -> bool:
        state = np.asarray(state, dtype=float).reshape(-1)
        return bool(
            _interval_array_union_contains_point(self.target_state_interval_union, state)
            or interval_array_contains_point(self.target_state_interval, state)
        )


@dataclass(frozen=True)
class TimeReversedValidatedEvaluation:
    """Evaluation facade for an exact time-reversal transform of an atlas."""

    forward_evaluation: object
    target_state_interval: Array
    final_state: Array
    target_state_interval_union: tuple[Array, ...] | None = None

    def target_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))


@dataclass(frozen=True)
class ParabolicHomotheticTotalCollisionChartCertificate:
    """Exact cubic-time homothetic total-collision chart certificate."""

    branch: HomotheticTotalCollisionBranch
    start_tau: float
    target_tau: float
    event_time: float
    max_newton_residual: float
    max_center_of_mass: float
    max_linear_momentum: float
    max_angular_momentum: float
    max_energy: float
    coefficient_tail_bound: float
    tolerance: float

    @property
    def central_configuration_certified(self) -> bool:
        return bool(self.branch.certified_scaled_central_configuration)

    @property
    def parabolic_exact_tail_certified(self) -> bool:
        return bool(
            abs(float(self.branch.energy_per_inertia)) <= self.tolerance
            and np.isfinite(self.coefficient_tail_bound)
            and self.coefficient_tail_bound <= self.tolerance
        )

    @property
    def residual_certified(self) -> bool:
        return bool(
            np.isfinite(self.max_newton_residual)
            and self.max_newton_residual <= self.tolerance
        )

    @property
    def invariants_certified(self) -> bool:
        return bool(
            np.isfinite(self.max_center_of_mass)
            and np.isfinite(self.max_linear_momentum)
            and np.isfinite(self.max_angular_momentum)
            and np.isfinite(self.max_energy)
            and self.max_center_of_mass <= self.tolerance
            and self.max_linear_momentum <= self.tolerance
            and self.max_angular_momentum <= self.tolerance
            and self.max_energy <= self.tolerance
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.start_tau < 0.0 < self.target_tau
            and np.isfinite(self.event_time)
            and self.central_configuration_certified
            and self.parabolic_exact_tail_certified
            and self.residual_certified
            and self.invariants_certified
        )


@dataclass(frozen=True)
class EnergyHomotheticTotalCollisionChartCertificate:
    """Finite-energy homothetic total-collision chart with scalar tail bounds."""

    branch: HomotheticTotalCollisionBranch
    scalar_majorant: HomotheticTotalCollisionScalarMajorantCertificate
    start_tau: float
    target_tau: float
    event_time: float
    max_newton_residual: float
    max_center_of_mass: float
    max_linear_momentum: float
    max_angular_momentum: float
    max_energy_error: float
    scalar_tail_bound: float
    derivative_tail_bound: float
    state_tail_bound: float
    tolerance: float

    @property
    def central_configuration_certified(self) -> bool:
        return bool(self.branch.certified_scaled_central_configuration)

    @property
    def scalar_tail_certified(self) -> bool:
        return bool(
            self.scalar_majorant.certified
            and np.isfinite(self.scalar_tail_bound)
            and np.isfinite(self.derivative_tail_bound)
            and np.isfinite(self.state_tail_bound)
            and self.scalar_tail_bound >= 0.0
            and self.derivative_tail_bound >= 0.0
            and self.state_tail_bound >= 0.0
        )

    @property
    def scalar_dynamics_certified(self) -> bool:
        return bool(
            self.scalar_tail_certified
            and np.linalg.norm(
                self.branch.energy_recurrence_residual_coefficients(),
                ord=np.inf,
            )
            <= self.tolerance
        )

    @property
    def residual_certified(self) -> bool:
        return bool(
            self.scalar_dynamics_certified
            and np.isfinite(self.max_newton_residual)
            and self.max_newton_residual <= self.tolerance
        )

    @property
    def invariants_certified(self) -> bool:
        return bool(
            np.isfinite(self.max_center_of_mass)
            and np.isfinite(self.max_linear_momentum)
            and np.isfinite(self.max_angular_momentum)
            and np.isfinite(self.max_energy_error)
            and self.max_center_of_mass <= self.tolerance
            and self.max_linear_momentum <= self.tolerance
            and self.max_angular_momentum <= self.tolerance
            and self.max_energy_error <= self.tolerance
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.start_tau < 0.0 < self.target_tau
            and np.isfinite(self.event_time)
            and self.central_configuration_certified
            and self.scalar_tail_certified
            and self.residual_certified
            and self.invariants_certified
        )


@dataclass(frozen=True)
class HomotheticTotalCollisionValidatedEvaluation:
    """Evaluation facade for an exact homothetic total-collision chart."""

    branch: HomotheticTotalCollisionBranch
    certificate: object
    initial_state: Array
    target_state: Array
    target_state_interval: Array

    @property
    def final_state(self) -> Array:
        return self.target_state

    def target_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))


def validated_atlas_from_parabolic_homothetic_total_collision_branch(
    branch: HomotheticTotalCollisionBranch,
    *,
    start_tau: float,
    target_tau: float,
    event_time: float = 0.0,
    tolerance: float = 1.0e-9,
    source: str = "parabolic_homothetic_total_collision",
) -> ValidatedAtlasSolution:
    """Expose the exact parabolic homothetic collision branch as an atlas.

    For ``energy_per_inertia=0`` the homothetic branch is exactly
    ``q=tau^2 Q`` with ``t=event_time+tau^3``.  The scaled central
    configuration equation ``A(Q)=-(2/9)Q`` gives Newton's equation on both
    punctured sides, while ``tau=0`` is handled as the explicit selector
    collision event in the regularized chart.
    """

    if not isinstance(branch, HomotheticTotalCollisionBranch):
        raise TypeError("branch must be a HomotheticTotalCollisionBranch")
    start_tau = float(start_tau)
    target_tau = float(target_tau)
    event_time = float(event_time)
    tolerance = float(tolerance)
    if not (
        np.isfinite(start_tau)
        and np.isfinite(target_tau)
        and start_tau < 0.0 < target_tau
    ):
        raise ValueError("start_tau and target_tau must straddle total collision")
    if not np.isfinite(event_time):
        raise ValueError("event_time must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if abs(float(branch.energy_per_inertia)) > tolerance:
        raise ValueError("only the exact parabolic homothetic branch has zero tail")

    masses = np.asarray(branch.masses, dtype=float).reshape(-1)
    if masses.shape != (3,) or np.any(masses <= 0.0) or not np.all(np.isfinite(masses)):
        raise ValueError("branch masses must be positive and finite")
    if branch.quadratic_coefficient.ndim != 2 or branch.quadratic_coefficient.shape[0] != 3:
        raise ValueError("branch quadratic coefficient must have three body rows")
    if branch.quadratic_coefficient.shape[1] < 2:
        raise ValueError("validated total-collision chart requires dimension at least two")

    coefficients = branch.coefficients
    coefficient_tail_bound = float(np.linalg.norm(coefficients[1:], ord=np.inf))
    sample_taus = (start_tau, 0.5 * start_tau, 0.5 * target_tau, target_tau)
    states = tuple(branch.state_at_tau(tau) for tau in sample_taus)
    newton_residuals = tuple(
        float(np.linalg.norm(branch.newton_residual_at_tau(tau), ord=np.inf))
        for tau in sample_taus
    )
    center_bounds = tuple(
        float(np.linalg.norm(center_of_mass(state, masses), ord=np.inf))
        for state in states
    )
    momentum_bounds = tuple(
        float(np.linalg.norm(linear_momentum(state, masses), ord=np.inf))
        for state in states
    )
    angular_bounds = tuple(
        float(np.linalg.norm(angular_momentum_components(state, masses), ord=np.inf))
        for state in states
    )
    energy_bounds = tuple(abs(float(energy(state, masses))) for state in states)
    certificate = ParabolicHomotheticTotalCollisionChartCertificate(
        branch=branch,
        start_tau=start_tau,
        target_tau=target_tau,
        event_time=event_time,
        max_newton_residual=float(max(newton_residuals, default=float("inf"))),
        max_center_of_mass=float(max(center_bounds, default=float("inf"))),
        max_linear_momentum=float(max(momentum_bounds, default=float("inf"))),
        max_angular_momentum=float(max(angular_bounds, default=float("inf"))),
        max_energy=float(max(energy_bounds, default=float("inf"))),
        coefficient_tail_bound=coefficient_tail_bound,
        tolerance=tolerance,
    )
    if not certificate.certified:
        raise ValueError("parabolic homothetic total-collision chart did not certify")

    initial_state = branch.state_at_tau(start_tau)
    target_state = branch.state_at_tau(target_tau)
    initial_interval = _point_interval_array(initial_state)
    target_interval = _point_interval_array(target_state)
    target_time = event_time + target_tau**3
    physical_interval = FloatInterval(
        min(event_time + start_tau**3, event_time + target_tau**3),
        max(event_time + start_tau**3, event_time + target_tau**3),
    )
    chart = ValidatedChart(
        chart_id="homothetic_total_collision_0",
        chart_type="finite_jet_identity_selector_total_collision",
        source=source,
        parameter_name="tau",
        parameter_interval=FloatInterval(start_tau, target_tau),
        physical_time_interval=physical_interval,
        dynamics_certified=certificate.certified,
        residual_certified=certificate.residual_certified,
        projection_certified=True,
        invariants_certified=certificate.invariants_certified,
        tail_certified=certificate.parabolic_exact_tail_certified,
        tail_bound=certificate.coefficient_tail_bound,
    )
    invariants = GlobalInvariantLedger(
        center_of_mass_certified=certificate.invariants_certified,
        linear_momentum_certified=certificate.invariants_certified,
        angular_momentum_certified=certificate.invariants_certified,
        energy_certified=certificate.invariants_certified,
        certified_chart_count=1,
        expected_chart_count=1,
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=certificate.coefficient_tail_bound,
        max_step_tail_bound=certificate.coefficient_tail_bound,
        certified=certificate.parabolic_exact_tail_certified,
    )
    residual_budget = NewtonResidualLedger(
        certified=certificate.residual_certified,
        certified_chart_count=1,
        expected_chart_count=1,
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="finite_time_exact_homothetic_total_collision_selector",
        binary_policy="no_binary_chart_in_exact_total_collision_branch",
        total_collision_policy="finite_time_identity_selector_total_collision",
        triple_collision_status="continued_by_explicit_homothetic_selector",
        triple_collision_reason=(
            "parabolic homothetic branch uses q=tau^2 Q, t=tau^3 with "
            "A(Q)=-(2/9)Q"
        ),
        certified=certificate.certified,
    )
    evaluation = HomotheticTotalCollisionValidatedEvaluation(
        branch=branch,
        certificate=certificate,
        initial_state=initial_state,
        target_state=target_state,
        target_state_interval=target_interval,
    )
    proof_ledger = ProofLedger(
        entries=(
            ProofLedgerEntry(
                "mass_domain",
                _masses_positive_finite(masses),
                source,
            ),
            ProofLedgerEntry(
                "initial_state_domain",
                _interval_array_finite_nonempty(initial_interval),
                source,
            ),
            ProofLedgerEntry(
                "parabolic_homothetic_central_configuration",
                certificate.central_configuration_certified,
                source,
                detail="A(Q)=-(2/9)Q",
            ),
            ProofLedgerEntry(
                "exact_parabolic_tail",
                certificate.parabolic_exact_tail_certified,
                source,
                detail="energy_per_inertia=0 gives u(tau)=1 and zero omitted tail",
            ),
            ProofLedgerEntry(
                "projection_ledger",
                True,
                source,
                detail="regularized tau chart projects by q=tau^2Q and t=tau^3 on punctured sides",
            ),
            ProofLedgerEntry(
                "newton_residuals",
                residual_budget.certified,
                source,
            ),
            ProofLedgerEntry(
                "invariant_ledger",
                invariants.certified,
                source,
            ),
            ProofLedgerEntry(
                "target_time",
                _target_time_in_final_chart_domain(target_time, (chart,)),
                source,
            ),
            ProofLedgerEntry(
                "target_containment",
                evaluation.target_state_contains(target_state),
                source,
            ),
            ProofLedgerEntry(
                "target_time_domain",
                _target_time_in_final_chart_domain(target_time, (chart,)),
                source,
            ),
            ProofLedgerEntry(
                "collision_policy",
                collision_policy.certified,
                source,
            ),
        )
    )
    return ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in masses),
        target_time=float(target_time),
        initial_state_interval=initial_interval,
        charts=(chart,),
        transitions=(),
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=proof_ledger,
        evaluation=evaluation,
    )


def validated_atlas_from_homothetic_total_collision_branch(
    branch: HomotheticTotalCollisionBranch,
    *,
    start_tau: float,
    target_tau: float,
    event_time: float = 0.0,
    scalar_majorant: HomotheticTotalCollisionScalarMajorantCertificate | None = None,
    z_radius: float | None = None,
    u_radius: float = 0.2,
    tolerance: float = 1.0e-8,
    source: str = "homothetic_total_collision_energy_series",
) -> ValidatedAtlasSolution:
    """Expose a finite-energy homothetic collision branch as a validated chart."""

    if not isinstance(branch, HomotheticTotalCollisionBranch):
        raise TypeError("branch must be a HomotheticTotalCollisionBranch")
    start_tau = float(start_tau)
    target_tau = float(target_tau)
    event_time = float(event_time)
    tolerance = float(tolerance)
    if not (
        np.isfinite(start_tau)
        and np.isfinite(target_tau)
        and start_tau < 0.0 < target_tau
    ):
        raise ValueError("start_tau and target_tau must straddle total collision")
    if not np.isfinite(event_time):
        raise ValueError("event_time must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    masses = np.asarray(branch.masses, dtype=float).reshape(-1)
    if masses.shape != (3,) or np.any(masses <= 0.0) or not np.all(np.isfinite(masses)):
        raise ValueError("branch masses must be positive and finite")
    if branch.quadratic_coefficient.ndim != 2 or branch.quadratic_coefficient.shape[0] != 3:
        raise ValueError("branch quadratic coefficient must have three body rows")
    if branch.quadratic_coefficient.shape[1] < 2:
        raise ValueError("validated total-collision chart requires dimension at least two")

    max_abs_tau = max(abs(start_tau), abs(target_tau))
    if scalar_majorant is None:
        if z_radius is None:
            z_radius = max(4.0 * max_abs_tau**2, 1.0e-6)
        scalar_majorant = certify_homothetic_total_collision_scalar_majorant(
            branch,
            z_radius=float(z_radius),
            u_radius=float(u_radius),
        )
    if not scalar_majorant.certified:
        raise ValueError("homothetic total-collision scalar majorant did not certify")

    scalar_tail = scalar_majorant.scalar_tail_bound(
        max_abs_tau=max_abs_tau,
        retained_order=branch.order,
    )
    derivative_tail = scalar_majorant.derivative_combination_tail_bound(
        max_abs_tau=max_abs_tau,
        retained_order=branch.order,
    )
    shape_bound = float(np.linalg.norm(branch.quadratic_coefficient, ord=np.inf))
    position_tail = max_abs_tau**2 * scalar_tail * shape_bound
    velocity_tail = (
        (2.0 / (3.0 * max_abs_tau)) * derivative_tail * shape_bound
        if max_abs_tau > 0.0
        else float("inf")
    )
    state_tail = float(max(position_tail, velocity_tail))
    sample_taus = (start_tau, 0.5 * start_tau, 0.5 * target_tau, target_tau)
    states = tuple(branch.state_at_tau(tau) for tau in sample_taus)
    newton_residuals = tuple(
        float(np.linalg.norm(branch.newton_residual_at_tau(tau), ord=np.inf))
        for tau in sample_taus
    )
    center_bounds = tuple(
        float(np.linalg.norm(center_of_mass(state, masses), ord=np.inf))
        for state in states
    )
    momentum_bounds = tuple(
        float(np.linalg.norm(linear_momentum(state, masses), ord=np.inf))
        for state in states
    )
    angular_bounds = tuple(
        float(np.linalg.norm(angular_momentum_components(state, masses), ord=np.inf))
        for state in states
    )
    expected_energy = float(branch.energy_per_inertia * branch.inertia)
    energy_errors = tuple(
        abs(float(energy(state, masses)) - expected_energy)
        for state in states
    )
    certificate = EnergyHomotheticTotalCollisionChartCertificate(
        branch=branch,
        scalar_majorant=scalar_majorant,
        start_tau=start_tau,
        target_tau=target_tau,
        event_time=event_time,
        max_newton_residual=float(max(newton_residuals, default=float("inf"))),
        max_center_of_mass=float(max(center_bounds, default=float("inf"))),
        max_linear_momentum=float(max(momentum_bounds, default=float("inf"))),
        max_angular_momentum=float(max(angular_bounds, default=float("inf"))),
        max_energy_error=float(max(energy_errors, default=float("inf"))),
        scalar_tail_bound=float(scalar_tail),
        derivative_tail_bound=float(derivative_tail),
        state_tail_bound=state_tail,
        tolerance=tolerance,
    )
    if not certificate.certified:
        raise ValueError("homothetic total-collision energy chart did not certify")

    initial_state = branch.state_at_tau(start_tau)
    target_state = branch.state_at_tau(target_tau)
    initial_interval = _inflate_interval_array(initial_state.reshape(-1), state_tail)
    target_interval = _inflate_interval_array(target_state.reshape(-1), state_tail)
    target_time = event_time + target_tau**3
    physical_interval = FloatInterval(
        min(event_time + start_tau**3, event_time + target_tau**3),
        max(event_time + start_tau**3, event_time + target_tau**3),
    )
    chart = ValidatedChart(
        chart_id="homothetic_total_collision_0",
        chart_type="finite_jet_identity_selector_total_collision",
        source=source,
        parameter_name="tau",
        parameter_interval=FloatInterval(start_tau, target_tau),
        physical_time_interval=physical_interval,
        dynamics_certified=certificate.scalar_dynamics_certified,
        residual_certified=certificate.residual_certified,
        projection_certified=True,
        invariants_certified=certificate.invariants_certified,
        tail_certified=certificate.scalar_tail_certified,
        tail_bound=certificate.state_tail_bound,
    )
    invariants = GlobalInvariantLedger(
        center_of_mass_certified=certificate.invariants_certified,
        linear_momentum_certified=certificate.invariants_certified,
        angular_momentum_certified=certificate.invariants_certified,
        energy_certified=certificate.invariants_certified,
        certified_chart_count=1,
        expected_chart_count=1,
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=certificate.state_tail_bound,
        max_step_tail_bound=certificate.state_tail_bound,
        certified=certificate.scalar_tail_certified,
    )
    residual_budget = NewtonResidualLedger(
        certified=certificate.residual_certified,
        certified_chart_count=1,
        expected_chart_count=1,
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="finite_time_homothetic_total_collision_energy_selector",
        binary_policy="no_binary_chart_in_homothetic_total_collision_branch",
        total_collision_policy="finite_time_identity_selector_total_collision",
        triple_collision_status="continued_by_explicit_homothetic_selector",
        triple_collision_reason=(
            "homothetic branch uses q=tau^2 u(tau^2)Q, t=T+tau^3 "
            "with constructor-certified scalar Cauchy tail"
        ),
        certified=certificate.certified,
    )
    evaluation = HomotheticTotalCollisionValidatedEvaluation(
        branch=branch,
        certificate=certificate,
        initial_state=initial_state,
        target_state=target_state,
        target_state_interval=target_interval,
    )
    proof_ledger = ProofLedger(
        entries=(
            ProofLedgerEntry("mass_domain", _masses_positive_finite(masses), source),
            ProofLedgerEntry(
                "initial_state_domain",
                _interval_array_finite_nonempty(initial_interval),
                source,
            ),
            ProofLedgerEntry(
                "homothetic_central_configuration",
                certificate.central_configuration_certified,
                source,
                detail="A(Q)=-(2/9)Q",
            ),
            ProofLedgerEntry(
                "homothetic_scalar_cauchy_tail",
                certificate.scalar_tail_certified,
                source,
                detail="Rouche/Cauchy majorant for u(tau^2)",
            ),
            ProofLedgerEntry(
                "projection_ledger",
                True,
                source,
                detail="regularized tau chart projects by q=tau^2 u(tau^2)Q and t=T+tau^3",
            ),
            ProofLedgerEntry("newton_residuals", residual_budget.certified, source),
            ProofLedgerEntry("invariant_ledger", invariants.certified, source),
            ProofLedgerEntry(
                "target_time",
                _target_time_in_final_chart_domain(target_time, (chart,)),
                source,
            ),
            ProofLedgerEntry(
                "target_containment",
                evaluation.target_state_contains(target_state),
                source,
            ),
            ProofLedgerEntry(
                "target_time_domain",
                _target_time_in_final_chart_domain(target_time, (chart,)),
                source,
            ),
            ProofLedgerEntry("collision_policy", collision_policy.certified, source),
        )
    )
    return ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in masses),
        target_time=float(target_time),
        initial_state_interval=initial_interval,
        charts=(chart,),
        transitions=(),
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=proof_ledger,
        evaluation=evaluation,
    )


def validated_atlas_from_unrestricted_evaluation(
    evaluation: object,
    *,
    masses: Array,
    target_time: float,
    source: str = "evaluate_unrestricted_solution",
) -> ValidatedAtlasSolution:
    """Build a shared atlas object from the finite-time evaluator output."""

    reduced_target = getattr(evaluation, "reduced_target", None)
    charts = _charts_from_reduced_target(evaluation, reduced_target, source=source)
    transitions = _transitions_from_charts(charts, source=source)
    initial_state_interval = _initial_state_interval_from_evaluation(evaluation)
    expected_chart_count = len(charts)
    invariants = _invariant_ledger_from_evaluation(
        evaluation,
        reduced_target,
        expected_chart_count=expected_chart_count,
    )
    residual_budget = _residual_ledger_from_target(
        evaluation,
        reduced_target,
        expected_chart_count=expected_chart_count,
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=float(getattr(evaluation, "local_tail_bound", 0.0)),
        max_step_tail_bound=float(getattr(evaluation, "max_step_tail_bound", 0.0)),
        certified=bool(getattr(evaluation, "tail_certified", False)),
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="finite_time_stop_or_selector_for_total_collision",
        binary_policy="ordinary_sundman_until_binary_atlas_integration",
        total_collision_policy="stop_or_require_explicit_selector",
        triple_collision_status=str(getattr(evaluation, "triple_collision_status", "missing")),
        triple_collision_reason=getattr(evaluation, "triple_collision_exclusion_reason", None),
        certified=str(getattr(evaluation, "triple_collision_status", "missing")) != "missing",
    )
    proof_ledger = ProofLedger(
        entries=(
            ProofLedgerEntry(
                "center_of_mass_reduction",
                bool(getattr(evaluation, "reduction_certified", False)),
                source,
            ),
            ProofLedgerEntry(
                "mass_domain",
                _masses_positive_finite(masses),
                source,
            ),
            ProofLedgerEntry(
                "initial_state_domain",
                _interval_array_finite_nonempty(initial_state_interval),
                source,
            ),
            ProofLedgerEntry(
                "target_time",
                bool(getattr(evaluation, "target_time_certified", False)),
                source,
            ),
            ProofLedgerEntry(
                "target_time_domain",
                _target_time_in_final_chart_domain(float(target_time), charts),
                source,
            ),
            ProofLedgerEntry(
                "chart_dynamics",
                bool(getattr(evaluation, "dynamics_certified", False)),
                source,
            ),
            ProofLedgerEntry(
                "tail_budget",
                tail_budget.certified and tail_budget.finite,
                source,
            ),
            ProofLedgerEntry(
                "newton_residuals",
                residual_budget.certified,
                source,
            ),
            ProofLedgerEntry(
                "invariant_ledger",
                invariants.certified,
                source,
            ),
            ProofLedgerEntry(
                "collision_policy",
                collision_policy.certified,
                source,
            ),
        )
    )
    return ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in np.asarray(masses, dtype=float).reshape(-1)),
        target_time=float(target_time),
        initial_state_interval=initial_state_interval,
        charts=charts,
        transitions=transitions,
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=proof_ledger,
        evaluation=evaluation,
    )


def time_reverse_validated_atlas_solution(
    atlas: ValidatedAtlasSolution,
    *,
    source: str = "time_reversal_symmetry",
) -> ValidatedAtlasSolution:
    """Transform a positive-time validated atlas into the reversed-time atlas.

    Newton's equations are invariant under ``(t,q,v) -> (-t,q,-v)``.  This
    constructor uses that exact symmetry to reuse a proof-certified forward
    atlas for reversed initial velocities as a proof-certified atlas for the
    original data at negative time.
    """

    if not isinstance(atlas, ValidatedAtlasSolution):
        raise TypeError("atlas must be a ValidatedAtlasSolution")
    charts = tuple(
        replace(
            chart,
            source=f"{source}:{chart.source}",
            physical_time_interval=_time_reverse_float_interval(
                chart.physical_time_interval
            ),
        )
        for chart in atlas.charts
    )
    transitions = tuple(
        replace(
            transition,
            source=f"{source}:{transition.source}",
        )
        for transition in atlas.transitions
    )
    initial_interval = _time_reverse_state_interval(atlas.initial_state_interval)
    initial_interval_union = (
        tuple(
            _time_reverse_state_interval(member)
            for member in atlas.initial_state_interval_union
        )
        if atlas.initial_state_interval_union is not None
        else None
    )
    target_interval = _time_reverse_state_interval(atlas.target_state_interval)
    target_interval_union = (
        tuple(
            _time_reverse_state_interval(member)
            for member in atlas.target_state_interval_union
        )
        if atlas.target_state_interval_union is not None
        else None
    )
    final_state = _time_reverse_state_vector(
        getattr(atlas.evaluation, "final_state", _interval_midpoint_state(atlas.target_state_interval))
    )
    target_time = -float(atlas.target_time)
    evaluation = TimeReversedValidatedEvaluation(
        forward_evaluation=atlas.evaluation,
        target_state_interval=target_interval,
        final_state=final_state,
        target_state_interval_union=target_interval_union,
    )
    proof_ledger = ProofLedger(
        entries=(
            *atlas.proof_ledger.entries,
            ProofLedgerEntry(
                "time_reversal_symmetry",
                bool(atlas.proof_certified),
                source,
                detail="Newton equations and invariant ledgers are preserved by (t,q,v)->(-t,q,-v)",
            ),
            ProofLedgerEntry(
                "target_time_domain",
                _target_time_in_final_chart_domain(target_time, charts),
                source,
                detail="time-reversed target time lies in the reversed final chart interval",
            ),
        )
    )
    return ValidatedAtlasSolution(
        masses=atlas.masses,
        target_time=target_time,
        initial_state_interval=initial_interval,
        charts=charts,
        transitions=transitions,
        invariants=atlas.invariants,
        tail_budget=atlas.tail_budget,
        residual_budget=atlas.residual_budget,
        collision_policy=replace(
            atlas.collision_policy,
            policy_id=f"time_reversed_{atlas.collision_policy.policy_id}",
        ),
        proof_ledger=proof_ledger,
        evaluation=evaluation,
        initial_state_interval_union=initial_interval_union,
    )


def validated_atlas_from_hybrid_solution(
    hybrid_solution: object,
    *,
    target_time: float | None = None,
    source: str = "continue_hybrid",
) -> ValidatedAtlasSolution:
    """Expose a planar hybrid ordinary/binary continuation as an atlas.

    Hybrid steps certify interval chart selection, event isolation, Cauchy tail
    bounds, coefficient residuals, projection, and invariant ledgers.  This
    adapter keeps those obligations explicit when exposing the hybrid path
    through the shared theorem-pipeline object.
    """

    target_time = (
        float(np.asarray(getattr(hybrid_solution, "times"), dtype=float)[-1])
        if target_time is None
        else float(target_time)
    )
    all_charts = _charts_from_hybrid_solution(hybrid_solution, source=source)
    has_binary = any(getattr(step, "chart", None) == "binary" for step in getattr(hybrid_solution, "steps", ()))
    collision_policy = CollisionPolicyWitness(
        policy_id="finite_time_planar_hybrid_stop_or_regularize_binary",
        binary_policy=(
            "planar_levi_civita_binary_regularized"
            if has_binary
            else "ordinary_hybrid_no_binary_event"
        ),
        total_collision_policy="finite_time_stop_before_total_collision",
        triple_collision_status="not_classified_by_planar_hybrid",
        triple_collision_reason=(
            "planar hybrid continuation certifies ordinary/binary chart choices, "
            "but does not prove a global triple-collision exclusion"
        ),
        certified=bool(getattr(hybrid_solution, "proof_certified", False)),
    )
    initial_state_interval = _hybrid_initial_state_interval(hybrid_solution)
    initial_state_interval_union = _hybrid_initial_state_interval_union(hybrid_solution)
    has_split_initial_union = len(initial_state_interval_union or ()) > 1
    lohner_enclosure = (
        None
        if has_split_initial_union
        else _hybrid_lohner_target_enclosure(hybrid_solution)
    )
    set_enclosure = (
        None
        if lohner_enclosure is not None
        else _hybrid_set_target_enclosure(hybrid_solution)
    )
    target_enclosure = _hybrid_target_enclosure_for_time(
        hybrid_solution,
        target_time=target_time,
        lohner_enclosure=lohner_enclosure,
        set_enclosure=set_enclosure,
    )
    charts = (
        all_charts[: target_enclosure.step_index + 1]
        if target_enclosure.step_index is not None
        else all_charts
    )
    transitions = _transitions_from_hybrid_steps(hybrid_solution, charts, source=source)
    expected_chart_count = len(charts)
    invariants = GlobalInvariantLedger(
        center_of_mass_certified=_hybrid_invariant_count(hybrid_solution, "center_of_mass") >= expected_chart_count,
        linear_momentum_certified=_hybrid_invariant_count(hybrid_solution, "linear_momentum") >= expected_chart_count,
        angular_momentum_certified=_hybrid_invariant_count(hybrid_solution, "angular_momentum") >= expected_chart_count,
        energy_certified=_hybrid_invariant_count(hybrid_solution, "energy") >= expected_chart_count,
        certified_chart_count=min(
            _hybrid_invariant_count(hybrid_solution, "center_of_mass"),
            _hybrid_invariant_count(hybrid_solution, "linear_momentum"),
            _hybrid_invariant_count(hybrid_solution, "angular_momentum"),
            _hybrid_invariant_count(hybrid_solution, "energy"),
        ),
        expected_chart_count=expected_chart_count,
    )
    residual_count = sum(_step_residual_certified(step) for step in getattr(hybrid_solution, "steps", ()))
    residual_budget = NewtonResidualLedger(
        certified=residual_count >= expected_chart_count,
        certified_chart_count=int(residual_count),
        expected_chart_count=expected_chart_count,
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=float(getattr(hybrid_solution, "local_tail_bound", 0.0)),
        max_step_tail_bound=float(getattr(hybrid_solution, "max_step_tail_bound", 0.0)),
        certified=bool(
            getattr(hybrid_solution, "proof_certified", False)
            and all(
                _step_tail_certified(step)
                for step in tuple(getattr(hybrid_solution, "steps", ()))[:expected_chart_count]
            )
        ),
    )
    evaluation = HybridValidatedEvaluation(
        hybrid_solution=hybrid_solution,
        target_state_interval=target_enclosure.target_state_interval,
        lohner_enclosure=lohner_enclosure,
        set_enclosure=set_enclosure,
        target_interval_source=target_enclosure.target_interval_source,
        initial_state_interval_union=initial_state_interval_union,
        target_state_interval_union=target_enclosure.target_state_interval_union,
        target_reference_state=target_enclosure.target_reference_state,
        target_step_index=target_enclosure.step_index,
    )
    proof_ledger = ProofLedger(
        entries=(
            ProofLedgerEntry(
                "hybrid_interval_chart_choices",
                bool(
                    expected_chart_count > 0
                    and getattr(hybrid_solution, "interval_chart_certified_step_count", 0)
                    >= expected_chart_count
                ),
                source,
            ),
            ProofLedgerEntry(
                "mass_domain",
                _masses_positive_finite(getattr(hybrid_solution, "masses", ())),
                source,
            ),
            ProofLedgerEntry(
                "hybrid_initial_state_domain",
                _interval_array_finite_nonempty(initial_state_interval)
                and _hybrid_initial_state_contains_point(hybrid_solution, initial_state_interval),
                source,
            ),
            ProofLedgerEntry(
                "hybrid_initial_state_union_domain",
                _interval_array_union_finite_nonempty(initial_state_interval_union)
                and _hybrid_initial_state_union_contains_point(hybrid_solution, initial_state_interval_union),
                source,
            ),
            ProofLedgerEntry(
                "hybrid_event_isolation",
                bool(all(getattr(step, "event_certified", False) for step in getattr(hybrid_solution, "steps", ()))),
                source,
            ),
            ProofLedgerEntry(
                "hybrid_chart_transitions",
                bool(
                    _transition_ledger_connects_charts(charts, transitions)
                    and all(transition.certified for transition in transitions)
                ),
                source,
                detail="adjacent hybrid charts must pass the constructor-derived endpoint interval union forward",
            ),
            ProofLedgerEntry(
                "hybrid_cauchy_tail_budget",
                tail_budget.certified and tail_budget.finite,
                source,
            ),
            ProofLedgerEntry(
                "hybrid_set_target_enclosure",
                target_enclosure.target_interval_source in {
                    "lohner_ordinary_set_propagation",
                    "hybrid_set_propagation",
                    "hybrid_chart_target_enclosure",
                },
                source,
                detail=f"target_interval_source={target_enclosure.target_interval_source}",
            ),
            ProofLedgerEntry(
                "hybrid_target_enclosure_chain",
                _hybrid_target_enclosure_chain_certified(
                    hybrid_solution,
                    target_enclosure,
                    lohner_enclosure=lohner_enclosure,
                    set_enclosure=set_enclosure,
                    expected_chart_count=expected_chart_count,
                ),
                source,
                detail=(
                    "target enclosure metadata must match the hybrid chart prefix "
                    "that produced the atlas"
                ),
            ),
            ProofLedgerEntry(
                "hybrid_target_containment",
                evaluation.target_state_contains(target_enclosure.target_reference_state),
                source,
            ),
            ProofLedgerEntry(
                "hybrid_target_state_union_domain",
                _interval_array_union_finite_nonempty(target_enclosure.target_state_interval_union)
                and _interval_array_union_contains_point(
                    target_enclosure.target_state_interval_union,
                    target_enclosure.target_reference_state,
                )
                and _interval_array_union_subsets(
                    target_enclosure.target_state_interval_union,
                    target_enclosure.target_state_interval,
                ),
                source,
            ),
            ProofLedgerEntry(
                "hybrid_requested_target_time",
                target_enclosure.certified and target_enclosure.target_time_in_chart,
                source,
                detail=target_enclosure.detail,
            ),
            ProofLedgerEntry(
                "target_time_domain",
                _target_time_in_final_chart_domain(target_time, charts),
                source,
                detail="requested target time must lie in the final hybrid chart time interval",
            ),
            ProofLedgerEntry(
                "projection_ledger",
                bool(all(_step_projection_certified(step) for step in getattr(hybrid_solution, "steps", ()))),
                source,
                detail="binary projection is certified only on punctured rho-positive intervals",
            ),
            ProofLedgerEntry(
                "newton_residuals",
                residual_budget.certified,
                source,
                detail="ordinary charts certify Newton residuals directly; binary charts certify the lifted Levi-Civita equations with punctured projection",
            ),
            ProofLedgerEntry(
                "invariant_ledger",
                invariants.certified,
                source,
                detail="binary charts use regularized finite center-of-mass, momentum, angular-momentum, and energy identities",
            ),
            ProofLedgerEntry(
                "collision_policy",
                collision_policy.certified,
                source,
            ),
        )
    )
    return ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in np.asarray(getattr(hybrid_solution, "masses"), dtype=float).reshape(-1)),
        target_time=target_time,
        initial_state_interval=initial_state_interval,
        charts=charts,
        transitions=transitions,
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=proof_ledger,
        evaluation=evaluation,
        initial_state_interval_union=initial_state_interval_union,
    )


def validated_atlas_from_spatial_ordinary_ks_handoff(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    *,
    pair: tuple[int, int],
    enter_distance: float,
    entry_time_upper: float,
    branch: str,
    s_endpoint: float | None = None,
    exit_rho: float | None = None,
    s_upper: float | None = None,
    ordinary_step_size: float | None = None,
    target_time: float | None = None,
    retained_order: int,
    guard_order: int,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    competing_pair_min_distance_required: float = 0.0,
    source: str = "spatial_ordinary_ks_local_handoff",
) -> ValidatedAtlasSolution:
    """Compose ordinary entry, spatial KS binary, and ordinary exit charts.

    The constructor derives the local ordinary-to-KS entry certificate before
    delegating the KS segment and ordinary exit handoff to
    :func:`validated_atlas_from_spatial_ks_binary_chart`.  It is deliberately
    still a local selected-binary certificate, not a public unrestricted
    finite-time theorem.
    """

    retained_order = int(retained_order)
    guard_order = int(guard_order)
    competing_pair_min_distance_required = float(competing_pair_min_distance_required)
    if retained_order < 1:
        raise ValueError("retained_order must be positive")
    if guard_order < 1:
        raise ValueError("guard_order must be positive")

    masses = np.asarray(masses, dtype=float)
    pair = tuple(pair)
    computed_order = retained_order + guard_order
    ordinary_positions, ordinary_velocities = _spatial_interval_state_arrays(state_interval)
    ordinary_entry_solution = construct_interval_taylor_solution_from_intervals(
        ordinary_positions,
        ordinary_velocities,
        masses,
        order=computed_order,
    )
    entry_event_certificate = certify_spatial_ordinary_ks_entry_event(
        ordinary_entry_solution,
        pair=pair,
        enter_distance=enter_distance,
        time_upper=entry_time_upper,
        coefficient_count=retained_order,
    )
    if not entry_event_certificate.certified:
        raise ValueError("ordinary-to-KS entry event could not be interval-certified")
    entry_time_interval = FloatInterval(*entry_event_certificate.root_interval)
    entry_ks_state = spatial_ordinary_entry_event_to_ks_chart_state(
        ordinary_entry_solution,
        entry_event_certificate,
        branch=branch,
    )
    if not entry_ks_state.certified:
        raise ValueError("ordinary-to-KS entry event did not lift to a certified KS branch")
    entry_projection = project_spatial_ks_binary_interval_chart_state_to_physical(
        entry_ks_state,
        physical_time=entry_time_interval,
    )
    if not entry_projection.certified:
        raise ValueError("ordinary-to-KS branch lift did not project to a rho-positive state")

    target_time_after_ks_start_interval = (
        None
        if target_time is None
        else FloatInterval.point(float(target_time)) - entry_time_interval
    )
    ks_atlas = validated_atlas_from_spatial_ks_binary_chart(
        entry_ks_state,
        s_endpoint=s_endpoint,
        exit_rho=exit_rho,
        s_upper=s_upper,
        ordinary_step_size=ordinary_step_size,
        target_time_after_ks_start_interval=target_time_after_ks_start_interval,
        retained_order=retained_order,
        guard_order=guard_order,
        ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
        ordinary_handoff_max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
        ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
        ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
        ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
        source=source,
    )

    ordinary_entry_residual = certify_ordinary_interval_taylor_equations(
        ordinary_entry_solution,
        coefficient_count=retained_order,
    )
    ordinary_time_series = _ordinary_physical_time_series(computed_order)
    ordinary_entry_center = certify_interval_center_of_mass_motion(
        ordinary_entry_solution.position,
        ordinary_entry_solution.velocity,
        ordinary_time_series,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_momentum = certify_interval_linear_momentum_conservation(
        ordinary_entry_solution.velocity,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_angular = certify_interval_centered_angular_momentum_conservation(
        ordinary_entry_solution.position,
        ordinary_entry_solution.velocity,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_energy = certify_interval_total_energy_conservation(
        ordinary_entry_solution.position,
        ordinary_entry_solution.velocity,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_tail = interval_guarded_tail_certificate(
        ordinary_interval_solution_arrays(ordinary_entry_solution),
        retained_order,
        entry_time_interval.upper,
    )
    ordinary_entry_domain = FloatInterval(0.0, entry_time_interval.upper)
    ordinary_entry_invariants_certified = bool(
        ordinary_entry_center.certified
        and ordinary_entry_momentum.certified
        and ordinary_entry_angular.certified
        and ordinary_entry_energy.certified
    )
    ordinary_entry_chart = ValidatedChart(
        chart_id="ordinary_before_spatial_ks_0",
        chart_type="spatial_ordinary_taylor_before_ks",
        source=source,
        parameter_name="t",
        parameter_interval=ordinary_entry_domain,
        physical_time_interval=ordinary_entry_domain,
        dynamics_certified=ordinary_entry_residual.certified,
        residual_certified=ordinary_entry_residual.certified,
        projection_certified=True,
        invariants_certified=ordinary_entry_invariants_certified,
        tail_certified=ordinary_entry_tail.is_nontrivial,
        tail_bound=ordinary_entry_tail.tail_bound,
    )

    shifted_ks_charts = tuple(
        replace(
            chart,
            chart_id=(
                "spatial_ks_1"
                if index == 0
                else "ordinary_after_spatial_ks_2"
            ),
            physical_time_interval=_shift_optional_interval(
                chart.physical_time_interval,
                entry_time_interval,
            ),
        )
        for index, chart in enumerate(ks_atlas.charts)
    )
    charts = (ordinary_entry_chart, *shifted_ks_charts)
    transitions = (
        ValidatedTransition(
            source_chart_id=charts[0].chart_id,
            target_chart_id=charts[1].chart_id,
            transition_type="spatial_ordinary_to_ks_decreasing_distance_entry",
            certified=bool(
                entry_event_certificate.certified
                and entry_ks_state.certified
                and entry_projection.certified
            ),
            source=source,
        ),
        *(
            replace(
                transition,
                source_chart_id=charts[index + 1].chart_id,
                target_chart_id=charts[index + 2].chart_id,
            )
            for index, transition in enumerate(ks_atlas.transitions)
        ),
    )
    invariants = GlobalInvariantLedger(
        center_of_mass_certified=bool(
            ordinary_entry_center.certified and ks_atlas.invariants.center_of_mass_certified
        ),
        linear_momentum_certified=bool(
            ordinary_entry_momentum.certified and ks_atlas.invariants.linear_momentum_certified
        ),
        angular_momentum_certified=bool(
            ordinary_entry_angular.certified and ks_atlas.invariants.angular_momentum_certified
        ),
        energy_certified=bool(
            ordinary_entry_energy.certified and ks_atlas.invariants.energy_certified
        ),
        certified_chart_count=int(ordinary_entry_chart.invariants_certified)
        + ks_atlas.invariants.certified_chart_count,
        expected_chart_count=len(charts),
    )
    residual_budget = NewtonResidualLedger(
        certified=bool(ordinary_entry_residual.certified and ks_atlas.residual_budget.certified),
        certified_chart_count=int(ordinary_entry_residual.certified)
        + ks_atlas.residual_budget.certified_chart_count,
        expected_chart_count=len(charts),
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=float(
            ordinary_entry_tail.tail_bound + ks_atlas.tail_budget.local_tail_bound
        ),
        max_step_tail_bound=float(
            max(ordinary_entry_tail.tail_bound, ks_atlas.tail_budget.max_step_tail_bound)
        ),
        certified=bool(ordinary_entry_tail.is_nontrivial and ks_atlas.tail_budget.certified),
    )
    target_time_interval = (
        FloatInterval.point(float(target_time))
        if target_time is not None
        else entry_time_interval + ks_atlas.evaluation.target_time_interval
    )
    has_ordinary_post_handoff = bool(
        len(ks_atlas.charts) > 1 and ks_atlas.evaluation.ordinary_solution is not None
    )
    collision_policy_certificate = _certify_spatial_local_collision_policy(
        pair=pair,
        masses=masses,
        ordinary_entry_solution=ordinary_entry_solution,
        ordinary_entry_domain=ordinary_entry_domain,
        ordinary_entry_tail_bound=ordinary_entry_tail.tail_bound,
        ks_solution=ks_atlas.evaluation.ks_solution,
        ks_parameter_interval=ks_atlas.charts[0].parameter_interval,
        ks_tail_bound=ks_atlas.charts[0].tail_bound,
        ordinary_post_solution=ks_atlas.evaluation.ordinary_solution,
        ordinary_post_parameter_interval=(
            ks_atlas.charts[1].parameter_interval if has_ordinary_post_handoff else None
        ),
        ordinary_post_tail_bound=(
            ks_atlas.charts[1].tail_bound if has_ordinary_post_handoff else 0.0
        ),
        retained_order=retained_order,
        competing_pair_min_distance_required=competing_pair_min_distance_required,
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="local_spatial_ordinary_ks_binary_handoff",
        binary_policy="spatial_ks_selected_binary_entry_and_local_regularization",
        total_collision_policy=(
            "selected_binary_only_local_third_body_separated"
            if collision_policy_certificate.certified
            else "not_classified_by_local_ordinary_ks_handoff"
        ),
        triple_collision_status=(
            "locally_excluded_on_handoff_charts"
            if collision_policy_certificate.certified
            else "not_classified_by_local_ordinary_ks_handoff"
        ),
        triple_collision_reason=(
            "ordinary chart domains avoid all pair collisions and the KS chart "
            "keeps the third body separated from the selected binary"
            if collision_policy_certificate.certified
            else "local ordinary/KS handoff has not certified competing-event "
            "separation on every chart domain"
        ),
        certified=collision_policy_certificate.certified,
    )
    evaluation = SpatialOrdinaryKSHandoffEvaluation(
        ordinary_entry_solution=ordinary_entry_solution,
        entry_event_certificate=entry_event_certificate,
        entry_ks_state=entry_ks_state,
        entry_projection=entry_projection,
        ks_evaluation=ks_atlas.evaluation,
        target_state_interval=ks_atlas.target_state_interval,
        target_time_interval=target_time_interval,
        collision_policy_certificate=collision_policy_certificate,
    )
    exit_event_certificate = ks_atlas.evaluation.exit_event_certificate
    proof_ledger = ProofLedger(
        entries=(
            ProofLedgerEntry(
                "spatial_ordinary_entry_residuals",
                ordinary_entry_residual.certified,
                source,
            ),
            ProofLedgerEntry(
                "mass_domain",
                _masses_positive_finite(masses),
                source,
            ),
            ProofLedgerEntry(
                "initial_state_domain",
                _interval_array_finite_nonempty(_flat_float_interval_array(state_interval)),
                source,
            ),
            ProofLedgerEntry(
                "spatial_ordinary_ks_entry_event_isolation",
                entry_event_certificate.certified,
                source,
            ),
            ProofLedgerEntry(
                "spatial_ordinary_to_ks_branch_lift",
                bool(entry_ks_state.certified and entry_projection.certified),
                source,
            ),
            ProofLedgerEntry(
                "spatial_ks_equation_residuals",
                charts[1].residual_certified,
                source,
            ),
            ProofLedgerEntry(
                "spatial_ks_projection_constraints",
                charts[1].projection_certified,
                source,
            ),
            ProofLedgerEntry(
                "spatial_ks_rho_positive_endpoint_projection",
                ks_atlas.evaluation.endpoint_projection.certified,
                source,
            ),
            *(
                (
                    ProofLedgerEntry(
                        "ordinary_post_handoff_residuals",
                        charts[2].residual_certified,
                        source,
                    ),
                    ProofLedgerEntry(
                        "ordinary_handoff_admissibility",
                        bool(
                            ks_atlas.evaluation.ordinary_handoff_admissibility is not None
                            and ks_atlas.evaluation.ordinary_handoff_admissibility.certified
                        ),
                        source,
                    ),
                )
                if has_ordinary_post_handoff
                else (
                    ProofLedgerEntry(
                        "spatial_ks_target_inside_regularized_chart",
                        True,
                        source,
                        detail="requested target is evaluated before ordinary handoff",
                    ),
                )
            ),
            ProofLedgerEntry(
                "spatial_ordinary_ks_and_post_invariants",
                invariants.certified,
                source,
            ),
            ProofLedgerEntry(
                "local_tail_budget",
                tail_budget.certified and tail_budget.finite,
                source,
            ),
            ProofLedgerEntry(
                "ordinary_to_ks_transition",
                transitions[0].certified,
                source,
            ),
            *(
                (
                    ProofLedgerEntry(
                        "rho_positive_handoff_transition",
                        transitions[1].certified,
                        source,
                    ),
                )
                if has_ordinary_post_handoff
                else ()
            ),
            ProofLedgerEntry(
                "spatial_ks_exit_event_isolation",
                bool(exit_event_certificate is not None and exit_event_certificate.certified),
                source,
                required=bool(exit_event_certificate is not None),
                detail=(
                    "rho-exit event certified"
                    if exit_event_certificate is not None and exit_event_certificate.certified
                    else (
                        "ordinary handoff is certified from a chosen rho-positive endpoint; "
                        "no rho-exit root isolation is claimed"
                    )
                ),
            ),
            ProofLedgerEntry(
                "finite_time_physical_targeting",
                target_time is not None,
                source,
                detail=(
                    (
                        "requested physical target time is enclosed by the post-exit ordinary chart"
                        if has_ordinary_post_handoff
                        else "requested physical target time is enclosed by the KS chart"
                    )
                    if target_time is not None
                    else "local handoff has not been selected by a requested physical target-time bracket"
                ),
            ),
            ProofLedgerEntry(
                "target_time_domain",
                _target_time_in_final_chart_domain(
                    0.5 * (target_time_interval.lower + target_time_interval.upper),
                    charts,
                ),
                source,
            ),
            ProofLedgerEntry(
                "spatial_collision_policy_scope",
                collision_policy.certified,
                source,
                detail=(
                    "local ordinary/KS/ordinary domains keep every non-regularized "
                    "pair distance strictly positive"
                    if collision_policy.certified
                    else "; ".join(collision_policy_certificate.missing_obligations)
                ),
            ),
        )
    )
    return ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in masses.reshape(-1)),
        target_time=0.5 * (target_time_interval.lower + target_time_interval.upper),
        initial_state_interval=_flat_float_interval_array(state_interval),
        charts=charts,
        transitions=transitions,
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=proof_ledger,
        evaluation=evaluation,
    )


def validated_atlas_from_spatial_initial_ks_handoff(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    *,
    pair: tuple[int, int],
    branch: str,
    s_endpoint: float | None = None,
    exit_rho: float | None = None,
    s_upper: float | None = None,
    target_time: float | FloatInterval | tuple[float, float],
    retained_order: int,
    guard_order: int,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    competing_pair_min_distance_required: float = 0.0,
    source: str = "spatial_initial_ks_local_handoff",
) -> ValidatedAtlasSolution:
    """Start finite-time validation inside a spatial KS binary chart."""

    retained_order = int(retained_order)
    guard_order = int(guard_order)
    competing_pair_min_distance_required = float(competing_pair_min_distance_required)
    if retained_order < 1:
        raise ValueError("retained_order must be positive")
    if guard_order < 1:
        raise ValueError("guard_order must be positive")
    target_time_interval = (
        target_time
        if isinstance(target_time, FloatInterval)
        else (
            _coerce_float_interval(target_time)
            if isinstance(target_time, tuple)
            else FloatInterval.point(float(target_time))
        )
    )
    if target_time_interval.upper <= 0.0:
        raise ValueError("initial spatial KS handoff currently advances positive target times")
    if target_time_interval.lower < 0.0:
        target_time_interval = FloatInterval(0.0, target_time_interval.upper)

    masses = np.asarray(masses, dtype=float)
    pair = tuple(pair)
    initial_ks_state = spatial_interval_to_ks_binary_chart_state(
        state_interval,
        masses,
        pair=pair,
        branch=branch,
    )
    if not initial_ks_state.certified:
        raise ValueError("initial spatial state did not lift to a certified KS branch")

    ks_atlas = validated_atlas_from_spatial_ks_binary_chart(
        initial_ks_state,
        s_endpoint=None if s_endpoint is None else float(s_endpoint),
        exit_rho=None if exit_rho is None else float(exit_rho),
        s_upper=None if s_upper is None else float(s_upper),
        target_time_after_ks_start_interval=target_time_interval,
        retained_order=retained_order,
        guard_order=guard_order,
        ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
        ordinary_handoff_max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
        ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
        ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
        ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
        source=source,
    )
    has_ordinary_post_handoff = bool(
        len(ks_atlas.charts) > 1 and ks_atlas.evaluation.ordinary_solution is not None
    )
    collision_policy_certificate = _certify_spatial_local_collision_policy(
        pair=pair,
        masses=masses,
        ordinary_entry_solution=None,
        ordinary_entry_domain=None,
        ordinary_entry_tail_bound=0.0,
        ks_solution=ks_atlas.evaluation.ks_solution,
        ks_parameter_interval=ks_atlas.charts[0].parameter_interval,
        ks_tail_bound=ks_atlas.charts[0].tail_bound,
        ordinary_post_solution=ks_atlas.evaluation.ordinary_solution,
        ordinary_post_parameter_interval=(
            ks_atlas.charts[1].parameter_interval if has_ordinary_post_handoff else None
        ),
        ordinary_post_tail_bound=(
            ks_atlas.charts[1].tail_bound if has_ordinary_post_handoff else 0.0
        ),
        retained_order=retained_order,
        require_ordinary_entry=False,
        competing_pair_min_distance_required=competing_pair_min_distance_required,
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="local_spatial_initial_ks_binary_handoff",
        binary_policy="spatial_ks_initial_close_binary_local_regularization",
        total_collision_policy=(
            "selected_binary_only_local_third_body_separated"
            if collision_policy_certificate.certified
            else "not_classified_by_initial_ks_handoff"
        ),
        triple_collision_status=(
            "locally_excluded_on_initial_ks_charts"
            if collision_policy_certificate.certified
            else "not_classified_by_initial_ks_handoff"
        ),
        triple_collision_reason=(
            "initial KS chart keeps the third body separated from the selected "
            "binary and every ordinary post-handoff chart avoids pair collision"
            if collision_policy_certificate.certified
            else "initial KS handoff has not certified competing-event separation "
            "on every chart domain"
        ),
        certified=collision_policy_certificate.certified,
    )
    proof_entries = tuple(
        entry
        for entry in ks_atlas.proof_ledger.entries
        if entry.name
        not in {
            "spatial_ks_entry_event_isolation",
            "spatial_collision_policy_scope",
        }
    )
    if not any(entry.name == "spatial_ks_and_ordinary_invariants" for entry in proof_entries):
        proof_entries = (
            *proof_entries,
            ProofLedgerEntry(
                "spatial_ks_and_ordinary_invariants",
                ks_atlas.invariants.certified,
                source,
            ),
        )
    if (
        not has_ordinary_post_handoff
        and not any(
            entry.name == "spatial_ks_target_inside_regularized_chart"
            for entry in proof_entries
        )
    ):
        proof_entries = (
            *proof_entries,
            ProofLedgerEntry(
                "spatial_ks_target_inside_regularized_chart",
                True,
                source,
                detail="requested target is evaluated before ordinary handoff",
            ),
        )
    proof_ledger = ProofLedger(
        entries=(
            *proof_entries,
            ProofLedgerEntry(
                "spatial_initial_ks_branch_lift",
                initial_ks_state.certified,
                source,
            ),
            ProofLedgerEntry(
                "spatial_collision_policy_scope",
                collision_policy.certified,
                source,
                detail=(
                    "initial KS and ordinary post-handoff domains keep every "
                    "non-regularized pair distance strictly positive"
                    if collision_policy.certified
                    else "; ".join(collision_policy_certificate.missing_obligations)
                ),
            ),
        )
    )
    return replace(
        ks_atlas,
        initial_state_interval=_flat_float_interval_array(state_interval),
        collision_policy=collision_policy,
        proof_ledger=proof_ledger,
    )


def validated_atlas_from_spatial_close_pair_branch_partition(
    partition: SimultaneousClosePairSplitCertificate,
    masses: Array,
    *,
    target_time: float | FloatInterval | tuple[float, float],
    s_endpoint: float,
    retained_order: int,
    guard_order: int,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    competing_pair_min_distance_required: float = 0.0,
    member_atlas_builder: object | None = None,
    source: str = "spatial_close_pair_branch_union",
) -> ValidatedAtlasSolution:
    """Consume certified spatial close-pair split leaves with KS branch charts.

    Each partition leaf is continued independently in the selected spatial KS
    chart until the requested target time, then the physical target boxes are
    exposed as a branch union.  This is still a finite-time constructor; it does
    not assert arbitrary repeat/split exhaustion beyond the supplied partition.
    """

    masses = np.asarray(masses, dtype=float)
    target_time_interval = (
        target_time
        if isinstance(target_time, FloatInterval)
        else (
            _coerce_float_interval(target_time)
            if isinstance(target_time, tuple)
            else FloatInterval.point(float(target_time))
        )
    )
    target_time_midpoint = 0.5 * (
        target_time_interval.lower + target_time_interval.upper
    )
    s_endpoint = float(s_endpoint)
    retained_order = int(retained_order)
    guard_order = int(guard_order)
    competing_pair_min_distance_required = float(competing_pair_min_distance_required)
    if not getattr(partition, "certified", False):
        raise ValueError("spatial close-pair branch partition is not certified")
    if target_time_interval.upper <= 0.0:
        raise ValueError("spatial close-pair branch union currently advances positive target times")
    if target_time_interval.lower < 0.0:
        target_time_interval = FloatInterval(0.0, target_time_interval.upper)
        target_time_midpoint = 0.5 * (
            target_time_interval.lower + target_time_interval.upper
        )
    if s_endpoint <= 0.0:
        raise ValueError("s_endpoint must be positive")
    if retained_order < 1:
        raise ValueError("retained_order must be positive")
    if guard_order < 1:
        raise ValueError("guard_order must be positive")

    member_atlases: list[ValidatedAtlasSolution] = []
    initial_union: list[Array] = []
    missing: list[str] = []
    for branch in partition.branches:
        if not branch.certified:
            missing.append(f"{branch.branch_id}:branch_not_certified")
            continue
        if branch.selected_pair is None:
            missing.append(f"{branch.branch_id}:selected_pair_missing")
            continue
        initial_union.append(_flat_float_interval_array(branch.state_interval))
        try:
            branch_states = spatial_interval_to_ks_binary_chart_state_atlas(
                branch.state_interval,
                masses,
                pair=branch.selected_pair,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            missing.append(f"{branch.branch_id}:spatial_ks_branch_lift:{exc}")
            continue
        labels = tuple(
            dict.fromkeys(
                str(getattr(getattr(state, "branch_certificate", None), "branch", ""))
                for state in branch_states
                if getattr(state, "certified", False)
            )
        )
        labels = tuple(label for label in labels if label)
        if not labels:
            missing.append(f"{branch.branch_id}:spatial_ks_branch_lift")
            continue
        branch_certified = False
        branch_errors: list[str] = []
        for label in labels:
            if member_atlas_builder is not None:
                try:
                    atlas = member_atlas_builder(
                        branch=branch,
                        branch_label=label,
                        target_time_interval=target_time_interval,
                        s_endpoint=s_endpoint,
                        retained_order=retained_order,
                        guard_order=guard_order,
                        source=f"{source}:{branch.branch_id}:{label}:member_builder",
                    )
                except (RuntimeError, ValueError, TypeError) as exc:
                    atlas = None
                    branch_errors.append(f"{label}:member_builder:{exc}")
                if atlas is not None:
                    if atlas.proof_certified:
                        member_atlases.append(atlas)
                        branch_certified = True
                        break
                    branch_errors.append(
                        f"{label}:member_builder:"
                        + ",".join(_atlas_certification_failure_obligations(atlas))
                    )
            try:
                atlas = validated_atlas_from_spatial_initial_ks_handoff(
                    branch.state_interval,
                    masses,
                    pair=branch.selected_pair,
                    branch=label,
                    s_endpoint=s_endpoint,
                    target_time=target_time_interval,
                    retained_order=retained_order,
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
                    competing_pair_min_distance_required=(
                        competing_pair_min_distance_required
                    ),
                    source=f"{source}:{branch.branch_id}:{label}",
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                branch_errors.append(f"{label}:{exc}")
                continue
            if atlas.proof_certified:
                member_atlases.append(atlas)
                branch_certified = True
                break
            branch_errors.append(
                f"{label}:"
                + ",".join(_atlas_certification_failure_obligations(atlas))
            )
        if not branch_certified:
            missing.append(
                f"{branch.branch_id}:spatial_ks_branch_target:"
                + "|".join(branch_errors)
            )

    if missing:
        raise ValueError(
            "spatial close-pair branch union did not certify: "
            + "; ".join(missing)
        )
    if not member_atlases or not initial_union:
        raise ValueError("spatial close-pair branch union produced no member atlases")

    target_union = tuple(
        np.asarray(atlas.target_state_interval, dtype=object).reshape(-1)
        for atlas in member_atlases
    )
    target_hull = _interval_array_union_hull(target_union)
    initial_union_tuple = tuple(initial_union)
    initial_hull = _flat_float_interval_array(partition.original_state_interval)
    tail_bound = float(
        max(float(atlas.tail_budget.max_step_tail_bound) for atlas in member_atlases)
    )
    chart = ValidatedChart(
        chart_id="spatial_branch_union_0",
        chart_type="spatial_branch_union",
        source=source,
        parameter_name="t",
        parameter_interval=FloatInterval(0.0, target_time_interval.upper),
        physical_time_interval=FloatInterval(0.0, target_time_interval.upper),
        dynamics_certified=all(atlas.proof_certified for atlas in member_atlases),
        residual_certified=all(atlas.residual_budget.certified for atlas in member_atlases),
        projection_certified=all(
            any(
                entry.certified
                and entry.name
                in {
                    "spatial_ks_target_time_projection",
                    "spatial_ks_rho_positive_endpoint_projection",
                    "projection_ledger",
                }
                for entry in atlas.proof_ledger.entries
            )
            for atlas in member_atlases
        ),
        invariants_certified=all(atlas.invariants.certified for atlas in member_atlases),
        tail_certified=all(atlas.tail_budget.certified for atlas in member_atlases),
        tail_bound=tail_bound,
    )
    invariants = GlobalInvariantLedger(
        center_of_mass_certified=all(
            atlas.invariants.center_of_mass_certified for atlas in member_atlases
        ),
        linear_momentum_certified=all(
            atlas.invariants.linear_momentum_certified for atlas in member_atlases
        ),
        angular_momentum_certified=all(
            atlas.invariants.angular_momentum_certified for atlas in member_atlases
        ),
        energy_certified=all(atlas.invariants.energy_certified for atlas in member_atlases),
        certified_chart_count=int(chart.invariants_certified),
        expected_chart_count=1,
    )
    residual_budget = NewtonResidualLedger(
        certified=all(atlas.residual_budget.certified for atlas in member_atlases),
        certified_chart_count=int(all(atlas.residual_budget.certified for atlas in member_atlases)),
        expected_chart_count=1,
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=float(
            max(float(atlas.tail_budget.local_tail_bound) for atlas in member_atlases)
        ),
        max_step_tail_bound=tail_bound,
        certified=all(atlas.tail_budget.certified for atlas in member_atlases),
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="finite_time_spatial_close_pair_branch_union",
        binary_policy="spatial_ks_branch_union_selected_binary_regularization",
        total_collision_policy="finite_time_branch_union_excludes_unselected_pair_collisions",
        triple_collision_status="locally_excluded_on_branch_union",
        triple_collision_reason=(
            "certified close-pair partition leaves have at most one possible "
            "close binary; each selected branch is continued in a certified KS chart"
        ),
        certified=all(atlas.collision_policy.certified for atlas in member_atlases),
    )
    evaluation = SpatialBranchUnionValidatedEvaluation(
        branch_partition=partition,
        branch_atlases=tuple(member_atlases),
        initial_state_interval=initial_hull,
        initial_state_interval_union=initial_union_tuple,
        target_state_interval=target_hull,
        target_state_interval_union=target_union,
        initial_state=_interval_midpoint_state(initial_union_tuple[0]),
        final_state=_interval_midpoint_state(target_union[0]),
        target_time_interval=target_time_interval,
    )
    proof_ledger = ProofLedger(
        entries=(
            ProofLedgerEntry("mass_domain", _masses_positive_finite(masses), source),
            ProofLedgerEntry(
                "initial_state_domain",
                _interval_array_finite_nonempty(initial_hull)
                and _interval_array_union_finite_nonempty(initial_union_tuple)
                and _interval_array_union_subsets(initial_union_tuple, initial_hull),
                source,
            ),
            ProofLedgerEntry(
                    "target_time_domain",
                    _target_time_in_final_chart_domain(target_time_midpoint, (chart,)),
                    source,
                ),
                ProofLedgerEntry(
                    "finite_time_physical_targeting",
                    bool(
                        _float_interval_finite_nonempty(target_time_interval)
                        and target_time_interval.lower >= -1.0e-14
                        and target_time_interval.upper
                        <= chart.physical_time_interval.upper + 1.0e-14
                    ),
                    source,
                ),
            ProofLedgerEntry(
                "projection_ledger",
                chart.projection_certified
                and _interval_array_union_subsets(target_union, target_hull),
                source,
            ),
            ProofLedgerEntry("newton_residuals", residual_budget.certified, source),
            ProofLedgerEntry("invariant_ledger", invariants.certified, source),
            ProofLedgerEntry(
                "local_tail_budget",
                tail_budget.certified and tail_budget.finite,
                source,
            ),
            ProofLedgerEntry("collision_policy", collision_policy.certified, source),
            ProofLedgerEntry(
                "simultaneous_close_pair_partition",
                partition.certified,
                source,
            ),
            ProofLedgerEntry(
                "finite_time_branch_union_consumption",
                bool(
                    partition.certified
                    and member_atlases
                    and len(member_atlases) >= len(partition.branches)
                    and all(atlas.proof_certified for atlas in member_atlases)
                    and _interval_array_union_finite_nonempty(target_union)
                    and _interval_array_union_subsets(target_union, target_hull)
                ),
                source,
                detail=(
                    f"consumed {len(member_atlases)} certified KS member atlases "
                    f"from {len(partition.branches)} close-pair partition leaves"
                ),
            ),
        )
    )
    return ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in masses.reshape(-1)),
        target_time=target_time_midpoint,
        initial_state_interval=initial_hull,
        charts=(chart,),
        transitions=(),
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=proof_ledger,
        evaluation=evaluation,
        initial_state_interval_union=initial_union_tuple,
    )


def validated_atlas_from_spatial_ks_event_order_partition(
    partition: KSEventOrderSplitCertificate,
    *,
    retained_order: int,
    guard_order: int,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    competing_pair_min_distance_required: float = 0.0,
    max_competing_repeats: int = 0,
    source: str = "spatial_ks_event_order_branch_union",
) -> ValidatedAtlasSolution:
    """Consume certified spatial-KS event-order split leaves as a branch union."""

    retained_order = int(retained_order)
    guard_order = int(guard_order)
    max_competing_repeats = int(max_competing_repeats)
    if not getattr(partition, "certified", False):
        raise ValueError("spatial KS event-order partition is not certified")
    if retained_order < 1:
        raise ValueError("retained_order must be positive")
    if guard_order < 1:
        raise ValueError("guard_order must be positive")
    if max_competing_repeats < 0:
        raise ValueError("max_competing_repeats cannot be negative")

    target_time_interval = partition.target_time_interval
    target_time = 0.5 * (target_time_interval.lower + target_time_interval.upper)
    member_atlases: list[ValidatedAtlasSolution] = []
    initial_union: list[Array] = []
    missing: list[str] = []
    for leaf in partition.leaves:
        if not leaf.certified:
            missing.append(f"{leaf.branch_id}:event_order_leaf_not_certified")
            continue
        projection = project_spatial_ks_binary_interval_chart_state_to_physical(
            leaf.ks_state,
            physical_time=FloatInterval.point(0.0),
        )
        if not projection.certified:
            missing.append(f"{leaf.branch_id}:initial_ks_projection")
            continue
        initial_union.append(_flat_float_interval_array(projection.state_interval))

        atlas_errors: list[str] = []
        if leaf.decision == "target_before_all_events":
            try:
                atlas = validated_atlas_from_spatial_ks_binary_chart(
                    leaf.ks_state,
                    target_time_after_ks_start_interval=target_time_interval,
                    retained_order=retained_order,
                    guard_order=guard_order,
                    ordinary_handoff_min_pair_distance_required=(
                        ordinary_handoff_min_pair_distance_required
                    ),
                    ordinary_handoff_max_acceleration_bound=(
                        ordinary_handoff_max_acceleration_bound
                    ),
                    ordinary_handoff_min_cauchy_radius=(
                        ordinary_handoff_min_cauchy_radius
                    ),
                    ordinary_handoff_max_residual_bound=(
                        ordinary_handoff_max_residual_bound
                    ),
                    ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
                    source=f"{source}:{leaf.branch_id}:target",
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                atlas = None
                atlas_errors.append(f"target:{exc}")
            if atlas is not None and atlas.proof_certified:
                member_atlases.append(atlas)
                continue
            if atlas is not None:
                atlas_errors.append(
                    "target:" + ",".join(_atlas_certification_failure_obligations(atlas))
                )
        elif leaf.decision == "unique_first_event" and leaf.first_event_pair is not None:
            expected_event_id = (
                "spatial_ks_competing_binary_entry:"
                f"{leaf.first_event_pair[0]}-{leaf.first_event_pair[1]}"
            )
            first_event = getattr(
                getattr(leaf, "event_set_certificate", None),
                "first_event",
                None,
            )
            if (
                leaf.first_event_id != expected_event_id
                or leaf.first_event_time_interval is None
                or first_event is None
                or getattr(first_event, "certificate", None) is None
            ):
                missing.append(f"{leaf.branch_id}:event_order_leaf_event_certificate")
                continue
            candidates = _derive_spatial_ks_competing_handoff_candidates_from_ks_state(
                leaf.ks_state,
                target_time_after_ks_start=target_time_interval.upper,
                binary_distance_threshold=partition.competing_enter_distance,
                s_upper=partition.competing_s_upper,
                retained_order=retained_order,
                guard_order=guard_order,
            )
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate[1] == leaf.first_event_pair
                and _float_in_interval_with_tolerance(
                    candidate[0],
                    leaf.first_event_time_interval,
                )
            )
            if not candidates:
                atlas_errors.append("branch_lift")
            for (
                _entry_time_estimate,
                competing_pair,
                label,
                enter_distance,
                entry_s_upper,
                next_s_endpoint,
            ) in candidates:
                try:
                    atlas = validated_atlas_from_spatial_ks_competing_binary_handoff(
                        leaf.ks_state,
                        competing_pair=competing_pair,
                        enter_distance=enter_distance,
                        entry_s_upper=entry_s_upper,
                        branch=label,
                        next_s_endpoint=next_s_endpoint,
                        target_time_after_ks_start_interval=target_time_interval,
                        retained_order=retained_order,
                        guard_order=guard_order,
                        ordinary_handoff_min_pair_distance_required=(
                            ordinary_handoff_min_pair_distance_required
                        ),
                        ordinary_handoff_max_acceleration_bound=(
                            ordinary_handoff_max_acceleration_bound
                        ),
                        ordinary_handoff_min_cauchy_radius=(
                            ordinary_handoff_min_cauchy_radius
                        ),
                        ordinary_handoff_max_residual_bound=(
                            ordinary_handoff_max_residual_bound
                        ),
                        ordinary_handoff_max_tail_bound=(
                            ordinary_handoff_max_tail_bound
                        ),
                        competing_pair_min_distance_required=(
                            competing_pair_min_distance_required
                        ),
                        max_competing_repeats=max_competing_repeats,
                        source=f"{source}:{leaf.branch_id}:{label}",
                    )
                except (RuntimeError, ValueError, TypeError) as exc:
                    atlas_errors.append(f"{label}:{exc}")
                    continue
                if atlas.proof_certified:
                    member_atlases.append(atlas)
                    break
                atlas_errors.append(
                    f"{label}:"
                    + ",".join(_atlas_certification_failure_obligations(atlas))
                )
            else:
                pass
            if len(member_atlases) >= len(initial_union):
                continue
        else:
            atlas_errors.append(f"unsupported_decision:{leaf.decision}")

        missing.append(f"{leaf.branch_id}:event_order_branch_target:" + "|".join(atlas_errors))

    if missing:
        raise ValueError(
            "spatial KS event-order branch union did not certify: "
            + "; ".join(missing)
        )
    if not member_atlases or len(member_atlases) != len(partition.leaves):
        raise ValueError("spatial KS event-order branch union produced incomplete members")

    target_union = tuple(
        np.asarray(atlas.target_state_interval, dtype=object).reshape(-1)
        for atlas in member_atlases
    )
    target_hull = _interval_array_union_hull(target_union)
    initial_union_tuple = tuple(initial_union)
    initial_hull = _interval_array_union_hull(initial_union_tuple)
    tail_bound = float(
        max(float(atlas.tail_budget.max_step_tail_bound) for atlas in member_atlases)
    )
    chart = ValidatedChart(
        chart_id="spatial_ks_event_order_branch_union_0",
        chart_type="spatial_ks_event_order_branch_union",
        source=source,
        parameter_name="t",
        parameter_interval=FloatInterval(0.0, target_time_interval.upper),
        physical_time_interval=FloatInterval(0.0, target_time_interval.upper),
        dynamics_certified=all(atlas.proof_certified for atlas in member_atlases),
        residual_certified=all(atlas.residual_budget.certified for atlas in member_atlases),
        projection_certified=all(
            _interval_array_finite_nonempty(atlas.target_state_interval)
            for atlas in member_atlases
        ),
        invariants_certified=all(atlas.invariants.certified for atlas in member_atlases),
        tail_certified=all(atlas.tail_budget.certified for atlas in member_atlases),
        tail_bound=tail_bound,
    )
    invariants = GlobalInvariantLedger(
        center_of_mass_certified=all(
            atlas.invariants.center_of_mass_certified for atlas in member_atlases
        ),
        linear_momentum_certified=all(
            atlas.invariants.linear_momentum_certified for atlas in member_atlases
        ),
        angular_momentum_certified=all(
            atlas.invariants.angular_momentum_certified for atlas in member_atlases
        ),
        energy_certified=all(atlas.invariants.energy_certified for atlas in member_atlases),
        certified_chart_count=int(chart.invariants_certified),
        expected_chart_count=1,
    )
    residual_budget = NewtonResidualLedger(
        certified=all(atlas.residual_budget.certified for atlas in member_atlases),
        certified_chart_count=int(all(atlas.residual_budget.certified for atlas in member_atlases)),
        expected_chart_count=1,
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=float(
            max(float(atlas.tail_budget.local_tail_bound) for atlas in member_atlases)
        ),
        max_step_tail_bound=tail_bound,
        certified=all(atlas.tail_budget.certified for atlas in member_atlases),
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="finite_time_spatial_ks_event_order_branch_union",
        binary_policy="spatial_ks_event_order_partition_member_regularization",
        total_collision_policy="finite_time_event_order_branch_union_excludes_dropped_branches",
        triple_collision_status="locally_excluded_on_event_order_branch_union",
        triple_collision_reason=(
            "certified KS event-order partition leaves are each continued by a "
            "constructor-certified member atlas without hulling across first-event ambiguity"
        ),
        certified=all(atlas.collision_policy.certified for atlas in member_atlases),
    )
    evaluation = SpatialBranchUnionValidatedEvaluation(
        branch_partition=partition,
        branch_atlases=tuple(member_atlases),
        initial_state_interval=initial_hull,
        initial_state_interval_union=initial_union_tuple,
        target_state_interval=target_hull,
        target_state_interval_union=target_union,
        initial_state=_interval_midpoint_state(initial_union_tuple[0]),
        final_state=_interval_midpoint_state(target_union[0]),
        target_time_interval=target_time_interval,
    )
    proof_ledger = ProofLedger(
        entries=(
            ProofLedgerEntry("mass_domain", _masses_positive_finite(partition.leaves[0].ks_state.masses), source),
            ProofLedgerEntry(
                "initial_state_domain",
                _interval_array_finite_nonempty(initial_hull)
                and _interval_array_union_finite_nonempty(initial_union_tuple)
                and _interval_array_union_subsets(initial_union_tuple, initial_hull),
                source,
            ),
            ProofLedgerEntry(
                "target_time_domain",
                _target_time_in_final_chart_domain(target_time, (chart,)),
                source,
            ),
            ProofLedgerEntry(
                "finite_time_physical_targeting",
                _target_time_in_final_chart_domain(target_time, (chart,)),
                source,
            ),
            ProofLedgerEntry(
                "projection_ledger",
                chart.projection_certified
                and _interval_array_union_subsets(target_union, target_hull),
                source,
            ),
            ProofLedgerEntry("newton_residuals", residual_budget.certified, source),
            ProofLedgerEntry("invariant_ledger", invariants.certified, source),
            ProofLedgerEntry(
                "local_tail_budget",
                tail_budget.certified and tail_budget.finite,
                source,
            ),
            ProofLedgerEntry("collision_policy", collision_policy.certified, source),
            ProofLedgerEntry("ks_event_order_partition", partition.certified, source),
            ProofLedgerEntry(
                "finite_time_event_order_branch_union_consumption",
                bool(
                    partition.certified
                    and len(member_atlases) == len(partition.leaves)
                    and all(atlas.proof_certified for atlas in member_atlases)
                    and _interval_array_union_finite_nonempty(target_union)
                    and _interval_array_union_subsets(target_union, target_hull)
                ),
                source,
                detail=(
                    f"consumed {len(member_atlases)} certified member atlases "
                    f"from {len(partition.leaves)} KS event-order partition leaves"
                ),
            ),
        )
    )
    masses = np.asarray(partition.leaves[0].ks_state.masses, dtype=float).reshape(-1)
    return ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in masses),
        target_time=target_time,
        initial_state_interval=initial_hull,
        charts=(chart,),
        transitions=(),
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=proof_ledger,
        evaluation=evaluation,
        initial_state_interval_union=initial_union_tuple,
    )


def _certify_spatial_ks_target_tail(
    initial_state: IntervalSpatialKSBinaryChartState,
    *,
    ks_solution: object,
    target_s: float,
    target_time_after_ks_start_interval: FloatInterval,
    s_search_upper: float | None = None,
    retained_order: int,
    guard_order: int,
) -> SpatialKSTargetTailEvaluation:
    target_s = float(target_s)
    s_search_upper = float(target_s if s_search_upper is None else s_search_upper)
    if s_search_upper < target_s:
        s_search_upper = target_s
    direct_projection = project_spatial_ks_binary_taylor_endpoint_to_physical(
        ks_solution,
        target_s,
    )
    if not direct_projection.certified:
        raise ValueError("KS target-time projection must certify rho.lower > 0")
    direct_tail = spatial_ks_binary_interval_tail_certificate(
        initial_state,
        retained_order=retained_order,
        guard_order=guard_order,
        step_size=target_s,
    )
    direct_physical_time = _inflate_interval(
        direct_projection.physical_time,
        direct_tail.tail_bound,
    )
    target_time_is_point = (
        target_time_after_ks_start_interval.lower
        == target_time_after_ks_start_interval.upper
    )
    target_time_containment_failed = False
    target_time_bracketing_failed = False
    if (
        target_time_is_point
        and direct_tail.is_nontrivial
        and _float_interval_contains_interval_with_tolerance(
            direct_physical_time,
            target_time_after_ks_start_interval,
            tolerance=1.0e-14,
        )
    ):
        direct_projection = replace(
            direct_projection,
            physical_time=direct_physical_time,
        )
        return SpatialKSTargetTailEvaluation(
            endpoint_projection=direct_projection,
            target_state_interval=_inflate_interval_array(
                _flat_float_interval_array(direct_projection.state_interval),
                direct_tail.tail_bound,
            ),
            tail_certificate=direct_tail,
            tail_bound=float(direct_tail.tail_bound),
            tail_certified=True,
            proof_detail=(
                "requested physical target is evaluated inside the regularized KS chart"
            ),
        )
    if direct_tail.is_nontrivial:
        target_time_containment_failed = True

    try:
        target_s_interval = _ks_physical_time_roots_for_target_interval(
            ks_solution,
            target_time_after_ks_start_interval,
            s_upper=s_search_upper,
        )
    except ValueError:
        target_s_interval = None
    if target_s_interval is not None:
        interval_width = float(target_s_interval.upper - target_s_interval.lower)
        base_padding = max(
            1.0e-14,
            64.0
            * np.finfo(float).eps
            * max(
                1.0,
                abs(target_s),
                abs(s_search_upper),
                abs(target_s_interval.lower),
                abs(target_s_interval.upper),
            ),
            0.5 * interval_width,
        )
        for padding_multiplier in (
            0.0,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
            32.0,
            64.0,
            128.0,
            256.0,
            512.0,
            1024.0,
            2048.0,
            4096.0,
            8192.0,
            16384.0,
            32768.0,
            65536.0,
            131072.0,
            262144.0,
            524288.0,
            1048576.0,
        ):
            candidate_s_interval = FloatInterval(
                max(
                    0.0,
                    target_s_interval.lower - padding_multiplier * base_padding,
                ),
                min(
                    s_search_upper,
                    target_s_interval.upper + padding_multiplier * base_padding,
                ),
            )
            interval_step = _interval_max_abs(candidate_s_interval)
            interval_tail = spatial_ks_binary_interval_tail_certificate(
                initial_state,
                retained_order=retained_order,
                guard_order=guard_order,
                step_size=interval_step,
            )
            if not interval_tail.is_nontrivial:
                continue
            interval_physical_time = _spatial_ks_physical_time_interval_from_solution(
                ks_solution,
                candidate_s_interval,
                tail_bound=interval_tail.tail_bound,
            )
            if not _float_interval_contains_interval_with_tolerance(
                interval_physical_time,
                target_time_after_ks_start_interval,
                tolerance=1.0e-14,
            ):
                target_time_containment_failed = True
                continue
            if not _spatial_ks_target_time_interval_bracket_certified(
                ks_solution,
                candidate_s_interval,
                target_time_after_ks_start_interval,
                tail_bound=interval_tail.tail_bound,
                tolerance=1.0e-14,
            ):
                target_time_bracketing_failed = True
                continue
            interval_state = _spatial_ks_interval_state_from_solution_interval(
                initial_state,
                ks_solution=ks_solution,
                parameter_interval=candidate_s_interval,
                tail_bound=interval_tail.tail_bound,
            )
            interval_projection = project_spatial_ks_binary_interval_chart_state_to_physical(
                interval_state,
                s_value=0.5
                * (candidate_s_interval.lower + candidate_s_interval.upper),
                physical_time=interval_physical_time,
            )
            if interval_projection.certified:
                return SpatialKSTargetTailEvaluation(
                    endpoint_projection=interval_projection,
                    target_state_interval=_flat_float_interval_array(
                        interval_projection.state_interval
                    ),
                    tail_certificate=interval_tail,
                    tail_bound=float(interval_tail.tail_bound),
                    tail_certified=True,
                    proof_detail=(
                        "requested physical target interval is evaluated inside "
                        "the regularized KS chart over an endpoint-bracketed "
                        "certified parameter interval"
                    ),
                )

    for segment_count in (2, 4, 8):
        segmented_tail = spatial_ks_binary_interval_segmented_tail_certificate(
            initial_state,
            retained_order=retained_order,
            guard_order=guard_order,
            step_size=target_s,
            segment_count=segment_count,
        )
        if not segmented_tail.is_nontrivial or segmented_tail.final_state is None:
            continue
        physical_time = segmented_tail.final_physical_time_delta
        if physical_time is None:
            continue
        if not target_time_is_point:
            target_time_bracketing_failed = True
            continue
        if not _float_interval_contains_interval_with_tolerance(
            physical_time,
            target_time_after_ks_start_interval,
            tolerance=1.0e-14,
        ):
            target_time_containment_failed = True
            continue
        segmented_projection = project_spatial_ks_binary_interval_chart_state_to_physical(
            segmented_tail.final_state,
            s_value=target_s,
            physical_time=physical_time,
        )
        if not segmented_projection.certified:
            continue
        return SpatialKSTargetTailEvaluation(
            endpoint_projection=segmented_projection,
            target_state_interval=_flat_float_interval_array(
                segmented_projection.state_interval
            ),
            tail_certificate=segmented_tail,
            tail_bound=float(segmented_tail.local_tail_bound),
            tail_certified=True,
            proof_detail=(
                "requested physical target is evaluated inside the regularized KS chart "
                f"with {segment_count} certified re-expanded KS tail segments"
            ),
        )

    detail = "KS target tail could not be certified by direct or segmented guards"
    if target_time_containment_failed:
        detail += ": spatial_ks_target_time_containment"
    if target_time_bracketing_failed:
        detail += ": spatial_ks_target_time_endpoint_bracketing"
    raise ValueError(detail)


def _spatial_ks_segment_state_from_solution(
    initial_state: IntervalSpatialKSBinaryChartState,
    *,
    ks_solution: object,
    step_size: float,
    tail_bound: float,
) -> IntervalSpatialKSBinaryChartState:
    evaluation_time = FloatInterval.point(float(step_size))
    return IntervalSpatialKSBinaryChartState(
        masses=initial_state.masses,
        pair=initial_state.pair,
        u=_inflate_interval_array(
            interval_array_series_eval(ks_solution.u, evaluation_time),
            tail_bound,
        ),
        u_velocity=_inflate_interval_array(
            interval_array_series_eval(ks_solution.u_velocity, evaluation_time),
            tail_bound,
        ),
        pair_energy=_inflate_interval(
            interval_array_series_eval(ks_solution.pair_energy[:, None], evaluation_time)[0],
            tail_bound,
        ),
        binary_center=_inflate_interval_array(
            interval_array_series_eval(ks_solution.binary_center, evaluation_time),
            tail_bound,
        ),
        binary_center_velocity=_inflate_interval_array(
            interval_array_series_eval(ks_solution.binary_center_velocity, evaluation_time),
            tail_bound,
        ),
        third_offset=_inflate_interval_array(
            interval_array_series_eval(ks_solution.third_offset, evaluation_time),
            tail_bound,
        ),
        third_offset_velocity=_inflate_interval_array(
            interval_array_series_eval(ks_solution.third_offset_velocity, evaluation_time),
            tail_bound,
        ),
        branch_certificate=None,
    )


def _certify_spatial_ks_segment_tail(
    initial_state: IntervalSpatialKSBinaryChartState,
    *,
    ks_solution: object,
    step_size: float,
    retained_order: int,
    guard_order: int,
) -> SpatialKSSegmentTailEvaluation:
    step_size = float(step_size)
    direct_tail = spatial_ks_binary_interval_tail_certificate(
        initial_state,
        retained_order=retained_order,
        guard_order=guard_order,
        step_size=step_size,
    )
    if direct_tail.is_nontrivial:
        physical_time_delta = _inflate_interval(
            interval_array_series_eval(
                ks_solution.physical_time[:, None],
                FloatInterval.point(step_size),
            )[0],
            direct_tail.tail_bound,
        )
        return SpatialKSSegmentTailEvaluation(
            tail_certificate=direct_tail,
            tail_bound=float(direct_tail.tail_bound),
            tail_certified=True,
            final_state=_spatial_ks_segment_state_from_solution(
                initial_state,
                ks_solution=ks_solution,
                step_size=step_size,
                tail_bound=direct_tail.tail_bound,
            ),
            physical_time_delta=physical_time_delta,
            proof_detail="KS segment tail certified by the direct interval guard",
        )

    for segment_count in (2, 4, 8):
        segmented_tail = spatial_ks_binary_interval_segmented_tail_certificate(
            initial_state,
            retained_order=retained_order,
            guard_order=guard_order,
            step_size=step_size,
            segment_count=segment_count,
        )
        if not segmented_tail.is_nontrivial:
            continue
        return SpatialKSSegmentTailEvaluation(
            tail_certificate=segmented_tail,
            tail_bound=float(segmented_tail.local_tail_bound),
            tail_certified=True,
            final_state=segmented_tail.final_state,
            physical_time_delta=segmented_tail.final_physical_time_delta,
            proof_detail=(
                f"KS segment tail certified with {segment_count} "
                "re-expanded interval KS subcharts"
            ),
        )

    return SpatialKSSegmentTailEvaluation(
        tail_certificate=direct_tail,
        tail_bound=float(direct_tail.tail_bound),
        tail_certified=False,
        final_state=None,
        physical_time_delta=None,
        proof_detail="KS segment tail was not certified by direct or segmented guards",
    )


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


def _finite_event_strictly_before(
    left: FloatInterval,
    right: FloatInterval,
    *,
    tolerance: float,
) -> bool:
    return bool(left.upper < right.lower - float(tolerance))


def _ks_event_physical_time_interval(
    ks_solution: object,
    parameter_interval: FloatInterval,
) -> FloatInterval:
    return _as_float_interval(
        interval_array_series_eval(
            np.asarray(ks_solution.physical_time, dtype=object)[:, None],
            parameter_interval,
        )[0]
    )


def certify_ambiguous_event_order_partition(
    *,
    target_time_interval: FloatInterval | tuple[float, float],
    ambiguous_events: tuple[FiniteTimeEventCandidate, ...],
    all_event_candidates: tuple[FiniteTimeEventCandidate, ...] = (),
    ordering_tolerance: float = 1.0e-14,
    state_partition_certified: bool = False,
) -> AmbiguousEventOrderSplitCertificate:
    """Derive event-order alternatives for overlapping first-event intervals."""

    target_time_interval = _coerce_float_interval(target_time_interval)
    ambiguous_events = tuple(ambiguous_events)
    all_event_candidates = tuple(all_event_candidates or ambiguous_events)
    missing: list[str] = []
    if len(ambiguous_events) < 2:
        missing.append("ambiguous_event_order_requires_multiple_events")
    ambiguous_ids = tuple(event.event_id for event in ambiguous_events)
    if len(set(ambiguous_ids)) != len(ambiguous_ids):
        missing.append("ambiguous_event_order_event_ids_unique")
    if not all(event.well_formed and event.certified for event in ambiguous_events):
        missing.append("ambiguous_event_order_events_certified")

    leaves: list[AmbiguousEventOrderPartitionLeaf] = []
    for event in ambiguous_events:
        competing_ids = tuple(
            other.event_id
            for other in ambiguous_events
            if other.event_id != event.event_id
            and not _finite_event_strictly_before(
                event.physical_time_interval,
                other.physical_time_interval,
                tolerance=ordering_tolerance,
            )
            and not _finite_event_strictly_before(
                other.physical_time_interval,
                event.physical_time_interval,
                tolerance=ordering_tolerance,
            )
        )
        leaf_missing = (
            ()
            if event.well_formed and event.certified
            else ("ambiguous_event_order_leaf_event_not_certified",)
        )
        leaves.append(
            AmbiguousEventOrderPartitionLeaf(
                leaf_id=f"event_order_leaf:{event.event_id}",
                assumed_first_event_id=event.event_id,
                event_type=event.event_type,
                physical_time_interval=event.physical_time_interval,
                parameter_interval=event.parameter_interval,
                pair=event.pair,
                competing_first_event_ids=competing_ids,
                source_event=event,
                certified=not leaf_missing,
                missing_obligations=leaf_missing,
            )
        )

    nonambiguous_candidates = tuple(
        event for event in all_event_candidates if event.event_id not in set(ambiguous_ids)
    )
    no_omitted_first_event = all(
        any(
            _finite_event_strictly_before(
                ambiguous_event.physical_time_interval,
                event.physical_time_interval,
                tolerance=ordering_tolerance,
            )
            for ambiguous_event in ambiguous_events
        )
        for event in nonambiguous_candidates
    )
    if not no_omitted_first_event:
        missing.append("ambiguous_event_order_omitted_possible_first_event")

    target_not_certified_first = not all(
        _finite_event_strictly_before(
            target_time_interval,
            event.physical_time_interval,
            tolerance=ordering_tolerance,
        )
        for event in ambiguous_events
    )
    if not target_not_certified_first:
        missing.append("ambiguous_event_order_target_precedes_all_events")

    if not state_partition_certified:
        missing.append("state_space_event_order_partition_not_constructed")

    event_alternative_cover_certified = bool(
        len(ambiguous_events) >= 2
        and len(set(ambiguous_ids)) == len(ambiguous_ids)
        and len(leaves) == len(ambiguous_events)
        and all(leaf.certified for leaf in leaves)
        and no_omitted_first_event
        and target_not_certified_first
    )
    return AmbiguousEventOrderSplitCertificate(
        target_time_interval=target_time_interval,
        ambiguous_events=ambiguous_events,
        branch_leaves=tuple(leaves),
        event_alternative_cover_certified=event_alternative_cover_certified,
        state_partition_certified=bool(state_partition_certified),
        missing_obligations=tuple(dict.fromkeys(missing)),
    )


def certify_next_finite_time_event_set(
    ks_solution: object,
    *,
    target_time_after_start_interval: FloatInterval | tuple[float, float],
    selected_pair: tuple[int, int] | None = None,
    competing_enter_distance: float | None = None,
    competing_s_upper: float | None = None,
    retained_order: int | None = None,
    competing_pairs: tuple[tuple[int, int], ...] | None = None,
    exit_rho: float | None = None,
    exit_s_upper: float | None = None,
    ordering_tolerance: float = 1.0e-14,
    require_all_event_certificates: bool = False,
) -> FiniteTimeEventSetCertificate:
    """Derive target/event ordering from certified KS root enclosures.

    Point roots may still seed the lower-level root isolators, but this selector
    only compares the resulting physical-time intervals.
    """

    target_time_interval = _coerce_float_interval(target_time_after_start_interval)
    retained_order = (
        int(retained_order)
        if retained_order is not None
        else int(getattr(ks_solution, "order", 0))
    )
    missing: list[str] = []
    if target_time_interval.lower < -ordering_tolerance:
        missing.append("finite_time_target_time_nonnegative")
    if retained_order < 1:
        missing.append("finite_time_event_retained_order_positive")

    target_candidate = FiniteTimeEventCandidate(
        event_id="target",
        event_type="target_time",
        physical_time_interval=target_time_interval,
        parameter_interval=None,
        pair=None,
        certificate=None,
        certified=target_time_interval.lower >= -ordering_tolerance,
        missing_obligations=(
            ()
            if target_time_interval.lower >= -ordering_tolerance
            else ("finite_time_target_time_nonnegative",)
        ),
    )
    event_candidates: list[FiniteTimeEventCandidate] = []

    if exit_rho is not None:
        if exit_s_upper is None:
            missing.append("spatial_ks_rho_exit_search_upper")
        else:
            try:
                exit_event = certify_spatial_ks_binary_rho_exit_event(
                    ks_solution,
                    exit_rho=float(exit_rho),
                    s_upper=float(exit_s_upper),
                    coefficient_count=retained_order,
                )
            except (RuntimeError, ValueError, TypeError):
                exit_event = None
            if (
                exit_event is not None
                and exit_event.certified
                and exit_event.root_interval is not None
            ):
                parameter_interval = FloatInterval(*exit_event.root_interval)
                event_candidates.append(
                    FiniteTimeEventCandidate(
                        event_id="spatial_ks_rho_exit",
                        event_type="spatial_ks_rho_exit",
                        physical_time_interval=_ks_event_physical_time_interval(
                            ks_solution,
                            parameter_interval,
                        ),
                        parameter_interval=parameter_interval,
                        pair=None,
                        certificate=exit_event,
                    )
                )
            elif require_all_event_certificates:
                missing.append("spatial_ks_rho_exit_event_not_certified")

    if competing_enter_distance is not None:
        if competing_s_upper is None:
            missing.append("spatial_ks_competing_entry_search_upper")
        else:
            selected_pair_set = set(tuple(selected_pair or ()))
            pairs = (
                tuple(tuple(pair) for pair in competing_pairs)
                if competing_pairs is not None
                else tuple(
                    pair
                    for pair in _three_body_pairs()
                    if set(pair) != selected_pair_set
                )
            )
            for pair in pairs:
                ordered_pair = _ordered_pair(*pair)
                try:
                    entry_event = certify_spatial_ks_competing_binary_entry_event(
                        ks_solution,
                        pair=ordered_pair,
                        enter_distance=float(competing_enter_distance),
                        s_upper=float(competing_s_upper),
                        coefficient_count=retained_order,
                    )
                except (RuntimeError, ValueError, TypeError):
                    entry_event = None
                if (
                    entry_event is not None
                    and entry_event.certified
                    and entry_event.root_interval is not None
                ):
                    parameter_interval = FloatInterval(*entry_event.root_interval)
                    event_candidates.append(
                        FiniteTimeEventCandidate(
                            event_id=(
                                "spatial_ks_competing_binary_entry:"
                                f"{ordered_pair[0]}-{ordered_pair[1]}"
                            ),
                            event_type="spatial_ks_competing_binary_entry",
                            physical_time_interval=_ks_event_physical_time_interval(
                                ks_solution,
                                parameter_interval,
                            ),
                            parameter_interval=parameter_interval,
                            pair=ordered_pair,
                            certificate=entry_event,
                        )
                    )
                elif require_all_event_certificates:
                    missing.append(
                        "spatial_ks_competing_entry_event_not_certified:"
                        f"{ordered_pair[0]}-{ordered_pair[1]}"
                    )

    target_before_all_events = bool(
        not missing
        and all(
            _finite_event_strictly_before(
                target_time_interval,
                candidate.physical_time_interval,
                tolerance=ordering_tolerance,
            )
            for candidate in event_candidates
        )
    )
    first_event = None
    unique_first_event_certified = False
    multiple_possible_first_events = False
    ambiguous_event_order_partition = None
    if not target_before_all_events:
        candidates_not_after_target = tuple(
            candidate
            for candidate in event_candidates
            if not _finite_event_strictly_before(
                target_time_interval,
                candidate.physical_time_interval,
                tolerance=ordering_tolerance,
            )
        )
        earliest = tuple(
            candidate
            for candidate in candidates_not_after_target
            if not any(
                other.event_id != candidate.event_id
                and _finite_event_strictly_before(
                    other.physical_time_interval,
                    candidate.physical_time_interval,
                    tolerance=ordering_tolerance,
                )
                for other in candidates_not_after_target
            )
        )
        if len(earliest) == 1:
            first_event = earliest[0]
            unique_first_event_certified = bool(
                not missing
                and _finite_event_strictly_before(
                    first_event.physical_time_interval,
                    target_time_interval,
                    tolerance=ordering_tolerance,
                )
            )
            if not unique_first_event_certified:
                missing.append(
                    "target_event_order_requires_split:"
                    f"{first_event.event_id}"
                )
        elif len(earliest) > 1:
            multiple_possible_first_events = True
            ambiguous_event_order_partition = certify_ambiguous_event_order_partition(
                target_time_interval=target_time_interval,
                ambiguous_events=earliest,
                all_event_candidates=tuple(event_candidates),
                ordering_tolerance=ordering_tolerance,
            )
            missing.append(
                "finite_time_event_order_requires_split:"
                + ",".join(candidate.event_id for candidate in earliest)
            )
            missing.extend(
                obligation
                for obligation in ambiguous_event_order_partition.missing_obligations
                if obligation != "state_space_event_order_partition_not_constructed"
            )
        elif event_candidates:
            missing.append("finite_time_event_order_not_certified")
    return FiniteTimeEventSetCertificate(
        target_time_interval=target_time_interval,
        target_candidate=target_candidate,
        event_candidates=tuple(event_candidates),
        first_event=first_event,
        target_before_all_events=target_before_all_events,
        unique_first_event_certified=unique_first_event_certified,
        multiple_possible_first_events_requiring_split=multiple_possible_first_events,
        missing_obligations=tuple(dict.fromkeys(missing)),
        ambiguous_event_order_partition=ambiguous_event_order_partition,
    )


def _ks_state_interval_tuple(
    state: IntervalSpatialKSBinaryChartState,
) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for field_name in (
        "u",
        "u_velocity",
        "binary_center",
        "binary_center_velocity",
        "third_offset",
        "third_offset_velocity",
    ):
        for interval in np.asarray(getattr(state, field_name), dtype=object).reshape(-1):
            intervals.append(_float_interval_bounds(interval))
    intervals.append(_float_interval_bounds(state.pair_energy))
    return tuple((float(lower), float(upper)) for lower, upper in intervals)


def _ks_state_from_interval_tuple(
    template: IntervalSpatialKSBinaryChartState,
    state_interval: tuple[tuple[float, float], ...],
) -> IntervalSpatialKSBinaryChartState:
    state_interval = _normalize_state_interval_tuple(state_interval, expected_length=21)
    index = 0

    def take(count: int, shape: tuple[int, ...]) -> Array:
        nonlocal index
        values = [
            FloatInterval(float(lower), float(upper))
            for lower, upper in state_interval[index : index + count]
        ]
        index += count
        return np.asarray(values, dtype=object).reshape(shape)

    u = take(4, (4,))
    u_velocity = take(4, (4,))
    binary_center = take(3, (3,))
    binary_center_velocity = take(3, (3,))
    third_offset = take(3, (3,))
    third_offset_velocity = take(3, (3,))
    lower, upper = state_interval[index]
    pair_energy = FloatInterval(float(lower), float(upper))
    return IntervalSpatialKSBinaryChartState(
        masses=np.asarray(template.masses, dtype=float),
        pair=tuple(template.pair),
        u=u,
        u_velocity=u_velocity,
        pair_energy=pair_energy,
        binary_center=binary_center,
        binary_center_velocity=binary_center_velocity,
        third_offset=third_offset,
        third_offset_velocity=third_offset_velocity,
        branch_certificate=template.branch_certificate,
    )


def _ks_event_order_split_axis(
    state_interval: tuple[tuple[float, float], ...],
    *,
    min_state_width: float,
) -> int | None:
    best_index = None
    best_width = -np.inf
    # Prefer geometric coordinates before velocities/energy: they control the
    # competing pair distances used by the event isolators.
    priority = (
        tuple(range(8, 17))
        + tuple(range(0, 8))
        + tuple(range(17, 20))
        + (20,)
    )
    for index in priority:
        lower, upper = state_interval[index]
        width = float(upper - lower)
        if width > best_width:
            best_width = width
            best_index = index
    if best_index is None or best_width <= float(min_state_width):
        return None
    return int(best_index)


def _ks_event_order_partition_path(branch_id: str) -> tuple[str, ...] | None:
    branch_id = str(branch_id)
    if branch_id == "root":
        return ()
    if not branch_id.startswith("root."):
        return None
    path = tuple(part for part in branch_id.split(".")[1:] if part)
    for token in path:
        if len(token) < 2 or token[0] not in {"L", "R"} or not token[1:].isdigit():
            return None
    return path


def _ks_event_order_bisection_paths_cover_root(
    branch_ids: tuple[str, ...],
) -> bool:
    paths: list[tuple[str, ...]] = []
    for branch_id in branch_ids:
        path = _ks_event_order_partition_path(branch_id)
        if path is None:
            return False
        paths.append(path)
    if not paths or len(set(paths)) != len(paths):
        return False

    def covers(paths_at_node: tuple[tuple[str, ...], ...]) -> bool:
        if not paths_at_node:
            return False
        if any(len(path) == 0 for path in paths_at_node):
            return len(paths_at_node) == 1
        first_tokens = {path[0] for path in paths_at_node}
        if len(first_tokens) != 2:
            return False
        directions = {token[0] for token in first_tokens}
        axes = {token[1:] for token in first_tokens}
        if directions != {"L", "R"} or len(axes) != 1:
            return False
        return all(
            covers(tuple(path[1:] for path in paths_at_node if path[0] == token))
            for token in first_tokens
        )

    return covers(tuple(paths))


def _ks_event_order_partition_leaf(
    state: IntervalSpatialKSBinaryChartState,
    *,
    branch_id: str,
    depth: int,
    target_time_interval: FloatInterval,
    selected_pair: tuple[int, int],
    competing_enter_distance: float,
    competing_s_upper: float,
    retained_order: int,
    computed_order: int,
    competing_pairs: tuple[tuple[int, int], ...] | None,
    ordering_tolerance: float,
) -> KSEventOrderPartitionLeaf:
    state_interval = _ks_state_interval_tuple(state)
    event_set = None
    missing: list[str] = []
    decision = "event_set_not_constructed"
    first_event_id = None
    first_event_pair = None
    first_event_time_interval = None
    try:
        ks_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
            state,
            order=computed_order,
        )
        event_set = certify_next_finite_time_event_set(
            ks_solution,
            target_time_after_start_interval=target_time_interval,
            selected_pair=selected_pair,
            competing_enter_distance=competing_enter_distance,
            competing_s_upper=competing_s_upper,
            retained_order=retained_order,
            competing_pairs=competing_pairs,
            ordering_tolerance=ordering_tolerance,
            require_all_event_certificates=True,
        )
    except (RuntimeError, ValueError, TypeError):
        missing.append("finite_time_event_set_not_constructed")

    if event_set is not None:
        first_event = getattr(event_set, "first_event", None)
        first_event_id = getattr(first_event, "event_id", None)
        first_event_pair = getattr(first_event, "pair", None)
        first_event_time_interval = getattr(first_event, "physical_time_interval", None)
        if getattr(event_set, "target_before_all_events", False):
            decision = "target_before_all_events"
        elif getattr(event_set, "unique_first_event_certified", False) and first_event is not None:
            decision = "unique_first_event"
        elif getattr(event_set, "multiple_possible_first_events_requiring_split", False):
            decision = "event_order_still_ambiguous"
            missing.append("ks_event_order_leaf_not_decided")
        else:
            decision = "event_order_not_certified"
            missing.append("ks_event_order_leaf_not_decided")
        if decision in {"event_order_still_ambiguous", "event_order_not_certified"}:
            missing.extend(
                obligation.split(":", 1)[0]
                for obligation in tuple(getattr(event_set, "missing_obligations", ()))
                if str(obligation)
            )
            ambiguous_partition = getattr(event_set, "ambiguous_event_order_partition", None)
            if ambiguous_partition is not None:
                missing.extend(
                    obligation.split(":", 1)[0]
                    for obligation in tuple(
                        getattr(ambiguous_partition, "missing_obligations", ())
                    )
                    if str(obligation)
                    and str(obligation) != "state_space_event_order_partition_not_constructed"
                )

    certified = bool(
        event_set is not None
        and (
            getattr(event_set, "target_before_all_events", False)
            or getattr(event_set, "unique_first_event_certified", False)
        )
        and not missing
    )
    return KSEventOrderPartitionLeaf(
        branch_id=branch_id,
        state_interval=state_interval,
        ks_state=state,
        event_set_certificate=event_set,
        decision=decision,
        depth=int(depth),
        first_event_id=first_event_id,
        first_event_pair=first_event_pair,
        first_event_time_interval=first_event_time_interval,
        certified=certified,
        missing_obligations=tuple(dict.fromkeys(missing)),
    )


def partition_ks_state_by_competing_event_order(
    initial_state: IntervalSpatialKSBinaryChartState,
    *,
    target_time_after_start_interval: FloatInterval | tuple[float, float],
    selected_pair: tuple[int, int] | None = None,
    competing_enter_distance: float,
    competing_s_upper: float,
    retained_order: int,
    guard_order: int = 0,
    competing_pairs: tuple[tuple[int, int], ...] | None = None,
    max_depth: int = 6,
    max_branches: int = 128,
    min_state_width: float = 1.0e-14,
    ordering_tolerance: float = 1.0e-14,
) -> KSEventOrderSplitCertificate:
    """Split a spatial-KS state box until each leaf has a first-event decision."""

    if not isinstance(initial_state, IntervalSpatialKSBinaryChartState):
        initial_state = interval_spatial_ks_binary_chart_state_from_point(initial_state)
    target_time_interval = _coerce_float_interval(target_time_after_start_interval)
    selected_pair = tuple(selected_pair or getattr(initial_state, "pair", ()))
    if selected_pair not in _three_body_pairs():
        raise ValueError("selected_pair must be one of the three body pairs")
    retained_order = int(retained_order)
    computed_order = retained_order + int(guard_order)
    max_depth = int(max_depth)
    max_branches = int(max_branches)
    min_state_width = float(min_state_width)
    competing_enter_distance = float(competing_enter_distance)
    competing_s_upper = float(competing_s_upper)
    if retained_order < 1 or computed_order < retained_order:
        raise ValueError("retained_order must be positive and guard_order nonnegative")
    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")
    if max_branches < 1:
        raise ValueError("max_branches must be positive")
    if min_state_width < 0.0:
        raise ValueError("min_state_width cannot be negative")
    if competing_enter_distance <= 0.0:
        raise ValueError("competing_enter_distance must be positive")
    if competing_s_upper <= 0.0:
        raise ValueError("competing_s_upper must be positive")

    original_state_interval = _ks_state_interval_tuple(initial_state)
    pending: list[
        tuple[tuple[tuple[float, float], ...], IntervalSpatialKSBinaryChartState, int, str]
    ] = [(original_state_interval, initial_state, 0, "root")]
    leaves: list[KSEventOrderPartitionLeaf] = []
    split_count = 0
    stopped_by_branch_limit = False
    while pending:
        state_interval, state, depth, branch_id = pending.pop(0)
        leaf = _ks_event_order_partition_leaf(
            state,
            branch_id=branch_id,
            depth=depth,
            target_time_interval=target_time_interval,
            selected_pair=selected_pair,
            competing_enter_distance=competing_enter_distance,
            competing_s_upper=competing_s_upper,
            retained_order=retained_order,
            computed_order=computed_order,
            competing_pairs=competing_pairs,
            ordering_tolerance=ordering_tolerance,
        )
        if leaf.certified:
            leaves.append(leaf)
            continue
        split_axis = (
            None
            if depth >= max_depth
            else _ks_event_order_split_axis(
                state_interval,
                min_state_width=min_state_width,
            )
        )
        if split_axis is None:
            if not any(
                state_interval[index][1] > state_interval[index][0]
                for index in range(21)
            ):
                leaf = replace(
                    leaf,
                    missing_obligations=tuple(
                        dict.fromkeys(
                            (
                                *leaf.missing_obligations,
                                "ks_event_order_state_width_exhausted",
                            )
                        )
                    ),
                )
            leaves.append(leaf)
            continue
        if len(leaves) + len(pending) + 2 > max_branches:
            stopped_by_branch_limit = True
            leaves.append(leaf)
            continue
        left_interval, right_interval = _bisect_state_interval_tuple(
            state_interval,
            split_axis,
        )
        next_depth = depth + 1
        pending.append(
            (
                left_interval,
                _ks_state_from_interval_tuple(initial_state, left_interval),
                next_depth,
                f"{branch_id}.L{split_axis}",
            )
        )
        pending.append(
            (
                right_interval,
                _ks_state_from_interval_tuple(initial_state, right_interval),
                next_depth,
                f"{branch_id}.R{split_axis}",
            )
        )
        split_count += 1

    state_union = tuple(leaf.state_interval for leaf in leaves)
    hull = _hull_state_interval_tuples(state_union) if state_union else ()
    tolerance = 16.0 * np.finfo(float).eps * max(
        1.0,
        max(abs(bound) for interval in original_state_interval for bound in interval),
    )
    recursive_cover = bool(
        leaves
        and len(leaves) == split_count + 1
        and _ks_event_order_bisection_paths_cover_root(
            tuple(leaf.branch_id for leaf in leaves)
        )
        and all(
            _state_interval_tuple_subset(leaf.state_interval, original_state_interval)
            for leaf in leaves
        )
        and _state_interval_tuples_same_bounds(
            hull,
            original_state_interval,
            tolerance=tolerance,
        )
    )
    leaf_decisions = bool(recursive_cover and all(leaf.certified for leaf in leaves))
    missing: list[str] = []
    if not recursive_cover:
        missing.append("ks_event_order_recursive_bisection_cover")
    if not leaf_decisions:
        missing.append("ks_event_order_leaf_decisions")
    for leaf in leaves:
        missing.extend(leaf.missing_obligations)
    if stopped_by_branch_limit:
        missing.append("ks_event_order_partition_branch_limit")
    return KSEventOrderSplitCertificate(
        original_state_interval=original_state_interval,
        target_time_interval=target_time_interval,
        selected_pair=selected_pair,
        competing_enter_distance=competing_enter_distance,
        competing_s_upper=competing_s_upper,
        retained_order=retained_order,
        computed_order=computed_order,
        leaves=tuple(leaves),
        split_count=split_count,
        max_depth=max_depth,
        recursive_bisection_cover_certified=recursive_cover,
        leaf_decisions_certified=leaf_decisions,
        missing_obligations=tuple(dict.fromkeys(missing)),
    )


def _derive_spatial_ks_competing_handoff_candidates_from_ks_state(
    initial_ks_state: IntervalSpatialKSBinaryChartState,
    *,
    target_time_after_ks_start: float,
    binary_distance_threshold: float,
    s_upper: float,
    retained_order: int,
    guard_order: int,
) -> tuple[tuple[float, tuple[int, int], str, float, float, float], ...]:
    target_time_after_ks_start = float(target_time_after_ks_start)
    threshold = float(binary_distance_threshold)
    s_upper = float(s_upper)
    if target_time_after_ks_start <= 0.0 or threshold <= 0.0 or s_upper <= 0.0:
        return ()
    retained_order = int(retained_order)
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

    selected_pair = tuple(initial_ks_state.pair)
    event_set = certify_next_finite_time_event_set(
        ks_solution,
        target_time_after_start_interval=FloatInterval.point(target_time_after_ks_start),
        selected_pair=selected_pair,
        competing_enter_distance=threshold,
        competing_s_upper=s_upper,
        retained_order=retained_order,
        require_all_event_certificates=True,
    )
    first_event = event_set.first_event
    if (
        event_set.target_before_all_events
        or not event_set.unique_first_event_certified
        or first_event is None
        or first_event.event_type != "spatial_ks_competing_binary_entry"
        or first_event.pair is None
        or not isinstance(first_event.certificate, SpatialKSEntryEventCertificate)
    ):
        return ()
    try:
        branch_states = spatial_ks_competing_entry_event_to_ks_chart_state_atlas(
            ks_solution,
            first_event.certificate,
        )
    except (RuntimeError, ValueError, TypeError):
        return ()
    candidates: list[tuple[float, tuple[int, int], str, float, float, float]] = []
    entry_time_lower = float(first_event.physical_time_interval.lower)
    if not np.isfinite(entry_time_lower):
        return ()
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


def _target_time_interval_from_validated_atlas(
    atlas: ValidatedAtlasSolution,
) -> FloatInterval:
    interval = getattr(atlas.evaluation, "target_time_interval", None)
    if _float_interval_finite_nonempty(interval):
        return _coerce_float_interval(interval)
    return FloatInterval.point(float(atlas.target_time))


def _atlas_certification_failure_obligations(
    atlas: ValidatedAtlasSolution,
) -> tuple[str, ...]:
    """Expose nested local-collision certificate failures in constructor errors."""

    missing: list[str] = list(atlas.missing_certification_obligations)
    evaluation = getattr(atlas, "evaluation", None)
    for name in (
        "collision_policy_certificate",
        "first_collision_policy_certificate",
        "next_collision_policy_certificate",
    ):
        certificate = getattr(evaluation, name, None)
        missing.extend(tuple(getattr(certificate, "missing_obligations", ())))
    return tuple(dict.fromkeys(obligation for obligation in missing if str(obligation)))


def _validate_spatial_ks_competing_next_atlas_override(
    next_atlas: ValidatedAtlasSolution,
    *,
    next_ks_state: IntervalSpatialKSBinaryChartState,
    next_target_interval: FloatInterval | None,
) -> None:
    """Verify that an injected suffix atlas starts from the derived next KS state."""

    if not next_atlas.proof_certified:
        raise ValueError("next atlas override is not proof-certified")
    if next_target_interval is None:
        raise ValueError("next atlas override requires a physical target interval")
    partition = getattr(next_atlas.evaluation, "branch_partition", None)
    expected_state_interval = _ks_state_interval_tuple(next_ks_state)
    tolerance = 16.0 * np.finfo(float).eps * max(
        1.0,
        max(abs(bound) for interval in expected_state_interval for bound in interval),
    )
    if isinstance(partition, KSEventOrderSplitCertificate):
        if not partition.certified:
            raise ValueError("next atlas override partition is not certified")
        state_matches = _state_interval_tuples_same_bounds(
            partition.original_state_interval,
            expected_state_interval,
            tolerance=tolerance,
        )
    elif isinstance(partition, SimultaneousClosePairSplitCertificate):
        if not partition.certified:
            raise ValueError("next atlas override partition is not certified")
        projected_next_state = project_spatial_ks_binary_interval_chart_state_to_physical(
            next_ks_state,
            physical_time=FloatInterval.point(0.0),
        )
        projected_interval = tuple(
            (float(lower), float(upper))
            for lower, upper in projected_next_state.state_interval
        )
        state_matches = bool(
            projected_next_state.certified
            and _state_interval_tuples_same_bounds(
                partition.original_state_interval,
                projected_interval,
                tolerance=tolerance,
            )
        )
    else:
        projected_next_state = project_spatial_ks_binary_interval_chart_state_to_physical(
            next_ks_state,
            physical_time=FloatInterval.point(0.0),
        )
        override_projection = getattr(next_atlas.evaluation, "initial_projection", None)
        state_matches = bool(
            projected_next_state.certified
            and getattr(override_projection, "certified", False)
            and _interval_arrays_equal(
                _flat_float_interval_array(projected_next_state.state_interval),
                _flat_float_interval_array(override_projection.state_interval),
            )
        )
    if not state_matches:
        raise ValueError("next atlas override does not start from the derived KS state")
    override_target = _target_time_interval_from_validated_atlas(next_atlas)
    target_tolerance = _physical_time_progress_tolerance(
        next_target_interval.lower,
        next_target_interval.upper,
        override_target.lower,
        override_target.upper,
    )
    if (
        abs(next_target_interval.lower - override_target.lower) > target_tolerance
        or abs(next_target_interval.upper - override_target.upper) > target_tolerance
    ):
        raise ValueError("next atlas override target interval does not match the prefix remainder")


def _spatial_ks_target_only_validated_atlas(
    *,
    initial_state: IntervalSpatialKSBinaryChartState,
    ks_solution: object,
    target_time_after_ks_start_interval: FloatInterval,
    s_search_upper: float,
    retained_order: int,
    guard_order: int,
    ks_residual: object,
    ks_horizontal: object,
    ks_pair_energy: object,
    ks_center: object,
    ks_momentum: object,
    ks_angular: object,
    ks_energy: object,
    exit_event_certificate: SpatialKSRhoExitEventCertificate | None,
    source: str,
) -> ValidatedAtlasSolution:
    """Return a local spatial KS atlas whose requested target stays regularized."""

    target_s = _ks_physical_time_root_for_target(
        ks_solution,
        target_time_after_ks_start_interval,
        s_upper=s_search_upper,
    )
    target_tail = _certify_spatial_ks_target_tail(
        initial_state,
        ks_solution=ks_solution,
        target_s=target_s,
        target_time_after_ks_start_interval=target_time_after_ks_start_interval,
        s_search_upper=s_search_upper,
        retained_order=retained_order,
        guard_order=guard_order,
    )
    ks_invariants_certified = bool(
        ks_center.certified
        and ks_momentum.certified
        and ks_angular.certified
        and ks_energy.certified
    )
    charts = (
        ValidatedChart(
            chart_id="spatial_ks_0",
            chart_type="spatial_ks_binary",
            source=source,
            parameter_name="s",
            parameter_interval=FloatInterval(min(0.0, target_s), max(0.0, target_s)),
            physical_time_interval=FloatInterval(
                min(0.0, target_tail.endpoint_projection.physical_time.lower),
                max(0.0, target_tail.endpoint_projection.physical_time.upper),
            ),
            dynamics_certified=bool(initial_state.certified and ks_residual.certified),
            residual_certified=ks_residual.certified,
            projection_certified=bool(
                target_tail.endpoint_projection.certified
                and ks_horizontal.certified
                and ks_pair_energy.certified
            ),
            invariants_certified=ks_invariants_certified,
            tail_certified=target_tail.tail_certified,
            tail_bound=target_tail.tail_bound,
        ),
    )
    invariants = GlobalInvariantLedger(
        center_of_mass_certified=ks_center.certified,
        linear_momentum_certified=ks_momentum.certified,
        angular_momentum_certified=ks_angular.certified,
        energy_certified=ks_energy.certified,
        certified_chart_count=sum(chart.invariants_certified for chart in charts),
        expected_chart_count=len(charts),
    )
    residual_budget = NewtonResidualLedger(
        certified=ks_residual.certified,
        certified_chart_count=int(ks_residual.certified),
        expected_chart_count=len(charts),
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=float(target_tail.tail_bound),
        max_step_tail_bound=float(target_tail.tail_bound),
        certified=target_tail.tail_certified,
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="local_spatial_ks_binary_target_inside_regularized_chart",
        binary_policy="spatial_ks_selected_binary_local_regularized_target_inside_ks",
        total_collision_policy="not_classified_by_local_ks_handoff",
        triple_collision_status="not_classified_by_local_ks_handoff",
        triple_collision_reason=(
            "local KS chart evaluates the requested physical target before "
            "ordinary handoff; it does not classify total collision"
        ),
        certified=False,
    )
    evaluation = SpatialKSValidatedEvaluation(
        ks_solution=ks_solution,
        ordinary_solution=None,
        endpoint_projection=target_tail.endpoint_projection,
        exit_event_certificate=exit_event_certificate,
        target_state_interval=target_tail.target_state_interval,
        target_time_interval=target_time_after_ks_start_interval,
        ks_tail_certificate=target_tail.tail_certificate,
    )
    proof_ledger = ProofLedger(
        entries=(
            ProofLedgerEntry("mass_domain", _masses_positive_finite(initial_state.masses), source),
            ProofLedgerEntry("spatial_ks_branch_domain", initial_state.certified, source),
            ProofLedgerEntry("spatial_ks_equation_residuals", ks_residual.certified, source),
            ProofLedgerEntry(
                "spatial_ks_projection_constraints",
                bool(ks_horizontal.certified and ks_pair_energy.certified),
                source,
            ),
            ProofLedgerEntry(
                "spatial_ks_target_time_projection",
                target_tail.endpoint_projection.certified,
                source,
                detail=target_tail.proof_detail,
            ),
            ProofLedgerEntry(
                "local_tail_budget",
                tail_budget.certified and tail_budget.finite,
                source,
            ),
            ProofLedgerEntry(
                "spatial_ks_exit_event_isolation",
                bool(exit_event_certificate is not None and exit_event_certificate.certified),
                source,
                required=bool(exit_event_certificate is not None),
                detail=(
                    "rho-exit event certified but ordinary handoff is not required for this target"
                    if exit_event_certificate is not None and exit_event_certificate.certified
                    else "missing certified rho-exit event constructor input"
                ),
            ),
            ProofLedgerEntry(
                "spatial_ks_entry_event_isolation",
                False,
                source,
                detail="local adapter starts inside a selected KS chart and does not certify ordinary-to-KS entry",
            ),
            ProofLedgerEntry(
                "finite_time_physical_targeting",
                True,
                source,
                detail="requested local physical target time is enclosed by the KS chart",
            ),
            ProofLedgerEntry(
                "target_time_domain",
                _target_time_in_final_chart_domain(
                    0.5
                    * (
                        target_time_after_ks_start_interval.lower
                        + target_time_after_ks_start_interval.upper
                    ),
                    charts,
                ),
                source,
            ),
            ProofLedgerEntry(
                "spatial_collision_policy_scope",
                collision_policy.certified,
                source,
                detail="selected-binary local handoff does not classify total collision or competing events",
            ),
        )
    )
    return ValidatedAtlasSolution(
        masses=tuple(
            float(mass)
            for mass in np.asarray(initial_state.masses, dtype=float).reshape(-1)
        ),
        target_time=0.5
        * (
            target_time_after_ks_start_interval.lower
            + target_time_after_ks_start_interval.upper
        ),
        initial_state_interval=None,
        charts=charts,
        transitions=(),
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=proof_ledger,
        evaluation=evaluation,
    )


def validated_atlas_from_spatial_ks_binary_chart(
    initial_state: IntervalSpatialKSBinaryChartState,
    *,
    s_endpoint: float | None = None,
    exit_rho: float | None = None,
    s_upper: float | None = None,
    ordinary_step_size: float | None = None,
    target_time_after_ks_start_interval: FloatInterval | tuple[float, float] | None = None,
    retained_order: int,
    guard_order: int,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    source: str = "spatial_ks_binary_local_handoff",
) -> ValidatedAtlasSolution:
    """Expose a local spatial KS binary exit and ordinary handoff as an atlas.

    This constructor derives every local certificate it records.  It does not
    claim finite-time spatial binary continuation: event-isolated KS entry/exit
    and physical target-time bracketing remain required ledger obligations.
    """

    if s_endpoint is not None:
        s_endpoint = float(s_endpoint)
    if exit_rho is not None:
        exit_rho = float(exit_rho)
    if s_upper is not None:
        s_upper = float(s_upper)
    if ordinary_step_size is not None:
        ordinary_step_size = float(ordinary_step_size)
    if target_time_after_ks_start_interval is not None:
        target_time_after_ks_start_interval = _coerce_float_interval(
            target_time_after_ks_start_interval
        )
    retained_order = int(retained_order)
    guard_order = int(guard_order)
    if retained_order < 1:
        raise ValueError("retained_order must be positive")
    if guard_order < 1:
        raise ValueError("guard_order must be positive")
    if ordinary_step_size is None and target_time_after_ks_start_interval is None:
        raise ValueError(
            "ordinary_step_size or target_time_after_ks_start_interval is required"
        )
    if ordinary_step_size is not None and ordinary_step_size <= 0.0:
        raise ValueError("ordinary_step_size must be positive")
    ordinary_handoff_min_pair_distance_required = float(
        ordinary_handoff_min_pair_distance_required
    )
    ordinary_handoff_max_acceleration_bound = float(ordinary_handoff_max_acceleration_bound)
    ordinary_handoff_min_cauchy_radius = float(ordinary_handoff_min_cauchy_radius)
    ordinary_handoff_max_residual_bound = float(ordinary_handoff_max_residual_bound)
    ordinary_handoff_max_tail_bound = float(ordinary_handoff_max_tail_bound)

    computed_order = retained_order + guard_order
    ks_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        initial_state,
        order=computed_order,
    )
    exit_event_certificate = None
    if exit_rho is not None:
        if s_upper is None:
            raise ValueError("s_upper is required when exit_rho is supplied")
        exit_event_certificate = certify_spatial_ks_binary_rho_exit_event(
            ks_solution,
            exit_rho=exit_rho,
            s_upper=s_upper,
            coefficient_count=retained_order,
        )
        if not exit_event_certificate.certified:
            raise ValueError("KS rho-exit event could not be interval-certified")
        s_endpoint = float(exit_event_certificate.root)
    if s_endpoint is None:
        raise ValueError("s_endpoint is required unless exit_rho and s_upper are supplied")
    ks_residual = certify_spatial_ks_binary_interval_taylor_equations(
        ks_solution,
        coefficient_count=retained_order,
    )
    ks_horizontal = certify_spatial_ks_binary_horizontal_constraint(
        ks_solution,
        coefficient_count=retained_order,
    )
    ks_pair_energy = certify_spatial_ks_binary_pair_energy_constraint(
        ks_solution,
        coefficient_count=retained_order,
    )
    ks_center = certify_spatial_ks_binary_center_of_mass_motion(
        ks_solution,
        coefficient_count=retained_order,
    )
    ks_momentum = certify_spatial_ks_binary_linear_momentum_conservation(
        ks_solution,
        coefficient_count=retained_order,
    )
    ks_angular = certify_spatial_ks_binary_centered_angular_momentum_conservation(
        ks_solution,
        coefficient_count=retained_order,
    )
    ks_energy = certify_spatial_ks_binary_total_energy_conservation(
        ks_solution,
        coefficient_count=retained_order,
    )
    ks_tail = spatial_ks_binary_interval_tail_certificate(
        initial_state,
        retained_order=retained_order,
        guard_order=guard_order,
        step_size=s_endpoint,
    )
    endpoint_projection = project_spatial_ks_binary_taylor_endpoint_to_physical(
        ks_solution,
        s_endpoint,
    )
    if not endpoint_projection.certified:
        raise ValueError("KS endpoint must certify rho.lower > 0 before ordinary handoff")

    finite_time_targeting_certified = target_time_after_ks_start_interval is not None
    target_inside_ks = bool(
        target_time_after_ks_start_interval is not None
        and target_time_after_ks_start_interval.upper
        <= endpoint_projection.physical_time.lower + 1.0e-14
    )
    if target_inside_ks:
        target_s = _ks_physical_time_root_for_target(
            ks_solution,
            target_time_after_ks_start_interval,
            s_upper=s_endpoint,
        )
        target_tail = _certify_spatial_ks_target_tail(
            initial_state,
            ks_solution=ks_solution,
            target_s=target_s,
            target_time_after_ks_start_interval=target_time_after_ks_start_interval,
            s_search_upper=s_endpoint,
            retained_order=retained_order,
            guard_order=guard_order,
        )
        ks_invariants_certified = bool(
            ks_center.certified
            and ks_momentum.certified
            and ks_angular.certified
            and ks_energy.certified
        )
        charts = (
            ValidatedChart(
                chart_id="spatial_ks_0",
                chart_type="spatial_ks_binary",
                source=source,
                parameter_name="s",
                parameter_interval=FloatInterval(min(0.0, target_s), max(0.0, target_s)),
                physical_time_interval=FloatInterval(
                    min(0.0, target_tail.endpoint_projection.physical_time.lower),
                    max(0.0, target_tail.endpoint_projection.physical_time.upper),
                ),
                dynamics_certified=bool(initial_state.certified and ks_residual.certified),
                residual_certified=ks_residual.certified,
                projection_certified=bool(
                    target_tail.endpoint_projection.certified
                    and ks_horizontal.certified
                    and ks_pair_energy.certified
                ),
                invariants_certified=ks_invariants_certified,
                tail_certified=target_tail.tail_certified,
                tail_bound=target_tail.tail_bound,
            ),
        )
        invariants = GlobalInvariantLedger(
            center_of_mass_certified=ks_center.certified,
            linear_momentum_certified=ks_momentum.certified,
            angular_momentum_certified=ks_angular.certified,
            energy_certified=ks_energy.certified,
            certified_chart_count=sum(chart.invariants_certified for chart in charts),
            expected_chart_count=len(charts),
        )
        residual_budget = NewtonResidualLedger(
            certified=ks_residual.certified,
            certified_chart_count=int(ks_residual.certified),
            expected_chart_count=len(charts),
        )
        tail_budget = TailBudgetLedger(
            local_tail_bound=float(target_tail.tail_bound),
            max_step_tail_bound=float(target_tail.tail_bound),
            certified=target_tail.tail_certified,
        )
        collision_policy = CollisionPolicyWitness(
            policy_id="local_spatial_ks_binary_target_inside_regularized_chart",
            binary_policy="spatial_ks_selected_binary_local_regularized_target_inside_ks",
            total_collision_policy="not_classified_by_local_ks_handoff",
            triple_collision_status="not_classified_by_local_ks_handoff",
            triple_collision_reason=(
                "local KS chart evaluates the requested physical target before "
                "ordinary handoff; it does not classify total collision"
            ),
            certified=False,
        )
        evaluation = SpatialKSValidatedEvaluation(
            ks_solution=ks_solution,
            ordinary_solution=None,
            endpoint_projection=target_tail.endpoint_projection,
            exit_event_certificate=exit_event_certificate,
            target_state_interval=target_tail.target_state_interval,
            target_time_interval=target_time_after_ks_start_interval,
            ks_tail_certificate=target_tail.tail_certificate,
        )
        proof_ledger = ProofLedger(
            entries=(
                ProofLedgerEntry("mass_domain", _masses_positive_finite(initial_state.masses), source),
                ProofLedgerEntry("spatial_ks_branch_domain", initial_state.certified, source),
                ProofLedgerEntry("spatial_ks_equation_residuals", ks_residual.certified, source),
                ProofLedgerEntry(
                    "spatial_ks_projection_constraints",
                    bool(ks_horizontal.certified and ks_pair_energy.certified),
                    source,
                ),
                ProofLedgerEntry(
                    "spatial_ks_target_time_projection",
                    target_tail.endpoint_projection.certified,
                    source,
                    detail=target_tail.proof_detail,
                ),
                ProofLedgerEntry(
                    "local_tail_budget",
                    tail_budget.certified and tail_budget.finite,
                    source,
                ),
                ProofLedgerEntry(
                "spatial_ks_exit_event_isolation",
                bool(exit_event_certificate is not None and exit_event_certificate.certified),
                source,
                required=bool(exit_event_certificate is not None),
                detail=(
                    "rho-exit event certified but ordinary handoff is not required for this target"
                    if exit_event_certificate is not None and exit_event_certificate.certified
                    else "missing certified rho-exit event constructor input"
                    ),
                ),
                ProofLedgerEntry(
                    "spatial_ks_entry_event_isolation",
                    False,
                    source,
                    detail="local adapter starts inside a selected KS chart and does not certify ordinary-to-KS entry",
                ),
                ProofLedgerEntry(
                    "finite_time_physical_targeting",
                    True,
                    source,
                    detail="requested local physical target time is enclosed by the KS chart",
                ),
                ProofLedgerEntry(
                    "target_time_domain",
                    _target_time_in_final_chart_domain(
                        0.5
                        * (
                            target_time_after_ks_start_interval.lower
                            + target_time_after_ks_start_interval.upper
                        ),
                        charts,
                    ),
                    source,
                ),
                ProofLedgerEntry(
                    "spatial_collision_policy_scope",
                    collision_policy.certified,
                    source,
                    detail="selected-binary local handoff does not classify total collision or competing events",
                ),
            )
        )
        return ValidatedAtlasSolution(
            masses=tuple(
                float(mass)
                for mass in np.asarray(initial_state.masses, dtype=float).reshape(-1)
            ),
            target_time=0.5
            * (
                target_time_after_ks_start_interval.lower
                + target_time_after_ks_start_interval.upper
            ),
            initial_state_interval=None,
            charts=charts,
            transitions=(),
            invariants=invariants,
            tail_budget=tail_budget,
            residual_budget=residual_budget,
            collision_policy=collision_policy,
            proof_ledger=proof_ledger,
            evaluation=evaluation,
        )

    ordinary_initial_state_interval = _inflate_state_interval_tuples(
        endpoint_projection.state_interval,
        ks_tail.tail_bound,
    )

    if target_time_after_ks_start_interval is not None:
        ordinary_parameter_interval = (
            target_time_after_ks_start_interval - endpoint_projection.physical_time
        )
        if ordinary_parameter_interval.lower < -1.0e-14:
            return _spatial_ks_target_only_validated_atlas(
                initial_state=initial_state,
                ks_solution=ks_solution,
                target_time_after_ks_start_interval=target_time_after_ks_start_interval,
                s_search_upper=float(s_upper if s_upper is not None else s_endpoint),
                retained_order=retained_order,
                guard_order=guard_order,
                ks_residual=ks_residual,
                ks_horizontal=ks_horizontal,
                ks_pair_energy=ks_pair_energy,
                ks_center=ks_center,
                ks_momentum=ks_momentum,
                ks_angular=ks_angular,
                ks_energy=ks_energy,
                exit_event_certificate=exit_event_certificate,
                source=source,
            )
        if ordinary_parameter_interval.lower < 0.0:
            ordinary_parameter_interval = FloatInterval(0.0, ordinary_parameter_interval.upper)
        target_time_interval = target_time_after_ks_start_interval
    else:
        ordinary_parameter_interval = FloatInterval.point(float(ordinary_step_size))
        target_time_interval = endpoint_projection.physical_time + ordinary_parameter_interval
    ordinary_chart_parameter_interval = _forward_chart_domain(ordinary_parameter_interval)
    ordinary_chart_physical_time_interval = FloatInterval(
        min(endpoint_projection.physical_time.lower, target_time_interval.lower),
        max(endpoint_projection.physical_time.upper, target_time_interval.upper),
    )
    pre_handoff_admissibility = certify_spatial_ks_to_ordinary_handoff_admissibility(
        endpoint_projection,
        retained_order=retained_order,
        requested_post_time_interval=ordinary_parameter_interval,
        ordinary_initial_state_interval=ordinary_initial_state_interval,
        min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
        max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
        min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
        max_residual_bound=ordinary_handoff_max_residual_bound,
        max_tail_bound=ordinary_handoff_max_tail_bound,
        require_ordinary_chart_evidence=False,
    )
    if not pre_handoff_admissibility.certified:
        if target_time_after_ks_start_interval is not None:
            return _spatial_ks_target_only_validated_atlas(
                initial_state=initial_state,
                ks_solution=ks_solution,
                target_time_after_ks_start_interval=target_time_after_ks_start_interval,
                s_search_upper=float(s_upper if s_upper is not None else s_endpoint),
                retained_order=retained_order,
                guard_order=guard_order,
                ks_residual=ks_residual,
                ks_horizontal=ks_horizontal,
                ks_pair_energy=ks_pair_energy,
                ks_center=ks_center,
                ks_momentum=ks_momentum,
                ks_angular=ks_angular,
                ks_energy=ks_energy,
                exit_event_certificate=exit_event_certificate,
                source=source,
            )
        raise ValueError(
            "ordinary handoff is not admissible: "
            + ", ".join(pre_handoff_admissibility.missing_obligations)
        )

    ordinary_positions, ordinary_velocities = _spatial_interval_state_arrays(
        ordinary_initial_state_interval,
    )
    ordinary_solution = construct_interval_taylor_solution_from_intervals(
        ordinary_positions,
        ordinary_velocities,
        initial_state.masses,
        order=computed_order,
    )
    ordinary_residual = certify_ordinary_interval_taylor_equations(
        ordinary_solution,
        coefficient_count=retained_order,
    )
    ordinary_time_series = _ordinary_physical_time_series(computed_order)
    ordinary_center = certify_interval_center_of_mass_motion(
        ordinary_solution.position,
        ordinary_solution.velocity,
        ordinary_time_series,
        initial_state.masses,
        coefficient_count=retained_order,
    )
    ordinary_momentum = certify_interval_linear_momentum_conservation(
        ordinary_solution.velocity,
        initial_state.masses,
        coefficient_count=retained_order,
    )
    ordinary_angular = certify_interval_centered_angular_momentum_conservation(
        ordinary_solution.position,
        ordinary_solution.velocity,
        initial_state.masses,
        coefficient_count=retained_order,
    )
    ordinary_energy = certify_interval_total_energy_conservation(
        ordinary_solution.position,
        ordinary_solution.velocity,
        initial_state.masses,
        coefficient_count=retained_order,
    )
    ordinary_step_bound = _interval_max_abs(ordinary_chart_parameter_interval)
    ordinary_tail = interval_guarded_tail_certificate(
        ordinary_interval_solution_arrays(ordinary_solution),
        retained_order,
        ordinary_step_bound,
    )
    ordinary_handoff_admissibility = certify_spatial_ks_to_ordinary_handoff_admissibility(
        endpoint_projection,
        retained_order=retained_order,
        requested_post_time_interval=ordinary_parameter_interval,
        ordinary_initial_state_interval=ordinary_initial_state_interval,
        ordinary_solution=ordinary_solution,
        ordinary_residual=ordinary_residual,
        ordinary_tail=ordinary_tail,
        min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
        max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
        min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
        max_residual_bound=ordinary_handoff_max_residual_bound,
        max_tail_bound=ordinary_handoff_max_tail_bound,
    )
    if not ordinary_handoff_admissibility.certified:
        raise ValueError(
            "ordinary handoff is not admissible: "
            + ", ".join(ordinary_handoff_admissibility.missing_obligations)
        )
    ordinary_target_state = _ordinary_target_state_interval_over_interval(
        ordinary_solution,
        ordinary_parameter_interval,
        retained_order=retained_order,
        tail_bound=ordinary_handoff_admissibility.tail_bound,
    )
    evaluation = SpatialKSValidatedEvaluation(
        ks_solution=ks_solution,
        ordinary_solution=ordinary_solution,
        endpoint_projection=endpoint_projection,
        exit_event_certificate=exit_event_certificate,
        target_state_interval=ordinary_target_state,
        target_time_interval=target_time_interval,
        ordinary_handoff_admissibility=ordinary_handoff_admissibility,
        ks_tail_certificate=ks_tail,
    )

    ks_invariants_certified = bool(
        ks_center.certified
        and ks_momentum.certified
        and ks_angular.certified
        and ks_energy.certified
    )
    ordinary_invariants_certified = bool(
        ordinary_center.certified
        and ordinary_momentum.certified
        and ordinary_angular.certified
        and ordinary_energy.certified
    )
    charts = (
        ValidatedChart(
            chart_id="spatial_ks_0",
            chart_type="spatial_ks_binary",
            source=source,
            parameter_name="s",
            parameter_interval=FloatInterval(min(0.0, s_endpoint), max(0.0, s_endpoint)),
            physical_time_interval=FloatInterval(
                min(0.0, endpoint_projection.physical_time.lower),
                max(0.0, endpoint_projection.physical_time.upper),
            ),
            dynamics_certified=bool(initial_state.certified and ks_residual.certified),
            residual_certified=ks_residual.certified,
            projection_certified=bool(
                endpoint_projection.certified
                and ks_horizontal.certified
                and ks_pair_energy.certified
            ),
            invariants_certified=ks_invariants_certified,
            tail_certified=ks_tail.is_nontrivial,
            tail_bound=ks_tail.tail_bound,
        ),
        ValidatedChart(
            chart_id="ordinary_after_spatial_ks_0",
            chart_type="spatial_ordinary_taylor_after_ks",
            source=source,
            parameter_name="t",
            parameter_interval=ordinary_chart_parameter_interval,
            physical_time_interval=ordinary_chart_physical_time_interval,
            dynamics_certified=ordinary_residual.certified,
            residual_certified=ordinary_residual.certified,
            projection_certified=True,
            invariants_certified=ordinary_invariants_certified,
            tail_certified=ordinary_handoff_admissibility.tail_certified,
            tail_bound=ordinary_handoff_admissibility.tail_bound,
        ),
    )
    transitions = (
        ValidatedTransition(
            source_chart_id=charts[0].chart_id,
            target_chart_id=charts[1].chart_id,
            transition_type=(
                "spatial_ks_exit_event_to_ordinary_rho_positive_projection"
                if exit_event_certificate is not None and exit_event_certificate.certified
                else "spatial_ks_to_ordinary_rho_positive_endpoint_projection"
            ),
            certified=bool(
                endpoint_projection.certified
                and ordinary_handoff_admissibility.certified
                and _interval_array_subset(
                    _flat_float_interval_array(endpoint_projection.state_interval),
                    _flat_float_interval_array(ordinary_initial_state_interval),
                )
                and _ordinary_initial_matches_projection(
                    ordinary_solution,
                    ordinary_initial_state_interval,
                )
            ),
            source=source,
        ),
    )
    invariants = GlobalInvariantLedger(
        center_of_mass_certified=bool(ks_center.certified and ordinary_center.certified),
        linear_momentum_certified=bool(ks_momentum.certified and ordinary_momentum.certified),
        angular_momentum_certified=bool(ks_angular.certified and ordinary_angular.certified),
        energy_certified=bool(ks_energy.certified and ordinary_energy.certified),
        certified_chart_count=sum(chart.invariants_certified for chart in charts),
        expected_chart_count=len(charts),
    )
    residual_budget = NewtonResidualLedger(
        certified=bool(ks_residual.certified and ordinary_residual.certified),
        certified_chart_count=int(ks_residual.certified) + int(ordinary_residual.certified),
        expected_chart_count=len(charts),
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=float(ks_tail.tail_bound + ordinary_handoff_admissibility.tail_bound),
        max_step_tail_bound=float(
            max(ks_tail.tail_bound, ordinary_handoff_admissibility.tail_bound)
        ),
        certified=bool(ks_tail.is_nontrivial and ordinary_handoff_admissibility.tail_certified),
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="local_spatial_ks_binary_to_ordinary_handoff",
        binary_policy="spatial_ks_selected_binary_local_regularized",
        total_collision_policy="not_classified_by_local_ks_handoff",
        triple_collision_status="not_classified_by_local_ks_handoff",
        triple_collision_reason=(
            "local KS handoff certifies one selected separated binary chart and "
            "rho-positive ordinary exit; it does not classify total collision"
        ),
        certified=False,
    )
    proof_ledger = ProofLedger(
        entries=(
            ProofLedgerEntry(
                "mass_domain",
                _masses_positive_finite(initial_state.masses),
                source,
            ),
            ProofLedgerEntry("spatial_ks_branch_domain", initial_state.certified, source),
            ProofLedgerEntry("spatial_ks_equation_residuals", ks_residual.certified, source),
            ProofLedgerEntry(
                "spatial_ks_projection_constraints",
                bool(ks_horizontal.certified and ks_pair_energy.certified),
                source,
            ),
            ProofLedgerEntry(
                "spatial_ks_rho_positive_endpoint_projection",
                endpoint_projection.certified,
                source,
            ),
            ProofLedgerEntry(
                "ordinary_handoff_admissibility",
                ordinary_handoff_admissibility.certified,
                source,
                detail=(
                    "post-KS ordinary chart has certified pair-distance, "
                    "acceleration, Cauchy-radius, residual, and tail bounds"
                ),
            ),
            ProofLedgerEntry("ordinary_post_handoff_residuals", ordinary_residual.certified, source),
            ProofLedgerEntry("spatial_ks_and_ordinary_invariants", invariants.certified, source),
            ProofLedgerEntry("local_tail_budget", tail_budget.certified and tail_budget.finite, source),
            ProofLedgerEntry("rho_positive_handoff_transition", transitions[0].certified, source),
            ProofLedgerEntry(
                "spatial_ks_exit_event_isolation",
                bool(exit_event_certificate is not None and exit_event_certificate.certified),
                source,
                required=bool(exit_event_certificate is not None),
                detail=(
                    "rho-exit event certified"
                    if exit_event_certificate is not None and exit_event_certificate.certified
                    else (
                        "ordinary handoff is certified from a chosen rho-positive endpoint; "
                        "no rho-exit root isolation is claimed"
                    )
                ),
            ),
            ProofLedgerEntry(
                "spatial_ks_entry_event_isolation",
                False,
                source,
                detail="local adapter starts inside a selected KS chart and does not certify ordinary-to-KS entry",
            ),
            ProofLedgerEntry(
                "finite_time_physical_targeting",
                finite_time_targeting_certified,
                source,
                detail=(
                    "requested local physical target time is enclosed by the post-exit ordinary chart"
                    if finite_time_targeting_certified
                    else "local s-endpoint handoff has not been selected by a physical target-time bracket"
                ),
            ),
            ProofLedgerEntry(
                "target_time_domain",
                _target_time_in_final_chart_domain(
                    0.5 * (target_time_interval.lower + target_time_interval.upper),
                    charts,
                ),
                source,
            ),
            ProofLedgerEntry(
                "spatial_collision_policy_scope",
                collision_policy.certified,
                source,
                detail="selected-binary local handoff does not classify total collision or competing events",
            ),
        )
    )
    return ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in np.asarray(initial_state.masses, dtype=float).reshape(-1)),
        target_time=0.5 * (target_time_interval.lower + target_time_interval.upper),
        initial_state_interval=None,
        charts=charts,
        transitions=transitions,
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=proof_ledger,
        evaluation=evaluation,
    )


def validated_atlas_from_spatial_ordinary_ks_competing_binary_handoff(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    *,
    pair: tuple[int, int],
    enter_distance: float,
    entry_time_upper: float,
    branch: str,
    competing_pair: tuple[int, int],
    competing_enter_distance: float,
    competing_entry_s_upper: float,
    competing_branch: str,
    next_s_endpoint: float | None = None,
    next_exit_rho: float | None = None,
    next_s_upper: float | None = None,
    ordinary_step_size: float | None = None,
    target_time: float | None = None,
    retained_order: int,
    guard_order: int,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    competing_pair_min_distance_required: float = 0.0,
    max_competing_repeats: int = 0,
    source: str = "spatial_ordinary_ks_competing_binary_handoff",
) -> ValidatedAtlasSolution:
    """Compose ordinary entry, selected KS, and competing-pair KS charts."""

    retained_order = int(retained_order)
    guard_order = int(guard_order)
    if retained_order < 1:
        raise ValueError("retained_order must be positive")
    if guard_order < 1:
        raise ValueError("guard_order must be positive")

    masses = np.asarray(masses, dtype=float)
    pair = tuple(pair)
    computed_order = retained_order + guard_order
    ordinary_positions, ordinary_velocities = _spatial_interval_state_arrays(state_interval)
    ordinary_entry_solution = construct_interval_taylor_solution_from_intervals(
        ordinary_positions,
        ordinary_velocities,
        masses,
        order=computed_order,
    )
    entry_event_certificate = certify_spatial_ordinary_ks_entry_event(
        ordinary_entry_solution,
        pair=pair,
        enter_distance=enter_distance,
        time_upper=entry_time_upper,
        coefficient_count=retained_order,
    )
    if not entry_event_certificate.certified:
        raise ValueError("ordinary-to-KS entry event could not be interval-certified")
    entry_time_interval = FloatInterval(*entry_event_certificate.root_interval)
    entry_ks_state = spatial_ordinary_entry_event_to_ks_chart_state(
        ordinary_entry_solution,
        entry_event_certificate,
        branch=branch,
    )
    if not entry_ks_state.certified:
        raise ValueError("ordinary-to-KS entry event did not lift to a certified KS branch")
    entry_projection = project_spatial_ks_binary_interval_chart_state_to_physical(
        entry_ks_state,
        physical_time=entry_time_interval,
    )
    if not entry_projection.certified:
        raise ValueError("ordinary-to-KS branch lift did not project to a rho-positive state")

    target_time_after_ks_start_interval = (
        None
        if target_time is None
        else FloatInterval.point(float(target_time)) - entry_time_interval
    )
    ks_competing_atlas = validated_atlas_from_spatial_ks_competing_binary_handoff(
        entry_ks_state,
        competing_pair=tuple(competing_pair),
        enter_distance=float(competing_enter_distance),
        entry_s_upper=float(competing_entry_s_upper),
        branch=str(competing_branch),
        next_s_endpoint=next_s_endpoint,
        next_exit_rho=next_exit_rho,
        next_s_upper=next_s_upper,
        ordinary_step_size=ordinary_step_size,
        target_time_after_ks_start_interval=target_time_after_ks_start_interval,
        retained_order=retained_order,
        guard_order=guard_order,
        ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
        ordinary_handoff_max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
        ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
        ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
        ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
        competing_pair_min_distance_required=competing_pair_min_distance_required,
        max_competing_repeats=max_competing_repeats,
        source=source,
    )

    ordinary_entry_residual = certify_ordinary_interval_taylor_equations(
        ordinary_entry_solution,
        coefficient_count=retained_order,
    )
    ordinary_time_series = _ordinary_physical_time_series(computed_order)
    ordinary_entry_center = certify_interval_center_of_mass_motion(
        ordinary_entry_solution.position,
        ordinary_entry_solution.velocity,
        ordinary_time_series,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_momentum = certify_interval_linear_momentum_conservation(
        ordinary_entry_solution.velocity,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_angular = certify_interval_centered_angular_momentum_conservation(
        ordinary_entry_solution.position,
        ordinary_entry_solution.velocity,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_energy = certify_interval_total_energy_conservation(
        ordinary_entry_solution.position,
        ordinary_entry_solution.velocity,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_tail = interval_guarded_tail_certificate(
        ordinary_interval_solution_arrays(ordinary_entry_solution),
        retained_order,
        entry_time_interval.upper,
    )
    ordinary_entry_domain = FloatInterval(0.0, entry_time_interval.upper)
    ordinary_entry_invariants_certified = bool(
        ordinary_entry_center.certified
        and ordinary_entry_momentum.certified
        and ordinary_entry_angular.certified
        and ordinary_entry_energy.certified
    )
    ordinary_entry_chart = ValidatedChart(
        chart_id="ordinary_before_spatial_ks_0",
        chart_type="spatial_ordinary_taylor_before_ks",
        source=source,
        parameter_name="t",
        parameter_interval=ordinary_entry_domain,
        physical_time_interval=ordinary_entry_domain,
        dynamics_certified=ordinary_entry_residual.certified,
        residual_certified=ordinary_entry_residual.certified,
        projection_certified=True,
        invariants_certified=ordinary_entry_invariants_certified,
        tail_certified=ordinary_entry_tail.is_nontrivial,
        tail_bound=ordinary_entry_tail.tail_bound,
    )
    shifted_ks_charts = tuple(
        replace(
            chart,
            chart_id=(
                f"spatial_ks_{index + 1}"
                if chart.chart_type == "spatial_ks_binary"
                else f"{chart.chart_id}_after_ordinary_ks_{index + 1}"
            ),
            physical_time_interval=_shift_optional_interval(
                chart.physical_time_interval,
                entry_time_interval,
            ),
        )
        for index, chart in enumerate(ks_competing_atlas.charts)
    )
    charts = (ordinary_entry_chart, *shifted_ks_charts)
    transitions = (
        ValidatedTransition(
            source_chart_id=charts[0].chart_id,
            target_chart_id=charts[1].chart_id,
            transition_type="spatial_ordinary_to_ks_decreasing_distance_entry",
            certified=bool(
                entry_event_certificate.certified
                and entry_ks_state.certified
                and entry_projection.certified
            ),
            source=source,
        ),
        *(
            replace(
                transition,
                source_chart_id=charts[index + 1].chart_id,
                target_chart_id=charts[index + 2].chart_id,
            )
            for index, transition in enumerate(ks_competing_atlas.transitions)
        ),
    )
    invariants = GlobalInvariantLedger(
        center_of_mass_certified=bool(
            ordinary_entry_center.certified
            and ks_competing_atlas.invariants.center_of_mass_certified
        ),
        linear_momentum_certified=bool(
            ordinary_entry_momentum.certified
            and ks_competing_atlas.invariants.linear_momentum_certified
        ),
        angular_momentum_certified=bool(
            ordinary_entry_angular.certified
            and ks_competing_atlas.invariants.angular_momentum_certified
        ),
        energy_certified=bool(
            ordinary_entry_energy.certified
            and ks_competing_atlas.invariants.energy_certified
        ),
        certified_chart_count=int(ordinary_entry_chart.invariants_certified)
        + ks_competing_atlas.invariants.certified_chart_count,
        expected_chart_count=len(charts),
    )
    residual_budget = NewtonResidualLedger(
        certified=bool(
            ordinary_entry_residual.certified
            and ks_competing_atlas.residual_budget.certified
        ),
        certified_chart_count=int(ordinary_entry_residual.certified)
        + ks_competing_atlas.residual_budget.certified_chart_count,
        expected_chart_count=len(charts),
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=float(
            ordinary_entry_tail.tail_bound
            + ks_competing_atlas.tail_budget.local_tail_bound
        ),
        max_step_tail_bound=float(
            max(
                ordinary_entry_tail.tail_bound,
                ks_competing_atlas.tail_budget.max_step_tail_bound,
            )
        ),
        certified=bool(
            ordinary_entry_tail.is_nontrivial
            and ks_competing_atlas.tail_budget.certified
        ),
    )
    target_time_interval = (
        FloatInterval.point(float(target_time))
        if target_time is not None
        else entry_time_interval + ks_competing_atlas.evaluation.target_time_interval
    )
    has_ordinary_post_handoff = any(
        chart.chart_type == "spatial_ordinary_taylor_after_ks"
        for chart in ks_competing_atlas.charts
    )
    has_event_order_branch_union = any(
        chart.chart_type == "spatial_ks_event_order_branch_union"
        for chart in ks_competing_atlas.charts
    )
    has_close_pair_branch_union = any(
        chart.chart_type == "spatial_branch_union"
        for chart in ks_competing_atlas.charts
    )
    has_branch_union_suffix = bool(
        has_event_order_branch_union or has_close_pair_branch_union
    )
    ordinary_first_collision_policy = _certify_spatial_local_collision_policy(
        pair=pair,
        masses=masses,
        ordinary_entry_solution=ordinary_entry_solution,
        ordinary_entry_domain=ordinary_entry_domain,
        ordinary_entry_tail_bound=ordinary_entry_tail.tail_bound,
        ks_solution=ks_competing_atlas.evaluation.first_ks_solution,
        ks_parameter_interval=ks_competing_atlas.charts[0].parameter_interval,
        ks_tail_bound=ks_competing_atlas.charts[0].tail_bound,
        ordinary_post_solution=None,
        ordinary_post_parameter_interval=None,
        ordinary_post_tail_bound=0.0,
        retained_order=retained_order,
    )
    collision_policy_certified = bool(
        ordinary_first_collision_policy.certified
        and ks_competing_atlas.evaluation.first_collision_policy_certificate is not None
        and ks_competing_atlas.evaluation.first_collision_policy_certificate.certified
        and (
            (
                has_branch_union_suffix
                and ks_competing_atlas.collision_policy.certified
            )
            or (
                getattr(
                    ks_competing_atlas.evaluation,
                    "next_collision_policy_certificate",
                    None,
                )
                is not None
                and ks_competing_atlas.evaluation.next_collision_policy_certificate.certified
            )
        )
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="local_spatial_ordinary_ks_to_competing_ks_handoff",
        binary_policy="spatial_ordinary_entry_then_competing_ks_regularization",
        total_collision_policy=(
            "finite_time_local_ordinary_ks_prefix_then_branch_union"
            if has_branch_union_suffix and collision_policy_certified
            else
            "finite_time_local_ordinary_ks_to_competing_ks_nonregularized_pairs_separated"
            if collision_policy_certified
            else "not_classified_by_local_ordinary_ks_to_competing_ks_handoff"
        ),
        triple_collision_status=(
            "locally_excluded_on_ordinary_ks_prefix_and_branch_union"
            if has_branch_union_suffix and collision_policy_certified
            else
            "locally_excluded_on_ordinary_and_ks_charts"
            if collision_policy_certified
            else "not_classified_by_local_ordinary_ks_to_competing_ks_handoff"
        ),
        triple_collision_reason=(
            "ordinary entry, KS prefix, and branch-union suffix certify local collision policy"
            if has_branch_union_suffix and collision_policy_certified
            else
            "ordinary and both KS chart domains keep non-regularized pair distances positive"
            if collision_policy_certified
            else "local ordinary/KS-to-KS handoff has not certified all competing-event separation"
        ),
        certified=collision_policy_certified,
    )
    evaluation = SpatialOrdinaryKSCompetingHandoffEvaluation(
        ordinary_entry_solution=ordinary_entry_solution,
        entry_event_certificate=entry_event_certificate,
        entry_ks_state=entry_ks_state,
        entry_projection=entry_projection,
        ks_competing_evaluation=ks_competing_atlas.evaluation,
        target_state_interval=ks_competing_atlas.target_state_interval,
        target_time_interval=target_time_interval,
        collision_policy_certificate=ordinary_first_collision_policy,
    )
    proof_entries: list[ProofLedgerEntry] = [
        ProofLedgerEntry("mass_domain", _masses_positive_finite(masses), source),
        ProofLedgerEntry(
            "initial_state_domain",
            _interval_array_finite_nonempty(_flat_float_interval_array(state_interval)),
            source,
        ),
        ProofLedgerEntry(
            "spatial_ordinary_entry_residuals",
            ordinary_entry_residual.certified,
            source,
        ),
        ProofLedgerEntry(
            "spatial_ordinary_ks_entry_event_isolation",
            entry_event_certificate.certified,
            source,
        ),
        ProofLedgerEntry(
            "spatial_ordinary_to_ks_branch_lift",
            bool(entry_ks_state.certified and entry_projection.certified),
            source,
        ),
        ProofLedgerEntry(
            "spatial_ks_equation_residuals",
            ks_competing_atlas.residual_budget.certified,
            source,
        ),
        ProofLedgerEntry(
            "spatial_ks_projection_constraints",
            all(chart.projection_certified for chart in ks_competing_atlas.charts),
            source,
        ),
        ProofLedgerEntry(
            "spatial_ks_competing_entry_event_isolation",
            ks_competing_atlas.evaluation.competing_entry_event_certificate.certified,
            source,
        ),
        ProofLedgerEntry(
            "spatial_ks_to_ks_branch_lift",
            ks_competing_atlas.evaluation.next_ks_state.certified,
            source,
        ),
        ProofLedgerEntry(
            "spatial_ks_to_ks_transition",
            any(
                transition.transition_type == "spatial_ks_to_ks_competing_binary_entry"
                and transition.certified
                for transition in transitions
            ),
            source,
        ),
        ProofLedgerEntry(
            "spatial_ordinary_ks_and_post_invariants",
            invariants.certified,
            source,
        ),
        ProofLedgerEntry(
            "local_tail_budget",
            tail_budget.certified and tail_budget.finite,
            source,
        ),
        ProofLedgerEntry("ordinary_to_ks_transition", transitions[0].certified, source),
    ]
    if has_ordinary_post_handoff:
        proof_entries.extend(
            [
                ProofLedgerEntry(
                    "spatial_ks_rho_positive_endpoint_projection",
                    ks_competing_atlas.evaluation.endpoint_projection.certified,
                    source,
                ),
                ProofLedgerEntry(
                    "ordinary_handoff_admissibility",
                    any(
                        entry.name == "ordinary_handoff_admissibility" and entry.certified
                        for entry in ks_competing_atlas.proof_ledger.entries
                    ),
                    source,
                ),
                ProofLedgerEntry(
                    "ordinary_post_handoff_residuals",
                    ks_competing_atlas.residual_budget.certified,
                    source,
                ),
                ProofLedgerEntry(
                    "rho_positive_handoff_transition",
                    any(
                        "rho_positive" in transition.transition_type and transition.certified
                        for transition in transitions
                    ),
                    source,
                ),
            ]
        )
    elif has_branch_union_suffix:
        transition_entry_name = (
            "spatial_ks_prefix_event_order_branch_union_transition"
            if has_event_order_branch_union
            else "spatial_ks_prefix_close_pair_branch_union_transition"
        )
        proof_entries.extend(
            [
                ProofLedgerEntry(
                    transition_entry_name,
                    bool(
                        _proof_ledger_has_certified_entry(
                            ks_competing_atlas.proof_ledger,
                            transition_entry_name,
                        )
                        and _physical_time_chain_progress_certified(
                            0.5
                            * (
                                target_time_interval.lower
                                + target_time_interval.upper
                            ),
                            charts,
                        )
                    ),
                    source,
                ),
            ]
        )
        if has_event_order_branch_union:
            proof_entries.extend(
                [
                    ProofLedgerEntry(
                        "ks_event_order_partition",
                        _proof_ledger_has_certified_entry(
                            ks_competing_atlas.proof_ledger,
                            "ks_event_order_partition",
                        ),
                        source,
                    ),
                    ProofLedgerEntry(
                        "finite_time_event_order_branch_union_consumption",
                        _proof_ledger_has_certified_entry(
                            ks_competing_atlas.proof_ledger,
                            "finite_time_event_order_branch_union_consumption",
                        ),
                        source,
                    ),
                ]
            )
        if has_close_pair_branch_union:
            proof_entries.extend(
                [
                    ProofLedgerEntry(
                        "simultaneous_close_pair_partition",
                        _proof_ledger_has_certified_entry(
                            ks_competing_atlas.proof_ledger,
                            "simultaneous_close_pair_partition",
                        ),
                        source,
                    ),
                    ProofLedgerEntry(
                        "finite_time_branch_union_consumption",
                        _proof_ledger_has_certified_entry(
                            ks_competing_atlas.proof_ledger,
                            "finite_time_branch_union_consumption",
                        ),
                        source,
                    ),
                ]
            )
    else:
        proof_entries.extend(
            [
                ProofLedgerEntry(
                    "spatial_ks_target_time_projection",
                    ks_competing_atlas.evaluation.endpoint_projection.certified,
                    source,
                ),
                ProofLedgerEntry(
                    "spatial_ks_target_inside_regularized_chart",
                    True,
                    source,
                    detail="requested target is evaluated before ordinary handoff from the competing KS chart",
                ),
            ]
        )
    proof_entries.extend(
        [
            ProofLedgerEntry(
                "finite_time_physical_targeting",
                target_time is not None,
                source,
            ),
            ProofLedgerEntry(
                "target_time_domain",
                _target_time_in_final_chart_domain(
                    0.5 * (target_time_interval.lower + target_time_interval.upper),
                    charts,
                ),
                source,
            ),
            ProofLedgerEntry(
                "spatial_collision_policy_scope",
                collision_policy.certified,
                source,
                detail=(
                    "ordinary/KS-to-KS domains keep all non-regularized pairs separated"
                    if collision_policy.certified
                    else "; ".join(
                        ordinary_first_collision_policy.missing_obligations
                        + tuple(
                            getattr(
                                ks_competing_atlas.evaluation.first_collision_policy_certificate,
                                "missing_obligations",
                                (),
                            )
                        )
                        + tuple(
                            getattr(
                                getattr(
                                    ks_competing_atlas.evaluation,
                                    "next_collision_policy_certificate",
                                    None,
                                ),
                                "missing_obligations",
                                (),
                            )
                        )
                    )
                ),
            ),
        ]
    )
    return ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in masses.reshape(-1)),
        target_time=0.5 * (target_time_interval.lower + target_time_interval.upper),
        initial_state_interval=_flat_float_interval_array(state_interval),
        charts=charts,
        transitions=transitions,
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=ProofLedger(entries=tuple(proof_entries)),
        evaluation=evaluation,
    )


def validated_atlas_from_spatial_ordinary_ks_suffix_atlas(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    *,
    pair: tuple[int, int],
    enter_distance: float,
    entry_time_upper: float,
    branch: str,
    ks_suffix_atlas: ValidatedAtlasSolution,
    target_time: float,
    retained_order: int,
    guard_order: int,
    source: str = "spatial_ordinary_ks_suffix_atlas",
) -> ValidatedAtlasSolution:
    """Prepend a certified ordinary-to-KS entry chart to a certified KS suffix."""

    retained_order = int(retained_order)
    guard_order = int(guard_order)
    if retained_order < 1:
        raise ValueError("retained_order must be positive")
    if guard_order < 1:
        raise ValueError("guard_order must be positive")
    if not getattr(ks_suffix_atlas, "proof_certified", False):
        raise ValueError("KS suffix atlas must be proof-certified")

    masses = np.asarray(masses, dtype=float)
    pair = tuple(pair)
    computed_order = retained_order + guard_order
    ordinary_positions, ordinary_velocities = _spatial_interval_state_arrays(state_interval)
    ordinary_entry_solution = construct_interval_taylor_solution_from_intervals(
        ordinary_positions,
        ordinary_velocities,
        masses,
        order=computed_order,
    )
    entry_event_certificate = certify_spatial_ordinary_ks_entry_event(
        ordinary_entry_solution,
        pair=pair,
        enter_distance=enter_distance,
        time_upper=entry_time_upper,
        coefficient_count=retained_order,
    )
    if not entry_event_certificate.certified:
        raise ValueError("ordinary-to-KS entry event could not be interval-certified")
    entry_time_interval = FloatInterval(*entry_event_certificate.root_interval)
    entry_ks_state = spatial_ordinary_entry_event_to_ks_chart_state(
        ordinary_entry_solution,
        entry_event_certificate,
        branch=branch,
    )
    if not entry_ks_state.certified:
        raise ValueError("ordinary-to-KS entry event did not lift to a certified KS branch")
    entry_projection = project_spatial_ks_binary_interval_chart_state_to_physical(
        entry_ks_state,
        physical_time=entry_time_interval,
    )
    if not entry_projection.certified:
        raise ValueError("ordinary-to-KS branch lift did not project to a rho-positive state")
    if ks_suffix_atlas.initial_state_interval is not None and not _interval_array_subset(
        _flat_float_interval_array(entry_projection.state_interval),
        ks_suffix_atlas.initial_state_interval,
    ):
        raise ValueError("KS suffix atlas initial interval does not contain the ordinary-entry projection")

    ordinary_entry_residual = certify_ordinary_interval_taylor_equations(
        ordinary_entry_solution,
        coefficient_count=retained_order,
    )
    ordinary_time_series = _ordinary_physical_time_series(computed_order)
    ordinary_entry_center = certify_interval_center_of_mass_motion(
        ordinary_entry_solution.position,
        ordinary_entry_solution.velocity,
        ordinary_time_series,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_momentum = certify_interval_linear_momentum_conservation(
        ordinary_entry_solution.velocity,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_angular = certify_interval_centered_angular_momentum_conservation(
        ordinary_entry_solution.position,
        ordinary_entry_solution.velocity,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_energy = certify_interval_total_energy_conservation(
        ordinary_entry_solution.position,
        ordinary_entry_solution.velocity,
        masses,
        coefficient_count=retained_order,
    )
    ordinary_entry_tail = interval_guarded_tail_certificate(
        ordinary_interval_solution_arrays(ordinary_entry_solution),
        retained_order,
        entry_time_interval.upper,
    )
    ordinary_entry_domain = FloatInterval(0.0, entry_time_interval.upper)
    ordinary_entry_lowers = _ordinary_pair_squared_distance_lowers(
        ordinary_entry_solution,
        ordinary_entry_domain,
        retained_order=retained_order,
        tail_bound=ordinary_entry_tail.tail_bound,
    )
    ordinary_entry_collision_certified = _all_squared_distance_lowers_positive(
        ordinary_entry_lowers,
    )
    ordinary_entry_invariants_certified = bool(
        ordinary_entry_center.certified
        and ordinary_entry_momentum.certified
        and ordinary_entry_angular.certified
        and ordinary_entry_energy.certified
    )
    ordinary_entry_chart = ValidatedChart(
        chart_id="ordinary_before_spatial_ks_0",
        chart_type="spatial_ordinary_taylor_before_ks",
        source=source,
        parameter_name="t",
        parameter_interval=ordinary_entry_domain,
        physical_time_interval=ordinary_entry_domain,
        dynamics_certified=ordinary_entry_residual.certified,
        residual_certified=ordinary_entry_residual.certified,
        projection_certified=True,
        invariants_certified=ordinary_entry_invariants_certified,
        tail_certified=ordinary_entry_tail.is_nontrivial,
        tail_bound=ordinary_entry_tail.tail_bound,
    )
    shifted_suffix_charts = tuple(
        replace(
            chart,
            chart_id=(
                f"spatial_ks_{index + 1}"
                if chart.chart_type == "spatial_ks_binary"
                else f"{chart.chart_id}_after_ordinary_ks_{index + 1}"
            ),
            physical_time_interval=_shift_optional_interval(
                chart.physical_time_interval,
                entry_time_interval,
            ),
        )
        for index, chart in enumerate(ks_suffix_atlas.charts)
    )
    charts = (ordinary_entry_chart, *shifted_suffix_charts)
    transitions = (
        ValidatedTransition(
            source_chart_id=charts[0].chart_id,
            target_chart_id=charts[1].chart_id,
            transition_type="spatial_ordinary_to_ks_decreasing_distance_entry",
            certified=bool(
                entry_event_certificate.certified
                and entry_ks_state.certified
                and entry_projection.certified
            ),
            source=source,
        ),
        *(
            replace(
                transition,
                source_chart_id=charts[index + 1].chart_id,
                target_chart_id=charts[index + 2].chart_id,
            )
            for index, transition in enumerate(ks_suffix_atlas.transitions)
        ),
    )
    invariants = GlobalInvariantLedger(
        center_of_mass_certified=bool(
            ordinary_entry_center.certified
            and ks_suffix_atlas.invariants.center_of_mass_certified
        ),
        linear_momentum_certified=bool(
            ordinary_entry_momentum.certified
            and ks_suffix_atlas.invariants.linear_momentum_certified
        ),
        angular_momentum_certified=bool(
            ordinary_entry_angular.certified
            and ks_suffix_atlas.invariants.angular_momentum_certified
        ),
        energy_certified=bool(
            ordinary_entry_energy.certified and ks_suffix_atlas.invariants.energy_certified
        ),
        certified_chart_count=int(ordinary_entry_chart.invariants_certified)
        + ks_suffix_atlas.invariants.certified_chart_count,
        expected_chart_count=len(charts),
    )
    residual_budget = NewtonResidualLedger(
        certified=bool(
            ordinary_entry_residual.certified and ks_suffix_atlas.residual_budget.certified
        ),
        certified_chart_count=int(ordinary_entry_residual.certified)
        + ks_suffix_atlas.residual_budget.certified_chart_count,
        expected_chart_count=len(charts),
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=float(
            ordinary_entry_tail.tail_bound + ks_suffix_atlas.tail_budget.local_tail_bound
        ),
        max_step_tail_bound=float(
            max(
                ordinary_entry_tail.tail_bound,
                ks_suffix_atlas.tail_budget.max_step_tail_bound,
            )
        ),
        certified=bool(
            ordinary_entry_tail.is_nontrivial and ks_suffix_atlas.tail_budget.certified
        ),
    )
    collision_policy_certified = bool(
        ordinary_entry_collision_certified and ks_suffix_atlas.collision_policy.certified
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="local_spatial_ordinary_ks_prefix_to_suffix_atlas",
        binary_policy="spatial_ordinary_entry_then_constructor_certified_ks_suffix",
        total_collision_policy=(
            "finite_time_local_ordinary_ks_prefix_then_suffix_atlas"
            if collision_policy_certified
            else "not_classified_by_local_ordinary_ks_suffix_atlas"
        ),
        triple_collision_status=(
            "locally_excluded_on_ordinary_prefix_and_ks_suffix"
            if collision_policy_certified
            else "not_classified_by_local_ordinary_ks_suffix_atlas"
        ),
        triple_collision_reason=(
            "ordinary entry chart avoids collisions and the KS suffix atlas "
            "carries its own constructor-certified collision policy"
            if collision_policy_certified
            else "ordinary entry chart or KS suffix atlas has not certified local collision policy"
        ),
        certified=collision_policy_certified,
    )
    target_time = float(target_time)
    target_union = ks_suffix_atlas.target_state_interval_union
    evaluation = SpatialOrdinaryKSSuffixAtlasEvaluation(
        ordinary_entry_solution=ordinary_entry_solution,
        entry_event_certificate=entry_event_certificate,
        entry_ks_state=entry_ks_state,
        entry_projection=entry_projection,
        ks_suffix_evaluation=ks_suffix_atlas.evaluation,
        target_state_interval=ks_suffix_atlas.target_state_interval,
        target_state_interval_union=target_union,
        target_time_interval=FloatInterval.point(target_time),
        collision_policy_certificate=SpatialLocalCollisionPolicyCertificate(
            pair=pair,
            ordinary_entry_squared_distance_lowers=ordinary_entry_lowers,
            ks_competing_squared_distance_lowers=(),
            ordinary_post_squared_distance_lowers=(),
            competing_pair_min_distance_required=0.0,
            missing_obligations=(
                () if ordinary_entry_collision_certified else ("ordinary_entry_domain_pair_collision_not_excluded",)
            ),
        ),
    )
    proof_entries: list[ProofLedgerEntry] = [
        ProofLedgerEntry("mass_domain", _masses_positive_finite(masses), source),
        ProofLedgerEntry(
            "initial_state_domain",
            _interval_array_finite_nonempty(_flat_float_interval_array(state_interval)),
            source,
        ),
        ProofLedgerEntry(
            "spatial_ordinary_entry_residuals",
            ordinary_entry_residual.certified,
            source,
        ),
        ProofLedgerEntry(
            "spatial_ordinary_ks_entry_event_isolation",
            entry_event_certificate.certified,
            source,
        ),
        ProofLedgerEntry(
            "spatial_ordinary_to_ks_branch_lift",
            bool(entry_ks_state.certified and entry_projection.certified),
            source,
        ),
        ProofLedgerEntry("ordinary_to_ks_transition", transitions[0].certified, source),
        ProofLedgerEntry("newton_residuals", residual_budget.certified, source),
        ProofLedgerEntry("invariant_ledger", invariants.certified, source),
        ProofLedgerEntry(
            "local_tail_budget",
            tail_budget.certified and tail_budget.finite,
            source,
        ),
        ProofLedgerEntry(
            "finite_time_physical_targeting",
            _target_time_in_final_chart_domain(target_time, charts),
            source,
        ),
        ProofLedgerEntry(
            "target_time_domain",
            _target_time_in_final_chart_domain(target_time, charts),
            source,
        ),
        ProofLedgerEntry(
            "spatial_collision_policy_scope",
            collision_policy.certified,
            source,
        ),
    ]
    for name in (
        "spatial_ks_prefix_event_order_branch_union_transition",
        "ks_event_order_partition",
        "finite_time_event_order_branch_union_consumption",
        "spatial_ks_prefix_close_pair_branch_union_transition",
        "simultaneous_close_pair_partition",
        "finite_time_branch_union_consumption",
    ):
        if _proof_ledger_has_certified_entry(ks_suffix_atlas.proof_ledger, name):
            proof_entries.append(ProofLedgerEntry(name, True, source))
    return ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in masses.reshape(-1)),
        target_time=target_time,
        initial_state_interval=_flat_float_interval_array(state_interval),
        charts=charts,
        transitions=transitions,
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=ProofLedger(entries=tuple(proof_entries)),
        evaluation=evaluation,
    )


def validated_atlas_from_spatial_ks_competing_binary_handoff(
    initial_state: IntervalSpatialKSBinaryChartState,
    *,
    competing_pair: tuple[int, int],
    enter_distance: float,
    entry_s_upper: float,
    branch: str,
    next_s_endpoint: float | None = None,
    next_exit_rho: float | None = None,
    next_s_upper: float | None = None,
    ordinary_step_size: float | None = None,
    target_time_after_ks_start_interval: FloatInterval | tuple[float, float] | None = None,
    retained_order: int,
    guard_order: int,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    competing_pair_min_distance_required: float = 0.0,
    max_competing_repeats: int = 0,
    next_atlas_override: ValidatedAtlasSolution | None = None,
    source: str = "spatial_ks_competing_binary_handoff",
) -> ValidatedAtlasSolution:
    """Compose a selected KS chart into a competing-pair KS chart.

    This is a local repeat-step constructor.  It certifies the first
    competing-binary entry event and the lift into the next KS branch, then
    delegates the next local segment to :func:`validated_atlas_from_spatial_ks_binary_chart`.
    It deliberately does not claim arbitrary multi-event spatial continuation.
    """

    retained_order = int(retained_order)
    guard_order = int(guard_order)
    if retained_order < 1:
        raise ValueError("retained_order must be positive")
    if guard_order < 1:
        raise ValueError("guard_order must be positive")
    competing_pair_min_distance_required = float(competing_pair_min_distance_required)
    max_competing_repeats = int(max_competing_repeats)
    if max_competing_repeats < 0:
        raise ValueError("max_competing_repeats cannot be negative")
    if target_time_after_ks_start_interval is not None:
        target_time_after_ks_start_interval = _coerce_float_interval(
            target_time_after_ks_start_interval
        )

    computed_order = retained_order + guard_order
    first_ks_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        initial_state,
        order=computed_order,
    )
    initial_projection = project_spatial_ks_binary_interval_chart_state_to_physical(
        initial_state,
        physical_time=FloatInterval.point(0.0),
    )
    initial_state_interval = (
        _flat_float_interval_array(initial_projection.state_interval)
        if initial_projection.certified
        else None
    )
    competing_entry_event_certificate = certify_spatial_ks_competing_binary_entry_event(
        first_ks_solution,
        pair=tuple(competing_pair),
        enter_distance=enter_distance,
        s_upper=entry_s_upper,
        coefficient_count=retained_order,
    )
    if not competing_entry_event_certificate.certified:
        raise ValueError("competing KS entry event could not be interval-certified")

    entry_s_interval = FloatInterval(*competing_entry_event_certificate.root_interval)
    entry_time_interval = interval_array_series_eval(
        first_ks_solution.physical_time[:, None],
        entry_s_interval,
    )[0]
    next_ks_state = spatial_ks_competing_entry_event_to_ks_chart_state(
        first_ks_solution,
        competing_entry_event_certificate,
        branch=branch,
    )
    if not next_ks_state.certified:
        raise ValueError("competing KS event did not lift to a certified next KS branch")

    next_target_interval = None
    if target_time_after_ks_start_interval is not None:
        next_target_interval = target_time_after_ks_start_interval - entry_time_interval
        if next_target_interval.upper < -1.0e-14:
            raise ValueError("target time occurs before the competing KS entry event")
        if next_target_interval.lower < 0.0:
            next_target_interval = FloatInterval(0.0, next_target_interval.upper)

    next_atlas = None
    next_atlas_error: Exception | None = None
    if next_atlas_override is not None:
        _validate_spatial_ks_competing_next_atlas_override(
            next_atlas_override,
            next_ks_state=next_ks_state,
            next_target_interval=next_target_interval,
        )
        next_atlas = next_atlas_override
    else:
        try:
            next_atlas = validated_atlas_from_spatial_ks_binary_chart(
                next_ks_state,
                s_endpoint=next_s_endpoint,
                exit_rho=next_exit_rho,
                s_upper=next_s_upper,
                ordinary_step_size=ordinary_step_size,
                target_time_after_ks_start_interval=next_target_interval,
                retained_order=retained_order,
                guard_order=guard_order,
                ordinary_handoff_min_pair_distance_required=ordinary_handoff_min_pair_distance_required,
                ordinary_handoff_max_acceleration_bound=ordinary_handoff_max_acceleration_bound,
                ordinary_handoff_min_cauchy_radius=ordinary_handoff_min_cauchy_radius,
                ordinary_handoff_max_residual_bound=ordinary_handoff_max_residual_bound,
                ordinary_handoff_max_tail_bound=ordinary_handoff_max_tail_bound,
                source=source,
            )
        except (RuntimeError, ValueError) as error:
            next_atlas_error = error

    should_try_close_pair_branch_union = next_atlas is None
    if (
        next_atlas is not None
        and competing_pair_min_distance_required > 0.0
        and next_atlas_override is None
    ):
        tentative_next_has_competing_handoff = any(
            "ks_to_ks_competing" in transition.transition_type
            for transition in next_atlas.transitions
        )
        tentative_next_is_branch_union = any(
            chart.chart_type
            in {"spatial_ks_event_order_branch_union", "spatial_branch_union"}
            for chart in next_atlas.charts
        )
        tentative_has_ordinary_post_handoff = bool(
            len(next_atlas.charts) > 1
            and not tentative_next_has_competing_handoff
            and not tentative_next_is_branch_union
            and getattr(next_atlas.evaluation, "ordinary_solution", None) is not None
        )
        tentative_next_first_ks_solution = getattr(
            next_atlas.evaluation,
            "ks_solution",
            None,
        )
        if tentative_next_first_ks_solution is None:
            tentative_next_first_ks_solution = getattr(
                next_atlas.evaluation,
                "first_ks_solution",
                None,
            )
        if not tentative_next_has_competing_handoff and not tentative_next_is_branch_union:
            tentative_next_collision_policy = _certify_spatial_local_collision_policy(
                pair=next_ks_state.pair,
                masses=initial_state.masses,
                ordinary_entry_solution=None,
                ordinary_entry_domain=None,
                ordinary_entry_tail_bound=0.0,
                ks_solution=tentative_next_first_ks_solution,
                ks_parameter_interval=next_atlas.charts[0].parameter_interval,
                ks_tail_bound=next_atlas.charts[0].tail_bound,
                ordinary_post_solution=(
                    next_atlas.evaluation.ordinary_solution
                    if tentative_has_ordinary_post_handoff
                    else None
                ),
                ordinary_post_parameter_interval=(
                    next_atlas.charts[1].parameter_interval
                    if tentative_has_ordinary_post_handoff
                    else None
                ),
                ordinary_post_tail_bound=(
                    next_atlas.charts[1].tail_bound
                    if tentative_has_ordinary_post_handoff
                    else 0.0
                ),
                retained_order=retained_order,
                require_ordinary_entry=False,
                competing_pair_min_distance_required=(
                    competing_pair_min_distance_required
                ),
            )
            should_try_close_pair_branch_union = bool(
                not tentative_next_collision_policy.certified
                and "ks_competing_close_binary_requires_next_regularized_chart_or_split"
                in tentative_next_collision_policy.missing_obligations
            )

    if (
        should_try_close_pair_branch_union
        and next_target_interval is not None
        and next_target_interval.upper > 0.0
        and competing_pair_min_distance_required > 0.0
        and next_atlas_override is None
    ):
        try:
            next_projection = project_spatial_ks_binary_interval_chart_state_to_physical(
                next_ks_state,
                physical_time=FloatInterval.point(0.0),
            )
            if next_projection.certified:
                next_partition_state = tuple(
                    (float(lower), float(upper))
                    for lower, upper in next_projection.state_interval
                )
                next_branch_partition = certify_simultaneous_close_pair_partition(
                    next_partition_state,
                    binary_distance_threshold=competing_pair_min_distance_required,
                    max_depth=6,
                    max_branches=64,
                    masses=initial_state.masses,
                )
                if next_branch_partition.certified:
                    branch_s_endpoint = float(
                        next_s_endpoint
                        if next_s_endpoint is not None
                        else (
                            next_s_upper
                            if next_s_upper is not None
                            else entry_s_upper
                        )
                    )
                    close_pair_atlas = validated_atlas_from_spatial_close_pair_branch_partition(
                        next_branch_partition,
                        next_ks_state.masses,
                        target_time=next_target_interval,
                        s_endpoint=branch_s_endpoint,
                        retained_order=retained_order,
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
                        competing_pair_min_distance_required=competing_pair_min_distance_required,
                        source=f"{source}:next_close_pair_branch_union",
                    )
                    if close_pair_atlas.proof_certified:
                        next_atlas = close_pair_atlas
        except (RuntimeError, ValueError, TypeError):
            pass

    should_try_competing_repeat = bool(
        max_competing_repeats > 0
        and target_time_after_ks_start_interval is not None
        and next_target_interval is not None
        and competing_pair_min_distance_required > 0.0
        and next_atlas_override is None
        and (
            next_atlas is None
            or not next_atlas.proof_certified
        )
    )
    if should_try_competing_repeat:
        repeat_candidates = _derive_spatial_ks_competing_handoff_candidates_from_ks_state(
            next_ks_state,
            target_time_after_ks_start=next_target_interval.upper,
            binary_distance_threshold=competing_pair_min_distance_required,
            s_upper=float(next_s_upper if next_s_upper is not None else entry_s_upper),
            retained_order=retained_order,
            guard_order=guard_order,
        )
        for (
            _repeat_time_estimate,
            repeat_pair,
            repeat_branch,
            repeat_enter_distance,
            repeat_entry_s_upper,
            repeat_next_s_endpoint,
        ) in repeat_candidates:
            try:
                repeat_atlas = validated_atlas_from_spatial_ks_competing_binary_handoff(
                    next_ks_state,
                    competing_pair=repeat_pair,
                    enter_distance=repeat_enter_distance,
                    entry_s_upper=repeat_entry_s_upper,
                    branch=repeat_branch,
                    next_s_endpoint=repeat_next_s_endpoint,
                    target_time_after_ks_start_interval=next_target_interval,
                    retained_order=retained_order,
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
                    competing_pair_min_distance_required=(
                        competing_pair_min_distance_required
                    ),
                    max_competing_repeats=max_competing_repeats - 1,
                    source=source,
                )
            except (RuntimeError, ValueError):
                continue
            if repeat_atlas.proof_certified:
                next_atlas = repeat_atlas
                break
    if next_atlas is None:
        if next_atlas_error is not None:
            raise next_atlas_error
        raise ValueError("next spatial KS chart could not be certified")

    first_ks_residual = certify_spatial_ks_binary_interval_taylor_equations(
        first_ks_solution,
        coefficient_count=retained_order,
    )
    first_ks_horizontal = certify_spatial_ks_binary_horizontal_constraint(
        first_ks_solution,
        coefficient_count=retained_order,
    )
    first_ks_pair_energy = certify_spatial_ks_binary_pair_energy_constraint(
        first_ks_solution,
        coefficient_count=retained_order,
    )
    first_ks_center = certify_spatial_ks_binary_center_of_mass_motion(
        first_ks_solution,
        coefficient_count=retained_order,
    )
    first_ks_momentum = certify_spatial_ks_binary_linear_momentum_conservation(
        first_ks_solution,
        coefficient_count=retained_order,
    )
    first_ks_angular = certify_spatial_ks_binary_centered_angular_momentum_conservation(
        first_ks_solution,
        coefficient_count=retained_order,
    )
    first_ks_energy = certify_spatial_ks_binary_total_energy_conservation(
        first_ks_solution,
        coefficient_count=retained_order,
    )
    first_ks_tail = _certify_spatial_ks_segment_tail(
        initial_state,
        ks_solution=first_ks_solution,
        retained_order=retained_order,
        guard_order=guard_order,
        step_size=float(competing_entry_event_certificate.root),
    )
    first_ks_invariants_certified = bool(
        first_ks_center.certified
        and first_ks_momentum.certified
        and first_ks_angular.certified
        and first_ks_energy.certified
    )

    first_chart = ValidatedChart(
        chart_id="spatial_ks_0",
        chart_type="spatial_ks_binary",
        source=source,
        parameter_name="s",
        parameter_interval=FloatInterval(0.0, float(competing_entry_event_certificate.root)),
        physical_time_interval=FloatInterval(
            min(0.0, entry_time_interval.lower),
            max(0.0, entry_time_interval.upper),
        ),
        dynamics_certified=bool(initial_state.certified and first_ks_residual.certified),
        residual_certified=first_ks_residual.certified,
        projection_certified=bool(
            first_ks_horizontal.certified and first_ks_pair_energy.certified
        ),
        invariants_certified=first_ks_invariants_certified,
        tail_certified=first_ks_tail.tail_certified,
        tail_bound=first_ks_tail.tail_bound,
    )
    shifted_next_charts = tuple(
        replace(
            chart,
            chart_id=(
                "spatial_ks_1"
                if index == 0 and chart.chart_type == "spatial_ks_binary"
                else f"{chart.chart_id}_after_competing_{index + 1}"
            ),
            physical_time_interval=_shift_optional_interval(
                chart.physical_time_interval,
                entry_time_interval,
            ),
        )
        for index, chart in enumerate(next_atlas.charts)
    )
    charts = (first_chart, *shifted_next_charts)
    transitions = (
        ValidatedTransition(
            source_chart_id=charts[0].chart_id,
            target_chart_id=charts[1].chart_id,
            transition_type="spatial_ks_to_ks_competing_binary_entry",
            certified=bool(
                competing_entry_event_certificate.certified and next_ks_state.certified
            ),
            source=source,
        ),
        *(
            replace(
                transition,
                source_chart_id=charts[index + 1].chart_id,
                target_chart_id=charts[index + 2].chart_id,
            )
            for index, transition in enumerate(next_atlas.transitions)
        ),
    )
    invariants = GlobalInvariantLedger(
        center_of_mass_certified=bool(
            first_ks_center.certified and next_atlas.invariants.center_of_mass_certified
        ),
        linear_momentum_certified=bool(
            first_ks_momentum.certified and next_atlas.invariants.linear_momentum_certified
        ),
        angular_momentum_certified=bool(
            first_ks_angular.certified and next_atlas.invariants.angular_momentum_certified
        ),
        energy_certified=bool(
            first_ks_energy.certified and next_atlas.invariants.energy_certified
        ),
        certified_chart_count=int(first_chart.invariants_certified)
        + next_atlas.invariants.certified_chart_count,
        expected_chart_count=len(charts),
    )
    residual_budget = NewtonResidualLedger(
        certified=bool(first_ks_residual.certified and next_atlas.residual_budget.certified),
        certified_chart_count=int(first_ks_residual.certified)
        + next_atlas.residual_budget.certified_chart_count,
        expected_chart_count=len(charts),
    )
    tail_budget = TailBudgetLedger(
        local_tail_bound=float(first_ks_tail.tail_bound + next_atlas.tail_budget.local_tail_bound),
        max_step_tail_bound=float(
            max(first_ks_tail.tail_bound, next_atlas.tail_budget.max_step_tail_bound)
        ),
        certified=bool(first_ks_tail.tail_certified and next_atlas.tail_budget.certified),
    )
    next_atlas_target_time_interval = _target_time_interval_from_validated_atlas(
        next_atlas
    )
    target_time_interval = entry_time_interval + next_atlas_target_time_interval
    next_has_competing_handoff = any(
        "ks_to_ks_competing" in transition.transition_type
        for transition in next_atlas.transitions
    )
    next_is_event_order_branch_union = any(
        chart.chart_type == "spatial_ks_event_order_branch_union"
        for chart in next_atlas.charts
    )
    next_is_close_pair_branch_union = any(
        chart.chart_type == "spatial_branch_union"
        for chart in next_atlas.charts
    )
    next_is_branch_union = bool(
        next_is_event_order_branch_union or next_is_close_pair_branch_union
    )
    has_ordinary_post_handoff = bool(
        len(next_atlas.charts) > 1
        and not next_has_competing_handoff
        and not next_is_branch_union
        and getattr(next_atlas.evaluation, "ordinary_solution", None) is not None
    )
    next_first_ks_solution = getattr(next_atlas.evaluation, "ks_solution", None)
    if next_first_ks_solution is None:
        next_first_ks_solution = getattr(next_atlas.evaluation, "first_ks_solution", None)
    first_collision_policy_certificate = _certify_spatial_local_collision_policy(
        pair=initial_state.pair,
        masses=initial_state.masses,
        ordinary_entry_solution=None,
        ordinary_entry_domain=None,
        ordinary_entry_tail_bound=0.0,
        ks_solution=first_ks_solution,
        ks_parameter_interval=first_chart.parameter_interval,
        ks_tail_bound=first_chart.tail_bound,
        ordinary_post_solution=None,
        ordinary_post_parameter_interval=None,
        ordinary_post_tail_bound=0.0,
        retained_order=retained_order,
        require_ordinary_entry=False,
    )
    next_collision_policy_certificate = (
        None
        if next_is_branch_union
        else _certify_spatial_local_collision_policy(
            pair=next_ks_state.pair,
            masses=initial_state.masses,
            ordinary_entry_solution=None,
            ordinary_entry_domain=None,
            ordinary_entry_tail_bound=0.0,
            ks_solution=next_first_ks_solution,
            ks_parameter_interval=next_atlas.charts[0].parameter_interval,
            ks_tail_bound=next_atlas.charts[0].tail_bound,
            ordinary_post_solution=(
                next_atlas.evaluation.ordinary_solution if has_ordinary_post_handoff else None
            ),
            ordinary_post_parameter_interval=(
                next_atlas.charts[1].parameter_interval if has_ordinary_post_handoff else None
            ),
            ordinary_post_tail_bound=(
                next_atlas.charts[1].tail_bound if has_ordinary_post_handoff else 0.0
            ),
            retained_order=retained_order,
            require_ordinary_entry=False,
            competing_pair_min_distance_required=(
                0.0 if next_has_competing_handoff else competing_pair_min_distance_required
            ),
        )
    )
    collision_policy_certified = bool(
        first_collision_policy_certificate.certified
        and (
            (
                next_is_branch_union
                and next_atlas.collision_policy.certified
            )
            or (
                next_collision_policy_certificate is not None
                and next_collision_policy_certificate.certified
            )
        )
        and (
            not next_has_competing_handoff
            or getattr(next_atlas.collision_policy, "certified", False)
        )
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="local_spatial_ks_to_competing_ks_handoff",
        binary_policy="spatial_ks_selected_binary_then_competing_binary",
        total_collision_policy=(
            "finite_time_ks_prefix_then_event_order_branch_union"
            if next_is_event_order_branch_union and collision_policy_certified
            else "finite_time_ks_prefix_then_close_pair_branch_union"
            if next_is_close_pair_branch_union and collision_policy_certified
            else (
                "finite_time_local_two_ks_nonregularized_pairs_separated"
                if collision_policy_certified
                else "not_classified_by_two_ks_local_handoff"
            )
        ),
        triple_collision_status=(
            "locally_excluded_by_prefix_and_event_order_branch_union"
            if next_is_event_order_branch_union and collision_policy_certified
            else "locally_excluded_by_prefix_and_close_pair_branch_union"
            if next_is_close_pair_branch_union and collision_policy_certified
            else "not_classified_by_two_ks_local_handoff"
        ),
        triple_collision_reason=(
            "prefix KS chart and event-order branch-union member atlases certify local collision policy"
            if next_is_event_order_branch_union and collision_policy_certified
            else "prefix KS chart and close-pair branch-union member atlases certify local collision policy"
            if next_is_close_pair_branch_union and collision_policy_certified
            else "both local KS chart domains keep non-regularized pair distances positive"
            if collision_policy_certified
            else (
                "local two-KS handoff certifies one competing binary transition but "
                "does not classify all future competing events or total collision"
            )
        ),
        certified=collision_policy_certified,
    )
    if next_is_branch_union:
        evaluation = SpatialKSPrefixEventOrderBranchUnionEvaluation(
            first_ks_solution=first_ks_solution,
            initial_projection=initial_projection,
            competing_entry_event_certificate=competing_entry_event_certificate,
            next_ks_state=next_ks_state,
            branch_union_evaluation=next_atlas.evaluation,
            target_state_interval=next_atlas.target_state_interval,
            target_state_interval_union=(
                next_atlas.target_state_interval_union or ()
            ),
            target_time_interval=target_time_interval,
            first_collision_policy_certificate=first_collision_policy_certificate,
            first_ks_tail_certificate=first_ks_tail.tail_certificate,
        )
    else:
        evaluation = SpatialKSCompetingHandoffEvaluation(
            first_ks_solution=first_ks_solution,
            initial_projection=initial_projection,
            competing_entry_event_certificate=competing_entry_event_certificate,
            next_ks_state=next_ks_state,
            next_ks_evaluation=next_atlas.evaluation,
            target_state_interval=next_atlas.target_state_interval,
            target_time_interval=target_time_interval,
            first_collision_policy_certificate=first_collision_policy_certificate,
            next_collision_policy_certificate=next_collision_policy_certificate,
            first_ks_tail_certificate=first_ks_tail.tail_certificate,
        )
    proof_entries: list[ProofLedgerEntry] = [
        ProofLedgerEntry("mass_domain", _masses_positive_finite(initial_state.masses), source),
        ProofLedgerEntry("spatial_ks_branch_domain", initial_state.certified, source),
        ProofLedgerEntry(
            "initial_state_domain",
            bool(
                initial_projection.certified
                and initial_state_interval is not None
                and _interval_array_finite_nonempty(initial_state_interval)
            ),
            source,
            detail=(
                "initial rho-positive KS state projects to a finite ordinary state box"
                if initial_projection.certified
                else "initial KS state has not been projected across a rho-positive domain"
            ),
        ),
        ProofLedgerEntry("spatial_ks_equation_residuals", residual_budget.certified, source),
        ProofLedgerEntry(
            "spatial_ks_projection_constraints",
            bool(
                first_ks_horizontal.certified
                and first_ks_pair_energy.certified
                and all(chart.projection_certified for chart in next_atlas.charts)
            ),
            source,
        ),
        ProofLedgerEntry(
            "spatial_ks_competing_entry_event_isolation",
            competing_entry_event_certificate.certified,
            source,
        ),
        ProofLedgerEntry(
            "spatial_ks_to_ks_branch_lift",
            next_ks_state.certified,
            source,
        ),
        ProofLedgerEntry(
            "spatial_ks_to_ks_transition",
            transitions[0].certified,
            source,
        ),
        ProofLedgerEntry(
            "spatial_ks_and_ordinary_invariants",
            invariants.certified,
            source,
            detail="first and next spatial KS charts certify invariant ledgers",
        ),
        ProofLedgerEntry(
            "local_tail_budget",
            tail_budget.certified and tail_budget.finite,
            source,
            detail=first_ks_tail.proof_detail,
        ),
    ]
    if has_ordinary_post_handoff:
        proof_entries.extend(
            [
                ProofLedgerEntry(
                    "spatial_ks_rho_positive_endpoint_projection",
                    next_atlas.evaluation.endpoint_projection.certified,
                    source,
                ),
                ProofLedgerEntry(
                    "ordinary_handoff_admissibility",
                    bool(
                        next_atlas.evaluation.ordinary_handoff_admissibility is not None
                        and next_atlas.evaluation.ordinary_handoff_admissibility.certified
                    ),
                    source,
                ),
                ProofLedgerEntry(
                    "ordinary_post_handoff_residuals",
                    next_atlas.residual_budget.certified,
                    source,
                ),
                ProofLedgerEntry(
                    "rho_positive_handoff_transition",
                    bool(len(transitions) > 1 and transitions[1].certified),
                    source,
                ),
            ]
        )
        if next_atlas.evaluation.exit_event_certificate is not None:
            proof_entries.append(
                ProofLedgerEntry(
                    "spatial_ks_exit_event_isolation",
                    next_atlas.evaluation.exit_event_certificate.certified,
                    source,
                )
            )
    elif next_is_branch_union:
        transition_entry_name = (
            "spatial_ks_prefix_event_order_branch_union_transition"
            if next_is_event_order_branch_union
            else "spatial_ks_prefix_close_pair_branch_union_transition"
        )
        transition_detail = (
            "prefix KS competing-entry handoff composes with a "
            "certified event-order branch-union suffix"
            if next_is_event_order_branch_union
            else "prefix KS competing-entry handoff composes with a "
            "certified close-pair branch-union suffix"
        )
        proof_entries.extend(
            [
                ProofLedgerEntry(
                    transition_entry_name,
                    bool(
                        transitions[0].certified
                        and next_ks_state.certified
                        and next_atlas.proof_certified
                        and _physical_time_chain_progress_certified(
                            0.5
                            * (
                                target_time_interval.lower
                                + target_time_interval.upper
                            ),
                            charts,
                        )
                    ),
                    source,
                    detail=transition_detail,
                ),
            ]
        )
        if next_is_event_order_branch_union:
            proof_entries.extend(
                [
                    ProofLedgerEntry(
                        "ks_event_order_partition",
                        _proof_ledger_has_certified_entry(
                            next_atlas.proof_ledger,
                            "ks_event_order_partition",
                        ),
                        f"{source}:next_event_order_branch_union",
                    ),
                    ProofLedgerEntry(
                        "finite_time_event_order_branch_union_consumption",
                        _proof_ledger_has_certified_entry(
                            next_atlas.proof_ledger,
                            "finite_time_event_order_branch_union_consumption",
                        ),
                        f"{source}:next_event_order_branch_union",
                    ),
                ]
            )
        if next_is_close_pair_branch_union:
            proof_entries.extend(
                [
                    ProofLedgerEntry(
                        "simultaneous_close_pair_partition",
                        _proof_ledger_has_certified_entry(
                            next_atlas.proof_ledger,
                            "simultaneous_close_pair_partition",
                        ),
                        f"{source}:next_close_pair_branch_union",
                    ),
                    ProofLedgerEntry(
                        "finite_time_branch_union_consumption",
                        _proof_ledger_has_certified_entry(
                            next_atlas.proof_ledger,
                            "finite_time_branch_union_consumption",
                        ),
                        f"{source}:next_close_pair_branch_union",
                    ),
                ]
            )
    else:
        proof_entries.extend(
            [
                ProofLedgerEntry(
                    "spatial_ks_target_time_projection",
                    next_atlas.evaluation.endpoint_projection.certified,
                    source,
                    detail="requested target is evaluated inside the second regularized KS chart",
                ),
                ProofLedgerEntry(
                    "spatial_ks_target_inside_regularized_chart",
                    True,
                    source,
                    detail="requested target is evaluated before ordinary handoff from the second KS chart",
                ),
            ]
        )
    proof_entries.extend(
        [
            ProofLedgerEntry(
                "finite_time_physical_targeting",
                target_time_after_ks_start_interval is not None,
                source,
                detail=(
                    "requested physical target time is enclosed after the competing KS handoff"
                    if target_time_after_ks_start_interval is not None
                    else "local two-KS handoff has not been selected by a requested physical target"
                ),
            ),
            ProofLedgerEntry(
                "target_time_domain",
                _target_time_in_final_chart_domain(
                    0.5 * (target_time_interval.lower + target_time_interval.upper),
                    charts,
                ),
                source,
            ),
            ProofLedgerEntry(
                "spatial_collision_policy_scope",
                collision_policy.certified,
                source,
                detail=(
                    "two-KS local handoff keeps all non-regularized pairs separated"
                    if collision_policy.certified
                    else (
                        "; ".join(
                            first_collision_policy_certificate.missing_obligations
                            + next_collision_policy_certificate.missing_obligations
                        )
                    )
                ),
            ),
        ]
    )
    return ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in np.asarray(initial_state.masses, dtype=float).reshape(-1)),
        target_time=0.5 * (target_time_interval.lower + target_time_interval.upper),
        initial_state_interval=initial_state_interval,
        charts=charts,
        transitions=transitions,
        invariants=invariants,
        tail_budget=tail_budget,
        residual_budget=residual_budget,
        collision_policy=collision_policy,
        proof_ledger=ProofLedger(entries=tuple(proof_entries)),
        evaluation=evaluation,
    )


def validated_atlas_from_spatial_ks_competing_binary_handoff_to_atlas(
    initial_state: IntervalSpatialKSBinaryChartState,
    *,
    next_atlas: ValidatedAtlasSolution,
    competing_pair: tuple[int, int],
    enter_distance: float,
    entry_s_upper: float,
    branch: str,
    target_time_after_ks_start_interval: FloatInterval | tuple[float, float],
    retained_order: int,
    guard_order: int,
    ordinary_handoff_min_pair_distance_required: float = 0.0,
    ordinary_handoff_max_acceleration_bound: float = np.inf,
    ordinary_handoff_min_cauchy_radius: float = 0.0,
    ordinary_handoff_max_residual_bound: float = np.inf,
    ordinary_handoff_max_tail_bound: float = np.inf,
    competing_pair_min_distance_required: float = 0.0,
    source: str = "spatial_ks_competing_binary_handoff_to_atlas",
) -> ValidatedAtlasSolution:
    """Prepend one certified spatial-KS competing event to an atlas suffix."""

    return validated_atlas_from_spatial_ks_competing_binary_handoff(
        initial_state,
        competing_pair=competing_pair,
        enter_distance=enter_distance,
        entry_s_upper=entry_s_upper,
        branch=branch,
        target_time_after_ks_start_interval=target_time_after_ks_start_interval,
        retained_order=retained_order,
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
        competing_pair_min_distance_required=competing_pair_min_distance_required,
        next_atlas_override=next_atlas,
        source=source,
    )


def _charts_from_reduced_target(
    evaluation: object,
    reduced_target: object | None,
    *,
    source: str,
) -> tuple[ValidatedChart, ...]:
    if reduced_target is None:
        certified = bool(getattr(evaluation, "proof_certified", False))
        return (
            ValidatedChart(
                chart_id="initial",
                chart_type="initial_identity",
                source=source,
                parameter_name="t",
                parameter_interval=FloatInterval.point(float(getattr(evaluation, "target_time", 0.0))),
                physical_time_interval=FloatInterval.point(float(getattr(evaluation, "target_time", 0.0))),
                dynamics_certified=certified,
                residual_certified=certified,
                projection_certified=certified,
                invariants_certified=certified,
                tail_certified=certified,
                tail_bound=0.0,
            ),
        )

    kind, parameter_name = _target_kind_and_parameter(reduced_target)
    charts: list[ValidatedChart] = []
    for index, step in enumerate(getattr(reduced_target, "steps", ())):
        charts.append(_chart_from_step(step, index, kind, parameter_name, source))
    charts.append(_target_chart_from_reduced_target(reduced_target, kind, parameter_name, source))
    return tuple(charts)


def _charts_from_hybrid_solution(
    hybrid_solution: object,
    *,
    source: str,
) -> tuple[ValidatedChart, ...]:
    charts = []
    for index, step in enumerate(getattr(hybrid_solution, "steps", ())):
        chart = str(getattr(step, "chart", "missing"))
        charts.append(
            ValidatedChart(
                chart_id=f"hybrid_{index}",
                chart_type=(
                    "planar_levi_civita_binary"
                    if chart == "binary"
                    else "planar_ordinary_taylor"
                ),
                source=source,
                parameter_name="s" if chart == "binary" else "t",
                parameter_interval=_hybrid_step_parameter_interval(step),
                physical_time_interval=FloatInterval(
                    float(getattr(step, "start_time")),
                    float(getattr(step, "end_time")),
                ),
                dynamics_certified=bool(getattr(step, "proof_certified", False)),
                residual_certified=_step_residual_certified(step),
                projection_certified=_step_projection_certified(step),
                invariants_certified=_step_invariants_certified(step),
                tail_certified=_step_tail_certified(step),
                tail_bound=_step_tail_bound(step),
            )
        )
    return tuple(charts)


def _hybrid_step_parameter_interval(step: object) -> FloatInterval:
    step_size = float(getattr(step, "parameter_step", getattr(step, "physical_step", 0.0)))
    if step_size < 0.0:
        return FloatInterval(step_size, 0.0)
    return FloatInterval(0.0, step_size)


def _transitions_from_hybrid_steps(
    hybrid_solution: object,
    charts: tuple[ValidatedChart, ...],
    *,
    source: str,
) -> tuple[ValidatedTransition, ...]:
    transitions = []
    steps = tuple(getattr(hybrid_solution, "steps", ()))
    states = np.asarray(getattr(hybrid_solution, "states"), dtype=float)
    for index, (left, right) in enumerate(zip(charts, charts[1:])):
        left_step = steps[index]
        right_step = steps[index + 1]
        shared_state = states[index + 1].reshape(-1)
        transitions.append(
            ValidatedTransition(
                source_chart_id=left.chart_id,
                target_chart_id=right.chart_id,
                transition_type=_hybrid_transition_type(left_step, right_step),
                certified=bool(
                    getattr(left_step, "proof_certified", False)
                    and getattr(right_step, "proof_certified", False)
                    and left_step.end_state_interval_contains(shared_state)
                    and right_step.start_state_interval_contains(shared_state)
                    and _hybrid_step_interval_handoff_certified(left_step, right_step)
                ),
                source=source,
            )
        )
    return tuple(transitions)


def _hybrid_step_interval_handoff_certified(left_step: object, right_step: object) -> bool:
    left_end = getattr(left_step, "end_state_interval", None)
    right_start = getattr(right_step, "start_state_interval", None)
    if not _state_interval_tuples_equal(left_end, right_start):
        return False

    left_union = getattr(left_step, "end_state_interval_union", None)
    right_union = getattr(right_step, "start_state_interval_union", None)
    if left_union is None and right_union is None:
        return True
    if left_union is None:
        left_union = (left_end,)
    if right_union is None:
        right_union = (right_start,)
    return _state_interval_union_tuples_equal(left_union, right_union)


def _state_interval_tuples_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return False
    try:
        left_values = tuple(left)
        right_values = tuple(right)
    except TypeError:
        return False
    if len(left_values) != len(right_values) or not left_values:
        return False
    for left_interval, right_interval in zip(left_values, right_values, strict=True):
        try:
            left_lower, left_upper = left_interval
            right_lower, right_upper = right_interval
        except (TypeError, ValueError):
            return False
        if float(left_lower) != float(right_lower) or float(left_upper) != float(right_upper):
            return False
    return True


def _state_interval_union_tuples_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return False
    try:
        left_members = tuple(left)
        right_members = tuple(right)
    except TypeError:
        return False
    if len(left_members) != len(right_members) or not left_members:
        return False
    return all(
        _state_interval_tuples_equal(left_member, right_member)
        for left_member, right_member in zip(left_members, right_members, strict=True)
    )


def _hybrid_transition_type(left_step: object, right_step: object) -> str:
    left_event = getattr(left_step, "event", None)
    left_chart = getattr(left_step, "chart", None)
    right_chart = getattr(right_step, "chart", None)
    if left_event == "enter_binary" and right_chart == "binary":
        return "ordinary_to_binary_event_handoff"
    if left_event == "exit_binary" and left_chart == "binary":
        return "binary_to_ordinary_event_handoff"
    if left_chart == right_chart:
        return f"{left_chart}_continuation_handoff"
    return "hybrid_chart_handoff"


def _hybrid_target_state_interval(hybrid_solution: object) -> Array:
    steps = tuple(getattr(hybrid_solution, "steps", ()))
    if steps and getattr(steps[-1], "end_state_interval", None) is not None:
        return _flat_float_interval_array(steps[-1].end_state_interval)
    return np.asarray(
        [FloatInterval.point(float(value)) for value in np.asarray(hybrid_solution.final_state, dtype=float).reshape(-1)],
        dtype=object,
    )


def _hybrid_target_state_interval_union(
    hybrid_solution: object,
    *,
    target_interval: Array,
    lohner_enclosure: object | None,
    set_enclosure: object | None,
) -> tuple[Array, ...] | None:
    if set_enclosure is not None:
        steps = tuple(getattr(set_enclosure, "steps", ()))
        if steps and getattr(steps[-1], "end_state_interval_union", None):
            return tuple(
                _flat_float_interval_array(state_interval)
                for state_interval in steps[-1].end_state_interval_union
            )
    if lohner_enclosure is not None:
        return (target_interval,)
    steps = tuple(getattr(hybrid_solution, "steps", ()))
    if steps and getattr(steps[-1], "end_state_interval_union", None) is not None:
        return tuple(
            _flat_float_interval_array(state_interval)
            for state_interval in steps[-1].end_state_interval_union
        )
    return (target_interval,)


def _target_time_in_final_chart_domain(
    target_time: float,
    charts: tuple[ValidatedChart, ...],
) -> bool:
    if not charts:
        return False
    interval = charts[-1].physical_time_interval
    if interval is None:
        return False
    try:
        lower, upper = interval.as_tuple()
    except AttributeError:
        lower = getattr(interval, "lower", np.nan)
        upper = getattr(interval, "upper", np.nan)
    target_time = float(target_time)
    lower = float(lower)
    upper = float(upper)
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(target_time),
        abs(lower),
        abs(upper),
    )
    return bool(
        np.isfinite(target_time)
        and np.isfinite(lower)
        and np.isfinite(upper)
        and lower <= upper
        and lower - tolerance <= target_time <= upper + tolerance
    )


def _physical_time_chain_progress_certified(
    target_time: float,
    charts: tuple[ValidatedChart, ...],
    *,
    require_zero_start: bool = False,
) -> bool:
    if not charts:
        return False
    target_time = float(target_time)
    if not np.isfinite(target_time):
        return False
    if not _target_time_in_final_chart_domain(target_time, charts):
        return False
    intervals: list[tuple[float, float]] = []
    for chart in charts:
        interval = getattr(chart, "physical_time_interval", None)
        if not _float_interval_finite_nonempty(interval):
            return False
        lower, upper = _float_interval_bounds(interval)
        intervals.append((lower, upper))
    first_lower, first_upper = intervals[0]
    tolerance = _physical_time_progress_tolerance(
        target_time,
        *(bound for interval in intervals for bound in interval),
    )
    if require_zero_start:
        if not (first_lower <= 0.0 + tolerance and first_upper >= 0.0 - tolerance):
            return False
        start_time = 0.0
    elif first_lower <= target_time <= first_upper:
        return True
    elif target_time >= first_upper:
        start_time = first_lower
    elif target_time <= first_lower:
        start_time = first_upper
    else:
        return False
    if abs(target_time - start_time) <= tolerance:
        return True
    if target_time > start_time:
        frontier = start_time
        for lower, upper in intervals:
            if lower > frontier + tolerance:
                return False
            if upper < frontier - tolerance:
                return False
            frontier = max(frontier, upper)
        return bool(frontier >= target_time - tolerance)
    frontier = start_time
    for lower, upper in intervals:
        if upper < frontier - tolerance:
            return False
        if lower > frontier + tolerance:
            return False
        frontier = min(frontier, lower)
    return bool(frontier <= target_time + tolerance)


def _physical_time_progress_tolerance(*values: float) -> float:
    scale = 1.0
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return np.inf
        if not np.isfinite(value):
            return np.inf
        scale = max(scale, abs(value))
    return float(128.0 * np.finfo(float).eps * scale)


def _float_interval_finite_nonempty(interval: object) -> bool:
    if interval is None:
        return False
    try:
        lower, upper = interval.as_tuple()  # type: ignore[attr-defined]
    except AttributeError:
        lower = getattr(interval, "lower", np.nan)
        upper = getattr(interval, "upper", np.nan)
    try:
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(lower) and np.isfinite(upper) and lower <= upper)


def _float_in_interval_with_tolerance(
    value: float,
    interval: object,
    *,
    tolerance: float | None = None,
) -> bool:
    if not _float_interval_finite_nonempty(interval):
        return False
    try:
        lower, upper = interval.as_tuple()  # type: ignore[attr-defined]
    except AttributeError:
        lower = getattr(interval, "lower", np.nan)
        upper = getattr(interval, "upper", np.nan)
    value = float(value)
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(value):
        return False
    if tolerance is None:
        tolerance = _physical_time_progress_tolerance(value, lower, upper)
    tolerance = float(tolerance)
    return bool(lower - tolerance <= value <= upper + tolerance)


def _float_interval_contains_interval_with_tolerance(
    container: object,
    contained: object,
    *,
    tolerance: float | None = None,
) -> bool:
    if not (
        _float_interval_finite_nonempty(container)
        and _float_interval_finite_nonempty(contained)
    ):
        return False
    try:
        container_lower, container_upper = container.as_tuple()  # type: ignore[attr-defined]
    except AttributeError:
        container_lower = getattr(container, "lower", np.nan)
        container_upper = getattr(container, "upper", np.nan)
    try:
        contained_lower, contained_upper = contained.as_tuple()  # type: ignore[attr-defined]
    except AttributeError:
        contained_lower = getattr(contained, "lower", np.nan)
        contained_upper = getattr(contained, "upper", np.nan)
    container_lower = float(container_lower)
    container_upper = float(container_upper)
    contained_lower = float(contained_lower)
    contained_upper = float(contained_upper)
    if tolerance is None:
        tolerance = _physical_time_progress_tolerance(
            0.5 * (contained_lower + contained_upper),
            container_lower,
            container_upper,
        )
    tolerance = float(tolerance)
    return bool(
        container_lower <= contained_lower + tolerance
        and contained_upper <= container_upper + tolerance
    )


def _hybrid_target_time_matches_endpoint(target_time: float, hybrid_solution: object) -> bool:
    try:
        times = np.asarray(getattr(hybrid_solution, "times"), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    if times.size == 0:
        return False
    endpoint_time = float(times[-1])
    target_time = float(target_time)
    tolerance = 8.0 * np.finfo(float).eps * max(
        1.0,
        abs(target_time),
        abs(endpoint_time),
    )
    return bool(
        np.isfinite(target_time)
        and np.isfinite(endpoint_time)
        and abs(target_time - endpoint_time) <= tolerance
    )


def _hybrid_target_enclosure_for_time(
    hybrid_solution: object,
    *,
    target_time: float,
    lohner_enclosure: object | None,
    set_enclosure: object | None,
) -> HybridTargetEnclosure:
    if _hybrid_target_time_matches_endpoint(target_time, hybrid_solution):
        return _hybrid_endpoint_target_enclosure(
            hybrid_solution,
            lohner_enclosure=lohner_enclosure,
            set_enclosure=set_enclosure,
        )

    step_index = _hybrid_step_index_for_target_time(hybrid_solution, target_time)
    if step_index is None:
        fallback = _hybrid_endpoint_target_enclosure(
            hybrid_solution,
            lohner_enclosure=lohner_enclosure,
            set_enclosure=set_enclosure,
        )
        return replace(
            fallback,
            step_index=None,
            target_time_in_chart=False,
            certified=False,
            detail=f"target_time={target_time!r} is not contained in any hybrid chart",
        )

    try:
        target_interval, target_union, reference = _hybrid_step_target_enclosure(
            hybrid_solution,
            step_index,
            target_time,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        fallback = _hybrid_endpoint_target_enclosure(
            hybrid_solution,
            lohner_enclosure=lohner_enclosure,
            set_enclosure=set_enclosure,
        )
        return replace(
            fallback,
            step_index=step_index,
            target_time_in_chart=True,
            certified=False,
            detail=f"hybrid chart target enclosure failed: {exc}",
        )

    target_union = target_union if target_union is not None else (target_interval,)
    certified = bool(
        _interval_array_finite_nonempty(target_interval)
        and _interval_array_union_finite_nonempty(target_union)
        and _interval_array_union_contains_point(target_union, reference)
        and _interval_array_union_subsets(target_union, target_interval)
        and interval_array_contains_point(target_interval, reference)
    )
    return HybridTargetEnclosure(
        target_state_interval=target_interval,
        target_state_interval_union=target_union,
        target_reference_state=reference,
        target_interval_source="hybrid_chart_target_enclosure",
        step_index=step_index,
        target_time_in_chart=True,
        certified=certified,
        detail=(
            f"target_time={target_time!r} evaluated inside hybrid chart "
            f"{step_index}"
        ),
    )


def _hybrid_endpoint_target_enclosure(
    hybrid_solution: object,
    *,
    lohner_enclosure: object | None,
    set_enclosure: object | None,
) -> HybridTargetEnclosure:
    if lohner_enclosure is not None:
        target_interval = _flat_float_interval_array(lohner_enclosure.final_state_interval)
        target_source = "lohner_ordinary_set_propagation"
    elif set_enclosure is not None:
        target_interval = _flat_float_interval_array(set_enclosure.final_state_interval)
        target_source = "hybrid_set_propagation"
    else:
        target_interval = _hybrid_target_state_interval(hybrid_solution)
        target_source = "hybrid_step_interval"
    target_union = _hybrid_target_state_interval_union(
        hybrid_solution,
        target_interval=target_interval,
        lohner_enclosure=lohner_enclosure,
        set_enclosure=set_enclosure,
    )
    reference = np.asarray(getattr(hybrid_solution, "final_state"), dtype=float).reshape(-1)
    certified = bool(
        _interval_array_finite_nonempty(target_interval)
        and _interval_array_union_finite_nonempty(target_union)
        and _interval_array_union_contains_point(target_union, reference)
        and _interval_array_union_subsets(target_union, target_interval)
        and interval_array_contains_point(target_interval, reference)
    )
    steps = tuple(getattr(hybrid_solution, "steps", ()))
    return HybridTargetEnclosure(
        target_state_interval=target_interval,
        target_state_interval_union=target_union,
        target_reference_state=reference,
        target_interval_source=target_source,
        step_index=len(steps) - 1 if steps else None,
        target_time_in_chart=bool(steps),
        certified=certified,
        detail="requested target time is the hybrid continuation endpoint",
    )


def _hybrid_step_index_for_target_time(
    hybrid_solution: object,
    target_time: float,
) -> int | None:
    target_time = float(target_time)
    steps = tuple(getattr(hybrid_solution, "steps", ()))
    for reverse_index, step in enumerate(reversed(steps)):
        index = len(steps) - reverse_index - 1
        start = float(getattr(step, "start_time"))
        end = float(getattr(step, "end_time"))
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0,
            abs(target_time),
            abs(start),
            abs(end),
        )
        if start - tolerance <= target_time <= end + tolerance:
            return index
    return None


def _hybrid_step_target_enclosure(
    hybrid_solution: object,
    step_index: int,
    target_time: float,
) -> tuple[Array, tuple[Array, ...] | None, Array]:
    steps = tuple(getattr(hybrid_solution, "steps", ()))
    states = np.asarray(getattr(hybrid_solution, "states"), dtype=float)
    if step_index < 0 or step_index >= len(steps):
        raise ValueError("target step index is outside the hybrid chart sequence")
    if states.shape[0] <= step_index:
        raise ValueError("hybrid point states do not cover the target step")
    step = steps[step_index]
    chart = str(getattr(step, "chart", ""))
    masses = np.asarray(getattr(hybrid_solution, "masses"), dtype=float).reshape(-1)
    if masses.shape != (3,):
        raise ValueError("hybrid solution is missing three positive masses")
    local_time = float(target_time) - float(getattr(step, "start_time"))
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(float(getattr(step, "physical_step", 0.0))),
        abs(local_time),
    )
    if local_time < -tolerance or local_time > float(getattr(step, "physical_step")) + tolerance:
        raise ValueError("target time is outside the selected hybrid chart")
    local_time = min(max(local_time, 0.0), float(getattr(step, "physical_step")))
    start_state = states[step_index].reshape(-1)
    if chart == "ordinary":
        return _hybrid_ordinary_step_target_enclosure(step, start_state, masses, local_time)
    if chart == "binary":
        return _hybrid_binary_step_target_enclosure(step, start_state, masses, local_time)
    raise ValueError(f"unsupported hybrid chart type {chart!r}")


def _hybrid_target_enclosure_chain_certified(
    hybrid_solution: object,
    target_enclosure: HybridTargetEnclosure,
    *,
    lohner_enclosure: object | None,
    set_enclosure: object | None,
    expected_chart_count: int,
) -> bool:
    steps = tuple(getattr(hybrid_solution, "steps", ()))[: int(expected_chart_count)]
    if not steps or len(steps) != int(expected_chart_count):
        return False
    if (
        target_enclosure.step_index is None
        or int(target_enclosure.step_index) != len(steps) - 1
    ):
        return False
    source = target_enclosure.target_interval_source
    if source == "hybrid_chart_target_enclosure":
        return bool(
            target_enclosure.certified
            and int(target_enclosure.step_index) == len(steps) - 1
        )
    if source == "lohner_ordinary_set_propagation":
        return _lohner_target_enclosure_matches_hybrid_prefix(
            lohner_enclosure,
            steps,
        )
    if source == "hybrid_set_propagation":
        return _set_target_enclosure_matches_hybrid_prefix(
            set_enclosure,
            steps,
        )
    if source == "hybrid_step_interval":
        return bool(
            target_enclosure.certified
            and _hybrid_target_time_matches_endpoint(
                float(getattr(hybrid_solution, "times")[-1]),
                hybrid_solution,
            )
        )
    return False


def _lohner_target_enclosure_matches_hybrid_prefix(
    enclosure: object | None,
    hybrid_steps: tuple[object, ...],
) -> bool:
    if enclosure is None or not getattr(enclosure, "proof_certified", False):
        return False
    enclosure_steps = tuple(getattr(enclosure, "steps", ()))
    if len(enclosure_steps) != len(hybrid_steps) or not enclosure_steps:
        return False
    for enclosure_step, hybrid_step in zip(enclosure_steps, hybrid_steps, strict=True):
        if getattr(hybrid_step, "chart", None) != "ordinary":
            return False
        if not _hybrid_enclosure_step_metadata_matches(
            enclosure_step,
            hybrid_step,
            compare_chart=False,
            compare_parameter=False,
        ):
            return False
        if not _optional_interval_tuples_equal(
            getattr(enclosure_step, "event_time_interval", None),
            getattr(hybrid_step, "event_time_interval", None),
        ):
            return False
    return True


def _set_target_enclosure_matches_hybrid_prefix(
    enclosure: object | None,
    hybrid_steps: tuple[object, ...],
) -> bool:
    if enclosure is None or not getattr(enclosure, "proof_certified", False):
        return False
    enclosure_steps = tuple(getattr(enclosure, "steps", ()))
    if len(enclosure_steps) != len(hybrid_steps) or not enclosure_steps:
        return False
    for index, (enclosure_step, hybrid_step) in enumerate(
        zip(enclosure_steps, hybrid_steps, strict=True)
    ):
        if not _hybrid_enclosure_step_metadata_matches(
            enclosure_step,
            hybrid_step,
            compare_chart=True,
            compare_parameter=True,
        ):
            return False
        if index == 0 and not _state_interval_union_tuples_equal(
            getattr(enclosure_step, "start_state_interval_union", None),
            getattr(hybrid_step, "start_state_interval_union", None)
            or (getattr(hybrid_step, "start_state_interval", None),),
        ):
            return False
        if not _optional_interval_tuples_equal(
            getattr(enclosure_step, "event_time_interval", None),
            getattr(hybrid_step, "event_time_interval", None),
        ):
            return False
        if not _optional_interval_tuples_equal(
            getattr(enclosure_step, "event_parameter_interval", None),
            _hybrid_step_event_parameter_interval(hybrid_step),
        ):
            return False
    return True


def _hybrid_enclosure_step_metadata_matches(
    enclosure_step: object,
    hybrid_step: object,
    *,
    compare_chart: bool,
    compare_parameter: bool,
) -> bool:
    if not _floats_close(
        getattr(enclosure_step, "start_time", np.nan),
        getattr(hybrid_step, "start_time", np.nan),
    ):
        return False
    if not _floats_close(
        getattr(enclosure_step, "physical_step", np.nan),
        getattr(hybrid_step, "physical_step", np.nan),
    ):
        return False
    if getattr(enclosure_step, "event", None) != getattr(hybrid_step, "event", None):
        return False
    if compare_chart and getattr(enclosure_step, "chart", None) != getattr(hybrid_step, "chart", None):
        return False
    if compare_parameter:
        if not _floats_close(
            getattr(enclosure_step, "parameter_step", np.nan),
            getattr(hybrid_step, "parameter_step", np.nan),
        ):
            return False
        enclosure_pair = getattr(enclosure_step, "pair", None)
        hybrid_pair = getattr(hybrid_step, "pair", None)
        if (
            None if enclosure_pair is None else tuple(enclosure_pair)
        ) != (None if hybrid_pair is None else tuple(hybrid_pair)):
            return False
    return True


def _hybrid_step_event_parameter_interval(step: object) -> tuple[float, float] | None:
    certificate = getattr(step, "event_certificate", None)
    root_enclosure = getattr(certificate, "root_enclosure", None)
    if root_enclosure is None:
        return None
    return getattr(root_enclosure, "interval", None)


def _optional_interval_tuples_equal(left: object, right: object) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    try:
        left_lower, left_upper = left
        right_lower, right_upper = right
    except (TypeError, ValueError):
        return False
    return bool(
        _floats_close(left_lower, right_lower)
        and _floats_close(left_upper, right_upper)
    )


def _floats_close(left: object, right: object) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(left_value),
        abs(right_value),
    )
    return bool(
        np.isfinite(left_value)
        and np.isfinite(right_value)
        and abs(left_value - right_value) <= tolerance
    )


def _hybrid_ordinary_step_target_enclosure(
    step: object,
    start_state: Array,
    masses: Array,
    local_time: float,
) -> tuple[Array, tuple[Array, ...] | None, Array]:
    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(start_state[:6], dtype=float).reshape(3, 2)
    velocities = np.asarray(start_state[6:], dtype=float).reshape(3, 2)
    point_solution = construct_taylor_solution(
        positions,
        velocities,
        masses,
        order=_hybrid_step_computed_order(step),
    )
    reference = np.concatenate(
        [
            point_solution.positions_at(local_time).reshape(-1),
            point_solution.velocities_at(local_time).reshape(-1),
        ]
    )
    retained_order = _hybrid_step_retained_order(step)
    tail_bound = _step_tail_bound(step)
    start_union = getattr(step, "start_state_interval_union", None)
    if start_union:
        members = tuple(
            _ordinary_interval_target_from_state_interval(
                state_interval,
                masses,
                local_time,
                retained_order=retained_order,
                computed_order=_hybrid_step_computed_order(step),
                tail_bound=tail_bound,
            )
            for state_interval in start_union
        )
        return _interval_array_union_hull(members), members, reference
    start_interval = getattr(step, "start_state_interval", None)
    if start_interval is None:
        start_interval = tuple((float(value), float(value)) for value in start_state)
    target = _ordinary_interval_target_from_state_interval(
        start_interval,
        masses,
        local_time,
        retained_order=retained_order,
        computed_order=_hybrid_step_computed_order(step),
        tail_bound=tail_bound,
    )
    return target, (target,), reference


def _hybrid_binary_step_target_enclosure(
    step: object,
    start_state: Array,
    masses: Array,
    local_time: float,
) -> tuple[Array, tuple[Array, ...] | None, Array]:
    masses = np.asarray(masses, dtype=float)
    pair = getattr(step, "pair", None)
    if masses.size != 3 or pair is None:
        raise ValueError("binary hybrid step is missing mass or pair data")
    positions = np.asarray(start_state[:6], dtype=float).reshape(3, 2)
    velocities = np.asarray(start_state[6:], dtype=float).reshape(3, 2)
    point_initial = _regularized_point_state_for_hybrid_binary_step(
        step,
        positions,
        velocities,
        masses,
        pair=tuple(pair),
    )
    point_solution = construct_regularized_binary_taylor_solution(
        point_initial,
        order=_hybrid_step_computed_order(step),
    )
    s_value = _hybrid_binary_parameter_for_local_time(
        point_solution,
        local_time,
        float(getattr(step, "parameter_step")),
    )
    target_positions, target_velocities = regularized_binary_collision_chart_to_planar(
        point_solution.state_at(s_value)
    )
    reference = np.concatenate(
        [target_positions.reshape(-1), target_velocities.reshape(-1)]
    )
    interval_union = getattr(step, "start_regularized_interval_state_union", ())
    if interval_union:
        target = _regularized_binary_atlas_end_state_interval(
            point_initial,
            tuple(interval_union),
            s_value,
            order=_hybrid_step_computed_order(step),
        )
    else:
        interval_initial = getattr(step, "start_regularized_interval_state", None)
        if interval_initial is None:
            raise ValueError("binary hybrid step is missing an interval LC lift")
        target = _regularized_binary_end_state_interval(
            point_initial,
            s_value,
            order=_hybrid_step_computed_order(step),
            interval_initial_state=interval_initial,
        )
    inflated = _inflate_interval_array(_flat_float_interval_array(target), _step_tail_bound(step))
    return inflated, (inflated,), reference


def _ordinary_interval_target_from_state_interval(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array,
    local_time: float,
    *,
    retained_order: int,
    computed_order: int,
    tail_bound: float,
) -> Array:
    positions, velocities = _planar_interval_state_arrays(state_interval)
    solution = construct_interval_taylor_solution_from_intervals(
        positions,
        velocities,
        masses,
        order=computed_order,
    )
    return _ordinary_target_state_interval_over_interval(
        solution,
        FloatInterval.point(float(local_time)),
        retained_order=retained_order,
        tail_bound=tail_bound,
    )


def _hybrid_binary_parameter_for_local_time(
    solution: object,
    local_time: float,
    parameter_step: float,
) -> float:
    local_time = float(local_time)
    parameter_step = float(parameter_step)
    if local_time <= 1.0e-18:
        return 0.0
    endpoint_time = float(solution.physical_time_at(parameter_step))
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(local_time), abs(endpoint_time))
    if abs(local_time - endpoint_time) <= tolerance:
        return parameter_step

    def residual(s_value: float) -> float:
        return float(solution.physical_time_at(s_value) - local_time)

    if residual(parameter_step) < -tolerance:
        raise ValueError("binary chart target time is beyond the regularized endpoint")
    return float(brentq(residual, 0.0, parameter_step, xtol=1e-15, rtol=1e-15))


def _regularized_point_state_for_hybrid_binary_step(
    step: object,
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    pair: tuple[int, int],
) -> RegularizedBinaryCollisionChartState:
    try:
        return planar_to_regularized_binary_collision_chart(
            positions,
            velocities,
            masses,
            pair=pair,
        )
    except ValueError:
        interval_state = getattr(step, "start_regularized_interval_state", None)
        if interval_state is None:
            interval_union = getattr(step, "start_regularized_interval_state_union", ())
            interval_state = tuple(interval_union)[0] if interval_union else None
        if interval_state is None:
            raise
        return _regularized_point_state_from_interval(interval_state)


def _regularized_point_state_from_interval(
    interval_state: IntervalRegularizedBinaryCollisionChartState,
) -> RegularizedBinaryCollisionChartState:
    return RegularizedBinaryCollisionChartState(
        masses=np.asarray(interval_state.masses, dtype=float),
        pair=tuple(interval_state.pair),
        z=_interval_midpoint_array(interval_state.z),
        z_velocity=_interval_midpoint_array(interval_state.z_velocity),
        pair_energy=_interval_midpoint(interval_state.pair_energy),
        binary_center=_interval_midpoint_array(interval_state.binary_center),
        binary_center_velocity=_interval_midpoint_array(interval_state.binary_center_velocity),
        third_offset=_interval_midpoint_array(interval_state.third_offset),
        third_offset_velocity=_interval_midpoint_array(interval_state.third_offset_velocity),
    )


def _interval_midpoint_array(values: Array) -> Array:
    return np.asarray(
        [_interval_midpoint(value) for value in np.asarray(values, dtype=object).reshape(-1)],
        dtype=float,
    ).reshape(np.asarray(values, dtype=object).shape)


def _interval_midpoint(value: object) -> float:
    lower, upper = _float_interval_bounds(value)
    return 0.5 * (lower + upper)


def _hybrid_step_retained_order(step: object) -> int:
    certificate = getattr(step, "truncation_certificate", None)
    return int(getattr(certificate, "retained_order", 0))


def _hybrid_step_computed_order(step: object) -> int:
    certificate = getattr(step, "truncation_certificate", None)
    computed = int(getattr(certificate, "computed_order", 0))
    if computed > 0:
        return computed
    retained = _hybrid_step_retained_order(step)
    return retained if retained > 0 else 1


def _time_reverse_float_interval(interval: FloatInterval | None) -> FloatInterval | None:
    if interval is None:
        return None
    lower, upper = interval.as_tuple()
    return FloatInterval(float(-upper), float(-lower))


def _time_reverse_state_interval(intervals: object) -> Array:
    if intervals is None:
        raise ValueError("cannot time-reverse a missing state interval")
    interval_array = np.asarray(intervals, dtype=object).reshape(-1)
    if interval_array.size == 0 or interval_array.size % 2 != 0:
        raise ValueError("state interval must contain position and velocity halves")
    half = interval_array.size // 2
    reversed_intervals = []
    for index, interval in enumerate(interval_array):
        lower, upper = _float_interval_bounds(interval)
        if index < half:
            reversed_intervals.append(FloatInterval(lower, upper))
        else:
            reversed_intervals.append(FloatInterval(float(-upper), float(-lower)))
    return np.asarray(reversed_intervals, dtype=object)


def _time_reverse_state_vector(state: object) -> Array:
    state_array = np.asarray(state, dtype=float).reshape(-1)
    if state_array.size == 0 or state_array.size % 2 != 0:
        raise ValueError("state vector must contain position and velocity halves")
    half = state_array.size // 2
    reversed_state = state_array.copy()
    reversed_state[half:] *= -1.0
    return reversed_state


def _interval_midpoint_state(intervals: object) -> Array:
    interval_array = np.asarray(intervals, dtype=object).reshape(-1)
    return np.asarray(
        [
            0.5 * (_float_interval_bounds(interval)[0] + _float_interval_bounds(interval)[1])
            for interval in interval_array
        ],
        dtype=float,
    )


def _float_interval_bounds(interval: object) -> tuple[float, float]:
    try:
        lower, upper = interval.as_tuple()  # type: ignore[attr-defined]
    except AttributeError:
        lower = getattr(interval, "lower")
        upper = getattr(interval, "upper")
    return float(lower), float(upper)


def _masses_positive_finite(masses: object) -> bool:
    try:
        masses_array = np.asarray(masses, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(
        masses_array.size > 0
        and np.all(np.isfinite(masses_array))
        and np.all(masses_array > 0.0)
    )


def _interval_array_finite_nonempty(intervals: object) -> bool:
    if intervals is None:
        return False
    try:
        interval_array = np.asarray(intervals, dtype=object).reshape(-1)
    except (TypeError, ValueError):
        return False
    if interval_array.size == 0:
        return False
    for interval in interval_array:
        try:
            lower, upper = interval.as_tuple()
        except AttributeError:
            lower = getattr(interval, "lower", np.nan)
            upper = getattr(interval, "upper", np.nan)
        lower = float(lower)
        upper = float(upper)
        if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
            return False
    return True


def _interval_array_union_finite_nonempty(interval_union: object) -> bool:
    if interval_union is None:
        return False
    try:
        members = tuple(interval_union)
    except TypeError:
        return False
    return bool(members and all(_interval_array_finite_nonempty(member) for member in members))


def _interval_array_union_contains_point(interval_union: object, state: object) -> bool:
    if interval_union is None:
        return False
    try:
        members = tuple(interval_union)
    except TypeError:
        return False
    return any(
        interval_array_contains_point(member, np.asarray(state, dtype=float).reshape(-1))
        for member in members
    )


def _interval_array_subset(inner: object, outer: object) -> bool:
    try:
        inner_array = np.asarray(inner, dtype=object).reshape(-1)
        outer_array = np.asarray(outer, dtype=object).reshape(-1)
    except (TypeError, ValueError):
        return False
    if inner_array.size != outer_array.size or inner_array.size == 0:
        return False
    for inner_interval, outer_interval in zip(inner_array, outer_array):
        inner_lower, inner_upper = _float_interval_bounds(inner_interval)
        outer_lower, outer_upper = _float_interval_bounds(outer_interval)
        if inner_lower < outer_lower or inner_upper > outer_upper:
            return False
    return True


def _interval_array_union_subsets(interval_union: object, outer: object) -> bool:
    if interval_union is None:
        return False
    try:
        members = tuple(interval_union)
    except TypeError:
        return False
    return bool(members and all(_interval_array_subset(member, outer) for member in members))


def _interval_array_union_hull(interval_union: tuple[Array, ...]) -> Array:
    if not interval_union:
        raise ValueError("cannot hull an empty interval-array union")
    arrays = tuple(np.asarray(member, dtype=object).reshape(-1) for member in interval_union)
    size = arrays[0].size
    if size == 0 or any(member.size != size for member in arrays):
        raise ValueError("interval-array union members must be nonempty and equal-sized")
    hull = []
    for index in range(size):
        bounds = [_float_interval_bounds(member[index]) for member in arrays]
        lower = min(bound[0] for bound in bounds)
        upper = max(bound[1] for bound in bounds)
        hull.append(
            FloatInterval(
                float(np.nextafter(lower, -np.inf)),
                float(np.nextafter(upper, np.inf)),
            )
        )
    return np.asarray(hull, dtype=object)


def _interval_array_unions_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return False
    try:
        left_members = tuple(left)
        right_members = tuple(right)
    except TypeError:
        return False
    if len(left_members) != len(right_members) or not left_members:
        return False
    return all(
        _interval_arrays_equal(
            np.asarray(left_member, dtype=object).reshape(-1),
            np.asarray(right_member, dtype=object).reshape(-1),
        )
        for left_member, right_member in zip(left_members, right_members, strict=True)
    )


def _expected_initial_state_interval_union_from_evaluation(
    evaluation: object,
) -> tuple[Array, ...] | None:
    current = evaluation
    seen: set[int] = set()
    reverse_count = 0
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeReversedValidatedEvaluation):
            reverse_count += 1
            current = current.forward_evaluation
            continue
        hybrid_solution = getattr(current, "hybrid_solution", None)
        if hybrid_solution is not None:
            return _apply_state_interval_union_time_reversals(
                _hybrid_initial_state_interval_union(hybrid_solution),
                reverse_count,
            )
        current = getattr(current, "evaluation", None)
    return None


def _expected_target_state_interval_union_from_evaluation(
    evaluation: object,
) -> tuple[Array, ...] | None:
    current = evaluation
    seen: set[int] = set()
    reverse_count = 0
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeReversedValidatedEvaluation):
            reverse_count += 1
            current = current.forward_evaluation
            continue
        target_union = _hybrid_target_state_interval_union_from_evaluation(current)
        if target_union is not None:
            return _apply_state_interval_union_time_reversals(
                target_union,
                reverse_count,
            )
        current = getattr(current, "evaluation", None)
    return None


def _apply_state_interval_union_time_reversals(
    interval_union: tuple[Array, ...] | None,
    reverse_count: int,
) -> tuple[Array, ...] | None:
    if interval_union is None:
        return None
    result = tuple(interval_union)
    for _ in range(reverse_count):
        result = tuple(_time_reverse_state_interval(member) for member in result)
    return result


def _apply_state_vector_time_reversals(state: Array, reverse_count: int) -> Array:
    result = np.asarray(state, dtype=float).reshape(-1)
    for _ in range(reverse_count):
        result = _time_reverse_state_vector(result)
    return result


def _hybrid_target_state_interval_union_from_evaluation(
    evaluation: object,
) -> tuple[Array, ...] | None:
    hybrid_solution = getattr(evaluation, "hybrid_solution", None)
    if hybrid_solution is None:
        return None
    target_union = getattr(evaluation, "target_state_interval_union", None)
    if (
        getattr(evaluation, "target_interval_source", None)
        == "hybrid_chart_target_enclosure"
        and target_union is not None
    ):
        return tuple(target_union)
    set_enclosure = getattr(evaluation, "set_enclosure", None)
    if set_enclosure is not None:
        steps = tuple(getattr(set_enclosure, "steps", ()))
        if steps and getattr(steps[-1], "end_state_interval_union", None):
            return tuple(
                _flat_float_interval_array(state_interval)
                for state_interval in steps[-1].end_state_interval_union
            )
    lohner_enclosure = getattr(evaluation, "lohner_enclosure", None)
    if (
        lohner_enclosure is not None
        and getattr(lohner_enclosure, "final_state_interval", None) is not None
    ):
        return (_flat_float_interval_array(lohner_enclosure.final_state_interval),)
    steps = tuple(getattr(hybrid_solution, "steps", ()))
    if steps and getattr(steps[-1], "end_state_interval_union", None) is not None:
        return tuple(
            _flat_float_interval_array(state_interval)
            for state_interval in steps[-1].end_state_interval_union
        )
    return (_hybrid_target_state_interval(hybrid_solution),)


def _target_state_point_from_evaluation(evaluation: object) -> Array | None:
    current = evaluation
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attribute in ("final_state", "target_reference_state", "target_state"):
            try:
                state = getattr(current, attribute)
            except (AttributeError, TypeError, ValueError):
                continue
            if callable(state):
                continue
            try:
                state_array = np.asarray(state, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if state_array.size and np.all(np.isfinite(state_array)):
                return state_array
        current = getattr(current, "evaluation", None)
    return None


def _initial_state_point_from_evaluation(evaluation: object) -> Array | None:
    current = evaluation
    seen: set[int] = set()
    reverse_count = 0
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeReversedValidatedEvaluation):
            reverse_count += 1
            current = current.forward_evaluation
            continue
        hybrid_solution = getattr(current, "hybrid_solution", None)
        if hybrid_solution is not None:
            states = np.asarray(getattr(hybrid_solution, "states", ()), dtype=float)
            if states.size == 0:
                return None
            return _apply_state_vector_time_reversals(states[0].reshape(-1), reverse_count)
        point_reduction = getattr(current, "point_reduction", None)
        if point_reduction is not None:
            try:
                positions = (
                    np.asarray(point_reduction.positions, dtype=float)
                    + np.asarray(point_reduction.center_position, dtype=float)
                )
                velocities = (
                    np.asarray(point_reduction.velocities, dtype=float)
                    + np.asarray(point_reduction.center_velocity, dtype=float)
                )
                state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
            except (AttributeError, TypeError, ValueError):
                state = None
            if state is not None and state.size and np.all(np.isfinite(state)):
                return _apply_state_vector_time_reversals(state, reverse_count)
        initial_state = getattr(current, "initial_state", None)
        if initial_state is not None and not callable(initial_state):
            try:
                state = np.asarray(initial_state, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                state = None
            if state is not None and state.size and np.all(np.isfinite(state)):
                return _apply_state_vector_time_reversals(state, reverse_count)
        current = getattr(current, "evaluation", None)
    return None


def _initial_state_interval_from_evaluation(evaluation: object) -> Array | None:
    """Recover the inertial initial-state interval from a reduced evaluator."""

    current = evaluation
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        interval_reduction = getattr(current, "interval_reduction", None)
        if interval_reduction is not None:
            positions, velocities = reconstruct_interval_from_center_of_mass_frame(
                interval_reduction.positions,
                interval_reduction.velocities,
                interval_reduction.center_position,
                interval_reduction.center_velocity,
                time=FloatInterval.point(0.0),
            )
            return _packed_interval_state_array(positions, velocities)
        current = getattr(current, "evaluation", None)
    return None


def _packed_interval_state_array(positions: object, velocities: object) -> Array:
    return np.concatenate(
        [
            np.asarray(positions, dtype=object).reshape(-1),
            np.asarray(velocities, dtype=object).reshape(-1),
        ]
    )


def _hybrid_initial_state_interval(hybrid_solution: object) -> Array | None:
    steps = tuple(getattr(hybrid_solution, "steps", ()))
    if steps and getattr(steps[0], "start_state_interval", None) is not None:
        return _flat_float_interval_array(steps[0].start_state_interval)
    states = np.asarray(getattr(hybrid_solution, "states", ()), dtype=float)
    if states.size == 0:
        return None
    return np.asarray(
        [FloatInterval.point(float(value)) for value in states[0].reshape(-1)],
        dtype=object,
    )


def _hybrid_initial_state_interval_union(hybrid_solution: object) -> tuple[Array, ...] | None:
    steps = tuple(getattr(hybrid_solution, "steps", ()))
    if steps and getattr(steps[0], "start_state_interval_union", None) is not None:
        return tuple(
            _flat_float_interval_array(state_interval)
            for state_interval in steps[0].start_state_interval_union
        )
    initial_interval = _hybrid_initial_state_interval(hybrid_solution)
    return (initial_interval,) if initial_interval is not None else None


def _hybrid_initial_state_contains_point(
    hybrid_solution: object,
    initial_state_interval: Array | None,
) -> bool:
    states = np.asarray(getattr(hybrid_solution, "states", ()), dtype=float)
    if states.size == 0 or initial_state_interval is None:
        return False
    return interval_array_contains_point(initial_state_interval, states[0].reshape(-1))


def _hybrid_initial_state_union_contains_point(
    hybrid_solution: object,
    initial_state_interval_union: tuple[Array, ...] | None,
) -> bool:
    states = np.asarray(getattr(hybrid_solution, "states", ()), dtype=float)
    if states.size == 0 or initial_state_interval_union is None:
        return False
    return _interval_array_union_contains_point(initial_state_interval_union, states[0].reshape(-1))


def _hybrid_lohner_target_enclosure(hybrid_solution: object) -> object | None:
    if not hasattr(hybrid_solution, "lohner_ordinary_set_propagated_interval_enclosure"):
        return None
    try:
        enclosure = hybrid_solution.lohner_ordinary_set_propagated_interval_enclosure()
    except (RuntimeError, ValueError):
        return None
    if (
        getattr(enclosure, "proof_certified", False)
        and getattr(enclosure, "final_state_interval", None) is not None
    ):
        return enclosure
    return None


def _hybrid_set_target_enclosure(hybrid_solution: object) -> object | None:
    if not hasattr(hybrid_solution, "ordinary_set_propagated_interval_enclosure"):
        return None
    try:
        enclosure = hybrid_solution.ordinary_set_propagated_interval_enclosure()
    except (RuntimeError, ValueError):
        return None
    if (
        getattr(enclosure, "proof_certified", False)
        and getattr(enclosure, "final_state_interval", None) is not None
    ):
        return enclosure
    return None


def _spatial_interval_state_arrays(
    state_interval: tuple[tuple[float, float], ...],
) -> tuple[Array, Array]:
    if len(state_interval) != 18:
        raise ValueError("spatial interval state must have length 18")
    values = np.asarray(
        [FloatInterval(float(lower), float(upper)) for lower, upper in state_interval],
        dtype=object,
    )
    return values[:9].reshape(3, 3), values[9:].reshape(3, 3)


def _planar_interval_state_arrays(
    state_interval: tuple[tuple[float, float], ...],
) -> tuple[Array, Array]:
    if len(state_interval) != 12:
        raise ValueError("planar interval state must have length 12")
    values = np.asarray(
        [FloatInterval(float(lower), float(upper)) for lower, upper in state_interval],
        dtype=object,
    )
    return values[:6].reshape(3, 2), values[6:].reshape(3, 2)


def _ordinary_physical_time_series(order: int) -> tuple[FloatInterval, ...]:
    if order < 1:
        raise ValueError("order must be at least one")
    return (
        FloatInterval.point(0.0),
        FloatInterval.point(1.0),
        *(FloatInterval.point(0.0) for _ in range(order - 1)),
    )


def _shift_optional_interval(
    interval: FloatInterval | None,
    offset: FloatInterval,
) -> FloatInterval | None:
    if interval is None:
        return None
    return interval + offset


def _coerce_float_interval(value: FloatInterval | tuple[float, float]) -> FloatInterval:
    if isinstance(value, FloatInterval):
        return value
    lower, upper = value
    return FloatInterval(float(lower), float(upper))


def _normalize_state_interval_tuple(
    state_interval: tuple[tuple[float, float], ...],
    *,
    expected_length: int,
) -> tuple[tuple[float, float], ...]:
    try:
        intervals = tuple(state_interval)
    except TypeError as exc:
        raise ValueError("state interval must be an iterable of scalar intervals") from exc
    if len(intervals) != int(expected_length):
        raise ValueError(f"state interval must have length {expected_length}")
    normalized = []
    for interval in intervals:
        lower, upper = interval
        lower = float(lower)
        upper = float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("state interval endpoints must be finite and ordered")
        normalized.append((lower, upper))
    return tuple(normalized)


def _state_interval_tuple_finite_nonempty(
    state_interval: object,
    *,
    expected_length: int | None = None,
) -> bool:
    try:
        intervals = tuple(state_interval)  # type: ignore[arg-type]
    except TypeError:
        return False
    if not intervals:
        return False
    if expected_length is not None and len(intervals) != int(expected_length):
        return False
    for interval in intervals:
        try:
            lower, upper = interval
        except (TypeError, ValueError):
            return False
        try:
            lower = float(lower)
            upper = float(upper)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            return False
    return True


def _state_interval_tuple_subset(
    inner: tuple[tuple[float, float], ...],
    outer: tuple[tuple[float, float], ...],
) -> bool:
    if len(inner) != len(outer) or not inner:
        return False
    return all(
        float(outer_lower) <= float(inner_lower)
        and float(inner_upper) <= float(outer_upper)
        for (inner_lower, inner_upper), (outer_lower, outer_upper) in zip(
            inner,
            outer,
            strict=True,
        )
    )


def _hull_state_interval_tuples(
    state_intervals: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[float, float], ...]:
    if not state_intervals:
        raise ValueError("cannot hull an empty state-interval union")
    length = len(state_intervals[0])
    if length == 0 or any(len(state_interval) != length for state_interval in state_intervals):
        raise ValueError("state-interval union members must have equal nonzero length")
    hull = []
    for index in range(length):
        lower = min(float(state_interval[index][0]) for state_interval in state_intervals)
        upper = max(float(state_interval[index][1]) for state_interval in state_intervals)
        hull.append((lower, upper))
    return tuple(hull)


def _state_interval_tuples_same_bounds(
    left: tuple[tuple[float, float], ...],
    right: tuple[tuple[float, float], ...],
    *,
    tolerance: float = 0.0,
) -> bool:
    if len(left) != len(right) or not left:
        return False
    tolerance = float(tolerance)
    return all(
        abs(float(left_lower) - float(right_lower)) <= tolerance
        and abs(float(left_upper) - float(right_upper)) <= tolerance
        for (left_lower, left_upper), (right_lower, right_upper) in zip(
            left,
            right,
            strict=True,
        )
    )


def _spatial_pair_distance_bounds_from_state_interval(
    state_interval: tuple[tuple[float, float], ...],
    pair: tuple[int, int],
) -> tuple[float, float]:
    positions, _velocities = _spatial_interval_state_arrays(state_interval)
    first, second = _ordered_pair(*pair)
    lower_squared = 0.0
    upper_squared = 0.0
    for dimension in range(3):
        difference = positions[second, dimension] - positions[first, dimension]
        lower = float(difference.lower)
        upper = float(difference.upper)
        if lower <= 0.0 <= upper:
            component_lower = 0.0
        else:
            component_lower = min(abs(lower), abs(upper))
        component_upper = max(abs(lower), abs(upper))
        lower_squared += component_lower * component_lower
        upper_squared += component_upper * component_upper
    return float(np.sqrt(max(0.0, lower_squared))), float(np.sqrt(max(0.0, upper_squared)))


def _threshold_adjacent_triple_close_cluster(
    pair_distance_bounds: tuple[tuple[tuple[int, int], tuple[float, float]], ...],
    *,
    possible_close_pairs: tuple[tuple[int, int], ...],
    certified_close_pairs: tuple[tuple[int, int], ...],
    binary_distance_threshold: float,
) -> bool:
    threshold = float(binary_distance_threshold)
    if threshold <= 0.0:
        return False
    if len(possible_close_pairs) != 3 or len(certified_close_pairs) != 1:
        return False
    bounds_by_pair = {pair: bounds for pair, bounds in pair_distance_bounds}
    unresolved_pairs = tuple(
        pair for pair in _three_body_pairs() if pair not in certified_close_pairs
    )
    if len(unresolved_pairs) != 2:
        return False
    if not all(pair in bounds_by_pair for pair in unresolved_pairs):
        return False
    threshold_band = _SPATIAL_TRIPLE_CLOSE_ADJACENCY_FACTOR * threshold
    return bool(
        all(
            lower <= threshold <= upper <= threshold_band
            for lower, upper in (bounds_by_pair[pair] for pair in unresolved_pairs)
        )
    )


def _triple_collision_exclusion_for_state_interval(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array | None,
) -> object | None:
    if masses is None:
        return None
    try:
        positions, velocities = _spatial_interval_state_arrays(state_interval)
        return certify_nonzero_angular_momentum_excludes_triple_collision(
            positions,
            velocities,
            np.asarray(masses, dtype=float),
        )
    except (TypeError, ValueError, FloatingPointError):
        return None


def _spatial_state_interval_midpoint_arrays(
    state_interval: tuple[tuple[float, float], ...],
) -> tuple[Array, Array]:
    state_interval = _normalize_state_interval_tuple(
        state_interval,
        expected_length=18,
    )
    midpoints = np.asarray(
        [0.5 * (float(lower) + float(upper)) for lower, upper in state_interval],
        dtype=float,
    )
    return midpoints[:9].reshape(3, 3), midpoints[9:].reshape(3, 3)


def _jacobi_cluster_coordinates_for_representative_state(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array | None,
    close_pairs: tuple[tuple[int, int], ...],
) -> tuple[object, ...]:
    if masses is None or not close_pairs:
        return ()
    try:
        positions, velocities = _spatial_state_interval_midpoint_arrays(state_interval)
        masses_array = np.asarray(masses, dtype=float)
        certificates = tuple(
            construct_jacobi_cluster_coordinates(
                positions,
                velocities,
                masses_array,
                pair=pair,
            )
            for pair in close_pairs
        )
        return certificates
    except (TypeError, ValueError, FloatingPointError):
        return ()


def _interval_jacobi_cluster_coordinates_for_state_interval(
    state_interval: tuple[tuple[float, float], ...],
    masses: Array | None,
    close_pairs: tuple[tuple[int, int], ...],
) -> tuple[object, ...]:
    if masses is None or not close_pairs:
        return ()
    try:
        positions, velocities = _spatial_interval_state_arrays(state_interval)
        masses_array = np.asarray(masses, dtype=float)
        return tuple(
            construct_interval_jacobi_cluster_coordinates(
                positions,
                velocities,
                masses_array,
                pair=pair,
            )
            for pair in close_pairs
        )
    except (TypeError, ValueError, FloatingPointError):
        return ()


def _triple_close_cluster_missing_obligations(
    exclusion_certificate: object | None,
) -> tuple[str, str]:
    if bool(
        exclusion_certificate is not None
        and getattr(exclusion_certificate, "certified", False)
    ):
        return (
            "simultaneous_close_pair_partitioning",
            "spatial_triple_close_cluster_requires_cluster_blowup_after_nonzero_angular_exclusion",
        )
    return (
        "simultaneous_close_pair_partitioning",
        "spatial_triple_close_cluster_requires_cluster_blowup_or_total_collision_selector",
    )


def _simultaneous_close_pair_branch(
    state_interval: tuple[tuple[float, float], ...],
    *,
    branch_id: str,
    binary_distance_threshold: float,
    depth: int,
    masses: Array | None = None,
) -> SimultaneousClosePairPartitionBranch:
    pair_distance_bounds = tuple(
        (pair, _spatial_pair_distance_bounds_from_state_interval(state_interval, pair))
        for pair in _three_body_pairs()
    )
    possible_close_pairs = tuple(
        pair
        for pair, (lower, _upper) in pair_distance_bounds
        if lower <= binary_distance_threshold
    )
    certified_close_pairs = tuple(
        pair
        for pair, (_lower, upper) in pair_distance_bounds
        if upper <= binary_distance_threshold
    )
    missing: list[str] = []
    selected_pair = possible_close_pairs[0] if len(possible_close_pairs) == 1 else None
    triple_collision_exclusion_certificate = None
    jacobi_cluster_coordinate_certificate = None
    jacobi_cluster_coordinate_certificates: tuple[object, ...] = ()
    interval_jacobi_cluster_coordinate_certificate = None
    interval_jacobi_cluster_coordinate_certificates: tuple[object, ...] = ()
    if not possible_close_pairs:
        leaf_type = "ordinary_separated"
    elif len(possible_close_pairs) == 1:
        leaf_type = "single_possible_binary"
    elif len(certified_close_pairs) >= 2:
        leaf_type = "certified_triple_close_cluster"
        triple_collision_exclusion_certificate = (
            _triple_collision_exclusion_for_state_interval(state_interval, masses)
        )
        jacobi_cluster_coordinate_certificates = (
            _jacobi_cluster_coordinates_for_representative_state(
                state_interval,
                masses,
                certified_close_pairs,
            )
        )
        interval_jacobi_cluster_coordinate_certificates = (
            _interval_jacobi_cluster_coordinates_for_state_interval(
                state_interval,
                masses,
                certified_close_pairs,
            )
        )
        missing.extend(
            _triple_close_cluster_missing_obligations(
                triple_collision_exclusion_certificate
            )
        )
    elif _threshold_adjacent_triple_close_cluster(
        pair_distance_bounds,
        possible_close_pairs=possible_close_pairs,
        certified_close_pairs=certified_close_pairs,
        binary_distance_threshold=binary_distance_threshold,
    ):
        leaf_type = "threshold_adjacent_triple_close_cluster"
        triple_collision_exclusion_certificate = (
            _triple_collision_exclusion_for_state_interval(state_interval, masses)
        )
        jacobi_cluster_coordinate_certificates = (
            _jacobi_cluster_coordinates_for_representative_state(
                state_interval,
                masses,
                certified_close_pairs,
            )
        )
        interval_jacobi_cluster_coordinate_certificates = (
            _interval_jacobi_cluster_coordinates_for_state_interval(
                state_interval,
                masses,
                certified_close_pairs,
            )
        )
        jacobi_cluster_coordinate_certificate = (
            jacobi_cluster_coordinate_certificates[0]
            if len(jacobi_cluster_coordinate_certificates) == 1
            else None
        )
        interval_jacobi_cluster_coordinate_certificate = (
            interval_jacobi_cluster_coordinate_certificates[0]
            if len(interval_jacobi_cluster_coordinate_certificates) == 1
            else None
        )
        missing.extend(
            _triple_close_cluster_missing_obligations(
                triple_collision_exclusion_certificate
            )
        )
    else:
        leaf_type = "ambiguous_simultaneous_close_pair"
        missing.append("simultaneous_close_pair_partitioning")
    return SimultaneousClosePairPartitionBranch(
        branch_id=branch_id,
        state_interval=state_interval,
        pair_distance_bounds=pair_distance_bounds,
        possible_close_pairs=possible_close_pairs,
        certified_close_pairs=certified_close_pairs,
        selected_pair=selected_pair,
        leaf_type=leaf_type,
        depth=int(depth),
        certified=not missing,
        missing_obligations=tuple(missing),
        triple_collision_exclusion_certificate=triple_collision_exclusion_certificate,
        jacobi_cluster_coordinate_certificate=jacobi_cluster_coordinate_certificate,
        jacobi_cluster_coordinate_certificates=jacobi_cluster_coordinate_certificates,
        interval_jacobi_cluster_coordinate_certificate=interval_jacobi_cluster_coordinate_certificate,
        interval_jacobi_cluster_coordinate_certificates=interval_jacobi_cluster_coordinate_certificates,
    )


def _simultaneous_close_pair_split_axis(
    branch: SimultaneousClosePairPartitionBranch,
    *,
    min_width: float,
) -> int | None:
    bodies = sorted({body for pair in branch.possible_close_pairs for body in pair})
    candidate_indices = tuple(body * 3 + dimension for body in bodies for dimension in range(3))
    best_index = None
    best_width = -np.inf
    for index in candidate_indices:
        lower, upper = branch.state_interval[index]
        width = float(upper - lower)
        if width > best_width:
            best_index = index
            best_width = width
    if best_index is None or best_width <= float(min_width):
        return None
    return int(best_index)


def _bisect_state_interval_tuple(
    state_interval: tuple[tuple[float, float], ...],
    axis_index: int,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    lower, upper = state_interval[axis_index]
    midpoint = 0.5 * (float(lower) + float(upper))
    left = list(state_interval)
    right = list(state_interval)
    left[axis_index] = (float(lower), midpoint)
    right[axis_index] = (midpoint, float(upper))
    return tuple(left), tuple(right)


def certify_simultaneous_close_pair_partition(
    state_interval: tuple[tuple[float, float], ...],
    *,
    binary_distance_threshold: float,
    max_depth: int = 8,
    max_branches: int = 256,
    min_position_width: float = 1.0e-14,
    masses: Array | None = None,
) -> SimultaneousClosePairSplitCertificate:
    """Split a spatial state box until close-pair ambiguity is branch-local."""

    state_interval = _normalize_state_interval_tuple(
        state_interval,
        expected_length=18,
    )
    binary_distance_threshold = float(binary_distance_threshold)
    max_depth = int(max_depth)
    max_branches = int(max_branches)
    min_position_width = float(min_position_width)
    if binary_distance_threshold <= 0.0:
        raise ValueError("binary_distance_threshold must be positive")
    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")
    if max_branches < 1:
        raise ValueError("max_branches must be positive")
    if min_position_width < 0.0:
        raise ValueError("min_position_width cannot be negative")

    pending: list[tuple[tuple[tuple[float, float], ...], int, str]] = [
        (state_interval, 0, "root")
    ]
    branches: list[SimultaneousClosePairPartitionBranch] = []
    split_count = 0
    stopped_by_branch_limit = False
    while pending:
        current_state, depth, branch_id = pending.pop(0)
        branch = _simultaneous_close_pair_branch(
            current_state,
            branch_id=branch_id,
            binary_distance_threshold=binary_distance_threshold,
            depth=depth,
            masses=masses,
        )
        if branch.certified:
            branches.append(branch)
            continue
        if branch.leaf_type in {
            "certified_triple_close_cluster",
            "threshold_adjacent_triple_close_cluster",
        }:
            branches.append(branch)
            continue
        split_axis = (
            None
            if depth >= max_depth
            else _simultaneous_close_pair_split_axis(
                branch,
                min_width=min_position_width,
            )
        )
        if split_axis is None:
            branches.append(branch)
            continue
        if len(branches) + len(pending) + 2 > max_branches:
            stopped_by_branch_limit = True
            branches.append(branch)
            continue
        left_state, right_state = _bisect_state_interval_tuple(current_state, split_axis)
        next_depth = depth + 1
        pending.append((left_state, next_depth, f"{branch_id}.L{split_axis}"))
        pending.append((right_state, next_depth, f"{branch_id}.R{split_axis}"))
        split_count += 1

    state_union = tuple(branch.state_interval for branch in branches)
    hull = _hull_state_interval_tuples(state_union)
    tolerance = 16.0 * np.finfo(float).eps * max(
        1.0,
        max(abs(bound) for interval in state_interval for bound in interval),
    )
    recursive_cover = bool(
        branches
        and all(_state_interval_tuple_subset(branch.state_interval, state_interval) for branch in branches)
        and _state_interval_tuples_same_bounds(hull, state_interval, tolerance=tolerance)
    )
    branch_cover = bool(recursive_cover and all(branch.well_formed for branch in branches))
    missing: list[str] = []
    if not recursive_cover:
        missing.append("simultaneous_close_pair_recursive_bisection_cover")
    if not branch_cover:
        missing.append("simultaneous_close_pair_branch_cover")
    for branch in branches:
        missing.extend(branch.missing_obligations)
    if stopped_by_branch_limit:
        missing.append("simultaneous_close_pair_branch_limit")
    return SimultaneousClosePairSplitCertificate(
        original_state_interval=state_interval,
        binary_distance_threshold=binary_distance_threshold,
        branches=tuple(branches),
        split_count=split_count,
        max_depth=max_depth,
        recursive_bisection_cover_certified=recursive_cover,
        branch_cover_certified=branch_cover,
        missing_obligations=tuple(dict.fromkeys(missing)),
    )


def certify_spatial_ks_to_ordinary_handoff_admissibility(
    endpoint_projection: SpatialKSBinaryPhysicalProjectionCertificate,
    *,
    retained_order: int,
    requested_post_time_interval: FloatInterval | tuple[float, float],
    ordinary_initial_state_interval: tuple[tuple[float, float], ...] | None = None,
    ordinary_solution: IntervalTaylorSolution | None = None,
    ordinary_residual: object | None = None,
    ordinary_tail: object | None = None,
    min_pair_distance_required: float = 0.0,
    max_acceleration_bound: float = np.inf,
    min_cauchy_radius: float = 0.0,
    max_residual_bound: float = np.inf,
    max_tail_bound: float = np.inf,
    require_ordinary_chart_evidence: bool = True,
) -> OrdinaryHandoffAdmissibilityCertificate:
    """Certify that a rho-positive KS endpoint is safe for an ordinary chart."""

    retained_order = int(retained_order)
    requested_post_time_interval = _coerce_float_interval(requested_post_time_interval)
    min_pair_distance_required = float(min_pair_distance_required)
    max_acceleration_bound = float(max_acceleration_bound)
    min_cauchy_radius = float(min_cauchy_radius)
    max_residual_bound = float(max_residual_bound)
    max_tail_bound = float(max_tail_bound)
    require_ordinary_chart_evidence = bool(require_ordinary_chart_evidence)
    missing: list[str] = []

    if retained_order < 1:
        missing.append("retained_order_not_positive")
    if requested_post_time_interval.lower < -1.0e-14:
        missing.append("post_handoff_time_interval_not_forward")
    if requested_post_time_interval.upper <= 0.0:
        missing.append("post_handoff_time_interval_not_positive")
    if min_pair_distance_required < 0.0:
        missing.append("negative_min_pair_distance_requirement")
    if min_cauchy_radius < 0.0:
        missing.append("negative_min_cauchy_radius_requirement")

    step_bound = _interval_max_abs(_forward_chart_domain(requested_post_time_interval))
    min_pair_distance_lower_bound = 0.0
    lower_squared_distance = 0.0
    acceleration_bound = np.inf
    cauchy_radius = 0.0
    cauchy_tail_bound = np.inf
    cauchy_majorant_certified = False
    cauchy_initial_state_interval = (
        endpoint_projection.state_interval
        if ordinary_initial_state_interval is None
        else ordinary_initial_state_interval
    )
    if endpoint_projection.certified and retained_order >= 1:
        try:
            cauchy = ordinary_interval_cauchy_majorant_tail_certificate(
                cauchy_initial_state_interval,
                endpoint_projection.masses,
                retained_order=retained_order,
                step_size=step_bound,
            )
            min_pair_distance_lower_bound = float(cauchy.min_pair_distance_lower_bound)
            lower_squared_distance = float(cauchy.lower_squared_distance)
            acceleration_bound = float(cauchy.acceleration_bound)
            cauchy_radius = float(cauchy.time_radius)
            cauchy_tail_bound = float(cauchy.tail_bound)
            cauchy_majorant_certified = cauchy.is_nontrivial
        except ValueError as exc:
            missing.append(f"ordinary_cauchy_majorant:{exc}")
    else:
        missing.append("rho_positive_endpoint_projection")

    residual_bound = np.inf
    residual_certified = not require_ordinary_chart_evidence and ordinary_residual is None
    if ordinary_residual is not None:
        residual_certified = bool(getattr(ordinary_residual, "certified", False))
        residual_bound = float(getattr(ordinary_residual, "max_residual_radius", np.inf))
    ordinary_initial_matches_projection = not require_ordinary_chart_evidence and ordinary_solution is None
    if ordinary_solution is not None:
        ordinary_initial_matches_projection = _ordinary_initial_matches_projection(
            ordinary_solution,
            cauchy_initial_state_interval,
        )

    guarded_tail_certified = False
    guarded_tail_bound = np.inf
    if ordinary_tail is not None:
        guarded_tail_certified = bool(getattr(ordinary_tail, "is_nontrivial", False))
        guarded_tail_bound = float(getattr(ordinary_tail, "tail_bound", np.inf))
    tail_candidates = [
        bound
        for bound, certified in (
            (cauchy_tail_bound, cauchy_majorant_certified),
            (guarded_tail_bound, guarded_tail_certified),
        )
        if certified and np.isfinite(bound)
    ]
    tail_certified = bool(tail_candidates)
    tail_bound = float(min(tail_candidates)) if tail_candidates else np.inf

    if min_cauchy_radius == 0.0 and step_bound > 0.0:
        min_cauchy_radius = float(step_bound)
    if not np.isfinite(max_acceleration_bound) and np.isfinite(acceleration_bound):
        max_acceleration_bound = float(acceleration_bound)
    if not np.isfinite(max_residual_bound) and np.isfinite(residual_bound):
        max_residual_bound = float(residual_bound)
    if not np.isfinite(max_tail_bound) and np.isfinite(tail_bound):
        max_tail_bound = float(tail_bound)

    if not cauchy_majorant_certified:
        missing.append("ordinary_cauchy_majorant_not_certified")
    if min_pair_distance_lower_bound < min_pair_distance_required:
        missing.append("ordinary_handoff_pair_distance_too_small")
    if acceleration_bound > max_acceleration_bound:
        missing.append("ordinary_handoff_acceleration_bound_too_large")
    if cauchy_radius < min_cauchy_radius:
        missing.append("ordinary_handoff_cauchy_radius_too_small")
    if require_ordinary_chart_evidence and ordinary_solution is None:
        missing.append("ordinary_handoff_constructed_chart_missing")
    if require_ordinary_chart_evidence and ordinary_residual is None:
        missing.append("ordinary_handoff_residual_certificate_missing")
    if not residual_certified:
        missing.append("ordinary_handoff_residual_not_certified")
    if residual_bound > max_residual_bound:
        missing.append("ordinary_handoff_residual_bound_too_large")
    if not tail_certified:
        missing.append("ordinary_handoff_tail_not_certified")
    if tail_bound > max_tail_bound:
        missing.append("ordinary_handoff_tail_bound_too_large")
    if not ordinary_initial_matches_projection:
        missing.append("ordinary_handoff_initial_state_not_projection")

    return OrdinaryHandoffAdmissibilityCertificate(
        retained_order=retained_order,
        requested_post_time_interval=requested_post_time_interval,
        min_pair_distance_required=min_pair_distance_required,
        max_acceleration_bound=max_acceleration_bound,
        min_cauchy_radius=min_cauchy_radius,
        max_residual_bound=max_residual_bound,
        max_tail_bound=max_tail_bound,
        min_pair_distance_lower_bound=float(min_pair_distance_lower_bound),
        lower_squared_distance=float(lower_squared_distance),
        acceleration_bound=float(acceleration_bound),
        cauchy_radius=float(cauchy_radius),
        residual_bound=float(residual_bound),
        tail_bound=float(tail_bound),
        feasible_retained_order=retained_order >= 1,
        endpoint_projection_certified=endpoint_projection.certified,
        ordinary_initial_matches_projection=ordinary_initial_matches_projection,
        ordinary_chart_evidence_required=require_ordinary_chart_evidence,
        residual_certified=residual_certified,
        tail_certified=tail_certified,
        cauchy_majorant_certified=cauchy_majorant_certified,
        missing_obligations=tuple(dict.fromkeys(missing)),
    )


def _interval_max_abs(interval: FloatInterval) -> float:
    return float(max(abs(interval.lower), abs(interval.upper)))


def _ks_physical_time_root_for_target(
    ks_solution: object,
    target_time: FloatInterval,
    *,
    s_upper: float,
) -> float:
    target = 0.5 * (target_time.lower + target_time.upper)
    if target < -1.0e-14:
        raise ValueError("KS target time cannot be negative")
    if abs(target) <= 1.0e-14:
        return 0.0
    coefficients = np.asarray(
        [
            0.5 * (
                _as_float_interval(coefficient).lower
                + _as_float_interval(coefficient).upper
            )
            for coefficient in np.asarray(ks_solution.physical_time, dtype=object)
        ],
        dtype=float,
    )

    def residual(s_value: float) -> float:
        return float(np.polynomial.polynomial.polyval(s_value, coefficients) - target)

    upper = float(s_upper)
    if upper <= 0.0:
        raise ValueError("KS target root requires positive s_upper")
    if residual(upper) < -1.0e-14:
        raise ValueError("KS target time is not reached before the certified endpoint")
    return float(brentq(residual, 0.0, upper, xtol=1e-15, rtol=1e-15))


def _ks_physical_time_roots_for_target_interval(
    ks_solution: object,
    target_time: FloatInterval,
    *,
    s_upper: float,
) -> FloatInterval:
    target_time = _coerce_float_interval(target_time)
    lower_target = FloatInterval.point(max(0.0, target_time.lower))
    upper_target = FloatInterval.point(max(0.0, target_time.upper))
    lower_root = _ks_physical_time_root_for_target(
        ks_solution,
        lower_target,
        s_upper=s_upper,
    )
    upper_root = _ks_physical_time_root_for_target(
        ks_solution,
        upper_target,
        s_upper=s_upper,
    )
    return FloatInterval(min(lower_root, upper_root), max(lower_root, upper_root))


def _spatial_ks_physical_time_interval_from_solution(
    ks_solution: object,
    parameter_interval: FloatInterval,
    *,
    tail_bound: float,
) -> FloatInterval:
    physical_time = _as_float_interval(
        interval_array_series_eval(
            np.asarray(ks_solution.physical_time, dtype=object)[:, None],
            parameter_interval,
        )[0]
    )
    return _inflate_interval(physical_time, tail_bound)


def _spatial_ks_target_time_interval_bracket_certified(
    ks_solution: object,
    parameter_interval: FloatInterval,
    target_time: FloatInterval,
    *,
    tail_bound: float,
    tolerance: float = 0.0,
) -> bool:
    parameter_interval = _coerce_float_interval(parameter_interval)
    target_time = _coerce_float_interval(target_time)
    tolerance = max(0.0, float(tolerance))
    left_time = _spatial_ks_physical_time_interval_from_solution(
        ks_solution,
        FloatInterval.point(parameter_interval.lower),
        tail_bound=tail_bound,
    )
    right_time = _spatial_ks_physical_time_interval_from_solution(
        ks_solution,
        FloatInterval.point(parameter_interval.upper),
        tail_bound=tail_bound,
    )
    return (
        left_time.upper <= target_time.lower + tolerance
        and right_time.lower >= target_time.upper - tolerance
    )


def _spatial_ks_interval_state_from_solution_interval(
    initial_state: IntervalSpatialKSBinaryChartState,
    *,
    ks_solution: object,
    parameter_interval: FloatInterval,
    tail_bound: float,
) -> IntervalSpatialKSBinaryChartState:
    return IntervalSpatialKSBinaryChartState(
        masses=initial_state.masses,
        pair=initial_state.pair,
        u=_inflate_interval_array(
            interval_array_series_eval(ks_solution.u, parameter_interval),
            tail_bound,
        ),
        u_velocity=_inflate_interval_array(
            interval_array_series_eval(ks_solution.u_velocity, parameter_interval),
            tail_bound,
        ),
        pair_energy=_inflate_interval(
            interval_array_series_eval(
                ks_solution.pair_energy[:, None],
                parameter_interval,
            )[0],
            tail_bound,
        ),
        binary_center=_inflate_interval_array(
            interval_array_series_eval(ks_solution.binary_center, parameter_interval),
            tail_bound,
        ),
        binary_center_velocity=_inflate_interval_array(
            interval_array_series_eval(
                ks_solution.binary_center_velocity,
                parameter_interval,
            ),
            tail_bound,
        ),
        third_offset=_inflate_interval_array(
            interval_array_series_eval(ks_solution.third_offset, parameter_interval),
            tail_bound,
        ),
        third_offset_velocity=_inflate_interval_array(
            interval_array_series_eval(
                ks_solution.third_offset_velocity,
                parameter_interval,
            ),
            tail_bound,
        ),
        branch_certificate=initial_state.branch_certificate,
    )


def _inflate_interval(interval: FloatInterval, radius: float) -> FloatInterval:
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0.0:
        return interval
    return FloatInterval(
        float(np.nextafter(interval.lower - radius, -np.inf)),
        float(np.nextafter(interval.upper + radius, np.inf)),
    )


def _ordinary_target_state_interval_over_interval(
    solution: IntervalTaylorSolution,
    time: FloatInterval,
    *,
    retained_order: int,
    tail_bound: float,
) -> Array:
    variable = _coerce_float_interval(time)
    retained_position = interval_array_series_eval(
        solution.position[: retained_order + 1],
        variable,
    )
    retained_velocity = interval_array_series_eval(
        solution.velocity[: retained_order + 1],
        variable,
    )
    state = np.concatenate([retained_position.reshape(-1), retained_velocity.reshape(-1)])
    return np.asarray(
        [_inflate_interval(component, tail_bound) for component in state],
        dtype=object,
    )


def _certify_spatial_local_collision_policy(
    *,
    pair: tuple[int, int],
    masses: Array,
    ordinary_entry_solution: IntervalTaylorSolution | None,
    ordinary_entry_domain: FloatInterval | None,
    ordinary_entry_tail_bound: float,
    ks_solution: object,
    ks_parameter_interval: FloatInterval | None,
    ks_tail_bound: float,
    ordinary_post_solution: IntervalTaylorSolution | None,
    ordinary_post_parameter_interval: FloatInterval | None,
    ordinary_post_tail_bound: float,
    retained_order: int,
    require_ordinary_entry: bool = True,
    competing_pair_min_distance_required: float = 0.0,
) -> SpatialLocalCollisionPolicyCertificate:
    pair = tuple(pair)
    competing_pair_min_distance_required = float(competing_pair_min_distance_required)
    missing: list[str] = []
    if require_ordinary_entry and ordinary_entry_solution is None:
        missing.append("ordinary_entry_solution_missing")
    if ordinary_entry_solution is not None and ordinary_entry_domain is None:
        missing.append("ordinary_entry_domain_missing")
        ordinary_entry_domain = FloatInterval.point(0.0)
    if ks_parameter_interval is None:
        missing.append("ks_parameter_domain_missing")
        ks_parameter_interval = FloatInterval.point(0.0)
    if ordinary_post_solution is not None and ordinary_post_parameter_interval is None:
        missing.append("ordinary_post_parameter_domain_missing")
        ordinary_post_parameter_interval = FloatInterval.point(0.0)
    if ordinary_entry_solution is not None and not np.isfinite(ordinary_entry_tail_bound):
        missing.append("ordinary_entry_tail_bound_not_finite")
    if not np.isfinite(ks_tail_bound):
        missing.append("ks_tail_bound_not_finite")
    if not np.isfinite(ordinary_post_tail_bound):
        missing.append("ordinary_post_tail_bound_not_finite")

    ordinary_entry_lowers = (
        ()
        if ordinary_entry_solution is None
        else _ordinary_pair_squared_distance_lowers(
            ordinary_entry_solution,
            ordinary_entry_domain,
            retained_order=retained_order,
            tail_bound=ordinary_entry_tail_bound,
        )
    )
    ks_competing_lowers = _ks_competing_pair_squared_distance_lowers(
        ks_solution,
        ks_parameter_interval,
        pair=pair,
        masses=masses,
        retained_order=retained_order,
        tail_bound=ks_tail_bound,
    )
    ks_competing_bounds = _ks_competing_pair_squared_distance_bounds(
        ks_solution,
        ks_parameter_interval,
        pair=pair,
        masses=masses,
        retained_order=retained_order,
        tail_bound=ks_tail_bound,
    )
    if ordinary_post_solution is None:
        ordinary_post_lowers = ()
    else:
        ordinary_post_domain = _forward_chart_domain(ordinary_post_parameter_interval)
        ordinary_post_lowers = _ordinary_pair_squared_distance_lowers(
            ordinary_post_solution,
            ordinary_post_domain,
            retained_order=retained_order,
            tail_bound=ordinary_post_tail_bound,
        )

    if require_ordinary_entry and not _all_squared_distance_lowers_positive(ordinary_entry_lowers):
        missing.append("ordinary_entry_domain_pair_collision_not_excluded")
    if not _all_squared_distance_lowers_positive(ks_competing_lowers):
        missing.append("ks_competing_third_body_collision_not_excluded")
    if competing_pair_min_distance_required < 0.0:
        missing.append("negative_competing_pair_min_distance_requirement")
    competing_squared_floor = float(competing_pair_min_distance_required) ** 2
    close_competing_pairs = tuple(
        competing_pair
        for competing_pair, lower in ks_competing_lowers
        if lower < competing_squared_floor
    )
    certified_close_competing_pairs = tuple(
        competing_pair
        for competing_pair, _lower, upper in ks_competing_bounds
        if np.isfinite(upper) and upper < competing_squared_floor
    )
    if competing_squared_floor > 0.0 and close_competing_pairs:
        missing.append(
            "ks_competing_close_binary_requires_next_regularized_chart_or_split"
        )
        if len(close_competing_pairs) > 1:
            missing.append("simultaneous_close_pair_partitioning")
        if len(certified_close_competing_pairs) > 1:
            missing.append(
                "spatial_triple_close_cluster_requires_cluster_blowup_or_total_collision_selector"
            )
    if ordinary_post_solution is not None and not _all_squared_distance_lowers_positive(ordinary_post_lowers):
        missing.append("ordinary_post_domain_pair_collision_not_excluded")

    return SpatialLocalCollisionPolicyCertificate(
        pair=pair,
        ordinary_entry_squared_distance_lowers=ordinary_entry_lowers,
        ks_competing_squared_distance_lowers=ks_competing_lowers,
        ordinary_post_squared_distance_lowers=ordinary_post_lowers,
        competing_pair_min_distance_required=competing_pair_min_distance_required,
        missing_obligations=tuple(missing),
    )


def _ordinary_pair_squared_distance_lowers(
    solution: IntervalTaylorSolution,
    time_domain: FloatInterval,
    *,
    retained_order: int,
    tail_bound: float,
) -> tuple[tuple[tuple[int, int], float], ...]:
    positions = interval_array_series_eval(
        solution.position[: retained_order + 1],
        time_domain,
    )
    positions = _inflate_interval_array(positions, tail_bound)
    lowers: list[tuple[tuple[int, int], float]] = []
    for first, second in _three_body_pairs():
        diff = np.array(
            [
                _as_float_interval(positions[second, axis])
                - _as_float_interval(positions[first, axis])
                for axis in range(3)
            ],
            dtype=object,
        )
        lowers.append(((first, second), _interval_vector_squared_norm(diff).lower))
    return tuple(lowers)


def _ks_competing_pair_squared_distance_lowers(
    solution: object,
    s_domain: FloatInterval,
    *,
    pair: tuple[int, int],
    masses: Array,
    retained_order: int,
    tail_bound: float,
) -> tuple[tuple[tuple[int, int], float], ...]:
    return tuple(
        (pair, lower)
        for pair, lower, _upper in _ks_competing_pair_squared_distance_bounds(
            solution,
            s_domain,
            pair=pair,
            masses=masses,
            retained_order=retained_order,
            tail_bound=tail_bound,
        )
    )


def _ks_competing_pair_squared_distance_bounds(
    solution: object,
    s_domain: FloatInterval,
    *,
    pair: tuple[int, int],
    masses: Array,
    retained_order: int,
    tail_bound: float,
) -> tuple[tuple[tuple[int, int], float, float], ...]:
    masses = np.asarray(masses, dtype=float)
    first, second = tuple(pair)
    third = _third_index_for_three_body_pair(pair)
    pair_mass = float(masses[first] + masses[second])
    alpha = float(masses[second] / pair_mass)
    beta = float(masses[first] / pair_mass)
    u_value = interval_array_series_eval(
        np.asarray(solution.u, dtype=object)[: retained_order + 1],
        s_domain,
    )
    u_value = _inflate_interval_array(u_value, tail_bound)
    relative_position = _ks_project_interval_value(u_value)
    third_offset = interval_array_series_eval(
        np.asarray(solution.third_offset, dtype=object)[: retained_order + 1],
        s_domain,
    )
    third_offset = _inflate_interval_array(third_offset, tail_bound)
    from_first_to_third = np.array(
        [
            _as_float_interval(third_offset[axis])
            + _as_float_interval(relative_position[axis]).scale(alpha)
            for axis in range(3)
        ],
        dtype=object,
    )
    from_second_to_third = np.array(
        [
            _as_float_interval(third_offset[axis])
            - _as_float_interval(relative_position[axis]).scale(beta)
            for axis in range(3)
        ],
        dtype=object,
    )
    return (
        (
            _ordered_pair(first, third),
            _interval_vector_squared_norm(from_first_to_third).lower,
            _interval_vector_squared_norm(from_first_to_third).upper,
        ),
        (
            _ordered_pair(second, third),
            _interval_vector_squared_norm(from_second_to_third).lower,
            _interval_vector_squared_norm(from_second_to_third).upper,
        ),
    )


def _all_squared_distance_lowers_positive(
    values: tuple[tuple[tuple[int, int], float], ...],
) -> bool:
    return bool(values and all(np.isfinite(lower) and lower > 0.0 for _pair, lower in values))


def _forward_chart_domain(interval: FloatInterval) -> FloatInterval:
    return FloatInterval(0.0, max(0.0, float(interval.upper)))


def _three_body_pairs() -> tuple[tuple[int, int], ...]:
    return ((0, 1), (0, 2), (1, 2))


def _ordered_pair(first: int, second: int) -> tuple[int, int]:
    return (min(int(first), int(second)), max(int(first), int(second)))


def _third_index_for_three_body_pair(pair: tuple[int, int]) -> int:
    remaining = {0, 1, 2} - set(tuple(pair))
    if len(remaining) != 1:
        raise ValueError("pair must contain two distinct body indices from {0, 1, 2}")
    return remaining.pop()


def _as_float_interval(value: object) -> FloatInterval:
    if isinstance(value, FloatInterval):
        return value
    return FloatInterval.point(float(value))


def _inflate_interval_array(values: Array, radius: float) -> Array:
    values = np.asarray(values, dtype=object)
    inflated = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        inflated[index] = _inflate_interval(_as_float_interval(values[index]), radius)
    return inflated


def _inflate_state_interval_tuples(
    state_interval: tuple[tuple[float, float], ...],
    radius: float,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        _inflate_interval(FloatInterval(float(lower), float(upper)), radius).as_tuple()
        for lower, upper in state_interval
    )


def _interval_square(interval: FloatInterval) -> FloatInterval:
    if interval.lower <= 0.0 <= interval.upper:
        high = max(interval.lower * interval.lower, interval.upper * interval.upper)
        return FloatInterval(0.0, float(np.nextafter(high, np.inf)))
    low = min(interval.lower * interval.lower, interval.upper * interval.upper)
    high = max(interval.lower * interval.lower, interval.upper * interval.upper)
    return FloatInterval(float(np.nextafter(low, -np.inf)), float(np.nextafter(high, np.inf)))


def _interval_vector_squared_norm(vector: Array) -> FloatInterval:
    total = FloatInterval.point(0.0)
    for value in np.asarray(vector, dtype=object).reshape(-1):
        total = total + _interval_square(_as_float_interval(value))
    return total


def _ks_project_interval_value(u_value: Array) -> Array:
    u_value = np.asarray(u_value, dtype=object).reshape(4)
    a, b, c, d = (_as_float_interval(value) for value in u_value)
    return np.array(
        [
            _interval_square(a) - _interval_square(b) - _interval_square(c) + _interval_square(d),
            (a * b - c * d).scale(2.0),
            (a * c + b * d).scale(2.0),
        ],
        dtype=object,
    )


def _ordinary_initial_matches_projection(
    solution: IntervalTaylorSolution,
    state_interval: tuple[tuple[float, float], ...],
) -> bool:
    expected_positions, expected_velocities = _spatial_interval_state_arrays(state_interval)
    return bool(
        _interval_arrays_equal(solution.position[0], expected_positions)
        and _interval_arrays_equal(solution.velocity[0], expected_velocities)
    )


def _interval_arrays_equal(left: Array, right: Array) -> bool:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape:
        return False
    for index in np.ndindex(left.shape):
        left_value = left[index]
        right_value = right[index]
        if (
            not isinstance(left_value, FloatInterval)
            or not isinstance(right_value, FloatInterval)
            or left_value.lower != right_value.lower
            or left_value.upper != right_value.upper
        ):
            return False
    return True


def _hybrid_invariant_count(hybrid_solution: object, name: str) -> int:
    step_name = f"{name}_certificate"
    return sum(
        bool(getattr(getattr(step, step_name, None), "certified", False))
        for step in getattr(hybrid_solution, "steps", ())
    )


def _flat_float_interval_array(
    intervals: tuple[tuple[float, float], ...],
) -> Array:
    return np.asarray(
        [
            FloatInterval(float(lower), float(upper))
            for lower, upper in intervals
        ],
        dtype=object,
    )


def _point_interval_array(values: object) -> Array:
    return np.asarray(
        [
            FloatInterval.point(float(value))
            for value in np.asarray(values, dtype=float).reshape(-1)
        ],
        dtype=object,
    )


def _chart_from_step(
    step: object,
    index: int,
    kind: str,
    parameter_name: str,
    source: str,
) -> ValidatedChart:
    return ValidatedChart(
        chart_id=f"chart_{index}",
        chart_type=kind,
        source=source,
        parameter_name=parameter_name,
        parameter_interval=_step_parameter_interval(step, parameter_name),
        physical_time_interval=_step_physical_time_interval(step),
        dynamics_certified=_step_dynamics_certified(step),
        residual_certified=_step_residual_certified(step),
        projection_certified=True,
        invariants_certified=_step_invariants_certified(step),
        tail_certified=_step_tail_certified(step),
        tail_bound=_step_tail_bound(step),
    )


def _target_chart_from_reduced_target(
    reduced_target: object,
    kind: str,
    parameter_name: str,
    source: str,
) -> ValidatedChart:
    return ValidatedChart(
        chart_id="target",
        chart_type=f"{kind}_target",
        source=source,
        parameter_name=parameter_name,
        parameter_interval=_target_parameter_interval(reduced_target, parameter_name),
        physical_time_interval=_target_physical_time_interval(reduced_target),
        dynamics_certified=bool(getattr(reduced_target, "certified", False)),
        residual_certified=_target_residual_certified(reduced_target),
        projection_certified=True,
        invariants_certified=_target_invariants_certified(reduced_target),
        tail_certified=_target_tail_certified(reduced_target),
        tail_bound=_target_tail_bound(reduced_target),
    )


def _target_kind_and_parameter(reduced_target: object) -> tuple[str, str]:
    if hasattr(reduced_target, "target_compact_parameter_interval"):
        return "compactified_sundman", "w"
    return "sundman", "s"


def _step_parameter_interval(step: object, parameter_name: str) -> FloatInterval | None:
    if parameter_name == "w" and hasattr(step, "start_compact_parameter"):
        lower = min(float(step.start_compact_parameter), float(step.end_compact_parameter))
        upper = max(float(step.start_compact_parameter), float(step.end_compact_parameter))
        return FloatInterval(lower, upper)
    if parameter_name == "s" and hasattr(step, "start_s"):
        lower = min(float(step.start_s), float(step.end_s))
        upper = max(float(step.start_s), float(step.end_s))
        return FloatInterval(lower, upper)
    return None


def _target_parameter_interval(reduced_target: object, parameter_name: str) -> FloatInterval | None:
    if parameter_name == "w" and hasattr(reduced_target, "target_compact_parameter_interval"):
        return reduced_target.target_compact_parameter_interval
    if parameter_name == "s" and hasattr(reduced_target, "target_s_interval"):
        return reduced_target.target_s_interval
    return None


def _target_physical_time_interval(reduced_target: object) -> FloatInterval:
    target_time = float(getattr(reduced_target, "target_time"))
    certificate = getattr(reduced_target, "target_certificate", None)
    bounds = [target_time]
    for name in (
        "start_time_interval",
        "global_time_at_lower",
        "global_time_at_upper",
    ):
        interval = getattr(certificate, name, None)
        if interval is None:
            continue
        if _float_interval_finite_nonempty(interval):
            lower, upper = _float_interval_bounds(interval)
            bounds.extend([lower, upper])
    return FloatInterval(float(min(bounds)), float(max(bounds)))


def _step_physical_time_interval(step: object) -> FloatInterval | None:
    if hasattr(step, "start_time_interval") and hasattr(step, "end_time_interval"):
        start = step.start_time_interval
        end = step.end_time_interval
        return FloatInterval(min(start.lower, end.lower), max(start.upper, end.upper))
    if hasattr(step, "physical_time_step_interval"):
        return step.physical_time_step_interval
    return None


def _step_dynamics_certified(step: object) -> bool:
    if hasattr(step, "proof_certified"):
        return bool(step.proof_certified)
    if hasattr(step, "certified"):
        return bool(step.certified)
    return bool(_step_residual_certified(step) and _step_invariants_certified(step))


def _step_residual_certified(step: object) -> bool:
    certificate = getattr(step, "residual_certificate", None)
    if certificate is None:
        certificate = getattr(step, "equation_residual_certificate", None)
    return bool(certificate is not None and certificate.certified)


def _step_projection_certified(step: object) -> bool:
    if hasattr(step, "projection_certified"):
        return bool(step.projection_certified)
    return True


def _step_invariants_certified(step: object) -> bool:
    names = (
        "angular_momentum_certificate",
        "energy_certificate",
        "linear_momentum_certificate",
        "center_of_mass_certificate",
    )
    return bool(all(getattr(getattr(step, name, None), "certified", False) for name in names))


def _step_tail_certified(step: object) -> bool:
    if hasattr(step, "tail_certified"):
        return bool(step.tail_certified)
    return getattr(step, "truncation_certificate", None) is not None


def _step_tail_bound(step: object) -> float:
    certificate = getattr(step, "tail_certificate", None)
    if certificate is None:
        certificate = getattr(step, "truncation_certificate", None)
    if certificate is None:
        return 0.0
    return float(getattr(certificate, "tail_bound", 0.0))


def _target_residual_certified(reduced_target: object) -> bool:
    certificate = getattr(reduced_target, "target_residual_certificate", None)
    if certificate is None:
        certificate = getattr(reduced_target, "target_equation_residual_certificate", None)
    return bool(certificate is not None and certificate.certified)


def _target_invariants_certified(reduced_target: object) -> bool:
    names = (
        "target_angular_momentum_certificate",
        "target_energy_certificate",
        "target_linear_momentum_certificate",
        "target_center_of_mass_certificate",
    )
    return bool(all(getattr(getattr(reduced_target, name, None), "certified", False) for name in names))


def _target_tail_certified(reduced_target: object) -> bool:
    certificate = getattr(reduced_target, "target_tail_certificate", None)
    if certificate is None:
        certificate = getattr(reduced_target, "target_truncation_certificate", None)
    return certificate is not None


def _target_tail_bound(reduced_target: object) -> float:
    certificate = getattr(reduced_target, "target_tail_certificate", None)
    if certificate is None:
        certificate = getattr(reduced_target, "target_truncation_certificate", None)
    if certificate is None:
        return 0.0
    return float(getattr(certificate, "tail_bound", 0.0))


def _transitions_from_charts(
    charts: tuple[ValidatedChart, ...],
    *,
    source: str,
) -> tuple[ValidatedTransition, ...]:
    transitions = []
    for left, right in zip(charts, charts[1:]):
        transitions.append(
            ValidatedTransition(
                source_chart_id=left.chart_id,
                target_chart_id=right.chart_id,
                transition_type="set_containment_handoff",
                certified=left.certified and right.certified,
                source=source,
            )
        )
    return tuple(transitions)


def _invariant_ledger_from_evaluation(
    evaluation: object,
    reduced_target: object | None,
    *,
    expected_chart_count: int,
) -> GlobalInvariantLedger:
    if reduced_target is None:
        certified = bool(getattr(evaluation, "reduction_certified", False))
        return GlobalInvariantLedger(
            center_of_mass_certified=certified,
            linear_momentum_certified=certified,
            angular_momentum_certified=certified,
            energy_certified=certified,
            certified_chart_count=1 if certified else 0,
            expected_chart_count=expected_chart_count,
        )

    center_count = _target_count(reduced_target, "center_of_mass")
    linear_count = _target_count(reduced_target, "linear_momentum")
    angular_count = _target_count(reduced_target, "angular_momentum")
    energy_count = _target_count(reduced_target, "energy")
    certified_chart_count = min(center_count, linear_count, angular_count, energy_count)
    return GlobalInvariantLedger(
        center_of_mass_certified=center_count >= expected_chart_count,
        linear_momentum_certified=linear_count >= expected_chart_count,
        angular_momentum_certified=angular_count >= expected_chart_count,
        energy_certified=energy_count >= expected_chart_count,
        certified_chart_count=certified_chart_count,
        expected_chart_count=expected_chart_count,
    )


def _residual_ledger_from_target(
    evaluation: object,
    reduced_target: object | None,
    *,
    expected_chart_count: int,
) -> NewtonResidualLedger:
    if reduced_target is None:
        certified = bool(getattr(evaluation, "dynamics_certified", False))
        return NewtonResidualLedger(
            certified=certified,
            certified_chart_count=1 if certified else 0,
            expected_chart_count=expected_chart_count,
        )
    count = sum(_step_residual_certified(step) for step in getattr(reduced_target, "steps", ())) + int(
        _target_residual_certified(reduced_target)
    )
    return NewtonResidualLedger(
        certified=count >= expected_chart_count,
        certified_chart_count=int(count),
        expected_chart_count=expected_chart_count,
    )


def _target_count(reduced_target: object, name: str) -> int:
    count_name = f"{name}_certified_step_count"
    if hasattr(reduced_target, count_name):
        return int(getattr(reduced_target, count_name))
    step_name = f"{name}_certificate"
    target_name = f"target_{name}_certificate"
    return sum(
        bool(getattr(getattr(step, step_name, None), "certified", False))
        for step in getattr(reduced_target, "steps", ())
    ) + int(bool(getattr(getattr(reduced_target, target_name, None), "certified", False)))
