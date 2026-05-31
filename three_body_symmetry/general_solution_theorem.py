"""Constructor-only theorem assembly for the Sundman-atlas route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .continuation import (
    certify_uniform_collision_free_ordinary_chart_ledgers,
    certify_uniformly_collision_free_taylor_recurrence,
)
from .escape_endpoint import (
    certify_homothetic_escape_branch_from_initial_data,
    certify_positive_energy_homothetic_all_real_gluing,
    certify_homothetic_escape_projection_invariants,
    certify_two_ended_scattering_invariant_match,
    derive_homothetic_escape_implicit_cauchy_majorant,
    derive_time_reversed_homothetic_escape_recurrence,
    construct_homothetic_escape_dyadic_recurrence,
)
from .event_regime_assembler import (
    derive_nonzero_angular_event_recurrence_from_uniform_envelopes,
    derive_nonzero_angular_event_recurrence_from_uniform_pair_envelopes,
)
from .global_invariants import certify_nonzero_angular_momentum_excludes_triple_collision
from .triple_collision import (
    certify_homothetic_total_collision_scalar_majorant,
    construct_homothetic_total_collision_branch,
)
from .validated_atlas import validated_atlas_from_homothetic_total_collision_branch
from .zero_angular_entry import certify_homothetic_finite_jet_identity_selector_entry


ORDINARY_FINITE_ATLAS_CHART_TYPES = (
    "initial_identity",
    "sundman",
    "sundman_target",
    "compactified_sundman",
    "compactified_sundman_target",
    "planar_ordinary_taylor",
    "spatial_ordinary_taylor_before_ks",
    "spatial_ordinary_taylor_after_ks",
)

PLANAR_BINARY_FINITE_ATLAS_CHART_TYPES = (
    "planar_levi_civita_binary",
)

SPATIAL_BINARY_FINITE_ATLAS_CHART_TYPES = (
    "spatial_ks_binary",
    "spatial_ks_event_order_branch_union",
    "spatial_branch_union",
)

BINARY_FINITE_ATLAS_CHART_TYPES = (
    *PLANAR_BINARY_FINITE_ATLAS_CHART_TYPES,
    *SPATIAL_BINARY_FINITE_ATLAS_CHART_TYPES,
)

TOTAL_COLLISION_FINITE_ATLAS_CHART_TYPES = (
    "finite_jet_identity_selector_total_collision",
    "automatic_identity_selector_total_collision",
    "identity_selector_total_collision",
)

REGIME_IDS = (
    "all_time_nonzero_angular",
    "compact_nonzero_angular_finite_events",
    "compact_zero_angular_finite_events_with_selector",
    "geometric_infinite_event_tail",
    "prescribed_two_ended_scattering",
    "positive_energy_homothetic_escape",
    "uniformly_noncollision_bounded_tail",
    "maximal_classical_until_total_collision",
)

GLOBAL_EXHAUSTION_REQUIRED_REGIME_IDS = (
    "all_time_nonzero_angular",
    "compact_nonzero_angular_finite_events",
    "compact_zero_angular_finite_events_with_selector",
    "geometric_infinite_event_tail",
    "prescribed_two_ended_scattering",
    "positive_energy_homothetic_escape",
    "uniformly_noncollision_bounded_tail",
    "maximal_classical_until_total_collision",
)


@dataclass(frozen=True)
class TheoremPipelineObligation:
    """One constructor-derived obligation in the theorem pipeline."""

    obligation: str
    certified: bool
    source: str
    required: bool = True
    detail: str = ""


@dataclass(frozen=True)
class PositiveMassNoncollisionInputDomainCertificate:
    """Input-domain certificate derived directly from masses and initial state."""

    masses: tuple[float, ...]
    positions: tuple[tuple[float, ...], ...]
    velocities: tuple[tuple[float, ...], ...]
    dimension: int
    min_pair_distance: float
    max_pair_distance: float
    max_body_speed: float
    finite_state: bool

    @property
    def positive_masses_certified(self) -> bool:
        return bool(self.masses and all(np.isfinite(mass) and mass > 0.0 for mass in self.masses))

    @property
    def noncollision_state_certified(self) -> bool:
        return bool(np.isfinite(self.min_pair_distance) and self.min_pair_distance > 0.0)

    @property
    def certified(self) -> bool:
        return bool(
            self.positive_masses_certified
            and self.dimension >= 1
            and self.finite_state
            and self.noncollision_state_certified
            and np.isfinite(self.max_pair_distance)
            and self.max_pair_distance >= self.min_pair_distance
            and np.isfinite(self.max_body_speed)
            and self.max_body_speed >= 0.0
        )


@dataclass(frozen=True)
class CompactTimeCoverageCertificate:
    """Scalar compact-time coverage derived from ``u=tanh(rate*t)``."""

    rate: float
    compact_lower: float = -1.0
    compact_upper: float = 1.0

    @property
    def certified(self) -> bool:
        return bool(
            np.isfinite(self.rate)
            and self.rate > 0.0
            and self.compact_lower == -1.0
            and self.compact_upper == 1.0
        )

    def compact_parameter(self, physical_time: float) -> float:
        physical_time = float(physical_time)
        if not np.isfinite(physical_time):
            raise ValueError("physical_time must be finite")
        return float(np.tanh(self.rate * physical_time))

    def physical_time(self, compact_parameter: float) -> float:
        compact_parameter = float(compact_parameter)
        if not -1.0 < compact_parameter < 1.0:
            raise ValueError("compact_parameter must lie in (-1, 1)")
        return float(np.arctanh(compact_parameter) / self.rate)


@dataclass(frozen=True)
class RegimeClassificationCertificate:
    """Global-regime classification assembled only from constructor outputs."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate
    compact_time_certificate: CompactTimeCoverageCertificate
    regime_id: str
    finite_middle_atlas: object | None
    event_regime_handoff: object | None
    event_tail_margin_certificate: object | None
    first_event_shell_prefix: object | None
    event_shell_invariance_certificate: object | None
    ordinary_gap_envelope: object | None
    separated_binary_envelope: object | None
    triple_collision_exclusion_certificate: object | None
    total_collision_selector_envelope: object | None
    escape_endpoint_envelope: object | None
    scattering_endpoint_envelope: object | None
    event_isolation_certificate: object | None
    primitive_cauchy_inputs: object | None
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def input_domain_certified(self) -> bool:
        return bool(getattr(self.input_domain_certificate, "certified", False))

    @property
    def compact_time_coverage_certified(self) -> bool:
        return bool(getattr(self.compact_time_certificate, "certified", False))

    @property
    def triple_collision_exclusion_certified(self) -> bool:
        return bool(getattr(self.triple_collision_exclusion_certificate, "certified", False))

    @property
    def certified(self) -> bool:
        return bool(
            self.regime_id in REGIME_IDS
            and self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class GlobalAtlasCertificate:
    """Regime-specific global atlas assembled from constructor certificates."""

    classification: RegimeClassificationCertificate
    validated_atlas: object | None
    finite_middle_atlas: object | None
    ordinary_gap_atlas: object | None
    event_budget: object | None
    scattering_atlas: object | None
    escape_atlas: object | None
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.classification.certified
            and self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        own_missing = tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )
        return (*self.classification.missing_obligations, *own_missing)


@dataclass(frozen=True)
class TwoSidedNonzeroAngularEventBudgetCertificate:
    """All-real event recurrence split into future and time-reversed-past halves."""

    future_event_budget: object | None
    past_event_budget: object | None
    source: str = "two_sided_nonzero_angular_event_budget"
    past_source: str = "future_recurrence_for_time_reversed_flow"

    @property
    def future_certified(self) -> bool:
        return _event_recurrence_certified(self.future_event_budget)

    @property
    def past_certified(self) -> bool:
        return _event_recurrence_certified(self.past_event_budget)

    @property
    def direction_provenance_certified(self) -> bool:
        future_direction = _event_budget_time_direction(self.future_event_budget)
        past_direction = _event_budget_time_direction(self.past_event_budget)
        future_ok = future_direction in {"future", "time_reversal_invariant"}
        past_ok = past_direction in {"time_reversed_past", "time_reversal_invariant"}
        same_budget = (
            self.future_event_budget is not None
            and self.future_event_budget is self.past_event_budget
        )
        return bool(
            future_ok
            and past_ok
            and (not same_budget or future_direction == "time_reversal_invariant")
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.future_certified
            and self.past_certified
            and self.direction_provenance_certified
            and self.future_event_budget is not None
            and self.past_event_budget is not None
        )

    @property
    def recurrence_closes(self) -> bool:
        return self.certified

    @property
    def family_kinds(self) -> tuple[str, ...]:
        names = set()
        for half in _event_recurrence_halves(self):
            names.update(_event_family_names(half))
        return tuple(sorted(names))

    @property
    def chart_family_counts(self) -> dict[str, int]:
        if self.future_event_budget is None:
            return {}
        counts = getattr(self.future_event_budget, "chart_family_counts", {})
        return {str(kind): int(count) for kind, count in dict(counts).items()}

    @property
    def chart_family_certificates(self) -> tuple[object, ...]:
        certificates = []
        for half in _event_recurrence_halves(self):
            certificates.extend(getattr(half, "chart_family_certificates", ()))
        return tuple(certificates)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = []
        if not self.future_certified:
            missing.append("future_all_future_event_budget")
        if not self.past_certified:
            missing.append("past_all_future_event_budget")
        if not self.direction_provenance_certified:
            missing.append("two_sided_event_time_direction_provenance")
        return tuple(missing)

    def family_certificate(self, kind: str) -> object:
        for half in _event_recurrence_halves(self):
            try:
                return half.family_certificate(kind)
            except Exception:
                continue
        raise KeyError(kind)


@dataclass(frozen=True)
class CompactNonzeroAngularFiniteAtlasCertificate:
    """Finite compact-interval atlas certified under nonzero angular momentum."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate
    validated_atlas: object
    chart_types: tuple[str, ...]
    ordinary_chart_count: int
    separated_binary_chart_count: int
    total_collision_chart_count: int
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class CompactOrdinaryBinaryFiniteAtlasCertificate:
    """Finite compact-interval ordinary/binary/selector atlas verifier."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate
    validated_atlas: object
    chart_types: tuple[str, ...]
    ordinary_chart_count: int
    separated_binary_chart_count: int
    total_collision_chart_count: int
    total_collision_policy_id: str
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class NonzeroAngularFiniteMiddleAtlasCertificate:
    """Two-sided compact middle atlas for an all-real nonzero-angular route."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate
    future_finite_atlas: CompactNonzeroAngularFiniteAtlasCertificate | None
    past_finite_atlas: CompactNonzeroAngularFiniteAtlasCertificate | None
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class NonzeroAngularEventRegimeHandoffCertificate:
    """Finite-middle endpoint containment in the event-tail ordinary envelope."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate
    compact_time_certificate: CompactTimeCoverageCertificate
    finite_middle_atlas: object | None
    future_envelope_spec: object | None
    past_envelope_spec: object | None
    future_compact_parameter: float
    past_compact_parameter: float
    future_metrics: Mapping[str, float]
    past_metrics: Mapping[str, float]
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def handoff_certified(self) -> bool:
        return self.certified

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class NonzeroAngularEventTailMarginCertificate:
    """All-future value-tail budget fits inside ordinary-envelope margins."""

    event_regime_handoff: object | None
    two_sided_event_budget: object | None
    future_value_tail_bound: float
    past_value_tail_bound: float
    future_required_coordinate_margin: float
    past_required_coordinate_margin: float
    future_metric_margins: Mapping[str, float]
    past_metric_margins: Mapping[str, float]
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def tail_margin_certified(self) -> bool:
        return self.certified

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class NonzeroAngularFirstEventShellPrefixCertificate:
    """Finite validated-atlas prefix through the first event shell."""

    event_regime_handoff: object | None
    future_prefix_atlas: object | None
    past_prefix_atlas: object | None
    future_end_compact_parameter: float
    past_end_compact_parameter: float
    future_end_metrics: Mapping[str, float]
    past_end_metrics: Mapping[str, float]
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def prefix_certified(self) -> bool:
        return self.certified

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class NonzeroAngularEventShellInvarianceCertificate:
    """Sufficient all-future shell-invariance bridge after the first shell."""

    event_regime_handoff: object | None
    event_tail_margin_certificate: object | None
    first_event_shell_prefix: object | None
    two_sided_event_budget: object | None
    future_remaining_value_tail_bound: float
    past_remaining_value_tail_bound: float
    future_required_coordinate_margin: float
    past_required_coordinate_margin: float
    future_prefix_metric_margins: Mapping[str, float]
    past_prefix_metric_margins: Mapping[str, float]
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def shell_invariance_certified(self) -> bool:
        return self.certified

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class CompactZeroAngularFiniteSelectorAtlasCertificate:
    """Finite compact atlas with explicit zero-angular total-collision selectors."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate
    validated_atlas: object
    total_collision_selector_envelopes: tuple[object, ...]
    chart_types: tuple[str, ...]
    ordinary_chart_count: int
    separated_binary_chart_count: int
    total_collision_chart_count: int
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def identity_selector_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class MaximalClassicalUntilTotalCollisionStopCertificate:
    """Finite classical atlas with an explicit stop-before-total-collision policy."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate
    validated_atlas: object
    finite_atlas_certificate: object
    chart_types: tuple[str, ...]
    total_collision_policy: str
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class PrescribedTwoEndedScatteringCertificate:
    """Prescribed scattering atlas plus invariant matching."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate
    scattering_atlas: object
    invariant_match_certificate: object
    middle_invariant_match_certificate: object
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def recurrence_closes(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class PrescribedScatteringMiddleInvariantMatchCertificate:
    """Finite middle state compatibility with a prescribed scattering endpoint."""

    middle_momentum: tuple[float, ...]
    endpoint_momentum: tuple[float, ...]
    middle_center_offset: tuple[float, ...]
    endpoint_center_offset: tuple[float, ...]
    middle_angular_momentum: float
    endpoint_angular_momentum: float
    middle_energy: float
    endpoint_energy: float
    tolerance: float
    dimension_supported: bool
    finite_invariants: bool

    @property
    def momentum_matches(self) -> bool:
        return (
            _max_abs_tuple_difference(
                self.middle_momentum,
                self.endpoint_momentum,
            )
            <= self.tolerance
        )

    @property
    def center_offset_matches(self) -> bool:
        return (
            _max_abs_tuple_difference(
                self.middle_center_offset,
                self.endpoint_center_offset,
            )
            <= self.tolerance
        )

    @property
    def angular_momentum_matches(self) -> bool:
        return (
            abs(self.middle_angular_momentum - self.endpoint_angular_momentum)
            <= self.tolerance
        )

    @property
    def energy_matches(self) -> bool:
        return abs(self.middle_energy - self.endpoint_energy) <= self.tolerance

    @property
    def certified(self) -> bool:
        return bool(
            self.tolerance > 0.0
            and self.dimension_supported
            and self.finite_invariants
            and self.momentum_matches
            and self.center_offset_matches
            and self.angular_momentum_matches
            and self.energy_matches
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.dimension_supported:
            missing.append("planar_middle_state")
        if not self.finite_invariants:
            missing.append("finite_middle_and_endpoint_invariants")
        if not self.momentum_matches:
            missing.append("middle_endpoint_momentum_match")
        if not self.center_offset_matches:
            missing.append("middle_endpoint_center_offset_match")
        if not self.angular_momentum_matches:
            missing.append("middle_endpoint_angular_momentum_match")
        if not self.energy_matches:
            missing.append("middle_endpoint_energy_match")
        return tuple(missing)


@dataclass(frozen=True)
class PositiveEnergyHomotheticEscapeCertificate:
    """Full-data positive-energy homothetic escape plus endpoint recurrence."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate
    branch_certificate: object
    projection_invariant_certificate: object
    recurrence: object | None
    past_recurrence: object | None
    total_collision_atlas: object | None
    selector_entry: object | None
    all_real_gluing: object | None
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def recurrence_closes(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class UniformlyNoncollisionBoundedTailCertificate:
    """Uniform collision-free ordinary recurrence for a bounded branch."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate
    recurrence: object
    ordinary_chart_ledgers: object
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def recurrence_closes(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


@dataclass(frozen=True)
class NonzeroAngularUniformPairEventEnvelopeSpec:
    """Uniform all-pair event-envelope inputs for one compact-time endpoint."""

    time_direction: str
    delta_initial: float
    theta: float
    event_isolation_initial: float
    boundary_clearance_initial: float
    ordinary_pair_distance_lower_bound: float
    ordinary_pair_diameter_upper_bound: float
    ordinary_speed_upper_bound: float
    binary_pair_envelopes: Mapping[tuple[int, int], Mapping[str, object]]
    step_ratio_bounds: Mapping[str, float]
    retained_order_initials: Mapping[str, int]
    retained_order_increments: Mapping[str, int]
    checked_prefix: int
    source: str = "uniform_pair_event_envelope_spec"


@dataclass(frozen=True)
class GeneralSolutionTheoremCertificate:
    """Scoped endpoint-regime assembler for the constructive Sundman-atlas route.

    The finite-target/open-time theorem is now the central unrestricted route.
    This object remains useful as an optional endpoint-compression theorem for
    regimes where stronger asymptotic certificates replace infinitely many
    compact finite-target certificates.
    """

    global_atlas: GlobalAtlasCertificate
    global_regime_exhaustion_certificate: object | None
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "constructive_sundman_atlas_general_solution"

    @property
    def regime_theorem_certified(self) -> bool:
        return bool(self.global_atlas.certified)

    @property
    def full_general_solution_certified(self) -> bool:
        return bool(
            self.regime_theorem_certified
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        atlas_missing = self.global_atlas.missing_obligations
        own_missing = tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )
        return (*atlas_missing, *own_missing)

    @property
    def route_summary(self) -> str:
        if self.full_general_solution_certified:
            return "constructive Sundman-atlas general solution theorem certified"
        if self.regime_theorem_certified:
            return "regime theorem certified; global regime exhaustion remains open"
        return "constructive Sundman-atlas theorem assembly remains incomplete"


@dataclass(frozen=True)
class GlobalRegimeExhaustionCertificate:
    """Typed exhaustion gate for the optional endpoint-compression theorem.

    A list of certified scoped regimes is not itself a partition theorem.  The
    endpoint-compression route therefore accepts only this typed object, and
    this constructor keeps the missing arbitrary-data partition theorem explicit
    until a real classifier proves the endpoint alternatives exhaustive.  The
    central unrestricted path instead goes through the finite-target/open-time
    atlas-or-stop theorem.
    """

    candidate_regimes: tuple[GlobalAtlasCertificate, ...]
    input_scope: str
    required_regime_ids: tuple[str, ...]
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_role: str = "optional_endpoint_compression_theorem"
    source: str = "global_regime_exhaustion_constructor"

    @property
    def covered_regime_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(getattr(atlas.classification, "regime_id", "missing"))
                    for atlas in self.candidate_regimes
                }
            )
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def candidate_input_domains_consistent(self) -> bool:
        return _candidate_regime_input_domains_consistent(self.candidate_regimes)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        )


def certify_positive_mass_noncollision_input_domain(
    masses: Any,
    positions: Any,
    velocities: Any,
) -> PositiveMassNoncollisionInputDomainCertificate:
    """Certify positive masses and a finite noncollision initial state."""

    masses_array = np.asarray(masses, dtype=float)
    positions_array = np.asarray(positions, dtype=float)
    velocities_array = np.asarray(velocities, dtype=float)
    if masses_array.ndim != 1 or positions_array.ndim != 2:
        raise ValueError("masses must be one-dimensional and positions two-dimensional")
    if velocities_array.shape != positions_array.shape:
        raise ValueError("positions and velocities must have matching shape")
    if positions_array.shape[0] != masses_array.shape[0]:
        raise ValueError("one mass is required for each body")
    if masses_array.shape[0] < 2:
        raise ValueError("at least two bodies are required")
    pair_distances = [
        float(np.linalg.norm(positions_array[j] - positions_array[i]))
        for i in range(positions_array.shape[0])
        for j in range(i + 1, positions_array.shape[0])
    ]
    if not pair_distances:
        raise ValueError("at least one pair distance is required")
    certificate = PositiveMassNoncollisionInputDomainCertificate(
        masses=tuple(float(mass) for mass in masses_array),
        positions=tuple(
            tuple(float(value) for value in row)
            for row in positions_array
        ),
        velocities=tuple(
            tuple(float(value) for value in row)
            for row in velocities_array
        ),
        dimension=int(positions_array.shape[1]),
        min_pair_distance=float(min(pair_distances)),
        max_pair_distance=float(max(pair_distances)),
        max_body_speed=float(max(np.linalg.norm(velocity) for velocity in velocities_array)),
        finite_state=bool(
            np.all(np.isfinite(masses_array))
            and np.all(np.isfinite(positions_array))
            and np.all(np.isfinite(velocities_array))
        ),
    )
    if not certificate.certified:
        raise ValueError("input domain is not positive-mass noncollision data")
    return certificate


def certify_prescribed_scattering_middle_invariant_match(
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    invariant_match_certificate: object,
    *,
    tolerance: float = 1.0e-10,
) -> PrescribedScatteringMiddleInvariantMatchCertificate:
    """Certify that the finite middle state lies on the endpoint invariant level."""

    _reject_raw_bool("input_domain_certificate", input_domain_certificate)
    _reject_raw_bool("invariant_match_certificate", invariant_match_certificate)
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    masses_array = np.asarray(input_domain_certificate.masses, dtype=float)
    positions_array = np.asarray(input_domain_certificate.positions, dtype=float)
    velocities_array = np.asarray(input_domain_certificate.velocities, dtype=float)
    dimension_supported = bool(positions_array.ndim == 2 and positions_array.shape[1] == 2)

    middle_momentum = np.sum(masses_array[:, None] * velocities_array, axis=0)
    middle_center_offset = np.sum(masses_array[:, None] * positions_array, axis=0)
    middle_angular = (
        _weighted_planar_wedge_sum(
            masses_array,
            positions_array,
            velocities_array,
        )
        if dimension_supported
        else float("nan")
    )
    middle_energy = _newtonian_energy(positions_array, velocities_array, masses_array)

    endpoint_momentum = _tuple_field(invariant_match_certificate, "future_momentum")
    endpoint_center_offset = _tuple_field(
        invariant_match_certificate,
        "future_center_offset",
    )
    endpoint_angular = float(
        getattr(invariant_match_certificate, "future_angular_momentum", float("nan"))
    )
    endpoint_energy = float(
        getattr(invariant_match_certificate, "future_energy", float("nan"))
    )
    finite_invariants = bool(
        np.all(np.isfinite(middle_momentum))
        and np.all(np.isfinite(middle_center_offset))
        and np.isfinite(middle_angular)
        and np.isfinite(middle_energy)
        and all(np.isfinite(value) for value in endpoint_momentum)
        and all(np.isfinite(value) for value in endpoint_center_offset)
        and np.isfinite(endpoint_angular)
        and np.isfinite(endpoint_energy)
    )

    return PrescribedScatteringMiddleInvariantMatchCertificate(
        middle_momentum=tuple(float(value) for value in middle_momentum),
        endpoint_momentum=endpoint_momentum,
        middle_center_offset=tuple(float(value) for value in middle_center_offset),
        endpoint_center_offset=endpoint_center_offset,
        middle_angular_momentum=float(middle_angular),
        endpoint_angular_momentum=endpoint_angular,
        middle_energy=float(middle_energy),
        endpoint_energy=endpoint_energy,
        tolerance=tolerance,
        dimension_supported=dimension_supported,
        finite_invariants=finite_invariants,
    )


def certify_compact_time_real_line_coverage(rate: float) -> CompactTimeCoverageCertificate:
    """Certify that ``u=tanh(rate*t)`` covers all finite real physical times."""

    certificate = CompactTimeCoverageCertificate(rate=float(rate))
    if not certificate.certified:
        raise ValueError("compact-time rate must be positive")
    return certificate


def certify_two_sided_nonzero_angular_event_budget(
    *,
    future_event_regime_assembly: object | None,
    past_event_regime_assembly: object | None,
    past_source: str = "future_recurrence_for_time_reversed_flow",
) -> TwoSidedNonzeroAngularEventBudgetCertificate:
    """Require separate future and time-reversed-past all-future recurrences.

    A geometric all-future event recurrence closes only the compact-time end
    reached as physical time tends to ``+infinity``.  The all-real
    nonzero-angular theorem needs the same recurrence at ``-infinity``.  The
    past half may be an independently constructed past certificate or the same
    absolute-envelope constructor applied to the time-reversed initial
    velocities.
    """

    _reject_raw_bool("future_event_regime_assembly", future_event_regime_assembly)
    _reject_raw_bool("past_event_regime_assembly", past_event_regime_assembly)
    return TwoSidedNonzeroAngularEventBudgetCertificate(
        future_event_budget=future_event_regime_assembly,
        past_event_budget=past_event_regime_assembly,
        past_source=str(past_source),
    )


def classify_global_regime(
    *,
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    compact_time_certificate: CompactTimeCoverageCertificate,
    regime_id: str,
    finite_middle_atlas: object | None = None,
    event_regime_handoff: object | None = None,
    event_tail_margin_certificate: object | None = None,
    first_event_shell_prefix: object | None = None,
    event_shell_invariance_certificate: object | None = None,
    ordinary_gap_envelope: object | None = None,
    separated_binary_envelope: object | None = None,
    triple_collision_exclusion_certificate: object | None = None,
    total_collision_selector_envelope: object | None = None,
    escape_endpoint_envelope: object | None = None,
    scattering_endpoint_envelope: object | None = None,
    event_isolation_certificate: object | None = None,
    primitive_cauchy_inputs: object | None = None,
) -> RegimeClassificationCertificate:
    """Classify one global regime from constructor-derived certificates."""

    if regime_id not in REGIME_IDS:
        raise ValueError(f"unknown regime_id {regime_id!r}")
    obligations = [
        _required_constructor_obligation(
            "positive_mass_noncollision_input_domain",
            input_domain_certificate,
            ("certified",),
        ),
        _required_constructor_obligation(
            "compact_time_real_line_coverage",
            compact_time_certificate,
            ("certified",),
        ),
    ]
    regime_requirements = _classification_requirements_for_regime(
        regime_id,
        input_domain_certificate=input_domain_certificate,
        compact_time_certificate=compact_time_certificate,
        finite_middle_atlas=finite_middle_atlas,
        event_regime_handoff=event_regime_handoff,
        event_tail_margin_certificate=event_tail_margin_certificate,
        first_event_shell_prefix=first_event_shell_prefix,
        event_shell_invariance_certificate=event_shell_invariance_certificate,
        ordinary_gap_envelope=ordinary_gap_envelope,
        separated_binary_envelope=separated_binary_envelope,
        triple_collision_exclusion_certificate=triple_collision_exclusion_certificate,
        total_collision_selector_envelope=total_collision_selector_envelope,
        escape_endpoint_envelope=escape_endpoint_envelope,
        scattering_endpoint_envelope=scattering_endpoint_envelope,
        event_isolation_certificate=event_isolation_certificate,
        primitive_cauchy_inputs=primitive_cauchy_inputs,
    )
    obligations.extend(regime_requirements)
    return RegimeClassificationCertificate(
        input_domain_certificate=input_domain_certificate,
        compact_time_certificate=compact_time_certificate,
        regime_id=str(regime_id),
        finite_middle_atlas=finite_middle_atlas,
        event_regime_handoff=event_regime_handoff,
        event_tail_margin_certificate=event_tail_margin_certificate,
        first_event_shell_prefix=first_event_shell_prefix,
        event_shell_invariance_certificate=event_shell_invariance_certificate,
        ordinary_gap_envelope=ordinary_gap_envelope,
        separated_binary_envelope=separated_binary_envelope,
        triple_collision_exclusion_certificate=triple_collision_exclusion_certificate,
        total_collision_selector_envelope=total_collision_selector_envelope,
        escape_endpoint_envelope=escape_endpoint_envelope,
        scattering_endpoint_envelope=scattering_endpoint_envelope,
        event_isolation_certificate=event_isolation_certificate,
        primitive_cauchy_inputs=primitive_cauchy_inputs,
        obligations=tuple(obligations),
    )


def construct_global_atlas_for_regime(
    classification: RegimeClassificationCertificate,
    *,
    validated_atlas: object | None = None,
    finite_middle_atlas: object | None = None,
    ordinary_gap_atlas: object | None = None,
    event_budget: object | None = None,
    scattering_atlas: object | None = None,
    escape_atlas: object | None = None,
) -> GlobalAtlasCertificate:
    """Assemble the regime-specific atlas object family."""

    _reject_raw_bool("classification", classification)
    obligations = [
        TheoremPipelineObligation(
            obligation="regime_classification",
            certified=classification.certified,
            source=type(classification).__name__,
            detail=classification.regime_id,
        )
    ]
    if classification.regime_id in (
        "geometric_infinite_event_tail",
        "all_time_nonzero_angular",
    ):
        selected_event_budget = event_budget or classification.primitive_cauchy_inputs
        selected_finite_middle = finite_middle_atlas or classification.finite_middle_atlas
        event_budget_obligation = (
            "two_sided_all_time_event_budget"
            if classification.regime_id == "all_time_nonzero_angular"
            else "all_future_event_budget"
        )
        obligations.append(
            _required_constructor_obligation(
                event_budget_obligation,
                selected_event_budget,
                ("recurrence_closes", "certified"),
            )
        )
        obligations.append(
            _classification_object_match_obligation(
                "global_atlas_event_budget_matches_classification",
                selected_event_budget,
                classification.primitive_cauchy_inputs,
                "primitive_cauchy_inputs",
            )
        )
        if classification.regime_id == "all_time_nonzero_angular":
            obligations.append(
                _classification_object_match_obligation(
                    "global_atlas_finite_middle_matches_classification",
                    selected_finite_middle,
                    classification.finite_middle_atlas,
                    "finite_middle_atlas",
                )
            )
        event_budget = selected_event_budget
        finite_middle_atlas = selected_finite_middle
    elif classification.regime_id == "uniformly_noncollision_bounded_tail":
        selected_ordinary_gap = ordinary_gap_atlas or classification.ordinary_gap_envelope
        obligations.append(
            _required_constructor_obligation(
                "uniform_collision_free_ordinary_recurrence",
                selected_ordinary_gap,
                ("recurrence_closes", "certified", "proof_certified"),
            )
        )
        obligations.append(
            _classification_object_match_obligation(
                "global_atlas_ordinary_gap_matches_classification",
                selected_ordinary_gap,
                classification.ordinary_gap_envelope,
                "ordinary_gap_envelope",
            )
        )
        ordinary_gap_atlas = selected_ordinary_gap
    elif classification.regime_id == "prescribed_two_ended_scattering":
        selected_scattering = scattering_atlas or classification.scattering_endpoint_envelope
        obligations.append(
            _required_constructor_obligation(
                "scattering_endpoint_atlas",
                selected_scattering,
                ("certified", "recurrence_closes"),
            )
        )
        obligations.append(
            _classification_object_match_obligation(
                "global_atlas_scattering_matches_classification",
                selected_scattering,
                classification.scattering_endpoint_envelope,
                "scattering_endpoint_envelope",
            )
        )
        scattering_atlas = selected_scattering
    elif classification.regime_id == "positive_energy_homothetic_escape":
        selected_escape = escape_atlas or classification.escape_endpoint_envelope
        obligations.append(
            _required_constructor_obligation(
                "escape_endpoint_atlas",
                selected_escape,
                ("certified", "recurrence_closes"),
            )
        )
        obligations.append(
            _classification_object_match_obligation(
                "global_atlas_escape_matches_classification",
                selected_escape,
                classification.escape_endpoint_envelope,
                "escape_endpoint_envelope",
            )
        )
        escape_atlas = selected_escape
    elif classification.regime_id == "maximal_classical_until_total_collision":
        selected_ordinary_gap = ordinary_gap_atlas or classification.ordinary_gap_envelope
        obligations.append(
            _required_constructor_obligation(
                "maximal_classical_until_total_collision_stop",
                selected_ordinary_gap,
                ("certified", "proof_certified"),
            )
        )
        obligations.append(
            _classification_object_match_obligation(
                "global_atlas_ordinary_gap_matches_classification",
                selected_ordinary_gap,
                classification.ordinary_gap_envelope,
                "ordinary_gap_envelope",
            )
        )
        obligations.append(
            _required_constructor_obligation(
                "validated_atlas_solution",
                validated_atlas,
                ("proof_certified", "certified"),
            )
        )
        obligations.append(
            _classification_validated_atlas_match_obligation(
                classification,
                validated_atlas,
            )
        )
        ordinary_gap_atlas = selected_ordinary_gap
    else:
        obligations.append(
            _required_constructor_obligation(
                "validated_atlas_solution",
                validated_atlas,
                ("proof_certified", "certified"),
            )
        )
        obligations.append(
            _classification_validated_atlas_match_obligation(
                classification,
                validated_atlas,
            )
        )
    return GlobalAtlasCertificate(
        classification=classification,
        validated_atlas=validated_atlas,
        finite_middle_atlas=finite_middle_atlas,
        ordinary_gap_atlas=ordinary_gap_atlas,
        event_budget=event_budget,
        scattering_atlas=scattering_atlas,
        escape_atlas=escape_atlas,
        obligations=tuple(obligations),
    )


def certify_compact_ordinary_binary_finite_atlas(
    *,
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object,
    total_collision_policy_id: str = "maximal_classical_stop",
) -> CompactOrdinaryBinaryFiniteAtlasCertificate:
    """Certify a finite ordinary/binary atlas independent of angular momentum."""

    _reject_raw_bool("input_domain_certificate", input_domain_certificate)
    _reject_raw_bool("validated_atlas", validated_atlas)
    total_collision_policy_id = str(total_collision_policy_id)
    chart_types = tuple(
        str(getattr(chart, "chart_type", "missing"))
        for chart in getattr(validated_atlas, "charts", ())
    )
    ordinary_types = set(ORDINARY_FINITE_ATLAS_CHART_TYPES)
    binary_types = set(BINARY_FINITE_ATLAS_CHART_TYPES)
    total_collision_types = set(TOTAL_COLLISION_FINITE_ATLAS_CHART_TYPES)
    planar_binary_types = set(PLANAR_BINARY_FINITE_ATLAS_CHART_TYPES)
    spatial_binary_types = set(SPATIAL_BINARY_FINITE_ATLAS_CHART_TYPES)
    ordinary_count = sum(chart_type in ordinary_types for chart_type in chart_types)
    binary_count = sum(chart_type in binary_types for chart_type in chart_types)
    total_collision_count = sum(chart_type in total_collision_types for chart_type in chart_types)
    planar_binary_count = sum(chart_type in planar_binary_types for chart_type in chart_types)
    spatial_binary_count = sum(chart_type in spatial_binary_types for chart_type in chart_types)
    unknown_chart_types = tuple(
        chart_type
        for chart_type in chart_types
        if chart_type not in ordinary_types
        and chart_type not in binary_types
        and chart_type not in total_collision_types
    )
    selected_total_collision_policy = total_collision_policy_id.startswith("selected_")
    obligations = (
        _required_constructor_obligation(
            "positive_mass_noncollision_input_domain",
            input_domain_certificate,
            ("certified",),
        ),
        _required_constructor_obligation(
            "validated_atlas_solution",
            validated_atlas,
            ("proof_certified",),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_mass_consistency",
            certified=_finite_atlas_mass_consistency_certified(
                input_domain_certificate,
                validated_atlas,
            ),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_mass_consistency_detail(
                input_domain_certificate,
                validated_atlas,
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_initial_state_consistency",
            certified=_finite_atlas_initial_state_consistency_certified(
                input_domain_certificate,
                validated_atlas,
            ),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_initial_state_consistency_detail(
                input_domain_certificate,
                validated_atlas,
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_newton_residual_ledger",
            certified=_finite_atlas_residual_ledger_certified(validated_atlas, len(chart_types)),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_residual_detail(validated_atlas),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_projection_domain_ledger",
            certified=_finite_atlas_projection_domain_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="all finite charts must carry certified dynamics, finite chart domains, and projection witnesses",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_invariant_ledger",
            certified=_finite_atlas_invariant_ledger_certified(validated_atlas, len(chart_types)),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_invariant_detail(validated_atlas),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_tail_budget",
            certified=_finite_atlas_tail_budget_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="global tail budget and every chart tail bound must be finite and certified",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_transition_ledger",
            certified=_finite_atlas_transition_ledger_certified(validated_atlas, len(chart_types)),
            source=type(validated_atlas).__name__,
            detail=(
                f"chart_count={len(chart_types)}; "
                f"transition_count={len(getattr(validated_atlas, 'transitions', ()))}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_collision_policy",
            certified=bool(getattr(getattr(validated_atlas, "collision_policy", None), "certified", False)),
            source=type(validated_atlas).__name__,
            detail=getattr(getattr(validated_atlas, "collision_policy", None), "binary_policy", "missing"),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_target_interval",
            certified=_finite_atlas_target_interval_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="target interval must be a nonempty finite coordinate interval array",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_target_time_domain",
            certified=bool(getattr(validated_atlas, "target_time_certified", False)),
            source=type(validated_atlas).__name__,
            detail="requested physical target time must lie in the final chart time interval",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_targeting_or_containment",
            certified=_finite_atlas_targeting_or_containment_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="requires target_time, finite_time_physical_targeting, or target_containment ledger evidence",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_proof_ledger",
            certified=bool(getattr(getattr(validated_atlas, "proof_ledger", None), "certified", False)),
            source=type(validated_atlas).__name__,
            detail="typed proof ledger must certify every required finite-atlas entry",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_selector_trace",
            certified=_finite_atlas_selector_trace_certified(validated_atlas, chart_types),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_selector_trace_detail(validated_atlas, chart_types),
        ),
        TheoremPipelineObligation(
            obligation="finite_chart_family_count",
            certified=bool(len(chart_types) > 0 and np.isfinite(len(chart_types))),
            source=type(validated_atlas).__name__,
            detail=f"chart_count={len(chart_types)}",
        ),
        TheoremPipelineObligation(
            obligation="ordinary_binary_or_selected_total_collision_chart_families",
            certified=bool(
                chart_types
                and ordinary_count + binary_count + total_collision_count == len(chart_types)
            ),
            source=type(validated_atlas).__name__,
            detail="unknown="
            + ",".join(unknown_chart_types)
            + "; chart_types="
            + ",".join(chart_types),
        ),
        TheoremPipelineObligation(
            obligation="total_collision_policy_matches_chart_family",
            certified=bool(total_collision_count == 0 or selected_total_collision_policy),
            source=type(validated_atlas).__name__,
            detail=(
                f"total_collision_chart_count={total_collision_count}; "
                f"total_collision_policy_id={total_collision_policy_id}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="spatial_binary_regularization_scope",
            certified=_finite_atlas_binary_scope_certified(
                input_domain_certificate=input_domain_certificate,
                validated_atlas=validated_atlas,
                planar_binary_count=int(planar_binary_count),
                spatial_binary_count=int(spatial_binary_count),
            ),
            source=type(validated_atlas).__name__,
            detail=(
                "planar Levi-Civita binary charts require dimension=2; "
                "spatial KS binary charts require dimension=3 and a certified "
                "local spatial KS collision policy; "
                f"dimension={input_domain_certificate.dimension}; "
                f"planar_binary_chart_count={planar_binary_count}; "
                f"spatial_binary_chart_count={spatial_binary_count}; "
                f"binary_policy={getattr(getattr(validated_atlas, 'collision_policy', None), 'binary_policy', 'missing')}"
            ),
        ),
    )
    return CompactOrdinaryBinaryFiniteAtlasCertificate(
        input_domain_certificate=input_domain_certificate,
        validated_atlas=validated_atlas,
        chart_types=chart_types,
        ordinary_chart_count=int(ordinary_count),
        separated_binary_chart_count=int(binary_count),
        total_collision_chart_count=int(total_collision_count),
        total_collision_policy_id=total_collision_policy_id,
        obligations=obligations,
    )


def certify_compact_nonzero_angular_finite_atlas(
    *,
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object,
) -> CompactNonzeroAngularFiniteAtlasCertificate:
    """Certify a finite ordinary/binary atlas for a nonzero-angular compact interval."""

    _reject_raw_bool("input_domain_certificate", input_domain_certificate)
    _reject_raw_bool("validated_atlas", validated_atlas)
    chart_types = tuple(
        str(getattr(chart, "chart_type", "missing"))
        for chart in getattr(validated_atlas, "charts", ())
    )
    ordinary_types = set(ORDINARY_FINITE_ATLAS_CHART_TYPES)
    binary_types = set(BINARY_FINITE_ATLAS_CHART_TYPES)
    planar_binary_types = set(PLANAR_BINARY_FINITE_ATLAS_CHART_TYPES)
    spatial_binary_types = set(SPATIAL_BINARY_FINITE_ATLAS_CHART_TYPES)
    ordinary_count = sum(chart_type in ordinary_types for chart_type in chart_types)
    binary_count = sum(chart_type in binary_types for chart_type in chart_types)
    planar_binary_count = sum(chart_type in planar_binary_types for chart_type in chart_types)
    spatial_binary_count = sum(chart_type in spatial_binary_types for chart_type in chart_types)
    total_collision_count = sum("total_collision" in chart_type for chart_type in chart_types)
    unknown_chart_types = tuple(
        chart_type
        for chart_type in chart_types
        if chart_type not in ordinary_types
        and chart_type not in binary_types
        and "total_collision" not in chart_type
    )
    obligations = (
        _required_constructor_obligation(
            "positive_mass_noncollision_input_domain",
            input_domain_certificate,
            ("certified",),
        ),
        _required_constructor_obligation(
            "validated_atlas_solution",
            validated_atlas,
            ("proof_certified",),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_mass_consistency",
            certified=_finite_atlas_mass_consistency_certified(
                input_domain_certificate,
                validated_atlas,
            ),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_mass_consistency_detail(
                input_domain_certificate,
                validated_atlas,
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_initial_state_consistency",
            certified=_finite_atlas_initial_state_consistency_certified(
                input_domain_certificate,
                validated_atlas,
            ),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_initial_state_consistency_detail(
                input_domain_certificate,
                validated_atlas,
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_newton_residual_ledger",
            certified=_finite_atlas_residual_ledger_certified(validated_atlas, len(chart_types)),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_residual_detail(validated_atlas),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_projection_domain_ledger",
            certified=_finite_atlas_projection_domain_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="all finite charts must carry certified dynamics, finite chart domains, and projection witnesses",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_invariant_ledger",
            certified=_finite_atlas_invariant_ledger_certified(validated_atlas, len(chart_types)),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_invariant_detail(validated_atlas),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_tail_budget",
            certified=_finite_atlas_tail_budget_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="global tail budget and every chart tail bound must be finite and certified",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_transition_ledger",
            certified=_finite_atlas_transition_ledger_certified(validated_atlas, len(chart_types)),
            source=type(validated_atlas).__name__,
            detail=(
                f"chart_count={len(chart_types)}; "
                f"transition_count={len(getattr(validated_atlas, 'transitions', ()))}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_collision_policy",
            certified=bool(getattr(getattr(validated_atlas, "collision_policy", None), "certified", False)),
            source=type(validated_atlas).__name__,
            detail=getattr(getattr(validated_atlas, "collision_policy", None), "binary_policy", "missing"),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_target_interval",
            certified=_finite_atlas_target_interval_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="target interval must be a nonempty finite coordinate interval array",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_target_time_domain",
            certified=bool(getattr(validated_atlas, "target_time_certified", False)),
            source=type(validated_atlas).__name__,
            detail="requested physical target time must lie in the final chart time interval",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_targeting_or_containment",
            certified=_finite_atlas_targeting_or_containment_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="requires target_time, finite_time_physical_targeting, or target_containment ledger evidence",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_proof_ledger",
            certified=bool(getattr(getattr(validated_atlas, "proof_ledger", None), "certified", False)),
            source=type(validated_atlas).__name__,
            detail="typed proof ledger must certify every required finite-atlas entry",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_selector_trace",
            certified=_finite_atlas_selector_trace_certified(validated_atlas, chart_types),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_selector_trace_detail(validated_atlas, chart_types),
        ),
        TheoremPipelineObligation(
            obligation="finite_chart_family_count",
            certified=bool(len(chart_types) > 0 and np.isfinite(len(chart_types))),
            source=type(validated_atlas).__name__,
            detail=f"chart_count={len(chart_types)}",
        ),
        TheoremPipelineObligation(
            obligation="ordinary_or_separated_binary_chart_families",
            certified=bool(
                chart_types
                and ordinary_count + binary_count == len(chart_types)
            ),
            source=type(validated_atlas).__name__,
            detail="unknown="
            + ",".join(unknown_chart_types)
            + "; chart_types="
            + ",".join(chart_types),
        ),
        TheoremPipelineObligation(
            obligation="no_total_collision_charts",
            certified=total_collision_count == 0,
            source=type(validated_atlas).__name__,
            detail=f"total_collision_chart_count={total_collision_count}",
        ),
        TheoremPipelineObligation(
            obligation="spatial_binary_regularization_scope",
            certified=_finite_atlas_binary_scope_certified(
                input_domain_certificate=input_domain_certificate,
                validated_atlas=validated_atlas,
                planar_binary_count=int(planar_binary_count),
                spatial_binary_count=int(spatial_binary_count),
            ),
            source=type(validated_atlas).__name__,
            detail=(
                "planar Levi-Civita binary charts require dimension=2; "
                "spatial KS binary charts require dimension=3 and a certified "
                "local spatial KS collision policy; "
                f"dimension={input_domain_certificate.dimension}; "
                f"planar_binary_chart_count={planar_binary_count}; "
                f"spatial_binary_chart_count={spatial_binary_count}; "
                f"binary_policy={getattr(getattr(validated_atlas, 'collision_policy', None), 'binary_policy', 'missing')}"
            ),
        ),
    )
    return CompactNonzeroAngularFiniteAtlasCertificate(
        input_domain_certificate=input_domain_certificate,
        validated_atlas=validated_atlas,
        chart_types=chart_types,
        ordinary_chart_count=int(ordinary_count),
        separated_binary_chart_count=int(binary_count),
        total_collision_chart_count=int(total_collision_count),
        obligations=obligations,
    )


def certify_maximal_classical_until_total_collision_stop(
    *,
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object,
) -> MaximalClassicalUntilTotalCollisionStopCertificate:
    """Certify a finite classical atlas that stops before total collision.

    This is a scoped maximal-classical certificate.  It accepts ordinary and
    binary-regularized finite charts, but it does not continue through a total
    collision and it does not interpret selector policies as classical
    uniqueness.
    """

    _reject_raw_bool("input_domain_certificate", input_domain_certificate)
    _reject_raw_bool("validated_atlas", validated_atlas)
    finite_atlas = certify_compact_nonzero_angular_finite_atlas(
        input_domain_certificate=input_domain_certificate,
        validated_atlas=validated_atlas,
    )
    chart_types = tuple(str(chart_type) for chart_type in finite_atlas.chart_types)
    total_collision_policy = str(
        getattr(
            getattr(validated_atlas, "collision_policy", None),
            "total_collision_policy",
            "",
        )
    )
    total_collision_chart_count = sum(
        "total_collision" in chart_type for chart_type in chart_types
    )
    obligations = (
        *finite_atlas.obligations,
        TheoremPipelineObligation(
            obligation="maximal_classical_finite_atlas",
            certified=finite_atlas.certified,
            source=type(finite_atlas).__name__,
            detail=_finite_middle_detail(finite_atlas),
        ),
        TheoremPipelineObligation(
            obligation="maximal_classical_stop_before_total_collision_policy",
            certified=total_collision_policy == "finite_time_stop_before_total_collision",
            source=type(getattr(validated_atlas, "collision_policy", None)).__name__,
            detail=f"total_collision_policy={total_collision_policy or 'missing'}",
        ),
        TheoremPipelineObligation(
            obligation="maximal_classical_no_selector_continuation_policy",
            certified="selector" not in total_collision_policy,
            source=type(getattr(validated_atlas, "collision_policy", None)).__name__,
            detail=f"total_collision_policy={total_collision_policy or 'missing'}",
        ),
        TheoremPipelineObligation(
            obligation="maximal_classical_no_total_collision_continuation_charts",
            certified=total_collision_chart_count == 0,
            source=type(validated_atlas).__name__,
            detail=f"total_collision_chart_count={total_collision_chart_count}",
        ),
    )
    return MaximalClassicalUntilTotalCollisionStopCertificate(
        input_domain_certificate=input_domain_certificate,
        validated_atlas=validated_atlas,
        finite_atlas_certificate=finite_atlas,
        chart_types=chart_types,
        total_collision_policy=total_collision_policy,
        obligations=obligations,
    )


def construct_maximal_classical_until_total_collision_atlas(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    compact_time_rate: float,
    validated_atlas: object,
) -> GlobalAtlasCertificate:
    """Build the scoped maximal-classical stop-before-total-collision theorem."""

    input_domain_certificate = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    compact_time_certificate = certify_compact_time_real_line_coverage(
        compact_time_rate,
    )
    stop_certificate = certify_maximal_classical_until_total_collision_stop(
        input_domain_certificate=input_domain_certificate,
        validated_atlas=validated_atlas,
    )
    classification = classify_global_regime(
        input_domain_certificate=input_domain_certificate,
        compact_time_certificate=compact_time_certificate,
        regime_id="maximal_classical_until_total_collision",
        ordinary_gap_envelope=stop_certificate,
    )
    return construct_global_atlas_for_regime(
        classification,
        validated_atlas=validated_atlas,
        ordinary_gap_atlas=stop_certificate,
    )


def certify_nonzero_angular_finite_middle_atlas(
    *,
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    future_validated_atlas: object | None,
    past_validated_atlas: object | None,
) -> NonzeroAngularFiniteMiddleAtlasCertificate:
    """Certify the compact finite middle needed before endpoint recurrences.

    The certificate deliberately proves only the finite middle charts from the
    supplied initial data in both time directions.  It does not identify the
    tail handoff time or prove the event-regime shell hypotheses.
    """

    _reject_raw_bool("input_domain_certificate", input_domain_certificate)
    _reject_raw_bool("future_validated_atlas", future_validated_atlas)
    _reject_raw_bool("past_validated_atlas", past_validated_atlas)
    future_certificate = (
        None
        if future_validated_atlas is None
        else certify_compact_nonzero_angular_finite_atlas(
            input_domain_certificate=input_domain_certificate,
            validated_atlas=future_validated_atlas,
        )
    )
    past_certificate = (
        None
        if past_validated_atlas is None
        else certify_compact_nonzero_angular_finite_atlas(
            input_domain_certificate=input_domain_certificate,
            validated_atlas=past_validated_atlas,
        )
    )
    obligations = (
        _required_constructor_obligation(
            "positive_mass_noncollision_input_domain",
            input_domain_certificate,
            ("certified",),
        ),
        TheoremPipelineObligation(
            obligation="finite_middle_future_validated_atlas",
            certified=bool(
                future_certificate is not None and future_certificate.certified
            ),
            source=type(future_certificate).__name__ if future_certificate is not None else "missing",
            detail=_finite_middle_detail(future_certificate),
        ),
        TheoremPipelineObligation(
            obligation="finite_middle_past_validated_atlas",
            certified=bool(
                past_certificate is not None and past_certificate.certified
            ),
            source=type(past_certificate).__name__ if past_certificate is not None else "missing",
            detail=_finite_middle_detail(past_certificate),
        ),
        TheoremPipelineObligation(
            obligation="finite_middle_future_time_direction",
            certified=_finite_middle_target_time_positive(future_validated_atlas),
            source=type(future_validated_atlas).__name__ if future_validated_atlas is not None else "missing",
            detail=f"target_time={getattr(future_validated_atlas, 'target_time', 'missing')}",
        ),
        TheoremPipelineObligation(
            obligation="finite_middle_past_time_direction",
            certified=_finite_middle_target_time_negative(past_validated_atlas),
            source=type(past_validated_atlas).__name__ if past_validated_atlas is not None else "missing",
            detail=f"target_time={getattr(past_validated_atlas, 'target_time', 'missing')}",
        ),
    )
    return NonzeroAngularFiniteMiddleAtlasCertificate(
        input_domain_certificate=input_domain_certificate,
        future_finite_atlas=future_certificate,
        past_finite_atlas=past_certificate,
        obligations=obligations,
    )


def construct_nonzero_angular_finite_middle_atlas_for_event_shell(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    compact_time_rate: float,
    future_envelope_spec: NonzeroAngularUniformPairEventEnvelopeSpec,
    past_envelope_spec: NonzeroAngularUniformPairEventEnvelopeSpec,
    order: int = 8,
    max_compact_step: float = 5.0e-5,
    max_s_step: float = 0.02,
    target_bisections: int = 20,
) -> NonzeroAngularFiniteMiddleAtlasCertificate:
    """Construct the two-sided finite middle ending at the first event shells."""

    future_spec = _require_uniform_pair_event_envelope_spec(
        "future_envelope_spec",
        future_envelope_spec,
        allowed_time_directions=("future", "time_reversal_invariant"),
    )
    past_spec = _require_uniform_pair_event_envelope_spec(
        "past_envelope_spec",
        past_envelope_spec,
        allowed_time_directions=("time_reversed_past", "time_reversal_invariant"),
    )
    input_domain_certificate = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    compact_time_certificate = certify_compact_time_real_line_coverage(
        compact_time_rate,
    )
    future_target_time = compact_time_certificate.physical_time(
        _future_event_shell_start(future_spec),
    )
    past_target_time = compact_time_certificate.physical_time(
        _past_event_shell_start(past_spec),
    )
    from .general_solution import evaluate_unrestricted_solution

    future_atlas = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        future_target_time,
        method="validated_atlas",
        order=int(order),
        max_compact_step=float(max_compact_step),
        max_s_step=float(max_s_step),
        target_bisections=int(target_bisections),
    )
    past_atlas = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        past_target_time,
        method="validated_atlas",
        order=int(order),
        max_compact_step=float(max_compact_step),
        max_s_step=float(max_s_step),
        target_bisections=int(target_bisections),
    )
    return certify_nonzero_angular_finite_middle_atlas(
        input_domain_certificate=input_domain_certificate,
        future_validated_atlas=future_atlas,
        past_validated_atlas=past_atlas,
    )


def certify_nonzero_angular_event_regime_handoff(
    *,
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    compact_time_certificate: CompactTimeCoverageCertificate,
    finite_middle_atlas: object | None,
    future_envelope_spec: NonzeroAngularUniformPairEventEnvelopeSpec,
    past_envelope_spec: NonzeroAngularUniformPairEventEnvelopeSpec,
) -> NonzeroAngularEventRegimeHandoffCertificate:
    """Certify finite endpoint containment in the ordinary event-tail envelopes.

    This is intentionally only the handoff into the first tail shell.  It does
    not prove that the ordinary/binary shell hypotheses remain invariant for
    all later shells.
    """

    _reject_raw_bool("input_domain_certificate", input_domain_certificate)
    _reject_raw_bool("compact_time_certificate", compact_time_certificate)
    _reject_raw_bool("finite_middle_atlas", finite_middle_atlas)
    future_spec = _require_uniform_pair_event_envelope_spec(
        "future_envelope_spec",
        future_envelope_spec,
        allowed_time_directions=("future", "time_reversal_invariant"),
    )
    past_spec = _require_uniform_pair_event_envelope_spec(
        "past_envelope_spec",
        past_envelope_spec,
        allowed_time_directions=("time_reversed_past", "time_reversal_invariant"),
    )
    future_validated_atlas = _finite_middle_validated_atlas(
        finite_middle_atlas,
        "future_finite_atlas",
    )
    past_validated_atlas = _finite_middle_validated_atlas(
        finite_middle_atlas,
        "past_finite_atlas",
    )
    future_metrics = _finite_middle_endpoint_metrics(
        input_domain_certificate,
        future_validated_atlas,
    )
    past_metrics = _finite_middle_endpoint_metrics(
        input_domain_certificate,
        past_validated_atlas,
    )
    future_compact_parameter = _finite_middle_target_compact_parameter(
        compact_time_certificate,
        future_validated_atlas,
    )
    past_compact_parameter = _finite_middle_target_compact_parameter(
        compact_time_certificate,
        past_validated_atlas,
    )
    obligations = (
        _required_constructor_obligation(
            "positive_mass_noncollision_input_domain",
            input_domain_certificate,
            ("certified",),
        ),
        _required_constructor_obligation(
            "compact_time_real_line_coverage",
            compact_time_certificate,
            ("certified",),
        ),
        _required_constructor_obligation(
            "finite_middle_validated_atlas",
            finite_middle_atlas,
            ("certified", "proof_certified"),
        ),
        TheoremPipelineObligation(
            obligation="finite_middle_input_domain_matches_handoff",
            certified=_input_domain_certificates_consistent(
                input_domain_certificate,
                getattr(finite_middle_atlas, "input_domain_certificate", None),
            ),
            source=type(finite_middle_atlas).__name__ if finite_middle_atlas is not None else "missing",
            detail="finite middle atlas must be certified from the same masses and initial state as this handoff",
        ),
        TheoremPipelineObligation(
            obligation="future_finite_middle_endpoint_available",
            certified=future_validated_atlas is not None,
            source=type(future_validated_atlas).__name__ if future_validated_atlas is not None else "missing",
            detail="future finite middle certificate must expose its ValidatedAtlasSolution",
        ),
        TheoremPipelineObligation(
            obligation="past_finite_middle_endpoint_available",
            certified=past_validated_atlas is not None,
            source=type(past_validated_atlas).__name__ if past_validated_atlas is not None else "missing",
            detail="past finite middle certificate must expose its ValidatedAtlasSolution",
        ),
        TheoremPipelineObligation(
            obligation="future_finite_middle_target_time",
            certified=_finite_middle_target_time_positive(future_validated_atlas),
            source=type(future_validated_atlas).__name__ if future_validated_atlas is not None else "missing",
            detail=f"target_time={getattr(future_validated_atlas, 'target_time', 'missing')}",
        ),
        TheoremPipelineObligation(
            obligation="past_finite_middle_target_time",
            certified=_finite_middle_target_time_negative(past_validated_atlas),
            source=type(past_validated_atlas).__name__ if past_validated_atlas is not None else "missing",
            detail=f"target_time={getattr(past_validated_atlas, 'target_time', 'missing')}",
        ),
        TheoremPipelineObligation(
            obligation="future_finite_middle_reaches_first_event_shell",
            certified=_compact_parameter_matches_event_shell_start(
                observed=future_compact_parameter,
                expected=_future_event_shell_start(future_spec),
            ),
            source=type(compact_time_certificate).__name__,
            detail=(
                f"observed_u={future_compact_parameter}; "
                f"required_u={_future_event_shell_start(future_spec)}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="past_finite_middle_reaches_first_event_shell",
            certified=_compact_parameter_matches_event_shell_start(
                observed=past_compact_parameter,
                expected=_past_event_shell_start(past_spec),
            ),
            source=type(compact_time_certificate).__name__,
            detail=(
                f"observed_u={past_compact_parameter}; "
                f"required_u={_past_event_shell_start(past_spec)}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="future_endpoint_ordinary_gap_envelope_containment",
            certified=_endpoint_metrics_inside_ordinary_gap(
                future_metrics,
                future_spec,
            ),
            source=future_spec.source,
            detail=_endpoint_envelope_detail(future_metrics, future_spec),
        ),
        TheoremPipelineObligation(
            obligation="past_endpoint_ordinary_gap_envelope_containment",
            certified=_endpoint_metrics_inside_ordinary_gap(
                past_metrics,
                past_spec,
            ),
            source=past_spec.source,
            detail=_endpoint_envelope_detail(past_metrics, past_spec),
        ),
    )
    return NonzeroAngularEventRegimeHandoffCertificate(
        input_domain_certificate=input_domain_certificate,
        compact_time_certificate=compact_time_certificate,
        finite_middle_atlas=finite_middle_atlas,
        future_envelope_spec=future_spec,
        past_envelope_spec=past_spec,
        future_compact_parameter=future_compact_parameter,
        past_compact_parameter=past_compact_parameter,
        future_metrics=future_metrics,
        past_metrics=past_metrics,
        obligations=obligations,
    )


def certify_nonzero_angular_event_tail_margin_from_handoff(
    *,
    event_regime_handoff: object | None,
    two_sided_event_budget: object | None,
) -> NonzeroAngularEventTailMarginCertificate:
    """Check the all-future value-tail budget fits the handoff metric margins.

    If every physical coordinate has projected value-tail error at most
    ``epsilon`` in a ``d``-dimensional state, pair distances and pair diameters
    change by at most ``2 sqrt(d) epsilon`` and body speeds by at most
    ``sqrt(d) epsilon``.  This constructor verifies that the infinite
    recurrence tail leaves those metric margins positive in both time
    directions.  It is not an event-isolation or shell-induction proof.
    """

    _reject_raw_bool("event_regime_handoff", event_regime_handoff)
    _reject_raw_bool("two_sided_event_budget", two_sided_event_budget)
    future_tail = _event_budget_component_infinite_bound(
        getattr(two_sided_event_budget, "future_event_budget", None),
        "value",
    )
    past_tail = _event_budget_component_infinite_bound(
        getattr(two_sided_event_budget, "past_event_budget", None),
        "value",
    )
    future_margins = _handoff_metric_margins(event_regime_handoff, "future")
    past_margins = _handoff_metric_margins(event_regime_handoff, "past")
    dimension = int(
        getattr(
            getattr(event_regime_handoff, "input_domain_certificate", None),
            "dimension",
            0,
        )
    )
    future_required = _required_coordinate_tail_margin(
        future_tail,
        dimension=dimension,
    )
    past_required = _required_coordinate_tail_margin(
        past_tail,
        dimension=dimension,
    )
    obligations = (
        _required_constructor_obligation(
            "finite_middle_to_event_envelope_handoff",
            event_regime_handoff,
            ("certified", "handoff_certified", "proof_certified"),
        ),
        _required_constructor_obligation(
            "two_sided_primitive_cauchy_all_time_budget",
            two_sided_event_budget,
            ("recurrence_closes", "certified"),
        ),
        TheoremPipelineObligation(
            obligation="future_all_future_value_tail_margin",
            certified=_tail_bound_fits_metric_margins(
                future_tail,
                future_margins,
                dimension=dimension,
            ),
            source=type(two_sided_event_budget).__name__ if two_sided_event_budget is not None else "missing",
            detail=_tail_margin_detail(future_tail, future_required, future_margins),
        ),
        TheoremPipelineObligation(
            obligation="past_all_future_value_tail_margin",
            certified=_tail_bound_fits_metric_margins(
                past_tail,
                past_margins,
                dimension=dimension,
            ),
            source=type(two_sided_event_budget).__name__ if two_sided_event_budget is not None else "missing",
            detail=_tail_margin_detail(past_tail, past_required, past_margins),
        ),
    )
    return NonzeroAngularEventTailMarginCertificate(
        event_regime_handoff=event_regime_handoff,
        two_sided_event_budget=two_sided_event_budget,
        future_value_tail_bound=future_tail,
        past_value_tail_bound=past_tail,
        future_required_coordinate_margin=future_required,
        past_required_coordinate_margin=past_required,
        future_metric_margins=future_margins,
        past_metric_margins=past_margins,
        obligations=obligations,
    )


def construct_nonzero_angular_first_event_shell_prefix(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    compact_time_rate: float,
    event_regime_handoff: object | None,
    order: int = 8,
    max_compact_step: float = 0.03,
    max_s_step: float = 0.02,
    target_bisections: int = 20,
) -> NonzeroAngularFirstEventShellPrefixCertificate:
    """Validate the finite atlas through shell zero into the next boundary."""

    _reject_raw_bool("event_regime_handoff", event_regime_handoff)
    input_domain_certificate = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    compact_time_certificate = certify_compact_time_real_line_coverage(
        compact_time_rate,
    )
    future_spec = getattr(event_regime_handoff, "future_envelope_spec", None)
    past_spec = getattr(event_regime_handoff, "past_envelope_spec", None)
    future_end_u = _future_event_shell_end(future_spec)
    past_end_u = _past_event_shell_end(past_spec)
    future_target_time = _compact_physical_time_or_nan(
        compact_time_certificate,
        future_end_u,
    )
    past_target_time = _compact_physical_time_or_nan(
        compact_time_certificate,
        past_end_u,
    )
    future_validated_atlas = None
    past_validated_atlas = None
    future_prefix_atlas = None
    past_prefix_atlas = None
    if np.isfinite(future_target_time):
        future_validated_atlas = _evaluate_nonzero_angular_validated_atlas(
            masses=masses,
            positions=positions,
            velocities=velocities,
            target_time=future_target_time,
            order=order,
            max_compact_step=max_compact_step,
            max_s_step=max_s_step,
            target_bisections=target_bisections,
        )
        future_prefix_atlas = certify_compact_nonzero_angular_finite_atlas(
            input_domain_certificate=input_domain_certificate,
            validated_atlas=future_validated_atlas,
        )
    if np.isfinite(past_target_time):
        past_validated_atlas = _evaluate_nonzero_angular_validated_atlas(
            masses=masses,
            positions=positions,
            velocities=velocities,
            target_time=past_target_time,
            order=order,
            max_compact_step=max_compact_step,
            max_s_step=max_s_step,
            target_bisections=target_bisections,
        )
        past_prefix_atlas = certify_compact_nonzero_angular_finite_atlas(
            input_domain_certificate=input_domain_certificate,
            validated_atlas=past_validated_atlas,
        )
    future_end_metrics = _finite_middle_endpoint_metrics(
        input_domain_certificate,
        future_validated_atlas,
    )
    past_end_metrics = _finite_middle_endpoint_metrics(
        input_domain_certificate,
        past_validated_atlas,
    )
    obligations = (
        _required_constructor_obligation(
            "finite_middle_to_event_envelope_handoff",
            event_regime_handoff,
            ("certified", "handoff_certified", "proof_certified"),
        ),
        TheoremPipelineObligation(
            obligation="first_event_shell_prefix_input_domain_matches_handoff",
            certified=_input_domain_certificates_consistent(
                input_domain_certificate,
                getattr(event_regime_handoff, "input_domain_certificate", None),
            ),
            source=type(event_regime_handoff).__name__ if event_regime_handoff is not None else "missing",
            detail="prefix atlas must be constructed from the same masses and initial state as the handoff",
        ),
        TheoremPipelineObligation(
            obligation="first_event_shell_prefix_compact_time_matches_handoff",
            certified=_compact_time_certificates_consistent(
                compact_time_certificate,
                getattr(event_regime_handoff, "compact_time_certificate", None),
            ),
            source=type(event_regime_handoff).__name__ if event_regime_handoff is not None else "missing",
            detail="prefix compact-time map must match the handoff compact-time map",
        ),
        _required_constructor_obligation(
            "future_first_event_shell_prefix_validated_atlas",
            future_prefix_atlas,
            ("certified", "proof_certified"),
        ),
        _required_constructor_obligation(
            "past_first_event_shell_prefix_validated_atlas",
            past_prefix_atlas,
            ("certified", "proof_certified"),
        ),
        TheoremPipelineObligation(
            obligation="future_first_event_shell_prefix_extends_handoff",
            certified=_prefix_extends_handoff(
                event_regime_handoff,
                future_validated_atlas,
                direction="future",
            ),
            source=type(future_validated_atlas).__name__ if future_validated_atlas is not None else "missing",
            detail=(
                f"handoff_t={_handoff_target_time(event_regime_handoff, 'future')}; "
                f"prefix_t={getattr(future_validated_atlas, 'target_time', 'missing')}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="past_first_event_shell_prefix_extends_handoff",
            certified=_prefix_extends_handoff(
                event_regime_handoff,
                past_validated_atlas,
                direction="past",
            ),
            source=type(past_validated_atlas).__name__ if past_validated_atlas is not None else "missing",
            detail=(
                f"handoff_t={_handoff_target_time(event_regime_handoff, 'past')}; "
                f"prefix_t={getattr(past_validated_atlas, 'target_time', 'missing')}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="future_first_event_shell_endpoint_ordinary_gap_envelope",
            certified=_endpoint_metrics_inside_ordinary_gap(
                future_end_metrics,
                future_spec,
            ),
            source=getattr(future_spec, "source", "missing"),
            detail=_endpoint_envelope_detail(future_end_metrics, future_spec),
        ),
        TheoremPipelineObligation(
            obligation="past_first_event_shell_endpoint_ordinary_gap_envelope",
            certified=_endpoint_metrics_inside_ordinary_gap(
                past_end_metrics,
                past_spec,
            ),
            source=getattr(past_spec, "source", "missing"),
            detail=_endpoint_envelope_detail(past_end_metrics, past_spec),
        ),
    )
    return NonzeroAngularFirstEventShellPrefixCertificate(
        event_regime_handoff=event_regime_handoff,
        future_prefix_atlas=future_prefix_atlas,
        past_prefix_atlas=past_prefix_atlas,
        future_end_compact_parameter=future_end_u,
        past_end_compact_parameter=past_end_u,
        future_end_metrics=future_end_metrics,
        past_end_metrics=past_end_metrics,
        obligations=obligations,
    )


def certify_nonzero_angular_event_shell_invariance_from_handoff(
    *,
    event_regime_handoff: object | None,
    event_tail_margin_certificate: object | None,
    first_event_shell_prefix: object | None,
    two_sided_event_budget: object | None,
) -> NonzeroAngularEventShellInvarianceCertificate:
    """Certify a sufficient all-future shell-invariance bridge.

    The analytic content is deliberately conditional: once the finite middle
    reaches the first compact event shell, the first shell is explicitly covered
    by a finite validated atlas.  Starting at the next shell boundary, the
    remaining geometric value tail from the event recurrence must fit inside
    the ordinary-gap metric margins at that boundary.  The recurrence geometry
    must also be the same compact shell geometry as the future and
    time-reversed-past event-envelope specifications, and its ordinary Cauchy
    source must use those same ordinary-gap bounds.
    """

    _reject_raw_bool("event_regime_handoff", event_regime_handoff)
    _reject_raw_bool("event_tail_margin_certificate", event_tail_margin_certificate)
    _reject_raw_bool("first_event_shell_prefix", first_event_shell_prefix)
    _reject_raw_bool("two_sided_event_budget", two_sided_event_budget)
    future_event_budget = getattr(two_sided_event_budget, "future_event_budget", None)
    past_event_budget = getattr(two_sided_event_budget, "past_event_budget", None)
    future_tail = _event_budget_component_tail_from_prefix(
        future_event_budget,
        "value",
        prefix_length=1,
    )
    past_tail = _event_budget_component_tail_from_prefix(
        past_event_budget,
        "value",
        prefix_length=1,
    )
    future_margins = _prefix_metric_margins(first_event_shell_prefix, "future")
    past_margins = _prefix_metric_margins(first_event_shell_prefix, "past")
    dimension = int(
        getattr(
            getattr(event_regime_handoff, "input_domain_certificate", None),
            "dimension",
            0,
        )
    )
    future_required = _required_coordinate_tail_margin(
        future_tail,
        dimension=dimension,
    )
    past_required = _required_coordinate_tail_margin(
        past_tail,
        dimension=dimension,
    )
    future_spec = getattr(event_regime_handoff, "future_envelope_spec", None)
    past_spec = getattr(event_regime_handoff, "past_envelope_spec", None)
    obligations = (
        _required_constructor_obligation(
            "finite_middle_to_event_envelope_handoff",
            event_regime_handoff,
            ("certified", "handoff_certified", "proof_certified"),
        ),
        _required_constructor_obligation(
            "nonzero_angular_all_future_value_tail_margin",
            event_tail_margin_certificate,
            ("certified", "tail_margin_certified", "proof_certified"),
        ),
        _required_constructor_obligation(
            "nonzero_angular_first_event_shell_prefix",
            first_event_shell_prefix,
            ("certified", "prefix_certified", "proof_certified"),
        ),
        _required_constructor_obligation(
            "two_sided_primitive_cauchy_all_time_budget",
            two_sided_event_budget,
            ("recurrence_closes", "certified"),
        ),
        TheoremPipelineObligation(
            obligation="event_tail_margin_uses_same_handoff",
            certified=_certificate_field_is_same_object(
                event_tail_margin_certificate,
                "event_regime_handoff",
                event_regime_handoff,
            ),
            source=(
                type(event_tail_margin_certificate).__name__
                if event_tail_margin_certificate is not None
                else "missing"
            ),
            detail="tail-margin certificate must be derived from the handoff being extended",
        ),
        TheoremPipelineObligation(
            obligation="event_tail_margin_uses_same_two_sided_budget",
            certified=_certificate_field_is_same_object(
                event_tail_margin_certificate,
                "two_sided_event_budget",
                two_sided_event_budget,
            ),
            source=(
                type(event_tail_margin_certificate).__name__
                if event_tail_margin_certificate is not None
                else "missing"
            ),
            detail="tail-margin certificate must use the same event recurrence budget as shell invariance",
        ),
        TheoremPipelineObligation(
            obligation="first_event_shell_prefix_uses_same_handoff",
            certified=_certificate_field_is_same_object(
                first_event_shell_prefix,
                "event_regime_handoff",
                event_regime_handoff,
            ),
            source=(
                type(first_event_shell_prefix).__name__
                if first_event_shell_prefix is not None
                else "missing"
            ),
            detail="first-shell prefix must extend the same handoff used by shell invariance",
        ),
        TheoremPipelineObligation(
            obligation="future_event_shell_geometry_matches_envelope",
            certified=_event_budget_shell_geometry_matches_spec(
                future_event_budget,
                future_spec,
            ),
            source=type(future_event_budget).__name__ if future_event_budget is not None else "missing",
            detail=_event_budget_shell_geometry_detail(future_event_budget, future_spec),
        ),
        TheoremPipelineObligation(
            obligation="past_event_shell_geometry_matches_envelope",
            certified=_event_budget_shell_geometry_matches_spec(
                past_event_budget,
                past_spec,
            ),
            source=type(past_event_budget).__name__ if past_event_budget is not None else "missing",
            detail=_event_budget_shell_geometry_detail(past_event_budget, past_spec),
        ),
        TheoremPipelineObligation(
            obligation="future_first_shell_remaining_value_tail_margin",
            certified=_tail_bound_fits_metric_margins(
                future_tail,
                future_margins,
                dimension=dimension,
            ),
            source=type(future_event_budget).__name__ if future_event_budget is not None else "missing",
            detail=_tail_margin_detail(future_tail, future_required, future_margins),
        ),
        TheoremPipelineObligation(
            obligation="past_first_shell_remaining_value_tail_margin",
            certified=_tail_bound_fits_metric_margins(
                past_tail,
                past_margins,
                dimension=dimension,
            ),
            source=type(past_event_budget).__name__ if past_event_budget is not None else "missing",
            detail=_tail_margin_detail(past_tail, past_required, past_margins),
        ),
        TheoremPipelineObligation(
            obligation="future_uniform_value_majorants",
            certified=_event_budget_component_majorant_growth_bounded(
                future_event_budget,
                "value",
                growth_bound=1.0,
            ),
            source=type(future_event_budget).__name__ if future_event_budget is not None else "missing",
            detail="all value majorant growth factors must be <= 1 for uniform envelope reuse",
        ),
        TheoremPipelineObligation(
            obligation="past_uniform_value_majorants",
            certified=_event_budget_component_majorant_growth_bounded(
                past_event_budget,
                "value",
                growth_bound=1.0,
            ),
            source=type(past_event_budget).__name__ if past_event_budget is not None else "missing",
            detail="all value majorant growth factors must be <= 1 for uniform envelope reuse",
        ),
        TheoremPipelineObligation(
            obligation="future_ordinary_gap_source_matches_envelope",
            certified=_ordinary_gap_source_matches_envelope(
                future_event_budget,
                future_spec,
            ),
            source=type(future_event_budget).__name__ if future_event_budget is not None else "missing",
            detail=_ordinary_gap_source_detail(future_event_budget, future_spec),
        ),
        TheoremPipelineObligation(
            obligation="past_ordinary_gap_source_matches_envelope",
            certified=_ordinary_gap_source_matches_envelope(
                past_event_budget,
                past_spec,
            ),
            source=type(past_event_budget).__name__ if past_event_budget is not None else "missing",
            detail=_ordinary_gap_source_detail(past_event_budget, past_spec),
        ),
        TheoremPipelineObligation(
            obligation="future_separated_binary_sources_match_envelope",
            certified=_separated_binary_sources_match_envelope(
                future_event_budget,
                future_spec,
            ),
            source=type(future_event_budget).__name__ if future_event_budget is not None else "missing",
            detail=_separated_binary_sources_detail(future_event_budget, future_spec),
        ),
        TheoremPipelineObligation(
            obligation="past_separated_binary_sources_match_envelope",
            certified=_separated_binary_sources_match_envelope(
                past_event_budget,
                past_spec,
            ),
            source=type(past_event_budget).__name__ if past_event_budget is not None else "missing",
            detail=_separated_binary_sources_detail(past_event_budget, past_spec),
        ),
    )
    return NonzeroAngularEventShellInvarianceCertificate(
        event_regime_handoff=event_regime_handoff,
        event_tail_margin_certificate=event_tail_margin_certificate,
        first_event_shell_prefix=first_event_shell_prefix,
        two_sided_event_budget=two_sided_event_budget,
        future_remaining_value_tail_bound=future_tail,
        past_remaining_value_tail_bound=past_tail,
        future_required_coordinate_margin=future_required,
        past_required_coordinate_margin=past_required,
        future_prefix_metric_margins=future_margins,
        past_prefix_metric_margins=past_margins,
        obligations=obligations,
    )


def certify_compact_zero_angular_finite_selector_atlas(
    *,
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object,
    total_collision_selector_envelopes: object,
) -> CompactZeroAngularFiniteSelectorAtlasCertificate:
    """Certify a finite compact atlas with explicit zero-angular selector charts."""

    _reject_raw_bool("input_domain_certificate", input_domain_certificate)
    _reject_raw_bool("validated_atlas", validated_atlas)
    selector_envelopes = _normalize_selector_envelopes(
        total_collision_selector_envelopes,
    )
    chart_types = tuple(
        str(getattr(chart, "chart_type", "missing"))
        for chart in getattr(validated_atlas, "charts", ())
    )
    ordinary_types = set(ORDINARY_FINITE_ATLAS_CHART_TYPES)
    binary_types = set(BINARY_FINITE_ATLAS_CHART_TYPES)
    total_collision_types = set(TOTAL_COLLISION_FINITE_ATLAS_CHART_TYPES)
    ordinary_count = sum(chart_type in ordinary_types for chart_type in chart_types)
    binary_count = sum(chart_type in binary_types for chart_type in chart_types)
    total_collision_count = sum(
        chart_type in total_collision_types for chart_type in chart_types
    )
    unknown_chart_types = tuple(
        chart_type
        for chart_type in chart_types
        if chart_type not in ordinary_types
        and chart_type not in binary_types
        and chart_type not in total_collision_types
    )
    collision_policy = getattr(validated_atlas, "collision_policy", None)
    obligations = (
        _required_constructor_obligation(
            "positive_mass_noncollision_input_domain",
            input_domain_certificate,
            ("certified",),
        ),
        _required_constructor_obligation(
            "validated_atlas_solution",
            validated_atlas,
            ("proof_certified",),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_mass_consistency",
            certified=_finite_atlas_mass_consistency_certified(
                input_domain_certificate,
                validated_atlas,
            ),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_mass_consistency_detail(
                input_domain_certificate,
                validated_atlas,
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_initial_state_consistency",
            certified=_finite_atlas_initial_state_consistency_certified(
                input_domain_certificate,
                validated_atlas,
            ),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_initial_state_consistency_detail(
                input_domain_certificate,
                validated_atlas,
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_newton_residual_ledger",
            certified=_finite_atlas_residual_ledger_certified(validated_atlas, len(chart_types)),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_residual_detail(validated_atlas),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_projection_domain_ledger",
            certified=_finite_atlas_projection_domain_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="all finite charts must carry certified dynamics, finite chart domains, and projection witnesses",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_invariant_ledger",
            certified=_finite_atlas_invariant_ledger_certified(validated_atlas, len(chart_types)),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_invariant_detail(validated_atlas),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_tail_budget",
            certified=_finite_atlas_tail_budget_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="global tail budget and every chart tail bound must be finite and certified",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_transition_ledger",
            certified=_finite_atlas_transition_ledger_certified(validated_atlas, len(chart_types)),
            source=type(validated_atlas).__name__,
            detail=(
                f"chart_count={len(chart_types)}; "
                f"transition_count={len(getattr(validated_atlas, 'transitions', ()))}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_target_interval",
            certified=_finite_atlas_target_interval_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="target interval must be a nonempty finite coordinate interval array",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_target_time_domain",
            certified=bool(getattr(validated_atlas, "target_time_certified", False)),
            source=type(validated_atlas).__name__,
            detail="requested physical target time must lie in the final chart time interval",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_targeting_or_containment",
            certified=_finite_atlas_targeting_or_containment_certified(validated_atlas),
            source=type(validated_atlas).__name__,
            detail="requires target_time, finite_time_physical_targeting, or target_containment ledger evidence",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_proof_ledger",
            certified=bool(getattr(getattr(validated_atlas, "proof_ledger", None), "certified", False)),
            source=type(validated_atlas).__name__,
            detail="typed proof ledger must certify every required finite-atlas entry",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_selector_trace",
            certified=_finite_atlas_selector_trace_certified(validated_atlas, chart_types),
            source=type(validated_atlas).__name__,
            detail=_finite_atlas_selector_trace_detail(validated_atlas, chart_types),
        ),
        TheoremPipelineObligation(
            obligation="finite_chart_family_count",
            certified=bool(len(chart_types) > 0 and np.isfinite(len(chart_types))),
            source=type(validated_atlas).__name__,
            detail=f"chart_count={len(chart_types)}",
        ),
        TheoremPipelineObligation(
            obligation="ordinary_binary_or_selector_chart_families",
            certified=bool(
                chart_types
                and ordinary_count + binary_count + total_collision_count == len(chart_types)
            ),
            source=type(validated_atlas).__name__,
            detail="unknown="
            + ",".join(unknown_chart_types)
            + "; chart_types="
            + ",".join(chart_types),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_total_collision_chart_present",
            certified=bool(total_collision_count > 0),
            source=type(validated_atlas).__name__,
            detail=f"total_collision_chart_count={total_collision_count}",
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_total_collision_selector_coverage",
            certified=bool(
                len(selector_envelopes) >= total_collision_count
                and all(_selector_envelope_certified(selector) for selector in selector_envelopes)
            ),
            source="finite_zero_angular_selector_entry_constructors",
            detail=(
                f"selector_count={len(selector_envelopes)}; "
                f"total_collision_chart_count={total_collision_count}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_total_collision_selector_mass_consistency",
            certified=_selector_envelopes_mass_consistent(
                input_domain_certificate,
                validated_atlas,
                selector_envelopes,
            ),
            source="finite_zero_angular_selector_entry_constructors",
            detail=_selector_envelope_mass_detail(
                input_domain_certificate,
                validated_atlas,
                selector_envelopes,
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_total_collision_selector_branch_compatibility",
            certified=_selector_envelopes_cover_total_collision_charts(
                validated_atlas,
                selector_envelopes,
                total_collision_count=total_collision_count,
            ),
            source="finite_zero_angular_selector_entry_constructors",
            detail=(
                "selector entries must match the total-collision chart branch "
                "central coefficient, energy limit, and punctured tau samples"
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_atlas_total_collision_selector_policy",
            certified=bool(
                total_collision_count > 0
                and getattr(collision_policy, "certified", False)
                and "selector" in str(getattr(collision_policy, "total_collision_policy", ""))
            ),
            source=type(validated_atlas).__name__,
            detail=str(getattr(collision_policy, "total_collision_policy", "missing")),
        ),
    )
    return CompactZeroAngularFiniteSelectorAtlasCertificate(
        input_domain_certificate=input_domain_certificate,
        validated_atlas=validated_atlas,
        total_collision_selector_envelopes=selector_envelopes,
        chart_types=chart_types,
        ordinary_chart_count=int(ordinary_count),
        separated_binary_chart_count=int(binary_count),
        total_collision_chart_count=int(total_collision_count),
        obligations=obligations,
    )


def _normalize_selector_envelopes(selector_envelopes: object) -> tuple[object, ...]:
    if isinstance(selector_envelopes, (bool, np.bool_)):
        raise TypeError("raw boolean selector envelopes are not constructor certificates")
    if selector_envelopes is None:
        return ()
    if isinstance(selector_envelopes, tuple):
        selectors = selector_envelopes
    elif isinstance(selector_envelopes, list):
        selectors = tuple(selector_envelopes)
    else:
        selectors = (selector_envelopes,)
    for selector in selectors:
        _reject_raw_bool("total_collision_selector_envelope", selector)
    return tuple(selectors)


def _selector_envelope_certified(selector: object) -> bool:
    return bool(
        getattr(selector, "certified", False)
        or getattr(selector, "identity_selector_certified", False)
        or getattr(selector, "proof_certified", False)
    )


def _selector_envelopes_mass_consistent(
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object,
    selector_envelopes: tuple[object, ...],
) -> bool:
    if not selector_envelopes:
        return False
    atlas_masses = getattr(validated_atlas, "masses", None)
    return bool(
        all(
            _mass_sequences_consistent(
                input_domain_certificate.masses,
                _selector_envelope_masses(selector),
            )
            and _mass_sequences_consistent(
                atlas_masses,
                _selector_envelope_masses(selector),
            )
            for selector in selector_envelopes
        )
    )


def _selector_envelope_mass_detail(
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object,
    selector_envelopes: tuple[object, ...],
) -> str:
    return (
        "input_masses="
        + ",".join(str(mass) for mass in input_domain_certificate.masses)
        + "; atlas_masses="
        + ",".join(str(mass) for mass in getattr(validated_atlas, "masses", ()))
        + "; selector_masses="
        + ";".join(
            ",".join(str(mass) for mass in (_selector_envelope_masses(selector) or ()))
            or "missing"
            for selector in selector_envelopes
        )
    )


def _selector_envelopes_cover_total_collision_charts(
    validated_atlas: object,
    selector_envelopes: tuple[object, ...],
    *,
    total_collision_count: int,
) -> bool:
    total_collision_charts = tuple(
        chart
        for chart in getattr(validated_atlas, "charts", ())
        if str(getattr(chart, "chart_type", ""))
        in TOTAL_COLLISION_FINITE_ATLAS_CHART_TYPES
    )
    if (
        total_collision_count <= 0
        or len(total_collision_charts) != int(total_collision_count)
        or not selector_envelopes
    ):
        return False
    branch = getattr(getattr(validated_atlas, "evaluation", None), "branch", None)
    return bool(
        branch is not None
        and all(
            any(
                _selector_envelope_covers_total_collision_chart(
                    selector,
                    chart,
                    branch,
                )
                for selector in selector_envelopes
            )
            for chart in total_collision_charts
        )
    )


def _selector_envelope_covers_total_collision_chart(
    selector: object,
    chart: object,
    branch: object,
) -> bool:
    if not _selector_envelope_certified(selector):
        return False
    branch_masses = getattr(branch, "masses", None)
    if not _mass_sequences_consistent(branch_masses, _selector_envelope_masses(selector)):
        return False
    try:
        expected_energy = float(getattr(branch, "energy_per_inertia")) * float(
            getattr(branch, "inertia")
        )
    except (TypeError, ValueError):
        return False
    selector_energy = _selector_envelope_selected_energy_limit(selector)
    if not _finite_close(selector_energy, expected_energy, tolerance=1.0e-9):
        return False
    selector_central = _selector_envelope_central_coefficient(selector)
    branch_central = getattr(branch, "quadratic_coefficient", None)
    if not _array_close(selector_central, branch_central, tolerance=1.0e-9):
        return False
    return _selector_sample_taus_inside_chart(selector, chart)


def _selector_envelope_masses(selector: object) -> tuple[float, ...] | None:
    masses = getattr(selector, "masses", None)
    if masses is None:
        entry = getattr(selector, "entry_certificate", None)
        masses = getattr(entry, "masses", None)
    if masses is None:
        selected_branch = getattr(selector, "selected_branch", None)
        masses = getattr(selected_branch, "masses", None)
    if masses is None:
        return None
    try:
        masses_array = np.asarray(masses, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return tuple(float(mass) for mass in masses_array)


def _selector_envelope_selected_energy_limit(selector: object) -> float:
    for candidate in (
        selector,
        getattr(selector, "entry_certificate", None),
    ):
        if candidate is None:
            continue
        try:
            return float(getattr(candidate, "selected_energy_limit"))
        except (AttributeError, TypeError, ValueError):
            pass
    return float("nan")


def _selector_envelope_central_coefficient(selector: object) -> object | None:
    for candidate in (
        selector,
        getattr(selector, "entry_certificate", None),
    ):
        if candidate is None:
            continue
        central = getattr(candidate, "selected_central_coefficient", None)
        if central is not None:
            return central
        central = getattr(candidate, "incoming_central_coefficient", None)
        if central is not None:
            return central
    selected_branch = getattr(selector, "selected_branch", None)
    coefficients = getattr(selected_branch, "coefficients", None)
    if coefficients is not None:
        try:
            coefficients_array = np.asarray(coefficients, dtype=float)
            return coefficients_array[0]
        except (TypeError, ValueError, IndexError):
            return None
    return None


def _selector_sample_taus_inside_chart(selector: object, chart: object) -> bool:
    sample_taus = getattr(selector, "sample_taus", None)
    if sample_taus is None:
        entry = getattr(selector, "entry_certificate", None)
        sample_taus = getattr(entry, "sample_taus", None)
    if sample_taus is None:
        return False
    try:
        tau_values = tuple(float(tau) for tau in sample_taus)
    except (TypeError, ValueError):
        return False
    if not tau_values or not any(tau < 0.0 for tau in tau_values) or not any(tau > 0.0 for tau in tau_values):
        return False
    try:
        lower, upper = getattr(chart, "parameter_interval").as_tuple()
    except AttributeError:
        return False
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(float(lower)),
        abs(float(upper)),
        *(abs(tau) for tau in tau_values),
    )
    return bool(
        all(np.isfinite(tau) and tau != 0.0 for tau in tau_values)
        and all(float(lower) - tolerance <= tau <= float(upper) + tolerance for tau in tau_values)
    )


def _array_close(left: object, right: object, *, tolerance: float) -> bool:
    try:
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
    except (TypeError, ValueError):
        return False
    if left_array.shape != right_array.shape or left_array.size == 0:
        return False
    scale = max(
        1.0,
        float(np.max(np.abs(left_array))),
        float(np.max(np.abs(right_array))),
    )
    return bool(
        np.all(np.isfinite(left_array))
        and np.all(np.isfinite(right_array))
        and np.max(np.abs(left_array - right_array)) <= float(tolerance) * scale
    )


def _finite_atlas_binary_scope_certified(
    *,
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object,
    planar_binary_count: int,
    spatial_binary_count: int,
) -> bool:
    if planar_binary_count < 0 or spatial_binary_count < 0:
        return False
    if planar_binary_count and input_domain_certificate.dimension != 2:
        return False
    if spatial_binary_count == 0:
        return True
    collision_policy = getattr(validated_atlas, "collision_policy", None)
    return bool(
        input_domain_certificate.dimension == 3
        and getattr(collision_policy, "certified", False)
        and getattr(collision_policy, "binary_policy", "")
        in {
            "spatial_ks_selected_binary_entry_and_local_regularization",
            "spatial_ks_initial_close_binary_local_regularization",
            "spatial_ks_selected_binary_local_regularized",
            "spatial_ks_selected_binary_local_regularized_target_inside_ks",
            "spatial_ordinary_entry_then_competing_ks_regularization",
            "spatial_ordinary_entry_then_constructor_certified_ks_suffix",
            "spatial_ks_selected_binary_then_competing_binary",
            "spatial_ks_event_order_partition_member_regularization",
            "spatial_ks_branch_union_selected_binary_regularization",
        }
    )


def _finite_atlas_mass_consistency_certified(
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object,
) -> bool:
    input_masses = np.asarray(input_domain_certificate.masses, dtype=float).reshape(-1)
    try:
        atlas_masses = np.asarray(getattr(validated_atlas, "masses"), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(input_masses))) if input_masses.size else 1.0,
        float(np.max(np.abs(atlas_masses))) if atlas_masses.size else 1.0,
    )
    return bool(
        input_masses.shape == atlas_masses.shape
        and input_masses.size > 0
        and np.all(np.isfinite(input_masses))
        and np.all(np.isfinite(atlas_masses))
        and np.all(input_masses > 0.0)
        and np.all(atlas_masses > 0.0)
        and np.all(np.abs(input_masses - atlas_masses) <= tolerance)
    )


def _finite_atlas_mass_consistency_detail(
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object,
) -> str:
    return (
        "input_masses="
        + ",".join(str(mass) for mass in input_domain_certificate.masses)
        + "; atlas_masses="
        + ",".join(str(mass) for mass in getattr(validated_atlas, "masses", ()))
    )


def _finite_atlas_initial_state_consistency_certified(
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object,
) -> bool:
    try:
        input_state = _packed_input_domain_initial_state(input_domain_certificate)
    except (TypeError, ValueError):
        return False
    return _interval_array_contains_finite_point(
        getattr(validated_atlas, "initial_state_interval", None),
        input_state,
    )


def _finite_atlas_initial_state_consistency_detail(
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object,
) -> str:
    try:
        input_state = _packed_input_domain_initial_state(input_domain_certificate)
    except (TypeError, ValueError):
        input_state = np.asarray([], dtype=float)
    atlas_interval = getattr(validated_atlas, "initial_state_interval", None)
    try:
        atlas_size = int(np.asarray(atlas_interval, dtype=object).reshape(-1).size)
    except (TypeError, ValueError):
        atlas_size = 0
    return (
        f"input_state_size={input_state.size}; "
        f"atlas_initial_interval_size={atlas_size}; "
        f"max_violation={_interval_point_max_violation(atlas_interval, input_state)}"
    )


def _packed_input_domain_initial_state(
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
) -> np.ndarray:
    positions = np.asarray(input_domain_certificate.positions, dtype=float)
    velocities = np.asarray(input_domain_certificate.velocities, dtype=float)
    if positions.shape != velocities.shape or positions.ndim != 2:
        raise ValueError("input-domain positions and velocities must have matching rank-two shape")
    return np.concatenate([positions.reshape(-1), velocities.reshape(-1)])


def _interval_array_contains_finite_point(intervals: object, point: np.ndarray) -> bool:
    try:
        interval_array = np.asarray(intervals, dtype=object).reshape(-1)
        point_array = np.asarray(point, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    if interval_array.size == 0 or interval_array.size != point_array.size:
        return False
    for interval, value in zip(interval_array, point_array, strict=True):
        lower, upper = _interval_bounds(interval)
        value = float(value)
        if not (
            np.isfinite(lower)
            and np.isfinite(upper)
            and np.isfinite(value)
            and lower <= upper
            and lower <= value <= upper
        ):
            return False
    return True


def _interval_point_max_violation(intervals: object, point: np.ndarray) -> float:
    try:
        interval_array = np.asarray(intervals, dtype=object).reshape(-1)
        point_array = np.asarray(point, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return float("inf")
    if interval_array.size == 0 or interval_array.size != point_array.size:
        return float("inf")
    violations: list[float] = []
    for interval, value in zip(interval_array, point_array, strict=True):
        lower, upper = _interval_bounds(interval)
        value = float(value)
        if not (np.isfinite(lower) and np.isfinite(upper) and np.isfinite(value)):
            return float("inf")
        violations.append(max(lower - value, value - upper, 0.0))
    return float(max(violations, default=float("inf")))


def _interval_bounds(interval: object) -> tuple[float, float]:
    if isinstance(interval, (tuple, list)) and len(interval) == 2:
        lower, upper = interval
    else:
        try:
            lower, upper = interval.as_tuple()
        except AttributeError:
            lower = getattr(interval, "lower", np.nan)
            upper = getattr(interval, "upper", np.nan)
    return float(lower), float(upper)


def _finite_atlas_residual_ledger_certified(
    validated_atlas: object,
    chart_count: int,
) -> bool:
    residual = getattr(validated_atlas, "residual_budget", None)
    return bool(
        getattr(residual, "certified", False)
        and int(getattr(residual, "expected_chart_count", -1)) == int(chart_count)
        and int(getattr(residual, "certified_chart_count", -1)) >= int(chart_count)
        and all(
            bool(getattr(chart, "residual_certified", False))
            for chart in getattr(validated_atlas, "charts", ())
        )
    )


def _finite_atlas_residual_detail(validated_atlas: object) -> str:
    residual = getattr(validated_atlas, "residual_budget", None)
    return (
        f"expected={getattr(residual, 'expected_chart_count', 'missing')}; "
        f"certified={getattr(residual, 'certified_chart_count', 'missing')}"
    )


def _finite_atlas_projection_domain_certified(validated_atlas: object) -> bool:
    charts = tuple(getattr(validated_atlas, "charts", ()))
    return bool(
        charts
        and all(
            bool(getattr(chart, "dynamics_certified", False))
            and bool(getattr(chart, "projection_certified", False))
            and _finite_scalar_interval_certified(getattr(chart, "parameter_interval", None))
            and _finite_scalar_interval_certified(getattr(chart, "physical_time_interval", None))
            for chart in charts
        )
    )


def _finite_atlas_invariant_ledger_certified(
    validated_atlas: object,
    chart_count: int,
) -> bool:
    invariants = getattr(validated_atlas, "invariants", None)
    return bool(
        getattr(invariants, "certified", False)
        and int(getattr(invariants, "expected_chart_count", -1)) == int(chart_count)
        and int(getattr(invariants, "certified_chart_count", -1)) >= int(chart_count)
        and all(
            bool(getattr(chart, "invariants_certified", False))
            for chart in getattr(validated_atlas, "charts", ())
        )
    )


def _finite_atlas_invariant_detail(validated_atlas: object) -> str:
    invariants = getattr(validated_atlas, "invariants", None)
    return (
        f"expected={getattr(invariants, 'expected_chart_count', 'missing')}; "
        f"certified={getattr(invariants, 'certified_chart_count', 'missing')}"
    )


def _finite_atlas_tail_budget_certified(validated_atlas: object) -> bool:
    tail_budget = getattr(validated_atlas, "tail_budget", None)
    charts = tuple(getattr(validated_atlas, "charts", ()))
    return bool(
        getattr(tail_budget, "certified", False)
        and getattr(tail_budget, "finite", False)
        and charts
        and all(
            bool(getattr(chart, "tail_certified", False))
            and np.isfinite(float(getattr(chart, "tail_bound", np.nan)))
            for chart in charts
        )
    )


def _finite_atlas_transition_ledger_certified(
    validated_atlas: object,
    chart_count: int,
) -> bool:
    charts = tuple(getattr(validated_atlas, "charts", ()))
    transitions = tuple(getattr(validated_atlas, "transitions", ()))
    return bool(
        len(charts) == int(chart_count)
        and len(transitions) == max(0, int(chart_count) - 1)
        and all(bool(getattr(transition, "certified", False)) for transition in transitions)
        and all(
            getattr(transition, "source_chart_id", None) == getattr(left, "chart_id", None)
            and getattr(transition, "target_chart_id", None) == getattr(right, "chart_id", None)
            for transition, left, right in zip(transitions, charts, charts[1:])
        )
    )


def _finite_atlas_target_interval_certified(validated_atlas: object) -> bool:
    try:
        target_interval = getattr(validated_atlas, "target_state_interval")
    except Exception:
        return False
    values = np.asarray(target_interval, dtype=object).reshape(-1)
    return bool(values.size > 0 and all(_finite_scalar_interval_certified(value) for value in values))


def _finite_atlas_targeting_or_containment_certified(validated_atlas: object) -> bool:
    return bool(
        _proof_ledger_has_certified_entry(
            validated_atlas,
            (
                "target_time",
                "finite_time_physical_targeting",
                "hybrid_target_containment",
            ),
        )
    )


def _finite_atlas_selector_trace_certified(
    validated_atlas: object,
    chart_types: tuple[str, ...],
) -> bool:
    trace = getattr(validated_atlas, "selector_trace", None)
    if trace is None:
        return True
    return bool(
        getattr(trace, "certified", False)
        and _finite_atlas_selector_route_matches_chart_family(trace, chart_types)
    )


def _finite_atlas_selector_trace_detail(
    validated_atlas: object,
    chart_types: tuple[str, ...],
) -> str:
    trace = getattr(validated_atlas, "selector_trace", None)
    if trace is None:
        return "no public finite-time selector trace supplied by this direct constructor"
    selected = str(getattr(trace, "selected_route_id", "missing"))
    attempts = ",".join(
        str(getattr(attempt, "route_id", "missing"))
        for attempt in getattr(trace, "attempts", ())
    )
    return (
        f"selected={selected}; attempts={attempts}; "
        f"trace_certified={bool(getattr(trace, 'certified', False))}; "
        f"chart_family_match={_finite_atlas_selector_route_matches_chart_family(trace, chart_types)}"
    )


def _finite_atlas_selector_route_matches_chart_family(
    trace: object,
    chart_types: tuple[str, ...],
) -> bool:
    selected = str(getattr(trace, "selected_route_id", ""))
    chart_type_set = set(chart_types)
    if selected in {"auto_spatial_ks", "explicit_spatial_ks"}:
        return "spatial_ks_binary" in chart_type_set
    if selected == "spatial_ks_event_order_branch_union":
        return bool(
            chart_types
            and chart_type_set <= {"spatial_ks_event_order_branch_union"}
        )
    if selected == "spatial_ks_prefix_event_order_branch_union":
        return bool(
            chart_types
            and "spatial_ks_event_order_branch_union" in chart_type_set
            and (
                "spatial_ks_binary" in chart_type_set
                or "spatial_ordinary_taylor_before_ks" in chart_type_set
            )
            and chart_type_set
            <= {
                "spatial_ordinary_taylor_before_ks",
                "spatial_ks_binary",
                "spatial_ks_event_order_branch_union",
            }
        )
    if selected == "spatial_ks_prefix_close_pair_branch_union":
        return bool(
            chart_types
            and "spatial_ks_binary" in chart_type_set
            and "spatial_branch_union" in chart_type_set
            and chart_type_set
            <= {
                "spatial_ordinary_taylor_before_ks",
                "spatial_ks_binary",
                "spatial_branch_union",
            }
        )
    if selected == "spatial_branch_union":
        return bool(chart_types and chart_type_set <= {"spatial_branch_union"})
    if selected == "planar_hybrid":
        return bool(
            chart_types
            and chart_type_set <= {
                "planar_ordinary_taylor",
                "planar_levi_civita_binary",
            }
        )
    if selected == "compactified_sundman":
        return bool(
            chart_types
            and chart_type_set <= {
                "compactified_sundman",
                "compactified_sundman_target",
            }
        )
    if selected == "sundman":
        return bool(
            chart_types
            and chart_type_set <= {
                "sundman",
                "sundman_target",
            }
        )
    if selected == "unrestricted_reduced_evaluation":
        return bool(chart_types)
    return False


def _proof_ledger_has_certified_entry(
    validated_atlas: object,
    names: tuple[str, ...],
) -> bool:
    ledger = getattr(validated_atlas, "proof_ledger", None)
    entries = tuple(getattr(ledger, "entries", ()))
    allowed = set(names)
    return any(
        str(getattr(entry, "name", "")) in allowed
        and bool(getattr(entry, "certified", False))
        for entry in entries
    )


def _finite_scalar_interval_certified(interval: object | None) -> bool:
    if interval is None:
        return False
    if hasattr(interval, "as_tuple"):
        lower, upper = interval.as_tuple()
    elif hasattr(interval, "lower") and hasattr(interval, "upper"):
        lower = getattr(interval, "lower")
        upper = getattr(interval, "upper")
    else:
        return False
    lower = float(lower)
    upper = float(upper)
    return bool(np.isfinite(lower) and np.isfinite(upper) and lower <= upper)


def _finite_middle_detail(certificate: object | None) -> str:
    if certificate is None:
        return "no finite validated atlas certificate supplied"
    missing = tuple(getattr(certificate, "missing_obligations", ()))
    if missing:
        return "missing=" + ",".join(str(obligation) for obligation in missing)
    return "finite compact atlas certificate is proof-certified"


def _finite_middle_target_time_positive(validated_atlas: object | None) -> bool:
    try:
        target_time = float(getattr(validated_atlas, "target_time"))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(target_time) and target_time > 0.0)


def _finite_middle_target_time_negative(validated_atlas: object | None) -> bool:
    try:
        target_time = float(getattr(validated_atlas, "target_time"))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(target_time) and target_time < 0.0)


def _finite_middle_validated_atlas(
    finite_middle_atlas: object | None,
    field_name: str,
) -> object | None:
    certificate = getattr(finite_middle_atlas, field_name, None)
    return getattr(certificate, "validated_atlas", None)


def _finite_middle_endpoint_metrics(
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    validated_atlas: object | None,
) -> Mapping[str, float]:
    if validated_atlas is None:
        return {}
    try:
        state_intervals = np.asarray(
            getattr(validated_atlas, "target_state_interval"),
            dtype=object,
        ).reshape(-1)
    except (TypeError, ValueError):
        return {}
    masses = tuple(input_domain_certificate.masses)
    body_count = len(masses)
    dimension = int(input_domain_certificate.dimension)
    expected_size = 2 * body_count * dimension
    if (
        body_count < 2
        or dimension < 1
        or state_intervals.size != expected_size
        or not all(_finite_scalar_interval_certified(value) for value in state_intervals)
    ):
        return {}
    position_intervals = state_intervals[: body_count * dimension].reshape(
        body_count,
        dimension,
    )
    velocity_intervals = state_intervals[body_count * dimension :].reshape(
        body_count,
        dimension,
    )
    pair_distance_lowers = []
    pair_distance_uppers = []
    for i in range(body_count):
        for j in range(i + 1, body_count):
            squared = _interval_vector_squared_norm(
                tuple(
                    _interval_subtract(position_intervals[j, axis], position_intervals[i, axis])
                    for axis in range(dimension)
                )
            )
            pair_distance_lowers.append(_safe_sqrt(squared[0]))
            pair_distance_uppers.append(_safe_sqrt(squared[1]))
    speed_uppers = []
    for body in range(body_count):
        squared = _interval_vector_squared_norm(
            tuple(velocity_intervals[body, axis] for axis in range(dimension))
        )
        speed_uppers.append(_safe_sqrt(squared[1]))
    metrics = {
        "pair_distance_lower": float(min(pair_distance_lowers, default=float("nan"))),
        "pair_diameter_upper": float(max(pair_distance_uppers, default=float("nan"))),
        "speed_upper": float(max(speed_uppers, default=float("nan"))),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        return {}
    return metrics


def _finite_middle_target_compact_parameter(
    compact_time_certificate: CompactTimeCoverageCertificate,
    validated_atlas: object | None,
) -> float:
    try:
        target_time = float(getattr(validated_atlas, "target_time"))
        compact_parameter = compact_time_certificate.compact_parameter(target_time)
    except (AttributeError, TypeError, ValueError):
        return float("nan")
    return float(compact_parameter)


def _future_event_shell_start(
    envelope_spec: NonzeroAngularUniformPairEventEnvelopeSpec,
) -> float:
    return float(1.0 - float(envelope_spec.delta_initial))


def _past_event_shell_start(
    envelope_spec: NonzeroAngularUniformPairEventEnvelopeSpec,
) -> float:
    return float(-1.0 + float(envelope_spec.delta_initial))


def _future_event_shell_end(
    envelope_spec: object | None,
) -> float:
    try:
        return float(
            1.0
            - float(envelope_spec.delta_initial) * float(envelope_spec.theta)
        )
    except (AttributeError, TypeError, ValueError):
        return float("nan")


def _past_event_shell_end(
    envelope_spec: object | None,
) -> float:
    try:
        return float(
            -1.0
            + float(envelope_spec.delta_initial) * float(envelope_spec.theta)
        )
    except (AttributeError, TypeError, ValueError):
        return float("nan")


def _compact_physical_time_or_nan(
    compact_time_certificate: CompactTimeCoverageCertificate,
    compact_parameter: float,
) -> float:
    try:
        return float(compact_time_certificate.physical_time(float(compact_parameter)))
    except (TypeError, ValueError):
        return float("nan")


def _compact_parameter_matches_event_shell_start(
    *,
    observed: float,
    expected: float,
    tolerance: float = 1.0e-10,
) -> bool:
    observed = float(observed)
    expected = float(expected)
    return bool(
        np.isfinite(observed)
        and np.isfinite(expected)
        and abs(observed - expected) <= tolerance * max(1.0, abs(expected))
    )


def _evaluate_nonzero_angular_validated_atlas(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    target_time: float,
    order: int,
    max_compact_step: float,
    max_s_step: float,
    target_bisections: int,
) -> object | None:
    from .general_solution import evaluate_unrestricted_solution

    try:
        return evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            float(target_time),
            method="validated_atlas",
            order=int(order),
            max_compact_step=float(max_compact_step),
            max_s_step=float(max_s_step),
            target_bisections=int(target_bisections),
        )
    except (TypeError, ValueError, RuntimeError, FloatingPointError):
        return None


def _handoff_target_time(
    event_regime_handoff: object | None,
    direction: str,
) -> float:
    atlas = _finite_middle_validated_atlas(
        getattr(event_regime_handoff, "finite_middle_atlas", None),
        "future_finite_atlas" if direction == "future" else "past_finite_atlas",
    )
    try:
        return float(getattr(atlas, "target_time"))
    except (TypeError, ValueError):
        return float("nan")


def _prefix_extends_handoff(
    event_regime_handoff: object | None,
    prefix_validated_atlas: object | None,
    *,
    direction: str,
) -> bool:
    handoff_time = _handoff_target_time(event_regime_handoff, direction)
    try:
        prefix_time = float(getattr(prefix_validated_atlas, "target_time"))
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(handoff_time) and np.isfinite(prefix_time)):
        return False
    if direction == "future":
        if not (0.0 < handoff_time < prefix_time):
            return False
    elif direction == "past":
        if not (prefix_time < handoff_time < 0.0):
            return False
    else:
        raise ValueError("direction must be future or past")
    return _validated_atlas_domain_contains_time(prefix_validated_atlas, handoff_time)


def _validated_atlas_domain_contains_time(
    validated_atlas: object | None,
    physical_time: float,
) -> bool:
    physical_time = float(physical_time)
    if not np.isfinite(physical_time):
        return False
    for chart in getattr(validated_atlas, "charts", ()):
        interval = getattr(chart, "physical_time_interval", None)
        if not _finite_scalar_interval_certified(interval):
            continue
        lower, upper = _interval_bounds(interval)
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(physical_time), abs(lower), abs(upper))
        if lower - tolerance <= physical_time <= upper + tolerance:
            return True
    return False


def _endpoint_metrics_inside_ordinary_gap(
    metrics: Mapping[str, float],
    envelope_spec: NonzeroAngularUniformPairEventEnvelopeSpec | object | None,
) -> bool:
    if envelope_spec is None:
        return False
    try:
        pair_distance_lower = float(metrics["pair_distance_lower"])
        pair_diameter_upper = float(metrics["pair_diameter_upper"])
        speed_upper = float(metrics["speed_upper"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(pair_distance_lower)
        and np.isfinite(pair_diameter_upper)
        and np.isfinite(speed_upper)
        and pair_distance_lower >= float(envelope_spec.ordinary_pair_distance_lower_bound)
        and pair_diameter_upper <= float(envelope_spec.ordinary_pair_diameter_upper_bound)
        and speed_upper <= float(envelope_spec.ordinary_speed_upper_bound)
    )


def _endpoint_envelope_detail(
    metrics: Mapping[str, float],
    envelope_spec: NonzeroAngularUniformPairEventEnvelopeSpec | object | None,
) -> str:
    if envelope_spec is None:
        return "no event-envelope spec supplied"
    return (
        "endpoint_pair_distance_lower="
        + str(metrics.get("pair_distance_lower", "missing"))
        + f"; required_pair_distance_lower={envelope_spec.ordinary_pair_distance_lower_bound}; "
        + "endpoint_pair_diameter_upper="
        + str(metrics.get("pair_diameter_upper", "missing"))
        + f"; required_pair_diameter_upper={envelope_spec.ordinary_pair_diameter_upper_bound}; "
        + "endpoint_speed_upper="
        + str(metrics.get("speed_upper", "missing"))
        + f"; required_speed_upper={envelope_spec.ordinary_speed_upper_bound}; "
        + f"time_direction={envelope_spec.time_direction}"
    )


def _event_budget_component_infinite_bound(
    event_regime_assembly: object | None,
    component: str,
) -> float:
    try:
        budget = event_regime_assembly.event_budget.event_budget
        return float(budget.component_infinite_scalar_bound(str(component)))
    except (AttributeError, KeyError, TypeError, ValueError):
        return float("inf")


def _event_budget_component_tail_from_prefix(
    event_regime_assembly: object | None,
    component: str,
    *,
    prefix_length: int,
) -> float:
    try:
        budget = event_regime_assembly.event_budget.event_budget
        return float(
            budget.component_scalar_tail_from_prefix(
                str(component),
                prefix_length=int(prefix_length),
            )
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return float("inf")


def _handoff_metric_margins(
    event_regime_handoff: object | None,
    direction: str,
) -> Mapping[str, float]:
    if direction == "future":
        metrics = getattr(event_regime_handoff, "future_metrics", {})
        spec = getattr(event_regime_handoff, "future_envelope_spec", None)
    elif direction == "past":
        metrics = getattr(event_regime_handoff, "past_metrics", {})
        spec = getattr(event_regime_handoff, "past_envelope_spec", None)
    else:
        raise ValueError("direction must be future or past")
    try:
        return {
            "pair_distance": float(metrics["pair_distance_lower"])
            - float(spec.ordinary_pair_distance_lower_bound),
            "pair_diameter": float(spec.ordinary_pair_diameter_upper_bound)
            - float(metrics["pair_diameter_upper"]),
            "speed": float(spec.ordinary_speed_upper_bound)
            - float(metrics["speed_upper"]),
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return {
            "pair_distance": float("-inf"),
            "pair_diameter": float("-inf"),
            "speed": float("-inf"),
        }


def _prefix_metric_margins(
    first_event_shell_prefix: object | None,
    direction: str,
) -> Mapping[str, float]:
    handoff = getattr(first_event_shell_prefix, "event_regime_handoff", None)
    if direction == "future":
        metrics = getattr(first_event_shell_prefix, "future_end_metrics", {})
        spec = getattr(handoff, "future_envelope_spec", None)
    elif direction == "past":
        metrics = getattr(first_event_shell_prefix, "past_end_metrics", {})
        spec = getattr(handoff, "past_envelope_spec", None)
    else:
        raise ValueError("direction must be future or past")
    try:
        return {
            "pair_distance": float(metrics["pair_distance_lower"])
            - float(spec.ordinary_pair_distance_lower_bound),
            "pair_diameter": float(spec.ordinary_pair_diameter_upper_bound)
            - float(metrics["pair_diameter_upper"]),
            "speed": float(spec.ordinary_speed_upper_bound)
            - float(metrics["speed_upper"]),
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return {
            "pair_distance": float("-inf"),
            "pair_diameter": float("-inf"),
            "speed": float("-inf"),
        }


def _required_coordinate_tail_margin(
    value_tail_bound: float,
    *,
    dimension: int,
) -> float:
    value_tail_bound = float(value_tail_bound)
    dimension = int(dimension)
    if not (np.isfinite(value_tail_bound) and value_tail_bound >= 0.0 and dimension > 0):
        return float("inf")
    return float(2.0 * np.sqrt(float(dimension)) * value_tail_bound)


def _tail_bound_fits_metric_margins(
    value_tail_bound: float,
    margins: Mapping[str, float],
    *,
    dimension: int,
) -> bool:
    value_tail_bound = float(value_tail_bound)
    dimension = int(dimension)
    if not (np.isfinite(value_tail_bound) and value_tail_bound >= 0.0 and dimension > 0):
        return False
    pair_required = 2.0 * np.sqrt(float(dimension)) * value_tail_bound
    speed_required = np.sqrt(float(dimension)) * value_tail_bound
    try:
        pair_distance_margin = float(margins["pair_distance"])
        pair_diameter_margin = float(margins["pair_diameter"])
        speed_margin = float(margins["speed"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(pair_distance_margin)
        and np.isfinite(pair_diameter_margin)
        and np.isfinite(speed_margin)
        and pair_distance_margin > pair_required
        and pair_diameter_margin > pair_required
        and speed_margin > speed_required
    )


def _tail_margin_detail(
    value_tail_bound: float,
    required_coordinate_margin: float,
    margins: Mapping[str, float],
) -> str:
    return (
        f"value_tail_bound={value_tail_bound}; "
        f"pair_metric_required_margin={required_coordinate_margin}; "
        f"speed_required_margin={0.5 * required_coordinate_margin}; "
        "margins="
        + ",".join(
            f"{name}:{margins.get(name, 'missing')}"
            for name in ("pair_distance", "pair_diameter", "speed")
        )
    )


def _event_budget_shell_geometry_matches_spec(
    event_regime_assembly: object | None,
    envelope_spec: object | None,
) -> bool:
    shell_isolation = getattr(event_regime_assembly, "shell_isolation", None)
    if not bool(getattr(shell_isolation, "certified", False)):
        return False
    return bool(
        _finite_close(
            getattr(shell_isolation, "delta_initial", float("nan")),
            getattr(envelope_spec, "delta_initial", float("nan")),
        )
        and _finite_close(
            getattr(shell_isolation, "theta", float("nan")),
            getattr(envelope_spec, "theta", float("nan")),
        )
        and _finite_close(
            getattr(shell_isolation, "event_isolation_initial", float("nan")),
            getattr(envelope_spec, "event_isolation_initial", float("nan")),
        )
        and _finite_close(
            getattr(shell_isolation, "boundary_clearance_initial", float("nan")),
            getattr(envelope_spec, "boundary_clearance_initial", float("nan")),
        )
    )


def _event_budget_shell_geometry_detail(
    event_regime_assembly: object | None,
    envelope_spec: object | None,
) -> str:
    shell_isolation = getattr(event_regime_assembly, "shell_isolation", None)
    return (
        "observed="
        f"delta:{getattr(shell_isolation, 'delta_initial', 'missing')},"
        f"theta:{getattr(shell_isolation, 'theta', 'missing')},"
        f"event_isolation:{getattr(shell_isolation, 'event_isolation_initial', 'missing')},"
        f"boundary_clearance:{getattr(shell_isolation, 'boundary_clearance_initial', 'missing')}; "
        "expected="
        f"delta:{getattr(envelope_spec, 'delta_initial', 'missing')},"
        f"theta:{getattr(envelope_spec, 'theta', 'missing')},"
        f"event_isolation:{getattr(envelope_spec, 'event_isolation_initial', 'missing')},"
        f"boundary_clearance:{getattr(envelope_spec, 'boundary_clearance_initial', 'missing')}"
    )


def _event_budget_component_majorant_growth_bounded(
    event_regime_assembly: object | None,
    component: str,
    *,
    growth_bound: float,
) -> bool:
    component = str(component)
    growth_bound = float(growth_bound)
    family_certificates = getattr(event_regime_assembly, "chart_family_certificates", ())
    if not family_certificates:
        return False
    observed = []
    for family_certificate in family_certificates:
        try:
            primitive = family_certificate.component_input(component)
            observed.append(float(primitive.majorant_growth))
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
    return bool(
        observed
        and np.isfinite(growth_bound)
        and all(
            np.isfinite(growth)
            and growth <= growth_bound * (1.0 + 1.0e-12)
            for growth in observed
        )
    )


def _ordinary_gap_source_matches_envelope(
    event_regime_assembly: object | None,
    envelope_spec: object | None,
) -> bool:
    source = _ordinary_gap_source_certificate(event_regime_assembly)
    if not bool(getattr(source, "certified", False)):
        return False
    return bool(
        _finite_close(
            getattr(source, "pair_distance_lower_bound", float("nan")),
            getattr(envelope_spec, "ordinary_pair_distance_lower_bound", float("nan")),
        )
        and _finite_close(
            getattr(source, "pair_diameter_upper_bound", float("nan")),
            getattr(envelope_spec, "ordinary_pair_diameter_upper_bound", float("nan")),
        )
        and _finite_close(
            getattr(source, "speed_upper_bound", float("nan")),
            getattr(envelope_spec, "ordinary_speed_upper_bound", float("nan")),
        )
    )


def _ordinary_gap_source_detail(
    event_regime_assembly: object | None,
    envelope_spec: object | None,
) -> str:
    source = _ordinary_gap_source_certificate(event_regime_assembly)
    return (
        "observed="
        f"pair_distance:{getattr(source, 'pair_distance_lower_bound', 'missing')},"
        f"pair_diameter:{getattr(source, 'pair_diameter_upper_bound', 'missing')},"
        f"speed:{getattr(source, 'speed_upper_bound', 'missing')}; "
        "expected="
        f"pair_distance:{getattr(envelope_spec, 'ordinary_pair_distance_lower_bound', 'missing')},"
        f"pair_diameter:{getattr(envelope_spec, 'ordinary_pair_diameter_upper_bound', 'missing')},"
        f"speed:{getattr(envelope_spec, 'ordinary_speed_upper_bound', 'missing')}"
    )


def _ordinary_gap_source_certificate(event_regime_assembly: object | None) -> object | None:
    try:
        family = event_regime_assembly.family_certificate("ordinary_gap_taylor")
        return getattr(family, "source_certificate", None)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


_SEPARATED_BINARY_ENVELOPE_FIELDS = (
    "z_bound",
    "z_velocity_bound",
    "pair_energy_bound",
    "binary_center_bound",
    "binary_center_velocity_bound",
    "third_offset_bound",
    "third_offset_velocity_bound",
    "third_body_nominal_distance_lower_bound",
)

_SEPARATED_BINARY_RADIUS_FIELDS = (
    "z",
    "z_velocity",
    "pair_energy",
    "binary_center",
    "binary_center_velocity",
    "third_offset",
    "third_offset_velocity",
)


def _separated_binary_sources_match_envelope(
    event_regime_assembly: object | None,
    envelope_spec: object | None,
) -> bool:
    normalized = _normalized_binary_pair_envelopes(envelope_spec)
    if set(normalized) != {(0, 1), (0, 2), (1, 2)}:
        return False
    return all(
        _separated_binary_source_matches_pair_envelope(
            _separated_binary_source_certificate(event_regime_assembly, pair),
            pair,
            pair_envelope,
        )
        for pair, pair_envelope in normalized.items()
    )


def _separated_binary_sources_detail(
    event_regime_assembly: object | None,
    envelope_spec: object | None,
) -> str:
    normalized = _normalized_binary_pair_envelopes(envelope_spec)
    if set(normalized) != {(0, 1), (0, 2), (1, 2)}:
        return "binary_pair_envelopes must contain exactly pairs 01, 02, and 12"
    fragments = []
    for pair, pair_envelope in normalized.items():
        source = _separated_binary_source_certificate(event_regime_assembly, pair)
        matches = _separated_binary_source_matches_pair_envelope(
            source,
            pair,
            pair_envelope,
        )
        fragments.append(
            f"{pair[0]}{pair[1]}:matches={matches}; "
            f"observed_z={getattr(source, 'z_bound', 'missing')}; "
            f"expected_z={_mapping_get(pair_envelope, 'z_bound')}; "
            f"observed_rz={_mapping_get(getattr(source, 'radii', {}), 'z')}; "
            f"expected_rz={_mapping_get(_mapping_get(pair_envelope, 'radii'), 'z')}"
        )
    return " | ".join(fragments)


def _separated_binary_source_certificate(
    event_regime_assembly: object | None,
    pair: tuple[int, int],
) -> object | None:
    kind = f"separated_binary_levi_civita_{pair[0]}{pair[1]}"
    try:
        family = event_regime_assembly.family_certificate(kind)
        return getattr(family, "source_certificate", None)
    except (AttributeError, KeyError, TypeError, ValueError):
        try:
            family = event_regime_assembly.family_certificate(
                "separated_binary_levi_civita",
            )
            source = getattr(family, "source_certificate", None)
            return (
                source
                if _binary_pair_from_object(getattr(source, "pair", None)) == pair
                else None
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None


def _separated_binary_source_matches_pair_envelope(
    source: object | None,
    pair: tuple[int, int],
    pair_envelope: Mapping[str, object],
) -> bool:
    if not bool(getattr(source, "certified", False)):
        return False
    if _binary_pair_from_object(getattr(source, "pair", None)) != pair:
        return False
    if not all(
        _finite_close(
            getattr(source, field, float("nan")),
            _mapping_get(pair_envelope, field),
        )
        for field in _SEPARATED_BINARY_ENVELOPE_FIELDS
    ):
        return False
    source_radii = getattr(source, "radii", {})
    envelope_radii = _mapping_get(pair_envelope, "radii")
    if not isinstance(source_radii, Mapping) or not isinstance(envelope_radii, Mapping):
        return False
    return all(
        _finite_close(
            _mapping_get(source_radii, radius_name),
            _mapping_get(envelope_radii, radius_name),
        )
        for radius_name in _SEPARATED_BINARY_RADIUS_FIELDS
    )


def _normalized_binary_pair_envelopes(
    envelope_spec: object | None,
) -> dict[tuple[int, int], Mapping[str, object]]:
    envelopes = getattr(envelope_spec, "binary_pair_envelopes", None)
    if not isinstance(envelopes, Mapping):
        return {}
    normalized: dict[tuple[int, int], Mapping[str, object]] = {}
    for pair, pair_envelope in envelopes.items():
        normalized_pair = _binary_pair_from_object(pair)
        if normalized_pair is None or not isinstance(pair_envelope, Mapping):
            return {}
        if normalized_pair in normalized:
            return {}
        normalized[normalized_pair] = pair_envelope
    return normalized


def _binary_pair_from_object(pair: object) -> tuple[int, int] | None:
    try:
        values = tuple(sorted(int(index) for index in pair))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(values) != 2 or values[0] == values[1]:
        return None
    if values[0] not in (0, 1, 2) or values[1] not in (0, 1, 2):
        return None
    return values


def _mapping_get(mapping: object, key: str) -> object:
    if not isinstance(mapping, Mapping):
        return float("nan")
    return mapping.get(key, float("nan"))


def _finite_close(left: object, right: object, *, tolerance: float = 1.0e-12) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    scale = max(1.0, abs(left_value), abs(right_value))
    return bool(
        np.isfinite(left_value)
        and np.isfinite(right_value)
        and abs(left_value - right_value) <= tolerance * scale
    )


def _interval_subtract(left: object, right: object) -> tuple[float, float]:
    left_lower, left_upper = _interval_bounds(left)
    right_lower, right_upper = _interval_bounds(right)
    return (
        float(np.nextafter(left_lower - right_upper, -np.inf)),
        float(np.nextafter(left_upper - right_lower, np.inf)),
    )


def _interval_square_bounds(interval: object) -> tuple[float, float]:
    lower, upper = _interval_bounds(interval)
    if lower <= 0.0 <= upper:
        squared_lower = 0.0
    else:
        squared_lower = min(lower * lower, upper * upper)
    squared_upper = max(lower * lower, upper * upper)
    return (
        float(np.nextafter(squared_lower, -np.inf)),
        float(np.nextafter(squared_upper, np.inf)),
    )


def _interval_vector_squared_norm(components: tuple[object, ...]) -> tuple[float, float]:
    lower = 0.0
    upper = 0.0
    for component in components:
        component_lower, component_upper = _interval_square_bounds(component)
        lower += component_lower
        upper += component_upper
    return (
        float(np.nextafter(max(0.0, lower), -np.inf)),
        float(np.nextafter(max(0.0, upper), np.inf)),
    )


def _safe_sqrt(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return float("nan")
    return float(np.sqrt(max(0.0, value)))


def construct_nonzero_angular_global_atlas(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    compact_time_rate: float,
    event_regime_assembly: object | None = None,
    past_event_regime_assembly: object | None = None,
    finite_middle_atlas: object | None = None,
    event_regime_handoff: object | None = None,
    event_tail_margin_certificate: object | None = None,
    first_event_shell_prefix: object | None = None,
    event_shell_invariance_certificate: object | None = None,
) -> GlobalAtlasCertificate:
    """Build the first all-time nonzero-angular atlas from derived certificates.

    This constructor derives the input-domain, compact-time, and Sundman
    nonzero-angular triple-collision exclusion certificates from the supplied
    initial data. It requires two all-future event-regime assemblies: one for
    the future endpoint of the supplied flow, and one for the future endpoint of
    the time-reversed flow, which is the past endpoint of the supplied flow.
    Absent or failed recurrence evidence is reported as a missing theorem
    obligation instead of being replaced by a hand-supplied boolean.
    """

    input_domain_certificate = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    compact_time_certificate = certify_compact_time_real_line_coverage(compact_time_rate)
    triple_collision_exclusion = certify_nonzero_angular_momentum_excludes_triple_collision(
        positions,
        velocities,
        masses,
    )
    two_sided_event_budget = certify_two_sided_nonzero_angular_event_budget(
        future_event_regime_assembly=event_regime_assembly,
        past_event_regime_assembly=past_event_regime_assembly,
    )
    _reject_raw_bool("finite_middle_atlas", finite_middle_atlas)
    _reject_raw_bool("event_regime_handoff", event_regime_handoff)
    _reject_raw_bool("event_tail_margin_certificate", event_tail_margin_certificate)
    _reject_raw_bool("first_event_shell_prefix", first_event_shell_prefix)
    _reject_raw_bool(
        "event_shell_invariance_certificate",
        event_shell_invariance_certificate,
    )
    classification = classify_global_regime(
        input_domain_certificate=input_domain_certificate,
        compact_time_certificate=compact_time_certificate,
        regime_id="all_time_nonzero_angular",
        finite_middle_atlas=finite_middle_atlas,
        event_regime_handoff=event_regime_handoff,
        event_tail_margin_certificate=event_tail_margin_certificate,
        first_event_shell_prefix=first_event_shell_prefix,
        event_shell_invariance_certificate=event_shell_invariance_certificate,
        triple_collision_exclusion_certificate=triple_collision_exclusion,
        event_isolation_certificate=two_sided_event_budget,
        primitive_cauchy_inputs=two_sided_event_budget,
    )
    return construct_global_atlas_for_regime(
        classification,
        event_budget=two_sided_event_budget,
        finite_middle_atlas=finite_middle_atlas,
    )


def construct_nonzero_angular_global_atlas_from_uniform_event_envelopes(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    compact_time_rate: float,
    delta_initial: float,
    theta: float,
    event_isolation_initial: float,
    boundary_clearance_initial: float,
    ordinary_pair_distance_lower_bound: float,
    ordinary_pair_diameter_upper_bound: float,
    ordinary_speed_upper_bound: float,
    binary_pair: tuple[int, int],
    binary_z_bound: float,
    binary_z_velocity_bound: float,
    binary_pair_energy_bound: float,
    binary_center_bound: float,
    binary_center_velocity_bound: float,
    binary_third_offset_bound: float,
    binary_third_offset_velocity_bound: float,
    binary_third_body_nominal_distance_lower_bound: float,
    binary_radii: Any,
    step_ratio_bounds: Any,
    retained_order_initials: Any,
    retained_order_increments: Any,
    checked_prefix: int,
    finite_middle_atlas: object | None = None,
) -> GlobalAtlasCertificate:
    """Build the nonzero-angular theorem from explicit uniform event envelopes.

    The supplied quantitative envelopes are still regime hypotheses, not an
    arbitrary-data classifier.  This constructor derives the binary-only
    all-future event recurrence from those envelopes before feeding the usual
    `all_time_nonzero_angular` theorem path.
    """

    event_regime_assembly = derive_nonzero_angular_event_recurrence_from_uniform_envelopes(
        masses=masses,
        delta_initial=delta_initial,
        theta=theta,
        event_isolation_initial=event_isolation_initial,
        boundary_clearance_initial=boundary_clearance_initial,
        ordinary_pair_distance_lower_bound=ordinary_pair_distance_lower_bound,
        ordinary_pair_diameter_upper_bound=ordinary_pair_diameter_upper_bound,
        ordinary_speed_upper_bound=ordinary_speed_upper_bound,
        binary_pair=binary_pair,
        binary_z_bound=binary_z_bound,
        binary_z_velocity_bound=binary_z_velocity_bound,
        binary_pair_energy_bound=binary_pair_energy_bound,
        binary_center_bound=binary_center_bound,
        binary_center_velocity_bound=binary_center_velocity_bound,
        binary_third_offset_bound=binary_third_offset_bound,
        binary_third_offset_velocity_bound=binary_third_offset_velocity_bound,
        binary_third_body_nominal_distance_lower_bound=binary_third_body_nominal_distance_lower_bound,
        binary_radii=binary_radii,
        step_ratio_bounds=step_ratio_bounds,
        retained_order_initials=retained_order_initials,
        retained_order_increments=retained_order_increments,
        checked_prefix=checked_prefix,
    )
    return construct_nonzero_angular_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=compact_time_rate,
        event_regime_assembly=event_regime_assembly,
        past_event_regime_assembly=event_regime_assembly,
        finite_middle_atlas=finite_middle_atlas,
    )


def construct_nonzero_angular_global_atlas_from_uniform_pair_event_envelopes(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    compact_time_rate: float,
    delta_initial: float,
    theta: float,
    event_isolation_initial: float,
    boundary_clearance_initial: float,
    ordinary_pair_distance_lower_bound: float,
    ordinary_pair_diameter_upper_bound: float,
    ordinary_speed_upper_bound: float,
    binary_pair_envelopes: Any,
    step_ratio_bounds: Any,
    retained_order_initials: Any,
    retained_order_increments: Any,
    checked_prefix: int,
    finite_middle_atlas: object | None = None,
    derive_finite_middle_atlas: bool = False,
    finite_middle_order: int = 8,
    finite_middle_max_compact_step: float = 5.0e-5,
    finite_middle_max_s_step: float = 0.02,
    finite_middle_target_bisections: int = 20,
    derive_first_event_shell_prefix: bool = False,
    first_event_shell_order: int = 8,
    first_event_shell_max_compact_step: float = 0.03,
    first_event_shell_max_s_step: float = 0.02,
    first_event_shell_target_bisections: int = 20,
    derive_event_shell_invariance: bool = False,
) -> GlobalAtlasCertificate:
    """Build the nonzero-angular theorem with one binary recurrence row per pair."""

    envelope_spec = NonzeroAngularUniformPairEventEnvelopeSpec(
        time_direction="time_reversal_invariant",
        delta_initial=float(delta_initial),
        theta=float(theta),
        event_isolation_initial=float(event_isolation_initial),
        boundary_clearance_initial=float(boundary_clearance_initial),
        ordinary_pair_distance_lower_bound=float(ordinary_pair_distance_lower_bound),
        ordinary_pair_diameter_upper_bound=float(ordinary_pair_diameter_upper_bound),
        ordinary_speed_upper_bound=float(ordinary_speed_upper_bound),
        binary_pair_envelopes=binary_pair_envelopes,
        step_ratio_bounds=step_ratio_bounds,
        retained_order_initials=retained_order_initials,
        retained_order_increments=retained_order_increments,
        checked_prefix=int(checked_prefix),
        source="shared_time_reversal_invariant_uniform_pair_event_envelope_spec",
    )
    return construct_nonzero_angular_global_atlas_from_two_sided_uniform_pair_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=compact_time_rate,
        future_envelope_spec=envelope_spec,
        past_envelope_spec=envelope_spec,
        finite_middle_atlas=finite_middle_atlas,
        derive_finite_middle_atlas=derive_finite_middle_atlas,
        finite_middle_order=finite_middle_order,
        finite_middle_max_compact_step=finite_middle_max_compact_step,
        finite_middle_max_s_step=finite_middle_max_s_step,
        finite_middle_target_bisections=finite_middle_target_bisections,
        derive_first_event_shell_prefix=derive_first_event_shell_prefix,
        first_event_shell_order=first_event_shell_order,
        first_event_shell_max_compact_step=first_event_shell_max_compact_step,
        first_event_shell_max_s_step=first_event_shell_max_s_step,
        first_event_shell_target_bisections=first_event_shell_target_bisections,
        derive_event_shell_invariance=derive_event_shell_invariance,
    )


def construct_nonzero_angular_global_atlas_from_two_sided_uniform_pair_event_envelopes(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    compact_time_rate: float,
    future_envelope_spec: NonzeroAngularUniformPairEventEnvelopeSpec,
    past_envelope_spec: NonzeroAngularUniformPairEventEnvelopeSpec,
    finite_middle_atlas: object | None = None,
    derive_finite_middle_atlas: bool = False,
    finite_middle_order: int = 8,
    finite_middle_max_compact_step: float = 5.0e-5,
    finite_middle_max_s_step: float = 0.02,
    finite_middle_target_bisections: int = 20,
    derive_first_event_shell_prefix: bool = False,
    first_event_shell_order: int = 8,
    first_event_shell_max_compact_step: float = 0.03,
    first_event_shell_max_s_step: float = 0.02,
    first_event_shell_target_bisections: int = 20,
    derive_event_shell_invariance: bool = False,
) -> GlobalAtlasCertificate:
    """Build the nonzero-angular theorem from explicit future and past envelopes."""

    future_spec = _require_uniform_pair_event_envelope_spec(
        "future_envelope_spec",
        future_envelope_spec,
        allowed_time_directions=("future", "time_reversal_invariant"),
    )
    past_spec = _require_uniform_pair_event_envelope_spec(
        "past_envelope_spec",
        past_envelope_spec,
        allowed_time_directions=("time_reversed_past", "time_reversal_invariant"),
    )
    future_event_regime_assembly = (
        _derive_nonzero_angular_event_recurrence_from_uniform_pair_spec(
            masses=masses,
            envelope_spec=future_spec,
        )
    )
    past_event_regime_assembly = (
        _derive_nonzero_angular_event_recurrence_from_uniform_pair_spec(
            masses=masses,
            envelope_spec=past_spec,
        )
    )
    _reject_raw_bool("finite_middle_atlas", finite_middle_atlas)
    if finite_middle_atlas is None and bool(derive_finite_middle_atlas):
        finite_middle_atlas = construct_nonzero_angular_finite_middle_atlas_for_event_shell(
            masses=masses,
            positions=positions,
            velocities=velocities,
            compact_time_rate=compact_time_rate,
            future_envelope_spec=future_spec,
            past_envelope_spec=past_spec,
            order=finite_middle_order,
            max_compact_step=finite_middle_max_compact_step,
            max_s_step=finite_middle_max_s_step,
            target_bisections=finite_middle_target_bisections,
        )
    input_domain_certificate = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    compact_time_certificate = certify_compact_time_real_line_coverage(
        compact_time_rate,
    )
    event_regime_handoff = (
        None
        if finite_middle_atlas is None
        else certify_nonzero_angular_event_regime_handoff(
            input_domain_certificate=input_domain_certificate,
            compact_time_certificate=compact_time_certificate,
            finite_middle_atlas=finite_middle_atlas,
            future_envelope_spec=future_spec,
            past_envelope_spec=past_spec,
        )
    )
    two_sided_event_budget = certify_two_sided_nonzero_angular_event_budget(
        future_event_regime_assembly=future_event_regime_assembly,
        past_event_regime_assembly=past_event_regime_assembly,
    )
    event_tail_margin_certificate = (
        None
        if event_regime_handoff is None
        else certify_nonzero_angular_event_tail_margin_from_handoff(
            event_regime_handoff=event_regime_handoff,
            two_sided_event_budget=two_sided_event_budget,
        )
    )
    first_event_shell_prefix = (
        None
        if event_regime_handoff is None or not bool(derive_first_event_shell_prefix)
        else construct_nonzero_angular_first_event_shell_prefix(
            masses=masses,
            positions=positions,
            velocities=velocities,
            compact_time_rate=compact_time_rate,
            event_regime_handoff=event_regime_handoff,
            order=first_event_shell_order,
            max_compact_step=first_event_shell_max_compact_step,
            max_s_step=first_event_shell_max_s_step,
            target_bisections=first_event_shell_target_bisections,
        )
    )
    event_shell_invariance_certificate = (
        None
        if (
            event_regime_handoff is None
            or event_tail_margin_certificate is None
            or first_event_shell_prefix is None
            or not bool(derive_event_shell_invariance)
        )
        else certify_nonzero_angular_event_shell_invariance_from_handoff(
            event_regime_handoff=event_regime_handoff,
            event_tail_margin_certificate=event_tail_margin_certificate,
            first_event_shell_prefix=first_event_shell_prefix,
            two_sided_event_budget=two_sided_event_budget,
        )
    )
    return construct_nonzero_angular_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=compact_time_rate,
        event_regime_assembly=future_event_regime_assembly,
        past_event_regime_assembly=past_event_regime_assembly,
        finite_middle_atlas=finite_middle_atlas,
        event_regime_handoff=event_regime_handoff,
        event_tail_margin_certificate=event_tail_margin_certificate,
        first_event_shell_prefix=first_event_shell_prefix,
        event_shell_invariance_certificate=event_shell_invariance_certificate,
    )


def _require_uniform_pair_event_envelope_spec(
    name: str,
    envelope_spec: object,
    *,
    allowed_time_directions: tuple[str, ...],
) -> NonzeroAngularUniformPairEventEnvelopeSpec:
    _reject_raw_bool(name, envelope_spec)
    if not isinstance(envelope_spec, NonzeroAngularUniformPairEventEnvelopeSpec):
        raise TypeError(
            f"{name} must be a NonzeroAngularUniformPairEventEnvelopeSpec"
        )
    if envelope_spec.time_direction not in allowed_time_directions:
        raise ValueError(
            f"{name} time_direction must be one of {allowed_time_directions}, "
            f"got {envelope_spec.time_direction!r}"
        )
    return envelope_spec


def _derive_nonzero_angular_event_recurrence_from_uniform_pair_spec(
    *,
    masses: Any,
    envelope_spec: NonzeroAngularUniformPairEventEnvelopeSpec,
) -> object:
    return derive_nonzero_angular_event_recurrence_from_uniform_pair_envelopes(
        masses=masses,
        delta_initial=envelope_spec.delta_initial,
        theta=envelope_spec.theta,
        event_isolation_initial=envelope_spec.event_isolation_initial,
        boundary_clearance_initial=envelope_spec.boundary_clearance_initial,
        ordinary_pair_distance_lower_bound=(
            envelope_spec.ordinary_pair_distance_lower_bound
        ),
        ordinary_pair_diameter_upper_bound=(
            envelope_spec.ordinary_pair_diameter_upper_bound
        ),
        ordinary_speed_upper_bound=envelope_spec.ordinary_speed_upper_bound,
        binary_pair_envelopes=envelope_spec.binary_pair_envelopes,
        step_ratio_bounds=envelope_spec.step_ratio_bounds,
        retained_order_initials=envelope_spec.retained_order_initials,
        retained_order_increments=envelope_spec.retained_order_increments,
        checked_prefix=envelope_spec.checked_prefix,
        time_direction=envelope_spec.time_direction,
        provenance=envelope_spec.source,
    )


def construct_nonzero_angular_compact_finite_atlas(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    compact_time_rate: float,
    validated_atlas: object,
) -> GlobalAtlasCertificate:
    """Build a compact finite-atlas theorem for nonzero-angular data."""

    input_domain_certificate = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    compact_time_certificate = certify_compact_time_real_line_coverage(compact_time_rate)
    triple_collision_exclusion = certify_nonzero_angular_momentum_excludes_triple_collision(
        positions,
        velocities,
        masses,
    )
    finite_atlas_certificate = certify_compact_nonzero_angular_finite_atlas(
        input_domain_certificate=input_domain_certificate,
        validated_atlas=validated_atlas,
    )
    classification = classify_global_regime(
        input_domain_certificate=input_domain_certificate,
        compact_time_certificate=compact_time_certificate,
        regime_id="compact_nonzero_angular_finite_events",
        ordinary_gap_envelope=finite_atlas_certificate,
        separated_binary_envelope=finite_atlas_certificate,
        triple_collision_exclusion_certificate=triple_collision_exclusion,
    )
    return construct_global_atlas_for_regime(
        classification,
        validated_atlas=validated_atlas,
    )


def construct_zero_angular_compact_finite_atlas_with_selector(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    compact_time_rate: float,
    validated_atlas: object,
    total_collision_selector_envelopes: object,
) -> GlobalAtlasCertificate:
    """Build a compact finite-atlas theorem with explicit zero-angular selectors."""

    input_domain_certificate = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    compact_time_certificate = certify_compact_time_real_line_coverage(compact_time_rate)
    finite_atlas_certificate = certify_compact_zero_angular_finite_selector_atlas(
        input_domain_certificate=input_domain_certificate,
        validated_atlas=validated_atlas,
        total_collision_selector_envelopes=total_collision_selector_envelopes,
    )
    classification = classify_global_regime(
        input_domain_certificate=input_domain_certificate,
        compact_time_certificate=compact_time_certificate,
        regime_id="compact_zero_angular_finite_events_with_selector",
        ordinary_gap_envelope=finite_atlas_certificate,
        total_collision_selector_envelope=finite_atlas_certificate,
    )
    return construct_global_atlas_for_regime(
        classification,
        validated_atlas=validated_atlas,
    )


def construct_zero_angular_parabolic_homothetic_total_collision_atlas(
    *,
    branch: object,
    start_tau: float,
    target_tau: float,
    compact_time_rate: float,
    event_time: float = 0.0,
    atlas_tolerance: float = 1.0e-9,
    selector_tolerance: float = 1.0e-8,
) -> GlobalAtlasCertificate:
    """Build the exact parabolic homothetic selector theorem path.

    This is a constructor-only local zero-angular continuation subcase.  The
    branch must be the zero-energy homothetic total-collision branch; the
    validated atlas adapter verifies the exact regularized projection and the
    selector-entry certificate is derived from the branch coefficients.
    """

    from .dynamics import split_state
    from .validated_atlas import (
        validated_atlas_from_parabolic_homothetic_total_collision_branch,
    )
    from .zero_angular_entry import (
        certify_homothetic_finite_jet_identity_selector_entry,
    )

    validated_atlas = validated_atlas_from_parabolic_homothetic_total_collision_branch(
        branch,  # type: ignore[arg-type]
        start_tau=start_tau,
        target_tau=target_tau,
        event_time=event_time,
        tolerance=atlas_tolerance,
    )
    positions, velocities = split_state(validated_atlas.evaluation.initial_state)
    selector_entry = certify_homothetic_finite_jet_identity_selector_entry(
        branch,  # type: ignore[arg-type]
        sample_taus=(float(start_tau), float(target_tau)),
        tolerance=selector_tolerance,
    )
    return construct_zero_angular_compact_finite_atlas_with_selector(
        masses=getattr(branch, "masses"),
        positions=positions,
        velocities=velocities,
        compact_time_rate=compact_time_rate,
        validated_atlas=validated_atlas,
        total_collision_selector_envelopes=(selector_entry,),
    )


def construct_uniformly_noncollision_bounded_tail_global_atlas(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    compact_time_rate: float,
    pair_distance_lower_bound: float,
    centered_position_upper_bound: float,
    centered_speed_upper_bound: float,
    retained_order: int,
    step_size: float | None = None,
    safety: float = 0.5,
) -> GlobalAtlasCertificate:
    """Build the all-future uniformly collision-free ordinary recurrence theorem.

    This constructor promotes the compact-cover lemma's all-future recurrence
    into the shared theorem pipeline.  The supplied bounds are explicit regime
    hypotheses on the centered branch; this is not arbitrary-data regime
    classification.
    """

    masses_array = np.asarray(masses, dtype=float)
    positions_array = np.asarray(positions, dtype=float)
    velocities_array = np.asarray(velocities, dtype=float)
    input_domain_certificate = certify_positive_mass_noncollision_input_domain(
        masses_array,
        positions_array,
        velocities_array,
    )
    compact_time_certificate = certify_compact_time_real_line_coverage(compact_time_rate)
    total_mass = float(np.sum(masses_array))
    center = np.sum(masses_array[:, None] * positions_array, axis=0) / total_mass
    center_velocity = np.sum(masses_array[:, None] * velocities_array, axis=0) / total_mass
    centered_positions = positions_array - center
    centered_velocities = velocities_array - center_velocity
    initial_pair_distance = input_domain_certificate.min_pair_distance
    initial_centered_position_bound = float(
        max(np.linalg.norm(position) for position in centered_positions)
    )
    initial_centered_speed_bound = float(
        max(np.linalg.norm(velocity) for velocity in centered_velocities)
    )
    recurrence = certify_uniformly_collision_free_taylor_recurrence(
        masses_array,
        pair_distance_lower_bound=pair_distance_lower_bound,
        position_upper_bound=centered_position_upper_bound,
        speed_upper_bound=centered_speed_upper_bound,
        retained_order=retained_order,
        step_size=step_size,
        safety=safety,
    )
    ordinary_chart_ledgers = certify_uniform_collision_free_ordinary_chart_ledgers(
        centered_positions,
        centered_velocities,
        masses_array,
        recurrence,
    )
    uniform_tail_certificate = UniformlyNoncollisionBoundedTailCertificate(
        input_domain_certificate=input_domain_certificate,
        recurrence=recurrence,
        ordinary_chart_ledgers=ordinary_chart_ledgers,
        obligations=(
            _required_constructor_obligation(
                "positive_mass_noncollision_input_domain",
                input_domain_certificate,
                ("certified",),
            ),
            TheoremPipelineObligation(
                obligation="initial_state_inside_uniform_centered_bounds",
                certified=bool(
                    initial_pair_distance >= float(pair_distance_lower_bound)
                    and initial_centered_position_bound
                    <= float(centered_position_upper_bound)
                    and initial_centered_speed_bound
                    <= float(centered_speed_upper_bound)
                ),
                source="construct_uniformly_noncollision_bounded_tail_global_atlas",
                detail=(
                    f"initial_pair_distance={initial_pair_distance}; "
                    f"initial_centered_position_bound={initial_centered_position_bound}; "
                    f"initial_centered_speed_bound={initial_centered_speed_bound}"
                ),
            ),
            _required_constructor_obligation(
                "uniform_collision_free_taylor_recurrence",
                recurrence,
                ("recurrence_closes", "certified", "proof_certified"),
            ),
            _required_constructor_obligation(
                "ordinary_chart_newton_projection_invariants_tail",
                ordinary_chart_ledgers,
                ("certified", "proof_certified"),
            ),
        ),
    )
    classification = classify_global_regime(
        input_domain_certificate=input_domain_certificate,
        compact_time_certificate=compact_time_certificate,
        regime_id="uniformly_noncollision_bounded_tail",
        ordinary_gap_envelope=uniform_tail_certificate,
    )
    return construct_global_atlas_for_regime(
        classification,
        ordinary_gap_atlas=uniform_tail_certificate,
    )


def construct_prescribed_two_ended_scattering_global_atlas(
    *,
    masses: Any,
    middle_positions: Any,
    middle_velocities: Any,
    compact_time_rate: float,
    scattering_atlas: object,
    invariant_tolerance: float = 1.0e-10,
) -> GlobalAtlasCertificate:
    """Build the prescribed two-ended scattering regime theorem."""

    input_domain_certificate = certify_positive_mass_noncollision_input_domain(
        masses,
        middle_positions,
        middle_velocities,
    )
    compact_time_certificate = certify_compact_time_real_line_coverage(compact_time_rate)
    _reject_raw_bool("scattering_atlas", scattering_atlas)
    invariant_match = certify_two_ended_scattering_invariant_match(
        scattering_atlas,
        tolerance=invariant_tolerance,
    )
    middle_invariant_match = certify_prescribed_scattering_middle_invariant_match(
        input_domain_certificate,
        invariant_match,
        tolerance=invariant_tolerance,
    )
    scattering_certificate = PrescribedTwoEndedScatteringCertificate(
        input_domain_certificate=input_domain_certificate,
        scattering_atlas=scattering_atlas,
        invariant_match_certificate=invariant_match,
        middle_invariant_match_certificate=middle_invariant_match,
        obligations=(
            _required_constructor_obligation(
                "positive_mass_noncollision_input_domain",
                input_domain_certificate,
                ("certified",),
            ),
            _required_constructor_obligation(
                "two_ended_scattering_recurrence",
                scattering_atlas,
                ("certified",),
            ),
            _required_constructor_obligation(
                "two_ended_scattering_invariant_match",
                invariant_match,
                ("certified",),
            ),
            _required_constructor_obligation(
                "two_ended_scattering_middle_invariant_match",
                middle_invariant_match,
                ("certified",),
            ),
        ),
    )
    classification = classify_global_regime(
        input_domain_certificate=input_domain_certificate,
        compact_time_certificate=compact_time_certificate,
        regime_id="prescribed_two_ended_scattering",
        scattering_endpoint_envelope=scattering_certificate,
    )
    return construct_global_atlas_for_regime(
        classification,
        scattering_atlas=scattering_certificate,
    )


def construct_positive_energy_homothetic_escape_global_atlas(
    *,
    masses: Any,
    positions: Any,
    velocities: Any,
    compact_time_rate: float,
    start_time: float,
    tau_radius: float,
    rho_radius: float,
    retained_degree: int,
    cauchy_majorant: float | None = None,
    derivative_order: int = 0,
    x_radius: float | None = None,
    homothetic_tolerance: float = 1.0e-10,
    total_collision_tau: float = 0.2,
    total_collision_order: int = 32,
    total_collision_z_radius: float | None = None,
    total_collision_u_radius: float = 0.2,
    middle_retained_order: int = 12,
    gluing_tolerance: float = 1.0e-5,
) -> GlobalAtlasCertificate:
    """Build the scoped positive-energy homothetic escape regime theorem."""

    input_domain_certificate = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    compact_time_certificate = certify_compact_time_real_line_coverage(compact_time_rate)
    branch_certificate = certify_homothetic_escape_branch_from_initial_data(
        masses,
        positions,
        velocities,
        tolerance=homothetic_tolerance,
    )
    projection_invariant_certificate = certify_homothetic_escape_projection_invariants(
        masses,
        positions,
        velocities,
        branch_certificate,
        tolerance=homothetic_tolerance,
    )
    recurrence = None
    past_recurrence = None
    total_collision_atlas = None
    selector_entry = None
    all_real_gluing = None
    majorant_certificate = None
    endpoint = getattr(branch_certificate, "endpoint", None)
    if endpoint is not None:
        try:
            if cauchy_majorant is None:
                if derivative_order != 0:
                    raise ValueError(
                        "automatic homothetic escape majorants currently cover "
                        "derivative_order=0"
                    )
                if x_radius is None:
                    x_radius = 0.25 * float(getattr(endpoint, "asymptotic_speed"))
                majorant_certificate = derive_homothetic_escape_implicit_cauchy_majorant(
                    endpoint,
                    tau_radius=tau_radius,
                    rho_radius=rho_radius,
                    x_radius=x_radius,
                )
                cauchy_majorant_value = majorant_certificate.cauchy_majorant
            else:
                cauchy_majorant_value = float(cauchy_majorant)
            recurrence = construct_homothetic_escape_dyadic_recurrence(
                endpoint,
                start_time=start_time,
                tau_radius=tau_radius,
                rho_radius=rho_radius,
                cauchy_majorant=cauchy_majorant_value,
                retained_degree=retained_degree,
                derivative_order=derivative_order,
                majorant_certificate=majorant_certificate,
            )
            past_recurrence = derive_time_reversed_homothetic_escape_recurrence(
                recurrence,
                compact_time_rate=compact_time_rate,
            )
        except ValueError:
            recurrence = None
            past_recurrence = None
        if recurrence is not None and past_recurrence is not None:
            try:
                masses_array = np.asarray(masses, dtype=float)
                positions_array = np.asarray(positions, dtype=float)
                total_mass = float(np.sum(masses_array))
                centered_positions = positions_array - (
                    np.sum(masses_array[:, None] * positions_array, axis=0)
                    / total_mass
                )
                central_scale = (
                    (9.0 / 2.0)
                    * float(branch_certificate.gravitational_parameter)
                ) ** (1.0 / 3.0)
                total_collision_branch = construct_homothetic_total_collision_branch(
                    centered_positions,
                    float(branch_certificate.gravitational_parameter),
                    masses_array,
                    energy_per_inertia=float(branch_certificate.energy)
                    / central_scale**2,
                    order=total_collision_order,
                )
                if total_collision_z_radius is None:
                    total_collision_z_radius_value = max(
                        1.5 * float(total_collision_tau) ** 2,
                        1.0e-6,
                    )
                else:
                    total_collision_z_radius_value = float(total_collision_z_radius)
                total_collision_majorant = (
                    certify_homothetic_total_collision_scalar_majorant(
                        total_collision_branch,
                        z_radius=total_collision_z_radius_value,
                        u_radius=total_collision_u_radius,
                    )
                )
                total_collision_atlas = (
                    validated_atlas_from_homothetic_total_collision_branch(
                        total_collision_branch,
                        start_tau=-float(total_collision_tau),
                        target_tau=float(total_collision_tau),
                        event_time=float(endpoint.time_shift),
                        scalar_majorant=total_collision_majorant,
                        tolerance=gluing_tolerance,
                    )
                )
                selector_entry = certify_homothetic_finite_jet_identity_selector_entry(
                    total_collision_branch,
                    sample_taus=(-float(total_collision_tau), float(total_collision_tau)),
                    tolerance=gluing_tolerance,
                )
                all_real_gluing = certify_positive_energy_homothetic_all_real_gluing(
                    masses=masses_array,
                    positions=positions_array,
                    branch_certificate=branch_certificate,
                    projection_invariant_certificate=projection_invariant_certificate,
                    future_recurrence=recurrence,
                    past_recurrence=past_recurrence,
                    total_collision_atlas=total_collision_atlas,
                    selector_entry=selector_entry,
                    compact_time_rate=compact_time_rate,
                    middle_retained_order=middle_retained_order,
                    tolerance=gluing_tolerance,
                )
            except (TypeError, ValueError, FloatingPointError):
                total_collision_atlas = None
                selector_entry = None
                all_real_gluing = None
    escape_certificate = PositiveEnergyHomotheticEscapeCertificate(
        input_domain_certificate=input_domain_certificate,
        branch_certificate=branch_certificate,
        projection_invariant_certificate=projection_invariant_certificate,
        recurrence=recurrence,
        past_recurrence=past_recurrence,
        total_collision_atlas=total_collision_atlas,
        selector_entry=selector_entry,
        all_real_gluing=all_real_gluing,
        obligations=(
            _required_constructor_obligation(
                "positive_mass_noncollision_input_domain",
                input_domain_certificate,
                ("certified",),
            ),
            _required_constructor_obligation(
                "homothetic_escape_branch",
                branch_certificate,
                ("certified",),
            ),
            _required_constructor_obligation(
                "positive_energy_homothetic_projection_newton_invariants",
                projection_invariant_certificate,
                ("certified",),
            ),
            _required_constructor_obligation(
                "positive_energy_homothetic_endpoint_recurrence",
                recurrence,
                ("certified",),
            ),
            _required_constructor_obligation(
                "positive_energy_homothetic_past_endpoint_time_reversal",
                past_recurrence,
                ("certified", "recurrence_closes"),
            ),
            _required_constructor_obligation(
                "positive_energy_homothetic_implicit_cauchy_majorant",
                majorant_certificate,
                ("certified",),
            ),
            _required_constructor_obligation(
                "positive_energy_homothetic_all_real_gluing",
                all_real_gluing,
                ("certified", "proof_certified", "recurrence_closes"),
            ),
        ),
    )
    classification = classify_global_regime(
        input_domain_certificate=input_domain_certificate,
        compact_time_certificate=compact_time_certificate,
        regime_id="positive_energy_homothetic_escape",
        escape_endpoint_envelope=escape_certificate,
    )
    return construct_global_atlas_for_regime(
        classification,
        escape_atlas=escape_certificate,
    )


def construct_general_solution_theorem_certificate(
    global_atlas: GlobalAtlasCertificate,
    *,
    global_regime_exhaustion_certificate: object | None = None,
) -> GeneralSolutionTheoremCertificate:
    """Assemble the top theorem while keeping global exhaustion explicit."""

    _reject_raw_bool("global_atlas", global_atlas)
    obligations = (
        _global_regime_exhaustion_obligation(
            global_regime_exhaustion_certificate,
        ),
    )
    return GeneralSolutionTheoremCertificate(
        global_atlas=global_atlas,
        global_regime_exhaustion_certificate=global_regime_exhaustion_certificate,
        obligations=obligations,
    )


def certify_global_regime_exhaustion(
    *,
    candidate_regimes: tuple[GlobalAtlasCertificate, ...],
    input_scope: str = "arbitrary_positive_mass_noncollision",
    required_regime_ids: tuple[str, ...] = GLOBAL_EXHAUSTION_REQUIRED_REGIME_IDS,
) -> GlobalRegimeExhaustionCertificate:
    """Build the typed full-theorem exhaustion witness.

    This intentionally does not certify today.  It checks the mechanical
    evidence that scoped regimes are present and certified, then records the
    still-missing analytic partition theorem as a required obligation.  That
    prevents a list of successful scoped examples from being mistaken for
    arbitrary-data exhaustion.
    """

    _reject_raw_bool("candidate_regimes", candidate_regimes)
    regimes = tuple(candidate_regimes)
    required = tuple(str(regime_id) for regime_id in required_regime_ids)
    covered = tuple(
        str(getattr(atlas.classification, "regime_id", "missing"))
        for atlas in regimes
    )
    duplicate_regimes = len(set(covered)) != len(covered)
    missing_required = tuple(regime_id for regime_id in required if regime_id not in covered)
    uncertified = tuple(
        regime_id
        for regime_id, atlas in zip(covered, regimes, strict=True)
        if not bool(getattr(atlas, "certified", False))
    )
    input_domains_consistent = _candidate_regime_input_domains_consistent(regimes)
    obligations = (
        TheoremPipelineObligation(
            obligation="global_exhaustion_input_scope",
            certified=str(input_scope) == "arbitrary_positive_mass_noncollision",
            source="GlobalRegimeExhaustionCertificate",
            detail=f"input_scope={input_scope}",
        ),
        TheoremPipelineObligation(
            obligation="global_exhaustion_unique_regime_ids",
            certified=bool(regimes and not duplicate_regimes),
            source="GlobalRegimeExhaustionCertificate",
            detail="covered=" + ",".join(covered),
        ),
        TheoremPipelineObligation(
            obligation="global_exhaustion_candidate_input_domains",
            certified=input_domains_consistent,
            source="GlobalRegimeExhaustionCertificate",
            detail=_candidate_regime_input_domain_detail(regimes),
        ),
        TheoremPipelineObligation(
            obligation="global_exhaustion_required_regime_ids",
            certified=not missing_required,
            source="GlobalRegimeExhaustionCertificate",
            detail="missing=" + ",".join(missing_required),
        ),
        TheoremPipelineObligation(
            obligation="global_exhaustion_candidate_regime_theorems",
            certified=not uncertified,
            source="GlobalRegimeExhaustionCertificate",
            detail="uncertified=" + ",".join(uncertified),
        ),
        TheoremPipelineObligation(
            obligation="arbitrary_initial_data_partition_theorem",
            certified=False,
            source="missing_global_regime_classifier",
            detail=(
                "still needs a proof that arbitrary positive-mass noncollision "
                "initial data enter exactly one certified compact, event-tail, "
                "escape/scattering, or selector-continuation regime"
            ),
        ),
    )
    return GlobalRegimeExhaustionCertificate(
        candidate_regimes=regimes,
        input_scope=str(input_scope),
        required_regime_ids=required,
        obligations=obligations,
    )


def _classification_requirements_for_regime(
    regime_id: str,
    *,
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    compact_time_certificate: CompactTimeCoverageCertificate,
    finite_middle_atlas: object | None,
    event_regime_handoff: object | None,
    event_tail_margin_certificate: object | None,
    first_event_shell_prefix: object | None,
    event_shell_invariance_certificate: object | None,
    ordinary_gap_envelope: object | None,
    separated_binary_envelope: object | None,
    triple_collision_exclusion_certificate: object | None,
    total_collision_selector_envelope: object | None,
    escape_endpoint_envelope: object | None,
    scattering_endpoint_envelope: object | None,
    event_isolation_certificate: object | None,
    primitive_cauchy_inputs: object | None,
) -> list[TheoremPipelineObligation]:
    if regime_id == "all_time_nonzero_angular":
        requirements = [
            _required_constructor_obligation(
                "nonzero_angular_triple_collision_exclusion",
                triple_collision_exclusion_certificate,
                ("certified",),
            ),
            _required_constructor_obligation(
                "finite_middle_validated_atlas",
                finite_middle_atlas,
                ("certified", "proof_certified"),
            ),
            _required_constructor_obligation(
                "finite_middle_to_event_envelope_handoff",
                event_regime_handoff,
                ("certified", "handoff_certified", "proof_certified"),
            ),
            _nonzero_angular_event_tail_margin_obligation(
                event_regime_handoff,
                event_tail_margin_certificate,
                primitive_cauchy_inputs,
            ),
            _nonzero_angular_first_event_shell_prefix_obligation(
                event_regime_handoff,
                first_event_shell_prefix,
            ),
            _nonzero_angular_event_shell_invariance_obligation(
                event_regime_handoff,
                event_shell_invariance_certificate,
                primitive_cauchy_inputs,
            ),
            _nonzero_angular_event_family_scope_obligation(
                event_isolation_certificate,
            ),
            _nonzero_angular_all_pair_binary_coverage_obligation(
                event_isolation_certificate,
            ),
            _nonzero_angular_event_family_mass_consistency_obligation(
                input_domain_certificate,
                event_isolation_certificate,
            ),
            _required_constructor_obligation(
                "event_isolation",
                event_isolation_certificate,
                ("certified",),
            ),
            _no_total_collision_event_family_obligation(
                event_isolation_certificate,
            ),
            _event_direction_obligation(
                "future_all_future_event_budget",
                primitive_cauchy_inputs,
                "future_event_budget",
            ),
            _event_direction_obligation(
                "past_all_future_event_budget",
                primitive_cauchy_inputs,
                "past_event_budget",
            ),
            _two_sided_event_time_direction_provenance_obligation(
                primitive_cauchy_inputs,
            ),
            _required_constructor_obligation(
                "two_sided_primitive_cauchy_all_time_budget",
                primitive_cauchy_inputs,
                ("recurrence_closes", "certified"),
            ),
        ]
        if finite_middle_atlas is not None:
            requirements.append(
                _embedded_input_domain_match_obligation(
                    "finite_middle_input_domain_matches_classification",
                    input_domain_certificate,
                    finite_middle_atlas,
                )
            )
        if event_regime_handoff is not None:
            requirements.extend(
                (
                    _embedded_input_domain_match_obligation(
                        "nonzero_angular_handoff_input_domain_matches_classification",
                        input_domain_certificate,
                        event_regime_handoff,
                    ),
                    _embedded_compact_time_match_obligation(
                        "nonzero_angular_handoff_compact_time_matches_classification",
                        event_regime_handoff,
                        compact_time_certificate,
                    ),
                    TheoremPipelineObligation(
                        obligation="nonzero_angular_handoff_finite_middle_matches_classification",
                        certified=bool(
                            finite_middle_atlas is not None
                            and getattr(event_regime_handoff, "finite_middle_atlas", None)
                            is finite_middle_atlas
                        ),
                        source=type(event_regime_handoff).__name__,
                        detail=(
                            "handoff certificate must consume the same finite middle "
                            "atlas object stored in the classification"
                        ),
                    ),
                )
            )
        return requirements
    if regime_id == "geometric_infinite_event_tail":
        return [
            _missing_global_regime_membership_obligation(
                "geometric_event_regime_membership_from_initial_data",
                (
                    "the geometric event-tail recurrence closes a conditional "
                    "all-future budget, but no constructor currently derives "
                    "the shell isolation and chart-family coverage hypotheses "
                    "from the supplied initial data"
                ),
            ),
            _required_constructor_obligation(
                "event_isolation",
                event_isolation_certificate,
                ("certified",),
            ),
            _required_constructor_obligation(
                "primitive_cauchy_all_future_budget",
                primitive_cauchy_inputs,
                ("recurrence_closes", "certified"),
            ),
        ]
    if regime_id == "prescribed_two_ended_scattering":
        requirements = [
            _required_constructor_obligation(
                "scattering_endpoint_envelope",
                scattering_endpoint_envelope,
                ("certified", "recurrence_closes"),
            )
        ]
        if scattering_endpoint_envelope is not None:
            requirements.append(
                _embedded_input_domain_match_obligation(
                    "scattering_endpoint_input_domain_matches_classification",
                    input_domain_certificate,
                    scattering_endpoint_envelope,
                )
            )
        return requirements
    if regime_id == "positive_energy_homothetic_escape":
        requirements = [
            _required_constructor_obligation(
                "escape_endpoint_envelope",
                escape_endpoint_envelope,
                ("certified", "recurrence_closes"),
            ),
            _required_constructor_obligation(
                "positive_energy_homothetic_all_real_gluing",
                getattr(escape_endpoint_envelope, "all_real_gluing", None),
                ("certified", "proof_certified", "recurrence_closes"),
            ),
        ]
        if escape_endpoint_envelope is not None:
            requirements.append(
                _embedded_input_domain_match_obligation(
                    "escape_endpoint_input_domain_matches_classification",
                    input_domain_certificate,
                    escape_endpoint_envelope,
                )
            )
        return requirements
    if regime_id == "compact_nonzero_angular_finite_events":
        requirements = [
            _required_constructor_obligation(
                "nonzero_angular_triple_collision_exclusion",
                triple_collision_exclusion_certificate,
                ("certified",),
            ),
            _required_constructor_obligation(
                "ordinary_gap_envelope",
                ordinary_gap_envelope,
                ("certified", "proof_certified"),
            ),
            _required_constructor_obligation(
                "separated_binary_envelope",
                separated_binary_envelope,
                ("certified", "proof_certified"),
            ),
        ]
        if ordinary_gap_envelope is not None:
            requirements.append(
                _embedded_input_domain_match_obligation(
                    "ordinary_gap_input_domain_matches_classification",
                    input_domain_certificate,
                    ordinary_gap_envelope,
                )
            )
        if separated_binary_envelope is not None:
            requirements.append(
                _embedded_input_domain_match_obligation(
                    "separated_binary_input_domain_matches_classification",
                    input_domain_certificate,
                    separated_binary_envelope,
                )
            )
        return requirements
    if regime_id == "compact_zero_angular_finite_events_with_selector":
        requirements = [
            _required_constructor_obligation(
                "ordinary_gap_envelope",
                ordinary_gap_envelope,
                ("certified", "proof_certified"),
            ),
            _required_constructor_obligation(
                "total_collision_selector_envelope",
                total_collision_selector_envelope,
                ("certified", "identity_selector_certified"),
            ),
        ]
        if ordinary_gap_envelope is not None:
            requirements.append(
                _embedded_input_domain_match_obligation(
                    "ordinary_gap_input_domain_matches_classification",
                    input_domain_certificate,
                    ordinary_gap_envelope,
                )
            )
        if total_collision_selector_envelope is not None:
            requirements.append(
                _embedded_input_domain_match_obligation(
                    "total_collision_selector_input_domain_matches_classification",
                    input_domain_certificate,
                    total_collision_selector_envelope,
                )
            )
        return requirements
    if regime_id == "uniformly_noncollision_bounded_tail":
        requirements = [
            _required_constructor_obligation(
                "ordinary_gap_envelope",
                ordinary_gap_envelope,
                ("certified", "proof_certified"),
            )
        ]
        if ordinary_gap_envelope is not None:
            requirements.append(
                _embedded_input_domain_match_obligation(
                    "ordinary_gap_input_domain_matches_classification",
                    input_domain_certificate,
                    ordinary_gap_envelope,
                )
            )
        return requirements
    if regime_id == "maximal_classical_until_total_collision":
        requirements = [
            _required_constructor_obligation(
                "ordinary_gap_envelope",
                ordinary_gap_envelope,
                ("certified", "proof_certified"),
            )
        ]
        if ordinary_gap_envelope is not None:
            requirements.append(
                _embedded_input_domain_match_obligation(
                    "ordinary_gap_input_domain_matches_classification",
                    input_domain_certificate,
                    ordinary_gap_envelope,
                )
            )
        return requirements
    raise ValueError(f"unknown regime_id {regime_id!r}")


def _missing_global_regime_membership_obligation(
    obligation: str,
    detail: str,
) -> TheoremPipelineObligation:
    return TheoremPipelineObligation(
        obligation=obligation,
        certified=False,
        source="missing_global_regime_classifier",
        detail=detail,
    )


def _classification_object_match_obligation(
    obligation: str,
    selected: object | None,
    expected: object | None,
    field_name: str,
) -> TheoremPipelineObligation:
    _reject_raw_bool(obligation, selected)
    return TheoremPipelineObligation(
        obligation=obligation,
        certified=bool(
            selected is not None
            and expected is not None
            and selected is expected
        ),
        source=type(selected).__name__ if selected is not None else "missing",
        detail=(
            f"assembled object must be the same constructor object stored in "
            f"classification.{field_name}"
        ),
    )


def _embedded_input_domain_match_obligation(
    obligation: str,
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    envelope: object | None,
) -> TheoremPipelineObligation:
    _reject_raw_bool(obligation, envelope)
    embedded = getattr(envelope, "input_domain_certificate", None)
    return TheoremPipelineObligation(
        obligation=obligation,
        certified=_input_domain_certificates_consistent(
            input_domain_certificate,
            embedded,
        ),
        source=type(envelope).__name__ if envelope is not None else "missing",
        detail=(
            "classification input domain must match the envelope's embedded "
            "input_domain_certificate"
        ),
    )


def _embedded_compact_time_match_obligation(
    obligation: str,
    envelope: object | None,
    compact_time_certificate: CompactTimeCoverageCertificate,
) -> TheoremPipelineObligation:
    _reject_raw_bool(obligation, envelope)
    embedded = getattr(envelope, "compact_time_certificate", None)
    return TheoremPipelineObligation(
        obligation=obligation,
        certified=_compact_time_certificates_consistent(
            compact_time_certificate,
            embedded,
        ),
        source=type(envelope).__name__ if envelope is not None else "missing",
        detail=(
            "classification compact-time certificate must match the envelope's "
            "embedded compact_time_certificate"
        ),
    )


def _classification_validated_atlas_match_obligation(
    classification: RegimeClassificationCertificate,
    validated_atlas: object | None,
) -> TheoremPipelineObligation:
    _reject_raw_bool("global_atlas_validated_atlas_matches_classification", validated_atlas)
    expected_atlases = _classification_validated_atlas_references(classification)
    certified = bool(
        validated_atlas is not None
        and expected_atlases
        and all(validated_atlas is expected for expected in expected_atlases)
    )
    return TheoremPipelineObligation(
        obligation="global_atlas_validated_atlas_matches_classification",
        certified=certified,
        source=type(validated_atlas).__name__ if validated_atlas is not None else "missing",
        detail=(
            f"regime_id={classification.regime_id}; "
            f"classification_validated_atlas_reference_count={len(expected_atlases)}"
        ),
    )


def _classification_validated_atlas_references(
    classification: RegimeClassificationCertificate,
) -> tuple[object, ...]:
    references = []
    for field_name in (
        "ordinary_gap_envelope",
        "separated_binary_envelope",
        "total_collision_selector_envelope",
    ):
        envelope = getattr(classification, field_name, None)
        validated_atlas = getattr(envelope, "validated_atlas", None)
        if validated_atlas is not None:
            references.append(validated_atlas)
    return tuple(references)


def _global_regime_exhaustion_obligation(
    certificate: object | None,
) -> TheoremPipelineObligation:
    _reject_raw_bool("global_regime_exhaustion", certificate)
    if certificate is None:
        return TheoremPipelineObligation(
            obligation="global_regime_exhaustion",
            certified=False,
            source="missing",
            detail="no constructor-derived GlobalRegimeExhaustionCertificate supplied",
        )
    if not isinstance(certificate, GlobalRegimeExhaustionCertificate):
        return TheoremPipelineObligation(
            obligation="global_regime_exhaustion",
            certified=False,
            source=type(certificate).__name__,
            detail=(
                "full theorem accepts only a typed "
                "GlobalRegimeExhaustionCertificate from "
                "certify_global_regime_exhaustion(...)"
            ),
        )
    return TheoremPipelineObligation(
        obligation="global_regime_exhaustion",
        certified=certificate.certified,
        source=type(certificate).__name__,
        detail="missing=" + ",".join(certificate.missing_obligations),
    )


def _candidate_regime_input_domains(
    candidate_regimes: tuple[GlobalAtlasCertificate, ...],
) -> tuple[object | None, ...]:
    domains = []
    for atlas in candidate_regimes:
        classification = getattr(atlas, "classification", None)
        domains.append(getattr(classification, "input_domain_certificate", None))
    return tuple(domains)


def _candidate_regime_input_domains_consistent(
    candidate_regimes: tuple[GlobalAtlasCertificate, ...],
) -> bool:
    domains = _candidate_regime_input_domains(candidate_regimes)
    if not domains:
        return False
    reference = domains[0]
    return bool(
        reference is not None
        and all(
            _input_domain_certificates_consistent(reference, domain)
            for domain in domains
        )
    )


def _candidate_regime_input_domain_detail(
    candidate_regimes: tuple[GlobalAtlasCertificate, ...],
) -> str:
    domains = _candidate_regime_input_domains(candidate_regimes)
    if not domains:
        return "no candidate input domains"
    fragments = []
    reference = domains[0]
    for index, domain in enumerate(domains):
        regime_id = str(
            getattr(
                getattr(candidate_regimes[index], "classification", None),
                "regime_id",
                "missing",
            )
        )
        consistent = _input_domain_certificates_consistent(reference, domain)
        fragments.append(f"{regime_id}:consistent={consistent}")
    return "; ".join(fragments)


def _nonzero_angular_event_shell_invariance_obligation(
    event_regime_handoff: object | None,
    certificate: object | None,
    expected_two_sided_event_budget: object | None,
) -> TheoremPipelineObligation:
    _reject_raw_bool("nonzero_angular_event_shell_invariance_from_handoff", certificate)
    handoff_certified = bool(
        event_regime_handoff is not None
        and (
            getattr(event_regime_handoff, "handoff_certified", False)
            or getattr(event_regime_handoff, "certified", False)
            or getattr(event_regime_handoff, "proof_certified", False)
        )
    )
    if not handoff_certified:
        return TheoremPipelineObligation(
            obligation="nonzero_angular_event_shell_invariance_from_handoff",
            certified=False,
            source="deferred",
            required=False,
            detail="deferred until finite-middle-to-envelope handoff is certified",
        )
    base = _required_constructor_obligation(
        "nonzero_angular_event_shell_invariance_from_handoff",
        certificate,
        ("certified", "shell_invariance_certified", "proof_certified"),
    )
    same_handoff = _certificate_field_is_same_object(
        certificate,
        "event_regime_handoff",
        event_regime_handoff,
    )
    same_budget_halves = _two_sided_event_budget_halves_match(
        getattr(certificate, "two_sided_event_budget", None),
        expected_two_sided_event_budget,
    )
    return TheoremPipelineObligation(
        obligation=base.obligation,
        certified=bool(base.certified and same_handoff and same_budget_halves),
        source=base.source,
        detail=(
            base.detail
            + f"; same_handoff={same_handoff}; "
            + f"same_event_budget_halves={same_budget_halves}"
        ),
    )


def _nonzero_angular_event_tail_margin_obligation(
    event_regime_handoff: object | None,
    certificate: object | None,
    expected_two_sided_event_budget: object | None,
) -> TheoremPipelineObligation:
    _reject_raw_bool("nonzero_angular_all_future_value_tail_margin", certificate)
    handoff_certified = bool(
        event_regime_handoff is not None
        and (
            getattr(event_regime_handoff, "handoff_certified", False)
            or getattr(event_regime_handoff, "certified", False)
            or getattr(event_regime_handoff, "proof_certified", False)
        )
    )
    if not handoff_certified:
        return TheoremPipelineObligation(
            obligation="nonzero_angular_all_future_value_tail_margin",
            certified=False,
            source="deferred",
            required=False,
            detail="deferred until finite-middle-to-envelope handoff is certified",
        )
    base = _required_constructor_obligation(
        "nonzero_angular_all_future_value_tail_margin",
        certificate,
        ("certified", "tail_margin_certified", "proof_certified"),
    )
    same_handoff = _certificate_field_is_same_object(
        certificate,
        "event_regime_handoff",
        event_regime_handoff,
    )
    same_budget_halves = _two_sided_event_budget_halves_match(
        getattr(certificate, "two_sided_event_budget", None),
        expected_two_sided_event_budget,
    )
    return TheoremPipelineObligation(
        obligation=base.obligation,
        certified=bool(base.certified and same_handoff and same_budget_halves),
        source=base.source,
        detail=(
            base.detail
            + f"; same_handoff={same_handoff}; "
            + f"same_event_budget_halves={same_budget_halves}"
        ),
    )


def _nonzero_angular_first_event_shell_prefix_obligation(
    event_regime_handoff: object | None,
    certificate: object | None,
) -> TheoremPipelineObligation:
    _reject_raw_bool("nonzero_angular_first_event_shell_prefix", certificate)
    handoff_certified = bool(
        event_regime_handoff is not None
        and (
            getattr(event_regime_handoff, "handoff_certified", False)
            or getattr(event_regime_handoff, "certified", False)
            or getattr(event_regime_handoff, "proof_certified", False)
        )
    )
    if not handoff_certified:
        return TheoremPipelineObligation(
            obligation="nonzero_angular_first_event_shell_prefix",
            certified=False,
            source="deferred",
            required=False,
            detail="deferred until finite-middle-to-envelope handoff is certified",
        )
    base = _required_constructor_obligation(
        "nonzero_angular_first_event_shell_prefix",
        certificate,
        ("certified", "prefix_certified", "proof_certified"),
    )
    same_handoff = _certificate_field_is_same_object(
        certificate,
        "event_regime_handoff",
        event_regime_handoff,
    )
    return TheoremPipelineObligation(
        obligation=base.obligation,
        certified=bool(base.certified and same_handoff),
        source=base.source,
        detail=base.detail + f"; same_handoff={same_handoff}",
    )


def _required_constructor_obligation(
    obligation: str,
    certificate: object | None,
    certified_fields: tuple[str, ...],
) -> TheoremPipelineObligation:
    _reject_raw_bool(obligation, certificate)
    if certificate is None:
        return TheoremPipelineObligation(
            obligation=obligation,
            certified=False,
            source="missing",
            detail="no constructor certificate supplied",
        )
    present_fields = tuple(field for field in certified_fields if hasattr(certificate, field))
    certified = any(bool(getattr(certificate, field)) for field in present_fields)
    return TheoremPipelineObligation(
        obligation=obligation,
        certified=certified,
        source=type(certificate).__name__,
        detail=(
            "checked fields "
            + ",".join(present_fields or certified_fields)
        ),
    )


def _event_direction_obligation(
    obligation: str,
    certificate: object | None,
    direction_field: str,
) -> TheoremPipelineObligation:
    _reject_raw_bool(obligation, certificate)
    if certificate is None:
        return TheoremPipelineObligation(
            obligation=obligation,
            certified=False,
            source="missing",
            detail="no two-sided event recurrence certificate supplied",
        )
    half = getattr(certificate, direction_field, None)
    certified = _event_recurrence_certified(half)
    return TheoremPipelineObligation(
        obligation=obligation,
        certified=certified,
        source=type(half).__name__ if half is not None else "missing",
        detail=(
            f"{direction_field} must close a constructor-derived all-future "
            "recurrence; the past half is interpreted after time reversal"
        ),
    )


def _two_sided_event_time_direction_provenance_obligation(
    certificate: object | None,
) -> TheoremPipelineObligation:
    _reject_raw_bool("two_sided_event_time_direction_provenance", certificate)
    if certificate is None:
        return TheoremPipelineObligation(
            obligation="two_sided_event_time_direction_provenance",
            certified=False,
            source="missing",
            detail="no two-sided event recurrence certificate supplied",
        )
    future = getattr(certificate, "future_event_budget", None)
    past = getattr(certificate, "past_event_budget", None)
    future_direction = _event_budget_time_direction(future)
    past_direction = _event_budget_time_direction(past)
    return TheoremPipelineObligation(
        obligation="two_sided_event_time_direction_provenance",
        certified=bool(getattr(certificate, "direction_provenance_certified", False)),
        source=type(certificate).__name__,
        detail=(
            f"future_time_direction={future_direction}; "
            f"past_time_direction={past_direction}; "
            "the past all-future recurrence must be produced for the time-reversed flow "
            "unless both halves use a time-reversal-invariant envelope constructor"
        ),
    )


def _no_total_collision_event_family_obligation(
    certificate: object | None,
) -> TheoremPipelineObligation:
    _reject_raw_bool("no_total_collision_event_family", certificate)
    if certificate is None:
        return TheoremPipelineObligation(
            obligation="no_total_collision_event_family",
            certified=False,
            source="missing",
            detail="no event-family certificate supplied",
        )
    family_names = set()
    halves = _event_recurrence_halves(certificate)
    for half in halves or (certificate,):
        family_names.update(_event_family_names(half))
    has_total_collision_family = any("total_collision" in name for name in family_names)
    return TheoremPipelineObligation(
        obligation="no_total_collision_event_family",
        certified=bool(
            _event_halves_recurrence_certified(certificate)
            and family_names
            and not has_total_collision_family
        ),
        source=type(certificate).__name__,
        detail="checked event families " + ",".join(sorted(family_names)),
    )


def _nonzero_angular_event_family_scope_obligation(
    certificate: object | None,
) -> TheoremPipelineObligation:
    """Check nonzero-angular recurrence uses constructor-derived ordinary/binary families."""

    _reject_raw_bool("nonzero_angular_event_family_scope", certificate)
    ordinary_family = "ordinary_gap_taylor"
    if certificate is None:
        return TheoremPipelineObligation(
            obligation="nonzero_angular_event_family_scope",
            certified=False,
            source="missing",
            detail="no event-family certificate supplied",
        )
    halves = _event_recurrence_halves(certificate)
    family_names_by_half = tuple(_event_family_names(half) for half in halves)
    family_names = set().union(*family_names_by_half) if family_names_by_half else set()
    half_scope_certified = []
    for half, half_family_names in zip(halves, family_names_by_half):
        binary_families = {
            family for family in half_family_names if _is_nonzero_angular_binary_family(family)
        }
        allowed_families = {ordinary_family, *binary_families}
        has_required_families = ordinary_family in half_family_names and bool(binary_families)
        no_extra_families = half_family_names == allowed_families
        source_certified = all(
            _event_family_source_certificate_certified(half, family)
            for family in allowed_families
        )
        half_scope_certified.append(
            bool(
                _event_recurrence_certified(half)
                and has_required_families
                and no_extra_families
                and source_certified
            )
        )
    source_certified = all(
        _event_family_source_certificate_certified(half, family)
        for half in halves
        for family in _event_family_names(half)
    )
    return TheoremPipelineObligation(
        obligation="nonzero_angular_event_family_scope",
        certified=bool(
            halves
            and all(half_scope_certified)
        ),
        source=type(certificate).__name__,
        detail=(
            "families="
            + ",".join(sorted(family_names))
            + "; each time direction requires ordinary_gap_taylor plus one or more separated_binary_levi_civita families; "
            + f"source_certified={source_certified}"
        ),
    )


def _nonzero_angular_all_pair_binary_coverage_obligation(
    certificate: object | None,
) -> TheoremPipelineObligation:
    """Require binary regularization rows for all three collision pairs."""

    _reject_raw_bool("nonzero_angular_all_pair_binary_coverage", certificate)
    required_pairs = {(0, 1), (0, 2), (1, 2)}
    if certificate is None:
        return TheoremPipelineObligation(
            obligation="nonzero_angular_all_pair_binary_coverage",
            certified=False,
            source="missing",
            detail="no event-family certificate supplied",
        )
    halves = _event_recurrence_halves(certificate)
    covered_by_half = tuple(_event_binary_pairs_covered(half) for half in halves)
    certified = bool(
        _event_halves_recurrence_certified(certificate)
        and covered_by_half
        and all(required_pairs <= covered for covered in covered_by_half)
    )
    return TheoremPipelineObligation(
        obligation="nonzero_angular_all_pair_binary_coverage",
        certified=certified,
        source=type(certificate).__name__,
        detail=(
            "required_pairs=01,02,12; covered_by_half="
            + ";".join(
                ",".join(f"{i}{j}" for i, j in sorted(covered)) or "none"
                for covered in covered_by_half
            )
        ),
    )


def _nonzero_angular_event_family_mass_consistency_obligation(
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate,
    certificate: object | None,
) -> TheoremPipelineObligation:
    """Check nonzero-angular event-family constructors use the theorem masses."""

    _reject_raw_bool("nonzero_angular_event_family_mass_consistency", certificate)
    if certificate is None:
        return TheoremPipelineObligation(
            obligation="nonzero_angular_event_family_mass_consistency",
            certified=False,
            source="missing",
            detail="no event-family certificate supplied",
        )
    halves = _event_recurrence_halves(certificate)
    family_mass_details = tuple(
        (family, _event_family_source_masses(half, family))
        for half in halves
        for family in sorted(
            family
            for family in _event_family_names(half)
            if family == "ordinary_gap_taylor"
            or _is_nonzero_angular_binary_family(family)
        )
    )
    certified = all(
        _mass_sequences_consistent(input_domain_certificate.masses, family_masses)
        for _family, family_masses in family_mass_details
    )
    return TheoremPipelineObligation(
        obligation="nonzero_angular_event_family_mass_consistency",
        certified=bool(_event_halves_recurrence_certified(certificate) and certified),
        source=type(certificate).__name__,
        detail=(
            "input_masses="
            + ",".join(str(mass) for mass in input_domain_certificate.masses)
            + "; family_masses="
            + ";".join(
                family + ":"
                + (
                    ",".join(str(mass) for mass in family_masses)
                    if family_masses is not None
                    else "missing"
                )
                for family, family_masses in family_mass_details
            )
        ),
    )


def _is_nonzero_angular_binary_family(family: str) -> bool:
    return bool(
        family == "separated_binary_levi_civita"
        or family.startswith("separated_binary_levi_civita_")
    )


def _event_binary_pairs_covered(certificate: object) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for family in _event_family_names(certificate):
        pair = _event_family_pair(certificate, family)
        if pair is not None:
            pairs.add(pair)
    return pairs


def _event_family_pair(
    certificate: object,
    family: str,
) -> tuple[int, int] | None:
    if family.startswith("separated_binary_levi_civita_"):
        suffix = family.removeprefix("separated_binary_levi_civita_")
        if len(suffix) == 2 and suffix.isdigit():
            pair = tuple(sorted((int(suffix[0]), int(suffix[1]))))
            if pair in ((0, 1), (0, 2), (1, 2)):
                return pair
    if family != "separated_binary_levi_civita":
        return None
    try:
        family_certificate = certificate.family_certificate(family)
    except Exception:
        return None
    source_certificate = getattr(family_certificate, "source_certificate", None)
    try:
        pair_values = tuple(int(index) for index in getattr(source_certificate, "pair"))
    except (AttributeError, TypeError, ValueError):
        return None
    if len(pair_values) != 2:
        return None
    pair = tuple(sorted(pair_values))
    if pair not in ((0, 1), (0, 2), (1, 2)):
        return None
    return pair


def _event_recurrence_halves(certificate: object | None) -> tuple[object, ...]:
    if certificate is None:
        return ()
    if hasattr(certificate, "future_event_budget") or hasattr(certificate, "past_event_budget"):
        return tuple(
            half
            for half in (
                getattr(certificate, "future_event_budget", None),
                getattr(certificate, "past_event_budget", None),
            )
            if half is not None
        )
    return (certificate,)


def _event_recurrence_certified(certificate: object | None) -> bool:
    return bool(
        certificate is not None
        and getattr(certificate, "certified", False)
        and getattr(certificate, "recurrence_closes", False)
    )


def _event_halves_recurrence_certified(certificate: object | None) -> bool:
    halves = _event_recurrence_halves(certificate)
    return bool(halves and all(_event_recurrence_certified(half) for half in halves))


def _event_budget_time_direction(certificate: object | None) -> str:
    if certificate is None:
        return "missing"
    return str(getattr(certificate, "time_direction", "unspecified"))


def _event_family_names(certificate: object) -> set[str]:
    family_names: set[str] = set()
    counts = getattr(certificate, "chart_family_counts", None)
    if counts is not None:
        family_names.update(str(kind) for kind in counts)
    family_kinds = getattr(certificate, "family_kinds", None)
    if family_kinds is not None:
        family_names.update(str(kind) for kind in family_kinds)
    return family_names


def _event_family_source_certificate_certified(
    certificate: object,
    family: str,
) -> bool:
    try:
        family_certificate = certificate.family_certificate(family)
    except Exception:
        return False
    source_certificate = getattr(family_certificate, "source_certificate", None)
    return bool(
        source_certificate is not None
        and getattr(family_certificate, "source_certified", False)
        and getattr(source_certificate, "certified", False)
    )


def _event_family_source_masses(
    certificate: object,
    family: str,
) -> tuple[float, ...] | None:
    try:
        family_certificate = certificate.family_certificate(family)
    except Exception:
        return None
    source_certificate = getattr(family_certificate, "source_certificate", None)
    if source_certificate is None:
        return None
    try:
        masses = np.asarray(getattr(source_certificate, "masses"), dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        return None
    return tuple(float(mass) for mass in masses)


def _mass_sequences_consistent(
    left: object,
    right: object,
) -> bool:
    if right is None:
        return False
    try:
        left_array = np.asarray(left, dtype=float).reshape(-1)
        right_array = np.asarray(right, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    if left_array.size == 0 or left_array.shape != right_array.shape:
        return False
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(left_array))),
        float(np.max(np.abs(right_array))),
    )
    return bool(
        np.all(np.isfinite(left_array))
        and np.all(np.isfinite(right_array))
        and np.all(left_array > 0.0)
        and np.all(right_array > 0.0)
        and np.all(np.abs(left_array - right_array) <= tolerance)
    )


def _input_domain_certificates_consistent(
    left: object | None,
    right: object | None,
) -> bool:
    if left is None or right is None:
        return False
    if not (
        bool(getattr(left, "certified", False))
        and bool(getattr(right, "certified", False))
    ):
        return False
    try:
        left_dimension = int(getattr(left, "dimension"))
        right_dimension = int(getattr(right, "dimension"))
    except (TypeError, ValueError):
        return False
    return bool(
        left_dimension == right_dimension
        and _mass_sequences_consistent(
            getattr(left, "masses", None),
            getattr(right, "masses", None),
        )
        and _array_close(
            getattr(left, "positions", None),
            getattr(right, "positions", None),
            tolerance=64.0 * np.finfo(float).eps,
        )
        and _array_close(
            getattr(left, "velocities", None),
            getattr(right, "velocities", None),
            tolerance=64.0 * np.finfo(float).eps,
        )
    )


def _compact_time_certificates_consistent(
    left: object | None,
    right: object | None,
) -> bool:
    if left is None or right is None:
        return False
    left_rate = float(getattr(left, "rate", float("nan")))
    right_rate = float(getattr(right, "rate", float("nan")))
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(left_rate),
        abs(right_rate),
    )
    return bool(
        bool(getattr(left, "certified", False))
        and bool(getattr(right, "certified", False))
        and np.isfinite(left_rate)
        and np.isfinite(right_rate)
        and abs(left_rate - right_rate) <= tolerance
        and float(getattr(left, "compact_lower", float("nan")))
        == float(getattr(right, "compact_lower", float("nan")))
        and float(getattr(left, "compact_upper", float("nan")))
        == float(getattr(right, "compact_upper", float("nan")))
    )


def _two_sided_event_budget_halves_match(
    observed: object | None,
    expected: object | None,
) -> bool:
    return bool(
        observed is not None
        and expected is not None
        and getattr(observed, "future_event_budget", None)
        is getattr(expected, "future_event_budget", None)
        and getattr(observed, "past_event_budget", None)
        is getattr(expected, "past_event_budget", None)
    )


def _weighted_planar_wedge_sum(
    masses: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    if left.shape[1] != 2 or right.shape[1] != 2:
        return float("nan")
    return float(
        sum(
            masses[index]
            * (
                left[index, 0] * right[index, 1]
                - left[index, 1] * right[index, 0]
            )
            for index in range(left.shape[0])
        )
    )


def _newtonian_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
) -> float:
    kinetic = 0.5 * float(np.sum(masses[:, None] * velocities**2))
    potential = 0.0
    for first in range(positions.shape[0]):
        for second in range(first + 1, positions.shape[0]):
            distance = float(np.linalg.norm(positions[first] - positions[second]))
            if distance <= 0.0 or not np.isfinite(distance):
                return float("nan")
            potential += float(masses[first] * masses[second] / distance)
    return kinetic - potential


def _tuple_field(certificate: object, field_name: str) -> tuple[float, ...]:
    value = getattr(certificate, field_name, ())
    try:
        return tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return ()


def _max_abs_tuple_difference(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    if len(left) != len(right):
        return float("inf")
    return float(max((abs(a - b) for a, b in zip(left, right)), default=0.0))


def _reject_raw_bool(name: str, certificate: object | None) -> None:
    if isinstance(certificate, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a constructor-derived certificate object, not a raw boolean"
        )


def _certificate_field_is_same_object(
    certificate: object | None,
    field_name: str,
    expected: object | None,
) -> bool:
    return bool(
        certificate is not None
        and expected is not None
        and getattr(certificate, field_name, None) is expected
    )
