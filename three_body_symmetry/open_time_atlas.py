"""Open-time locally finite atlas theorem surface.

This module implements the theorem direction that does not require classifying
the endpoint behavior of every trajectory.  The constructive target is local in
physical time: for a finite target, either construct a certified finite atlas,
return a certified total-collision stop/selected continuation, or expose a
typed proof-grade obstruction.  The all-real object is then the countable
locally finite exhaustion by compact physical-time intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .certificate_checker import (
    IndependentChartVerifierCertificate,
    attach_independent_chart_verifier,
    check_branch_union,
    verify_chart_certificates,
)
from .certificate_language import (
    BranchUnionCertificate,
    ChartChainCertificate,
    GeneralizedFuchsianRemainderMajorantCertificate,
    PrimitiveCauchyTailInputCertificate,
    ordinary_taylor_chart_certificate_from_solution,
    planar_hybrid_chart_chain_certificates_from_solution,
    spatial_ks_binary_chart_certificate_from_solution,
    spatial_ks_transition_certificate,
    total_collision_generalized_fuchsian_stop_chart_certificate_from_branch,
)
from .event_recurrence import PrimitiveCauchyTailInput
from .finite_time_regime import (
    FiniteTimeRegimeClassificationCertificate,
    classify_finite_time_regime,
)
from .finite_target_completeness import (
    FiniteTargetCertificateSearchCompletenessCertificate,
    FiniteTargetCompletenessTheoremCertificate,
    SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate,
    UniformMarginSetValuedConstructorCompletenessCertificate,
    ValidatedSetValuedConstructorCompletenessTheoremCertificate,
    certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness,
    certify_constructor_pair_derived_recursive_stratified_set_valued_constructor_completeness,
    certify_finite_target_certificate_search_completeness,
    certify_finite_target_completeness_theorem,
    recursive_constructor_source_scope,
)
from .general_solution_theorem import (
    CompactOrdinaryBinaryFiniteAtlasCertificate,
    CompactTimeCoverageCertificate,
    PositiveMassNoncollisionInputDomainCertificate,
    TheoremPipelineObligation,
    _theorem_pipeline_obligation_ledger_certified,
    _theorem_pipeline_obligation_ledger_missing,
    certify_compact_ordinary_binary_finite_atlas,
    certify_compact_time_real_line_coverage,
    certify_positive_mass_noncollision_input_domain,
)
from .fuchsian import construct_fuchsian_shape_branch
from .ks_binary_chart import SpatialKSBinaryChartState, ks_binary_chart_to_spatial
from .ks_binary_series import construct_spatial_ks_binary_taylor_solution
from .series import construct_taylor_solution
from .validated_atlas import finite_time_selector_trace_binding_token


FINITE_TARGET_OUTCOMES = (
    "finite_atlas_reaches_target",
    "unselected_total_collision_before_target",
    "selected_total_collision_continuation",
    "proof_grade_obstruction",
)

COMPACT_INTERVAL_OUTCOMES = (
    "compact_interval_atlas_reaches_both_endpoints",
    "compact_interval_total_collision_stop",
    "compact_interval_selected_total_collision_continuation",
    "proof_grade_obstruction",
)

TOTAL_COLLISION_POLICY_IDS = (
    "maximal_classical_stop",
    "maximal_classical_stop_at_total_collision",
    "selected_identity_selector",
    "selected_finite_jet_selector",
    "selected_fuchsian_log_selector",
    "selected_homothetic_selector",
)


def _open_time_obligation_ledger_certified(
    obligations: tuple[Any, ...],
) -> bool:
    return _theorem_pipeline_obligation_ledger_certified(obligations)


def _open_time_obligation_ledger_missing(
    obligations: tuple[Any, ...],
    *,
    ledger_name: str,
) -> tuple[str, ...]:
    return _theorem_pipeline_obligation_ledger_missing(
        obligations,
        ledger_name=ledger_name,
    )


@dataclass(frozen=True)
class AnalyticTheoremCertificate:
    """A theorem-level analytic lemma with the proof idea kept in prose."""

    theorem_id: str
    statement: str
    proof_sketch: str
    prerequisites: tuple[str, ...] = ()
    source: str = "open_time_atlas_theorem"
    proof_mode: str = "declared_only"
    internally_proven: bool = False
    externally_audited: bool = False
    machine_checked: bool = False

    @property
    def certified(self) -> bool:
        return bool(self.theorem_id and self.statement and self.proof_sketch)

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and (self.externally_audited or self.machine_checked)
        )

    @property
    def internally_supported(self) -> bool:
        return bool(
            self.certified
            and self.internally_proven
            and self.proof_mode != "declared_only"
        )


@dataclass(frozen=True)
class TotalCollisionPolicyCertificate:
    """Explicit policy for total collision in the finite-target theorem."""

    policy_id: str
    stop_at_unselected_total_collision: bool
    selected_continuation_allowed: bool
    selector_policy_id: str | None = None

    @property
    def certified(self) -> bool:
        stop_policy = self.policy_id.startswith("maximal_classical_stop")
        selected_policy = self.policy_id.startswith("selected_")
        return bool(
            self.policy_id in TOTAL_COLLISION_POLICY_IDS
            and self.stop_at_unselected_total_collision is stop_policy
            and self.selected_continuation_allowed is selected_policy
            and (
                self.selected_continuation_allowed is not True
                or bool(self.selector_policy_id)
            )
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified


@dataclass(frozen=True)
class FiniteTargetAtlasOrStopCertificate:
    """Theorem-facing finite-target atlas-or-stop outcome."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate | None
    target_time: float
    total_collision_policy: TotalCollisionPolicyCertificate
    finite_time_classification: FiniteTimeRegimeClassificationCertificate
    finite_atlas_certificate: CompactOrdinaryBinaryFiniteAtlasCertificate | None
    painleve_certificate: AnalyticTheoremCertificate
    binary_regularization_certificate: AnalyticTheoremCertificate
    binary_isolation_certificate: AnalyticTheoremCertificate
    binary_accumulation_certificate: AnalyticTheoremCertificate
    compact_collision_free_cover_certificate: AnalyticTheoremCertificate
    outcome_id: str
    validated_atlas: object | None
    stop_certificate: object | None
    obstruction_obligations: tuple[str, ...]
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def outcome_certified(self) -> bool:
        return self.outcome_id in {
            "finite_atlas_reaches_target",
            "unselected_total_collision_before_target",
            "selected_total_collision_continuation",
        }

    @property
    def proof_grade_response_certified(self) -> bool:
        return bool(
            self.outcome_certified
            or (
                self.outcome_id == "proof_grade_obstruction"
                and self.obstruction_obligations
            )
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.outcome_certified
            and _open_time_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return False

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _open_time_obligation_ledger_missing(
            self.obligations,
            ledger_name="finite_target_atlas_or_stop",
        )

    @property
    def chart_types(self) -> tuple[str, ...]:
        return tuple(
            str(getattr(chart, "chart_type", ""))
            for chart in getattr(self.validated_atlas, "charts", ())
        )


@dataclass(frozen=True)
class TotalCollisionStopCertificate:
    """A certified maximal-classical stop at an unselected total collision."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate | None
    validated_atlas: object | None
    total_collision_chart_type: str | None
    stop_time: float | None
    target_time: float
    total_collision_policy: TotalCollisionPolicyCertificate
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def certified(self) -> bool:
        return _open_time_obligation_ledger_certified(
            self.obligations,
        )

    @property
    def proof_certified(self) -> bool:
        return False

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _open_time_obligation_ledger_missing(
            self.obligations,
            ledger_name="total_collision_stop",
        )


@dataclass(frozen=True)
class CompactIntervalAtlasOrStopCertificate:
    """Two-sided compact physical-time certificate around the initial instant."""

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate | None
    interval_lower: float
    interval_upper: float
    total_collision_policy: TotalCollisionPolicyCertificate
    past_target_certificate: FiniteTargetAtlasOrStopCertificate | None
    future_target_certificate: FiniteTargetAtlasOrStopCertificate | None
    stop_certificate: object | None
    outcome_id: str
    obstruction_obligations: tuple[str, ...]
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def finite_target_certificates(self) -> tuple[FiniteTargetAtlasOrStopCertificate, ...]:
        return tuple(
            certificate
            for certificate in (
                self.past_target_certificate,
                self.future_target_certificate,
            )
            if isinstance(certificate, FiniteTargetAtlasOrStopCertificate)
        )

    @property
    def component_types_certified(self) -> bool:
        return bool(
            isinstance(self.total_collision_policy, TotalCollisionPolicyCertificate)
            and (
                self.past_target_certificate is None
                or isinstance(
                    self.past_target_certificate,
                    FiniteTargetAtlasOrStopCertificate,
                )
            )
            and (
                self.future_target_certificate is None
                or isinstance(
                    self.future_target_certificate,
                    FiniteTargetAtlasOrStopCertificate,
                )
            )
            and (
                self.stop_certificate is None
                or isinstance(self.stop_certificate, TotalCollisionStopCertificate)
            )
        )

    @property
    def outcome_certified(self) -> bool:
        return self.outcome_id in {
            "compact_interval_atlas_reaches_both_endpoints",
            "compact_interval_total_collision_stop",
            "compact_interval_selected_total_collision_continuation",
        }

    @property
    def proof_grade_response_certified(self) -> bool:
        return bool(
            self.outcome_certified
            or (
                self.outcome_id == "proof_grade_obstruction"
                and self.obstruction_obligations
            )
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.component_types_certified
            and self.outcome_certified
            and _open_time_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.component_types_certified
            and self.finite_target_certificates
            and all(certificate.proof_certified for certificate in self.finite_target_certificates)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        type_missing = []
        if not isinstance(self.total_collision_policy, TotalCollisionPolicyCertificate):
            type_missing.append("compact_interval_total_collision_policy_type")
        if not (
            self.past_target_certificate is None
            or isinstance(
                self.past_target_certificate,
                FiniteTargetAtlasOrStopCertificate,
            )
        ):
            type_missing.append("compact_interval_past_finite_target_type")
        if not (
            self.future_target_certificate is None
            or isinstance(
                self.future_target_certificate,
                FiniteTargetAtlasOrStopCertificate,
            )
        ):
            type_missing.append("compact_interval_future_finite_target_type")
        if not (
            self.stop_certificate is None
            or isinstance(self.stop_certificate, TotalCollisionStopCertificate)
        ):
            type_missing.append("compact_interval_stop_certificate_type")
        ledger_missing = _open_time_obligation_ledger_missing(
            self.obligations,
            ledger_name="compact_interval_atlas_or_stop",
        )
        return tuple(dict.fromkeys((*type_missing, *ledger_missing)))


@dataclass(frozen=True)
class CompactIntervalExhaustionFamilyCertificate:
    """Nested compact intervals used by the open-time theorem."""

    compact_time_certificate: CompactTimeCoverageCertificate
    base_time_radius: float
    prefix_count: int
    prefix_certificates: tuple[CompactIntervalAtlasOrStopCertificate, ...]
    exhaustion_formula: str
    finite_target_reduction_certificate: AnalyticTheoremCertificate
    local_finiteness_certificate: AnalyticTheoremCertificate
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def component_types_certified(self) -> bool:
        return bool(
            isinstance(self.compact_time_certificate, CompactTimeCoverageCertificate)
            and self.prefix_certificates
            and all(
                isinstance(certificate, CompactIntervalAtlasOrStopCertificate)
                for certificate in self.prefix_certificates
            )
            and isinstance(
                self.finite_target_reduction_certificate,
                AnalyticTheoremCertificate,
            )
            and isinstance(
                self.local_finiteness_certificate,
                AnalyticTheoremCertificate,
            )
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.component_types_certified
            and _open_time_obligation_ledger_certified(
                self.obligations,
            )
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.component_types_certified
            and self.prefix_certificates
            and all(certificate.proof_certified for certificate in self.prefix_certificates)
            and self.finite_target_reduction_certificate.proof_certified
            and self.local_finiteness_certificate.proof_certified
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        type_missing = []
        if not isinstance(
            self.compact_time_certificate,
            CompactTimeCoverageCertificate,
        ):
            type_missing.append("compact_interval_exhaustion_compact_time_type")
        if not self.prefix_certificates:
            type_missing.append("compact_interval_exhaustion_prefixes_present")
        for certificate in self.prefix_certificates:
            if not isinstance(certificate, CompactIntervalAtlasOrStopCertificate):
                type_missing.append("compact_interval_exhaustion_prefix_type")
                break
        if not isinstance(
            self.finite_target_reduction_certificate,
            AnalyticTheoremCertificate,
        ):
            type_missing.append("compact_interval_exhaustion_reduction_type")
        if not isinstance(
            self.local_finiteness_certificate,
            AnalyticTheoremCertificate,
        ):
            type_missing.append("compact_interval_exhaustion_local_finiteness_type")
        ledger_missing = _open_time_obligation_ledger_missing(
            self.obligations,
            ledger_name="compact_interval_exhaustion_family",
        )
        return tuple(dict.fromkeys((*type_missing, *ledger_missing)))

    @property
    def interval_radii(self) -> tuple[float, ...]:
        return tuple(
            0.5
            * (
                float(certificate.interval_upper)
                - float(certificate.interval_lower)
            )
            for certificate in self.prefix_certificates
        )


@dataclass(frozen=True)
class CountableCompactExhaustionCertificate:
    """Countable locally finite exhaustion of an open physical-time interval."""

    compact_time_certificate: CompactTimeCoverageCertificate
    finite_target_certificate: FiniteTargetAtlasOrStopCertificate
    exhaustion_id: str
    obligations: tuple[TheoremPipelineObligation, ...]
    compact_interval_certificate: CompactIntervalAtlasOrStopCertificate | None = None
    exhaustion_family_certificate: CompactIntervalExhaustionFamilyCertificate | None = None

    @property
    def component_types_certified(self) -> bool:
        return bool(
            isinstance(self.compact_time_certificate, CompactTimeCoverageCertificate)
            and isinstance(
                self.finite_target_certificate,
                FiniteTargetAtlasOrStopCertificate,
            )
            and (
                self.compact_interval_certificate is None
                or isinstance(
                    self.compact_interval_certificate,
                    CompactIntervalAtlasOrStopCertificate,
                )
            )
            and (
                self.exhaustion_family_certificate is None
                or isinstance(
                    self.exhaustion_family_certificate,
                    CompactIntervalExhaustionFamilyCertificate,
                )
            )
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.component_types_certified
            and _open_time_obligation_ledger_certified(
                self.obligations,
            )
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.component_types_certified
            and self.finite_target_certificate.proof_certified
            and (
                self.compact_interval_certificate is None
                or self.compact_interval_certificate.proof_certified
            )
            and (
                self.exhaustion_family_certificate is None
                or self.exhaustion_family_certificate.proof_certified
            )
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        type_missing = []
        if not isinstance(
            self.compact_time_certificate,
            CompactTimeCoverageCertificate,
        ):
            type_missing.append("countable_exhaustion_compact_time_type")
        if not isinstance(
            self.finite_target_certificate,
            FiniteTargetAtlasOrStopCertificate,
        ):
            type_missing.append("countable_exhaustion_finite_target_type")
        if not (
            self.compact_interval_certificate is None
            or isinstance(
                self.compact_interval_certificate,
                CompactIntervalAtlasOrStopCertificate,
            )
        ):
            type_missing.append("countable_exhaustion_compact_interval_type")
        if not (
            self.exhaustion_family_certificate is None
            or isinstance(
                self.exhaustion_family_certificate,
                CompactIntervalExhaustionFamilyCertificate,
            )
        ):
            type_missing.append("countable_exhaustion_family_type")
        ledger_missing = _open_time_obligation_ledger_missing(
            self.obligations,
            ledger_name="countable_compact_exhaustion",
        )
        return tuple(dict.fromkeys((*type_missing, *ledger_missing)))


@dataclass(frozen=True)
class FiniteTargetCompletenessReductionCertificate:
    """Reduction from pointwise finite-target evidence to the universal theorem.

    This is intentionally stricter than a checked example and narrower than
    the point-input theorem.  The pointwise atlas-or-stop theorem and fair
    exact-point search can certify today; the arbitrary-input constructor still
    needs proof that set-valued branch partitions and ambiguous event-order
    trees are recursively consumed.
    """

    finite_target_certificate: FiniteTargetAtlasOrStopCertificate
    compact_interval_certificate: CompactIntervalAtlasOrStopCertificate | None
    exhaustion_family_certificate: CompactIntervalExhaustionFamilyCertificate | None
    statement: str
    proof_sketch: str
    obligations: tuple[TheoremPipelineObligation, ...]
    pointwise_completeness_theorem: (
        FiniteTargetCompletenessTheoremCertificate | None
    ) = None
    certificate_search_completeness: (
        FiniteTargetCertificateSearchCompletenessCertificate | None
    ) = None
    set_valued_constructor_completeness_certificate: (
        UniformMarginSetValuedConstructorCompletenessCertificate
        | SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate
        | ValidatedSetValuedConstructorCompletenessTheoremCertificate
        | None
    ) = None

    @property
    def component_types_certified(self) -> bool:
        return bool(
            isinstance(
                self.finite_target_certificate,
                FiniteTargetAtlasOrStopCertificate,
            )
            and (
                self.compact_interval_certificate is None
                or isinstance(
                    self.compact_interval_certificate,
                    CompactIntervalAtlasOrStopCertificate,
                )
            )
            and (
                self.exhaustion_family_certificate is None
                or isinstance(
                    self.exhaustion_family_certificate,
                    CompactIntervalExhaustionFamilyCertificate,
                )
            )
            and isinstance(
                self.pointwise_completeness_theorem,
                FiniteTargetCompletenessTheoremCertificate,
            )
            and isinstance(
                self.certificate_search_completeness,
                FiniteTargetCertificateSearchCompletenessCertificate,
            )
            and (
                self.set_valued_constructor_completeness_certificate is None
                or isinstance(
                    self.set_valued_constructor_completeness_certificate,
                    (
                        UniformMarginSetValuedConstructorCompletenessCertificate,
                        SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate,
                        ValidatedSetValuedConstructorCompletenessTheoremCertificate,
                    ),
                )
            )
        )

    @property
    def source_matches(self) -> bool:
        if not self.component_types_certified:
            return False
        input_domain = self.finite_target_certificate.input_domain_certificate
        total_collision_policy = self.finite_target_certificate.total_collision_policy
        return bool(
            int(self.pointwise_completeness_theorem.dimension)
            == int(getattr(input_domain, "dimension", 0))
            and str(self.pointwise_completeness_theorem.total_collision_policy_id)
            == str(getattr(total_collision_policy, "policy_id", ""))
            and self.certificate_search_completeness.theorem_certificate
            is self.pointwise_completeness_theorem
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.component_types_certified
            and self.source_matches
            and _open_time_obligation_ledger_certified(
                self.obligations,
            )
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.component_types_certified
            and self.source_matches
            and self.finite_target_certificate.proof_certified
            and (
                self.pointwise_completeness_theorem is not None
                and self.pointwise_completeness_theorem.proof_certified
            )
            and (
                self.certificate_search_completeness is not None
                and self.certificate_search_completeness.proof_certified
            )
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        type_missing = []
        if not isinstance(
            self.finite_target_certificate,
            FiniteTargetAtlasOrStopCertificate,
        ):
            type_missing.append("finite_target_reduction_finite_target_type")
        if not (
            self.compact_interval_certificate is None
            or isinstance(
                self.compact_interval_certificate,
                CompactIntervalAtlasOrStopCertificate,
            )
        ):
            type_missing.append("finite_target_reduction_compact_interval_type")
        if not (
            self.exhaustion_family_certificate is None
            or isinstance(
                self.exhaustion_family_certificate,
                CompactIntervalExhaustionFamilyCertificate,
            )
        ):
            type_missing.append("finite_target_reduction_exhaustion_family_type")
        if not isinstance(
            self.pointwise_completeness_theorem,
            FiniteTargetCompletenessTheoremCertificate,
        ):
            type_missing.append("finite_target_reduction_pointwise_theorem_type")
        if not isinstance(
            self.certificate_search_completeness,
            FiniteTargetCertificateSearchCompletenessCertificate,
        ):
            type_missing.append("finite_target_reduction_search_completeness_type")
        if not (
            self.set_valued_constructor_completeness_certificate is None
            or isinstance(
                self.set_valued_constructor_completeness_certificate,
                (
                    UniformMarginSetValuedConstructorCompletenessCertificate,
                    SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate,
                    ValidatedSetValuedConstructorCompletenessTheoremCertificate,
                ),
            )
        ):
            type_missing.append("finite_target_reduction_set_valued_constructor_type")
        if self.component_types_certified and not self.source_matches:
            type_missing.append("finite_target_reduction_source_match")
        own_missing = _open_time_obligation_ledger_missing(
            self.obligations,
            ledger_name="finite_target_completeness_reduction",
        )
        pointwise_missing = (
            self.pointwise_completeness_theorem.missing_obligations
            if isinstance(
                self.pointwise_completeness_theorem,
                FiniteTargetCompletenessTheoremCertificate,
            )
            else ("pointwise_finite_target_atlas_or_stop_completeness",)
        )
        search_missing = (
            self.certificate_search_completeness.missing_obligations
            if isinstance(
                self.certificate_search_completeness,
                FiniteTargetCertificateSearchCompletenessCertificate,
            )
            else ("finite_target_certificate_search_completeness",)
        )
        return tuple(
            dict.fromkeys(
                (
                    *type_missing,
                    *pointwise_missing,
                    *search_missing,
                    *own_missing,
                )
            )
        )

    @property
    def unaudited_analytic_lemma_ids(self) -> tuple[str, ...]:
        if not isinstance(
            self.pointwise_completeness_theorem,
            FiniteTargetCompletenessTheoremCertificate,
        ):
            return ()
        return self.pointwise_completeness_theorem.unaudited_analytic_lemma_ids

    @property
    def critical_unaudited_analytic_lemma_ids(self) -> tuple[str, ...]:
        if not isinstance(
            self.pointwise_completeness_theorem,
            FiniteTargetCompletenessTheoremCertificate,
        ):
            return ()
        return (
            self.pointwise_completeness_theorem
            .critical_unaudited_analytic_lemma_ids
        )

    @property
    def analytic_lemma_audit_blockers(self) -> tuple[str, ...]:
        if not isinstance(
            self.pointwise_completeness_theorem,
            FiniteTargetCompletenessTheoremCertificate,
        ):
            return ()
        return self.pointwise_completeness_theorem.analytic_lemma_audit_blockers

    @property
    def set_valued_constructor_input_scope_id(self) -> str | None:
        certificate = self.set_valued_constructor_completeness_certificate
        if certificate is None:
            return None
        return getattr(
            certificate,
            "input_scope_id",
            getattr(certificate, "theorem_id", None),
        )

    @property
    def set_valued_constructor_arbitrary_partition_generation_claimed(self) -> bool:
        certificate = self.set_valued_constructor_completeness_certificate
        if certificate is None:
            return False
        return bool(
            getattr(certificate, "arbitrary_partition_generation_claimed", False)
        )

    @property
    def scoped_set_valued_constructor_only(self) -> bool:
        return bool(
            self.set_valued_constructor_completeness_certificate is not None
            and not self.set_valued_constructor_arbitrary_partition_generation_claimed
        )


@dataclass(frozen=True)
class OpenTimeLocallyFiniteAtlasTheoremCertificate:
    """Open-time theorem using finite-target atlas-or-stop certificates."""

    finite_target_certificate: FiniteTargetAtlasOrStopCertificate
    compact_interval_certificate: CompactIntervalAtlasOrStopCertificate | None
    exhaustion_family_certificate: CompactIntervalExhaustionFamilyCertificate | None
    countable_exhaustion_certificate: CountableCompactExhaustionCertificate
    endpoint_regime_partition_required: bool
    obligations: tuple[TheoremPipelineObligation, ...]
    finite_target_completeness_certificate: (
        FiniteTargetCompletenessReductionCertificate | None
    ) = None
    independent_chart_verifier_certificate: object | None = None
    independent_chart_verifier_certified: bool = False
    theorem_id: str = "open_time_locally_finite_atlas"

    @property
    def component_types_certified(self) -> bool:
        return bool(
            isinstance(
                self.finite_target_certificate,
                FiniteTargetAtlasOrStopCertificate,
            )
            and (
                self.compact_interval_certificate is None
                or isinstance(
                    self.compact_interval_certificate,
                    CompactIntervalAtlasOrStopCertificate,
                )
            )
            and (
                self.exhaustion_family_certificate is None
                or isinstance(
                    self.exhaustion_family_certificate,
                    CompactIntervalExhaustionFamilyCertificate,
                )
            )
            and isinstance(
                self.countable_exhaustion_certificate,
                CountableCompactExhaustionCertificate,
            )
            and isinstance(
                self.finite_target_completeness_certificate,
                FiniteTargetCompletenessReductionCertificate,
            )
            and (
                self.independent_chart_verifier_certificate is None
                or type(self.independent_chart_verifier_certificate)
                is IndependentChartVerifierCertificate
            )
        )

    @property
    def theorem_prefix_obligations_certified(self) -> bool:
        return _open_time_obligation_ledger_certified(
            self.obligations,
        )

    @property
    def independent_checked_prefix_certified(self) -> bool:
        verifier = self.independent_chart_verifier_certificate
        return bool(
            self.independent_chart_verifier_certified is True
            and type(verifier) is IndependentChartVerifierCertificate
            and verifier.certified is True
            and verifier.proof_grade_finite_atlas_bundle_certified is True
        )

    @property
    def checked_prefix_certified(self) -> bool:
        return bool(
            self.theorem_prefix_obligations_certified
            or self.independent_checked_prefix_certified
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.component_types_certified
            and self.theorem_prefix_obligations_certified
            and self.arbitrary_finite_target_completeness_certified
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.component_types_certified
            and not self.scoped_set_valued_constructor_only
            and self.countable_exhaustion_certificate.proof_certified
            and (
                self.finite_target_completeness_certificate is not None
                and self.finite_target_completeness_certificate.proof_certified
            )
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        type_missing = []
        if not isinstance(
            self.finite_target_certificate,
            FiniteTargetAtlasOrStopCertificate,
        ):
            type_missing.append("open_time_finite_target_certificate_type")
        if not (
            self.compact_interval_certificate is None
            or isinstance(
                self.compact_interval_certificate,
                CompactIntervalAtlasOrStopCertificate,
            )
        ):
            type_missing.append("open_time_compact_interval_certificate_type")
        if not (
            self.exhaustion_family_certificate is None
            or isinstance(
                self.exhaustion_family_certificate,
                CompactIntervalExhaustionFamilyCertificate,
            )
        ):
            type_missing.append("open_time_exhaustion_family_certificate_type")
        if not isinstance(
            self.countable_exhaustion_certificate,
            CountableCompactExhaustionCertificate,
        ):
            type_missing.append("open_time_countable_exhaustion_certificate_type")
        if not isinstance(
            self.finite_target_completeness_certificate,
            FiniteTargetCompletenessReductionCertificate,
        ):
            type_missing.append("open_time_finite_target_completeness_certificate_type")
        if not (
            self.independent_chart_verifier_certificate is None
            or type(self.independent_chart_verifier_certificate)
            is IndependentChartVerifierCertificate
        ):
            type_missing.append("open_time_independent_chart_verifier_certificate_type")
        own_missing = _open_time_obligation_ledger_missing(
            self.obligations,
            ledger_name="open_time_locally_finite_atlas",
        )
        return tuple(
            dict.fromkeys(
                (
                    *type_missing,
                    *own_missing,
                    *self.finite_target_completeness_missing_obligations,
                )
            )
        )

    @property
    def arbitrary_finite_target_completeness_certified(self) -> bool:
        return bool(
            isinstance(
                self.finite_target_completeness_certificate,
                FiniteTargetCompletenessReductionCertificate,
            )
            and self.finite_target_completeness_certificate.certified
        )

    @property
    def finite_target_completeness_missing_obligations(self) -> tuple[str, ...]:
        if self.finite_target_completeness_certificate is None:
            return ("finite_target_completeness_reduction_certificate",)
        if not isinstance(
            self.finite_target_completeness_certificate,
            FiniteTargetCompletenessReductionCertificate,
        ):
            return ("finite_target_completeness_reduction_certificate_type",)
        return self.finite_target_completeness_certificate.missing_obligations

    @property
    def set_valued_constructor_input_scope_id(self) -> str | None:
        if self.finite_target_completeness_certificate is None:
            return None
        if not isinstance(
            self.finite_target_completeness_certificate,
            FiniteTargetCompletenessReductionCertificate,
        ):
            return None
        return getattr(
            self.finite_target_completeness_certificate,
            "set_valued_constructor_input_scope_id",
            None,
        )

    @property
    def set_valued_constructor_arbitrary_partition_generation_claimed(self) -> bool:
        if self.finite_target_completeness_certificate is None:
            return False
        if not isinstance(
            self.finite_target_completeness_certificate,
            FiniteTargetCompletenessReductionCertificate,
        ):
            return False
        return bool(
            getattr(
                self.finite_target_completeness_certificate,
                "set_valued_constructor_arbitrary_partition_generation_claimed",
                False,
            )
        )

    @property
    def scoped_set_valued_constructor_only(self) -> bool:
        if self.finite_target_completeness_certificate is None:
            return False
        if not isinstance(
            self.finite_target_completeness_certificate,
            FiniteTargetCompletenessReductionCertificate,
        ):
            return False
        return bool(
            getattr(
                self.finite_target_completeness_certificate,
                "scoped_set_valued_constructor_only",
                False,
            )
        )

    @property
    def independent_chart_verifier_arithmetic_certified(self) -> bool:
        return bool(
            self.independent_chart_verifier_certified is True
            and type(self.independent_chart_verifier_certificate)
            is IndependentChartVerifierCertificate
            and (
                self.independent_chart_verifier_certificate
                .proof_grade_finite_atlas_bundle_certified
                is True
            )
        )

    @property
    def independent_chart_verifier_arithmetic_blockers(self) -> tuple[str, ...]:
        if (
            not self.independent_chart_verifier_certified
            or self.independent_chart_verifier_certificate is None
        ):
            return ()
        if (
            type(self.independent_chart_verifier_certificate)
            is not IndependentChartVerifierCertificate
        ):
            return ("independent_chart_verifier_certificate_type",)
        return tuple(
            str(blocker)
            for blocker in getattr(
                self.independent_chart_verifier_certificate,
                "proof_grade_finite_atlas_blockers",
                (),
            )
            if blocker
        )

    @property
    def unaudited_analytic_lemma_ids(self) -> tuple[str, ...]:
        if self.finite_target_completeness_certificate is None:
            return ()
        if not isinstance(
            self.finite_target_completeness_certificate,
            FiniteTargetCompletenessReductionCertificate,
        ):
            return ()
        return (
            self.finite_target_completeness_certificate
            .unaudited_analytic_lemma_ids
        )

    @property
    def critical_unaudited_analytic_lemma_ids(self) -> tuple[str, ...]:
        if self.finite_target_completeness_certificate is None:
            return ()
        if not isinstance(
            self.finite_target_completeness_certificate,
            FiniteTargetCompletenessReductionCertificate,
        ):
            return ()
        return (
            self.finite_target_completeness_certificate
            .critical_unaudited_analytic_lemma_ids
        )

    @property
    def analytic_lemma_audit_blockers(self) -> tuple[str, ...]:
        if self.finite_target_completeness_certificate is None:
            return ()
        if not isinstance(
            self.finite_target_completeness_certificate,
            FiniteTargetCompletenessReductionCertificate,
        ):
            return ()
        return (
            self.finite_target_completeness_certificate
            .analytic_lemma_audit_blockers
        )

    @property
    def route_summary(self) -> str:
        if self.certified:
            if self.scoped_set_valued_constructor_only:
                return (
                    "open-time locally finite atlas theorem certified for "
                    f"scoped set-valued constructor input class "
                    f"{self.set_valued_constructor_input_scope_id}; arbitrary "
                    "interval-input partition generation remains separate"
                )
            return "open-time locally finite atlas theorem certified for arbitrary finite targets"
        if self.checked_prefix_certified:
            return "compact-interval atlas-or-stop prefix certified; arbitrary finite-target completeness remains open"
        return "open-time locally finite atlas theorem has a typed finite-target obstruction"


@dataclass(frozen=True)
class PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate:
    """Exact-input open-time theorem derived from finite-target completeness.

    This is the mathematical compact-exhaustion theorem for exact point
    initial data.  It deliberately does not certify today's interval-box
    branch/event-order recursion backend.
    """

    dimension: int
    input_model: str
    total_collision_policy_id: str
    compact_time_certificate: CompactTimeCoverageCertificate
    finite_target_theorem: FiniteTargetCompletenessTheoremCertificate
    finite_target_reduction_certificate: AnalyticTheoremCertificate
    local_finiteness_certificate: AnalyticTheoremCertificate
    endpoint_regime_partition_required: bool
    statement: str
    proof_sketch: str
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "pointwise_open_time_locally_finite_atlas"

    @property
    def component_types_certified(self) -> bool:
        return bool(
            isinstance(self.compact_time_certificate, CompactTimeCoverageCertificate)
            and isinstance(
                self.finite_target_theorem,
                FiniteTargetCompletenessTheoremCertificate,
            )
            and isinstance(
                self.finite_target_reduction_certificate,
                AnalyticTheoremCertificate,
            )
            and isinstance(
                self.local_finiteness_certificate,
                AnalyticTheoremCertificate,
            )
        )

    @property
    def finite_target_theorem_source_matches(self) -> bool:
        return bool(
            isinstance(
                self.finite_target_theorem,
                FiniteTargetCompletenessTheoremCertificate,
            )
            and int(self.finite_target_theorem.dimension) == int(self.dimension)
            and str(self.finite_target_theorem.input_model) == str(self.input_model)
            and str(self.finite_target_theorem.total_collision_policy_id)
            == str(self.total_collision_policy_id)
        )

    @property
    def theorem_scope_parameters_certified(self) -> bool:
        return bool(
            self.theorem_id == "pointwise_open_time_locally_finite_atlas"
            and self.endpoint_regime_partition_required is False
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.component_types_certified
            and self.finite_target_theorem_source_matches
            and self.theorem_scope_parameters_certified
            and _open_time_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.component_types_certified
            and self.finite_target_theorem_source_matches
            and self.finite_target_theorem.proof_certified
            and getattr(
                self.compact_time_certificate,
                "proof_certified",
                self.compact_time_certificate.certified,
            )
            and self.finite_target_reduction_certificate.proof_certified
            and self.local_finiteness_certificate.proof_certified
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        type_missing = []
        if not isinstance(
            self.compact_time_certificate,
            CompactTimeCoverageCertificate,
        ):
            type_missing.append("pointwise_open_time_compact_time_type")
        if self.theorem_id != "pointwise_open_time_locally_finite_atlas":
            type_missing.append("pointwise_open_time_theorem_id")
        if self.endpoint_regime_partition_required is not False:
            type_missing.append("endpoint_regime_partition_not_required")
        if not isinstance(
            self.finite_target_theorem,
            FiniteTargetCompletenessTheoremCertificate,
        ):
            type_missing.append("pointwise_open_time_finite_target_theorem_type")
        elif not self.finite_target_theorem_source_matches:
            type_missing.append("pointwise_open_time_finite_target_theorem_source_match")
        elif not self.finite_target_theorem.proof_certified:
            type_missing.append("pointwise_open_time_finite_target_theorem_proof")
            type_missing.extend(
                self.finite_target_theorem.analytic_lemma_audit_blockers
            )
        if not isinstance(
            self.finite_target_reduction_certificate,
            AnalyticTheoremCertificate,
        ):
            type_missing.append(
                "pointwise_open_time_finite_target_reduction_type"
            )
        if not isinstance(
            self.local_finiteness_certificate,
            AnalyticTheoremCertificate,
        ):
            type_missing.append("pointwise_open_time_local_finiteness_type")
        if isinstance(
            self.compact_time_certificate,
            CompactTimeCoverageCertificate,
        ) and not getattr(
            self.compact_time_certificate,
            "proof_certified",
            self.compact_time_certificate.certified,
        ):
            type_missing.append("pointwise_open_time_compact_time_proof")
        if isinstance(
            self.finite_target_reduction_certificate,
            AnalyticTheoremCertificate,
        ) and not self.finite_target_reduction_certificate.proof_certified:
            type_missing.append(
                "audited_or_machine_checked:"
                + self.finite_target_reduction_certificate.theorem_id
            )
        if isinstance(
            self.local_finiteness_certificate,
            AnalyticTheoremCertificate,
        ) and not self.local_finiteness_certificate.proof_certified:
            type_missing.append(
                "audited_or_machine_checked:"
                + self.local_finiteness_certificate.theorem_id
            )
        ledger_missing = _open_time_obligation_ledger_missing(
            self.obligations,
            ledger_name="pointwise_open_time_locally_finite_atlas",
        )
        return tuple(dict.fromkeys((*type_missing, *ledger_missing)))

    @property
    def unaudited_analytic_lemma_ids(self) -> tuple[str, ...]:
        if not isinstance(
            self.finite_target_theorem,
            FiniteTargetCompletenessTheoremCertificate,
        ):
            return ()
        return self.finite_target_theorem.unaudited_analytic_lemma_ids

    @property
    def critical_unaudited_analytic_lemma_ids(self) -> tuple[str, ...]:
        if not isinstance(
            self.finite_target_theorem,
            FiniteTargetCompletenessTheoremCertificate,
        ):
            return ()
        return self.finite_target_theorem.critical_unaudited_analytic_lemma_ids

    @property
    def analytic_lemma_audit_blockers(self) -> tuple[str, ...]:
        if not isinstance(
            self.finite_target_theorem,
            FiniteTargetCompletenessTheoremCertificate,
        ):
            return ()
        return self.finite_target_theorem.analytic_lemma_audit_blockers

    @property
    def route_summary(self) -> str:
        if self.proof_certified:
            return (
                "pointwise open-time locally finite atlas-or-stop theorem "
                "proof-certified by finite-target compact exhaustion"
            )
        if self.certified:
            return (
                "pointwise open-time atlas theorem scaffold assembled; "
                "nested analytic proof obligations remain unaudited"
            )
        return "pointwise open-time theorem has missing finite-target obligations"


def construct_finite_target_atlas_or_stop(
    masses: Any,
    positions: Any,
    velocities: Any,
    target_time: float,
    *,
    total_collision_policy: str = "maximal_classical_stop",
    selector_policy_id: str | None = None,
    **solver_options: Any,
) -> FiniteTargetAtlasOrStopCertificate:
    """Construct the finite-target atlas-or-stop theorem object."""

    target_time = float(target_time)
    finite_time_classification = classify_finite_time_regime(
        masses,
        positions,
        velocities,
        target_time,
        **solver_options,
    )
    return _finite_target_atlas_or_stop_from_classification(
        finite_time_classification,
        total_collision_policy=total_collision_policy,
        selector_policy_id=selector_policy_id,
    )


def certify_finite_target_atlas_or_stop_from_validated_atlas(
    masses: Any,
    positions: Any,
    velocities: Any,
    target_time: float,
    validated_atlas: object,
    *,
    total_collision_policy: str = "maximal_classical_stop",
    selector_policy_id: str | None = None,
    independent_chart_verifier_certificate: object | None = None,
) -> FiniteTargetAtlasOrStopCertificate:
    """Certify a finite-target theorem response from a constructor-derived atlas."""

    target_time = float(target_time)
    input_domain = _certify_input_domain_or_none(masses, positions, velocities)
    atlas_or_checker_certified = bool(
        getattr(validated_atlas, "proof_certified", False) is True
        or _independent_verifier_covers_validated_atlas(
            independent_chart_verifier_certificate,
            validated_atlas,
        )
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="positive_mass_noncollision_input_domain",
            certified=bool(getattr(input_domain, "certified", False)),
            source="certify_positive_mass_noncollision_input_domain",
            detail="finite-target atlas adapter input domain",
        ),
        TheoremPipelineObligation(
            obligation="finite_target_time",
            certified=bool(np.isfinite(target_time)),
            source="certify_finite_target_atlas_or_stop_from_validated_atlas",
            detail=f"target_time={target_time!r}",
        ),
        TheoremPipelineObligation(
            obligation="supplied_validated_atlas_proof_certified",
            certified=atlas_or_checker_certified,
            source=(
                type(independent_chart_verifier_certificate).__name__
                if independent_chart_verifier_certificate is not None
                else type(validated_atlas).__name__
            ),
            detail=(
                "adapter consumes constructor proof or independent serialized "
                "checker evidence for the supplied ValidatedAtlasSolution-like object"
            ),
        ),
        TheoremPipelineObligation(
            obligation="supplied_validated_atlas_matches_input_domain",
            certified=_validated_atlas_initial_state_matches_input_domain(
                validated_atlas,
                input_domain,
            ),
            source=type(validated_atlas).__name__,
            detail="atlas initial state and masses bind to the supplied finite-target input",
        ),
    )
    classification = FiniteTimeRegimeClassificationCertificate(
        input_domain_certificate=input_domain,
        target_time=target_time,
        validated_atlas=validated_atlas if atlas_or_checker_certified else None,
        selected_route_id="supplied_validated_atlas",
        selector_trace=getattr(validated_atlas, "selector_trace", None),
        branch_partition=None,
        obligations=obligations,
        failure_reason=None,
        failure_obligations=(),
    )
    return _finite_target_atlas_or_stop_from_classification(
        classification,
        total_collision_policy=total_collision_policy,
        selector_policy_id=selector_policy_id,
        independent_chart_verifier_certificate=independent_chart_verifier_certificate,
    )


def _finite_target_atlas_or_stop_from_classification(
    finite_time_classification: FiniteTimeRegimeClassificationCertificate,
    *,
    total_collision_policy: str,
    selector_policy_id: str | None,
    independent_chart_verifier_certificate: object | None = None,
) -> FiniteTargetAtlasOrStopCertificate:
    target_time = float(finite_time_classification.target_time)
    input_domain = finite_time_classification.input_domain_certificate
    dimension = int(getattr(input_domain, "dimension", 0))
    policy = certify_explicit_total_collision_policy(
        total_collision_policy,
        selector_policy_id=selector_policy_id,
    )
    painleve = certify_three_body_painleve_no_noncollision_singularities()
    binary_regularization = certify_all_pair_binary_regularization(
        dimension=dimension,
    )
    binary_isolation = certify_binary_collision_isolation(
        dimension=dimension,
    )
    binary_accumulation = certify_binary_accumulation_implies_total_collision()
    compact_cover = certify_compact_collision_free_taylor_cover()
    validated_atlas = finite_time_classification.validated_atlas
    finite_atlas_certificate = None
    if input_domain is not None and validated_atlas is not None:
        finite_atlas_certificate = certify_compact_ordinary_binary_finite_atlas(
            input_domain_certificate=input_domain,
            validated_atlas=validated_atlas,
            total_collision_policy_id=policy.policy_id,
        )
    stop_certificate = certify_total_collision_stop_from_validated_atlas(
        input_domain_certificate=input_domain,
        validated_atlas=validated_atlas,
        target_time=target_time,
        total_collision_policy=policy,
        independent_chart_verifier_certificate=independent_chart_verifier_certificate,
    )
    chart_types = tuple(
        str(getattr(chart, "chart_type", ""))
        for chart in getattr(validated_atlas, "charts", ())
    )
    has_total_collision_chart = any("total_collision" in chart_type for chart_type in chart_types)
    finite_atlas_certified = bool(getattr(finite_atlas_certificate, "certified", False))
    total_collision_stop_certified = bool(stop_certificate.certified)
    stop_certified = _classification_certifies_unselected_total_collision_stop(
        finite_time_classification,
    )
    if finite_atlas_certified and has_total_collision_chart:
        outcome_id = (
            "selected_total_collision_continuation"
            if policy.selected_continuation_allowed
            else "proof_grade_obstruction"
        )
    elif finite_atlas_certified:
        outcome_id = "finite_atlas_reaches_target"
    elif total_collision_stop_certified:
        outcome_id = "unselected_total_collision_before_target"
    elif stop_certified and policy.stop_at_unselected_total_collision:
        outcome_id = "unselected_total_collision_before_target"
    else:
        outcome_id = "proof_grade_obstruction"
    obstruction_obligations = _finite_target_obstruction_obligations(
        finite_time_classification,
        outcome_id=outcome_id,
        has_total_collision_chart=has_total_collision_chart,
        total_collision_policy=policy,
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="positive_mass_noncollision_input_domain",
            certified=bool(getattr(input_domain, "certified", False)),
            source=type(input_domain).__name__ if input_domain is not None else "missing",
            detail="finite target theorem consumes arbitrary positive-mass noncollision data",
        ),
        TheoremPipelineObligation(
            obligation="finite_time_classification",
            certified=finite_time_classification.certified,
            source=type(finite_time_classification).__name__,
            detail=(
                "missing="
                + ",".join(finite_time_classification.missing_obligations)
            )
            if not finite_time_classification.certified
            else finite_time_classification.selected_route_id or "finite-time route certified",
        ),
        TheoremPipelineObligation(
            obligation="finite_target_time",
            certified=bool(np.isfinite(target_time)),
            source="construct_finite_target_atlas_or_stop",
            detail=f"target_time={target_time!r}",
        ),
        TheoremPipelineObligation(
            obligation="explicit_total_collision_policy",
            certified=policy.certified,
            source=type(policy).__name__,
            detail=policy.policy_id,
        ),
        TheoremPipelineObligation(
            obligation="three_body_painleve_no_noncollision_singularities",
            certified=painleve.certified,
            source=painleve.source,
            detail=painleve.statement,
        ),
        TheoremPipelineObligation(
            obligation="all_pair_binary_regularization",
            certified=binary_regularization.certified,
            source=binary_regularization.source,
            detail=binary_regularization.statement,
        ),
        TheoremPipelineObligation(
            obligation="binary_collision_isolation",
            certified=binary_isolation.certified,
            source=binary_isolation.source,
            detail=binary_isolation.statement,
        ),
        TheoremPipelineObligation(
            obligation="binary_accumulation_implies_total_collision",
            certified=binary_accumulation.certified,
            source=binary_accumulation.source,
            detail=binary_accumulation.statement,
        ),
        TheoremPipelineObligation(
            obligation="compact_collision_free_taylor_cover",
            certified=compact_cover.certified,
            source=compact_cover.source,
            detail=compact_cover.statement,
        ),
        TheoremPipelineObligation(
            obligation="compact_ordinary_binary_finite_atlas",
            certified=finite_atlas_certified or total_collision_stop_certified,
            source=(
                type(stop_certificate).__name__
                if total_collision_stop_certified
                else type(finite_atlas_certificate).__name__
                if finite_atlas_certificate is not None
                else "missing"
            ),
            detail=(
                "missing="
                + ",".join(getattr(finite_atlas_certificate, "missing_obligations", ()))
            )
            if finite_atlas_certificate is not None and not total_collision_stop_certified
            else "maximal classical stop certified before unselected total collision"
            if total_collision_stop_certified
            else "validated atlas was unavailable",
        ),
        TheoremPipelineObligation(
            obligation="certified_total_collision_stop_before_target",
            certified=bool(
                outcome_id != "unselected_total_collision_before_target"
                or total_collision_stop_certified
                or stop_certified
            ),
            source=type(stop_certificate).__name__,
            detail=(
                "missing=" + ",".join(stop_certificate.missing_obligations)
                if not total_collision_stop_certified
                else f"stop_time={stop_certificate.stop_time!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_target_atlas_or_total_collision_stop",
            certified=outcome_id != "proof_grade_obstruction",
            source="construct_finite_target_atlas_or_stop",
            detail=(
                f"outcome={outcome_id}; "
                f"obstructions={','.join(obstruction_obligations)}"
            ),
        ),
    )
    return FiniteTargetAtlasOrStopCertificate(
        input_domain_certificate=input_domain,
        target_time=target_time,
        total_collision_policy=policy,
        finite_time_classification=finite_time_classification,
        finite_atlas_certificate=finite_atlas_certificate,
        painleve_certificate=painleve,
        binary_regularization_certificate=binary_regularization,
        binary_isolation_certificate=binary_isolation,
        binary_accumulation_certificate=binary_accumulation,
        compact_collision_free_cover_certificate=compact_cover,
        outcome_id=outcome_id,
        validated_atlas=validated_atlas,
        stop_certificate=stop_certificate if total_collision_stop_certified else None,
        obstruction_obligations=obstruction_obligations,
        obligations=obligations,
    )


def construct_compact_interval_atlas_or_stop(
    masses: Any,
    positions: Any,
    velocities: Any,
    time_radius: float,
    *,
    total_collision_policy: str = "maximal_classical_stop",
    selector_policy_id: str | None = None,
    **solver_options: Any,
) -> CompactIntervalAtlasOrStopCertificate:
    """Construct two finite-target certificates covering ``[-R, R]``.

    The open-time theorem uses compact physical-time intervals rather than
    endpoint-regime classification.  This constructor makes that reduction
    explicit by certifying the future endpoint ``+R`` and past endpoint ``-R``
    with the same total-collision policy.
    """

    radius = float(time_radius)
    certify_explicit_total_collision_policy(
        total_collision_policy,
        selector_policy_id=selector_policy_id,
    )
    future_target: FiniteTargetAtlasOrStopCertificate | None = None
    past_target: FiniteTargetAtlasOrStopCertificate | None = None
    if np.isfinite(radius) and radius > 0.0:
        future_target = construct_finite_target_atlas_or_stop(
            masses,
            positions,
            velocities,
            radius,
            total_collision_policy=total_collision_policy,
            selector_policy_id=selector_policy_id,
            **solver_options,
        )
        past_target = construct_finite_target_atlas_or_stop(
            masses,
            positions,
            velocities,
            -radius,
            total_collision_policy=total_collision_policy,
            selector_policy_id=selector_policy_id,
            **solver_options,
        )

    return certify_compact_interval_atlas_or_stop_from_finite_targets(
        past_target,
        future_target,
        radius,
        total_collision_policy=total_collision_policy,
        selector_policy_id=selector_policy_id,
    )


def certify_compact_interval_atlas_or_stop_from_finite_targets(
    past_target: FiniteTargetAtlasOrStopCertificate | None,
    future_target: FiniteTargetAtlasOrStopCertificate | None,
    time_radius: float,
    *,
    total_collision_policy: str = "maximal_classical_stop",
    selector_policy_id: str | None = None,
) -> CompactIntervalAtlasOrStopCertificate:
    """Certify a compact-interval response from two finite-target responses."""

    radius = float(time_radius)
    interval_lower = -radius
    interval_upper = radius
    policy = certify_explicit_total_collision_policy(
        total_collision_policy,
        selector_policy_id=selector_policy_id,
    )
    certificates = tuple(
        certificate
        for certificate in (past_target, future_target)
        if isinstance(certificate, FiniteTargetAtlasOrStopCertificate)
    )
    input_domain = (
        future_target.input_domain_certificate
        if isinstance(future_target, FiniteTargetAtlasOrStopCertificate)
        else past_target.input_domain_certificate
        if isinstance(past_target, FiniteTargetAtlasOrStopCertificate)
        else None
    )
    endpoint_base_certified = bool(
        _finite_target_atlas_or_stop_certified(past_target)
        and _finite_target_atlas_or_stop_certified(future_target)
    )
    endpoint_policy_consistent = _compact_interval_endpoint_policy_consistent(
        past_target,
        future_target,
        policy,
    )
    endpoint_input_consistent = _compact_interval_endpoint_input_domains_consistent(
        past_target,
        future_target,
    )
    endpoint_target_times_consistent = _compact_interval_endpoint_targets_match_radius(
        past_target,
        future_target,
        radius=radius,
    )
    endpoint_certified = bool(
        endpoint_base_certified
        and endpoint_policy_consistent
        and endpoint_input_consistent
        and endpoint_target_times_consistent
    )
    endpoint_outcomes = tuple(certificate.outcome_id for certificate in certificates)
    stop_certificate = _compact_interval_stop_certificate(past_target, future_target)
    if endpoint_certified and all(
        outcome == "finite_atlas_reaches_target" for outcome in endpoint_outcomes
    ):
        outcome_id = "compact_interval_atlas_reaches_both_endpoints"
    elif endpoint_certified and any(
        outcome == "unselected_total_collision_before_target"
        for outcome in endpoint_outcomes
    ):
        outcome_id = "compact_interval_total_collision_stop"
    elif endpoint_certified and any(
        outcome == "selected_total_collision_continuation"
        for outcome in endpoint_outcomes
    ):
        outcome_id = "compact_interval_selected_total_collision_continuation"
    else:
        outcome_id = "proof_grade_obstruction"
    obstruction_obligations = _compact_interval_obstruction_obligations(
        past_target,
        future_target,
        radius=radius,
        total_collision_policy=policy,
        outcome_id=outcome_id,
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="positive_time_radius_compact_interval",
            certified=bool(np.isfinite(radius) and radius > 0.0),
            source="construct_compact_interval_atlas_or_stop",
            detail=f"time_radius={radius!r}",
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_contains_initial_time",
            certified=bool(
                np.isfinite(interval_lower)
                and np.isfinite(interval_upper)
                and interval_lower < 0.0 < interval_upper
            ),
            source="construct_compact_interval_atlas_or_stop",
            detail=f"interval=[{interval_lower!r}, {interval_upper!r}]",
        ),
        TheoremPipelineObligation(
            obligation="explicit_total_collision_policy",
            certified=policy.certified,
            source=type(policy).__name__,
            detail=policy.policy_id,
        ),
        TheoremPipelineObligation(
            obligation="past_finite_target_atlas_or_stop_theorem",
            certified=_finite_target_atlas_or_stop_certified(past_target),
            source=type(past_target).__name__ if past_target is not None else "missing",
            detail=(
                past_target.outcome_id
                if past_target is not None
                else "past endpoint certificate was not constructed"
            ),
        ),
        TheoremPipelineObligation(
            obligation="future_finite_target_atlas_or_stop_theorem",
            certified=_finite_target_atlas_or_stop_certified(future_target),
            source=(
                type(future_target).__name__ if future_target is not None else "missing"
            ),
            detail=(
                future_target.outcome_id
                if future_target is not None
                else "future endpoint certificate was not constructed"
            ),
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_endpoint_policy_consistency",
            certified=endpoint_policy_consistent,
            source="certify_compact_interval_atlas_or_stop_from_finite_targets",
            detail=f"policy={policy.policy_id}",
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_endpoint_input_domain_consistency",
            certified=endpoint_input_consistent,
            source="certify_compact_interval_atlas_or_stop_from_finite_targets",
            detail="past and future finite-target certificates bind to the same masses and initial state",
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_endpoint_target_time_consistency",
            certified=endpoint_target_times_consistent,
            source="certify_compact_interval_atlas_or_stop_from_finite_targets",
            detail=f"expected targets=({-radius!r}, {radius!r})",
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_total_collision_stop_certificate",
            certified=(
                outcome_id != "compact_interval_total_collision_stop"
                or _total_collision_stop_certified(stop_certificate)
            ),
            source=type(stop_certificate).__name__ if stop_certificate is not None else "missing",
            detail=(
                f"stop_time={getattr(stop_certificate, 'stop_time', None)!r}"
                if stop_certificate is not None
                else "compact interval stop needs a certified endpoint stop certificate"
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_target_theorem_covers_compact_interval_endpoints",
            certified=endpoint_certified,
            source="construct_compact_interval_atlas_or_stop",
            detail=f"endpoint_outcomes={endpoint_outcomes!r}",
        ),
    )
    return CompactIntervalAtlasOrStopCertificate(
        input_domain_certificate=input_domain,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        total_collision_policy=policy,
        past_target_certificate=past_target,
        future_target_certificate=future_target,
        stop_certificate=stop_certificate,
        outcome_id=outcome_id,
        obstruction_obligations=obstruction_obligations,
        obligations=obligations,
    )


def construct_compact_interval_exhaustion_family(
    masses: Any,
    positions: Any,
    velocities: Any,
    base_time_radius: float,
    *,
    prefix_count: int = 1,
    compact_time_rate: float = 1.0,
    total_collision_policy: str = "maximal_classical_stop",
    selector_policy_id: str | None = None,
    **solver_options: Any,
) -> CompactIntervalExhaustionFamilyCertificate:
    """Construct a checked prefix of the countable compact exhaustion.

    The infinite family is the explicit nested schedule ``K_n=[-nR,nR]``.
    The finite prefix is not a substitute for the analytic theorem; it binds
    the formula to actual finite-target constructors for the supplied data.
    """

    base_radius = float(base_time_radius)
    prefix_count = int(prefix_count)
    compact_time = certify_compact_time_real_line_coverage(compact_time_rate)
    prefix_certificates: list[CompactIntervalAtlasOrStopCertificate] = []
    if np.isfinite(base_radius) and base_radius > 0.0 and prefix_count > 0:
        for index in range(prefix_count):
            prefix_certificates.append(
                construct_compact_interval_atlas_or_stop(
                    masses,
                    positions,
                    velocities,
                    base_radius * float(index + 1),
                    total_collision_policy=total_collision_policy,
                    selector_policy_id=selector_policy_id,
                    **solver_options,
                )
            )
    prefix_tuple = tuple(prefix_certificates)
    finite_target_reduction = (
        certify_finite_target_theorem_reduces_compact_interval_exhaustion()
    )
    local_finiteness = certify_countable_nested_compact_interval_local_finiteness()
    obligations = (
        TheoremPipelineObligation(
            obligation="compact_time_real_line_coverage",
            certified=bool(compact_time.certified),
            source=type(compact_time).__name__,
            detail="compact time remains the ambient open-time coordinate",
        ),
        TheoremPipelineObligation(
            obligation="positive_base_time_radius",
            certified=bool(np.isfinite(base_radius) and base_radius > 0.0),
            source="construct_compact_interval_exhaustion_family",
            detail=f"base_time_radius={base_radius!r}",
        ),
        TheoremPipelineObligation(
            obligation="positive_exhaustion_prefix_count",
            certified=prefix_count > 0,
            source="construct_compact_interval_exhaustion_family",
            detail=f"prefix_count={prefix_count!r}",
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_prefix_constructed",
            certified=len(prefix_tuple) == prefix_count and prefix_count > 0,
            source="construct_compact_interval_exhaustion_family",
            detail=f"constructed={len(prefix_tuple)}",
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_prefix_certified",
            certified=bool(
                prefix_tuple
                and all(certificate.certified for certificate in prefix_tuple)
            ),
            source="construct_compact_interval_exhaustion_family",
            detail=",".join(certificate.outcome_id for certificate in prefix_tuple),
        ),
        TheoremPipelineObligation(
            obligation="linear_radius_schedule_prefix",
            certified=_prefix_radii_follow_linear_schedule(
                prefix_tuple,
                base_radius=base_radius,
            ),
            source="construct_compact_interval_exhaustion_family",
            detail="K_n=[-nR,nR]",
        ),
        TheoremPipelineObligation(
            obligation="nested_compact_interval_prefix",
            certified=_prefix_intervals_are_nested(prefix_tuple),
            source="construct_compact_interval_exhaustion_family",
            detail="K_n subset K_{n+1} on the checked prefix",
        ),
        TheoremPipelineObligation(
            obligation="finite_target_theorem_reduces_compact_interval_exhaustion",
            certified=finite_target_reduction.certified,
            source=finite_target_reduction.source,
            detail=finite_target_reduction.statement,
        ),
        TheoremPipelineObligation(
            obligation="countable_nested_compact_interval_local_finiteness",
            certified=local_finiteness.certified,
            source=local_finiteness.source,
            detail=local_finiteness.statement,
        ),
    )
    return CompactIntervalExhaustionFamilyCertificate(
        compact_time_certificate=compact_time,
        base_time_radius=base_radius,
        prefix_count=prefix_count,
        prefix_certificates=prefix_tuple,
        exhaustion_formula="K_n=[-nR,nR], n=1,2,...",
        finite_target_reduction_certificate=finite_target_reduction,
        local_finiteness_certificate=local_finiteness,
        obligations=obligations,
    )


def construct_open_time_locally_finite_atlas_theorem(
    masses: Any,
    positions: Any,
    velocities: Any,
    target_time: float,
    *,
    compact_time_rate: float = 1.0,
    exhaustion_prefix_count: int = 1,
    total_collision_policy: str = "maximal_classical_stop",
    selector_policy_id: str | None = None,
    **solver_options: Any,
) -> OpenTimeLocallyFiniteAtlasTheoremCertificate:
    """Build the open-time theorem around compact interval certificates."""

    checked_prefix_strategy = solver_options.pop("checked_prefix_strategy", None)
    checked_prefix_order = int(
        solver_options.pop(
            "checked_prefix_order",
            solver_options.get("order", 10),
        )
    )
    checked_prefix_coefficient_tolerance = float(
        solver_options.pop("checked_prefix_coefficient_tolerance", 1.0e-11)
    )
    checked_prefix_residual_tolerance = float(
        solver_options.pop("checked_prefix_residual_tolerance", 1.0e-8)
    )
    checked_prefix_regularized_residual_tolerance = float(
        solver_options.pop(
            "checked_prefix_regularized_residual_tolerance",
            1.0e-8,
        )
    )
    checked_prefix_projected_residual_tolerance = float(
        solver_options.pop(
            "checked_prefix_projected_residual_tolerance",
            2.0e-1,
        )
    )
    checked_prefix_constraint_tolerance = float(
        solver_options.pop("checked_prefix_constraint_tolerance", 1.0e-8)
    )
    checked_prefix_tail_bound = float(
        solver_options.pop("checked_prefix_tail_bound", 1.0e-9)
    )
    checked_prefix_sample_count = int(
        solver_options.pop("checked_prefix_sample_count", 7)
    )
    checked_prefix_physical_time_tolerance = float(
        solver_options.pop("checked_prefix_physical_time_tolerance", 1.0e-10)
    )
    checked_prefix_position_tolerance = float(
        solver_options.pop("checked_prefix_position_tolerance", 1.0e-8)
    )
    checked_prefix_velocity_tolerance = float(
        solver_options.pop("checked_prefix_velocity_tolerance", 1.0e-8)
    )
    checked_prefix_validated_atlas = solver_options.pop(
        "checked_prefix_validated_atlas",
        None,
    )
    search_recursive_branch_refinement_certificate = solver_options.pop(
        "certificate_search_recursive_branch_refinement_certificate",
        None,
    )
    search_event_order_refinement_certificate = solver_options.pop(
        "certificate_search_event_order_refinement_certificate",
        None,
    )
    search_stratified_branch_tree_certificate = solver_options.pop(
        "certificate_search_stratified_branch_tree_certificate",
        None,
    )
    search_stratified_event_order_tree_certificate = solver_options.pop(
        "certificate_search_stratified_event_order_tree_certificate",
        None,
    )
    search_recursive_stratified_branch_consumption_certificate = solver_options.pop(
        "certificate_search_recursive_stratified_branch_consumption_certificate",
        None,
    )
    search_recursive_stratified_event_order_consumption_certificate = (
        solver_options.pop(
            "certificate_search_recursive_stratified_event_order_consumption_certificate",
            None,
        )
    )
    search_set_valued_constructor_completeness_certificate = solver_options.pop(
        "certificate_search_set_valued_constructor_completeness_certificate",
        None,
    )
    search_constructor_certificate = solver_options.pop(
        "certificate_search_constructor_certificate",
        None,
    )
    search_branch_constructor_certificate = solver_options.pop(
        "certificate_search_branch_constructor_certificate",
        None,
    )
    search_event_order_constructor_certificate = solver_options.pop(
        "certificate_search_event_order_constructor_certificate",
        None,
    )
    search_constructor_root_dimension = solver_options.pop(
        "certificate_search_constructor_root_dimension",
        None,
    )
    search_constructor_root_rank = solver_options.pop(
        "certificate_search_constructor_root_rank",
        None,
    )
    search_branch_constructor_root_dimension = solver_options.pop(
        "certificate_search_branch_constructor_root_dimension",
        search_constructor_root_dimension,
    )
    search_branch_constructor_root_rank = solver_options.pop(
        "certificate_search_branch_constructor_root_rank",
        search_constructor_root_rank,
    )
    search_event_order_constructor_root_dimension = solver_options.pop(
        "certificate_search_event_order_constructor_root_dimension",
        search_constructor_root_dimension,
    )
    search_event_order_constructor_root_rank = solver_options.pop(
        "certificate_search_event_order_constructor_root_rank",
        search_constructor_root_rank,
    )
    exhaustion_family = construct_compact_interval_exhaustion_family(
        masses,
        positions,
        velocities,
        abs(float(target_time)),
        prefix_count=exhaustion_prefix_count,
        compact_time_rate=compact_time_rate,
        total_collision_policy=total_collision_policy,
        selector_policy_id=selector_policy_id,
        **solver_options,
    )
    compact_interval = (
        exhaustion_family.prefix_certificates[0]
        if exhaustion_family.prefix_certificates
        else None
    )
    finite_target = (
        compact_interval.future_target_certificate
        if compact_interval is not None
        and compact_interval.future_target_certificate is not None
        else compact_interval.past_target_certificate
        if compact_interval is not None
        else None
    )
    if finite_target is None:
        finite_target = construct_finite_target_atlas_or_stop(
            masses,
            positions,
            velocities,
            target_time,
            total_collision_policy=total_collision_policy,
            selector_policy_id=selector_policy_id,
            **solver_options,
        )
    compact_time = certify_compact_time_real_line_coverage(compact_time_rate)
    exhaustion = certify_countable_compact_time_exhaustion(
        compact_time_certificate=compact_time,
        finite_target_certificate=finite_target,
        compact_interval_certificate=compact_interval,
        exhaustion_family_certificate=exhaustion_family,
    )
    if search_set_valued_constructor_completeness_certificate is None:
        input_dimension = int(
            getattr(
                finite_target.input_domain_certificate,
                "dimension",
                3,
            )
        )
        constructor_theorem = certify_finite_target_completeness_theorem(
            dimension=input_dimension,
            total_collision_policy_id=finite_target.total_collision_policy.policy_id,
        )
        if (
            search_branch_constructor_certificate is not None
            or search_event_order_constructor_certificate is not None
        ):
            branch_constructor = (
                search_branch_constructor_certificate
                if search_branch_constructor_certificate is not None
                else search_constructor_certificate
            )
            event_constructor = (
                search_event_order_constructor_certificate
                if search_event_order_constructor_certificate is not None
                else search_constructor_certificate
            )
            if branch_constructor is None or event_constructor is None:
                raise ValueError(
                    "constructor-derived certificate search requires either "
                    "certificate_search_constructor_certificate or both branch "
                    "and event-order constructor certificates"
                )
            search_set_valued_constructor_completeness_certificate = (
                certify_constructor_pair_derived_recursive_stratified_set_valued_constructor_completeness(
                    constructor_theorem,
                    branch_constructor_certificate=branch_constructor,
                    event_order_constructor_certificate=event_constructor,
                    branch_root_dimension=search_branch_constructor_root_dimension,
                    branch_root_rank=search_branch_constructor_root_rank,
                    event_order_root_dimension=search_event_order_constructor_root_dimension,
                    event_order_root_rank=search_event_order_constructor_root_rank,
                )
            )
        elif search_constructor_certificate is not None:
            search_set_valued_constructor_completeness_certificate = (
                certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness(
                    constructor_theorem,
                    constructor_certificate=search_constructor_certificate,
                    root_dimension=search_constructor_root_dimension,
                    root_rank=search_constructor_root_rank,
                )
            )
    if search_set_valued_constructor_completeness_certificate is not None:
        scoped_set_valued_certificate = _scoped_set_valued_constructor_certificate(
            search_set_valued_constructor_completeness_certificate,
        )
        if search_recursive_stratified_branch_consumption_certificate is None:
            search_recursive_stratified_branch_consumption_certificate = getattr(
                scoped_set_valued_certificate,
                "branch_consumption_certificate",
                None,
            )
        if search_recursive_stratified_event_order_consumption_certificate is None:
            search_recursive_stratified_event_order_consumption_certificate = getattr(
                scoped_set_valued_certificate,
                "event_order_consumption_certificate",
                None,
            )
    finite_target_completeness = certify_finite_target_completeness_reduction(
        finite_target_certificate=finite_target,
        compact_interval_certificate=compact_interval,
        exhaustion_family_certificate=exhaustion_family,
        recursive_branch_refinement_certificate=(
            search_recursive_branch_refinement_certificate
        ),
        event_order_refinement_certificate=(
            search_event_order_refinement_certificate
        ),
        stratified_branch_tree_certificate=(
            search_stratified_branch_tree_certificate
        ),
        stratified_event_order_tree_certificate=(
            search_stratified_event_order_tree_certificate
        ),
        recursive_stratified_branch_consumption_certificate=(
            search_recursive_stratified_branch_consumption_certificate
        ),
        recursive_stratified_event_order_consumption_certificate=(
            search_recursive_stratified_event_order_consumption_certificate
        ),
        set_valued_constructor_completeness_certificate=(
            search_set_valued_constructor_completeness_certificate
        ),
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="finite_target_atlas_or_stop_theorem",
            certified=finite_target.certified,
            source=type(finite_target).__name__,
            detail=finite_target.outcome_id,
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_atlas_or_stop_theorem",
            certified=bool(compact_interval is not None and compact_interval.certified),
            source=(
                type(compact_interval).__name__
                if compact_interval is not None
                else "missing"
            ),
            detail=(
                compact_interval.outcome_id
                if compact_interval is not None
                else "compact interval certificate unavailable"
            ),
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_exhaustion_family",
            certified=exhaustion_family.certified,
            source=type(exhaustion_family).__name__,
            detail=exhaustion_family.exhaustion_formula,
        ),
        TheoremPipelineObligation(
            obligation="countable_compact_time_exhaustion",
            certified=exhaustion.certified,
            source=type(exhaustion).__name__,
            detail=exhaustion.exhaustion_id,
        ),
        TheoremPipelineObligation(
            obligation="endpoint_regime_partition_not_required",
            certified=True,
            source="open_time_locally_finite_atlas_theorem",
            detail="endpoint regimes are optional compression certificates",
        ),
    )
    theorem = OpenTimeLocallyFiniteAtlasTheoremCertificate(
        finite_target_certificate=finite_target,
        compact_interval_certificate=compact_interval,
        exhaustion_family_certificate=exhaustion_family,
        countable_exhaustion_certificate=exhaustion,
        endpoint_regime_partition_required=False,
        obligations=obligations,
        finite_target_completeness_certificate=finite_target_completeness,
    )
    if checked_prefix_strategy in (None, False):
        return theorem
    if str(checked_prefix_strategy) in {
        "finite_target_atlas",
        "validated_atlas",
        "planar_hybrid",
        "supplied_validated_atlas",
    }:
        checked_finite_target = finite_target
        if checked_prefix_validated_atlas is not None:
            checked_finite_target = (
                certify_finite_target_atlas_or_stop_from_validated_atlas(
                    masses,
                    positions,
                    velocities,
                    target_time,
                    checked_prefix_validated_atlas,
                    total_collision_policy=total_collision_policy,
                    selector_policy_id=selector_policy_id,
                )
            )
        elif str(checked_prefix_strategy) == "supplied_validated_atlas":
            raise ValueError(
                "checked_prefix_strategy='supplied_validated_atlas' requires "
                "checked_prefix_validated_atlas"
            )
        verifier = construct_independent_finite_target_checked_atlas(
            checked_finite_target,
            coefficient_tolerance=checked_prefix_coefficient_tolerance,
            ordinary_residual_tolerance=checked_prefix_residual_tolerance,
            regularized_residual_tolerance=(
                checked_prefix_regularized_residual_tolerance
            ),
            projected_residual_tolerance=(
                checked_prefix_projected_residual_tolerance
            ),
            constraint_tolerance=checked_prefix_constraint_tolerance,
            physical_time_tolerance=checked_prefix_physical_time_tolerance,
            position_tolerance=checked_prefix_position_tolerance,
            velocity_tolerance=checked_prefix_velocity_tolerance,
            sample_count=checked_prefix_sample_count,
        )
        return attach_independent_chart_verifier(theorem, verifier)
    if str(checked_prefix_strategy) not in {"ordinary_taylor", "ordinary"}:
        raise ValueError(
            "checked_prefix_strategy currently supports 'ordinary_taylor' "
            "'finite_target_atlas', or 'supplied_validated_atlas'"
        )
    verifier = construct_independent_ordinary_taylor_checked_prefix(
        masses,
        positions,
        velocities,
        target_time,
        order=checked_prefix_order,
        coefficient_tolerance=checked_prefix_coefficient_tolerance,
        residual_tolerance=checked_prefix_residual_tolerance,
        tail_bound=checked_prefix_tail_bound,
        sample_count=checked_prefix_sample_count,
    )
    return attach_independent_chart_verifier(theorem, verifier)


def _float_interval_pair(interval: Any) -> tuple[float, float]:
    if hasattr(interval, "lower") and hasattr(interval, "upper"):
        return (float(interval.lower), float(interval.upper))
    lower, upper = interval
    return (float(lower), float(upper))


def _union_float_interval_pair(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    return (
        min(float(left[0]), float(right[0])),
        max(float(left[1]), float(right[1])),
    )


def _spatial_ks_chart_certificate_with_interval_time_enclosure(
    solution: object,
    *,
    physical_time_interval: Any,
    **kwargs: Any,
) -> object:
    """Serialize a KS chart with metadata and checker enclosures reconciled."""

    provisional = spatial_ks_binary_chart_certificate_from_solution(
        solution,
        **kwargs,
    )
    return replace(
        provisional,
        physical_time_interval=_union_float_interval_pair(
            _float_interval_pair(physical_time_interval),
            _float_interval_pair(provisional.physical_time_interval),
        ),
    )


def _bind_independent_verifier_to_validated_atlas(
    verifier: IndependentChartVerifierCertificate,
    validated_atlas: object,
) -> IndependentChartVerifierCertificate:
    return replace(
        verifier,
        atlas_binding_token=finite_time_selector_trace_binding_token(validated_atlas),
    )


def construct_independent_finite_target_checked_atlas(
    finite_target_certificate: FiniteTargetAtlasOrStopCertificate,
    *,
    coefficient_tolerance: float = 1.0e-6,
    ordinary_residual_tolerance: float = 1.0e-6,
    regularized_residual_tolerance: float = 1.0e-8,
    projected_residual_tolerance: float = 2.0e-1,
    constraint_tolerance: float = 1.0e-8,
    physical_time_tolerance: float = 1.0e-10,
    position_tolerance: float = 1.0e-8,
    velocity_tolerance: float = 1.0e-8,
    sample_count: int = 7,
) -> object:
    """Serialize and independently check the finite-target atlas response."""

    atlas = getattr(finite_target_certificate, "validated_atlas", None)
    if atlas is None or getattr(atlas, "proof_certified", False) is not True:
        raise ValueError("finite target certificate does not carry a proof-certified atlas")
    return construct_independent_validated_atlas_checked_chain(
        atlas,
        certificate_id_prefix="finite-target-atlas",
        coefficient_tolerance=coefficient_tolerance,
        ordinary_residual_tolerance=ordinary_residual_tolerance,
        regularized_residual_tolerance=regularized_residual_tolerance,
        projected_residual_tolerance=projected_residual_tolerance,
        constraint_tolerance=constraint_tolerance,
        physical_time_tolerance=physical_time_tolerance,
        position_tolerance=position_tolerance,
        velocity_tolerance=velocity_tolerance,
        sample_count=sample_count,
    )


def construct_independent_validated_atlas_checked_chain(
    validated_atlas: object,
    *,
    certificate_id_prefix: str = "validated-atlas",
    coefficient_tolerance: float = 1.0e-6,
    ordinary_residual_tolerance: float = 1.0e-6,
    regularized_residual_tolerance: float = 1.0e-8,
    projected_residual_tolerance: float = 2.0e-1,
    constraint_tolerance: float = 1.0e-8,
    physical_time_tolerance: float = 1.0e-10,
    position_tolerance: float = 1.0e-8,
    velocity_tolerance: float = 1.0e-8,
    sample_count: int = 7,
) -> object:
    """Run the independent checker on a supported ValidatedAtlasSolution.

    This intentionally starts with the planar hybrid route, where the atlas
    already has ordinary/Levi-Civita chart semantics and the checker has a
    finite chart-chain grammar.  Unsupported chart families raise instead of
    returning placeholder verifier evidence.
    """

    evaluation = getattr(validated_atlas, "evaluation", None)
    atlas_charts = tuple(getattr(validated_atlas, "charts", ()))
    atlas_transitions = tuple(getattr(validated_atlas, "transitions", ()))
    if (
        len(atlas_charts) == 1
        and not atlas_transitions
        and str(getattr(atlas_charts[0], "chart_type", "")) == "spatial_ks_binary"
        and getattr(evaluation, "ks_solution", None) is not None
        and getattr(evaluation, "ordinary_solution", None) is None
    ):
        chart_meta = atlas_charts[0]
        chart_id = str(getattr(chart_meta, "chart_id", "spatial_ks_0"))
        ks_solution = _spatial_ks_checker_representative_solution(
            getattr(evaluation, "ks_solution"),
        )
        chart = _spatial_ks_chart_certificate_with_interval_time_enclosure(
            ks_solution,
            certificate_id=f"{certificate_id_prefix}-spatial-ks-chart-0",
            chart_id=chart_id,
            parameter_interval=_float_interval_pair(chart_meta.parameter_interval),
            physical_time_interval=_float_interval_pair(
                chart_meta.physical_time_interval,
            ),
            coefficient_tolerance=coefficient_tolerance,
            regularized_residual_tolerance=regularized_residual_tolerance,
            projected_residual_tolerance=projected_residual_tolerance,
            constraint_tolerance=constraint_tolerance,
            tail_bound=float(getattr(chart_meta, "tail_bound", 0.0)),
            sample_count=sample_count,
            projection_rho_lower_bound=0.0,
            source="finite_target_spatial_ks_validated_atlas_independent_serialization",
        )
        target_value = getattr(validated_atlas, "target_time", None)
        target = np.inf if target_value is None else float(target_value)
        target_interval = (
            (target, target)
            if np.isfinite(target)
            else _float_interval_pair(chart_meta.physical_time_interval)
        )
        chain = ChartChainCertificate(
            certificate_id=f"{certificate_id_prefix}-spatial-ks-chain-0",
            chain_id=f"{certificate_id_prefix}-spatial-ks-chain",
            chain_type="regularized_atlas_chart_chain",
            chart_ids=(chart.chart_id,),
            transition_ids=(),
            target_physical_time_interval=target_interval,
        )
        return _bind_independent_verifier_to_validated_atlas(
            verify_chart_certificates(
                (chart,),
                chart_chains=(chain,),
            ),
            validated_atlas,
        )

    if _spatial_ordinary_ks_handoff_supported_for_checker(
        evaluation,
        atlas_charts,
        atlas_transitions,
    ):
        return _bind_independent_verifier_to_validated_atlas(
            _construct_independent_spatial_ordinary_ks_checked_chain(
                validated_atlas,
                certificate_id_prefix=certificate_id_prefix,
                coefficient_tolerance=coefficient_tolerance,
                ordinary_residual_tolerance=ordinary_residual_tolerance,
                regularized_residual_tolerance=regularized_residual_tolerance,
                projected_residual_tolerance=projected_residual_tolerance,
                constraint_tolerance=constraint_tolerance,
                physical_time_tolerance=physical_time_tolerance,
                position_tolerance=position_tolerance,
                velocity_tolerance=velocity_tolerance,
                sample_count=sample_count,
            ),
            validated_atlas,
        )

    if _spatial_ks_competing_handoff_supported_for_checker(
        evaluation,
        atlas_charts,
        atlas_transitions,
    ):
        return _bind_independent_verifier_to_validated_atlas(
            _construct_independent_spatial_ks_competing_checked_chain(
                validated_atlas,
                certificate_id_prefix=certificate_id_prefix,
                coefficient_tolerance=coefficient_tolerance,
                regularized_residual_tolerance=regularized_residual_tolerance,
                projected_residual_tolerance=projected_residual_tolerance,
                constraint_tolerance=constraint_tolerance,
                physical_time_tolerance=physical_time_tolerance,
                position_tolerance=position_tolerance,
                velocity_tolerance=velocity_tolerance,
                sample_count=sample_count,
            ),
            validated_atlas,
        )

    if _homothetic_total_collision_supported_for_checker(
        evaluation,
        atlas_charts,
        atlas_transitions,
    ):
        chart_meta = atlas_charts[0]
        chart = _homothetic_total_collision_stop_chart_for_checker(
            evaluation,
            chart_meta,
            certificate_id=f"{certificate_id_prefix}-homothetic-total-stop-chart-0",
            residual_tolerance=regularized_residual_tolerance,
            projected_residual_tolerance=projected_residual_tolerance,
            angular_momentum_tolerance=regularized_residual_tolerance,
            sample_count=sample_count,
        )
        target_value = getattr(validated_atlas, "target_time", None)
        target = np.inf if target_value is None else float(target_value)
        target_interval = (
            (target, target)
            if np.isfinite(target)
            else _float_interval_pair(chart_meta.physical_time_interval)
        )
        chain = ChartChainCertificate(
            certificate_id=f"{certificate_id_prefix}-homothetic-total-stop-chain-0",
            chain_id=f"{certificate_id_prefix}-homothetic-total-stop-chain",
            chain_type="regularized_atlas_chart_chain",
            chart_ids=(chart.chart_id,),
            transition_ids=(),
            target_physical_time_interval=target_interval,
        )
        return _bind_independent_verifier_to_validated_atlas(
            verify_chart_certificates(
                (chart,),
                chart_chains=(chain,),
            ),
            validated_atlas,
        )

    branch_atlases = tuple(getattr(evaluation, "branch_atlases", ()) or ())
    if branch_atlases:
        return _bind_independent_verifier_to_validated_atlas(
            _construct_independent_branch_union_checked_atlas(
                validated_atlas,
                branch_atlases=branch_atlases,
                certificate_id_prefix=certificate_id_prefix,
                coefficient_tolerance=coefficient_tolerance,
                ordinary_residual_tolerance=ordinary_residual_tolerance,
                regularized_residual_tolerance=regularized_residual_tolerance,
                projected_residual_tolerance=projected_residual_tolerance,
                constraint_tolerance=constraint_tolerance,
                physical_time_tolerance=physical_time_tolerance,
                position_tolerance=position_tolerance,
                velocity_tolerance=velocity_tolerance,
                sample_count=sample_count,
            ),
            validated_atlas,
        )

    hybrid_solution = getattr(evaluation, "hybrid_solution", None)
    if hybrid_solution is None:
        raise ValueError(
            "independent validated-atlas checking currently supports only "
            "planar hybrid atlas evaluations and one-chart spatial KS "
            "target-inside-regularized-chart evaluations, plus supplied "
            "spatial ordinary/KS/ordinary handoff chains"
        )
    charts, transitions, chain = planar_hybrid_chart_chain_certificates_from_solution(
        hybrid_solution,
        certificate_id_prefix=certificate_id_prefix,
        coefficient_tolerance=coefficient_tolerance,
        ordinary_residual_tolerance=ordinary_residual_tolerance,
        regularized_residual_tolerance=regularized_residual_tolerance,
        projected_residual_tolerance=projected_residual_tolerance,
        physical_time_tolerance=physical_time_tolerance,
        position_tolerance=position_tolerance,
        velocity_tolerance=velocity_tolerance,
        sample_count=sample_count,
        source="finite_target_validated_atlas_independent_serialization",
    )
    return _bind_independent_verifier_to_validated_atlas(
        verify_chart_certificates(
            charts,
            transitions=transitions,
            chart_chains=(chain,),
        ),
        validated_atlas,
    )


_EXACT_HOMOTHETIC_REMAINDER_COMPONENT_EXPONENTS = (
    ("value", 2.0),
    ("first_jet", 1.0),
    ("lifted_residual", 1.0),
    ("physical_residual", 1.0),
    ("regularized_position_value", 2.0),
)


def _homothetic_total_collision_supported_for_checker(
    evaluation: object,
    atlas_charts: tuple[object, ...],
    atlas_transitions: tuple[object, ...],
) -> bool:
    if len(atlas_charts) != 1 or atlas_transitions:
        return False
    chart = atlas_charts[0]
    certificate = getattr(evaluation, "certificate", None)
    branch = getattr(evaluation, "branch", None)
    return bool(
        str(getattr(chart, "chart_type", ""))
        == "finite_jet_identity_selector_total_collision"
        and branch is not None
        and certificate is not None
        and getattr(certificate, "certified", False)
        and hasattr(branch, "quadratic_coefficient")
        and hasattr(branch, "masses")
        and (
            getattr(certificate, "parabolic_exact_tail_certified", False)
            or getattr(getattr(certificate, "scalar_majorant", None), "certified", False)
        )
    )


def _homothetic_total_collision_stop_chart_for_checker(
    evaluation: object,
    chart_meta: object,
    *,
    certificate_id: str,
    residual_tolerance: float,
    angular_momentum_tolerance: float,
    sample_count: int,
    projected_residual_tolerance: float | None = None,
):
    """Serialize an exact homothetic selector as a generalized stop chart.

    Exact parabolic branches get a zero remainder.  Nonzero-energy homothetic
    series use the scalar Rouche/Cauchy majorant already carried by the
    constructor; the serialized checker remains responsible for accepting or
    rejecting the resulting interval residual and tail obligations.
    """

    branch = getattr(evaluation, "branch", None)
    certificate = getattr(evaluation, "certificate", None)
    if branch is None or certificate is None:
        raise ValueError("homothetic checker bridge requires a validated branch")
    exact_tail = bool(getattr(certificate, "parabolic_exact_tail_certified", False))
    scalar_majorant = getattr(certificate, "scalar_majorant", None)
    if not exact_tail and not bool(getattr(scalar_majorant, "certified", False)):
        raise ValueError(
            "homothetic checker bridge requires exact tail or scalar majorant",
        )

    central_shape = np.asarray(getattr(branch, "quadratic_coefficient"), dtype=float)
    masses = np.asarray(getattr(branch, "masses"), dtype=float).reshape(-1)
    if central_shape.ndim != 2 or central_shape.shape[0] != 3:
        raise ValueError("homothetic central shape must have three body rows")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("homothetic masses must be positive three-body masses")

    tau_interval = _float_interval_pair(getattr(chart_meta, "parameter_interval", ()))
    if not (tau_interval[0] < 0.0 < tau_interval[1]):
        raise ValueError("homothetic total-collision chart must straddle tau=0")
    isolation_radius = float(max(abs(tau_interval[0]), abs(tau_interval[1])))
    if not np.isfinite(isolation_radius) or isolation_radius <= 0.0:
        raise ValueError("homothetic isolation radius must be positive")

    scalar_coefficients = np.asarray(getattr(branch, "coefficients"), dtype=float)
    max_total_degree = int(len(scalar_coefficients) - 1)
    if max_total_degree < 1:
        raise ValueError("homothetic branch must carry at least the scale row")
    scale_index = (1,)
    selected_coefficients = {
        scale_index: float(scalar_coefficients[1]) * central_shape,
    }
    generalized_branch = construct_fuchsian_shape_branch(
        masses=masses,
        central_shape=central_shape,
        powers=(2.0,),
        selected_coefficients=selected_coefficients,
        max_total_degree=max_total_degree,
        scale_index=scale_index,
    )
    central_floor = _minimum_pair_distance_for_checker(central_shape)
    shape_deviation = _generalized_fuchsian_shape_deviation_bound_for_bridge(
        generalized_branch,
        isolation_radius,
    )
    shape_floor = central_floor - 2.0 * np.sqrt(float(central_shape.shape[1])) * shape_deviation
    if shape_floor <= 0.0:
        raise ValueError("homothetic Fuchsian shape isolation is not positive")
    remainder_majorant = (
        _exact_zero_remainder_majorant_for_checker(initial_radius=isolation_radius)
        if exact_tail
        else _homothetic_scalar_remainder_majorant_for_checker(
            scalar_majorant,
            initial_radius=isolation_radius,
            retained_order=max_total_degree + 4,
        )
    )
    tail_bound = _generalized_remainder_tail_bound_for_bridge(remainder_majorant)
    return total_collision_generalized_fuchsian_stop_chart_certificate_from_branch(
        generalized_branch,
        certificate_id=str(certificate_id),
        chart_id=str(getattr(chart_meta, "chart_id", "homothetic_total_collision_0")),
        isolation_radius=isolation_radius,
        central_shape_pair_distance_floor=central_floor,
        shape_deviation_bound=shape_deviation,
        shape_pair_distance_floor=shape_floor,
        tau_interval=tau_interval,
        event_physical_time=float(getattr(certificate, "event_time", 0.0)),
        residual_tolerance=float(residual_tolerance),
        projected_residual_tolerance=projected_residual_tolerance,
        angular_momentum_tolerance=float(angular_momentum_tolerance),
        tail_bound=tail_bound,
        sample_count=int(sample_count),
        remainder_majorant=remainder_majorant,
        source=(
            "exact_homothetic_total_collision_independent_serialization"
            if exact_tail
            else "homothetic_energy_total_collision_independent_serialization"
        ),
    )


def _exact_zero_remainder_majorant_for_checker(
    *,
    initial_radius: float,
) -> GeneralizedFuchsianRemainderMajorantCertificate:
    shell_contraction = 0.5
    step_ratio = 0.25
    retained_order = 4
    inputs = []
    for component, exponent in _EXACT_HOMOTHETIC_REMAINDER_COMPONENT_EXPONENTS:
        primitive = PrimitiveCauchyTailInput(
            majorant_initial=0.0,
            majorant_growth=shell_contraction**float(exponent),
            step_ratio_bound=step_ratio,
            retained_order_initial=retained_order,
            retained_order_increment=1,
        )
        inputs.append((component, PrimitiveCauchyTailInputCertificate.from_input(primitive)))
    return GeneralizedFuchsianRemainderMajorantCertificate(
        initial_radius=float(initial_radius),
        shell_contraction=shell_contraction,
        analytic_disk_fraction=0.25,
        defect_bound=0.0,
        linear_inverse_bound=1.0,
        nonlinear_lipschitz_bound=0.0,
        remainder_ball_radius=0.0,
        component_effective_exponents=_EXACT_HOMOTHETIC_REMAINDER_COMPONENT_EXPONENTS,
        component_inputs=tuple(inputs),
    )


def _homothetic_scalar_remainder_majorant_for_checker(
    scalar_majorant: object,
    *,
    initial_radius: float,
    retained_order: int,
) -> GeneralizedFuchsianRemainderMajorantCertificate:
    shell_contraction = 0.5
    analytic_disk_fraction = 0.25
    step_ratio = 0.25
    remainder_ball_radius = float(getattr(scalar_majorant, "cauchy_majorant"))
    if not np.isfinite(remainder_ball_radius) or remainder_ball_radius <= 0.0:
        raise ValueError("homothetic scalar majorant must have positive Cauchy bound")
    inputs = []
    for component, exponent in _EXACT_HOMOTHETIC_REMAINDER_COMPONENT_EXPONENTS:
        radius_factor = initial_radius * (1.0 + analytic_disk_fraction)
        primitive = PrimitiveCauchyTailInput(
            majorant_initial=remainder_ball_radius * radius_factor ** float(exponent),
            majorant_growth=shell_contraction ** float(exponent),
            step_ratio_bound=step_ratio,
            retained_order_initial=int(retained_order),
            retained_order_increment=2,
        )
        inputs.append((component, PrimitiveCauchyTailInputCertificate.from_input(primitive)))
    return GeneralizedFuchsianRemainderMajorantCertificate(
        initial_radius=float(initial_radius),
        shell_contraction=shell_contraction,
        analytic_disk_fraction=analytic_disk_fraction,
        defect_bound=0.0,
        linear_inverse_bound=1.0,
        nonlinear_lipschitz_bound=0.0,
        remainder_ball_radius=remainder_ball_radius,
        component_effective_exponents=_EXACT_HOMOTHETIC_REMAINDER_COMPONENT_EXPONENTS,
        component_inputs=tuple(inputs),
    )


def _generalized_fuchsian_shape_deviation_bound_for_bridge(
    branch: object,
    radius: float,
) -> float:
    radius = float(radius)
    if radius <= 0.0:
        return float("inf")
    bound = 0.0
    for index, coefficient in getattr(branch, "coefficients", {}).items():
        if index == getattr(branch, "zero_index", None):
            continue
        exponent = float(branch.exponent(index))
        if not np.isfinite(exponent) or exponent <= 0.0:
            return float("inf")
        bound += float(np.linalg.norm(np.asarray(coefficient, dtype=float), ord=np.inf)) * radius**exponent
    return float(bound)


def _generalized_remainder_tail_bound_for_bridge(
    majorant: GeneralizedFuchsianRemainderMajorantCertificate,
) -> float:
    return float(
        max(
            float(input_.first_shell_tail_bound)
            for _component, input_ in majorant.component_inputs
        )
    )


def _minimum_pair_distance_for_checker(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    return float(
        min(
            np.linalg.norm(points[left] - points[right])
            for left in range(points.shape[0])
            for right in range(left + 1, points.shape[0])
        )
    )


def _construct_independent_branch_union_checked_atlas(
    validated_atlas: object,
    *,
    branch_atlases: tuple[object, ...],
    certificate_id_prefix: str,
    coefficient_tolerance: float,
    ordinary_residual_tolerance: float,
    regularized_residual_tolerance: float,
    projected_residual_tolerance: float,
    constraint_tolerance: float,
    physical_time_tolerance: float,
    position_tolerance: float,
    velocity_tolerance: float,
    sample_count: int,
) -> IndependentChartVerifierCertificate:
    """Serialize a finite branch-union atlas through checked leaf responses."""

    if not branch_atlases:
        raise ValueError("branch-union atlas does not contain member atlases")
    member_verifiers = tuple(
        construct_independent_validated_atlas_checked_chain(
            atlas,
            certificate_id_prefix=f"{certificate_id_prefix}-branch-{index}",
            coefficient_tolerance=coefficient_tolerance,
            ordinary_residual_tolerance=ordinary_residual_tolerance,
            regularized_residual_tolerance=regularized_residual_tolerance,
            projected_residual_tolerance=projected_residual_tolerance,
            constraint_tolerance=constraint_tolerance,
            physical_time_tolerance=physical_time_tolerance,
            position_tolerance=position_tolerance,
            velocity_tolerance=velocity_tolerance,
            sample_count=sample_count,
        )
        for index, atlas in enumerate(branch_atlases)
    )
    chart_results = tuple(
        result
        for verifier in member_verifiers
        for result in verifier.chart_results
    )
    transition_results = tuple(
        result
        for verifier in member_verifiers
        for result in verifier.transition_results
    )
    event_results = tuple(
        result
        for verifier in member_verifiers
        for result in verifier.event_results
    )
    nested_branch_union_results = tuple(
        result
        for verifier in member_verifiers
        for result in verifier.branch_union_results
    )
    chart_chain_results = tuple(
        result
        for verifier in member_verifiers
        for result in verifier.chart_chain_results
    )
    leaf_response_ids = tuple(
        _checked_leaf_response_identifier(verifier)
        for verifier in member_verifiers
    )
    if any(not response_id for response_id in leaf_response_ids):
        raise ValueError("branch-union member did not expose a checked response id")
    leaf_target_intervals = tuple(
        _validated_atlas_target_time_interval(atlas)
        for atlas in branch_atlases
    )
    aggregate_target_interval = _branch_union_aggregate_target_interval(
        validated_atlas,
        leaf_target_intervals,
    )
    branch_union = BranchUnionCertificate(
        certificate_id=f"{certificate_id_prefix}-branch-union-certificate",
        union_id=f"{certificate_id_prefix}-branch-union",
        union_type=_validated_atlas_branch_union_type(validated_atlas),
        leaf_response_certificate_ids=leaf_response_ids,
        leaf_target_intervals=leaf_target_intervals,
        aggregate_target_interval=aggregate_target_interval,
        leaf_kinds=_branch_union_leaf_kinds(
            getattr(getattr(validated_atlas, "evaluation", None), "branch_partition", None),
            len(branch_atlases),
        ),
        source="finite_target_validated_atlas_branch_union_independent_serialization",
    )
    branch_union_result = check_branch_union(
        branch_union,
        (*chart_results, *nested_branch_union_results, *chart_chain_results),
    )
    return IndependentChartVerifierCertificate(
        checker_id="independent_chart_verifier_v1",
        chart_results=chart_results,
        transition_results=transition_results,
        event_results=event_results,
        branch_union_results=(*nested_branch_union_results, branch_union_result),
        chart_chain_results=chart_chain_results,
    )


def _spatial_ks_checker_representative_solution(ks_solution: object) -> object:
    """Return a point KS Taylor chart suitable for independent checking.

    Interval-seeded KS Taylor charts carry set-valued coefficients.  The
    independent checker currently verifies point coefficient recurrences, so
    the serialized chart uses the midpoint KS initial data and recomputes the
    pair energy from the KS constraint before reconstructing a point recurrence.
    Point-seeded charts are returned unchanged.
    """

    if not _contains_interval_coefficients(getattr(ks_solution, "u", ())):
        return ks_solution
    masses = np.asarray(getattr(ks_solution, "masses"), dtype=float)
    pair = tuple(int(value) for value in getattr(ks_solution, "pair"))
    u0 = _coefficient_midpoint_array(getattr(ks_solution, "u")[0])
    u_velocity0 = _coefficient_midpoint_array(getattr(ks_solution, "u_velocity")[0])
    pair_energy0 = float(
        _coefficient_midpoint_array(getattr(ks_solution, "pair_energy"))[0]
    )
    rho0 = float(np.dot(u0, u0))
    if rho0 > 0.0:
        pair_mass = float(masses[pair[0]] + masses[pair[1]])
        pair_energy0 = (2.0 * float(np.dot(u_velocity0, u_velocity0)) - pair_mass) / rho0
    state = SpatialKSBinaryChartState(
        masses=masses,
        pair=pair,
        u=u0,
        u_velocity=u_velocity0,
        pair_energy=pair_energy0,
        binary_center=_coefficient_midpoint_array(
            getattr(ks_solution, "binary_center")[0],
        ),
        binary_center_velocity=_coefficient_midpoint_array(
            getattr(ks_solution, "binary_center_velocity")[0],
        ),
        third_offset=_coefficient_midpoint_array(
            getattr(ks_solution, "third_offset")[0],
        ),
        third_offset_velocity=_coefficient_midpoint_array(
            getattr(ks_solution, "third_offset_velocity")[0],
        ),
    )
    return construct_spatial_ks_binary_taylor_solution(
        state,
        order=int(getattr(ks_solution, "order")),
    )


def _contains_interval_coefficients(values: object) -> bool:
    array = np.asarray(values, dtype=object)
    return any(
        hasattr(value, "lower") and hasattr(value, "upper")
        for value in array.reshape(-1)
    )


def _coefficient_midpoint_array(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    midpoint_values = [
        0.5 * (float(value.lower) + float(value.upper))
        if hasattr(value, "lower") and hasattr(value, "upper")
        else float(value)
        for value in array.reshape(-1)
    ]
    return np.asarray(midpoint_values, dtype=float).reshape(array.shape)


def _checked_leaf_response_identifier(verifier: object) -> str:
    chart_chain_results = tuple(getattr(verifier, "chart_chain_results", ()) or ())
    for result in chart_chain_results:
        if getattr(result, "certified", False) and getattr(result, "chain_id", ""):
            return str(result.chain_id)
    branch_union_results = tuple(getattr(verifier, "branch_union_results", ()) or ())
    for result in branch_union_results:
        if getattr(result, "certified", False) and getattr(result, "union_id", ""):
            return str(result.union_id)
    chart_results = tuple(getattr(verifier, "chart_results", ()) or ())
    for result in chart_results:
        if (
            getattr(result, "certified", False)
            and getattr(result, "certificate_id", "")
        ):
            return str(result.certificate_id)
    return ""


def _validated_atlas_target_time_interval(atlas: object) -> tuple[float, float]:
    evaluation = getattr(atlas, "evaluation", None)
    interval = getattr(evaluation, "target_time_interval", None)
    if interval is not None:
        return _float_interval_pair(interval)
    target_time = getattr(atlas, "target_time", np.inf)
    if np.isfinite(float(target_time)):
        value = float(target_time)
        return (value, value)
    charts = tuple(getattr(atlas, "charts", ()) or ())
    if charts:
        return _float_interval_pair(charts[-1].physical_time_interval)
    return (float("inf"), float("inf"))


def _branch_union_aggregate_target_interval(
    atlas: object,
    leaf_target_intervals: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    evaluation = getattr(atlas, "evaluation", None)
    interval = getattr(evaluation, "target_time_interval", None)
    if interval is not None:
        return _float_interval_pair(interval)
    if not leaf_target_intervals:
        return (float("inf"), float("inf"))
    return (
        float(min(interval[0] for interval in leaf_target_intervals)),
        float(max(interval[1] for interval in leaf_target_intervals)),
    )


def _validated_atlas_branch_union_type(atlas: object) -> str:
    partition = getattr(getattr(atlas, "evaluation", None), "branch_partition", None)
    if _partition_has_stratified_leaf_kinds(partition):
        return "finite_time_stratified_branch_union"
    chart_types = {
        str(getattr(chart, "chart_type", ""))
        for chart in tuple(getattr(atlas, "charts", ()) or ())
    }
    if "spatial_ks_event_order_branch_union" in chart_types:
        return "finite_time_event_order_branch_union"
    return "finite_time_branch_union"


def _partition_has_stratified_leaf_kinds(partition: object | None) -> bool:
    leaves = tuple(getattr(partition, "leaf_certificates", ()) or ())
    return bool(leaves and all(hasattr(leaf, "leaf_kind") for leaf in leaves))


def _branch_union_leaf_kinds(
    partition: object | None,
    fallback_count: int,
) -> tuple[str, ...]:
    leaves = tuple(getattr(partition, "branches", ()) or ())
    if not leaves:
        leaves = tuple(getattr(partition, "leaves", ()) or ())
    if not leaves:
        leaves = tuple(getattr(partition, "branch_leaves", ()) or ())
    if not leaves:
        leaves = tuple(getattr(partition, "leaf_certificates", ()) or ())
    kinds = tuple(
        str(
            getattr(
                leaf,
                "leaf_kind",
                getattr(
                    leaf,
                    "leaf_type",
                    getattr(leaf, "decision", type(leaf).__name__),
                ),
            )
        )
        for leaf in leaves
    )
    if len(kinds) == int(fallback_count):
        return kinds
    return tuple("certified_branch_leaf" for _ in range(int(fallback_count)))


def _spatial_ordinary_ks_handoff_supported_for_checker(
    evaluation: object,
    atlas_charts: tuple[object, ...],
    atlas_transitions: tuple[object, ...],
) -> bool:
    chart_types = tuple(str(getattr(chart, "chart_type", "")) for chart in atlas_charts)
    return bool(
        evaluation is not None
        and getattr(evaluation, "ordinary_entry_solution", None) is not None
        and getattr(evaluation, "ks_evaluation", None) is not None
        and getattr(evaluation.ks_evaluation, "ks_solution", None) is not None
        and len(atlas_transitions) == max(0, len(atlas_charts) - 1)
        and chart_types
        in {
            ("spatial_ordinary_taylor_before_ks", "spatial_ks_binary"),
            (
                "spatial_ordinary_taylor_before_ks",
                "spatial_ks_binary",
                "spatial_ordinary_taylor_after_ks",
            ),
        }
        and (
            len(atlas_charts) == 2
            or getattr(evaluation.ks_evaluation, "ordinary_solution", None) is not None
        )
    )


def _spatial_ks_competing_handoff_supported_for_checker(
    evaluation: object,
    atlas_charts: tuple[object, ...],
    atlas_transitions: tuple[object, ...],
) -> bool:
    chart_types = tuple(str(getattr(chart, "chart_type", "")) for chart in atlas_charts)
    transition_types = tuple(
        str(getattr(transition, "transition_type", ""))
        for transition in atlas_transitions
    )
    return bool(
        evaluation is not None
        and getattr(evaluation, "first_ks_solution", None) is not None
        and getattr(evaluation, "next_ks_evaluation", None) is not None
        and getattr(evaluation.next_ks_evaluation, "ks_solution", None) is not None
        and chart_types == ("spatial_ks_binary", "spatial_ks_binary")
        and transition_types == ("spatial_ks_to_ks_competing_binary_entry",)
    )


def _construct_independent_spatial_ks_competing_checked_chain(
    validated_atlas: object,
    *,
    certificate_id_prefix: str,
    coefficient_tolerance: float,
    regularized_residual_tolerance: float,
    projected_residual_tolerance: float,
    constraint_tolerance: float,
    physical_time_tolerance: float,
    position_tolerance: float,
    velocity_tolerance: float,
    sample_count: int,
) -> object:
    evaluation = getattr(validated_atlas, "evaluation", None)
    atlas_charts = tuple(getattr(validated_atlas, "charts", ()))
    first_solution = _spatial_ks_checker_representative_solution(
        evaluation.first_ks_solution,
    )
    next_solution = _spatial_ks_checker_representative_solution(
        evaluation.next_ks_evaluation.ks_solution,
    )
    first_meta, next_meta = atlas_charts
    entry_parameter = _event_root_value(evaluation.competing_entry_event_certificate)
    handoff_time = float(first_solution.physical_time_at(entry_parameter))
    first_chart = _spatial_ks_chart_certificate_with_interval_time_enclosure(
        first_solution,
        certificate_id=f"{certificate_id_prefix}-spatial-ks-chart-0",
        chart_id=str(getattr(first_meta, "chart_id", "spatial_ks_0")),
        parameter_interval=_float_interval_pair(first_meta.parameter_interval),
        physical_time_interval=_float_interval_pair(first_meta.physical_time_interval),
        coefficient_tolerance=coefficient_tolerance,
        regularized_residual_tolerance=regularized_residual_tolerance,
        projected_residual_tolerance=projected_residual_tolerance,
        constraint_tolerance=constraint_tolerance,
        tail_bound=float(getattr(first_meta, "tail_bound", 0.0)),
        sample_count=sample_count,
        projection_rho_lower_bound=0.0,
        source="finite_target_spatial_ks_competing_source_independent_serialization",
    )
    next_chart = _spatial_ks_chart_certificate_with_interval_time_enclosure(
        next_solution,
        certificate_id=f"{certificate_id_prefix}-spatial-ks-chart-1",
        chart_id=str(getattr(next_meta, "chart_id", "spatial_ks_1")),
        parameter_interval=_float_interval_pair(next_meta.parameter_interval),
        physical_time_interval=_float_interval_pair(next_meta.physical_time_interval),
        coefficient_tolerance=coefficient_tolerance,
        regularized_residual_tolerance=regularized_residual_tolerance,
        projected_residual_tolerance=projected_residual_tolerance,
        constraint_tolerance=constraint_tolerance,
        tail_bound=float(getattr(next_meta, "tail_bound", 0.0)),
        sample_count=sample_count,
        projection_rho_lower_bound=0.0,
        physical_time_shift=handoff_time,
        source="finite_target_spatial_ks_competing_target_independent_serialization",
    )
    transition = spatial_ks_transition_certificate(
        transition_id=f"{certificate_id_prefix}-ks-to-ks-transition-0",
        source_chart_id=first_chart.chart_id,
        target_chart_id=next_chart.chart_id,
        transition_type="spatial_ks_to_ks_competing_binary_entry",
        handoff_time=handoff_time,
        source_parameter=entry_parameter,
        target_parameter=0.0,
        position_tolerance=position_tolerance,
        velocity_tolerance=velocity_tolerance,
        physical_time_tolerance=physical_time_tolerance,
        source="finite_target_spatial_ks_to_ks_competing_transition_serialization",
    )
    target_value = getattr(validated_atlas, "target_time", None)
    target = np.inf if target_value is None else float(target_value)
    target_interval = (
        (target, target)
        if np.isfinite(target)
        else _float_interval_pair(next_meta.physical_time_interval)
    )
    chain = ChartChainCertificate(
        certificate_id=f"{certificate_id_prefix}-spatial-ks-competing-chain-0",
        chain_id=f"{certificate_id_prefix}-spatial-ks-competing-chain",
        chain_type="regularized_atlas_chart_chain",
        chart_ids=(first_chart.chart_id, next_chart.chart_id),
        transition_ids=(transition.transition_id,),
        target_physical_time_interval=target_interval,
    )
    return verify_chart_certificates(
        (first_chart, next_chart),
        transitions=(transition,),
        chart_chains=(chain,),
    )


def _construct_independent_spatial_ordinary_ks_checked_chain(
    validated_atlas: object,
    *,
    certificate_id_prefix: str,
    coefficient_tolerance: float,
    ordinary_residual_tolerance: float,
    regularized_residual_tolerance: float,
    projected_residual_tolerance: float,
    constraint_tolerance: float,
    physical_time_tolerance: float,
    position_tolerance: float,
    velocity_tolerance: float,
    sample_count: int,
) -> object:
    evaluation = getattr(validated_atlas, "evaluation", None)
    atlas_charts = tuple(getattr(validated_atlas, "charts", ()))
    ks_evaluation = evaluation.ks_evaluation
    entry_event = evaluation.entry_event_certificate
    entry_time = _event_root_value(entry_event)
    ordinary_entry_point = _point_taylor_from_interval_solution(
        evaluation.ordinary_entry_solution,
    )
    ks_point_state = _point_spatial_ks_state_from_interval_state(
        evaluation.entry_ks_state,
    )
    ks_point_solution = construct_spatial_ks_binary_taylor_solution(
        ks_point_state,
        order=int(getattr(ks_evaluation.ks_solution, "order", 0)),
    )

    ordinary_entry_meta = atlas_charts[0]
    ks_meta = atlas_charts[1]
    ordinary_entry_chart = ordinary_taylor_chart_certificate_from_solution(
        ordinary_entry_point,
        certificate_id=f"{certificate_id_prefix}-ordinary-before-ks-chart-0",
        chart_id=str(getattr(ordinary_entry_meta, "chart_id", "ordinary_before_spatial_ks_0")),
        parameter_interval=(0.0, entry_time),
        physical_time_interval=(0.0, entry_time),
        coefficient_tolerance=coefficient_tolerance,
        residual_tolerance=ordinary_residual_tolerance,
        tail_bound=float(getattr(ordinary_entry_meta, "tail_bound", 0.0)),
        sample_count=sample_count,
        source="finite_target_spatial_ordinary_ks_entry_independent_serialization",
    )
    ks_chart = _spatial_ks_chart_certificate_with_interval_time_enclosure(
        ks_point_solution,
        certificate_id=f"{certificate_id_prefix}-spatial-ks-chart-1",
        chart_id=str(getattr(ks_meta, "chart_id", "spatial_ks_1")),
        parameter_interval=_float_interval_pair(ks_meta.parameter_interval),
        physical_time_interval=_float_interval_pair(ks_meta.physical_time_interval),
        coefficient_tolerance=coefficient_tolerance,
        regularized_residual_tolerance=regularized_residual_tolerance,
        projected_residual_tolerance=projected_residual_tolerance,
        constraint_tolerance=constraint_tolerance,
        tail_bound=float(getattr(ks_meta, "tail_bound", 0.0)),
        sample_count=sample_count,
        projection_rho_lower_bound=0.0,
        physical_time_shift=entry_time,
        source="finite_target_spatial_ordinary_ks_chart_independent_serialization",
    )
    charts: list[object] = [ordinary_entry_chart, ks_chart]
    transitions = [
        spatial_ks_transition_certificate(
            transition_id=f"{certificate_id_prefix}-ordinary-to-ks-transition-0",
            source_chart_id=ordinary_entry_chart.chart_id,
            target_chart_id=ks_chart.chart_id,
            transition_type="spatial_ordinary_to_ks_decreasing_distance_entry",
            handoff_time=entry_time,
            source_parameter=entry_time,
            target_parameter=0.0,
            position_tolerance=position_tolerance,
            velocity_tolerance=velocity_tolerance,
            physical_time_tolerance=physical_time_tolerance,
            source="finite_target_spatial_ordinary_to_ks_transition_serialization",
        )
    ]
    if len(atlas_charts) == 3:
        ordinary_after_meta = atlas_charts[2]
        exit_parameter = float(ks_evaluation.endpoint_projection.s_value)
        exit_time = float(ks_point_solution.physical_time_at(exit_parameter)) + entry_time
        exit_positions, exit_velocities = ks_binary_chart_to_spatial(
            ks_point_solution.state_at(exit_parameter),
        )
        ordinary_after_point = construct_taylor_solution(
            exit_positions,
            exit_velocities,
            np.asarray(validated_atlas.masses, dtype=float),
            order=int(getattr(ks_evaluation.ordinary_solution, "order", 0)),
        )
        ordinary_after_interval = _float_interval_pair(
            ordinary_after_meta.parameter_interval,
        )
        ordinary_after_physical = (
            exit_time,
            exit_time + max(
                0.0,
                ordinary_after_interval[1] - ordinary_after_interval[0],
            ),
        )
        ordinary_after_chart = ordinary_taylor_chart_certificate_from_solution(
            ordinary_after_point,
            certificate_id=f"{certificate_id_prefix}-ordinary-after-ks-chart-2",
            chart_id=str(
                getattr(ordinary_after_meta, "chart_id", "ordinary_after_spatial_ks_2")
            ),
            parameter_interval=ordinary_after_interval,
            physical_time_interval=ordinary_after_physical,
            coefficient_tolerance=coefficient_tolerance,
            residual_tolerance=ordinary_residual_tolerance,
            tail_bound=float(getattr(ordinary_after_meta, "tail_bound", 0.0)),
            sample_count=sample_count,
            source="finite_target_spatial_ordinary_after_ks_independent_serialization",
        )
        charts.append(ordinary_after_chart)
        transitions.append(
            spatial_ks_transition_certificate(
                transition_id=f"{certificate_id_prefix}-ks-to-ordinary-transition-1",
                source_chart_id=ks_chart.chart_id,
                target_chart_id=ordinary_after_chart.chart_id,
                transition_type="spatial_ks_to_ordinary_safe_handoff",
                handoff_time=exit_time,
                source_parameter=exit_parameter,
                target_parameter=0.0,
                position_tolerance=position_tolerance,
                velocity_tolerance=velocity_tolerance,
                physical_time_tolerance=physical_time_tolerance,
                source="finite_target_spatial_ks_to_ordinary_transition_serialization",
            )
        )
    target_value = getattr(validated_atlas, "target_time", None)
    target = np.inf if target_value is None else float(target_value)
    target_interval = (
        (target, target)
        if np.isfinite(target)
        else _float_interval_pair(charts[-1].physical_time_interval)
    )
    chain = ChartChainCertificate(
        certificate_id=f"{certificate_id_prefix}-spatial-ordinary-ks-chain-0",
        chain_id=f"{certificate_id_prefix}-spatial-ordinary-ks-chain",
        chain_type="regularized_atlas_chart_chain",
        chart_ids=tuple(chart.chart_id for chart in charts),
        transition_ids=tuple(transition.transition_id for transition in transitions),
        target_physical_time_interval=target_interval,
    )
    return verify_chart_certificates(
        tuple(charts),
        transitions=tuple(transitions),
        chart_chains=(chain,),
    )


def _event_root_value(event: object) -> float:
    root = getattr(event, "root", None)
    if root is not None:
        return float(root)
    root_interval = getattr(event, "root_interval", None)
    if root_interval is not None:
        return _midpoint_value(root_interval)
    raise ValueError("spatial KS event certificate does not carry a root")


def _point_taylor_from_interval_solution(solution: object) -> object:
    positions = _midpoint_array(getattr(solution, "position"))
    velocities = _midpoint_array(getattr(solution, "velocity"))
    return construct_taylor_solution(
        positions[0],
        velocities[0],
        np.asarray(getattr(solution, "masses"), dtype=float),
        order=int(getattr(solution, "order", 0)),
    )


def _point_spatial_ks_state_from_interval_state(state: object) -> SpatialKSBinaryChartState:
    return SpatialKSBinaryChartState(
        masses=np.asarray(getattr(state, "masses"), dtype=float),
        pair=tuple(int(value) for value in getattr(state, "pair")),
        u=_midpoint_array(getattr(state, "u")),
        u_velocity=_midpoint_array(getattr(state, "u_velocity")),
        pair_energy=_midpoint_value(getattr(state, "pair_energy")),
        binary_center=_midpoint_array(getattr(state, "binary_center")),
        binary_center_velocity=_midpoint_array(getattr(state, "binary_center_velocity")),
        third_offset=_midpoint_array(getattr(state, "third_offset")),
        third_offset_velocity=_midpoint_array(getattr(state, "third_offset_velocity")),
    )


def _midpoint_array(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    return np.asarray(
        [_midpoint_value(value) for value in array.reshape(-1)],
        dtype=float,
    ).reshape(array.shape)


def _midpoint_value(value: Any) -> float:
    if hasattr(value, "lower") and hasattr(value, "upper"):
        return 0.5 * (float(value.lower) + float(value.upper))
    lower, upper = value
    return 0.5 * (float(lower) + float(upper))


def construct_independent_ordinary_taylor_checked_prefix(
    masses: Any,
    positions: Any,
    velocities: Any,
    target_time: float,
    *,
    order: int = 10,
    coefficient_tolerance: float = 1.0e-11,
    residual_tolerance: float = 1.0e-8,
    tail_bound: float = 1.0e-9,
    sample_count: int = 7,
) -> object:
    """Build and independently check a one-chart ordinary finite-target prefix.

    This is the first verified-kernel path for the open-time theorem surface:
    construct a local ordinary Taylor chart from the raw finite-target data,
    serialize it into the independent certificate language, add a chart-chain
    coverage certificate, and run the checker.  The result is only checked
    prefix evidence; analytic theorem and set-valued constructor blockers
    remain the responsibility of the surrounding theorem certificate.
    """

    target = float(target_time)
    interval = (min(0.0, target), max(0.0, target))
    solution = construct_taylor_solution(
        np.asarray(positions, dtype=float),
        np.asarray(velocities, dtype=float),
        np.asarray(masses, dtype=float),
        order=int(order),
    )
    chart = ordinary_taylor_chart_certificate_from_solution(
        solution,
        certificate_id="checked-prefix-ordinary-chart-0",
        chart_id="checked-prefix-ordinary-0",
        parameter_interval=interval,
        physical_time_interval=interval,
        coefficient_tolerance=float(coefficient_tolerance),
        residual_tolerance=float(residual_tolerance),
        tail_bound=float(tail_bound),
        sample_count=int(sample_count),
        source="open_time_checked_prefix_ordinary_taylor",
    )
    chain = ChartChainCertificate(
        certificate_id="checked-prefix-chart-chain-0",
        chain_id="checked-prefix-ordinary-chain",
        chain_type="ordinary_chart_chain",
        chart_ids=(chart.chart_id,),
        transition_ids=(),
        target_physical_time_interval=interval,
    )
    return verify_chart_certificates(
        (chart,),
        chart_chains=(chain,),
    )


def certify_pointwise_open_time_locally_finite_atlas_theorem(
    *,
    dimension: int,
    compact_time_rate: float = 1.0,
    total_collision_policy_id: str = "maximal_classical_stop",
    input_model: str = "exact_point_positive_mass_noncollision",
) -> PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate:
    """Derive the exact-input open-time theorem from finite-target completeness.

    The theorem applies the finite-target atlas-or-stop theorem to a countable
    compact exhaustion of physical time.  It is intentionally pointwise: plain
    interval boxes still need the separate set-valued branch/event-order
    recursion theorem recorded by ``FiniteTargetCompletenessReductionCertificate``.
    """

    dimension = int(dimension)
    input_model = str(input_model)
    total_collision_policy_id = str(total_collision_policy_id)
    maximal_classical_policy = total_collision_policy_id in {
        "maximal_classical_stop",
        "maximal_classical_stop_at_total_collision",
    }
    finite_target_theorem = certify_finite_target_completeness_theorem(
        dimension=dimension,
        total_collision_policy_id=total_collision_policy_id,
        input_model=input_model,
    )
    compact_time = certify_compact_time_real_line_coverage(compact_time_rate)
    finite_target_reduction = (
        certify_finite_target_theorem_reduces_compact_interval_exhaustion()
    )
    local_finiteness = certify_countable_nested_compact_interval_local_finiteness()
    obligations = (
        TheoremPipelineObligation(
            obligation="maximal_classical_total_collision_policy",
            certified=maximal_classical_policy,
            source="pointwise_open_time_locally_finite_atlas_theorem",
            detail=(
                "open-time theorem is stated for maximal-classical stop "
                f"semantics; policy={total_collision_policy_id!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="pointwise_finite_target_atlas_or_stop_completeness",
            certified=finite_target_theorem.certified,
            source=type(finite_target_theorem).__name__,
            detail="missing=" + ",".join(finite_target_theorem.missing_obligations),
        ),
        TheoremPipelineObligation(
            obligation="compact_time_real_line_coverage",
            certified=bool(compact_time.certified),
            source=type(compact_time).__name__,
            detail="u=tanh(rate*t) gives a bounded coordinate for physical time",
        ),
        TheoremPipelineObligation(
            obligation="finite_target_theorem_reduces_compact_interval_exhaustion",
            certified=finite_target_reduction.certified,
            source=finite_target_reduction.source,
            detail=finite_target_reduction.statement,
        ),
        TheoremPipelineObligation(
            obligation="countable_nested_compact_interval_local_finiteness",
            certified=local_finiteness.certified,
            source=local_finiteness.source,
            detail=local_finiteness.statement,
        ),
        TheoremPipelineObligation(
            obligation="endpoint_regime_partition_not_required",
            certified=True,
            source="pointwise_open_time_locally_finite_atlas_theorem",
            detail=(
                "finite compact intervals are certified directly; endpoint "
                "regimes remain optional compression certificates"
            ),
        ),
        TheoremPipelineObligation(
            obligation="set_valued_constructor_not_claimed",
            certified=True,
            source="pointwise_open_time_locally_finite_atlas_theorem",
            detail=(
                "the theorem is for exact/computable point inputs; interval-box "
                "branch recursion remains a separate implementation theorem"
            ),
        ),
    )
    return PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate(
        dimension=dimension,
        input_model=input_model,
        total_collision_policy_id=total_collision_policy_id,
        compact_time_certificate=compact_time,
        finite_target_theorem=finite_target_theorem,
        finite_target_reduction_certificate=finite_target_reduction,
        local_finiteness_certificate=local_finiteness,
        endpoint_regime_partition_required=False,
        statement=(
            "For d in {2,3}, positive masses, exact noncollision finite "
            "initial data, and the maximal-classical total-collision policy, "
            "the binary-regularized solution admits a countable locally finite "
            "open-time atlas-or-stop description obtained by applying the "
            "finite-target atlas-or-stop theorem on K_n=[-nR,nR]."
        ),
        proof_sketch=(
            "Choose any R>0 and the nested compact exhaustion K_n=[-nR,nR]. "
            "For each n, apply the pointwise finite-target theorem to +nR and "
            "-nR.  If both directions reach their endpoints, the two finite "
            "chains cover K_n.  If either direction reaches a first unselected "
            "total collision, the maximal-classical theorem stops there and "
            "does not assert continuation through that singularity.  Every "
            "finite physical time lies in some K_n, and every compact time "
            "subinterval lies in one K_n, so these finite certificates form a "
            "countable locally finite atlas-or-stop family without endpoint "
            "regime classification.  This exact-input theorem does not assert "
            "that a plain interval box containing multiple outcome strata can "
            "be consumed without the separate set-valued branch-recursion "
            "backend."
        ),
        obligations=obligations,
    )


def _set_valued_constructor_completeness_detail(certificate: object) -> str:
    theorem_id = str(getattr(certificate, "theorem_id", ""))
    if theorem_id == "validated_set_valued_constructor_completeness":
        nested = getattr(certificate, "set_valued_constructor_certificate", None)
        nested_detail = ""
        if nested is not None:
            nested_detail = (
                "; underlying constructor detail: "
                + _set_valued_constructor_completeness_detail(nested)
            )
        return (
            "validated set-valued constructor theorem certifies the represented "
            f"{getattr(certificate, 'input_scope_id', 'interval-box')} input "
            "class; arbitrary recursive partition generation remains outside "
            "its scope"
            f"{nested_detail}"
        )
    if theorem_id == "uniform_margin_set_valued_constructor_branch_event_completeness":
        return (
            "explicit uniform-margin set-valued constructor theorem certifies "
            "this represented positive-margin input class; equality strata "
            "remain outside its scope"
        )
    if (
        theorem_id
        == "supplied_recursive_stratified_set_valued_constructor_branch_event_completeness"
    ):
        branch_tree_kind = _recursive_consumption_source_tree_kind(
            getattr(certificate, "branch_consumption_certificate", None)
        )
        event_tree_kind = _recursive_consumption_source_tree_kind(
            getattr(certificate, "event_order_consumption_certificate", None)
        )
        branch_source_type = _recursive_consumption_source_type(
            getattr(certificate, "branch_consumption_certificate", None)
        )
        event_source_type = _recursive_consumption_source_type(
            getattr(certificate, "event_order_consumption_certificate", None)
        )
        branch_source_scope = recursive_constructor_source_scope(
            getattr(certificate, "branch_consumption_certificate", None)
        )
        event_source_scope = recursive_constructor_source_scope(
            getattr(certificate, "event_order_consumption_certificate", None)
        )
        if (
            branch_tree_kind == "simultaneous_close_pair_branch_partition"
            and event_tree_kind == "simultaneous_close_pair_branch_partition"
        ):
            return (
                "constructor-derived simultaneous close-pair branch partition "
                "is consumed as a finite recursive stratified set-valued input "
                "class; each branch has an explicit ordinary/separated-binary "
                "leaf taxonomy, and arbitrary recursive partition generation "
                "remains outside its scope"
            )
        if (
            branch_source_type
            and event_source_type
            and branch_source_type != event_source_type
            and branch_source_scope is not None
            and event_source_scope is not None
        ):
            return (
                "constructor-derived mixed branch/event recursive "
                "partition is consumed as a finite stratified set-valued "
                f"input class: branch source {branch_source_type} "
                f"({branch_source_scope[3]}) and event-order source "
                f"{event_source_type} ({event_source_scope[3]}); arbitrary recursive "
                "partition generation remains outside its scope"
            )
        if (
            branch_source_type == event_source_type
            and branch_source_scope is not None
        ):
            return (
                f"constructor-derived {branch_source_scope[3]} "
                f"({branch_source_scope[2]}) is consumed as a represented "
                "equality-tree input class and finite recursive stratified "
                "set-valued input class; arbitrary recursive partition "
                "generation remains outside its scope"
            )
        return (
            "supplied recursive stratified set-valued constructor theorem "
            "certifies this represented equality-tree input class; arbitrary "
            "recursive partition generation remains outside its scope"
        )
    if (
        theorem_id
        == "affine_halfspace_arrangement_set_valued_constructor_branch_event_completeness"
    ):
        arrangement = getattr(certificate, "arrangement_certificate", None)
        dimension = getattr(arrangement, "dimension", None)
        descent_detail = _recursive_consumption_descent_detail(
            getattr(certificate, "branch_consumption_certificate", None)
        )
        if dimension == 3:
            return (
                "constructor-derived affine halfspace arrangement theorem "
                "certifies this represented finite 3D oblique arrangement "
                "input class with polyhedron value bounds, volume-cover "
                f"evidence, and recursive equality children ({descent_detail}); arbitrary "
                "recursive partition generation remains outside its scope"
            )
        return (
            "constructor-derived affine halfspace arrangement theorem certifies "
            "this represented finite 2D oblique arrangement input class with "
            "polygon value bounds, area-cover evidence, and recursive equality "
            f"children ({descent_detail}); arbitrary recursive partition "
            "generation remains outside its scope"
        )
    return (
        f"{type(certificate).__name__} certifies the represented set-valued "
        "finite-target constructor input class"
    )


def _recursive_consumption_source_tree_kind(certificate: object | None) -> str:
    stratified_tree = getattr(certificate, "source_tree", None)
    source_tree = getattr(stratified_tree, "source_tree", None)
    return str(getattr(source_tree, "tree_kind", ""))


def _recursive_consumption_source_type(certificate: object | None) -> str:
    stratified_tree = getattr(certificate, "source_tree", None)
    source_tree = getattr(stratified_tree, "source_tree", None)
    return str(getattr(source_tree, "source_type", ""))


def _recursive_consumption_descent_detail(certificate: object | None) -> str:
    if certificate is None:
        return "missing recursive descent certificate"
    child_sources = tuple(
        str(source)
        for source in getattr(certificate, "child_constructor_source_types", ())
        if source
    )
    child_source_detail = (
        "; child_constructor_sources=" + ",".join(child_sources)
        if child_sources
        else ""
    )
    return (
        f"strict_descent_edge_count="
        f"{getattr(certificate, 'strict_descent_edge_count', 'missing')}; "
        f"unresolved_descent_edge_count="
        f"{getattr(certificate, 'unresolved_descent_edge_count', 'missing')}; "
        f"descent_well_founded="
        f"{getattr(certificate, 'descent_well_founded', 'missing')}"
        f"{child_source_detail}"
    )


def _scoped_set_valued_constructor_certificate(certificate: object | None) -> object | None:
    if certificate is None:
        return None
    return getattr(certificate, "set_valued_constructor_certificate", certificate)


def certify_finite_target_completeness_reduction(
    *,
    finite_target_certificate: FiniteTargetAtlasOrStopCertificate,
    compact_interval_certificate: CompactIntervalAtlasOrStopCertificate | None = None,
    exhaustion_family_certificate: CompactIntervalExhaustionFamilyCertificate | None = None,
    recursive_branch_refinement_certificate: Any | None = None,
    event_order_refinement_certificate: Any | None = None,
    stratified_branch_tree_certificate: Any | None = None,
    stratified_event_order_tree_certificate: Any | None = None,
    recursive_stratified_branch_consumption_certificate: Any | None = None,
    recursive_stratified_event_order_consumption_certificate: Any | None = None,
    set_valued_constructor_completeness_certificate: (
        UniformMarginSetValuedConstructorCompletenessCertificate
        | SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate
        | ValidatedSetValuedConstructorCompletenessTheoremCertificate
        | None
    ) = None,
) -> FiniteTargetCompletenessReductionCertificate:
    """Expose the real gap between checked finite targets and universality."""

    analytic_certificates = (
        finite_target_certificate.painleve_certificate,
        finite_target_certificate.binary_regularization_certificate,
        finite_target_certificate.binary_isolation_certificate,
        finite_target_certificate.binary_accumulation_certificate,
        finite_target_certificate.compact_collision_free_cover_certificate,
    )
    observed_prefix_failures = _observed_finite_target_prefix_failures(
        finite_target_certificate,
        compact_interval_certificate,
        exhaustion_family_certificate,
    )
    dimension = int(
        getattr(
            finite_target_certificate.input_domain_certificate,
            "dimension",
            0,
        )
    )
    pointwise_theorem = certify_finite_target_completeness_theorem(
        dimension=dimension,
        total_collision_policy_id=(
            finite_target_certificate.total_collision_policy.policy_id
        ),
    )
    scoped_set_valued_certificate = _scoped_set_valued_constructor_certificate(
        set_valued_constructor_completeness_certificate,
    )
    if recursive_branch_refinement_certificate is None:
        recursive_branch_refinement_certificate = getattr(
            scoped_set_valued_certificate,
            "branch_refinement_certificate",
            None,
        )
    if event_order_refinement_certificate is None:
        event_order_refinement_certificate = getattr(
            scoped_set_valued_certificate,
            "event_order_refinement_certificate",
            None,
        )
    if recursive_stratified_branch_consumption_certificate is None:
        recursive_stratified_branch_consumption_certificate = getattr(
            scoped_set_valued_certificate,
            "branch_consumption_certificate",
            None,
        )
    if recursive_stratified_event_order_consumption_certificate is None:
        recursive_stratified_event_order_consumption_certificate = getattr(
            scoped_set_valued_certificate,
            "event_order_consumption_certificate",
            None,
        )
    search_completeness = certify_finite_target_certificate_search_completeness(
        pointwise_theorem,
        observed_prefix_failures=observed_prefix_failures,
        recursive_branch_refinement_certificate=recursive_branch_refinement_certificate,
        event_order_refinement_certificate=event_order_refinement_certificate,
        stratified_branch_tree_certificate=stratified_branch_tree_certificate,
        stratified_event_order_tree_certificate=stratified_event_order_tree_certificate,
        recursive_stratified_branch_consumption_certificate=(
            recursive_stratified_branch_consumption_certificate
        ),
        recursive_stratified_event_order_consumption_certificate=(
            recursive_stratified_event_order_consumption_certificate
        ),
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="finite_target_analytic_reduction_lemmas",
            certified=all(
                getattr(certificate, "certified", False)
                for certificate in analytic_certificates
            ),
            source="open_time_atlas_theorem",
            detail="Painleve, all-pair binary regularization, binary isolation, binary accumulation, and compact Taylor cover",
        ),
        TheoremPipelineObligation(
            obligation="pointwise_finite_target_atlas_or_stop_completeness",
            certified=pointwise_theorem.certified,
            source=type(pointwise_theorem).__name__,
            detail=(
                "missing="
                + ",".join(pointwise_theorem.missing_obligations)
            ),
        ),
        TheoremPipelineObligation(
            obligation="point_input_finite_target_certificate_search",
            certified=all(
                name not in search_completeness.missing_obligations
                for name in (
                    "fair_adaptive_chart_search",
                    "finite_time_loop_budget_elimination",
                    "certificate_search_completeness_for_point_inputs",
                )
            ),
            source=type(search_completeness).__name__,
            detail=(
                "exact/computable point search is closed; set-valued "
                "constructor gaps="
                + ",".join(search_completeness.missing_obligations)
            ),
        ),
        TheoremPipelineObligation(
            obligation="pointwise_finite_target_constructor",
            certified=finite_target_certificate.certified,
            source=type(finite_target_certificate).__name__,
            detail=finite_target_certificate.outcome_id,
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_prefix_constructor",
            certified=bool(
                compact_interval_certificate is not None
                and compact_interval_certificate.certified
            ),
            source=(
                type(compact_interval_certificate).__name__
                if compact_interval_certificate is not None
                else "missing"
            ),
            detail=(
                compact_interval_certificate.outcome_id
                if compact_interval_certificate is not None
                else "compact interval certificate was not supplied"
            ),
        ),
        TheoremPipelineObligation(
            obligation="countable_exhaustion_prefix_constructor",
            certified=bool(
                exhaustion_family_certificate is not None
                and exhaustion_family_certificate.certified
            ),
            source=(
                type(exhaustion_family_certificate).__name__
                if exhaustion_family_certificate is not None
                else "missing"
            ),
            detail=(
                exhaustion_family_certificate.exhaustion_formula
                if exhaustion_family_certificate is not None
                else "nested compact prefix was not supplied"
            ),
        ),
        TheoremPipelineObligation(
            obligation="set_valued_constructor_branch_event_completeness",
            certified=bool(
                set_valued_constructor_completeness_certificate is not None
                and set_valued_constructor_completeness_certificate.certified
            ),
            source=(
                type(set_valued_constructor_completeness_certificate).__name__
                if set_valued_constructor_completeness_certificate is not None
                else "missing_set_valued_constructor_branch_event_theorem"
            ),
            detail=(
                _set_valued_constructor_completeness_detail(
                    set_valued_constructor_completeness_certificate
                )
                if set_valued_constructor_completeness_certificate is not None
                and getattr(
                    set_valued_constructor_completeness_certificate,
                    "certified",
                    False,
                )
                else (
                    "supplied certificate-search refinement evidence has "
                    "consumed the displayed recursive branch/event obligations; "
                    "the public set-valued constructor still needs a theorem "
                    "deriving such evidence from arbitrary positive-mass "
                    "noncollision interval inputs"
                )
                if search_completeness.certified
                else (
                    "the pointwise atlas-or-stop theorem and fair exact-point "
                    "certificate search are certified; the public set-valued "
                    "constructor still needs recursive branch-partition and "
                    "event-order consumption for arbitrary positive-mass "
                    "noncollision inputs"
                )
            ),
        ),
    )
    return FiniteTargetCompletenessReductionCertificate(
        finite_target_certificate=finite_target_certificate,
        compact_interval_certificate=compact_interval_certificate,
        exhaustion_family_certificate=exhaustion_family_certificate,
        pointwise_completeness_theorem=pointwise_theorem,
        certificate_search_completeness=search_completeness,
        set_valued_constructor_completeness_certificate=(
            set_valued_constructor_completeness_certificate
        ),
        statement=(
            "For every positive-mass noncollision initial state and finite "
            "target time, the finite-target constructor returns a certified "
            "ordinary/binary/selected atlas, or a certified maximal-classical "
            "total-collision stop."
        ),
        proof_sketch=(
            "Painleve reduces finite singularities to collisions. Compact "
            "collision-free segments have finite ordinary Taylor covers. "
            "Separated binary collisions are regularized pairwise and isolated; "
            "binary accumulation in finite time forces total collision. The "
            "pointwise theorem now eliminates finite loop budget and exact "
            "point certificate search as mathematical obstructions. What "
            "remains is the constructive set-valued search theorem: branch "
            "partitions and event-order ambiguities must be recursively "
            "consumed by a fair certificate search, or converted into the "
            "explicit total-collision policy."
        ),
        obligations=obligations,
    )


def certify_explicit_total_collision_policy(
    policy_id: str,
    *,
    selector_policy_id: str | None = None,
) -> TotalCollisionPolicyCertificate:
    policy_id = str(policy_id)
    selected = policy_id.startswith("selected_")
    return TotalCollisionPolicyCertificate(
        policy_id=policy_id,
        stop_at_unselected_total_collision=policy_id.startswith("maximal_classical_stop"),
        selected_continuation_allowed=selected,
        selector_policy_id=selector_policy_id or (policy_id if selected else None),
    )


def certify_total_collision_stop_from_validated_atlas(
    *,
    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate | None,
    validated_atlas: object | None,
    target_time: float,
    total_collision_policy: TotalCollisionPolicyCertificate,
    independent_chart_verifier_certificate: object | None = None,
) -> TotalCollisionStopCertificate:
    """Extract a maximal-classical stop from a certified total-collision chart."""

    target_time = float(target_time)
    chart_type, stop_time = _first_total_collision_stop_event(validated_atlas)
    atlas_or_checker_certified = bool(
        getattr(validated_atlas, "proof_certified", False) is True
        or _independent_verifier_covers_validated_atlas(
            independent_chart_verifier_certificate,
            validated_atlas,
        )
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="positive_mass_noncollision_input_domain",
            certified=bool(getattr(input_domain_certificate, "certified", False)),
            source=(
                type(input_domain_certificate).__name__
                if input_domain_certificate is not None
                else "missing"
            ),
            detail="total-collision stop consumes noncollision initial data",
        ),
        TheoremPipelineObligation(
            obligation="validated_atlas_proof_certified",
            certified=atlas_or_checker_certified,
            source=(
                type(independent_chart_verifier_certificate).__name__
                if independent_chart_verifier_certificate is not None
                else type(validated_atlas).__name__
                if validated_atlas is not None
                else "missing"
            ),
            detail=(
                "regularized atlas identifies the total-collision event; "
                "constructor proof or independent checker evidence is accepted"
            ),
        ),
        TheoremPipelineObligation(
            obligation="validated_atlas_input_domain_matches_stop_input",
            certified=_validated_atlas_initial_state_matches_input_domain(
                validated_atlas,
                input_domain_certificate,
            ),
            source=type(validated_atlas).__name__ if validated_atlas is not None else "missing",
            detail="stop certificate is bound to the supplied initial condition",
        ),
        TheoremPipelineObligation(
            obligation="total_collision_chart_present",
            certified=bool(chart_type),
            source=type(validated_atlas).__name__ if validated_atlas is not None else "missing",
            detail=f"chart_type={chart_type or 'missing'}",
        ),
        TheoremPipelineObligation(
            obligation="total_collision_event_in_chart_domain",
            certified=bool(stop_time is not None and np.isfinite(stop_time)),
            source=type(validated_atlas).__name__ if validated_atlas is not None else "missing",
            detail=f"stop_time={stop_time!r}",
        ),
        TheoremPipelineObligation(
            obligation="total_collision_stop_before_target",
            certified=_stop_time_lies_before_target(stop_time, target_time),
            source="certify_total_collision_stop_from_validated_atlas",
            detail=f"stop_time={stop_time!r}; target_time={target_time!r}",
        ),
        TheoremPipelineObligation(
            obligation="maximal_classical_stop_policy",
            certified=bool(
                total_collision_policy.certified
                and total_collision_policy.stop_at_unselected_total_collision
                and not total_collision_policy.selected_continuation_allowed
            ),
            source=type(total_collision_policy).__name__,
            detail=total_collision_policy.policy_id,
        ),
    )
    return TotalCollisionStopCertificate(
        input_domain_certificate=input_domain_certificate,
        validated_atlas=validated_atlas,
        total_collision_chart_type=chart_type,
        stop_time=stop_time,
        target_time=target_time,
        total_collision_policy=total_collision_policy,
        obligations=obligations,
    )


def certify_three_body_painleve_no_noncollision_singularities() -> AnalyticTheoremCertificate:
    return AnalyticTheoremCertificate(
        theorem_id="three_body_painleve_no_noncollision_singularities",
        statement="For three positive masses, every finite-time singularity is a collision singularity.",
        proof_sketch=(
            "Painleve's N=3 theorem excludes finite noncollision singularities: "
            "if all pair distances stay bounded below on a compact time interval, "
            "the Newtonian vector field and its derivatives remain bounded, so "
            "ordinary analytic continuation extends the branch past the endpoint."
        ),
        prerequisites=("positive_masses", "finite_energy_initial_state"),
    )


def certify_all_pair_binary_regularization(*, dimension: int) -> AnalyticTheoremCertificate:
    dimension = int(dimension)
    certified = dimension in (2, 3)
    statement = (
        "Every isolated binary collision in the supported three-body dimension "
        "has a regularized chart for each pair: planar Levi-Civita in dimension "
        "two and KS binary charts in dimension three."
    )
    return AnalyticTheoremCertificate(
        theorem_id="all_pair_binary_regularization" if certified else "",
        statement=statement,
        proof_sketch=(
            "Lift the colliding pair to a regularized binary variable, use "
            "dt/ds=rho to remove the inverse-square singularity, keep the third "
            "body separated, then project the regularized solution back to "
            "Newtonian coordinates away from the collision instant."
        )
        if certified
        else "",
        prerequisites=("separated_third_body", "pair_index_coverage"),
    )


def certify_binary_collision_isolation(*, dimension: int) -> AnalyticTheoremCertificate:
    dimension = int(dimension)
    certified = dimension in (2, 3)
    return AnalyticTheoremCertificate(
        theorem_id="binary_collision_isolation" if certified else "",
        statement=(
            "A separated binary event is isolated in a regularized binary chart "
            "unless the compact endpoint is total collision."
        ),
        proof_sketch=(
            "In a regularized LC/KS chart the lifted vector field is analytic "
            "with the third body separated, so zeros of the binary separation "
            "are isolated unless the same compact endpoint degenerates into a "
            "multi-pair collision."
        )
        if certified
        else "",
        prerequisites=("all_pair_binary_regularization",),
    )


def certify_binary_accumulation_implies_total_collision() -> AnalyticTheoremCertificate:
    return AnalyticTheoremCertificate(
        theorem_id="binary_accumulation_implies_total_collision",
        statement="Infinitely many separated binary events in a compact interval can accumulate only at total collision.",
        proof_sketch=(
            "A compact sequence of binary event times has an accumulation time. "
            "Painleve reduces a finite singularity there to collision. If only "
            "one pair vanished, the separated-binary regularized chart would "
            "isolate the event and contradict accumulation, so at least two pair "
            "distances vanish and all three bodies collide."
        ),
        prerequisites=(
            "three_body_painleve_no_noncollision_singularities",
            "binary_collision_isolation",
        ),
    )


def certify_compact_collision_free_taylor_cover() -> AnalyticTheoremCertificate:
    return AnalyticTheoremCertificate(
        theorem_id="compact_collision_free_taylor_cover",
        statement="A compact collision-free segment is covered by finitely many ordinary analytic Taylor charts.",
        proof_sketch=(
            "The minimum pair distance on a compact collision-free segment is "
            "positive, so the Newtonian vector field is analytic on a uniform "
            "complex neighborhood of the segment. Local Taylor charts cover the "
            "segment, and compactness selects a finite subcover."
        ),
        prerequisites=("positive_pair_distance_floor",),
    )


def certify_finite_target_theorem_reduces_compact_interval_exhaustion() -> AnalyticTheoremCertificate:
    return AnalyticTheoremCertificate(
        theorem_id="finite_target_theorem_reduces_compact_interval_exhaustion",
        statement=(
            "A finite-target atlas-or-stop theorem for every finite time "
            "induces a compact-interval atlas-or-stop response on every "
            "symmetric interval K_n=[-nR,nR]."
        ),
        proof_sketch=(
            "Fix R>0 and n>=1, and write K_n=[-nR,nR].  Apply the "
            "pointwise finite-target atlas-or-stop theorem to the exact "
            "initial state with targets +nR and -nR, using the same explicit "
            "total-collision policy.  If both responses reach their endpoint "
            "targets, reversing the parameter direction on the negative-time "
            "finite chain gives a finite certified atlas on [-nR,0], while the "
            "positive-time chain covers [0,nR]; their common initial chart is "
            "the same initial state, so finite chart-chain concatenation gives "
            "a certified compact-interval atlas on K_n.  If either one-sided "
            "response reaches the first unselected total collision before its "
            "target, the compact response is exactly the earliest such "
            "maximal-classical stop and no continuation through that singular "
            "time is asserted.  Thus no endpoint regime classification is used: "
            "each compact interval is reduced to two finite-target theorem "
            "applications plus the explicit stop policy."
        ),
        prerequisites=("finite_target_atlas_or_stop_theorem",),
        proof_mode="internal_two_sided_finite_target_compact_interval_reduction",
        internally_proven=True,
    )


def certify_countable_nested_compact_interval_local_finiteness() -> AnalyticTheoremCertificate:
    return AnalyticTheoremCertificate(
        theorem_id="countable_nested_compact_interval_local_finiteness",
        statement=(
            "The family K_n=[-nR,nR] is a countable exhaustion of the real "
            "line by compact physical-time intervals, and every compact "
            "subinterval is contained in some K_n."
        ),
        proof_sketch=(
            "Let R>0 and K_n=[-nR,nR].  The index set N is countable, each "
            "K_n is compact, K_n is contained in K_{n+1}, and for every finite "
            "physical time t there is an integer n>=ceil(|t|/R) with t in K_n; "
            "hence the union of the K_n is the whole real line.  If J=[a,b] is "
            "any compact physical-time subinterval, M=max(|a|,|b|) is finite, "
            "so every n>=ceil(M/R) contains J in K_n.  Therefore a chart family "
            "obtained by applying the finite-target theorem on this cofinal "
            "nested compact exhaustion is locally finite in the theorem sense: "
            "every compact J is already covered by one finite prefix response "
            "K_n, and evaluating any finite target needs only the finite "
            "atlas-or-stop certificate for such an n."
        ),
        prerequisites=("positive_base_time_radius",),
        proof_mode="internal_countable_nested_compact_exhaustion_local_finiteness",
        internally_proven=True,
    )


def certify_countable_compact_time_exhaustion(
    *,
    compact_time_certificate: CompactTimeCoverageCertificate,
    finite_target_certificate: FiniteTargetAtlasOrStopCertificate,
    compact_interval_certificate: CompactIntervalAtlasOrStopCertificate | None = None,
    exhaustion_family_certificate: CompactIntervalExhaustionFamilyCertificate | None = None,
) -> CountableCompactExhaustionCertificate:
    finite_certificate_certified = bool(finite_target_certificate.certified)
    compact_interval_certified = bool(
        compact_interval_certificate is not None
        and compact_interval_certificate.certified
    )
    exhaustion_family_certified = bool(
        exhaustion_family_certificate is not None
        and exhaustion_family_certificate.certified
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="compact_time_real_line_coverage",
            certified=bool(compact_time_certificate.certified),
            source=type(compact_time_certificate).__name__,
            detail="u=tanh(rate*t) maps real physical time to (-1,1)",
        ),
        TheoremPipelineObligation(
            obligation="finite_target_atlas_or_stop_theorem",
            certified=finite_certificate_certified,
            source=type(finite_target_certificate).__name__,
            detail=finite_target_certificate.outcome_id,
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_atlas_or_stop_theorem",
            certified=compact_interval_certified,
            source=(
                type(compact_interval_certificate).__name__
                if compact_interval_certificate is not None
                else "missing"
            ),
            detail=(
                compact_interval_certificate.outcome_id
                if compact_interval_certificate is not None
                else "compact interval certificate was not supplied"
            ),
        ),
        TheoremPipelineObligation(
            obligation="compact_interval_exhaustion_family",
            certified=exhaustion_family_certified,
            source=(
                type(exhaustion_family_certificate).__name__
                if exhaustion_family_certificate is not None
                else "missing"
            ),
            detail=(
                exhaustion_family_certificate.exhaustion_formula
                if exhaustion_family_certificate is not None
                else "nested compact exhaustion family was not supplied"
            ),
        ),
        TheoremPipelineObligation(
            obligation="locally_finite_compact_interval_exhaustion",
            certified=exhaustion_family_certified,
            source="open_time_locally_finite_atlas_theorem",
            detail=(
                "each compact interval [-n,n] is covered by two finite-target "
                "atlas-or-stop certificates; compact-time cores "
                "[-1+2^-n,1-2^-n] are countable and locally finite"
            ),
        ),
    )
    return CountableCompactExhaustionCertificate(
        compact_time_certificate=compact_time_certificate,
        finite_target_certificate=finite_target_certificate,
        exhaustion_id="countable_compact_physical_time_exhaustion",
        obligations=obligations,
        compact_interval_certificate=compact_interval_certificate,
        exhaustion_family_certificate=exhaustion_family_certificate,
    )


def _classification_certifies_unselected_total_collision_stop(
    classification: FiniteTimeRegimeClassificationCertificate,
) -> bool:
    obligations = set(classification.failure_obligations)
    return bool(
        "unselected_total_collision_before_target" in obligations
        or "certified_total_collision_stop" in obligations
    )


def _finite_target_obstruction_obligations(
    classification: FiniteTimeRegimeClassificationCertificate,
    *,
    outcome_id: str,
    has_total_collision_chart: bool,
    total_collision_policy: TotalCollisionPolicyCertificate,
) -> tuple[str, ...]:
    if outcome_id != "proof_grade_obstruction":
        return ()
    missing = list(classification.missing_obligations)
    missing.extend(classification.failure_obligations)
    if has_total_collision_chart and not total_collision_policy.selected_continuation_allowed:
        missing.append("selected_total_collision_policy_required")
    if not total_collision_policy.certified:
        missing.append("explicit_total_collision_policy")
    if not missing:
        missing.append("finite_time_constructor_did_not_certify_target_or_stop")
    return tuple(dict.fromkeys(str(item) for item in missing if str(item)))


def _finite_target_atlas_or_stop_certified(
    target: object,
) -> bool:
    return (
        isinstance(target, FiniteTargetAtlasOrStopCertificate)
        and target.certified is True
    )


def _total_collision_stop_certified(stop_certificate: object) -> bool:
    return (
        isinstance(stop_certificate, TotalCollisionStopCertificate)
        and stop_certificate.certified is True
    )


def _compact_interval_obstruction_obligations(
    past_target: FiniteTargetAtlasOrStopCertificate | None,
    future_target: FiniteTargetAtlasOrStopCertificate | None,
    *,
    radius: float,
    total_collision_policy: TotalCollisionPolicyCertificate,
    outcome_id: str,
) -> tuple[str, ...]:
    if outcome_id != "proof_grade_obstruction":
        return ()
    missing: list[str] = []
    if not (np.isfinite(radius) and radius > 0.0):
        missing.append("positive_time_radius_compact_interval")
    if past_target is None:
        missing.append("past_finite_target_atlas_or_stop_theorem")
    elif not isinstance(past_target, FiniteTargetAtlasOrStopCertificate):
        missing.append("past_finite_target_atlas_or_stop_theorem_type")
    elif past_target.certified is not True:
        missing.extend(
            f"past:{obligation}" for obligation in past_target.missing_obligations
        )
        missing.extend(
            f"past:{obligation}" for obligation in past_target.obstruction_obligations
        )
    if future_target is None:
        missing.append("future_finite_target_atlas_or_stop_theorem")
    elif not isinstance(future_target, FiniteTargetAtlasOrStopCertificate):
        missing.append("future_finite_target_atlas_or_stop_theorem_type")
    elif future_target.certified is not True:
        missing.extend(
            f"future:{obligation}" for obligation in future_target.missing_obligations
        )
        missing.extend(
            f"future:{obligation}"
            for obligation in future_target.obstruction_obligations
        )
    if not _compact_interval_endpoint_policy_consistent(
        past_target,
        future_target,
        total_collision_policy,
    ):
        missing.append("compact_interval_endpoint_policy_consistency")
    if not _compact_interval_endpoint_input_domains_consistent(
        past_target,
        future_target,
    ):
        missing.append("compact_interval_endpoint_input_domain_consistency")
    if not _compact_interval_endpoint_targets_match_radius(
        past_target,
        future_target,
        radius=radius,
    ):
        missing.append("compact_interval_endpoint_target_time_consistency")
    if not missing:
        missing.append("compact_interval_endpoint_certificates_not_certified")
    return tuple(dict.fromkeys(str(item) for item in missing if str(item)))


def _observed_finite_target_prefix_failures(
    finite_target_certificate: FiniteTargetAtlasOrStopCertificate | None,
    compact_interval_certificate: CompactIntervalAtlasOrStopCertificate | None,
    exhaustion_family_certificate: CompactIntervalExhaustionFamilyCertificate | None,
) -> tuple[str, ...]:
    failures: list[str] = []
    targets: list[FiniteTargetAtlasOrStopCertificate] = []
    if finite_target_certificate is not None:
        targets.append(finite_target_certificate)
    if compact_interval_certificate is not None:
        targets.extend(compact_interval_certificate.finite_target_certificates)
    if exhaustion_family_certificate is not None:
        for prefix in exhaustion_family_certificate.prefix_certificates:
            targets.extend(prefix.finite_target_certificates)
    for target in targets:
        failures.extend(tuple(getattr(target, "obstruction_obligations", ())))
        classification = getattr(target, "finite_time_classification", None)
        failures.extend(tuple(getattr(classification, "failure_obligations", ())))
        failures.extend(tuple(getattr(classification, "missing_obligations", ())))
    return tuple(dict.fromkeys(str(item) for item in failures if str(item)))


def _compact_interval_stop_certificate(
    past_target: FiniteTargetAtlasOrStopCertificate | None,
    future_target: FiniteTargetAtlasOrStopCertificate | None,
) -> object | None:
    for target in (past_target, future_target):
        stop_certificate = getattr(target, "stop_certificate", None)
        if _total_collision_stop_certified(stop_certificate):
            return stop_certificate
    return None


def _compact_interval_endpoint_policy_consistent(
    past_target: FiniteTargetAtlasOrStopCertificate | None,
    future_target: FiniteTargetAtlasOrStopCertificate | None,
    policy: TotalCollisionPolicyCertificate,
) -> bool:
    if not isinstance(past_target, FiniteTargetAtlasOrStopCertificate):
        return False
    if not isinstance(future_target, FiniteTargetAtlasOrStopCertificate):
        return False
    if not isinstance(policy, TotalCollisionPolicyCertificate):
        return False
    return bool(
        isinstance(past_target.total_collision_policy, TotalCollisionPolicyCertificate)
        and isinstance(
            future_target.total_collision_policy,
            TotalCollisionPolicyCertificate,
        )
        and past_target.total_collision_policy.certified is True
        and future_target.total_collision_policy.certified is True
        and policy.certified is True
        and past_target.total_collision_policy.policy_id == policy.policy_id
        and future_target.total_collision_policy.policy_id == policy.policy_id
        and past_target.total_collision_policy.selector_policy_id
        == policy.selector_policy_id
        and future_target.total_collision_policy.selector_policy_id
        == policy.selector_policy_id
    )


def _compact_interval_endpoint_targets_match_radius(
    past_target: FiniteTargetAtlasOrStopCertificate | None,
    future_target: FiniteTargetAtlasOrStopCertificate | None,
    *,
    radius: float,
) -> bool:
    if past_target is None or future_target is None:
        return False
    try:
        past_time = float(past_target.target_time)
        future_time = float(future_target.target_time)
        radius = float(radius)
    except (TypeError, ValueError):
        return False
    if not (
        np.isfinite(past_time)
        and np.isfinite(future_time)
        and np.isfinite(radius)
        and radius > 0.0
    ):
        return False
    tolerance = max(1.0e-14, 128.0 * np.finfo(float).eps * abs(radius))
    return bool(
        abs(past_time + radius) <= tolerance
        and abs(future_time - radius) <= tolerance
    )


def _compact_interval_endpoint_input_domains_consistent(
    past_target: FiniteTargetAtlasOrStopCertificate | None,
    future_target: FiniteTargetAtlasOrStopCertificate | None,
) -> bool:
    if past_target is None or future_target is None:
        return False
    return _input_domain_certificates_match(
        past_target.input_domain_certificate,
        future_target.input_domain_certificate,
    )


def _input_domain_certificates_match(
    left: PositiveMassNoncollisionInputDomainCertificate | None,
    right: PositiveMassNoncollisionInputDomainCertificate | None,
) -> bool:
    if not isinstance(left, PositiveMassNoncollisionInputDomainCertificate):
        return False
    if not isinstance(right, PositiveMassNoncollisionInputDomainCertificate):
        return False
    if not (left.certified is True and right.certified is True):
        return False
    return bool(
        left.masses == right.masses
        and left.positions == right.positions
        and left.velocities == right.velocities
        and left.dimension == right.dimension
    )


def _certify_input_domain_or_none(
    masses: Any,
    positions: Any,
    velocities: Any,
) -> PositiveMassNoncollisionInputDomainCertificate | None:
    try:
        return certify_positive_mass_noncollision_input_domain(
            masses,
            positions,
            velocities,
        )
    except (TypeError, ValueError):
        return None


def _validated_atlas_initial_state_matches_input_domain(
    validated_atlas: object | None,
    input_domain: PositiveMassNoncollisionInputDomainCertificate | None,
) -> bool:
    if validated_atlas is None or input_domain is None:
        return False
    if not getattr(input_domain, "certified", False):
        return False
    try:
        atlas_masses = np.asarray(getattr(validated_atlas, "masses"), dtype=float).reshape(-1)
        domain_masses = np.asarray(input_domain.masses, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        return False
    if (
        atlas_masses.shape != domain_masses.shape
        or not np.allclose(atlas_masses, domain_masses, rtol=0.0, atol=1.0e-12)
    ):
        return False
    try:
        input_state = np.concatenate(
            [
                np.asarray(input_domain.positions, dtype=float).reshape(-1),
                np.asarray(input_domain.velocities, dtype=float).reshape(-1),
            ]
        )
        atlas_state = np.asarray(
            getattr(getattr(validated_atlas, "evaluation", None), "initial_state"),
            dtype=float,
        ).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        input_state.shape == atlas_state.shape
        and input_state.size > 0
        and np.allclose(input_state, atlas_state, rtol=1.0e-10, atol=1.0e-10)
    )


def _independent_verifier_covers_validated_atlas(
    verifier_certificate: object | None,
    validated_atlas: object | None,
) -> bool:
    if validated_atlas is None:
        return False
    if type(verifier_certificate) is not IndependentChartVerifierCertificate:
        return False
    if verifier_certificate.certified is not True:
        return False
    if verifier_certificate.proof_grade_arithmetic_checked_bundle_certified is not True:
        return False
    if (
        verifier_certificate.atlas_binding_token
        != finite_time_selector_trace_binding_token(validated_atlas)
    ):
        return False
    chart_count = len(tuple(getattr(validated_atlas, "charts", ())))
    verifier_chart_count = int(
        verifier_certificate.checked_certificate_count,
    )
    verifier_chain_count = int(
        verifier_certificate.checked_chart_chain_count,
    )
    return bool(
        chart_count > 0
        and verifier_chart_count >= chart_count
        and verifier_chain_count > 0
    )


def _first_total_collision_stop_event(
    validated_atlas: object | None,
) -> tuple[str | None, float | None]:
    for chart in getattr(validated_atlas, "charts", ()):
        chart_type = str(getattr(chart, "chart_type", ""))
        if "total_collision" not in chart_type:
            continue
        interval = getattr(chart, "physical_time_interval", None)
        lower = getattr(interval, "lower", None)
        upper = getattr(interval, "upper", None)
        try:
            lower = float(lower)
            upper = float(upper)
        except (TypeError, ValueError):
            continue
        if np.isfinite(lower) and np.isfinite(upper) and lower <= 0.0 <= upper:
            return chart_type, 0.0
    return None, None


def _stop_time_lies_before_target(stop_time: float | None, target_time: float) -> bool:
    if stop_time is None:
        return False
    try:
        stop = float(stop_time)
        target = float(target_time)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(stop) and np.isfinite(target)):
        return False
    tolerance = 1.0e-14
    if abs(target) <= tolerance:
        return abs(stop) <= tolerance
    if target > 0.0:
        return -tolerance <= stop <= target + tolerance
    return target - tolerance <= stop <= tolerance


def _prefix_radii_follow_linear_schedule(
    prefix_certificates: tuple[CompactIntervalAtlasOrStopCertificate, ...],
    *,
    base_radius: float,
) -> bool:
    if not prefix_certificates or not (np.isfinite(base_radius) and base_radius > 0.0):
        return False
    tolerance = max(1.0e-14, 128.0 * np.finfo(float).eps * abs(base_radius))
    for index, certificate in enumerate(prefix_certificates, start=1):
        expected = base_radius * float(index)
        lower = float(certificate.interval_lower)
        upper = float(certificate.interval_upper)
        if not (
            np.isfinite(lower)
            and np.isfinite(upper)
            and abs(lower + expected) <= tolerance * max(1.0, float(index))
            and abs(upper - expected) <= tolerance * max(1.0, float(index))
        ):
            return False
    return True


def _prefix_intervals_are_nested(
    prefix_certificates: tuple[CompactIntervalAtlasOrStopCertificate, ...],
) -> bool:
    if not prefix_certificates:
        return False
    previous_lower = float(prefix_certificates[0].interval_lower)
    previous_upper = float(prefix_certificates[0].interval_upper)
    if not (
        np.isfinite(previous_lower)
        and np.isfinite(previous_upper)
        and previous_lower < previous_upper
    ):
        return False
    for certificate in prefix_certificates[1:]:
        lower = float(certificate.interval_lower)
        upper = float(certificate.interval_upper)
        if not (
            np.isfinite(lower)
            and np.isfinite(upper)
            and lower <= previous_lower
            and upper >= previous_upper
        ):
            return False
        previous_lower = lower
        previous_upper = upper
    return True
