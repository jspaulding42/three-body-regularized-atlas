"""Closed-form target classification for the general three-body problem.

The numerical harness can verify finite target-time enclosures and conditional
Sundman-series obligations. It should not silently treat every meaning of
"closed form" as the same theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .certificate_checker import IndependentChartVerifierCertificate
from .general_solution_theorem import (
    GeneralSolutionTheoremCertificate,
    TheoremPipelineObligation,
)
from .obstructions import certify_classical_integrals_do_not_determine_vector_field
from .open_time_atlas import (
    OpenTimeLocallyFiniteAtlasTheoremCertificate,
    PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate,
)


BRUNS_THEOREM_URL = "https://scienceworld.wolfram.com/physics/BrunsTheorem.html"
YAGASAKI_NONINTEGRABILITY_URL = "https://arxiv.org/abs/2106.04925"
SUNDMAN_MEMOIR_URL = (
    "https://archive.ymsc.tsinghua.edu.cn/pacm_download/117/"
    "5229-11511_2006_Article_BF02422379.pdf"
)
POINTWISE_FINITE_TARGET_CHART_FAMILIES = (
    "ordinary_taylor",
    "planar_levi_civita_binary",
    "spatial_ks_binary",
    "total_collision_stop",
)
POINTWISE_FINITE_TARGET_ALLOWED_OUTCOMES = (
    "finite_atlas_reaches_target",
    "unselected_total_collision_before_target",
)
POINTWISE_REGULARIZED_ATLAS_REQUIRED_OBLIGATIONS = (
    "regularized_locally_finite_atlas_class",
    "pointwise_open_time_atlas_proof",
    "certificate_language_soundness",
    "computable_atlas_certificate_enumeration",
    "maximal_classical_total_collision_policy",
    "pointwise_closed_form_chart_primitives",
    "pointwise_closed_form_finite_target_outcomes",
    "endpoint_regime_partition_not_required",
)


def _strict_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")
    return value


def _strict_optional_bool(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    return _strict_bool(value, field_name)


def _pipeline_obligation_ledger_certified(
    obligations: tuple[Any, ...],
) -> bool:
    return bool(
        obligations
        and any(
            isinstance(obligation, TheoremPipelineObligation)
            and obligation.required is True
            for obligation in obligations
        )
        and all(
            isinstance(obligation, TheoremPipelineObligation)
            for obligation in obligations
        )
        and all(
            obligation.certified is True
            for obligation in obligations
            if obligation.required
        )
    )


def _pipeline_obligation_ledger_missing(
    obligations: tuple[Any, ...],
    *,
    ledger_name: str,
) -> tuple[str, ...]:
    missing: list[str] = []
    if not obligations:
        missing.append(f"{ledger_name}_obligations_present")
    if obligations and not any(
        isinstance(obligation, TheoremPipelineObligation)
        and obligation.required is True
        for obligation in obligations
    ):
        missing.append(f"{ledger_name}_required_obligation_present")
    for obligation in obligations:
        if not isinstance(obligation, TheoremPipelineObligation):
            missing.append(f"{ledger_name}_obligation_type")
            continue
        if obligation.required and obligation.certified is not True:
            missing.append(obligation.obligation)
    return tuple(dict.fromkeys(missing))


def _pipeline_obligation_manifest_exact(
    obligations: tuple[Any, ...],
    required_obligation_ids: tuple[str, ...],
) -> bool:
    return bool(
        tuple(
            obligation.obligation
            for obligation in obligations
            if isinstance(obligation, TheoremPipelineObligation)
        )
        == tuple(required_obligation_ids)
        and len(obligations) == len(required_obligation_ids)
    )


def _requirement_status_ledger_certified(
    statuses: tuple[Any, ...],
) -> bool:
    return bool(
        statuses
        and all(
            type(status) is GeneralSolutionRequirementStatus
            and status.certified is True
            for status in statuses
        )
    )


def _requirement_status_ledger_missing(
    statuses: tuple[Any, ...],
    *,
    ledger_name: str,
) -> tuple[str, ...]:
    missing: list[str] = []
    if not statuses:
        missing.append(f"{ledger_name}_requirements_present")
    for status in statuses:
        if type(status) is not GeneralSolutionRequirementStatus:
            missing.append(f"{ledger_name}_requirement_status_type")
            continue
        if status.certified is not True:
            missing.append(status.requirement)
    return tuple(dict.fromkeys(missing))


def _requirement_status_ledger_structure_missing(
    statuses: tuple[Any, ...],
    *,
    ledger_name: str,
) -> tuple[str, ...]:
    return tuple(
        missing
        for missing in _requirement_status_ledger_missing(
            statuses,
            ledger_name=ledger_name,
        )
        if missing.endswith("_requirements_present")
        or missing.endswith("_requirement_status_type")
    )


@dataclass(frozen=True)
class ClosedFormTheoremReference:
    """External theorem used to classify a closed-form route."""

    theorem_id: str
    statement: str
    source_url: str


@dataclass(frozen=True)
class ClosedFormTargetCertificate:
    """Certificate describing whether a requested closed-form class can be the target."""

    requested_class: str
    normalized_class: str
    status: str
    reason: str
    theorem_references: tuple[ClosedFormTheoremReference, ...]
    missing_requirements: tuple[str, ...] = ()
    compact_sundman_obligations: tuple[str, ...] = ()
    finite_closed_form_obstructed: bool = False
    internal_obstruction_certified: bool = False
    internal_obstruction_evidence: str | None = None
    infinite_series_route: bool = False
    series_route_certified: bool = False
    regularized_atlas_route: bool = False
    atlas_route_certified: bool = False
    general_solution_scope_certified: bool = False
    general_solution_certified: bool = False

    @property
    def classification_certified(self) -> bool:
        return self.status in {
            "obstructed",
            "conditional_infinite_series_route",
            "certified_infinite_series_route",
            "conditional_regularized_atlas_route",
            "certified_regularized_atlas_route",
            "certified_pointwise_regularized_atlas_route",
            "certified_set_valued_constructor_regularized_atlas_route",
            "definition_required",
        }

    @property
    def obstruction_certified(self) -> bool:
        return bool(self.status == "obstructed" and self.finite_closed_form_obstructed)

    @property
    def definition_required(self) -> bool:
        return self.status == "definition_required"

    @property
    def route_summary(self) -> str:
        if self.general_solution_certified:
            if self.status == "certified_pointwise_regularized_atlas_route":
                return "pointwise regularized locally finite atlas route certified for exact/computable inputs"
            if self.status == "certified_set_valued_constructor_regularized_atlas_route":
                return "set-valued constructor regularized locally finite atlas route certified for interval boxes"
            if self.atlas_route_certified:
                return "regularized locally finite atlas route certified; general-solution scope is included"
            return "general solution certified in the requested infinite-series class"
        if self.series_route_certified:
            return "global series route certified; general-solution scope still needs proof"
        if self.atlas_route_certified:
            if self.status == "certified_pointwise_regularized_atlas_route":
                return "pointwise regularized locally finite atlas route certified for exact/computable inputs"
            if self.status == "certified_set_valued_constructor_regularized_atlas_route":
                return "set-valued constructor regularized locally finite atlas route certified for interval boxes"
            return "regularized locally finite atlas route certified; general-solution scope still needs proof"
        if self.finite_closed_form_obstructed:
            return "finite first-integral closed-form route is obstructed"
        if self.regularized_atlas_route:
            return "regularized locally finite atlas route remains conditional"
        if self.infinite_series_route:
            return "Sundman-style infinite-series route remains conditional"
        return "closed-form class is not yet formalized"


@dataclass(frozen=True)
class GeneralSolutionRequirementStatus:
    """One requirement in the full general-solution theorem target."""

    requirement: str
    certified: bool
    reason: str
    witness_field: str | None = None
    required: str | None = None
    observed: str | None = None
    blocking_obligations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "certified", self.certified is True)


@dataclass(frozen=True)
class ClosedFormProofObligationDetail:
    """Machine-readable proof obligation used by top-level theorem witnesses."""

    obligation: str
    certified: bool
    required: str | None = None
    observed: str | None = None
    witness_field: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "certified", self.certified is True)


@dataclass(frozen=True)
class ClosedFormFunctionClassWitnessCertificate:
    """Typed definition of the function class meant by "closed form"."""

    class_id: str = "unspecified"
    representation_semantics_certified: bool = False
    evaluation_semantics_certified: bool = False
    convergence_semantics_certified: bool = False
    equation_verification_semantics_certified: bool = False
    witness_source: str = "manual_closed_form_function_class"

    @property
    def normalized_class(self) -> str:
        return _normalize_closed_form_class(self.class_id)

    @property
    def class_id_allowed(self) -> bool:
        return self.normalized_class != "unspecified"

    @property
    def function_class_definition_certified(self) -> bool:
        return bool(
            self.class_id_allowed
            and self.representation_semantics_certified
            and self.evaluation_semantics_certified
            and self.convergence_semantics_certified
            and self.equation_verification_semantics_certified
        )

    @property
    def theorem_compatible_infinite_series_class_certified(self) -> bool:
        return bool(
            self.function_class_definition_certified
            and self.normalized_class == "sundman_global_series"
        )

    @property
    def theorem_compatible_regularized_atlas_class_certified(self) -> bool:
        return bool(
            self.function_class_definition_certified
            and self.normalized_class == "regularized_locally_finite_atlas"
        )

    @property
    def function_class_obligation_details(
        self,
    ) -> tuple[ClosedFormProofObligationDetail, ...]:
        return (
            ClosedFormProofObligationDetail(
                obligation="closed_form_function_class_id_allowed",
                certified=self.class_id_allowed,
                required="function class id maps to a known theorem route",
                observed=f"class_id={self.class_id}; normalized_class={self.normalized_class}",
                witness_field="class_id",
            ),
            ClosedFormProofObligationDetail(
                obligation="closed_form_representation_semantics",
                certified=self.representation_semantics_certified,
                required="allowed expressions or series representations are specified",
                observed=(
                    "representation_semantics_certified="
                    f"{self.representation_semantics_certified}"
                ),
                witness_field="representation_semantics_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="closed_form_evaluation_semantics",
                certified=self.evaluation_semantics_certified,
                required="evaluation rule for the representation is specified",
                observed=(
                    "evaluation_semantics_certified="
                    f"{self.evaluation_semantics_certified}"
                ),
                witness_field="evaluation_semantics_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="closed_form_convergence_semantics",
                certified=self.convergence_semantics_certified,
                required="finite, limiting, or series convergence semantics are specified",
                observed=(
                    "convergence_semantics_certified="
                    f"{self.convergence_semantics_certified}"
                ),
                witness_field="convergence_semantics_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="closed_form_equation_verification_semantics",
                certified=self.equation_verification_semantics_certified,
                required="meaning of satisfying Newton's equations is specified for the class",
                observed=(
                    "equation_verification_semantics_certified="
                    f"{self.equation_verification_semantics_certified}"
                ),
                witness_field="equation_verification_semantics_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="closed_form_function_class_definition",
                certified=self.function_class_definition_certified,
                required="all function-class semantics are certified",
                observed=(
                    "function_class_definition_certified="
                    f"{self.function_class_definition_certified}"
                ),
                witness_field="function_class_definition_certified",
            ),
        )

    @property
    def missing_function_class_obligations(self) -> tuple[str, ...]:
        return tuple(
            detail.obligation
            for detail in self.function_class_obligation_details
            if not detail.certified
        )


@dataclass(frozen=True)
class CertificateLanguageSoundnessCertificate:
    """Soundness evidence for the serialized atlas certificate language."""

    ordinary_taylor_sound: bool = False
    levi_civita_sound: bool = False
    spatial_ks_sound: bool = False
    total_stop_sound: bool = False
    fuchsian_stop_sound: bool = False
    generalized_fuchsian_stop_sound: bool = False
    transition_sound: bool = False
    branch_union_sound: bool = False
    chart_chain_sound: bool = False
    verifier_kernel_sound: bool = False
    proof_grade_arithmetic_backend_sound: bool = False
    witness_source: str = "certificate_language_soundness"

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.ordinary_taylor_sound is True
            and self.levi_civita_sound is True
            and self.spatial_ks_sound is True
            and self.fuchsian_stop_sound is True
            and self.generalized_fuchsian_stop_sound is True
            and self.transition_sound is True
            and self.branch_union_sound is True
            and self.chart_chain_sound is True
            and self.verifier_kernel_sound is True
            and self.proof_grade_arithmetic_backend_sound is True
        )

    @property
    def checker_kernel_derived(self) -> bool:
        return bool(
            self.proof_certified
            and self.witness_source
            == "checker_kernel_support_certificate_language_soundness"
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        fields = (
            ("ordinary_taylor_sound", self.ordinary_taylor_sound),
            ("levi_civita_sound", self.levi_civita_sound),
            ("spatial_ks_sound", self.spatial_ks_sound),
            ("fuchsian_stop_sound", self.fuchsian_stop_sound),
            (
                "generalized_fuchsian_stop_sound",
                self.generalized_fuchsian_stop_sound,
            ),
            ("transition_sound", self.transition_sound),
            ("branch_union_sound", self.branch_union_sound),
            ("chart_chain_sound", self.chart_chain_sound),
            ("verifier_kernel_sound", self.verifier_kernel_sound),
            (
                "proof_grade_arithmetic_backend_sound",
                self.proof_grade_arithmetic_backend_sound,
            ),
        )
        return tuple(name for name, certified in fields if certified is not True)


@dataclass(frozen=True)
class ComputableAtlasCertificateEnumerationCertificate:
    """Fair enumeration evidence for computable-input atlas certificates."""

    chart_family_words_enumerated: bool = False
    pair_labels_enumerated: bool = False
    rational_domains_enumerated: bool = False
    truncation_orders_enumerated: bool = False
    rational_or_interval_coefficients_enumerated: bool = False
    rational_tail_budgets_enumerated: bool = False
    generalized_fuchsian_exponent_data_enumerated: bool = False
    fuchsian_selector_constants_enumerated: bool = False
    cauchy_majorants_enumerated: bool = False
    transition_witnesses_enumerated: bool = False
    collision_policy_data_enumerated: bool = False
    independent_checker_dovetailed: bool = False
    dovetailing_fairness_certified: bool = False
    finite_target_query_terminates_certified: bool = False
    source_theorem_id: str = ""
    source_theorem_proof_certified: bool = False
    source_theorem_dimension: int = 0
    source_theorem_input_model: str = ""
    source_total_collision_policy_id: str = ""
    finite_target_chart_families: tuple[str, ...] = ()
    finite_target_allowed_outcomes: tuple[str, ...] = ()
    witness_source: str = "computable_atlas_certificate_enumeration"

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.chart_family_words_enumerated is True
            and self.pair_labels_enumerated is True
            and self.rational_domains_enumerated is True
            and self.truncation_orders_enumerated is True
            and self.rational_or_interval_coefficients_enumerated is True
            and self.rational_tail_budgets_enumerated is True
            and self.generalized_fuchsian_exponent_data_enumerated is True
            and self.fuchsian_selector_constants_enumerated is True
            and self.cauchy_majorants_enumerated is True
            and self.transition_witnesses_enumerated is True
            and self.collision_policy_data_enumerated is True
            and self.independent_checker_dovetailed is True
            and self.dovetailing_fairness_certified is True
            and self.finite_target_query_terminates_certified is True
        )

    @property
    def pointwise_theorem_derived(self) -> bool:
        return bool(
            self.proof_certified
            and self.source_theorem_id == "pointwise_open_time_locally_finite_atlas"
            and self.source_theorem_proof_certified is True
            and self.source_theorem_dimension in {2, 3}
            and (
                "computable" in self.source_theorem_input_model
                or "exact_point" in self.source_theorem_input_model
            )
            and self.source_total_collision_policy_id
            in {
                "maximal_classical_stop",
                "maximal_classical_stop_at_total_collision",
            }
            and (
                set(self.finite_target_chart_families)
                == set(POINTWISE_FINITE_TARGET_CHART_FAMILIES)
            )
            and (
                tuple(self.finite_target_allowed_outcomes)
                == POINTWISE_FINITE_TARGET_ALLOWED_OUTCOMES
            )
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        fields = (
            ("chart_family_words_enumerated", self.chart_family_words_enumerated),
            ("pair_labels_enumerated", self.pair_labels_enumerated),
            ("rational_domains_enumerated", self.rational_domains_enumerated),
            ("truncation_orders_enumerated", self.truncation_orders_enumerated),
            (
                "rational_or_interval_coefficients_enumerated",
                self.rational_or_interval_coefficients_enumerated,
            ),
            ("rational_tail_budgets_enumerated", self.rational_tail_budgets_enumerated),
            (
                "generalized_fuchsian_exponent_data_enumerated",
                self.generalized_fuchsian_exponent_data_enumerated,
            ),
            (
                "fuchsian_selector_constants_enumerated",
                self.fuchsian_selector_constants_enumerated,
            ),
            ("cauchy_majorants_enumerated", self.cauchy_majorants_enumerated),
            ("transition_witnesses_enumerated", self.transition_witnesses_enumerated),
            ("collision_policy_data_enumerated", self.collision_policy_data_enumerated),
            ("independent_checker_dovetailed", self.independent_checker_dovetailed),
            ("dovetailing_fairness_certified", self.dovetailing_fairness_certified),
            (
                "finite_target_query_terminates_certified",
                self.finite_target_query_terminates_certified,
            ),
        )
        return tuple(name for name, certified in fields if certified is not True)


@dataclass(frozen=True)
class MaximalClassicalTotalCollisionPolicyCertificate:
    """Maximal-classical stop semantics for unselected total collision."""

    policy_id: str = "maximal_classical_stop"
    stop_at_unselected_total_collision: bool = False
    selected_continuation_forbidden: bool = False
    maximal_classical_domain_certified: bool = False
    pointwise_theorem_policy_matches: bool = False
    source_theorem_id: str = ""
    source_theorem_proof_certified: bool = False
    source_theorem_dimension: int = 0
    source_theorem_input_model: str = ""
    source_total_collision_policy_id: str = ""
    witness_source: str = "maximal_classical_total_collision_policy"

    @property
    def allowed_policy_id(self) -> bool:
        return self.policy_id in {
            "maximal_classical_stop",
            "maximal_classical_stop_at_total_collision",
        }

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.allowed_policy_id
            and self.stop_at_unselected_total_collision is True
            and self.selected_continuation_forbidden is True
            and self.maximal_classical_domain_certified is True
            and self.pointwise_theorem_policy_matches is True
            and self.pointwise_theorem_derived
        )

    @property
    def pointwise_theorem_derived(self) -> bool:
        return bool(
            self.source_theorem_id == "pointwise_open_time_locally_finite_atlas"
            and self.source_theorem_proof_certified is True
            and self.source_theorem_dimension in {2, 3}
            and (
                "computable" in self.source_theorem_input_model
                or "exact_point" in self.source_theorem_input_model
            )
            and self.source_total_collision_policy_id == self.policy_id
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        fields = (
            ("maximal_classical_policy_id", self.allowed_policy_id),
            (
                "stop_at_unselected_total_collision",
                self.stop_at_unselected_total_collision,
            ),
            ("selected_continuation_forbidden", self.selected_continuation_forbidden),
            ("maximal_classical_domain_certified", self.maximal_classical_domain_certified),
            ("pointwise_theorem_policy_matches", self.pointwise_theorem_policy_matches),
            ("pointwise_theorem_policy_source", self.pointwise_theorem_derived),
        )
        return tuple(name for name, certified in fields if certified is not True)


@dataclass(frozen=True)
class PointwiseRegularizedAtlasClosedFormTheoremCertificate:
    """Closed-form theorem for exact/computable inputs via pointwise atlas proof."""

    closed_form_class_id: str
    pointwise_open_time_theorem: Any
    certificate_language_soundness: CertificateLanguageSoundnessCertificate
    computable_certificate_enumeration: ComputableAtlasCertificateEnumerationCertificate
    maximal_classical_total_collision_policy: (
        MaximalClassicalTotalCollisionPolicyCertificate
    )
    statement: str
    proof_sketch: str
    obligations: tuple[TheoremPipelineObligation, ...]
    chart_primitives: tuple[str, ...] = ()
    finite_target_certificate_outcomes: tuple[str, ...] = ()
    endpoint_regime_partition_required: bool = False
    theorem_id: str = "pointwise_regularized_atlas_closed_form"

    @property
    def normalized_class(self) -> str:
        return _normalize_closed_form_class(self.closed_form_class_id)

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.theorem_id == "pointwise_regularized_atlas_closed_form"
            and self.normalized_class == "regularized_locally_finite_atlas"
            and type(
                self.pointwise_open_time_theorem,
            )
            is PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate
            and getattr(self.pointwise_open_time_theorem, "proof_certified", False)
            is True
            and type(
                self.certificate_language_soundness,
            )
            is CertificateLanguageSoundnessCertificate
            and getattr(
                self.certificate_language_soundness,
                "proof_certified",
                False,
            )
            is True
            and getattr(
                self.certificate_language_soundness,
                "checker_kernel_derived",
                False,
            )
            is True
            and type(
                self.computable_certificate_enumeration,
            )
            is ComputableAtlasCertificateEnumerationCertificate
            and getattr(
                self.computable_certificate_enumeration,
                "proof_certified",
                False,
            )
            is True
            and getattr(
                self.computable_certificate_enumeration,
                "pointwise_theorem_derived",
                False,
            )
            is True
            and _enumeration_source_matches_pointwise_theorem(
                self.computable_certificate_enumeration,
                self.pointwise_open_time_theorem,
            )
            and type(
                self.maximal_classical_total_collision_policy,
            )
            is MaximalClassicalTotalCollisionPolicyCertificate
            and getattr(
                self.maximal_classical_total_collision_policy,
                "proof_certified",
                False,
            )
            is True
            and _policy_source_matches_pointwise_theorem(
                self.maximal_classical_total_collision_policy,
                self.pointwise_open_time_theorem,
            )
            and self.statement
            and self.proof_sketch
            and _pipeline_obligation_ledger_certified(self.obligations)
            and self.obligation_manifest_certified
            and self.chart_primitive_scope_certified
            and self.finite_target_outcome_scope_certified
            and not self.endpoint_regime_partition_required
        )

    @property
    def certified(self) -> bool:
        return self.proof_certified

    @property
    def obligation_manifest_certified(self) -> bool:
        return _pipeline_obligation_manifest_exact(
            self.obligations,
            POINTWISE_REGULARIZED_ATLAS_REQUIRED_OBLIGATIONS,
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.theorem_id != "pointwise_regularized_atlas_closed_form":
            missing.append("pointwise_regularized_atlas_closed_form_theorem_id")
        if self.normalized_class != "regularized_locally_finite_atlas":
            missing.append("regularized_locally_finite_atlas_class")
        if type(
            self.pointwise_open_time_theorem,
        ) is not PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate:
            missing.append("pointwise_open_time_atlas_constructor_certificate")
        if getattr(self.pointwise_open_time_theorem, "proof_certified", False) is not True:
            missing.append("pointwise_open_time_atlas_proof")
        if type(
            self.certificate_language_soundness,
        ) is not CertificateLanguageSoundnessCertificate:
            missing.append("certificate_language_soundness_constructor_certificate")
        if (
            getattr(self.certificate_language_soundness, "proof_certified", False)
            is not True
        ):
            missing.append("certificate_language_soundness")
            missing.extend(
                f"certificate_language_soundness:{obligation}"
                for obligation in getattr(
                    self.certificate_language_soundness,
                    "missing_obligations",
                    (),
                )
            )
        if (
            getattr(self.certificate_language_soundness, "proof_certified", False)
            is True
            and getattr(
                self.certificate_language_soundness,
                "checker_kernel_derived",
                False,
            )
            is not True
        ):
            missing.append("certificate_language_soundness_checker_kernel_derived")
        if type(
            self.computable_certificate_enumeration,
        ) is not ComputableAtlasCertificateEnumerationCertificate:
            missing.append("computable_atlas_certificate_enumeration_constructor_certificate")
        if (
            getattr(self.computable_certificate_enumeration, "proof_certified", False)
            is not True
        ):
            missing.append("computable_atlas_certificate_enumeration")
            missing.extend(
                f"computable_atlas_certificate_enumeration:{obligation}"
                for obligation in getattr(
                    self.computable_certificate_enumeration,
                    "missing_obligations",
                    (),
                )
            )
        if (
            getattr(self.computable_certificate_enumeration, "proof_certified", False)
            is True
            and getattr(
                self.computable_certificate_enumeration,
                "pointwise_theorem_derived",
                False,
            )
            is not True
        ):
            missing.append(
                "computable_atlas_certificate_enumeration_pointwise_theorem_derived"
            )
        if (
            type(
                self.computable_certificate_enumeration,
            )
            is ComputableAtlasCertificateEnumerationCertificate
            and getattr(
                self.computable_certificate_enumeration,
                "pointwise_theorem_derived",
                False,
            )
            is True
            and not _enumeration_source_matches_pointwise_theorem(
                self.computable_certificate_enumeration,
                self.pointwise_open_time_theorem,
            )
        ):
            missing.append(
                "computable_atlas_certificate_enumeration_source_matches_theorem"
            )
        if type(
            self.maximal_classical_total_collision_policy,
        ) is not MaximalClassicalTotalCollisionPolicyCertificate:
            missing.append("maximal_classical_total_collision_policy_constructor_certificate")
        if (
            getattr(
                self.maximal_classical_total_collision_policy,
                "proof_certified",
                False,
            )
            is not True
        ):
            missing.append("maximal_classical_total_collision_policy")
            missing.extend(
                f"maximal_classical_total_collision_policy:{obligation}"
                for obligation in getattr(
                    self.maximal_classical_total_collision_policy,
                    "missing_obligations",
                    (),
                )
            )
        if (
            type(
                self.maximal_classical_total_collision_policy,
            )
            is MaximalClassicalTotalCollisionPolicyCertificate
            and getattr(
                self.maximal_classical_total_collision_policy,
                "pointwise_theorem_derived",
                False,
            )
            is True
            and not _policy_source_matches_pointwise_theorem(
                self.maximal_classical_total_collision_policy,
                self.pointwise_open_time_theorem,
            )
        ):
            missing.append("maximal_classical_total_collision_policy_source_matches_theorem")
        missing.extend(
            _pipeline_obligation_ledger_missing(
                self.obligations,
                ledger_name="pointwise_regularized_atlas_closed_form",
            )
        )
        if not self.obligation_manifest_certified:
            missing.append("pointwise_regularized_atlas_closed_form_obligation_manifest")
        if not self.chart_primitive_scope_certified:
            missing.append("pointwise_closed_form_chart_primitives")
        if not self.finite_target_outcome_scope_certified:
            missing.append("pointwise_closed_form_finite_target_outcomes")
        if self.endpoint_regime_partition_required:
            missing.append("endpoint_regime_partition_not_required")
        return tuple(dict.fromkeys(str(obligation) for obligation in missing))

    @property
    def route_summary(self) -> str:
        if self.proof_certified:
            return (
                "pointwise regularized locally finite atlas closed-form theorem "
                "certified for exact/computable inputs"
            )
        return "pointwise closed-form theorem has missing soundness/enumeration/policy obligations"

    @property
    def chart_primitive_scope_certified(self) -> bool:
        return set(self.chart_primitives) == {
            "ordinary_taylor",
            "planar_levi_civita_binary",
            "spatial_ks_binary",
            "generalized_fuchsian_puiseux_log_total_stop",
        }

    @property
    def finite_target_outcome_scope_certified(self) -> bool:
        return tuple(self.finite_target_certificate_outcomes) == (
            "finite_ordinary_lc_ks_chart_chain_reaches_target",
            "finite_ordinary_lc_ks_total_stop_chain_certifies_first_unselected_total_collision",
        )


@dataclass(frozen=True)
class GeneralSolutionScopeWitnessCertificate:
    """Typed witness for the arbitrary-data and all-time scope of a general solution."""

    arbitrary_positive_masses_certified: bool = False
    arbitrary_noncollision_initial_data_certified: bool = False
    all_real_target_times_certified: bool = False
    lift_construct_project_verify_certified: bool = False
    newton_equations_full_interval_certified: bool = False
    witness_source: str = "manual"

    @property
    def general_solution_scope_certified(self) -> bool:
        return (
            self.arbitrary_positive_masses_certified is True
            and self.arbitrary_noncollision_initial_data_certified is True
            and self.all_real_target_times_certified is True
            and self.lift_construct_project_verify_certified is True
            and self.newton_equations_full_interval_certified is True
        )

    @property
    def requirement_statuses(self) -> tuple[GeneralSolutionRequirementStatus, ...]:
        return (
            GeneralSolutionRequirementStatus(
                requirement="arbitrary_positive_masses",
                certified=self.arbitrary_positive_masses_certified,
                reason="the theorem must cover every positive mass triple, not only tested masses",
                witness_field="arbitrary_positive_masses_certified",
                required="all positive masses are covered by the proof",
                observed=f"arbitrary_positive_masses_certified={self.arbitrary_positive_masses_certified}",
            ),
            GeneralSolutionRequirementStatus(
                requirement="arbitrary_noncollision_initial_data",
                certified=self.arbitrary_noncollision_initial_data_certified,
                reason="the theorem must cover every non-collision initial position/velocity state",
                witness_field="arbitrary_noncollision_initial_data_certified",
                required="all non-collision initial states are covered by the proof",
                observed=(
                    "arbitrary_noncollision_initial_data_certified="
                    f"{self.arbitrary_noncollision_initial_data_certified}"
                ),
            ),
            GeneralSolutionRequirementStatus(
                requirement="all_real_target_times",
                certified=self.all_real_target_times_certified,
                reason="the theorem must return positions for every real target time in the claimed continuation",
                witness_field="all_real_target_times_certified",
                required="all real target times in the claimed continuation are covered",
                observed=f"all_real_target_times_certified={self.all_real_target_times_certified}",
            ),
            GeneralSolutionRequirementStatus(
                requirement="lift_construct_project_verify_pipeline",
                certified=self.lift_construct_project_verify_certified,
                reason="the lift, construction, projection, and Newtonian verification steps must all be certified",
                witness_field="lift_construct_project_verify_certified",
                required="lift, construction, projection, and verification are certified as one pipeline",
                observed=(
                    "lift_construct_project_verify_certified="
                    f"{self.lift_construct_project_verify_certified}"
                ),
            ),
            GeneralSolutionRequirementStatus(
                requirement="newton_equations_full_interval",
                certified=self.newton_equations_full_interval_certified,
                reason="projected positions must satisfy Newton's equations on the full claimed interval",
                witness_field="newton_equations_full_interval_certified",
                required="projected positions satisfy Newton's equations over the full claimed interval",
                observed=(
                    "newton_equations_full_interval_certified="
                    f"{self.newton_equations_full_interval_certified}"
                ),
            ),
        )

    @property
    def missing_scope_requirements(self) -> tuple[str, ...]:
        return tuple(status.requirement for status in self.requirement_statuses if not status.certified)


@dataclass(frozen=True)
class CompactTimeRealLineBijectionWitnessCertificate:
    """Typed evidence that compact physical time covers every real target time."""

    positive_rate_parameter_certified: bool = False
    forward_map_all_real_certified: bool = False
    inverse_map_open_interval_certified: bool = False
    strict_monotonicity_certified: bool = False
    endpoint_limits_certified: bool = False
    witness_source: str = "manual_compact_time_real_line_bijection"

    @property
    def compact_time_real_line_bijection_certified(self) -> bool:
        return bool(
            self.positive_rate_parameter_certified
            and self.forward_map_all_real_certified
            and self.inverse_map_open_interval_certified
            and self.strict_monotonicity_certified
            and self.endpoint_limits_certified
        )

    @property
    def compact_time_obligation_details(
        self,
    ) -> tuple[ClosedFormProofObligationDetail, ...]:
        return (
            ClosedFormProofObligationDetail(
                obligation="compact_time_positive_rate_parameter",
                certified=self.positive_rate_parameter_certified,
                required="compact-time rate parameter is positive",
                observed=(
                    "positive_rate_parameter_certified="
                    f"{self.positive_rate_parameter_certified}"
                ),
                witness_field="positive_rate_parameter_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="compact_time_forward_map_all_real",
                certified=self.forward_map_all_real_certified,
                required="u = tanh(rate * t) is defined for every real physical time",
                observed=(
                    "forward_map_all_real_certified="
                    f"{self.forward_map_all_real_certified}"
                ),
                witness_field="forward_map_all_real_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="compact_time_inverse_map_open_interval",
                certified=self.inverse_map_open_interval_certified,
                required="t = atanh(u) / rate is defined for every u in (-1, 1)",
                observed=(
                    "inverse_map_open_interval_certified="
                    f"{self.inverse_map_open_interval_certified}"
                ),
                witness_field="inverse_map_open_interval_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="compact_time_strict_monotonicity",
                certified=self.strict_monotonicity_certified,
                required="compact-time map is strictly monotone and invertible",
                observed=(
                    "strict_monotonicity_certified="
                    f"{self.strict_monotonicity_certified}"
                ),
                witness_field="strict_monotonicity_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="compact_time_endpoint_limits",
                certified=self.endpoint_limits_certified,
                required="limits t -> +/-infinity map to compact endpoints +/-1",
                observed=f"endpoint_limits_certified={self.endpoint_limits_certified}",
                witness_field="endpoint_limits_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="compact_time_real_line_bijection_witness",
                certified=self.compact_time_real_line_bijection_certified,
                required="all compact-time real-line bijection subclaims are certified",
                observed=(
                    "compact_time_real_line_bijection_certified="
                    f"{self.compact_time_real_line_bijection_certified}"
                ),
                witness_field="compact_time_real_line_bijection_certified",
            ),
        )

    @property
    def missing_compact_time_obligations(self) -> tuple[str, ...]:
        return tuple(
            detail.obligation
            for detail in self.compact_time_obligation_details
            if not detail.certified
        )


@dataclass(frozen=True)
class SundmanTimeTargetingGlobalWitnessCertificate:
    """Typed evidence that Sundman time can target every claimed physical time."""

    compact_sundman_parameter_domain_certified: bool = False
    positive_physical_time_derivative_certified: bool = False
    finite_target_bracketing_certified: bool = False
    target_interval_evaluation_certified: bool = False
    global_target_range_certified: bool = False
    witness_source: str = "manual_sundman_time_targeting"

    @property
    def sundman_time_targeting_global_certified(self) -> bool:
        return bool(
            self.compact_sundman_parameter_domain_certified
            and self.positive_physical_time_derivative_certified
            and self.finite_target_bracketing_certified
            and self.target_interval_evaluation_certified
            and self.global_target_range_certified
        )

    @property
    def sundman_target_obligation_details(
        self,
    ) -> tuple[ClosedFormProofObligationDetail, ...]:
        return (
            ClosedFormProofObligationDetail(
                obligation="sundman_target_compact_parameter_domain",
                certified=self.compact_sundman_parameter_domain_certified,
                required="compactified Sundman parameter domain supports target search",
                observed=(
                    "compact_sundman_parameter_domain_certified="
                    f"{self.compact_sundman_parameter_domain_certified}"
                ),
                witness_field="compact_sundman_parameter_domain_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="sundman_target_positive_time_derivative",
                certified=self.positive_physical_time_derivative_certified,
                required="physical time is strictly monotone along certified Sundman charts",
                observed=(
                    "positive_physical_time_derivative_certified="
                    f"{self.positive_physical_time_derivative_certified}"
                ),
                witness_field="positive_physical_time_derivative_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="sundman_target_finite_time_bracketing",
                certified=self.finite_target_bracketing_certified,
                required="every finite target time is enclosed by a certified time bracket",
                observed=(
                    "finite_target_bracketing_certified="
                    f"{self.finite_target_bracketing_certified}"
                ),
                witness_field="finite_target_bracketing_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="sundman_target_interval_evaluation",
                certified=self.target_interval_evaluation_certified,
                required="target state is evaluated over the certified target interval",
                observed=(
                    "target_interval_evaluation_certified="
                    f"{self.target_interval_evaluation_certified}"
                ),
                witness_field="target_interval_evaluation_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="sundman_target_global_range",
                certified=self.global_target_range_certified,
                required="targeting proof covers the full claimed physical-time range",
                observed=f"global_target_range_certified={self.global_target_range_certified}",
                witness_field="global_target_range_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="sundman_time_targeting_global_witness",
                certified=self.sundman_time_targeting_global_certified,
                required="all Sundman target-time subclaims are certified",
                observed=(
                    "sundman_time_targeting_global_certified="
                    f"{self.sundman_time_targeting_global_certified}"
                ),
                witness_field="sundman_time_targeting_global_certified",
            ),
        )

    @property
    def missing_sundman_target_obligations(self) -> tuple[str, ...]:
        return tuple(
            detail.obligation
            for detail in self.sundman_target_obligation_details
            if not detail.certified
        )


@dataclass(frozen=True)
class InertialProjectionWitnessCertificate:
    """Typed evidence that the lifted solution projects back to inertial coordinates."""

    center_of_mass_affine_motion_certified: bool = False
    reduced_to_inertial_coordinate_map_certified: bool = False
    physical_time_series_compatibility_certified: bool = False
    interval_projection_containment_certified: bool = False
    mass_weighted_reconstruction_certified: bool = False
    witness_source: str = "manual_inertial_projection"

    @property
    def inertial_projection_certified(self) -> bool:
        return bool(
            self.center_of_mass_affine_motion_certified
            and self.reduced_to_inertial_coordinate_map_certified
            and self.physical_time_series_compatibility_certified
            and self.interval_projection_containment_certified
            and self.mass_weighted_reconstruction_certified
        )

    @property
    def inertial_projection_obligation_details(
        self,
    ) -> tuple[ClosedFormProofObligationDetail, ...]:
        return (
            ClosedFormProofObligationDetail(
                obligation="inertial_projection_center_of_mass_affine_motion",
                certified=self.center_of_mass_affine_motion_certified,
                required="center of mass follows the certified affine inertial motion",
                observed=(
                    "center_of_mass_affine_motion_certified="
                    f"{self.center_of_mass_affine_motion_certified}"
                ),
                witness_field="center_of_mass_affine_motion_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="inertial_projection_coordinate_map",
                certified=self.reduced_to_inertial_coordinate_map_certified,
                required="reduced coordinates are mapped back to inertial body coordinates",
                observed=(
                    "reduced_to_inertial_coordinate_map_certified="
                    f"{self.reduced_to_inertial_coordinate_map_certified}"
                ),
                witness_field="reduced_to_inertial_coordinate_map_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="inertial_projection_physical_time_series",
                certified=self.physical_time_series_compatibility_certified,
                required="projection uses the certified physical-time series, not fictitious time",
                observed=(
                    "physical_time_series_compatibility_certified="
                    f"{self.physical_time_series_compatibility_certified}"
                ),
                witness_field="physical_time_series_compatibility_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="inertial_projection_interval_containment",
                certified=self.interval_projection_containment_certified,
                required="projected interval enclosures contain the reconstructed inertial states",
                observed=(
                    "interval_projection_containment_certified="
                    f"{self.interval_projection_containment_certified}"
                ),
                witness_field="interval_projection_containment_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="inertial_projection_mass_weighted_reconstruction",
                certified=self.mass_weighted_reconstruction_certified,
                required="mass-weighted reconstruction preserves the center-of-mass constraints",
                observed=(
                    "mass_weighted_reconstruction_certified="
                    f"{self.mass_weighted_reconstruction_certified}"
                ),
                witness_field="mass_weighted_reconstruction_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="inertial_projection_witness",
                certified=self.inertial_projection_certified,
                required="all inertial-projection subclaims are certified",
                observed=(
                    "inertial_projection_certified="
                    f"{self.inertial_projection_certified}"
                ),
                witness_field="inertial_projection_certified",
            ),
        )

    @property
    def missing_inertial_projection_obligations(self) -> tuple[str, ...]:
        return tuple(
            detail.obligation
            for detail in self.inertial_projection_obligation_details
            if not detail.certified
        )


@dataclass(frozen=True)
class GeneralSolutionTheoremScopeWitnessCertificate:
    """Granular theorem-scope witness for arbitrary data and all target times."""

    positive_mass_domain_quantified_certified: bool = False
    mass_parameter_regularity_certified: bool = False
    noncollision_initial_domain_quantified_certified: bool = False
    center_of_mass_reduction_global_certified: bool = False
    interval_initial_data_lift_global_certified: bool = False
    compact_time_real_line_bijection_certified: bool = False
    compact_time_real_line_bijection_witness: (
        CompactTimeRealLineBijectionWitnessCertificate | None
    ) = None
    sundman_time_targeting_global_certified: bool = False
    sundman_time_targeting_global_witness: (
        SundmanTimeTargetingGlobalWitnessCertificate | None
    ) = None
    compact_sundman_lift_certified: bool = False
    global_series_construction_scope_certified: bool = False
    inertial_projection_certified: bool = False
    inertial_projection_witness: InertialProjectionWitnessCertificate | None = None
    chain_rule_newton_equations_certified: bool = False
    full_interval_residual_verification_certified: bool = False
    witness_source: str = "manual_theorem_scope"

    @property
    def arbitrary_positive_masses_certified(self) -> bool:
        return bool(
            self.positive_mass_domain_quantified_certified
            and self.mass_parameter_regularity_certified
        )

    @property
    def arbitrary_noncollision_initial_data_certified(self) -> bool:
        return bool(
            self.noncollision_initial_domain_quantified_certified
            and self.center_of_mass_reduction_global_certified
            and self.interval_initial_data_lift_global_certified
        )

    @property
    def all_real_target_times_certified(self) -> bool:
        return bool(
            self.compact_time_real_line_bijection_effectively_certified
            and self.sundman_time_targeting_effectively_certified
        )

    @property
    def compact_time_real_line_bijection_effectively_certified(self) -> bool:
        if self.compact_time_real_line_bijection_witness is not None:
            return (
                self.compact_time_real_line_bijection_witness
                .compact_time_real_line_bijection_certified
            )
        return self.compact_time_real_line_bijection_certified

    @property
    def sundman_time_targeting_effectively_certified(self) -> bool:
        if self.sundman_time_targeting_global_witness is not None:
            return (
                self.sundman_time_targeting_global_witness
                .sundman_time_targeting_global_certified
            )
        return self.sundman_time_targeting_global_certified

    @property
    def inertial_projection_effectively_certified(self) -> bool:
        if self.inertial_projection_witness is not None:
            return self.inertial_projection_witness.inertial_projection_certified
        return self.inertial_projection_certified

    @property
    def lift_construct_project_verify_certified(self) -> bool:
        return bool(
            self.compact_sundman_lift_certified
            and self.global_series_construction_scope_certified
            and self.inertial_projection_effectively_certified
        )

    @property
    def newton_equations_full_interval_certified(self) -> bool:
        return bool(
            self.chain_rule_newton_equations_certified
            and self.full_interval_residual_verification_certified
        )

    @property
    def general_solution_scope_certified(self) -> bool:
        return bool(
            self.arbitrary_positive_masses_certified
            and self.arbitrary_noncollision_initial_data_certified
            and self.all_real_target_times_certified
            and self.lift_construct_project_verify_certified
            and self.newton_equations_full_interval_certified
        )

    @property
    def scope_obligation_details(self) -> tuple[ClosedFormProofObligationDetail, ...]:
        compact_time_details = (
            ()
            if self.compact_time_real_line_bijection_witness is None
            else self.compact_time_real_line_bijection_witness.compact_time_obligation_details
        )
        sundman_target_details = (
            ()
            if self.sundman_time_targeting_global_witness is None
            else self.sundman_time_targeting_global_witness.sundman_target_obligation_details
        )
        inertial_projection_details = (
            ()
            if self.inertial_projection_witness is None
            else self.inertial_projection_witness.inertial_projection_obligation_details
        )
        return (
            self._proof_detail(
                "positive_mass_domain_quantified",
                self.positive_mass_domain_quantified_certified,
                "proof quantifies over every positive mass triple",
                "positive_mass_domain_quantified_certified",
            ),
            self._proof_detail(
                "mass_parameter_regularity",
                self.mass_parameter_regularity_certified,
                "mass-dependent construction remains regular on the positive mass domain",
                "mass_parameter_regularity_certified",
            ),
            self._proof_detail(
                "noncollision_initial_domain_quantified",
                self.noncollision_initial_domain_quantified_certified,
                "proof quantifies over every non-collision initial state",
                "noncollision_initial_domain_quantified_certified",
            ),
            self._proof_detail(
                "center_of_mass_reduction_global",
                self.center_of_mass_reduction_global_certified,
                "center-of-mass reduction and reconstruction are certified for the full data domain",
                "center_of_mass_reduction_global_certified",
            ),
            self._proof_detail(
                "interval_initial_data_lift_global",
                self.interval_initial_data_lift_global_certified,
                "interval initial-data lift covers arbitrary non-collision input neighborhoods",
                "interval_initial_data_lift_global_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="compact_time_real_line_bijection",
                certified=self.compact_time_real_line_bijection_effectively_certified,
                required="compact physical-time variable covers every real target time",
                observed=(
                    "compact_time_real_line_bijection_certified="
                    f"{self.compact_time_real_line_bijection_certified}; "
                    "typed_witness_certified="
                    f"{self.compact_time_real_line_bijection_effectively_certified}"
                ),
                witness_field="compact_time_real_line_bijection_certified",
            ),
            *compact_time_details,
            ClosedFormProofObligationDetail(
                obligation="sundman_time_targeting_global",
                certified=self.sundman_time_targeting_effectively_certified,
                required="Sundman target-time map reaches every claimed physical target time",
                observed=(
                    "sundman_time_targeting_global_certified="
                    f"{self.sundman_time_targeting_global_certified}; "
                    "typed_witness_certified="
                    f"{self.sundman_time_targeting_effectively_certified}"
                ),
                witness_field="sundman_time_targeting_global_certified",
            ),
            *sundman_target_details,
            self._proof_detail(
                "compact_sundman_lift",
                self.compact_sundman_lift_certified,
                "initial-value problem is lifted into compactified Sundman variables",
                "compact_sundman_lift_certified",
            ),
            self._proof_detail(
                "global_series_construction_scope",
                self.global_series_construction_scope_certified,
                "global series construction is certified uniformly over the theorem scope",
                "global_series_construction_scope_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="inertial_projection",
                certified=self.inertial_projection_effectively_certified,
                required="constructed reduced solution projects back to inertial body coordinates",
                observed=(
                    "inertial_projection_certified="
                    f"{self.inertial_projection_certified}; "
                    "typed_witness_certified="
                    f"{self.inertial_projection_effectively_certified}"
                ),
                witness_field="inertial_projection_certified",
            ),
            *inertial_projection_details,
            self._proof_detail(
                "chain_rule_newton_equations",
                self.chain_rule_newton_equations_certified,
                "lifted compact-Sundman equations imply Newton's equations after projection",
                "chain_rule_newton_equations_certified",
            ),
            self._proof_detail(
                "full_interval_residual_verification",
                self.full_interval_residual_verification_certified,
                "Newton-equation residual verification holds over the full claimed interval",
                "full_interval_residual_verification_certified",
            ),
        )

    def _proof_detail(
        self,
        obligation: str,
        certified: bool,
        required: str,
        witness_field: str,
    ) -> ClosedFormProofObligationDetail:
        return ClosedFormProofObligationDetail(
            obligation=obligation,
            certified=bool(certified),
            required=required,
            observed=f"{witness_field}={bool(certified)}",
            witness_field=witness_field,
        )

    def _missing_obligations(self, obligation_ids: tuple[str, ...]) -> tuple[str, ...]:
        details = {detail.obligation: detail for detail in self.scope_obligation_details}
        return tuple(
            obligation_id
            for obligation_id in obligation_ids
            if not details[obligation_id].certified
        )

    @property
    def requirement_statuses(self) -> tuple[GeneralSolutionRequirementStatus, ...]:
        positive_mass_blockers = self._missing_obligations(
            ("positive_mass_domain_quantified", "mass_parameter_regularity")
        )
        noncollision_blockers = self._missing_obligations(
            (
                "noncollision_initial_domain_quantified",
                "center_of_mass_reduction_global",
                "interval_initial_data_lift_global",
            )
        )
        time_blockers = self._missing_obligations(
            self._time_obligation_ids()
        )
        pipeline_blockers = self._missing_obligations(
            self._pipeline_obligation_ids()
        )
        newton_blockers = self._missing_obligations(
            ("chain_rule_newton_equations", "full_interval_residual_verification")
        )
        return (
            GeneralSolutionRequirementStatus(
                requirement="arbitrary_positive_masses",
                certified=self.arbitrary_positive_masses_certified,
                reason="the theorem must cover every positive mass triple, not only tested masses",
                witness_field="positive_mass_domain_quantified_certified and mass_parameter_regularity_certified",
                required="all positive masses are covered by the proof",
                observed=f"blocking_obligations={positive_mass_blockers}",
                blocking_obligations=positive_mass_blockers,
            ),
            GeneralSolutionRequirementStatus(
                requirement="arbitrary_noncollision_initial_data",
                certified=self.arbitrary_noncollision_initial_data_certified,
                reason="the theorem must cover every non-collision initial position/velocity state",
                witness_field=(
                    "noncollision_initial_domain_quantified_certified, "
                    "center_of_mass_reduction_global_certified, "
                    "interval_initial_data_lift_global_certified"
                ),
                required="all non-collision initial states are covered by the proof",
                observed=f"blocking_obligations={noncollision_blockers}",
                blocking_obligations=noncollision_blockers,
            ),
            GeneralSolutionRequirementStatus(
                requirement="all_real_target_times",
                certified=self.all_real_target_times_certified,
                reason="the theorem must return positions for every real target time in the claimed continuation",
                witness_field=(
                    "compact_time_real_line_bijection_certified and "
                    "sundman_time_targeting_global_certified"
                ),
                required="all real target times in the claimed continuation are covered",
                observed=f"blocking_obligations={time_blockers}",
                blocking_obligations=time_blockers,
            ),
            GeneralSolutionRequirementStatus(
                requirement="lift_construct_project_verify_pipeline",
                certified=self.lift_construct_project_verify_certified,
                reason="the lift, construction, projection, and Newtonian verification steps must all be certified",
                witness_field=(
                    "compact_sundman_lift_certified, "
                    "global_series_construction_scope_certified, "
                    "inertial_projection_certified"
                ),
                required="lift, construction, projection, and verification are certified as one pipeline",
                observed=f"blocking_obligations={pipeline_blockers}",
                blocking_obligations=pipeline_blockers,
            ),
            GeneralSolutionRequirementStatus(
                requirement="newton_equations_full_interval",
                certified=self.newton_equations_full_interval_certified,
                reason="projected positions must satisfy Newton's equations on the full claimed interval",
                witness_field=(
                    "chain_rule_newton_equations_certified and "
                    "full_interval_residual_verification_certified"
                ),
                required="projected positions satisfy Newton's equations over the full claimed interval",
                observed=f"blocking_obligations={newton_blockers}",
                blocking_obligations=newton_blockers,
            ),
        )

    @property
    def missing_scope_requirements(self) -> tuple[str, ...]:
        return tuple(status.requirement for status in self.requirement_statuses if not status.certified)

    @property
    def missing_scope_obligations(self) -> tuple[str, ...]:
        return tuple(
            detail.obligation for detail in self.scope_obligation_details if not detail.certified
        )

    def _time_obligation_ids(self) -> tuple[str, ...]:
        compact_time_obligations = (
            ("compact_time_real_line_bijection",)
            if self.compact_time_real_line_bijection_witness is None
            else self.compact_time_real_line_bijection_witness.missing_compact_time_obligations
        )
        sundman_target_obligations = (
            ("sundman_time_targeting_global",)
            if self.sundman_time_targeting_global_witness is None
            else (
                self.sundman_time_targeting_global_witness
                .missing_sundman_target_obligations
            )
        )
        return (*compact_time_obligations, *sundman_target_obligations)

    def _pipeline_obligation_ids(self) -> tuple[str, ...]:
        inertial_projection_obligations = (
            ("inertial_projection",)
            if self.inertial_projection_witness is None
            else self.inertial_projection_witness.missing_inertial_projection_obligations
        )
        return (
            "compact_sundman_lift",
            "global_series_construction_scope",
            *inertial_projection_obligations,
        )


_ALLOWED_TRIPLE_COLLISION_CONTINUATION_CONVENTIONS = (
    "sundman_total_collision_regularization",
    "regularized_second_jet_branch",
)
_REQUIRED_BINARY_COLLISION_PAIRS = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class ZeroAngularMomentumTripleCollisionConventionCertificate:
    """Typed convention for the zero-angular-momentum triple-collision branch."""

    convention_id: str = "unspecified"
    regularized_time_parameter_certified: bool = False
    terminal_collision_value_certified: bool = False
    continuation_selection_rule_certified: bool = False
    witness_source: str = "manual_zero_angular_triple_convention"

    @property
    def convention_id_allowed(self) -> bool:
        return self.convention_id in _ALLOWED_TRIPLE_COLLISION_CONTINUATION_CONVENTIONS

    @property
    def convention_certified(self) -> bool:
        return bool(
            self.convention_id_allowed
            and self.regularized_time_parameter_certified
            and self.terminal_collision_value_certified
            and self.continuation_selection_rule_certified
        )

    @property
    def convention_obligation_details(self) -> tuple[ClosedFormProofObligationDetail, ...]:
        return (
            ClosedFormProofObligationDetail(
                obligation="zero_angular_triple_convention_id_allowed",
                certified=self.convention_id_allowed,
                required=(
                    "convention id is one of "
                    f"{_ALLOWED_TRIPLE_COLLISION_CONTINUATION_CONVENTIONS}"
                ),
                observed=f"convention_id={self.convention_id}",
                witness_field="convention_id",
            ),
            ClosedFormProofObligationDetail(
                obligation="zero_angular_triple_regularized_time_parameter",
                certified=self.regularized_time_parameter_certified,
                required="zero-angular triple branch has a certified regularized time parameter",
                observed=(
                    "regularized_time_parameter_certified="
                    f"{self.regularized_time_parameter_certified}"
                ),
                witness_field="regularized_time_parameter_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="zero_angular_triple_terminal_collision_value",
                certified=self.terminal_collision_value_certified,
                required="triple-collision terminal value is specified in regularized variables",
                observed=(
                    "terminal_collision_value_certified="
                    f"{self.terminal_collision_value_certified}"
                ),
                witness_field="terminal_collision_value_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="zero_angular_triple_continuation_selection_rule",
                certified=self.continuation_selection_rule_certified,
                required=(
                    "continuation branch selection rule is explicit and certified, "
                    "including the regularized second-jet branch datum when that "
                    "convention is used"
                ),
                observed=(
                    "continuation_selection_rule_certified="
                    f"{self.continuation_selection_rule_certified}"
                ),
                witness_field="continuation_selection_rule_certified",
            ),
        )

    @property
    def missing_convention_obligations(self) -> tuple[str, ...]:
        return tuple(
            detail.obligation
            for detail in self.convention_obligation_details
            if not detail.certified
        )


@dataclass(frozen=True)
class BinaryCollisionContinuationWitnessCertificate:
    """Typed binary-collision continuation evidence for all body pairs."""

    regularized_pairs: tuple[tuple[int, int], ...] = ()
    pair_regularization_charts_certified: bool = False
    branch_atlas_certified: bool = False
    projection_back_to_newtonian_certified: bool = False
    regularized_time_parameter_certified: bool = False
    witness_source: str = "manual_binary_collision_continuation"

    @property
    def normalized_regularized_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(tuple(sorted(pair)) for pair in self.regularized_pairs)

    @property
    def regularized_pairs_valid(self) -> bool:
        required_pairs = set(_REQUIRED_BINARY_COLLISION_PAIRS)
        return bool(
            self.normalized_regularized_pairs
            and all(pair in required_pairs for pair in self.normalized_regularized_pairs)
        )

    @property
    def all_required_pairs_covered(self) -> bool:
        return bool(
            self.regularized_pairs_valid
            and set(self.normalized_regularized_pairs) == set(_REQUIRED_BINARY_COLLISION_PAIRS)
        )

    @property
    def all_binary_pairs_regularized_certified(self) -> bool:
        return bool(
            self.all_required_pairs_covered
            and self.pair_regularization_charts_certified
        )

    @property
    def binary_collision_continuation_certified(self) -> bool:
        return bool(
            self.all_binary_pairs_regularized_certified
            and self.branch_atlas_certified
            and self.projection_back_to_newtonian_certified
            and self.regularized_time_parameter_certified
        )

    @property
    def binary_obligation_details(self) -> tuple[ClosedFormProofObligationDetail, ...]:
        return (
            ClosedFormProofObligationDetail(
                obligation="binary_required_pairs_covered",
                certified=self.all_required_pairs_covered,
                required=f"regularized pairs exactly cover {_REQUIRED_BINARY_COLLISION_PAIRS}",
                observed=f"regularized_pairs={self.normalized_regularized_pairs}",
                witness_field="regularized_pairs",
            ),
            ClosedFormProofObligationDetail(
                obligation="binary_pair_regularization_charts",
                certified=self.pair_regularization_charts_certified,
                required="regularized binary-collision charts are certified for each required pair",
                observed=(
                    "pair_regularization_charts_certified="
                    f"{self.pair_regularization_charts_certified}"
                ),
                witness_field="pair_regularization_charts_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="binary_branch_atlas",
                certified=self.branch_atlas_certified,
                required="Levi-Civita branch atlas covers binary collision branches",
                observed=f"branch_atlas_certified={self.branch_atlas_certified}",
                witness_field="branch_atlas_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="binary_projection_back_to_newtonian",
                certified=self.projection_back_to_newtonian_certified,
                required="regularized binary charts project back to Newtonian motion away from collision",
                observed=(
                    "projection_back_to_newtonian_certified="
                    f"{self.projection_back_to_newtonian_certified}"
                ),
                witness_field="projection_back_to_newtonian_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="binary_collision_time_parameter",
                certified=self.regularized_time_parameter_certified,
                required="regularized time parameter gives a certified continuation through binary collision",
                observed=(
                    "regularized_time_parameter_certified="
                    f"{self.regularized_time_parameter_certified}"
                ),
                witness_field="regularized_time_parameter_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="binary_collision_continuation",
                certified=self.binary_collision_continuation_certified,
                required="all typed binary-collision continuation subclaims are certified",
                observed=(
                    "binary_collision_continuation_certified="
                    f"{self.binary_collision_continuation_certified}"
                ),
                witness_field="binary_collision_continuation_certified",
            ),
        )

    @property
    def missing_binary_obligations(self) -> tuple[str, ...]:
        return tuple(
            detail.obligation
            for detail in self.binary_obligation_details
            if not detail.certified
        )


@dataclass(frozen=True)
class NonzeroAngularMomentumTripleExclusionWitnessCertificate:
    """Typed evidence for excluding triple collision on the nonzero-angular branch."""

    nonzero_branch_domain_quantified_certified: bool = False
    centered_angular_momentum_conservation_certified: bool = False
    triple_collision_zero_angular_momentum_lemma_certified: bool = False
    positive_lower_bound_predicate_certified: bool = False
    witness_source: str = "manual_nonzero_angular_triple_exclusion"

    @property
    def nonzero_angular_momentum_triple_exclusion_certified(self) -> bool:
        return bool(
            self.nonzero_branch_domain_quantified_certified
            and self.centered_angular_momentum_conservation_certified
            and self.triple_collision_zero_angular_momentum_lemma_certified
            and self.positive_lower_bound_predicate_certified
        )

    @property
    def nonzero_angular_obligation_details(
        self,
    ) -> tuple[ClosedFormProofObligationDetail, ...]:
        return (
            ClosedFormProofObligationDetail(
                obligation="nonzero_angular_branch_domain_quantified",
                certified=self.nonzero_branch_domain_quantified_certified,
                required="proof quantifies over the whole nonzero centered-angular-momentum branch",
                observed=(
                    "nonzero_branch_domain_quantified_certified="
                    f"{self.nonzero_branch_domain_quantified_certified}"
                ),
                witness_field="nonzero_branch_domain_quantified_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="centered_angular_momentum_conservation",
                certified=self.centered_angular_momentum_conservation_certified,
                required="centered angular momentum is conserved by the lifted/projected flow",
                observed=(
                    "centered_angular_momentum_conservation_certified="
                    f"{self.centered_angular_momentum_conservation_certified}"
                ),
                witness_field="centered_angular_momentum_conservation_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="triple_collision_zero_angular_momentum_lemma",
                certified=self.triple_collision_zero_angular_momentum_lemma_certified,
                required="total collision implies zero translation-reduced angular momentum",
                observed=(
                    "triple_collision_zero_angular_momentum_lemma_certified="
                    f"{self.triple_collision_zero_angular_momentum_lemma_certified}"
                ),
                witness_field="triple_collision_zero_angular_momentum_lemma_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="nonzero_angular_lower_bound_predicate",
                certified=self.positive_lower_bound_predicate_certified,
                required="interval lower-bound predicate soundly proves nonzero angular momentum",
                observed=(
                    "positive_lower_bound_predicate_certified="
                    f"{self.positive_lower_bound_predicate_certified}"
                ),
                witness_field="positive_lower_bound_predicate_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="nonzero_angular_momentum_triple_exclusion",
                certified=self.nonzero_angular_momentum_triple_exclusion_certified,
                required="all typed nonzero-angular triple-exclusion subclaims are certified",
                observed=(
                    "nonzero_angular_momentum_triple_exclusion_certified="
                    f"{self.nonzero_angular_momentum_triple_exclusion_certified}"
                ),
                witness_field="nonzero_angular_momentum_triple_exclusion_certified",
            ),
        )

    @property
    def missing_nonzero_angular_obligations(self) -> tuple[str, ...]:
        return tuple(
            detail.obligation
            for detail in self.nonzero_angular_obligation_details
            if not detail.certified
        )


@dataclass(frozen=True)
class CollisionContinuationWitnessCertificate:
    """Typed collision-continuation evidence for the Sundman global-series route."""

    all_binary_pairs_regularized_certified: bool = False
    binary_branch_atlas_certified: bool = False
    binary_projection_back_to_newtonian_certified: bool = False
    binary_collision_time_parameter_certified: bool = False
    binary_collision_continuation_witness: (
        BinaryCollisionContinuationWitnessCertificate | None
    ) = None
    nonzero_angular_momentum_triple_exclusion_certified: bool = False
    nonzero_angular_momentum_triple_exclusion_witness: (
        NonzeroAngularMomentumTripleExclusionWitnessCertificate | None
    ) = None
    zero_angular_momentum_triple_collision_convention_certified: bool = False
    triple_collision_continuation_convention: str = "unspecified"
    zero_angular_momentum_triple_collision_convention_witness: (
        ZeroAngularMomentumTripleCollisionConventionCertificate | None
    ) = None
    witness_source: str = "manual_collision_continuation"

    @property
    def all_binary_pairs_regularized_effectively_certified(self) -> bool:
        if self.binary_collision_continuation_witness is not None:
            return self.binary_collision_continuation_witness.all_binary_pairs_regularized_certified
        return self.all_binary_pairs_regularized_certified

    @property
    def binary_branch_atlas_effectively_certified(self) -> bool:
        if self.binary_collision_continuation_witness is not None:
            return self.binary_collision_continuation_witness.branch_atlas_certified
        return self.binary_branch_atlas_certified

    @property
    def binary_projection_back_to_newtonian_effectively_certified(self) -> bool:
        if self.binary_collision_continuation_witness is not None:
            return self.binary_collision_continuation_witness.projection_back_to_newtonian_certified
        return self.binary_projection_back_to_newtonian_certified

    @property
    def binary_collision_time_parameter_effectively_certified(self) -> bool:
        if self.binary_collision_continuation_witness is not None:
            return self.binary_collision_continuation_witness.regularized_time_parameter_certified
        return self.binary_collision_time_parameter_certified

    @property
    def binary_collision_continuation_certified(self) -> bool:
        if self.binary_collision_continuation_witness is not None:
            return self.binary_collision_continuation_witness.binary_collision_continuation_certified
        return bool(
            self.all_binary_pairs_regularized_certified
            and self.binary_branch_atlas_certified
            and self.binary_projection_back_to_newtonian_certified
            and self.binary_collision_time_parameter_certified
        )

    @property
    def nonzero_angular_momentum_triple_exclusion_effectively_certified(self) -> bool:
        if self.nonzero_angular_momentum_triple_exclusion_witness is not None:
            return (
                self.nonzero_angular_momentum_triple_exclusion_witness
                .nonzero_angular_momentum_triple_exclusion_certified
            )
        return self.nonzero_angular_momentum_triple_exclusion_certified

    @property
    def effective_triple_collision_continuation_convention(self) -> str:
        if self.zero_angular_momentum_triple_collision_convention_witness is not None:
            return self.zero_angular_momentum_triple_collision_convention_witness.convention_id
        return self.triple_collision_continuation_convention

    @property
    def zero_angular_momentum_triple_convention_effectively_certified(self) -> bool:
        if self.zero_angular_momentum_triple_collision_convention_witness is not None:
            return (
                self.zero_angular_momentum_triple_collision_convention_witness.convention_certified
            )
        return bool(
            self.zero_angular_momentum_triple_collision_convention_certified
            and self.triple_collision_continuation_convention
            in _ALLOWED_TRIPLE_COLLISION_CONTINUATION_CONVENTIONS
        )

    @property
    def triple_collision_continuation_certified(self) -> bool:
        return bool(
            self.nonzero_angular_momentum_triple_exclusion_effectively_certified
            and self.zero_angular_momentum_triple_convention_effectively_certified
        )

    @property
    def collision_continuation_certified(self) -> bool:
        return bool(
            self.binary_collision_continuation_certified
            and self.triple_collision_continuation_certified
        )

    @property
    def collision_continuation_obligations_certified(self) -> bool:
        return self.collision_continuation_certified

    @property
    def global_collision_continuation_certified(self) -> bool:
        return self.collision_continuation_certified

    @property
    def collision_obligation_details(self) -> tuple[ClosedFormProofObligationDetail, ...]:
        binary_details = (
            ()
            if self.binary_collision_continuation_witness is None
            else self.binary_collision_continuation_witness.binary_obligation_details
        )
        nonzero_angular_details = (
            ()
            if self.nonzero_angular_momentum_triple_exclusion_witness is None
            else (
                self.nonzero_angular_momentum_triple_exclusion_witness
                .nonzero_angular_obligation_details
            )
        )
        convention_details = (
            ()
            if self.zero_angular_momentum_triple_collision_convention_witness is None
            else self.zero_angular_momentum_triple_collision_convention_witness.convention_obligation_details
        )
        return (
            ClosedFormProofObligationDetail(
                obligation="all_binary_pairs_regularized",
                certified=self.all_binary_pairs_regularized_effectively_certified,
                required="regularized binary-collision charts cover every pair",
                observed=(
                    "all_binary_pairs_regularized_certified="
                    f"{self.all_binary_pairs_regularized_certified}; "
                    "typed_witness_certified="
                    f"{self.all_binary_pairs_regularized_effectively_certified}"
                ),
                witness_field="all_binary_pairs_regularized_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="binary_branch_atlas",
                certified=self.binary_branch_atlas_effectively_certified,
                required="Levi-Civita branch atlas covers binary collision branches",
                observed=(
                    f"binary_branch_atlas_certified={self.binary_branch_atlas_certified}; "
                    "typed_witness_certified="
                    f"{self.binary_branch_atlas_effectively_certified}"
                ),
                witness_field="binary_branch_atlas_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="binary_projection_back_to_newtonian",
                certified=self.binary_projection_back_to_newtonian_effectively_certified,
                required="regularized binary charts project back to Newtonian motion away from collision",
                observed=(
                    "binary_projection_back_to_newtonian_certified="
                    f"{self.binary_projection_back_to_newtonian_certified}; "
                    "typed_witness_certified="
                    f"{self.binary_projection_back_to_newtonian_effectively_certified}"
                ),
                witness_field="binary_projection_back_to_newtonian_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="binary_collision_time_parameter",
                certified=self.binary_collision_time_parameter_effectively_certified,
                required="regularized time parameter gives a certified continuation through binary collision",
                observed=(
                    "binary_collision_time_parameter_certified="
                    f"{self.binary_collision_time_parameter_certified}; "
                    "typed_witness_certified="
                    f"{self.binary_collision_time_parameter_effectively_certified}"
                ),
                witness_field="binary_collision_time_parameter_certified",
            ),
            *binary_details,
            ClosedFormProofObligationDetail(
                obligation="nonzero_angular_momentum_triple_exclusion",
                certified=(
                    self.nonzero_angular_momentum_triple_exclusion_effectively_certified
                ),
                required="nonzero centered angular momentum excludes triple collision",
                observed=(
                    "nonzero_angular_momentum_triple_exclusion_certified="
                    f"{self.nonzero_angular_momentum_triple_exclusion_certified}; "
                    "typed_witness_certified="
                    f"{self.nonzero_angular_momentum_triple_exclusion_effectively_certified}"
                ),
                witness_field="nonzero_angular_momentum_triple_exclusion_certified",
            ),
            *nonzero_angular_details,
            ClosedFormProofObligationDetail(
                obligation="zero_angular_momentum_triple_collision_convention",
                certified=self.zero_angular_momentum_triple_convention_effectively_certified,
                required=(
                    "zero-angular-momentum triple-collision branch has an explicit "
                    "allowed continuation convention"
                ),
                observed=(
                    "zero_angular_momentum_triple_collision_convention_certified="
                    f"{self.zero_angular_momentum_triple_collision_convention_certified}; "
                    f"convention={self.effective_triple_collision_continuation_convention}"
                ),
                witness_field="zero_angular_momentum_triple_collision_convention_certified",
            ),
            *convention_details,
            ClosedFormProofObligationDetail(
                obligation="binary_collision_continuation",
                certified=self.binary_collision_continuation_certified,
                required="all binary-collision continuation subclaims are certified",
                observed=f"binary_collision_continuation_certified={self.binary_collision_continuation_certified}",
                witness_field="binary_collision_continuation_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="triple_collision_continuation",
                certified=self.triple_collision_continuation_certified,
                required="nonzero-angular triple exclusion and zero-angular continuation convention are certified",
                observed=f"triple_collision_continuation_certified={self.triple_collision_continuation_certified}",
                witness_field="triple_collision_continuation_certified",
            ),
            ClosedFormProofObligationDetail(
                obligation="collision_continuation",
                certified=self.collision_continuation_certified,
                required="binary and triple collision continuation are certified",
                observed=f"collision_continuation_certified={self.collision_continuation_certified}",
                witness_field="collision_continuation_certified",
            ),
        )

    @property
    def missing_collision_obligations(self) -> tuple[str, ...]:
        obligations: list[str] = []
        seen: set[str] = set()
        for detail in self.collision_obligation_details:
            if detail.certified or detail.obligation in seen:
                continue
            seen.add(detail.obligation)
            obligations.append(detail.obligation)
        return tuple(obligations)


@dataclass(frozen=True)
class SundmanGeneralSolutionTheoremWitnessCertificate:
    """Typed witness for the Sundman-series route to the full theorem target.

    The witness composes three independent pieces: compact-Sundman future-shell
    recurrence closure, collision continuation, and arbitrary-data/all-time
    theorem scope.  It intentionally delegates global-series certification to
    the compact-Sundman witness produced by the recurrence closure.
    """

    recurrence_closure: Any | None = None
    scope_witness: Any | None = None
    collision_witness: CollisionContinuationWitnessCertificate | None = None
    collision_continuation_certified: bool = False
    binary_collision_continuation_certified: bool = False
    triple_collision_continuation_certified: bool = False
    witness_source: str = "manual_sundman_theorem_witness"

    @property
    def effective_collision_continuation_certified(self) -> bool:
        if self.collision_witness is not None:
            return self.collision_witness.collision_continuation_certified
        return self.collision_continuation_certified

    @property
    def effective_binary_collision_continuation_certified(self) -> bool:
        if self.collision_witness is not None:
            return self.collision_witness.binary_collision_continuation_certified
        return self.binary_collision_continuation_certified

    @property
    def effective_triple_collision_continuation_certified(self) -> bool:
        if self.collision_witness is not None:
            return self.collision_witness.triple_collision_continuation_certified
        return self.triple_collision_continuation_certified

    @property
    def compact_sundman_witness(self) -> Any | None:
        if self.recurrence_closure is None:
            return None
        if hasattr(self.recurrence_closure, "to_accelerated_induction_witness"):
            return self.recurrence_closure.to_accelerated_induction_witness(
                collision_continuation_certified=self.effective_collision_continuation_certified,
                binary_collision_continuation_certified=(
                    self.effective_binary_collision_continuation_certified
                ),
                triple_collision_continuation_certified=(
                    self.effective_triple_collision_continuation_certified
                ),
                witness_source=self.witness_source,
            )
        return self.recurrence_closure

    @property
    def global_series_certified(self) -> bool:
        return getattr(
            self.compact_sundman_witness,
            "global_series_certified",
            False,
        ) is True

    @property
    def collision_continuation_obligations_certified(self) -> bool:
        return (
            getattr(
                self.compact_sundman_witness,
                "collision_continuation_obligations_certified",
                False,
            )
            is True
        )

    @property
    def global_collision_continuation_certified(self) -> bool:
        return self.collision_continuation_obligations_certified

    @property
    def missing_recurrence_obligations(self) -> tuple[str, ...]:
        if self.recurrence_closure is None:
            return ("compact_sundman_recurrence_closure",)
        return tuple(getattr(self.recurrence_closure, "missing_recurrence_obligations", ()))

    @property
    def missing_collision_obligations(self) -> tuple[str, ...]:
        if self.collision_witness is None:
            return ()
        return self.collision_witness.missing_collision_obligations

    @property
    def missing_global_proof_obligations(self) -> tuple[str, ...]:
        compact_obligations = tuple(
            getattr(self.compact_sundman_witness, "missing_global_proof_obligations", ())
        )
        obligations: list[str] = []
        seen: set[str] = set()
        for obligation in (
            *self.missing_recurrence_obligations,
            *self.missing_collision_obligations,
            *compact_obligations,
        ):
            if obligation not in seen:
                seen.add(obligation)
                obligations.append(obligation)
        return tuple(obligations)

    @property
    def global_proof_obligation_details(self) -> tuple[Any, ...]:
        details: list[Any] = []
        seen: set[str] = set()
        for detail in (
            *tuple(getattr(self.recurrence_closure, "recurrence_obligation_details", ())),
            *tuple(getattr(self.collision_witness, "collision_obligation_details", ())),
            *tuple(getattr(self.compact_sundman_witness, "global_proof_obligation_details", ())),
        ):
            obligation = getattr(detail, "obligation", None)
            if obligation is None or obligation in seen:
                continue
            seen.add(obligation)
            details.append(detail)
        return tuple(details)

    @property
    def general_solution_scope_certified(self) -> bool:
        return (
            self.scope_witness is not None
            and getattr(self.scope_witness, "general_solution_scope_certified", False)
            is True
        )

    @property
    def arbitrary_positive_masses_certified(self) -> bool:
        return (
            self.scope_witness is not None
            and getattr(
                self.scope_witness,
                "arbitrary_positive_masses_certified",
                False,
            )
            is True
        )

    @property
    def arbitrary_noncollision_initial_data_certified(self) -> bool:
        return (
            self.scope_witness is not None
            and getattr(
                self.scope_witness,
                "arbitrary_noncollision_initial_data_certified",
                False,
            )
            is True
        )

    @property
    def all_real_target_times_certified(self) -> bool:
        return (
            self.scope_witness is not None
            and getattr(self.scope_witness, "all_real_target_times_certified", False)
            is True
        )

    @property
    def lift_construct_project_verify_certified(self) -> bool:
        return (
            self.scope_witness is not None
            and getattr(
                self.scope_witness,
                "lift_construct_project_verify_certified",
                False,
            )
            is True
        )

    @property
    def newton_equations_full_interval_certified(self) -> bool:
        return (
            self.scope_witness is not None
            and getattr(
                self.scope_witness,
                "newton_equations_full_interval_certified",
                False,
            )
            is True
        )


@dataclass(frozen=True)
class GeneralClosedFormSolutionCertificate:
    """Top-level certificate for the requested general closed-form solution theorem."""

    closed_form_certificate: ClosedFormTargetCertificate
    requirement_statuses: tuple[GeneralSolutionRequirementStatus, ...]

    @property
    def proof_certified(self) -> bool:
        return bool(
            isinstance(self.closed_form_certificate, ClosedFormTargetCertificate)
            and self.closed_form_certificate.general_solution_certified is True
            and _requirement_status_ledger_certified(self.requirement_statuses)
        )

    @property
    def certified(self) -> bool:
        return self.proof_certified

    @property
    def status(self) -> str:
        if self.proof_certified:
            return "certified"
        if (
            isinstance(self.closed_form_certificate, ClosedFormTargetCertificate)
            and self.closed_form_certificate.definition_required
        ):
            return "definition_required"
        if (
            isinstance(self.closed_form_certificate, ClosedFormTargetCertificate)
            and self.closed_form_certificate.obstruction_certified
        ):
            return "obstructed_requested_class"
        return "incomplete"

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        closed_form_missing = (
            self.closed_form_certificate.missing_requirements
            if isinstance(self.closed_form_certificate, ClosedFormTargetCertificate)
            else ("general_closed_form_solution_closed_form_certificate_type",)
        )
        return (
            *closed_form_missing,
            *_requirement_status_ledger_missing(
                self.requirement_statuses,
                ledger_name="general_closed_form_solution",
            ),
        )

    @property
    def missing_requirement_details(self) -> tuple[GeneralSolutionRequirementStatus, ...]:
        return tuple(
            status
            for status in self.requirement_statuses
            if type(status) is GeneralSolutionRequirementStatus
            and status.certified is not True
        )

    @property
    def blocking_obligations(self) -> tuple[str, ...]:
        obligations: list[str] = []
        seen: set[str] = set()
        for obligation in _requirement_status_ledger_structure_missing(
            self.requirement_statuses,
            ledger_name="general_closed_form_solution",
        ):
            if obligation not in seen:
                seen.add(obligation)
                obligations.append(obligation)
        for status in self.missing_requirement_details:
            for obligation in status.blocking_obligations:
                if obligation not in seen:
                    seen.add(obligation)
                    obligations.append(obligation)
        return tuple(obligations)

    @property
    def route_summary(self) -> str:
        if self.proof_certified:
            return "full general closed-form solution target is certified"
        if self.status == "obstructed_requested_class":
            return self.closed_form_certificate.route_summary
        if self.status == "definition_required":
            return self.closed_form_certificate.route_summary
        return "constructive route remains incomplete"


_CLASS_ALIASES = {
    "": "unspecified",
    "unspecified": "unspecified",
    "closed_form": "unspecified",
    "closed-form": "unspecified",
    "finite_first_integral": "finite_first_integral_closed_form",
    "finite_first_integrals": "finite_first_integral_closed_form",
    "finite_first_integral_closed_form": "finite_first_integral_closed_form",
    "finite_integral_closed_form": "finite_first_integral_closed_form",
    "algebraic_first_integrals": "finite_first_integral_closed_form",
    "finite_algebraic_first_integrals": "finite_first_integral_closed_form",
    "analytic_extra_integral": "analytic_or_meromorphic_extra_integral",
    "analytic_first_integral": "analytic_or_meromorphic_extra_integral",
    "meromorphic_first_integral": "analytic_or_meromorphic_extra_integral",
    "analytic_or_meromorphic_extra_integral": "analytic_or_meromorphic_extra_integral",
    "sundman": "sundman_global_series",
    "sundman_series": "sundman_global_series",
    "sundman_global_series": "sundman_global_series",
    "infinite_convergent_series": "sundman_global_series",
    "global_convergent_series": "sundman_global_series",
    "regularized_atlas": "regularized_locally_finite_atlas",
    "regularized_locally_finite_atlas": "regularized_locally_finite_atlas",
    "certified_atlas_closed_form": "regularized_locally_finite_atlas",
    "open_time_atlas_series": "regularized_locally_finite_atlas",
    "piecewise_analytic_regularized_atlas": "regularized_locally_finite_atlas",
    "finite_target_atlas_series": "regularized_locally_finite_atlas",
}


def _normalize_closed_form_class(closed_form_class: str) -> str:
    key = closed_form_class.strip().lower().replace(" ", "_").replace("-", "_")
    return _CLASS_ALIASES.get(key, "unspecified")


def _bruns_reference() -> ClosedFormTheoremReference:
    return ClosedFormTheoremReference(
        theorem_id="bruns_algebraic_integrals",
        statement=(
            "The only algebraically independent algebraic first integrals of the "
            "three-body problem are the classical center-of-mass, linear-momentum, "
            "angular-momentum, and energy integrals."
        ),
        source_url=BRUNS_THEOREM_URL,
    )


def _poincare_yagasaki_reference() -> ClosedFormTheoremReference:
    return ClosedFormTheoremReference(
        theorem_id="poincare_yagasaki_analytic_nonintegrability",
        statement=(
            "Poincare-type nonintegrability results rule out additional analytic "
            "or meromorphic first integrals in representative three-body settings."
        ),
        source_url=YAGASAKI_NONINTEGRABILITY_URL,
    )


def _sundman_reference() -> ClosedFormTheoremReference:
    return ClosedFormTheoremReference(
        theorem_id="sundman_global_convergent_series",
        statement=(
            "Sundman's regularized construction gives a global convergent-series "
            "route under its collision-continuation and convergence hypotheses."
        ),
        source_url=SUNDMAN_MEMOIR_URL,
    )


def _regularized_atlas_reference() -> ClosedFormTheoremReference:
    return ClosedFormTheoremReference(
        theorem_id="regularized_locally_finite_atlas_closed_form",
        statement=(
            "A computably enumerable, locally finite family of regularized "
            "analytic chart primitives is a theorem-compatible closed-form "
            "target when every finite target is resolved by a finite "
            "verifier-checkable atlas or maximal-classical total-collision stop."
        ),
        source_url=SUNDMAN_MEMOIR_URL,
    )


def _witness_flag(witness: Any | None, *field_names: str) -> bool:
    if witness is None:
        return False
    return any(getattr(witness, field_name, False) is True for field_name in field_names)


def _observed_witness_flags(witness: Any | None, *field_names: str) -> str:
    if witness is None:
        return "no witness supplied"
    observed: list[str] = []
    for field_name in field_names:
        value = getattr(witness, field_name, False)
        rendered = str(value) if type(value) is bool else f"{value!r} (not literal True)"
        observed.append(f"{field_name}={rendered}")
    return ", ".join(observed)


def certify_closed_form_target(
    closed_form_class: str = "unspecified",
    *,
    compact_sundman_witness: Any | None = None,
    closed_form_function_class_witness: (
        ClosedFormFunctionClassWitnessCertificate | None
    ) = None,
) -> ClosedFormTargetCertificate:
    """Classify the requested closed-form theorem target.

    This function deliberately certifies only the theorem shape. It does not
    turn the current finite atlas into a general solution unless the supplied
    compact-Sundman witness itself certifies the global series.
    """

    normalized = _normalize_closed_form_class(closed_form_class)
    if normalized == "unspecified" and closed_form_function_class_witness is not None:
        normalized = closed_form_function_class_witness.normalized_class
        if not closed_form_function_class_witness.function_class_definition_certified:
            missing = (
                closed_form_function_class_witness.missing_function_class_obligations
                or ("define_allowed_closed_form_function_class",)
            )
            return ClosedFormTargetCertificate(
                requested_class=closed_form_class,
                normalized_class=normalized,
                status="definition_required",
                reason=(
                    "The supplied closed-form function-class witness does not yet "
                    "define the representation, evaluation, convergence, and "
                    "Newton-equation verification semantics needed by a theorem."
                ),
                theorem_references=(),
                missing_requirements=missing,
            )
    if normalized == "finite_first_integral_closed_form":
        internal_obstruction = certify_classical_integrals_do_not_determine_vector_field()
        return ClosedFormTargetCertificate(
            requested_class=closed_form_class,
            normalized_class=normalized,
            status="obstructed",
            reason=(
                "A finite closed form based on algebraic first integrals would need "
                "additional independent algebraic integrals beyond the classical ten; "
                "Bruns' theorem rules those out for the general three-body problem."
            ),
            theorem_references=(_bruns_reference(),),
            missing_requirements=("switch to an allowed infinite-series or local-numerical certificate class",),
            finite_closed_form_obstructed=True,
            internal_obstruction_certified=internal_obstruction.certified,
            internal_obstruction_evidence=internal_obstruction.reason,
        )
    if normalized == "analytic_or_meromorphic_extra_integral":
        return ClosedFormTargetCertificate(
            requested_class=closed_form_class,
            normalized_class=normalized,
            status="obstructed",
            reason=(
                "A general analytic or meromorphic extra-integral route is incompatible "
                "with Poincare-type nonintegrability obstructions in three-body settings."
            ),
            theorem_references=(_poincare_yagasaki_reference(),),
            missing_requirements=("switch to a non-finite global series theorem or restrict the data class",),
            finite_closed_form_obstructed=True,
        )
    if normalized == "sundman_global_series":
        if compact_sundman_witness is not None and (
            getattr(compact_sundman_witness, "global_series_certified", False)
            is True
        ) and not tuple(
            getattr(compact_sundman_witness, "missing_global_proof_obligations", ())
        ):
            general_scope_certified = _witness_flag(
                compact_sundman_witness,
                "general_solution_scope_certified",
            )
            return ClosedFormTargetCertificate(
                requested_class=closed_form_class,
                normalized_class=normalized,
                status="certified_infinite_series_route",
                reason=(
                    "The supplied compact-Sundman witness certifies the global series obligations; "
                    "the full general-solution theorem still requires explicit arbitrary-data scope "
                    "unless that scope is separately certified."
                ),
                theorem_references=(_sundman_reference(),),
                infinite_series_route=True,
                series_route_certified=True,
                general_solution_scope_certified=general_scope_certified,
                general_solution_certified=general_scope_certified,
            )

        witness_obligations: tuple[str, ...] = ()
        if compact_sundman_witness is not None:
            witness_obligations = tuple(
                getattr(compact_sundman_witness, "missing_global_proof_obligations", ())
            )
        missing = witness_obligations or (
            "compact_sundman_global_induction_witness",
            "binary_collision_continuation",
            "triple_collision_continuation",
        )
        return ClosedFormTargetCertificate(
            requested_class=closed_form_class,
            normalized_class=normalized,
            status="conditional_infinite_series_route",
            reason=(
                "A Sundman-style infinite convergent series is the theorem-compatible "
                "global route, but the current harness has not proved every all-future "
                "compact-Sundman induction and collision-continuation obligation."
            ),
            theorem_references=(_sundman_reference(),),
            missing_requirements=missing,
            compact_sundman_obligations=witness_obligations,
            infinite_series_route=True,
        )

    if normalized == "regularized_locally_finite_atlas":
        return ClosedFormTargetCertificate(
            requested_class=closed_form_class,
            normalized_class=normalized,
            status="conditional_regularized_atlas_route",
            reason=(
                "The theorem-compatible target is a computably enumerable, "
                "locally finite family of ordinary/LC/KS/total-stop analytic "
                "charts with finite verifier-checkable evaluation for every "
                "finite target time.  The route remains conditional until the "
                "open-time atlas proof, finite-target atlas-or-stop "
                "completeness, collision semantics, and independent chart "
                "verifier are all certified."
            ),
            theorem_references=(_regularized_atlas_reference(),),
            missing_requirements=(
                "open_time_locally_finite_atlas_proof",
                "finite_target_atlas_or_stop_completeness",
                "independent_chart_verifier",
                "binary_collision_continuation",
                "total_collision_stop_policy",
            ),
            regularized_atlas_route=True,
        )

    return ClosedFormTargetCertificate(
        requested_class=closed_form_class,
        normalized_class=normalized,
        status="definition_required",
        reason=(
            "The phrase closed form is not a theorem-compatible target until its "
            "allowed function class is specified."
        ),
        theorem_references=(),
        missing_requirements=("define_allowed_closed_form_function_class",),
    )


def _requirement_status(
    witness: Any | None,
    requirement: str,
    reason: str,
    *field_names: str,
    required: str | None = None,
) -> GeneralSolutionRequirementStatus:
    certified = _witness_flag(witness, *field_names)
    return GeneralSolutionRequirementStatus(
        requirement=requirement,
        certified=certified,
        reason=reason,
        witness_field=" or ".join(field_names),
        required=required or "at least one witness field is certified",
        observed=_observed_witness_flags(witness, *field_names),
    )


def certify_general_solution_scope_witness(
    *,
    arbitrary_positive_masses_certified: bool = False,
    arbitrary_noncollision_initial_data_certified: bool = False,
    all_real_target_times_certified: bool = False,
    lift_construct_project_verify_certified: bool = False,
    newton_equations_full_interval_certified: bool = False,
    witness_source: str = "manual",
) -> GeneralSolutionScopeWitnessCertificate:
    """Record the arbitrary-data/all-time scope witness needed by a general solution."""

    return GeneralSolutionScopeWitnessCertificate(
        arbitrary_positive_masses_certified=_strict_bool(
            arbitrary_positive_masses_certified,
            "arbitrary_positive_masses_certified",
        ),
        arbitrary_noncollision_initial_data_certified=_strict_bool(
            arbitrary_noncollision_initial_data_certified,
            "arbitrary_noncollision_initial_data_certified",
        ),
        all_real_target_times_certified=_strict_bool(
            all_real_target_times_certified,
            "all_real_target_times_certified",
        ),
        lift_construct_project_verify_certified=_strict_bool(
            lift_construct_project_verify_certified,
            "lift_construct_project_verify_certified",
        ),
        newton_equations_full_interval_certified=_strict_bool(
            newton_equations_full_interval_certified,
            "newton_equations_full_interval_certified",
        ),
        witness_source=str(witness_source),
    )


def certify_general_solution_theorem_scope_witness(
    *,
    positive_mass_domain_quantified_certified: bool = False,
    mass_parameter_regularity_certified: bool = False,
    noncollision_initial_domain_quantified_certified: bool = False,
    center_of_mass_reduction_global_certified: bool = False,
    interval_initial_data_lift_global_certified: bool = False,
    compact_time_real_line_bijection_certified: bool = False,
    compact_time_real_line_bijection_witness: (
        CompactTimeRealLineBijectionWitnessCertificate | None
    ) = None,
    sundman_time_targeting_global_certified: bool = False,
    sundman_time_targeting_global_witness: (
        SundmanTimeTargetingGlobalWitnessCertificate | None
    ) = None,
    compact_sundman_lift_certified: bool = False,
    global_series_construction_scope_certified: bool = False,
    inertial_projection_certified: bool = False,
    inertial_projection_witness: InertialProjectionWitnessCertificate | None = None,
    chain_rule_newton_equations_certified: bool = False,
    full_interval_residual_verification_certified: bool = False,
    witness_source: str = "manual_theorem_scope",
) -> GeneralSolutionTheoremScopeWitnessCertificate:
    """Record granular arbitrary-data/all-time theorem-scope evidence."""

    return GeneralSolutionTheoremScopeWitnessCertificate(
        positive_mass_domain_quantified_certified=bool(
            positive_mass_domain_quantified_certified
        ),
        mass_parameter_regularity_certified=bool(mass_parameter_regularity_certified),
        noncollision_initial_domain_quantified_certified=bool(
            noncollision_initial_domain_quantified_certified
        ),
        center_of_mass_reduction_global_certified=bool(
            center_of_mass_reduction_global_certified
        ),
        interval_initial_data_lift_global_certified=bool(
            interval_initial_data_lift_global_certified
        ),
        compact_time_real_line_bijection_certified=bool(
            compact_time_real_line_bijection_certified
        ),
        compact_time_real_line_bijection_witness=(
            compact_time_real_line_bijection_witness
        ),
        sundman_time_targeting_global_certified=bool(
            sundman_time_targeting_global_certified
        ),
        sundman_time_targeting_global_witness=sundman_time_targeting_global_witness,
        compact_sundman_lift_certified=bool(compact_sundman_lift_certified),
        global_series_construction_scope_certified=bool(
            global_series_construction_scope_certified
        ),
        inertial_projection_certified=bool(inertial_projection_certified),
        inertial_projection_witness=inertial_projection_witness,
        chain_rule_newton_equations_certified=bool(
            chain_rule_newton_equations_certified
        ),
        full_interval_residual_verification_certified=bool(
            full_interval_residual_verification_certified
        ),
        witness_source=str(witness_source),
    )


def certify_inertial_projection_witness(
    *,
    center_of_mass_affine_motion_certified: bool = False,
    reduced_to_inertial_coordinate_map_certified: bool = False,
    physical_time_series_compatibility_certified: bool = False,
    interval_projection_containment_certified: bool = False,
    mass_weighted_reconstruction_certified: bool = False,
    witness_source: str = "manual_inertial_projection",
) -> InertialProjectionWitnessCertificate:
    """Record typed evidence for projecting a lifted solution back to inertial coordinates."""

    return InertialProjectionWitnessCertificate(
        center_of_mass_affine_motion_certified=bool(
            center_of_mass_affine_motion_certified
        ),
        reduced_to_inertial_coordinate_map_certified=bool(
            reduced_to_inertial_coordinate_map_certified
        ),
        physical_time_series_compatibility_certified=bool(
            physical_time_series_compatibility_certified
        ),
        interval_projection_containment_certified=bool(
            interval_projection_containment_certified
        ),
        mass_weighted_reconstruction_certified=bool(
            mass_weighted_reconstruction_certified
        ),
        witness_source=str(witness_source),
    )


def certify_sundman_time_targeting_global_witness(
    *,
    compact_sundman_parameter_domain_certified: bool = False,
    positive_physical_time_derivative_certified: bool = False,
    finite_target_bracketing_certified: bool = False,
    target_interval_evaluation_certified: bool = False,
    global_target_range_certified: bool = False,
    witness_source: str = "manual_sundman_time_targeting",
) -> SundmanTimeTargetingGlobalWitnessCertificate:
    """Record typed evidence that Sundman time can target physical time."""

    return SundmanTimeTargetingGlobalWitnessCertificate(
        compact_sundman_parameter_domain_certified=bool(
            compact_sundman_parameter_domain_certified
        ),
        positive_physical_time_derivative_certified=bool(
            positive_physical_time_derivative_certified
        ),
        finite_target_bracketing_certified=bool(finite_target_bracketing_certified),
        target_interval_evaluation_certified=bool(target_interval_evaluation_certified),
        global_target_range_certified=bool(global_target_range_certified),
        witness_source=str(witness_source),
    )


def certify_compact_time_real_line_bijection_witness(
    *,
    positive_rate_parameter_certified: bool = False,
    forward_map_all_real_certified: bool = False,
    inverse_map_open_interval_certified: bool = False,
    strict_monotonicity_certified: bool = False,
    endpoint_limits_certified: bool = False,
    witness_source: str = "manual_compact_time_real_line_bijection",
) -> CompactTimeRealLineBijectionWitnessCertificate:
    """Record typed evidence that compact physical time covers the real line."""

    return CompactTimeRealLineBijectionWitnessCertificate(
        positive_rate_parameter_certified=bool(positive_rate_parameter_certified),
        forward_map_all_real_certified=bool(forward_map_all_real_certified),
        inverse_map_open_interval_certified=bool(inverse_map_open_interval_certified),
        strict_monotonicity_certified=bool(strict_monotonicity_certified),
        endpoint_limits_certified=bool(endpoint_limits_certified),
        witness_source=str(witness_source),
    )


def certify_closed_form_function_class_witness(
    *,
    class_id: str = "unspecified",
    representation_semantics_certified: bool = False,
    evaluation_semantics_certified: bool = False,
    convergence_semantics_certified: bool = False,
    equation_verification_semantics_certified: bool = False,
    witness_source: str = "manual_closed_form_function_class",
) -> ClosedFormFunctionClassWitnessCertificate:
    """Record typed semantics for the function class meant by closed form."""

    return ClosedFormFunctionClassWitnessCertificate(
        class_id=str(class_id),
        representation_semantics_certified=bool(representation_semantics_certified),
        evaluation_semantics_certified=bool(evaluation_semantics_certified),
        convergence_semantics_certified=bool(convergence_semantics_certified),
        equation_verification_semantics_certified=bool(
            equation_verification_semantics_certified
        ),
        witness_source=str(witness_source),
    )


def certify_certificate_language_soundness(
    *,
    ordinary_taylor_sound: bool = False,
    levi_civita_sound: bool = False,
    spatial_ks_sound: bool = False,
    total_stop_sound: bool = False,
    fuchsian_stop_sound: bool | None = None,
    generalized_fuchsian_stop_sound: bool | None = None,
    transition_sound: bool = False,
    branch_union_sound: bool = False,
    chart_chain_sound: bool = False,
    verifier_kernel_sound: bool = False,
    proof_grade_arithmetic_backend_sound: bool = False,
    witness_source: str = "certificate_language_soundness",
) -> CertificateLanguageSoundnessCertificate:
    """Record soundness evidence for the regularized-atlas certificate language."""

    fuchsian_stop_sound = _strict_optional_bool(
        fuchsian_stop_sound,
        "fuchsian_stop_sound",
    )
    generalized_fuchsian_stop_sound = _strict_optional_bool(
        generalized_fuchsian_stop_sound,
        "generalized_fuchsian_stop_sound",
    )
    return CertificateLanguageSoundnessCertificate(
        ordinary_taylor_sound=_strict_bool(
            ordinary_taylor_sound,
            "ordinary_taylor_sound",
        ),
        levi_civita_sound=_strict_bool(levi_civita_sound, "levi_civita_sound"),
        spatial_ks_sound=_strict_bool(spatial_ks_sound, "spatial_ks_sound"),
        total_stop_sound=_strict_bool(total_stop_sound, "total_stop_sound"),
        fuchsian_stop_sound=(
            False if fuchsian_stop_sound is None else fuchsian_stop_sound
        ),
        generalized_fuchsian_stop_sound=(
            False
            if generalized_fuchsian_stop_sound is None
            else generalized_fuchsian_stop_sound
        ),
        transition_sound=_strict_bool(transition_sound, "transition_sound"),
        branch_union_sound=_strict_bool(branch_union_sound, "branch_union_sound"),
        chart_chain_sound=_strict_bool(chart_chain_sound, "chart_chain_sound"),
        verifier_kernel_sound=_strict_bool(
            verifier_kernel_sound,
            "verifier_kernel_sound",
        ),
        proof_grade_arithmetic_backend_sound=_strict_bool(
            proof_grade_arithmetic_backend_sound,
            "proof_grade_arithmetic_backend_sound",
        ),
        witness_source=str(witness_source),
    )


def derive_certificate_language_soundness_from_checker_kernel(
    checker_kernel_support: Any,
    *,
    witness_source: str = "checker_kernel_support_certificate_language_soundness",
) -> CertificateLanguageSoundnessCertificate:
    """Derive the closed-form soundness gate from a checker-kernel manifest.

    This is the theorem-facing constructor for the language-soundness gate.  It
    reads the supported chart/checker families from a kernel support
    certificate instead of accepting separate chart-family booleans.  The
    proof-grade arithmetic backend remains an explicit obligation on the kernel
    support certificate.
    """

    return certify_certificate_language_soundness(
        ordinary_taylor_sound=getattr(
            checker_kernel_support,
            "ordinary_taylor_sound",
            False,
        ),
        levi_civita_sound=getattr(
            checker_kernel_support,
            "levi_civita_sound",
            False,
        ),
        spatial_ks_sound=getattr(checker_kernel_support, "spatial_ks_sound", False),
        fuchsian_stop_sound=getattr(
            checker_kernel_support,
            "fuchsian_stop_sound",
            False,
        ),
        generalized_fuchsian_stop_sound=getattr(
            checker_kernel_support,
            "generalized_fuchsian_stop_sound",
            False,
        ),
        transition_sound=getattr(checker_kernel_support, "transition_sound", False),
        branch_union_sound=getattr(
            checker_kernel_support,
            "branch_union_sound",
            False,
        ),
        chart_chain_sound=getattr(
            checker_kernel_support,
            "chart_chain_sound",
            False,
        ),
        verifier_kernel_sound=getattr(
            checker_kernel_support,
            "verifier_kernel_sound",
            False,
        ),
        proof_grade_arithmetic_backend_sound=getattr(
            checker_kernel_support,
            "proof_grade_arithmetic_backend_sound",
            False,
        ),
        witness_source=str(witness_source),
    )


def certify_computable_atlas_certificate_enumeration(
    *,
    chart_family_words_enumerated: bool = False,
    pair_labels_enumerated: bool = False,
    rational_domains_enumerated: bool = False,
    truncation_orders_enumerated: bool = False,
    rational_or_interval_coefficients_enumerated: bool = False,
    rational_tail_budgets_enumerated: bool = False,
    generalized_fuchsian_exponent_data_enumerated: bool = False,
    fuchsian_selector_constants_enumerated: bool = False,
    cauchy_majorants_enumerated: bool = False,
    transition_witnesses_enumerated: bool = False,
    collision_policy_data_enumerated: bool = False,
    independent_checker_dovetailed: bool = False,
    dovetailing_fairness_certified: bool = False,
    finite_target_query_terminates_certified: bool = False,
    source_theorem_id: str = "",
    source_theorem_proof_certified: bool = False,
    source_theorem_dimension: int = 0,
    source_theorem_input_model: str = "",
    source_total_collision_policy_id: str = "",
    finite_target_chart_families: tuple[str, ...] = (),
    finite_target_allowed_outcomes: tuple[str, ...] = (),
    witness_source: str = "computable_atlas_certificate_enumeration",
) -> ComputableAtlasCertificateEnumerationCertificate:
    """Record fair enumeration evidence for computable-input atlas certificates."""

    return ComputableAtlasCertificateEnumerationCertificate(
        chart_family_words_enumerated=_strict_bool(
            chart_family_words_enumerated,
            "chart_family_words_enumerated",
        ),
        pair_labels_enumerated=_strict_bool(
            pair_labels_enumerated,
            "pair_labels_enumerated",
        ),
        rational_domains_enumerated=_strict_bool(
            rational_domains_enumerated,
            "rational_domains_enumerated",
        ),
        truncation_orders_enumerated=_strict_bool(
            truncation_orders_enumerated,
            "truncation_orders_enumerated",
        ),
        rational_or_interval_coefficients_enumerated=_strict_bool(
            rational_or_interval_coefficients_enumerated,
            "rational_or_interval_coefficients_enumerated",
        ),
        rational_tail_budgets_enumerated=_strict_bool(
            rational_tail_budgets_enumerated,
            "rational_tail_budgets_enumerated",
        ),
        generalized_fuchsian_exponent_data_enumerated=_strict_bool(
            generalized_fuchsian_exponent_data_enumerated,
            "generalized_fuchsian_exponent_data_enumerated",
        ),
        fuchsian_selector_constants_enumerated=_strict_bool(
            fuchsian_selector_constants_enumerated,
            "fuchsian_selector_constants_enumerated",
        ),
        cauchy_majorants_enumerated=_strict_bool(
            cauchy_majorants_enumerated,
            "cauchy_majorants_enumerated",
        ),
        transition_witnesses_enumerated=_strict_bool(
            transition_witnesses_enumerated,
            "transition_witnesses_enumerated",
        ),
        collision_policy_data_enumerated=_strict_bool(
            collision_policy_data_enumerated,
            "collision_policy_data_enumerated",
        ),
        independent_checker_dovetailed=_strict_bool(
            independent_checker_dovetailed,
            "independent_checker_dovetailed",
        ),
        dovetailing_fairness_certified=_strict_bool(
            dovetailing_fairness_certified,
            "dovetailing_fairness_certified",
        ),
        finite_target_query_terminates_certified=_strict_bool(
            finite_target_query_terminates_certified,
            "finite_target_query_terminates_certified",
        ),
        source_theorem_id=str(source_theorem_id),
        source_theorem_proof_certified=_strict_bool(
            source_theorem_proof_certified,
            "source_theorem_proof_certified",
        ),
        source_theorem_dimension=int(source_theorem_dimension),
        source_theorem_input_model=str(source_theorem_input_model),
        source_total_collision_policy_id=str(source_total_collision_policy_id),
        finite_target_chart_families=tuple(str(item) for item in finite_target_chart_families),
        finite_target_allowed_outcomes=tuple(str(item) for item in finite_target_allowed_outcomes),
        witness_source=str(witness_source),
    )


def derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
    pointwise_open_time_theorem: Any,
    *,
    witness_source: str = "pointwise_open_time_theorem_fair_certificate_enumeration",
) -> ComputableAtlasCertificateEnumerationCertificate:
    """Derive fair certificate enumeration from the proof-certified pointwise theorem.

    This is the constructor-backed version of the enumeration gate.  It does
    not ask the caller for raw enumeration booleans; it reads the finite-target
    chart grammar and the two atlas-or-stop outcomes from the pointwise theorem
    and certifies the usual dovetailed enumeration obligations only when the
    theorem itself is proof-certified.
    """

    if not isinstance(
        pointwise_open_time_theorem,
        PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate,
    ):
        return certify_computable_atlas_certificate_enumeration(
            source_theorem_id=str(getattr(pointwise_open_time_theorem, "theorem_id", "")),
            source_theorem_proof_certified=(
                getattr(pointwise_open_time_theorem, "proof_certified", False)
                is True
            ),
            source_theorem_dimension=int(
                getattr(pointwise_open_time_theorem, "dimension", 0)
            ),
            source_theorem_input_model=str(
                getattr(pointwise_open_time_theorem, "input_model", "")
            ),
            source_total_collision_policy_id=str(
                getattr(pointwise_open_time_theorem, "total_collision_policy_id", "")
            ),
            finite_target_chart_families=tuple(
                str(item)
                for item in getattr(
                    getattr(pointwise_open_time_theorem, "finite_target_theorem", None),
                    "chart_families",
                    (),
                )
                or ()
            ),
            finite_target_allowed_outcomes=tuple(
                str(item)
                for item in getattr(
                    getattr(pointwise_open_time_theorem, "finite_target_theorem", None),
                    "allowed_outcomes",
                    (),
                )
                or ()
            ),
            witness_source=str(witness_source),
        )
    theorem_id = str(getattr(pointwise_open_time_theorem, "theorem_id", ""))
    proof_certified = (
        getattr(pointwise_open_time_theorem, "proof_certified", False) is True
    )
    finite_target_theorem = getattr(
        pointwise_open_time_theorem,
        "finite_target_theorem",
        None,
    )
    chart_families = tuple(
        str(item)
        for item in getattr(finite_target_theorem, "chart_families", ()) or ()
    )
    allowed_outcomes = tuple(
        str(item)
        for item in getattr(finite_target_theorem, "allowed_outcomes", ()) or ()
    )
    chart_grammar_certified = set(chart_families) == set(
        POINTWISE_FINITE_TARGET_CHART_FAMILIES
    )
    outcome_grammar_certified = (
        tuple(allowed_outcomes) == POINTWISE_FINITE_TARGET_ALLOWED_OUTCOMES
    )
    input_model = str(getattr(pointwise_open_time_theorem, "input_model", ""))
    computable_point_input = "computable" in input_model or "exact_point" in input_model
    dimension_supported = int(getattr(pointwise_open_time_theorem, "dimension", 0)) in {
        2,
        3,
    }
    base_certified = bool(
        theorem_id == "pointwise_open_time_locally_finite_atlas"
        and proof_certified
        and chart_grammar_certified
        and outcome_grammar_certified
        and computable_point_input
        and dimension_supported
    )
    return certify_computable_atlas_certificate_enumeration(
        chart_family_words_enumerated=base_certified,
        pair_labels_enumerated=base_certified,
        rational_domains_enumerated=base_certified,
        truncation_orders_enumerated=base_certified,
        rational_or_interval_coefficients_enumerated=base_certified,
        rational_tail_budgets_enumerated=base_certified,
        generalized_fuchsian_exponent_data_enumerated=base_certified,
        fuchsian_selector_constants_enumerated=base_certified,
        cauchy_majorants_enumerated=base_certified,
        transition_witnesses_enumerated=base_certified,
        collision_policy_data_enumerated=base_certified,
        independent_checker_dovetailed=base_certified,
        dovetailing_fairness_certified=base_certified,
        finite_target_query_terminates_certified=base_certified,
        source_theorem_id=theorem_id,
        source_theorem_proof_certified=proof_certified,
        source_theorem_dimension=int(
            getattr(pointwise_open_time_theorem, "dimension", 0)
        ),
        source_theorem_input_model=input_model,
        source_total_collision_policy_id=str(
            getattr(pointwise_open_time_theorem, "total_collision_policy_id", "")
        ),
        finite_target_chart_families=chart_families,
        finite_target_allowed_outcomes=allowed_outcomes,
        witness_source=str(witness_source),
    )


def certify_maximal_classical_total_collision_policy(
    *,
    policy_id: str = "maximal_classical_stop",
    stop_at_unselected_total_collision: bool = True,
    selected_continuation_forbidden: bool = True,
    maximal_classical_domain_certified: bool = True,
    pointwise_open_time_theorem: Any | None = None,
    witness_source: str = "maximal_classical_total_collision_policy",
) -> MaximalClassicalTotalCollisionPolicyCertificate:
    """Record maximal-classical total-collision stop semantics."""

    policy_id = str(policy_id)
    source_theorem_id = str(getattr(pointwise_open_time_theorem, "theorem_id", ""))
    source_theorem_proof_certified = bool(
        isinstance(
            pointwise_open_time_theorem,
            PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate,
        )
        and getattr(pointwise_open_time_theorem, "proof_certified", False) is True
    )
    theorem_policy_id = (
        str(getattr(pointwise_open_time_theorem, "total_collision_policy_id", ""))
        if pointwise_open_time_theorem is not None
        else ""
    )
    return MaximalClassicalTotalCollisionPolicyCertificate(
        policy_id=policy_id,
        stop_at_unselected_total_collision=_strict_bool(
            stop_at_unselected_total_collision,
            "stop_at_unselected_total_collision",
        ),
        selected_continuation_forbidden=_strict_bool(
            selected_continuation_forbidden,
            "selected_continuation_forbidden",
        ),
        maximal_classical_domain_certified=_strict_bool(
            maximal_classical_domain_certified,
            "maximal_classical_domain_certified",
        ),
        pointwise_theorem_policy_matches=bool(
            source_theorem_proof_certified and theorem_policy_id == policy_id
        ),
        source_theorem_id=source_theorem_id,
        source_theorem_proof_certified=source_theorem_proof_certified,
        source_theorem_dimension=int(
            getattr(pointwise_open_time_theorem, "dimension", 0)
        ),
        source_theorem_input_model=str(
            getattr(pointwise_open_time_theorem, "input_model", "")
        ),
        source_total_collision_policy_id=str(theorem_policy_id),
        witness_source=str(witness_source),
    )


def certify_pointwise_regularized_atlas_closed_form_theorem(
    *,
    pointwise_open_time_theorem: Any,
    certificate_language_soundness: CertificateLanguageSoundnessCertificate,
    computable_certificate_enumeration: ComputableAtlasCertificateEnumerationCertificate,
    maximal_classical_total_collision_policy: (
        MaximalClassicalTotalCollisionPolicyCertificate | None
    ) = None,
    closed_form_class_id: str = "regularized_locally_finite_atlas",
    statement: str | None = None,
    proof_sketch: str | None = None,
) -> PointwiseRegularizedAtlasClosedFormTheoremCertificate:
    """Assemble the exact/computable-input regularized-atlas closed-form theorem."""

    policy = maximal_classical_total_collision_policy
    if policy is None:
        policy = certify_maximal_classical_total_collision_policy(
            policy_id=str(
                getattr(
                    pointwise_open_time_theorem,
                    "total_collision_policy_id",
                    "maximal_classical_stop",
                )
            ),
            pointwise_open_time_theorem=pointwise_open_time_theorem,
        )
    if statement is None:
        statement = (
            "For positive computable masses and computable noncollision initial "
            "data in dimension two or three, the maximal-classical Newtonian "
            "three-body solution admits a computably enumerable locally finite "
            "regularized analytic atlas in the class "
            "regularized_locally_finite_atlas.  The chart primitives are "
            "ordinary Taylor charts, planar Levi-Civita binary charts, spatial "
            "KS binary charts, and generalized Fuchsian/Puiseux-log "
            "total-collision stop charts.  For every finite physical target "
            "time, a finite certificate either reaches the target by an "
            "ordinary/LC/KS chart chain or certifies the first unselected "
            "total collision before or at the target by an ordinary/LC/KS/"
            "total-stop chain."
        )
    if proof_sketch is None:
        proof_sketch = (
            "Use the pointwise finite-target atlas-or-stop theorem on each "
            "compact physical-time interval, exhaust the real line by compact "
            "intervals, enumerate verifier-checkable ordinary, LC, KS, and "
            "generalized Fuchsian/Puiseux-log total-stop certificates fairly, "
            "and use certificate-language soundness to project every accepted "
            "chart chain back to a Newtonian solution or a maximal-classical "
            "total-collision stop.  Endpoint classification into scattering, "
            "bounded, homothetic, oscillatory, or other final-motion regimes "
            "is not required; those regimes remain optional compression "
            "certificates outside the closed-form proof."
        )
    chart_primitives = (
        "ordinary_taylor",
        "planar_levi_civita_binary",
        "spatial_ks_binary",
        "generalized_fuchsian_puiseux_log_total_stop",
    )
    finite_target_outcomes = (
        "finite_ordinary_lc_ks_chart_chain_reaches_target",
        "finite_ordinary_lc_ks_total_stop_chain_certifies_first_unselected_total_collision",
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="regularized_locally_finite_atlas_class",
            certified=_normalize_closed_form_class(closed_form_class_id)
            == "regularized_locally_finite_atlas",
            source="closed_form_class_id",
            detail=(
                "required=regularized_locally_finite_atlas; observed="
                f"{_normalize_closed_form_class(closed_form_class_id)}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="pointwise_open_time_atlas_proof",
            certified=(
                isinstance(
                    pointwise_open_time_theorem,
                    PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate,
                )
                and getattr(pointwise_open_time_theorem, "proof_certified", False)
                is True
            ),
            source=(
                "PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate."
                "proof_certified"
            ),
            detail=str(getattr(pointwise_open_time_theorem, "route_summary", "")),
        ),
        TheoremPipelineObligation(
            obligation="certificate_language_soundness",
            certified=(
                isinstance(
                    certificate_language_soundness,
                    CertificateLanguageSoundnessCertificate,
                )
                and certificate_language_soundness.proof_certified is True
                and certificate_language_soundness.checker_kernel_derived is True
            ),
            source="CertificateLanguageSoundnessCertificate.checker_kernel_derived",
            detail=(
                "missing="
                f"{certificate_language_soundness.missing_obligations}; "
                "checker_kernel_derived="
                f"{certificate_language_soundness.checker_kernel_derived}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="computable_atlas_certificate_enumeration",
            certified=(
                isinstance(
                    computable_certificate_enumeration,
                    ComputableAtlasCertificateEnumerationCertificate,
                )
                and computable_certificate_enumeration.proof_certified is True
                and computable_certificate_enumeration.pointwise_theorem_derived
                is True
                and _enumeration_source_matches_pointwise_theorem(
                    computable_certificate_enumeration,
                    pointwise_open_time_theorem,
                )
            ),
            source=(
                "ComputableAtlasCertificateEnumerationCertificate."
                "pointwise_theorem_derived"
            ),
            detail=(
                "missing="
                f"{computable_certificate_enumeration.missing_obligations}; "
                "pointwise_theorem_derived="
                f"{computable_certificate_enumeration.pointwise_theorem_derived}; "
                "source_matches_theorem="
                f"{_enumeration_source_matches_pointwise_theorem(computable_certificate_enumeration, pointwise_open_time_theorem)}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="maximal_classical_total_collision_policy",
            certified=(
                isinstance(
                    policy,
                    MaximalClassicalTotalCollisionPolicyCertificate,
                )
                and policy.proof_certified is True
                and _policy_source_matches_pointwise_theorem(
                    policy,
                    pointwise_open_time_theorem,
                )
            ),
            source=(
                "MaximalClassicalTotalCollisionPolicyCertificate.proof_certified"
            ),
            detail=(
                f"policy_id={policy.policy_id}; "
                f"missing={policy.missing_obligations}; "
                "source_matches_theorem="
                f"{_policy_source_matches_pointwise_theorem(policy, pointwise_open_time_theorem)}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="pointwise_closed_form_chart_primitives",
            certified=True,
            source="PointwiseRegularizedAtlasClosedFormTheoremCertificate",
            detail=",".join(chart_primitives),
        ),
        TheoremPipelineObligation(
            obligation="pointwise_closed_form_finite_target_outcomes",
            certified=True,
            source="PointwiseRegularizedAtlasClosedFormTheoremCertificate",
            detail=",".join(finite_target_outcomes),
        ),
        TheoremPipelineObligation(
            obligation="endpoint_regime_partition_not_required",
            certified=True,
            source="PointwiseRegularizedAtlasClosedFormTheoremCertificate",
            detail="endpoint regimes are optional compression certificates",
        ),
    )
    return PointwiseRegularizedAtlasClosedFormTheoremCertificate(
        closed_form_class_id=str(closed_form_class_id),
        pointwise_open_time_theorem=pointwise_open_time_theorem,
        certificate_language_soundness=certificate_language_soundness,
        computable_certificate_enumeration=computable_certificate_enumeration,
        maximal_classical_total_collision_policy=policy,
        statement=str(statement),
        proof_sketch=str(proof_sketch),
        obligations=obligations,
        chart_primitives=chart_primitives,
        finite_target_certificate_outcomes=finite_target_outcomes,
        endpoint_regime_partition_required=False,
    )


def certify_zero_angular_momentum_triple_collision_convention(
    *,
    convention_id: str = "unspecified",
    regularized_time_parameter_certified: bool = False,
    terminal_collision_value_certified: bool = False,
    continuation_selection_rule_certified: bool = False,
    witness_source: str = "manual_zero_angular_triple_convention",
) -> ZeroAngularMomentumTripleCollisionConventionCertificate:
    """Record the zero-angular-momentum triple-collision continuation convention."""

    return ZeroAngularMomentumTripleCollisionConventionCertificate(
        convention_id=str(convention_id),
        regularized_time_parameter_certified=bool(regularized_time_parameter_certified),
        terminal_collision_value_certified=bool(terminal_collision_value_certified),
        continuation_selection_rule_certified=bool(continuation_selection_rule_certified),
        witness_source=str(witness_source),
    )


def certify_binary_collision_continuation_witness(
    *,
    regularized_pairs: tuple[tuple[int, int], ...] = (),
    pair_regularization_charts_certified: bool = False,
    branch_atlas_certified: bool = False,
    projection_back_to_newtonian_certified: bool = False,
    regularized_time_parameter_certified: bool = False,
    witness_source: str = "manual_binary_collision_continuation",
) -> BinaryCollisionContinuationWitnessCertificate:
    """Record typed binary-collision continuation evidence for all body pairs."""

    return BinaryCollisionContinuationWitnessCertificate(
        regularized_pairs=tuple(tuple(pair) for pair in regularized_pairs),
        pair_regularization_charts_certified=bool(
            pair_regularization_charts_certified
        ),
        branch_atlas_certified=bool(branch_atlas_certified),
        projection_back_to_newtonian_certified=bool(
            projection_back_to_newtonian_certified
        ),
        regularized_time_parameter_certified=bool(
            regularized_time_parameter_certified
        ),
        witness_source=str(witness_source),
    )


def certify_nonzero_angular_momentum_triple_exclusion_witness(
    *,
    nonzero_branch_domain_quantified_certified: bool = False,
    centered_angular_momentum_conservation_certified: bool = False,
    triple_collision_zero_angular_momentum_lemma_certified: bool = False,
    positive_lower_bound_predicate_certified: bool = False,
    witness_source: str = "manual_nonzero_angular_triple_exclusion",
) -> NonzeroAngularMomentumTripleExclusionWitnessCertificate:
    """Record typed evidence for the nonzero-angular triple-exclusion branch."""

    return NonzeroAngularMomentumTripleExclusionWitnessCertificate(
        nonzero_branch_domain_quantified_certified=bool(
            nonzero_branch_domain_quantified_certified
        ),
        centered_angular_momentum_conservation_certified=bool(
            centered_angular_momentum_conservation_certified
        ),
        triple_collision_zero_angular_momentum_lemma_certified=bool(
            triple_collision_zero_angular_momentum_lemma_certified
        ),
        positive_lower_bound_predicate_certified=bool(
            positive_lower_bound_predicate_certified
        ),
        witness_source=str(witness_source),
    )


def certify_collision_continuation_witness(
    *,
    all_binary_pairs_regularized_certified: bool = False,
    binary_branch_atlas_certified: bool = False,
    binary_projection_back_to_newtonian_certified: bool = False,
    binary_collision_time_parameter_certified: bool = False,
    binary_collision_continuation_witness: (
        BinaryCollisionContinuationWitnessCertificate | None
    ) = None,
    nonzero_angular_momentum_triple_exclusion_certified: bool = False,
    nonzero_angular_momentum_triple_exclusion_witness: (
        NonzeroAngularMomentumTripleExclusionWitnessCertificate | None
    ) = None,
    zero_angular_momentum_triple_collision_convention_certified: bool = False,
    triple_collision_continuation_convention: str = "unspecified",
    zero_angular_momentum_triple_collision_convention_witness: (
        ZeroAngularMomentumTripleCollisionConventionCertificate | None
    ) = None,
    witness_source: str = "manual_collision_continuation",
) -> CollisionContinuationWitnessCertificate:
    """Record typed collision-continuation evidence for the Sundman route."""

    return CollisionContinuationWitnessCertificate(
        all_binary_pairs_regularized_certified=bool(all_binary_pairs_regularized_certified),
        binary_branch_atlas_certified=bool(binary_branch_atlas_certified),
        binary_projection_back_to_newtonian_certified=bool(
            binary_projection_back_to_newtonian_certified
        ),
        binary_collision_time_parameter_certified=bool(
            binary_collision_time_parameter_certified
        ),
        binary_collision_continuation_witness=binary_collision_continuation_witness,
        nonzero_angular_momentum_triple_exclusion_certified=bool(
            nonzero_angular_momentum_triple_exclusion_certified
        ),
        nonzero_angular_momentum_triple_exclusion_witness=(
            nonzero_angular_momentum_triple_exclusion_witness
        ),
        zero_angular_momentum_triple_collision_convention_certified=bool(
            zero_angular_momentum_triple_collision_convention_certified
        ),
        triple_collision_continuation_convention=str(
            triple_collision_continuation_convention
        ),
        zero_angular_momentum_triple_collision_convention_witness=(
            zero_angular_momentum_triple_collision_convention_witness
        ),
        witness_source=str(witness_source),
    )


def certify_sundman_general_solution_theorem_witness(
    *,
    recurrence_closure: Any | None = None,
    general_scope_witness: Any | None = None,
    collision_witness: CollisionContinuationWitnessCertificate | None = None,
    collision_continuation_certified: bool = False,
    binary_collision_continuation_certified: bool = False,
    triple_collision_continuation_certified: bool = False,
    witness_source: str = "manual_sundman_theorem_witness",
) -> SundmanGeneralSolutionTheoremWitnessCertificate:
    """Compose the typed evidence needed by the Sundman general-solution route."""

    return SundmanGeneralSolutionTheoremWitnessCertificate(
        recurrence_closure=recurrence_closure,
        scope_witness=general_scope_witness,
        collision_witness=collision_witness,
        collision_continuation_certified=_strict_bool(
            collision_continuation_certified,
            "collision_continuation_certified",
        ),
        binary_collision_continuation_certified=_strict_bool(
            binary_collision_continuation_certified,
            "binary_collision_continuation_certified",
        ),
        triple_collision_continuation_certified=_strict_bool(
            triple_collision_continuation_certified,
            "triple_collision_continuation_certified",
        ),
        witness_source=str(witness_source),
    )


def _scope_requirement_statuses_from_witness(
    witness: Any | None,
) -> tuple[GeneralSolutionRequirementStatus, ...]:
    return (
        _requirement_status(
            witness,
            "arbitrary_positive_masses",
            "the theorem must cover every positive mass triple, not only tested masses",
            "arbitrary_positive_masses_certified",
            "general_solution_scope_certified",
            required="all positive masses are covered by the proof",
        ),
        _requirement_status(
            witness,
            "arbitrary_noncollision_initial_data",
            "the theorem must cover every non-collision initial position/velocity state",
            "arbitrary_noncollision_initial_data_certified",
            "general_solution_scope_certified",
            required="all non-collision initial states are covered by the proof",
        ),
        _requirement_status(
            witness,
            "all_real_target_times",
            "the theorem must return positions for every real target time in the claimed continuation",
            "all_real_target_times_certified",
            "general_solution_scope_certified",
            required="all real target times in the claimed continuation are covered",
        ),
        _requirement_status(
            witness,
            "lift_construct_project_verify_pipeline",
            "the lift, construction, projection, and Newtonian verification steps must all be certified",
            "lift_construct_project_verify_certified",
            "general_solution_scope_certified",
            required="lift, construction, projection, and verification are certified as one pipeline",
        ),
        _requirement_status(
            witness,
            "newton_equations_full_interval",
            "projected positions must satisfy Newton's equations on the full claimed interval",
            "newton_equations_full_interval_certified",
            "general_solution_scope_certified",
            required="projected positions satisfy Newton's equations over the full claimed interval",
        ),
    )


def _scope_requirement_statuses_from_constructor_theorem(
    theorem_certificate: Any,
    *,
    override_certified: bool | None = None,
    override_blockers: tuple[str, ...] | None = None,
) -> tuple[GeneralSolutionRequirementStatus, ...]:
    full_certified = (
        bool(override_certified)
        if override_certified is not None
        else _constructor_theorem_full_general_solution_certified(theorem_certificate)
    )
    blockers = (
        tuple(override_blockers)
        if override_blockers is not None
        else _constructor_theorem_blocking_obligations(theorem_certificate)
    )
    observed = _constructor_theorem_route_summary(theorem_certificate)
    witness_field = _constructor_theorem_scope_witness_field(theorem_certificate)
    if _is_pointwise_regularized_atlas_route_certificate(theorem_certificate):
        positive_reason = (
            "pointwise open-time theorem quantifies positive masses for exact "
            "computable inputs"
        )
        noncollision_reason = (
            "pointwise open-time theorem applies the finite-target atlas-or-stop "
            "theorem to every exact noncollision initial state"
        )
        all_time_reason = (
            "pointwise open-time theorem gives the countable compact exhaustion "
            "for every real finite target"
        )
        pipeline_reason = (
            "pointwise route requires finite verifier-checkable atlas-or-stop "
            "construction plus certificate-language soundness"
        )
        newton_reason = (
            "Newton residual semantics are supplied by the certificate language "
            "soundness theorem for every enumerated chart certificate"
        )
    elif _is_open_time_locally_finite_theorem(theorem_certificate):
        positive_reason = (
            "open-time theorem evidence must be upgraded from a pointwise "
            "finite-target constructor to arbitrary positive masses"
        )
        noncollision_reason = (
            "open-time theorem evidence must certify finite-target "
            "atlas-or-stop completeness for every noncollision initial state"
        )
        all_time_reason = (
            "open-time theorem evidence must certify the countable compact "
            "exhaustion for arbitrary finite targets, not only the checked prefix"
        )
        pipeline_reason = (
            "open-time theorem evidence must certify ordinary, binary, and "
            "total-collision policy selection as one arbitrary-input pipeline"
        )
        newton_reason = (
            "open-time theorem evidence must verify Newton residuals on every "
            "chart in the countable maximal-interval atlas"
        )
    else:
        positive_reason = (
            "constructor theorem certificate must exhaust all admissible "
            "positive-mass regimes"
        )
        noncollision_reason = (
            "constructor theorem certificate must classify every noncollision "
            "initial state"
        )
        all_time_reason = (
            "constructor theorem certificate must provide all-time compact-atlas "
            "coverage"
        )
        pipeline_reason = (
            "constructor theorem certificate must certify the integrated atlas "
            "pipeline"
        )
        newton_reason = (
            "constructor theorem certificate must verify Newton residuals on the "
            "full atlas"
        )
    return (
        GeneralSolutionRequirementStatus(
            requirement="arbitrary_positive_masses",
            certified=full_certified,
            reason=positive_reason,
            witness_field=witness_field,
            required="all positive masses are covered by the constructor theorem",
            observed=observed,
            blocking_obligations=blockers,
        ),
        GeneralSolutionRequirementStatus(
            requirement="arbitrary_noncollision_initial_data",
            certified=full_certified,
            reason=noncollision_reason,
            witness_field=witness_field,
            required="all noncollision initial states are covered by the constructor theorem",
            observed=observed,
            blocking_obligations=blockers,
        ),
        GeneralSolutionRequirementStatus(
            requirement="all_real_target_times",
            certified=full_certified,
            reason=all_time_reason,
            witness_field=witness_field,
            required="all real target times are covered by the constructor theorem",
            observed=observed,
            blocking_obligations=blockers,
        ),
        GeneralSolutionRequirementStatus(
            requirement="lift_construct_project_verify_pipeline",
            certified=full_certified,
            reason=pipeline_reason,
            witness_field=witness_field,
            required="lift, construction, projection, and verification are certified",
            observed=observed,
            blocking_obligations=blockers,
        ),
        GeneralSolutionRequirementStatus(
            requirement="newton_equations_full_interval",
            certified=full_certified,
            reason=newton_reason,
            witness_field=witness_field,
            required="projected positions satisfy Newton's equations over the full interval",
            observed=observed,
            blocking_obligations=blockers,
        ),
    )


def _is_open_time_locally_finite_theorem(theorem_certificate: Any) -> bool:
    return bool(
        isinstance(
            theorem_certificate,
            OpenTimeLocallyFiniteAtlasTheoremCertificate,
        )
        and getattr(theorem_certificate, "theorem_id", None)
        == "open_time_locally_finite_atlas"
    )


def _is_pointwise_open_time_locally_finite_theorem(theorem_certificate: Any) -> bool:
    return bool(
        isinstance(
            theorem_certificate,
            PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate,
        )
        and getattr(theorem_certificate, "theorem_id", None)
        == "pointwise_open_time_locally_finite_atlas"
    )


def _is_pointwise_regularized_atlas_closed_form_theorem(
    theorem_certificate: Any,
) -> bool:
    return bool(
        isinstance(
            theorem_certificate,
            PointwiseRegularizedAtlasClosedFormTheoremCertificate,
        )
        and getattr(theorem_certificate, "theorem_id", None)
        == "pointwise_regularized_atlas_closed_form"
    )


def _is_general_solution_theorem(theorem_certificate: Any) -> bool:
    return bool(
        isinstance(theorem_certificate, GeneralSolutionTheoremCertificate)
        and getattr(theorem_certificate, "theorem_id", None)
        == "constructive_sundman_atlas_general_solution"
    )


def _enumeration_source_matches_pointwise_theorem(
    enumeration: Any,
    theorem_certificate: Any,
) -> bool:
    if not isinstance(
        theorem_certificate,
        PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate,
    ):
        return False
    finite_target_theorem = getattr(theorem_certificate, "finite_target_theorem", None)
    theorem_chart_families = tuple(
        str(item)
        for item in getattr(finite_target_theorem, "chart_families", ()) or ()
    )
    theorem_allowed_outcomes = tuple(
        str(item)
        for item in getattr(finite_target_theorem, "allowed_outcomes", ()) or ()
    )
    return bool(
        isinstance(enumeration, ComputableAtlasCertificateEnumerationCertificate)
        and enumeration.source_theorem_id == theorem_certificate.theorem_id
        and enumeration.source_theorem_proof_certified is True
        and theorem_certificate.proof_certified is True
        and enumeration.source_theorem_dimension == theorem_certificate.dimension
        and enumeration.source_theorem_input_model == theorem_certificate.input_model
        and enumeration.source_total_collision_policy_id
        == theorem_certificate.total_collision_policy_id
        and tuple(enumeration.finite_target_chart_families) == theorem_chart_families
        and (
            tuple(enumeration.finite_target_allowed_outcomes)
            == theorem_allowed_outcomes
        )
    )


def _policy_source_matches_pointwise_theorem(
    policy: Any,
    theorem_certificate: Any,
) -> bool:
    if not isinstance(
        theorem_certificate,
        PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate,
    ):
        return False
    return bool(
        isinstance(policy, MaximalClassicalTotalCollisionPolicyCertificate)
        and policy.source_theorem_id == theorem_certificate.theorem_id
        and policy.source_theorem_proof_certified is True
        and theorem_certificate.proof_certified is True
        and policy.source_theorem_dimension == theorem_certificate.dimension
        and policy.source_theorem_input_model == theorem_certificate.input_model
        and policy.source_total_collision_policy_id
        == theorem_certificate.total_collision_policy_id
        and policy.policy_id == theorem_certificate.total_collision_policy_id
    )


def _is_pointwise_regularized_atlas_route_certificate(
    theorem_certificate: Any,
) -> bool:
    return bool(
        _is_pointwise_open_time_locally_finite_theorem(theorem_certificate)
        or _is_pointwise_regularized_atlas_closed_form_theorem(theorem_certificate)
    )


def _is_regularized_atlas_theorem(theorem_certificate: Any) -> bool:
    return bool(
        _is_open_time_locally_finite_theorem(theorem_certificate)
        or _is_pointwise_open_time_locally_finite_theorem(theorem_certificate)
        or _is_pointwise_regularized_atlas_closed_form_theorem(theorem_certificate)
    )


def _constructor_theorem_blocking_obligations(
    theorem_certificate: Any,
) -> tuple[str, ...]:
    missing = list(getattr(theorem_certificate, "missing_obligations", ()))
    if not (
        _is_open_time_locally_finite_theorem(theorem_certificate)
        or _is_pointwise_open_time_locally_finite_theorem(theorem_certificate)
        or _is_pointwise_regularized_atlas_closed_form_theorem(theorem_certificate)
        or _is_general_solution_theorem(theorem_certificate)
    ):
        missing.append("general_solution_theorem_constructor_certificate")
    missing.extend(
        tuple(getattr(theorem_certificate, "analytic_lemma_audit_blockers", ()))
    )
    if _is_open_time_locally_finite_theorem(theorem_certificate):
        missing.extend(
            tuple(
                getattr(
                    theorem_certificate,
                    "finite_target_completeness_missing_obligations",
                    (),
                )
            )
        )
        if getattr(theorem_certificate, "scoped_set_valued_constructor_only", False):
            missing.append("arbitrary_interval_input_partition_generation")
        missing.extend(
            f"independent_chart_verifier_arithmetic:{blocker}"
            for blocker in getattr(
                theorem_certificate,
                "independent_chart_verifier_arithmetic_blockers",
                (),
            )
        )
    return tuple(dict.fromkeys(str(obligation) for obligation in missing if obligation))


def _constructor_theorem_route_summary(theorem_certificate: Any) -> str:
    return str(
        getattr(
            theorem_certificate,
            "route_summary",
            "constructor theorem certificate supplied",
        )
    )


def _constructor_theorem_scope_witness_field(theorem_certificate: Any) -> str:
    if _is_pointwise_regularized_atlas_closed_form_theorem(theorem_certificate):
        return (
            "PointwiseRegularizedAtlasClosedFormTheoremCertificate."
            "proof_certified"
        )
    if _is_pointwise_open_time_locally_finite_theorem(theorem_certificate):
        return (
            "PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate."
            "pointwise_open_time_proof_certified"
        )
    if _is_open_time_locally_finite_theorem(theorem_certificate):
        return (
            "OpenTimeLocallyFiniteAtlasTheoremCertificate."
            "arbitrary_finite_target_completeness_certified"
        )
    if not _is_general_solution_theorem(theorem_certificate):
        return "unsupported_constructor_theorem_certificate"
    return "GeneralSolutionTheoremCertificate.full_general_solution_certified"


def _constructor_theorem_global_series_certified(theorem_certificate: Any) -> bool:
    if _is_open_time_locally_finite_theorem(theorem_certificate):
        return False
    return bool(
        _is_general_solution_theorem(theorem_certificate)
        and theorem_certificate.regime_theorem_certified
    )


def _constructor_theorem_full_general_solution_certified(
    theorem_certificate: Any,
) -> bool:
    if _is_pointwise_regularized_atlas_closed_form_theorem(theorem_certificate):
        return bool(
            getattr(theorem_certificate, "proof_certified", False) is True
            and not _constructor_theorem_blocking_obligations(theorem_certificate)
        )
    if _is_pointwise_open_time_locally_finite_theorem(theorem_certificate):
        return bool(
            getattr(theorem_certificate, "proof_certified", False) is True
            and not _constructor_theorem_blocking_obligations(theorem_certificate)
        )
    if _is_open_time_locally_finite_theorem(theorem_certificate):
        return bool(
            getattr(theorem_certificate, "proof_certified", False) is True
            and not _constructor_theorem_blocking_obligations(theorem_certificate)
        )
    return bool(
        _is_general_solution_theorem(theorem_certificate)
        and theorem_certificate.full_general_solution_certified
    )


def _constructor_theorem_independent_verifier_certified(
    theorem_certificate: Any,
) -> bool:
    verifier = getattr(
        theorem_certificate,
        "independent_chart_verifier_certificate",
        None,
    )
    return bool(
        getattr(theorem_certificate, "independent_chart_verifier_certified", False)
        is True
        and type(verifier) is IndependentChartVerifierCertificate
        and verifier.certified is True
        and verifier.proof_grade_finite_atlas_bundle_certified is True
    )


def _constructor_theorem_regularized_atlas_blockers(
    theorem_certificate: Any,
) -> tuple[str, ...]:
    blockers = list(_constructor_theorem_blocking_obligations(theorem_certificate))
    if _is_pointwise_regularized_atlas_closed_form_theorem(theorem_certificate):
        if getattr(theorem_certificate, "proof_certified", False) is not True:
            blockers.append("pointwise_regularized_atlas_closed_form_proof")
        return tuple(dict.fromkeys(str(blocker) for blocker in blockers if blocker))
    if _is_pointwise_open_time_locally_finite_theorem(theorem_certificate):
        if getattr(theorem_certificate, "proof_certified", False) is not True:
            blockers.append("pointwise_open_time_atlas_proof")
        return tuple(dict.fromkeys(str(blocker) for blocker in blockers if blocker))
    if not _is_open_time_locally_finite_theorem(theorem_certificate):
        blockers.append("open_time_locally_finite_atlas_proof")
    if getattr(theorem_certificate, "proof_certified", False) is not True:
        blockers.append("audited_or_machine_checked_open_time_atlas_proof")
    if not _constructor_theorem_independent_verifier_certified(theorem_certificate):
        blockers.append("independent_chart_verifier")
    return tuple(dict.fromkeys(str(blocker) for blocker in blockers if blocker))


def _constructor_theorem_regularized_atlas_certified(
    theorem_certificate: Any,
) -> bool:
    return bool(
        _is_regularized_atlas_theorem(theorem_certificate)
        and not _constructor_theorem_regularized_atlas_blockers(theorem_certificate)
    )


def _constructor_theorem_global_witness_field(theorem_certificate: Any) -> str:
    if _is_pointwise_regularized_atlas_closed_form_theorem(theorem_certificate):
        return "PointwiseRegularizedAtlasClosedFormTheoremCertificate"
    if _is_pointwise_open_time_locally_finite_theorem(theorem_certificate):
        return "PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate"
    if _is_open_time_locally_finite_theorem(theorem_certificate):
        return "OpenTimeLocallyFiniteAtlasTheoremCertificate"
    return "GeneralSolutionTheoremCertificate"


def _constructor_theorem_global_required(theorem_certificate: Any) -> str:
    if _is_open_time_locally_finite_theorem(theorem_certificate):
        return (
            "open-time finite-target atlas-or-stop theorem has no "
            "arbitrary-input completeness blockers"
        )
    return "constructor theorem certificate has no missing global proof obligations"


def certify_general_closed_form_solution_target(
    closed_form_class: str = "unspecified",
    *,
    compact_sundman_witness: Any | None = None,
    general_scope_witness: Any | None = None,
    general_theorem_certificate: Any | None = None,
    certificate_language_soundness_certificate: (
        CertificateLanguageSoundnessCertificate | None
    ) = None,
    computable_atlas_enumeration_certificate: (
        ComputableAtlasCertificateEnumerationCertificate | None
    ) = None,
    closed_form_function_class_witness: (
        ClosedFormFunctionClassWitnessCertificate | None
    ) = None,
) -> GeneralClosedFormSolutionCertificate:
    """Audit the full requested theorem target against the supplied evidence."""

    if isinstance(general_theorem_certificate, bool) or (
        general_theorem_certificate is not None
        and type(general_theorem_certificate).__name__ == "bool_"
    ):
        raise TypeError(
            "general_theorem_certificate must be a constructor-derived certificate object"
        )

    closed_form_certificate = certify_closed_form_target(
        closed_form_class,
        compact_sundman_witness=compact_sundman_witness,
        closed_form_function_class_witness=closed_form_function_class_witness,
    )
    regularized_atlas_requested = (
        closed_form_certificate.normalized_class == "regularized_locally_finite_atlas"
    )
    route_requirement = (
        "regularized_locally_finite_atlas"
        if regularized_atlas_requested
        else "compact_sundman_global_series"
    )
    route_required = (
        "open-time locally finite atlas proof, finite-target completeness, "
        "collision semantics, and independent chart verifier are certified"
        if regularized_atlas_requested
        else "global_series_certified=True with no compact-Sundman proof-obligation blockers"
    )
    collision_requirement = (
        "collision_semantics"
        if regularized_atlas_requested
        else "collision_continuation"
    )
    collision_required = (
        "binary collisions are regularized and total collisions stop under the maximal-classical policy"
        if regularized_atlas_requested
        else "binary and triple collision continuation are certified globally"
    )
    witness = compact_sundman_witness
    if general_theorem_certificate is not None:
        if regularized_atlas_requested:
            theorem_missing = _constructor_theorem_regularized_atlas_blockers(
                general_theorem_certificate,
            )
            if _is_pointwise_open_time_locally_finite_theorem(
                general_theorem_certificate
            ):
                if not isinstance(
                    certificate_language_soundness_certificate,
                    CertificateLanguageSoundnessCertificate,
                ):
                    theorem_missing = (
                        *theorem_missing,
                        "certificate_language_soundness_constructor_certificate",
                    )
                if not bool(
                    getattr(
                        certificate_language_soundness_certificate,
                        "proof_certified",
                        False,
                    )
                ):
                    theorem_missing = (
                        *theorem_missing,
                        "certificate_language_soundness",
                        *(
                            f"certificate_language_soundness:{obligation}"
                            for obligation in getattr(
                                certificate_language_soundness_certificate,
                                "missing_obligations",
                                (),
                            )
                        ),
                    )
                elif not bool(
                    getattr(
                        certificate_language_soundness_certificate,
                        "checker_kernel_derived",
                        False,
                    )
                ):
                    theorem_missing = (
                        *theorem_missing,
                        "certificate_language_soundness_checker_kernel_derived",
                    )
                if not isinstance(
                    computable_atlas_enumeration_certificate,
                    ComputableAtlasCertificateEnumerationCertificate,
                ):
                    theorem_missing = (
                        *theorem_missing,
                        (
                            "computable_atlas_certificate_enumeration_"
                            "constructor_certificate"
                        ),
                    )
                if not bool(
                    getattr(
                        computable_atlas_enumeration_certificate,
                        "proof_certified",
                        False,
                    )
                ):
                    theorem_missing = (
                        *theorem_missing,
                        "computable_atlas_certificate_enumeration",
                        *(
                            f"computable_atlas_certificate_enumeration:{obligation}"
                            for obligation in getattr(
                                computable_atlas_enumeration_certificate,
                                "missing_obligations",
                                (),
                            )
                        ),
                    )
                elif not bool(
                    getattr(
                        computable_atlas_enumeration_certificate,
                        "pointwise_theorem_derived",
                        False,
                    )
                ):
                    theorem_missing = (
                        *theorem_missing,
                        (
                            "computable_atlas_certificate_enumeration_"
                            "pointwise_theorem_derived"
                        ),
                    )
                elif not _enumeration_source_matches_pointwise_theorem(
                    computable_atlas_enumeration_certificate,
                    general_theorem_certificate,
                ):
                    theorem_missing = (
                        *theorem_missing,
                        (
                            "computable_atlas_certificate_enumeration_"
                            "source_matches_theorem"
                        ),
                    )
                theorem_missing = tuple(dict.fromkeys(theorem_missing))
            global_series_certified = _constructor_theorem_regularized_atlas_certified(
                general_theorem_certificate,
            ) and not theorem_missing
            collision_continuation_certified = global_series_certified
        else:
            theorem_missing = _constructor_theorem_blocking_obligations(
                general_theorem_certificate,
            )
            global_series_certified = _constructor_theorem_global_series_certified(
                general_theorem_certificate,
            )
            collision_continuation_certified = (
                _constructor_theorem_full_general_solution_certified(
                    general_theorem_certificate,
                )
            )
        compact_blockers = theorem_missing
        global_reason = _constructor_theorem_route_summary(general_theorem_certificate)
        global_witness_field = _constructor_theorem_global_witness_field(
            general_theorem_certificate,
        )
        global_required = (
            route_required
            if regularized_atlas_requested
            else _constructor_theorem_global_required(general_theorem_certificate)
        )
        global_observed = global_reason
        collision_witness_field = _constructor_theorem_scope_witness_field(
            general_theorem_certificate,
        )
        collision_observed = global_reason
        if (
            (
                closed_form_certificate.infinite_series_route
                or closed_form_certificate.regularized_atlas_route
            )
            and not theorem_missing
            and (
                global_series_certified
                if _is_pointwise_regularized_atlas_route_certificate(
                    general_theorem_certificate
                )
                else _constructor_theorem_full_general_solution_certified(
                    general_theorem_certificate,
                )
            )
        ):
            certified_status = (
                (
                    "certified_pointwise_regularized_atlas_route"
                    if _is_pointwise_regularized_atlas_route_certificate(
                        general_theorem_certificate
                    )
                    else (
                        "certified_set_valued_constructor_regularized_atlas_route"
                        if _is_open_time_locally_finite_theorem(
                            general_theorem_certificate
                        )
                        else "certified_regularized_atlas_route"
                    )
                )
                if regularized_atlas_requested
                else "certified_infinite_series_route"
            )
            closed_form_certificate = ClosedFormTargetCertificate(
                requested_class=closed_form_certificate.requested_class,
                normalized_class=closed_form_certificate.normalized_class,
                status=certified_status,
                reason=global_reason,
                theorem_references=(
                    (_regularized_atlas_reference(),)
                    if regularized_atlas_requested
                    else (_sundman_reference(),)
                ),
                infinite_series_route=closed_form_certificate.infinite_series_route,
                series_route_certified=closed_form_certificate.infinite_series_route,
                regularized_atlas_route=closed_form_certificate.regularized_atlas_route,
                atlas_route_certified=closed_form_certificate.regularized_atlas_route,
                general_solution_scope_certified=True,
                general_solution_certified=True,
            )
    else:
        global_series_certified = _witness_flag(witness, "global_series_certified")
        collision_continuation_certified = _witness_flag(
            witness,
            "collision_continuation_obligations_certified",
            "global_collision_continuation_certified",
        )
        missing_obligations = tuple(
            getattr(witness, "missing_global_proof_obligations", ())
        )
        detailed_missing_obligations = tuple(
            detail.obligation
            for detail in getattr(witness, "global_proof_obligation_details", ())
            if not bool(getattr(detail, "certified", False))
        )
        compact_blockers = detailed_missing_obligations or missing_obligations
        if global_series_certified and not missing_obligations:
            global_reason = "compact-Sundman global series obligations are certified"
        elif missing_obligations:
            global_reason = "missing compact-Sundman obligations: " + ", ".join(
                missing_obligations
            )
        else:
            global_reason = "no compact-Sundman global series witness was supplied"
        global_witness_field = "global_series_certified"
        global_required = route_required
        global_observed = (
            "global_series_certified=True"
            if global_series_certified
            else _observed_witness_flags(witness, "global_series_certified")
        )
        collision_witness_field = (
            "collision_continuation_obligations_certified "
            "or global_collision_continuation_certified"
        )
        collision_observed = _observed_witness_flags(
            witness,
            "collision_continuation_obligations_certified",
            "global_collision_continuation_certified",
        )
    delegated_scope_witness = getattr(witness, "scope_witness", None)
    if general_scope_witness is not None:
        if hasattr(general_scope_witness, "requirement_statuses"):
            scope_requirement_statuses = general_scope_witness.requirement_statuses
        else:
            scope_requirement_statuses = _scope_requirement_statuses_from_witness(
                general_scope_witness,
            )
    elif general_theorem_certificate is not None:
        scope_requirement_statuses = _scope_requirement_statuses_from_constructor_theorem(
            general_theorem_certificate,
            override_certified=(
                global_series_certified
                if (
                    regularized_atlas_requested
                    and _is_pointwise_regularized_atlas_route_certificate(
                        general_theorem_certificate
                    )
                )
                else None
            ),
            override_blockers=(
                compact_blockers
                if (
                    regularized_atlas_requested
                    and _is_pointwise_regularized_atlas_route_certificate(
                        general_theorem_certificate
                    )
                )
                else None
            ),
        )
    elif delegated_scope_witness is not None and hasattr(
        delegated_scope_witness, "requirement_statuses"
    ):
        scope_requirement_statuses = delegated_scope_witness.requirement_statuses
    else:
        scope_requirement_statuses = _scope_requirement_statuses_from_witness(witness)

    requirement_statuses = (
        GeneralSolutionRequirementStatus(
            requirement="allowed_closed_form_class",
            certified=bool(
                not closed_form_certificate.definition_required
                and not closed_form_certificate.finite_closed_form_obstructed
                and (
                    closed_form_certificate.infinite_series_route
                    or closed_form_certificate.regularized_atlas_route
                )
            ),
            reason=closed_form_certificate.reason,
            witness_field="requested_class",
            required="requested class must be a theorem-compatible infinite series or regularized atlas class",
            observed=closed_form_certificate.normalized_class,
            blocking_obligations=closed_form_certificate.missing_requirements,
        ),
        *scope_requirement_statuses[:-1],
        GeneralSolutionRequirementStatus(
            requirement=route_requirement,
            certified=bool(global_series_certified and not compact_blockers),
            reason=global_reason,
            witness_field=global_witness_field,
            required=route_required if regularized_atlas_requested else global_required,
            observed=global_observed,
            blocking_obligations=compact_blockers,
        ),
        GeneralSolutionRequirementStatus(
            requirement=collision_requirement,
            certified=collision_continuation_certified,
            reason=(
                "binary continuation and maximal-classical total-collision stop semantics must be certified"
                if regularized_atlas_requested
                else "binary and triple collision continuation must be certified globally"
            ),
            witness_field=collision_witness_field,
            required=collision_required,
            observed=collision_observed,
            blocking_obligations=tuple(
                compact_blockers
                if general_theorem_certificate is not None
                else (
                    obligation
                    for obligation in (
                        "binary_collision_continuation",
                        "triple_collision_continuation",
                    )
                    if obligation in compact_blockers
                )
            ),
        ),
        scope_requirement_statuses[-1],
    )

    if (
        (
            closed_form_certificate.series_route_certified
            or closed_form_certificate.atlas_route_certified
        )
        and _requirement_status_ledger_certified(requirement_statuses)
    ):
        closed_form_certificate = ClosedFormTargetCertificate(
            requested_class=closed_form_certificate.requested_class,
            normalized_class=closed_form_certificate.normalized_class,
            status=closed_form_certificate.status,
            reason=closed_form_certificate.reason,
            theorem_references=closed_form_certificate.theorem_references,
            missing_requirements=closed_form_certificate.missing_requirements,
            compact_sundman_obligations=closed_form_certificate.compact_sundman_obligations,
            finite_closed_form_obstructed=closed_form_certificate.finite_closed_form_obstructed,
            internal_obstruction_certified=closed_form_certificate.internal_obstruction_certified,
            internal_obstruction_evidence=closed_form_certificate.internal_obstruction_evidence,
            infinite_series_route=closed_form_certificate.infinite_series_route,
            series_route_certified=closed_form_certificate.series_route_certified,
            regularized_atlas_route=closed_form_certificate.regularized_atlas_route,
            atlas_route_certified=closed_form_certificate.atlas_route_certified,
            general_solution_scope_certified=True,
            general_solution_certified=True,
        )

    return GeneralClosedFormSolutionCertificate(
        closed_form_certificate=closed_form_certificate,
        requirement_statuses=requirement_statuses,
    )
