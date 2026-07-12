"""Pointwise finite-target completeness theorem scaffold.

This module states the theorem that now sits between finite-time constructor
examples and the open-time compact-exhaustion theorem.  It deliberately
separates the mathematical atlas-or-stop statement from the stronger claim
that today's search procedure can always find the certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .branch_event_tree import (
    BranchEventTreeCertificate,
    certify_supplied_branch_event_tree,
)
from .stratified_branch_tree import (
    AffineBoxDecisionFunctionSpec,
    AffineBoxDecisionArrangementStratificationCertificate,
    AffineHalfspaceArrangement3DStratificationCertificate,
    AffineHalfspaceArrangementStratificationCertificate,
    AffineHalfspaceDecisionStratificationCertificate,
    PolynomialDecisionFunctionSpec,
    PolynomialDecisionArrangementStratificationCertificate,
    PolynomialDecisionStratificationCertificate,
    RationalDecisionFunctionSpec,
    RationalDecisionArrangementStratificationCertificate,
    RationalDecisionStratificationCertificate,
    RecursiveStratifiedBranchEventConsumptionCertificate,
    StratifiedBranchTreeCertificate,
    TaylorModelDecisionFunctionSpec,
    TaylorModelDecisionArrangementStratificationCertificate,
    TaylorModelDecisionStratificationCertificate,
    certify_affine_box_decision_arrangement_stratified_branch_event_tree,
    certify_affine_decision_arrangement_stratified_branch_event_tree,
    certify_affine_decision_stratified_branch_event_tree,
    certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree,
    certify_affine_box_decision_arrangement_recursive_consumption,
    certify_affine_halfspace_3d_arrangement_recursive_consumption,
    certify_affine_halfspace_arrangement_stratified_branch_event_tree,
    certify_affine_halfspace_arrangement_recursive_consumption,
    certify_affine_halfspace_decision_stratified_branch_event_tree,
    certify_affine_halfspace_decision_recursive_consumption,
    certify_polynomial_decision_arrangement_stratified_branch_event_tree,
    certify_polynomial_decision_arrangement_recursive_consumption,
    certify_polynomial_decision_stratified_branch_event_tree,
    certify_polynomial_decision_recursive_consumption,
    certify_quadratic_decision_arrangement_stratified_branch_event_tree,
    certify_rational_decision_arrangement_stratified_branch_event_tree,
    certify_rational_decision_arrangement_recursive_consumption,
    certify_rational_decision_stratified_branch_event_tree,
    certify_rational_decision_recursive_consumption,
    certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree,
    certify_sturm_polynomial_decision_stratified_branch_event_tree,
    certify_sturm_rational_decision_arrangement_stratified_branch_event_tree,
    certify_sturm_rational_decision_stratified_branch_event_tree,
    certify_taylor_model_decision_arrangement_stratified_branch_event_tree,
    certify_taylor_model_decision_arrangement_recursive_consumption,
    certify_taylor_model_decision_stratified_branch_event_tree,
    certify_taylor_model_decision_recursive_consumption,
)
from .certificate_checker import (
    CertificateCheckObligation,
    check_total_collision_fuchsian_stop_chart,
    check_total_collision_generalized_fuchsian_stop_chart,
)
from .certificate_language import (
    GeneralizedFuchsianRemainderMajorantCertificate,
    PrimitiveCauchyTailInputCertificate,
    total_collision_fuchsian_stop_chart_certificate_from_branch,
    total_collision_generalized_fuchsian_stop_chart_certificate_from_branch,
)
from .dynamics import energy
from .event_recurrence import (
    PrimitiveCauchyTailInput,
    construct_primitive_cauchy_tail_input,
)
from .fuchsian import (
    FiniteFuchsianLogBranch,
    FiniteFuchsianLogCompactTimeIsolationCertificate,
    FiniteFuchsianLogPrimitiveCauchyInputs,
    FiniteFuchsianLogTotalCollisionIsolationCertificate,
    FuchsianShapeBranch,
    certify_finite_fuchsian_log_total_collision_isolation,
)
from .general_solution_theorem import TheoremPipelineObligation


FINITE_TARGET_COMPLETENESS_CHART_FAMILIES = (
    "ordinary_taylor",
    "planar_levi_civita_binary",
    "spatial_ks_binary",
    "total_collision_stop",
)

FINITE_TARGET_COMPLETENESS_OUTCOMES = (
    "finite_atlas_reaches_target",
    "unselected_total_collision_before_target",
)

FINITE_TARGET_SUPPORTED_DIMENSIONS = (2, 3)

FINITE_TARGET_POINT_INPUT_MODELS = (
    "exact_point_positive_mass_noncollision",
    "computable_point_positive_mass_noncollision",
)

FINITE_TARGET_MAXIMAL_CLASSICAL_POLICIES = (
    "maximal_classical_stop",
    "maximal_classical_stop_at_total_collision",
)

FINITE_TARGET_TIER_A_ANALYTIC_LEMMA_IDS = (
    "three_body_painleve_no_noncollision_singularities",
    "all_pair_binary_regularization",
    "binary_collision_isolation",
    "binary_accumulation_implies_total_collision",
    "compact_collision_free_taylor_cover",
    "finite_chart_chain_concatenation",
    "target_or_stop_dichotomy",
)

FINITE_TARGET_CRITICAL_ANALYTIC_LEMMA_IDS = (
    "binary_degenerate_total_collision_exclusion",
    "reduced_hyperbolic_total_collision_entry",
    "poincare_dulac_fuchsian_log_selector_completeness",
    "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data",
    "arbitrary_total_collision_germ_entry_to_stop_chart",
    "total_collision_stop_chart_existence",
)

FINITE_TARGET_TIER_B_ANALYTIC_LEMMA_IDS = FINITE_TARGET_CRITICAL_ANALYTIC_LEMMA_IDS

FINITE_TARGET_ANALYTIC_LEMMA_TIER_BY_ID = {
    **{
        lemma_id: "tier_a_classical_or_structural"
        for lemma_id in FINITE_TARGET_TIER_A_ANALYTIC_LEMMA_IDS
    },
    **{
        lemma_id: "tier_b_total_collision_entry_frontier"
        for lemma_id in FINITE_TARGET_TIER_B_ANALYTIC_LEMMA_IDS
    },
}

CONSTRUCTOR_DERIVED_RECURSIVE_SOURCE_SCOPES = {
    "SturmPolynomialDecisionStratification": (
        "finite_sturm_polynomial_decision_stratification_interval_boxes",
        (
            "constructor-derived single Sturm polynomial decision "
            "stratification with exact rational root isolation, "
            "multiplicity-preserving equality strata, strict sign cells, and "
            "recursive equality children"
        ),
        "single Sturm polynomial decision stratification",
    ),
    "SturmPolynomialDecisionArrangement": (
        "finite_sturm_polynomial_decision_arrangement_interval_boxes",
        (
            "constructor-derived Sturm polynomial decision arrangement "
            "with exact rational root isolation and recursive equality "
            "children"
        ),
        "Sturm polynomial decision arrangement",
    ),
    "AffineDecisionStratification": (
        "finite_affine_decision_stratification_interval_boxes",
        (
            "constructor-derived one-dimensional affine decision "
            "stratification with coefficient-derived root brackets, strict "
            "sign cells, and recursive equality children"
        ),
        "single affine decision stratification",
    ),
    "AffineDecisionArrangement": (
        "finite_affine_decision_arrangement_interval_boxes",
        (
            "constructor-derived one-dimensional affine decision arrangement "
            "with coefficient-derived roots, one-sided boundary brackets, "
            "grouped simultaneous equality strata, and recursive equality "
            "children"
        ),
        "affine decision arrangement",
    ),
    "QuadraticDoubleRootArrangement": (
        "finite_quadratic_double_root_decision_arrangement_interval_boxes",
        (
            "constructor-derived quadratic double-root decision arrangement "
            "with tangent equality strata and recursive equality children"
        ),
        "quadratic double-root decision arrangement",
    ),
    "PolynomialDecisionArrangement": (
        "finite_polynomial_decision_arrangement_interval_boxes",
        (
            "constructor-derived polynomial decision arrangement with "
            "verified simple root brackets and recursive equality children"
        ),
        "polynomial decision arrangement",
    ),
    "PolynomialDecisionStratification": (
        "finite_polynomial_decision_stratification_interval_boxes",
        (
            "constructor-derived single-polynomial decision stratification "
            "with verified simple root brackets, strict sign cells, and "
            "recursive equality children"
        ),
        "single-polynomial decision stratification",
    ),
    "RationalDecisionStratification": (
        "finite_rational_decision_stratification_interval_boxes",
        (
            "constructor-derived one-dimensional rational decision "
            "stratification with denominator exclusion, numerator-root "
            "sign cells, and recursive equality children"
        ),
        "rational decision stratification",
    ),
    "SturmRationalDecisionStratification": (
        "finite_sturm_rational_decision_stratification_interval_boxes",
        (
            "constructor-derived one-dimensional rational decision "
            "stratification with exact Sturm numerator-root isolation, "
            "closed-interval Sturm denominator exclusion, rational sign "
            "cells, and recursive numerator-root equality children"
        ),
        "Sturm rational decision stratification",
    ),
    "RationalDecisionArrangement": (
        "finite_rational_decision_arrangement_interval_boxes",
        (
            "constructor-derived one-dimensional rational decision "
            "arrangement with denominator exclusion for every discriminator, "
            "rational sign-vector cells, and recursive numerator-root "
            "equality children"
        ),
        "rational decision arrangement",
    ),
    "SturmRationalDecisionArrangement": (
        "finite_sturm_rational_decision_arrangement_interval_boxes",
        (
            "constructor-derived one-dimensional rational decision "
            "arrangement with exact Sturm numerator-root isolation, "
            "closed-interval Sturm denominator exclusion for every "
            "discriminator, rational sign-vector cells, and recursive "
            "numerator-root equality children"
        ),
        "Sturm rational decision arrangement",
    ),
    "TaylorModelDecisionStratification": (
        "finite_taylor_model_decision_interval_inputs_with_weierstrass_certificate",
        (
            "constructor-derived finite Taylor-model decision stratification "
            "with explicit value and derivative remainder bounds, supported "
            "monotone or Weierstrass equality-root witnesses, and recursive "
            "equality children"
        ),
        "finite Taylor-model decision stratification",
    ),
    "TaylorModelDecisionArrangement": (
        "finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate",
        (
            "constructor-derived finite Taylor-model decision arrangement "
            "with explicit value and derivative remainder bounds, supported "
            "monotone or Weierstrass equality-root witnesses for each "
            "discriminator, sign-vector cells, and recursive equality children"
        ),
        "finite Taylor-model decision arrangement",
    ),
    "PolynomialRootChild": (
        "finite_polynomial_root_child_interval_boxes",
        (
            "constructor-derived zero-dimensional polynomial root child "
            "from an isolated equality root stratum"
        ),
        "polynomial root child",
    ),
    "AxisAlignedAffineBoxArrangement": (
        "finite_axis_aligned_affine_box_arrangement_interval_boxes",
        (
            "constructor-derived axis-aligned affine box arrangement with "
            "coordinate equality slabs, Cartesian sign boxes, and recursive "
            "equality children"
        ),
        "axis-aligned affine box arrangement",
    ),
    "AxisAlignedAffineBoxChild": (
        "finite_axis_aligned_affine_box_child_interval_boxes",
        (
            "constructor-derived lower-dimensional axis-aligned affine box "
            "child from coordinate equality slabs"
        ),
        "axis-aligned affine box child",
    ),
    "AffineHalfspaceDecision": (
        "finite_affine_halfspace_decision_interval_boxes",
        (
            "constructor-derived affine halfspace decision stratification "
            "with separated halfspace cells, an explicit central equality "
            "slab, and recursive equality children"
        ),
        "affine halfspace decision stratification",
    ),
    "AffineHalfspaceDecisionChild": (
        "finite_affine_halfspace_decision_child_interval_boxes",
        (
            "constructor-derived lower-dimensional affine halfspace decision "
            "child from the exact central hyperplane"
        ),
        "affine halfspace decision child",
    ),
    "AffineHalfspaceArrangement": (
        "finite_2d_affine_halfspace_arrangement_interval_boxes",
        (
            "constructor-derived finite 2D oblique affine halfspace "
            "arrangement with polygon value bounds, area-cover evidence, "
            "and recursive equality children"
        ),
        "2D oblique affine halfspace arrangement",
    ),
    "AffineHalfspace3DArrangement": (
        "finite_3d_affine_halfspace_arrangement_interval_boxes",
        (
            "constructor-derived finite 3D oblique affine halfspace "
            "arrangement with polyhedron value bounds, volume-cover "
            "evidence, and recursive equality children"
        ),
        "3D oblique affine halfspace arrangement",
    ),
    "AffineHalfspacePlaneChild": (
        "finite_2d_affine_halfspace_plane_child_interval_boxes",
        (
            "constructor-derived two-dimensional affine halfspace plane child "
            "from a single or coincident 3D oblique affine equality boundary"
        ),
        "2D oblique affine plane child",
    ),
    "AffineHalfspaceSpatialLineChild": (
        "finite_1d_affine_halfspace_spatial_line_child_interval_boxes",
        (
            "constructor-derived one-dimensional affine halfspace spatial "
            "line child from independent 3D oblique affine equality "
            "boundaries"
        ),
        "1D oblique affine spatial line child",
    ),
    "AffineHalfspaceSpatialPointChild": (
        "finite_0d_affine_halfspace_spatial_point_child_interval_boxes",
        (
            "constructor-derived zero-dimensional affine halfspace spatial "
            "point child from independent 3D oblique affine equality "
            "boundaries"
        ),
        "0D oblique affine spatial point child",
    ),
    "AffineHalfspaceLineChild": (
        "finite_1d_affine_halfspace_line_child_interval_boxes",
        (
            "constructor-derived one-dimensional affine halfspace line child "
            "from a single or coincident 2D oblique affine equality boundary"
        ),
        "1D oblique affine line child",
    ),
    "AffineHalfspacePointChild": (
        "finite_0d_affine_halfspace_point_child_interval_boxes",
        (
            "constructor-derived zero-dimensional affine halfspace point "
            "child from independent 2D oblique affine equality boundaries"
        ),
        "0D oblique affine point child",
    ),
    "PolynomialRootArrangement": (
        "finite_computed_polynomial_root_arrangement_interval_boxes",
        (
            "constructor-derived computed polynomial-root arrangement with "
            "grouped coincident/multiple equality strata and recursive "
            "equality children"
        ),
        "computed polynomial-root arrangement",
    ),
}

SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS = {
    "AffineDecisionStratification": "finite_affine_decision_interval_inputs",
    "AffineDecisionArrangement": "finite_affine_decision_arrangement_interval_inputs",
    "PolynomialDecisionStratification": "finite_polynomial_decision_interval_inputs",
    "SturmPolynomialDecisionStratification": (
        "finite_sturm_polynomial_decision_interval_inputs"
    ),
    "RationalDecisionStratification": (
        "finite_rational_decision_interval_inputs_with_denominator_exclusion"
    ),
    "SturmRationalDecisionStratification": (
        "finite_sturm_rational_decision_interval_inputs_with_denominator_exclusion"
    ),
    "RationalDecisionArrangement": (
        "finite_rational_decision_arrangement_interval_inputs_with_denominator_exclusion"
    ),
    "SturmRationalDecisionArrangement": (
        "finite_sturm_rational_decision_arrangement_interval_inputs_with_denominator_exclusion"
    ),
    "PolynomialDecisionArrangement": (
        "finite_polynomial_decision_arrangement_interval_inputs"
    ),
    "SturmPolynomialDecisionArrangement": (
        "finite_sturm_polynomial_decision_arrangement_interval_inputs"
    ),
    "QuadraticDoubleRootArrangement": (
        "finite_quadratic_double_root_arrangement_interval_inputs"
    ),
    "PolynomialRootArrangement": (
        "finite_computed_polynomial_root_arrangement_interval_inputs"
    ),
    "TaylorModelDecisionStratification": (
        "finite_taylor_model_decision_interval_inputs_with_weierstrass_certificate"
    ),
    "TaylorModelDecisionArrangement": (
        "finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate"
    ),
    "AxisAlignedAffineBoxArrangement": (
        "finite_axis_aligned_affine_box_arrangement_interval_inputs"
    ),
    "AffineHalfspaceDecision": "finite_affine_halfspace_decision_interval_inputs",
    "AffineHalfspaceArrangement": (
        "finite_2d_affine_halfspace_arrangement_interval_inputs"
    ),
    "AffineHalfspace3DArrangement": (
        "finite_3d_affine_halfspace_arrangement_interval_inputs"
    ),
}


SUPPORTED_EVENT_FUNCTION_STRATIFICATION_CONSTRUCTOR_TYPES = (
    PolynomialDecisionStratificationCertificate,
    PolynomialDecisionArrangementStratificationCertificate,
    RationalDecisionStratificationCertificate,
    RationalDecisionArrangementStratificationCertificate,
    TaylorModelDecisionStratificationCertificate,
    TaylorModelDecisionArrangementStratificationCertificate,
    AffineBoxDecisionArrangementStratificationCertificate,
    AffineHalfspaceDecisionStratificationCertificate,
    AffineHalfspaceArrangementStratificationCertificate,
    AffineHalfspaceArrangement3DStratificationCertificate,
)


def _theorem_obligation_ledger_certified(
    obligations: tuple[object, ...],
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
            if isinstance(obligation, TheoremPipelineObligation)
            and obligation.required is True
        )
    )


def _theorem_obligation_ledger_missing(
    obligations: tuple[object, ...],
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
        if obligation.required is True and obligation.certified is not True:
            missing.append(obligation.obligation)
    return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class FiniteTargetAnalyticLemmaCertificate:
    """A named analytic lemma used by the finite-target theorem.

    The certificate distinguishes a declared theorem statement from a proof
    row that has been externally audited, machine checked, or internally
    proven by a short structural argument recorded in the theorem scaffold.
    """

    lemma_id: str
    statement: str
    proof_sketch: str
    prerequisites: tuple[str, ...] = ()
    proof_mode: str = "declared_prose"
    externally_audited: bool = False
    machine_checkable: bool = False
    machine_checked: bool = False
    internally_proven: bool = False

    @property
    def declared(self) -> bool:
        return bool(self.lemma_id and self.statement and self.proof_sketch)

    @property
    def audited(self) -> bool:
        """Whether independent evidence closes this analytic lemma.

        ``internally_proven`` records that the repository contains a proposed
        proof sketch.  ``machine_checkable`` records that supplied finite data
        can be checked.  Neither establishes the universal analytic statement.
        Only an external audit or an actually completed machine proof closes
        the theorem-facing obligation.
        """

        return bool(self.externally_audited or self.machine_checked)

    @property
    def internally_supported(self) -> bool:
        """Whether a nontrivial first-party proof argument is recorded."""

        return bool(
            self.declared
            and self.internally_proven
            and self.proof_mode != "declared_prose"
        )

    @property
    def certified(self) -> bool:
        return self.declared

    @property
    def proof_certified(self) -> bool:
        return self.audited


@dataclass(frozen=True)
class AnalyticLemmaAuditRecord:
    """Audit-facing inventory row for a declared analytic theorem dependency."""

    lemma_id: str
    statement: str
    hypotheses: tuple[str, ...]
    source_theorem: str
    proof_mode: str
    status: str
    normalization_translation: str
    checker_inputs: tuple[str, ...]
    failure_modes: tuple[str, ...]
    audit_tier: str = "untiered_supporting_lemma"

    @property
    def declared(self) -> bool:
        return bool(self.lemma_id and self.statement)

    @property
    def audited(self) -> bool:
        return self.status in {
            "externally_audited",
            "machine_checked",
        }

    @property
    def internally_supported(self) -> bool:
        return self.status == "internally_proven"


@dataclass(frozen=True)
class AnalyticLemmaRegistry:
    """Proof audit registry for theorem-level analytic lemmas.

    The registry is not a proof certificate.  It records which theorem
    dependencies are only declared in prose so top-level proof audits can keep
    those lemma ids visible until an external audit, machine checker, or
    recorded internal proof consumes them.
    """

    theorem_id: str
    records: tuple[AnalyticLemmaAuditRecord, ...]
    critical_lemma_ids: tuple[str, ...] = FINITE_TARGET_CRITICAL_ANALYTIC_LEMMA_IDS

    @property
    def lemma_ids(self) -> tuple[str, ...]:
        return tuple(record.lemma_id for record in self.records if record.lemma_id)

    @property
    def missing_critical_lemma_ids(self) -> tuple[str, ...]:
        present = set(self.lemma_ids)
        return tuple(
            lemma_id
            for lemma_id in self.critical_lemma_ids
            if lemma_id not in present
        )

    @property
    def unaudited_lemma_ids(self) -> tuple[str, ...]:
        return tuple(
            record.lemma_id
            for record in self.records
            if record.declared and not record.audited
        )

    @property
    def critical_unaudited_lemma_ids(self) -> tuple[str, ...]:
        unaudited = set(self.unaudited_lemma_ids)
        return tuple(
            lemma_id
            for lemma_id in self.critical_lemma_ids
            if lemma_id in unaudited
        )

    def unaudited_lemma_ids_for_tier(self, audit_tier: str) -> tuple[str, ...]:
        return tuple(
            record.lemma_id
            for record in self.records
            if (
                record.declared
                and not record.audited
                and record.audit_tier == audit_tier
            )
        )

    @property
    def tier_a_unaudited_lemma_ids(self) -> tuple[str, ...]:
        return self.unaudited_lemma_ids_for_tier(
            "tier_a_classical_or_structural"
        )

    @property
    def tier_b_unaudited_lemma_ids(self) -> tuple[str, ...]:
        return self.unaudited_lemma_ids_for_tier(
            "tier_b_total_collision_entry_frontier"
        )

    @property
    def untiered_unaudited_lemma_ids(self) -> tuple[str, ...]:
        return tuple(
            record.lemma_id
            for record in self.records
            if (
                record.declared
                and not record.audited
                and record.audit_tier == "untiered_supporting_lemma"
            )
        )

    @property
    def audit_complete(self) -> bool:
        return bool(
            self.records
            and not self.missing_critical_lemma_ids
            and not self.unaudited_lemma_ids
        )

    @property
    def blocking_obligations(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.tier_b_unaudited_lemma_ids,
                    *self.tier_a_unaudited_lemma_ids,
                    *self.untiered_unaudited_lemma_ids,
                )
            )
        )


@dataclass(frozen=True)
class FiniteTargetCompletenessTheoremCertificate:
    """Point-input finite-target atlas-or-stop completeness statement."""

    dimension: int
    input_model: str
    total_collision_policy_id: str
    painleve_no_noncollision_singularities: FiniteTargetAnalyticLemmaCertificate
    all_pair_binary_regularization: FiniteTargetAnalyticLemmaCertificate
    binary_collision_isolation: FiniteTargetAnalyticLemmaCertificate
    binary_accumulation_forces_total_collision: FiniteTargetAnalyticLemmaCertificate
    compact_collision_free_taylor_cover: FiniteTargetAnalyticLemmaCertificate
    total_collision_zero_angular_momentum_condition: (
        FiniteTargetAnalyticLemmaCertificate
    )
    total_collision_central_configuration_asymptotic: (
        FiniteTargetAnalyticLemmaCertificate
    )
    cubic_time_total_collision_scaling: FiniteTargetAnalyticLemmaCertificate
    finite_fuchsian_log_stop_chart_for_admissible_entry_data: (
        FiniteTargetAnalyticLemmaCertificate
    )
    binary_degenerate_total_collision_exclusion: (
        FiniteTargetAnalyticLemmaCertificate
    )
    reduced_hyperbolic_total_collision_entry: (
        FiniteTargetAnalyticLemmaCertificate
    )
    poincare_dulac_fuchsian_log_selector_completeness: (
        FiniteTargetAnalyticLemmaCertificate
    )
    arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data: (
        FiniteTargetAnalyticLemmaCertificate
    )
    arbitrary_total_collision_germ_entry_to_stop_chart: (
        FiniteTargetAnalyticLemmaCertificate
    )
    homothetic_total_collision_stop_chart_existence: (
        FiniteTargetAnalyticLemmaCertificate
    )
    total_collision_stop_chart_existence: FiniteTargetAnalyticLemmaCertificate
    finite_chart_chain_concatenation: FiniteTargetAnalyticLemmaCertificate
    target_or_stop_dichotomy: FiniteTargetAnalyticLemmaCertificate
    chart_families: tuple[str, ...]
    allowed_outcomes: tuple[str, ...]
    statement: str
    proof_sketch: str
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "pointwise_finite_target_atlas_or_stop_completeness"

    @property
    def analytic_lemma_statements_declared(self) -> bool:
        return all(lemma.declared for lemma in self.analytic_lemmas)

    @property
    def analytic_lemma_proofs_audited(self) -> bool:
        return all(lemma.proof_certified for lemma in self.analytic_lemmas)

    @property
    def dimension_supported(self) -> bool:
        return self.dimension in FINITE_TARGET_SUPPORTED_DIMENSIONS

    @property
    def input_model_supported(self) -> bool:
        return self.input_model in FINITE_TARGET_POINT_INPUT_MODELS

    @property
    def maximal_classical_policy_supported(self) -> bool:
        return self.total_collision_policy_id in FINITE_TARGET_MAXIMAL_CLASSICAL_POLICIES

    @property
    def theorem_scope_parameters_certified(self) -> bool:
        return bool(
            self.theorem_id == "pointwise_finite_target_atlas_or_stop_completeness"
            and self.dimension_supported
            and self.input_model_supported
            and self.maximal_classical_policy_supported
        )

    @property
    def statement_certified(self) -> bool:
        return bool(
            self.theorem_scope_parameters_certified
            and self.statement
            and self.proof_sketch
            and self.analytic_lemma_statements_declared
            and self.chart_families == FINITE_TARGET_COMPLETENESS_CHART_FAMILIES
            and self.allowed_outcomes == FINITE_TARGET_COMPLETENESS_OUTCOMES
        )

    @property
    def statement_declared(self) -> bool:
        return self.statement_certified

    @property
    def scaffold_certified(self) -> bool:
        return self.certified

    @property
    def certified(self) -> bool:
        return bool(
            self.statement_certified
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and self.analytic_lemma_proofs_audited)

    @property
    def analytic_lemma_registry(self) -> AnalyticLemmaRegistry:
        return build_analytic_lemma_registry_for_finite_target_theorem(self)

    @property
    def unaudited_analytic_lemma_ids(self) -> tuple[str, ...]:
        return self.analytic_lemma_registry.unaudited_lemma_ids

    @property
    def critical_unaudited_analytic_lemma_ids(self) -> tuple[str, ...]:
        return self.analytic_lemma_registry.critical_unaudited_lemma_ids

    @property
    def tier_a_unaudited_analytic_lemma_ids(self) -> tuple[str, ...]:
        return self.analytic_lemma_registry.tier_a_unaudited_lemma_ids

    @property
    def tier_b_unaudited_analytic_lemma_ids(self) -> tuple[str, ...]:
        return self.analytic_lemma_registry.tier_b_unaudited_lemma_ids

    @property
    def core_analytic_lemma_audit_blockers(self) -> tuple[str, ...]:
        return self.tier_b_unaudited_analytic_lemma_ids

    @property
    def analytic_lemma_audit_blockers(self) -> tuple[str, ...]:
        return self.analytic_lemma_registry.blocking_obligations

    @property
    def arbitrary_total_collision_germ_finite_fuchsian_log_entry_data(
        self,
    ) -> FiniteTargetAnalyticLemmaCertificate:
        """Compatibility alias for the generalized Fuchsian entry theorem.

        The theorem was renamed because noninteger Fuchsian/Puiseux rows are
        part of the intended entry language, not an optional extension.
        """

        return self.arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data

    @property
    def analytic_lemmas(self) -> tuple[FiniteTargetAnalyticLemmaCertificate, ...]:
        return (
            self.painleve_no_noncollision_singularities,
            self.all_pair_binary_regularization,
            self.binary_collision_isolation,
            self.binary_accumulation_forces_total_collision,
            self.compact_collision_free_taylor_cover,
            self.total_collision_zero_angular_momentum_condition,
            self.total_collision_central_configuration_asymptotic,
            self.cubic_time_total_collision_scaling,
            self.finite_fuchsian_log_stop_chart_for_admissible_entry_data,
            self.binary_degenerate_total_collision_exclusion,
            self.reduced_hyperbolic_total_collision_entry,
            self.poincare_dulac_fuchsian_log_selector_completeness,
            self.arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data,
            self.arbitrary_total_collision_germ_entry_to_stop_chart,
            self.homothetic_total_collision_stop_chart_existence,
            self.total_collision_stop_chart_existence,
            self.finite_chart_chain_concatenation,
            self.target_or_stop_dichotomy,
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _theorem_obligation_ledger_missing(
                self.obligations,
                ledger_name="finite_target_completeness",
            )
        )
        if self.theorem_id != "pointwise_finite_target_atlas_or_stop_completeness":
            missing.append("pointwise_finite_target_theorem_id")
        if not self.dimension_supported:
            missing.append("finite_target_dimension_supported")
        if not self.input_model_supported:
            missing.append("point_input_model_or_computable_name")
        if not self.maximal_classical_policy_supported:
            missing.append("maximal_classical_total_collision_policy")
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class FiniteTargetCertificateSearchCompletenessCertificate:
    """Implementation-side search completeness, kept separate from the theorem."""

    theorem_certificate: FiniteTargetCompletenessTheoremCertificate
    observed_prefix_failures: tuple[str, ...]
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "certificate_search_completeness_for_point_inputs"

    @property
    def component_types_certified(self) -> bool:
        return isinstance(
            self.theorem_certificate,
            FiniteTargetCompletenessTheoremCertificate,
        )

    @property
    def source_matches_theorem(self) -> bool:
        return bool(
            self.component_types_certified
            and self.theorem_certificate.theorem_id
            == "pointwise_finite_target_atlas_or_stop_completeness"
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.component_types_certified
            and self.source_matches_theorem
            and self.theorem_certificate.certified
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified and self.theorem_certificate.proof_certified is True
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _theorem_obligation_ledger_missing(
                self.obligations,
                ledger_name="finite_target_certificate_search",
            )
        )
        if not self.component_types_certified:
            missing.append("pointwise_finite_target_theorem_type")
        if not self.source_matches_theorem:
            missing.append("pointwise_finite_target_theorem_source")
        if not (
            self.component_types_certified
            and self.theorem_certificate.proof_certified is True
        ):
            missing.append("pointwise_finite_target_theorem_proof_certified")
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class UniformMarginSetValuedConstructorCompletenessCertificate:
    """Set-valued finite-target constructor theorem under positive margins.

    This is not the arbitrary interval-input theorem.  It proves the open
    subset where every branch and event-order discriminator is separated from
    its equality boundary by explicit uniform margins, so recursive refinement
    terminates before any equality stratum must be analyzed.
    """

    theorem_certificate: FiniteTargetCompletenessTheoremCertificate
    search_completeness_certificate: FiniteTargetCertificateSearchCompletenessCertificate
    branch_refinement_certificate: "UniformMarginBranchRefinementTerminationCertificate"
    event_order_refinement_certificate: "UniformMarginBranchRefinementTerminationCertificate"
    statement: str
    proof_sketch: str
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "uniform_margin_set_valued_constructor_branch_event_completeness"

    @property
    def component_types_certified(self) -> bool:
        return bool(
            isinstance(
                self.theorem_certificate,
                FiniteTargetCompletenessTheoremCertificate,
            )
            and isinstance(
                self.search_completeness_certificate,
                FiniteTargetCertificateSearchCompletenessCertificate,
            )
            and isinstance(
                self.branch_refinement_certificate,
                UniformMarginBranchRefinementTerminationCertificate,
            )
            and isinstance(
                self.event_order_refinement_certificate,
                UniformMarginBranchRefinementTerminationCertificate,
            )
        )

    @property
    def source_matches_theorem(self) -> bool:
        return bool(
            self.component_types_certified
            and self.search_completeness_certificate.theorem_certificate
            is self.theorem_certificate
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.component_types_certified
            and self.source_matches_theorem
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.theorem_certificate.proof_certified is True
            and self.search_completeness_certificate.proof_certified is True
            and self.branch_refinement_certificate.certified is True
            and self.event_order_refinement_certificate.certified is True
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _theorem_obligation_ledger_missing(
                self.obligations,
                ledger_name="uniform_margin_set_valued_constructor",
            )
        )
        if not self.component_types_certified:
            missing.append("uniform_margin_component_types")
        if not self.source_matches_theorem:
            missing.append("uniform_margin_search_source_matches_theorem")
        if not (
            isinstance(
                self.theorem_certificate,
                FiniteTargetCompletenessTheoremCertificate,
            )
            and self.theorem_certificate.proof_certified is True
        ):
            missing.append("pointwise_finite_target_theorem_proof_certified")
        if not (
            isinstance(
                self.search_completeness_certificate,
                FiniteTargetCertificateSearchCompletenessCertificate,
            )
            and self.search_completeness_certificate.proof_certified is True
        ):
            missing.append("certificate_search_completeness_proof_certified")
        if not (
            isinstance(
                self.branch_refinement_certificate,
                UniformMarginBranchRefinementTerminationCertificate,
            )
            and self.branch_refinement_certificate.certified is True
        ):
            missing.append("uniform_margin_branch_refinement_certified")
        if not (
            isinstance(
                self.event_order_refinement_certificate,
                UniformMarginBranchRefinementTerminationCertificate,
            )
            and self.event_order_refinement_certificate.certified is True
        ):
            missing.append("uniform_margin_event_order_refinement_certified")
        return tuple(dict.fromkeys(missing))

    @property
    def equality_strata_claimed(self) -> bool:
        return False


@dataclass(frozen=True)
class SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate:
    """Set-valued constructor theorem for a supplied recursive equality tree.

    This certificate covers the represented finite recursive stratification:
    positive-margin leaves are terminal, zero-margin/equality leaves are
    consumed by constructor-derived lower-dimensional or lower-rank child
    certificates, and finite union glues the leaf responses.  It deliberately
    does not derive that recursive stratified tree from arbitrary interval
    inputs.
    """

    theorem_certificate: FiniteTargetCompletenessTheoremCertificate
    search_completeness_certificate: FiniteTargetCertificateSearchCompletenessCertificate
    branch_consumption_certificate: RecursiveStratifiedBranchEventConsumptionCertificate
    event_order_consumption_certificate: RecursiveStratifiedBranchEventConsumptionCertificate
    statement: str
    proof_sketch: str
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "supplied_recursive_stratified_set_valued_constructor_branch_event_completeness"

    @property
    def component_types_certified(self) -> bool:
        return bool(
            isinstance(
                self.theorem_certificate,
                FiniteTargetCompletenessTheoremCertificate,
            )
            and isinstance(
                self.search_completeness_certificate,
                FiniteTargetCertificateSearchCompletenessCertificate,
            )
            and isinstance(
                self.branch_consumption_certificate,
                RecursiveStratifiedBranchEventConsumptionCertificate,
            )
            and isinstance(
                self.event_order_consumption_certificate,
                RecursiveStratifiedBranchEventConsumptionCertificate,
            )
        )

    @property
    def source_matches_theorem(self) -> bool:
        return bool(
            self.component_types_certified
            and self.search_completeness_certificate.theorem_certificate
            is self.theorem_certificate
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.component_types_certified
            and self.source_matches_theorem
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.theorem_certificate.proof_certified is True
            and self.search_completeness_certificate.proof_certified is True
            and self.branch_consumption_certificate.proof_certified is True
            and self.event_order_consumption_certificate.proof_certified is True
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _theorem_obligation_ledger_missing(
                self.obligations,
                ledger_name="supplied_recursive_set_valued_constructor",
            )
        )
        if not self.component_types_certified:
            missing.append("supplied_recursive_component_types")
        if not self.source_matches_theorem:
            missing.append("supplied_recursive_search_source_matches_theorem")
        if not (
            isinstance(
                self.theorem_certificate,
                FiniteTargetCompletenessTheoremCertificate,
            )
            and self.theorem_certificate.proof_certified is True
        ):
            missing.append("pointwise_finite_target_theorem_proof_certified")
        if not (
            isinstance(
                self.search_completeness_certificate,
                FiniteTargetCertificateSearchCompletenessCertificate,
            )
            and self.search_completeness_certificate.proof_certified is True
        ):
            missing.append("certificate_search_completeness_proof_certified")
        if not (
            isinstance(
                self.branch_consumption_certificate,
                RecursiveStratifiedBranchEventConsumptionCertificate,
            )
            and self.branch_consumption_certificate.proof_certified is True
        ):
            missing.append("branch_partition_consumption_proof_certified")
        if not (
            isinstance(
                self.event_order_consumption_certificate,
                RecursiveStratifiedBranchEventConsumptionCertificate,
            )
            and self.event_order_consumption_certificate.proof_certified is True
        ):
            missing.append("event_order_partition_consumption_proof_certified")
        return tuple(dict.fromkeys(missing))

    @property
    def equality_strata_claimed(self) -> bool:
        return True

    @property
    def arbitrary_partition_generation_claimed(self) -> bool:
        return False


@dataclass(frozen=True)
class AffineHalfspaceArrangementSetValuedConstructorCompletenessCertificate:
    """Set-valued theorem for a constructor-derived affine arrangement.

    This certificate is narrower than the arbitrary interval-input theorem but
    stronger than a purely supplied recursive tree: the top-level stratification
    is constructed from explicit oblique affine discriminants, convex-cell
    clipping, value-bound checks, and an area/volume-cover guard before
    recursive equality-stratum consumption is applied.
    """

    theorem_certificate: FiniteTargetCompletenessTheoremCertificate
    search_completeness_certificate: FiniteTargetCertificateSearchCompletenessCertificate
    arrangement_certificate: (
        AffineHalfspaceArrangementStratificationCertificate
        | AffineHalfspaceArrangement3DStratificationCertificate
    )
    branch_consumption_certificate: RecursiveStratifiedBranchEventConsumptionCertificate
    event_order_consumption_certificate: RecursiveStratifiedBranchEventConsumptionCertificate
    statement: str
    proof_sketch: str
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "affine_halfspace_arrangement_set_valued_constructor_branch_event_completeness"

    @property
    def component_types_certified(self) -> bool:
        return bool(
            isinstance(
                self.theorem_certificate,
                FiniteTargetCompletenessTheoremCertificate,
            )
            and isinstance(
                self.search_completeness_certificate,
                FiniteTargetCertificateSearchCompletenessCertificate,
            )
            and isinstance(
                self.arrangement_certificate,
                (
                    AffineHalfspaceArrangementStratificationCertificate,
                    AffineHalfspaceArrangement3DStratificationCertificate,
                ),
            )
            and isinstance(
                self.branch_consumption_certificate,
                RecursiveStratifiedBranchEventConsumptionCertificate,
            )
            and isinstance(
                self.event_order_consumption_certificate,
                RecursiveStratifiedBranchEventConsumptionCertificate,
            )
        )

    @property
    def source_matches_theorem(self) -> bool:
        return bool(
            self.component_types_certified
            and self.search_completeness_certificate.theorem_certificate
            is self.theorem_certificate
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.component_types_certified
            and self.source_matches_theorem
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.theorem_certificate.proof_certified is True
            and self.search_completeness_certificate.proof_certified is True
            and self.arrangement_certificate.proof_certified is True
            and self.branch_consumption_certificate.proof_certified is True
            and self.event_order_consumption_certificate.proof_certified is True
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _theorem_obligation_ledger_missing(
                self.obligations,
                ledger_name="affine_halfspace_arrangement_set_valued_constructor",
            )
        )
        if not self.component_types_certified:
            missing.append("affine_halfspace_arrangement_component_types")
        if not self.source_matches_theorem:
            missing.append("affine_halfspace_arrangement_search_source_matches_theorem")
        if not (
            isinstance(
                self.theorem_certificate,
                FiniteTargetCompletenessTheoremCertificate,
            )
            and self.theorem_certificate.proof_certified is True
        ):
            missing.append("pointwise_finite_target_theorem_proof_certified")
        if not (
            isinstance(
                self.search_completeness_certificate,
                FiniteTargetCertificateSearchCompletenessCertificate,
            )
            and self.search_completeness_certificate.proof_certified is True
        ):
            missing.append("certificate_search_completeness_proof_certified")
        if not (
            isinstance(
                self.arrangement_certificate,
                (
                    AffineHalfspaceArrangementStratificationCertificate,
                    AffineHalfspaceArrangement3DStratificationCertificate,
                ),
            )
            and self.arrangement_certificate.proof_certified is True
        ):
            missing.append("affine_halfspace_arrangement_proof_certified")
        if not (
            isinstance(
                self.branch_consumption_certificate,
                RecursiveStratifiedBranchEventConsumptionCertificate,
            )
            and self.branch_consumption_certificate.proof_certified is True
        ):
            missing.append("branch_partition_consumption_proof_certified")
        if not (
            isinstance(
                self.event_order_consumption_certificate,
                RecursiveStratifiedBranchEventConsumptionCertificate,
            )
            and self.event_order_consumption_certificate.proof_certified is True
        ):
            missing.append("event_order_partition_consumption_proof_certified")
        return tuple(dict.fromkeys(missing))

    @property
    def equality_strata_claimed(self) -> bool:
        return True

    @property
    def arbitrary_partition_generation_claimed(self) -> bool:
        return False


def _validated_set_valued_constructor_scope(
    certificate: object,
) -> tuple[str, str]:
    """Scope label derived from the actual scoped constructor certificate."""

    if isinstance(certificate, UniformMarginSetValuedConstructorCompletenessCertificate):
        return (
            "positive_margin_interval_boxes",
            "explicit branch and event-order margins exclude equality strata",
        )
    if isinstance(
        certificate,
        SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate,
    ):
        constructor_scope = _shared_recursive_constructor_scope_detail(
            certificate.branch_consumption_certificate,
            certificate.event_order_consumption_certificate,
        )
        if constructor_scope is None:
            return (
                "supplied_recursive_stratified_interval_boxes",
                "a finite recursive equality tree is supplied and consumed by "
                "strict descent",
            )
        return constructor_scope
    if isinstance(
        certificate,
        AffineHalfspaceArrangementSetValuedConstructorCompletenessCertificate,
    ):
        arrangement = certificate.arrangement_certificate
        if isinstance(arrangement, AffineHalfspaceArrangement3DStratificationCertificate):
            return (
                "finite_3d_affine_halfspace_arrangement_interval_boxes",
                "a constructor-derived convex-polyhedron affine halfspace "
                "arrangement is volume-cover certified and recursively consumed",
            )
        return (
            "finite_2d_affine_halfspace_arrangement_interval_boxes",
            "a constructor-derived convex-polygon affine halfspace arrangement "
            "is area-cover certified and recursively consumed",
        )
    return (
        "",
        "unsupported scoped set-valued constructor certificate",
    )


@dataclass(frozen=True)
class ValidatedSetValuedConstructorCompletenessTheoremCertificate:
    """Named implementation theorem for interval-box constructor completeness.

    This is the separate validated-numerics theorem surface requested by the
    pointwise closed-form split.  It wraps a scoped constructor theorem such as
    the positive-margin theorem or a supplied recursive stratified theorem.  It
    does not turn those scoped certificates into an arbitrary interval-input
    partition-generation theorem.
    """

    theorem_certificate: FiniteTargetCompletenessTheoremCertificate
    search_completeness_certificate: FiniteTargetCertificateSearchCompletenessCertificate
    set_valued_constructor_certificate: (
        UniformMarginSetValuedConstructorCompletenessCertificate
        | SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate
        | AffineHalfspaceArrangementSetValuedConstructorCompletenessCertificate
    )
    input_scope_id: str
    statement: str
    proof_sketch: str
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "validated_set_valued_constructor_completeness"

    @property
    def component_types_certified(self) -> bool:
        return bool(
            isinstance(
                self.theorem_certificate,
                FiniteTargetCompletenessTheoremCertificate,
            )
            and isinstance(
                self.search_completeness_certificate,
                FiniteTargetCertificateSearchCompletenessCertificate,
            )
            and isinstance(
                self.set_valued_constructor_certificate,
                (
                    UniformMarginSetValuedConstructorCompletenessCertificate,
                    SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate,
                    AffineHalfspaceArrangementSetValuedConstructorCompletenessCertificate,
                ),
            )
        )

    @property
    def source_matches_theorem(self) -> bool:
        return bool(
            self.component_types_certified
            and self.search_completeness_certificate.theorem_certificate
            is self.theorem_certificate
            and self.set_valued_constructor_certificate.theorem_certificate
            is self.theorem_certificate
            and (
                self.set_valued_constructor_certificate.search_completeness_certificate
                is self.search_completeness_certificate
            )
        )

    @property
    def expected_input_scope_id(self) -> str:
        if not self.component_types_certified:
            return ""
        return _validated_set_valued_constructor_scope(
            self.set_valued_constructor_certificate,
        )[0]

    @property
    def input_scope_matches_constructor(self) -> bool:
        return bool(
            self.component_types_certified
            and self.expected_input_scope_id
            and self.input_scope_id == self.expected_input_scope_id
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.input_scope_id
            and self.component_types_certified
            and self.source_matches_theorem
            and self.input_scope_matches_constructor
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.theorem_certificate.proof_certified is True
            and self.search_completeness_certificate.proof_certified is True
            and self.set_valued_constructor_certificate.proof_certified is True
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _theorem_obligation_ledger_missing(
                self.obligations,
                ledger_name="validated_set_valued_constructor",
            )
        )
        if not isinstance(
            self.theorem_certificate,
            FiniteTargetCompletenessTheoremCertificate,
        ):
            missing.append("pointwise_finite_target_theorem_type")
        if not (
            isinstance(
                self.theorem_certificate,
                FiniteTargetCompletenessTheoremCertificate,
            )
            and self.theorem_certificate.proof_certified is True
        ):
            missing.append("pointwise_finite_target_theorem_proof_certified")
        if not isinstance(
            self.search_completeness_certificate,
            FiniteTargetCertificateSearchCompletenessCertificate,
        ):
            missing.append("certificate_search_completeness_type")
        if not isinstance(
            self.set_valued_constructor_certificate,
            (
                UniformMarginSetValuedConstructorCompletenessCertificate,
                SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate,
                AffineHalfspaceArrangementSetValuedConstructorCompletenessCertificate,
            ),
        ):
            missing.append("scoped_set_valued_constructor_certificate_type")
        if not self.source_matches_theorem:
            missing.append("validated_set_valued_sources_match")
        if not self.input_scope_matches_constructor:
            missing.append("validated_interval_input_scope_matches_constructor")
        if not (
            isinstance(
                self.search_completeness_certificate,
                FiniteTargetCertificateSearchCompletenessCertificate,
            )
            and self.search_completeness_certificate.proof_certified is True
        ):
            missing.append("certificate_search_completeness_proof_certified")
        if not (
            isinstance(
                self.set_valued_constructor_certificate,
                (
                    UniformMarginSetValuedConstructorCompletenessCertificate,
                    SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate,
                    AffineHalfspaceArrangementSetValuedConstructorCompletenessCertificate,
                ),
            )
            and self.set_valued_constructor_certificate.proof_certified is True
        ):
            missing.append("scoped_set_valued_constructor_certificate_proof_certified")
        return tuple(dict.fromkeys(missing))

    @property
    def arbitrary_partition_generation_claimed(self) -> bool:
        return False


@dataclass(frozen=True)
class SupportedEventFunctionGrammarInput:
    """Raw finite event-function data for a supported stratification grammar.

    This is the implementation-side input that the interval-box milestone
    needs: it contains discriminator data, not an already-built constructor
    certificate.  The generator below dispatches this data through the
    existing proof-producing constructors and then through the scoped
    arbitrary interval-input bridge.
    """

    source_type: str
    decision_id: str = ""
    arrangement_id: str = ""
    coefficients: tuple[float, ...] = ()
    root_brackets: tuple[tuple[float, float], ...] = ()
    domain: tuple[float, float] | None = None
    domain_box: tuple[tuple[float, float], ...] = ()
    polynomial_decision_functions: tuple[PolynomialDecisionFunctionSpec, ...] = ()
    rational_decision_function: RationalDecisionFunctionSpec | None = None
    rational_decision_functions: tuple[RationalDecisionFunctionSpec, ...] = ()
    taylor_model_decision_function: TaylorModelDecisionFunctionSpec | None = None
    taylor_model_decision_functions: tuple[
        TaylorModelDecisionFunctionSpec, ...
    ] = ()
    affine_box_decision_functions: tuple[AffineBoxDecisionFunctionSpec, ...] = ()
    slab_half_width: float | None = None
    equality_resolution_policy: str = "lower_dimensional_recursive_stratum"
    max_bisection_depth: int = 96

    @property
    def grammar_id(self) -> str:
        return SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS.get(
            self.source_type,
            "",
        )

    @property
    def supported(self) -> bool:
        return bool(self.grammar_id)


def _signature_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return tuple(_signature_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_signature_value(item) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return (
            type(value).__name__,
            tuple(
                (field_name, _signature_value(getattr(value, field_name)))
                for field_name in value.__dataclass_fields__
            ),
        )
    return repr(value)


def _polynomial_decision_function_signature(
    decision_function: PolynomialDecisionFunctionSpec,
    *,
    include_root_brackets: bool,
) -> tuple[object, ...]:
    signature: tuple[object, ...] = (
        decision_function.decision_id,
        _signature_value(decision_function.coefficients),
    )
    if include_root_brackets:
        signature = signature + (_signature_value(decision_function.root_brackets),)
    return signature


def _polynomial_decision_functions_signature(
    decision_functions: tuple[PolynomialDecisionFunctionSpec, ...],
    *,
    include_root_brackets: bool,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        _polynomial_decision_function_signature(
            decision_function,
            include_root_brackets=include_root_brackets,
        )
        for decision_function in decision_functions
    )


def _rational_decision_function_signature(
    decision_function: RationalDecisionFunctionSpec,
    *,
    include_root_brackets: bool,
) -> tuple[object, ...]:
    signature: tuple[object, ...] = (
        decision_function.decision_id,
        _signature_value(decision_function.numerator_coefficients),
        _signature_value(decision_function.denominator_coefficients),
    )
    if include_root_brackets:
        signature = signature + (_signature_value(decision_function.root_brackets),)
    return signature


def _rational_decision_functions_signature(
    decision_functions: tuple[RationalDecisionFunctionSpec, ...],
    *,
    include_root_brackets: bool,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        _rational_decision_function_signature(
            decision_function,
            include_root_brackets=include_root_brackets,
        )
        for decision_function in decision_functions
    )


def _supported_grammar_input_signature(
    grammar_input: SupportedEventFunctionGrammarInput,
) -> tuple[object, ...]:
    source_type = str(grammar_input.source_type)
    if source_type in {
        "AffineDecisionStratification",
        "PolynomialDecisionStratification",
        "SturmPolynomialDecisionStratification",
    }:
        values: tuple[object, ...] = (
            source_type,
            grammar_input.decision_id,
            _signature_value(grammar_input.coefficients),
            _signature_value(grammar_input.domain),
        )
        if source_type == "PolynomialDecisionStratification":
            values = values + (_signature_value(grammar_input.root_brackets),)
        return values
    if source_type in {
        "AffineDecisionArrangement",
        "PolynomialDecisionArrangement",
        "SturmPolynomialDecisionArrangement",
        "QuadraticDoubleRootArrangement",
        "PolynomialRootArrangement",
    }:
        include_root_brackets = source_type == "PolynomialDecisionArrangement"
        return (
            source_type,
            grammar_input.arrangement_id,
            _polynomial_decision_functions_signature(
                grammar_input.polynomial_decision_functions,
                include_root_brackets=include_root_brackets,
            ),
            _signature_value(grammar_input.domain),
        )
    if source_type in {
        "RationalDecisionStratification",
        "SturmRationalDecisionStratification",
    }:
        include_root_brackets = source_type == "RationalDecisionStratification"
        return (
            source_type,
            (
                _rational_decision_function_signature(
                    grammar_input.rational_decision_function,
                    include_root_brackets=include_root_brackets,
                )
                if grammar_input.rational_decision_function is not None
                else None
            ),
            _signature_value(grammar_input.domain),
        )
    if source_type in {
        "RationalDecisionArrangement",
        "SturmRationalDecisionArrangement",
    }:
        include_root_brackets = source_type == "RationalDecisionArrangement"
        return (
            source_type,
            grammar_input.arrangement_id,
            _rational_decision_functions_signature(
                grammar_input.rational_decision_functions,
                include_root_brackets=include_root_brackets,
            ),
            _signature_value(grammar_input.domain),
        )
    if source_type == "TaylorModelDecisionStratification":
        return (
            source_type,
            _signature_value(grammar_input.taylor_model_decision_function),
            _signature_value(grammar_input.domain),
        )
    if source_type == "TaylorModelDecisionArrangement":
        return (
            source_type,
            grammar_input.arrangement_id,
            _signature_value(grammar_input.taylor_model_decision_functions),
            _signature_value(grammar_input.domain),
        )
    if source_type == "AxisAlignedAffineBoxArrangement":
        return (
            source_type,
            grammar_input.arrangement_id,
            _signature_value(grammar_input.affine_box_decision_functions),
            _signature_value(grammar_input.domain_box),
        )
    if source_type == "AffineHalfspaceDecision":
        return (
            source_type,
            grammar_input.decision_id,
            _signature_value(grammar_input.coefficients),
            _signature_value(grammar_input.domain_box),
            _signature_value(grammar_input.slab_half_width),
        )
    if source_type in {
        "AffineHalfspaceArrangement",
        "AffineHalfspace3DArrangement",
    }:
        return (
            source_type,
            grammar_input.arrangement_id,
            _signature_value(grammar_input.affine_box_decision_functions),
            _signature_value(grammar_input.domain_box),
            _signature_value(grammar_input.slab_half_width),
        )
    return (source_type,)


def _supported_grammar_payload_signature(
    grammar_input: SupportedEventFunctionGrammarInput,
) -> tuple[object, ...]:
    return (
        _supported_grammar_input_signature(grammar_input),
        str(grammar_input.equality_resolution_policy),
        int(grammar_input.max_bisection_depth),
    )


def _constructor_input_signature(constructor_certificate: object) -> tuple[object, ...]:
    source_type = str(
        getattr(
            getattr(constructor_certificate, "source_tree", None),
            "source_type",
            "",
        )
    )
    if source_type in {
        "AffineDecisionStratification",
        "PolynomialDecisionStratification",
        "SturmPolynomialDecisionStratification",
    }:
        values: tuple[object, ...] = (
            source_type,
            getattr(constructor_certificate, "decision_id", ""),
            _signature_value(getattr(constructor_certificate, "coefficients", ())),
            _signature_value(getattr(constructor_certificate, "domain", None)),
        )
        if source_type == "PolynomialDecisionStratification":
            values = values + (
                _signature_value(
                    getattr(constructor_certificate, "root_brackets", ()),
                ),
            )
        return values
    if source_type in {
        "AffineDecisionArrangement",
        "PolynomialDecisionArrangement",
        "SturmPolynomialDecisionArrangement",
        "QuadraticDoubleRootArrangement",
        "PolynomialRootArrangement",
    }:
        include_root_brackets = source_type == "PolynomialDecisionArrangement"
        return (
            source_type,
            getattr(constructor_certificate, "arrangement_id", ""),
            _polynomial_decision_functions_signature(
                getattr(constructor_certificate, "decision_functions", ()),
                include_root_brackets=include_root_brackets,
            ),
            _signature_value(getattr(constructor_certificate, "domain", None)),
        )
    if source_type in {
        "RationalDecisionStratification",
        "SturmRationalDecisionStratification",
    }:
        include_root_brackets = source_type == "RationalDecisionStratification"
        decision_function = getattr(constructor_certificate, "decision_function", None)
        return (
            source_type,
            (
                _rational_decision_function_signature(
                    decision_function,
                    include_root_brackets=include_root_brackets,
                )
                if decision_function is not None
                else None
            ),
            _signature_value(getattr(constructor_certificate, "domain", None)),
        )
    if source_type in {
        "RationalDecisionArrangement",
        "SturmRationalDecisionArrangement",
    }:
        include_root_brackets = source_type == "RationalDecisionArrangement"
        return (
            source_type,
            getattr(constructor_certificate, "arrangement_id", ""),
            _rational_decision_functions_signature(
                getattr(constructor_certificate, "decision_functions", ()),
                include_root_brackets=include_root_brackets,
            ),
            _signature_value(getattr(constructor_certificate, "domain", None)),
        )
    if source_type == "TaylorModelDecisionStratification":
        return (
            source_type,
            _signature_value(
                getattr(constructor_certificate, "decision_function", None),
            ),
            _signature_value(getattr(constructor_certificate, "domain", None)),
        )
    if source_type == "TaylorModelDecisionArrangement":
        return (
            source_type,
            getattr(constructor_certificate, "arrangement_id", ""),
            _signature_value(
                getattr(constructor_certificate, "decision_functions", ()),
            ),
            _signature_value(getattr(constructor_certificate, "domain", None)),
        )
    if source_type == "AxisAlignedAffineBoxArrangement":
        return (
            source_type,
            getattr(constructor_certificate, "arrangement_id", ""),
            _signature_value(
                getattr(constructor_certificate, "decision_functions", ()),
            ),
            _signature_value(getattr(constructor_certificate, "domain_box", ())),
        )
    if source_type == "AffineHalfspaceDecision":
        return (
            source_type,
            getattr(constructor_certificate, "decision_id", ""),
            _signature_value(getattr(constructor_certificate, "coefficients", ())),
            _signature_value(getattr(constructor_certificate, "domain_box", ())),
            _signature_value(
                getattr(constructor_certificate, "slab_half_width", None),
            ),
        )
    if source_type in {
        "AffineHalfspaceArrangement",
        "AffineHalfspace3DArrangement",
    }:
        return (
            source_type,
            getattr(constructor_certificate, "arrangement_id", ""),
            _signature_value(
                getattr(constructor_certificate, "decision_functions", ()),
            ),
            _signature_value(getattr(constructor_certificate, "domain_box", ())),
            _signature_value(
                getattr(constructor_certificate, "slab_half_width", None),
            ),
        )
    return (source_type,)


def _constructor_generated_evidence_signature(
    constructor_certificate: object,
) -> tuple[object, ...]:
    source_type = str(
        getattr(
            getattr(constructor_certificate, "source_tree", None),
            "source_type",
            "",
        )
    )
    decision_functions = tuple(
        (
            getattr(decision_function, "decision_id", ""),
            _signature_value(getattr(decision_function, "root_brackets", ())),
        )
        for decision_function in tuple(
            getattr(constructor_certificate, "decision_functions", ()) or ()
        )
    )
    strata = tuple(
        _signature_value(stratum)
        for stratum in tuple(getattr(constructor_certificate, "strata", ()) or ())
    )
    cells = tuple(
        _signature_value(cell)
        for cell in tuple(getattr(constructor_certificate, "cells", ()) or ())
    )
    return (
        source_type,
        _signature_value(getattr(constructor_certificate, "root_brackets", ())),
        _signature_value(getattr(constructor_certificate, "denominator_interval", ())),
        _signature_value(getattr(constructor_certificate, "denominator_sign", None)),
        _signature_value(getattr(constructor_certificate, "denominator_intervals", ())),
        _signature_value(getattr(constructor_certificate, "denominator_signs", ())),
        decision_functions,
        strata,
        cells,
    )


def _constructor_resolution_policy_signature(
    constructor_certificate: object,
) -> tuple[tuple[object, ...], ...]:
    """Policies actually emitted on equality/selector strata by a constructor."""

    stratified_tree = getattr(constructor_certificate, "stratified_tree", None)
    rows: list[tuple[object, ...]] = []
    for leaf in tuple(getattr(stratified_tree, "leaf_certificates", ()) or ()):
        equality_strata = [
            getattr(leaf, "equality_stratum", None),
            getattr(getattr(leaf, "event_order_tie", None), "equality_stratum", None),
        ]
        for equality in equality_strata:
            policy = str(getattr(equality, "resolution_policy", ""))
            if not policy:
                continue
            rows.append(
                (
                    str(getattr(leaf, "leaf_id", "")),
                    str(getattr(equality, "stratum_id", "")),
                    _signature_value(
                        getattr(equality, "defining_function_ids", ()),
                    ),
                    policy,
                )
            )
    return tuple(rows)


def _supported_constructor_replay_matches_grammar_input(
    *,
    grammar_input: object,
    constructor_certificate: object,
) -> bool:
    """Regenerate the supported constructor from raw grammar input.

    Stored payload tuples are useful diagnostics, but they are certificate
    fields and can be replaced.  Proof certification therefore also requires
    the current raw grammar input to replay the same constructor evidence,
    including equality-resolution policies and Sturm root-isolation outcomes.
    """

    if not (
        isinstance(grammar_input, SupportedEventFunctionGrammarInput)
        and isinstance(
            constructor_certificate,
            SUPPORTED_EVENT_FUNCTION_STRATIFICATION_CONSTRUCTOR_TYPES,
        )
    ):
        return False
    try:
        replayed_constructor = _construct_supported_event_function_stratification(
            grammar_input,
        )
    except (TypeError, ValueError, ArithmeticError, FloatingPointError):
        return False
    return bool(
        str(
            getattr(
                getattr(replayed_constructor, "source_tree", None),
                "source_type",
                "",
            )
        )
        == str(
            getattr(
                getattr(constructor_certificate, "source_tree", None),
                "source_type",
                "",
            )
        )
        and _constructor_input_signature(replayed_constructor)
        == _constructor_input_signature(constructor_certificate)
        and _constructor_generated_evidence_signature(replayed_constructor)
        == _constructor_generated_evidence_signature(constructor_certificate)
        and _constructor_resolution_policy_signature(replayed_constructor)
        == _constructor_resolution_policy_signature(constructor_certificate)
    )


def _consumption_source_tree_matches_generated(
    consumption: RecursiveStratifiedBranchEventConsumptionCertificate,
    generated_stratified_tree: object,
) -> bool:
    generated_trees = (
        tuple(generated_stratified_tree)
        if isinstance(generated_stratified_tree, tuple)
        else (generated_stratified_tree,)
    )
    return any(consumption.source_tree == tree for tree in generated_trees)


@dataclass(frozen=True)
class ArbitraryIntervalInputPartitionGenerationCertificate:
    """Scoped interval-input partition generation for explicit grammars.

    This is not an arbitrary analytic partition theorem.  It certifies only
    finite interval inputs whose event discriminants are already represented in
    one of the supported constructor grammars listed by
    ``event_function_grammar_id``.  Unsupported analytic strata are surfaced as
    blockers rather than hulled back into ambient interval boxes.
    """

    input_scope_id: str
    event_function_grammar_id: str
    branch_function_grammar_id: str
    event_order_function_grammar_id: str
    branch_constructor_source_type: str
    event_order_constructor_source_type: str
    branch_constructor_input_scope_id: str
    event_order_constructor_input_scope_id: str
    generated_stratified_tree: object
    branch_consumption_certificate: (
        RecursiveStratifiedBranchEventConsumptionCertificate
    )
    event_order_consumption_certificate: (
        RecursiveStratifiedBranchEventConsumptionCertificate
    )
    unsupported_strata: tuple[str, ...]
    arbitrary_partition_generation_claimed: bool
    set_valued_constructor_certificate: (
        SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate
    )
    validated_set_valued_constructor_certificate: (
        ValidatedSetValuedConstructorCompletenessTheoremCertificate
    )
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "scoped_arbitrary_interval_input_partition_generation"

    @property
    def component_types_certified(self) -> bool:
        return bool(
            isinstance(
                self.branch_consumption_certificate,
                RecursiveStratifiedBranchEventConsumptionCertificate,
            )
            and isinstance(
                self.event_order_consumption_certificate,
                RecursiveStratifiedBranchEventConsumptionCertificate,
            )
            and isinstance(
                self.set_valued_constructor_certificate,
                SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate,
            )
            and isinstance(
                self.validated_set_valued_constructor_certificate,
                ValidatedSetValuedConstructorCompletenessTheoremCertificate,
            )
        )

    @property
    def source_matches_constructor_chain(self) -> bool:
        return bool(
            self.component_types_certified
            and (
                self.set_valued_constructor_certificate.branch_consumption_certificate
                is self.branch_consumption_certificate
            )
            and (
                self.set_valued_constructor_certificate.event_order_consumption_certificate
                is self.event_order_consumption_certificate
            )
            and (
                self.validated_set_valued_constructor_certificate
                .set_valued_constructor_certificate
                is self.set_valued_constructor_certificate
            )
            and _consumption_source_tree_matches_generated(
                self.branch_consumption_certificate,
                self.generated_stratified_tree,
            )
            and _consumption_source_tree_matches_generated(
                self.event_order_consumption_certificate,
                self.generated_stratified_tree,
            )
        )

    @property
    def expected_branch_constructor_source_type(self) -> str:
        if not self.component_types_certified:
            return ""
        return recursive_constructor_source_type(self.branch_consumption_certificate)

    @property
    def expected_event_order_constructor_source_type(self) -> str:
        if not self.component_types_certified:
            return ""
        return recursive_constructor_source_type(
            self.event_order_consumption_certificate,
        )

    @property
    def expected_branch_constructor_input_scope_id(self) -> str:
        if not self.component_types_certified:
            return ""
        scope = recursive_constructor_source_scope(
            self.branch_consumption_certificate,
        )
        return scope[1] if scope is not None else ""

    @property
    def expected_event_order_constructor_input_scope_id(self) -> str:
        if not self.component_types_certified:
            return ""
        scope = recursive_constructor_source_scope(
            self.event_order_consumption_certificate,
        )
        return scope[1] if scope is not None else ""

    @property
    def expected_branch_function_grammar_id(self) -> str:
        return SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS.get(
            self.expected_branch_constructor_source_type,
            "",
        )

    @property
    def expected_event_order_function_grammar_id(self) -> str:
        return SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS.get(
            self.expected_event_order_constructor_source_type,
            "",
        )

    @property
    def expected_event_function_grammar_id(self) -> str:
        return _scoped_partition_grammar_id(
            self.expected_branch_function_grammar_id,
            self.expected_event_order_function_grammar_id,
        )

    @property
    def expected_input_scope_id(self) -> str:
        if not isinstance(
            self.validated_set_valued_constructor_certificate,
            ValidatedSetValuedConstructorCompletenessTheoremCertificate,
        ):
            return ""
        return self.validated_set_valued_constructor_certificate.input_scope_id

    @property
    def scope_fields_match_constructor_chain(self) -> bool:
        return bool(
            self.component_types_certified
            and self.expected_event_function_grammar_id
            and self.expected_branch_function_grammar_id
            and self.expected_event_order_function_grammar_id
            and self.expected_branch_constructor_input_scope_id
            and self.expected_event_order_constructor_input_scope_id
            and self.expected_input_scope_id
            and self.branch_constructor_source_type
            == self.expected_branch_constructor_source_type
            and self.event_order_constructor_source_type
            == self.expected_event_order_constructor_source_type
            and self.branch_constructor_input_scope_id
            == self.expected_branch_constructor_input_scope_id
            and self.event_order_constructor_input_scope_id
            == self.expected_event_order_constructor_input_scope_id
            and self.branch_function_grammar_id
            == self.expected_branch_function_grammar_id
            and self.event_order_function_grammar_id
            == self.expected_event_order_function_grammar_id
            and self.event_function_grammar_id
            == self.expected_event_function_grammar_id
            and self.input_scope_id == self.expected_input_scope_id
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.input_scope_id
            and self.event_function_grammar_id
            and self.branch_function_grammar_id
            and self.event_order_function_grammar_id
            and self.branch_constructor_source_type
            and self.event_order_constructor_source_type
            and self.branch_constructor_input_scope_id
            and self.event_order_constructor_input_scope_id
            and self.arbitrary_partition_generation_claimed is True
            and not self.unsupported_strata
            and self.component_types_certified is True
            and self.source_matches_constructor_chain is True
            and self.scope_fields_match_constructor_chain is True
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.set_valued_constructor_certificate.proof_certified is True
            and (
                self.validated_set_valued_constructor_certificate.proof_certified
                is True
            )
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _theorem_obligation_ledger_missing(
                self.obligations,
                ledger_name="scoped_arbitrary_interval_partition",
            )
        )
        if not isinstance(
            self.branch_consumption_certificate,
            RecursiveStratifiedBranchEventConsumptionCertificate,
        ):
            missing.append("branch_partition_consumption_type")
        if not isinstance(
            self.event_order_consumption_certificate,
            RecursiveStratifiedBranchEventConsumptionCertificate,
        ):
            missing.append("event_order_partition_consumption_type")
        if not isinstance(
            self.set_valued_constructor_certificate,
            SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate,
        ):
            missing.append("scoped_set_valued_constructor_type")
        if not isinstance(
            self.validated_set_valued_constructor_certificate,
            ValidatedSetValuedConstructorCompletenessTheoremCertificate,
        ):
            missing.append("validated_set_valued_scope_type")
        if not self.source_matches_constructor_chain:
            missing.append("scoped_partition_sources_match_constructor_chain")
        if not self.scope_fields_match_constructor_chain:
            missing.append("scoped_partition_scope_fields_match_constructor_chain")
        if not (
            isinstance(
                self.set_valued_constructor_certificate,
                SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate,
            )
            and self.set_valued_constructor_certificate.proof_certified is True
        ):
            missing.append("scoped_set_valued_constructor_proof_certified")
        if not (
            isinstance(
                self.validated_set_valued_constructor_certificate,
                ValidatedSetValuedConstructorCompletenessTheoremCertificate,
            )
            and (
                self.validated_set_valued_constructor_certificate.proof_certified
                is True
            )
        ):
            missing.append("validated_set_valued_scope_proof_certified")
        missing.extend(f"unsupported:{item}" for item in self.unsupported_strata)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class SupportedEventFunctionStratificationGenerationCertificate:
    """Generated stratification plus scoped partition theorem for one grammar."""

    grammar_input: SupportedEventFunctionGrammarInput
    constructor_certificate: object
    partition_generation_certificate: (
        ArbitraryIntervalInputPartitionGenerationCertificate
    )
    obligations: tuple[TheoremPipelineObligation, ...]
    event_order_grammar_input: SupportedEventFunctionGrammarInput | None = None
    event_order_constructor_certificate: object | None = None
    branch_generated_evidence_signature: tuple[object, ...] = ()
    event_order_generated_evidence_signature: tuple[object, ...] = ()
    branch_grammar_payload_signature: tuple[object, ...] = ()
    event_order_grammar_payload_signature: tuple[object, ...] = ()
    theorem_id: str = "supported_event_function_stratification_generation"

    @property
    def grammar_inputs_typed(self) -> bool:
        return bool(
            isinstance(self.grammar_input, SupportedEventFunctionGrammarInput)
            and (
                self.event_order_grammar_input is None
                or isinstance(
                    self.event_order_grammar_input,
                    SupportedEventFunctionGrammarInput,
                )
            )
        )

    @property
    def constructor_certificate_type_certified(self) -> bool:
        return isinstance(
            self.constructor_certificate,
            SUPPORTED_EVENT_FUNCTION_STRATIFICATION_CONSTRUCTOR_TYPES,
        )

    @property
    def event_order_constructor_certificate_type_certified(self) -> bool:
        return bool(
            self.event_order_constructor_certificate is None
            or isinstance(
                self.event_order_constructor_certificate,
                SUPPORTED_EVENT_FUNCTION_STRATIFICATION_CONSTRUCTOR_TYPES,
            )
        )

    @property
    def partition_generation_certificate_type_certified(self) -> bool:
        return isinstance(
            self.partition_generation_certificate,
            ArbitraryIntervalInputPartitionGenerationCertificate,
        )

    @property
    def constructor_source_type(self) -> str:
        return str(
            getattr(
                getattr(self.constructor_certificate, "source_tree", None),
                "source_type",
                "",
            )
        )

    @property
    def event_order_constructor_source_type(self) -> str:
        constructor = (
            self.event_order_constructor_certificate
            if self.event_order_constructor_certificate is not None
            else self.constructor_certificate
        )
        return str(
            getattr(
                getattr(constructor, "source_tree", None),
                "source_type",
                "",
            )
        )

    @property
    def input_scope_id(self) -> str:
        return str(
            getattr(
                self.partition_generation_certificate,
                "input_scope_id",
                "",
            )
        )

    @property
    def event_function_grammar_id(self) -> str:
        return str(
            getattr(
                self.partition_generation_certificate,
                "event_function_grammar_id",
                "",
            )
        )

    @property
    def expected_branch_grammar_id(self) -> str:
        if not isinstance(self.grammar_input, SupportedEventFunctionGrammarInput):
            return ""
        return self.grammar_input.grammar_id

    @property
    def expected_event_order_grammar_id(self) -> str:
        if not isinstance(self.grammar_input, SupportedEventFunctionGrammarInput):
            return ""
        if self.event_order_grammar_input is None:
            return self.grammar_input.grammar_id
        if not isinstance(
            self.event_order_grammar_input,
            SupportedEventFunctionGrammarInput,
        ):
            return ""
        return self.event_order_grammar_input.grammar_id

    @property
    def partition_bridge_branch_source_matches_constructor(self) -> bool:
        return bool(
            self.partition_generation_certificate_type_certified
            and (
                self.partition_generation_certificate.branch_constructor_source_type
                == self.constructor_source_type
            )
        )

    @property
    def partition_bridge_event_order_source_matches_constructor(self) -> bool:
        return bool(
            self.partition_generation_certificate_type_certified
            and (
                self.partition_generation_certificate.event_order_constructor_source_type
                == self.event_order_constructor_source_type
            )
        )

    @property
    def partition_bridge_branch_grammar_matches_input(self) -> bool:
        return bool(
            self.partition_generation_certificate_type_certified
            and (
                self.partition_generation_certificate.branch_function_grammar_id
                == self.expected_branch_grammar_id
            )
        )

    @property
    def partition_bridge_event_order_grammar_matches_input(self) -> bool:
        return bool(
            self.partition_generation_certificate_type_certified
            and (
                self.partition_generation_certificate.event_order_function_grammar_id
                == self.expected_event_order_grammar_id
            )
        )

    @property
    def constructor_input_matches_grammar_input(self) -> bool:
        if not (
            isinstance(self.grammar_input, SupportedEventFunctionGrammarInput)
            and self.constructor_certificate_type_certified
        ):
            return False
        return bool(
            _constructor_input_signature(self.constructor_certificate)
            == _supported_grammar_input_signature(self.grammar_input)
        )

    @property
    def event_order_constructor_input_matches_grammar_input(self) -> bool:
        if not self.grammar_inputs_typed:
            return False
        if self.event_order_grammar_input is None:
            event_order_input = self.grammar_input
            event_order_constructor = self.constructor_certificate
        else:
            event_order_input = self.event_order_grammar_input
            event_order_constructor = self.event_order_constructor_certificate
        if not (
            isinstance(event_order_input, SupportedEventFunctionGrammarInput)
            and isinstance(
                event_order_constructor,
                SUPPORTED_EVENT_FUNCTION_STRATIFICATION_CONSTRUCTOR_TYPES,
            )
        ):
            return False
        return bool(
            _constructor_input_signature(event_order_constructor)
            == _supported_grammar_input_signature(event_order_input)
        )

    @property
    def grammar_payload_matches_generation(self) -> bool:
        return bool(
            isinstance(self.grammar_input, SupportedEventFunctionGrammarInput)
            and self.branch_grammar_payload_signature
            and _supported_grammar_payload_signature(self.grammar_input)
            == tuple(self.branch_grammar_payload_signature)
        )

    @property
    def event_order_grammar_payload_matches_generation(self) -> bool:
        if self.event_order_grammar_input is None:
            return True
        return bool(
            isinstance(
                self.event_order_grammar_input,
                SupportedEventFunctionGrammarInput,
            )
            and self.event_order_grammar_payload_signature
            and _supported_grammar_payload_signature(self.event_order_grammar_input)
            == tuple(self.event_order_grammar_payload_signature)
        )

    @property
    def constructor_generated_evidence_matches_generation(self) -> bool:
        return bool(
            self.constructor_certificate_type_certified
            and self.branch_generated_evidence_signature
            and _constructor_generated_evidence_signature(self.constructor_certificate)
            == tuple(self.branch_generated_evidence_signature)
        )

    @property
    def constructor_replays_from_grammar_input(self) -> bool:
        return _supported_constructor_replay_matches_grammar_input(
            grammar_input=self.grammar_input,
            constructor_certificate=self.constructor_certificate,
        )

    @property
    def event_order_constructor_generated_evidence_matches_generation(self) -> bool:
        if self.event_order_grammar_input is None:
            return True
        if not self.event_order_constructor_certificate_type_certified:
            return False
        constructor = self.event_order_constructor_certificate
        return bool(
            self.event_order_generated_evidence_signature
            and _constructor_generated_evidence_signature(constructor)
            == tuple(self.event_order_generated_evidence_signature)
        )

    @property
    def event_order_constructor_replays_from_grammar_input(self) -> bool:
        if self.event_order_grammar_input is None:
            return True
        constructor = self.event_order_constructor_certificate
        return _supported_constructor_replay_matches_grammar_input(
            grammar_input=self.event_order_grammar_input,
            constructor_certificate=constructor,
        )

    @property
    def partition_bridge_branch_tree_matches_constructor(self) -> bool:
        if not (
            self.partition_generation_certificate_type_certified
            and self.constructor_certificate_type_certified
        ):
            return False
        return bool(
            self.partition_generation_certificate
            .branch_consumption_certificate
            .source_tree
            == getattr(self.constructor_certificate, "stratified_tree", None)
        )

    @property
    def partition_bridge_event_order_tree_matches_constructor(self) -> bool:
        if not (
            self.partition_generation_certificate_type_certified
            and self.event_order_constructor_certificate_type_certified
        ):
            return False
        constructor = (
            self.event_order_constructor_certificate
            if self.event_order_constructor_certificate is not None
            else self.constructor_certificate
        )
        return bool(
            self.partition_generation_certificate
            .event_order_consumption_certificate
            .source_tree
            == getattr(constructor, "stratified_tree", None)
        )

    @property
    def certified(self) -> bool:
        event_order_matches = bool(
            (
                self.grammar_inputs_typed
                and self.event_order_grammar_input is None
            )
            or (
                isinstance(
                    self.event_order_grammar_input,
                    SupportedEventFunctionGrammarInput,
                )
                and self.event_order_grammar_input.supported
                and self.event_order_constructor_certificate_type_certified
                and self.event_order_constructor_source_type
                == self.event_order_grammar_input.source_type
                and _object_proof_certified(
                    self.event_order_constructor_certificate,
                )
            )
        )
        return bool(
            self.grammar_inputs_typed
            and self.constructor_certificate_type_certified
            and self.partition_generation_certificate_type_certified
            and self.grammar_input.supported
            and self.constructor_source_type == self.grammar_input.source_type
            and self.constructor_input_matches_grammar_input
            and self.grammar_payload_matches_generation
            and self.constructor_generated_evidence_matches_generation
            and self.constructor_replays_from_grammar_input
            and _object_proof_certified(self.constructor_certificate)
            and event_order_matches
            and self.event_order_constructor_input_matches_grammar_input
            and self.event_order_grammar_payload_matches_generation
            and self.event_order_constructor_generated_evidence_matches_generation
            and self.event_order_constructor_replays_from_grammar_input
            and self.partition_generation_certificate.certified is True
            and self.partition_bridge_branch_source_matches_constructor
            and self.partition_bridge_event_order_source_matches_constructor
            and self.partition_bridge_branch_grammar_matches_input
            and self.partition_bridge_event_order_grammar_matches_input
            and self.partition_bridge_branch_tree_matches_constructor
            and self.partition_bridge_event_order_tree_matches_constructor
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.partition_generation_certificate.proof_certified is True
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _theorem_obligation_ledger_missing(
                self.obligations,
                ledger_name="supported_event_function_generation",
            )
        )
        if not isinstance(self.grammar_input, SupportedEventFunctionGrammarInput):
            missing.append("supported_event_function_grammar_input_type")
        if self.event_order_grammar_input is not None and not isinstance(
            self.event_order_grammar_input,
            SupportedEventFunctionGrammarInput,
        ):
            missing.append("supported_event_order_grammar_input_type")
        if not self.constructor_certificate_type_certified:
            missing.append("generated_constructor_certificate_type")
        if not self.event_order_constructor_certificate_type_certified:
            missing.append("generated_event_order_constructor_certificate_type")
        if not self.partition_generation_certificate_type_certified:
            missing.append("generated_partition_bridge_type")
        if not self.partition_bridge_branch_source_matches_constructor:
            missing.append("generated_partition_bridge_branch_source_matches_constructor")
        if not self.partition_bridge_event_order_source_matches_constructor:
            missing.append(
                "generated_partition_bridge_event_order_source_matches_constructor"
            )
        if not self.partition_bridge_branch_grammar_matches_input:
            missing.append("generated_partition_bridge_branch_grammar_matches_input")
        if not self.partition_bridge_event_order_grammar_matches_input:
            missing.append(
                "generated_partition_bridge_event_order_grammar_matches_input"
            )
        if not self.constructor_input_matches_grammar_input:
            missing.append("generated_constructor_input_matches_grammar_input")
        if not self.grammar_payload_matches_generation:
            missing.append("grammar_input_payload_matches_generation")
        if not self.constructor_generated_evidence_matches_generation:
            missing.append(
                "generated_constructor_evidence_matches_generation"
            )
        if not self.constructor_replays_from_grammar_input:
            missing.append("generated_constructor_replays_from_grammar_input")
        if (
            self.event_order_grammar_input is not None
            and not self.event_order_constructor_input_matches_grammar_input
        ):
            missing.append(
                "generated_event_order_constructor_input_matches_grammar_input"
            )
        if (
            self.event_order_grammar_input is not None
            and not self.event_order_grammar_payload_matches_generation
        ):
            missing.append(
                "event_order_grammar_input_payload_matches_generation"
            )
        if (
            self.event_order_grammar_input is not None
            and not self.event_order_constructor_generated_evidence_matches_generation
        ):
            missing.append(
                "generated_event_order_constructor_evidence_matches_generation"
            )
        if (
            self.event_order_grammar_input is not None
            and not self.event_order_constructor_replays_from_grammar_input
        ):
            missing.append(
                "generated_event_order_constructor_replays_from_grammar_input"
            )
        if not self.partition_bridge_branch_tree_matches_constructor:
            missing.append("generated_partition_bridge_branch_tree_matches_constructor")
        if not self.partition_bridge_event_order_tree_matches_constructor:
            missing.append(
                "generated_partition_bridge_event_order_tree_matches_constructor"
            )
        if not _object_proof_certified(self.constructor_certificate):
            missing.append("generated_constructor_proof_certified")
            missing.extend(
                f"constructor:{item}"
                for item in getattr(
                    self.constructor_certificate,
                    "missing_obligations",
                    (),
                )
            )
        if (
            self.event_order_constructor_certificate is not None
            and not _object_proof_certified(
                self.event_order_constructor_certificate,
            )
        ):
            missing.append("generated_event_order_constructor_proof_certified")
            missing.extend(
                f"event_order_constructor:{item}"
                for item in getattr(
                    self.event_order_constructor_certificate,
                    "missing_obligations",
                    (),
                )
            )
        if not (
            self.partition_generation_certificate_type_certified
            and self.partition_generation_certificate.proof_certified is True
        ):
            missing.append("generated_partition_bridge_proof_certified")
            if self.partition_generation_certificate_type_certified:
                missing.extend(
                    self.partition_generation_certificate.missing_obligations
                )
        return tuple(dict.fromkeys(str(item) for item in missing))


@dataclass(frozen=True)
class FiniteSuppliedBranchTreeConsumptionCertificate:
    """Finite set-valued branch-tree consumption theorem.

    This certificate is deliberately not an arbitrary interval-box termination
    theorem.  It proves the finite gluing step: once a finite certified branch
    partition covers the input set and each leaf has a proof-certified
    atlas/stop response, their union is a certified set-valued response.
    """

    partition_kind: str
    consumption_kind: str
    branch_count: int
    certified_leaf_count: int
    statement: str
    proof_sketch: str
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "finite_supplied_set_valued_branch_tree_consumption"

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.branch_count > 0
            and self.certified_leaf_count >= self.branch_count
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _theorem_obligation_ledger_missing(
            self.obligations,
            ledger_name="finite_supplied_branch_tree_consumption",
        )


@dataclass(frozen=True)
class UniformMarginBranchRefinementTerminationCertificate:
    """Termination theorem for recursive branch refinement under margins.

    This covers the compact-set case away from decision boundaries.  It does
    not handle equality strata such as simultaneous event times or total-
    collision selector entry; those remain separate stop/selector obligations.
    """

    refinement_kind: str
    uniform_decision_margin: float
    local_decision_lipschitz_bound: float
    initial_width_bound: float
    refinement_factor: float
    max_depth: int
    terminal_width_bound: float
    required_width_bound: float
    statement: str
    proof_sketch: str
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "uniform_margin_recursive_branch_refinement_termination"

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _theorem_obligation_ledger_missing(
            self.obligations,
            ledger_name="uniform_margin_branch_refinement",
        )


@dataclass(frozen=True)
class SuppliedFiniteFuchsianLogStopChartCertificate:
    """Executable stop-chart theorem for supplied finite Fuchsian-log entry data.

    This is intentionally not the arbitrary total-collision entry theorem.  It
    consumes finite entry data that has already been constructed: a branch, a
    punctured isolation radius, primitive Cauchy inputs, and the independent
    serialized stop-chart checker.
    """

    branch: FiniteFuchsianLogBranch
    isolation: FiniteFuchsianLogTotalCollisionIsolationCertificate
    cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputs
    compact_isolation: FiniteFuchsianLogCompactTimeIsolationCertificate | None
    stop_chart_certificate: object
    independent_checker_result: object
    sample_taus: tuple[float, ...]
    max_projection_identity_residual: float
    max_angular_momentum: float
    total_collision_policy_id: str
    tolerance: float
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data"

    @property
    def certified(self) -> bool:
        return bool(
            self.theorem_id
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _theorem_obligation_ledger_missing(
            self.obligations,
            ledger_name="supplied_finite_fuchsian_log_stop_chart",
        )

    @property
    def arbitrary_entry_theorem_claimed(self) -> bool:
        return False


@dataclass(frozen=True)
class SuppliedGeneralizedFuchsianEntryCertificate:
    """Executable supplied-entry theorem for nonresonant generalized branches.

    This consumes an already constructed finite-dimensional
    ``FuchsianShapeBranch`` with possibly noninteger powers.  It verifies the
    branch as local total-collision entry data, not as a full stop chart: a
    Cauchy tail majorant and independent serialized stop-chart checker are
    still separate obligations.
    """

    branch: FuchsianShapeBranch
    radius: float
    sample_taus: tuple[float, ...]
    central_shape_pair_distance_floor: float
    shape_deviation_bound: float
    shape_pair_distance_floor: float
    max_lifted_residual: float
    max_angular_momentum: float
    max_energy_gap: float
    tolerance: float
    energy_tolerance: float
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "supplied_generalized_fuchsian_entry_data"

    @property
    def certified(self) -> bool:
        return bool(
            self.theorem_id
            and self.shape_pair_distance_floor > 0.0
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _theorem_obligation_ledger_missing(
            self.obligations,
            ledger_name="supplied_generalized_fuchsian_entry_data",
        )

    @property
    def arbitrary_entry_theorem_claimed(self) -> bool:
        return False

    @property
    def stop_chart_theorem_claimed(self) -> bool:
        return False


@dataclass(frozen=True)
class SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate:
    """Finite-row truncation budget for supplied generalized Fuchsian branches.

    This is intentionally weaker than a primitive Cauchy remainder theorem.  It
    budgets the rows already present in a constructor-derived
    ``FuchsianShapeBranch`` that are omitted by a retained total degree, and it
    records that an analytic remainder majorant and serialized total-stop chart
    remain separate proof obligations.
    """

    entry_certificate: SuppliedGeneralizedFuchsianEntryCertificate
    retained_total_degree: int
    radius: float
    omitted_indices: tuple[tuple[int, ...], ...]
    component_tail_bounds: Mapping[str, float]
    max_finite_row_tail_bound: float
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "supplied_generalized_fuchsian_finite_row_tail_budget"

    @property
    def certified(self) -> bool:
        return bool(
            self.theorem_id
            and self.retained_total_degree >= 0
            and np.isfinite(self.radius)
            and self.radius > 0.0
            and self.component_tail_bounds
            and all(
                np.isfinite(value) and value >= 0.0
                for value in self.component_tail_bounds.values()
            )
            and np.isfinite(self.max_finite_row_tail_bound)
            and self.max_finite_row_tail_bound >= 0.0
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _theorem_obligation_ledger_missing(
            self.obligations,
            ledger_name="supplied_generalized_fuchsian_finite_row_tail_budget",
        )

    @property
    def analytic_remainder_tail_claimed(self) -> bool:
        return False

    @property
    def stop_chart_theorem_claimed(self) -> bool:
        return False


@dataclass(frozen=True)
class SuppliedGeneralizedFuchsianAnalyticRemainderMajorantCertificate:
    """Banach-majorant certificate for a supplied generalized remainder.

    The certificate proves a local analytic remainder only after finite
    generalized Fuchsian rows and explicit contraction constants have been
    supplied.  It is the local implication

    ``entry rows + finite-row budget + defect/inverse/Lipschitz bounds``
    ``=>`` ``Cauchy-majorized analytic remainder``.

    It still does not derive those constants from an arbitrary incoming
    total-collision germ, and it is not a serialized total-stop chart.
    """

    entry_certificate: SuppliedGeneralizedFuchsianEntryCertificate
    finite_row_budget: SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate
    initial_radius: float
    shell_contraction: float
    analytic_disk_fraction: float
    defect_bound: float
    linear_inverse_bound: float
    nonlinear_lipschitz_bound: float
    remainder_ball_radius: float
    contraction_factor: float
    self_map_bound: float
    banach_contraction_slack: float
    banach_self_map_margin: float
    retained_weight_cutoff: int
    first_omitted_weight: int
    cauchy_polydisc_certified: bool
    polydisc_source_scope: str
    uniform_interval_box_constants_claimed: bool
    component_effective_exponents: Mapping[str, float]
    component_inputs: Mapping[str, PrimitiveCauchyTailInput]
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "supplied_generalized_fuchsian_analytic_remainder_majorant"

    @property
    def certified(self) -> bool:
        return bool(
            self.theorem_id
            and self.initial_radius > 0.0
            and np.isfinite(self.initial_radius)
            and 0.0 < self.shell_contraction < 1.0
            and np.isfinite(self.shell_contraction)
            and 0.0 < self.analytic_disk_fraction < 1.0
            and np.isfinite(self.analytic_disk_fraction)
            and np.isfinite(self.defect_bound)
            and self.defect_bound >= 0.0
            and np.isfinite(self.linear_inverse_bound)
            and self.linear_inverse_bound >= 0.0
            and np.isfinite(self.nonlinear_lipschitz_bound)
            and self.nonlinear_lipschitz_bound >= 0.0
            and np.isfinite(self.remainder_ball_radius)
            and self.remainder_ball_radius >= 0.0
            and np.isfinite(self.contraction_factor)
            and self.contraction_factor < 1.0
            and np.isfinite(self.self_map_bound)
            and self.self_map_bound <= self.remainder_ball_radius * (1.0 + 1e-12)
            and np.isfinite(self.banach_contraction_slack)
            and self.banach_contraction_slack > 0.0
            and np.isfinite(self.banach_self_map_margin)
            and self.banach_self_map_margin >= -1.0e-12
            and self.retained_weight_cutoff >= 0
            and self.first_omitted_weight > self.retained_weight_cutoff
            and self.cauchy_polydisc_certified
            and self.polydisc_source_scope == "pointwise_supplied_entry"
            and not self.uniform_interval_box_constants_claimed
            and self.component_inputs
            and all(
                getattr(input_, "certified", False) is True
                for input_ in self.component_inputs.values()
            )
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _theorem_obligation_ledger_missing(
            self.obligations,
            ledger_name="supplied_generalized_fuchsian_analytic_remainder_majorant",
        )

    @property
    def analytic_remainder_tail_claimed(self) -> bool:
        return self.certified

    @property
    def arbitrary_entry_theorem_claimed(self) -> bool:
        return False

    @property
    def stop_chart_theorem_claimed(self) -> bool:
        return False

    @property
    def pointwise_supplied_entry_scope_certified(self) -> bool:
        return bool(
            self.polydisc_source_scope == "pointwise_supplied_entry"
            and not self.uniform_interval_box_constants_claimed
        )

    def component_input(self, component: str) -> PrimitiveCauchyTailInput:
        return self.component_inputs[str(component)]


@dataclass(frozen=True)
class SuppliedGeneralizedFuchsianStopChartCertificate:
    """Supplied-entry generalized Fuchsian maximal-classical stop chart.

    This is the generalized analogue of the finite Fuchsian-log stop theorem,
    but only for already supplied local data.  It consumes the entry rows,
    finite-row tail budget, Banach-majorized analytic remainder, and the
    independent serialized generalized stop-chart checker.  It does not claim
    the arbitrary incoming-germ theorem.
    """

    entry_certificate: SuppliedGeneralizedFuchsianEntryCertificate
    finite_row_budget: SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate
    remainder_majorant: SuppliedGeneralizedFuchsianAnalyticRemainderMajorantCertificate
    tau_interval: tuple[float, float]
    physical_time_interval: tuple[float, float]
    event_physical_time: float
    total_collision_policy_id: str
    residual_tolerance: float
    angular_momentum_tolerance: float
    tail_bound: float
    endpoint_position_tail_bound: float
    residual_tail_bound: float
    obligations: tuple[TheoremPipelineObligation, ...]
    stop_chart_certificate: object | None = None
    independent_checker_result: object | None = None
    theorem_id: str = "supplied_generalized_fuchsian_stop_chart"

    @property
    def certified(self) -> bool:
        return bool(
            self.theorem_id
            and self.total_collision_policy_id == "maximal_classical_stop"
            and _finite_nonempty_interval(self.tau_interval)
            and _finite_nonempty_interval(self.physical_time_interval)
            and self.tau_interval[0] < 0.0 < self.tau_interval[1]
            and self.physical_time_interval[0] < self.event_physical_time < self.physical_time_interval[1]
            and np.isfinite(self.residual_tolerance)
            and self.residual_tolerance >= 0.0
            and np.isfinite(self.angular_momentum_tolerance)
            and self.angular_momentum_tolerance >= 0.0
            and np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
            and np.isfinite(self.endpoint_position_tail_bound)
            and self.endpoint_position_tail_bound >= 0.0
            and np.isfinite(self.residual_tail_bound)
            and self.residual_tail_bound >= 0.0
            and _theorem_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _theorem_obligation_ledger_missing(
            self.obligations,
            ledger_name="supplied_generalized_fuchsian_stop_chart",
        )

    @property
    def arbitrary_entry_theorem_claimed(self) -> bool:
        return False

    @property
    def independent_serialized_checker_claimed(self) -> bool:
        return getattr(self.independent_checker_result, "certified", False) is True

    @property
    def stop_chart_theorem_claimed(self) -> bool:
        return self.certified


def certify_supplied_generalized_fuchsian_entry_data(
    *,
    branch: FuchsianShapeBranch,
    radius: float,
    sample_taus: tuple[float, ...] = (-0.035, 0.035),
    tolerance: float = 1.0e-5,
    energy_tolerance: float = 1.0e-8,
) -> SuppliedGeneralizedFuchsianEntryCertificate:
    """Certify supplied generalized Fuchsian entry data.

    The local implication here is

    ``constructor-derived nonresonant Fuchsian branch + small isolation radius``
    ``=>`` ``finite-dimensional generalized total-collision entry data``.

    The constructor handles fractional Fuchsian powers already represented by
    ``FuchsianShapeBranch``.  It does not derive such a branch from an
    arbitrary incoming total-collision germ, and it does not build the
    primitive Cauchy tail required by the independent stop-chart checker.
    """

    if isinstance(branch, (bool, np.bool_)):
        raise TypeError("branch must be a constructor-derived FuchsianShapeBranch")
    radius = float(radius)
    tolerance = float(tolerance)
    energy_tolerance = float(energy_tolerance)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if not np.isfinite(energy_tolerance) or energy_tolerance <= 0.0:
        raise ValueError("energy_tolerance must be positive and finite")
    sample_taus = tuple(float(tau) for tau in sample_taus)

    central_floor = (
        _minimum_pair_distance(branch.central_shape)
        if isinstance(branch, FuchsianShapeBranch)
        else float("inf")
    )
    shape_deviation = (
        _generalized_fuchsian_shape_deviation_bound(branch, radius)
        if isinstance(branch, FuchsianShapeBranch)
        else float("inf")
    )
    dimension = (
        int(np.asarray(branch.central_shape).shape[1])
        if isinstance(branch, FuchsianShapeBranch)
        and np.asarray(branch.central_shape).ndim == 2
        else 0
    )
    shape_floor = (
        central_floor - 2.0 * math.sqrt(float(dimension)) * shape_deviation
        if dimension > 0
        else float("-inf")
    )
    valid_sample_taus = bool(
        sample_taus
        and all(
            np.isfinite(tau)
            and 0.0 < abs(tau) <= radius
            for tau in sample_taus
        )
    )
    residuals: tuple[float, ...] = ()
    angular: tuple[float, ...] = ()
    energy_gaps: tuple[float, ...] = ()
    if valid_sample_taus and isinstance(branch, FuchsianShapeBranch):
        residuals = tuple(
            float(np.linalg.norm(branch.shape_equation_residual_at_tau(tau), ord=np.inf))
            for tau in sample_taus
        )
        angular = tuple(
            abs(float(branch.centered_angular_momentum_scalar_at_tau(tau)))
            for tau in sample_taus
        )
        if np.isfinite(branch.finite_energy_limit):
            energy_gaps = tuple(
                abs(
                    float(energy(branch.state_at_tau(tau), branch.masses))
                    - branch.finite_energy_limit
                )
                for tau in sample_taus
            )
    max_residual = float(max(residuals, default=float("inf")))
    max_angular = float(max(angular, default=float("inf")))
    max_energy_gap = float(max(energy_gaps, default=float("inf")))

    obligations = (
        TheoremPipelineObligation(
            obligation="supplied_generalized_fuchsian_branch",
            certified=isinstance(branch, FuchsianShapeBranch),
            source=type(branch).__name__,
            detail="finite-dimensional Fuchsian entry data is supplied by a constructor",
        ),
        TheoremPipelineObligation(
            obligation="nonresonant_fuchsian_recurrence",
            certified=bool(
                isinstance(branch, FuchsianShapeBranch)
                and branch.recurrence_certified
            ),
            source=type(branch).__name__,
            detail=(
                f"max_recurrence_residual={getattr(branch, 'max_recurrence_residual_norm', 'missing')!r}; "
                f"min_singular_floor={getattr(branch, 'min_solved_singular_value_floor', 'missing')!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_energy_scale_row",
            certified=bool(
                isinstance(branch, FuchsianShapeBranch)
                and branch.scale_index is not None
                and np.isfinite(branch.finite_energy_limit)
            ),
            source=type(branch).__name__,
            detail=(
                f"scale_index={getattr(branch, 'scale_index', None)!r}; "
                f"finite_energy_limit={getattr(branch, 'finite_energy_limit', float('nan'))!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="planar_generalized_fuchsian_shape",
            certified=dimension == 2,
            source=type(branch).__name__,
            detail=(
                "this supplied-entry checker currently verifies the planar "
                f"angular scalar; shape_dimension={dimension}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generalized_fuchsian_punctured_isolation",
            certified=bool(
                np.isfinite(central_floor)
                and central_floor > 0.0
                and np.isfinite(shape_deviation)
                and shape_floor > 0.0
            ),
            source="finite_fuchsian_coefficient_majorant",
            detail=(
                f"central_floor={central_floor!r}; "
                f"shape_deviation_bound={shape_deviation!r}; "
                f"shape_pair_distance_floor={shape_floor!r}; radius={radius!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generalized_fuchsian_lifted_residual_from_recurrence",
            certified=bool(
                isinstance(branch, FuchsianShapeBranch)
                and branch.recurrence_certified
            ),
            source=type(branch).__name__,
            detail=(
                "finite supplied rows satisfy the constructor coefficient "
                "recurrence; sampled lifted residuals are diagnostic only"
            ),
        ),
        TheoremPipelineObligation(
            obligation="diagnostic_sample_taus_inside_generalized_fuchsian_radius",
            certified=valid_sample_taus,
            source="certify_supplied_generalized_fuchsian_entry_data",
            required=False,
            detail="sample_taus=" + ",".join(str(tau) for tau in sample_taus),
        ),
        TheoremPipelineObligation(
            obligation="diagnostic_sampled_generalized_fuchsian_lifted_residual",
            certified=bool(max_residual <= tolerance),
            source=type(branch).__name__,
            required=False,
            detail=f"max_lifted_residual={max_residual!r}",
        ),
        TheoremPipelineObligation(
            obligation="diagnostic_sampled_generalized_fuchsian_zero_angular_momentum",
            certified=bool(max_angular <= tolerance),
            source=type(branch).__name__,
            required=False,
            detail=f"max_angular_momentum={max_angular!r}",
        ),
        TheoremPipelineObligation(
            obligation="diagnostic_sampled_generalized_fuchsian_finite_energy_matching",
            certified=bool(max_energy_gap <= energy_tolerance),
            source=type(branch).__name__,
            required=False,
            detail=f"max_energy_gap={max_energy_gap!r}",
        ),
        TheoremPipelineObligation(
            obligation="generalized_fuchsian_tail_majorant_not_claimed",
            certified=True,
            source="supplied_generalized_fuchsian_entry_data",
            required=False,
            detail=(
                "this certificate verifies finite supplied entry rows only; "
                "primitive Cauchy tail inputs and a serialized stop chart are "
                "separate obligations"
            ),
        ),
        TheoremPipelineObligation(
            obligation="arbitrary_total_collision_entry_not_claimed",
            certified=True,
            source="supplied_generalized_fuchsian_entry_data",
            required=False,
            detail=(
                "does not derive generalized Fuchsian data from an arbitrary "
                "incoming total-collision germ"
            ),
        ),
    )
    return SuppliedGeneralizedFuchsianEntryCertificate(
        branch=branch,
        radius=radius,
        sample_taus=sample_taus,
        central_shape_pair_distance_floor=float(central_floor),
        shape_deviation_bound=float(shape_deviation),
        shape_pair_distance_floor=float(shape_floor),
        max_lifted_residual=max_residual,
        max_angular_momentum=max_angular,
        max_energy_gap=max_energy_gap,
        tolerance=tolerance,
        energy_tolerance=energy_tolerance,
        obligations=obligations,
    )


def certify_supplied_generalized_fuchsian_finite_row_tail_budget(
    *,
    entry_certificate: SuppliedGeneralizedFuchsianEntryCertificate,
    retained_total_degree: int,
    radius: float | None = None,
) -> SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate:
    """Budget finite generalized Fuchsian rows omitted after truncation.

    The constructor is a stopgap for the generalized-entry path: it gives the
    theorem pipeline a machine-checkable accounting of finite rows already in
    the supplied branch, while refusing to claim the Cauchy majorant needed for
    the unknown analytic remainder.  It therefore cannot feed the independent
    total-collision stop-chart checker by itself.
    """

    if isinstance(entry_certificate, (bool, np.bool_)):
        raise TypeError(
            "entry_certificate must be a SuppliedGeneralizedFuchsianEntryCertificate"
        )
    if not isinstance(
        entry_certificate,
        SuppliedGeneralizedFuchsianEntryCertificate,
    ):
        raise TypeError(
            "entry_certificate must be a SuppliedGeneralizedFuchsianEntryCertificate"
        )
    retained_total_degree = int(retained_total_degree)
    radius = (
        float(entry_certificate.radius)
        if radius is None
        else float(radius)
    )
    branch = entry_certificate.branch
    omitted_indices = tuple(
        sorted(
            (
                tuple(index)
                for index in branch.coefficients
                if sum(index) > retained_total_degree
            ),
            key=lambda index: (sum(index), index),
        )
    )
    component_tail_bounds = {
        "shape_value": _generalized_fuchsian_finite_row_tail_bound(
            branch,
            radius,
            retained_total_degree=retained_total_degree,
            lift_power=0.0,
            derivative_order=0,
        ),
        "shape_first_tau_derivative": _generalized_fuchsian_finite_row_tail_bound(
            branch,
            radius,
            retained_total_degree=retained_total_degree,
            lift_power=0.0,
            derivative_order=1,
        ),
        "regularized_position_value": _generalized_fuchsian_finite_row_tail_bound(
            branch,
            radius,
            retained_total_degree=retained_total_degree,
            lift_power=2.0,
            derivative_order=0,
        ),
        "regularized_position_first_jet": _generalized_fuchsian_finite_row_tail_bound(
            branch,
            radius,
            retained_total_degree=retained_total_degree,
            lift_power=2.0,
            derivative_order=1,
        ),
    }
    max_tail = float(max(component_tail_bounds.values(), default=float("inf")))
    obligations = (
        TheoremPipelineObligation(
            obligation="constructor_supplied_generalized_entry_certificate",
            certified=entry_certificate.certified,
            source=entry_certificate.theorem_id,
            detail=(
                "finite-row budget consumes the supplied generalized entry "
                "certificate, not raw branch booleans"
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_row_retained_degree_valid",
            certified=retained_total_degree >= 0,
            source="certify_supplied_generalized_fuchsian_finite_row_tail_budget",
            detail=(
                f"retained_total_degree={retained_total_degree!r}; "
                f"branch_max_total_degree={branch.max_total_degree!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_row_tail_radius_inside_entry",
            certified=bool(
                np.isfinite(radius)
                and 0.0 < radius <= entry_certificate.radius
            ),
            source=entry_certificate.theorem_id,
            detail=f"radius={radius!r}; entry_radius={entry_certificate.radius!r}",
        ),
        TheoremPipelineObligation(
            obligation="finite_row_tail_bounds_finite",
            certified=bool(
                component_tail_bounds
                and all(
                    np.isfinite(value) and value >= 0.0
                    for value in component_tail_bounds.values()
                )
            ),
            source="finite_supplied_fuchsian_rows",
            detail=(
                f"omitted_row_count={len(omitted_indices)}; "
                f"max_finite_row_tail_bound={max_tail!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="analytic_remainder_cauchy_majorant_not_claimed",
            certified=True,
            source="supplied_generalized_fuchsian_finite_row_tail_budget",
            required=False,
            detail=(
                "bounds only rows present in the supplied finite branch; no "
                "infinite analytic remainder tail is certified"
            ),
        ),
        TheoremPipelineObligation(
            obligation="serialized_total_stop_chart_not_claimed",
            certified=True,
            source="supplied_generalized_fuchsian_finite_row_tail_budget",
            required=False,
            detail=(
                "finite-row budget is not accepted as primitive Cauchy input "
                "for the independent total-collision stop-chart checker"
            ),
        ),
    )
    return SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate(
        entry_certificate=entry_certificate,
        retained_total_degree=retained_total_degree,
        radius=radius,
        omitted_indices=omitted_indices,
        component_tail_bounds=component_tail_bounds,
        max_finite_row_tail_bound=max_tail,
        obligations=obligations,
    )


def certify_supplied_generalized_fuchsian_analytic_remainder_majorant(
    *,
    entry_certificate: SuppliedGeneralizedFuchsianEntryCertificate,
    finite_row_budget: SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate,
    defect_bound: float,
    linear_inverse_bound: float,
    nonlinear_lipschitz_bound: float,
    component_effective_exponents: Mapping[str, float],
    step_ratio_bounds: Mapping[str, float],
    retained_order_initials: Mapping[str, int],
    retained_order_increments: Mapping[str, int],
    remainder_ball_radius: float | None = None,
    initial_radius: float | None = None,
    shell_contraction: float = 0.5,
    analytic_disk_fraction: float = 0.25,
    component_multipliers: Mapping[str, float] | None = None,
) -> SuppliedGeneralizedFuchsianAnalyticRemainderMajorantCertificate:
    """Certify a supplied generalized Fuchsian analytic remainder majorant.

    The normalized remainder equation is represented as a fixed-point problem
    ``w = T(w)`` on a Banach ball.  If ``B`` is a supplied right-inverse bound,
    ``D`` a supplied finite-row defect bound, and ``L`` a supplied nonlinear
    Lipschitz bound on the ball, the constructor checks

    ``q = B L < 1`` and ``B D + q R <= R``.

    It then converts the radius ``R`` into primitive Cauchy tail inputs for the
    named components using their supplied effective shell exponents.  The
    result is local supplied-entry evidence only; arbitrary germ entry and a
    serialized generalized stop checker remain separate obligations.
    """

    if isinstance(entry_certificate, (bool, np.bool_)):
        raise TypeError(
            "entry_certificate must be a SuppliedGeneralizedFuchsianEntryCertificate"
        )
    if isinstance(finite_row_budget, (bool, np.bool_)):
        raise TypeError(
            "finite_row_budget must be a SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate"
        )
    if not isinstance(
        entry_certificate,
        SuppliedGeneralizedFuchsianEntryCertificate,
    ):
        raise TypeError(
            "entry_certificate must be a SuppliedGeneralizedFuchsianEntryCertificate"
        )
    if not isinstance(
        finite_row_budget,
        SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate,
    ):
        raise TypeError(
            "finite_row_budget must be a SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate"
        )

    defect_bound = float(defect_bound)
    linear_inverse_bound = float(linear_inverse_bound)
    nonlinear_lipschitz_bound = float(nonlinear_lipschitz_bound)
    shell_contraction = float(shell_contraction)
    analytic_disk_fraction = float(analytic_disk_fraction)
    initial_radius = (
        min(float(entry_certificate.radius), float(finite_row_budget.radius))
        if initial_radius is None
        else float(initial_radius)
    )
    contraction_factor = float(linear_inverse_bound * nonlinear_lipschitz_bound)
    if remainder_ball_radius is None:
        if (
            np.isfinite(contraction_factor)
            and contraction_factor < 1.0
            and np.isfinite(linear_inverse_bound)
            and np.isfinite(defect_bound)
        ):
            remainder_ball_radius = (
                linear_inverse_bound * defect_bound / (1.0 - contraction_factor)
            )
        else:
            remainder_ball_radius = float("inf")
    remainder_ball_radius = float(remainder_ball_radius)
    self_map_bound = float(
        linear_inverse_bound * defect_bound
        + contraction_factor * remainder_ball_radius
    )
    banach_contraction_slack = float(1.0 - contraction_factor)
    banach_self_map_margin = float(remainder_ball_radius - self_map_bound)
    retained_weight_cutoff = int(finite_row_budget.retained_total_degree)
    first_omitted_weight = retained_weight_cutoff + 1

    exponents = {
        str(component): float(exponent)
        for component, exponent in component_effective_exponents.items()
    }
    multipliers = {component: 1.0 for component in exponents}
    if component_multipliers is not None:
        multipliers.update(
            {
                str(component): float(multiplier)
                for component, multiplier in component_multipliers.items()
            }
        )

    component_inputs: dict[str, PrimitiveCauchyTailInput] = {}
    component_input_errors: list[str] = []
    for component, exponent in exponents.items():
        try:
            if component not in step_ratio_bounds:
                raise ValueError(f"missing step ratio bound for {component}")
            if component not in retained_order_initials:
                raise ValueError(f"missing retained-order initial for {component}")
            if component not in retained_order_increments:
                raise ValueError(f"missing retained-order increment for {component}")
            multiplier = float(multipliers.get(component, 1.0))
            if not np.isfinite(multiplier) or multiplier < 0.0:
                raise ValueError(f"invalid component multiplier for {component}")
            majorant_initial, majorant_growth = (
                _generalized_remainder_shell_majorant(
                    remainder_ball_radius=remainder_ball_radius,
                    effective_exponent=exponent,
                    initial_radius=initial_radius,
                    shell_contraction=shell_contraction,
                    analytic_disk_fraction=analytic_disk_fraction,
                    multiplier=multiplier,
                )
            )
            component_inputs[component] = construct_primitive_cauchy_tail_input(
                majorant_initial=majorant_initial,
                majorant_growth=majorant_growth,
                step_ratio_bound=float(step_ratio_bounds[component]),
                retained_order_initial=int(retained_order_initials[component]),
                retained_order_increment=int(retained_order_increments[component]),
            )
        except (TypeError, ValueError, FloatingPointError) as error:
            component_input_errors.append(f"{component}: {error}")

    same_entry = finite_row_budget.entry_certificate == entry_certificate
    radius_inside = bool(
        np.isfinite(initial_radius)
        and initial_radius > 0.0
        and initial_radius <= entry_certificate.radius
        and initial_radius <= finite_row_budget.radius
    )
    cauchy_polydisc_certified = bool(
        radius_inside
        and np.isfinite(shell_contraction)
        and 0.0 < shell_contraction < 1.0
        and np.isfinite(analytic_disk_fraction)
        and 0.0 < analytic_disk_fraction < 1.0
    )
    contraction_constants_valid = bool(
        np.isfinite(defect_bound)
        and defect_bound >= 0.0
        and np.isfinite(linear_inverse_bound)
        and linear_inverse_bound >= 0.0
        and np.isfinite(nonlinear_lipschitz_bound)
        and nonlinear_lipschitz_bound >= 0.0
        and np.isfinite(remainder_ball_radius)
        and remainder_ball_radius >= 0.0
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="constructor_supplied_generalized_entry_certificate",
            certified=entry_certificate.certified,
            source=entry_certificate.theorem_id,
            detail="analytic remainder majorant consumes supplied entry data",
        ),
        TheoremPipelineObligation(
            obligation="constructor_supplied_finite_row_tail_budget",
            certified=finite_row_budget.certified and same_entry,
            source=finite_row_budget.theorem_id,
            detail=(
                "finite-row defect source is paired with the same entry "
                f"certificate; same_entry={same_entry!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="remainder_shell_radius_inside_entry",
            certified=radius_inside,
            source="certify_supplied_generalized_fuchsian_analytic_remainder_majorant",
            detail=(
                f"initial_radius={initial_radius!r}; "
                f"entry_radius={entry_certificate.radius!r}; "
                f"finite_row_radius={finite_row_budget.radius!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="retained_weight_cutoff_from_finite_rows",
            certified=bool(
                retained_weight_cutoff >= 0
                and first_omitted_weight > retained_weight_cutoff
            ),
            source=finite_row_budget.theorem_id,
            detail=(
                f"W_*={retained_weight_cutoff}; "
                f"first_omitted_weight={first_omitted_weight}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="closed_lifted_cauchy_polydisc_from_supplied_entry",
            certified=cauchy_polydisc_certified,
            source="supplied_entry_radius_and_cauchy_majorant_inputs",
            detail=(
                f"initial_radius={initial_radius!r}; "
                f"shell_contraction={shell_contraction!r}; "
                f"analytic_disk_fraction={analytic_disk_fraction!r}; "
                "scope=pointwise_supplied_entry"
            ),
        ),
        TheoremPipelineObligation(
            obligation="banach_majorant_constants_finite",
            certified=contraction_constants_valid,
            source="supplied_remainder_fixed_point_bounds",
            detail=(
                f"defect={defect_bound!r}; inverse={linear_inverse_bound!r}; "
                f"lipschitz={nonlinear_lipschitz_bound!r}; radius={remainder_ball_radius!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="banach_contraction_factor",
            certified=bool(np.isfinite(contraction_factor) and contraction_factor < 1.0),
            source="supplied_remainder_fixed_point_bounds",
            detail=(
                f"q=B*L={contraction_factor!r}; "
                f"slack=1-q={banach_contraction_slack!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="banach_self_map_ball",
            certified=bool(
                np.isfinite(self_map_bound)
                and np.isfinite(remainder_ball_radius)
                and self_map_bound <= remainder_ball_radius * (1.0 + 1e-12)
            ),
            source="supplied_remainder_fixed_point_bounds",
            detail=(
                f"B*D+q*R={self_map_bound!r}; "
                f"R={remainder_ball_radius!r}; "
                f"margin=R-(B*D+q*R)={banach_self_map_margin!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="primitive_cauchy_inputs_from_remainder_majorant",
            certified=bool(
                component_inputs
                and set(component_inputs) == set(exponents)
                and all(input_.certified for input_ in component_inputs.values())
            ),
            source="generalized_remainder_shell_majorant",
            detail=(
                "components="
                + ",".join(sorted(component_inputs))
                + (
                    "; errors=" + "; ".join(component_input_errors)
                    if component_input_errors
                    else ""
                )
            ),
        ),
        TheoremPipelineObligation(
            obligation="pointwise_supplied_entry_scope_only",
            certified=True,
            source="supplied_generalized_fuchsian_analytic_remainder_majorant",
            detail=(
                "Banach constants are certified for this supplied entry "
                "polydisc only; uniform interval-box constants are not claimed"
            ),
        ),
        TheoremPipelineObligation(
            obligation="uniform_interval_box_constants_not_claimed",
            certified=True,
            source="supplied_generalized_fuchsian_analytic_remainder_majorant",
            required=False,
            detail=(
                "set-valued interval-box total-collision entry constants "
                "remain a separate theorem obligation"
            ),
        ),
        TheoremPipelineObligation(
            obligation="arbitrary_total_collision_entry_not_claimed",
            certified=True,
            source="supplied_generalized_fuchsian_analytic_remainder_majorant",
            required=False,
            detail=(
                "does not derive fixed-point constants from arbitrary "
                "incoming total-collision germs"
            ),
        ),
        TheoremPipelineObligation(
            obligation="serialized_total_stop_chart_not_claimed",
            certified=True,
            source="supplied_generalized_fuchsian_analytic_remainder_majorant",
            required=False,
            detail=(
                "majorant supplies primitive local tails but is not itself an "
                "independently serialized generalized total-stop chart"
            ),
        ),
    )
    return SuppliedGeneralizedFuchsianAnalyticRemainderMajorantCertificate(
        entry_certificate=entry_certificate,
        finite_row_budget=finite_row_budget,
        initial_radius=initial_radius,
        shell_contraction=shell_contraction,
        analytic_disk_fraction=analytic_disk_fraction,
        defect_bound=defect_bound,
        linear_inverse_bound=linear_inverse_bound,
        nonlinear_lipschitz_bound=nonlinear_lipschitz_bound,
        remainder_ball_radius=remainder_ball_radius,
        contraction_factor=contraction_factor,
        self_map_bound=self_map_bound,
        banach_contraction_slack=banach_contraction_slack,
        banach_self_map_margin=banach_self_map_margin,
        retained_weight_cutoff=retained_weight_cutoff,
        first_omitted_weight=first_omitted_weight,
        cauchy_polydisc_certified=cauchy_polydisc_certified,
        polydisc_source_scope="pointwise_supplied_entry",
        uniform_interval_box_constants_claimed=False,
        component_effective_exponents=exponents,
        component_inputs=component_inputs,
        obligations=obligations,
    )


def certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data(
    *,
    entry_certificate: SuppliedGeneralizedFuchsianEntryCertificate,
    finite_row_budget: SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate,
    remainder_majorant: SuppliedGeneralizedFuchsianAnalyticRemainderMajorantCertificate,
    tau_interval: tuple[float, float] | None = None,
    event_physical_time: float = 0.0,
    total_collision_policy_id: str = "maximal_classical_stop",
    residual_tolerance: float = 1.0e-5,
    angular_momentum_tolerance: float = 1.0e-5,
) -> SuppliedGeneralizedFuchsianStopChartCertificate:
    """Certify a supplied generalized Fuchsian maximal-classical stop chart.

    The constructor proves the supplied-entry implication

    ``generalized entry rows + punctured isolation + Banach remainder majorant``
    ``=>`` ``finite maximal-classical total-collision stop chart``.

    It deliberately stops short of two harder claims: deriving the entry data
    from an arbitrary incoming germ, and checking a serialized generalized
    chart in the independent certificate kernel.
    """

    for name, value, expected_type in (
        (
            "entry_certificate",
            entry_certificate,
            SuppliedGeneralizedFuchsianEntryCertificate,
        ),
        (
            "finite_row_budget",
            finite_row_budget,
            SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate,
        ),
        (
            "remainder_majorant",
            remainder_majorant,
            SuppliedGeneralizedFuchsianAnalyticRemainderMajorantCertificate,
        ),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a {expected_type.__name__}")
        if not isinstance(value, expected_type):
            raise TypeError(f"{name} must be a {expected_type.__name__}")

    event_physical_time = float(event_physical_time)
    residual_tolerance = float(residual_tolerance)
    angular_momentum_tolerance = float(angular_momentum_tolerance)
    if tau_interval is None:
        radius = min(
            float(entry_certificate.radius),
            float(finite_row_budget.radius),
            float(remainder_majorant.initial_radius),
        )
        tau_interval = (-radius, radius)
    tau_interval = tuple(float(value) for value in tau_interval)
    physical_time_interval = (
        event_physical_time + tau_interval[0] ** 3,
        event_physical_time + tau_interval[1] ** 3,
    )
    same_entry = (
        finite_row_budget.entry_certificate == entry_certificate
        and remainder_majorant.entry_certificate == entry_certificate
        and remainder_majorant.finite_row_budget == finite_row_budget
    )
    tau_inside = bool(
        _finite_nonempty_interval(tau_interval)
        and tau_interval[0] < 0.0 < tau_interval[1]
        and max(abs(tau_interval[0]), abs(tau_interval[1]))
        <= min(
            entry_certificate.radius,
            finite_row_budget.radius,
            remainder_majorant.initial_radius,
        )
        * (1.0 + 1.0e-12)
    )
    physical_time_ok = bool(
        np.isfinite(event_physical_time)
        and _finite_nonempty_interval(physical_time_interval)
        and physical_time_interval[0] < event_physical_time < physical_time_interval[1]
    )
    endpoint_tail = _generalized_stop_endpoint_tail_bound(
        finite_row_budget,
        remainder_majorant,
    )
    residual_tail = _generalized_stop_residual_tail_bound(
        finite_row_budget,
        remainder_majorant,
    )
    primitive_tail = max(
        (
            input_.first_shell_tail_bound
            for input_ in remainder_majorant.component_inputs.values()
        ),
        default=float("inf"),
    )
    tail_bound = float(
        max(
            finite_row_budget.max_finite_row_tail_bound,
            primitive_tail,
        )
    )
    residual_tail_ok = bool(
        np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
        and residual_tail
        <= residual_tolerance * (1.0 + 1.0e-12)
    )
    stop_chart_certificate = None
    checker_result = None
    checker_error = ""
    try:
        stop_chart_certificate = (
            total_collision_generalized_fuchsian_stop_chart_certificate_from_branch(
                entry_certificate.branch,
                certificate_id="supplied-generalized-fuchsian-stop-chart",
                chart_id="supplied-generalized-fuchsian-stop",
                isolation_radius=entry_certificate.radius,
                central_shape_pair_distance_floor=(
                    entry_certificate.central_shape_pair_distance_floor
                ),
                shape_deviation_bound=entry_certificate.shape_deviation_bound,
                shape_pair_distance_floor=entry_certificate.shape_pair_distance_floor,
                tau_interval=tau_interval,
                event_physical_time=event_physical_time,
                residual_tolerance=residual_tolerance,
                angular_momentum_tolerance=angular_momentum_tolerance,
                tail_bound=tail_bound,
                sample_count=7,
                remainder_majorant=(
                    _generalized_remainder_majorant_certificate_from_supplied(
                        remainder_majorant,
                    )
                ),
            )
        )
        checker_result = check_total_collision_generalized_fuchsian_stop_chart(
            stop_chart_certificate,
        )
    except (FloatingPointError, TypeError, ValueError) as error:
        checker_error = str(error)
    residual_ok = bool(
        residual_tail_ok
        and _checker_obligation_certified(
            checker_result,
            "interval_generalized_fuchsian_lifted_residual_on_punctured_shells",
        )
    )
    angular_ok = bool(
        np.isfinite(angular_momentum_tolerance)
        and angular_momentum_tolerance >= 0.0
        and _checker_obligation_certified(
            checker_result,
            "interval_generalized_zero_angular_momentum_on_punctured_shells",
        )
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="constructor_supplied_generalized_entry_certificate",
            certified=entry_certificate.certified,
            source=entry_certificate.theorem_id,
            detail="supplied generalized entry rows are certified",
        ),
        TheoremPipelineObligation(
            obligation="constructor_supplied_finite_row_tail_budget",
            certified=finite_row_budget.certified,
            source=finite_row_budget.theorem_id,
            detail="finite supplied rows have a truncation budget",
        ),
        TheoremPipelineObligation(
            obligation="constructor_supplied_analytic_remainder_majorant",
            certified=remainder_majorant.certified,
            source=remainder_majorant.theorem_id,
            detail="Banach majorant supplies primitive Cauchy remainder tails",
        ),
        TheoremPipelineObligation(
            obligation="generalized_stop_inputs_refer_to_same_entry",
            certified=same_entry,
            source="certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data",
            detail=f"same_entry={same_entry!r}",
        ),
        TheoremPipelineObligation(
            obligation="generalized_stop_tau_interval_inside_isolation",
            certified=tau_inside,
            source="certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data",
            detail=f"tau_interval={tau_interval!r}",
        ),
        TheoremPipelineObligation(
            obligation="generalized_stop_physical_time_matches_cubic_time",
            certified=physical_time_ok,
            source="cubic_total_collision_time",
            detail=f"physical_time_interval={physical_time_interval!r}",
        ),
        TheoremPipelineObligation(
            obligation="maximal_classical_total_collision_stop_policy",
            certified=str(total_collision_policy_id) == "maximal_classical_stop",
            source="total_collision_policy",
            detail=f"policy={total_collision_policy_id!r}",
        ),
        TheoremPipelineObligation(
            obligation="generalized_stop_residual_tail_within_tolerance",
            certified=residual_ok,
            source="finite_rows_plus_remainder_majorant",
            detail=(
                f"residual_tail={residual_tail!r}; tolerance={residual_tolerance!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generalized_stop_zero_angular_momentum",
            certified=angular_ok,
            source=type(checker_result).__name__ if checker_result is not None else "missing",
            detail=(
                "interval_generalized_zero_angular_momentum_on_punctured_shells"
                if checker_result is not None
                else checker_error or "checker did not run"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generalized_stop_endpoint_collapse_tail",
            certified=bool(np.isfinite(endpoint_tail) and endpoint_tail >= 0.0),
            source="finite_rows_plus_remainder_majorant",
            detail=f"endpoint_position_tail_bound={endpoint_tail!r}",
        ),
        TheoremPipelineObligation(
            obligation="independent_generalized_total_collision_stop_chart_checker",
            certified=getattr(checker_result, "certified", False) is True,
            source=type(checker_result).__name__ if checker_result is not None else "missing",
            detail=(
                "missing="
                + ",".join(getattr(checker_result, "missing_obligations", ()))
                if checker_result is not None
                else checker_error or "checker did not run"
            ),
        ),
        TheoremPipelineObligation(
            obligation="arbitrary_total_collision_entry_not_claimed",
            certified=True,
            source="supplied_generalized_fuchsian_stop_chart",
            required=False,
            detail="does not derive supplied data from arbitrary incoming germs",
        ),
    )
    return SuppliedGeneralizedFuchsianStopChartCertificate(
        entry_certificate=entry_certificate,
        finite_row_budget=finite_row_budget,
        remainder_majorant=remainder_majorant,
        tau_interval=tau_interval,
        physical_time_interval=physical_time_interval,
        event_physical_time=event_physical_time,
        total_collision_policy_id=str(total_collision_policy_id),
        residual_tolerance=residual_tolerance,
        angular_momentum_tolerance=angular_momentum_tolerance,
        tail_bound=tail_bound,
        endpoint_position_tail_bound=endpoint_tail,
        residual_tail_bound=residual_tail,
        obligations=obligations,
        stop_chart_certificate=stop_chart_certificate,
        independent_checker_result=checker_result,
    )


def certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data(
    *,
    branch: FiniteFuchsianLogBranch,
    isolation: FiniteFuchsianLogTotalCollisionIsolationCertificate,
    cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputs,
    compact_isolation: FiniteFuchsianLogCompactTimeIsolationCertificate | None = None,
    sample_taus: tuple[float, ...] = (-0.04, 0.04),
    total_collision_policy_id: str = "maximal_classical_stop",
    tolerance: float = 1.0e-8,
) -> SuppliedFiniteFuchsianLogStopChartCertificate:
    """Certify the supplied-entry finite Fuchsian-log stop-chart theorem.

    The theorem proved here is the local implication

    ``finite Fuchsian-log entry data + punctured isolation + Cauchy inputs``
    ``=>`` ``verified maximal-classical total-collision stop chart``.

    It does not derive finite entry data from an arbitrary incoming total-
    collision germ; that remains a separate analytic obligation.
    """

    if isinstance(branch, (bool, np.bool_)):
        raise TypeError("branch must be a constructor-derived Fuchsian-log branch")
    if isinstance(isolation, (bool, np.bool_)):
        raise TypeError("isolation must be a constructor-derived certificate")
    if isinstance(cauchy_inputs, (bool, np.bool_)):
        raise TypeError("cauchy_inputs must be a constructor-derived certificate")
    if not isinstance(branch, FiniteFuchsianLogBranch):
        raise TypeError("branch must be a FiniteFuchsianLogBranch")
    if not isinstance(isolation, FiniteFuchsianLogTotalCollisionIsolationCertificate):
        raise TypeError(
            "isolation must be a FiniteFuchsianLogTotalCollisionIsolationCertificate"
        )
    if not isinstance(cauchy_inputs, FiniteFuchsianLogPrimitiveCauchyInputs):
        raise TypeError("cauchy_inputs must be FiniteFuchsianLogPrimitiveCauchyInputs")
    if compact_isolation is not None and not isinstance(
        compact_isolation,
        FiniteFuchsianLogCompactTimeIsolationCertificate,
    ):
        raise TypeError(
            "compact_isolation must be a FiniteFuchsianLogCompactTimeIsolationCertificate"
        )
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    sample_taus = tuple(float(tau) for tau in sample_taus)

    recomputed_isolation: FiniteFuchsianLogTotalCollisionIsolationCertificate | None = None
    recomputed_isolation_error: str | None = None
    try:
        recomputed_isolation = certify_finite_fuchsian_log_total_collision_isolation(
            branch,
            radius=float(isolation.radius),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        recomputed_isolation_error = str(exc)

    valid_sample_taus = _sample_taus_inside_fuchsian_isolation(
        sample_taus,
        isolation,
    )
    projection_bounds: tuple[float, ...] = ()
    angular_bounds: tuple[float, ...] = ()
    if valid_sample_taus:
        projection_bounds = tuple(
            _fuchsian_projection_identity_residual_bound(branch, tau)
            for tau in sample_taus
        )
        angular_bounds = tuple(
            abs(float(branch.centered_angular_momentum_scalar_at_tau(tau)))
            for tau in sample_taus
        )
    max_projection_residual = float(max(projection_bounds, default=float("inf")))
    max_angular_momentum = float(max(angular_bounds, default=float("inf")))

    stop_chart_certificate = None
    checker_result = None
    checker_error: str | None = None
    if recomputed_isolation is not None:
        try:
            stop_chart_certificate = total_collision_fuchsian_stop_chart_certificate_from_branch(
                branch,
                recomputed_isolation,
                certificate_id="supplied-fuchsian-log-entry-stop-chart",
                chart_id="supplied-fuchsian-log-entry-stop",
                tau_interval=(
                    -float(recomputed_isolation.radius),
                    float(recomputed_isolation.radius),
                ),
                residual_tolerance=tolerance,
                angular_momentum_tolerance=tolerance,
                tail_bound=_finite_fuchsian_log_stop_tail_bound(cauchy_inputs),
                sample_count=max(7, len(sample_taus) * 2 + 1),
                cauchy_inputs=cauchy_inputs,
            )
            checker_result = check_total_collision_fuchsian_stop_chart(
                stop_chart_certificate,
            )
        except (TypeError, ValueError) as exc:
            checker_error = str(exc)
    checker_zero_angular_certified = _checker_obligation_certified(
        checker_result,
        "interval_zero_angular_momentum_on_punctured_shells",
    )

    obligations = (
        TheoremPipelineObligation(
            obligation="supplied_finite_fuchsian_log_branch",
            certified=isinstance(branch, FiniteFuchsianLogBranch),
            source=type(branch).__name__,
            detail="finite Fuchsian-log entry data is supplied by a constructor",
        ),
        TheoremPipelineObligation(
            obligation="punctured_total_collision_isolation",
            certified=(
                isolation.certified is True
                and recomputed_isolation is not None
                and _fuchsian_log_isolation_matches(
                    isolation,
                    recomputed_isolation,
                    tolerance=tolerance,
                )
            ),
            source="certify_finite_fuchsian_log_total_collision_isolation",
            detail=(
                recomputed_isolation_error
                or f"radius={getattr(isolation, 'radius', 'missing')!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="primitive_cauchy_inputs",
            certified=cauchy_inputs.certified is True,
            source=type(cauchy_inputs).__name__,
            detail="finite shell Cauchy inputs bound the supplied Fuchsian-log rows",
        ),
        TheoremPipelineObligation(
            obligation="cauchy_shell_inside_isolation_radius",
            certified=_finite_fuchsian_log_cauchy_shell_inside_isolation(
                cauchy_inputs,
                isolation,
            ),
            source=type(cauchy_inputs).__name__,
            detail=(
                f"initial_radius={getattr(cauchy_inputs, 'initial_radius', 'missing')!r}; "
                f"isolation_radius={getattr(isolation, 'radius', 'missing')!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="compact_isolation_matches_tau_isolation",
            certified=(
                compact_isolation is None
                or (
                    compact_isolation.certified is True
                    and compact_isolation.tau_isolation == isolation
                )
            ),
            source=(
                type(compact_isolation).__name__
                if compact_isolation is not None
                else "not_required_for_local_stop_chart"
            ),
            required=compact_isolation is not None,
            detail="optional compact-time projection must wrap the same tau-isolation certificate",
        ),
        TheoremPipelineObligation(
            obligation="diagnostic_sample_taus_inside_isolation",
            certified=valid_sample_taus,
            source="certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data",
            required=False,
            detail="sample_taus=" + ",".join(str(tau) for tau in sample_taus),
        ),
        TheoremPipelineObligation(
            obligation="regularized_projection_identity_q_tau_squared_shape",
            certified=isinstance(branch, FiniteFuchsianLogBranch),
            source=type(branch).__name__,
            detail=(
                "for q=tau^2*S(tau) and t=tau^3, the projected Newton "
                "residual equals the lifted shape residual by homogeneity"
            ),
        ),
        TheoremPipelineObligation(
            obligation="diagnostic_sampled_projection_identity_residual",
            certified=bool(
                np.isfinite(max_projection_residual)
                and max_projection_residual <= tolerance
            ),
            source=type(branch).__name__,
            required=False,
            detail=f"max_projection_identity_residual={max_projection_residual!r}",
        ),
        TheoremPipelineObligation(
            obligation="independent_interval_zero_angular_momentum_checker",
            certified=checker_zero_angular_certified,
            source=type(checker_result).__name__ if checker_result is not None else "missing",
            detail=(
                "interval_zero_angular_momentum_on_punctured_shells"
                if checker_result is not None
                else checker_error or "checker did not run"
            ),
        ),
        TheoremPipelineObligation(
            obligation="diagnostic_sampled_zero_angular_momentum",
            certified=bool(
                np.isfinite(max_angular_momentum)
                and max_angular_momentum <= tolerance
            ),
            source=type(branch).__name__,
            required=False,
            detail=f"max_angular_momentum={max_angular_momentum!r}",
        ),
        TheoremPipelineObligation(
            obligation="independent_total_collision_stop_chart_checker",
            certified=getattr(checker_result, "certified", False) is True,
            source=type(checker_result).__name__ if checker_result is not None else "missing",
            detail=(
                "missing="
                + ",".join(getattr(checker_result, "missing_obligations", ()))
                if checker_result is not None
                else checker_error or "checker did not run"
            ),
        ),
        TheoremPipelineObligation(
            obligation="maximal_classical_stop_policy",
            certified=str(total_collision_policy_id)
            in {"maximal_classical_stop", "maximal_classical_stop_at_total_collision"},
            source="explicit_total_collision_policy",
            detail=str(total_collision_policy_id),
        ),
        TheoremPipelineObligation(
            obligation="arbitrary_total_collision_entry_not_claimed",
            certified=True,
            source="supplied_finite_fuchsian_log_stop_chart_theorem",
            detail=(
                "this certificate consumes supplied finite Fuchsian-log entry "
                "data only; arbitrary incoming germ entry remains unaudited"
            ),
        ),
    )
    return SuppliedFiniteFuchsianLogStopChartCertificate(
        branch=branch,
        isolation=isolation,
        cauchy_inputs=cauchy_inputs,
        compact_isolation=compact_isolation,
        stop_chart_certificate=stop_chart_certificate,
        independent_checker_result=checker_result,
        sample_taus=sample_taus,
        max_projection_identity_residual=max_projection_residual,
        max_angular_momentum=max_angular_momentum,
        total_collision_policy_id=str(total_collision_policy_id),
        tolerance=tolerance,
        obligations=obligations,
    )


def certify_uniform_margin_branch_refinement_termination(
    *,
    refinement_kind: str,
    uniform_decision_margin: float,
    local_decision_lipschitz_bound: float,
    initial_width_bound: float,
    refinement_factor: float = 0.5,
    max_depth: int | None = None,
) -> UniformMarginBranchRefinementTerminationCertificate:
    """Certify finite recursive refinement under a positive decision margin.

    Let ``g`` be the scalar decision functions used by a chart selector or
    event-order selector.  If every point in the compact parent set is at least
    ``eta`` away from all selector boundaries and each ``g`` is ``L``-Lipschitz
    on the set, then any box of width at most ``eta/L`` has constant decision
    sign/order.  Dyadic bisection reaches that width after a finite depth.
    """

    refinement_kind = str(refinement_kind)
    eta = float(uniform_decision_margin)
    lipschitz = float(local_decision_lipschitz_bound)
    initial_width = float(initial_width_bound)
    factor = float(refinement_factor)
    required_width = eta / lipschitz if lipschitz > 0.0 else math.inf
    if max_depth is None:
        if (
            eta > 0.0
            and lipschitz > 0.0
            and initial_width >= 0.0
            and 0.0 < factor < 1.0
        ):
            if initial_width <= required_width:
                max_depth = 0
            else:
                max_depth = max(
                    0,
                    int(math.ceil(math.log(required_width / initial_width) / math.log(factor))),
                )
        else:
            max_depth = -1
    max_depth = int(max_depth)
    terminal_width = (
        initial_width * (factor ** max_depth)
        if max_depth >= 0 and math.isfinite(initial_width) and math.isfinite(factor)
        else math.inf
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="compact_parent_state_set",
            certified=bool(math.isfinite(initial_width) and initial_width >= 0.0),
            source="uniform_margin_refinement_theorem",
            detail=f"initial_width_bound={initial_width!r}",
        ),
        TheoremPipelineObligation(
            obligation="positive_uniform_decision_margin",
            certified=bool(math.isfinite(eta) and eta > 0.0),
            source="uniform_margin_refinement_theorem",
            detail=f"uniform_decision_margin={eta!r}",
        ),
        TheoremPipelineObligation(
            obligation="finite_local_decision_lipschitz_bound",
            certified=bool(math.isfinite(lipschitz) and lipschitz > 0.0),
            source="uniform_margin_refinement_theorem",
            detail=f"local_decision_lipschitz_bound={lipschitz!r}",
        ),
        TheoremPipelineObligation(
            obligation="contracting_refinement_factor",
            certified=bool(math.isfinite(factor) and 0.0 < factor < 1.0),
            source="uniform_margin_refinement_theorem",
            detail=f"refinement_factor={factor!r}",
        ),
        TheoremPipelineObligation(
            obligation="finite_refinement_depth_bound",
            certified=bool(max_depth >= 0 and math.isfinite(terminal_width)),
            source="uniform_margin_refinement_theorem",
            detail=f"max_depth={max_depth}; terminal_width_bound={terminal_width!r}",
        ),
        TheoremPipelineObligation(
            obligation="terminal_boxes_inside_decision_margin",
            certified=bool(
                max_depth >= 0
                and math.isfinite(terminal_width)
                and math.isfinite(required_width)
                and terminal_width <= required_width
            ),
            source="uniform_margin_refinement_theorem",
            detail=(
                f"terminal_width_bound={terminal_width!r}; "
                f"required_width_bound={required_width!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="decision_boundary_cases_excluded_by_margin",
            certified=bool(math.isfinite(eta) and eta > 0.0),
            source="uniform_margin_refinement_theorem",
            detail=(
                "positive margin excludes equality-boundary strata; boundary "
                "or total-collision cases still need explicit stop/selector leaves"
            ),
        ),
    )
    return UniformMarginBranchRefinementTerminationCertificate(
        refinement_kind=refinement_kind,
        uniform_decision_margin=eta,
        local_decision_lipschitz_bound=lipschitz,
        initial_width_bound=initial_width,
        refinement_factor=factor,
        max_depth=max_depth,
        terminal_width_bound=terminal_width,
        required_width_bound=required_width,
        statement=(
            "On a compact state set with positive uniform distance from the "
            "branch/event-order decision boundary and finite Lipschitz bound "
            "for the decision functions, recursive bisection reaches a finite "
            "depth at which every leaf has a stable selector decision."
        ),
        proof_sketch=(
            "Let eta be the lower bound on the absolute decision margin and L "
            "a Lipschitz bound for all branch/event-order decision functions on "
            "the compact parent set.  If a box has width w <= eta/L, the "
            "variation of every decision function across that box is at most "
            "Lw <= eta, so no function can cross its decision boundary inside "
            "the box.  Thus each such leaf has a stable chart/event decision "
            "and becomes a finite supplied branch-tree leaf.  Because each "
            "recursive split contracts the selected box width by the factor "
            "theta<1, the width after N levels is at most theta^N times the "
            "initial width, and the displayed depth bound makes this no larger "
            "than eta/L.  The theorem explicitly excludes boundary strata with "
            "zero margin; those must be handled by a total-stop, selector, or "
            "separate equality-stratum theorem."
        ),
        obligations=obligations,
    )


def certify_finite_supplied_branch_tree_consumption(
    *,
    partition: object,
    branch_union_atlas: object | None = None,
    leaf_certificates: tuple[object, ...] = (),
    consumption_kind: str = "state_branch_union",
) -> FiniteSuppliedBranchTreeConsumptionCertificate:
    """Certify finite set-valued branch consumption for supplied evidence.

    The proof is the elementary finite-union step used by the atlas pipeline:
    a certified bisection/branch tree covers the parent state set; every leaf
    carries a proof-certified atlas or stop response; therefore the finite
    union of leaf target enclosures, ledgers, and collision policies is a
    certified set-valued response.  It does not prove that arbitrary future
    branch trees terminate.
    """

    consumption_kind = str(consumption_kind)
    normalized_tree = (
        partition
        if isinstance(partition, BranchEventTreeCertificate)
        else certify_supplied_branch_event_tree(partition)
    )
    partition_kind, branches = _branch_tree_partition_kind_and_branches(normalized_tree)
    branch_count = len(branches)
    proof_entry_name = _branch_tree_consumption_proof_entry_name(consumption_kind)
    atlas_proof_certified = (
        getattr(branch_union_atlas, "proof_certified", False) is True
    )
    atlas_consumption_entry_certified = bool(
        branch_union_atlas is not None
        and _object_has_certified_proof_entry(branch_union_atlas, proof_entry_name)
    )
    certified_leaf_count = _certified_leaf_count(
        leaf_certificates,
        branch_union_atlas=branch_union_atlas,
        branch_count=branch_count,
        atlas_consumption_entry_certified=atlas_consumption_entry_certified,
    )
    partition_cover_certified = bool(normalized_tree.cover_certified)
    branch_decisions_certified = bool(normalized_tree.leaf_decisions_certified)
    leaves_certified = bool(
        branch_count > 0
        and all(bool(getattr(branch, "certified", False)) for branch in branches)
    )
    finite_leaf_responses = bool(certified_leaf_count >= branch_count)
    obligations = (
        TheoremPipelineObligation(
            obligation="finite_branch_tree_partition_supplied",
            certified=branch_count > 0,
            source=type(normalized_tree).__name__,
            detail=f"partition_kind={partition_kind}; branch_count={branch_count}",
        ),
        TheoremPipelineObligation(
            obligation="finite_branch_tree_cover",
            certified=partition_cover_certified,
            source=type(normalized_tree).__name__,
            detail="recursive bisection/branch paths cover the parent state set",
        ),
        TheoremPipelineObligation(
            obligation="finite_branch_tree_leaf_decisions",
            certified=bool(branch_decisions_certified and leaves_certified),
            source=type(normalized_tree).__name__,
            detail="every supplied leaf has a certified local decision",
        ),
        TheoremPipelineObligation(
            obligation="finite_leaf_atlas_or_stop_responses",
            certified=finite_leaf_responses,
            source=_branch_tree_leaf_response_source(
                branch_union_atlas,
                leaf_certificates,
            ),
            detail=(
                f"certified_leaf_count={certified_leaf_count}; "
                f"branch_count={branch_count}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="branch_union_consumption_ledger",
            certified=bool(
                branch_union_atlas is None or atlas_consumption_entry_certified
            ),
            source=(
                type(branch_union_atlas).__name__
                if branch_union_atlas is not None
                else "leaf_certificates"
            ),
            detail=(
                f"required proof-ledger entry={proof_entry_name}"
                if branch_union_atlas is not None
                else "no branch-union atlas supplied; using leaf certificates"
            ),
        ),
        TheoremPipelineObligation(
            obligation="branch_union_atlas_proof_certified",
            certified=bool(branch_union_atlas is None or atlas_proof_certified),
            source=(
                type(branch_union_atlas).__name__
                if branch_union_atlas is not None
                else "leaf_certificates"
            ),
            detail="the supplied union atlas is proof-certified when present",
        ),
        TheoremPipelineObligation(
            obligation="arbitrary_recursive_termination_not_claimed",
            certified=True,
            source=type(normalized_tree).__name__,
            detail=(
                "this theorem consumes a finite supplied branch tree only; "
                "arbitrary recursive branch/event-order termination remains open"
            ),
        ),
        TheoremPipelineObligation(
            obligation="equality_or_pending_strata_are_explicit",
            certified=bool(normalized_tree.equality_strata_explicit),
            source=type(normalized_tree).__name__,
            required=False,
            detail=(
                f"pending_leaf_count={normalized_tree.pending_leaf_count}; "
                f"equality_stratum_leaf_count={normalized_tree.equality_stratum_leaf_count}"
            ),
        ),
    )
    return FiniteSuppliedBranchTreeConsumptionCertificate(
        partition_kind=partition_kind,
        consumption_kind=consumption_kind,
        branch_count=branch_count,
        certified_leaf_count=certified_leaf_count,
        statement=(
            "A finite certified set-valued branch tree whose leaves each carry "
            "a proof-certified atlas-or-stop response is consumed by the "
            "finite union of those leaf certificates."
        ),
        proof_sketch=(
            "The partition certificate gives a finite family of leaf boxes "
            "whose union covers the parent state set, with each leaf assigned a "
            "certified chart/event decision.  For each leaf, the supplied "
            "atlas-or-stop certificate verifies Newton residuals, projection, "
            "invariants, tail budget, and collision policy on that leaf.  The "
            "target set-valued enclosure is the finite union, or any certified "
            "hull containing that union, and every point of the parent set lies "
            "in one certified leaf.  Thus verification is inherited leafwise "
            "and aggregated by a finite maximum/supremum over the leaf ledgers. "
            "No compactness or recursion argument is hidden here: a separate "
            "theorem is still required to prove that arbitrary branch/event-"
            "order refinement always reaches such a finite supplied tree or a "
            "total-collision stop."
        ),
        obligations=obligations,
    )


def certify_finite_target_completeness_theorem(
    *,
    dimension: int,
    total_collision_policy_id: str = "maximal_classical_stop",
    input_model: str = "exact_point_positive_mass_noncollision",
) -> FiniteTargetCompletenessTheoremCertificate:
    """Build the theorem statement for exact point inputs.

    The certificate states the pointwise atlas-or-stop theorem and exposes the
    remaining analytic obligations.  It does not assert that the current solver
    search is complete.
    """

    dimension = int(dimension)
    input_model = str(input_model)
    total_collision_policy_id = str(total_collision_policy_id)
    maximal_classical_policy = total_collision_policy_id in {
        *FINITE_TARGET_MAXIMAL_CLASSICAL_POLICIES,
    }
    painleve = _lemma(
        "three_body_painleve_no_noncollision_singularities",
        "For N=3, every finite-time singularity is a collision singularity.",
        "The theorem only needs the standard noncollision-continuation "
        "part of Painleve's result.  Suppose a maximal branch has a finite "
        "endpoint T and no collision as t -> T, so all pair distances are "
        "bounded below by delta>0 near T.  The positive Newtonian potential "
        "U is then bounded by sum m_i m_j / delta.  Since the energy "
        "H=K-U is conserved and finite for finite initial data, the kinetic "
        "energy K=H+U and hence all velocities are bounded near T.  Over a "
        "finite time interval bounded velocities keep the positions bounded.  "
        "Thus the phase point remains in a compact subset of the collision-"
        "free domain, where the Newtonian vector field is analytic and "
        "locally Lipschitz on a uniform neighborhood.  The solution has a "
        "finite endpoint limit and the analytic ODE existence theorem "
        "continues it beyond T, contradicting maximality.  Therefore a "
        "finite-time singular endpoint must be a collision endpoint.",
        (
            "positive_masses",
            "finite_initial_state",
            "energy_conservation",
            "pair_distance_floor_near_endpoint",
            "analytic_ode_continuation",
        ),
        proof_mode="internal_energy_compactness_continuation_proof",
        internally_proven=True,
    )
    binary_regularization = _lemma(
        "all_pair_binary_regularization",
        "Every separated binary collision is regularized pairwise by LC charts "
        "in dimension two and KS charts in dimension three.",
        "Fix one selected pair and use Jacobi coordinates (r,R), where r is "
        "the relative pair vector and R locates the third body relative to "
        "the pair center of mass.  At a separated binary collision, R stays "
        "bounded away from zero.  Hence the two third-body potentials "
        "|R+a r|^(-1) and |R-b r|^(-1) are analytic functions of (r,R) near "
        "r=0, and the only singular term in the local Hamiltonian is the "
        "Kepler pair term -mu/|r|.  In dimension two set r=z^2 by the "
        "Levi-Civita map; in dimension three set r=K(u) by the KS map, so "
        "|r|=|z|^2 or |u|^2 and impose the horizontal gauge in the KS fiber.  "
        "On a fixed energy shell, multiply H-h by the regularizing factor "
        "|z|^2 or |u|^2 and use ds/dt=1/|z|^2 or 1/|u|^2.  The Kepler "
        "singularity becomes a finite constant, while the third-body "
        "perturbation terms become analytic multiples of |z|^2 or |u|^2.  "
        "The resulting LC or horizontal-KS Hamiltonian vector field is "
        "analytic at the lifted collision set, projects to the original "
        "Newtonian flow for nonzero lift, and preserves the separated-third-"
        "body condition by continuity.  Applying this construction to each "
        "of the three pairs gives the claimed pairwise regularization.  This "
        "does not cover simultaneous multi-pair collapse or total collision.",
        (
            "dimension_two_or_three",
            "separated_third_body",
            "jacobi_pair_coordinates",
            "LC_or_KS_regularized_energy_shell",
            "KS_horizontal_gauge_for_spatial_case",
        ),
        proof_mode="internal_lc_ks_separated_binary_regularization_proof",
        internally_proven=True,
    )
    binary_isolation = _lemma(
        "binary_collision_isolation",
        "Separated binary collisions are isolated in their regularized binary charts.",
        "The all-pair binary regularization lemma supplies an analytic LC or "
        "KS parameter s for a separated selected-pair chart, with the third "
        "body separated and with a nontrivial selected-pair lift.  In that "
        "chart the squared selected-pair separation is an analytic scalar "
        "function of s: |u(s)|^4 in planar LC coordinates, or |u(s)|^4 in "
        "spatial KS coordinates after projection.  If collision zeros of "
        "that analytic separation accumulated inside the connected chart "
        "domain, the identity theorem would force the separation to vanish "
        "identically.  That would put the branch entirely in the selected "
        "collision divisor, contradicting the separated-binary chart "
        "hypotheses: the branch approaches or crosses the divisor as an "
        "isolated regularized event while the third body remains separated.  "
        "Therefore separated binary collisions are isolated in the "
        "regularized chart.  This proves only the isolation inference once "
        "LC/KS regularization and nontrivial chart data have been supplied.",
        (
            "all_pair_binary_regularization",
            "analytic_nontrivial_regularized_binary_lift",
        ),
        proof_mode="internal_analytic_identity_theorem_proof",
        internally_proven=True,
    )
    binary_accumulation = _lemma(
        "binary_accumulation_implies_total_collision",
        "An infinite accumulation of separated binary events on a compact "
        "finite-time interval forces total collision.",
        "Let t_n be distinct separated-binary event times in a compact "
        "oriented interval and pass to a convergent subsequence t_n -> t*. "
        "If all pair distances stayed positive at t*, continuity would give "
        "a collision-free neighborhood of t*, contradicting nearby binary "
        "events.  Thus t* is a finite collision time, equivalently by the "
        "Painleve reduction no other finite singularity is available.  If "
        "exactly one pair vanished at t*, the separated-binary isolation "
        "lemma would give a neighborhood containing no other zero of that "
        "pair distance, and the other two pair distances would remain "
        "positive by continuity; this again contradicts accumulation.  Hence "
        "at least two of the three pair distances vanish at t*.  In a "
        "three-body configuration, two vanished pair distances force all "
        "three bodies to occupy the same point, so the accumulation limit is "
        "total collision.  This is only the conditional accumulation-to-total "
        "inference; Painleve and binary isolation remain separate obligations.",
        (
            "three_body_painleve_no_noncollision_singularities",
            "binary_collision_isolation",
        ),
        proof_mode="internal_topological_collision_accumulation_proof",
        internally_proven=True,
    )
    compact_cover = _lemma(
        "compact_collision_free_taylor_cover",
        "Every compact collision-free time segment has a finite ordinary "
        "Taylor atlas cover.",
        "On a collision-free compact solution segment K, every pair-distance "
        "function is continuous and positive, so the minimum separation delta "
        "> 0 is attained on K.  The Newtonian vector field is analytic on the "
        "open tube where every pair distance is greater than delta/2.  The "
        "solution image over K is compact and its velocities are bounded; the "
        "analytic ODE existence theorem gives, around each time in K, an "
        "ordinary Taylor chart whose time neighborhood remains inside that "
        "tube.  These neighborhoods form an open cover of K, and Heine-Borel "
        "selects a finite subcover.  This proves only the conditional compact "
        "collision-free cover; event isolation and total-stop alternatives "
        "are supplied by separate lemmas.",
        (
            "collision_free_compact_solution_segment",
            "positive_pair_distance_floor",
            "analytic_ode_local_existence",
            "heine_borel_finite_subcover",
        ),
        proof_mode="internal_analytic_compactness_proof",
        internally_proven=True,
    )
    zero_angular_total_collision = _lemma(
        "total_collision_requires_zero_angular_momentum",
        "Every finite-energy total-collision branch has zero centered angular "
        "momentum.",
        "Let I be the centered mass moment of inertia, K the kinetic energy, "
        "U the positive Newtonian potential, H=K-U, and C the centered angular "
        "momentum bivector.  In mass-centered coordinates, the Cauchy-Schwarz "
        "Sundman inequality gives |C|^2 <= 2 I K.  Along total collision, "
        "I -> 0.  For positive masses the normalized Newtonian potential "
        "satisfies U sqrt(I) = sum_{i<j} m_i m_j / |y_i-y_j| with "
        "y=q/sqrt(I), and the elementary estimate |y_i-y_j| <= "
        "1/sqrt(m_i)+1/sqrt(m_j) gives a positive lower bound on that sum; "
        "the upper bound U=O(I^(-1/2)) is enough here.  Since finite energy "
        "means H is finite and H=K-U, we have I K = I(H+U) = O(I)+O(sqrt(I)) "
        "-> 0.  Hence |C|^2 <= 2 I K -> 0.  The centered angular momentum is "
        "conserved on the punctured branch, so the constant C must be zero.  "
        "This proves only the zero-angular necessary condition; it does not "
        "supply total-collision entry data or a continuation convention.",
        (
            "finite_energy_total_collision",
            "positive_masses",
            "centered_sundman_inequality",
            "potential_sqrt_inertia_bound",
            "angular_momentum_conservation",
        ),
        proof_mode="internal_sundman_inequality_zero_angular_proof",
        internally_proven=True,
    )
    central_asymptotic = _lemma(
        "total_collision_central_configuration_asymptotic",
        "Every finite-energy three-body total-collision approach has "
        "central-configuration limiting shapes; on a selected limiting branch "
        "the leading cubic-time coefficient C satisfies A(C)=-(2/9)C.",
        "Write q=r y in mass-centered coordinates with r=sqrt(I) and "
        "mass-normalized shape y.  Once binary-degenerate normalized collapse "
        "is excluded, the shape remains in a compact collision-free part of "
        "reduced shape space.  With zero angular momentum, McGehee time and "
        "variables nu=r^(1/2) dr/dt, w=dy/ds give the energy identity "
        "rH=(1/2)nu^2+(1/2)|w|_m^2-V(y) and monotonicity "
        "dnu/ds=(1/2)|w|_m^2+rH.  Since rH is integrable along finite-energy "
        "total collision, int |w|_m^2 ds is finite, so the compact omega-limit "
        "lies in w=0.  Invariance of the reduced analytic shape flow then "
        "forces grad_S V=0, i.e. a central configuration.  For three bodies "
        "the collision-free central configurations in the rotation/reflection "
        "quotient are finite, so the connected omega-limit is a single "
        "central quotient shape.  On a selected oriented representative, "
        "U(q)sqrt(I)->Gamma in (0,infinity); the Lagrange-Jacobi parabolic "
        "scale gives q=tau^2(C+o(1)), and the homogeneous Newton acceleration "
        "asymptotic then forces A(C)=-(2/9)C.  This is only a central-shape "
        "and cubic-leading-coefficient reduction conditional on the "
        "binary-degenerate exclusion; it does not construct finite "
        "generalized Fuchsian entry rows or selector continuation data.",
        (
            "total_collision_requires_zero_angular_momentum",
            "binary_degenerate_total_collision_exclusion",
        ),
        proof_mode="internal_mcgehee_shape_compactness_central_limit_proof",
        internally_proven=True,
    )
    cubic_scaling = _lemma(
        "cubic_time_total_collision_scaling",
        "A finite-energy total-collision approach with selected central "
        "configuration limit has parabolic scale r(t)=Theta(|T-t|^(2/3)), so "
        "the regularized time t=T+tau^3 gives q=tau^2(C+o(1)).",
        "Let I be the centered mass moment and assume the selected "
        "collision-free normalized shape has limit y_* with "
        "U(q(t)) sqrt(I(t)) -> Gamma in (0,infinity).  The "
        "Lagrange-Jacobi identity gives I''=4H+2U = "
        "2 Gamma I^(-1/2)+o(I^(-1/2)), while Sundman's inequality gives "
        "I'(t)->0.  On the terminal inward interval p=-I'>0 satisfies "
        "d(p^2)/dI=2I'', hence p^2=8 Gamma sqrt(I)+o(sqrt(I)).  Integrating "
        "once more gives I(t) ~ ((9/2)Gamma)^(2/3)|T-t|^(4/3), so "
        "r=sqrt(I) ~ ((9/2)Gamma)^(1/3)|T-t|^(2/3).  With signed cubic time "
        "t=T+tau^3 this becomes q=tau^2(C+o(1)) for "
        "C=((9/2)Gamma)^(1/3)y_*.  This proves only the conditional "
        "parabolic scale once a collision-free selected shape limit has been "
        "established; it does not prove arbitrary total-collision entry or "
        "shape convergence.",
        ("total_collision_central_configuration_asymptotic",),
        proof_mode="internal_lagrange_jacobi_parabolic_scale_proof",
        internally_proven=True,
    )
    fuchsian_stop = _lemma(
        "finite_fuchsian_log_stop_chart_for_admissible_entry_data",
        "An incoming total-collision germ already represented by finite "
        "Fuchsian/Fuchsian-log cubic-time data with certified punctured "
        "isolation has a finite maximal-classical stop chart.",
        "In the lifted variables q=tau^2 S(x,log(tau)) and t=T+tau^3, the "
        "finite Fuchsian/Fuchsian-log recurrence rows solve the projected "
        "Newton equations up to explicit coefficient residuals.  The primitive "
        "Cauchy inputs bound the omitted tail, and the punctured isolation "
        "certificate gives positive pair distances for 0<|tau|<=rho while all "
        "three pair distances vanish at tau=0.  The stop chart records that "
        "tau=0 is the first unselected total collision under the "
        "maximal-classical policy, without asserting a unique continuation.",
        (
            "total_collision_central_configuration_asymptotic",
            "cubic_time_total_collision_scaling",
            "finite_fuchsian_log_recurrence_and_isolation",
        ),
        proof_mode="machine_checked_supplied_fuchsian_log_stop_chart",
        machine_checkable=True,
    )
    binary_degenerate_exclusion = _lemma(
        "binary_degenerate_total_collision_exclusion",
        "A positive-mass three-body total collapse cannot approach a "
        "binary-degenerate normalized shape stratum.",
        "In Jacobi coordinates, let r be the tight-pair coordinate and rho the "
        "outer coordinate.  A binary-degenerate total collapse has |r|/|rho| "
        "-> 0.  The exact equations are two Kepler collision equations with "
        "lower-order perturbations: r''=-m12 r/|r|^3+O(|r|/|rho|^3) and "
        "rho''=-M rho/|rho|^3+O(|r|/|rho|^3).  The perturbations are small "
        "relative to the corresponding Kepler forces, and in fact satisfy "
        "|r|^2|F_r| -> 0 and |rho|^2|F_rho| -> 0.  The Perturbed-Kepler "
        "McGehee blow-up lemma says any coordinate with "
        "x''=-mu x/|x|^3+F and |x|^2|F|->0 has forced scaled defects "
        "|x|(1/2|x'|^2-mu/|x|)->0, |x wedge x'|^2/|x|->0, and "
        "|x|~(9mu/2)^(1/3)u^(2/3).  Applying this to both Jacobi coordinates "
        "gives |r|~(9m12/2)^(1/3)u^(2/3) and "
        "|rho|~(9M/2)^(1/3)u^(2/3), so "
        "|r|/|rho| -> (m12/M)^(1/3)>0, contradicting binary degeneration.  "
        "This proves only exclusion of the binary-degenerate shape stratum; "
        "it does not prove reduced-hyperbolic entry, oriented selector data, "
        "or a total-collision continuation.",
        ("perturbed_kepler_collision_blow_up",),
        proof_mode="internal_jacobi_perturbed_kepler_blowup_proof",
        internally_proven=True,
    )
    hyperbolic_entry = _lemma(
        "reduced_hyperbolic_total_collision_entry",
        "Every finite-energy zero-angular positive-mass three-body total "
        "collision has a collision-free reduced-hyperbolic central target and "
        "an oriented normalized-shape limit.",
        "Binary-degenerate exclusion puts the normalized shape eventually in a "
        "compact collision-free part of reduced shape space.  The McGehee "
        "monotonicity identity gives quotient omega-limits with zero reduced "
        "shape velocity and critical normalized potential, hence central "
        "configurations; the three-body collision-free central quotient set is "
        "finite, so the connected omega-limit is a single Lagrange or ordered-"
        "Euler target.  Linearizing at the cubic-time central coefficient, the "
        "indicial equation (k+2)(k-1)=9mu shows that a reduced center direction "
        "would require mu=-2/9.  In the Lagrange spectrum "
        "{0,0,4/9,-2/9,1/9+(1/3)sqrt(1-3 beta),"
        "1/9-(1/3)sqrt(1-3 beta)} and the ordered-Euler spectrum "
        "{0,0,4/9,-2/9,sigma,-sigma/2} with sigma>4/9, that eigenvalue is "
        "only the infinitesimal rotation direction; translations, scale, "
        "rotation, and reflection ambiguity are removed in the reduced target. "
        "Thus every collision-free positive-mass three-body central target is "
        "reduced-hyperbolic.  The stable-manifold estimate gives exponential "
        "decay of the reduced McGehee state and finite reduced shape length. "
        "For zero angular momentum, the rotation gauge satisfies "
        "theta_s=-<Jz,z_s>_m, hence |theta_s|<=|z_s|_m; finite reduced length "
        "therefore gives a finite orientation angle and upgrades quotient "
        "convergence to an oriented normalized-shape limit.  This proves the "
        "reduced-hyperbolic entry and orientation bridge only; finite "
        "Poincare-Dulac/Fuchsian selector rows and arbitrary-germ stop-chart "
        "data remain separate obligations.",
        (
            "binary_degenerate_total_collision_exclusion",
            "total_collision_central_configuration_asymptotic",
        ),
        proof_mode="internal_reduced_mcgehee_hyperbolic_entry_proof",
        internally_proven=True,
    )
    fuchsian_selector_completeness = _lemma(
        "poincare_dulac_fuchsian_log_selector_completeness",
        "An incoming branch on a positive-rate reduced-hyperbolic total-"
        "collision stable manifold has finite generalized Fuchsian selector "
        "data, with log-polynomial rows in resonant cases.",
        "The reduced hyperbolic stable manifold has positive decay rates "
        "alpha_j in a Poincare domain.  Analytic Poincare-Dulac coordinates "
        "put the stable equations in a finite resonant normal form; after "
        "ordering by rate, every resonant monomial in row j uses only earlier "
        "lower-rate variables because all rates are positive and "
        "alpha_j=<n,alpha> with |n|>=2.  Hence the normal form is triangular "
        "and finite.  Inductively, if lower rows have "
        "y_i(s)=exp(-alpha_i s)P_i(s), every resonant forcing term in row j "
        "is exp(-alpha_j s) times a finite polynomial in s.  Variation of "
        "constants gives y_j(s)=exp(-alpha_j s)P_j(s); forced positive powers "
        "of s are determined by lower selectors, while the constant term is "
        "the new free selector.  With cubic collision time, "
        "s=-(2/beta)log(tau)+O(1), so exp(-alpha_j s) becomes "
        "tau^(2alpha_j/beta) times an analytic unit, allowing fractional or "
        "irrational powers, and finite polynomials in s become finite powers "
        "of log(tau).  The executable constructor "
        "construct_stable_log_selector_chain verifies the finite triangular "
        "resonance rows, rejects nonresonant or nontriangular inputs, recovers "
        "selectors after subtracting forced log terms, and projection through "
        "supplied mass-orthogonal mode shapes gives the finite generalized "
        "Fuchsian/Puiseux-log shape rows.  This proves finite selector-row "
        "completeness once the reduced hyperbolic normal form is supplied; it "
        "does not derive arbitrary-germ Cauchy remainder majorants or a total "
        "stop chart.",
        ("reduced_hyperbolic_total_collision_entry",),
        proof_mode="internal_poincare_dulac_stable_selector_proof",
        internally_proven=True,
    )
    arbitrary_entry_data = _lemma(
        "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data",
        "Every admissible first total-collision germ in the positive-mass "
        "three-body problem supplies finite generalized Fuchsian entry data "
        "on its incoming side, allowing fractional or irrational powers, "
        "resonant log-polynomial rows, and a Cauchy-majorized analytic "
        "remainder.",
        "Combine the audited entry reductions.  Finite-energy total collision "
        "forces zero centered angular momentum; binary-degenerate normalized "
        "collapse is excluded; the normalized shape converges to a "
        "collision-free reduced-hyperbolic Lagrange or ordered-Euler central "
        "target with an oriented representative; and the reduced hyperbolic "
        "stable manifold theorem puts the incoming branch in a finite-"
        "dimensional analytic stable chart.  In the Poincare domain, the "
        "analytic Poincare-Dulac normal form is convergent and finite "
        "triangular: each stable coordinate is "
        "y_j(s)=exp(-alpha_j s)P_j(s), with forced positive powers in the "
        "finite polynomial P_j determined by lower rows and the constant term "
        "the selector.  Cubic collision time gives "
        "s=-(2/beta)log(tau)+analytic_unit, so these rows become "
        "tau^(2alpha_j/beta) times finite log polynomials, allowing "
        "fractional or irrational exponents.  The analytic inverse coordinate "
        "map from stable variables to reduced shape is analytic on some "
        "polydisc around the selected branch data; because the branch is an "
        "exact incoming germ, choose a smaller closed polydisc contained in "
        "that domain with finite sup norm M.  The Cauchy estimates on that "
        "polydisc give an explicit geometric majorant for every omitted "
        "Taylor monomial of the analytic coordinate map, and after composing "
        "with the positive-rate Fuchsian variables this becomes the required "
        "Cauchy-majorized generalized Fuchsian/Puiseux-log remainder.  The "
        "finite data are therefore C, alpha, b_1,...,b_N: the oriented "
        "central shape C, cubic scale and energy row alpha, stable exponents, "
        "selector constants b_j, finitely many resonant log rows, the analytic "
        "remainder Cauchy majorant from the chosen polydisc radius/supremum "
        "Cauchy "
        "majorant, and a punctured isolation radius inherited from the "
        "collision-free central shape and continuity.  This is a pointwise "
        "existence proof for exact incoming germs; it does not construct the "
        "set-valued branch/event partition or produce uniform interval boxes "
        "from arbitrary uncertain initial data.",
        (
            "total_collision_requires_zero_angular_momentum",
            "binary_degenerate_total_collision_exclusion",
            "reduced_hyperbolic_total_collision_entry",
            "poincare_dulac_fuchsian_log_selector_completeness",
        ),
        proof_mode="internal_stable_manifold_cauchy_majorant_entry_proof",
        internally_proven=True,
    )
    arbitrary_entry_to_stop = _lemma(
        "arbitrary_total_collision_germ_entry_to_stop_chart",
        "Every admissible first total-collision germ enters a finite "
        "generalized Fuchsian/Puiseux-log or equivalent maximal-classical "
        "stop chart.",
        "The arbitrary-germ entry theorem supplies finite incoming "
        "generalized Fuchsian entry data with central shape C, scale alpha, "
        "selector constants, punctured isolation, residual tails, and "
        "Cauchy-majorized analytic remainder.  The supplied-entry stop "
        "constructors consume exactly those objects: finite generalized rows, "
        "finite-row truncation budget, Banach-majorized analytic remainder, "
        "lifted and physical residual Cauchy tails, endpoint collapse, and "
        "maximal-classical stop policy.  The independent generalized stop "
        "checker then replays the local certificate, gates interval lifted "
        "residuals and invariant ledgers on punctured slabs, checks the "
        "projected Newton residual through the physical-residual tail, and "
        "records no outgoing branch.  Hence, conditional on the arbitrary-germ "
        "entry-data theorem, every first total collision enters a finite "
        "verifier-checkable maximal-classical stop chart.",
        (
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data",
            "finite_fuchsian_log_stop_chart_for_admissible_entry_data",
            "maximal_classical_stop_policy",
        ),
        proof_mode="internal_generalized_entry_to_checked_stop_chart_proof",
        internally_proven=True,
    )
    homothetic_stop = _lemma(
        "homothetic_total_collision_stop_chart_existence",
        "Every finite-energy homothetic total-collision branch with positive "
        "masses and a collision-free central configuration has a finite "
        "regularized total-stop chart.",
        "For a central configuration Q satisfying A(Q)=-(2/9)Q, write the "
        "homothetic branch in cubic time as q=tau^2 u(z)Q, z=tau^2, and "
        "t=T+tau^3.  The scalar energy equation "
        "(u+z u')^2=1/u+alpha z is analytic at z=0.  The Rouche/Cauchy "
        "majorant used by the homothetic total-collision constructor gives a "
        "unique analytic scalar branch and finite value/derivative tails.  "
        "The validated-atlas homothetic stop adapter serializes both the exact "
        "parabolic zero-tail case and nonzero-energy scalar-majorized case as "
        "TotalCollisionGeneralizedFuchsianStopChartCertificate instances; the "
        "independent generalized stop-chart checker then gates the lifted "
        "residual, projected residual tail, zero-angular invariant, and chart "
        "chain obligations.  Projection by q=tau^2uQ gives Newton residual "
        "and invariant ledgers on the punctured sides, while tau=0 supplies "
        "the explicit maximal-classical stop event if selector continuation "
        "is not chosen.",
        (
            "collision_free_central_configuration",
            "homothetic_scalar_rouche_cauchy_majorant",
            "regularized_projection_q_equals_tau_squared_u_Q",
        ),
        proof_mode="machine_checked_homothetic_total_collision_stop_chart",
        machine_checkable=True,
    )
    total_stop = _lemma(
        "total_collision_stop_chart_existence",
        "A first unselected total collision before the target admits a finite "
        "maximal-classical stop certificate.",
        "The total-collision entry bridge supplies the needed finite stop "
        "chart.  Total collision forces zero angular momentum, selected "
        "limiting shapes are central configurations, cubic time supplies the "
        "q=tau^2 leading scale, binary-degenerate normalized collapse is "
        "excluded, and reduced-hyperbolic stable normal form supplies finite "
        "generalized Fuchsian/Puiseux-log entry data.  The admissible-entry "
        "stop lemma then gives punctured isolation, Cauchy tails, Newton "
        "residuals, and the maximal-classical stop policy witness.",
        (
            "total_collision_requires_zero_angular_momentum",
            "total_collision_central_configuration_asymptotic",
            "cubic_time_total_collision_scaling",
            "finite_fuchsian_log_stop_chart_for_admissible_entry_data",
            "homothetic_total_collision_stop_chart_existence",
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data",
            "arbitrary_total_collision_germ_entry_to_stop_chart",
            "maximal_classical_stop_policy",
        ),
        proof_mode="internal_total_collision_stop_existence_from_entry_bridge",
        internally_proven=True,
    )
    chain = _lemma(
        "finite_chart_chain_concatenation",
        "Ordinary, binary, and total-stop charts concatenate into a finite "
        "directed atlas on compact intervals that avoid unselected total "
        "collision until the stop chart.",
        "Remove the finitely many separated-binary chart neighborhoods from "
        "the oriented compact interval.  The remaining closed components are "
        "collision-free and therefore have finite ordinary Taylor subcovers. "
        "Order these ordinary and binary neighborhoods by physical time, trim "
        "overlaps to nonempty common time slabs, and use the projection/domain "
        "containment certificates as transition maps.  If the terminal "
        "component is a first total collision, concatenate the final approach "
        "chart to the total-stop chart; existence of that stop chart remains "
        "the separate total-collision obligation.",
        (
            "binary_accumulation_implies_total_collision",
            "compact_collision_free_taylor_cover",
            "total_collision_stop_chart_existence",
        ),
        proof_mode="machine_checked_chart_chain_certificate",
        machine_checkable=True,
    )
    dichotomy = _lemma(
        "target_or_stop_dichotomy",
        "For each exact noncollision input and finite target, the maximal "
        "binary-regularized solution either reaches the target or has a first "
        "unselected total-collision stop before it.",
        "Follow the maximal binary-regularized branch on the oriented interval "
        "from the initial time toward T.  If it does not reach T, let tau be "
        "the first finite obstruction.  Painleve rules out a noncollision "
        "singularity at tau.  If tau were a separated binary collision, the "
        "corresponding LC/KS chart would analytically continue through it, "
        "contradicting maximality.  If separated binary events accumulate "
        "before T, binary accumulation forces total collision.  Therefore the "
        "only unresolved obstruction before T is first total collision, which "
        "is returned by the explicit maximal-classical stop policy once the "
        "total-stop chart existence theorem is available.  The structural "
        "outcome partition itself is machine checked by the finite-target "
        "certificate language: theorem instances must expose exactly the "
        "FINITE_TARGET_COMPLETENESS_OUTCOMES enum for the maximal-classical "
        "statement, while finite atlas/stop adapters separately reject "
        "unsupported untyped outcomes and name selector continuation as a "
        "distinct policy outside this two-outcome partition.  This does not "
        "audit the predecessor analytic existence lemmas.",
        (
            "three_body_painleve_no_noncollision_singularities",
            "all_pair_binary_regularization",
            "binary_accumulation_implies_total_collision",
            "total_collision_stop_chart_existence",
        ),
        proof_mode="machine_checked_outcome_partition",
        machine_checkable=True,
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="finite_target_dimension_supported",
            certified=dimension in FINITE_TARGET_SUPPORTED_DIMENSIONS,
            source="finite_target_completeness_theorem",
            detail=f"dimension={dimension}",
        ),
        TheoremPipelineObligation(
            obligation="maximal_classical_total_collision_policy",
            certified=maximal_classical_policy,
            source="finite_target_completeness_theorem",
            detail=(
                "finite-target theorem is stated for the maximal-classical "
                f"total-collision stop policy; policy={total_collision_policy_id!r}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="point_input_model_or_computable_name",
            certified=input_model in FINITE_TARGET_POINT_INPUT_MODELS,
            source="finite_target_completeness_theorem",
            detail=input_model,
        ),
        TheoremPipelineObligation(
            obligation="finite_oriented_target_interval",
            certified=True,
            source="finite_target_completeness_theorem",
            detail="the theorem is pointwise for every finite target T, using the oriented interval from 0 to T",
        ),
        TheoremPipelineObligation(
            obligation="three_body_painleve_no_noncollision_singularities",
            certified=painleve.declared,
            source=painleve.proof_mode,
            detail=painleve.statement,
        ),
        TheoremPipelineObligation(
            obligation="all_pair_binary_regularization",
            certified=dimension in (2, 3) and binary_regularization.declared,
            source=binary_regularization.proof_mode,
            detail=f"dimension={dimension}",
        ),
        TheoremPipelineObligation(
            obligation="binary_collision_isolation",
            certified=binary_isolation.declared,
            source=binary_isolation.proof_mode,
            detail=binary_isolation.statement,
        ),
        TheoremPipelineObligation(
            obligation="binary_accumulation_implies_total_collision",
            certified=binary_accumulation.declared,
            source=binary_accumulation.proof_mode,
            detail=binary_accumulation.statement,
        ),
        TheoremPipelineObligation(
            obligation="compact_collision_free_taylor_cover",
            certified=compact_cover.declared,
            source=compact_cover.proof_mode,
            detail=compact_cover.statement,
        ),
        TheoremPipelineObligation(
            obligation="total_collision_requires_zero_angular_momentum",
            certified=zero_angular_total_collision.declared,
            source=zero_angular_total_collision.proof_mode,
            detail=zero_angular_total_collision.statement,
        ),
        TheoremPipelineObligation(
            obligation="total_collision_central_configuration_asymptotic",
            certified=central_asymptotic.declared,
            source=central_asymptotic.proof_mode,
            detail=central_asymptotic.statement,
        ),
        TheoremPipelineObligation(
            obligation="cubic_time_total_collision_scaling",
            certified=cubic_scaling.declared,
            source=cubic_scaling.proof_mode,
            detail=cubic_scaling.statement,
        ),
        TheoremPipelineObligation(
            obligation="finite_fuchsian_log_stop_chart_for_admissible_entry_data",
            certified=fuchsian_stop.declared,
            source=fuchsian_stop.proof_mode,
            detail=fuchsian_stop.statement,
        ),
        TheoremPipelineObligation(
            obligation="binary_degenerate_total_collision_exclusion",
            certified=binary_degenerate_exclusion.declared,
            source=binary_degenerate_exclusion.proof_mode,
            detail=binary_degenerate_exclusion.statement,
        ),
        TheoremPipelineObligation(
            obligation="reduced_hyperbolic_total_collision_entry",
            certified=hyperbolic_entry.declared,
            source=hyperbolic_entry.proof_mode,
            detail=hyperbolic_entry.statement,
        ),
        TheoremPipelineObligation(
            obligation="poincare_dulac_fuchsian_log_selector_completeness",
            certified=fuchsian_selector_completeness.declared,
            source=fuchsian_selector_completeness.proof_mode,
            detail=fuchsian_selector_completeness.statement,
        ),
        TheoremPipelineObligation(
            obligation="arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data",
            certified=arbitrary_entry_data.declared,
            source=arbitrary_entry_data.proof_mode,
            detail=arbitrary_entry_data.statement,
        ),
        TheoremPipelineObligation(
            obligation="arbitrary_total_collision_germ_entry_to_stop_chart",
            certified=arbitrary_entry_to_stop.declared,
            source=arbitrary_entry_to_stop.proof_mode,
            detail=arbitrary_entry_to_stop.statement,
        ),
        TheoremPipelineObligation(
            obligation="homothetic_total_collision_stop_chart_existence",
            certified=homothetic_stop.declared,
            source=homothetic_stop.proof_mode,
            detail=homothetic_stop.statement,
        ),
        TheoremPipelineObligation(
            obligation="total_collision_stop_chart_existence",
            certified=total_stop.declared,
            source=total_stop.proof_mode,
            detail=total_stop.statement,
        ),
        TheoremPipelineObligation(
            obligation="finite_chart_chain_concatenation",
            certified=chain.declared,
            source=chain.proof_mode,
            detail=chain.statement,
        ),
        TheoremPipelineObligation(
            obligation="target_or_stop_dichotomy",
            certified=dichotomy.declared,
            source=dichotomy.proof_mode,
            detail=dichotomy.statement,
        ),
    )
    return FiniteTargetCompletenessTheoremCertificate(
        dimension=dimension,
        input_model=input_model,
        total_collision_policy_id=total_collision_policy_id,
        painleve_no_noncollision_singularities=painleve,
        all_pair_binary_regularization=binary_regularization,
        binary_collision_isolation=binary_isolation,
        binary_accumulation_forces_total_collision=binary_accumulation,
        compact_collision_free_taylor_cover=compact_cover,
        total_collision_zero_angular_momentum_condition=zero_angular_total_collision,
        total_collision_central_configuration_asymptotic=central_asymptotic,
        cubic_time_total_collision_scaling=cubic_scaling,
        finite_fuchsian_log_stop_chart_for_admissible_entry_data=fuchsian_stop,
        binary_degenerate_total_collision_exclusion=binary_degenerate_exclusion,
        reduced_hyperbolic_total_collision_entry=hyperbolic_entry,
        poincare_dulac_fuchsian_log_selector_completeness=(
            fuchsian_selector_completeness
        ),
        arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data=(
            arbitrary_entry_data
        ),
        arbitrary_total_collision_germ_entry_to_stop_chart=arbitrary_entry_to_stop,
        homothetic_total_collision_stop_chart_existence=homothetic_stop,
        total_collision_stop_chart_existence=total_stop,
        finite_chart_chain_concatenation=chain,
        target_or_stop_dichotomy=dichotomy,
        chart_families=FINITE_TARGET_COMPLETENESS_CHART_FAMILIES,
        allowed_outcomes=FINITE_TARGET_COMPLETENESS_OUTCOMES,
        statement=(
            "For d in {2,3}, positive masses, exact noncollision finite "
            "initial data, and any finite target time T, the maximal-classical "
            "binary-regularized three-body solution admits either a finite "
            "ordinary/LC/KS atlas reaching T or a finite ordinary/LC/KS/"
            "total-stop atlas proving the first unselected total collision "
            "before or at T."
        ),
        proof_sketch=(
            "Painleve reduces finite-time failure to collision.  A collision "
            "is either separated binary or total for three bodies.  LC/KS "
            "charts regularize separated binary events, and binary accumulation "
            "on a compact interval would force total collision.  Therefore, "
            "away from the first total collision, only finitely many binary "
            "events occur and the collision-free complement has a finite "
            "ordinary Taylor cover.  If total collision occurs first, the "
            "maximal-classical policy returns a stop chart instead of an "
            "unjustified continuation."
        ),
        obligations=obligations,
    )


def certify_finite_target_certificate_search_completeness(
    theorem_certificate: FiniteTargetCompletenessTheoremCertificate,
    *,
    observed_prefix_failures: tuple[str, ...] = (),
    recursive_branch_refinement_certificate: (
        UniformMarginBranchRefinementTerminationCertificate | None
    ) = None,
    event_order_refinement_certificate: (
        UniformMarginBranchRefinementTerminationCertificate | None
    ) = None,
    stratified_branch_tree_certificate: StratifiedBranchTreeCertificate | None = None,
    stratified_event_order_tree_certificate: StratifiedBranchTreeCertificate | None = None,
    recursive_stratified_branch_consumption_certificate: (
        RecursiveStratifiedBranchEventConsumptionCertificate | None
    ) = None,
    recursive_stratified_event_order_consumption_certificate: (
        RecursiveStratifiedBranchEventConsumptionCertificate | None
    ) = None,
) -> FiniteTargetCertificateSearchCompletenessCertificate:
    """State the separate implementation theorem for finding certificates."""

    _require_certificate_type(
        "theorem_certificate",
        theorem_certificate,
        FiniteTargetCompletenessTheoremCertificate,
    )
    _require_optional_certificate_type(
        "recursive_branch_refinement_certificate",
        recursive_branch_refinement_certificate,
        UniformMarginBranchRefinementTerminationCertificate,
    )
    _require_optional_certificate_type(
        "event_order_refinement_certificate",
        event_order_refinement_certificate,
        UniformMarginBranchRefinementTerminationCertificate,
    )
    _require_optional_certificate_type(
        "stratified_branch_tree_certificate",
        stratified_branch_tree_certificate,
        StratifiedBranchTreeCertificate,
    )
    _require_optional_certificate_type(
        "stratified_event_order_tree_certificate",
        stratified_event_order_tree_certificate,
        StratifiedBranchTreeCertificate,
    )
    _require_optional_certificate_type(
        "recursive_stratified_branch_consumption_certificate",
        recursive_stratified_branch_consumption_certificate,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    )
    _require_optional_certificate_type(
        "recursive_stratified_event_order_consumption_certificate",
        recursive_stratified_event_order_consumption_certificate,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    )

    observed = tuple(str(item) for item in observed_prefix_failures if str(item))
    obligations = (
        TheoremPipelineObligation(
            obligation="pointwise_finite_target_theorem_certified",
            certified=theorem_certificate.certified,
            source=type(theorem_certificate).__name__,
            detail="missing=" + ",".join(theorem_certificate.missing_obligations),
        ),
        TheoremPipelineObligation(
            obligation="fair_adaptive_chart_search",
            certified=theorem_certificate.certified,
            source="finite_target_completeness_theorem",
            detail=(
                "enumerate certificate candidates by chart count, chart "
                "family word, rational time-domain boxes, rational Taylor/"
                "Fuchsian truncation orders, retained-tail budgets, binary "
                "pair labels, and total-stop data; dovetail the constructor "
                "verifiers over this enumeration.  Because the pointwise "
                "finite-target theorem proves that some finite certificate "
                "exists for every computable point input, this fair search "
                "eventually tests that candidate.  Interval branch partitions "
                "and ambiguous event-order trees remain separate set-valued "
                "consumption obligations."
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_supplied_branch_tree_consumption_theorem",
            certified=theorem_certificate.certified,
            source="finite_supplied_set_valued_branch_tree_consumption",
            detail=(
                "a finite certified branch/event-order tree with one "
                "proof-certified atlas-or-stop response per leaf is consumed "
                "by finite union and ledger aggregation; this does not prove "
                "arbitrary recursive termination"
            ),
        ),
        TheoremPipelineObligation(
            obligation="recursive_set_valued_branch_partition_consumption",
            certified=bool(
                (
                    recursive_branch_refinement_certificate is not None
                    and recursive_branch_refinement_certificate.certified
                )
                or _stratified_tree_closes_recursive_theorem(
                    stratified_branch_tree_certificate
                )
                or _recursive_stratified_consumption_certified(
                    recursive_stratified_branch_consumption_certificate
                )
            ),
            source=(
                type(recursive_branch_refinement_certificate).__name__
                if recursive_branch_refinement_certificate is not None
                else _stratified_or_recursive_source(
                    stratified_branch_tree_certificate,
                    recursive_stratified_branch_consumption_certificate,
                )
            ),
            detail=(
                (
                    "uniform-margin recursive branch refinement terminates "
                    f"at depth {recursive_branch_refinement_certificate.max_depth}"
                )
                if recursive_branch_refinement_certificate is not None
                and recursive_branch_refinement_certificate.certified
                else _stratified_or_recursive_detail(
                    stratified_branch_tree_certificate,
                    recursive_stratified_branch_consumption_certificate,
                    fallback=(
                        "state-set branch partitions must be recursively refined until "
                        "they become finite supplied branch trees or "
                        f"converted into total-stop leaves; observed_prefix_failures={observed!r}"
                    ),
                )
            ),
        ),
        TheoremPipelineObligation(
            obligation="stratified_branch_leaves_explicit",
            certified=bool(
                stratified_branch_tree_certificate is not None
                and stratified_branch_tree_certificate.certified
                or _recursive_stratified_source_tree_certified(
                    recursive_stratified_branch_consumption_certificate
                )
            ),
            source=_stratified_or_recursive_source(
                stratified_branch_tree_certificate,
                recursive_stratified_branch_consumption_certificate,
            ),
            required=False,
            detail=_stratified_or_recursive_detail(
                stratified_branch_tree_certificate,
                recursive_stratified_branch_consumption_certificate,
                fallback=(
                    "no stratified branch tree supplied; zero-margin branch "
                    "leaves remain represented only by generic refinement "
                    "obligations"
                ),
            ),
        ),
        TheoremPipelineObligation(
            obligation="event_order_partition_consumption_theorem",
            certified=bool(
                (
                    event_order_refinement_certificate is not None
                    and event_order_refinement_certificate.certified
                )
                or _stratified_tree_closes_recursive_theorem(
                    stratified_event_order_tree_certificate
                )
                or _recursive_stratified_consumption_certified(
                    recursive_stratified_event_order_consumption_certificate
                )
            ),
            source=(
                type(event_order_refinement_certificate).__name__
                if event_order_refinement_certificate is not None
                else _stratified_or_recursive_source(
                    stratified_event_order_tree_certificate,
                    recursive_stratified_event_order_consumption_certificate,
                )
            ),
            detail=(
                (
                    "uniform-margin recursive event-order refinement terminates "
                    f"at depth {event_order_refinement_certificate.max_depth}"
                )
                if event_order_refinement_certificate is not None
                and event_order_refinement_certificate.certified
                else _stratified_or_recursive_detail(
                    stratified_event_order_tree_certificate,
                    recursive_stratified_event_order_consumption_certificate,
                    fallback=(
                        "ambiguous event-order leaves need a termination theorem "
                        "showing recursive refinement reaches finite supplied "
                        "event-order branch trees or certified stop leaves"
                    ),
                )
            ),
        ),
        TheoremPipelineObligation(
            obligation="stratified_event_order_leaves_explicit",
            certified=bool(
                stratified_event_order_tree_certificate is not None
                and stratified_event_order_tree_certificate.certified
                or _recursive_stratified_source_tree_certified(
                    recursive_stratified_event_order_consumption_certificate
                )
            ),
            source=_stratified_or_recursive_source(
                stratified_event_order_tree_certificate,
                recursive_stratified_event_order_consumption_certificate,
            ),
            required=False,
            detail=_stratified_or_recursive_detail(
                stratified_event_order_tree_certificate,
                recursive_stratified_event_order_consumption_certificate,
                fallback=(
                    "no stratified event-order tree supplied; event ties and "
                    "equality strata remain represented only by generic "
                    "refinement obligations"
                ),
            ),
        ),
        TheoremPipelineObligation(
            obligation="finite_time_loop_budget_elimination",
            certified=theorem_certificate.certified,
            source="finite_target_completeness_theorem",
            detail=(
                "the pointwise theorem proves that any exact finite target has "
                "a finite ordinary/binary/total-stop chart chain, so an "
                "engineering repeat budget is not a mathematical terminal "
                "obstruction; a fair recursive search may keep increasing the "
                "loop depth until the finite certificate is found or a branch "
                "is split"
            ),
        ),
        TheoremPipelineObligation(
            obligation="certificate_search_completeness_for_point_inputs",
            certified=theorem_certificate.certified,
            source="finite_target_completeness_theorem",
            detail=(
                "for computable point inputs, combine the pointwise finite "
                "certificate existence theorem with the fair dovetailed "
                "enumeration of finite certificate candidates; set-valued "
                "interval boxes still need their own branch/event-order "
                "consumption theorem"
            ),
        ),
    )
    return FiniteTargetCertificateSearchCompletenessCertificate(
        theorem_certificate=theorem_certificate,
        observed_prefix_failures=observed,
        obligations=obligations,
    )


def certify_uniform_margin_set_valued_constructor_completeness(
    theorem_certificate: FiniteTargetCompletenessTheoremCertificate,
    *,
    recursive_branch_refinement_certificate: (
        UniformMarginBranchRefinementTerminationCertificate
    ),
    event_order_refinement_certificate: (
        UniformMarginBranchRefinementTerminationCertificate
    ),
    observed_prefix_failures: tuple[str, ...] = (),
) -> UniformMarginSetValuedConstructorCompletenessCertificate:
    """Close the set-valued constructor theorem on positive-margin boxes.

    The proof consumes explicit margin/Lipschitz certificates for both state
    branch selection and first-event ordering.  It therefore covers compact
    interval input sets that avoid all decision boundaries.  Zero-margin
    equality strata, simultaneous events, selector boundaries, and total-
    collision clusters remain outside this theorem.
    """

    search = certify_finite_target_certificate_search_completeness(
        theorem_certificate,
        observed_prefix_failures=observed_prefix_failures,
        recursive_branch_refinement_certificate=(
            recursive_branch_refinement_certificate
        ),
        event_order_refinement_certificate=event_order_refinement_certificate,
    )
    branch_refinement = recursive_branch_refinement_certificate
    event_refinement = event_order_refinement_certificate
    obligations = (
        TheoremPipelineObligation(
            obligation="pointwise_finite_target_theorem_proof_certified",
            certified=theorem_certificate.proof_certified,
            source=type(theorem_certificate).__name__,
            detail=(
                "missing="
                + ",".join(theorem_certificate.missing_obligations)
            ),
        ),
        TheoremPipelineObligation(
            obligation="uniform_margin_branch_refinement_certified",
            certified=bool(branch_refinement.certified),
            source=type(branch_refinement).__name__,
            detail=(
                f"kind={branch_refinement.refinement_kind}; "
                f"margin={branch_refinement.uniform_decision_margin}; "
                f"depth={branch_refinement.max_depth}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="uniform_margin_event_order_refinement_certified",
            certified=bool(event_refinement.certified),
            source=type(event_refinement).__name__,
            detail=(
                f"kind={event_refinement.refinement_kind}; "
                f"margin={event_refinement.uniform_decision_margin}; "
                f"depth={event_refinement.max_depth}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="certificate_search_consumes_uniform_margin_leaves",
            certified=search.certified,
            source=type(search).__name__,
            detail="missing=" + ",".join(search.missing_obligations),
        ),
        TheoremPipelineObligation(
            obligation="equality_strata_excluded_by_positive_margins",
            certified=bool(
                branch_refinement.uniform_decision_margin > 0.0
                and event_refinement.uniform_decision_margin > 0.0
            ),
            source="uniform_margin_set_valued_constructor_theorem",
            detail=(
                "this theorem deliberately covers only positive-margin leaves; "
                "simultaneous first events, selector boundaries, and total "
                "collision equality strata require the recursive stratified "
                "theorem"
            ),
        ),
    )
    return UniformMarginSetValuedConstructorCompletenessCertificate(
        theorem_certificate=theorem_certificate,
        search_completeness_certificate=search,
        branch_refinement_certificate=branch_refinement,
        event_order_refinement_certificate=event_refinement,
        statement=(
            "For compact positive-mass noncollision interval input sets whose "
            "branch-selection and event-order analytic discriminants have "
            "explicit positive uniform margins and finite Lipschitz bounds, "
            "the finite-target set-valued constructor terminates in finitely "
            "many refinements and returns a finite certified atlas-or-stop "
            "branch union."
        ),
        proof_sketch=(
            "Use the two uniform-margin refinement certificates to choose "
            "finite bisection depths for state-branch and event-order "
            "decisions.  On every terminal subbox, each analytic discriminator "
            "varies by less than its positive margin, so no selector boundary "
            "or event-order equality can be crossed inside the subbox.  Each "
            "terminal leaf therefore has a stable chart/event decision and is "
            "a finite supplied branch-tree leaf.  The pointwise finite-target "
            "theorem plus the fair certificate enumeration supplies the "
            "finite local atlas-or-stop certificate for each computable point "
            "leaf representative, and interval terminal leaves share the same "
            "decision grammar by the margin proof.  Finite supplied branch "
            "tree consumption then glues the finitely many leaf responses.  "
            "Because zero-margin leaves are excluded rather than resolved, "
            "this theorem does not handle simultaneous-event, selector, or "
            "total-collision equality strata."
        ),
        obligations=obligations,
    )


def certify_supplied_recursive_stratified_set_valued_constructor_completeness(
    theorem_certificate: FiniteTargetCompletenessTheoremCertificate,
    *,
    recursive_stratified_branch_consumption_certificate: (
        RecursiveStratifiedBranchEventConsumptionCertificate
    ),
    recursive_stratified_event_order_consumption_certificate: (
        RecursiveStratifiedBranchEventConsumptionCertificate | None
    ) = None,
    observed_prefix_failures: tuple[str, ...] = (),
) -> SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate:
    """Close the set-valued theorem for a displayed recursive stratification.

    This is the equality-stratum companion to the uniform-margin theorem.  It
    consumes already constructed finite recursive stratified branch/event
    trees; it does not claim the missing arbitrary interval-input partition
    generation theorem.
    """

    branch_consumption = recursive_stratified_branch_consumption_certificate
    event_consumption = (
        recursive_stratified_event_order_consumption_certificate
        if recursive_stratified_event_order_consumption_certificate is not None
        else branch_consumption
    )
    if isinstance(branch_consumption, (bool, np.bool_)) or not isinstance(
        branch_consumption,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ):
        raise TypeError(
            "recursive_stratified_branch_consumption_certificate must be a "
            "RecursiveStratifiedBranchEventConsumptionCertificate"
        )
    if isinstance(event_consumption, (bool, np.bool_)) or not isinstance(
        event_consumption,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ):
        raise TypeError(
            "recursive_stratified_event_order_consumption_certificate must be a "
            "RecursiveStratifiedBranchEventConsumptionCertificate"
        )
    search = certify_finite_target_certificate_search_completeness(
        theorem_certificate,
        observed_prefix_failures=observed_prefix_failures,
        recursive_stratified_branch_consumption_certificate=branch_consumption,
        recursive_stratified_event_order_consumption_certificate=event_consumption,
    )
    constructor_scope = _shared_recursive_constructor_scope_detail(
        branch_consumption,
        event_consumption,
    )
    if constructor_scope is None:
        scope_statement = "supplied finite recursive stratified trees"
        scope_proof = (
            "the supplied branch tree and event-order tree. Positive-margin "
            "leaves are terminal atlas/stop/selector responses. "
            "Simultaneous-event, selector, and total-collision equality "
            "leaves are nonterminal only when they carry a child certificate "
            "on a strictly lower dimension or lower rank stratum."
        )
        scope_source = (
            "the finite recursive tree displayed by the constructors"
        )
    else:
        _scope_id, scope_phrase = constructor_scope
        scope_statement = f"supplied finite recursive stratified trees from {scope_phrase}"
        scope_proof = (
            f"the constructor-derived {scope_phrase}. Positive-margin leaves "
            "are terminal atlas/stop/selector responses. Equality leaves are "
            "nonterminal only when they carry a child certificate on a "
            "strictly lower dimension or lower rank stratum."
        )
        scope_source = f"the constructor-derived {scope_phrase}"
    obligations = (
        TheoremPipelineObligation(
            obligation="pointwise_finite_target_theorem_proof_certified",
            certified=theorem_certificate.proof_certified,
            source=type(theorem_certificate).__name__,
            detail=(
                "missing="
                + ",".join(theorem_certificate.missing_obligations)
            ),
        ),
        TheoremPipelineObligation(
            obligation="recursive_stratified_branch_consumption_certified",
            certified=_recursive_stratified_consumption_certified(
                branch_consumption
            ),
            source=type(branch_consumption).__name__,
            detail=_recursive_stratified_detail(branch_consumption),
        ),
        TheoremPipelineObligation(
            obligation="recursive_stratified_event_order_consumption_certified",
            certified=_recursive_stratified_consumption_certified(
                event_consumption
            ),
            source=type(event_consumption).__name__,
            detail=_recursive_stratified_detail(event_consumption),
        ),
        TheoremPipelineObligation(
            obligation="certificate_search_consumes_recursive_stratified_trees",
            certified=search.certified,
            source=type(search).__name__,
            detail="missing=" + ",".join(search.missing_obligations),
        ),
        TheoremPipelineObligation(
            obligation="finite_recursive_descent_not_manual_flag",
            certified=bool(
                branch_consumption.recursive_theorem_certified
                and event_consumption.recursive_theorem_certified
            ),
            source="recursive_stratified_branch_event_consumption",
            detail=(
                "branch_depth="
                f"{branch_consumption.recursion_depth}; "
                "event_depth="
                f"{event_consumption.recursion_depth}; "
                "manual recursive_exhaustion_certified flags are not accepted"
            ),
        ),
        TheoremPipelineObligation(
            obligation="arbitrary_partition_generation_not_claimed",
            certified=True,
            source="supplied_recursive_stratified_set_valued_constructor_theorem",
            required=False,
            detail=(
                f"the theorem consumes {scope_source}; deriving such a tree from arbitrary "
                "interval inputs remains a separate theorem"
            ),
        ),
    )
    return SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate(
        theorem_certificate=theorem_certificate,
        search_completeness_certificate=search,
        branch_consumption_certificate=branch_consumption,
        event_order_consumption_certificate=event_consumption,
        statement=(
            "For compact positive-mass noncollision interval input sets whose "
            "branch and event-order ambiguities are represented by "
            f"{scope_statement}, the finite-target set-valued "
            "constructor consumes every displayed equality stratum by terminal "
            "proof certificates or strict dimension/rank descent and returns "
            "a finite certified atlas-or-stop branch union."
        ),
        proof_sketch=(
            "Apply the recursive stratified consumption theorem to "
            f"{scope_proof} Because both recursive trees are "
            "finite and every child edge descends, induction consumes all "
            "leaves. The pointwise finite-target theorem and fair exact-point "
            "certificate search supply the leaf atlas-or-stop certificates, "
            "and finite supplied branch-tree consumption glues the finite "
            "leaf responses. This proves the displayed constructor input "
            "class; it does not construct the recursive stratification from "
            "arbitrary interval boxes outside that class."
        ),
        obligations=obligations,
    )


def certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness(
    theorem_certificate: FiniteTargetCompletenessTheoremCertificate,
    *,
    constructor_certificate: object,
    root_dimension: int | None = None,
    root_rank: int | None = None,
    recursive_stratified_event_order_consumption_certificate: (
        RecursiveStratifiedBranchEventConsumptionCertificate | None
    ) = None,
    observed_prefix_failures: tuple[str, ...] = (),
) -> SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate:
    """Close the supplied-recursive theorem from a displayed constructor.

    This is a thin adapter over existing constructor and recursive-consumption
    proofs.  It derives the recursive branch-consumption certificate from one
    supported displayed stratification, then feeds the existing supplied
    recursive theorem.  It does not generate that displayed constructor from
    arbitrary interval input data.
    """

    recursive = _derive_recursive_consumption_from_displayed_constructor(
        constructor_certificate,
        root_dimension=root_dimension,
        root_rank=root_rank,
    )
    return certify_supplied_recursive_stratified_set_valued_constructor_completeness(
        theorem_certificate,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=(
            recursive_stratified_event_order_consumption_certificate
        ),
        observed_prefix_failures=observed_prefix_failures,
    )


def certify_constructor_pair_derived_recursive_stratified_set_valued_constructor_completeness(
    theorem_certificate: FiniteTargetCompletenessTheoremCertificate,
    *,
    branch_constructor_certificate: object,
    event_order_constructor_certificate: object,
    branch_root_dimension: int | None = None,
    branch_root_rank: int | None = None,
    event_order_root_dimension: int | None = None,
    event_order_root_rank: int | None = None,
    observed_prefix_failures: tuple[str, ...] = (),
) -> SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate:
    """Close the supplied-recursive theorem from branch/event constructors."""

    branch_recursive = _derive_recursive_consumption_from_displayed_constructor(
        branch_constructor_certificate,
        root_dimension=branch_root_dimension,
        root_rank=branch_root_rank,
    )
    event_recursive = _derive_recursive_consumption_from_displayed_constructor(
        event_order_constructor_certificate,
        root_dimension=event_order_root_dimension,
        root_rank=event_order_root_rank,
    )
    return certify_supplied_recursive_stratified_set_valued_constructor_completeness(
        theorem_certificate,
        recursive_stratified_branch_consumption_certificate=branch_recursive,
        recursive_stratified_event_order_consumption_certificate=event_recursive,
        observed_prefix_failures=observed_prefix_failures,
    )


def _scoped_partition_grammar_id(
    branch_grammar: str,
    event_grammar: str,
) -> str:
    if not branch_grammar or not event_grammar:
        return ""
    if branch_grammar == event_grammar:
        return branch_grammar
    return (
        "finite_mixed_supported_constructor_interval_inputs"
        f"(branch={branch_grammar};event={event_grammar})"
    )


def certify_arbitrary_interval_input_partition_generation(
    theorem_certificate: FiniteTargetCompletenessTheoremCertificate,
    *,
    branch_constructor_certificate: object,
    event_order_constructor_certificate: object | None = None,
    branch_root_dimension: int | None = None,
    branch_root_rank: int | None = None,
    event_order_root_dimension: int | None = None,
    event_order_root_rank: int | None = None,
    observed_prefix_failures: tuple[str, ...] = (),
) -> ArbitraryIntervalInputPartitionGenerationCertificate:
    """Certify scoped interval-input partition generation for supported grammars.

    The word "arbitrary" here is scoped by ``event_function_grammar_id``: the
    constructor may refine any interval input represented in that finite
    grammar.  It deliberately refuses to promote unsupported analytic strata or
    nonlisted grammars to the full arbitrary interval-box theorem.
    """

    event_constructor = (
        event_order_constructor_certificate
        if event_order_constructor_certificate is not None
        else branch_constructor_certificate
    )
    if isinstance(branch_constructor_certificate, (bool, np.bool_)):
        raise TypeError("branch_constructor_certificate must be a constructor certificate")
    if isinstance(event_constructor, (bool, np.bool_)):
        raise TypeError("event_order_constructor_certificate must be a constructor certificate")
    if event_order_constructor_certificate is None:
        scoped_constructor = (
            certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness(
                theorem_certificate,
                constructor_certificate=branch_constructor_certificate,
                root_dimension=branch_root_dimension,
                root_rank=branch_root_rank,
                observed_prefix_failures=observed_prefix_failures,
            )
        )
        generated_tree: object = getattr(
            branch_constructor_certificate,
            "stratified_tree",
            None,
        )
    else:
        scoped_constructor = (
            certify_constructor_pair_derived_recursive_stratified_set_valued_constructor_completeness(
                theorem_certificate,
                branch_constructor_certificate=branch_constructor_certificate,
                event_order_constructor_certificate=event_constructor,
                branch_root_dimension=branch_root_dimension,
                branch_root_rank=branch_root_rank,
                event_order_root_dimension=event_order_root_dimension,
                event_order_root_rank=event_order_root_rank,
                observed_prefix_failures=observed_prefix_failures,
            )
        )
        generated_tree = (
            getattr(branch_constructor_certificate, "stratified_tree", None),
            getattr(event_constructor, "stratified_tree", None),
        )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped_constructor,
    )
    branch_consumption = scoped_constructor.branch_consumption_certificate
    event_consumption = scoped_constructor.event_order_consumption_certificate
    branch_source = recursive_constructor_source_type(branch_consumption)
    event_source = recursive_constructor_source_type(event_consumption)
    branch_scope = recursive_constructor_source_scope(branch_consumption)
    event_scope = recursive_constructor_source_scope(event_consumption)
    branch_scope_id = branch_scope[1] if branch_scope is not None else ""
    event_scope_id = event_scope[1] if event_scope is not None else ""
    branch_grammar = SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS.get(
        branch_source,
        "",
    )
    event_grammar = SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS.get(
        event_source,
        "",
    )
    grammar_id = _scoped_partition_grammar_id(branch_grammar, event_grammar)
    unsupported_strata = tuple(
        dict.fromkeys(
            (
                *_constructor_unsupported_strata(branch_constructor_certificate),
                *_constructor_unsupported_strata(event_constructor),
            )
        )
    )
    supported_grammar = bool(
        branch_grammar
        and event_grammar
        and grammar_id
        and branch_scope_id
        and event_scope_id
    )
    arbitrary_claimed = bool(
        supported_grammar
        and scoped_constructor.proof_certified is True
        and validated.proof_certified is True
        and not unsupported_strata
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="supported_event_function_grammar",
            certified=supported_grammar,
            source="scoped_arbitrary_interval_input_partition_generation",
            detail=(
                f"branch_source={branch_source}; "
                f"branch_grammar={branch_grammar or 'unsupported'}; "
                f"event_source={event_source}; "
                f"event_grammar={event_grammar or 'unsupported'}; "
                f"grammar={grammar_id or 'unsupported'}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="branch_constructor_scope_resolved",
            certified=bool(branch_scope_id),
            source="scoped_arbitrary_interval_input_partition_generation",
            detail=(
                f"branch_source={branch_source}; "
                f"branch_scope={branch_scope_id or 'unsupported'}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="event_order_constructor_scope_resolved",
            certified=bool(event_scope_id),
            source="scoped_arbitrary_interval_input_partition_generation",
            detail=(
                f"event_source={event_source}; "
                f"event_scope={event_scope_id or 'unsupported'}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_stratified_tree_present",
            certified=bool(generated_tree),
            source="scoped_arbitrary_interval_input_partition_generation",
            detail=type(generated_tree).__name__,
        ),
        TheoremPipelineObligation(
            obligation="branch_partition_consumption_certified",
            certified=branch_consumption.proof_certified is True,
            source=type(branch_consumption).__name__,
            detail=_recursive_stratified_detail(branch_consumption),
        ),
        TheoremPipelineObligation(
            obligation="event_order_partition_consumption_certified",
            certified=event_consumption.proof_certified is True,
            source=type(event_consumption).__name__,
            detail=_recursive_stratified_detail(event_consumption),
        ),
        TheoremPipelineObligation(
            obligation="no_unsupported_analytic_strata",
            certified=not unsupported_strata,
            source="scoped_arbitrary_interval_input_partition_generation",
            detail=",".join(unsupported_strata),
        ),
        TheoremPipelineObligation(
            obligation="scoped_set_valued_constructor_proof_certified",
            certified=scoped_constructor.proof_certified is True,
            source=type(scoped_constructor).__name__,
            detail="missing=" + ",".join(scoped_constructor.missing_obligations),
        ),
        TheoremPipelineObligation(
            obligation="validated_set_valued_scope_proof_certified",
            certified=validated.proof_certified is True,
            source=type(validated).__name__,
            detail=(
                f"input_scope_id={validated.input_scope_id}; "
                f"missing={','.join(validated.missing_obligations)}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="unqualified_arbitrary_analytic_inputs_not_claimed",
            certified=True,
            source="scoped_arbitrary_interval_input_partition_generation",
            required=False,
            detail=(
                "partition generation is claimed only for the finite grammar "
                f"{grammar_id or 'unsupported'}"
            ),
        ),
    )
    return ArbitraryIntervalInputPartitionGenerationCertificate(
        input_scope_id=validated.input_scope_id,
        event_function_grammar_id=grammar_id,
        branch_function_grammar_id=branch_grammar,
        event_order_function_grammar_id=event_grammar,
        branch_constructor_source_type=branch_source,
        event_order_constructor_source_type=event_source,
        branch_constructor_input_scope_id=branch_scope_id,
        event_order_constructor_input_scope_id=event_scope_id,
        generated_stratified_tree=generated_tree,
        branch_consumption_certificate=branch_consumption,
        event_order_consumption_certificate=event_consumption,
        unsupported_strata=unsupported_strata,
        arbitrary_partition_generation_claimed=arbitrary_claimed,
        set_valued_constructor_certificate=scoped_constructor,
        validated_set_valued_constructor_certificate=validated,
        obligations=obligations,
    )


def certify_supported_event_function_stratification_generation(
    theorem_certificate: FiniteTargetCompletenessTheoremCertificate,
    *,
    grammar_input: SupportedEventFunctionGrammarInput,
    event_order_grammar_input: SupportedEventFunctionGrammarInput | None = None,
    branch_root_dimension: int | None = None,
    branch_root_rank: int | None = None,
    event_order_root_dimension: int | None = None,
    event_order_root_rank: int | None = None,
    observed_prefix_failures: tuple[str, ...] = (),
) -> SupportedEventFunctionStratificationGenerationCertificate:
    """Generate a scoped partition from raw supported grammar data.

    This is the first implementation-theorem step beyond accepting a
    preconstructed displayed stratification.  The function takes finite
    discriminator data in one of ``SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS``,
    runs the matching constructor, checks that the emitted source type is the
    requested grammar source, and then feeds the result to
    ``certify_arbitrary_interval_input_partition_generation``.  When
    ``event_order_grammar_input`` is supplied, branch and event-order
    stratifications are generated independently and consumed as a mixed
    constructor pair.
    """

    if isinstance(grammar_input, (bool, np.bool_)) or not isinstance(
        grammar_input,
        SupportedEventFunctionGrammarInput,
    ):
        raise TypeError(
            "grammar_input must be a SupportedEventFunctionGrammarInput"
        )
    if event_order_grammar_input is not None and (
        isinstance(event_order_grammar_input, (bool, np.bool_))
        or not isinstance(
            event_order_grammar_input,
            SupportedEventFunctionGrammarInput,
        )
    ):
        raise TypeError(
            "event_order_grammar_input must be a SupportedEventFunctionGrammarInput"
        )
    constructor = _construct_supported_event_function_stratification(
        grammar_input,
    )
    event_order_constructor = (
        _construct_supported_event_function_stratification(event_order_grammar_input)
        if event_order_grammar_input is not None
        else None
    )
    generated_source = str(
        getattr(getattr(constructor, "source_tree", None), "source_type", "")
    )
    event_generated_source = (
        str(
            getattr(
                getattr(event_order_constructor, "source_tree", None),
                "source_type",
                "",
            )
        )
        if event_order_constructor is not None
        else generated_source
    )
    source_matches = generated_source == grammar_input.source_type
    event_source_matches = bool(
        event_order_grammar_input is None
        or event_generated_source == event_order_grammar_input.source_type
    )
    partition = certify_arbitrary_interval_input_partition_generation(
        theorem_certificate,
        branch_constructor_certificate=constructor,
        event_order_constructor_certificate=event_order_constructor,
        branch_root_dimension=branch_root_dimension,
        branch_root_rank=branch_root_rank,
        event_order_root_dimension=event_order_root_dimension,
        event_order_root_rank=event_order_root_rank,
        observed_prefix_failures=observed_prefix_failures,
    )
    expected_branch_grammar = grammar_input.grammar_id
    expected_event_order_grammar = (
        grammar_input.grammar_id
        if event_order_grammar_input is None
        else event_order_grammar_input.grammar_id
    )
    branch_input_signature = _supported_grammar_input_signature(grammar_input)
    branch_payload_signature = _supported_grammar_payload_signature(grammar_input)
    generated_branch_signature = _constructor_input_signature(constructor)
    event_order_input_signature = (
        branch_input_signature
        if event_order_grammar_input is None
        else _supported_grammar_input_signature(event_order_grammar_input)
    )
    event_order_payload_signature = (
        branch_payload_signature
        if event_order_grammar_input is None
        else _supported_grammar_payload_signature(event_order_grammar_input)
    )
    generated_event_order_signature = _constructor_input_signature(
        event_order_constructor
        if event_order_constructor is not None
        else constructor
    )
    branch_generated_evidence_signature = (
        _constructor_generated_evidence_signature(constructor)
    )
    event_order_generated_evidence_signature = (
        _constructor_generated_evidence_signature(
            event_order_constructor
            if event_order_constructor is not None
            else constructor
        )
    )
    branch_constructor_replays = _supported_constructor_replay_matches_grammar_input(
        grammar_input=grammar_input,
        constructor_certificate=constructor,
    )
    event_order_constructor_replays = (
        True
        if event_order_grammar_input is None
        else _supported_constructor_replay_matches_grammar_input(
            grammar_input=event_order_grammar_input,
            constructor_certificate=event_order_constructor,
        )
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="supported_grammar_input_declared",
            certified=grammar_input.supported,
            source="supported_event_function_stratification_generation",
            detail=(
                f"source_type={grammar_input.source_type}; "
                f"grammar={grammar_input.grammar_id or 'unsupported'}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="constructor_generated_from_grammar_input",
            certified=bool(constructor is not None),
            source=type(constructor).__name__,
            detail=f"generated_source={generated_source or 'missing'}",
        ),
        TheoremPipelineObligation(
            obligation="generated_constructor_source_matches_requested_grammar",
            certified=source_matches,
            source=type(constructor).__name__,
            detail=(
                f"requested={grammar_input.source_type}; "
                f"generated={generated_source or 'missing'}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_constructor_input_matches_grammar_input",
            certified=generated_branch_signature == branch_input_signature,
            source=type(constructor).__name__,
            detail=(
                f"requested_source={grammar_input.source_type}; "
                f"generated_source={generated_source or 'missing'}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="grammar_input_payload_matches_generation",
            certified=branch_payload_signature
            == _supported_grammar_payload_signature(grammar_input),
            source="supported_event_function_stratification_generation",
            detail=(
                f"source_type={grammar_input.source_type}; "
                f"policy={grammar_input.equality_resolution_policy}; "
                f"max_bisection_depth={grammar_input.max_bisection_depth}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_constructor_evidence_matches_generation",
            certified=(
                _constructor_generated_evidence_signature(constructor)
                == branch_generated_evidence_signature
            ),
            source=type(constructor).__name__,
            detail=f"generated_source={generated_source or 'missing'}",
        ),
        TheoremPipelineObligation(
            obligation="generated_constructor_replays_from_grammar_input",
            certified=branch_constructor_replays,
            source=type(constructor).__name__,
            detail=(
                f"source_type={grammar_input.source_type}; "
                f"policy={grammar_input.equality_resolution_policy}; "
                f"max_bisection_depth={grammar_input.max_bisection_depth}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="event_order_grammar_input_declared",
            certified=bool(
                event_order_grammar_input is None
                or event_order_grammar_input.supported
            ),
            source="supported_event_function_stratification_generation",
            required=event_order_grammar_input is not None,
            detail=(
                "shared branch/event grammar"
                if event_order_grammar_input is None
                else (
                    f"source_type={event_order_grammar_input.source_type}; "
                    "grammar="
                    f"{event_order_grammar_input.grammar_id or 'unsupported'}"
                )
            ),
        ),
        TheoremPipelineObligation(
            obligation="event_order_constructor_generated_from_grammar_input",
            certified=bool(
                event_order_grammar_input is None
                or event_order_constructor is not None
            ),
            source=(
                type(event_order_constructor).__name__
                if event_order_constructor is not None
                else type(constructor).__name__
            ),
            required=event_order_grammar_input is not None,
            detail=f"generated_source={event_generated_source or 'missing'}",
        ),
        TheoremPipelineObligation(
            obligation="generated_event_order_constructor_source_matches_requested_grammar",
            certified=event_source_matches,
            source=(
                type(event_order_constructor).__name__
                if event_order_constructor is not None
                else type(constructor).__name__
            ),
            required=event_order_grammar_input is not None,
            detail=(
                "shared branch/event grammar"
                if event_order_grammar_input is None
                else (
                    f"requested={event_order_grammar_input.source_type}; "
                    f"generated={event_generated_source or 'missing'}"
                )
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_event_order_constructor_input_matches_grammar_input",
            certified=generated_event_order_signature == event_order_input_signature,
            source=(
                type(event_order_constructor).__name__
                if event_order_constructor is not None
                else type(constructor).__name__
            ),
            required=event_order_grammar_input is not None,
            detail=(
                "shared branch/event grammar"
                if event_order_grammar_input is None
                else (
                    f"requested_source={event_order_grammar_input.source_type}; "
                    f"generated_source={event_generated_source or 'missing'}"
                )
            ),
        ),
        TheoremPipelineObligation(
            obligation="event_order_grammar_input_payload_matches_generation",
            certified=event_order_payload_signature
            == (
                branch_payload_signature
                if event_order_grammar_input is None
                else _supported_grammar_payload_signature(event_order_grammar_input)
            ),
            source="supported_event_function_stratification_generation",
            required=event_order_grammar_input is not None,
            detail=(
                "shared branch/event grammar"
                if event_order_grammar_input is None
                else (
                    f"source_type={event_order_grammar_input.source_type}; "
                    "policy="
                    f"{event_order_grammar_input.equality_resolution_policy}; "
                    "max_bisection_depth="
                    f"{event_order_grammar_input.max_bisection_depth}"
                )
            ),
        ),
        TheoremPipelineObligation(
            obligation=(
                "generated_event_order_constructor_evidence_matches_generation"
            ),
            certified=(
                _constructor_generated_evidence_signature(
                    event_order_constructor
                    if event_order_constructor is not None
                    else constructor
                )
                == event_order_generated_evidence_signature
            ),
            source=(
                type(event_order_constructor).__name__
                if event_order_constructor is not None
                else type(constructor).__name__
            ),
            required=event_order_grammar_input is not None,
            detail=(
                "shared branch/event grammar"
                if event_order_grammar_input is None
                else f"generated_source={event_generated_source or 'missing'}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_event_order_constructor_replays_from_grammar_input",
            certified=event_order_constructor_replays,
            source=(
                type(event_order_constructor).__name__
                if event_order_constructor is not None
                else type(constructor).__name__
            ),
            required=event_order_grammar_input is not None,
            detail=(
                "shared branch/event grammar"
                if event_order_grammar_input is None
                else (
                    f"source_type={event_order_grammar_input.source_type}; "
                    "policy="
                    f"{event_order_grammar_input.equality_resolution_policy}; "
                    "max_bisection_depth="
                    f"{event_order_grammar_input.max_bisection_depth}"
                )
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_event_order_constructor_proof_certified",
            certified=bool(
                event_order_grammar_input is None
                or _object_proof_certified(event_order_constructor)
            ),
            source=(
                type(event_order_constructor).__name__
                if event_order_constructor is not None
                else type(constructor).__name__
            ),
            required=event_order_grammar_input is not None,
            detail=(
                "shared branch/event grammar"
                if event_order_constructor is None
                else (
                    "missing="
                    + ",".join(
                        getattr(
                            event_order_constructor,
                            "missing_obligations",
                            (),
                        )
                    )
                )
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_partition_bridge_branch_source_matches_constructor",
            certified=partition.branch_constructor_source_type == generated_source,
            source=type(partition).__name__,
            detail=(
                f"partition_branch_source={partition.branch_constructor_source_type}; "
                f"generated_branch_source={generated_source or 'missing'}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_partition_bridge_event_order_source_matches_constructor",
            certified=(
                partition.event_order_constructor_source_type
                == event_generated_source
            ),
            source=type(partition).__name__,
            detail=(
                "partition_event_source="
                f"{partition.event_order_constructor_source_type}; "
                f"generated_event_source={event_generated_source or 'missing'}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_partition_bridge_branch_grammar_matches_input",
            certified=partition.branch_function_grammar_id == expected_branch_grammar,
            source=type(partition).__name__,
            detail=(
                f"partition_branch_grammar={partition.branch_function_grammar_id}; "
                f"expected_branch_grammar={expected_branch_grammar or 'missing'}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_partition_bridge_event_order_grammar_matches_input",
            certified=(
                partition.event_order_function_grammar_id
                == expected_event_order_grammar
            ),
            source=type(partition).__name__,
            detail=(
                "partition_event_grammar="
                f"{partition.event_order_function_grammar_id}; "
                f"expected_event_grammar={expected_event_order_grammar or 'missing'}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_constructor_proof_certified",
            certified=_object_proof_certified(constructor),
            source=type(constructor).__name__,
            detail=(
                "missing="
                + ",".join(getattr(constructor, "missing_obligations", ()))
            ),
        ),
        TheoremPipelineObligation(
            obligation="generated_partition_bridge_proof_certified",
            certified=partition.proof_certified is True,
            source=type(partition).__name__,
            detail="missing=" + ",".join(partition.missing_obligations),
        ),
    )
    return SupportedEventFunctionStratificationGenerationCertificate(
        grammar_input=grammar_input,
        constructor_certificate=constructor,
        partition_generation_certificate=partition,
        obligations=obligations,
        event_order_grammar_input=event_order_grammar_input,
        event_order_constructor_certificate=event_order_constructor,
        branch_generated_evidence_signature=branch_generated_evidence_signature,
        event_order_generated_evidence_signature=(
            event_order_generated_evidence_signature
        ),
        branch_grammar_payload_signature=branch_payload_signature,
        event_order_grammar_payload_signature=event_order_payload_signature,
    )


def _construct_supported_event_function_stratification(
    grammar_input: SupportedEventFunctionGrammarInput,
) -> object:
    source_type = str(grammar_input.source_type)
    if source_type not in SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS:
        raise ValueError(f"unsupported event-function grammar source: {source_type}")
    if source_type == "AffineDecisionStratification":
        return certify_affine_decision_stratified_branch_event_tree(
            decision_id=grammar_input.decision_id,
            coefficients=grammar_input.coefficients,
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
        )
    if source_type == "PolynomialDecisionStratification":
        return certify_polynomial_decision_stratified_branch_event_tree(
            decision_id=grammar_input.decision_id,
            coefficients=grammar_input.coefficients,
            domain=_require_interval_domain(grammar_input, source_type),
            root_brackets=grammar_input.root_brackets,
            equality_resolution_policy=grammar_input.equality_resolution_policy,
        )
    if source_type == "SturmPolynomialDecisionStratification":
        return certify_sturm_polynomial_decision_stratified_branch_event_tree(
            decision_id=grammar_input.decision_id,
            coefficients=grammar_input.coefficients,
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
            max_bisection_depth=grammar_input.max_bisection_depth,
        )
    if source_type == "AffineDecisionArrangement":
        return certify_affine_decision_arrangement_stratified_branch_event_tree(
            arrangement_id=_require_arrangement_id(grammar_input, source_type),
            decision_functions=_require_polynomial_decision_functions(
                grammar_input,
                source_type,
            ),
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
        )
    if source_type == "PolynomialDecisionArrangement":
        return certify_polynomial_decision_arrangement_stratified_branch_event_tree(
            arrangement_id=_require_arrangement_id(grammar_input, source_type),
            decision_functions=_require_polynomial_decision_functions(
                grammar_input,
                source_type,
            ),
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
        )
    if source_type == "SturmPolynomialDecisionArrangement":
        return certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree(
            arrangement_id=_require_arrangement_id(grammar_input, source_type),
            decision_functions=_require_polynomial_decision_functions(
                grammar_input,
                source_type,
            ),
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
            max_bisection_depth=grammar_input.max_bisection_depth,
        )
    if source_type in {"QuadraticDoubleRootArrangement", "PolynomialRootArrangement"}:
        return certify_quadratic_decision_arrangement_stratified_branch_event_tree(
            arrangement_id=_require_arrangement_id(grammar_input, source_type),
            decision_functions=_require_polynomial_decision_functions(
                grammar_input,
                source_type,
            ),
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
        )
    if source_type == "RationalDecisionStratification":
        return certify_rational_decision_stratified_branch_event_tree(
            decision_function=_require_rational_decision_function(
                grammar_input,
                source_type,
            ),
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
        )
    if source_type == "SturmRationalDecisionStratification":
        return certify_sturm_rational_decision_stratified_branch_event_tree(
            decision_function=_require_rational_decision_function(
                grammar_input,
                source_type,
            ),
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
            max_bisection_depth=grammar_input.max_bisection_depth,
        )
    if source_type == "RationalDecisionArrangement":
        return certify_rational_decision_arrangement_stratified_branch_event_tree(
            arrangement_id=_require_arrangement_id(grammar_input, source_type),
            decision_functions=_require_rational_decision_functions(
                grammar_input,
                source_type,
            ),
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
        )
    if source_type == "SturmRationalDecisionArrangement":
        return certify_sturm_rational_decision_arrangement_stratified_branch_event_tree(
            arrangement_id=_require_arrangement_id(grammar_input, source_type),
            decision_functions=_require_rational_decision_functions(
                grammar_input,
                source_type,
            ),
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
            max_bisection_depth=grammar_input.max_bisection_depth,
        )
    if source_type == "TaylorModelDecisionStratification":
        return certify_taylor_model_decision_stratified_branch_event_tree(
            decision_function=_require_taylor_model_decision_function(
                grammar_input,
                source_type,
            ),
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
        )
    if source_type == "TaylorModelDecisionArrangement":
        return certify_taylor_model_decision_arrangement_stratified_branch_event_tree(
            arrangement_id=_require_arrangement_id(grammar_input, source_type),
            decision_functions=_require_taylor_model_decision_functions(
                grammar_input,
                source_type,
            ),
            domain=_require_interval_domain(grammar_input, source_type),
            equality_resolution_policy=grammar_input.equality_resolution_policy,
        )
    if source_type == "AxisAlignedAffineBoxArrangement":
        return certify_affine_box_decision_arrangement_stratified_branch_event_tree(
            arrangement_id=_require_arrangement_id(grammar_input, source_type),
            decision_functions=_require_affine_box_decision_functions(
                grammar_input,
                source_type,
            ),
            domain_box=_require_domain_box(grammar_input, source_type),
        )
    if source_type == "AffineHalfspaceDecision":
        return certify_affine_halfspace_decision_stratified_branch_event_tree(
            decision_id=grammar_input.decision_id,
            coefficients=grammar_input.coefficients,
            domain_box=_require_domain_box(grammar_input, source_type),
            slab_half_width=_require_slab_half_width(grammar_input, source_type),
        )
    if source_type == "AffineHalfspaceArrangement":
        return certify_affine_halfspace_arrangement_stratified_branch_event_tree(
            arrangement_id=_require_arrangement_id(grammar_input, source_type),
            decision_functions=_require_affine_box_decision_functions(
                grammar_input,
                source_type,
            ),
            domain_box=_require_domain_box(grammar_input, source_type),
            slab_half_width=_require_slab_half_width(grammar_input, source_type),
        )
    if source_type == "AffineHalfspace3DArrangement":
        return certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
            arrangement_id=_require_arrangement_id(grammar_input, source_type),
            decision_functions=_require_affine_box_decision_functions(
                grammar_input,
                source_type,
            ),
            domain_box=_require_domain_box(grammar_input, source_type),
            slab_half_width=_require_slab_half_width(grammar_input, source_type),
        )
    raise ValueError(f"supported grammar has no generator: {source_type}")


def _require_interval_domain(
    grammar_input: SupportedEventFunctionGrammarInput,
    source_type: str,
) -> tuple[float, float]:
    if grammar_input.domain is None:
        raise ValueError(f"{source_type} requires a compact interval domain")
    lower, upper = grammar_input.domain
    return float(lower), float(upper)


def _require_domain_box(
    grammar_input: SupportedEventFunctionGrammarInput,
    source_type: str,
) -> tuple[tuple[float, float], ...]:
    if not grammar_input.domain_box:
        raise ValueError(f"{source_type} requires a compact domain_box")
    return tuple(
        (float(lower), float(upper))
        for lower, upper in grammar_input.domain_box
    )


def _require_arrangement_id(
    grammar_input: SupportedEventFunctionGrammarInput,
    source_type: str,
) -> str:
    arrangement_id = str(grammar_input.arrangement_id)
    if not arrangement_id:
        raise ValueError(f"{source_type} requires arrangement_id")
    return arrangement_id


def _require_slab_half_width(
    grammar_input: SupportedEventFunctionGrammarInput,
    source_type: str,
) -> float:
    if grammar_input.slab_half_width is None:
        raise ValueError(f"{source_type} requires slab_half_width")
    return float(grammar_input.slab_half_width)


def _require_polynomial_decision_functions(
    grammar_input: SupportedEventFunctionGrammarInput,
    source_type: str,
) -> tuple[PolynomialDecisionFunctionSpec, ...]:
    specs = tuple(grammar_input.polynomial_decision_functions)
    if not specs:
        raise ValueError(f"{source_type} requires polynomial_decision_functions")
    return specs


def _require_rational_decision_function(
    grammar_input: SupportedEventFunctionGrammarInput,
    source_type: str,
) -> RationalDecisionFunctionSpec:
    spec = grammar_input.rational_decision_function
    if spec is None:
        raise ValueError(f"{source_type} requires rational_decision_function")
    return spec


def _require_rational_decision_functions(
    grammar_input: SupportedEventFunctionGrammarInput,
    source_type: str,
) -> tuple[RationalDecisionFunctionSpec, ...]:
    specs = tuple(grammar_input.rational_decision_functions)
    if not specs:
        raise ValueError(f"{source_type} requires rational_decision_functions")
    return specs


def _require_taylor_model_decision_function(
    grammar_input: SupportedEventFunctionGrammarInput,
    source_type: str,
) -> TaylorModelDecisionFunctionSpec:
    spec = grammar_input.taylor_model_decision_function
    if spec is None:
        raise ValueError(f"{source_type} requires taylor_model_decision_function")
    return spec


def _require_taylor_model_decision_functions(
    grammar_input: SupportedEventFunctionGrammarInput,
    source_type: str,
) -> tuple[TaylorModelDecisionFunctionSpec, ...]:
    specs = tuple(grammar_input.taylor_model_decision_functions)
    if not specs:
        raise ValueError(f"{source_type} requires taylor_model_decision_functions")
    return specs


def _require_affine_box_decision_functions(
    grammar_input: SupportedEventFunctionGrammarInput,
    source_type: str,
) -> tuple[AffineBoxDecisionFunctionSpec, ...]:
    specs = tuple(grammar_input.affine_box_decision_functions)
    if not specs:
        raise ValueError(f"{source_type} requires affine_box_decision_functions")
    return specs


def certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
    theorem_certificate: FiniteTargetCompletenessTheoremCertificate,
    *,
    arrangement_certificate: (
        AffineHalfspaceArrangementStratificationCertificate
        | AffineHalfspaceArrangement3DStratificationCertificate
    ),
    recursive_stratified_branch_consumption_certificate: (
        RecursiveStratifiedBranchEventConsumptionCertificate
        | None
    ) = None,
    recursive_stratified_event_order_consumption_certificate: (
        RecursiveStratifiedBranchEventConsumptionCertificate | None
    ) = None,
    observed_prefix_failures: tuple[str, ...] = (),
) -> AffineHalfspaceArrangementSetValuedConstructorCompletenessCertificate:
    """Close the represented finite affine-halfspace arrangement input class.

    The arrangement itself must be constructor-derived from affine
    discriminants and must pass its convex-cell value-bound and measure-cover
    checks.  Equality slab cells are then consumed by derived or supplied
    recursive descent certificates.  This still does not generate such
    arrangements from arbitrary interval inputs.
    """

    arrangement = arrangement_certificate
    arrangement_types = (
        AffineHalfspaceArrangementStratificationCertificate,
        AffineHalfspaceArrangement3DStratificationCertificate,
    )
    if isinstance(arrangement, (bool, np.bool_)) or not isinstance(
        arrangement,
        arrangement_types,
    ):
        raise TypeError(
            "arrangement_certificate must be an "
            "AffineHalfspaceArrangementStratificationCertificate or "
            "AffineHalfspaceArrangement3DStratificationCertificate"
        )
    if recursive_stratified_branch_consumption_certificate is None:
        if isinstance(arrangement, AffineHalfspaceArrangement3DStratificationCertificate):
            recursive_stratified_branch_consumption_certificate = (
                certify_affine_halfspace_3d_arrangement_recursive_consumption(
                    arrangement,
                    root_dimension=3,
                    root_rank=2,
                )
            )
        else:
            recursive_stratified_branch_consumption_certificate = (
                certify_affine_halfspace_arrangement_recursive_consumption(
                    arrangement,
                    root_dimension=2,
                    root_rank=2,
                )
            )
    branch_consumption = recursive_stratified_branch_consumption_certificate
    event_consumption = (
        recursive_stratified_event_order_consumption_certificate
        if recursive_stratified_event_order_consumption_certificate is not None
        else branch_consumption
    )
    if isinstance(branch_consumption, (bool, np.bool_)) or not isinstance(
        branch_consumption,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ):
        raise TypeError(
            "recursive_stratified_branch_consumption_certificate must be a "
            "RecursiveStratifiedBranchEventConsumptionCertificate"
        )
    if isinstance(event_consumption, (bool, np.bool_)) or not isinstance(
        event_consumption,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ):
        raise TypeError(
            "recursive_stratified_event_order_consumption_certificate must be a "
            "RecursiveStratifiedBranchEventConsumptionCertificate"
        )
    if isinstance(arrangement, AffineHalfspaceArrangement3DStratificationCertificate):
        measure_kind = "volume"
        measure_cover_certified = arrangement.volume_cover_certified
        measure_gap = arrangement.cover_volume_gap_upper_bound
        domain_measure = arrangement.domain_volume
        cell_measure_sum = arrangement.cell_volume_sum
        measure_obligation = "affine_halfspace_arrangement_volume_cover_certified"
        cell_kind_phrase = "convex polyhedral cells"
        scope_phrase = "three-dimensional"
    else:
        measure_kind = "area"
        measure_cover_certified = arrangement.area_cover_certified
        measure_gap = arrangement.cover_area_gap_upper_bound
        domain_measure = arrangement.domain_area
        cell_measure_sum = arrangement.cell_area_sum
        measure_obligation = "affine_halfspace_arrangement_area_cover_certified"
        cell_kind_phrase = "convex polygon cells"
        scope_phrase = "two-dimensional"
    branch_matches = branch_consumption.source_tree == arrangement.stratified_tree
    event_matches = event_consumption.source_tree == arrangement.stratified_tree
    search = certify_finite_target_certificate_search_completeness(
        theorem_certificate,
        observed_prefix_failures=observed_prefix_failures,
        recursive_stratified_branch_consumption_certificate=branch_consumption,
        recursive_stratified_event_order_consumption_certificate=event_consumption,
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="pointwise_finite_target_theorem_proof_certified",
            certified=theorem_certificate.proof_certified,
            source=type(theorem_certificate).__name__,
            detail="missing=" + ",".join(theorem_certificate.missing_obligations),
        ),
        TheoremPipelineObligation(
            obligation="affine_halfspace_arrangement_proof_certified",
            certified=arrangement.proof_certified,
            source=type(arrangement).__name__,
            detail=(
                f"dimension={arrangement.dimension}; "
                f"sign_strata={arrangement.sign_stratum_count}; "
                f"equality_strata={arrangement.equality_stratum_count}; "
                f"missing={','.join(arrangement.missing_obligations)}"
            ),
        ),
        TheoremPipelineObligation(
            obligation=measure_obligation,
            certified=bool(
                measure_cover_certified
                and measure_gap <= 1.0e-8 * max(1.0, domain_measure)
            ),
            source=type(arrangement).__name__,
            detail=(
                f"domain_{measure_kind}={domain_measure:g}; "
                f"cell_{measure_kind}_sum={cell_measure_sum:g}; "
                f"cover_gap={measure_gap:g}"
            ),
        ),
        TheoremPipelineObligation(
            obligation="branch_consumption_uses_arrangement_stratified_tree",
            certified=bool(branch_matches and branch_consumption.certified),
            source=type(branch_consumption).__name__,
            detail=_recursive_stratified_detail(branch_consumption),
        ),
        TheoremPipelineObligation(
            obligation="event_order_consumption_uses_arrangement_stratified_tree",
            certified=bool(event_matches and event_consumption.certified),
            source=type(event_consumption).__name__,
            detail=_recursive_stratified_detail(event_consumption),
        ),
        TheoremPipelineObligation(
            obligation="certificate_search_consumes_affine_halfspace_arrangement",
            certified=search.certified,
            source=type(search).__name__,
            detail="missing=" + ",".join(search.missing_obligations),
        ),
        TheoremPipelineObligation(
            obligation="arbitrary_partition_generation_not_claimed",
            certified=True,
            source="affine_halfspace_arrangement_set_valued_constructor_theorem",
            required=False,
            detail=(
                "the theorem consumes a displayed finite affine halfspace "
                "arrangement with recursive equality children; deriving such "
                "arrangements from arbitrary interval inputs remains a "
                "separate theorem"
            ),
        ),
    )
    return AffineHalfspaceArrangementSetValuedConstructorCompletenessCertificate(
        theorem_certificate=theorem_certificate,
        search_completeness_certificate=search,
        arrangement_certificate=arrangement,
        branch_consumption_certificate=branch_consumption,
        event_order_consumption_certificate=event_consumption,
        statement=(
            "For compact interval input sets whose branch/event-order "
            f"discriminants are represented by a finite {scope_phrase} "
            "oblique affine halfspace arrangement, the finite-target "
            "set-valued constructor consumes every strict-sign cell and every "
            "displayed equality slab through recursive descent."
        ),
        proof_sketch=(
            "The affine halfspace arrangement constructor enumerates the "
            "trichotomy for every discriminator, clips the input box into "
            f"{cell_kind_phrase}, verifies affine value bounds on each cell, "
            f"and checks the cell {measure_kind} sum against the domain "
            f"{measure_kind}. "
            "Strict sign cells are positive-margin terminal leaves. Equality "
            "slab cells are named simultaneous-event strata and are consumed "
            "by the recursive descent certificates. The finite-target theorem "
            "and certificate-search theorem then provide and find the finite "
            "atlas-or-stop leaf certificates, while finite union glues the "
            "represented set-valued response."
        ),
        obligations=obligations,
    )


def _constructor_unsupported_strata(constructor_certificate: object) -> tuple[str, ...]:
    tree = getattr(constructor_certificate, "stratified_tree", None)
    leaves = tuple(getattr(tree, "leaf_certificates", ()) or ())
    unsupported: list[str] = []
    for leaf in leaves:
        leaf_kind = str(getattr(leaf, "leaf_kind", ""))
        missing = tuple(str(item) for item in getattr(leaf, "missing_obligations", ()))
        if leaf_kind == "unsupported_analytic_stratum":
            unsupported.append(str(getattr(leaf, "leaf_id", "unsupported")))
        elif any("unsupported_analytic_stratum" in item for item in missing):
            unsupported.append(str(getattr(leaf, "leaf_id", "unsupported")))
    strata = tuple(getattr(constructor_certificate, "strata", ()) or ())
    for stratum in strata:
        missing = tuple(
            str(item) for item in getattr(stratum, "missing_obligations", ())
        )
        if any("unsupported_analytic_stratum" in item for item in missing):
            unsupported.append(str(getattr(stratum, "stratum_id", "unsupported")))
    return tuple(dict.fromkeys(unsupported))


def _derive_recursive_consumption_from_displayed_constructor(
    constructor_certificate: object,
    *,
    root_dimension: int | None,
    root_rank: int | None,
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    dimension = (
        int(root_dimension)
        if root_dimension is not None
        else int(getattr(constructor_certificate, "dimension", 1))
    )
    rank = None if root_rank is None else int(root_rank)
    if isinstance(
        constructor_certificate,
        PolynomialDecisionStratificationCertificate,
    ):
        return certify_polynomial_decision_recursive_consumption(
            constructor_certificate,
            root_dimension=dimension,
            root_rank=rank,
        )
    if isinstance(
        constructor_certificate,
        RationalDecisionStratificationCertificate,
    ):
        return certify_rational_decision_recursive_consumption(
            constructor_certificate,
            root_dimension=dimension,
            root_rank=rank,
        )
    if isinstance(
        constructor_certificate,
        RationalDecisionArrangementStratificationCertificate,
    ):
        return certify_rational_decision_arrangement_recursive_consumption(
            constructor_certificate,
            root_dimension=dimension,
            root_rank=rank,
        )
    if isinstance(
        constructor_certificate,
        PolynomialDecisionArrangementStratificationCertificate,
    ):
        return certify_polynomial_decision_arrangement_recursive_consumption(
            constructor_certificate,
            root_dimension=dimension,
            root_rank=rank,
        )
    if isinstance(
        constructor_certificate,
        TaylorModelDecisionStratificationCertificate,
    ):
        return certify_taylor_model_decision_recursive_consumption(
            constructor_certificate,
            root_dimension=dimension,
            root_rank=rank,
        )
    if isinstance(
        constructor_certificate,
        TaylorModelDecisionArrangementStratificationCertificate,
    ):
        return certify_taylor_model_decision_arrangement_recursive_consumption(
            constructor_certificate,
            root_dimension=dimension,
            root_rank=rank,
        )
    if isinstance(
        constructor_certificate,
        AffineBoxDecisionArrangementStratificationCertificate,
    ):
        return certify_affine_box_decision_arrangement_recursive_consumption(
            constructor_certificate,
            root_dimension=dimension,
            root_rank=rank,
        )
    if isinstance(
        constructor_certificate,
        AffineHalfspaceDecisionStratificationCertificate,
    ):
        return certify_affine_halfspace_decision_recursive_consumption(
            constructor_certificate,
            root_dimension=dimension,
            root_rank=rank,
        )
    if isinstance(
        constructor_certificate,
        AffineHalfspaceArrangement3DStratificationCertificate,
    ):
        return certify_affine_halfspace_3d_arrangement_recursive_consumption(
            constructor_certificate,
            root_dimension=dimension,
            root_rank=rank,
        )
    if isinstance(
        constructor_certificate,
        AffineHalfspaceArrangementStratificationCertificate,
    ):
        return certify_affine_halfspace_arrangement_recursive_consumption(
            constructor_certificate,
            root_dimension=dimension,
            root_rank=rank,
        )
    raise TypeError(
        "constructor_certificate must be a supported displayed stratification "
        "certificate"
    )


def certify_validated_set_valued_constructor_completeness_theorem(
    set_valued_constructor_certificate: (
        UniformMarginSetValuedConstructorCompletenessCertificate
        | SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate
        | AffineHalfspaceArrangementSetValuedConstructorCompletenessCertificate
    ),
) -> ValidatedSetValuedConstructorCompletenessTheoremCertificate:
    """Name the separate validated-numerics theorem for interval boxes.

    The returned certificate certifies exactly the represented interval-input
    class of the supplied constructor theorem.  It records that arbitrary
    interval partition generation remains outside this wrapper unless supplied
    by a future stronger constructor certificate.
    """

    certificate = set_valued_constructor_certificate
    if isinstance(certificate, (bool, np.bool_)) or not isinstance(
        certificate,
        (
            UniformMarginSetValuedConstructorCompletenessCertificate,
            SuppliedRecursiveStratifiedSetValuedConstructorCompletenessCertificate,
            AffineHalfspaceArrangementSetValuedConstructorCompletenessCertificate,
        ),
    ):
        raise TypeError(
            "set_valued_constructor_certificate must be a scoped set-valued "
            "constructor completeness certificate"
        )
    theorem = certificate.theorem_certificate
    search = certificate.search_completeness_certificate
    input_scope_id, scope_detail = _validated_set_valued_constructor_scope(certificate)
    obligations = (
        TheoremPipelineObligation(
            obligation="pointwise_finite_target_theorem_proof_certified",
            certified=theorem.proof_certified,
            source=type(theorem).__name__,
            detail="missing=" + ",".join(theorem.missing_obligations),
        ),
        TheoremPipelineObligation(
            obligation="certificate_search_completeness_proof_certified",
            certified=search.proof_certified,
            source=type(search).__name__,
            detail="missing=" + ",".join(search.missing_obligations),
        ),
        TheoremPipelineObligation(
            obligation="scoped_set_valued_constructor_certificate_proof_certified",
            certified=certificate.proof_certified,
            source=type(certificate).__name__,
            detail="missing=" + ",".join(certificate.missing_obligations),
        ),
        TheoremPipelineObligation(
            obligation="validated_interval_input_scope_declared",
            certified=bool(input_scope_id),
            source="validated_set_valued_constructor_completeness_theorem",
            detail=f"input_scope_id={input_scope_id}; {scope_detail}",
        ),
        TheoremPipelineObligation(
            obligation="arbitrary_interval_partition_generation_not_claimed",
            certified=True,
            source="validated_set_valued_constructor_completeness_theorem",
            required=False,
            detail=(
                "this named theorem preserves the pointwise/interval-box "
                "split: it certifies the represented interval-input class, "
                "while arbitrary recursive partition generation remains open"
            ),
        ),
    )
    return ValidatedSetValuedConstructorCompletenessTheoremCertificate(
        theorem_certificate=theorem,
        search_completeness_certificate=search,
        set_valued_constructor_certificate=certificate,
        input_scope_id=input_scope_id,
        statement=(
            "The validated set-valued finite-target constructor theorem "
            f"certifies the interval-box input class `{input_scope_id}` by "
            "combining the pointwise finite-target atlas-or-stop theorem, "
            "fair certificate search, and a scoped branch/event-order "
            "constructor theorem."
        ),
        proof_sketch=(
            "Use the pointwise finite-target theorem to guarantee finite "
            "atlas-or-stop certificates on exact representatives, use fair "
            "certificate search to find them, and use the supplied scoped "
            "set-valued constructor certificate to prove that the same finite "
            "branch/event grammar covers the represented interval input set. "
            f"The represented scope is {scope_detail}. "
            "Finite branch-union and ledger aggregation then produce the "
            "validated set-valued response for that scope.  This wrapper does "
            "not derive recursive branch/event-order partitions from arbitrary "
            "interval boxes; arbitrary recursive partition generation remains "
            "open."
        ),
        obligations=obligations,
    )


def _stratified_tree_closes_recursive_theorem(
    certificate: StratifiedBranchTreeCertificate | None,
) -> bool:
    return bool(
        certificate is not None
        and certificate.recursive_theorem_certified
    )


def _require_certificate_type(name: str, value: object, expected_type: type) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__}")


def _require_optional_certificate_type(
    name: str,
    value: object | None,
    expected_type: type,
) -> None:
    if value is None:
        return
    _require_certificate_type(name, value, expected_type)


def _recursive_stratified_consumption_certified(
    certificate: RecursiveStratifiedBranchEventConsumptionCertificate | None,
) -> bool:
    return bool(certificate is not None and certificate.certified)


def _recursive_stratified_source_tree_certified(
    certificate: RecursiveStratifiedBranchEventConsumptionCertificate | None,
) -> bool:
    return bool(
        certificate is not None
        and certificate.source_tree.certified
    )


def _recursive_stratified_source_type(
    certificate: RecursiveStratifiedBranchEventConsumptionCertificate | None,
) -> str:
    return recursive_constructor_source_type(certificate)


def recursive_constructor_source_type(
    certificate: RecursiveStratifiedBranchEventConsumptionCertificate | None,
) -> str:
    if certificate is None:
        return ""
    return str(getattr(certificate, "constructor_source_type", ""))


def recursive_constructor_source_scope(
    certificate: RecursiveStratifiedBranchEventConsumptionCertificate | None,
) -> tuple[str, str, str, str] | None:
    source_type = recursive_constructor_source_type(certificate)
    scope = CONSTRUCTOR_DERIVED_RECURSIVE_SOURCE_SCOPES.get(source_type)
    if scope is None:
        return None
    input_scope_id, detail, label = scope
    return source_type, input_scope_id, detail, label


def _shared_recursive_constructor_scope_detail(
    branch_consumption: RecursiveStratifiedBranchEventConsumptionCertificate | None,
    event_consumption: RecursiveStratifiedBranchEventConsumptionCertificate | None,
) -> tuple[str, str] | None:
    branch_source = recursive_constructor_source_type(branch_consumption)
    event_source = recursive_constructor_source_type(event_consumption)
    branch_kind = str(getattr(branch_consumption, "recursion_kind", ""))
    event_kind = str(getattr(event_consumption, "recursion_kind", ""))
    if not branch_source or not event_source:
        return None
    branch_scope = recursive_constructor_source_scope(branch_consumption)
    event_scope = recursive_constructor_source_scope(event_consumption)
    if branch_source != event_source or branch_kind != event_kind:
        if branch_scope is not None and event_scope is not None:
            return (
                "finite_mixed_constructor_branch_event_interval_boxes",
                (
                    "constructor-derived mixed branch/event recursive "
                    f"partition with branch source {branch_source} "
                    f"({branch_scope[1]}) and event-order source "
                    f"{event_source} ({event_scope[1]}); each side is "
                    "recursively consumed without claiming arbitrary "
                    "interval-input partition generation"
                ),
            )
        return None
    if branch_scope is not None:
        return branch_scope[1], branch_scope[2]
    return None


def _stratified_or_recursive_source(
    stratified: StratifiedBranchTreeCertificate | None,
    recursive: RecursiveStratifiedBranchEventConsumptionCertificate | None,
) -> str:
    if recursive is not None:
        return type(recursive).__name__
    return _stratified_tree_source(stratified)


def _stratified_or_recursive_detail(
    stratified: StratifiedBranchTreeCertificate | None,
    recursive: RecursiveStratifiedBranchEventConsumptionCertificate | None,
    *,
    fallback: str,
) -> str:
    if recursive is not None:
        return _recursive_stratified_detail(recursive)
    return _stratified_tree_detail(stratified, fallback=fallback)


def _recursive_stratified_detail(
    certificate: RecursiveStratifiedBranchEventConsumptionCertificate,
) -> str:
    return (
        f"recursion_kind={certificate.recursion_kind}; "
        f"root_dimension={certificate.root_dimension}; "
        f"root_rank={certificate.root_rank}; "
        f"leaf_count={certificate.leaf_count}; "
        f"terminal_leaf_count={certificate.terminal_leaf_count}; "
        f"recursive_leaf_count={certificate.recursive_leaf_count}; "
        f"strict_descent_edge_count={certificate.strict_descent_edge_count}; "
        f"unresolved_descent_edge_count={certificate.unresolved_descent_edge_count}; "
        f"descent_well_founded={certificate.descent_well_founded}; "
        f"unsupported_leaf_count={certificate.unsupported_leaf_count}; "
        f"node_count={certificate.node_count}; "
        f"recursion_depth={certificate.recursion_depth}; "
        f"missing={','.join(certificate.missing_obligations)}"
    )


def _stratified_tree_source(
    certificate: StratifiedBranchTreeCertificate | None,
) -> str:
    if certificate is None:
        return "missing_certificate_search_theorem"
    return type(certificate).__name__


def _stratified_tree_detail(
    certificate: StratifiedBranchTreeCertificate | None,
    *,
    fallback: str,
) -> str:
    if certificate is None:
        return fallback
    return (
        f"leaf_kinds={certificate.leaf_kinds}; "
        f"leaf_count={certificate.leaf_count}; "
        f"zero_margin_leaf_count={certificate.zero_margin_leaf_count}; "
        f"unsupported_leaf_count={certificate.unsupported_leaf_count}; "
        f"recursive_exhaustion_certified={certificate.recursive_exhaustion_certified}; "
        f"missing={','.join(certificate.missing_obligations)}"
    )


def _minimum_pair_distance(positions: np.ndarray) -> float:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2:
        return float("inf")
    return float(
        min(
            np.linalg.norm(positions[j] - positions[i])
            for i in range(positions.shape[0])
            for j in range(i + 1, positions.shape[0])
        )
    )


def _generalized_fuchsian_shape_deviation_bound(
    branch: FuchsianShapeBranch,
    radius: float,
) -> float:
    radius = float(radius)
    if radius <= 0.0:
        return float("inf")
    bound = 0.0
    for index, coefficient in branch.coefficients.items():
        if index == branch.zero_index:
            continue
        exponent = branch.exponent(index)
        if not np.isfinite(exponent) or exponent <= 0.0:
            return float("inf")
        bound += float(np.linalg.norm(coefficient, ord=np.inf)) * radius**exponent
    return float(bound)


def _generalized_fuchsian_finite_row_tail_bound(
    branch: FuchsianShapeBranch,
    radius: float,
    *,
    retained_total_degree: int,
    lift_power: float,
    derivative_order: int,
) -> float:
    radius = float(radius)
    retained_total_degree = int(retained_total_degree)
    lift_power = float(lift_power)
    derivative_order = int(derivative_order)
    if not np.isfinite(radius) or radius <= 0.0:
        return float("inf")
    if retained_total_degree < 0 or derivative_order < 0:
        return float("inf")
    if not np.isfinite(lift_power):
        return float("inf")
    bound = 0.0
    for index, coefficient in branch.coefficients.items():
        if sum(index) <= retained_total_degree:
            continue
        exponent = branch.exponent(index) + lift_power
        if not np.isfinite(exponent):
            return float("inf")
        falling_factorial = 1.0
        for offset in range(derivative_order):
            falling_factorial *= exponent - offset
        bound += (
            abs(falling_factorial)
            * float(np.linalg.norm(coefficient, ord=np.inf))
            * radius ** (exponent - derivative_order)
        )
    return float(bound)


def _generalized_remainder_shell_majorant(
    *,
    remainder_ball_radius: float,
    effective_exponent: float,
    initial_radius: float,
    shell_contraction: float,
    analytic_disk_fraction: float,
    multiplier: float = 1.0,
) -> tuple[float, float]:
    remainder_ball_radius = float(remainder_ball_radius)
    effective_exponent = float(effective_exponent)
    initial_radius = float(initial_radius)
    shell_contraction = float(shell_contraction)
    analytic_disk_fraction = float(analytic_disk_fraction)
    multiplier = float(multiplier)
    if not (
        np.isfinite(remainder_ball_radius)
        and remainder_ball_radius >= 0.0
        and np.isfinite(effective_exponent)
        and np.isfinite(initial_radius)
        and initial_radius > 0.0
        and np.isfinite(shell_contraction)
        and 0.0 < shell_contraction < 1.0
        and np.isfinite(analytic_disk_fraction)
        and 0.0 < analytic_disk_fraction < 1.0
        and np.isfinite(multiplier)
        and multiplier >= 0.0
    ):
        raise ValueError("invalid generalized remainder shell-majorant inputs")
    radius_factor = (
        initial_radius * (1.0 + analytic_disk_fraction)
        if effective_exponent >= 0.0
        else initial_radius * (1.0 - analytic_disk_fraction)
    )
    return (
        float(multiplier * remainder_ball_radius * radius_factor**effective_exponent),
        float(shell_contraction**effective_exponent),
    )


def _generalized_remainder_majorant_certificate_from_supplied(
    majorant: SuppliedGeneralizedFuchsianAnalyticRemainderMajorantCertificate,
) -> GeneralizedFuchsianRemainderMajorantCertificate:
    return GeneralizedFuchsianRemainderMajorantCertificate(
        initial_radius=float(majorant.initial_radius),
        shell_contraction=float(majorant.shell_contraction),
        analytic_disk_fraction=float(majorant.analytic_disk_fraction),
        defect_bound=float(majorant.defect_bound),
        linear_inverse_bound=float(majorant.linear_inverse_bound),
        nonlinear_lipschitz_bound=float(majorant.nonlinear_lipschitz_bound),
        remainder_ball_radius=float(majorant.remainder_ball_radius),
        component_effective_exponents=tuple(
            (component, float(exponent))
            for component, exponent in sorted(
                majorant.component_effective_exponents.items(),
            )
        ),
        component_inputs=tuple(
            (component, PrimitiveCauchyTailInputCertificate.from_input(input_))
            for component, input_ in sorted(majorant.component_inputs.items())
        ),
    )


def _generalized_stop_endpoint_tail_bound(
    finite_row_budget: SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate,
    remainder_majorant: SuppliedGeneralizedFuchsianAnalyticRemainderMajorantCertificate,
) -> float:
    finite_row_tail = float(
        finite_row_budget.component_tail_bounds.get(
            "regularized_position_value",
            finite_row_budget.max_finite_row_tail_bound,
        )
    )
    remainder_input = remainder_majorant.component_inputs.get(
        "regularized_position_value",
    )
    if remainder_input is None:
        remainder_input = remainder_majorant.component_inputs.get("value")
    if remainder_input is None:
        return float("inf")
    return float(finite_row_tail + remainder_input.first_shell_tail_bound)


def _generalized_stop_residual_tail_bound(
    finite_row_budget: SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate,
    remainder_majorant: SuppliedGeneralizedFuchsianAnalyticRemainderMajorantCertificate,
) -> float:
    lifted_finite_row_tail = float(
        finite_row_budget.component_tail_bounds.get("lifted_residual", 0.0)
    )
    physical_finite_row_tail = float(
        finite_row_budget.component_tail_bounds.get("physical_residual", 0.0)
    )
    lifted_remainder_input = remainder_majorant.component_inputs.get(
        "lifted_residual",
    )
    physical_remainder_input = remainder_majorant.component_inputs.get(
        "physical_residual",
    )
    if lifted_remainder_input is None or physical_remainder_input is None:
        return float("inf")
    lifted_tail = float(
        lifted_finite_row_tail + lifted_remainder_input.first_shell_tail_bound
    )
    physical_tail = float(
        physical_finite_row_tail + physical_remainder_input.first_shell_tail_bound
    )
    return float(max(lifted_tail, physical_tail))


def _finite_nonempty_interval(interval: tuple[float, float]) -> bool:
    try:
        left, right = (float(interval[0]), float(interval[1]))
    except (TypeError, ValueError, IndexError):
        return False
    return bool(np.isfinite(left) and np.isfinite(right) and left < right)


def _fuchsian_projection_identity_residual_bound(
    branch: FiniteFuchsianLogBranch,
    tau: float,
) -> float:
    try:
        residual = branch.shape_projection_identity_residual(float(tau))
    except (AttributeError, TypeError, ValueError, FloatingPointError):
        return float("inf")
    return float(np.linalg.norm(residual, ord=np.inf))


def _checker_obligation_certified(checker_result: object, obligation: str) -> bool:
    for item in tuple(getattr(checker_result, "obligations", ()) or ()):
        if getattr(item, "obligation", None) == obligation:
            return (
                isinstance(item, CertificateCheckObligation)
                and item.certified is True
            )
    return False


def _sample_taus_inside_fuchsian_isolation(
    sample_taus: tuple[float, ...],
    isolation: FiniteFuchsianLogTotalCollisionIsolationCertificate,
) -> bool:
    try:
        radius = float(isolation.radius)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        sample_taus
        and np.isfinite(radius)
        and radius > 0.0
        and all(
            np.isfinite(tau)
            and tau != 0.0
            and abs(float(tau)) <= radius
            for tau in sample_taus
        )
        and any(tau < 0.0 for tau in sample_taus)
        and any(tau > 0.0 for tau in sample_taus)
    )


def _finite_fuchsian_log_stop_tail_bound(
    cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputs,
) -> float:
    try:
        values = tuple(
            float(input_.first_shell_tail_bound)
            for input_ in cauchy_inputs.component_inputs.values()
        )
    except (AttributeError, TypeError, ValueError):
        return float("inf")
    if not values:
        return float("inf")
    return float(max(values))


def _finite_fuchsian_log_cauchy_shell_inside_isolation(
    cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputs,
    isolation: FiniteFuchsianLogTotalCollisionIsolationCertificate,
) -> bool:
    if not isinstance(cauchy_inputs, FiniteFuchsianLogPrimitiveCauchyInputs):
        return False
    if not isinstance(isolation, FiniteFuchsianLogTotalCollisionIsolationCertificate):
        return False
    try:
        initial_radius = float(cauchy_inputs.initial_radius)
        isolation_radius = float(isolation.radius)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        cauchy_inputs.certified is True
        and isolation.certified is True
        and np.isfinite(initial_radius)
        and np.isfinite(isolation_radius)
        and 0.0 < initial_radius <= isolation_radius
    )


def _fuchsian_log_isolation_matches(
    supplied: FiniteFuchsianLogTotalCollisionIsolationCertificate,
    recomputed: FiniteFuchsianLogTotalCollisionIsolationCertificate,
    *,
    tolerance: float,
) -> bool:
    if not isinstance(supplied, FiniteFuchsianLogTotalCollisionIsolationCertificate):
        return False
    if not isinstance(recomputed, FiniteFuchsianLogTotalCollisionIsolationCertificate):
        return False
    if not (supplied.certified is True and recomputed.certified is True):
        return False
    for attribute in (
        "radius",
        "central_shape_pair_distance_floor",
        "shape_deviation_bound",
        "shape_pair_distance_floor",
    ):
        try:
            left = float(getattr(supplied, attribute))
            right = float(getattr(recomputed, attribute))
        except (AttributeError, TypeError, ValueError):
            return False
        scale = max(1.0, abs(left), abs(right))
        if abs(left - right) > tolerance * scale:
            return False
    return int(getattr(supplied, "dimension", -1)) == int(
        getattr(recomputed, "dimension", -2)
    )


def build_analytic_lemma_registry_for_finite_target_theorem(
    theorem_certificate: FiniteTargetCompletenessTheoremCertificate,
    *,
    critical_lemma_ids: tuple[str, ...] = FINITE_TARGET_CRITICAL_ANALYTIC_LEMMA_IDS,
) -> AnalyticLemmaRegistry:
    """Create audit rows for every finite-target analytic lemma dependency."""

    records = tuple(
        _analytic_lemma_audit_record(
            lemma,
            source_theorem=theorem_certificate.theorem_id,
        )
        for lemma in theorem_certificate.analytic_lemmas
    )
    return AnalyticLemmaRegistry(
        theorem_id=theorem_certificate.theorem_id,
        records=records,
        critical_lemma_ids=tuple(critical_lemma_ids),
    )


def _branch_tree_partition_kind_and_branches(
    partition: object,
) -> tuple[str, tuple[object, ...]]:
    if partition is None:
        return "missing_partition", ()
    if isinstance(partition, BranchEventTreeCertificate):
        return partition.tree_kind, partition.leaf_certificates
    branches = tuple(getattr(partition, "branches", ()) or ())
    if branches:
        return "simultaneous_close_pair_branch_partition", branches
    leaves = tuple(getattr(partition, "leaves", ()) or ())
    if leaves:
        return "ks_event_order_partition", leaves
    branch_leaves = tuple(getattr(partition, "branch_leaves", ()) or ())
    if branch_leaves:
        return "ambiguous_event_order_partition", branch_leaves
    return type(partition).__name__, ()


def _branch_tree_consumption_proof_entry_name(consumption_kind: str) -> str:
    if "event" in str(consumption_kind):
        return "finite_time_event_order_branch_union_consumption"
    return "finite_time_branch_union_consumption"


def _object_has_certified_proof_entry(value: object, name: str) -> bool:
    proof_ledger = getattr(value, "proof_ledger", None)
    return any(
        getattr(entry, "name", None) == name
        and getattr(entry, "certified", False) is True
        for entry in getattr(proof_ledger, "entries", ())
    )


def _object_proof_certified(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    return getattr(value, "proof_certified", False) is True


def _certified_leaf_count(
    leaf_certificates: tuple[object, ...],
    *,
    branch_union_atlas: object | None,
    branch_count: int,
    atlas_consumption_entry_certified: bool,
) -> int:
    explicit_count = sum(1 for certificate in leaf_certificates if _object_proof_certified(certificate))
    if explicit_count:
        return explicit_count
    if (
        branch_union_atlas is not None
        and getattr(branch_union_atlas, "proof_certified", False) is True
        and atlas_consumption_entry_certified
    ):
        return int(branch_count)
    return 0


def _branch_tree_leaf_response_source(
    branch_union_atlas: object | None,
    leaf_certificates: tuple[object, ...],
) -> str:
    if leaf_certificates:
        return "leaf_certificates"
    if branch_union_atlas is not None:
        return type(branch_union_atlas).__name__
    return "missing_leaf_certificates"


def _lemma(
    lemma_id: str,
    statement: str,
    proof_sketch: str,
    prerequisites: tuple[str, ...] = (),
    *,
    proof_mode: str = "declared_prose",
    externally_audited: bool = False,
    machine_checkable: bool = False,
    machine_checked: bool = False,
    internally_proven: bool = False,
) -> FiniteTargetAnalyticLemmaCertificate:
    return FiniteTargetAnalyticLemmaCertificate(
        lemma_id=lemma_id,
        statement=statement,
        proof_sketch=proof_sketch,
        prerequisites=tuple(prerequisites),
        proof_mode=str(proof_mode),
        externally_audited=bool(externally_audited),
        machine_checkable=bool(machine_checkable),
        machine_checked=bool(machine_checked),
        internally_proven=bool(internally_proven),
    )


def _analytic_lemma_audit_record(
    lemma: FiniteTargetAnalyticLemmaCertificate,
    *,
    source_theorem: str,
) -> AnalyticLemmaAuditRecord:
    if lemma.machine_checked:
        status = "machine_checked"
    elif lemma.externally_audited:
        status = "externally_audited"
    elif lemma.internally_proven:
        status = "internally_proven"
    elif lemma.machine_checkable:
        status = "machine_checkable"
    elif lemma.declared:
        status = "declared"
    else:
        status = "missing_statement"
    return AnalyticLemmaAuditRecord(
        lemma_id=lemma.lemma_id,
        statement=lemma.statement,
        hypotheses=tuple(lemma.prerequisites),
        source_theorem=source_theorem,
        proof_mode=lemma.proof_mode,
        status=status,
        normalization_translation=_analytic_lemma_normalization_translation(
            lemma.lemma_id
        ),
        checker_inputs=_analytic_lemma_checker_inputs(lemma.lemma_id),
        failure_modes=_analytic_lemma_failure_modes(lemma.lemma_id),
        audit_tier=_analytic_lemma_audit_tier(lemma.lemma_id),
    )


def _analytic_lemma_audit_tier(lemma_id: str) -> str:
    return FINITE_TARGET_ANALYTIC_LEMMA_TIER_BY_ID.get(
        lemma_id,
        "untiered_supporting_lemma",
    )


def _analytic_lemma_normalization_translation(lemma_id: str) -> str:
    if lemma_id in {
        "total_collision_requires_zero_angular_momentum",
        "total_collision_central_configuration_asymptotic",
        "cubic_time_total_collision_scaling",
    }:
        return (
            "translate physical total-collision time to centered mass "
            "coordinates and cubic time tau with q=tau^2 C+lower terms"
        )
    if "fuchsian" in lemma_id or "total_collision_germ" in lemma_id:
        return (
            "translate incoming cubic-time total-collision germs into finite "
            "Fuchsian/Fuchsian-log selector data"
        )
    if "binary" in lemma_id:
        return (
            "translate physical pair separation to the selected LC or KS "
            "binary chart with the third body separated"
        )
    if "taylor" in lemma_id or "collision_free" in lemma_id:
        return (
            "translate a positive pair-distance floor into an ordinary "
            "analytic Taylor chart domain"
        )
    return "uses the theorem statement's native Newtonian normalization"


def _analytic_lemma_checker_inputs(lemma_id: str) -> tuple[str, ...]:
    if lemma_id == "finite_fuchsian_log_stop_chart_for_admissible_entry_data":
        return (
            "finite_fuchsian_log_branch",
            "punctured_isolation_certificate",
            "primitive_cauchy_inputs",
            "serialized_total_collision_stop_chart_checker",
            "maximal_classical_stop_policy",
        )
    if lemma_id == "homothetic_total_collision_stop_chart_existence":
        return (
            "construct_homothetic_total_collision_branch",
            "homothetic_scalar_rouche_cauchy_majorant",
            "validated_atlas_from_homothetic_total_collision_branch",
            "TotalCollisionGeneralizedFuchsianStopChartCertificate",
            "check_total_collision_generalized_fuchsian_stop_chart",
            "construct_independent_validated_atlas_checked_chain",
        )
    if lemma_id == "total_collision_stop_chart_existence":
        return (
            "arbitrary_total_collision_germ_entry_to_stop_chart",
            "finite_generalized_fuchsian_entry_data",
            "punctured_isolation_certificate",
            "analytic_remainder_majorant_certificate",
            "TotalCollisionGeneralizedFuchsianStopChartCertificate",
            "check_total_collision_generalized_fuchsian_stop_chart",
            "newton_residual_ledger",
            "maximal_classical_stop_policy",
        )
    if lemma_id == "arbitrary_total_collision_germ_entry_to_stop_chart":
        return (
            "arbitrary_germ_entry_certificate",
            "admissible_entry_stop_chart_certificate",
            "supplied_generalized_fuchsian_stop_chart_certificate",
            "independent_generalized_stop_chart_checker",
            "maximal_classical_stop_policy",
        )
    if lemma_id == "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data":
        return (
            "zero_angular_total_collision_evidence",
            "reduced_hyperbolic_entry_certificate",
            "poincare_dulac_selector_rows",
            "analytic_stable_manifold_chart",
            "convergent_poincare_dulac_coordinates",
            "stable_coordinate_selector_constants",
            "analytic_inverse_coordinate_polydisc",
            "cauchy_estimates_for_analytic_remainder",
            "punctured_isolation_from_collision_free_central_shape",
            "supplied_generalized_fuchsian_entry_certificate",
            "supplied_generalized_fuchsian_finite_row_tail_budget",
            "supplied_generalized_fuchsian_analytic_remainder_majorant",
            "analytic_remainder_cauchy_majorant",
        )
    if lemma_id == "poincare_dulac_fuchsian_log_selector_completeness":
        return (
            "reduced_hyperbolic_entry_certificate",
            "positive_stable_eigenrate_spectrum",
            "poincare_domain_stable_normal_form",
            "finite_resonant_triangular_rows",
            "resonant_log_row_recurrence",
            "construct_stable_log_selector_chain",
            "fuchsian_log_branch_projection_from_mode_shapes",
        )
    if lemma_id == "reduced_hyperbolic_total_collision_entry":
        return (
            "binary_degenerate_total_collision_exclusion",
            "mcgehee_quotient_shape_convergence",
            "collision_free_central_configuration_target",
            "lagrange_reduced_spectrum",
            "ordered_euler_reduced_spectrum",
            "quotiented_shape_linearization",
            "stable_manifold_finite_reduced_length",
            "zero_angular_rotation_gauge",
            "orientation_gauge_limit",
        )
    if lemma_id == "binary_degenerate_total_collision_exclusion":
        return (
            "jacobi_coordinate_force_scale_bounds",
            "perturbed_kepler_collision_blow_up",
            "coordinate_energy_defect_forced_by_blow_up",
            "coordinate_angular_defect_forced_by_blow_up",
            "positive_jacobi_kepler_scale_ratio_floor",
        )
    if lemma_id == "three_body_painleve_no_noncollision_singularities":
        return (
            "pair_distance_floor_near_finite_endpoint",
            "bounded_positive_newtonian_potential",
            "energy_conservation_bounds_kinetic_energy",
            "compact_collision_free_phase_domain",
            "analytic_ode_continuation_theorem",
        )
    if lemma_id == "total_collision_requires_zero_angular_momentum":
        return (
            "centered_mass_moment_of_inertia",
            "sundman_angular_momentum_inequality",
            "finite_energy_identity_K_equals_H_plus_U",
            "positive_mass_potential_sqrt_inertia_bound",
            "angular_momentum_conservation",
        )
    if lemma_id == "total_collision_central_configuration_asymptotic":
        return (
            "zero_angular_total_collision_evidence",
            "binary_degenerate_total_collision_exclusion",
            "collision_free_shape_compactness",
            "mcgehee_shape_flow_monotonicity",
            "finite_three_body_central_configuration_quotient",
            "selected_oriented_shape_representative",
            "homogeneous_acceleration_limit",
        )
    if lemma_id == "cubic_time_total_collision_scaling":
        return (
            "selected_collision_free_normalized_shape_limit",
            "normalized_potential_limit_Gamma",
            "lagrange_jacobi_identity_I_second_derivative",
            "sundman_inertia_velocity_limit",
            "terminal_inward_monotonicity",
            "integrated_parabolic_inertia_scale",
        )
    if lemma_id == "finite_chart_chain_concatenation":
        return (
            "serialized_chart_certificates",
            "serialized_transition_certificates",
            "chart_chain_certificate_checker",
            "finite_branch_union_certificate_checker",
        )
    if lemma_id == "compact_collision_free_taylor_cover":
        return (
            "collision_free_compact_solution_segment",
            "pair_distance_floor",
            "ordinary_analytic_ode_local_existence",
            "finite_subcover_selection",
            "ordinary_taylor_chart_checker_for_supplied_charts",
        )
    if lemma_id == "binary_accumulation_implies_total_collision":
        return (
            "compact_time_interval",
            "convergent_binary_event_subsequence",
            "three_body_painleve_no_noncollision_singularities",
            "binary_collision_isolation",
            "two_pair_zero_implies_total_collision",
        )
    if lemma_id == "binary_collision_isolation":
        return (
            "analytic_LC_or_KS_regularized_branch",
            "nontrivial_selected_pair_lift",
            "separated_third_body",
            "analytic_identity_theorem",
        )
    if lemma_id == "all_pair_binary_regularization":
        return (
            "jacobi_pair_coordinates",
            "planar_levi_civita_square_map",
            "spatial_ks_quadratic_map",
            "regularized_energy_shell_H_minus_h_times_pair_radius",
            "KS_horizontal_gauge_constraint",
            "planar_lc_and_spatial_ks_chart_checkers_for_supplied_charts",
        )
    if lemma_id == "target_or_stop_dichotomy":
        return (
            "FINITE_TARGET_COMPLETENESS_OUTCOMES",
            "FINITE_TARGET_OUTCOMES",
            "FiniteTargetCompletenessTheoremCertificate.statement_certified",
            "FiniteTargetCompletenessTheoremCertificate.allowed_outcomes",
            "FiniteTargetAtlasOrStopCertificate.outcome_certified",
        )
    if "binary" in lemma_id:
        return ("selected_pair", "separated_third_body", "LC_or_KS_chart")
    if "collision_free" in lemma_id:
        return ("pair_distance_floor", "ordinary_cauchy_majorant")
    return ()


def _analytic_lemma_failure_modes(lemma_id: str) -> tuple[str, ...]:
    if lemma_id == "finite_fuchsian_log_stop_chart_for_admissible_entry_data":
        return (
            "supplied_entry_data_must_be_constructor_derived",
            "arbitrary_total_collision_germ_entry_not_derived",
        )
    if lemma_id == "homothetic_total_collision_stop_chart_existence":
        return (
            "homothetic_subcase_only",
            "requires_collision_free_central_configuration",
            "requires_exact_tail_or_scalar_majorant",
            "does_not_prove_arbitrary_total_collision_entry",
            "does_not_select_nonhomothetic_zero_angular_branch",
        )
    if lemma_id == "binary_degenerate_total_collision_exclusion":
        return (
            "requires_three_body_jacobi_cluster_alternative",
            "conditional_on_perturbed_kepler_blow_up",
            "excludes_only_binary_degenerate_shape_stratum",
            "does_not_prove_reduced_hyperbolic_entry",
            "does_not_construct_fuchsian_entry_data",
        )
    if lemma_id == "reduced_hyperbolic_total_collision_entry":
        return (
            "conditional_on_binary_degenerate_exclusion",
            "conditional_on_selected_central_quotient_limit",
            "requires_three_body_central_target_spectrum",
            "proves_oriented_shape_limit_not_fuchsian_entry_data",
            "does_not_prove_poincare_dulac_selector_completeness",
            "does_not_construct_total_stop_chart",
        )
    if lemma_id == "poincare_dulac_fuchsian_log_selector_completeness":
        return (
            "conditional_on_reduced_hyperbolic_entry",
            "requires_analytic_poincare_dulac_normal_form",
            "requires_supplied_mode_shapes_for_physical_projection",
            "proves_finite_selector_rows_not_remainder_majorant",
            "does_not_derive_arbitrary_total_collision_germ_entry_data",
            "does_not_construct_total_stop_chart",
        )
    if lemma_id == "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data":
        return (
            "pointwise_exact_germ_theorem_not_interval_constructor",
            "requires_choice_of_local_stable_chart_polydisc",
            "cauchy_constants_exist_but_are_not_uniform_in_input_boxes",
            "does_not_construct_recursive_branch_event_partition",
            "does_not_choose_zero_angular_continuation",
        )
    if lemma_id == "arbitrary_total_collision_germ_entry_to_stop_chart":
        return (
            "conditional_on_arbitrary_generalized_fuchsian_entry_data",
            "requires_cauchy_majorized_analytic_remainder",
            "requires_constructor_derived_punctured_isolation",
            "proves_stop_chart_bridge_not_entry_data",
            "does_not_choose_zero_angular_continuation",
        )
    if lemma_id == "total_collision_stop_chart_existence":
        return (
            "conditional_on_arbitrary_generalized_fuchsian_entry_data",
            "conditional_on_entry_to_stop_bridge",
            "maximal_classical_stop_policy_only",
            "does_not_choose_zero_angular_continuation",
        )
    if lemma_id in FINITE_TARGET_CRITICAL_ANALYTIC_LEMMA_IDS:
        return (
            "declared_prose_not_audited",
            "no_machine_checker_for_total_collision_entry_chain",
        )
    if lemma_id == "finite_chart_chain_concatenation":
        return (
            "checker_language_supports_only_serialized_finite_chains",
            "arbitrary_chart_existence_still_supplied_by_other_lemmas",
        )
    if lemma_id == "compact_collision_free_taylor_cover":
        return (
            "requires_segment_already_collision_free",
            "does_not_isolate_events_or_total_collision",
            "supplied_chart_coefficients_still_checked_separately",
        )
    if lemma_id == "binary_accumulation_implies_total_collision":
        return (
            "conditional_on_painleve_reduction",
            "conditional_on_binary_isolation",
            "does_not_construct_total_collision_stop_chart",
        )
    if lemma_id == "binary_collision_isolation":
        return (
            "conditional_on_all_pair_binary_regularization",
            "requires_nontrivial_regularized_binary_lift",
            "does_not_cover_multi_pair_collision_endpoint",
        )
    if lemma_id == "all_pair_binary_regularization":
        return (
            "requires_selected_pair_and_separated_third_body",
            "does_not_cover_total_collision_or_multi_pair_collapse",
            "spatial_case_requires_horizontal_KS_gauge",
        )
    if lemma_id == "three_body_painleve_no_noncollision_singularities":
        return (
            "requires_energy_conservation_for_finite_initial_data",
            "concludes_only_finite_endpoint_must_be_collision",
            "does_not_regularize_or_continue_collision_endpoint",
        )
    if lemma_id == "total_collision_requires_zero_angular_momentum":
        return (
            "requires_finite_energy_total_collision",
            "necessary_condition_only",
            "does_not_construct_total_collision_entry_or_continuation",
        )
    if lemma_id == "total_collision_central_configuration_asymptotic":
        return (
            "conditional_on_binary_degenerate_exclusion",
            "proves_quotient_central_shape_not_fuchsian_entry",
            "selected_oriented_representative_still_required",
            "does_not_prove_reduced_hyperbolic_selector_normal_form",
        )
    if lemma_id == "cubic_time_total_collision_scaling":
        return (
            "conditional_on_selected_collision_free_shape_limit",
            "conditional_on_normalized_potential_limit",
            "does_not_prove_shape_convergence",
            "does_not_construct_fuchsian_entry_data",
        )
    if lemma_id == "target_or_stop_dichotomy":
        return (
            "depends_on_predecessor_existence_lemmas",
            "unsupported_outcome_id_rejected_by_certificate",
        )
    if "binary" in lemma_id:
        return (
            "declared_prose_not_audited",
            "chart_hypotheses_must_be_verified_for_each_event",
        )
    return ("declared_prose_not_audited",)
