from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from three_body_symmetry.certificate_checker import (
    check_total_collision_fuchsian_stop_chart,
)
from three_body_symmetry.certificate_language import (
    TotalCollisionFuchsianStopChartCertificate,
    total_collision_fuchsian_stop_chart_certificate_from_branch,
)
from three_body_symmetry.fuchsian import (
    FiniteFuchsianLogBranch,
    certify_finite_fuchsian_log_compact_time_isolation,
    certify_finite_fuchsian_log_total_collision_isolation,
    construct_fuchsian_selector_continuation,
    derive_finite_fuchsian_log_branch_primitive_cauchy_inputs,
    linearized_acceleration_matrix,
    mass_inner_product,
)
from three_body_symmetry.branch_event_tree import (
    certify_supplied_branch_event_tree,
)
from three_body_symmetry.stratified_branch_tree import (
    AffineBoxDecisionFunctionSpec,
    AnalyticDecisionFunctionCertificate,
    EqualityStratumCertificate,
    EventOrderTieLeafCertificate,
    PolynomialDecisionFunctionSpec,
    RationalDecisionArrangementStratificationCertificate,
    RationalDecisionFunctionSpec,
    RationalDecisionStratificationCertificate,
    SelectorPolicyLeafCertificate,
    StratifiedBranchLeafCertificate,
    TaylorModelDecisionArrangementStratificationCertificate,
    TaylorModelDecisionFunctionSpec,
    TotalCollisionClusterLeafCertificate,
    certify_affine_halfspace_arrangement_recursive_consumption,
    certify_affine_halfspace_arrangement_stratified_branch_event_tree,
    derive_affine_halfspace_arrangement_child_consumptions,
    derive_affine_halfspace_arrangement_line_child_consumptions,
    derive_affine_halfspace_arrangement_point_child_consumptions,
    certify_affine_halfspace_3d_arrangement_recursive_consumption,
    certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree,
    derive_affine_halfspace_3d_arrangement_child_consumptions,
    derive_affine_halfspace_3d_arrangement_line_child_consumptions,
    derive_affine_halfspace_3d_arrangement_plane_child_consumptions,
    derive_affine_halfspace_3d_arrangement_point_child_consumptions,
    certify_affine_box_decision_arrangement_recursive_consumption,
    certify_affine_box_decision_arrangement_stratified_branch_event_tree,
    derive_affine_box_decision_arrangement_child_consumptions,
    certify_affine_halfspace_decision_recursive_consumption,
    certify_affine_halfspace_decision_stratified_branch_event_tree,
    derive_affine_halfspace_decision_child_consumptions,
    certify_affine_decision_arrangement_stratified_branch_event_tree,
    certify_affine_decision_stratified_branch_event_tree,
    certify_polynomial_decision_arrangement_recursive_consumption,
    certify_polynomial_decision_arrangement_stratified_branch_event_tree,
    derive_polynomial_decision_arrangement_child_consumptions,
    certify_polynomial_decision_recursive_consumption,
    certify_polynomial_decision_stratified_branch_event_tree,
    certify_taylor_model_decision_recursive_consumption,
    certify_taylor_model_decision_arrangement_recursive_consumption,
    certify_taylor_model_decision_arrangement_stratified_branch_event_tree,
    certify_taylor_model_decision_stratified_branch_event_tree,
    derive_taylor_model_decision_arrangement_child_consumptions,
    certify_rational_decision_arrangement_stratified_branch_event_tree,
    certify_rational_decision_stratified_branch_event_tree,
    certify_sturm_rational_decision_stratified_branch_event_tree,
    certify_sturm_rational_decision_arrangement_stratified_branch_event_tree,
    derive_taylor_model_decision_child_consumptions,
    certify_quadratic_decision_arrangement_stratified_branch_event_tree,
    certify_recursive_stratified_branch_event_consumption,
    certify_sturm_polynomial_decision_stratified_branch_event_tree,
    certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree,
    certify_stratified_branch_event_tree,
    certify_terminal_policy_stratified_branch_event_tree,
)
from three_body_symmetry.finite_target_completeness import (
    CONSTRUCTOR_DERIVED_RECURSIVE_SOURCE_SCOPES,
    FINITE_TARGET_COMPLETENESS_CHART_FAMILIES,
    FINITE_TARGET_COMPLETENESS_OUTCOMES,
    FINITE_TARGET_CRITICAL_ANALYTIC_LEMMA_IDS,
    FINITE_TARGET_TIER_A_ANALYTIC_LEMMA_IDS,
    FINITE_TARGET_TIER_B_ANALYTIC_LEMMA_IDS,
    SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS,
    ArbitraryIntervalInputPartitionGenerationCertificate,
    SupportedEventFunctionGrammarInput,
    SupportedEventFunctionStratificationGenerationCertificate,
    build_analytic_lemma_registry_for_finite_target_theorem,
    certify_affine_halfspace_arrangement_set_valued_constructor_completeness,
    certify_arbitrary_interval_input_partition_generation,
    certify_supported_event_function_stratification_generation,
    certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness,
    certify_constructor_pair_derived_recursive_stratified_set_valued_constructor_completeness,
    certify_finite_target_certificate_search_completeness,
    certify_finite_target_completeness_theorem,
    certify_finite_supplied_branch_tree_consumption,
    certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data,
    certify_supplied_generalized_fuchsian_analytic_remainder_majorant,
    certify_supplied_generalized_fuchsian_entry_data,
    certify_supplied_generalized_fuchsian_finite_row_tail_budget,
    certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data,
    certify_supplied_recursive_stratified_set_valued_constructor_completeness,
    certify_uniform_margin_set_valued_constructor_completeness,
    certify_uniform_margin_branch_refinement_termination,
    certify_validated_set_valued_constructor_completeness_theorem,
    recursive_constructor_source_scope,
    recursive_constructor_source_type,
)


def _supplied_fuchsian_log_entry_inputs():
    masses = np.ones(3)
    configuration = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ],
    )
    central_lambda = 1.0 / np.sqrt(3.0)
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    branch = FiniteFuchsianLogBranch(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=0.0,
        terms=(),
    )
    isolation = certify_finite_fuchsian_log_total_collision_isolation(
        branch,
        radius=0.025,
    )
    cauchy_inputs = derive_finite_fuchsian_log_branch_primitive_cauchy_inputs(
        branch,
        initial_radius=0.02,
        shell_contraction=0.5,
        analytic_disk_fraction=0.25,
        log_growth_factor=1.2,
        step_ratio_bounds={
            "value": 0.3,
            "first_jet": 0.3,
            "lifted_residual": 0.3,
            "physical_residual": 0.3,
        },
        retained_order_initials={
            "value": 12,
            "first_jet": 12,
            "lifted_residual": 12,
            "physical_residual": 12,
        },
        retained_order_increments={
            "value": 2,
            "first_jet": 2,
            "lifted_residual": 2,
            "physical_residual": 2,
        },
    )
    compact_isolation = certify_finite_fuchsian_log_compact_time_isolation(
        isolation,
        event_physical_time=0.0,
        compact_time_rate=1.0,
    )
    return branch, isolation, cauchy_inputs, compact_isolation


@lru_cache(maxsize=1)
def _supplied_generalized_fuchsian_branch():
    masses = np.array([1.0, 0.7, 1.4])
    configuration = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ],
    )
    configuration = configuration - np.average(configuration, axis=0, weights=masses)
    central_lambda = float(np.sum(masses) / (np.sqrt(3.0) ** 3))
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    derivative_matrix = linearized_acceleration_matrix(central_shape, masses)
    beta = float(
        sum(masses[i] * masses[j] for i in range(3) for j in range(i + 1, 3))
        / np.sum(masses) ** 2
    )
    shape_eigenvalues = (
        1.0 / 9.0 + (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
        1.0 / 9.0 - (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
    )
    powers = (
        2.0,
        *(
            0.5 * (-1.0 + np.sqrt(9.0 + 36.0 * eigenvalue))
            for eigenvalue in shape_eigenvalues
        ),
    )
    eigenvalues, eigenvectors = np.linalg.eig(derivative_matrix)
    fractional_modes = []
    for eigenvalue in shape_eigenvalues:
        eigenvector_index = int(
            np.argmin(
                np.abs(eigenvalues.real - eigenvalue)
                + np.abs(eigenvalues.imag)
            )
        )
        mode = eigenvectors[:, eigenvector_index].real.reshape(3, 2)
        mode /= np.sqrt(mass_inner_product(masses, mode, mode))
        fractional_modes.append(mode)
    continuation = construct_fuchsian_selector_continuation(
        masses=masses,
        central_shape=central_shape,
        powers=powers,
        incoming_selected_coefficients={
            (1, 0, 0): 0.02 * central_shape,
            (0, 1, 0): -0.018 * fractional_modes[0],
            (0, 0, 1): 0.022 * fractional_modes[1],
        },
        max_total_degree=4,
        scale_index=(1, 0, 0),
    )
    return continuation.incoming


@lru_cache(maxsize=1)
def _supplied_generalized_entry_certificate():
    return certify_supplied_generalized_fuchsian_entry_data(
        branch=_supplied_generalized_fuchsian_branch(),
        radius=0.035,
        sample_taus=(-0.03, 0.03),
        tolerance=1.0e-5,
        energy_tolerance=1.0e-8,
    )


@lru_cache(maxsize=4)
def _supplied_generalized_finite_row_tail_budget(retained_total_degree):
    return certify_supplied_generalized_fuchsian_finite_row_tail_budget(
        entry_certificate=_supplied_generalized_entry_certificate(),
        retained_total_degree=int(retained_total_degree),
        radius=0.03,
    )


@lru_cache(maxsize=1)
def _supplied_generalized_remainder_majorant():
    return certify_supplied_generalized_fuchsian_analytic_remainder_majorant(
        entry_certificate=_supplied_generalized_entry_certificate(),
        finite_row_budget=_supplied_generalized_finite_row_tail_budget(
            _supplied_generalized_fuchsian_branch().max_total_degree,
        ),
        defect_bound=1.0e-14,
        linear_inverse_bound=2.0,
        nonlinear_lipschitz_bound=0.1,
        component_effective_exponents={
            "value": 1.4,
            "first_jet": 0.4,
            "lifted_residual": 1.1,
            "physical_residual": 0.7,
            "regularized_position_value": 2.1,
        },
        step_ratio_bounds={
            "value": 0.2,
            "first_jet": 0.2,
            "lifted_residual": 0.2,
            "physical_residual": 0.2,
            "regularized_position_value": 0.2,
        },
        retained_order_initials={
            "value": 10,
            "first_jet": 10,
            "lifted_residual": 10,
            "physical_residual": 10,
            "regularized_position_value": 10,
        },
        retained_order_increments={
            "value": 2,
            "first_jet": 2,
            "lifted_residual": 2,
            "physical_residual": 2,
            "regularized_position_value": 2,
        },
        initial_radius=0.025,
        shell_contraction=0.5,
        analytic_disk_fraction=0.2,
    )


def test_constructor_source_scope_manifest_covers_emitted_stratified_sources():
    source = (
        Path(__file__).resolve().parents[1]
        / "three_body_symmetry"
        / "stratified_branch_tree.py"
    ).read_text()
    emitted_source_types = set(re.findall(r'source_type="([^"]+)"', source))
    manifest_source_types = set(CONSTRUCTOR_DERIVED_RECURSIVE_SOURCE_SCOPES)
    missing = sorted(emitted_source_types - manifest_source_types)
    extra = sorted(manifest_source_types - emitted_source_types)
    input_scope_ids = tuple(
        scope[0] for scope in CONSTRUCTOR_DERIVED_RECURSIVE_SOURCE_SCOPES.values()
    )
    finite_non_box_interval_input_scopes = {
        "finite_taylor_model_decision_interval_inputs_with_weierstrass_certificate",
        "finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate",
    }

    assert emitted_source_types
    assert not missing
    assert not extra
    assert len(input_scope_ids) == len(set(input_scope_ids))
    assert all(scope_id.startswith("finite_") for scope_id in input_scope_ids)
    assert all(
        scope_id.endswith("_interval_boxes")
        or scope_id in finite_non_box_interval_input_scopes
        for scope_id in input_scope_ids
    )


def test_recursive_constructor_source_scope_helper_exposes_manifest_scope():
    arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="manifest_helper_affine_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_total",
                coefficients=(0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
    )
    source_type, input_scope_id, detail, label = (
        recursive_constructor_source_scope(recursive)
    )

    assert recursive.constructor_source_type == "AffineDecisionArrangement"
    assert recursive.source_tree_kind == "ambiguous_event_order_partition"
    assert recursive.recursion_kind == "affine_decision_arrangement"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert recursive_constructor_source_type(recursive) == "AffineDecisionArrangement"
    assert source_type == "AffineDecisionArrangement"
    assert input_scope_id == "finite_affine_decision_arrangement_interval_boxes"
    assert "coefficient-derived roots" in detail
    assert label == "affine decision arrangement"


def test_recursive_stratified_consumption_exposes_well_founded_descent_counts():
    arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="descent_accounting_affine_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_total",
                coefficients=(0.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="ks_exit_minus_target",
                coefficients=(-0.5, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
    )
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    details = {obligation.obligation: obligation for obligation in search.obligations}

    assert recursive.recursive_leaf_count == 2
    assert recursive.strict_descent_edge_count == 2
    assert recursive.unresolved_descent_edge_count == 0
    assert recursive.descent_well_founded
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert recursive.certified
    assert details["recursive_set_valued_branch_partition_consumption"].certified
    assert "strict_descent_edge_count=2" in (
        details["recursive_set_valued_branch_partition_consumption"].detail
    )
    assert "descent_well_founded=True" in (
        details["recursive_set_valued_branch_partition_consumption"].detail
    )


def test_recursive_stratified_branch_event_theorem_note_preserves_scope_boundary():
    theorem_note = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "recursive-stratified-branch-event-consumption-theorem.md"
    ).read_text()

    assert "Recursive Stratified Branch/Event Consumption Theorem" in theorem_note
    assert "mu = (dimension, rank)" in theorem_note
    assert "strictly smaller" in theorem_note
    assert "No equality stratum is hulled back into an ambient box" in theorem_note
    assert "2D/3D oblique affine halfspace arrangements" in theorem_note
    assert "does not prove arbitrary interval-input recursive partition" in (
        theorem_note
    )
    assert "simultaneous first events" in theorem_note
    assert "total-collision clusters" in theorem_note
    assert "strict_descent_edge_count" in theorem_note
    assert "unresolved_descent_edge_count" in theorem_note


def test_pointwise_finite_target_atlas_or_stop_theorem_is_a_scaffold_until_proofs_are_audited():
    theorem = certify_finite_target_completeness_theorem(
        dimension=3,
        total_collision_policy_id="maximal_classical_stop",
    )

    assert theorem.theorem_id == "pointwise_finite_target_atlas_or_stop_completeness"
    assert theorem.statement_certified
    assert theorem.certified
    assert theorem.chart_families == FINITE_TARGET_COMPLETENESS_CHART_FAMILIES
    assert theorem.allowed_outcomes == FINITE_TARGET_COMPLETENESS_OUTCOMES
    assert "spatial_ks_binary" in theorem.chart_families
    assert "total_collision_stop" in theorem.chart_families
    assert theorem.total_collision_stop_chart_existence.declared
    assert theorem.homothetic_total_collision_stop_chart_existence.declared
    assert theorem.total_collision_zero_angular_momentum_condition.declared
    assert theorem.total_collision_central_configuration_asymptotic.declared
    assert theorem.cubic_time_total_collision_scaling.declared
    assert (
        theorem.finite_fuchsian_log_stop_chart_for_admissible_entry_data.declared
    )
    assert theorem.binary_degenerate_total_collision_exclusion.declared
    assert theorem.reduced_hyperbolic_total_collision_entry.declared
    assert theorem.poincare_dulac_fuchsian_log_selector_completeness.declared
    assert (
        theorem.arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data.declared
    )
    assert (
        theorem.arbitrary_total_collision_germ_finite_fuchsian_log_entry_data
        is theorem.arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data
    )
    assert theorem.arbitrary_total_collision_germ_entry_to_stop_chart.declared
    assert theorem.analytic_lemma_statements_declared
    assert not theorem.analytic_lemma_proofs_audited
    assert theorem.statement_declared
    assert theorem.scaffold_certified
    assert not theorem.proof_certified
    assert theorem.analytic_lemma_audit_blockers
    assert (
        "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
        in theorem.analytic_lemma_audit_blockers
    )
    assert "total_collision_requires_zero_angular_momentum" not in (
        theorem.missing_obligations
    )
    assert "total_collision_central_configuration_asymptotic" not in (
        theorem.missing_obligations
    )
    assert "cubic_time_total_collision_scaling" not in (
        theorem.missing_obligations
    )
    assert "finite_fuchsian_log_stop_chart_for_admissible_entry_data" not in (
        theorem.missing_obligations
    )
    assert "homothetic_total_collision_stop_chart_existence" not in (
        theorem.missing_obligations
    )
    assert "binary_degenerate_total_collision_exclusion" not in (
        theorem.missing_obligations
    )
    assert "reduced_hyperbolic_total_collision_entry" not in (
        theorem.missing_obligations
    )
    assert "poincare_dulac_fuchsian_log_selector_completeness" not in (
        theorem.missing_obligations
    )
    assert "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data" not in (
        theorem.missing_obligations
    )
    assert "arbitrary_total_collision_germ_entry_to_stop_chart" not in (
        theorem.missing_obligations
    )
    assert "total_collision_stop_chart_existence" not in theorem.missing_obligations
    assert "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data" in (
        theorem.total_collision_stop_chart_existence.prerequisites
    )
    assert "arbitrary_total_collision_germ_entry_to_stop_chart" in (
        theorem.total_collision_stop_chart_existence.prerequisites
    )
    assert "I K = I(H+U)" in (
        theorem.total_collision_zero_angular_momentum_condition.proof_sketch
    )
    assert "A(C)=-(2/9)C" in (
        theorem.total_collision_central_configuration_asymptotic.proof_sketch
    )
    assert "t=T+tau^3" in theorem.cubic_time_total_collision_scaling.proof_sketch
    assert "q=tau^2 S" in (
        theorem.finite_fuchsian_log_stop_chart_for_admissible_entry_data.proof_sketch
    )
    assert "Perturbed-Kepler" in (
        theorem.binary_degenerate_total_collision_exclusion.proof_sketch
    )
    assert "oriented normalized-shape limit" in (
        theorem.reduced_hyperbolic_total_collision_entry.proof_sketch
    )
    assert "Poincare-Dulac" in (
        theorem.poincare_dulac_fuchsian_log_selector_completeness.proof_sketch
    )
    assert "C, alpha, b_1,...,b_N" in (
        theorem.arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data.proof_sketch
    )
    assert "analytic remainder Cauchy majorant" in (
        theorem.arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data.proof_sketch
    )
    assert "q=tau^2 u(z)Q" in (
        theorem.homothetic_total_collision_stop_chart_existence.proof_sketch
    )
    assert "finite_chart_chain_concatenation" not in theorem.missing_obligations
    assert "total_collision_stop_chart_existence" in (
        theorem.finite_chart_chain_concatenation.prerequisites
    )
    assert "trim overlaps to nonempty common time slabs" in (
        theorem.finite_chart_chain_concatenation.proof_sketch
    )
    assert "target_or_stop_dichotomy" not in theorem.missing_obligations
    assert "binary accumulation forces total collision" in (
        theorem.target_or_stop_dichotomy.proof_sketch
    )
    assert "total_collision_stop_chart_existence" in (
        theorem.target_or_stop_dichotomy.prerequisites
    )
    assert "global_regime_exhaustion" not in theorem.missing_obligations
    assert "arbitrary_initial_data_partition_theorem" not in (
        theorem.missing_obligations
    )


def test_pointwise_finite_target_theorem_requires_maximal_classical_policy():
    theorem = certify_finite_target_completeness_theorem(
        dimension=3,
        total_collision_policy_id="selected_identity_selector",
    )

    assert not theorem.certified
    assert not theorem.proof_certified
    assert "maximal_classical_total_collision_policy" in (
        theorem.missing_obligations
    )


def test_pointwise_finite_target_theorem_rejects_forged_scope_parameters():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    forged_dimension = replace(theorem, dimension=99)
    forged_input_model = replace(theorem, input_model="arbitrary_interval_boxes")
    forged_policy = replace(
        theorem,
        total_collision_policy_id="selected_identity_selector",
    )
    forged_theorem_id = replace(theorem, theorem_id="spoofed_finite_target_theorem")

    assert not theorem.proof_certified
    assert forged_dimension.missing_obligations == (
        "finite_target_dimension_supported",
    )
    assert forged_input_model.missing_obligations == (
        "point_input_model_or_computable_name",
    )
    assert forged_policy.missing_obligations == (
        "maximal_classical_total_collision_policy",
    )
    assert forged_theorem_id.missing_obligations == (
        "pointwise_finite_target_theorem_id",
    )
    for forged in (
        forged_dimension,
        forged_input_model,
        forged_policy,
        forged_theorem_id,
    ):
        assert not forged.certified
        assert not forged.proof_certified


def test_total_collision_generalized_fuchsian_proof_note_spells_out_tc5_tc6():
    proof_note = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "total-collision-generalized-fuchsian-stop-proof.md"
    ).read_text()

    assert "## TC5. Stable Branches Admit Finite Generalized Fuchsian/Puiseux-Log Data" in proof_note
    assert "## TC6. Cauchy Estimates Produce a Finite Analytic Remainder Majorant" in proof_note
    assert "Poincare-Dulac resonant polynomial" in proof_note
    assert "(<m,alpha>-alpha_j)c_{j,m,ell}" in proof_note
    assert "constant row is the selector parameter" in proof_note
    assert "B D + q R_0 <= R_0" in proof_note
    assert "(C_0, Lambda, sigma, p_0, d)" in proof_note
    assert "ratio is `Lambda sigma^d`" in proof_note


def test_finite_target_analytic_lemma_registry_exposes_unaudited_prose():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    registry = theorem.analytic_lemma_registry
    rebuilt = build_analytic_lemma_registry_for_finite_target_theorem(theorem)

    assert registry == rebuilt
    assert registry.theorem_id == theorem.theorem_id
    assert len(registry.records) == len(theorem.analytic_lemmas)
    assert registry.audit_complete is False
    assert registry.missing_critical_lemma_ids == ()
    assert theorem.unaudited_analytic_lemma_ids == tuple(
        lemma.lemma_id for lemma in theorem.analytic_lemmas if not lemma.audited
    )
    assert theorem.tier_a_unaudited_analytic_lemma_ids == (
        FINITE_TARGET_TIER_A_ANALYTIC_LEMMA_IDS
    )
    expected_tier_b_blockers = FINITE_TARGET_TIER_B_ANALYTIC_LEMMA_IDS
    assert theorem.tier_b_unaudited_analytic_lemma_ids == expected_tier_b_blockers
    assert theorem.core_analytic_lemma_audit_blockers == (
        expected_tier_b_blockers
    )
    assert theorem.analytic_lemma_audit_blockers[: len(
        expected_tier_b_blockers
    )] == expected_tier_b_blockers
    assert set(theorem.analytic_lemma_audit_blockers) == set(
        theorem.unaudited_analytic_lemma_ids
    )
    expected_critical_blockers = FINITE_TARGET_CRITICAL_ANALYTIC_LEMMA_IDS
    assert theorem.critical_unaudited_analytic_lemma_ids == expected_critical_blockers

    records = {record.lemma_id: record for record in registry.records}
    for lemma_id in expected_critical_blockers:
        record = records[lemma_id]
        assert record.audit_tier == "tier_b_total_collision_entry_frontier"
        assert record.declared
        assert not record.audited
        assert record.hypotheses
        assert record.normalization_translation
        assert record.failure_modes

    assert records["binary_degenerate_total_collision_exclusion"].status == (
        "internally_proven"
    )
    assert records["binary_degenerate_total_collision_exclusion"].proof_mode == (
        "internal_jacobi_perturbed_kepler_blowup_proof"
    )
    assert records["binary_degenerate_total_collision_exclusion"].internally_supported
    assert records["binary_degenerate_total_collision_exclusion"].audit_tier == (
        "tier_b_total_collision_entry_frontier"
    )
    assert "perturbed_kepler_collision_blow_up" in (
        records["binary_degenerate_total_collision_exclusion"].checker_inputs
    )
    assert "positive_jacobi_kepler_scale_ratio_floor" in (
        records["binary_degenerate_total_collision_exclusion"].checker_inputs
    )
    assert "excludes_only_binary_degenerate_shape_stratum" in (
        records["binary_degenerate_total_collision_exclusion"].failure_modes
    )
    assert "does_not_construct_fuchsian_entry_data" in (
        records["binary_degenerate_total_collision_exclusion"].failure_modes
    )

    assert records["reduced_hyperbolic_total_collision_entry"].status == (
        "internally_proven"
    )
    assert records["reduced_hyperbolic_total_collision_entry"].proof_mode == (
        "internal_reduced_mcgehee_hyperbolic_entry_proof"
    )
    assert records["reduced_hyperbolic_total_collision_entry"].internally_supported
    assert records["reduced_hyperbolic_total_collision_entry"].audit_tier == (
        "tier_b_total_collision_entry_frontier"
    )
    assert "lagrange_reduced_spectrum" in (
        records["reduced_hyperbolic_total_collision_entry"].checker_inputs
    )
    assert "ordered_euler_reduced_spectrum" in (
        records["reduced_hyperbolic_total_collision_entry"].checker_inputs
    )
    assert "stable_manifold_finite_reduced_length" in (
        records["reduced_hyperbolic_total_collision_entry"].checker_inputs
    )
    assert "proves_oriented_shape_limit_not_fuchsian_entry_data" in (
        records["reduced_hyperbolic_total_collision_entry"].failure_modes
    )
    assert "does_not_prove_poincare_dulac_selector_completeness" in (
        records["reduced_hyperbolic_total_collision_entry"].failure_modes
    )

    assert records[
        "poincare_dulac_fuchsian_log_selector_completeness"
    ].status == "internally_proven"
    assert records[
        "poincare_dulac_fuchsian_log_selector_completeness"
    ].proof_mode == "internal_poincare_dulac_stable_selector_proof"
    assert records[
        "poincare_dulac_fuchsian_log_selector_completeness"
    ].internally_supported
    assert records[
        "poincare_dulac_fuchsian_log_selector_completeness"
    ].audit_tier == "tier_b_total_collision_entry_frontier"
    assert "construct_stable_log_selector_chain" in (
        records[
            "poincare_dulac_fuchsian_log_selector_completeness"
        ].checker_inputs
    )
    assert "finite_resonant_triangular_rows" in (
        records[
            "poincare_dulac_fuchsian_log_selector_completeness"
        ].checker_inputs
    )
    assert "proves_finite_selector_rows_not_remainder_majorant" in (
        records[
            "poincare_dulac_fuchsian_log_selector_completeness"
        ].failure_modes
    )
    assert "does_not_derive_arbitrary_total_collision_germ_entry_data" in (
        records[
            "poincare_dulac_fuchsian_log_selector_completeness"
        ].failure_modes
    )

    assert records[
        "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
    ].status == "internally_proven"
    assert records[
        "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
    ].proof_mode == "internal_stable_manifold_cauchy_majorant_entry_proof"
    assert records[
        "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
    ].internally_supported
    assert "analytic_stable_manifold_chart" in (
        records[
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
        ].checker_inputs
    )
    assert "cauchy_estimates_for_analytic_remainder" in (
        records[
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
        ].checker_inputs
    )
    assert "pointwise_exact_germ_theorem_not_interval_constructor" in (
        records[
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
        ].failure_modes
    )
    assert "does_not_construct_recursive_branch_event_partition" in (
        records[
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
        ].failure_modes
    )

    assert records[
        "arbitrary_total_collision_germ_entry_to_stop_chart"
    ].status == "internally_proven"
    assert records[
        "arbitrary_total_collision_germ_entry_to_stop_chart"
    ].proof_mode == "internal_generalized_entry_to_checked_stop_chart_proof"
    assert records[
        "arbitrary_total_collision_germ_entry_to_stop_chart"
    ].internally_supported
    assert "supplied_generalized_fuchsian_stop_chart_certificate" in (
        records[
            "arbitrary_total_collision_germ_entry_to_stop_chart"
        ].checker_inputs
    )
    assert "conditional_on_arbitrary_generalized_fuchsian_entry_data" in (
        records[
            "arbitrary_total_collision_germ_entry_to_stop_chart"
        ].failure_modes
    )
    assert "proves_stop_chart_bridge_not_entry_data" in (
        records[
            "arbitrary_total_collision_germ_entry_to_stop_chart"
        ].failure_modes
    )

    assert records["total_collision_stop_chart_existence"].status == (
        "internally_proven"
    )
    assert records["total_collision_stop_chart_existence"].proof_mode == (
        "internal_total_collision_stop_existence_from_entry_bridge"
    )
    assert records["total_collision_stop_chart_existence"].internally_supported
    assert "check_total_collision_generalized_fuchsian_stop_chart" in (
        records["total_collision_stop_chart_existence"].checker_inputs
    )
    assert "conditional_on_arbitrary_generalized_fuchsian_entry_data" in (
        records["total_collision_stop_chart_existence"].failure_modes
    )

    for lemma_id in FINITE_TARGET_TIER_A_ANALYTIC_LEMMA_IDS:
        assert records[lemma_id].audit_tier == "tier_a_classical_or_structural"
    assert records["compact_collision_free_taylor_cover"].status == (
        "internally_proven"
    )
    assert records["compact_collision_free_taylor_cover"].proof_mode == (
        "internal_analytic_compactness_proof"
    )
    assert records["compact_collision_free_taylor_cover"].internally_supported
    assert "finite_subcover_selection" in (
        records["compact_collision_free_taylor_cover"].checker_inputs
    )
    assert "requires_segment_already_collision_free" in (
        records["compact_collision_free_taylor_cover"].failure_modes
    )
    assert records["binary_accumulation_implies_total_collision"].status == (
        "internally_proven"
    )
    assert records["binary_accumulation_implies_total_collision"].proof_mode == (
        "internal_topological_collision_accumulation_proof"
    )
    assert records["binary_accumulation_implies_total_collision"].internally_supported
    assert "convergent_binary_event_subsequence" in (
        records["binary_accumulation_implies_total_collision"].checker_inputs
    )
    assert "conditional_on_binary_isolation" in (
        records["binary_accumulation_implies_total_collision"].failure_modes
    )
    assert records["binary_collision_isolation"].status == "internally_proven"
    assert records["binary_collision_isolation"].proof_mode == (
        "internal_analytic_identity_theorem_proof"
    )
    assert records["binary_collision_isolation"].internally_supported
    assert "analytic_identity_theorem" in (
        records["binary_collision_isolation"].checker_inputs
    )
    assert "conditional_on_all_pair_binary_regularization" in (
        records["binary_collision_isolation"].failure_modes
    )
    assert records["all_pair_binary_regularization"].status == (
        "internally_proven"
    )
    assert records["all_pair_binary_regularization"].proof_mode == (
        "internal_lc_ks_separated_binary_regularization_proof"
    )
    assert records["all_pair_binary_regularization"].internally_supported
    assert "spatial_ks_quadratic_map" in (
        records["all_pair_binary_regularization"].checker_inputs
    )
    assert "does_not_cover_total_collision_or_multi_pair_collapse" in (
        records["all_pair_binary_regularization"].failure_modes
    )
    assert records["three_body_painleve_no_noncollision_singularities"].status == (
        "internally_proven"
    )
    assert records["three_body_painleve_no_noncollision_singularities"].proof_mode == (
        "internal_energy_compactness_continuation_proof"
    )
    assert records["three_body_painleve_no_noncollision_singularities"].internally_supported
    assert "energy_conservation_bounds_kinetic_energy" in (
        records["three_body_painleve_no_noncollision_singularities"].checker_inputs
    )
    assert "does_not_regularize_or_continue_collision_endpoint" in (
        records[
            "three_body_painleve_no_noncollision_singularities"
        ].failure_modes
    )
    assert records["total_collision_requires_zero_angular_momentum"].status == (
        "internally_proven"
    )
    assert records["total_collision_requires_zero_angular_momentum"].proof_mode == (
        "internal_sundman_inequality_zero_angular_proof"
    )
    assert records["total_collision_requires_zero_angular_momentum"].internally_supported
    assert "sundman_angular_momentum_inequality" in (
        records["total_collision_requires_zero_angular_momentum"].checker_inputs
    )
    assert "necessary_condition_only" in (
        records["total_collision_requires_zero_angular_momentum"].failure_modes
    )
    assert records["total_collision_central_configuration_asymptotic"].status == (
        "internally_proven"
    )
    assert records["total_collision_central_configuration_asymptotic"].proof_mode == (
        "internal_mcgehee_shape_compactness_central_limit_proof"
    )
    assert records["total_collision_central_configuration_asymptotic"].internally_supported
    assert "binary_degenerate_total_collision_exclusion" in (
        records["total_collision_central_configuration_asymptotic"].hypotheses
    )
    assert "mcgehee_shape_flow_monotonicity" in (
        records["total_collision_central_configuration_asymptotic"].checker_inputs
    )
    assert "conditional_on_binary_degenerate_exclusion" in (
        records["total_collision_central_configuration_asymptotic"].failure_modes
    )
    assert "does_not_prove_reduced_hyperbolic_selector_normal_form" in (
        records["total_collision_central_configuration_asymptotic"].failure_modes
    )
    assert records["cubic_time_total_collision_scaling"].status == (
        "internally_proven"
    )
    assert records["cubic_time_total_collision_scaling"].proof_mode == (
        "internal_lagrange_jacobi_parabolic_scale_proof"
    )
    assert records["cubic_time_total_collision_scaling"].internally_supported
    assert "lagrange_jacobi_identity_I_second_derivative" in (
        records["cubic_time_total_collision_scaling"].checker_inputs
    )
    assert "conditional_on_selected_collision_free_shape_limit" in (
        records["cubic_time_total_collision_scaling"].failure_modes
    )
    assert "does_not_construct_fuchsian_entry_data" in (
        records["cubic_time_total_collision_scaling"].failure_modes
    )
    assert records[
        "finite_fuchsian_log_stop_chart_for_admissible_entry_data"
    ].status == "machine_checkable"
    assert records[
        "finite_fuchsian_log_stop_chart_for_admissible_entry_data"
    ].proof_mode == "machine_checked_supplied_fuchsian_log_stop_chart"
    assert records[
        "finite_fuchsian_log_stop_chart_for_admissible_entry_data"
    ].audited is False
    assert "serialized_total_collision_stop_chart_checker" in (
        records[
            "finite_fuchsian_log_stop_chart_for_admissible_entry_data"
        ].checker_inputs
    )
    assert "arbitrary_total_collision_germ_entry_not_derived" in (
        records[
            "finite_fuchsian_log_stop_chart_for_admissible_entry_data"
        ].failure_modes
    )
    assert records[
        "homothetic_total_collision_stop_chart_existence"
    ].status == "machine_checkable"
    assert records[
        "homothetic_total_collision_stop_chart_existence"
    ].proof_mode == "machine_checked_homothetic_total_collision_stop_chart"
    assert not records["homothetic_total_collision_stop_chart_existence"].audited
    assert "construct_homothetic_total_collision_branch" in (
        records["homothetic_total_collision_stop_chart_existence"].checker_inputs
    )
    assert "check_total_collision_generalized_fuchsian_stop_chart" in (
        records["homothetic_total_collision_stop_chart_existence"].checker_inputs
    )
    assert "homothetic_subcase_only" in (
        records["homothetic_total_collision_stop_chart_existence"].failure_modes
    )
    assert "does_not_prove_arbitrary_total_collision_entry" in (
        records["homothetic_total_collision_stop_chart_existence"].failure_modes
    )
    assert records["finite_chart_chain_concatenation"].status == "machine_checkable"
    assert records["finite_chart_chain_concatenation"].proof_mode == (
        "machine_checked_chart_chain_certificate"
    )
    assert not records["finite_chart_chain_concatenation"].audited
    assert "chart_chain_certificate_checker" in (
        records["finite_chart_chain_concatenation"].checker_inputs
    )
    assert records["target_or_stop_dichotomy"].status == "machine_checkable"
    assert records["target_or_stop_dichotomy"].proof_mode == (
        "machine_checked_outcome_partition"
    )
    assert not records["target_or_stop_dichotomy"].audited
    assert "FINITE_TARGET_COMPLETENESS_OUTCOMES" in (
        records["target_or_stop_dichotomy"].checker_inputs
    )
    assert "depends_on_predecessor_existence_lemmas" in (
        records["target_or_stop_dichotomy"].failure_modes
    )

    assert "finite_generalized_fuchsian_entry_data" in (
        records["total_collision_stop_chart_existence"].checker_inputs
    )
    assert "arbitrary_total_collision_germ_entry_to_stop_chart" in (
        records["total_collision_stop_chart_existence"].checker_inputs
    )
    assert "poincare_dulac_selector_rows" in (
        records[
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
        ].checker_inputs
    )
    assert "supplied_generalized_fuchsian_entry_certificate" in (
        records[
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
        ].checker_inputs
    )
    assert "supplied_generalized_fuchsian_finite_row_tail_budget" in (
        records[
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
        ].checker_inputs
    )
    assert "supplied_generalized_fuchsian_analytic_remainder_majorant" in (
        records[
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
        ].checker_inputs
    )
    assert "analytic_remainder_cauchy_majorant" in (
        records[
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data"
        ].checker_inputs
    )
    assert "serialized_total_collision_stop_chart_checker" in (
        records[
            "finite_fuchsian_log_stop_chart_for_admissible_entry_data"
        ].checker_inputs
    )


def test_supplied_generalized_fuchsian_entry_data_covers_fractional_branch_only():
    certificate = _supplied_generalized_entry_certificate()

    assert certificate.certified
    assert certificate.proof_certified
    assert certificate.shape_pair_distance_floor > 0.0
    assert certificate.max_lifted_residual < 1.0e-8
    assert certificate.max_angular_momentum < 1.0e-5
    assert certificate.max_energy_gap < 1.0e-8
    assert not certificate.arbitrary_entry_theorem_claimed
    assert not certificate.stop_chart_theorem_claimed
    obligations = {obligation.obligation: obligation for obligation in certificate.obligations}
    assert obligations["nonresonant_fuchsian_recurrence"].certified
    assert obligations["finite_energy_scale_row"].certified
    assert obligations["generalized_fuchsian_punctured_isolation"].certified
    assert obligations["generalized_fuchsian_tail_majorant_not_claimed"].certified
    assert not obligations["generalized_fuchsian_tail_majorant_not_claimed"].required


def test_supplied_generalized_fuchsian_entry_rejects_nonisolated_radius():
    branch = _supplied_generalized_fuchsian_branch()
    certificate = certify_supplied_generalized_fuchsian_entry_data(
        branch=branch,
        radius=5.0,
        sample_taus=(-0.03, 0.03),
        tolerance=1.0e-5,
        energy_tolerance=1.0e-8,
    )

    assert not certificate.certified
    assert "generalized_fuchsian_punctured_isolation" in (
        certificate.missing_obligations
    )


def test_supplied_generalized_fuchsian_entry_keeps_samples_diagnostic():
    branch = _supplied_generalized_fuchsian_branch()
    certificate = certify_supplied_generalized_fuchsian_entry_data(
        branch=branch,
        radius=0.035,
        sample_taus=(0.0, 0.03),
        tolerance=1.0e-5,
        energy_tolerance=1.0e-8,
    )
    obligations = {obligation.obligation: obligation for obligation in certificate.obligations}

    assert certificate.certified
    assert certificate.missing_obligations == ()
    assert obligations["generalized_fuchsian_lifted_residual_from_recurrence"].certified
    assert not obligations[
        "diagnostic_sample_taus_inside_generalized_fuchsian_radius"
    ].required
    assert not obligations[
        "diagnostic_sample_taus_inside_generalized_fuchsian_radius"
    ].certified
    assert not obligations[
        "diagnostic_sampled_generalized_fuchsian_lifted_residual"
    ].required
    assert not obligations[
        "diagnostic_sampled_generalized_fuchsian_lifted_residual"
    ].certified


def test_supplied_generalized_fuchsian_finite_row_tail_budget_tracks_omitted_rows():
    branch = _supplied_generalized_fuchsian_branch()
    entry = _supplied_generalized_entry_certificate()
    partial = _supplied_generalized_finite_row_tail_budget(2)
    full = _supplied_generalized_finite_row_tail_budget(branch.max_total_degree)

    assert entry.certified
    assert partial.certified
    assert partial.proof_certified
    assert partial.missing_obligations == ()
    assert partial.omitted_indices
    assert partial.max_finite_row_tail_bound > 0.0
    assert partial.component_tail_bounds["shape_value"] > 0.0
    assert partial.component_tail_bounds["regularized_position_first_jet"] > 0.0
    assert not partial.analytic_remainder_tail_claimed
    assert not partial.stop_chart_theorem_claimed
    obligations = {obligation.obligation: obligation for obligation in partial.obligations}
    assert obligations["constructor_supplied_generalized_entry_certificate"].certified
    assert obligations["finite_row_tail_bounds_finite"].certified
    assert obligations["analytic_remainder_cauchy_majorant_not_claimed"].certified
    assert not obligations["analytic_remainder_cauchy_majorant_not_claimed"].required
    assert full.certified
    assert full.omitted_indices == ()
    assert full.max_finite_row_tail_bound == 0.0
    assert all(value == 0.0 for value in full.component_tail_bounds.values())


def test_supplied_generalized_fuchsian_finite_row_tail_budget_requires_entry_certificate():
    branch = _supplied_generalized_fuchsian_branch()
    bad_entry = certify_supplied_generalized_fuchsian_entry_data(
        branch=branch,
        radius=5.0,
        sample_taus=(-0.03, 0.03),
        tolerance=1.0e-5,
        energy_tolerance=1.0e-8,
    )
    budget = certify_supplied_generalized_fuchsian_finite_row_tail_budget(
        entry_certificate=bad_entry,
        retained_total_degree=2,
        radius=0.03,
    )

    assert not bad_entry.certified
    assert not budget.certified
    assert "constructor_supplied_generalized_entry_certificate" in (
        budget.missing_obligations
    )
    try:
        certify_supplied_generalized_fuchsian_finite_row_tail_budget(
            entry_certificate=True,
            retained_total_degree=2,
        )
    except TypeError as error:
        assert "SuppliedGeneralizedFuchsianEntryCertificate" in str(error)
    else:
        raise AssertionError("raw boolean entry certificate was accepted")


def test_supplied_generalized_fuchsian_analytic_remainder_majorant_closes_contraction():
    branch = _supplied_generalized_fuchsian_branch()
    _ = _supplied_generalized_entry_certificate()
    _ = _supplied_generalized_finite_row_tail_budget(branch.max_total_degree)
    majorant = _supplied_generalized_remainder_majorant()

    assert majorant.certified
    assert majorant.proof_certified
    assert majorant.missing_obligations == ()
    assert majorant.contraction_factor == 0.2
    assert abs(majorant.banach_contraction_slack - 0.8) < 1.0e-15
    assert majorant.self_map_bound <= majorant.remainder_ball_radius * (
        1.0 + 1.0e-12
    )
    assert majorant.banach_self_map_margin >= -1.0e-15
    assert majorant.retained_weight_cutoff == branch.max_total_degree
    assert majorant.first_omitted_weight == branch.max_total_degree + 1
    assert majorant.cauchy_polydisc_certified
    assert majorant.polydisc_source_scope == "pointwise_supplied_entry"
    assert majorant.pointwise_supplied_entry_scope_certified
    assert not majorant.uniform_interval_box_constants_claimed
    assert majorant.analytic_remainder_tail_claimed
    assert not majorant.arbitrary_entry_theorem_claimed
    assert not majorant.stop_chart_theorem_claimed
    for component in ("value", "first_jet", "lifted_residual", "physical_residual"):
        component_input = majorant.component_input(component)
        assert component_input.certified
        assert component_input.first_shell_tail_bound > 0.0
        assert component_input.shell_ratio < 1.0
    obligations = {obligation.obligation: obligation for obligation in majorant.obligations}
    assert obligations["retained_weight_cutoff_from_finite_rows"].certified
    assert obligations["closed_lifted_cauchy_polydisc_from_supplied_entry"].certified
    assert obligations["banach_contraction_factor"].certified
    assert "slack=1-q" in obligations["banach_contraction_factor"].detail
    assert obligations["banach_self_map_ball"].certified
    assert "margin=R-(B*D+q*R)" in obligations["banach_self_map_ball"].detail
    assert obligations[
        "primitive_cauchy_inputs_from_remainder_majorant"
    ].certified
    assert obligations["pointwise_supplied_entry_scope_only"].certified
    assert obligations["uniform_interval_box_constants_not_claimed"].certified
    assert not obligations["uniform_interval_box_constants_not_claimed"].required
    assert obligations["arbitrary_total_collision_entry_not_claimed"].certified
    assert not obligations["arbitrary_total_collision_entry_not_claimed"].required


def test_supplied_generalized_fuchsian_analytic_remainder_majorant_rejects_bad_contraction():
    branch = _supplied_generalized_fuchsian_branch()
    entry = _supplied_generalized_entry_certificate()
    finite_rows = _supplied_generalized_finite_row_tail_budget(branch.max_total_degree)
    majorant = certify_supplied_generalized_fuchsian_analytic_remainder_majorant(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        defect_bound=1.0e-10,
        linear_inverse_bound=3.0,
        nonlinear_lipschitz_bound=0.5,
        remainder_ball_radius=1.0e-8,
        component_effective_exponents={"value": 1.0},
        step_ratio_bounds={"value": 0.25},
        retained_order_initials={"value": 8},
        retained_order_increments={"value": 2},
        initial_radius=0.025,
    )

    assert not majorant.certified
    assert "banach_contraction_factor" in majorant.missing_obligations
    assert majorant.banach_contraction_slack < 0.0

    unit_contraction = certify_supplied_generalized_fuchsian_analytic_remainder_majorant(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        defect_bound=1.0e-10,
        linear_inverse_bound=2.0,
        nonlinear_lipschitz_bound=0.5,
        remainder_ball_radius=1.0e-8,
        component_effective_exponents={"value": 1.0},
        step_ratio_bounds={"value": 0.25},
        retained_order_initials={"value": 8},
        retained_order_increments={"value": 2},
        initial_radius=0.025,
    )

    assert not unit_contraction.certified
    assert unit_contraction.contraction_factor == 1.0
    assert unit_contraction.banach_contraction_slack == 0.0
    assert "banach_contraction_factor" in unit_contraction.missing_obligations
    try:
        certify_supplied_generalized_fuchsian_analytic_remainder_majorant(
            entry_certificate=True,
            finite_row_budget=finite_rows,
            defect_bound=1.0e-10,
            linear_inverse_bound=2.0,
            nonlinear_lipschitz_bound=0.1,
            component_effective_exponents={"value": 1.0},
            step_ratio_bounds={"value": 0.25},
            retained_order_initials={"value": 8},
            retained_order_increments={"value": 2},
        )
    except TypeError as error:
        assert "SuppliedGeneralizedFuchsianEntryCertificate" in str(error)
    else:
        raise AssertionError("raw boolean entry certificate was accepted")


def test_supplied_generalized_fuchsian_analytic_remainder_majorant_rejects_bad_self_map():
    branch = _supplied_generalized_fuchsian_branch()
    entry = _supplied_generalized_entry_certificate()
    finite_rows = _supplied_generalized_finite_row_tail_budget(branch.max_total_degree)
    majorant = certify_supplied_generalized_fuchsian_analytic_remainder_majorant(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        defect_bound=1.0e-7,
        linear_inverse_bound=2.0,
        nonlinear_lipschitz_bound=0.1,
        remainder_ball_radius=1.0e-8,
        component_effective_exponents={"value": 1.0},
        step_ratio_bounds={"value": 0.25},
        retained_order_initials={"value": 8},
        retained_order_increments={"value": 2},
        initial_radius=0.025,
    )

    assert not majorant.certified
    assert majorant.banach_contraction_slack > 0.0
    assert majorant.banach_self_map_margin < 0.0
    assert "banach_self_map_ball" in majorant.missing_obligations
    obligations = {obligation.obligation: obligation for obligation in majorant.obligations}
    assert obligations["banach_contraction_factor"].certified
    assert not obligations["banach_self_map_ball"].certified
    assert "margin=R-(B*D+q*R)" in obligations["banach_self_map_ball"].detail


def test_supplied_generalized_fuchsian_cached_inputs_are_reused():
    branch = _supplied_generalized_fuchsian_branch()

    assert _supplied_generalized_entry_certificate() is (
        _supplied_generalized_entry_certificate()
    )
    assert _supplied_generalized_finite_row_tail_budget(2) is (
        _supplied_generalized_finite_row_tail_budget(2)
    )
    assert _supplied_generalized_finite_row_tail_budget(branch.max_total_degree) is (
        _supplied_generalized_finite_row_tail_budget(branch.max_total_degree)
    )
    assert _supplied_generalized_remainder_majorant() is (
        _supplied_generalized_remainder_majorant()
    )


@lru_cache(maxsize=1)
def _supplied_generalized_stop_chart_inputs():
    entry = _supplied_generalized_entry_certificate()
    finite_rows = _supplied_generalized_finite_row_tail_budget(
        _supplied_generalized_fuchsian_branch().max_total_degree,
    )
    majorant = _supplied_generalized_remainder_majorant()
    return entry, finite_rows, majorant


def test_supplied_generalized_fuchsian_stop_chart_consumes_remainder_majorant():
    entry, finite_rows, majorant = _supplied_generalized_stop_chart_inputs()
    stop_chart = certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        remainder_majorant=majorant,
        residual_tolerance=5.0e3,
        angular_momentum_tolerance=1.0e-5,
    )

    assert stop_chart.certified
    assert stop_chart.proof_certified
    assert stop_chart.missing_obligations == ()
    assert stop_chart.stop_chart_theorem_claimed
    assert not stop_chart.arbitrary_entry_theorem_claimed
    assert stop_chart.independent_serialized_checker_claimed
    assert stop_chart.stop_chart_certificate is not None
    assert stop_chart.independent_checker_result is not None
    assert stop_chart.independent_checker_result.certified
    assert "interval_generalized_fuchsian_lifted_residual_on_punctured_shells" not in (
        stop_chart.independent_checker_result.missing_obligations
    )
    assert "interval_generalized_zero_angular_momentum_on_punctured_shells" not in (
        stop_chart.independent_checker_result.missing_obligations
    )
    assert (
        "interval_generalized_center_of_mass_and_linear_momentum_on_punctured_shells"
        not in stop_chart.independent_checker_result.missing_obligations
    )
    assert stop_chart.tau_interval[0] < 0.0 < stop_chart.tau_interval[1]
    assert stop_chart.physical_time_interval[0] < 0.0 < stop_chart.physical_time_interval[1]
    assert stop_chart.residual_tail_bound >= 0.0
    assert stop_chart.endpoint_position_tail_bound >= 0.0
    obligations = {obligation.obligation: obligation for obligation in stop_chart.obligations}
    assert obligations["constructor_supplied_analytic_remainder_majorant"].certified
    assert obligations["generalized_stop_residual_tail_within_tolerance"].certified
    assert obligations["generalized_stop_endpoint_collapse_tail"].certified
    assert obligations[
        "independent_generalized_total_collision_stop_chart_checker"
    ].certified
    assert obligations[
        "independent_generalized_total_collision_stop_chart_checker"
    ].required


def test_supplied_generalized_fuchsian_stop_chart_rejects_loose_residual_budget():
    entry, finite_rows, majorant = _supplied_generalized_stop_chart_inputs()
    stop_chart = certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        remainder_majorant=majorant,
        residual_tolerance=1.0e-15,
        angular_momentum_tolerance=1.0e-5,
    )

    assert not stop_chart.certified
    assert "generalized_stop_residual_tail_within_tolerance" in (
        stop_chart.missing_obligations
    )
    try:
        certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data(
            entry_certificate=True,
            finite_row_budget=finite_rows,
            remainder_majorant=majorant,
        )
    except TypeError as error:
        assert "SuppliedGeneralizedFuchsianEntryCertificate" in str(error)
    else:
        raise AssertionError("raw boolean entry certificate was accepted")


def test_supplied_generalized_fuchsian_certificates_reject_spoofed_obligation_ledgers():
    fake_obligation = SimpleNamespace(
        obligation="fake_generalized_fuchsian_obligation",
        certified=True,
        required=True,
    )
    entry, finite_rows, majorant = _supplied_generalized_stop_chart_inputs()
    stop_chart = certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        remainder_majorant=majorant,
        residual_tolerance=5.0e3,
        angular_momentum_tolerance=1.0e-5,
    )

    spoofed_entry = replace(entry, obligations=(fake_obligation,))
    spoofed_finite_rows = replace(finite_rows, obligations=(fake_obligation,))
    spoofed_majorant = replace(majorant, obligations=(fake_obligation,))
    spoofed_stop_chart = replace(stop_chart, obligations=(fake_obligation,))
    truthy_checker_stop_chart = replace(
        stop_chart,
        independent_checker_result=SimpleNamespace(certified="yes"),
    )

    assert entry.proof_certified
    assert finite_rows.proof_certified
    assert majorant.proof_certified
    assert stop_chart.proof_certified
    assert not spoofed_entry.proof_certified
    assert not spoofed_finite_rows.proof_certified
    assert not spoofed_majorant.proof_certified
    assert not spoofed_stop_chart.proof_certified
    assert "supplied_generalized_fuchsian_entry_data_obligation_type" in (
        spoofed_entry.missing_obligations
    )
    assert "supplied_generalized_fuchsian_finite_row_tail_budget_obligation_type" in (
        spoofed_finite_rows.missing_obligations
    )
    assert (
        "supplied_generalized_fuchsian_analytic_remainder_majorant_obligation_type"
        in spoofed_majorant.missing_obligations
    )
    assert "supplied_generalized_fuchsian_stop_chart_obligation_type" in (
        spoofed_stop_chart.missing_obligations
    )
    assert not truthy_checker_stop_chart.independent_serialized_checker_claimed


def test_supplied_finite_fuchsian_log_stop_chart_consumes_constructor_inputs_only():
    branch, isolation, cauchy_inputs, compact_isolation = (
        _supplied_fuchsian_log_entry_inputs()
    )
    certificate = (
        certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data(
            branch=branch,
            isolation=isolation,
            cauchy_inputs=cauchy_inputs,
            compact_isolation=compact_isolation,
            sample_taus=(-0.02, 0.02),
            tolerance=1.0e-6,
        )
    )

    assert certificate.certified
    assert certificate.proof_certified
    assert certificate.missing_obligations == ()
    assert certificate.independent_checker_result.certified
    assert certificate.stop_chart_certificate.primitive_cauchy_inputs is not None
    assert certificate.max_projection_identity_residual < 1.0e-12
    assert certificate.max_angular_momentum == 0.0
    assert not certificate.arbitrary_entry_theorem_claimed
    obligations = {obligation.obligation: obligation for obligation in certificate.obligations}
    assert obligations["arbitrary_total_collision_entry_not_claimed"].certified
    checker_obligations = {
        obligation.obligation: obligation
        for obligation in certificate.independent_checker_result.obligations
    }
    assert checker_obligations["fuchsian_primitive_cauchy_inputs_certify"].certified
    assert checker_obligations[
        "fuchsian_tail_bound_covers_primitive_cauchy_tail"
    ].certified
    assert checker_obligations[
        "fuchsian_primitive_cauchy_residual_tail_within_tolerance"
    ].certified
    round_trip_stop_chart = TotalCollisionFuchsianStopChartCertificate.from_dict(
        certificate.stop_chart_certificate.to_dict(),
    )
    round_trip_check = check_total_collision_fuchsian_stop_chart(
        round_trip_stop_chart,
    )
    assert round_trip_stop_chart.primitive_cauchy_inputs is not None
    assert round_trip_check.certified


def test_supplied_finite_fuchsian_log_stop_chart_rejects_spoofed_obligation_ledger():
    branch, isolation, cauchy_inputs, compact_isolation = (
        _supplied_fuchsian_log_entry_inputs()
    )
    certificate = (
        certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data(
            branch=branch,
            isolation=isolation,
            cauchy_inputs=cauchy_inputs,
            compact_isolation=compact_isolation,
            sample_taus=(-0.02, 0.02),
            tolerance=1.0e-6,
        )
    )
    spoofed = replace(
        certificate,
        obligations=(
            SimpleNamespace(
                obligation="fake_finite_fuchsian_log_stop_obligation",
                certified=True,
                required=True,
            ),
        ),
    )

    assert certificate.proof_certified
    assert not spoofed.proof_certified
    assert "supplied_finite_fuchsian_log_stop_chart_obligation_type" in (
        spoofed.missing_obligations
    )


def test_supplied_finite_fuchsian_log_stop_chart_rejects_fake_constructor_inputs():
    branch, isolation, cauchy_inputs, compact_isolation = (
        _supplied_fuchsian_log_entry_inputs()
    )

    fake_isolation = SimpleNamespace(
        certified=True,
        radius=isolation.radius,
        central_shape_pair_distance_floor=isolation.central_shape_pair_distance_floor,
        shape_deviation_bound=isolation.shape_deviation_bound,
        shape_pair_distance_floor=isolation.shape_pair_distance_floor,
        dimension=isolation.dimension,
    )
    fake_cauchy_inputs = SimpleNamespace(
        certified=True,
        initial_radius=cauchy_inputs.initial_radius,
        component_inputs=cauchy_inputs.component_inputs,
    )
    fake_compact_isolation = SimpleNamespace(
        certified=True,
        tau_isolation=isolation,
    )

    cases = (
        (
            {"isolation": fake_isolation},
            "FiniteFuchsianLogTotalCollisionIsolationCertificate",
        ),
        (
            {"cauchy_inputs": fake_cauchy_inputs},
            "FiniteFuchsianLogPrimitiveCauchyInputs",
        ),
        (
            {"compact_isolation": fake_compact_isolation},
            "FiniteFuchsianLogCompactTimeIsolationCertificate",
        ),
    )

    for overrides, expected_message in cases:
        kwargs = dict(
            branch=branch,
            isolation=isolation,
            cauchy_inputs=cauchy_inputs,
            compact_isolation=compact_isolation,
            sample_taus=(-0.02, 0.02),
            tolerance=1.0e-6,
        )
        kwargs.update(overrides)
        with pytest.raises(TypeError, match=expected_message):
            certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data(
                **kwargs,
            )


def test_supplied_finite_fuchsian_log_stop_chart_checker_rejects_corrupt_cauchy_ratio():
    branch, isolation, cauchy_inputs, _compact_isolation = (
        _supplied_fuchsian_log_entry_inputs()
    )
    certificate = (
        certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data(
            branch=branch,
            isolation=isolation,
            cauchy_inputs=cauchy_inputs,
            sample_taus=(-0.02, 0.02),
            tolerance=1.0e-6,
        )
    )
    primitive_inputs = certificate.stop_chart_certificate.primitive_cauchy_inputs
    component_name, component_input = primitive_inputs.component_inputs[0]
    corrupt_component_input = replace(
        component_input,
        shell_ratio=component_input.shell_ratio * 1.25 + 0.01,
    )
    corrupt_primitive_inputs = replace(
        primitive_inputs,
        component_inputs=(
            (component_name, corrupt_component_input),
            *primitive_inputs.component_inputs[1:],
        ),
    )
    corrupt_stop_chart = replace(
        certificate.stop_chart_certificate,
        primitive_cauchy_inputs=corrupt_primitive_inputs,
    )
    checked = check_total_collision_fuchsian_stop_chart(corrupt_stop_chart)

    assert not checked.certified
    assert "fuchsian_primitive_cauchy_inputs_certify" in (
        checked.missing_obligations
    )


def test_supplied_finite_fuchsian_log_stop_chart_checker_rejects_underreported_branch_envelope():
    branch, isolation, cauchy_inputs, _compact_isolation = (
        _supplied_fuchsian_log_entry_inputs()
    )
    certificate = (
        certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data(
            branch=branch,
            isolation=isolation,
            cauchy_inputs=cauchy_inputs,
            sample_taus=(-0.02, 0.02),
            tolerance=1.0e-6,
        )
    )
    primitive_inputs = certificate.stop_chart_certificate.primitive_cauchy_inputs
    component_entries = dict(primitive_inputs.component_inputs)
    value_input = component_entries["value"]
    underreported_majorant = 0.5 * value_input.majorant_initial
    underreported_value = replace(
        value_input,
        majorant_initial=underreported_majorant,
        first_shell_tail_bound=(
            underreported_majorant
            * value_input.step_ratio_bound ** (value_input.retained_order_initial + 1)
            / (1.0 - value_input.step_ratio_bound)
        ),
    )
    corrupt_primitive_inputs = replace(
        primitive_inputs,
        component_inputs=tuple(
            (
                component,
                underreported_value if component == "value" else component_input,
            )
            for component, component_input in primitive_inputs.component_inputs
        ),
    )
    corrupt_stop_chart = replace(
        certificate.stop_chart_certificate,
        primitive_cauchy_inputs=corrupt_primitive_inputs,
    )
    checked = check_total_collision_fuchsian_stop_chart(corrupt_stop_chart)

    assert not checked.certified
    assert "fuchsian_primitive_cauchy_inputs_certify" in (
        checked.missing_obligations
    )
    assert "fuchsian_primitive_cauchy_residual_tail_within_tolerance" not in (
        checked.missing_obligations
    )


def test_supplied_fuchsian_log_stop_chart_checker_rejects_large_residual_tail():
    branch, isolation, _cauchy_inputs, _compact_isolation = (
        _supplied_fuchsian_log_entry_inputs()
    )
    loose_cauchy_inputs = derive_finite_fuchsian_log_branch_primitive_cauchy_inputs(
        branch,
        initial_radius=0.02,
        shell_contraction=0.5,
        analytic_disk_fraction=0.25,
        log_growth_factor=1.2,
        step_ratio_bounds={
            "value": 0.3,
            "first_jet": 0.3,
            "lifted_residual": 0.3,
            "physical_residual": 0.3,
        },
        retained_order_initials={
            "value": 4,
            "first_jet": 4,
            "lifted_residual": 4,
            "physical_residual": 4,
        },
        retained_order_increments={
            "value": 2,
            "first_jet": 2,
            "lifted_residual": 2,
            "physical_residual": 2,
        },
    )
    loose_tail = max(
        input_.first_shell_tail_bound
        for input_ in loose_cauchy_inputs.component_inputs.values()
    )
    stop_chart = total_collision_fuchsian_stop_chart_certificate_from_branch(
        branch,
        isolation,
        certificate_id="loose-fuchsian-log-entry-stop-chart",
        chart_id="loose-fuchsian-log-entry-stop",
        residual_tolerance=1.0e-6,
        angular_momentum_tolerance=1.0e-6,
        tail_bound=loose_tail,
        cauchy_inputs=loose_cauchy_inputs,
    )
    checked = check_total_collision_fuchsian_stop_chart(stop_chart)

    assert not checked.certified
    assert "fuchsian_primitive_cauchy_inputs_certify" not in (
        checked.missing_obligations
    )
    assert "fuchsian_tail_bound_covers_primitive_cauchy_tail" not in (
        checked.missing_obligations
    )
    assert "fuchsian_primitive_cauchy_residual_tail_within_tolerance" in (
        checked.missing_obligations
    )


def test_supplied_fuchsian_log_stop_chart_checker_rejects_mislabeled_residual_tail():
    branch, isolation, cauchy_inputs, _compact_isolation = (
        _supplied_fuchsian_log_entry_inputs()
    )
    certificate = (
        certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data(
            branch=branch,
            isolation=isolation,
            cauchy_inputs=cauchy_inputs,
            sample_taus=(-0.02, 0.02),
            tolerance=1.0e-6,
        )
    )
    primitive_inputs = certificate.stop_chart_certificate.primitive_cauchy_inputs
    corrupt_primitive_inputs = replace(
        primitive_inputs,
        component_derivative_orders=tuple(
            (
                component,
                1 if component == "lifted_residual" else derivative_order,
            )
            for component, derivative_order in primitive_inputs.component_derivative_orders
        ),
    )
    corrupt_stop_chart = replace(
        certificate.stop_chart_certificate,
        primitive_cauchy_inputs=corrupt_primitive_inputs,
    )
    checked = check_total_collision_fuchsian_stop_chart(corrupt_stop_chart)

    assert not checked.certified
    assert "fuchsian_primitive_cauchy_inputs_certify" not in (
        checked.missing_obligations
    )
    assert "fuchsian_primitive_cauchy_residual_tail_within_tolerance" in (
        checked.missing_obligations
    )


def test_supplied_finite_fuchsian_log_stop_chart_rejects_nonisolated_branch():
    branch, isolation, cauchy_inputs, _compact_isolation = (
        _supplied_fuchsian_log_entry_inputs()
    )
    stale_isolation = replace(
        isolation,
        shape_pair_distance_floor=isolation.shape_pair_distance_floor + 0.5,
    )
    certificate = (
        certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data(
            branch=branch,
            isolation=stale_isolation,
            cauchy_inputs=cauchy_inputs,
            sample_taus=(-0.02, 0.02),
            tolerance=1.0e-6,
        )
    )

    assert not certificate.certified
    assert "punctured_total_collision_isolation" in certificate.missing_obligations
    assert "independent_total_collision_stop_chart_checker" not in (
        certificate.missing_obligations
    )


def test_supplied_finite_fuchsian_log_stop_chart_keeps_samples_diagnostic():
    branch, isolation, cauchy_inputs, _compact_isolation = (
        _supplied_fuchsian_log_entry_inputs()
    )
    certificate = (
        certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data(
            branch=branch,
            isolation=isolation,
            cauchy_inputs=cauchy_inputs,
            sample_taus=(0.0, 0.02),
            tolerance=1.0e-6,
        )
    )

    obligations = {obligation.obligation: obligation for obligation in certificate.obligations}

    assert certificate.certified
    assert certificate.missing_obligations == ()
    assert not obligations["diagnostic_sample_taus_inside_isolation"].required
    assert not obligations["diagnostic_sample_taus_inside_isolation"].certified
    assert not obligations["diagnostic_sampled_projection_identity_residual"].required
    assert not obligations["diagnostic_sampled_projection_identity_residual"].certified
    assert obligations["regularized_projection_identity_q_tau_squared_shape"].certified
    assert obligations["independent_interval_zero_angular_momentum_checker"].certified


def test_supplied_finite_fuchsian_log_stop_chart_does_not_audit_arbitrary_entry_theorem():
    branch, isolation, cauchy_inputs, _compact_isolation = (
        _supplied_fuchsian_log_entry_inputs()
    )
    certificate = (
        certify_supplied_finite_fuchsian_log_stop_chart_for_admissible_entry_data(
            branch=branch,
            isolation=isolation,
            cauchy_inputs=cauchy_inputs,
            sample_taus=(-0.02, 0.02),
            tolerance=1.0e-6,
        )
    )
    theorem = certify_finite_target_completeness_theorem(dimension=3)

    assert certificate.certified
    assert theorem.proof_certified is False
    assert "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data" in (
        theorem.critical_unaudited_analytic_lemma_ids
    )
    assert "arbitrary_total_collision_germ_entry_to_stop_chart" in (
        theorem.critical_unaudited_analytic_lemma_ids
    )
    assert "total_collision_stop_chart_existence" in (
        theorem.critical_unaudited_analytic_lemma_ids
    )


def test_certificate_search_completeness_keeps_search_gap_separate():
    theorem = certify_finite_target_completeness_theorem(dimension=2)
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        observed_prefix_failures=("finite_time_branch_union_consumption",),
    )

    assert search.theorem_certificate is theorem
    assert not search.certified
    assert search.observed_prefix_failures == (
        "finite_time_branch_union_consumption",
    )
    assert "pointwise_finite_target_theorem_certified" not in (
        search.missing_obligations
    )
    assert "fair_adaptive_chart_search" not in search.missing_obligations
    assert "finite_supplied_branch_tree_consumption_theorem" not in (
        search.missing_obligations
    )
    assert "recursive_set_valued_branch_partition_consumption" in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" in search.missing_obligations
    assert "finite_time_loop_budget_elimination" not in search.missing_obligations
    details = {obligation.obligation: obligation for obligation in search.obligations}
    assert details["fair_adaptive_chart_search"].certified
    assert "dovetail the constructor verifiers" in (
        details["fair_adaptive_chart_search"].detail
    )
    assert details["finite_time_loop_budget_elimination"].certified
    assert "engineering repeat budget is not a mathematical terminal obstruction" in (
        details["finite_time_loop_budget_elimination"].detail
    )
    assert "certificate_search_completeness_for_point_inputs" not in (
        search.missing_obligations
    )
    assert details["certificate_search_completeness_for_point_inputs"].certified
    assert "computable point inputs" in (
        details["certificate_search_completeness_for_point_inputs"].detail
    )
    assert details["finite_supplied_branch_tree_consumption_theorem"].certified
    assert "finite certified branch/event-order tree" in (
        details["finite_supplied_branch_tree_consumption_theorem"].detail
    )


def test_certificate_search_completeness_rejects_raw_boolean_constructor_witnesses():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    boolean_cases = (
        ("recursive_branch_refinement_certificate", True),
        ("event_order_refinement_certificate", True),
        ("stratified_branch_tree_certificate", True),
        ("stratified_event_order_tree_certificate", True),
        ("recursive_stratified_branch_consumption_certificate", True),
        ("recursive_stratified_event_order_consumption_certificate", True),
    )

    for field_name, value in boolean_cases:
        try:
            certify_finite_target_certificate_search_completeness(
                theorem,
                **{field_name: value},
            )
        except TypeError as error:
            assert field_name in str(error)
        else:
            raise AssertionError(f"raw boolean {field_name} was accepted")

    try:
        certify_finite_target_certificate_search_completeness(True)
    except TypeError as error:
        assert "theorem_certificate" in str(error)
    else:
        raise AssertionError("raw boolean theorem_certificate was accepted")


def test_finite_supplied_branch_tree_consumption_certifies_leafwise_union_only():
    partition = SimpleNamespace(
        certified=True,
        recursive_bisection_cover_certified=True,
        branch_cover_certified=True,
        branches=(
            SimpleNamespace(certified=True),
            SimpleNamespace(certified=True),
        ),
    )
    tree = certify_supplied_branch_event_tree(partition)
    branch_union_atlas = SimpleNamespace(
        proof_certified=True,
        proof_ledger=SimpleNamespace(
            entries=(
                SimpleNamespace(
                    name="finite_time_branch_union_consumption",
                    certified=True,
                ),
            )
        ),
    )

    certificate = certify_finite_supplied_branch_tree_consumption(
        partition=tree,
        branch_union_atlas=branch_union_atlas,
    )

    assert tree.certified
    assert tree.tree_kind == "simultaneous_close_pair_branch_partition"
    assert tree.leaf_count == 2
    assert tree.pending_leaf_count == 0
    assert certificate.certified
    assert certificate.partition_kind == "simultaneous_close_pair_branch_partition"
    assert certificate.branch_count == 2
    assert certificate.certified_leaf_count == 2
    assert certificate.missing_obligations == ()
    details = {obligation.obligation: obligation for obligation in certificate.obligations}
    assert details["finite_branch_tree_cover"].certified
    assert details["branch_union_consumption_ledger"].certified
    assert details["arbitrary_recursive_termination_not_claimed"].certified
    assert details["equality_or_pending_strata_are_explicit"].certified
    assert "separate theorem is still required" in certificate.proof_sketch


def test_supplied_branch_event_tree_rejects_truthy_partition_and_leaf_flags():
    partition = SimpleNamespace(
        certified="yes",
        recursive_bisection_cover_certified="yes",
        branch_cover_certified="yes",
        branches=(
            SimpleNamespace(branch_id="truthy_leaf", certified="yes"),
        ),
    )

    tree = certify_supplied_branch_event_tree(partition)

    assert not tree.certified
    assert tree.cover_certified is False
    assert tree.leaf_decisions_certified is False
    assert tree.leaf_certificates[0].certified is False
    assert "finite_branch_event_tree_cover" in tree.missing_obligations
    assert "finite_branch_event_tree_leaf_decisions" in tree.missing_obligations


def test_finite_supplied_branch_tree_consumption_rejects_raw_certified_leaf_objects():
    partition = SimpleNamespace(
        certified=True,
        recursive_bisection_cover_certified=True,
        branch_cover_certified=True,
        branches=(
            SimpleNamespace(certified=True),
            SimpleNamespace(certified=True),
        ),
    )
    tree = certify_supplied_branch_event_tree(partition)
    certificate = certify_finite_supplied_branch_tree_consumption(
        partition=tree,
        leaf_certificates=(
            SimpleNamespace(certified=True),
            SimpleNamespace(certified=True),
        ),
    )

    assert tree.certified
    assert not certificate.certified
    assert certificate.certified_leaf_count == 0
    assert "finite_leaf_atlas_or_stop_responses" in certificate.missing_obligations

    truthy = certify_finite_supplied_branch_tree_consumption(
        partition=tree,
        leaf_certificates=(
            SimpleNamespace(proof_certified="yes"),
            SimpleNamespace(proof_certified=1),
        ),
    )

    assert not truthy.certified
    assert truthy.certified_leaf_count == 0
    assert "finite_leaf_atlas_or_stop_responses" in truthy.missing_obligations


def test_finite_supplied_branch_tree_consumption_rejects_truthy_union_ledger():
    partition = SimpleNamespace(
        certified=True,
        recursive_bisection_cover_certified=True,
        branch_cover_certified=True,
        branches=(
            SimpleNamespace(certified=True),
            SimpleNamespace(certified=True),
        ),
    )
    tree = certify_supplied_branch_event_tree(partition)
    truthy_union_atlas = SimpleNamespace(
        proof_certified="yes",
        proof_ledger=SimpleNamespace(
            entries=(
                SimpleNamespace(
                    name="finite_time_branch_union_consumption",
                    certified="yes",
                ),
            )
        ),
    )

    certificate = certify_finite_supplied_branch_tree_consumption(
        partition=tree,
        branch_union_atlas=truthy_union_atlas,
    )

    assert tree.certified
    assert not certificate.certified
    assert certificate.certified_leaf_count == 0
    assert "finite_leaf_atlas_or_stop_responses" in certificate.missing_obligations
    assert "branch_union_consumption_ledger" in certificate.missing_obligations
    assert "branch_union_atlas_proof_certified" in certificate.missing_obligations


def test_finite_supplied_event_order_tree_requires_event_consumption_ledger():
    partition = SimpleNamespace(
        certified=True,
        recursive_bisection_cover_certified=True,
        leaf_decisions_certified=True,
        leaves=(SimpleNamespace(certified=True),),
    )
    tree = certify_supplied_branch_event_tree(partition)
    wrong_union_atlas = SimpleNamespace(
        proof_certified=True,
        proof_ledger=SimpleNamespace(
            entries=(
                SimpleNamespace(
                    name="finite_time_branch_union_consumption",
                    certified=True,
                ),
            )
        ),
    )

    blocked = certify_finite_supplied_branch_tree_consumption(
        partition=tree,
        branch_union_atlas=wrong_union_atlas,
        consumption_kind="event_order_branch_union",
    )
    assert not blocked.certified
    assert "branch_union_consumption_ledger" in blocked.missing_obligations

    event_union_atlas = SimpleNamespace(
        proof_certified=True,
        proof_ledger=SimpleNamespace(
            entries=(
                SimpleNamespace(
                    name="finite_time_event_order_branch_union_consumption",
                    certified=True,
                ),
            )
        ),
    )
    certified = certify_finite_supplied_branch_tree_consumption(
        partition=tree,
        branch_union_atlas=event_union_atlas,
        consumption_kind="event_order_branch_union",
    )

    assert tree.certified
    assert tree.tree_kind == "ks_event_order_partition"
    assert certified.certified
    assert certified.partition_kind == "ks_event_order_partition"
    assert certified.branch_count == 1


def test_ambiguous_event_order_tree_keeps_equality_strata_pending():
    partition = SimpleNamespace(
        event_alternative_cover_certified=True,
        state_partition_certified=False,
        branch_leaves=(
            SimpleNamespace(
                leaf_id="event_order_leaf:ks-exit",
                event_type="event_order_tie_leaf",
                decision="ambiguous_event_order_tie",
                certified=False,
                missing_obligations=(
                    "state_space_event_order_partition_not_constructed",
                ),
            ),
            SimpleNamespace(
                leaf_id="event_order_leaf:total-stop",
                event_type="event_order_tie_leaf",
                decision="ambiguous_event_order_tie",
                certified=False,
                missing_obligations=(
                    "state_space_event_order_partition_not_constructed",
                ),
            ),
        ),
    )

    tree = certify_supplied_branch_event_tree(partition)
    certificate = certify_finite_supplied_branch_tree_consumption(
        partition=tree,
        leaf_certificates=(),
        consumption_kind="event_order_branch_union",
    )

    assert not tree.certified
    assert tree.tree_kind == "ambiguous_event_order_partition"
    assert tree.pending_leaf_count == 2
    assert tree.equality_stratum_leaf_count == 2
    assert tree.equality_strata_explicit
    assert not certificate.certified
    assert "finite_branch_tree_leaf_decisions" in certificate.missing_obligations
    assert "finite_leaf_atlas_or_stop_responses" in certificate.missing_obligations
    details = {obligation.obligation: obligation for obligation in certificate.obligations}
    assert details["finite_branch_tree_cover"].certified
    assert details["equality_or_pending_strata_are_explicit"].certified


def test_stratified_event_order_tree_surfaces_tie_leaf_without_recursive_claim():
    partition = SimpleNamespace(
        event_alternative_cover_certified=True,
        state_partition_certified=False,
        branch_leaves=(
            SimpleNamespace(
                leaf_id="event_order_leaf:ks-exit-total-stop",
                event_type="event_order_tie_leaf",
                decision="ambiguous_event_order_tie",
                certified=False,
                missing_obligations=(
                    "state_space_event_order_partition_not_constructed",
                ),
            ),
        ),
    )
    branch_tree = certify_supplied_branch_event_tree(partition)
    equality = EqualityStratumCertificate(
        stratum_id="tie:ks-exit-total-stop",
        defining_function_ids=("t_ks_exit-t_total_stop",),
        leaf_kind="simultaneous_event_equality",
        isolation_certified=True,
        resolution_policy="requires_stop_or_selector_leaf",
        certified=True,
    )
    tie_leaf = EventOrderTieLeafCertificate(
        leaf_id="event_order_leaf:ks-exit-total-stop",
        tied_event_ids=("ks_exit", "total_stop"),
        equality_stratum=equality,
        certified=True,
    )
    stratified_leaf = StratifiedBranchLeafCertificate(
        leaf_id="stratified:event_order_leaf:ks-exit-total-stop",
        source_leaf_id="event_order_leaf:ks-exit-total-stop",
        leaf_kind="simultaneous_event_equality",
        terminal_response_kind="pending_equality_stratum_response",
        terminal_response_certified=False,
        source_leaf_certified=False,
        event_order_tie=tie_leaf,
        equality_stratum=equality,
        missing_obligations=("equality_stratum_terminal_response_missing",),
    )

    stratified = certify_stratified_branch_event_tree(
        branch_tree,
        leaf_certificates=(stratified_leaf,),
    )
    search = certify_finite_target_certificate_search_completeness(
        certify_finite_target_completeness_theorem(dimension=3),
        stratified_event_order_tree_certificate=stratified,
    )

    assert stratified.certified
    assert not stratified.proof_certified
    assert not stratified.recursive_theorem_certified
    assert stratified.leaf_kinds == ("simultaneous_event_equality",)
    assert stratified.zero_margin_leaf_count == 1
    assert "arbitrary_recursive_stratified_exhaustion_not_claimed" in (
        stratified.missing_obligations
    )
    assert (
        "stratified:event_order_leaf:ks-exit-total-stop:terminal_response_missing"
        in stratified.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" in search.missing_obligations
    details = {obligation.obligation: obligation for obligation in search.obligations}
    assert details["stratified_event_order_leaves_explicit"].certified
    assert "simultaneous_event_equality" in (
        details["event_order_partition_consumption_theorem"].detail
    )
    assert "equality_stratum_terminal_response_missing" in (
        details["event_order_partition_consumption_theorem"].detail
    )


def _positive_margin_stratified_tree(source_id: str):
    source_tree = certify_supplied_branch_event_tree(
        SimpleNamespace(
            certified=True,
            recursive_bisection_cover_certified=True,
            branch_cover_certified=True,
            branches=(
                SimpleNamespace(
                    branch_id=source_id,
                    leaf_type="ordinary_positive_margin_leaf",
                    decision="ordinary_chart_response",
                    certified=True,
                ),
            ),
        )
    )
    leaf = StratifiedBranchLeafCertificate(
        leaf_id=f"stratified:{source_id}",
        source_leaf_id=source_id,
        leaf_kind="positive_margin_unique_event",
        terminal_response_kind="ordinary_chart_response",
        terminal_response_certified=True,
        source_leaf_certified=True,
        decision_functions=(
            AnalyticDecisionFunctionCertificate(
                function_id=f"margin:{source_id}",
                function_kind="event_order_gap",
                margin_lower_bound=0.02,
                lipschitz_bound=3.0,
                certified=True,
            ),
        ),
    )
    return certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=(leaf,),
    )


def test_stratified_branch_tree_rejects_truthy_supplied_leaf_and_tree_flags():
    branch_tree = certify_supplied_branch_event_tree(
        SimpleNamespace(
            certified=True,
            recursive_bisection_cover_certified=True,
            branch_cover_certified=True,
            branches=(
                SimpleNamespace(
                    branch_id="truthy:positive-margin",
                    leaf_type="ordinary_positive_margin_leaf",
                    decision="ordinary_chart_response",
                    certified=True,
                ),
            ),
        )
    )
    leaf = StratifiedBranchLeafCertificate(
        leaf_id="stratified:truthy:positive-margin",
        source_leaf_id="truthy:positive-margin",
        leaf_kind="positive_margin_unique_event",
        terminal_response_kind="ordinary_chart_response",
        terminal_response_certified="yes",
        source_leaf_certified="yes",
        decision_functions=(
            AnalyticDecisionFunctionCertificate(
                function_id="truthy:gap",
                function_kind="event_order_gap",
                margin_lower_bound=0.1,
                lipschitz_bound=1.0,
                certified=True,
            ),
        ),
    )
    stratified = certify_stratified_branch_event_tree(
        branch_tree,
        leaf_certificates=(leaf,),
    )
    spoofed_tree_flags = replace(
        _positive_margin_stratified_tree("truthy-tree-flags"),
        cover_certified="yes",
        preserves_branch_tree="yes",
        leaf_taxonomy_certified="yes",
    )

    assert branch_tree.certified
    assert stratified.certified
    assert stratified.terminal_response_count == 0
    assert not stratified.proof_certified
    assert "stratified:truthy:positive-margin:source_leaf_not_certified" in (
        stratified.missing_obligations
    )
    assert "stratified:truthy:positive-margin:terminal_response_missing" in (
        stratified.missing_obligations
    )
    assert not spoofed_tree_flags.certified
    assert "stratified_branch_tree_cover" in spoofed_tree_flags.missing_obligations
    assert "stratified_branch_tree_preserves_source_leaves" in (
        spoofed_tree_flags.missing_obligations
    )
    assert "stratified_branch_tree_leaf_taxonomy" in (
        spoofed_tree_flags.missing_obligations
    )


def test_branch_event_tree_rejects_attribute_compatible_obligation_spoof():
    tree = certify_supplied_branch_event_tree(
        SimpleNamespace(
            certified=True,
            recursive_bisection_cover_certified=True,
            branch_cover_certified=True,
            branches=(
                SimpleNamespace(
                    branch_id="branch-obligation-spoof",
                    leaf_type="ordinary_positive_margin_leaf",
                    decision="ordinary_chart_response",
                    certified=True,
                ),
            ),
        )
    )
    spoofed = replace(
        tree,
        obligations=(
            SimpleNamespace(
                obligation="fake_branch_event_obligation",
                certified=True,
                required=True,
            ),
        ),
    )
    empty_ledger = replace(tree, obligations=())
    optional_only_ledger = replace(
        tree,
        obligations=tuple(
            replace(obligation, required=False) for obligation in tree.obligations
        ),
    )

    assert tree.certified
    assert not spoofed.certified
    assert "finite_branch_event_tree_obligation_type" in (
        spoofed.missing_obligations
    )
    assert not empty_ledger.certified
    assert "finite_branch_event_tree_obligations_present" in (
        empty_ledger.missing_obligations
    )
    assert not optional_only_ledger.certified
    assert "finite_branch_event_tree_required_obligation_present" in (
        optional_only_ledger.missing_obligations
    )


def _parent_tree_with_recursive_tie():
    source_tree = certify_supplied_branch_event_tree(
        SimpleNamespace(
            certified=True,
            recursive_bisection_cover_certified=True,
            branch_cover_certified=False,
            branches=(
                SimpleNamespace(
                    branch_id="parent:ordinary",
                    leaf_type="ordinary_positive_margin_leaf",
                    decision="ordinary_chart_response",
                    certified=True,
                ),
                SimpleNamespace(
                    branch_id="parent:tie",
                    leaf_type="event_order_tie_leaf",
                    decision="ambiguous_event_order_tie",
                    certified=False,
                ),
            ),
        )
    )
    ordinary_leaf = StratifiedBranchLeafCertificate(
        leaf_id="stratified:parent:ordinary",
        source_leaf_id="parent:ordinary",
        leaf_kind="positive_margin_unique_event",
        terminal_response_kind="ordinary_chart_response",
        terminal_response_certified=True,
        source_leaf_certified=True,
        decision_functions=(
            AnalyticDecisionFunctionCertificate(
                function_id="margin:parent:ordinary",
                function_kind="event_order_gap",
                margin_lower_bound=0.03,
                lipschitz_bound=2.0,
                certified=True,
            ),
        ),
    )
    equality = EqualityStratumCertificate(
        stratum_id="tie:parent",
        defining_function_ids=("t_binary-t_total",),
        leaf_kind="simultaneous_event_equality",
        isolation_certified=True,
        resolution_policy="refine_on_lower_dimensional_stratum",
        certified=True,
    )
    tie_leaf = EventOrderTieLeafCertificate(
        leaf_id="parent:tie",
        tied_event_ids=("binary_entry", "total_stop"),
        equality_stratum=equality,
        certified=True,
    )
    recursive_leaf = StratifiedBranchLeafCertificate(
        leaf_id="stratified:parent:tie",
        source_leaf_id="parent:tie",
        leaf_kind="simultaneous_event_equality",
        terminal_response_kind="recursive_equality_stratum",
        terminal_response_certified=False,
        source_leaf_certified=False,
        equality_stratum=equality,
        event_order_tie=tie_leaf,
    )
    return certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=(ordinary_leaf, recursive_leaf),
    )


def test_recursive_stratified_consumption_closes_lower_dimensional_equality_tree():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    child_tree = _positive_margin_stratified_tree("child:ordinary")
    child_consumption = certify_recursive_stratified_branch_event_consumption(
        child_tree,
        root_dimension=2,
        root_rank=1,
    )
    parent_tree = _parent_tree_with_recursive_tie()
    parent_consumption = certify_recursive_stratified_branch_event_consumption(
        parent_tree,
        root_dimension=3,
        root_rank=1,
        child_consumptions={
            "stratified:parent:tie": child_consumption,
        },
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=parent_consumption,
        recursive_stratified_event_order_consumption_certificate=parent_consumption,
    )

    assert child_consumption.certified
    assert parent_consumption.certified
    assert parent_consumption.terminal_leaf_count == 1
    assert parent_consumption.recursive_leaf_count == 1
    assert parent_consumption.node_count == 2
    assert parent_consumption.missing_obligations == ()
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )
    details = {obligation.obligation: obligation for obligation in search.obligations}
    assert details["recursive_set_valued_branch_partition_consumption"].certified
    assert "recursive_leaf_count=1" in (
        details["recursive_set_valued_branch_partition_consumption"].detail
    )


def test_recursive_stratified_consumption_rejects_truthy_child_consumption():
    child_consumption = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("child:truthy-consumption"),
        root_dimension=2,
        root_rank=1,
    )
    parent_tree = _parent_tree_with_recursive_tie()
    truthy_child = SimpleNamespace(
        certified="yes",
        proof_certified="yes",
        root_dimension=2,
        root_rank=1,
        constructor_source_type="TruthinessChild",
    )

    assert child_consumption.certified
    assert parent_tree.certified
    with pytest.raises(
        TypeError,
        match="child_consumptions must map leaf ids",
    ):
        certify_recursive_stratified_branch_event_consumption(
            parent_tree,
            root_dimension=3,
            root_rank=1,
            child_consumptions={
                "stratified:parent:tie": truthy_child,
            },
        )


def test_recursive_stratified_consumption_rejects_attribute_compatible_obligation_spoof():
    consumption = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("recursive-obligation-spoof"),
        root_dimension=2,
        root_rank=1,
    )
    spoofed = replace(
        consumption,
        obligations=(
            SimpleNamespace(
                obligation="fake_recursive_stratified_obligation",
                certified=True,
                required=True,
            ),
        ),
    )
    empty_ledger = replace(consumption, obligations=())
    optional_only_ledger = replace(
        consumption,
        obligations=tuple(
            replace(obligation, required=False)
            for obligation in consumption.obligations
        ),
    )

    assert consumption.certified
    assert not spoofed.certified
    assert "recursive_stratified_consumption_obligation_type" in (
        spoofed.missing_obligations
    )
    assert not empty_ledger.certified
    assert "recursive_stratified_consumption_obligations_present" in (
        empty_ledger.missing_obligations
    )
    assert not optional_only_ledger.certified
    assert "recursive_stratified_consumption_required_obligation_present" in (
        optional_only_ledger.missing_obligations
    )


def test_recursive_stratified_consumption_rejects_unsupported_stratum():
    tree = certify_stratified_branch_event_tree(
        certify_supplied_branch_event_tree(
            SimpleNamespace(
                certified=True,
                recursive_bisection_cover_certified=True,
                branch_cover_certified=False,
                branches=(
                    SimpleNamespace(
                        branch_id="unsupported:leaf",
                        leaf_type="unsupported_analytic_stratum",
                        decision="unsupported_analytic_stratum",
                        certified=False,
                    ),
                ),
            )
        ),
        leaf_certificates=(
            StratifiedBranchLeafCertificate(
                leaf_id="stratified:unsupported:leaf",
                source_leaf_id="unsupported:leaf",
                leaf_kind="unsupported_analytic_stratum",
                terminal_response_kind="unsupported",
                terminal_response_certified=False,
                source_leaf_certified=False,
            ),
        ),
    )
    certificate = certify_recursive_stratified_branch_event_consumption(
        tree,
        root_dimension=3,
    )

    assert not certificate.certified
    assert certificate.unsupported_leaf_count == 1
    assert (
        "stratified:unsupported:leaf:unsupported_analytic_stratum"
        in certificate.missing_obligations
    )


def test_recursive_stratified_consumption_accepts_terminal_selector_policy_leaf():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    source_tree = certify_supplied_branch_event_tree(
        SimpleNamespace(
            certified=True,
            recursive_bisection_cover_certified=True,
            branch_cover_certified=True,
            branches=(
                SimpleNamespace(
                    branch_id="selector:identity",
                    leaf_type="selector_policy_leaf",
                    decision="selected_identity_selector",
                    certified=True,
                ),
            ),
        )
    )
    stratified = certify_terminal_policy_stratified_branch_event_tree(
        source_tree,
        selector_policies=(
            SelectorPolicyLeafCertificate(
                leaf_id="selector:identity",
                selector_policy_id="selected_identity_selector",
                defining_function_ids=("selector_boundary",),
                isolation_certified=True,
                certified=True,
            ),
        ),
    )
    recursive = certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=3,
        root_rank=1,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert stratified.proof_certified
    assert stratified.leaf_kinds == ("selector_policy",)
    assert recursive.certified
    assert recursive.terminal_leaf_count == 1
    assert recursive.recursive_leaf_count == 0
    assert recursive.missing_obligations == ()
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_recursive_stratified_consumption_accepts_terminal_total_collision_cluster():
    source_tree = certify_supplied_branch_event_tree(
        SimpleNamespace(
            certified=True,
            recursive_bisection_cover_certified=True,
            branch_cover_certified=True,
            branches=(
                SimpleNamespace(
                    branch_id="total:cluster",
                    leaf_type="total_collision_cluster_leaf",
                    decision="maximal_classical_stop",
                    certified=True,
                ),
            ),
        )
    )
    stratified = certify_terminal_policy_stratified_branch_event_tree(
        source_tree,
        total_collision_clusters=(
            TotalCollisionClusterLeafCertificate(
                leaf_id="total:cluster",
                cluster_pair_ids=("pair:0-1", "pair:0-2", "pair:1-2"),
                stop_or_selector_policy="maximal_classical_stop",
                entry_certificate=SimpleNamespace(proof_certified=True),
                certified=True,
            ),
        ),
    )
    recursive = certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=3,
        root_rank=1,
    )

    assert stratified.proof_certified
    assert stratified.leaf_kinds == ("total_collision_cluster",)
    assert recursive.certified
    assert recursive.terminal_leaf_count == 1
    assert recursive.recursive_leaf_count == 0
    assert recursive.missing_obligations == ()


def test_terminal_policy_stratification_rejects_truthy_terminal_policy_evidence():
    source_tree = certify_supplied_branch_event_tree(
        SimpleNamespace(
            certified=True,
            recursive_bisection_cover_certified=True,
            branch_cover_certified=True,
            branches=(
                SimpleNamespace(
                    branch_id="selector:truthy",
                    leaf_type="selector_policy_leaf",
                    decision="selected_identity_selector",
                    certified=True,
                ),
                SimpleNamespace(
                    branch_id="total:truthy",
                    leaf_type="total_collision_cluster_leaf",
                    decision="maximal_classical_stop",
                    certified=True,
                ),
            ),
        )
    )
    stratified = certify_terminal_policy_stratified_branch_event_tree(
        source_tree,
        selector_policies=(
            SelectorPolicyLeafCertificate(
                leaf_id="selector:truthy",
                selector_policy_id="selected_identity_selector",
                defining_function_ids=("selector_boundary",),
                isolation_certified="yes",
                certified=True,
            ),
        ),
        total_collision_clusters=(
            TotalCollisionClusterLeafCertificate(
                leaf_id="total:truthy",
                cluster_pair_ids=("pair:0-1", "pair:0-2", "pair:1-2"),
                stop_or_selector_policy="maximal_classical_stop",
                entry_certificate=SimpleNamespace(certified=True),
                certified="yes",
            ),
        ),
    )
    recursive = certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=3,
        root_rank=1,
    )

    assert source_tree.certified
    assert not stratified.proof_certified
    assert "stratified:selector:truthy:stratum_not_certified" in (
        stratified.missing_obligations
    )
    assert "stratified:selector:truthy:terminal_response_missing" in (
        stratified.missing_obligations
    )
    assert "stratified:total:truthy:stratum_not_certified" in (
        stratified.missing_obligations
    )
    assert "stratified:total:truthy:terminal_response_missing" in (
        stratified.missing_obligations
    )
    assert not recursive.certified
    assert "every_leaf_terminal_or_descending_child" in recursive.missing_obligations
    assert (
        "stratified:selector:truthy:terminal_or_recursive_response_missing"
        in recursive.missing_obligations
    )
    assert (
        "stratified:total:truthy:terminal_or_recursive_response_missing"
        in recursive.missing_obligations
    )


def test_recursive_stratified_consumption_requires_strict_dimension_or_rank_descent():
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("child:same-rank"),
        root_dimension=3,
        root_rank=1,
    )
    parent = certify_recursive_stratified_branch_event_consumption(
        _parent_tree_with_recursive_tie(),
        root_dimension=3,
        root_rank=1,
        child_consumptions={
            "stratified:parent:tie": child,
        },
    )

    assert child.certified
    assert not parent.certified
    assert (
        "stratified:parent:tie:child_stratum_not_strictly_lower_dimension_or_rank"
        in parent.missing_obligations
    )


def test_polynomial_decision_stratifier_derives_equality_leaf_without_hulling():
    stratification = certify_polynomial_decision_stratified_branch_event_tree(
        decision_id="event_tie_discriminant",
        coefficients=(0.0, 1.0),
        domain=(-1.0, 1.0),
        root_brackets=((-0.01, 0.01),),
    )

    assert stratification.certified
    assert stratification.sign_stratum_count == 2
    assert stratification.equality_stratum_count == 1
    assert stratification.source_tree.cover_certified
    assert not stratification.source_tree.certified
    assert stratification.stratified_tree.certified
    assert not stratification.stratified_tree.proof_certified
    assert stratification.stratified_tree.leaf_kinds == (
        "positive_margin_unique_event",
        "simultaneous_event_equality",
        "positive_margin_unique_event",
    )
    assert "arbitrary_recursive_stratified_exhaustion_not_claimed" in (
        stratification.stratified_tree.missing_obligations
    )


def test_polynomial_decision_stratifier_feeds_recursive_consumption_with_child_stratum():
    stratification = certify_polynomial_decision_stratified_branch_event_tree(
        decision_id="event_tie_discriminant",
        coefficients=(0.0, 1.0),
        domain=(-1.0, 1.0),
        root_brackets=((-0.01, 0.01),),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("polynomial-child:ordinary"),
        root_dimension=2,
        root_rank=1,
    )
    parent = certify_recursive_stratified_branch_event_consumption(
        stratification.stratified_tree,
        root_dimension=3,
        root_rank=1,
        child_consumptions={
            "stratified:event_tie_discriminant:root:0:leaf": child,
        },
    )

    assert child.certified
    assert parent.certified
    assert parent.terminal_leaf_count == 2
    assert parent.recursive_leaf_count == 1
    assert parent.missing_obligations == ()


def test_polynomial_decision_set_valued_constructor_scope_is_constructor_derived():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    stratification = certify_polynomial_decision_stratified_branch_event_tree(
        decision_id="validated_event_tie_discriminant",
        coefficients=(0.0, 1.0),
        domain=(-1.0, 1.0),
        root_brackets=((-0.01, 0.01),),
    )
    scoped = (
        certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness(
            theorem,
            constructor_certificate=stratification,
            root_dimension=3,
            root_rank=1,
        )
    )
    recursive = scoped.branch_consumption_certificate
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert scoped.event_order_consumption_certificate is recursive
    assert stratification.source_tree.source_type == "PolynomialDecisionStratification"
    assert stratification.proof_certified
    assert recursive.proof_certified
    assert recursive.recursion_kind == "polynomial_decision_stratification"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert recursive.terminal_leaf_count == 2
    assert recursive.recursive_leaf_count == 1
    assert not scoped.proof_certified
    assert "single-polynomial decision stratification" in scoped.statement
    assert "strict sign cells" in scoped.proof_sketch
    assert not validated.proof_certified
    assert (
        validated.input_scope_id
        == "finite_polynomial_decision_stratification_interval_boxes"
    )
    assert "verified simple root brackets" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


def test_sturm_polynomial_decision_stratification_proves_root_free_positive_cell():
    interval_only = certify_polynomial_decision_stratified_branch_event_tree(
        decision_id="interval_only_quadratic_event",
        coefficients=(1.0, 0.0, 1.0),
        domain=(-1.0, 1.0),
    )
    stratification = certify_sturm_polynomial_decision_stratified_branch_event_tree(
        decision_id="single_sturm_quadratic_event",
        coefficients=(1.0, 0.0, 1.0),
        domain=(-1.0, 1.0),
    )
    recursive = certify_polynomial_decision_recursive_consumption(
        stratification,
        root_dimension=3,
        root_rank=1,
    )

    assert not interval_only.proof_certified
    assert any(
        "polynomial_sign_cell_not_separated_from_zero" in item
        for item in interval_only.missing_obligations
    )
    assert stratification.proof_certified
    assert stratification.source_tree.source_type == (
        "SturmPolynomialDecisionStratification"
    )
    assert stratification.sign_stratum_count == 1
    assert stratification.equality_stratum_count == 0
    assert stratification.root_brackets == ()
    assert recursive.proof_certified
    assert recursive.constructor_source_type == (
        "SturmPolynomialDecisionStratification"
    )
    assert recursive.terminal_leaf_count == 1


def test_sturm_polynomial_decision_stratification_preserves_double_root():
    stratification = certify_sturm_polynomial_decision_stratified_branch_event_tree(
        decision_id="single_sturm_tangent_event",
        coefficients=(0.0, 0.0, 1.0),
        domain=(-1.0, 1.0),
    )
    recursive = certify_polynomial_decision_recursive_consumption(
        stratification,
        root_dimension=3,
        root_rank=1,
    )
    equality_strata = tuple(
        stratum for stratum in stratification.strata if stratum.equality
    )

    assert stratification.proof_certified
    assert stratification.sign_stratum_count == 2
    assert stratification.equality_stratum_count == 1
    assert len(equality_strata) == 1
    assert equality_strata[0].stratum_kind == "sturm_polynomial_multiple_equality_root"
    assert equality_strata[0].root_multiplicities == (2,)
    assert equality_strata[0].second_derivative_interval is not None
    assert recursive.proof_certified
    assert recursive.recursive_leaf_count == 1
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)


def test_sturm_polynomial_decision_stratification_blocks_identically_zero_polynomial():
    stratification = certify_sturm_polynomial_decision_stratified_branch_event_tree(
        decision_id="single_sturm_zero_event",
        coefficients=(0.0,),
        domain=(-1.0, 1.0),
    )

    assert not stratification.proof_certified
    assert stratification.sign_stratum_count == 1
    assert any(
        "sturm_polynomial_identically_zero_on_cell" in item
        for item in stratification.missing_obligations
    )


def test_scoped_arbitrary_interval_partition_generation_accepts_single_sturm_polynomial_grammar():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    stratification = certify_sturm_polynomial_decision_stratified_branch_event_tree(
        decision_id="scoped_sturm_polynomial_event",
        coefficients=(1.0, 0.0, 1.0),
        domain=(-1.0, 1.0),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=stratification,
        branch_root_dimension=3,
        branch_root_rank=1,
    )

    assert not certificate.proof_certified
    assert certificate.branch_constructor_source_type == (
        "SturmPolynomialDecisionStratification"
    )
    assert certificate.branch_function_grammar_id == (
        "finite_sturm_polynomial_decision_interval_inputs"
    )
    assert certificate.input_scope_id == (
        "finite_sturm_polynomial_decision_stratification_interval_boxes"
    )
    assert certificate.unsupported_strata == ()
    assert certificate.missing_obligations == (
        "scoped_set_valued_constructor_proof_certified",
        "validated_set_valued_scope_proof_certified",
    )


def test_taylor_model_decision_with_weierstrass_witness_is_recursively_consumed():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    stratification = certify_taylor_model_decision_stratified_branch_event_tree(
        decision_function=TaylorModelDecisionFunctionSpec(
            decision_id="taylor_event_tie_discriminant",
            coefficients=(-0.25, 1.0),
            expansion_center=0.0,
            remainder_bound=1.0e-4,
            derivative_remainder_bound=5.0e-2,
            root_brackets=((0.24, 0.26),),
            witness_kind="weierstrass_simple_root",
        ),
        domain=(-1.0, 1.0),
    )
    children = derive_taylor_model_decision_child_consumptions(stratification)
    recursive = certify_taylor_model_decision_recursive_consumption(
        stratification,
        root_dimension=3,
        root_rank=1,
        child_consumptions=children,
    )
    scoped = (
        certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness(
            theorem,
            constructor_certificate=stratification,
            root_dimension=3,
            root_rank=1,
        )
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert stratification.source_tree.source_type == "TaylorModelDecisionStratification"
    assert stratification.proof_certified
    assert stratification.sign_stratum_count == 2
    assert stratification.equality_stratum_count == 1
    assert children
    assert recursive.proof_certified
    assert recursive.recursion_kind == "taylor_model_decision_stratification"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not scoped.proof_certified
    assert not validated.proof_certified
    assert validated.input_scope_id == (
        "finite_taylor_model_decision_interval_inputs_with_weierstrass_certificate"
    )
    assert "finite Taylor-model decision stratification" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


def test_taylor_model_decision_without_root_witness_returns_unsupported_stratum():
    stratification = certify_taylor_model_decision_stratified_branch_event_tree(
        decision_function=TaylorModelDecisionFunctionSpec(
            decision_id="unsupported_taylor_event_tie",
            coefficients=(-0.25, 1.0),
            expansion_center=0.0,
            remainder_bound=1.0e-4,
            derivative_remainder_bound=5.0e-2,
            root_brackets=((0.24, 0.26),),
            witness_kind="unsupported",
        ),
        domain=(-1.0, 1.0),
    )

    assert not stratification.proof_certified
    assert stratification.stratified_tree.unsupported_leaf_count == 1
    assert any(
        leaf.leaf_kind == "unsupported_analytic_stratum"
        for leaf in stratification.stratified_tree.leaf_certificates
    )
    assert any(
        "taylor_model_weierstrass_or_monotone_witness_missing" in item
        for item in stratification.missing_obligations
    )


def test_taylor_model_decision_arrangement_scope_is_constructor_derived():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_taylor_model_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="taylor_model_event_order_arrangement",
        decision_functions=(
            TaylorModelDecisionFunctionSpec(
                decision_id="first_event_tie",
                coefficients=(-0.25, 1.0),
                expansion_center=0.0,
                remainder_bound=1.0e-4,
                derivative_remainder_bound=5.0e-2,
                root_brackets=((0.24, 0.26),),
                witness_kind="weierstrass_simple_root",
            ),
            TaylorModelDecisionFunctionSpec(
                decision_id="second_event_tie",
                coefficients=(0.5, 1.0),
                expansion_center=0.0,
                remainder_bound=1.0e-4,
                derivative_remainder_bound=5.0e-2,
                root_brackets=((-0.51, -0.49),),
                witness_kind="monotone_root_isolation",
            ),
        ),
        domain=(-1.0, 1.0),
    )
    children = derive_taylor_model_decision_arrangement_child_consumptions(
        arrangement,
    )
    recursive = certify_taylor_model_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=1,
        child_consumptions=children,
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=arrangement,
        branch_root_dimension=3,
        branch_root_rank=1,
    )

    assert isinstance(arrangement, TaylorModelDecisionArrangementStratificationCertificate)
    assert arrangement.source_tree.source_type == "TaylorModelDecisionArrangement"
    assert arrangement.proof_certified
    assert arrangement.sign_stratum_count == 3
    assert arrangement.equality_stratum_count == 2
    assert arrangement.unsupported_stratum_count == 0
    assert children
    assert recursive.proof_certified
    assert recursive.recursion_kind == "taylor_model_decision_arrangement"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not certificate.proof_certified
    assert not certificate.arbitrary_partition_generation_claimed
    assert certificate.event_function_grammar_id == (
        "finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate"
    )
    assert certificate.input_scope_id == (
        "finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate"
    )
    assert certificate.unsupported_strata == ()


def test_taylor_model_decision_arrangement_groups_matching_root_brackets():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_taylor_model_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="coincident_taylor_model_event_order",
        decision_functions=(
            TaylorModelDecisionFunctionSpec(
                decision_id="binary_12_minus_13",
                coefficients=(-0.25, 1.0),
                expansion_center=0.0,
                remainder_bound=1.0e-4,
                derivative_remainder_bound=5.0e-2,
                root_brackets=((0.24, 0.26),),
                witness_kind="weierstrass_simple_root",
            ),
            TaylorModelDecisionFunctionSpec(
                decision_id="binary_12_minus_23",
                coefficients=(-0.25, 1.0),
                expansion_center=0.0,
                remainder_bound=1.0e-4,
                derivative_remainder_bound=5.0e-2,
                root_brackets=((0.24, 0.26),),
                witness_kind="monotone_root_isolation",
            ),
        ),
        domain=(-1.0, 1.0),
    )
    simultaneous = [stratum for stratum in arrangement.strata if stratum.equality]
    child_consumptions = derive_taylor_model_decision_arrangement_child_consumptions(
        arrangement,
    )
    recursive = certify_taylor_model_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=child_consumptions,
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=arrangement,
        branch_root_dimension=3,
        branch_root_rank=2,
    )

    assert arrangement.proof_certified
    assert arrangement.sign_stratum_count == 2
    assert len(simultaneous) == 1
    assert simultaneous[0].stratum_kind == "simultaneous_taylor_model_equality_root"
    assert set(simultaneous[0].stratum_id.split(":")[-1].split("+")) == {
        "binary_12_minus_13",
        "binary_12_minus_23",
    }
    equality_leaf = [
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    ][0]
    assert set(equality_leaf.equality_stratum.defining_function_ids) == {
        "binary_12_minus_13",
        "binary_12_minus_23",
    }
    assert child_consumptions
    assert recursive.proof_certified
    assert not certificate.proof_certified
    assert not certificate.arbitrary_partition_generation_claimed
    assert certificate.missing_obligations == (
        "scoped_set_valued_constructor_proof_certified",
        "validated_set_valued_scope_proof_certified",
    )


def test_taylor_model_decision_arrangement_rejects_unproved_overlapping_roots():
    try:
        certify_taylor_model_decision_arrangement_stratified_branch_event_tree(
            arrangement_id="overlapping_taylor_model_event_order",
            decision_functions=(
                TaylorModelDecisionFunctionSpec(
                    decision_id="first_overlap",
                    coefficients=(-0.25, 1.0),
                    expansion_center=0.0,
                    remainder_bound=1.0e-4,
                    derivative_remainder_bound=5.0e-2,
                    root_brackets=((0.24, 0.27),),
                    witness_kind="weierstrass_simple_root",
                ),
                TaylorModelDecisionFunctionSpec(
                    decision_id="second_overlap",
                    coefficients=(-0.25, 1.0),
                    expansion_center=0.0,
                    remainder_bound=1.0e-4,
                    derivative_remainder_bound=5.0e-2,
                    root_brackets=((0.23, 0.26),),
                    witness_kind="monotone_root_isolation",
                ),
            ),
            domain=(-1.0, 1.0),
        )
    except ValueError as error:
        assert "separated root brackets or exactly matching brackets" in str(error)
    else:
        raise AssertionError("overlapping Taylor-model root brackets were accepted")


def test_taylor_model_decision_arrangement_blocks_unsupported_root_witness():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_taylor_model_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="unsupported_taylor_model_arrangement",
        decision_functions=(
            TaylorModelDecisionFunctionSpec(
                decision_id="supported_event_tie",
                coefficients=(-0.25, 1.0),
                expansion_center=0.0,
                remainder_bound=1.0e-4,
                derivative_remainder_bound=5.0e-2,
                root_brackets=((0.24, 0.26),),
                witness_kind="weierstrass_simple_root",
            ),
            TaylorModelDecisionFunctionSpec(
                decision_id="unsupported_event_tie",
                coefficients=(0.5, 1.0),
                expansion_center=0.0,
                remainder_bound=1.0e-4,
                derivative_remainder_bound=5.0e-2,
                root_brackets=((-0.51, -0.49),),
                witness_kind="unsupported",
            ),
        ),
        domain=(-1.0, 1.0),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=arrangement,
        branch_root_dimension=3,
        branch_root_rank=1,
    )

    assert not arrangement.proof_certified
    assert arrangement.unsupported_stratum_count == 1
    assert arrangement.stratified_tree.unsupported_leaf_count == 1
    assert not certificate.proof_certified
    assert not certificate.arbitrary_partition_generation_claimed
    assert certificate.event_function_grammar_id == (
        "finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate"
    )
    assert certificate.unsupported_strata
    assert "no_unsupported_analytic_strata" in certificate.missing_obligations


def test_scoped_arbitrary_interval_partition_generation_accepts_supported_polynomial_grammar():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    stratification = certify_polynomial_decision_stratified_branch_event_tree(
        decision_id="arbitrary_scoped_polynomial_event_tie",
        coefficients=(0.0, 1.0),
        domain=(-1.0, 1.0),
        root_brackets=((-0.01, 0.01),),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=stratification,
        branch_root_dimension=3,
        branch_root_rank=1,
    )

    assert isinstance(certificate, ArbitraryIntervalInputPartitionGenerationCertificate)
    assert not certificate.proof_certified
    assert not certificate.certified
    assert not certificate.arbitrary_partition_generation_claimed
    assert certificate.unsupported_strata == ()
    assert certificate.generated_stratified_tree == stratification.stratified_tree
    assert certificate.event_function_grammar_id == (
        "finite_polynomial_decision_interval_inputs"
    )
    assert certificate.input_scope_id == (
        "finite_polynomial_decision_stratification_interval_boxes"
    )
    assert certificate.branch_consumption_certificate.proof_certified
    assert certificate.event_order_consumption_certificate.proof_certified
    assert "scoped_set_valued_constructor_proof_certified" in (
        certificate.missing_obligations
    )
    assert "validated_set_valued_scope_proof_certified" in (
        certificate.missing_obligations
    )


def test_rational_decision_stratification_reduces_to_numerator_with_denominator_exclusion():
    stratification = certify_rational_decision_stratified_branch_event_tree(
        decision_function=RationalDecisionFunctionSpec(
            decision_id="rational_event_threshold",
            numerator_coefficients=(-0.25, 1.0),
            denominator_coefficients=(2.0, 1.0),
            root_brackets=((0.24, 0.26),),
        ),
        domain=(-1.0, 1.0),
    )

    assert isinstance(stratification, RationalDecisionStratificationCertificate)
    assert stratification.proof_certified
    assert stratification.source_tree.source_type == "RationalDecisionStratification"
    assert stratification.denominator_interval[0] > 0.99
    assert stratification.denominator_interval[1] < 3.01
    assert stratification.denominator_sign == 1
    assert stratification.sign_stratum_count == 2
    assert stratification.equality_stratum_count == 1
    assert tuple(stratum.sign for stratum in stratification.strata) == (-1, 0, 1)
    assert all(
        leaf.source_leaf_certified
        for leaf in stratification.stratified_tree.leaf_certificates
    )


def test_rational_decision_stratification_blocks_denominator_zero():
    stratification = certify_rational_decision_stratified_branch_event_tree(
        decision_function=RationalDecisionFunctionSpec(
            decision_id="rational_with_pole",
            numerator_coefficients=(-0.25, 1.0),
            denominator_coefficients=(0.0, 1.0),
            root_brackets=((0.24, 0.26),),
        ),
        domain=(-1.0, 1.0),
    )

    assert not stratification.proof_certified
    assert stratification.denominator_sign == 0
    assert "rational_denominator_sign_not_isolated" in (
        stratification.missing_obligations
    )


def test_scoped_arbitrary_interval_partition_generation_accepts_supported_rational_grammar():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    stratification = certify_rational_decision_stratified_branch_event_tree(
        decision_function=RationalDecisionFunctionSpec(
            decision_id="scoped_rational_event_threshold",
            numerator_coefficients=(-0.25, 1.0),
            denominator_coefficients=(2.0, 1.0),
            root_brackets=((0.24, 0.26),),
        ),
        domain=(-1.0, 1.0),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=stratification,
        branch_root_dimension=3,
        branch_root_rank=1,
    )

    assert not certificate.proof_certified
    assert certificate.branch_constructor_source_type == (
        "RationalDecisionStratification"
    )
    assert certificate.branch_function_grammar_id == (
        "finite_rational_decision_interval_inputs_with_denominator_exclusion"
    )
    assert certificate.input_scope_id == (
        "finite_rational_decision_stratification_interval_boxes"
    )
    assert certificate.unsupported_strata == ()
    assert "scoped_set_valued_constructor_proof_certified" in (
        certificate.missing_obligations
    )
    assert "validated_set_valued_scope_proof_certified" in (
        certificate.missing_obligations
    )


def test_sturm_rational_decision_stratification_proves_denominator_exclusion_without_interval_sign():
    interval_only = certify_rational_decision_stratified_branch_event_tree(
        decision_function=RationalDecisionFunctionSpec(
            decision_id="interval_only_single_rational_event",
            numerator_coefficients=(-0.25, 1.0),
            denominator_coefficients=(1.0, 0.0, 1.0),
            root_brackets=((0.24, 0.26),),
        ),
        domain=(-1.0, 1.0),
    )
    stratification = certify_sturm_rational_decision_stratified_branch_event_tree(
        decision_function=RationalDecisionFunctionSpec(
            decision_id="single_sturm_rational_event",
            numerator_coefficients=(-0.25, 1.0),
            denominator_coefficients=(1.0, 0.0, 1.0),
        ),
        domain=(-1.0, 1.0),
    )

    assert not interval_only.proof_certified
    assert interval_only.denominator_sign == 0
    assert isinstance(stratification, RationalDecisionStratificationCertificate)
    assert stratification.proof_certified
    assert stratification.source_tree.source_type == (
        "SturmRationalDecisionStratification"
    )
    assert stratification.denominator_sign == 1
    assert stratification.sign_stratum_count == 2
    assert stratification.equality_stratum_count == 1
    assert tuple(stratum.sign for stratum in stratification.strata) == (-1, 0, 1)
    assert tuple(
        stratum.root_multiplicities
        for stratum in stratification.strata
        if stratum.equality
    ) == ((1,),)


def test_sturm_rational_decision_stratification_blocks_denominator_root():
    stratification = certify_sturm_rational_decision_stratified_branch_event_tree(
        decision_function=RationalDecisionFunctionSpec(
            decision_id="single_sturm_rational_with_pole",
            numerator_coefficients=(-0.25, 1.0),
            denominator_coefficients=(0.0, 1.0),
        ),
        domain=(-1.0, 1.0),
    )

    assert not stratification.proof_certified
    assert stratification.denominator_sign == 0
    assert any(
        "sturm_rational_denominator_zero_not_excluded" in item
        for item in stratification.missing_obligations
    )


def test_sturm_rational_decision_stratification_names_identically_zero_numerator_blocker():
    stratification = certify_sturm_rational_decision_stratified_branch_event_tree(
        decision_function=RationalDecisionFunctionSpec(
            decision_id="single_sturm_rational_zero_numerator",
            numerator_coefficients=(0.0,),
            denominator_coefficients=(1.0, 0.0, 1.0),
        ),
        domain=(-1.0, 1.0),
    )

    assert not stratification.proof_certified
    assert stratification.denominator_sign == 1
    assert any(
        "sturm_polynomial_identically_zero_on_cell" in item
        for item in stratification.missing_obligations
    )


def test_scoped_arbitrary_interval_partition_generation_accepts_supported_sturm_rational_grammar():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    stratification = certify_sturm_rational_decision_stratified_branch_event_tree(
        decision_function=RationalDecisionFunctionSpec(
            decision_id="scoped_sturm_rational_event",
            numerator_coefficients=(-0.25, 1.0),
            denominator_coefficients=(1.0, 0.0, 1.0),
        ),
        domain=(-1.0, 1.0),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=stratification,
        branch_root_dimension=3,
        branch_root_rank=1,
    )

    assert not certificate.proof_certified
    assert certificate.branch_constructor_source_type == (
        "SturmRationalDecisionStratification"
    )
    assert certificate.branch_function_grammar_id == (
        "finite_sturm_rational_decision_interval_inputs_with_denominator_exclusion"
    )
    assert certificate.input_scope_id == (
        "finite_sturm_rational_decision_stratification_interval_boxes"
    )
    assert certificate.unsupported_strata == ()
    assert "scoped_set_valued_constructor_proof_certified" in (
        certificate.missing_obligations
    )
    assert "validated_set_valued_scope_proof_certified" in (
        certificate.missing_obligations
    )


def test_rational_decision_arrangement_reduces_to_numerator_root_arrangement():
    arrangement = certify_rational_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="rational_event_order_pair",
        decision_functions=(
            RationalDecisionFunctionSpec(
                decision_id="early_rational_event",
                numerator_coefficients=(-0.25, 1.0),
                denominator_coefficients=(2.0, 1.0),
                root_brackets=((0.24, 0.26),),
            ),
            RationalDecisionFunctionSpec(
                decision_id="late_rational_event",
                numerator_coefficients=(-0.75, 1.0),
                denominator_coefficients=(3.0, -1.0),
                root_brackets=((0.74, 0.76),),
            ),
        ),
        domain=(-1.0, 1.0),
    )

    assert isinstance(arrangement, RationalDecisionArrangementStratificationCertificate)
    assert arrangement.proof_certified
    assert arrangement.source_tree.source_type == "RationalDecisionArrangement"
    assert arrangement.denominators_excluded_from_zero
    assert arrangement.sign_stratum_count == 3
    assert arrangement.equality_stratum_count == 2
    assert tuple(stratum.sign for stratum in arrangement.strata) == (
        1,
        0,
        1,
        0,
        1,
    )


def test_rational_decision_arrangement_blocks_any_denominator_zero():
    arrangement = certify_rational_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="rational_event_order_with_pole",
        decision_functions=(
            RationalDecisionFunctionSpec(
                decision_id="safe_rational_event",
                numerator_coefficients=(-0.25, 1.0),
                denominator_coefficients=(2.0, 1.0),
                root_brackets=((0.24, 0.26),),
            ),
            RationalDecisionFunctionSpec(
                decision_id="pole_rational_event",
                numerator_coefficients=(-0.75, 1.0),
                denominator_coefficients=(0.0, 1.0),
                root_brackets=((0.74, 0.76),),
            ),
        ),
        domain=(-1.0, 1.0),
    )

    assert not arrangement.proof_certified
    assert not arrangement.denominators_excluded_from_zero
    assert "rational_arrangement_denominator_sign_not_isolated" in (
        arrangement.missing_obligations
    )


def test_scoped_arbitrary_interval_partition_generation_accepts_supported_rational_arrangement_grammar():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_rational_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="scoped_rational_event_order_pair",
        decision_functions=(
            RationalDecisionFunctionSpec(
                decision_id="first_scoped_rational_event",
                numerator_coefficients=(-0.25, 1.0),
                denominator_coefficients=(2.0, 1.0),
                root_brackets=((0.24, 0.26),),
            ),
            RationalDecisionFunctionSpec(
                decision_id="second_scoped_rational_event",
                numerator_coefficients=(-0.75, 1.0),
                denominator_coefficients=(3.0, -1.0),
                root_brackets=((0.74, 0.76),),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=arrangement,
        branch_root_dimension=3,
        branch_root_rank=1,
    )

    assert not certificate.proof_certified
    assert certificate.branch_constructor_source_type == "RationalDecisionArrangement"
    assert certificate.branch_function_grammar_id == (
        "finite_rational_decision_arrangement_interval_inputs_with_denominator_exclusion"
    )
    assert certificate.input_scope_id == (
        "finite_rational_decision_arrangement_interval_boxes"
    )
    assert certificate.unsupported_strata == ()
    assert "scoped_set_valued_constructor_proof_certified" in (
        certificate.missing_obligations
    )
    assert "validated_set_valued_scope_proof_certified" in (
        certificate.missing_obligations
    )


def test_sturm_rational_decision_arrangement_proves_denominator_exclusion_without_interval_sign():
    specs = (
        RationalDecisionFunctionSpec(
            decision_id="first_sturm_rational_event",
            numerator_coefficients=(-0.25, 1.0),
            denominator_coefficients=(1.0, 0.0, 1.0),
        ),
        RationalDecisionFunctionSpec(
            decision_id="second_sturm_rational_event",
            numerator_coefficients=(0.5, 1.0),
            denominator_coefficients=(0.25, 0.0, 1.0),
        ),
    )
    interval_only = certify_rational_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="interval_only_rational_event_order_pair",
        decision_functions=specs,
        domain=(-1.0, 1.0),
    )
    arrangement = certify_sturm_rational_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="sturm_rational_event_order_pair",
        decision_functions=specs,
        domain=(-1.0, 1.0),
    )

    assert not interval_only.proof_certified
    assert interval_only.denominator_signs == (0, 0)
    assert isinstance(arrangement, RationalDecisionArrangementStratificationCertificate)
    assert arrangement.proof_certified
    assert arrangement.source_tree.source_type == "SturmRationalDecisionArrangement"
    assert arrangement.denominator_signs == (1, 1)
    assert arrangement.denominators_excluded_from_zero
    assert arrangement.sign_stratum_count == 3
    assert arrangement.equality_stratum_count == 2
    assert tuple(
        stratum.root_multiplicities
        for stratum in arrangement.strata
        if stratum.equality
    ) == ((1,), (1,))


def test_sturm_rational_decision_arrangement_blocks_denominator_root():
    arrangement = certify_sturm_rational_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="sturm_rational_event_order_with_pole",
        decision_functions=(
            RationalDecisionFunctionSpec(
                decision_id="pole_sturm_rational_event",
                numerator_coefficients=(-0.25, 1.0),
                denominator_coefficients=(0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )

    assert not arrangement.proof_certified
    assert arrangement.denominator_signs == (0,)
    assert not arrangement.denominators_excluded_from_zero
    assert "rational_arrangement_denominator_sign_not_isolated" in (
        arrangement.missing_obligations
    )


def test_sturm_rational_decision_arrangement_projects_tangent_root_derivative_data():
    arrangement = certify_sturm_rational_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="sturm_rational_tangent_event",
        decision_functions=(
            RationalDecisionFunctionSpec(
                decision_id="tangent_sturm_rational_event",
                numerator_coefficients=(0.0, 0.0, 1.0),
                denominator_coefficients=(2.0, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    tangent_stratum = next(stratum for stratum in arrangement.strata if stratum.equality)

    assert arrangement.proof_certified
    assert tangent_stratum.root_multiplicities == (2,)
    assert tangent_stratum.second_derivative_interval is not None
    assert tangent_stratum.second_derivative_interval[0] > 0.99
    assert tangent_stratum.second_derivative_interval[1] < 1.01


def test_sturm_rational_decision_arrangement_names_identically_zero_numerator_blocker():
    arrangement = certify_sturm_rational_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="sturm_rational_zero_numerator_arrangement",
        decision_functions=(
            RationalDecisionFunctionSpec(
                decision_id="zero_numerator_arrangement_event",
                numerator_coefficients=(0.0,),
                denominator_coefficients=(1.0, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )

    assert not arrangement.proof_certified
    assert arrangement.denominator_signs == (1,)
    assert any(
        "sturm_polynomial_identically_zero_on_cell" in item
        for item in arrangement.missing_obligations
    )


def test_scoped_arbitrary_interval_partition_generation_accepts_supported_sturm_rational_arrangement_grammar():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_sturm_rational_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="scoped_sturm_rational_event_order_pair",
        decision_functions=(
            RationalDecisionFunctionSpec(
                decision_id="first_scoped_sturm_rational_event",
                numerator_coefficients=(-0.25, 1.0),
                denominator_coefficients=(1.0, 0.0, 1.0),
            ),
            RationalDecisionFunctionSpec(
                decision_id="second_scoped_sturm_rational_event",
                numerator_coefficients=(0.5, 1.0),
                denominator_coefficients=(0.25, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=arrangement,
        branch_root_dimension=3,
        branch_root_rank=1,
    )

    assert not certificate.proof_certified
    assert certificate.branch_constructor_source_type == (
        "SturmRationalDecisionArrangement"
    )
    assert certificate.branch_function_grammar_id == (
        "finite_sturm_rational_decision_arrangement_interval_inputs_with_denominator_exclusion"
    )
    assert certificate.input_scope_id == (
        "finite_sturm_rational_decision_arrangement_interval_boxes"
    )
    assert certificate.unsupported_strata == ()
    assert "scoped_set_valued_constructor_proof_certified" in (
        certificate.missing_obligations
    )
    assert "validated_set_valued_scope_proof_certified" in (
        certificate.missing_obligations
    )


def test_scoped_arbitrary_interval_partition_generation_blocks_unsupported_taylor_stratum():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    stratification = certify_taylor_model_decision_stratified_branch_event_tree(
        decision_function=TaylorModelDecisionFunctionSpec(
            decision_id="unsupported_scoped_taylor_event_tie",
            coefficients=(-0.25, 1.0),
            expansion_center=0.0,
            remainder_bound=1.0e-4,
            derivative_remainder_bound=5.0e-2,
            root_brackets=((0.24, 0.26),),
            witness_kind="unsupported",
        ),
        domain=(-1.0, 1.0),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=stratification,
        branch_root_dimension=3,
        branch_root_rank=1,
    )

    assert not certificate.proof_certified
    assert not certificate.certified
    assert not certificate.arbitrary_partition_generation_claimed
    assert certificate.event_function_grammar_id == (
        "finite_taylor_model_decision_interval_inputs_with_weierstrass_certificate"
    )
    assert certificate.input_scope_id == (
        "finite_taylor_model_decision_interval_inputs_with_weierstrass_certificate"
    )
    assert certificate.unsupported_strata
    assert "no_unsupported_analytic_strata" in certificate.missing_obligations
    assert any(
        item.startswith("unsupported:")
        for item in certificate.missing_obligations
    )


def test_scoped_arbitrary_interval_partition_generation_rejects_raw_boolean_constructor():
    theorem = certify_finite_target_completeness_theorem(dimension=3)

    try:
        certify_arbitrary_interval_input_partition_generation(
            theorem,
            branch_constructor_certificate=True,
            branch_root_dimension=3,
            branch_root_rank=1,
        )
    except TypeError as error:
        assert "constructor certificate" in str(error)
    else:
        raise AssertionError("raw boolean constructor was accepted")


def test_scoped_arbitrary_interval_partition_manifest_names_supported_grammars():
    assert set(SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS).issubset(
        set(CONSTRUCTOR_DERIVED_RECURSIVE_SOURCE_SCOPES)
    )
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "AffineDecisionStratification"
    ] == "finite_affine_decision_interval_inputs"
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "AffineDecisionArrangement"
    ] == "finite_affine_decision_arrangement_interval_inputs"
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "PolynomialDecisionStratification"
    ] == "finite_polynomial_decision_interval_inputs"
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "PolynomialDecisionArrangement"
    ] == "finite_polynomial_decision_arrangement_interval_inputs"
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "SturmPolynomialDecisionStratification"
    ] == "finite_sturm_polynomial_decision_interval_inputs"
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "SturmPolynomialDecisionArrangement"
    ] == "finite_sturm_polynomial_decision_arrangement_interval_inputs"
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "RationalDecisionStratification"
    ] == "finite_rational_decision_interval_inputs_with_denominator_exclusion"
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "SturmRationalDecisionStratification"
    ] == (
        "finite_sturm_rational_decision_interval_inputs_with_denominator_exclusion"
    )
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "RationalDecisionArrangement"
    ] == (
        "finite_rational_decision_arrangement_interval_inputs_with_denominator_exclusion"
    )
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "SturmRationalDecisionArrangement"
    ] == (
        "finite_sturm_rational_decision_arrangement_interval_inputs_with_denominator_exclusion"
    )
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "TaylorModelDecisionStratification"
    ] == (
        "finite_taylor_model_decision_interval_inputs_with_weierstrass_certificate"
    )
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "TaylorModelDecisionArrangement"
    ] == (
        "finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate"
    )
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "QuadraticDoubleRootArrangement"
    ] == "finite_quadratic_double_root_arrangement_interval_inputs"
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "PolynomialRootArrangement"
    ] == "finite_computed_polynomial_root_arrangement_interval_inputs"
    for source_type in SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS:
        scope_id, scope_detail, scope_label = (
            CONSTRUCTOR_DERIVED_RECURSIVE_SOURCE_SCOPES[source_type]
        )
        assert scope_id
        assert scope_detail
        assert scope_label


def test_scoped_arbitrary_interval_partition_generation_covers_declared_grammars():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    cases = (
        (
            "AffineDecisionStratification",
            certify_affine_decision_stratified_branch_event_tree(
                decision_id="scoped_single_affine_event",
                coefficients=(0.0, 1.0),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_affine_decision_stratification_interval_boxes",
        ),
        (
            "AffineDecisionArrangement",
            certify_affine_decision_arrangement_stratified_branch_event_tree(
                arrangement_id="scoped_all_affine_event",
                decision_functions=(
                    PolynomialDecisionFunctionSpec(
                        decision_id="affine_threshold",
                        coefficients=(0.0, 1.0),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            2,
            "finite_affine_decision_arrangement_interval_boxes",
        ),
        (
            "PolynomialDecisionStratification",
            certify_polynomial_decision_stratified_branch_event_tree(
                decision_id="scoped_all_polynomial_event",
                coefficients=(0.0, 1.0),
                domain=(-1.0, 1.0),
                root_brackets=((-0.01, 0.01),),
            ),
            3,
            1,
            "finite_polynomial_decision_stratification_interval_boxes",
        ),
        (
            "PolynomialDecisionArrangement",
            certify_polynomial_decision_arrangement_stratified_branch_event_tree(
                arrangement_id="scoped_all_polynomial_arrangement",
                decision_functions=(
                    PolynomialDecisionFunctionSpec(
                        decision_id="polynomial_threshold",
                        coefficients=(0.0, 1.0),
                        root_brackets=((-0.01, 0.01),),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            2,
            "finite_polynomial_decision_arrangement_interval_boxes",
        ),
        (
            "SturmPolynomialDecisionStratification",
            certify_sturm_polynomial_decision_stratified_branch_event_tree(
                decision_id="scoped_all_sturm_polynomial_event",
                coefficients=(1.0, 0.0, 1.0),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_sturm_polynomial_decision_stratification_interval_boxes",
        ),
        (
            "RationalDecisionStratification",
            certify_rational_decision_stratified_branch_event_tree(
                decision_function=RationalDecisionFunctionSpec(
                    decision_id="scoped_all_rational_event",
                    numerator_coefficients=(-0.25, 1.0),
                    denominator_coefficients=(2.0, 1.0),
                    root_brackets=((0.24, 0.26),),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_rational_decision_stratification_interval_boxes",
        ),
        (
            "SturmRationalDecisionStratification",
            certify_sturm_rational_decision_stratified_branch_event_tree(
                decision_function=RationalDecisionFunctionSpec(
                    decision_id="scoped_all_sturm_rational_event",
                    numerator_coefficients=(-0.25, 1.0),
                    denominator_coefficients=(1.0, 0.0, 1.0),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_sturm_rational_decision_stratification_interval_boxes",
        ),
        (
            "RationalDecisionArrangement",
            certify_rational_decision_arrangement_stratified_branch_event_tree(
                arrangement_id="scoped_all_rational_arrangement",
                decision_functions=(
                    RationalDecisionFunctionSpec(
                        decision_id="first_scoped_all_rational_event",
                        numerator_coefficients=(-0.25, 1.0),
                        denominator_coefficients=(2.0, 1.0),
                        root_brackets=((0.24, 0.26),),
                    ),
                    RationalDecisionFunctionSpec(
                        decision_id="second_scoped_all_rational_event",
                        numerator_coefficients=(-0.75, 1.0),
                        denominator_coefficients=(3.0, -1.0),
                        root_brackets=((0.74, 0.76),),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_rational_decision_arrangement_interval_boxes",
        ),
        (
            "SturmRationalDecisionArrangement",
            certify_sturm_rational_decision_arrangement_stratified_branch_event_tree(
                arrangement_id="scoped_all_sturm_rational_arrangement",
                decision_functions=(
                    RationalDecisionFunctionSpec(
                        decision_id="first_scoped_all_sturm_rational_event",
                        numerator_coefficients=(-0.25, 1.0),
                        denominator_coefficients=(1.0, 0.0, 1.0),
                    ),
                    RationalDecisionFunctionSpec(
                        decision_id="second_scoped_all_sturm_rational_event",
                        numerator_coefficients=(0.5, 1.0),
                        denominator_coefficients=(0.25, 0.0, 1.0),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_sturm_rational_decision_arrangement_interval_boxes",
        ),
        (
            "SturmPolynomialDecisionArrangement",
            certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree(
                arrangement_id="scoped_all_sturm_arrangement",
                decision_functions=(
                    PolynomialDecisionFunctionSpec(
                        decision_id="sturm_cubic_threshold",
                        coefficients=(0.0, -0.25, 0.0, 1.0),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            2,
            "finite_sturm_polynomial_decision_arrangement_interval_boxes",
        ),
        (
            "QuadraticDoubleRootArrangement",
            certify_quadratic_decision_arrangement_stratified_branch_event_tree(
                arrangement_id="scoped_all_quadratic_double_root",
                decision_functions=(
                    PolynomialDecisionFunctionSpec(
                        decision_id="quadratic_double_threshold",
                        coefficients=(0.0, 0.0, 1.0),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            2,
            "finite_quadratic_double_root_decision_arrangement_interval_boxes",
        ),
        (
            "PolynomialRootArrangement",
            certify_quadratic_decision_arrangement_stratified_branch_event_tree(
                arrangement_id="scoped_all_computed_polynomial_root",
                decision_functions=(
                    PolynomialDecisionFunctionSpec(
                        decision_id="left_quadratic_threshold",
                        coefficients=(-0.25, 0.0, 1.0),
                    ),
                    PolynomialDecisionFunctionSpec(
                        decision_id="shifted_quadratic_threshold",
                        coefficients=(-0.375, 0.25, 1.0),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            2,
            "finite_computed_polynomial_root_arrangement_interval_boxes",
        ),
        (
            "TaylorModelDecisionStratification",
            certify_taylor_model_decision_stratified_branch_event_tree(
                decision_function=TaylorModelDecisionFunctionSpec(
                    decision_id="scoped_all_taylor_event",
                    coefficients=(-0.25, 1.0),
                    expansion_center=0.0,
                    remainder_bound=1.0e-4,
                    derivative_remainder_bound=5.0e-2,
                    root_brackets=((0.24, 0.26),),
                    witness_kind="weierstrass_simple_root",
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_taylor_model_decision_interval_inputs_with_weierstrass_certificate",
        ),
        (
            "TaylorModelDecisionArrangement",
            certify_taylor_model_decision_arrangement_stratified_branch_event_tree(
                arrangement_id="scoped_all_taylor_arrangement",
                decision_functions=(
                    TaylorModelDecisionFunctionSpec(
                        decision_id="first_taylor_event",
                        coefficients=(-0.25, 1.0),
                        expansion_center=0.0,
                        remainder_bound=1.0e-4,
                        derivative_remainder_bound=5.0e-2,
                        root_brackets=((0.24, 0.26),),
                        witness_kind="weierstrass_simple_root",
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate",
        ),
        (
            "AxisAlignedAffineBoxArrangement",
            certify_affine_box_decision_arrangement_stratified_branch_event_tree(
                arrangement_id="scoped_all_axis_box",
                decision_functions=(
                    AffineBoxDecisionFunctionSpec(
                        decision_id="x_threshold",
                        coefficients=(0.0, 1.0, 0.0),
                    ),
                ),
                domain_box=((-1.0, 1.0), (-1.0, 1.0)),
            ),
            2,
            2,
            "finite_axis_aligned_affine_box_arrangement_interval_boxes",
        ),
        (
            "AffineHalfspaceDecision",
            certify_affine_halfspace_decision_stratified_branch_event_tree(
                decision_id="scoped_all_halfspace_decision",
                coefficients=(0.0, 1.0, 1.0),
                domain_box=((-1.0, 1.0), (-1.0, 1.0)),
                slab_half_width=0.25,
            ),
            2,
            2,
            "finite_affine_halfspace_decision_interval_boxes",
        ),
        (
            "AffineHalfspaceArrangement",
            certify_affine_halfspace_arrangement_stratified_branch_event_tree(
                arrangement_id="scoped_all_halfspace_2d",
                decision_functions=(
                    AffineBoxDecisionFunctionSpec(
                        decision_id="diagonal_boundary",
                        coefficients=(0.0, 1.0, 1.0),
                    ),
                ),
                domain_box=((-1.0, 1.0), (-1.0, 1.0)),
                slab_half_width=0.2,
            ),
            2,
            2,
            "finite_2d_affine_halfspace_arrangement_interval_boxes",
        ),
        (
            "AffineHalfspace3DArrangement",
            certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
                arrangement_id="scoped_all_halfspace_3d",
                decision_functions=(
                    AffineBoxDecisionFunctionSpec(
                        decision_id="diagonal_plane_boundary",
                        coefficients=(0.0, 1.0, 1.0, 1.0),
                    ),
                ),
                domain_box=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
                slab_half_width=0.2,
            ),
            3,
            2,
            "finite_3d_affine_halfspace_arrangement_interval_boxes",
        ),
    )
    observed_sources = set()

    for source_type, constructor, root_dimension, root_rank, input_scope in cases:
        certificate = certify_arbitrary_interval_input_partition_generation(
            theorem,
            branch_constructor_certificate=constructor,
            branch_root_dimension=root_dimension,
            branch_root_rank=root_rank,
        )
        observed_sources.add(source_type)

        assert constructor.source_tree.source_type == source_type
        assert not certificate.proof_certified
        assert not certificate.arbitrary_partition_generation_claimed
        assert certificate.branch_constructor_source_type == source_type
        assert certificate.event_order_constructor_source_type == source_type
        assert certificate.branch_constructor_input_scope_id == input_scope
        assert certificate.event_order_constructor_input_scope_id == input_scope
        assert certificate.branch_function_grammar_id == (
            SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[source_type]
        )
        assert certificate.event_order_function_grammar_id == (
            SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[source_type]
        )
        assert certificate.event_function_grammar_id == (
            SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[source_type]
        )
        assert certificate.input_scope_id == input_scope
        assert certificate.unsupported_strata == ()
        assert "scoped_set_valued_constructor_proof_certified" in (
            certificate.missing_obligations
        )
        assert "validated_set_valued_scope_proof_certified" in (
            certificate.missing_obligations
        )

    assert observed_sources == set(SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS)


def test_supported_event_function_stratification_generation_covers_declared_grammars():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    cases = (
        (
            "AffineDecisionStratification",
            SupportedEventFunctionGrammarInput(
                source_type="AffineDecisionStratification",
                decision_id="generated_single_affine_event",
                coefficients=(0.0, 1.0),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_affine_decision_stratification_interval_boxes",
        ),
        (
            "AffineDecisionArrangement",
            SupportedEventFunctionGrammarInput(
                source_type="AffineDecisionArrangement",
                arrangement_id="generated_affine_event",
                polynomial_decision_functions=(
                    PolynomialDecisionFunctionSpec(
                        decision_id="affine_threshold",
                        coefficients=(0.0, 1.0),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            2,
            "finite_affine_decision_arrangement_interval_boxes",
        ),
        (
            "PolynomialDecisionStratification",
            SupportedEventFunctionGrammarInput(
                source_type="PolynomialDecisionStratification",
                decision_id="generated_polynomial_event",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_polynomial_decision_stratification_interval_boxes",
        ),
        (
            "SturmPolynomialDecisionStratification",
            SupportedEventFunctionGrammarInput(
                source_type="SturmPolynomialDecisionStratification",
                decision_id="generated_sturm_polynomial_event",
                coefficients=(1.0, 0.0, 1.0),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_sturm_polynomial_decision_stratification_interval_boxes",
        ),
        (
            "PolynomialDecisionArrangement",
            SupportedEventFunctionGrammarInput(
                source_type="PolynomialDecisionArrangement",
                arrangement_id="generated_polynomial_arrangement",
                polynomial_decision_functions=(
                    PolynomialDecisionFunctionSpec(
                        decision_id="polynomial_threshold",
                        coefficients=(0.0, 1.0),
                        root_brackets=((-0.01, 0.01),),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            2,
            "finite_polynomial_decision_arrangement_interval_boxes",
        ),
        (
            "RationalDecisionStratification",
            SupportedEventFunctionGrammarInput(
                source_type="RationalDecisionStratification",
                rational_decision_function=RationalDecisionFunctionSpec(
                    decision_id="generated_rational_event",
                    numerator_coefficients=(-0.25, 1.0),
                    denominator_coefficients=(2.0, 1.0),
                    root_brackets=((0.24, 0.26),),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_rational_decision_stratification_interval_boxes",
        ),
        (
            "SturmRationalDecisionStratification",
            SupportedEventFunctionGrammarInput(
                source_type="SturmRationalDecisionStratification",
                rational_decision_function=RationalDecisionFunctionSpec(
                    decision_id="generated_sturm_rational_event",
                    numerator_coefficients=(-0.25, 1.0),
                    denominator_coefficients=(1.0, 0.0, 1.0),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_sturm_rational_decision_stratification_interval_boxes",
        ),
        (
            "RationalDecisionArrangement",
            SupportedEventFunctionGrammarInput(
                source_type="RationalDecisionArrangement",
                arrangement_id="generated_rational_arrangement",
                rational_decision_functions=(
                    RationalDecisionFunctionSpec(
                        decision_id="first_generated_rational_event",
                        numerator_coefficients=(-0.25, 1.0),
                        denominator_coefficients=(2.0, 1.0),
                        root_brackets=((0.24, 0.26),),
                    ),
                    RationalDecisionFunctionSpec(
                        decision_id="second_generated_rational_event",
                        numerator_coefficients=(-0.75, 1.0),
                        denominator_coefficients=(3.0, -1.0),
                        root_brackets=((0.74, 0.76),),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_rational_decision_arrangement_interval_boxes",
        ),
        (
            "SturmRationalDecisionArrangement",
            SupportedEventFunctionGrammarInput(
                source_type="SturmRationalDecisionArrangement",
                arrangement_id="generated_sturm_rational_arrangement",
                rational_decision_functions=(
                    RationalDecisionFunctionSpec(
                        decision_id="first_generated_sturm_rational_event",
                        numerator_coefficients=(-0.25, 1.0),
                        denominator_coefficients=(1.0, 0.0, 1.0),
                    ),
                    RationalDecisionFunctionSpec(
                        decision_id="second_generated_sturm_rational_event",
                        numerator_coefficients=(0.5, 1.0),
                        denominator_coefficients=(0.25, 0.0, 1.0),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_sturm_rational_decision_arrangement_interval_boxes",
        ),
        (
            "SturmPolynomialDecisionArrangement",
            SupportedEventFunctionGrammarInput(
                source_type="SturmPolynomialDecisionArrangement",
                arrangement_id="generated_sturm_polynomial_arrangement",
                polynomial_decision_functions=(
                    PolynomialDecisionFunctionSpec(
                        decision_id="sturm_cubic_threshold",
                        coefficients=(0.0, -0.25, 0.0, 1.0),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            2,
            "finite_sturm_polynomial_decision_arrangement_interval_boxes",
        ),
        (
            "QuadraticDoubleRootArrangement",
            SupportedEventFunctionGrammarInput(
                source_type="QuadraticDoubleRootArrangement",
                arrangement_id="generated_quadratic_double_root",
                polynomial_decision_functions=(
                    PolynomialDecisionFunctionSpec(
                        decision_id="quadratic_double_threshold",
                        coefficients=(0.0, 0.0, 1.0),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            2,
            "finite_quadratic_double_root_decision_arrangement_interval_boxes",
        ),
        (
            "PolynomialRootArrangement",
            SupportedEventFunctionGrammarInput(
                source_type="PolynomialRootArrangement",
                arrangement_id="generated_computed_polynomial_root",
                polynomial_decision_functions=(
                    PolynomialDecisionFunctionSpec(
                        decision_id="left_quadratic_threshold",
                        coefficients=(-0.25, 0.0, 1.0),
                    ),
                    PolynomialDecisionFunctionSpec(
                        decision_id="shifted_quadratic_threshold",
                        coefficients=(-0.375, 0.25, 1.0),
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            2,
            "finite_computed_polynomial_root_arrangement_interval_boxes",
        ),
        (
            "TaylorModelDecisionStratification",
            SupportedEventFunctionGrammarInput(
                source_type="TaylorModelDecisionStratification",
                taylor_model_decision_function=TaylorModelDecisionFunctionSpec(
                    decision_id="generated_taylor_event",
                    coefficients=(-0.25, 1.0),
                    expansion_center=0.0,
                    remainder_bound=1.0e-4,
                    derivative_remainder_bound=5.0e-2,
                    root_brackets=((0.24, 0.26),),
                    witness_kind="weierstrass_simple_root",
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_taylor_model_decision_interval_inputs_with_weierstrass_certificate",
        ),
        (
            "TaylorModelDecisionArrangement",
            SupportedEventFunctionGrammarInput(
                source_type="TaylorModelDecisionArrangement",
                arrangement_id="generated_taylor_arrangement",
                taylor_model_decision_functions=(
                    TaylorModelDecisionFunctionSpec(
                        decision_id="first_generated_taylor_event",
                        coefficients=(-0.25, 1.0),
                        expansion_center=0.0,
                        remainder_bound=1.0e-4,
                        derivative_remainder_bound=5.0e-2,
                        root_brackets=((0.24, 0.26),),
                        witness_kind="weierstrass_simple_root",
                    ),
                ),
                domain=(-1.0, 1.0),
            ),
            3,
            1,
            "finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate",
        ),
        (
            "AxisAlignedAffineBoxArrangement",
            SupportedEventFunctionGrammarInput(
                source_type="AxisAlignedAffineBoxArrangement",
                arrangement_id="generated_axis_box",
                affine_box_decision_functions=(
                    AffineBoxDecisionFunctionSpec(
                        decision_id="x_threshold",
                        coefficients=(0.0, 1.0, 0.0),
                    ),
                ),
                domain_box=((-1.0, 1.0), (-1.0, 1.0)),
            ),
            2,
            2,
            "finite_axis_aligned_affine_box_arrangement_interval_boxes",
        ),
        (
            "AffineHalfspaceDecision",
            SupportedEventFunctionGrammarInput(
                source_type="AffineHalfspaceDecision",
                decision_id="generated_halfspace_decision",
                coefficients=(0.0, 1.0, 1.0),
                domain_box=((-1.0, 1.0), (-1.0, 1.0)),
                slab_half_width=0.25,
            ),
            2,
            2,
            "finite_affine_halfspace_decision_interval_boxes",
        ),
        (
            "AffineHalfspaceArrangement",
            SupportedEventFunctionGrammarInput(
                source_type="AffineHalfspaceArrangement",
                arrangement_id="generated_halfspace_2d",
                affine_box_decision_functions=(
                    AffineBoxDecisionFunctionSpec(
                        decision_id="diagonal_boundary",
                        coefficients=(0.0, 1.0, 1.0),
                    ),
                ),
                domain_box=((-1.0, 1.0), (-1.0, 1.0)),
                slab_half_width=0.2,
            ),
            2,
            2,
            "finite_2d_affine_halfspace_arrangement_interval_boxes",
        ),
        (
            "AffineHalfspace3DArrangement",
            SupportedEventFunctionGrammarInput(
                source_type="AffineHalfspace3DArrangement",
                arrangement_id="generated_halfspace_3d",
                affine_box_decision_functions=(
                    AffineBoxDecisionFunctionSpec(
                        decision_id="diagonal_plane_boundary",
                        coefficients=(0.0, 1.0, 1.0, 1.0),
                    ),
                ),
                domain_box=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
                slab_half_width=0.2,
            ),
            3,
            2,
            "finite_3d_affine_halfspace_arrangement_interval_boxes",
        ),
    )
    observed_sources = set()

    for source_type, grammar_input, root_dimension, root_rank, input_scope in cases:
        certificate = certify_supported_event_function_stratification_generation(
            theorem,
            grammar_input=grammar_input,
            branch_root_dimension=root_dimension,
            branch_root_rank=root_rank,
        )
        observed_sources.add(source_type)

        assert isinstance(
            certificate,
            SupportedEventFunctionStratificationGenerationCertificate,
        )
        assert not certificate.proof_certified
        assert not certificate.certified
        assert certificate.constructor_source_type == source_type
        assert certificate.event_function_grammar_id == (
            SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[source_type]
        )
        assert certificate.input_scope_id == input_scope
        assert not (
            certificate.partition_generation_certificate
            .arbitrary_partition_generation_claimed
        )
        assert "generated_partition_bridge_proof_certified" in (
            certificate.missing_obligations
        )

    assert observed_sources == set(SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS)


def test_supported_event_function_generation_blocks_source_mismatch():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialRootArrangement",
            arrangement_id="mismatched_generated_quadratic_double_root",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="quadratic_double_threshold",
                    coefficients=(0.0, 0.0, 1.0),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )

    assert not certificate.proof_certified
    assert certificate.constructor_source_type == "QuadraticDoubleRootArrangement"
    assert "generated_constructor_source_matches_requested_grammar" in (
        certificate.missing_obligations
    )


def test_supported_event_function_generation_preserves_mixed_branch_event_sources():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="AffineHalfspaceDecision",
            decision_id="generated_mixed_branch_boundary",
            coefficients=(0.0, 1.0, 1.0),
            domain_box=((-1.0, 1.0), (-1.0, 1.0)),
            slab_half_width=0.25,
        ),
        event_order_grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_mixed_event_arrangement",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="event_minus_target",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=2,
        branch_root_rank=2,
        event_order_root_dimension=3,
        event_order_root_rank=2,
    )
    partition = certificate.partition_generation_certificate

    # The mixed constructor metadata is preserved, but composing it with the
    # unaudited finite-target source does not promote the generated bridge to
    # a theorem certificate.
    assert not certificate.proof_certified
    assert not certificate.certified
    assert certificate.constructor_source_type == "AffineHalfspaceDecision"
    assert (
        certificate.event_order_constructor_source_type
        == "PolynomialDecisionArrangement"
    )
    assert partition.branch_constructor_source_type == "AffineHalfspaceDecision"
    assert (
        partition.event_order_constructor_source_type
        == "PolynomialDecisionArrangement"
    )
    assert partition.branch_constructor_input_scope_id == (
        "finite_affine_halfspace_decision_interval_boxes"
    )
    assert partition.event_order_constructor_input_scope_id == (
        "finite_polynomial_decision_arrangement_interval_boxes"
    )
    assert partition.event_function_grammar_id == (
        "finite_mixed_supported_constructor_interval_inputs"
        "(branch=finite_affine_halfspace_decision_interval_inputs;"
        "event=finite_polynomial_decision_arrangement_interval_inputs)"
    )
    assert certificate.input_scope_id == (
        "finite_mixed_constructor_branch_event_interval_boxes"
    )
    assert "generated_partition_bridge_proof_certified" in (
        certificate.missing_obligations
    )


def test_supported_event_function_generation_rejects_stale_partition_bridge_provenance():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_polynomial_arrangement_for_stale_bridge",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="polynomial_threshold",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )
    stale_source = replace(
        certificate,
        partition_generation_certificate=replace(
            certificate.partition_generation_certificate,
            branch_constructor_source_type="AffineDecisionArrangement",
        ),
    )
    stale_grammar = replace(
        certificate,
        partition_generation_certificate=replace(
            certificate.partition_generation_certificate,
            event_order_function_grammar_id=(
                "finite_affine_decision_arrangement_interval_inputs"
            ),
        ),
    )

    assert not certificate.proof_certified
    assert "generated_partition_bridge_proof_certified" in (
        certificate.missing_obligations
    )
    assert not stale_source.proof_certified
    assert "generated_partition_bridge_branch_source_matches_constructor" in (
        stale_source.missing_obligations
    )
    assert not stale_grammar.proof_certified
    assert "generated_partition_bridge_event_order_grammar_matches_input" in (
        stale_grammar.missing_obligations
    )


def test_scoped_partition_rejects_forged_scope_fields_directly():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_polynomial_arrangement_for_scope_forgery",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="polynomial_threshold",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )
    partition = certificate.partition_generation_certificate
    forged_scope = replace(
        partition,
        input_scope_id="forged_interval_scope",
        event_function_grammar_id="forged_event_function_grammar",
        branch_function_grammar_id="forged_branch_grammar",
        event_order_function_grammar_id="forged_event_order_grammar",
        branch_constructor_input_scope_id="forged_branch_scope",
        event_order_constructor_input_scope_id="forged_event_scope",
    )

    assert not certificate.proof_certified
    assert not partition.proof_certified
    assert partition.scope_fields_match_constructor_chain
    assert not forged_scope.proof_certified
    assert not forged_scope.scope_fields_match_constructor_chain
    assert "scoped_partition_scope_fields_match_constructor_chain" in (
        forged_scope.missing_obligations
    )


def test_validated_set_valued_constructor_rejects_forged_scope_directly():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_polynomial_arrangement_for_validated_scope",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="polynomial_threshold",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )
    validated = (
        certificate.partition_generation_certificate
        .validated_set_valued_constructor_certificate
    )
    forged_scope = replace(
        validated,
        input_scope_id="forged_validated_interval_scope",
    )

    assert not validated.proof_certified
    assert validated.input_scope_matches_constructor
    assert not forged_scope.proof_certified
    assert not forged_scope.input_scope_matches_constructor
    assert "validated_interval_input_scope_matches_constructor" in (
        forged_scope.missing_obligations
    )


def test_supported_event_function_generation_rejects_same_source_constructor_partition_swap():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_polynomial_arrangement_for_swap_a",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="polynomial_threshold_a",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )
    other_certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_polynomial_arrangement_for_swap_b",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="polynomial_threshold_b",
                    coefficients=(0.25, 1.0),
                    root_brackets=((-0.26, -0.24),),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )
    swapped = replace(
        certificate,
        constructor_certificate=other_certificate.constructor_certificate,
        partition_generation_certificate=(
            other_certificate.partition_generation_certificate
        ),
    )

    assert not certificate.proof_certified
    assert not other_certificate.proof_certified
    assert not swapped.proof_certified
    assert "generated_constructor_input_matches_grammar_input" in (
        swapped.missing_obligations
    )


def test_supported_event_function_generation_rejects_replayed_branch_payload_policy():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_polynomial_arrangement_for_payload_policy",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="polynomial_threshold",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
            equality_resolution_policy="lower_dimensional_recursive_stratum",
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )
    replayed_policy = replace(
        certificate,
        grammar_input=replace(
            certificate.grammar_input,
            equality_resolution_policy="selector_policy",
        ),
    )

    assert not certificate.proof_certified
    assert replayed_policy.constructor_input_matches_grammar_input
    assert not replayed_policy.proof_certified
    assert "grammar_input_payload_matches_generation" in (
        replayed_policy.missing_obligations
    )


def test_supported_event_function_generation_rejects_forged_branch_payload_policy():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_polynomial_arrangement_for_forged_payload",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="polynomial_threshold",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
            equality_resolution_policy="lower_dimensional_recursive_stratum",
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )
    forged_policy = replace(
        certificate,
        grammar_input=replace(
            certificate.grammar_input,
            equality_resolution_policy="selector_policy",
        ),
        branch_grammar_payload_signature=(
            certificate.branch_grammar_payload_signature[0],
            "selector_policy",
            certificate.branch_grammar_payload_signature[2],
        ),
    )

    assert not certificate.proof_certified
    assert forged_policy.constructor_input_matches_grammar_input
    assert forged_policy.grammar_payload_matches_generation
    assert not forged_policy.proof_certified
    assert "generated_constructor_replays_from_grammar_input" in (
        forged_policy.missing_obligations
    )


def test_supported_event_function_generation_rejects_replayed_event_payload_budget():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="AffineDecisionStratification",
            decision_id="generated_branch_affine_for_event_payload",
            coefficients=(0.0, 1.0),
            domain=(-1.0, 1.0),
        ),
        event_order_grammar_input=SupportedEventFunctionGrammarInput(
            source_type="SturmPolynomialDecisionStratification",
            decision_id="generated_event_sturm_for_payload_budget",
            coefficients=(1.0, 0.0, 1.0),
            domain=(-1.0, 1.0),
            max_bisection_depth=64,
        ),
        branch_root_dimension=3,
        branch_root_rank=1,
        event_order_root_dimension=3,
        event_order_root_rank=1,
    )
    replayed_event_budget = replace(
        certificate,
        event_order_grammar_input=replace(
            certificate.event_order_grammar_input,
            max_bisection_depth=32,
        ),
    )

    assert not certificate.proof_certified
    assert replayed_event_budget.event_order_constructor_input_matches_grammar_input
    assert not replayed_event_budget.proof_certified
    assert "event_order_grammar_input_payload_matches_generation" in (
        replayed_event_budget.missing_obligations
    )


def test_supported_event_function_generation_rejects_forged_event_payload_budget():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="AffineDecisionStratification",
            decision_id="generated_branch_affine_for_forged_event_payload",
            coefficients=(0.0, 1.0),
            domain=(-1.0, 1.0),
        ),
        event_order_grammar_input=SupportedEventFunctionGrammarInput(
            source_type="SturmPolynomialDecisionStratification",
            decision_id="generated_event_sturm_for_forged_payload_budget",
            coefficients=(1.0, 0.0, 1.0),
            domain=(-1.0, 1.0),
            max_bisection_depth=64,
        ),
        branch_root_dimension=3,
        branch_root_rank=1,
        event_order_root_dimension=3,
        event_order_root_rank=1,
    )
    forged_event_budget = replace(
        certificate,
        event_order_grammar_input=replace(
            certificate.event_order_grammar_input,
            max_bisection_depth=0,
        ),
        event_order_grammar_payload_signature=(
            certificate.event_order_grammar_payload_signature[0],
            certificate.event_order_grammar_payload_signature[1],
            0,
        ),
    )

    assert not certificate.proof_certified
    assert forged_event_budget.event_order_constructor_input_matches_grammar_input
    assert forged_event_budget.event_order_grammar_payload_matches_generation
    assert not forged_event_budget.proof_certified
    assert "generated_event_order_constructor_replays_from_grammar_input" in (
        forged_event_budget.missing_obligations
    )


def test_supported_event_function_generation_rejects_mutated_generated_sturm_evidence():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="SturmPolynomialDecisionStratification",
            decision_id="generated_sturm_for_mutation_guard",
            coefficients=(1.0, 0.0, 1.0),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=1,
    )
    mutated = replace(
        certificate,
        constructor_certificate=replace(
            certificate.constructor_certificate,
            root_brackets=((-0.1, 0.1),),
        ),
    )

    assert not certificate.proof_certified
    assert mutated.constructor_input_matches_grammar_input
    assert not mutated.proof_certified
    assert "generated_constructor_evidence_matches_generation" in (
        mutated.missing_obligations
    )


def test_supported_event_function_generation_rejects_mutated_event_order_sturm_evidence():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="AffineDecisionStratification",
            decision_id="generated_branch_affine_for_event_mutation_guard",
            coefficients=(0.0, 1.0),
            domain=(-1.0, 1.0),
        ),
        event_order_grammar_input=SupportedEventFunctionGrammarInput(
            source_type="SturmPolynomialDecisionStratification",
            decision_id="generated_event_sturm_for_mutation_guard",
            coefficients=(1.0, 0.0, 1.0),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=1,
        event_order_root_dimension=3,
        event_order_root_rank=1,
    )
    mutated = replace(
        certificate,
        event_order_constructor_certificate=replace(
            certificate.event_order_constructor_certificate,
            root_brackets=((-0.1, 0.1),),
        ),
    )

    assert not certificate.proof_certified
    assert mutated.event_order_constructor_input_matches_grammar_input
    assert not mutated.proof_certified
    assert "generated_event_order_constructor_evidence_matches_generation" in (
        mutated.missing_obligations
    )


def test_supported_event_function_generation_rejects_stale_constructor_source_tree():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="AffineDecisionStratification",
            decision_id="generated_affine_for_source_tree_guard",
            coefficients=(0.0, 1.0),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=1,
    )
    alternate = certify_affine_decision_stratified_branch_event_tree(
        decision_id="alternate_affine_source_tree",
        coefficients=(-0.25, 1.0),
        domain=(-1.0, 1.0),
    )
    mutated_constructor = replace(
        certificate.constructor_certificate,
        source_tree=alternate.source_tree,
    )
    mutated = replace(certificate, constructor_certificate=mutated_constructor)

    assert not certificate.proof_certified
    assert mutated.constructor_input_matches_grammar_input
    assert not mutated_constructor.proof_certified
    assert "source_tree_matches_stratified_tree" in (
        mutated_constructor.missing_obligations
    )
    assert not mutated.proof_certified
    assert "generated_constructor_proof_certified" in mutated.missing_obligations
    assert "constructor:source_tree_matches_stratified_tree" in (
        mutated.missing_obligations
    )


def test_supported_event_function_generation_rejects_mutated_generated_stratum_interval_evidence():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="SturmPolynomialDecisionStratification",
            decision_id="generated_sturm_for_stratum_interval_guard",
            coefficients=(1.0, 0.0, -0.25),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=1,
    )
    mutated_strata = tuple(
        (
            replace(stratum, value_interval=(999.0, 1000.0))
            if index == 0
            else stratum
        )
        for index, stratum in enumerate(certificate.constructor_certificate.strata)
    )
    mutated_constructor = replace(
        certificate.constructor_certificate,
        strata=mutated_strata,
    )
    mutated = replace(certificate, constructor_certificate=mutated_constructor)

    assert not certificate.proof_certified
    assert mutated_constructor.proof_certified
    assert mutated.constructor_input_matches_grammar_input
    assert not mutated.proof_certified
    assert "generated_constructor_evidence_matches_generation" in (
        mutated.missing_obligations
    )


def test_supported_event_function_generation_rejects_truthy_constructor_proof_flags():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_polynomial_arrangement_for_truthy_guard",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="polynomial_threshold",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )
    spoofed_constructor = replace(
        certificate,
        constructor_certificate=SimpleNamespace(
            source_tree=SimpleNamespace(source_type="PolynomialDecisionArrangement"),
            proof_certified="yes",
            missing_obligations=(),
        ),
    )
    spoofed_partition = replace(
        certificate,
        partition_generation_certificate=replace(
            certificate.partition_generation_certificate,
            set_valued_constructor_certificate=SimpleNamespace(
                proof_certified="yes",
            ),
        ),
    )

    assert not certificate.proof_certified
    assert not spoofed_constructor.proof_certified
    assert "generated_constructor_proof_certified" in (
        spoofed_constructor.missing_obligations
    )
    assert not spoofed_partition.partition_generation_certificate.certified
    assert "scoped_set_valued_constructor_type" in (
        spoofed_partition.partition_generation_certificate.missing_obligations
    )
    assert not spoofed_partition.partition_generation_certificate.proof_certified
    assert "scoped_set_valued_constructor_proof_certified" in (
        spoofed_partition.partition_generation_certificate.missing_obligations
    )
    assert not spoofed_partition.proof_certified
    assert "generated_partition_bridge_proof_certified" in (
        spoofed_partition.missing_obligations
    )


def test_supported_event_function_generation_rejects_attribute_compatible_certificate_types():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_polynomial_arrangement_for_type_guard",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="polynomial_threshold",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )
    other_certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_polynomial_arrangement_for_stale_partition",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="polynomial_threshold",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )
    partition = certificate.partition_generation_certificate
    fake_partition = SimpleNamespace(
        certified=True,
        proof_certified=True,
        input_scope_id=partition.input_scope_id,
        event_function_grammar_id=partition.event_function_grammar_id,
        branch_function_grammar_id=partition.branch_function_grammar_id,
        event_order_function_grammar_id=partition.event_order_function_grammar_id,
        branch_constructor_source_type=partition.branch_constructor_source_type,
        event_order_constructor_source_type=(
            partition.event_order_constructor_source_type
        ),
        missing_obligations=(),
    )

    spoofed_constructor = replace(
        certificate,
        constructor_certificate=SimpleNamespace(
            source_tree=SimpleNamespace(source_type="PolynomialDecisionArrangement"),
            proof_certified=True,
            missing_obligations=(),
        ),
    )
    spoofed_partition_bridge = replace(
        certificate,
        partition_generation_certificate=fake_partition,
    )
    spoofed_grammar_input = replace(
        certificate,
        grammar_input=SimpleNamespace(
            source_type="PolynomialDecisionArrangement",
            grammar_id=(
                certificate.partition_generation_certificate
                .branch_function_grammar_id
            ),
            supported=True,
        ),
    )
    spoofed_set_valued = replace(
        certificate,
        partition_generation_certificate=replace(
            partition,
            set_valued_constructor_certificate=SimpleNamespace(
                proof_certified=True,
                missing_obligations=(),
            ),
        ),
    )
    spoofed_validated = replace(
        certificate,
        partition_generation_certificate=replace(
            partition,
            validated_set_valued_constructor_certificate=SimpleNamespace(
                proof_certified=True,
                missing_obligations=(),
            ),
        ),
    )
    stale_real_partition_branch = replace(
        certificate,
        partition_generation_certificate=replace(
            partition,
            branch_consumption_certificate=(
                other_certificate.partition_generation_certificate
                .branch_consumption_certificate
            ),
        ),
    )
    stale_real_partition_bridge = replace(
        certificate,
        partition_generation_certificate=(
            other_certificate.partition_generation_certificate
        ),
    )

    assert not certificate.proof_certified
    assert not other_certificate.proof_certified
    assert not spoofed_constructor.proof_certified
    assert "generated_constructor_certificate_type" in (
        spoofed_constructor.missing_obligations
    )
    assert not spoofed_partition_bridge.proof_certified
    assert "generated_partition_bridge_type" in (
        spoofed_partition_bridge.missing_obligations
    )
    assert not spoofed_grammar_input.proof_certified
    assert "supported_event_function_grammar_input_type" in (
        spoofed_grammar_input.missing_obligations
    )
    assert not spoofed_set_valued.proof_certified
    assert not spoofed_set_valued.partition_generation_certificate.certified
    assert "scoped_set_valued_constructor_type" in (
        spoofed_set_valued.missing_obligations
    )
    assert not spoofed_validated.proof_certified
    assert not spoofed_validated.partition_generation_certificate.certified
    assert "validated_set_valued_scope_type" in (
        spoofed_validated.missing_obligations
    )
    assert not stale_real_partition_branch.proof_certified
    assert not stale_real_partition_branch.partition_generation_certificate.certified
    assert "scoped_partition_sources_match_constructor_chain" in (
        stale_real_partition_branch
        .partition_generation_certificate
        .missing_obligations
    )
    assert "generated_partition_bridge_proof_certified" in (
        stale_real_partition_branch.missing_obligations
    )
    assert not stale_real_partition_bridge.proof_certified
    assert "generated_partition_bridge_branch_tree_matches_constructor" in (
        stale_real_partition_bridge.missing_obligations
    )
    assert "generated_partition_bridge_event_order_tree_matches_constructor" in (
        stale_real_partition_bridge.missing_obligations
    )


def test_supported_event_function_generation_rejects_spoofed_obligation_ledgers():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="generated_polynomial_arrangement_for_ledger_spoof",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="polynomial_threshold",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=3,
        branch_root_rank=2,
    )
    partition = certificate.partition_generation_certificate
    set_valued = partition.set_valued_constructor_certificate
    validated = partition.validated_set_valued_constructor_certificate
    search = set_valued.search_completeness_certificate

    spoofed_theorem = replace(
        theorem,
        obligations=(
            SimpleNamespace(
                obligation="fake_finite_target_obligation",
                certified=True,
                required=True,
            ),
        ),
    )
    spoofed_generation = replace(
        certificate,
        obligations=(
            SimpleNamespace(
                obligation="fake_supported_generation_obligation",
                certified=True,
                required=True,
            ),
        ),
    )
    optional_generation = replace(
        certificate,
        obligations=tuple(
            replace(obligation, required=False)
            for obligation in certificate.obligations
        ),
    )
    spoofed_partition = replace(
        partition,
        obligations=(
            SimpleNamespace(
                obligation="fake_scoped_partition_obligation",
                certified=True,
                required=True,
            ),
        ),
    )
    optional_partition = replace(
        partition,
        obligations=tuple(
            replace(obligation, required=False) for obligation in partition.obligations
        ),
    )
    truthy_partition_claim = replace(
        partition,
        arbitrary_partition_generation_claimed="yes",
    )
    spoofed_search = replace(
        search,
        obligations=(
            SimpleNamespace(
                obligation="fake_search_obligation",
                certified=True,
                required=True,
            ),
        ),
    )
    spoofed_set_valued = replace(
        set_valued,
        obligations=(
            SimpleNamespace(
                obligation="fake_set_valued_obligation",
                certified=True,
                required=True,
            ),
        ),
    )
    spoofed_validated = replace(
        validated,
        obligations=(
            SimpleNamespace(
                obligation="fake_validated_set_valued_obligation",
                certified=True,
                required=True,
            ),
        ),
    )

    assert not theorem.proof_certified
    assert not certificate.proof_certified
    assert not spoofed_theorem.certified
    assert "finite_target_completeness_obligation_type" in (
        spoofed_theorem.missing_obligations
    )
    assert not spoofed_generation.proof_certified
    assert "supported_event_function_generation_obligation_type" in (
        spoofed_generation.missing_obligations
    )
    assert not optional_generation.proof_certified
    assert "supported_event_function_generation_required_obligation_present" in (
        optional_generation.missing_obligations
    )
    assert not spoofed_partition.certified
    assert "scoped_arbitrary_interval_partition_obligation_type" in (
        spoofed_partition.missing_obligations
    )
    assert not optional_partition.certified
    assert "scoped_arbitrary_interval_partition_required_obligation_present" in (
        optional_partition.missing_obligations
    )
    assert not truthy_partition_claim.certified
    assert not spoofed_search.certified
    assert "finite_target_certificate_search_obligation_type" in (
        spoofed_search.missing_obligations
    )
    assert not spoofed_set_valued.certified
    assert "supplied_recursive_set_valued_constructor_obligation_type" in (
        spoofed_set_valued.missing_obligations
    )
    assert not spoofed_validated.certified
    assert "validated_set_valued_constructor_obligation_type" in (
        spoofed_validated.missing_obligations
    )


def test_supported_event_function_generation_rejects_truthy_event_order_proof_flag():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="AffineHalfspaceDecision",
            decision_id="truthy_event_branch_boundary",
            coefficients=(0.0, 1.0, 1.0),
            domain_box=((-1.0, 1.0), (-1.0, 1.0)),
            slab_half_width=0.25,
        ),
        event_order_grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialDecisionArrangement",
            arrangement_id="truthy_event_order_arrangement",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="event_minus_target",
                    coefficients=(0.0, 1.0),
                    root_brackets=((-0.01, 0.01),),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=2,
        branch_root_rank=2,
        event_order_root_dimension=3,
        event_order_root_rank=2,
    )
    spoofed_event = replace(
        certificate,
        event_order_constructor_certificate=SimpleNamespace(
            source_tree=SimpleNamespace(source_type="PolynomialDecisionArrangement"),
            proof_certified="yes",
            missing_obligations=(),
        ),
    )
    spoofed_event_type = replace(
        certificate,
        event_order_constructor_certificate=SimpleNamespace(
            source_tree=SimpleNamespace(source_type="PolynomialDecisionArrangement"),
            proof_certified=True,
            missing_obligations=(),
        ),
    )

    assert not certificate.proof_certified
    assert not spoofed_event.proof_certified
    assert "generated_event_order_constructor_proof_certified" in (
        spoofed_event.missing_obligations
    )
    assert not spoofed_event_type.proof_certified
    assert "generated_event_order_constructor_certificate_type" in (
        spoofed_event_type.missing_obligations
    )


def test_supported_event_function_generation_blocks_event_source_mismatch():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    certificate = certify_supported_event_function_stratification_generation(
        theorem,
        grammar_input=SupportedEventFunctionGrammarInput(
            source_type="AffineHalfspaceDecision",
            decision_id="mismatch_event_branch_boundary",
            coefficients=(0.0, 1.0, 1.0),
            domain_box=((-1.0, 1.0), (-1.0, 1.0)),
            slab_half_width=0.25,
        ),
        event_order_grammar_input=SupportedEventFunctionGrammarInput(
            source_type="PolynomialRootArrangement",
            arrangement_id="mismatch_event_quadratic_double_root",
            polynomial_decision_functions=(
                PolynomialDecisionFunctionSpec(
                    decision_id="event_quadratic_double_threshold",
                    coefficients=(0.0, 0.0, 1.0),
                ),
            ),
            domain=(-1.0, 1.0),
        ),
        branch_root_dimension=2,
        branch_root_rank=2,
        event_order_root_dimension=3,
        event_order_root_rank=2,
    )

    assert not certificate.proof_certified
    assert certificate.constructor_source_type == "AffineHalfspaceDecision"
    assert (
        certificate.event_order_constructor_source_type
        == "QuadraticDoubleRootArrangement"
    )
    assert (
        "generated_event_order_constructor_source_matches_requested_grammar"
        in certificate.missing_obligations
    )


def test_supported_event_function_generation_rejects_unsupported_grammar():
    theorem = certify_finite_target_completeness_theorem(dimension=3)

    try:
        certify_supported_event_function_stratification_generation(
            theorem,
            grammar_input=SupportedEventFunctionGrammarInput(
                source_type="arbitrary_analytic_black_box",
                decision_id="unsupported",
                domain=(-1.0, 1.0),
            ),
            branch_root_dimension=3,
            branch_root_rank=1,
        )
    except ValueError as error:
        assert "unsupported event-function grammar source" in str(error)
    else:
        raise AssertionError("unsupported grammar generated a partition")


def test_supported_event_function_generation_rejects_raw_event_order_grammar_input():
    theorem = certify_finite_target_completeness_theorem(dimension=3)

    try:
        certify_supported_event_function_stratification_generation(
            theorem,
            grammar_input=SupportedEventFunctionGrammarInput(
                source_type="PolynomialDecisionStratification",
                decision_id="valid_branch_event",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
                domain=(-1.0, 1.0),
            ),
            event_order_grammar_input=True,
            branch_root_dimension=3,
            branch_root_rank=1,
            event_order_root_dimension=3,
            event_order_root_rank=1,
        )
    except TypeError as error:
        assert (
            "event_order_grammar_input must be a SupportedEventFunctionGrammarInput"
            in str(error)
        )
    else:
        raise AssertionError("raw event-order grammar input was accepted")


def test_constructor_derived_set_valued_helper_rejects_unsupported_constructor():
    theorem = certify_finite_target_completeness_theorem(dimension=3)

    try:
        certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness(
            theorem,
            constructor_certificate=object(),
            root_dimension=3,
            root_rank=1,
        )
    except TypeError as error:
        assert "supported displayed stratification" in str(error)
    else:
        raise AssertionError("unsupported displayed constructor was accepted")


def test_mixed_constructor_set_valued_scope_preserves_branch_and_event_sources():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    branch_stratification = certify_affine_halfspace_decision_stratified_branch_event_tree(
        decision_id="branch_oblique_boundary",
        coefficients=(0.0, 1.0, 1.0),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )
    event_arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="event_polynomial_arrangement",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_target",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    scoped = (
        certify_constructor_pair_derived_recursive_stratified_set_valued_constructor_completeness(
            theorem,
            branch_constructor_certificate=branch_stratification,
            event_order_constructor_certificate=event_arrangement,
            branch_root_dimension=2,
            branch_root_rank=2,
            event_order_root_dimension=3,
            event_order_root_rank=2,
        )
    )
    branch_recursive = scoped.branch_consumption_certificate
    event_recursive = scoped.event_order_consumption_certificate
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert branch_recursive.certified
    assert event_recursive.certified
    assert branch_recursive.child_constructor_source_types == (
        "AffineHalfspaceDecisionChild",
    )
    assert event_recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert recursive_constructor_source_scope(branch_recursive)[1] == (
        "finite_affine_halfspace_decision_interval_boxes"
    )
    assert recursive_constructor_source_scope(event_recursive)[1] == (
        "finite_polynomial_decision_arrangement_interval_boxes"
    )
    assert not scoped.proof_certified
    assert not validated.proof_certified
    assert validated.input_scope_id == (
        "finite_mixed_constructor_branch_event_interval_boxes"
    )
    assert "branch source AffineHalfspaceDecision" in validated.proof_sketch
    assert "event-order source PolynomialDecisionArrangement" in (
        validated.proof_sketch
    )
    assert not validated.arbitrary_partition_generation_claimed


def test_constructor_pair_derived_set_valued_helper_rejects_bad_event_constructor():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    branch_stratification = certify_affine_halfspace_decision_stratified_branch_event_tree(
        decision_id="bad_event_helper_branch_boundary",
        coefficients=(0.0, 1.0, 1.0),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )

    try:
        certify_constructor_pair_derived_recursive_stratified_set_valued_constructor_completeness(
            theorem,
            branch_constructor_certificate=branch_stratification,
            event_order_constructor_certificate=object(),
            branch_root_dimension=2,
            branch_root_rank=2,
            event_order_root_dimension=3,
            event_order_root_rank=2,
        )
    except TypeError as error:
        assert "supported displayed stratification" in str(error)
    else:
        raise AssertionError("unsupported event-order constructor was accepted")


def test_mixed_oblique_affine_arrangement_scope_preserves_volume_source():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    branch_arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="mixed_spatial_oblique_branch",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="sum_boundary",
                coefficients=(0.0, 1.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="mixed_boundary",
                coefficients=(0.0, 1.0, -1.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    event_arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="mixed_affine_event",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_target",
                coefficients=(0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    scoped = (
        certify_constructor_pair_derived_recursive_stratified_set_valued_constructor_completeness(
            theorem,
            branch_constructor_certificate=branch_arrangement,
            event_order_constructor_certificate=event_arrangement,
            branch_root_dimension=3,
            branch_root_rank=2,
            event_order_root_dimension=3,
            event_order_root_rank=2,
        )
    )
    branch_recursive = scoped.branch_consumption_certificate
    event_recursive = scoped.event_order_consumption_certificate
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert branch_arrangement.source_tree.source_type == "AffineHalfspace3DArrangement"
    assert branch_arrangement.volume_cover_certified
    assert event_arrangement.source_tree.source_type == "AffineDecisionArrangement"
    assert "AffineHalfspacePlaneChild" in (
        branch_recursive.child_constructor_source_types
    )
    assert event_recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert branch_recursive.certified
    assert event_recursive.certified
    assert not scoped.proof_certified
    assert not validated.proof_certified
    assert validated.input_scope_id == (
        "finite_mixed_constructor_branch_event_interval_boxes"
    )
    assert "branch source AffineHalfspace3DArrangement" in validated.proof_sketch
    assert "finite_3d_affine_halfspace_arrangement_interval_boxes" in (
        validated.proof_sketch
    )
    assert "event-order source AffineDecisionArrangement" in (
        validated.proof_sketch
    )
    assert not validated.arbitrary_partition_generation_claimed


def test_scoped_arbitrary_interval_partition_generation_preserves_mixed_grammar_pair():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    branch_stratification = certify_affine_halfspace_decision_stratified_branch_event_tree(
        decision_id="scoped_mixed_branch_oblique_boundary",
        coefficients=(0.0, 1.0, 1.0),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )
    event_arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="scoped_mixed_event_polynomial_arrangement",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="event_minus_target",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=branch_stratification,
        event_order_constructor_certificate=event_arrangement,
        branch_root_dimension=2,
        branch_root_rank=2,
        event_order_root_dimension=3,
        event_order_root_rank=2,
    )
    support_obligation = next(
        obligation
        for obligation in certificate.obligations
        if obligation.obligation == "supported_event_function_grammar"
    )

    assert not certificate.proof_certified
    assert not certificate.arbitrary_partition_generation_claimed
    assert certificate.branch_constructor_source_type == "AffineHalfspaceDecision"
    assert (
        certificate.event_order_constructor_source_type
        == "PolynomialDecisionArrangement"
    )
    assert certificate.branch_constructor_input_scope_id == (
        "finite_affine_halfspace_decision_interval_boxes"
    )
    assert certificate.event_order_constructor_input_scope_id == (
        "finite_polynomial_decision_arrangement_interval_boxes"
    )
    assert certificate.branch_function_grammar_id == (
        "finite_affine_halfspace_decision_interval_inputs"
    )
    assert certificate.event_order_function_grammar_id == (
        "finite_polynomial_decision_arrangement_interval_inputs"
    )
    assert certificate.event_function_grammar_id == (
        "finite_mixed_supported_constructor_interval_inputs"
        "(branch=finite_affine_halfspace_decision_interval_inputs;"
        "event=finite_polynomial_decision_arrangement_interval_inputs)"
    )
    assert "branch_grammar=finite_affine_halfspace_decision_interval_inputs" in (
        support_obligation.detail
    )
    assert "event_grammar=finite_polynomial_decision_arrangement_interval_inputs" in (
        support_obligation.detail
    )
    assert "unqualified_arbitrary_analytic_inputs_not_claimed" not in (
        certificate.missing_obligations
    )
    assert "scoped_set_valued_constructor_proof_certified" in (
        certificate.missing_obligations
    )
    assert "validated_set_valued_scope_proof_certified" in (
        certificate.missing_obligations
    )


def test_scoped_arbitrary_interval_partition_generation_surfaces_event_unsupported_strata():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    branch_stratification = certify_affine_halfspace_decision_stratified_branch_event_tree(
        decision_id="scoped_supported_branch_for_unsupported_event",
        coefficients=(0.0, 1.0, 1.0),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )
    event_stratification = certify_taylor_model_decision_stratified_branch_event_tree(
        decision_function=TaylorModelDecisionFunctionSpec(
            decision_id="unsupported_event_order_taylor_root",
            coefficients=(-0.25, 1.0),
            expansion_center=0.0,
            remainder_bound=1.0e-4,
            derivative_remainder_bound=5.0e-2,
            root_brackets=((0.24, 0.26),),
            witness_kind="unsupported",
        ),
        domain=(-1.0, 1.0),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=branch_stratification,
        event_order_constructor_certificate=event_stratification,
        branch_root_dimension=2,
        branch_root_rank=2,
        event_order_root_dimension=3,
        event_order_root_rank=1,
    )

    assert not event_stratification.proof_certified
    assert not certificate.proof_certified
    assert not certificate.arbitrary_partition_generation_claimed
    assert certificate.branch_function_grammar_id == (
        "finite_affine_halfspace_decision_interval_inputs"
    )
    assert certificate.event_order_function_grammar_id == (
        "finite_taylor_model_decision_interval_inputs_with_weierstrass_certificate"
    )
    assert certificate.event_function_grammar_id == (
        "finite_mixed_supported_constructor_interval_inputs"
        "(branch=finite_affine_halfspace_decision_interval_inputs;"
        "event=finite_taylor_model_decision_interval_inputs_with_weierstrass_certificate)"
    )
    assert certificate.unsupported_strata
    assert "no_unsupported_analytic_strata" in certificate.missing_obligations
    assert any(
        item.startswith("unsupported:")
        for item in certificate.missing_obligations
    )


def test_polynomial_decision_arrangement_derives_sign_vector_and_equality_leaves():
    arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="event_order_arrangement",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_total",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="ks_exit_minus_target",
                coefficients=(-0.5, 1.0),
                root_brackets=((0.49, 0.51),),
            ),
        ),
        domain=(-1.0, 1.0),
    )

    assert arrangement.certified
    assert arrangement.sign_stratum_count == 3
    assert arrangement.equality_stratum_count == 2
    assert arrangement.source_tree.cover_certified
    assert not arrangement.source_tree.certified
    assert arrangement.stratified_tree.certified
    assert arrangement.stratified_tree.zero_margin_leaf_count == 2
    assert arrangement.stratified_tree.leaf_kinds == (
        "positive_margin_unique_event",
        "simultaneous_event_equality",
        "positive_margin_unique_event",
        "simultaneous_event_equality",
        "positive_margin_unique_event",
    )


def test_supported_event_function_generation_rejects_nested_stratification_spoofs():
    arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="nested_type_guard_arrangement",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="event_minus_target",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    fake_stratum = replace(
        arrangement,
        strata=(
            SimpleNamespace(
                stratum_id="fake_polynomial_stratum",
                proof_certified=True,
                missing_obligations=(),
                equality=False,
            ),
        ),
    )
    truthy_stratum = replace(
        arrangement,
        strata=(
            replace(arrangement.strata[0], certified="yes"),
            *arrangement.strata[1:],
        ),
    )
    fake_source = replace(
        arrangement,
        source_tree=SimpleNamespace(cover_certified=True),
    )
    fake_stratified_tree = replace(
        arrangement,
        stratified_tree=SimpleNamespace(
            certified=True,
            missing_obligations=(),
            unsupported_leaf_count=0,
        ),
    )
    halfspace = certify_affine_halfspace_decision_stratified_branch_event_tree(
        decision_id="nested_type_guard_halfspace",
        coefficients=(0.0, 1.0, 1.0),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )
    fake_cell = replace(
        halfspace,
        cells=(
            SimpleNamespace(
                cell_id="fake_halfspace_cell",
                proof_certified=True,
                missing_obligations=(),
                equality=False,
            ),
        ),
    )

    assert arrangement.certified
    assert not fake_stratum.certified
    assert "polynomial_arrangement_stratum:fake_polynomial_stratum:type" in (
        fake_stratum.missing_obligations
    )
    assert not truthy_stratum.certified
    assert "polynomial_arrangement_stratum:" in " ".join(
        truthy_stratum.missing_obligations
    )
    assert not fake_source.certified
    assert "source_tree_type" in fake_source.missing_obligations
    assert not fake_stratified_tree.certified
    assert "stratified_tree_type" in fake_stratified_tree.missing_obligations
    assert halfspace.certified
    assert not fake_cell.certified
    assert "affine_halfspace_decision_cell:fake_halfspace_cell:type" in (
        fake_cell.missing_obligations
    )


def test_affine_decision_arrangement_derives_root_brackets_from_coefficients():
    arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="affine_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_total",
                coefficients=(0.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="ks_exit_minus_target",
                coefficients=(-0.5, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("affine-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            "affine_event_order:root:0:binary_minus_total": child,
            "affine_event_order:root:1:ks_exit_minus_target": child,
        },
    )

    assert arrangement.certified
    assert arrangement.source_tree.source_type == "AffineDecisionArrangement"
    assert arrangement.equality_stratum_count == 2
    assert arrangement.sign_stratum_count == 3
    assert all(spec.root_brackets for spec in arrangement.decision_functions)
    assert "without supplying root brackets" in arrangement.statement
    assert "one-quarter-gap brackets" in arrangement.proof_sketch
    assert parent.certified
    assert parent.recursion_kind == "affine_decision_arrangement"
    assert parent.recursive_leaf_count == 2
    assert parent.terminal_leaf_count == 3


def test_affine_decision_arrangement_groups_coincident_roots_as_simultaneous_stratum():
    arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="coincident_affine_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_total",
                coefficients=(0.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="ks_exit_minus_target",
                coefficients=(0.0, 2.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("coincident-affine-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            "coincident_affine_event_order:root:0:binary_minus_total+ks_exit_minus_target": (
                child
            ),
        },
    )

    assert arrangement.certified
    assert arrangement.source_tree.source_type == "AffineDecisionArrangement"
    assert arrangement.equality_stratum_count == 1
    assert arrangement.sign_stratum_count == 2
    equality_leaf = arrangement.stratified_tree.leaf_certificates[1]
    assert equality_leaf.leaf_kind == "simultaneous_event_equality"
    assert equality_leaf.equality_stratum.defining_function_ids == (
        "binary_minus_total",
        "ks_exit_minus_target",
    )
    assert "coincident roots are grouped" in arrangement.statement
    assert "higher-codimension equality stratum" in arrangement.proof_sketch
    assert parent.certified
    assert parent.recursion_kind == "affine_decision_arrangement"
    assert parent.recursive_leaf_count == 1
    assert parent.terminal_leaf_count == 2


def test_affine_decision_arrangement_derives_one_sided_boundary_root_bracket():
    arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="boundary_affine_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="target_starts_on_event_boundary",
                coefficients=(1.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="interior_binary_minus_target",
                coefficients=(-0.5, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("boundary-affine-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            "boundary_affine_event_order:root:0:target_starts_on_event_boundary": (
                child
            ),
            "boundary_affine_event_order:root:1:interior_binary_minus_target": (
                child
            ),
        },
    )

    assert arrangement.certified
    assert arrangement.source_tree.source_type == "AffineDecisionArrangement"
    assert arrangement.equality_stratum_count == 2
    assert arrangement.sign_stratum_count == 2
    boundary_bracket = arrangement.decision_functions[0].root_brackets[0]
    assert boundary_bracket[0] == -1.0
    assert boundary_bracket[1] > boundary_bracket[0]
    assert "one-sided brackets" in arrangement.statement
    assert parent.certified
    assert parent.recursion_kind == "affine_decision_arrangement"
    assert parent.recursive_leaf_count == 2
    assert parent.terminal_leaf_count == 2


def test_affine_decision_set_valued_constructor_scope_is_constructor_derived():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="validated_affine_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="target_starts_on_event_boundary",
                coefficients=(1.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="interior_binary_minus_target",
                coefficients=(-0.5, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    scoped = (
        certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness(
            theorem,
            constructor_certificate=arrangement,
            root_dimension=3,
            root_rank=2,
        )
    )
    recursive = scoped.branch_consumption_certificate
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert scoped.event_order_consumption_certificate is recursive
    assert arrangement.source_tree.source_type == "AffineDecisionArrangement"
    assert recursive.recursion_kind == "affine_decision_arrangement"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not scoped.proof_certified
    assert "affine decision arrangement" in scoped.statement
    assert "coefficient-derived roots" in scoped.proof_sketch
    assert not validated.proof_certified
    assert (
        validated.input_scope_id
        == "finite_affine_decision_arrangement_interval_boxes"
    )
    assert "one-sided boundary brackets" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


def test_quadratic_decision_arrangement_derives_simple_root_brackets_from_coefficients():
    arrangement = certify_quadratic_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="quadratic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="quadratic_binary_threshold",
                coefficients=(-0.25, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("quadratic-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            "quadratic_event_order:root:0:quadratic_binary_threshold": child,
            "quadratic_event_order:root:1:quadratic_binary_threshold": child,
        },
    )

    assert arrangement.certified
    assert arrangement.equality_stratum_count == 2
    assert arrangement.sign_stratum_count == 3
    brackets = arrangement.decision_functions[0].root_brackets
    assert len(brackets) == 2
    assert brackets[0][0] < -0.5 < brackets[0][1]
    assert brackets[1][0] < 0.5 < brackets[1][1]
    assert "affine/quadratic decision" in arrangement.statement
    assert "quadratic formula" in arrangement.proof_sketch
    assert parent.certified
    assert parent.recursive_leaf_count == 2
    assert parent.terminal_leaf_count == 3


def test_quadratic_decision_arrangement_derives_double_root_tangent_stratum():
    arrangement = certify_quadratic_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="double_quadratic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="double_root_threshold",
                coefficients=(0.0, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("double-quadratic-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            "double_quadratic_event_order:root:0:double_root_threshold": child,
        },
    )

    assert arrangement.certified
    assert arrangement.equality_stratum_count == 1
    assert arrangement.sign_stratum_count == 2
    tangent = arrangement.strata[1]
    assert tangent.stratum_kind == "quadratic_double_equality_root"
    assert tangent.second_derivative_interval[0] > 0.0
    assert "double root" in arrangement.statement
    assert "multiplicity-two stratum" in arrangement.proof_sketch
    assert parent.certified
    assert parent.recursive_leaf_count == 1
    assert parent.terminal_leaf_count == 2


def test_quadratic_decision_arrangement_groups_coincident_simple_roots():
    arrangement = certify_quadratic_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="coincident_quadratic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="left_quadratic_threshold",
                coefficients=(-0.25, 0.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="shifted_quadratic_threshold",
                coefficients=(-0.375, 0.25, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("coincident-quadratic-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            "coincident_quadratic_event_order:root:0:shifted_quadratic_threshold": child,
            "coincident_quadratic_event_order:root:1:left_quadratic_threshold": child,
            "coincident_quadratic_event_order:root:2:left_quadratic_threshold+shifted_quadratic_threshold": (
                child
            ),
        },
    )

    assert arrangement.certified
    assert arrangement.equality_stratum_count == 3
    assert arrangement.sign_stratum_count == 4
    shared_leaf = arrangement.stratified_tree.leaf_certificates[5]
    assert shared_leaf.leaf_kind == "simultaneous_event_equality"
    assert shared_leaf.equality_stratum.defining_function_ids == (
        "left_quadratic_threshold",
        "shifted_quadratic_threshold",
    )
    assert "coincident simple roots" in arrangement.statement
    assert "higher-codimension equality stratum" in arrangement.proof_sketch
    assert parent.certified
    assert parent.recursive_leaf_count == 3
    assert parent.terminal_leaf_count == 4


def test_quadratic_decision_arrangement_groups_mixed_simple_double_root():
    arrangement = certify_quadratic_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="mixed_double_quadratic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="left_quadratic_threshold",
                coefficients=(-0.25, 0.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="double_quadratic_threshold",
                coefficients=(0.25, -1.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("mixed-double-quadratic-child"),
        root_dimension=2,
        root_rank=1,
    )
    equality_leaves = tuple(
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            leaf.equality_stratum.stratum_id: child
            for leaf in equality_leaves
        },
    )
    multiple = next(
        stratum
        for stratum in arrangement.strata
        if stratum.stratum_kind == "simultaneous_polynomial_multiple_equality_root"
    )
    multiple_leaf = next(
        leaf
        for leaf in equality_leaves
        if leaf.equality_stratum.stratum_id == multiple.stratum_id
    )

    assert arrangement.certified
    assert arrangement.equality_stratum_count == 2
    assert arrangement.sign_stratum_count == 3
    assert multiple.second_derivative_interval[0] > 0.0
    assert multiple_leaf.equality_stratum.defining_function_ids == (
        "double_quadratic_threshold",
        "left_quadratic_threshold",
    )
    assert "mixed simple/double coincidences" in arrangement.statement
    assert "multiple equality stratum" in arrangement.proof_sketch
    assert parent.certified
    assert parent.recursive_leaf_count == 2
    assert parent.terminal_leaf_count == 3


def test_quadratic_decision_arrangement_groups_coincident_double_roots():
    arrangement = certify_quadratic_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="coincident_double_quadratic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="double_quadratic_a",
                coefficients=(0.25, -1.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="double_quadratic_b",
                coefficients=(1.0, -4.0, 4.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("coincident-double-quadratic-child"),
        root_dimension=2,
        root_rank=1,
    )
    equality_leaf = next(
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={equality_leaf.equality_stratum.stratum_id: child},
    )
    multiple = arrangement.strata[1]

    assert arrangement.certified
    assert arrangement.equality_stratum_count == 1
    assert arrangement.sign_stratum_count == 2
    assert multiple.stratum_kind == "simultaneous_polynomial_multiple_equality_root"
    assert multiple.second_derivative_interval[0] > 0.0
    assert equality_leaf.equality_stratum.defining_function_ids == (
        "double_quadratic_a",
        "double_quadratic_b",
    )
    assert parent.certified
    assert parent.recursive_leaf_count == 1
    assert parent.terminal_leaf_count == 2


def test_sturm_polynomial_decision_arrangement_derives_cubic_roots():
    arrangement = certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="sturm_cubic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="cubic_threshold",
                coefficients=(0.0, -0.25, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("sturm-cubic-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    equality_leaves = tuple(
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            leaf.equality_stratum.stratum_id: child
            for leaf in equality_leaves
        },
    )
    brackets = arrangement.decision_functions[0].root_brackets

    assert arrangement.certified
    assert arrangement.equality_stratum_count == 3
    assert arrangement.sign_stratum_count == 4
    assert brackets[0][0] < -0.5 < brackets[0][1]
    assert brackets[1][0] < 0.0 < brackets[1][1]
    assert brackets[2][0] < 0.5 < brackets[2][1]
    assert "Sturm sequences" in arrangement.statement
    assert "Sturm variation" in arrangement.proof_sketch
    assert parent.certified
    assert parent.recursive_leaf_count == 3
    assert parent.terminal_leaf_count == 4


def test_sturm_polynomial_decision_arrangement_derives_multiple_root_stratum():
    arrangement = certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="sturm_multiple_root_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="multiple_threshold",
                coefficients=(0.0, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("sturm-multiple-root-child"),
        root_dimension=2,
        root_rank=1,
    )
    equality_leaves = tuple(
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            leaf.equality_stratum.stratum_id: child
            for leaf in equality_leaves
        },
    )
    multiple_stratum = next(stratum for stratum in arrangement.strata if stratum.equality)

    assert arrangement.certified
    assert arrangement.equality_stratum_count == 1
    assert arrangement.sign_stratum_count == 2
    assert multiple_stratum.stratum_kind == "sturm_polynomial_multiple_equality_root"
    assert multiple_stratum.root_multiplicities == (2,)
    assert multiple_stratum.derivative_interval[0] < 0.0 < multiple_stratum.derivative_interval[1]
    assert multiple_stratum.second_derivative_interval[0] > 0.0
    assert parent.certified
    assert parent.recursive_leaf_count == 1
    assert parent.terminal_leaf_count == 2


def test_sturm_polynomial_decision_arrangement_certifies_quartic_sign_cells():
    arrangement = certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="sturm_quartic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="quartic_threshold",
                coefficients=(0.03515625, 0.0, -0.625, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("sturm-quartic-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    equality_leaves = tuple(
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            leaf.equality_stratum.stratum_id: child
            for leaf in equality_leaves
        },
    )
    brackets = arrangement.decision_functions[0].root_brackets

    assert arrangement.certified
    assert arrangement.equality_stratum_count == 4
    assert arrangement.sign_stratum_count == 5
    assert brackets[0][0] < -0.75 < brackets[0][1]
    assert brackets[1][0] < -0.25 < brackets[1][1]
    assert brackets[2][0] < 0.25 < brackets[2][1]
    assert brackets[3][0] < 0.75 < brackets[3][1]
    assert all(
        leaf.terminal_response_certified
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.leaf_kind == "positive_margin_unique_event"
    )
    assert parent.certified
    assert parent.recursive_leaf_count == 4
    assert parent.terminal_leaf_count == 5


def test_sturm_polynomial_decision_arrangement_derives_boundary_root_brackets():
    arrangement = certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="sturm_boundary_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="boundary_threshold",
                coefficients=(-0.25, 0.75, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("sturm-boundary-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    equality_leaves = tuple(
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            leaf.equality_stratum.stratum_id: child
            for leaf in equality_leaves
        },
    )
    brackets = arrangement.decision_functions[0].root_brackets

    assert arrangement.certified
    assert arrangement.equality_stratum_count == 2
    assert arrangement.sign_stratum_count == 2
    assert brackets[0][0] == -1.0
    assert brackets[0][0] < -1.0 + 1.0e-12 < brackets[0][1]
    assert brackets[1][0] < 0.25 < brackets[1][1]
    assert parent.certified
    assert parent.recursive_leaf_count == 2
    assert parent.terminal_leaf_count == 2


def test_sturm_polynomial_decision_arrangement_groups_coincident_roots():
    arrangement = certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="sturm_coincident_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="cubic_threshold",
                coefficients=(0.0, -0.25, 0.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="quadratic_common_threshold",
                coefficients=(0.0, -0.5, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("sturm-coincident-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    equality_leaves = tuple(
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            leaf.equality_stratum.stratum_id: child
            for leaf in equality_leaves
        },
    )
    simultaneous_leaves = tuple(
        leaf
        for leaf in equality_leaves
        if len(leaf.equality_stratum.defining_function_ids) == 2
    )

    assert arrangement.certified
    assert arrangement.equality_stratum_count == 3
    assert arrangement.sign_stratum_count == 4
    assert len(simultaneous_leaves) == 2
    assert all(
        leaf.equality_stratum.defining_function_ids
        == ("cubic_threshold", "quadratic_common_threshold")
        for leaf in simultaneous_leaves
    )
    assert all(
        stratum.stratum_kind == "simultaneous_polynomial_equality_root"
        for stratum in arrangement.strata
        if stratum.stratum_id
        in {leaf.equality_stratum.stratum_id for leaf in simultaneous_leaves}
    )
    assert parent.certified
    assert parent.recursive_leaf_count == 3
    assert parent.terminal_leaf_count == 4


def test_sturm_polynomial_decision_arrangement_groups_mixed_multiplicity_root():
    arrangement = certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="sturm_mixed_multiplicity_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="simple_threshold",
                coefficients=(0.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="double_threshold",
                coefficients=(0.0, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("sturm-mixed-multiple-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    equality_leaves = tuple(
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            leaf.equality_stratum.stratum_id: child
            for leaf in equality_leaves
        },
    )
    multiple_stratum = next(stratum for stratum in arrangement.strata if stratum.equality)

    assert arrangement.certified
    assert arrangement.equality_stratum_count == 1
    assert arrangement.sign_stratum_count == 2
    assert multiple_stratum.stratum_kind == "simultaneous_polynomial_multiple_equality_root"
    assert multiple_stratum.root_multiplicities == (2, 1)
    assert set(equality_leaves[0].equality_stratum.defining_function_ids) == {
        "double_threshold",
        "simple_threshold",
    }
    assert parent.certified
    assert parent.recursive_leaf_count == 1
    assert parent.terminal_leaf_count == 2


def test_polynomial_decision_arrangement_recursive_consumes_equality_children():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="event_order_arrangement",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_total",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="ks_exit_minus_target",
                coefficients=(-0.5, 1.0),
                root_brackets=((0.49, 0.51),),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    binary_total_child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("arrangement-child:binary-total"),
        root_dimension=2,
        root_rank=1,
    )
    ks_target_child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("arrangement-child:ks-target"),
        root_dimension=2,
        root_rank=1,
    )
    parent = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            "event_order_arrangement:root:0:binary_minus_total": binary_total_child,
            "event_order_arrangement:root:1:ks_exit_minus_target:leaf": ks_target_child,
        },
    )
    partial = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            "stratified:event_order_arrangement:root:0:binary_minus_total:leaf": (
                binary_total_child
            ),
        },
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=parent,
        recursive_stratified_event_order_consumption_certificate=parent,
    )

    assert arrangement.certified
    assert binary_total_child.certified
    assert ks_target_child.certified
    assert parent.certified
    assert parent.proof_certified
    assert parent.recursion_kind == "polynomial_decision_arrangement"
    assert parent.terminal_leaf_count == 3
    assert parent.recursive_leaf_count == 2
    assert parent.node_count == 3
    assert parent.missing_obligations == ()
    assert not partial.certified
    assert (
        "stratified:event_order_arrangement:root:1:ks_exit_minus_target:leaf:"
        "terminal_or_recursive_response_missing"
        in partial.missing_obligations
    )
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_verified_polynomial_arrangement_set_valued_scope_is_constructor_derived():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="validated_polynomial_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_total",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="ks_exit_minus_target",
                coefficients=(-0.5, 1.0),
                root_brackets=((0.49, 0.51),),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    equality_leaves = tuple(
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    )
    child_consumptions = derive_polynomial_decision_arrangement_child_consumptions(
        arrangement,
    )
    scoped = (
        certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness(
            theorem,
            constructor_certificate=arrangement,
            root_dimension=3,
            root_rank=2,
        )
    )
    recursive = scoped.branch_consumption_certificate
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert arrangement.proof_certified
    assert len(child_consumptions) == len(equality_leaves)
    assert scoped.event_order_consumption_certificate is recursive
    assert recursive.proof_certified
    assert recursive.recursion_kind == "polynomial_decision_arrangement"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not scoped.proof_certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        scoped.missing_obligations
    )
    assert "constructor-derived polynomial decision arrangement" in scoped.statement
    assert "verified simple root brackets" in scoped.proof_sketch
    assert not validated.proof_certified
    assert "scoped_set_valued_constructor_certificate_proof_certified" in (
        validated.missing_obligations
    )
    assert (
        validated.input_scope_id
        == "finite_polynomial_decision_arrangement_interval_boxes"
    )
    assert "verified simple root brackets" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


def test_polynomial_arrangement_child_consumption_rejects_swapped_leaf_binding():
    arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="swapped_child_polynomial_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="left_event",
                coefficients=(0.5, 1.0),
                root_brackets=((-0.51, -0.49),),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="right_event",
                coefficients=(-0.5, 1.0),
                root_brackets=((0.49, 0.51),),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    children = derive_polynomial_decision_arrangement_child_consumptions(
        arrangement,
    )
    equality_strata = tuple(stratum for stratum in arrangement.strata if stratum.equality)
    swapped_children = {
        equality_strata[0].stratum_id: children[equality_strata[1].stratum_id],
        equality_strata[1].stratum_id: children[equality_strata[0].stratum_id],
    }
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=swapped_children,
    )

    assert arrangement.proof_certified
    assert len(equality_strata) == 2
    assert not recursive.proof_certified
    assert any(
        item.endswith(":child_consumption_source_leaf_binding")
        for item in recursive.missing_obligations
    )


def test_quadratic_double_root_set_valued_scope_is_constructor_derived():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_quadratic_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="validated_double_quadratic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="double_root_threshold",
                coefficients=(0.0, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
    )
    scoped = certify_supplied_recursive_stratified_set_valued_constructor_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert arrangement.source_tree.source_type == "QuadraticDoubleRootArrangement"
    assert recursive.proof_certified
    assert recursive.recursion_kind == "quadratic_double_root_decision_arrangement"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not scoped.proof_certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        scoped.missing_obligations
    )
    assert "quadratic double-root decision arrangement" in scoped.statement
    assert "tangent equality strata" in scoped.proof_sketch
    assert not validated.proof_certified
    assert "scoped_set_valued_constructor_certificate_proof_certified" in (
        validated.missing_obligations
    )
    assert (
        validated.input_scope_id
        == "finite_quadratic_double_root_decision_arrangement_interval_boxes"
    )
    assert "tangent equality strata" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


def test_scoped_arbitrary_interval_partition_generation_accepts_quadratic_double_root_arrangement():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_quadratic_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="scoped_double_quadratic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="double_root_threshold",
                coefficients=(0.0, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=arrangement,
        branch_root_dimension=3,
        branch_root_rank=2,
    )

    assert arrangement.source_tree.source_type == "QuadraticDoubleRootArrangement"
    assert not certificate.proof_certified
    assert not certificate.arbitrary_partition_generation_claimed
    assert certificate.branch_constructor_source_type == "QuadraticDoubleRootArrangement"
    assert (
        certificate.event_order_constructor_source_type
        == "QuadraticDoubleRootArrangement"
    )
    assert certificate.event_function_grammar_id == (
        "finite_quadratic_double_root_arrangement_interval_inputs"
    )
    assert certificate.input_scope_id == (
        "finite_quadratic_double_root_decision_arrangement_interval_boxes"
    )
    assert certificate.unsupported_strata == ()
    assert "scoped_set_valued_constructor_proof_certified" in (
        certificate.missing_obligations
    )
    assert "validated_set_valued_scope_proof_certified" in (
        certificate.missing_obligations
    )


def test_computed_polynomial_root_set_valued_scope_is_constructor_derived():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_quadratic_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="validated_coincident_quadratic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="left_quadratic_threshold",
                coefficients=(-0.25, 0.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="shifted_quadratic_threshold",
                coefficients=(-0.375, 0.25, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_polynomial_decision_arrangement_child_consumptions(
            arrangement,
        ),
    )
    scoped = certify_supplied_recursive_stratified_set_valued_constructor_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert arrangement.source_tree.source_type == "PolynomialRootArrangement"
    assert recursive.proof_certified
    assert recursive.recursion_kind == "computed_polynomial_root_decision_arrangement"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not scoped.proof_certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        scoped.missing_obligations
    )
    assert "computed polynomial-root arrangement" in scoped.statement
    assert "grouped coincident/multiple equality strata" in scoped.proof_sketch
    assert not validated.proof_certified
    assert "scoped_set_valued_constructor_certificate_proof_certified" in (
        validated.missing_obligations
    )
    assert (
        validated.input_scope_id
        == "finite_computed_polynomial_root_arrangement_interval_boxes"
    )
    assert "grouped coincident/multiple equality strata" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


def test_scoped_arbitrary_interval_partition_generation_accepts_computed_polynomial_root_arrangement():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_quadratic_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="scoped_coincident_quadratic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="left_quadratic_threshold",
                coefficients=(-0.25, 0.0, 1.0),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="shifted_quadratic_threshold",
                coefficients=(-0.375, 0.25, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    certificate = certify_arbitrary_interval_input_partition_generation(
        theorem,
        branch_constructor_certificate=arrangement,
        branch_root_dimension=3,
        branch_root_rank=2,
    )

    assert arrangement.source_tree.source_type == "PolynomialRootArrangement"
    assert not certificate.proof_certified
    assert not certificate.arbitrary_partition_generation_claimed
    assert certificate.branch_constructor_source_type == "PolynomialRootArrangement"
    assert (
        certificate.event_order_constructor_source_type
        == "PolynomialRootArrangement"
    )
    assert certificate.event_function_grammar_id == (
        "finite_computed_polynomial_root_arrangement_interval_inputs"
    )
    assert certificate.input_scope_id == (
        "finite_computed_polynomial_root_arrangement_interval_boxes"
    )
    assert certificate.unsupported_strata == ()
    assert "scoped_set_valued_constructor_proof_certified" in (
        certificate.missing_obligations
    )
    assert "validated_set_valued_scope_proof_certified" in (
        certificate.missing_obligations
    )


def test_sturm_polynomial_set_valued_constructor_scope_is_constructor_derived():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="validated_sturm_cubic_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="cubic_threshold",
                coefficients=(0.0, -0.25, 0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_polynomial_decision_arrangement_child_consumptions(
            arrangement,
        ),
    )
    scoped = certify_supplied_recursive_stratified_set_valued_constructor_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert arrangement.proof_certified
    assert recursive.proof_certified
    assert recursive.recursion_kind == "sturm_polynomial_decision_arrangement"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not scoped.proof_certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        scoped.missing_obligations
    )
    assert "constructor-derived Sturm polynomial decision arrangement" in (
        scoped.statement
    )
    assert "exact rational root isolation" in scoped.proof_sketch
    assert not validated.proof_certified
    assert "scoped_set_valued_constructor_certificate_proof_certified" in (
        validated.missing_obligations
    )
    assert (
        validated.input_scope_id
        == "finite_sturm_polynomial_decision_arrangement_interval_boxes"
    )
    assert "exact rational root isolation" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


def test_affine_box_decision_arrangement_derives_box_slabs_and_recursive_children():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_box_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="affine_box_event_order",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="x_binary_threshold",
                coefficients=(-0.1, 1.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="y_target_threshold",
                coefficients=(0.2, 0.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
    )
    child_consumptions = derive_affine_box_decision_arrangement_child_consumptions(
        arrangement,
    )
    recursive = certify_affine_box_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=2,
        root_rank=2,
        child_consumptions=child_consumptions,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert arrangement.certified
    assert arrangement.dimension == 2
    assert arrangement.sign_stratum_count == 4
    assert arrangement.equality_stratum_count == 5
    assert len(child_consumptions) == arrangement.equality_stratum_count
    assert recursive.certified
    assert recursive.recursion_kind == "axis_aligned_affine_box_decision_arrangement"
    assert recursive.terminal_leaf_count == 4
    assert recursive.recursive_leaf_count == 5
    assert recursive.strict_descent_edge_count == 5
    assert recursive.child_constructor_source_types == ("AxisAlignedAffineBoxChild",)
    assert {child.root_dimension for child in child_consumptions.values()} == {0, 1}
    simultaneous = [
        stratum
        for stratum in arrangement.strata
        if set(stratum.defining_function_ids)
        == {"x_binary_threshold", "y_target_threshold"}
    ]
    assert len(simultaneous) == 1
    assert simultaneous[0].stratum_kind == "axis_aligned_affine_equality_slab"
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_affine_box_set_valued_constructor_scope_is_constructor_derived():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_box_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="validated_affine_box_event_order",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="x_binary_threshold",
                coefficients=(-0.1, 1.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="y_target_threshold",
                coefficients=(0.2, 0.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
    )
    recursive = certify_affine_box_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=2,
        root_rank=2,
    )
    scoped = certify_supplied_recursive_stratified_set_valued_constructor_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert arrangement.source_tree.source_type == "AxisAlignedAffineBoxArrangement"
    assert arrangement.proof_certified
    assert recursive.proof_certified
    assert recursive.recursion_kind == "axis_aligned_affine_box_decision_arrangement"
    assert recursive.child_constructor_source_types == ("AxisAlignedAffineBoxChild",)
    assert not scoped.proof_certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        scoped.missing_obligations
    )
    assert scoped.event_order_consumption_certificate is recursive
    assert "axis-aligned affine box arrangement" in scoped.statement
    assert "coordinate equality slabs" in scoped.proof_sketch
    assert not validated.proof_certified
    assert "scoped_set_valued_constructor_certificate_proof_certified" in (
        validated.missing_obligations
    )
    assert (
        validated.input_scope_id
        == "finite_axis_aligned_affine_box_arrangement_interval_boxes"
    )
    assert "Cartesian sign boxes" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


def test_affine_box_decision_arrangement_derives_coincident_axis_child():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_box_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="coincident_axis_affine_box_event_order",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="first_x_threshold",
                coefficients=(0.0, 1.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="second_x_threshold",
                coefficients=(0.0, 2.0, 0.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
    )
    child_consumptions = derive_affine_box_decision_arrangement_child_consumptions(
        arrangement,
    )
    recursive = certify_affine_box_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=2,
        root_rank=2,
        child_consumptions=child_consumptions,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert arrangement.certified
    assert arrangement.sign_stratum_count == 2
    assert arrangement.equality_stratum_count == 1
    simultaneous = [stratum for stratum in arrangement.strata if stratum.equality]
    assert len(simultaneous) == 1
    assert set(simultaneous[0].defining_function_ids) == {
        "first_x_threshold",
        "second_x_threshold",
    }
    assert len(child_consumptions) == 1
    child = child_consumptions[simultaneous[0].stratum_id]
    assert child.constructor_source_type == "AxisAlignedAffineBoxChild"
    assert child.root_dimension == 1
    assert child.proof_certified
    assert recursive.strict_descent_edge_count == 1
    assert recursive.proof_certified
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_affine_box_decision_arrangement_rejects_oblique_hyperplane_hulling():
    try:
        certify_affine_box_decision_arrangement_stratified_branch_event_tree(
            arrangement_id="oblique_box_event_order",
            decision_functions=(
                AffineBoxDecisionFunctionSpec(
                    decision_id="oblique_event_boundary",
                    coefficients=(0.0, 1.0, 1.0),
                ),
            ),
            domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        )
    except ValueError as exc:
        assert "axis-aligned decisions" in str(exc)
    else:
        raise AssertionError("oblique affine box decision was hulled into boxes")


def test_affine_halfspace_decision_lifts_oblique_boundary_without_hulling():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    stratification = certify_affine_halfspace_decision_stratified_branch_event_tree(
        decision_id="oblique_event_boundary",
        coefficients=(0.0, 1.0, 1.0),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )
    recursive = certify_affine_halfspace_decision_recursive_consumption(
        stratification,
        root_dimension=2,
        root_rank=2,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert stratification.certified
    assert stratification.sign_stratum_count == 2
    assert stratification.equality_stratum_count == 1
    assert stratification.slab_half_width == 0.25
    assert recursive.certified
    assert recursive.recursion_kind == "affine_halfspace_decision"
    assert recursive.terminal_leaf_count == 2
    assert recursive.recursive_leaf_count == 1
    assert recursive.strict_descent_edge_count == 1
    assert recursive.child_constructor_source_types == ("AffineHalfspaceDecisionChild",)
    slab = [cell for cell in stratification.cells if cell.equality][0]
    assert slab.constraints == (
        "oblique_event_boundary >= -0.25",
        "oblique_event_boundary <= 0.25",
    )
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_affine_halfspace_decision_set_valued_scope_is_constructor_derived():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    stratification = certify_affine_halfspace_decision_stratified_branch_event_tree(
        decision_id="validated_oblique_event_boundary",
        coefficients=(0.0, 1.0, 1.0),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )
    recursive = certify_affine_halfspace_decision_recursive_consumption(
        stratification,
        root_dimension=2,
        root_rank=2,
        child_consumptions=derive_affine_halfspace_decision_child_consumptions(
            stratification,
        ),
    )
    scoped = certify_supplied_recursive_stratified_set_valued_constructor_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert stratification.source_tree.source_type == "AffineHalfspaceDecision"
    assert stratification.proof_certified
    assert recursive.proof_certified
    assert recursive.recursion_kind == "affine_halfspace_decision"
    assert recursive.child_constructor_source_types == ("AffineHalfspaceDecisionChild",)
    assert not scoped.proof_certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        scoped.missing_obligations
    )
    assert scoped.event_order_consumption_certificate is recursive
    assert "affine halfspace decision stratification" in scoped.statement
    assert "central equality slab" in scoped.proof_sketch
    assert not validated.proof_certified
    assert "scoped_set_valued_constructor_certificate_proof_certified" in (
        validated.missing_obligations
    )
    assert (
        validated.input_scope_id
        == "finite_affine_halfspace_decision_interval_boxes"
    )
    assert "separated halfspace cells" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


def test_affine_halfspace_arrangement_derives_intersecting_oblique_polygon_cells():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="oblique_two_line_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="sum_boundary",
                coefficients=(0.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="difference_boundary",
                coefficients=(0.0, 1.0, -1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("oblique-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    recursive = certify_affine_halfspace_arrangement_recursive_consumption(
        arrangement,
        root_dimension=2,
        root_rank=2,
        child_consumptions={
            cell.cell_id: child
            for cell in arrangement.cells
            if cell.equality
        },
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert arrangement.certified
    assert arrangement.dimension == 2
    assert arrangement.sign_stratum_count == 4
    assert arrangement.equality_stratum_count == 5
    assert arrangement.area_cover_certified
    assert arrangement.cover_area_gap_upper_bound <= 1.0e-8
    assert abs(arrangement.cell_area_sum - arrangement.domain_area) <= 1.0e-8
    assert all(cell.area_lower_bound > 0.0 for cell in arrangement.cells)
    for cell in arrangement.cells:
        assert len(cell.value_bounds) == 2
        if not cell.equality:
            for sign_entry, (lower, upper) in zip(
                cell.sign_vector,
                cell.value_bounds,
            ):
                if sign_entry.endswith(":+"):
                    assert lower >= arrangement.slab_half_width - 1.0e-8
                elif sign_entry.endswith(":-"):
                    assert upper <= -arrangement.slab_half_width + 1.0e-8
    assert any(
        set(cell.defining_function_ids)
        == {"sum_boundary", "difference_boundary"}
        for cell in arrangement.cells
        if cell.equality
    )
    assert recursive.certified
    assert recursive.recursion_kind == "affine_halfspace_arrangement"
    assert recursive.terminal_leaf_count == 4
    assert recursive.recursive_leaf_count == 5
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_affine_halfspace_arrangement_derives_line_child_consumptions():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="line_child_oblique_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="vertical_boundary",
                coefficients=(0.0, 1.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="positive_offset_boundary",
                coefficients=(2.0, 0.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    child_consumptions = derive_affine_halfspace_arrangement_line_child_consumptions(
        arrangement,
    )
    recursive = certify_affine_halfspace_arrangement_recursive_consumption(
        arrangement,
        root_dimension=2,
        root_rank=2,
        child_consumptions=child_consumptions,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert arrangement.proof_certified
    assert arrangement.equality_stratum_count == 1
    assert len(child_consumptions) == 1
    child = next(iter(child_consumptions.values()))
    assert child.constructor_source_type == "AffineDecisionArrangement"
    assert child.root_dimension == 1
    assert child.terminal_leaf_count == 1
    assert child.recursive_leaf_count == 0
    assert child.proof_certified
    assert recursive.proof_certified
    assert recursive.strict_descent_edge_count == 1
    assert recursive.descent_well_founded
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_affine_halfspace_arrangement_derives_terminal_line_child_without_remaining_decisions():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="terminal_line_child_oblique_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="diagonal_boundary",
                coefficients=(0.0, 1.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    child_consumptions = derive_affine_halfspace_arrangement_line_child_consumptions(
        arrangement,
    )
    recursive = certify_affine_halfspace_arrangement_recursive_consumption(
        arrangement,
        root_dimension=2,
        root_rank=2,
        child_consumptions=child_consumptions,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert arrangement.equality_stratum_count == 1
    assert len(child_consumptions) == 1
    child = next(iter(child_consumptions.values()))
    assert child.constructor_source_type == "AffineHalfspaceLineChild"
    assert child.root_dimension == 1
    assert child.terminal_leaf_count == 1
    assert child.recursive_leaf_count == 0
    assert child.proof_certified
    assert recursive.strict_descent_edge_count == 1
    assert recursive.proof_certified
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_affine_halfspace_arrangement_derives_point_child_consumptions():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="point_child_oblique_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="sum_boundary",
                coefficients=(0.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="difference_boundary",
                coefficients=(0.0, 1.0, -1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    point_children = derive_affine_halfspace_arrangement_point_child_consumptions(
        arrangement,
    )
    all_children = derive_affine_halfspace_arrangement_child_consumptions(
        arrangement,
    )
    recursive = certify_affine_halfspace_arrangement_recursive_consumption(
        arrangement,
        root_dimension=2,
        root_rank=2,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert arrangement.equality_stratum_count == 5
    assert len(point_children) == 1
    point_child = next(iter(point_children.values()))
    assert point_child.constructor_source_type == "AffineHalfspacePointChild"
    assert point_child.root_dimension == 0
    assert point_child.terminal_leaf_count == 1
    assert point_child.recursive_leaf_count == 0
    assert point_child.proof_certified
    assert len(all_children) == arrangement.equality_stratum_count
    assert recursive.recursive_leaf_count == arrangement.equality_stratum_count
    assert recursive.strict_descent_edge_count == arrangement.equality_stratum_count
    assert recursive.proof_certified
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_affine_halfspace_arrangement_groups_coincident_oblique_boundaries():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="coincident_oblique_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="first_same_boundary",
                coefficients=(0.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="second_same_boundary",
                coefficients=(0.0, 1.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )
    children = derive_affine_halfspace_arrangement_line_child_consumptions(
        arrangement,
    )
    recursive = certify_affine_halfspace_arrangement_recursive_consumption(
        arrangement,
        root_dimension=2,
        root_rank=2,
        child_consumptions=children,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert arrangement.certified
    assert arrangement.area_cover_certified
    assert arrangement.sign_stratum_count == 2
    assert arrangement.equality_stratum_count == 1
    simultaneous = [cell for cell in arrangement.cells if cell.equality]
    assert len(simultaneous) == 1
    assert set(simultaneous[0].defining_function_ids) == {
        "first_same_boundary",
        "second_same_boundary",
    }
    assert len(children) == 1
    child = children[simultaneous[0].cell_id]
    assert child.constructor_source_type == "AffineHalfspaceLineChild"
    assert child.root_dimension == 1
    assert child.terminal_leaf_count == 1
    assert child.proof_certified
    assert recursive.strict_descent_edge_count == 1
    assert recursive.proof_certified
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_affine_halfspace_arrangement_line_child_rejects_distinct_parallel_boundaries():
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="parallel_distinct_oblique_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="first_parallel_boundary",
                coefficients=(0.0, 1.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="second_parallel_boundary",
                coefficients=(-0.1, 1.0, 0.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )
    children = derive_affine_halfspace_arrangement_line_child_consumptions(
        arrangement,
    )

    simultaneous = [
        cell for cell in arrangement.cells if len(cell.defining_function_ids) == 2
    ]
    assert len(simultaneous) == 1
    assert simultaneous[0].cell_id not in children
    assert all(
        len(cell.defining_function_ids) == 1
        for cell in arrangement.cells
        if cell.cell_id in children
    )


def test_affine_halfspace_arrangement_certification_depends_on_area_cover():
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="area_guarded_oblique_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="sum_boundary",
                coefficients=(0.0, 1.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )

    assert arrangement.certified
    assert not replace(arrangement, area_cover_certified=False).certified
    assert not replace(
        arrangement,
        cover_area_gap_upper_bound=arrangement.domain_area,
    ).certified


def test_affine_halfspace_3d_arrangement_derives_spatial_oblique_cells():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="spatial_oblique_two_plane_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="sum_boundary",
                coefficients=(0.0, 1.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="mixed_boundary",
                coefficients=(0.0, 1.0, -1.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    automatic_children = derive_affine_halfspace_3d_arrangement_child_consumptions(
        arrangement,
    )
    recursive = certify_affine_halfspace_3d_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert arrangement.certified
    assert arrangement.dimension == 3
    assert arrangement.sign_stratum_count == 4
    assert arrangement.equality_stratum_count == 5
    assert arrangement.volume_cover_certified
    assert arrangement.cover_volume_gap_upper_bound <= 1.0e-8
    assert abs(arrangement.cell_volume_sum - arrangement.domain_volume) <= 1.0e-8
    assert all(cell.volume_lower_bound > 0.0 for cell in arrangement.cells)
    for cell in arrangement.cells:
        assert len(cell.value_bounds) == 2
        assert all(len(vertex) == 3 for vertex in cell.vertices)
        if not cell.equality:
            for sign_entry, (lower, upper) in zip(
                cell.sign_vector,
                cell.value_bounds,
            ):
                if sign_entry.endswith(":+"):
                    assert lower >= arrangement.slab_half_width - 1.0e-8
                elif sign_entry.endswith(":-"):
                    assert upper <= -arrangement.slab_half_width + 1.0e-8
    assert any(
        set(cell.defining_function_ids) == {"sum_boundary", "mixed_boundary"}
        for cell in arrangement.cells
        if cell.equality
    )
    assert recursive.certified
    assert recursive.recursion_kind == "affine_halfspace_3d_arrangement"
    assert recursive.terminal_leaf_count == 4
    assert recursive.recursive_leaf_count == 5
    assert len(automatic_children) == arrangement.equality_stratum_count
    assert "AffineHalfspacePlaneChild" in recursive.child_constructor_source_types
    assert "AffineHalfspaceSpatialLineChild" in recursive.child_constructor_source_types
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_affine_halfspace_3d_arrangement_derives_terminal_plane_child_without_remaining_decisions():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="terminal_plane_child_spatial_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="diagonal_plane_boundary",
                coefficients=(0.0, 1.0, 1.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )

    child_consumptions = derive_affine_halfspace_3d_arrangement_plane_child_consumptions(
        arrangement,
    )
    recursive = certify_affine_halfspace_3d_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=child_consumptions,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert arrangement.certified
    assert arrangement.sign_stratum_count == 2
    assert arrangement.equality_stratum_count == 1
    assert len(child_consumptions) == 1
    child = next(iter(child_consumptions.values()))
    assert child.constructor_source_type == "AffineHalfspacePlaneChild"
    assert recursive_constructor_source_scope(child)[1] == (
        "finite_2d_affine_halfspace_plane_child_interval_boxes"
    )
    assert child.root_dimension == 2
    assert child.terminal_leaf_count == 1
    assert child.recursive_leaf_count == 0
    assert child.proof_certified
    assert recursive.recursive_leaf_count == 1
    assert recursive.strict_descent_edge_count == 1
    assert recursive.descent_well_founded
    assert recursive.proof_certified
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_affine_halfspace_3d_arrangement_derives_coincident_plane_child():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="coincident_plane_child_spatial_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="first_same_plane",
                coefficients=(0.0, 1.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="second_same_plane",
                coefficients=(0.0, 1.0, 1.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )

    child_consumptions = derive_affine_halfspace_3d_arrangement_plane_child_consumptions(
        arrangement,
    )
    recursive = certify_affine_halfspace_3d_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=child_consumptions,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert arrangement.certified
    assert arrangement.volume_cover_certified
    assert abs(arrangement.cell_volume_sum - arrangement.domain_volume) <= 1.0e-8
    assert arrangement.sign_stratum_count == 2
    assert arrangement.equality_stratum_count == 1
    simultaneous = [cell for cell in arrangement.cells if cell.equality]
    assert len(simultaneous) == 1
    assert set(simultaneous[0].defining_function_ids) == {
        "first_same_plane",
        "second_same_plane",
    }
    assert len(child_consumptions) == 1
    child = child_consumptions[simultaneous[0].cell_id]
    assert child.constructor_source_type == "AffineHalfspacePlaneChild"
    assert child.root_dimension == 2
    assert child.terminal_leaf_count == 1
    assert child.proof_certified
    assert recursive.strict_descent_edge_count == 1
    assert recursive.proof_certified
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )


def test_affine_halfspace_3d_plane_child_rejects_distinct_parallel_planes():
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="parallel_distinct_plane_spatial_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="first_parallel_plane",
                coefficients=(0.0, 1.0, 0.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="second_parallel_plane",
                coefficients=(-0.1, 1.0, 0.0, 0.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )
    children = derive_affine_halfspace_3d_arrangement_plane_child_consumptions(
        arrangement,
    )

    simultaneous = [
        cell for cell in arrangement.cells if len(cell.defining_function_ids) == 2
    ]
    assert arrangement.certified
    assert arrangement.volume_cover_certified
    assert len(simultaneous) == 1
    assert simultaneous[0].cell_id not in children


def test_affine_halfspace_3d_arrangement_derives_spatial_line_child():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="spatial_line_child_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="x_plane",
                coefficients=(0.0, 1.0, 0.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="y_plane",
                coefficients=(0.0, 0.0, 1.0, 0.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )

    automatic_children = derive_affine_halfspace_3d_arrangement_child_consumptions(
        arrangement,
    )
    line_children = derive_affine_halfspace_3d_arrangement_line_child_consumptions(
        arrangement,
    )
    recursive = certify_affine_halfspace_3d_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=automatic_children,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert len(line_children) == 1
    child = next(iter(line_children.values()))
    assert child.constructor_source_type == "AffineHalfspaceSpatialLineChild"
    assert recursive_constructor_source_scope(child)[1] == (
        "finite_1d_affine_halfspace_spatial_line_child_interval_boxes"
    )
    assert child.root_dimension == 1
    assert child.terminal_leaf_count == 1
    assert child.proof_certified
    assert "AffineHalfspacePlaneChild" in recursive.child_constructor_source_types
    assert "AffineHalfspaceSpatialLineChild" in recursive.child_constructor_source_types
    assert recursive.strict_descent_edge_count == arrangement.equality_stratum_count
    assert recursive.proof_certified
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )


def test_affine_halfspace_3d_arrangement_derives_spatial_point_child():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="spatial_point_child_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="x_plane",
                coefficients=(0.0, 1.0, 0.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="y_plane",
                coefficients=(0.0, 0.0, 1.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="z_plane",
                coefficients=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )

    automatic_children = derive_affine_halfspace_3d_arrangement_child_consumptions(
        arrangement,
    )
    point_children = derive_affine_halfspace_3d_arrangement_point_child_consumptions(
        arrangement,
    )
    recursive = certify_affine_halfspace_3d_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=automatic_children,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )

    assert len(point_children) == 1
    point_child = next(iter(point_children.values()))
    assert point_child.constructor_source_type == "AffineHalfspaceSpatialPointChild"
    assert recursive_constructor_source_scope(point_child)[1] == (
        "finite_0d_affine_halfspace_spatial_point_child_interval_boxes"
    )
    assert point_child.root_dimension == 0
    assert point_child.terminal_leaf_count == 1
    assert point_child.proof_certified
    assert "AffineHalfspacePlaneChild" in recursive.child_constructor_source_types
    assert "AffineDecisionArrangement" in recursive.child_constructor_source_types
    assert "AffineHalfspaceSpatialPointChild" in recursive.child_constructor_source_types
    assert recursive.strict_descent_edge_count == arrangement.equality_stratum_count
    assert recursive.proof_certified
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )


def test_affine_halfspace_decision_requires_positive_slab_width():
    try:
        certify_affine_halfspace_decision_stratified_branch_event_tree(
            decision_id="bad_oblique_event_boundary",
            coefficients=(0.0, 1.0, 1.0),
            domain_box=((-1.0, 1.0), (-1.0, 1.0)),
            slab_half_width=0.0,
        )
    except ValueError as exc:
        assert "slab_half_width" in str(exc)
    else:
        raise AssertionError("zero-width affine halfspace slab was accepted")


def test_stratified_tree_rejects_manual_recursive_exhaustion_flag():
    try:
        certify_stratified_branch_event_tree(
            _positive_margin_stratified_tree("manual:flag"),
            recursive_exhaustion_certified=True,
        )
    except TypeError as exc:
        assert "certify_recursive_stratified_branch_event_consumption" in str(exc)
    else:
        raise AssertionError("manual recursive_exhaustion_certified flag was accepted")


def test_uniform_margin_refinement_closes_search_obligations_under_explicit_margins():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    branch_refinement = certify_uniform_margin_branch_refinement_termination(
        refinement_kind="state_branch_partition",
        uniform_decision_margin=0.02,
        local_decision_lipschitz_bound=4.0,
        initial_width_bound=0.5,
        refinement_factor=0.5,
    )
    event_refinement = certify_uniform_margin_branch_refinement_termination(
        refinement_kind="event_order_partition",
        uniform_decision_margin=0.01,
        local_decision_lipschitz_bound=5.0,
        initial_width_bound=0.2,
        refinement_factor=0.5,
    )

    assert branch_refinement.certified
    assert event_refinement.certified
    assert branch_refinement.max_depth > 0
    assert branch_refinement.terminal_width_bound <= (
        branch_refinement.required_width_bound
    )
    assert "zero margin" in branch_refinement.proof_sketch

    branch_only = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_branch_refinement_certificate=branch_refinement,
    )
    assert not branch_only.certified
    assert "recursive_set_valued_branch_partition_consumption" not in (
        branch_only.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" in (
        branch_only.missing_obligations
    )

    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_branch_refinement_certificate=branch_refinement,
        event_order_refinement_certificate=event_refinement,
    )

    assert search.certified
    assert not search.proof_certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        search.missing_obligations
    )
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )
    details = {obligation.obligation: obligation for obligation in search.obligations}
    assert details["recursive_set_valued_branch_partition_consumption"].certified
    assert details["event_order_partition_consumption_theorem"].certified
    assert "uniform-margin recursive branch refinement terminates" in (
        details["recursive_set_valued_branch_partition_consumption"].detail
    )


def test_uniform_margin_set_valued_constructor_completeness_closes_positive_margin_subset():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    branch_refinement = certify_uniform_margin_branch_refinement_termination(
        refinement_kind="state_branch_partition",
        uniform_decision_margin=0.02,
        local_decision_lipschitz_bound=4.0,
        initial_width_bound=0.5,
        refinement_factor=0.5,
    )
    event_refinement = certify_uniform_margin_branch_refinement_termination(
        refinement_kind="event_order_partition",
        uniform_decision_margin=0.01,
        local_decision_lipschitz_bound=5.0,
        initial_width_bound=0.2,
        refinement_factor=0.5,
    )
    certificate = certify_uniform_margin_set_valued_constructor_completeness(
        theorem,
        recursive_branch_refinement_certificate=branch_refinement,
        event_order_refinement_certificate=event_refinement,
    )

    assert not certificate.certified
    assert not certificate.proof_certified
    assert not certificate.equality_strata_claimed
    assert certificate.search_completeness_certificate.certified
    assert not certificate.search_completeness_certificate.proof_certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        certificate.missing_obligations
    )
    assert "positive uniform margins" in certificate.statement
    assert "does not handle simultaneous-event" in certificate.proof_sketch
    details = {obligation.obligation: obligation for obligation in certificate.obligations}
    assert details["equality_strata_excluded_by_positive_margins"].certified
    assert "recursive stratified theorem" in (
        details["equality_strata_excluded_by_positive_margins"].detail
    )


def test_validated_set_valued_constructor_theorem_wraps_scoped_interval_box_certificate():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    branch_refinement = certify_uniform_margin_branch_refinement_termination(
        refinement_kind="state_branch_partition",
        uniform_decision_margin=0.02,
        local_decision_lipschitz_bound=4.0,
        initial_width_bound=0.5,
        refinement_factor=0.5,
    )
    event_refinement = certify_uniform_margin_branch_refinement_termination(
        refinement_kind="event_order_partition",
        uniform_decision_margin=0.01,
        local_decision_lipschitz_bound=5.0,
        initial_width_bound=0.2,
        refinement_factor=0.5,
    )
    scoped = certify_uniform_margin_set_valued_constructor_completeness(
        theorem,
        recursive_branch_refinement_certificate=branch_refinement,
        event_order_refinement_certificate=event_refinement,
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )

    assert not scoped.certified
    assert not scoped.proof_certified
    assert validated.theorem_id == "validated_set_valued_constructor_completeness"
    assert validated.input_scope_id == "positive_margin_interval_boxes"
    assert not validated.certified
    assert not validated.proof_certified
    assert validated.set_valued_constructor_certificate is scoped
    assert not validated.arbitrary_partition_generation_claimed
    assert "pointwise_finite_target_theorem_proof_certified" in (
        validated.missing_obligations
    )
    assert "certificate_search_completeness_proof_certified" in (
        validated.missing_obligations
    )
    assert "scoped_set_valued_constructor_certificate_proof_certified" in (
        validated.missing_obligations
    )
    assert "arbitrary recursive partition generation remains open" in (
        validated.proof_sketch
    )


def test_validated_set_valued_constructor_rejects_truthy_nested_proof_flags():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    branch_refinement = certify_uniform_margin_branch_refinement_termination(
        refinement_kind="state_branch_partition",
        uniform_decision_margin=0.02,
        local_decision_lipschitz_bound=4.0,
        initial_width_bound=0.5,
        refinement_factor=0.5,
    )
    event_refinement = certify_uniform_margin_branch_refinement_termination(
        refinement_kind="event_order_partition",
        uniform_decision_margin=0.01,
        local_decision_lipschitz_bound=5.0,
        initial_width_bound=0.2,
        refinement_factor=0.5,
    )
    scoped = certify_uniform_margin_set_valued_constructor_completeness(
        theorem,
        recursive_branch_refinement_certificate=branch_refinement,
        event_order_refinement_certificate=event_refinement,
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    truthy_scoped = replace(
        scoped,
        theorem_certificate=SimpleNamespace(proof_certified="yes"),
        search_completeness_certificate=SimpleNamespace(proof_certified="yes"),
    )
    truthy_validated_theorem = replace(
        validated,
        theorem_certificate=SimpleNamespace(proof_certified="yes"),
    )
    truthy_validated_search = replace(
        validated,
        search_completeness_certificate=SimpleNamespace(proof_certified="yes"),
    )
    truthy_validated_scoped = replace(
        validated,
        set_valued_constructor_certificate=SimpleNamespace(proof_certified="yes"),
    )
    stale_validated_search = replace(
        validated,
        search_completeness_certificate=certify_finite_target_certificate_search_completeness(
            theorem,
            recursive_branch_refinement_certificate=branch_refinement,
            event_order_refinement_certificate=event_refinement,
        ),
    )

    assert not scoped.certified
    assert not scoped.proof_certified
    assert not validated.proof_certified
    assert not truthy_scoped.certified
    assert not truthy_scoped.proof_certified
    assert "uniform_margin_component_types" in truthy_scoped.missing_obligations
    assert not truthy_validated_theorem.certified
    assert not truthy_validated_theorem.proof_certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        truthy_validated_theorem.missing_obligations
    )
    assert "pointwise_finite_target_theorem_type" in (
        truthy_validated_theorem.missing_obligations
    )
    assert not truthy_validated_search.certified
    assert not truthy_validated_search.proof_certified
    assert "certificate_search_completeness_proof_certified" in (
        truthy_validated_search.missing_obligations
    )
    assert "certificate_search_completeness_type" in (
        truthy_validated_search.missing_obligations
    )
    assert not truthy_validated_scoped.certified
    assert not truthy_validated_scoped.proof_certified
    assert "scoped_set_valued_constructor_certificate_proof_certified" in (
        truthy_validated_scoped.missing_obligations
    )
    assert "scoped_set_valued_constructor_certificate_type" in (
        truthy_validated_scoped.missing_obligations
    )
    assert not stale_validated_search.certified
    assert not stale_validated_search.proof_certified
    assert "validated_set_valued_sources_match" in (
        stale_validated_search.missing_obligations
    )


def test_finite_target_search_completeness_rejects_spoofed_nested_theorem():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    branch_refinement = certify_uniform_margin_branch_refinement_termination(
        refinement_kind="state_branch_partition",
        uniform_decision_margin=0.02,
        local_decision_lipschitz_bound=4.0,
        initial_width_bound=0.5,
        refinement_factor=0.5,
    )
    event_refinement = certify_uniform_margin_branch_refinement_termination(
        refinement_kind="event_order_partition",
        uniform_decision_margin=0.01,
        local_decision_lipschitz_bound=5.0,
        initial_width_bound=0.2,
        refinement_factor=0.5,
    )
    search = certify_finite_target_certificate_search_completeness(
        theorem,
        recursive_branch_refinement_certificate=branch_refinement,
        event_order_refinement_certificate=event_refinement,
    )
    fake_theorem_search = replace(
        search,
        theorem_certificate=SimpleNamespace(
            certified=True,
            proof_certified=True,
            theorem_id="pointwise_finite_target_atlas_or_stop_completeness",
            missing_obligations=(),
        ),
    )
    wrong_source_search = replace(
        search,
        theorem_certificate=replace(theorem, theorem_id="spoofed_pointwise_theorem"),
    )

    assert search.certified
    assert not search.proof_certified
    assert not fake_theorem_search.certified
    assert not fake_theorem_search.proof_certified
    assert "pointwise_finite_target_theorem_type" in (
        fake_theorem_search.missing_obligations
    )
    assert not wrong_source_search.certified
    assert not wrong_source_search.proof_certified
    assert "pointwise_finite_target_theorem_source" in (
        wrong_source_search.missing_obligations
    )


def test_affine_halfspace_arrangement_set_valued_constructor_completeness_feeds_validated_scope():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="validated_oblique_two_line_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="sum_boundary",
                coefficients=(0.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="difference_boundary",
                coefficients=(0.0, 1.0, -1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem,
        arrangement_certificate=arrangement,
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    recursive = scoped.branch_consumption_certificate

    assert not scoped.certified
    assert not scoped.proof_certified
    assert scoped.equality_strata_claimed
    assert not scoped.arbitrary_partition_generation_claimed
    assert scoped.event_order_consumption_certificate is recursive
    assert "AffineDecisionArrangement" in recursive.child_constructor_source_types
    assert "AffineHalfspacePointChild" in recursive.child_constructor_source_types
    assert scoped.search_completeness_certificate.certified
    assert not scoped.search_completeness_certificate.proof_certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        scoped.missing_obligations
    )
    details = {obligation.obligation: obligation for obligation in scoped.obligations}
    assert details["affine_halfspace_arrangement_area_cover_certified"].certified
    assert details["branch_consumption_uses_arrangement_stratified_tree"].certified
    assert details["event_order_consumption_uses_arrangement_stratified_tree"].certified
    assert not validated.certified
    assert not validated.proof_certified
    assert (
        validated.input_scope_id
        == "finite_2d_affine_halfspace_arrangement_interval_boxes"
    )
    assert validated.set_valued_constructor_certificate is scoped
    assert not validated.arbitrary_partition_generation_claimed


def test_affine_halfspace_3d_arrangement_set_valued_constructor_scope_is_spatial():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="validated_spatial_two_plane_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="sum_boundary",
                coefficients=(0.0, 1.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="mixed_boundary",
                coefficients=(0.0, 1.0, -1.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem,
        arrangement_certificate=arrangement,
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    recursive = scoped.branch_consumption_certificate

    assert not scoped.certified
    assert not scoped.proof_certified
    assert scoped.event_order_consumption_certificate is recursive
    assert "AffineHalfspacePlaneChild" in recursive.child_constructor_source_types
    assert "AffineHalfspaceSpatialLineChild" in recursive.child_constructor_source_types
    details = {obligation.obligation: obligation for obligation in scoped.obligations}
    assert details["affine_halfspace_arrangement_volume_cover_certified"].certified
    assert details["branch_consumption_uses_arrangement_stratified_tree"].certified
    assert details["event_order_consumption_uses_arrangement_stratified_tree"].certified
    assert not validated.certified
    assert not validated.proof_certified
    assert (
        validated.input_scope_id
        == "finite_3d_affine_halfspace_arrangement_interval_boxes"
    )
    assert validated.set_valued_constructor_certificate is scoped
    assert not validated.arbitrary_partition_generation_claimed


def test_supplied_recursive_stratified_set_valued_constructor_completeness_closes_displayed_equality_tree():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="event_order_arrangement",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_total",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
            ),
            PolynomialDecisionFunctionSpec(
                decision_id="ks_exit_minus_target",
                coefficients=(-0.5, 1.0),
                root_brackets=((0.49, 0.51),),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("set-valued-arrangement-child"),
        root_dimension=2,
        root_rank=1,
    )
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            "event_order_arrangement:root:0:binary_minus_total": child,
            "event_order_arrangement:root:1:ks_exit_minus_target": child,
        },
    )
    certificate = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem,
            recursive_stratified_branch_consumption_certificate=recursive,
        )
    )

    assert recursive.certified
    assert not certificate.certified
    assert not certificate.proof_certified
    assert certificate.event_order_consumption_certificate is recursive
    assert certificate.equality_strata_claimed
    assert not certificate.arbitrary_partition_generation_claimed
    assert certificate.search_completeness_certificate.certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        certificate.missing_obligations
    )
    assert "certificate_search_completeness_proof_certified" in (
        certificate.missing_obligations
    )
    assert "supplied finite recursive stratified trees" in certificate.statement
    assert "does not construct the recursive stratification" in (
        certificate.proof_sketch
    )
    details = {obligation.obligation: obligation for obligation in certificate.obligations}
    assert details["recursive_stratified_branch_consumption_certified"].certified
    assert details["recursive_stratified_event_order_consumption_certified"].certified
    assert details["arbitrary_partition_generation_not_claimed"].certified
    assert not details["arbitrary_partition_generation_not_claimed"].required

    try:
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem,
            recursive_stratified_branch_consumption_certificate=True,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    except TypeError as error:
        assert "RecursiveStratifiedBranchEventConsumptionCertificate" in str(error)
    else:
        raise AssertionError("raw boolean recursive consumption was accepted")

    try:
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=True,
        )
    except TypeError as error:
        assert "RecursiveStratifiedBranchEventConsumptionCertificate" in str(error)
    else:
        raise AssertionError("raw boolean event-order consumption was accepted")
