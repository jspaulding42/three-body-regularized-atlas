from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

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
    SelectorPolicyLeafCertificate,
    StratifiedBranchLeafCertificate,
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
    certify_polynomial_decision_arrangement_recursive_consumption,
    certify_polynomial_decision_arrangement_stratified_branch_event_tree,
    derive_polynomial_decision_arrangement_child_consumptions,
    certify_polynomial_decision_recursive_consumption,
    certify_polynomial_decision_stratified_branch_event_tree,
    derive_polynomial_decision_child_consumptions,
    certify_quadratic_decision_arrangement_stratified_branch_event_tree,
    certify_recursive_stratified_branch_event_consumption,
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
    build_analytic_lemma_registry_for_finite_target_theorem,
    certify_affine_halfspace_arrangement_set_valued_constructor_completeness,
    certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness,
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

    assert emitted_source_types
    assert not missing
    assert not extra
    assert len(input_scope_ids) == len(set(input_scope_ids))
    assert all(scope_id.startswith("finite_") for scope_id in input_scope_ids)
    assert all(scope_id.endswith("_interval_boxes") for scope_id in input_scope_ids)


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


def test_pointwise_finite_target_atlas_or_stop_theorem_closes_total_stop_gap():
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
    assert theorem.analytic_lemma_proofs_audited
    assert theorem.statement_declared
    assert theorem.scaffold_certified
    assert theorem.proof_certified
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
    assert registry.audit_complete is True
    assert registry.missing_critical_lemma_ids == ()
    assert theorem.unaudited_analytic_lemma_ids == tuple(
        lemma.lemma_id for lemma in theorem.analytic_lemmas if not lemma.audited
    )
    assert theorem.tier_a_unaudited_analytic_lemma_ids == (
        tuple(
            lemma_id
            for lemma_id in FINITE_TARGET_TIER_A_ANALYTIC_LEMMA_IDS
            if lemma_id
            not in {
                "three_body_painleve_no_noncollision_singularities",
                "all_pair_binary_regularization",
                "binary_accumulation_implies_total_collision",
                "binary_collision_isolation",
                "compact_collision_free_taylor_cover",
                "finite_chart_chain_concatenation",
                "target_or_stop_dichotomy",
            }
        )
    )
    expected_tier_b_blockers = tuple(
        lemma_id
        for lemma_id in FINITE_TARGET_TIER_B_ANALYTIC_LEMMA_IDS
        if lemma_id
        not in {
            "binary_degenerate_total_collision_exclusion",
            "reduced_hyperbolic_total_collision_entry",
            "poincare_dulac_fuchsian_log_selector_completeness",
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data",
            "arbitrary_total_collision_germ_entry_to_stop_chart",
            "total_collision_stop_chart_existence",
        }
    )
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
    expected_critical_blockers = tuple(
        lemma_id
        for lemma_id in FINITE_TARGET_CRITICAL_ANALYTIC_LEMMA_IDS
        if lemma_id
        not in {
            "binary_degenerate_total_collision_exclusion",
            "reduced_hyperbolic_total_collision_entry",
            "poincare_dulac_fuchsian_log_selector_completeness",
            "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data",
            "arbitrary_total_collision_germ_entry_to_stop_chart",
            "total_collision_stop_chart_existence",
        }
    )
    assert theorem.critical_unaudited_analytic_lemma_ids == expected_critical_blockers

    records = {record.lemma_id: record for record in registry.records}
    for lemma_id in expected_critical_blockers:
        record = records[lemma_id]
        assert record.status == "declared"
        assert record.proof_mode == "declared_prose"
        assert record.audit_tier == "tier_b_total_collision_entry_frontier"
        assert record.declared
        assert not record.audited
        assert record.hypotheses
        assert record.normalization_translation
        assert "declared_prose_not_audited" in record.failure_modes

    assert records["binary_degenerate_total_collision_exclusion"].status == (
        "internally_proven"
    )
    assert records["binary_degenerate_total_collision_exclusion"].proof_mode == (
        "internal_jacobi_perturbed_kepler_blowup_proof"
    )
    assert records["binary_degenerate_total_collision_exclusion"].audited
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
    assert records["reduced_hyperbolic_total_collision_entry"].audited
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
    ].audited
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
    ].audited
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
    ].audited
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
    assert records["total_collision_stop_chart_existence"].audited
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
    assert records["compact_collision_free_taylor_cover"].audited
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
    assert records["binary_accumulation_implies_total_collision"].audited
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
    assert records["binary_collision_isolation"].audited
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
    assert records["all_pair_binary_regularization"].audited
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
    assert records["three_body_painleve_no_noncollision_singularities"].audited
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
    assert records["total_collision_requires_zero_angular_momentum"].audited
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
    assert records["total_collision_central_configuration_asymptotic"].audited
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
    assert records["cubic_time_total_collision_scaling"].audited
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
    ].status == "machine_checked"
    assert records[
        "finite_fuchsian_log_stop_chart_for_admissible_entry_data"
    ].proof_mode == "machine_checked_supplied_fuchsian_log_stop_chart"
    assert records[
        "finite_fuchsian_log_stop_chart_for_admissible_entry_data"
    ].audited
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
    ].status == "machine_checked"
    assert records[
        "homothetic_total_collision_stop_chart_existence"
    ].proof_mode == "machine_checked_homothetic_total_collision_stop_chart"
    assert records["homothetic_total_collision_stop_chart_existence"].audited
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
    assert records["finite_chart_chain_concatenation"].status == "machine_checked"
    assert records["finite_chart_chain_concatenation"].proof_mode == (
        "machine_checked_chart_chain_certificate"
    )
    assert records["finite_chart_chain_concatenation"].audited
    assert "chart_chain_certificate_checker" in (
        records["finite_chart_chain_concatenation"].checker_inputs
    )
    assert records["target_or_stop_dichotomy"].status == "machine_checked"
    assert records["target_or_stop_dichotomy"].proof_mode == (
        "machine_checked_outcome_partition"
    )
    assert records["target_or_stop_dichotomy"].audited
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
    branch = _supplied_generalized_fuchsian_branch()
    certificate = certify_supplied_generalized_fuchsian_entry_data(
        branch=branch,
        radius=0.035,
        sample_taus=(-0.03, 0.03),
        tolerance=1.0e-5,
        energy_tolerance=1.0e-8,
    )

    assert certificate.certified
    assert certificate.proof_certified
    assert certificate.missing_obligations == ()
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
    entry = certify_supplied_generalized_fuchsian_entry_data(
        branch=branch,
        radius=0.035,
        sample_taus=(-0.03, 0.03),
        tolerance=1.0e-5,
        energy_tolerance=1.0e-8,
    )

    partial = certify_supplied_generalized_fuchsian_finite_row_tail_budget(
        entry_certificate=entry,
        retained_total_degree=2,
        radius=0.03,
    )
    full = certify_supplied_generalized_fuchsian_finite_row_tail_budget(
        entry_certificate=entry,
        retained_total_degree=branch.max_total_degree,
        radius=0.03,
    )

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
    entry = certify_supplied_generalized_fuchsian_entry_data(
        branch=branch,
        radius=0.035,
        sample_taus=(-0.03, 0.03),
        tolerance=1.0e-5,
        energy_tolerance=1.0e-8,
    )
    finite_rows = certify_supplied_generalized_fuchsian_finite_row_tail_budget(
        entry_certificate=entry,
        retained_total_degree=branch.max_total_degree,
        radius=0.03,
    )
    majorant = certify_supplied_generalized_fuchsian_analytic_remainder_majorant(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        defect_bound=1.0e-10,
        linear_inverse_bound=2.0,
        nonlinear_lipschitz_bound=0.15,
        component_effective_exponents={
            "value": 1.4,
            "first_jet": 0.4,
            "lifted_residual": 1.1,
            "physical_residual": 0.7,
        },
        step_ratio_bounds={
            "value": 0.25,
            "first_jet": 0.25,
            "lifted_residual": 0.25,
            "physical_residual": 0.25,
        },
        retained_order_initials={
            "value": 8,
            "first_jet": 8,
            "lifted_residual": 8,
            "physical_residual": 8,
        },
        retained_order_increments={
            "value": 2,
            "first_jet": 2,
            "lifted_residual": 2,
            "physical_residual": 2,
        },
        initial_radius=0.025,
        shell_contraction=0.5,
        analytic_disk_fraction=0.2,
    )

    assert majorant.certified
    assert majorant.proof_certified
    assert majorant.missing_obligations == ()
    assert majorant.contraction_factor == 0.3
    assert abs(majorant.banach_contraction_slack - 0.7) < 1.0e-15
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
    entry = certify_supplied_generalized_fuchsian_entry_data(
        branch=branch,
        radius=0.035,
        sample_taus=(-0.03, 0.03),
        tolerance=1.0e-5,
        energy_tolerance=1.0e-8,
    )
    finite_rows = certify_supplied_generalized_fuchsian_finite_row_tail_budget(
        entry_certificate=entry,
        retained_total_degree=branch.max_total_degree,
        radius=0.03,
    )
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
    entry = certify_supplied_generalized_fuchsian_entry_data(
        branch=branch,
        radius=0.035,
        sample_taus=(-0.03, 0.03),
        tolerance=1.0e-5,
        energy_tolerance=1.0e-8,
    )
    finite_rows = certify_supplied_generalized_fuchsian_finite_row_tail_budget(
        entry_certificate=entry,
        retained_total_degree=branch.max_total_degree,
        radius=0.03,
    )
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


@lru_cache(maxsize=1)
def _supplied_generalized_stop_chart_inputs():
    branch = _supplied_generalized_fuchsian_branch()
    entry = certify_supplied_generalized_fuchsian_entry_data(
        branch=branch,
        radius=0.035,
        sample_taus=(-0.03, 0.03),
        tolerance=1.0e-5,
        energy_tolerance=1.0e-8,
    )
    finite_rows = certify_supplied_generalized_fuchsian_finite_row_tail_budget(
        entry_certificate=entry,
        retained_total_degree=branch.max_total_degree,
        radius=0.03,
    )
    majorant = certify_supplied_generalized_fuchsian_analytic_remainder_majorant(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
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
    return entry, finite_rows, majorant


def test_supplied_generalized_fuchsian_stop_chart_consumes_remainder_majorant():
    entry, finite_rows, majorant = _supplied_generalized_stop_chart_inputs()
    stop_chart = certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        remainder_majorant=majorant,
        residual_tolerance=1.0e-5,
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
    assert theorem.proof_certified is True
    assert "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data" not in (
        theorem.critical_unaudited_analytic_lemma_ids
    )
    assert "arbitrary_total_collision_germ_entry_to_stop_chart" not in (
        theorem.critical_unaudited_analytic_lemma_ids
    )
    assert "total_collision_stop_chart_existence" not in (
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
    assert scoped.proof_certified
    assert "single-polynomial decision stratification" in scoped.statement
    assert "strict sign cells" in scoped.proof_sketch
    assert validated.proof_certified
    assert (
        validated.input_scope_id
        == "finite_polynomial_decision_stratification_interval_boxes"
    )
    assert "verified simple root brackets" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


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
    branch_recursive = certify_affine_halfspace_decision_recursive_consumption(
        branch_stratification,
        root_dimension=2,
        root_rank=2,
        child_consumptions=derive_affine_halfspace_decision_child_consumptions(
            branch_stratification,
        ),
    )
    event_recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        event_arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_polynomial_decision_arrangement_child_consumptions(
            event_arrangement,
        ),
    )
    scoped = certify_supplied_recursive_stratified_set_valued_constructor_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=branch_recursive,
        recursive_stratified_event_order_consumption_certificate=event_recursive,
    )
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
    assert scoped.proof_certified
    assert validated.proof_certified
    assert validated.input_scope_id == (
        "finite_mixed_constructor_branch_event_interval_boxes"
    )
    assert "branch source AffineHalfspaceDecision" in validated.proof_sketch
    assert "event-order source PolynomialDecisionArrangement" in (
        validated.proof_sketch
    )
    assert not validated.arbitrary_partition_generation_claimed


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
    branch_recursive = certify_affine_halfspace_3d_arrangement_recursive_consumption(
        branch_arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_affine_halfspace_3d_arrangement_child_consumptions(
            branch_arrangement,
        ),
    )
    event_recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        event_arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_polynomial_decision_arrangement_child_consumptions(
            event_arrangement,
        ),
    )
    scoped = certify_supplied_recursive_stratified_set_valued_constructor_completeness(
        theorem,
        recursive_stratified_branch_consumption_certificate=branch_recursive,
        recursive_stratified_event_order_consumption_certificate=event_recursive,
    )
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
    assert scoped.proof_certified
    assert validated.proof_certified
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
    assert scoped.proof_certified
    assert "affine decision arrangement" in scoped.statement
    assert "coefficient-derived roots" in scoped.proof_sketch
    assert validated.proof_certified
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
    assert scoped.proof_certified
    assert "constructor-derived polynomial decision arrangement" in scoped.statement
    assert "verified simple root brackets" in scoped.proof_sketch
    assert validated.proof_certified
    assert (
        validated.input_scope_id
        == "finite_polynomial_decision_arrangement_interval_boxes"
    )
    assert "verified simple root brackets" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


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
    assert scoped.proof_certified
    assert "quadratic double-root decision arrangement" in scoped.statement
    assert "tangent equality strata" in scoped.proof_sketch
    assert validated.proof_certified
    assert (
        validated.input_scope_id
        == "finite_quadratic_double_root_decision_arrangement_interval_boxes"
    )
    assert "tangent equality strata" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


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
    assert scoped.proof_certified
    assert "computed polynomial-root arrangement" in scoped.statement
    assert "grouped coincident/multiple equality strata" in scoped.proof_sketch
    assert validated.proof_certified
    assert (
        validated.input_scope_id
        == "finite_computed_polynomial_root_arrangement_interval_boxes"
    )
    assert "grouped coincident/multiple equality strata" in validated.proof_sketch
    assert not validated.arbitrary_partition_generation_claimed


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
    assert scoped.proof_certified
    assert "constructor-derived Sturm polynomial decision arrangement" in (
        scoped.statement
    )
    assert "exact rational root isolation" in scoped.proof_sketch
    assert validated.proof_certified
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
    assert scoped.proof_certified
    assert scoped.event_order_consumption_certificate is recursive
    assert "axis-aligned affine box arrangement" in scoped.statement
    assert "coordinate equality slabs" in scoped.proof_sketch
    assert validated.proof_certified
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
    assert scoped.proof_certified
    assert scoped.event_order_consumption_certificate is recursive
    assert "affine halfspace decision stratification" in scoped.statement
    assert "central equality slab" in scoped.proof_sketch
    assert validated.proof_certified
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
    assert search.proof_certified
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

    assert certificate.certified
    assert certificate.proof_certified
    assert not certificate.equality_strata_claimed
    assert certificate.search_completeness_certificate.certified
    assert certificate.missing_obligations == ()
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

    assert scoped.proof_certified
    assert validated.theorem_id == "validated_set_valued_constructor_completeness"
    assert validated.input_scope_id == "positive_margin_interval_boxes"
    assert validated.certified
    assert validated.proof_certified
    assert validated.set_valued_constructor_certificate is scoped
    assert not validated.arbitrary_partition_generation_claimed
    assert validated.missing_obligations == ()
    assert "arbitrary recursive partition generation remains open" in (
        validated.proof_sketch
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

    assert scoped.certified
    assert scoped.proof_certified
    assert scoped.equality_strata_claimed
    assert not scoped.arbitrary_partition_generation_claimed
    assert scoped.event_order_consumption_certificate is recursive
    assert "AffineDecisionArrangement" in recursive.child_constructor_source_types
    assert "AffineHalfspacePointChild" in recursive.child_constructor_source_types
    assert scoped.search_completeness_certificate.certified
    assert scoped.missing_obligations == ()
    details = {obligation.obligation: obligation for obligation in scoped.obligations}
    assert details["affine_halfspace_arrangement_area_cover_certified"].certified
    assert details["branch_consumption_uses_arrangement_stratified_tree"].certified
    assert details["event_order_consumption_uses_arrangement_stratified_tree"].certified
    assert validated.certified
    assert validated.proof_certified
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

    assert scoped.certified
    assert scoped.proof_certified
    assert scoped.event_order_consumption_certificate is recursive
    assert "AffineHalfspacePlaneChild" in recursive.child_constructor_source_types
    assert "AffineHalfspaceSpatialLineChild" in recursive.child_constructor_source_types
    details = {obligation.obligation: obligation for obligation in scoped.obligations}
    assert details["affine_halfspace_arrangement_volume_cover_certified"].certified
    assert details["branch_consumption_uses_arrangement_stratified_tree"].certified
    assert details["event_order_consumption_uses_arrangement_stratified_tree"].certified
    assert validated.certified
    assert validated.proof_certified
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
    assert certificate.certified
    assert certificate.proof_certified
    assert certificate.event_order_consumption_certificate is recursive
    assert certificate.equality_strata_claimed
    assert not certificate.arbitrary_partition_generation_claimed
    assert certificate.search_completeness_certificate.certified
    assert certificate.missing_obligations == ()
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
