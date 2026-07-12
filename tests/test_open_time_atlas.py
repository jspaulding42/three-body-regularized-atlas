from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from three_body_symmetry.branch_event_tree import certify_supplied_branch_event_tree
from three_body_symmetry.certificate_checker import IndependentChartVerifierCertificate
from three_body_symmetry.dynamics import accelerations, split_state
from three_body_symmetry.finite_target_completeness import (
    certify_affine_halfspace_arrangement_set_valued_constructor_completeness,
    certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness,
    certify_constructor_pair_derived_recursive_stratified_set_valued_constructor_completeness,
    certify_finite_target_completeness_theorem,
    certify_supplied_recursive_stratified_set_valued_constructor_completeness,
    certify_uniform_margin_branch_refinement_termination,
    certify_uniform_margin_set_valued_constructor_completeness,
    certify_validated_set_valued_constructor_completeness_theorem,
)
from three_body_symmetry.general_solution_theorem import (
    TheoremPipelineObligation,
    certify_compact_ordinary_binary_finite_atlas,
    certify_positive_mass_noncollision_input_domain,
)
from three_body_symmetry.ks_binary_chart import (
    SpatialKSBinaryChartState,
    ks_binary_chart_to_spatial,
    spatial_to_ks_binary_chart,
)
from three_body_symmetry.ks_binary_series import (
    certify_spatial_ks_competing_binary_entry_event,
    construct_interval_spatial_ks_binary_taylor_solution,
    construct_spatial_ks_binary_taylor_solution,
    interval_spatial_ks_binary_chart_state_from_point,
)
from three_body_symmetry.intervals import FloatInterval
from three_body_symmetry.open_time_atlas import (
    TotalCollisionPolicyCertificate,
    certify_compact_interval_atlas_or_stop_from_finite_targets,
    certify_finite_target_atlas_or_stop_from_validated_atlas,
    certify_finite_target_completeness_reduction,
    certify_pointwise_open_time_locally_finite_atlas_theorem,
    construct_compact_interval_atlas_or_stop,
    construct_compact_interval_exhaustion_family,
    construct_finite_target_atlas_or_stop,
    construct_independent_finite_target_checked_atlas,
    construct_independent_validated_atlas_checked_chain,
    construct_independent_ordinary_taylor_checked_prefix,
    construct_open_time_locally_finite_atlas_theorem,
)
from three_body_symmetry.stratified_branch_tree import (
    AffineBoxDecisionFunctionSpec,
    AnalyticDecisionFunctionCertificate,
    PolynomialDecisionFunctionSpec,
    SelectorPolicyLeafCertificate,
    StratifiedBranchLeafCertificate,
    TotalCollisionClusterLeafCertificate,
    certify_affine_decision_arrangement_stratified_branch_event_tree,
    certify_affine_box_decision_arrangement_recursive_consumption,
    certify_affine_box_decision_arrangement_stratified_branch_event_tree,
    derive_affine_box_decision_arrangement_child_consumptions,
    certify_affine_halfspace_3d_arrangement_recursive_consumption,
    certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree,
    derive_affine_halfspace_3d_arrangement_child_consumptions,
    derive_affine_halfspace_3d_arrangement_line_child_consumptions,
    derive_affine_halfspace_3d_arrangement_plane_child_consumptions,
    derive_affine_halfspace_3d_arrangement_point_child_consumptions,
    certify_affine_halfspace_decision_recursive_consumption,
    certify_affine_halfspace_decision_stratified_branch_event_tree,
    derive_affine_halfspace_decision_child_consumptions,
    certify_affine_halfspace_arrangement_recursive_consumption,
    certify_affine_halfspace_arrangement_stratified_branch_event_tree,
    derive_affine_halfspace_arrangement_child_consumptions,
    derive_affine_halfspace_arrangement_line_child_consumptions,
    derive_affine_halfspace_arrangement_point_child_consumptions,
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
from three_body_symmetry.triple_collision import construct_homothetic_total_collision_branch
from three_body_symmetry.validated_atlas import (
    certify_simultaneous_close_pair_partition,
    validated_atlas_from_spatial_close_pair_branch_partition,
    validated_atlas_from_homothetic_total_collision_branch,
    validated_atlas_from_spatial_ks_competing_binary_handoff,
    validated_atlas_from_spatial_ordinary_ks_handoff,
    validated_atlas_from_parabolic_homothetic_total_collision_branch,
)


def _spatial_initial_data():
    masses = np.array([1.0, 0.7, 1.4])
    positions = np.array(
        [
            [0.8, -0.2, 0.1],
            [-0.4, 0.6, -0.3],
            [0.1, -0.5, 0.7],
        ]
    )
    velocities = np.array(
        [
            [0.05, 0.11, -0.02],
            [-0.07, 0.03, 0.04],
            [0.02, -0.08, 0.01],
        ]
    )
    return masses, positions, velocities


def _planar_initial_data():
    masses = np.array([1.0, 0.7, 1.4])
    positions = np.array(
        [
            [0.8, -0.2],
            [-0.4, 0.6],
            [0.1, -0.5],
        ]
    )
    velocities = np.array(
        [
            [0.05, 0.11],
            [-0.07, 0.03],
            [0.02, -0.08],
        ]
    )
    return masses, positions, velocities


def _spatial_ks_close_binary_initial_data():
    masses = np.array([0.8, 1.2, 1.7])
    pair_mass = masses[0] + masses[1]
    initial = SpatialKSBinaryChartState(
        masses=masses,
        pair=(0, 1),
        u=np.zeros(4),
        u_velocity=np.array([np.sqrt(pair_mass / 2.0), 0.0, 0.0, 0.0]),
        pair_energy=-0.3,
        binary_center=np.array([0.0, 0.0, 0.0]),
        binary_center_velocity=np.array([0.2, -0.1, 0.03]),
        third_offset=np.array([1.5, 0.25, -0.35]),
        third_offset_velocity=np.array([-0.03, 0.07, 0.02]),
    )
    pre_collision = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    positions, velocities = ks_binary_chart_to_spatial(pre_collision.state_at(-0.04))
    return masses, positions, velocities


def _state_box(positions, velocities, half_width=0.0):
    state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
    return tuple((float(value - half_width), float(value + half_width)) for value in state)


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


def _spatial_ordinary_ks_handoff_data():
    masses = np.array([1.0, 1.2, 1.5])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.02, 0.01],
            [2.5, 0.8, 0.5],
        ]
    )
    velocities = np.array(
        [
            [0.03, 0.0, 0.0],
            [-0.03, -0.001, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    probe = validated_atlas_from_spatial_ordinary_ks_handoff(
        _state_box(positions, velocities),
        masses,
        pair=(0, 1),
        enter_distance=1.0,
        entry_time_upper=10.0,
        branch="positive_x",
        s_endpoint=1.0e-4,
        ordinary_step_size=1.0e-5,
        retained_order=8,
        guard_order=4,
    )
    atlas = validated_atlas_from_spatial_ordinary_ks_handoff(
        _state_box(positions, velocities),
        masses,
        pair=(0, 1),
        enter_distance=1.0,
        entry_time_upper=10.0,
        branch="positive_x",
        s_endpoint=1.0e-4,
        target_time=probe.target_time,
        retained_order=8,
        guard_order=4,
    )
    return masses, positions, velocities, atlas


def _spatial_close_pair_branch_union_atlas_data():
    masses = np.array([1.0, 1.2, 1.5])
    state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.02, 0.18),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.20, 0.20),
        (0.0, 0.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 9
    partition = certify_simultaneous_close_pair_partition(
        state_interval,
        binary_distance_threshold=0.05,
        max_depth=5,
    )
    atlas = validated_atlas_from_spatial_close_pair_branch_partition(
        partition,
        masses,
        target_time=1.0e-6,
        s_endpoint=1.0e-4,
        retained_order=8,
        guard_order=2,
    )
    initial_state = np.asarray(atlas.evaluation.initial_state, dtype=float).reshape(2, 3, 3)
    return masses, initial_state[0], initial_state[1], partition, atlas


def _spatial_two_ks_competing_handoff_atlas():
    masses = np.array([0.8, 1.2, 1.7])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.03, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-100.0, 0.0, 0.0],
        ]
    )
    retained_order = 20
    guard_order = 8
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    initial_interval = interval_spatial_ks_binary_chart_state_from_point(initial)
    first_point = construct_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    first_interval = construct_interval_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    first_event = certify_spatial_ks_competing_binary_entry_event(
        first_interval,
        pair=(0, 2),
        enter_distance=0.025,
        s_upper=0.02,
        coefficient_count=retained_order,
    )
    event_positions, event_velocities = ks_binary_chart_to_spatial(
        first_point.state_at(first_event.root),
    )
    next_point = spatial_to_ks_binary_chart(
        event_positions,
        event_velocities,
        masses,
        pair=(0, 2),
    )
    target_s = 1.0e-4
    next_point_solution = construct_spatial_ks_binary_taylor_solution(
        next_point,
        order=retained_order + guard_order,
    )
    target_time = (
        first_point.physical_time_at(first_event.root)
        + next_point_solution.physical_time_at(target_s)
    )
    atlas = validated_atlas_from_spatial_ks_competing_binary_handoff(
        initial_interval,
        competing_pair=(0, 2),
        enter_distance=0.025,
        entry_s_upper=0.02,
        branch="positive_x",
        next_s_endpoint=2.0e-4,
        target_time_after_ks_start_interval=FloatInterval.point(target_time),
        retained_order=retained_order,
        guard_order=guard_order,
    )
    return atlas


def _solver_options():
    return {
        "initial_radius": 1.0e-15,
        "order": 10,
        "sundman_rate": 1.15,
        "max_compact_step": 0.025,
        "radius_fraction": 0.2,
        "guard_order": 6,
        "target_bisections": 42,
    }


def _parabolic_homothetic_total_collision_atlas():
    masses = np.ones(3)
    central_shape = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ]
    )
    acceleration = accelerations(central_shape, masses)
    central_lambda = -float(
        np.sum(masses[:, None] * central_shape * acceleration)
        / np.sum(masses[:, None] * central_shape * central_shape)
    )
    branch = construct_homothetic_total_collision_branch(
        central_shape,
        central_lambda,
        masses,
        energy_per_inertia=0.0,
        order=8,
    )
    atlas = validated_atlas_from_parabolic_homothetic_total_collision_branch(
        branch,
        start_tau=-0.04,
        target_tau=0.04,
        tolerance=1.0e-8,
    )
    positions, velocities = split_state(atlas.evaluation.initial_state)
    return masses, positions, velocities, atlas


def _nonzero_energy_homothetic_total_collision_atlas():
    masses = np.ones(3)
    central_shape = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ]
    )
    acceleration = accelerations(central_shape, masses)
    central_lambda = -float(
        np.sum(masses[:, None] * central_shape * acceleration)
        / np.sum(masses[:, None] * central_shape * central_shape)
    )
    branch = construct_homothetic_total_collision_branch(
        central_shape,
        central_lambda,
        masses,
        energy_per_inertia=0.02,
        order=8,
    )
    atlas = validated_atlas_from_homothetic_total_collision_branch(
        branch,
        start_tau=-0.04,
        target_tau=0.04,
        tolerance=1.0e-8,
    )
    positions, velocities = split_state(atlas.evaluation.initial_state)
    return masses, positions, velocities, atlas


def test_finite_target_atlas_or_stop_certifies_reached_target_theorem():
    masses, positions, velocities = _spatial_initial_data()
    theorem = construct_finite_target_atlas_or_stop(
        masses,
        positions,
        velocities,
        1.0e-4,
        **_solver_options(),
    )

    assert theorem.certified
    assert theorem.outcome_id == "finite_atlas_reaches_target"
    assert theorem.proof_grade_response_certified
    assert theorem.total_collision_policy.policy_id == "maximal_classical_stop"
    assert theorem.finite_time_classification.certified
    assert theorem.finite_atlas_certificate.certified
    assert theorem.finite_atlas_certificate.total_collision_chart_count == 0
    assert theorem.validated_atlas.proof_certified
    assert "compactified_sundman_target" in theorem.chart_types
    assert theorem.missing_obligations == ()
    assert theorem.obstruction_obligations == ()
    assert theorem.painleve_certificate.certified
    assert theorem.binary_regularization_certificate.certified
    assert theorem.binary_accumulation_certificate.certified
    assert theorem.compact_collision_free_cover_certificate.certified


def test_open_time_locally_finite_atlas_theorem_uses_compact_exhaustion_not_endpoint_partition():
    masses, positions, velocities = _spatial_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=2,
        **_solver_options(),
    )

    assert theorem.checked_prefix_certified
    assert not theorem.certified
    assert theorem.theorem_id == "open_time_locally_finite_atlas"
    assert not theorem.endpoint_regime_partition_required
    assert theorem.finite_target_certificate.certified
    assert not theorem.finite_target_certificate.proof_certified
    assert theorem.compact_interval_certificate.certified
    assert not theorem.compact_interval_certificate.proof_certified
    assert theorem.compact_interval_certificate.interval_lower == -1.0e-4
    assert theorem.compact_interval_certificate.interval_upper == 1.0e-4
    assert theorem.compact_interval_certificate.past_target_certificate.certified
    assert theorem.compact_interval_certificate.future_target_certificate.certified
    assert theorem.exhaustion_family_certificate.certified
    assert not theorem.exhaustion_family_certificate.proof_certified
    assert not theorem.countable_exhaustion_certificate.proof_certified
    assert not theorem.proof_certified
    assert theorem.finite_target_completeness_certificate is not None
    assert not theorem.arbitrary_finite_target_completeness_certified
    completeness = theorem.finite_target_completeness_certificate
    assert completeness.statement.startswith(
        "For every positive-mass noncollision initial state"
    )
    assert "Painleve reduces finite singularities to collisions" in (
        completeness.proof_sketch
    )
    assert "finite_target_analytic_reduction_lemmas" not in (
        completeness.missing_obligations
    )
    assert completeness.pointwise_completeness_theorem.statement_certified
    assert completeness.pointwise_completeness_theorem.certified
    assert "total_collision_stop_chart_existence" not in (
        completeness.missing_obligations
    )
    assert "total_collision_requires_zero_angular_momentum" not in (
        completeness.missing_obligations
    )
    assert "total_collision_central_configuration_asymptotic" not in (
        completeness.missing_obligations
    )
    assert "cubic_time_total_collision_scaling" not in (
        completeness.missing_obligations
    )
    assert "finite_fuchsian_log_stop_chart_for_admissible_entry_data" not in (
        completeness.missing_obligations
    )
    assert "homothetic_total_collision_stop_chart_existence" not in (
        completeness.missing_obligations
    )
    assert "binary_degenerate_total_collision_exclusion" not in (
        completeness.missing_obligations
    )
    assert "reduced_hyperbolic_total_collision_entry" not in (
        completeness.missing_obligations
    )
    assert "poincare_dulac_fuchsian_log_selector_completeness" not in (
        completeness.missing_obligations
    )
    assert "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data" not in (
        completeness.missing_obligations
    )
    assert "arbitrary_total_collision_germ_entry_to_stop_chart" not in (
        completeness.missing_obligations
    )
    assert "finite_chart_chain_concatenation" not in (
        completeness.missing_obligations
    )
    assert "target_or_stop_dichotomy" not in completeness.missing_obligations
    assert not completeness.certificate_search_completeness.certified
    assert "certificate_search_completeness_for_point_inputs" not in (
        completeness.missing_obligations
    )
    assert "fair_adaptive_chart_search" not in completeness.missing_obligations
    assert "finite_target_certificate_search_completeness" not in (
        completeness.missing_obligations
    )
    assert "point_input_finite_target_certificate_search" not in (
        completeness.missing_obligations
    )
    assert "recursive_set_valued_branch_partition_consumption" in (
        completeness.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" in (
        completeness.missing_obligations
    )
    assert "finite_time_loop_budget_elimination" not in (
        completeness.missing_obligations
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.finite_target_completeness_missing_obligations
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert len(theorem.exhaustion_family_certificate.prefix_certificates) == 2
    assert np.allclose(
        theorem.exhaustion_family_certificate.interval_radii,
        (1.0e-4, 2.0e-4),
        rtol=0.0,
        atol=1.0e-18,
    )
    assert (
        theorem.countable_exhaustion_certificate.exhaustion_family_certificate
        is theorem.exhaustion_family_certificate
    )
    assert (
        theorem.compact_interval_certificate.outcome_id
        == "compact_interval_atlas_reaches_both_endpoints"
    )
    assert theorem.countable_exhaustion_certificate.certified
    assert (
        theorem.countable_exhaustion_certificate.compact_interval_certificate
        is theorem.compact_interval_certificate
    )
    assert "global_regime_exhaustion" not in theorem.missing_obligations
    assert "arbitrary_initial_data_partition_theorem" not in theorem.missing_obligations
    assert theorem.route_summary.startswith("compact-interval atlas-or-stop prefix certified")


def test_open_time_reduction_consumes_supplied_search_refinement_evidence():
    masses, positions, velocities = _spatial_initial_data()
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
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_branch_refinement_certificate=(
            branch_refinement
        ),
        certificate_search_event_order_refinement_certificate=event_refinement,
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate

    assert branch_refinement.certified
    assert event_refinement.certified
    assert completeness.certificate_search_completeness.certified
    assert "recursive_set_valued_branch_partition_consumption" not in (
        completeness.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        completeness.missing_obligations
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.finite_target_completeness_missing_obligations
    )
    details = {obligation.obligation: obligation for obligation in completeness.obligations}
    assert "supplied certificate-search refinement evidence" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )
    assert not theorem.certified
    assert not theorem.proof_certified


def test_open_time_reduction_can_certify_uniform_margin_set_valued_subset():
    masses, positions, velocities = _spatial_initial_data()
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
    set_valued = certify_uniform_margin_set_valued_constructor_completeness(
        certify_finite_target_completeness_theorem(dimension=3),
        recursive_branch_refinement_certificate=branch_refinement,
        event_order_refinement_certificate=event_refinement,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_branch_refinement_certificate=(
            branch_refinement
        ),
        certificate_search_event_order_refinement_certificate=event_refinement,
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert branch_refinement.certified
    assert event_refinement.certified
    assert not set_valued.certified
    assert not completeness.certified
    assert not theorem.certified
    assert not theorem.proof_certified
    assert theorem.scoped_set_valued_constructor_only
    assert theorem.proof_certified is False
    assert theorem.set_valued_constructor_input_scope_id == (
        "uniform_margin_set_valued_constructor_branch_event_completeness"
    )
    assert not theorem.set_valued_constructor_arbitrary_partition_generation_claimed
    assert "arbitrary finite-target completeness remains open" in theorem.route_summary
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert not details["set_valued_constructor_branch_event_completeness"].certified
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )
    assert completeness.set_valued_constructor_completeness_certificate is set_valued


def test_open_time_reduction_can_consume_named_validated_set_valued_theorem():
    masses, positions, velocities = _spatial_initial_data()
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
        certify_finite_target_completeness_theorem(dimension=3),
        recursive_branch_refinement_certificate=branch_refinement,
        event_order_refinement_certificate=event_refinement,
    )
    set_valued = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_branch_refinement_certificate=(
            branch_refinement
        ),
        certificate_search_event_order_refinement_certificate=event_refinement,
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert branch_refinement.certified
    assert event_refinement.certified
    assert not scoped.proof_certified
    assert not set_valued.proof_certified
    assert not theorem.certified
    assert theorem.scoped_set_valued_constructor_only
    assert theorem.set_valued_constructor_input_scope_id == (
        "positive_margin_interval_boxes"
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert not details["set_valued_constructor_branch_event_completeness"].certified
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )
    assert completeness.set_valued_constructor_completeness_certificate is set_valued


def test_open_time_reduction_can_consume_affine_halfspace_arrangement_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_oblique_affine_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="sum_event_boundary",
                coefficients=(0.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="difference_event_boundary",
                coefficients=(0.0, 1.0, -1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    child = certify_recursive_stratified_branch_event_consumption(
        _positive_margin_stratified_tree("open-time-oblique-arrangement-child"),
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
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem_certificate,
        arrangement_certificate=arrangement,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    set_valued = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.area_cover_certified
    assert recursive.certified
    assert not scoped.proof_certified
    assert not set_valued.proof_certified
    assert not theorem.certified
    assert not theorem.proof_certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert not details["set_valued_constructor_branch_event_completeness"].certified
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )
    assert completeness.set_valued_constructor_completeness_certificate is set_valued


def test_open_time_reduction_consumes_affine_line_child_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_line_child_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="vertical_event_boundary",
                coefficients=(0.0, 1.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="positive_offset_event_boundary",
                coefficients=(2.0, 0.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    recursive = certify_affine_halfspace_arrangement_recursive_consumption(
        arrangement,
        root_dimension=2,
        root_rank=2,
        child_consumptions=(
            derive_affine_halfspace_arrangement_line_child_consumptions(arrangement)
        ),
    )
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem_certificate,
        arrangement_certificate=arrangement,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    set_valued = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.equality_stratum_count == 1
    assert recursive.strict_descent_edge_count == 1
    assert recursive.proof_certified
    assert not scoped.proof_certified
    assert not set_valued.proof_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_terminal_affine_line_child_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_terminal_line_child_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="diagonal_event_boundary",
                coefficients=(0.0, 1.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
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
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem_certificate,
        arrangement_certificate=arrangement,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    set_valued = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert len(children) == 1
    assert next(iter(children.values())).constructor_source_type == (
        "AffineHalfspaceLineChild"
    )
    assert recursive.proof_certified
    assert not scoped.proof_certified
    assert not set_valued.proof_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_coincident_affine_line_child_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_coincident_line_child_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="first_same_event_boundary",
                coefficients=(0.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="second_same_event_boundary",
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
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem_certificate,
        arrangement_certificate=arrangement,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    set_valued = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.equality_stratum_count == 1
    assert len(children) == 1
    assert next(iter(children.values())).constructor_source_type == (
        "AffineHalfspaceLineChild"
    )
    assert recursive.child_constructor_source_types == ("AffineHalfspaceLineChild",)
    assert recursive.strict_descent_edge_count == 1
    assert recursive.proof_certified
    assert not scoped.proof_certified
    assert not set_valued.proof_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    detail = details["set_valued_constructor_branch_event_completeness"].detail
    assert "arbitrary positive-mass noncollision interval inputs" in detail


def test_open_time_reduction_consumes_affine_point_child_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_point_child_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="sum_event_boundary",
                coefficients=(0.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="difference_event_boundary",
                coefficients=(0.0, 1.0, -1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    point_children = derive_affine_halfspace_arrangement_point_child_consumptions(
        arrangement,
    )
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem_certificate,
        arrangement_certificate=arrangement,
    )
    recursive = scoped.branch_consumption_certificate
    set_valued = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert len(point_children) == 1
    assert recursive.strict_descent_edge_count == arrangement.equality_stratum_count
    assert recursive.proof_certified
    assert not scoped.proof_certified
    assert not set_valued.proof_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_can_consume_spatial_affine_halfspace_arrangement_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_spatial_oblique_affine_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="sum_event_boundary",
                coefficients=(0.0, 1.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="mixed_event_boundary",
                coefficients=(0.0, 1.0, -1.0, 1.0),
            ),
        ),
        domain_box=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.2,
    )
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem_certificate,
        arrangement_certificate=arrangement,
    )
    recursive = scoped.branch_consumption_certificate
    set_valued = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.volume_cover_certified
    assert recursive.certified
    assert "AffineHalfspacePlaneChild" in recursive.child_constructor_source_types
    assert "AffineHalfspaceSpatialLineChild" in recursive.child_constructor_source_types
    assert not scoped.proof_certified
    assert not set_valued.proof_certified
    assert not theorem.certified
    assert not theorem.proof_certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert not details["set_valued_constructor_branch_event_completeness"].certified
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )
    assert completeness.set_valued_constructor_completeness_certificate is set_valued


def test_open_time_reduction_consumes_terminal_spatial_affine_plane_child_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_terminal_spatial_plane_child_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="diagonal_event_plane",
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
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem_certificate,
        arrangement_certificate=arrangement,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    set_valued = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert len(child_consumptions) == 1
    assert next(iter(child_consumptions.values())).constructor_source_type == (
        "AffineHalfspacePlaneChild"
    )
    assert recursive.proof_certified
    assert not scoped.proof_certified
    assert not set_valued.proof_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_coincident_spatial_affine_plane_child_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_coincident_spatial_plane_child_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="first_same_event_plane",
                coefficients=(0.0, 1.0, 1.0, 1.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="second_same_event_plane",
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
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem_certificate,
        arrangement_certificate=arrangement,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    set_valued = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.equality_stratum_count == 1
    assert len(child_consumptions) == 1
    assert next(iter(child_consumptions.values())).constructor_source_type == (
        "AffineHalfspacePlaneChild"
    )
    assert recursive.child_constructor_source_types == ("AffineHalfspacePlaneChild",)
    assert recursive.strict_descent_edge_count == 1
    assert recursive.proof_certified
    assert not scoped.proof_certified
    assert not set_valued.proof_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    detail = details["set_valued_constructor_branch_event_completeness"].detail
    assert "arbitrary positive-mass noncollision interval inputs" in detail


def test_open_time_reduction_reports_spatial_point_child_source():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_spatial_line_point_child_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="x_event_plane",
                coefficients=(0.0, 1.0, 0.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="y_event_plane",
                coefficients=(0.0, 0.0, 1.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="z_event_plane",
                coefficients=(0.0, 0.0, 0.0, 1.0),
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
        child_consumptions=automatic_children,
    )
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem_certificate,
        arrangement_certificate=arrangement,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    set_valued = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}
    detail = details["set_valued_constructor_branch_event_completeness"].detail

    assert "AffineHalfspacePlaneChild" in recursive.child_constructor_source_types
    assert "AffineDecisionArrangement" in recursive.child_constructor_source_types
    assert "AffineHalfspaceSpatialPointChild" in recursive.child_constructor_source_types
    assert recursive.proof_certified
    assert not scoped.proof_certified
    assert not set_valued.proof_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in detail


def test_open_time_reduction_reports_spatial_line_child_source():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_spatial_line_child_arrangement",
        decision_functions=(
            AffineBoxDecisionFunctionSpec(
                decision_id="x_event_plane",
                coefficients=(0.0, 1.0, 0.0, 0.0),
            ),
            AffineBoxDecisionFunctionSpec(
                decision_id="y_event_plane",
                coefficients=(0.0, 0.0, 1.0, 0.0),
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
        child_consumptions=automatic_children,
    )
    scoped = certify_affine_halfspace_arrangement_set_valued_constructor_completeness(
        theorem_certificate,
        arrangement_certificate=arrangement,
        recursive_stratified_branch_consumption_certificate=recursive,
        recursive_stratified_event_order_consumption_certificate=recursive,
    )
    set_valued = certify_validated_set_valued_constructor_completeness_theorem(
        scoped,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}
    detail = details["set_valued_constructor_branch_event_completeness"].detail

    assert "AffineHalfspaceSpatialLineChild" in recursive.child_constructor_source_types
    assert "AffineHalfspacePlaneChild" in recursive.child_constructor_source_types
    assert recursive.proof_certified
    assert not scoped.proof_certified
    assert not set_valued.proof_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in detail


def test_open_time_reduction_can_certify_supplied_recursive_stratified_set_valued_subset():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="event_order_arrangement",
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
        _positive_margin_stratified_tree("open-time-arrangement-child"),
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
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.source_tree.source_type == "AffineDecisionArrangement"
    assert recursive.certified
    assert recursive.recursion_kind == "affine_decision_arrangement"
    assert not set_valued.certified
    assert not completeness.certified
    assert not theorem.certified
    assert not theorem.proof_certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert not details["set_valued_constructor_branch_event_completeness"].certified
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )
    assert completeness.set_valued_constructor_completeness_certificate is set_valued


def test_open_time_reduction_consumes_verified_polynomial_root_bracket_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="verified_polynomial_event_order",
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
        _positive_margin_stratified_tree("open-time-verified-polynomial-child"),
        root_dimension=2,
        root_rank=1,
    )
    equality_leaves = tuple(
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    )
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            leaf.equality_stratum.stratum_id: child
            for leaf in equality_leaves
        },
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.source_tree.source_type == "PolynomialDecisionArrangement"
    assert recursive.recursion_kind == "polynomial_decision_arrangement"
    assert recursive.certified
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_finite_target_reduction_extracts_recursive_evidence_from_validated_set_valued_certificate():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    finite_target = construct_finite_target_atlas_or_stop(
        masses,
        positions,
        velocities,
        1.0e-4,
        **_solver_options(),
    )
    arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="direct_reduction_verified_polynomial_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_total",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    scoped = (
        certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            constructor_certificate=arrangement,
            root_dimension=3,
            root_rank=2,
        )
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(scoped)
    reduction = certify_finite_target_completeness_reduction(
        finite_target_certificate=finite_target,
        set_valued_constructor_completeness_certificate=validated,
    )
    search = reduction.certificate_search_completeness

    assert scoped.branch_consumption_certificate.certified
    assert not scoped.certified
    assert not validated.certified
    assert search.certified
    assert "recursive_set_valued_branch_partition_consumption" not in (
        search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        search.missing_obligations
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        reduction.missing_obligations
    )
    assert not reduction.certified
    assert reduction.set_valued_constructor_completeness_certificate is validated


def test_open_time_reduction_can_consume_constructor_close_pair_partition_scope():
    masses, positions, velocities = _spatial_initial_data()
    _branch_masses, _branch_positions, _branch_velocities, partition, _branch_union_atlas = (
        _spatial_close_pair_branch_union_atlas_data()
    )
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    source_tree = certify_supplied_branch_event_tree(partition)
    stratified = certify_stratified_branch_event_tree(source_tree)
    recursive = certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=3,
        root_rank=2,
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert partition.certified
    assert source_tree.tree_kind == "simultaneous_close_pair_branch_partition"
    assert stratified.leaf_kinds == ("separated_binary_entry", "separated_binary_entry")
    assert recursive.certified
    assert recursive.terminal_leaf_count == 2
    assert not set_valued.proof_certified
    assert not completeness.certified
    assert not theorem.certified
    assert not theorem.proof_certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert not details["set_valued_constructor_branch_event_completeness"].certified
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_axis_aligned_affine_box_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_box_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_affine_box_event_order",
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
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.source_tree.source_type == "AxisAlignedAffineBoxArrangement"
    assert recursive.recursion_kind == "axis_aligned_affine_box_decision_arrangement"
    assert recursive.certified
    assert recursive.child_constructor_source_types == ("AxisAlignedAffineBoxChild",)
    assert not set_valued.certified
    assert set_valued.event_order_consumption_certificate is recursive
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_affine_halfspace_decision_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    stratification = certify_affine_halfspace_decision_stratified_branch_event_tree(
        decision_id="open_time_oblique_event_boundary",
        coefficients=(0.0, 1.0, 1.0),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )
    recursive = certify_affine_halfspace_decision_recursive_consumption(
        stratification,
        root_dimension=2,
        root_rank=2,
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert stratification.source_tree.source_type == "AffineHalfspaceDecision"
    assert recursive.recursion_kind == "affine_halfspace_decision"
    assert recursive.certified
    assert recursive.child_constructor_source_types == ("AffineHalfspaceDecisionChild",)
    assert not set_valued.certified
    assert set_valued.event_order_consumption_certificate is recursive
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_polynomial_decision_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    stratification = certify_polynomial_decision_stratified_branch_event_tree(
        decision_id="open_time_event_tie_discriminant",
        coefficients=(0.0, 1.0),
        domain=(-1.0, 1.0),
        root_brackets=((-0.01, 0.01),),
    )
    set_valued = (
        certify_constructor_derived_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            constructor_certificate=stratification,
            root_dimension=3,
            root_rank=1,
        )
    )
    recursive = set_valued.branch_consumption_certificate
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert stratification.source_tree.source_type == "PolynomialDecisionStratification"
    assert recursive.recursion_kind == "polynomial_decision_stratification"
    assert recursive.certified
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    # The polynomial partition is locally certified, but does not prove that
    # the constructor covers arbitrary admissible three-body inputs.
    assert not set_valued.certified
    assert set_valued.event_order_consumption_certificate is recursive
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_derives_constructor_search_scope_internally():
    masses, positions, velocities = _spatial_initial_data()
    stratification = certify_polynomial_decision_stratified_branch_event_tree(
        decision_id="open_time_internal_event_tie_discriminant",
        coefficients=(0.0, 1.0),
        domain=(-1.0, 1.0),
        root_brackets=((-0.01, 0.01),),
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_constructor_certificate=stratification,
        certificate_search_constructor_root_dimension=3,
        certificate_search_constructor_root_rank=1,
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    set_valued = completeness.set_valued_constructor_completeness_certificate
    recursive = set_valued.branch_consumption_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert recursive.recursion_kind == "polynomial_decision_stratification"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert set_valued.event_order_consumption_certificate is recursive
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_mixed_constructor_branch_event_scope():
    masses, positions, velocities = _spatial_initial_data()
    branch_stratification = certify_affine_halfspace_decision_stratified_branch_event_tree(
        decision_id="open_time_branch_oblique_boundary",
        coefficients=(0.0, 1.0, 1.0),
        domain_box=((-1.0, 1.0), (-1.0, 1.0)),
        slab_half_width=0.25,
    )
    event_arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_event_polynomial_arrangement",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_target",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_branch_constructor_certificate=branch_stratification,
        certificate_search_event_order_constructor_certificate=event_arrangement,
        certificate_search_branch_constructor_root_dimension=2,
        certificate_search_branch_constructor_root_rank=2,
        certificate_search_event_order_constructor_root_dimension=3,
        certificate_search_event_order_constructor_root_rank=2,
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    set_valued = completeness.set_valued_constructor_completeness_certificate
    branch_recursive = set_valued.branch_consumption_certificate
    event_recursive = set_valued.event_order_consumption_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert branch_recursive.certified
    assert event_recursive.certified
    assert branch_recursive.child_constructor_source_types == (
        "AffineHalfspaceDecisionChild",
    )
    assert event_recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_mixed_oblique_affine_arrangement_scope():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    branch_arrangement = certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
        arrangement_id="open_time_mixed_spatial_oblique_branch",
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
        arrangement_id="open_time_mixed_affine_event",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="binary_minus_target",
                coefficients=(0.0, 1.0),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    set_valued = (
        certify_constructor_pair_derived_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            branch_constructor_certificate=branch_arrangement,
            event_order_constructor_certificate=event_arrangement,
            branch_root_dimension=3,
            branch_root_rank=2,
            event_order_root_dimension=3,
            event_order_root_rank=2,
        )
    )
    branch_recursive = set_valued.branch_consumption_certificate
    event_recursive = set_valued.event_order_consumption_certificate
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        set_valued,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            branch_recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            event_recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            validated
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert branch_arrangement.source_tree.source_type == "AffineHalfspace3DArrangement"
    assert branch_arrangement.volume_cover_certified
    assert event_arrangement.source_tree.source_type == "AffineDecisionArrangement"
    assert "AffineHalfspacePlaneChild" in (
        branch_recursive.child_constructor_source_types
    )
    assert event_recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert validated.input_scope_id == (
        "finite_mixed_constructor_branch_event_interval_boxes"
    )
    assert not set_valued.certified
    assert not validated.proof_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_simultaneous_affine_equality_stratum():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="simultaneous_event_order",
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
        _positive_margin_stratified_tree("open-time-simultaneous-child"),
        root_dimension=2,
        root_rank=1,
    )
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            "simultaneous_event_order:root:0:binary_minus_total+ks_exit_minus_target": (
                child
            ),
        },
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.equality_stratum_count == 1
    assert arrangement.source_tree.source_type == "AffineDecisionArrangement"
    assert recursive.recursion_kind == "affine_decision_arrangement"
    assert recursive.recursive_leaf_count == 1
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert not details["set_valued_constructor_branch_event_completeness"].certified
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )
    assert completeness.set_valued_constructor_completeness_certificate is set_valued


def test_open_time_reduction_consumes_boundary_affine_equality_stratum():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
    arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="boundary_event_order",
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
        _positive_margin_stratified_tree("open-time-boundary-child"),
        root_dimension=2,
        root_rank=1,
    )
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions={
            "boundary_event_order:root:0:target_starts_on_event_boundary": child,
            "boundary_event_order:root:1:interior_binary_minus_target": child,
        },
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )

    assert arrangement.decision_functions[0].root_brackets[0][0] == -1.0
    assert recursive.certified
    # Boundary-stratum consumption is valid local evidence, but composing it
    # with the unaudited finite-target theorem does not establish the
    # arbitrary-input constructor theorem.
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_open_time_reduction_consumes_quadratic_simple_root_equality_strata():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
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
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.equality_stratum_count == 2
    assert recursive.certified
    assert recursive.recursion_kind == "polynomial_decision_arrangement"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_quadratic_double_root_tangent_stratum():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
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
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.equality_stratum_count == 1
    assert arrangement.strata[1].stratum_kind == "quadratic_double_equality_root"
    assert recursive.certified
    assert recursive.recursion_kind == "quadratic_double_root_decision_arrangement"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_coincident_quadratic_simple_root_stratum():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
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
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_polynomial_decision_arrangement_child_consumptions(
            arrangement,
        ),
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.equality_stratum_count == 3
    assert recursive.certified
    assert recursive.recursion_kind == "computed_polynomial_root_decision_arrangement"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_mixed_quadratic_multiple_root_stratum():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
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
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_polynomial_decision_arrangement_child_consumptions(
            arrangement,
        ),
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    multiple = next(
        stratum
        for stratum in arrangement.strata
        if stratum.stratum_kind == "simultaneous_polynomial_multiple_equality_root"
    )

    assert arrangement.equality_stratum_count == 2
    assert multiple.second_derivative_interval[0] > 0.0
    assert recursive.certified
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_open_time_reduction_consumes_sturm_cubic_root_strata():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
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
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_polynomial_decision_arrangement_child_consumptions(
            arrangement,
        ),
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )
    completeness = theorem.finite_target_completeness_certificate
    details = {obligation.obligation: obligation for obligation in completeness.obligations}

    assert arrangement.equality_stratum_count == 3
    assert arrangement.sign_stratum_count == 4
    assert recursive.certified
    assert recursive.recursion_kind == "sturm_polynomial_decision_arrangement"
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary positive-mass noncollision interval inputs" in (
        details["set_valued_constructor_branch_event_completeness"].detail
    )


def test_open_time_reduction_consumes_sturm_quartic_root_strata():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
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
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_polynomial_decision_arrangement_child_consumptions(
            arrangement,
        ),
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )

    assert arrangement.equality_stratum_count == 4
    assert arrangement.sign_stratum_count == 5
    assert recursive.certified
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_open_time_reduction_consumes_sturm_boundary_root_strata():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
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
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_polynomial_decision_arrangement_child_consumptions(
            arrangement,
        ),
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )

    assert arrangement.equality_stratum_count == 2
    assert arrangement.decision_functions[0].root_brackets[0][0] == -1.0
    assert recursive.certified
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_open_time_reduction_consumes_sturm_coincident_root_strata():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
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
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_polynomial_decision_arrangement_child_consumptions(
            arrangement,
        ),
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )

    assert arrangement.equality_stratum_count == 3
    assert any(
        stratum.stratum_kind == "simultaneous_polynomial_equality_root"
        for stratum in arrangement.strata
    )
    assert recursive.certified
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_open_time_reduction_consumes_sturm_multiple_root_strata():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
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
    recursive = certify_polynomial_decision_arrangement_recursive_consumption(
        arrangement,
        root_dimension=3,
        root_rank=2,
        child_consumptions=derive_polynomial_decision_arrangement_child_consumptions(
            arrangement,
        ),
    )
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )

    assert arrangement.equality_stratum_count == 1
    assert arrangement.strata[1].root_multiplicities == (2,)
    assert recursive.certified
    assert recursive.child_constructor_source_types == ("PolynomialRootChild",)
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_open_time_reduction_consumes_terminal_selector_policy_strata():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
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
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_recursive_stratified_branch_consumption_certificate=(
            recursive
        ),
        certificate_search_recursive_stratified_event_order_consumption_certificate=(
            recursive
        ),
        certificate_search_set_valued_constructor_completeness_certificate=(
            set_valued
        ),
        **_solver_options(),
    )

    assert stratified.proof_certified
    assert recursive.certified
    assert recursive.terminal_leaf_count == 1
    assert not set_valued.certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_terminal_policy_stratification_rejects_stale_or_duplicate_leaf_evidence():
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
    selector = SelectorPolicyLeafCertificate(
        leaf_id="selector:identity",
        selector_policy_id="selected_identity_selector",
        defining_function_ids=("selector_boundary",),
        isolation_certified=True,
        certified=True,
    )
    stale_selector = SelectorPolicyLeafCertificate(
        leaf_id="selector:stale",
        selector_policy_id="selected_identity_selector",
        defining_function_ids=("selector_boundary",),
        isolation_certified=True,
        certified=True,
    )
    overlapping_cluster = TotalCollisionClusterLeafCertificate(
        leaf_id="selector:identity",
        cluster_pair_ids=("pair:0-1", "pair:0-2"),
        stop_or_selector_policy="maximal_classical_stop",
        entry_certificate=SimpleNamespace(proof_certified=True),
        certified=True,
    )

    with pytest.raises(ValueError, match="duplicate selector policy"):
        certify_terminal_policy_stratified_branch_event_tree(
            source_tree,
            selector_policies=(selector, selector),
        )
    with pytest.raises(ValueError, match="unknown source leaves"):
        certify_terminal_policy_stratified_branch_event_tree(
            source_tree,
            selector_policies=(stale_selector,),
        )
    with pytest.raises(ValueError, match="both selector and total cluster"):
        certify_terminal_policy_stratified_branch_event_tree(
            source_tree,
            selector_policies=(selector,),
            total_collision_clusters=(overlapping_cluster,),
        )


def test_branch_event_tree_rejects_duplicate_leaf_ids_before_policy_attachment():
    duplicate_tree = certify_supplied_branch_event_tree(
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
                SimpleNamespace(
                    branch_id="selector:identity",
                    leaf_type="selector_policy_leaf",
                    decision="selected_identity_selector",
                    certified=True,
                ),
            ),
        )
    )
    selector = SelectorPolicyLeafCertificate(
        leaf_id="selector:identity",
        selector_policy_id="selected_identity_selector",
        defining_function_ids=("selector_boundary",),
        isolation_certified=True,
        certified=True,
    )

    assert not duplicate_tree.certified
    assert "finite_branch_event_tree_leaf_ids_unique" in (
        duplicate_tree.missing_obligations
    )
    with pytest.raises(ValueError, match="duplicate leaf ids"):
        certify_terminal_policy_stratified_branch_event_tree(
            duplicate_tree,
            selector_policies=(selector,),
        )


def test_recursive_child_consumption_rejects_duplicate_and_alias_colliding_keys():
    arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id="alias_collision_event_order",
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id="linear_threshold",
                coefficients=(0.0, 1.0),
                root_brackets=((-0.01, 0.01),),
            ),
        ),
        domain=(-1.0, 1.0),
    )
    equality_leaf = next(
        leaf
        for leaf in arrangement.stratified_tree.leaf_certificates
        if leaf.equality_stratum is not None
    )
    child = derive_polynomial_decision_arrangement_child_consumptions(
        arrangement,
    )[equality_leaf.equality_stratum.stratum_id]

    with pytest.raises(ValueError, match="duplicate child consumption ids"):
        certify_polynomial_decision_arrangement_recursive_consumption(
            arrangement,
            root_dimension=3,
            root_rank=2,
            child_consumptions=(
                (equality_leaf.leaf_id, child),
                (equality_leaf.leaf_id, child),
            ),
        )
    with pytest.raises(ValueError, match="aliases collide"):
        certify_polynomial_decision_arrangement_recursive_consumption(
            arrangement,
            root_dimension=3,
            root_rank=2,
            child_consumptions=(
                (equality_leaf.equality_stratum.stratum_id, child),
                (equality_leaf.source_leaf_id, child),
            ),
        )


def test_terminal_policy_stratification_rejects_incompatible_source_leaf_kind():
    source_tree = certify_supplied_branch_event_tree(
        SimpleNamespace(
            certified=True,
            recursive_bisection_cover_certified=True,
            branch_cover_certified=True,
            branches=(
                SimpleNamespace(
                    branch_id="ordinary:positive",
                    leaf_type="positive_margin_unique_event_leaf",
                    decision="ordinary_chart_reaches_target",
                    certified=True,
                ),
            ),
        )
    )
    selector = SelectorPolicyLeafCertificate(
        leaf_id="ordinary:positive",
        selector_policy_id="selected_identity_selector",
        defining_function_ids=("selector_boundary",),
        isolation_certified=True,
        certified=True,
    )
    cluster = TotalCollisionClusterLeafCertificate(
        leaf_id="ordinary:positive",
        cluster_pair_ids=("pair:0-1", "pair:0-2"),
        stop_or_selector_policy="maximal_classical_stop",
        entry_certificate=SimpleNamespace(proof_certified=True),
        certified=True,
    )

    with pytest.raises(ValueError, match="selector source leaves"):
        certify_terminal_policy_stratified_branch_event_tree(
            source_tree,
            selector_policies=(selector,),
        )
    with pytest.raises(ValueError, match="total-collision source leaves"):
        certify_terminal_policy_stratified_branch_event_tree(
            source_tree,
            total_collision_clusters=(cluster,),
        )


def test_open_time_reduction_consumes_terminal_total_collision_cluster_strata_without_proof_promotion():
    masses, positions, velocities = _spatial_initial_data()
    theorem_certificate = certify_finite_target_completeness_theorem(dimension=3)
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
    set_valued = (
        certify_supplied_recursive_stratified_set_valued_constructor_completeness(
            theorem_certificate,
            recursive_stratified_branch_consumption_certificate=recursive,
            recursive_stratified_event_order_consumption_certificate=recursive,
        )
    )
    validated = certify_validated_set_valued_constructor_completeness_theorem(
        set_valued,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        certificate_search_set_valued_constructor_completeness_certificate=(
            validated
        ),
        **_solver_options(),
    )

    assert stratified.proof_certified
    assert stratified.leaf_kinds == ("total_collision_cluster",)
    assert recursive.certified
    assert recursive.terminal_leaf_count == 1
    assert not set_valued.certified
    assert not validated.proof_certified
    assert not theorem.certified
    assert not theorem.proof_certified
    assert theorem.scoped_set_valued_constructor_only
    assert theorem.set_valued_constructor_input_scope_id == (
        "supplied_recursive_stratified_interval_boxes"
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )
    assert "arbitrary finite-target completeness remains open" in theorem.route_summary


def test_pointwise_open_time_theorem_derives_compact_exhaustion_without_set_valued_claim():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )

    assert theorem.theorem_id == "pointwise_open_time_locally_finite_atlas"
    assert theorem.certified
    assert theorem.finite_target_theorem.certified
    assert not theorem.finite_target_theorem.proof_certified
    assert not theorem.finite_target_reduction_certificate.proof_certified
    assert theorem.finite_target_reduction_certificate.proof_mode == (
        "internal_two_sided_finite_target_compact_interval_reduction"
    )
    assert not theorem.local_finiteness_certificate.proof_certified
    assert theorem.local_finiteness_certificate.proof_mode == (
        "internal_countable_nested_compact_exhaustion_local_finiteness"
    )
    assert not theorem.proof_certified
    assert theorem.compact_time_certificate.certified
    assert not theorem.endpoint_regime_partition_required
    assert "pointwise_open_time_finite_target_theorem_proof" in (
        theorem.missing_obligations
    )
    assert "binary_degenerate_total_collision_exclusion" in (
        theorem.missing_obligations
    )
    assert (
        "audited_or_machine_checked:"
        "finite_target_theorem_reduces_compact_interval_exhaustion"
    ) in theorem.missing_obligations
    assert (
        "audited_or_machine_checked:"
        "countable_nested_compact_interval_local_finiteness"
    ) in theorem.missing_obligations
    assert "recursive_set_valued_branch_partition_consumption" not in (
        theorem.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        theorem.missing_obligations
    )
    assert "set-valued branch-recursion backend" in theorem.proof_sketch
    details = {obligation.obligation: obligation for obligation in theorem.obligations}
    assert details["set_valued_constructor_not_claimed"].certified
    assert "interval-box branch recursion remains a separate" in (
        details["set_valued_constructor_not_claimed"].detail
    )
    assert theorem.route_summary.startswith("pointwise open-time")


def test_pointwise_open_time_theorem_requires_maximal_classical_policy():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="selected_identity_selector",
    )

    assert not theorem.certified
    assert not theorem.proof_certified
    assert "maximal_classical_total_collision_policy" in theorem.missing_obligations
    assert "pointwise_finite_target_atlas_or_stop_completeness" in (
        theorem.missing_obligations
    )
    assert "maximal_classical_total_collision_policy" in (
        theorem.finite_target_theorem.missing_obligations
    )


def test_pointwise_open_time_theorem_rejects_forged_scope_parameters():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    forged_theorem_id = replace(
        theorem,
        theorem_id="spoofed_open_time_theorem",
    )
    forged_endpoint_scope = replace(
        theorem,
        endpoint_regime_partition_required=True,
    )

    assert theorem.certified
    assert not theorem.proof_certified
    assert "pointwise_open_time_theorem_id" in (
        forged_theorem_id.missing_obligations
    )
    assert "endpoint_regime_partition_not_required" in (
        forged_endpoint_scope.missing_obligations
    )
    for forged in (forged_theorem_id, forged_endpoint_scope):
        assert not forged.certified
        assert not forged.proof_certified


def test_pointwise_open_time_theorem_rejects_attribute_compatible_nested_components():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    spoofed = replace(
        theorem,
        compact_time_certificate=SimpleNamespace(
            certified=True,
            proof_certified=True,
        ),
        finite_target_theorem=SimpleNamespace(
            certified=True,
            proof_certified=True,
            missing_obligations=(),
            unaudited_analytic_lemma_ids=(),
            critical_unaudited_analytic_lemma_ids=(),
            analytic_lemma_audit_blockers=(),
        ),
        finite_target_reduction_certificate=SimpleNamespace(
            certified=True,
            proof_certified=True,
        ),
        local_finiteness_certificate=SimpleNamespace(
            certified=True,
            proof_certified=True,
        ),
    )

    assert theorem.component_types_certified
    assert theorem.certified
    assert not theorem.proof_certified
    assert not spoofed.component_types_certified
    assert not spoofed.certified
    assert not spoofed.proof_certified
    assert "pointwise_open_time_compact_time_type" in spoofed.missing_obligations
    assert "pointwise_open_time_finite_target_theorem_type" in (
        spoofed.missing_obligations
    )
    assert "pointwise_open_time_finite_target_reduction_type" in (
        spoofed.missing_obligations
    )
    assert "pointwise_open_time_local_finiteness_type" in (
        spoofed.missing_obligations
    )
    assert spoofed.unaudited_analytic_lemma_ids == ()
    assert spoofed.critical_unaudited_analytic_lemma_ids == ()
    assert spoofed.analytic_lemma_audit_blockers == ()


def test_pointwise_open_time_theorem_rejects_stale_finite_target_theorem_source():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    stale_finite_target = certify_finite_target_completeness_theorem(
        dimension=2,
        total_collision_policy_id="maximal_classical_stop",
    )
    stale = replace(theorem, finite_target_theorem=stale_finite_target)

    assert theorem.finite_target_theorem_source_matches
    assert stale_finite_target.certified
    assert not stale_finite_target.proof_certified
    assert not stale.finite_target_theorem_source_matches
    assert not stale.certified
    assert not stale.proof_certified
    assert "pointwise_open_time_finite_target_theorem_source_match" in (
        stale.missing_obligations
    )


def test_open_time_theorem_surfaces_reject_spoofed_obligation_ledgers():
    masses, positions, velocities = _spatial_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        **_solver_options(),
    )
    fake_obligation = SimpleNamespace(
        obligation="attribute_compatible_fake",
        certified=True,
        required=True,
    )
    optional_only = TheoremPipelineObligation(
        obligation="optional_only",
        certified=True,
        source="test",
        required=False,
    )

    spoofed_finite_target = replace(
        theorem.finite_target_certificate,
        obligations=(fake_obligation,),
    )
    optional_finite_target = replace(
        theorem.finite_target_certificate,
        obligations=(optional_only,),
    )
    spoofed_reduction = replace(
        theorem.finite_target_completeness_certificate,
        obligations=(fake_obligation,),
    )
    spoofed_theorem = replace(theorem, obligations=(fake_obligation,))
    spoofed_nested_theorem = replace(
        theorem,
        finite_target_certificate=SimpleNamespace(
            certified=True,
            proof_certified=True,
            outcome_id=theorem.finite_target_certificate.outcome_id,
        ),
        countable_exhaustion_certificate=SimpleNamespace(
            certified=True,
            proof_certified=True,
        ),
        finite_target_completeness_certificate=SimpleNamespace(
            certified=True,
            proof_certified=True,
            missing_obligations=(),
        ),
        independent_chart_verifier_certificate=SimpleNamespace(
            certified=True,
            proof_grade_arithmetic_checked_bundle_certified=True,
        ),
        independent_chart_verifier_certified=True,
    )
    pointwise = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    spoofed_pointwise = replace(pointwise, obligations=(fake_obligation,))

    assert theorem.checked_prefix_certified
    assert theorem.finite_target_certificate.certified
    assert not spoofed_finite_target.certified
    assert "finite_target_atlas_or_stop_obligation_type" in (
        spoofed_finite_target.missing_obligations
    )
    assert not optional_finite_target.certified
    assert "finite_target_atlas_or_stop_required_obligation_present" in (
        optional_finite_target.missing_obligations
    )
    assert not spoofed_reduction.certified
    assert "finite_target_completeness_reduction_obligation_type" in (
        spoofed_reduction.missing_obligations
    )
    assert not spoofed_theorem.checked_prefix_certified
    assert "open_time_locally_finite_atlas_obligation_type" in (
        spoofed_theorem.missing_obligations
    )
    assert not spoofed_nested_theorem.component_types_certified
    assert not spoofed_nested_theorem.certified
    assert not spoofed_nested_theorem.proof_certified
    assert "open_time_finite_target_certificate_type" in (
        spoofed_nested_theorem.missing_obligations
    )
    assert "open_time_countable_exhaustion_certificate_type" in (
        spoofed_nested_theorem.missing_obligations
    )
    assert "open_time_finite_target_completeness_certificate_type" in (
        spoofed_nested_theorem.missing_obligations
    )
    assert "open_time_independent_chart_verifier_certificate_type" in (
        spoofed_nested_theorem.missing_obligations
    )
    assert pointwise.certified
    assert not spoofed_pointwise.certified
    assert "pointwise_open_time_locally_finite_atlas_obligation_type" in (
        spoofed_pointwise.missing_obligations
    )


def test_total_collision_policy_certificate_requires_exact_policy_flags():
    truthy_stop = TotalCollisionPolicyCertificate(
        policy_id="maximal_classical_stop",
        stop_at_unselected_total_collision="yes",
        selected_continuation_allowed=False,
    )
    mismatched_selected = TotalCollisionPolicyCertificate(
        policy_id="selected_identity_selector",
        stop_at_unselected_total_collision=True,
        selected_continuation_allowed=True,
        selector_policy_id="selected_identity_selector",
    )
    selected_without_selector = TotalCollisionPolicyCertificate(
        policy_id="selected_identity_selector",
        stop_at_unselected_total_collision=False,
        selected_continuation_allowed=True,
    )

    assert not truthy_stop.certified
    assert not truthy_stop.proof_certified
    assert not mismatched_selected.certified
    assert not selected_without_selector.certified


def test_constructor_attaches_independent_ordinary_checked_prefix_without_overclaiming():
    masses, positions, velocities = _spatial_initial_data()
    verifier = construct_independent_ordinary_taylor_checked_prefix(
        masses,
        positions,
        velocities,
        1.0e-4,
        order=10,
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        checked_prefix_strategy="ordinary_taylor",
        **_solver_options(),
    )

    assert verifier.certified
    assert verifier.ordinary_taylor_chart_count == 1
    assert verifier.checked_chart_chain_count == 1
    assert theorem.independent_chart_verifier_certified
    assert theorem.independent_chart_verifier_certificate.certified
    assert (
        theorem.independent_chart_verifier_certificate.checked_chart_chain_count
        == 1
    )
    assert theorem.independent_chart_verifier_arithmetic_certified
    assert theorem.independent_chart_verifier_arithmetic_blockers == ()
    assert theorem.checked_prefix_certified
    assert not theorem.certified
    assert not theorem.proof_certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_open_time_can_attach_independent_checked_finite_planar_atlas():
    masses, positions, velocities = _planar_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-3,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        initial_radius=1.0e-15,
        order=10,
        max_compact_step=1.0e-3,
        binary_distance_threshold=0.05,
        target_bisections=42,
        checked_prefix_strategy="finite_target_atlas",
    )

    assert theorem.finite_target_certificate.certified
    assert theorem.finite_target_certificate.finite_time_classification.selected_route_id == (
        "planar_hybrid"
    )
    assert theorem.independent_chart_verifier_certified
    assert theorem.independent_chart_verifier_certificate.certified
    assert theorem.independent_chart_verifier_certificate.checked_chart_chain_count == 1
    assert theorem.independent_chart_verifier_certificate.ordinary_taylor_chart_count >= 1
    assert theorem.checked_prefix_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_open_time_can_attach_independent_checked_spatial_ks_target_chart():
    masses, positions, velocities = _spatial_ks_close_binary_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        5.0e-5,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        order=24,
        guard_order=6,
        binary_distance_threshold=1.0e-2,
        checked_prefix_strategy="finite_target_atlas",
        checked_prefix_coefficient_tolerance=1.0e-6,
        checked_prefix_regularized_residual_tolerance=1.0e-6,
        checked_prefix_constraint_tolerance=1.0e-6,
        checked_prefix_projected_residual_tolerance=1.0e18,
    )

    assert theorem.finite_target_certificate.certified
    assert theorem.finite_target_certificate.finite_time_classification.selected_route_id == (
        "auto_spatial_ks"
    )
    assert theorem.finite_target_certificate.validated_atlas.evaluation.__class__.__name__ == (
        "SpatialKSValidatedEvaluation"
    )
    assert theorem.independent_chart_verifier_certified
    assert theorem.independent_chart_verifier_certificate.certified
    assert theorem.independent_chart_verifier_certificate.spatial_ks_binary_chart_count == 1
    assert theorem.independent_chart_verifier_certificate.checked_chart_chain_count == 1
    assert theorem.checked_prefix_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_independent_checker_accepts_supplied_spatial_ordinary_ks_handoff_chain():
    masses, positions, velocities, atlas = _spatial_ordinary_ks_handoff_data()
    finite_target = certify_finite_target_atlas_or_stop_from_validated_atlas(
        masses,
        positions,
        velocities,
        atlas.target_time,
        atlas,
    )
    verifier = construct_independent_finite_target_checked_atlas(
        finite_target,
        coefficient_tolerance=1.0e-4,
        ordinary_residual_tolerance=1.0e-4,
        regularized_residual_tolerance=1.0e-5,
        projected_residual_tolerance=1.0e4,
        constraint_tolerance=1.0e-6,
        physical_time_tolerance=1.0e-8,
        position_tolerance=1.0e-5,
        velocity_tolerance=1.0e-5,
    )

    assert atlas.proof_certified
    assert finite_target.certified
    assert finite_target.finite_time_classification.selected_route_id == (
        "supplied_validated_atlas"
    )
    assert verifier.certified
    assert verifier.ordinary_taylor_chart_count == 2
    assert verifier.spatial_ks_binary_chart_count == 1
    assert len(verifier.transition_results) == 2
    assert verifier.checked_chart_chain_count == 1
    assert verifier.missing_obligations == ()


def test_open_time_accepts_supplied_spatial_ordinary_ks_checked_prefix():
    masses, positions, velocities, atlas = _spatial_ordinary_ks_handoff_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        atlas.target_time,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        checked_prefix_strategy="supplied_validated_atlas",
        checked_prefix_validated_atlas=atlas,
        checked_prefix_coefficient_tolerance=1.0e-4,
        checked_prefix_residual_tolerance=1.0e-4,
        checked_prefix_regularized_residual_tolerance=1.0e-5,
        checked_prefix_projected_residual_tolerance=1.0e4,
        checked_prefix_constraint_tolerance=1.0e-6,
        checked_prefix_physical_time_tolerance=1.0e-8,
        checked_prefix_position_tolerance=1.0e-5,
        checked_prefix_velocity_tolerance=1.0e-5,
        initial_radius=1.0e-15,
        order=10,
        target_bisections=42,
    )

    assert theorem.independent_chart_verifier_certified
    assert theorem.independent_chart_verifier_certificate.certified
    assert theorem.independent_chart_verifier_certificate.ordinary_taylor_chart_count == 2
    assert theorem.independent_chart_verifier_certificate.spatial_ks_binary_chart_count == 1
    assert len(theorem.independent_chart_verifier_certificate.transition_results) == 2
    assert theorem.independent_chart_verifier_certificate.checked_chart_chain_count == 1
    assert theorem.independent_chart_verifier_arithmetic_certified
    assert theorem.independent_chart_verifier_arithmetic_blockers == ()
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_independent_checker_serializes_homothetic_total_collision_stop_chain():
    masses, positions, velocities, selector_atlas = (
        _parabolic_homothetic_total_collision_atlas()
    )
    finite_target = certify_finite_target_atlas_or_stop_from_validated_atlas(
        masses,
        positions,
        velocities,
        selector_atlas.target_time,
        selector_atlas,
        total_collision_policy="maximal_classical_stop",
    )
    verifier = construct_independent_finite_target_checked_atlas(
        finite_target,
        regularized_residual_tolerance=1.0e-8,
        projected_residual_tolerance=1.0e-8,
        sample_count=1,
    )

    assert finite_target.certified
    assert finite_target.outcome_id == "unselected_total_collision_before_target"
    assert verifier.certified
    assert verifier.total_collision_generalized_fuchsian_stop_chart_count == 1
    assert verifier.checked_chart_chain_count == 1
    assert verifier.chart_results[0].checker_id == (
        "independent_total_collision_generalized_fuchsian_stop_checker_interval_cauchy_projected_v3"
    )
    obligation_ids = {
        obligation.obligation
        for obligation in verifier.chart_results[0].obligations
    }
    assert "interval_generalized_fuchsian_lifted_residual_on_punctured_shells" in (
        obligation_ids
    )
    assert "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells" in (
        obligation_ids
    )
    assert "sampled_generalized_fuchsian_lifted_residual_with_tail" not in (
        obligation_ids
    )


def test_independent_checker_serializes_nonzero_energy_homothetic_stop_chain():
    masses, positions, velocities, selector_atlas = (
        _nonzero_energy_homothetic_total_collision_atlas()
    )
    finite_target = certify_finite_target_atlas_or_stop_from_validated_atlas(
        masses,
        positions,
        velocities,
        selector_atlas.target_time,
        selector_atlas,
        total_collision_policy="maximal_classical_stop",
    )
    too_tight = construct_independent_finite_target_checked_atlas(
        finite_target,
        regularized_residual_tolerance=1.0e-8,
        projected_residual_tolerance=1.0e-8,
        sample_count=1,
    )
    verifier = construct_independent_finite_target_checked_atlas(
        finite_target,
        regularized_residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
        sample_count=1,
    )

    assert finite_target.certified
    assert finite_target.outcome_id == "unselected_total_collision_before_target"
    assert not too_tight.certified
    assert "interval_generalized_fuchsian_lifted_residual_on_punctured_shells" in (
        too_tight.missing_obligations
    )
    assert verifier.certified
    assert verifier.total_collision_generalized_fuchsian_stop_chart_count == 1
    assert verifier.checked_chart_chain_count == 1
    assert verifier.chart_results[0].max_coefficient_residual <= 1.0e-5
    assert verifier.chart_results[0].max_sampled_newton_residual <= 2.0e4
    obligation_ids = {
        obligation.obligation
        for obligation in verifier.chart_results[0].obligations
    }
    assert "generalized_fuchsian_remainder_majorant_certifies" in obligation_ids
    assert "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells" in (
        obligation_ids
    )
    assert "sampled_generalized_fuchsian_lifted_residual_with_tail" not in (
        obligation_ids
    )


def test_homothetic_stop_adapter_accepts_independent_checker_without_sample_gate():
    masses, positions, velocities, selector_atlas = (
        _nonzero_energy_homothetic_total_collision_atlas()
    )
    sample_blocked_atlas = replace(
        selector_atlas,
        charts=(replace(selector_atlas.charts[0], residual_certified=False),),
    )
    blocked = certify_finite_target_atlas_or_stop_from_validated_atlas(
        masses,
        positions,
        velocities,
        sample_blocked_atlas.target_time,
        sample_blocked_atlas,
        total_collision_policy="maximal_classical_stop",
    )
    verifier = construct_independent_validated_atlas_checked_chain(
        sample_blocked_atlas,
        certificate_id_prefix="checker-first-homothetic-stop",
        regularized_residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
        sample_count=1,
    )
    checked = certify_finite_target_atlas_or_stop_from_validated_atlas(
        masses,
        positions,
        velocities,
        sample_blocked_atlas.target_time,
        sample_blocked_atlas,
        total_collision_policy="maximal_classical_stop",
        independent_chart_verifier_certificate=verifier,
    )

    assert not sample_blocked_atlas.proof_certified
    assert not blocked.certified
    assert "supplied_validated_atlas_proof_certified" in (
        blocked.obstruction_obligations
    )
    assert verifier.certified
    assert checked.certified
    assert checked.outcome_id == "unselected_total_collision_before_target"
    assert checked.stop_certificate.certified
    assert checked.finite_time_classification.validated_atlas is sample_blocked_atlas
    assert "supplied_validated_atlas_proof_certified" not in (
        checked.missing_obligations
    )
    assert "validated_atlas_proof_certified" not in (
        checked.stop_certificate.missing_obligations
    )


def test_independent_checker_bridge_rejects_stale_real_verifier_for_different_atlas():
    masses, positions, velocities, selector_atlas = (
        _nonzero_energy_homothetic_total_collision_atlas()
    )
    sample_blocked_atlas = replace(
        selector_atlas,
        charts=(replace(selector_atlas.charts[0], residual_certified=False),),
    )
    stale_verifier = construct_independent_validated_atlas_checked_chain(
        selector_atlas,
        certificate_id_prefix="stale-homothetic-stop",
        regularized_residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
        sample_count=1,
    )
    blocked = certify_finite_target_atlas_or_stop_from_validated_atlas(
        masses,
        positions,
        velocities,
        sample_blocked_atlas.target_time,
        sample_blocked_atlas,
        total_collision_policy="maximal_classical_stop",
        independent_chart_verifier_certificate=stale_verifier,
    )

    assert stale_verifier.certified
    assert not sample_blocked_atlas.proof_certified
    assert not blocked.certified
    assert "supplied_validated_atlas_proof_certified" in (
        blocked.obstruction_obligations
    )
    assert blocked.stop_certificate is None


def test_independent_checker_bridge_rejects_attribute_compatible_fake_verifier():
    masses, positions, velocities, selector_atlas = (
        _nonzero_energy_homothetic_total_collision_atlas()
    )
    sample_blocked_atlas = replace(
        selector_atlas,
        charts=(replace(selector_atlas.charts[0], residual_certified=False),),
    )
    fake_verifier = SimpleNamespace(
        certified=True,
        proof_grade_arithmetic_checked_bundle_certified=True,
        checked_certificate_count=len(sample_blocked_atlas.charts),
        checked_chart_chain_count=1,
    )
    blocked = certify_finite_target_atlas_or_stop_from_validated_atlas(
        masses,
        positions,
        velocities,
        sample_blocked_atlas.target_time,
        sample_blocked_atlas,
        total_collision_policy="maximal_classical_stop",
        independent_chart_verifier_certificate=fake_verifier,
    )

    assert not sample_blocked_atlas.proof_certified
    assert not blocked.certified
    assert "supplied_validated_atlas_proof_certified" in (
        blocked.obstruction_obligations
    )
    assert blocked.stop_certificate is None


def test_open_time_arithmetic_gate_rejects_attribute_compatible_fake_verifier():
    masses, positions, velocities = _spatial_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        **_solver_options(),
    )
    fake_verifier = SimpleNamespace(
        certified=True,
        proof_grade_arithmetic_checked_bundle_certified=True,
        checked_certificate_count=1,
        checked_chart_chain_count=1,
    )
    spoofed = replace(
        theorem,
        independent_chart_verifier_certificate=fake_verifier,
        independent_chart_verifier_certified=True,
    )

    assert not spoofed.independent_chart_verifier_arithmetic_certified


def test_open_time_arithmetic_gate_rejects_subclassed_fake_verifier():
    class SpoofedIndependentChartVerifierCertificate(
        IndependentChartVerifierCertificate
    ):
        @property
        def certified(self):
            return True

        @property
        def proof_grade_finite_atlas_bundle_certified(self):
            return True

        @property
        def proof_grade_finite_atlas_blockers(self):
            return ()

    masses, positions, velocities = _spatial_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        **_solver_options(),
    )
    spoofed_verifier = SpoofedIndependentChartVerifierCertificate(
        checker_id="independent_chart_verifier_v1",
        chart_results=(),
    )
    spoofed = replace(
        theorem,
        independent_chart_verifier_certificate=spoofed_verifier,
        independent_chart_verifier_certified=True,
    )

    assert spoofed_verifier.certified
    assert spoofed_verifier.proof_grade_finite_atlas_bundle_certified
    assert not spoofed.component_types_certified
    assert not spoofed.independent_checked_prefix_certified
    assert spoofed.checked_prefix_certified is theorem.theorem_prefix_obligations_certified
    assert not spoofed.independent_chart_verifier_arithmetic_certified
    assert spoofed.independent_chart_verifier_arithmetic_blockers == (
        "independent_chart_verifier_certificate_type",
    )
    assert "open_time_independent_chart_verifier_certificate_type" in (
        spoofed.missing_obligations
    )


def test_independent_finite_target_checker_rejects_truthy_atlas_proof_flag():
    fake_finite_target = SimpleNamespace(
        validated_atlas=SimpleNamespace(proof_certified="yes"),
    )

    with pytest.raises(
        ValueError,
        match="finite target certificate does not carry a proof-certified atlas",
    ):
        construct_independent_finite_target_checked_atlas(fake_finite_target)


def test_compact_interval_rejects_attribute_compatible_finite_target_certificates():
    masses, positions, velocities = _spatial_initial_data()
    input_domain = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    policy = TotalCollisionPolicyCertificate(
        policy_id="maximal_classical_stop",
        stop_at_unselected_total_collision=True,
        selected_continuation_allowed=False,
    )
    fake_past = SimpleNamespace(
        certified="yes",
        target_time=-0.25,
        outcome_id="finite_atlas_reaches_target",
        input_domain_certificate=input_domain,
        total_collision_policy=policy,
        missing_obligations=(),
        obstruction_obligations=(),
        stop_certificate=None,
    )
    fake_future = SimpleNamespace(
        certified="yes",
        target_time=0.25,
        outcome_id="finite_atlas_reaches_target",
        input_domain_certificate=input_domain,
        total_collision_policy=policy,
        missing_obligations=(),
        obstruction_obligations=(),
        stop_certificate=None,
    )

    compact = certify_compact_interval_atlas_or_stop_from_finite_targets(
        fake_past,
        fake_future,
        0.25,
        total_collision_policy="maximal_classical_stop",
    )

    assert not compact.certified
    assert compact.outcome_id == "proof_grade_obstruction"
    assert compact.finite_target_certificates == ()
    assert "past_finite_target_atlas_or_stop_theorem" in compact.missing_obligations
    assert "future_finite_target_atlas_or_stop_theorem" in compact.missing_obligations
    assert "past_finite_target_atlas_or_stop_theorem_type" in (
        compact.obstruction_obligations
    )
    assert "future_finite_target_atlas_or_stop_theorem_type" in (
        compact.obstruction_obligations
    )


def test_open_time_child_certificates_reject_attribute_compatible_nested_components():
    masses, positions, velocities = _spatial_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        **_solver_options(),
    )
    fake_finite_target = SimpleNamespace(
        certified=True,
        proof_certified=True,
        target_time=1.0e-4,
        outcome_id="finite_atlas_reaches_target",
        missing_obligations=(),
    )
    fake_compact_interval = SimpleNamespace(
        certified=True,
        proof_certified=True,
        missing_obligations=(),
    )

    countable = replace(
        theorem.countable_exhaustion_certificate,
        finite_target_certificate=fake_finite_target,
        compact_interval_certificate=fake_compact_interval,
    )
    compact = replace(
        theorem.compact_interval_certificate,
        past_target_certificate=fake_finite_target,
        future_target_certificate=fake_finite_target,
    )
    family = replace(
        theorem.exhaustion_family_certificate,
        prefix_certificates=(fake_compact_interval,),
        finite_target_reduction_certificate=SimpleNamespace(proof_certified=True),
        local_finiteness_certificate=SimpleNamespace(proof_certified=True),
    )

    assert not countable.component_types_certified
    assert not countable.certified
    assert not countable.proof_certified
    assert "countable_exhaustion_finite_target_type" in countable.missing_obligations
    assert "countable_exhaustion_compact_interval_type" in (
        countable.missing_obligations
    )
    assert not compact.component_types_certified
    assert not compact.certified
    assert not compact.proof_certified
    assert "compact_interval_past_finite_target_type" in compact.missing_obligations
    assert "compact_interval_future_finite_target_type" in (
        compact.missing_obligations
    )
    assert not family.component_types_certified
    assert not family.certified
    assert not family.proof_certified
    assert "compact_interval_exhaustion_prefix_type" in family.missing_obligations
    assert "compact_interval_exhaustion_reduction_type" in family.missing_obligations
    assert "compact_interval_exhaustion_local_finiteness_type" in (
        family.missing_obligations
    )


def test_finite_target_completeness_reduction_rejects_spoofed_nested_theorems():
    masses, positions, velocities = _spatial_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        **_solver_options(),
    )
    reduction = theorem.finite_target_completeness_certificate
    spoofed = replace(
        reduction,
        pointwise_completeness_theorem=SimpleNamespace(
            proof_certified=True,
            certified=True,
            missing_obligations=(),
            unaudited_analytic_lemma_ids=(),
            critical_unaudited_analytic_lemma_ids=(),
            analytic_lemma_audit_blockers=(),
        ),
        certificate_search_completeness=SimpleNamespace(
            proof_certified=True,
            certified=True,
            missing_obligations=(),
        ),
    )
    stale_pointwise = certify_finite_target_completeness_theorem(
        dimension=2,
        total_collision_policy_id="maximal_classical_stop",
    )
    stale = replace(reduction, pointwise_completeness_theorem=stale_pointwise)

    assert reduction.component_types_certified
    assert reduction.source_matches
    assert not spoofed.component_types_certified
    assert not spoofed.certified
    assert not spoofed.proof_certified
    assert "finite_target_reduction_pointwise_theorem_type" in (
        spoofed.missing_obligations
    )
    assert "finite_target_reduction_search_completeness_type" in (
        spoofed.missing_obligations
    )
    assert stale.component_types_certified
    assert not stale.source_matches
    assert not stale.certified
    assert not stale.proof_certified
    assert "finite_target_reduction_source_match" in stale.missing_obligations


def test_independent_checker_serializes_validated_branch_union_leaf_chains():
    _masses, _positions, _velocities, partition, branch_union_atlas = (
        _spatial_close_pair_branch_union_atlas_data()
    )
    verifier = construct_independent_validated_atlas_checked_chain(
        branch_union_atlas,
        certificate_id_prefix="branch-union-checked-prefix",
        coefficient_tolerance=1.0e-4,
        ordinary_residual_tolerance=1.0e-4,
        regularized_residual_tolerance=1.0e-5,
        projected_residual_tolerance=1.0e20,
        constraint_tolerance=1.0e-6,
        physical_time_tolerance=1.0e-8,
        position_tolerance=1.0e-5,
        velocity_tolerance=1.0e-5,
    )

    assert partition.certified
    assert branch_union_atlas.proof_certified
    assert verifier.certified
    assert verifier.checked_branch_union_count == 1
    assert verifier.checked_chart_chain_count == len(partition.branches)
    assert verifier.ordinary_taylor_chart_count == 0
    assert verifier.spatial_ks_binary_chart_count == len(partition.branches)
    assert verifier.missing_obligations == ()


def test_independent_checker_serializes_stratified_branch_union_taxonomy():
    _masses, _positions, _velocities, partition, branch_union_atlas = (
        _spatial_close_pair_branch_union_atlas_data()
    )
    source_tree = certify_supplied_branch_event_tree(partition)
    stratified_partition = certify_stratified_branch_event_tree(source_tree)
    stratified_evaluation = replace(
        branch_union_atlas.evaluation,
        branch_partition=stratified_partition,
    )
    stratified_atlas = replace(
        branch_union_atlas,
        evaluation=stratified_evaluation,
    )

    verifier = construct_independent_validated_atlas_checked_chain(
        stratified_atlas,
        certificate_id_prefix="stratified-branch-union-checked-prefix",
        coefficient_tolerance=1.0e-4,
        ordinary_residual_tolerance=1.0e-4,
        regularized_residual_tolerance=1.0e-5,
        projected_residual_tolerance=1.0e20,
        constraint_tolerance=1.0e-6,
        physical_time_tolerance=1.0e-8,
        position_tolerance=1.0e-5,
        velocity_tolerance=1.0e-5,
    )
    result = verifier.branch_union_results[-1]

    assert stratified_partition.proof_certified
    assert verifier.certified
    assert result.union_type == "finite_time_stratified_branch_union"
    taxonomy_obligation = next(
        obligation
        for obligation in result.obligations
        if obligation.obligation == "stratified_branch_union_leaf_kinds_supported"
    )
    assert "separated_binary_entry" in taxonomy_obligation.detail
    assert "stratified_branch_union_leaf_kinds_supported" not in (
        result.missing_obligations
    )
    assert "stratified_branch_union_unsupported_strata_absent" not in (
        result.missing_obligations
    )


def test_open_time_accepts_supplied_branch_union_checked_prefix():
    masses, positions, velocities, partition, atlas = (
        _spatial_close_pair_branch_union_atlas_data()
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        atlas.target_time,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        checked_prefix_strategy="supplied_validated_atlas",
        checked_prefix_validated_atlas=atlas,
        checked_prefix_coefficient_tolerance=1.0e-4,
        checked_prefix_residual_tolerance=1.0e-4,
        checked_prefix_regularized_residual_tolerance=1.0e-5,
        checked_prefix_projected_residual_tolerance=1.0e20,
        checked_prefix_constraint_tolerance=1.0e-6,
        checked_prefix_physical_time_tolerance=1.0e-8,
        checked_prefix_position_tolerance=1.0e-5,
        checked_prefix_velocity_tolerance=1.0e-5,
        initial_radius=1.0e-15,
        order=10,
        target_bisections=42,
    )

    assert theorem.independent_chart_verifier_certified
    assert theorem.independent_chart_verifier_certificate.certified
    assert theorem.independent_chart_verifier_certificate.checked_branch_union_count == 1
    assert theorem.independent_chart_verifier_certificate.checked_chart_chain_count == (
        len(partition.branches)
    )
    assert theorem.independent_chart_verifier_certificate.spatial_ks_binary_chart_count == (
        len(partition.branches)
    )
    assert theorem.independent_chart_verifier_arithmetic_certified
    assert theorem.independent_chart_verifier_arithmetic_blockers == ()
    assert theorem.independent_checked_prefix_certified
    assert theorem.checked_prefix_certified
    assert not theorem.certified
    assert "compact_interval_atlas_or_stop_theorem" in theorem.missing_obligations
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_open_time_accepts_stratified_branch_union_checked_prefix():
    masses, positions, velocities, partition, atlas = (
        _spatial_close_pair_branch_union_atlas_data()
    )
    stratified_partition = certify_stratified_branch_event_tree(
        certify_supplied_branch_event_tree(partition),
    )
    stratified_atlas = replace(
        atlas,
        evaluation=replace(atlas.evaluation, branch_partition=stratified_partition),
    )

    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        stratified_atlas.target_time,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        checked_prefix_strategy="supplied_validated_atlas",
        checked_prefix_validated_atlas=stratified_atlas,
        checked_prefix_coefficient_tolerance=1.0e-4,
        checked_prefix_residual_tolerance=1.0e-4,
        checked_prefix_regularized_residual_tolerance=1.0e-5,
        checked_prefix_projected_residual_tolerance=1.0e20,
        checked_prefix_constraint_tolerance=1.0e-6,
        checked_prefix_physical_time_tolerance=1.0e-8,
        checked_prefix_position_tolerance=1.0e-5,
        checked_prefix_velocity_tolerance=1.0e-5,
        initial_radius=1.0e-15,
        order=10,
        target_bisections=42,
    )
    result = theorem.independent_chart_verifier_certificate.branch_union_results[-1]

    assert theorem.independent_chart_verifier_certified
    assert theorem.independent_chart_verifier_certificate.certified
    assert result.union_type == "finite_time_stratified_branch_union"
    assert result.missing_obligations == ()
    assert theorem.independent_chart_verifier_arithmetic_certified
    assert theorem.independent_chart_verifier_arithmetic_blockers == ()
    assert theorem.independent_checked_prefix_certified
    assert theorem.checked_prefix_certified
    assert not theorem.certified
    assert "set_valued_constructor_branch_event_completeness" in (
        theorem.missing_obligations
    )


def test_branch_union_checked_prefix_rejects_attribute_compatible_fake_verifier():
    masses, positions, velocities, _partition, atlas = (
        _spatial_close_pair_branch_union_atlas_data()
    )
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        atlas.target_time,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        checked_prefix_strategy="supplied_validated_atlas",
        checked_prefix_validated_atlas=atlas,
        checked_prefix_coefficient_tolerance=1.0e-4,
        checked_prefix_residual_tolerance=1.0e-4,
        checked_prefix_regularized_residual_tolerance=1.0e-5,
        checked_prefix_projected_residual_tolerance=1.0e20,
        checked_prefix_constraint_tolerance=1.0e-6,
        checked_prefix_physical_time_tolerance=1.0e-8,
        checked_prefix_position_tolerance=1.0e-5,
        checked_prefix_velocity_tolerance=1.0e-5,
        initial_radius=1.0e-15,
        order=10,
        target_bisections=42,
    )
    fake_verifier = SimpleNamespace(
        certified=True,
        proof_grade_arithmetic_checked_bundle_certified=True,
        checked_certificate_count=0,
        checked_branch_union_count=1,
        checked_chart_chain_count=1,
    )
    spoofed = replace(
        theorem,
        independent_chart_verifier_certificate=fake_verifier,
        independent_chart_verifier_certified=True,
    )

    assert theorem.independent_checked_prefix_certified
    assert theorem.checked_prefix_certified
    assert not spoofed.independent_checked_prefix_certified
    assert not spoofed.checked_prefix_certified
    assert not spoofed.certified


def test_independent_checker_serializes_spatial_ks_to_ks_competing_chain():
    atlas = _spatial_two_ks_competing_handoff_atlas()
    verifier = construct_independent_validated_atlas_checked_chain(
        atlas,
        certificate_id_prefix="spatial-ks-competing-checked-prefix",
        coefficient_tolerance=1.0e-4,
        regularized_residual_tolerance=1.0e-5,
        projected_residual_tolerance=1.0e20,
        constraint_tolerance=1.0e-6,
        physical_time_tolerance=1.0e-8,
        position_tolerance=1.0e-5,
        velocity_tolerance=1.0e-5,
    )

    assert atlas.proof_certified
    assert [chart.chart_type for chart in atlas.charts] == [
        "spatial_ks_binary",
        "spatial_ks_binary",
    ]
    assert verifier.certified
    assert verifier.spatial_ks_binary_chart_count == 2
    assert len(verifier.transition_results) == 1
    assert verifier.transition_results[0].transition_type == (
        "spatial_ks_to_ks_competing_binary_entry"
    )
    assert verifier.checked_chart_chain_count == 1
    assert verifier.missing_obligations == ()


def test_proof_certified_properties_do_not_raise_or_promote_scaffolds():
    masses, positions, velocities = _spatial_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        **_solver_options(),
    )
    pointwise = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    objects = (
        theorem.finite_target_certificate.total_collision_policy,
        theorem.finite_target_certificate.painleve_certificate,
        theorem.finite_target_certificate,
        theorem.compact_interval_certificate,
        theorem.exhaustion_family_certificate,
        theorem.countable_exhaustion_certificate,
        theorem.finite_target_completeness_certificate,
        theorem,
        pointwise.finite_target_reduction_certificate,
        pointwise.local_finiteness_certificate,
        pointwise,
    )
    statuses = tuple(bool(obj.proof_certified) for obj in objects if obj is not None)

    assert statuses[0]
    assert not any(statuses[1:8])
    assert not any(statuses[8:])
    assert theorem.checked_prefix_certified
    assert not theorem.finite_target_completeness_certificate.certified
    assert not theorem.finite_target_completeness_certificate.proof_certified
    assert pointwise.certified
    assert not pointwise.proof_certified


def test_compact_interval_exhaustion_family_checks_nested_prefix_and_analytic_reduction():
    masses, positions, velocities = _spatial_initial_data()
    family = construct_compact_interval_exhaustion_family(
        masses,
        positions,
        velocities,
        1.0e-4,
        prefix_count=2,
        compact_time_rate=1.3,
        **_solver_options(),
    )

    assert family.certified
    assert family.exhaustion_formula == "K_n=[-nR,nR], n=1,2,..."
    assert np.allclose(family.interval_radii, (1.0e-4, 2.0e-4), rtol=0.0, atol=1.0e-18)
    assert len(family.prefix_certificates) == 2
    assert family.prefix_certificates[0].interval_lower == -1.0e-4
    assert family.prefix_certificates[1].interval_lower == -2.0e-4
    assert family.finite_target_reduction_certificate.certified
    assert family.local_finiteness_certificate.certified
    assert family.missing_obligations == ()


def test_compact_interval_exhaustion_family_blocks_empty_prefix():
    masses, positions, velocities = _spatial_initial_data()
    family = construct_compact_interval_exhaustion_family(
        masses,
        positions,
        velocities,
        1.0e-4,
        prefix_count=0,
        compact_time_rate=1.3,
        **_solver_options(),
    )

    assert not family.certified
    assert family.prefix_certificates == ()
    assert "positive_exhaustion_prefix_count" in family.missing_obligations
    assert "compact_interval_prefix_constructed" in family.missing_obligations
    assert "compact_interval_prefix_certified" in family.missing_obligations
    assert "linear_radius_schedule_prefix" in family.missing_obligations
    assert "nested_compact_interval_prefix" in family.missing_obligations


def test_compact_interval_atlas_or_stop_uses_forward_and_backward_finite_targets():
    masses, positions, velocities = _spatial_initial_data()
    compact = construct_compact_interval_atlas_or_stop(
        masses,
        positions,
        velocities,
        1.0e-4,
        **_solver_options(),
    )

    assert compact.certified
    assert compact.outcome_id == "compact_interval_atlas_reaches_both_endpoints"
    assert compact.proof_grade_response_certified
    assert compact.past_target_certificate.target_time == -1.0e-4
    assert compact.future_target_certificate.target_time == 1.0e-4
    assert compact.past_target_certificate.certified
    assert compact.future_target_certificate.certified
    assert compact.missing_obligations == ()
    assert compact.obstruction_obligations == ()


def test_compact_interval_atlas_or_stop_returns_typed_obstruction_for_bad_radius():
    masses, positions, velocities = _spatial_initial_data()
    compact = construct_compact_interval_atlas_or_stop(
        masses,
        positions,
        velocities,
        0.0,
        **_solver_options(),
    )

    assert not compact.certified
    assert compact.outcome_id == "proof_grade_obstruction"
    assert compact.proof_grade_response_certified
    assert "positive_time_radius_compact_interval" in compact.missing_obligations
    assert "positive_time_radius_compact_interval" in compact.obstruction_obligations
    assert "past_finite_target_atlas_or_stop_theorem" in compact.obstruction_obligations
    assert "future_finite_target_atlas_or_stop_theorem" in compact.obstruction_obligations


def test_finite_target_atlas_or_stop_returns_typed_obstruction_for_bad_input():
    masses, positions, velocities = _spatial_initial_data()
    theorem = construct_finite_target_atlas_or_stop(
        np.array([1.0, -0.7, 1.4]),
        positions,
        velocities,
        1.0e-4,
        **_solver_options(),
    )

    assert not theorem.certified
    assert theorem.outcome_id == "proof_grade_obstruction"
    assert theorem.proof_grade_response_certified
    assert "positive_mass_noncollision_input_domain" in theorem.missing_obligations
    assert "finite_target_atlas_or_total_collision_stop" in theorem.missing_obligations
    assert "positive_mass_noncollision_input_domain" in theorem.obstruction_obligations


def test_compact_ordinary_binary_finite_atlas_policy_controls_total_collision_charts():
    masses, positions, velocities, selector_atlas = _parabolic_homothetic_total_collision_atlas()
    input_domain = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )

    stop_policy_certificate = certify_compact_ordinary_binary_finite_atlas(
        input_domain_certificate=input_domain,
        validated_atlas=selector_atlas,
        total_collision_policy_id="maximal_classical_stop",
    )
    selected_policy_certificate = certify_compact_ordinary_binary_finite_atlas(
        input_domain_certificate=input_domain,
        validated_atlas=selector_atlas,
        total_collision_policy_id="selected_identity_selector",
    )

    assert selector_atlas.proof_certified
    assert stop_policy_certificate.total_collision_chart_count == 1
    assert not stop_policy_certificate.certified
    assert "total_collision_policy_matches_chart_family" in (
        stop_policy_certificate.missing_obligations
    )
    assert selected_policy_certificate.certified
    assert selected_policy_certificate.total_collision_chart_count == 1
    assert "finite_jet_identity_selector_total_collision" in (
        selected_policy_certificate.chart_types
    )


def test_finite_target_adapter_turns_unselected_total_collision_chart_into_stop():
    masses, positions, velocities, selector_atlas = _parabolic_homothetic_total_collision_atlas()
    theorem = certify_finite_target_atlas_or_stop_from_validated_atlas(
        masses,
        positions,
        velocities,
        selector_atlas.target_time,
        selector_atlas,
        total_collision_policy="maximal_classical_stop",
    )

    assert theorem.certified
    assert theorem.outcome_id == "unselected_total_collision_before_target"
    assert theorem.stop_certificate.certified
    assert theorem.stop_certificate.total_collision_chart_type == (
        "finite_jet_identity_selector_total_collision"
    )
    assert theorem.stop_certificate.stop_time == 0.0
    assert theorem.finite_atlas_certificate is not None
    assert not theorem.finite_atlas_certificate.certified
    assert "selected_total_collision_policy_required" not in theorem.obstruction_obligations
    assert theorem.missing_obligations == ()


def test_finite_target_adapter_preserves_selected_total_collision_continuation_policy():
    masses, positions, velocities, selector_atlas = _parabolic_homothetic_total_collision_atlas()
    theorem = certify_finite_target_atlas_or_stop_from_validated_atlas(
        masses,
        positions,
        velocities,
        selector_atlas.target_time,
        selector_atlas,
        total_collision_policy="selected_identity_selector",
    )

    assert theorem.certified
    assert theorem.outcome_id == "selected_total_collision_continuation"
    assert theorem.stop_certificate is None
    assert theorem.finite_atlas_certificate.certified


def test_compact_interval_from_finite_targets_carries_total_collision_stop_certificate():
    masses, positions, velocities, selector_atlas = _parabolic_homothetic_total_collision_atlas()
    future_stop = certify_finite_target_atlas_or_stop_from_validated_atlas(
        masses,
        positions,
        velocities,
        selector_atlas.target_time,
        selector_atlas,
        total_collision_policy="maximal_classical_stop",
    )
    past_stop = certify_finite_target_atlas_or_stop_from_validated_atlas(
        masses,
        positions,
        velocities,
        -selector_atlas.target_time,
        selector_atlas,
        total_collision_policy="maximal_classical_stop",
    )
    compact = certify_compact_interval_atlas_or_stop_from_finite_targets(
        past_stop,
        future_stop,
        selector_atlas.target_time,
        total_collision_policy="maximal_classical_stop",
    )

    assert future_stop.stop_certificate.certified
    assert past_stop.stop_certificate.certified
    assert compact.certified
    assert compact.outcome_id == "compact_interval_total_collision_stop"
    assert compact.stop_certificate.certified
    assert compact.stop_certificate.stop_time == 0.0
    assert "compact_interval_total_collision_stop_certificate" not in (
        compact.missing_obligations
    )


def test_compact_interval_from_finite_targets_rejects_policy_splicing():
    masses, positions, velocities = _spatial_initial_data()
    future = construct_finite_target_atlas_or_stop(
        masses,
        positions,
        velocities,
        1.0e-4,
        **_solver_options(),
    )
    past = construct_finite_target_atlas_or_stop(
        masses,
        positions,
        velocities,
        -1.0e-4,
        **_solver_options(),
    )
    compact = certify_compact_interval_atlas_or_stop_from_finite_targets(
        past,
        future,
        1.0e-4,
        total_collision_policy="selected_identity_selector",
    )

    assert future.certified
    assert past.certified
    assert not compact.certified
    assert compact.outcome_id == "proof_grade_obstruction"
    assert "compact_interval_endpoint_policy_consistency" in compact.missing_obligations
    assert (
        "compact_interval_endpoint_policy_consistency"
        in compact.obstruction_obligations
    )


def test_compact_interval_from_finite_targets_rejects_target_time_splicing():
    masses, positions, velocities = _spatial_initial_data()
    future = construct_finite_target_atlas_or_stop(
        masses,
        positions,
        velocities,
        1.0e-4,
        **_solver_options(),
    )
    wrong_past = construct_finite_target_atlas_or_stop(
        masses,
        positions,
        velocities,
        1.0e-4,
        **_solver_options(),
    )
    compact = certify_compact_interval_atlas_or_stop_from_finite_targets(
        wrong_past,
        future,
        1.0e-4,
        total_collision_policy="maximal_classical_stop",
    )

    assert future.certified
    assert wrong_past.certified
    assert not compact.certified
    assert compact.outcome_id == "proof_grade_obstruction"
    assert (
        "compact_interval_endpoint_target_time_consistency"
        in compact.missing_obligations
    )
    assert (
        "compact_interval_endpoint_target_time_consistency"
        in compact.obstruction_obligations
    )


def test_compact_interval_from_finite_targets_rejects_input_domain_splicing():
    masses, positions, velocities = _spatial_initial_data()
    future = construct_finite_target_atlas_or_stop(
        masses,
        positions,
        velocities,
        1.0e-4,
        **_solver_options(),
    )
    past = construct_finite_target_atlas_or_stop(
        masses,
        positions,
        velocities,
        -1.0e-4,
        **_solver_options(),
    )
    foreign_input_domain = certify_positive_mass_noncollision_input_domain(
        masses,
        positions + 0.02,
        velocities,
    )
    spliced_future = replace(
        future,
        input_domain_certificate=foreign_input_domain,
    )
    compact = certify_compact_interval_atlas_or_stop_from_finite_targets(
        past,
        spliced_future,
        1.0e-4,
        total_collision_policy="maximal_classical_stop",
    )

    assert future.certified
    assert past.certified
    assert spliced_future.certified
    assert not compact.certified
    assert compact.outcome_id == "proof_grade_obstruction"
    assert (
        "compact_interval_endpoint_input_domain_consistency"
        in compact.missing_obligations
    )
    assert (
        "compact_interval_endpoint_input_domain_consistency"
        in compact.obstruction_obligations
    )
