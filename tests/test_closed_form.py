from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import three_body_symmetry as three_body_api

from three_body_symmetry.closed_form import (
    certify_binary_collision_continuation_witness,
    certify_collision_continuation_witness,
    certify_closed_form_target,
    certify_closed_form_function_class_witness,
    certify_certificate_language_soundness,
    certify_computable_atlas_certificate_enumeration,
    certify_compact_time_real_line_bijection_witness,
    certify_general_closed_form_solution_target,
    certify_general_solution_scope_witness,
    certify_general_solution_theorem_scope_witness,
    certify_inertial_projection_witness,
    certify_maximal_classical_total_collision_policy,
    certify_nonzero_angular_momentum_triple_exclusion_witness,
    certify_pointwise_regularized_atlas_closed_form_theorem,
    certify_sundman_general_solution_theorem_witness,
    certify_sundman_time_targeting_global_witness,
    certify_zero_angular_momentum_triple_collision_convention,
    derive_certificate_language_soundness_from_checker_kernel,
    derive_computable_atlas_certificate_enumeration_from_pointwise_theorem,
)
from three_body_symmetry.certificate_checker import (
    certify_certificate_checker_kernel_support,
    certify_rational_interval_arithmetic_backend_soundness,
)
from three_body_symmetry.branch_event_tree import certify_supplied_branch_event_tree
from three_body_symmetry.event_recurrence import (
    derive_all_future_event_budget_from_geometric_shell_isolation,
    derive_geometric_shell_event_isolation,
)
from three_body_symmetry.general_solution_theorem import (
    TheoremPipelineObligation,
    certify_compact_time_real_line_coverage,
    certify_positive_mass_noncollision_input_domain,
    classify_global_regime,
    construct_general_solution_theorem_certificate,
    construct_global_atlas_for_regime,
)
from three_body_symmetry.open_time_atlas import (
    OpenTimeLocallyFiniteAtlasTheoremCertificate,
    certify_pointwise_open_time_locally_finite_atlas_theorem,
    construct_open_time_locally_finite_atlas_theorem,
)
from three_body_symmetry.stratified_branch_tree import (
    certify_stratified_branch_event_tree,
)
from three_body_symmetry.finite_target_completeness import (
    FINITE_TARGET_CRITICAL_ANALYTIC_LEMMA_IDS,
    certify_uniform_margin_branch_refinement_termination,
)
from three_body_symmetry.ks_binary_chart import (
    SpatialKSBinaryChartState,
    ks_binary_chart_to_spatial,
)
from three_body_symmetry.ks_binary_series import (
    construct_spatial_ks_binary_taylor_solution,
)
from three_body_symmetry.validated_atlas import (
    certify_simultaneous_close_pair_partition,
    validated_atlas_from_spatial_close_pair_branch_partition,
    validated_atlas_from_spatial_ordinary_ks_handoff,
)


def _complete_certificate_language_soundness(**overrides):
    fields = dict(
        ordinary_taylor_sound=True,
        levi_civita_sound=True,
        spatial_ks_sound=True,
        fuchsian_stop_sound=True,
        generalized_fuchsian_stop_sound=True,
        transition_sound=True,
        branch_union_sound=True,
        chart_chain_sound=True,
        verifier_kernel_sound=True,
        proof_grade_arithmetic_backend_sound=True,
    )
    fields.update(overrides)
    return certify_certificate_language_soundness(**fields)


def _complete_computable_atlas_certificate_enumeration(**overrides):
    fields = dict(
        chart_family_words_enumerated=True,
        pair_labels_enumerated=True,
        rational_domains_enumerated=True,
        truncation_orders_enumerated=True,
        rational_or_interval_coefficients_enumerated=True,
        rational_tail_budgets_enumerated=True,
        generalized_fuchsian_exponent_data_enumerated=True,
        fuchsian_selector_constants_enumerated=True,
        cauchy_majorants_enumerated=True,
        transition_witnesses_enumerated=True,
        collision_policy_data_enumerated=True,
        independent_checker_dovetailed=True,
        dovetailing_fairness_certified=True,
        finite_target_query_terminates_certified=True,
    )
    fields.update(overrides)
    return certify_computable_atlas_certificate_enumeration(**fields)


@dataclass(frozen=True)
class _ObligationDetail:
    obligation: str
    certified: bool


@dataclass(frozen=True)
class _CompactSundmanWitness:
    global_series_certified: bool
    missing_global_proof_obligations: tuple[str, ...] = ()
    global_proof_obligation_details: tuple[_ObligationDetail, ...] = ()
    collision_continuation_obligations_certified: bool = False
    general_solution_scope_certified: bool = False
    arbitrary_positive_masses_certified: bool = False
    arbitrary_noncollision_initial_data_certified: bool = False
    all_real_target_times_certified: bool = False
    lift_construct_project_verify_certified: bool = False
    newton_equations_full_interval_certified: bool = False


@dataclass(frozen=True)
class _RecurrenceClosure:
    compact_witness: _CompactSundmanWitness
    missing_recurrence_obligations: tuple[str, ...] = ()
    recurrence_obligation_details: tuple[_ObligationDetail, ...] = ()

    def to_accelerated_induction_witness(
        self,
        *,
        collision_continuation_certified: bool = False,
        binary_collision_continuation_certified: bool = False,
        triple_collision_continuation_certified: bool = False,
        witness_source: str | None = None,
    ) -> _CompactSundmanWitness:
        del witness_source
        collision_certified = bool(
            collision_continuation_certified
            and binary_collision_continuation_certified
            and triple_collision_continuation_certified
        )
        collision_obligations = tuple(
            obligation
            for obligation, certified in (
                ("collision_continuation", collision_continuation_certified),
                ("binary_collision_continuation", binary_collision_continuation_certified),
                ("triple_collision_continuation", triple_collision_continuation_certified),
            )
            if not certified
        )
        missing = (*self.compact_witness.missing_global_proof_obligations, *collision_obligations)
        return _CompactSundmanWitness(
            global_series_certified=bool(
                self.compact_witness.global_series_certified and collision_certified and not missing
            ),
            missing_global_proof_obligations=missing,
            global_proof_obligation_details=self.compact_witness.global_proof_obligation_details,
            collision_continuation_obligations_certified=collision_certified,
        )


def _open_time_spatial_initial_data():
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


def _open_time_spatial_ks_initial_data():
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


def _state_box(positions, velocities, radius=1.0e-14):
    state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
    return tuple((float(value - radius), float(value + radius)) for value in state)


def _open_time_spatial_ordinary_ks_handoff_data():
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


def _open_time_spatial_close_pair_branch_union_atlas_data():
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


def _open_time_solver_options():
    return {
        "initial_radius": 1.0e-15,
        "order": 10,
        "sundman_rate": 1.15,
        "max_compact_step": 0.025,
        "radius_fraction": 0.2,
        "guard_order": 6,
        "target_bisections": 42,
    }


def _complete_zero_angular_triple_convention():
    return certify_zero_angular_momentum_triple_collision_convention(
        convention_id="sundman_total_collision_regularization",
        regularized_time_parameter_certified=True,
        terminal_collision_value_certified=True,
        continuation_selection_rule_certified=True,
        witness_source="complete_zero_angular_triple_convention_test",
    )


def _complete_binary_collision_continuation():
    return certify_binary_collision_continuation_witness(
        regularized_pairs=((0, 1), (0, 2), (1, 2)),
        pair_regularization_charts_certified=True,
        branch_atlas_certified=True,
        projection_back_to_newtonian_certified=True,
        regularized_time_parameter_certified=True,
        witness_source="complete_binary_collision_continuation_test",
    )


def _complete_nonzero_angular_triple_exclusion():
    return certify_nonzero_angular_momentum_triple_exclusion_witness(
        nonzero_branch_domain_quantified_certified=True,
        centered_angular_momentum_conservation_certified=True,
        triple_collision_zero_angular_momentum_lemma_certified=True,
        positive_lower_bound_predicate_certified=True,
        witness_source="complete_nonzero_angular_triple_exclusion_test",
    )


def _complete_sundman_function_class():
    return certify_closed_form_function_class_witness(
        class_id="sundman global series",
        representation_semantics_certified=True,
        evaluation_semantics_certified=True,
        convergence_semantics_certified=True,
        equation_verification_semantics_certified=True,
        witness_source="complete_sundman_function_class_test",
    )


def _complete_regularized_atlas_function_class():
    return certify_closed_form_function_class_witness(
        class_id="regularized locally finite atlas",
        representation_semantics_certified=True,
        evaluation_semantics_certified=True,
        convergence_semantics_certified=True,
        equation_verification_semantics_certified=True,
        witness_source="complete_regularized_atlas_function_class_test",
    )


def _complete_compact_time_real_line_bijection():
    return certify_compact_time_real_line_bijection_witness(
        positive_rate_parameter_certified=True,
        forward_map_all_real_certified=True,
        inverse_map_open_interval_certified=True,
        strict_monotonicity_certified=True,
        endpoint_limits_certified=True,
        witness_source="complete_compact_time_bijection_test",
    )


def _constructor_theorem_with_geometric_event_tail():
    shell_isolation = derive_geometric_shell_event_isolation(
        delta_initial=0.2,
        theta=0.5,
        event_isolation_initial=0.018,
        boundary_clearance_initial=0.02,
    )
    primitive_inputs = {
        "ordinary_gap_taylor": {
            "value": (0.9e-5, 1.06, 0.30, 5, 3),
            "first_jet": (2.1e-5, 1.08, 0.32, 5, 3),
            "lifted_residual": (4.6e-5, 1.05, 0.28, 6, 3),
            "physical_residual": (2.4e-9, 1.03, 0.22, 6, 3),
        },
        "separated_binary_levi_civita": {
            "value": (1.4e-5, 1.10, 0.33, 5, 3),
            "first_jet": (3.3e-5, 1.12, 0.34, 5, 3),
            "lifted_residual": (6.8e-5, 1.08, 0.31, 6, 3),
            "physical_residual": (4.1e-9, 1.04, 0.23, 6, 3),
        },
        "automatic_identity_selector_total_collision": {
            "value": (1.8e-5, 1.12, 0.32, 5, 3),
            "first_jet": (4.0e-5, 1.13, 0.35, 5, 3),
            "lifted_residual": (7.8e-5, 1.09, 0.32, 6, 3),
            "physical_residual": (4.9e-9, 1.05, 0.24, 6, 3),
        },
    }
    event_budget = derive_all_future_event_budget_from_geometric_shell_isolation(
        shell_isolation=shell_isolation,
        primitive_inputs=primitive_inputs,
        checked_prefix=6,
    )
    classification = classify_global_regime(
        input_domain_certificate=certify_positive_mass_noncollision_input_domain(
            masses=[1.0, 0.7, 1.4],
            positions=[[0.8, -0.2], [-0.4, 0.6], [0.1, -0.5]],
            velocities=[[0.05, 0.11], [-0.07, 0.03], [0.02, -0.08]],
        ),
        compact_time_certificate=certify_compact_time_real_line_coverage(1.3),
        regime_id="geometric_infinite_event_tail",
        event_isolation_certificate=shell_isolation,
        primitive_cauchy_inputs=event_budget,
    )
    return construct_general_solution_theorem_certificate(
        construct_global_atlas_for_regime(classification)
    )


def _complete_sundman_time_targeting():
    return certify_sundman_time_targeting_global_witness(
        compact_sundman_parameter_domain_certified=True,
        positive_physical_time_derivative_certified=True,
        finite_target_bracketing_certified=True,
        target_interval_evaluation_certified=True,
        global_target_range_certified=True,
        witness_source="complete_sundman_time_targeting_test",
    )


def _complete_inertial_projection():
    return certify_inertial_projection_witness(
        center_of_mass_affine_motion_certified=True,
        reduced_to_inertial_coordinate_map_certified=True,
        physical_time_series_compatibility_certified=True,
        interval_projection_containment_certified=True,
        mass_weighted_reconstruction_certified=True,
        witness_source="complete_inertial_projection_test",
    )


def test_unspecified_closed_form_target_requires_a_function_class():
    certificate = certify_closed_form_target("closed form")

    assert certificate.classification_certified
    assert certificate.definition_required
    assert not certificate.general_solution_certified
    assert not certificate.finite_closed_form_obstructed
    assert certificate.missing_requirements == ("define_allowed_closed_form_function_class",)
    assert "not yet formalized" in certificate.route_summary


def test_closed_form_function_class_witness_reports_missing_semantics():
    witness = certify_closed_form_function_class_witness(
        class_id="sundman global series",
        representation_semantics_certified=True,
        evaluation_semantics_certified=False,
        convergence_semantics_certified=True,
        equation_verification_semantics_certified=False,
        witness_source="partial_function_class_test",
    )

    assert witness.normalized_class == "sundman_global_series"
    assert not witness.function_class_definition_certified
    assert not witness.theorem_compatible_infinite_series_class_certified
    assert witness.missing_function_class_obligations == (
        "closed_form_evaluation_semantics",
        "closed_form_equation_verification_semantics",
        "closed_form_function_class_definition",
    )
    details = {
        detail.obligation: detail for detail in witness.function_class_obligation_details
    }
    assert details["closed_form_convergence_semantics"].certified
    assert details["closed_form_evaluation_semantics"].required == (
        "evaluation rule for the representation is specified"
    )


def test_regularized_locally_finite_atlas_function_class_is_first_class():
    witness = _complete_regularized_atlas_function_class()
    certificate = certify_closed_form_target("certified atlas closed form")

    assert witness.normalized_class == "regularized_locally_finite_atlas"
    assert witness.function_class_definition_certified
    assert witness.theorem_compatible_regularized_atlas_class_certified
    assert not witness.theorem_compatible_infinite_series_class_certified
    assert certificate.classification_certified
    assert certificate.status == "conditional_regularized_atlas_route"
    assert certificate.regularized_atlas_route
    assert not certificate.infinite_series_route
    assert "independent_chart_verifier" in certificate.missing_requirements
    assert "regularized locally finite atlas route remains conditional" in (
        certificate.route_summary
    )


def test_unspecified_closed_form_can_use_typed_sundman_function_class_witness():
    certificate = certify_closed_form_target(
        "closed form",
        closed_form_function_class_witness=_complete_sundman_function_class(),
    )

    assert certificate.classification_certified
    assert certificate.infinite_series_route
    assert not certificate.definition_required
    assert certificate.normalized_class == "sundman_global_series"
    assert certificate.status == "conditional_infinite_series_route"
    assert certificate.missing_requirements == (
        "compact_sundman_global_induction_witness",
        "binary_collision_continuation",
        "triple_collision_continuation",
    )


def test_unspecified_closed_form_rejects_incomplete_function_class_witness():
    witness = certify_closed_form_function_class_witness(
        class_id="sundman global series",
        representation_semantics_certified=True,
        evaluation_semantics_certified=True,
        convergence_semantics_certified=False,
        equation_verification_semantics_certified=True,
        witness_source="incomplete_function_class_for_target_test",
    )

    certificate = certify_closed_form_target(
        "closed form",
        closed_form_function_class_witness=witness,
    )

    assert certificate.definition_required
    assert certificate.normalized_class == "sundman_global_series"
    assert certificate.missing_requirements == (
        "closed_form_convergence_semantics",
        "closed_form_function_class_definition",
    )


def test_finite_first_integral_closed_form_route_is_obstructed_by_bruns_theorem():
    certificate = certify_closed_form_target("finite algebraic first integrals")

    assert certificate.classification_certified
    assert certificate.obstruction_certified
    assert certificate.finite_closed_form_obstructed
    assert certificate.internal_obstruction_certified
    assert certificate.internal_obstruction_evidence == (
        "classical integrals match but the Newtonian vector field is different"
    )
    assert not certificate.general_solution_certified
    assert certificate.normalized_class == "finite_first_integral_closed_form"
    assert certificate.theorem_references[0].theorem_id == "bruns_algebraic_integrals"
    assert "Bruns" in certificate.reason
    assert "finite first-integral" in certificate.route_summary


def test_analytic_extra_integral_route_is_obstructed_by_nonintegrability_results():
    certificate = certify_closed_form_target("meromorphic first integral")

    assert certificate.classification_certified
    assert certificate.obstruction_certified
    assert certificate.finite_closed_form_obstructed
    assert not certificate.general_solution_certified
    assert certificate.normalized_class == "analytic_or_meromorphic_extra_integral"
    assert certificate.theorem_references[0].theorem_id == "poincare_yagasaki_analytic_nonintegrability"
    assert "Poincare" in certificate.reason


def test_sundman_global_series_route_remains_conditional_without_global_witness():
    certificate = certify_closed_form_target("sundman global series")

    assert certificate.classification_certified
    assert certificate.infinite_series_route
    assert not certificate.finite_closed_form_obstructed
    assert not certificate.general_solution_certified
    assert certificate.status == "conditional_infinite_series_route"
    assert certificate.missing_requirements == (
        "compact_sundman_global_induction_witness",
        "binary_collision_continuation",
        "triple_collision_continuation",
    )
    assert "conditional" in certificate.route_summary


def test_sundman_route_imports_missing_obligations_from_compact_witness():
    witness = _CompactSundmanWitness(
        global_series_certified=False,
        missing_global_proof_obligations=(
            "future_domain_exhaustion",
            "quantitative_witness_dominates_accelerated_bounds",
            "triple_collision_continuation",
        ),
    )

    certificate = certify_closed_form_target(
        "infinite convergent series",
        compact_sundman_witness=witness,
    )

    assert certificate.status == "conditional_infinite_series_route"
    assert certificate.compact_sundman_obligations == witness.missing_global_proof_obligations
    assert certificate.missing_requirements == witness.missing_global_proof_obligations
    assert "quantitative_witness_dominates_accelerated_bounds" in certificate.missing_requirements


def test_sundman_route_certifies_series_but_not_general_scope_without_scope_witness():
    witness = _CompactSundmanWitness(global_series_certified=True)

    certificate = certify_closed_form_target("sundman", compact_sundman_witness=witness)

    assert certificate.classification_certified
    assert certificate.status == "certified_infinite_series_route"
    assert certificate.infinite_series_route
    assert certificate.series_route_certified
    assert not certificate.general_solution_certified
    assert certificate.missing_requirements == ()
    assert "general-solution scope" in certificate.route_summary


def test_general_solution_target_rejects_finite_closed_form_obstruction():
    certificate = certify_general_closed_form_solution_target("finite algebraic first integrals")

    assert certificate.status == "obstructed_requested_class"
    assert not certificate.proof_certified
    assert certificate.closed_form_certificate.obstruction_certified
    assert "allowed_closed_form_class" in certificate.missing_requirements
    details = {detail.requirement: detail for detail in certificate.missing_requirement_details}
    assert details["allowed_closed_form_class"].required == (
        "requested class must be a theorem-compatible infinite series or regularized atlas class"
    )
    assert details["allowed_closed_form_class"].observed == "finite_first_integral_closed_form"
    assert details["allowed_closed_form_class"].blocking_obligations == (
        "switch to an allowed infinite-series or local-numerical certificate class",
    )
    assert certificate.route_summary == "finite first-integral closed-form route is obstructed"


def test_general_solution_target_blocks_incomplete_function_class_witness():
    compact_witness = _CompactSundmanWitness(
        global_series_certified=True,
        collision_continuation_obligations_certified=True,
    )
    scope = certify_general_solution_scope_witness(
        arbitrary_positive_masses_certified=True,
        arbitrary_noncollision_initial_data_certified=True,
        all_real_target_times_certified=True,
        lift_construct_project_verify_certified=True,
        newton_equations_full_interval_certified=True,
        witness_source="complete_scope_for_function_class_blocker_test",
    )
    function_class = certify_closed_form_function_class_witness(
        class_id="sundman global series",
        representation_semantics_certified=True,
        evaluation_semantics_certified=True,
        convergence_semantics_certified=False,
        equation_verification_semantics_certified=True,
        witness_source="partial_function_class_for_general_target_test",
    )

    certificate = certify_general_closed_form_solution_target(
        "closed form",
        compact_sundman_witness=compact_witness,
        general_scope_witness=scope,
        closed_form_function_class_witness=function_class,
    )

    assert certificate.status == "definition_required"
    assert not certificate.proof_certified
    assert certificate.missing_requirements == (
        "closed_form_convergence_semantics",
        "closed_form_function_class_definition",
        "allowed_closed_form_class",
    )
    details = {detail.requirement: detail for detail in certificate.missing_requirement_details}
    assert details["allowed_closed_form_class"].blocking_obligations == (
        "closed_form_convergence_semantics",
        "closed_form_function_class_definition",
    )


def test_general_solution_target_can_certify_closed_form_with_typed_sundman_class():
    compact_witness = _CompactSundmanWitness(
        global_series_certified=True,
        collision_continuation_obligations_certified=True,
    )
    scope = certify_general_solution_scope_witness(
        arbitrary_positive_masses_certified=True,
        arbitrary_noncollision_initial_data_certified=True,
        all_real_target_times_certified=True,
        lift_construct_project_verify_certified=True,
        newton_equations_full_interval_certified=True,
        witness_source="complete_scope_for_typed_function_class_test",
    )

    certificate = certify_general_closed_form_solution_target(
        "closed form",
        compact_sundman_witness=compact_witness,
        general_scope_witness=scope,
        closed_form_function_class_witness=_complete_sundman_function_class(),
    )

    assert certificate.status == "certified"
    assert certificate.proof_certified
    assert certificate.closed_form_certificate.normalized_class == "sundman_global_series"
    assert certificate.missing_requirements == ()


def test_general_solution_target_reports_sundman_witness_obligations():
    witness = _CompactSundmanWitness(
        global_series_certified=False,
        missing_global_proof_obligations=(
            "future_domain_exhaustion",
            "binary_collision_continuation",
        ),
        global_proof_obligation_details=(
            _ObligationDetail("future_domain_exhaustion", False),
            _ObligationDetail("binary_collision_continuation", False),
            _ObligationDetail("triple_collision_continuation", True),
        ),
    )

    certificate = certify_general_closed_form_solution_target(
        "sundman global series",
        compact_sundman_witness=witness,
    )

    assert certificate.status == "incomplete"
    assert not certificate.proof_certified
    assert "future_domain_exhaustion" in certificate.missing_requirements
    assert "compact_sundman_global_series" in certificate.missing_requirements
    assert certificate.blocking_obligations == (
        "future_domain_exhaustion",
        "binary_collision_continuation",
    )
    details = {detail.requirement: detail for detail in certificate.missing_requirement_details}
    assert "future_domain_exhaustion" in details["compact_sundman_global_series"].reason
    assert details["compact_sundman_global_series"].witness_field == "global_series_certified"
    assert details["compact_sundman_global_series"].required == (
        "global_series_certified=True with no compact-Sundman proof-obligation blockers"
    )
    assert details["compact_sundman_global_series"].observed == "global_series_certified=False"
    assert details["compact_sundman_global_series"].blocking_obligations == (
        "future_domain_exhaustion",
        "binary_collision_continuation",
    )
    assert details["collision_continuation"].blocking_obligations == (
        "binary_collision_continuation",
    )


def test_general_solution_target_requires_arbitrary_scope_even_with_global_series():
    witness = _CompactSundmanWitness(
        global_series_certified=True,
        collision_continuation_obligations_certified=True,
    )

    certificate = certify_general_closed_form_solution_target(
        "sundman",
        compact_sundman_witness=witness,
    )

    assert certificate.status == "incomplete"
    assert not certificate.proof_certified
    assert "compact_sundman_global_series" not in certificate.missing_requirements
    assert "collision_continuation" not in certificate.missing_requirements
    assert "arbitrary_positive_masses" in certificate.missing_requirements
    assert "newton_equations_full_interval" in certificate.missing_requirements
    details = {detail.requirement: detail for detail in certificate.missing_requirement_details}
    assert details["arbitrary_positive_masses"].required == (
        "all positive masses are covered by the proof"
    )
    assert details["arbitrary_positive_masses"].observed == (
        "arbitrary_positive_masses_certified=False, general_solution_scope_certified=False"
    )
    assert details["newton_equations_full_interval"].observed == (
        "newton_equations_full_interval_certified=False, general_solution_scope_certified=False"
    )


def test_general_solution_scope_witness_reports_missing_items_independently():
    scope = certify_general_solution_scope_witness(
        arbitrary_positive_masses_certified=True,
        all_real_target_times_certified=True,
        witness_source="partial_scope_test",
    )

    assert not scope.general_solution_scope_certified
    assert scope.witness_source == "partial_scope_test"
    assert scope.missing_scope_requirements == (
        "arbitrary_noncollision_initial_data",
        "lift_construct_project_verify_pipeline",
        "newton_equations_full_interval",
    )
    details = {detail.requirement: detail for detail in scope.requirement_statuses}
    assert details["arbitrary_positive_masses"].certified
    assert details["arbitrary_positive_masses"].observed == "arbitrary_positive_masses_certified=True"
    assert not details["arbitrary_noncollision_initial_data"].certified
    assert details["arbitrary_noncollision_initial_data"].observed == (
        "arbitrary_noncollision_initial_data_certified=False"
    )


def test_general_solution_theorem_scope_witness_reports_granular_blockers():
    scope = certify_general_solution_theorem_scope_witness(
        positive_mass_domain_quantified_certified=True,
        noncollision_initial_domain_quantified_certified=True,
        compact_time_real_line_bijection_certified=True,
        compact_sundman_lift_certified=True,
        inertial_projection_certified=True,
        chain_rule_newton_equations_certified=True,
        witness_source="partial_granular_scope_test",
    )

    assert not scope.general_solution_scope_certified
    assert scope.witness_source == "partial_granular_scope_test"
    assert scope.missing_scope_requirements == (
        "arbitrary_positive_masses",
        "arbitrary_noncollision_initial_data",
        "all_real_target_times",
        "lift_construct_project_verify_pipeline",
        "newton_equations_full_interval",
    )
    assert scope.missing_scope_obligations == (
        "mass_parameter_regularity",
        "center_of_mass_reduction_global",
        "interval_initial_data_lift_global",
        "sundman_time_targeting_global",
        "global_series_construction_scope",
        "full_interval_residual_verification",
    )
    details = {detail.obligation: detail for detail in scope.scope_obligation_details}
    assert details["mass_parameter_regularity"].required == (
        "mass-dependent construction remains regular on the positive mass domain"
    )
    statuses = {status.requirement: status for status in scope.requirement_statuses}
    assert statuses["arbitrary_positive_masses"].blocking_obligations == (
        "mass_parameter_regularity",
    )
    assert statuses["lift_construct_project_verify_pipeline"].blocking_obligations == (
        "global_series_construction_scope",
    )


def test_compact_time_real_line_bijection_witness_reports_granular_blockers():
    compact_time = certify_compact_time_real_line_bijection_witness(
        positive_rate_parameter_certified=True,
        forward_map_all_real_certified=True,
        inverse_map_open_interval_certified=False,
        strict_monotonicity_certified=True,
        endpoint_limits_certified=False,
        witness_source="partial_compact_time_bijection_test",
    )

    assert not compact_time.compact_time_real_line_bijection_certified
    assert compact_time.missing_compact_time_obligations == (
        "compact_time_inverse_map_open_interval",
        "compact_time_endpoint_limits",
        "compact_time_real_line_bijection_witness",
    )
    details = {
        detail.obligation: detail
        for detail in compact_time.compact_time_obligation_details
    }
    assert details["compact_time_forward_map_all_real"].required == (
        "u = tanh(rate * t) is defined for every real physical time"
    )
    assert details["compact_time_endpoint_limits"].observed == (
        "endpoint_limits_certified=False"
    )


def test_theorem_scope_consumes_typed_compact_time_blockers():
    compact_time = certify_compact_time_real_line_bijection_witness(
        positive_rate_parameter_certified=True,
        forward_map_all_real_certified=True,
        inverse_map_open_interval_certified=True,
        strict_monotonicity_certified=True,
        endpoint_limits_certified=False,
        witness_source="partial_typed_compact_time_scope_test",
    )
    scope = certify_general_solution_theorem_scope_witness(
        positive_mass_domain_quantified_certified=True,
        mass_parameter_regularity_certified=True,
        noncollision_initial_domain_quantified_certified=True,
        center_of_mass_reduction_global_certified=True,
        interval_initial_data_lift_global_certified=True,
        compact_time_real_line_bijection_certified=True,
        compact_time_real_line_bijection_witness=compact_time,
        sundman_time_targeting_global_certified=True,
        compact_sundman_lift_certified=True,
        global_series_construction_scope_certified=True,
        inertial_projection_certified=True,
        chain_rule_newton_equations_certified=True,
        full_interval_residual_verification_certified=True,
        witness_source="typed_compact_time_scope_blocker_test",
    )

    assert not scope.all_real_target_times_certified
    assert not scope.general_solution_scope_certified
    assert scope.missing_scope_requirements == ("all_real_target_times",)
    statuses = {status.requirement: status for status in scope.requirement_statuses}
    assert statuses["all_real_target_times"].blocking_obligations == (
        "compact_time_endpoint_limits",
        "compact_time_real_line_bijection_witness",
    )
    details = {detail.obligation: detail for detail in scope.scope_obligation_details}
    assert details["compact_time_real_line_bijection"].observed == (
        "compact_time_real_line_bijection_certified=True; "
        "typed_witness_certified=False"
    )


def test_sundman_time_targeting_witness_reports_granular_blockers():
    targeting = certify_sundman_time_targeting_global_witness(
        compact_sundman_parameter_domain_certified=True,
        positive_physical_time_derivative_certified=False,
        finite_target_bracketing_certified=True,
        target_interval_evaluation_certified=False,
        global_target_range_certified=True,
        witness_source="partial_sundman_time_targeting_test",
    )

    assert not targeting.sundman_time_targeting_global_certified
    assert targeting.missing_sundman_target_obligations == (
        "sundman_target_positive_time_derivative",
        "sundman_target_interval_evaluation",
        "sundman_time_targeting_global_witness",
    )
    details = {
        detail.obligation: detail
        for detail in targeting.sundman_target_obligation_details
    }
    assert details["sundman_target_finite_time_bracketing"].required == (
        "every finite target time is enclosed by a certified time bracket"
    )
    assert details["sundman_target_interval_evaluation"].observed == (
        "target_interval_evaluation_certified=False"
    )


def test_theorem_scope_consumes_typed_sundman_targeting_blockers():
    targeting = certify_sundman_time_targeting_global_witness(
        compact_sundman_parameter_domain_certified=True,
        positive_physical_time_derivative_certified=True,
        finite_target_bracketing_certified=False,
        target_interval_evaluation_certified=True,
        global_target_range_certified=True,
        witness_source="partial_typed_sundman_targeting_scope_test",
    )
    scope = certify_general_solution_theorem_scope_witness(
        positive_mass_domain_quantified_certified=True,
        mass_parameter_regularity_certified=True,
        noncollision_initial_domain_quantified_certified=True,
        center_of_mass_reduction_global_certified=True,
        interval_initial_data_lift_global_certified=True,
        compact_time_real_line_bijection_witness=(
            _complete_compact_time_real_line_bijection()
        ),
        sundman_time_targeting_global_certified=True,
        sundman_time_targeting_global_witness=targeting,
        compact_sundman_lift_certified=True,
        global_series_construction_scope_certified=True,
        inertial_projection_certified=True,
        chain_rule_newton_equations_certified=True,
        full_interval_residual_verification_certified=True,
        witness_source="typed_sundman_targeting_scope_blocker_test",
    )

    assert not scope.all_real_target_times_certified
    assert not scope.general_solution_scope_certified
    assert scope.missing_scope_requirements == ("all_real_target_times",)
    statuses = {status.requirement: status for status in scope.requirement_statuses}
    assert statuses["all_real_target_times"].blocking_obligations == (
        "sundman_target_finite_time_bracketing",
        "sundman_time_targeting_global_witness",
    )
    details = {detail.obligation: detail for detail in scope.scope_obligation_details}
    assert details["sundman_time_targeting_global"].observed == (
        "sundman_time_targeting_global_certified=True; "
        "typed_witness_certified=False"
    )


def test_inertial_projection_witness_reports_granular_blockers():
    projection = certify_inertial_projection_witness(
        center_of_mass_affine_motion_certified=True,
        reduced_to_inertial_coordinate_map_certified=True,
        physical_time_series_compatibility_certified=False,
        interval_projection_containment_certified=False,
        mass_weighted_reconstruction_certified=True,
        witness_source="partial_inertial_projection_test",
    )

    assert not projection.inertial_projection_certified
    assert projection.missing_inertial_projection_obligations == (
        "inertial_projection_physical_time_series",
        "inertial_projection_interval_containment",
        "inertial_projection_witness",
    )
    details = {
        detail.obligation: detail
        for detail in projection.inertial_projection_obligation_details
    }
    assert details["inertial_projection_coordinate_map"].required == (
        "reduced coordinates are mapped back to inertial body coordinates"
    )
    assert details["inertial_projection_physical_time_series"].observed == (
        "physical_time_series_compatibility_certified=False"
    )


def test_theorem_scope_consumes_typed_inertial_projection_blockers():
    projection = certify_inertial_projection_witness(
        center_of_mass_affine_motion_certified=True,
        reduced_to_inertial_coordinate_map_certified=True,
        physical_time_series_compatibility_certified=False,
        interval_projection_containment_certified=True,
        mass_weighted_reconstruction_certified=True,
        witness_source="partial_typed_inertial_projection_scope_test",
    )
    scope = certify_general_solution_theorem_scope_witness(
        positive_mass_domain_quantified_certified=True,
        mass_parameter_regularity_certified=True,
        noncollision_initial_domain_quantified_certified=True,
        center_of_mass_reduction_global_certified=True,
        interval_initial_data_lift_global_certified=True,
        compact_time_real_line_bijection_witness=(
            _complete_compact_time_real_line_bijection()
        ),
        sundman_time_targeting_global_witness=_complete_sundman_time_targeting(),
        compact_sundman_lift_certified=True,
        global_series_construction_scope_certified=True,
        inertial_projection_certified=True,
        inertial_projection_witness=projection,
        chain_rule_newton_equations_certified=True,
        full_interval_residual_verification_certified=True,
        witness_source="typed_inertial_projection_scope_blocker_test",
    )

    assert not scope.lift_construct_project_verify_certified
    assert not scope.general_solution_scope_certified
    assert scope.missing_scope_requirements == (
        "lift_construct_project_verify_pipeline",
    )
    statuses = {status.requirement: status for status in scope.requirement_statuses}
    assert statuses["lift_construct_project_verify_pipeline"].blocking_obligations == (
        "inertial_projection_physical_time_series",
        "inertial_projection_witness",
    )
    details = {detail.obligation: detail for detail in scope.scope_obligation_details}
    assert details["inertial_projection"].observed == (
        "inertial_projection_certified=True; typed_witness_certified=False"
    )


def test_general_solution_target_consumes_typed_scope_witness():
    compact_witness = _CompactSundmanWitness(
        global_series_certified=True,
        collision_continuation_obligations_certified=True,
    )
    scope = certify_general_solution_scope_witness(
        arbitrary_positive_masses_certified=True,
        arbitrary_noncollision_initial_data_certified=True,
        all_real_target_times_certified=True,
        lift_construct_project_verify_certified=True,
        newton_equations_full_interval_certified=False,
        witness_source="typed_scope_test",
    )

    certificate = certify_general_closed_form_solution_target(
        "sundman",
        compact_sundman_witness=compact_witness,
        general_scope_witness=scope,
    )

    assert certificate.status == "incomplete"
    assert not certificate.proof_certified
    assert "arbitrary_positive_masses" not in certificate.missing_requirements
    assert "arbitrary_noncollision_initial_data" not in certificate.missing_requirements
    assert "all_real_target_times" not in certificate.missing_requirements
    assert "lift_construct_project_verify_pipeline" not in certificate.missing_requirements
    assert "newton_equations_full_interval" in certificate.missing_requirements
    details = {detail.requirement: detail for detail in certificate.missing_requirement_details}
    assert details["newton_equations_full_interval"].witness_field == (
        "newton_equations_full_interval_certified"
    )
    assert details["newton_equations_full_interval"].observed == (
        "newton_equations_full_interval_certified=False"
    )


def test_sundman_theorem_witness_exposes_recurrence_blockers_to_top_level_audit():
    recurrence = _RecurrenceClosure(
        compact_witness=_CompactSundmanWitness(
            global_series_certified=False,
            missing_global_proof_obligations=("future_cauchy_denominator_growth_bound",),
            global_proof_obligation_details=(
                _ObligationDetail("future_cauchy_denominator_growth_bound", False),
            ),
        ),
        missing_recurrence_obligations=("denominator_growth_recurrence",),
        recurrence_obligation_details=(
            _ObligationDetail("denominator_growth_recurrence", False),
        ),
    )
    scope = certify_general_solution_scope_witness(
        arbitrary_positive_masses_certified=True,
        arbitrary_noncollision_initial_data_certified=True,
        all_real_target_times_certified=True,
        lift_construct_project_verify_certified=True,
        newton_equations_full_interval_certified=True,
        witness_source="complete_scope_for_blocker_test",
    )
    theorem_witness = certify_sundman_general_solution_theorem_witness(
        recurrence_closure=recurrence,
        general_scope_witness=scope,
        collision_continuation_certified=True,
        binary_collision_continuation_certified=True,
        triple_collision_continuation_certified=True,
        witness_source="typed_sundman_blocker_test",
    )

    certificate = certify_general_closed_form_solution_target(
        "sundman",
        compact_sundman_witness=theorem_witness,
    )

    assert theorem_witness.missing_global_proof_obligations == (
        "denominator_growth_recurrence",
        "future_cauchy_denominator_growth_bound",
    )
    assert not theorem_witness.global_series_certified
    assert certificate.status == "incomplete"
    assert "compact_sundman_global_series" in certificate.missing_requirements
    assert certificate.blocking_obligations == (
        "denominator_growth_recurrence",
        "future_cauchy_denominator_growth_bound",
    )
    details = {detail.requirement: detail for detail in certificate.missing_requirement_details}
    assert details["compact_sundman_global_series"].blocking_obligations == (
        "denominator_growth_recurrence",
        "future_cauchy_denominator_growth_bound",
    )
    assert "arbitrary_positive_masses" not in certificate.missing_requirements
    assert "newton_equations_full_interval" not in certificate.missing_requirements


def test_sundman_theorem_witness_consumes_granular_scope_blockers():
    recurrence = _RecurrenceClosure(
        compact_witness=_CompactSundmanWitness(global_series_certified=True),
    )
    collision = certify_collision_continuation_witness(
        binary_collision_continuation_witness=_complete_binary_collision_continuation(),
        nonzero_angular_momentum_triple_exclusion_witness=(
            _complete_nonzero_angular_triple_exclusion()
        ),
        zero_angular_momentum_triple_collision_convention_witness=(
            _complete_zero_angular_triple_convention()
        ),
        witness_source="complete_collision_for_scope_blocker_test",
    )
    scope = certify_general_solution_theorem_scope_witness(
        positive_mass_domain_quantified_certified=True,
        mass_parameter_regularity_certified=True,
        noncollision_initial_domain_quantified_certified=True,
        center_of_mass_reduction_global_certified=True,
        interval_initial_data_lift_global_certified=True,
        compact_time_real_line_bijection_certified=True,
        compact_time_real_line_bijection_witness=(
            _complete_compact_time_real_line_bijection()
        ),
        sundman_time_targeting_global_certified=True,
        compact_sundman_lift_certified=True,
        inertial_projection_certified=True,
        chain_rule_newton_equations_certified=True,
        witness_source="partial_granular_scope_for_theorem_test",
    )
    theorem_witness = certify_sundman_general_solution_theorem_witness(
        recurrence_closure=recurrence,
        general_scope_witness=scope,
        collision_witness=collision,
        witness_source="typed_sundman_scope_blocker_test",
    )

    certificate = certify_general_closed_form_solution_target(
        "sundman global series",
        compact_sundman_witness=theorem_witness,
    )

    assert theorem_witness.global_series_certified
    assert theorem_witness.collision_continuation_obligations_certified
    assert not theorem_witness.general_solution_scope_certified
    assert "compact_sundman_global_series" not in certificate.missing_requirements
    assert "collision_continuation" not in certificate.missing_requirements
    assert certificate.status == "incomplete"
    assert "lift_construct_project_verify_pipeline" in certificate.missing_requirements
    assert "newton_equations_full_interval" in certificate.missing_requirements
    assert certificate.blocking_obligations == (
        "global_series_construction_scope",
        "full_interval_residual_verification",
    )
    details = {detail.requirement: detail for detail in certificate.missing_requirement_details}
    assert details["lift_construct_project_verify_pipeline"].blocking_obligations == (
        "global_series_construction_scope",
    )
    assert details["newton_equations_full_interval"].blocking_obligations == (
        "full_interval_residual_verification",
    )


def test_closed_form_audit_consumes_constructor_theorem_without_overclaiming():
    theorem = _constructor_theorem_with_geometric_event_tail()
    certificate = certify_general_closed_form_solution_target(
        "sundman global series",
        general_theorem_certificate=theorem,
    )
    atlas_certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
    )

    assert not theorem.regime_theorem_certified
    assert not theorem.full_general_solution_certified
    assert certificate.status == "incomplete"
    assert "compact_sundman_global_series" in certificate.missing_requirements
    assert "geometric_event_regime_membership_from_initial_data" in (
        certificate.blocking_obligations
    )
    assert "regime_classification" in certificate.blocking_obligations
    assert "global_regime_exhaustion" in certificate.blocking_obligations
    details = {detail.requirement: detail for detail in certificate.missing_requirement_details}
    assert details["compact_sundman_global_series"].witness_field == (
        "GeneralSolutionTheoremCertificate"
    )
    assert details["compact_sundman_global_series"].blocking_obligations == (
        "geometric_event_regime_membership_from_initial_data",
        "regime_classification",
        "global_regime_exhaustion",
    )
    assert details["arbitrary_positive_masses"].witness_field == (
        "GeneralSolutionTheoremCertificate.full_general_solution_certified"
    )
    assert details["arbitrary_positive_masses"].blocking_obligations == (
        "geometric_event_regime_membership_from_initial_data",
        "regime_classification",
        "global_regime_exhaustion",
    )


def test_closed_form_audit_consumes_open_time_theorem_without_endpoint_partition_blocker():
    masses, positions, velocities = _open_time_spatial_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        **_open_time_solver_options(),
    )
    certificate = certify_general_closed_form_solution_target(
        "sundman global series",
        general_theorem_certificate=theorem,
    )
    atlas_certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
    )

    assert theorem.checked_prefix_certified
    assert not theorem.certified
    assert theorem.theorem_id == "open_time_locally_finite_atlas"
    assert not theorem.endpoint_regime_partition_required
    assert certificate.status == "incomplete"
    assert atlas_certificate.status == "incomplete"
    assert "regularized_locally_finite_atlas" in (
        atlas_certificate.missing_requirements
    )
    assert "compact_sundman_global_series" not in (
        atlas_certificate.missing_requirements
    )
    assert "independent_chart_verifier" in atlas_certificate.blocking_obligations
    assert "audited_or_machine_checked_open_time_atlas_proof" in (
        atlas_certificate.blocking_obligations
    )
    assert "global_regime_exhaustion" not in certificate.blocking_obligations
    assert "arbitrary_initial_data_partition_theorem" not in (
        certificate.blocking_obligations
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        certificate.blocking_obligations
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
    for lemma_id in expected_critical_blockers:
        assert lemma_id in certificate.blocking_obligations
        assert lemma_id in theorem.critical_unaudited_analytic_lemma_ids
    assert "binary_degenerate_total_collision_exclusion" not in (
        certificate.blocking_obligations
    )
    assert "binary_degenerate_total_collision_exclusion" not in (
        theorem.critical_unaudited_analytic_lemma_ids
    )
    assert "reduced_hyperbolic_total_collision_entry" not in (
        certificate.blocking_obligations
    )
    assert "reduced_hyperbolic_total_collision_entry" not in (
        theorem.critical_unaudited_analytic_lemma_ids
    )
    assert "poincare_dulac_fuchsian_log_selector_completeness" not in (
        certificate.blocking_obligations
    )
    assert "poincare_dulac_fuchsian_log_selector_completeness" not in (
        theorem.critical_unaudited_analytic_lemma_ids
    )
    assert "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data" not in (
        certificate.blocking_obligations
    )
    assert "arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data" not in (
        theorem.critical_unaudited_analytic_lemma_ids
    )
    assert "arbitrary_total_collision_germ_entry_to_stop_chart" not in (
        certificate.blocking_obligations
    )
    assert "arbitrary_total_collision_germ_entry_to_stop_chart" not in (
        theorem.critical_unaudited_analytic_lemma_ids
    )
    assert "total_collision_stop_chart_existence" not in (
        certificate.blocking_obligations
    )
    assert "total_collision_stop_chart_existence" not in (
        theorem.critical_unaudited_analytic_lemma_ids
    )
    assert "three_body_painleve_no_noncollision_singularities" not in (
        certificate.blocking_obligations
    )
    assert "total_collision_requires_zero_angular_momentum" not in (
        certificate.blocking_obligations
    )
    assert "total_collision_central_configuration_asymptotic" not in (
        certificate.blocking_obligations
    )
    assert "finite_fuchsian_log_stop_chart_for_admissible_entry_data" not in (
        certificate.blocking_obligations
    )
    assert "homothetic_total_collision_stop_chart_existence" not in (
        certificate.blocking_obligations
    )
    assert "cubic_time_total_collision_scaling" not in (
        certificate.blocking_obligations
    )
    assert "all_pair_binary_regularization" not in certificate.blocking_obligations
    assert "binary_accumulation_implies_total_collision" not in (
        certificate.blocking_obligations
    )
    assert "binary_collision_isolation" not in certificate.blocking_obligations
    assert "compact_collision_free_taylor_cover" not in certificate.blocking_obligations
    assert "finite_chart_chain_concatenation" not in certificate.blocking_obligations
    assert "target_or_stop_dichotomy" not in certificate.blocking_obligations
    assert "certificate_search_completeness_for_point_inputs" not in (
        certificate.blocking_obligations
    )
    assert "fair_adaptive_chart_search" not in certificate.blocking_obligations
    assert "finite_target_certificate_search_completeness" not in (
        certificate.blocking_obligations
    )
    assert "point_input_finite_target_certificate_search" not in (
        certificate.blocking_obligations
    )
    assert "recursive_set_valued_branch_partition_consumption" in (
        certificate.blocking_obligations
    )
    assert "event_order_partition_consumption_theorem" in (
        certificate.blocking_obligations
    )
    assert "finite_time_loop_budget_elimination" not in certificate.blocking_obligations
    details = {detail.requirement: detail for detail in certificate.missing_requirement_details}
    assert details["compact_sundman_global_series"].witness_field == (
        "OpenTimeLocallyFiniteAtlasTheoremCertificate"
    )
    assert set(details["compact_sundman_global_series"].blocking_obligations) >= {
        "recursive_set_valued_branch_partition_consumption",
        "event_order_partition_consumption_theorem",
        "set_valued_constructor_branch_event_completeness",
    }
    assert set(details["compact_sundman_global_series"].blocking_obligations) >= set(
        expected_critical_blockers
    )
    assert "certificate_search_completeness_for_point_inputs" not in (
        details["compact_sundman_global_series"].blocking_obligations
    )
    assert "fair_adaptive_chart_search" not in (
        details["compact_sundman_global_series"].blocking_obligations
    )
    assert "finite_target_certificate_search_completeness" not in (
        details["compact_sundman_global_series"].blocking_obligations
    )
    assert "point_input_finite_target_certificate_search" not in (
        details["compact_sundman_global_series"].blocking_obligations
    )
    assert details["arbitrary_positive_masses"].witness_field == (
        "OpenTimeLocallyFiniteAtlasTheoremCertificate."
        "arbitrary_finite_target_completeness_certified"
    )
    assert set(details["arbitrary_positive_masses"].blocking_obligations) >= {
        "recursive_set_valued_branch_partition_consumption",
        "event_order_partition_consumption_theorem",
        "set_valued_constructor_branch_event_completeness",
    }
    assert set(details["arbitrary_positive_masses"].blocking_obligations) >= set(
        expected_critical_blockers
    )
    assert "certificate_search_completeness_for_point_inputs" not in (
        details["arbitrary_positive_masses"].blocking_obligations
    )
    assert "fair_adaptive_chart_search" not in (
        details["arbitrary_positive_masses"].blocking_obligations
    )
    assert "finite_target_certificate_search_completeness" not in (
        details["arbitrary_positive_masses"].blocking_obligations
    )
    assert "point_input_finite_target_certificate_search" not in (
        details["arbitrary_positive_masses"].blocking_obligations
    )

    atlas_details = {
        detail.requirement: detail
        for detail in atlas_certificate.missing_requirement_details
    }
    assert atlas_details["regularized_locally_finite_atlas"].witness_field == (
        "OpenTimeLocallyFiniteAtlasTheoremCertificate"
    )
    assert "independent_chart_verifier" in (
        atlas_details["regularized_locally_finite_atlas"].blocking_obligations
    )
    assert atlas_details["collision_semantics"].required == (
        "binary collisions are regularized and total collisions stop under the maximal-classical policy"
    )


def test_closed_form_audit_consumes_constructor_checked_prefix_without_proof_promotion():
    masses, positions, velocities = _open_time_spatial_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        checked_prefix_strategy="ordinary_taylor",
        **_open_time_solver_options(),
    )
    atlas_certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
    )

    assert theorem.independent_chart_verifier_certified
    assert theorem.independent_chart_verifier_certificate.certified
    assert atlas_certificate.status == "incomplete"
    assert not atlas_certificate.proof_certified
    assert "independent_chart_verifier" not in (
        atlas_certificate.blocking_obligations
    )
    assert "audited_or_machine_checked_open_time_atlas_proof" in (
        atlas_certificate.blocking_obligations
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        atlas_certificate.blocking_obligations
    )
    assert not any(
        str(obligation).startswith("independent_chart_verifier_arithmetic:")
        for obligation in atlas_certificate.blocking_obligations
    )
    assert not any(
        str(obligation).startswith("independent_chart_verifier_arithmetic:")
        for obligation in atlas_certificate.blocking_obligations
    )


def test_pointwise_closed_form_route_requires_soundness_and_enumeration_gates():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
    )

    assert theorem.proof_certified
    assert certificate.status == "incomplete"
    assert not certificate.proof_certified
    assert "set_valued_constructor_branch_event_completeness" not in (
        certificate.blocking_obligations
    )
    assert "certificate_language_soundness" in certificate.blocking_obligations
    assert "computable_atlas_certificate_enumeration" in (
        certificate.blocking_obligations
    )


def test_pointwise_closed_form_route_requires_explicit_generalized_stop_soundness():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    soundness = _complete_certificate_language_soundness(
        total_stop_sound=False,
        fuchsian_stop_sound=True,
        generalized_fuchsian_stop_sound=False,
    )
    enumeration = _complete_computable_atlas_certificate_enumeration()
    certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
        certificate_language_soundness_certificate=soundness,
        computable_atlas_enumeration_certificate=enumeration,
    )

    assert not soundness.proof_certified
    assert soundness.missing_obligations == ("generalized_fuchsian_stop_sound",)
    assert not certificate.proof_certified
    assert "certificate_language_soundness:generalized_fuchsian_stop_sound" in (
        certificate.blocking_obligations
    )


def test_certificate_language_soundness_rejects_aggregate_total_stop_shortcut():
    soundness = certify_certificate_language_soundness(
        ordinary_taylor_sound=True,
        levi_civita_sound=True,
        spatial_ks_sound=True,
        total_stop_sound=True,
        transition_sound=True,
        branch_union_sound=True,
        chart_chain_sound=True,
        verifier_kernel_sound=True,
        proof_grade_arithmetic_backend_sound=True,
    )

    assert not soundness.proof_certified
    assert soundness.missing_obligations == (
        "fuchsian_stop_sound",
        "generalized_fuchsian_stop_sound",
    )


def test_pointwise_closed_form_route_requires_proof_grade_arithmetic_soundness():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    soundness = _complete_certificate_language_soundness(
        proof_grade_arithmetic_backend_sound=False,
    )
    certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
        certificate_language_soundness_certificate=soundness,
        computable_atlas_enumeration_certificate=(
            _complete_computable_atlas_certificate_enumeration()
        ),
    )

    assert not soundness.proof_certified
    assert soundness.missing_obligations == (
        "proof_grade_arithmetic_backend_sound",
    )
    assert not certificate.proof_certified
    assert (
        "certificate_language_soundness:proof_grade_arithmetic_backend_sound"
        in certificate.blocking_obligations
    )


def test_certificate_language_soundness_derives_from_checker_kernel_manifest():
    kernel = certify_certificate_checker_kernel_support(
        proof_grade_arithmetic_backend_certificate=(
            certify_rational_interval_arithmetic_backend_soundness()
        ),
    )
    soundness = derive_certificate_language_soundness_from_checker_kernel(kernel)

    assert kernel.proof_certified
    assert soundness.proof_certified
    assert soundness.missing_obligations == ()
    assert three_body_api.CertificateCheckerKernelSupportCertificate is not None
    assert three_body_api.ProofGradeArithmeticBackendCertificate is not None
    assert three_body_api.certify_certificate_checker_kernel_support
    assert three_body_api.certify_rational_interval_arithmetic_backend_soundness
    assert three_body_api.derive_certificate_language_soundness_from_checker_kernel


def test_certificate_language_soundness_derivation_keeps_arithmetic_gate_visible():
    kernel = certify_certificate_checker_kernel_support(
        proof_grade_arithmetic_backend_sound=False,
    )
    soundness = derive_certificate_language_soundness_from_checker_kernel(kernel)

    assert not kernel.proof_certified
    assert kernel.missing_obligations == ("proof_grade_arithmetic_backend_sound",)
    assert not soundness.proof_certified
    assert soundness.missing_obligations == (
        "proof_grade_arithmetic_backend_sound",
    )


def test_checker_kernel_rejects_raw_arithmetic_soundness_flag():
    try:
        certify_certificate_checker_kernel_support(
            proof_grade_arithmetic_backend_sound=True,
        )
    except TypeError as error:
        assert "ProofGradeArithmeticBackendCertificate" in str(error)
    else:
        raise AssertionError("raw arithmetic soundness flag was accepted")


def test_pointwise_closed_form_route_requires_generalized_fuchsian_enumeration_data():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    enumeration = _complete_computable_atlas_certificate_enumeration(
        generalized_fuchsian_exponent_data_enumerated=False,
        fuchsian_selector_constants_enumerated=False,
        cauchy_majorants_enumerated=False,
    )
    certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
        certificate_language_soundness_certificate=(
            _complete_certificate_language_soundness()
        ),
        computable_atlas_enumeration_certificate=enumeration,
    )

    assert not enumeration.proof_certified
    assert enumeration.missing_obligations == (
        "generalized_fuchsian_exponent_data_enumerated",
        "fuchsian_selector_constants_enumerated",
        "cauchy_majorants_enumerated",
    )
    assert not certificate.proof_certified
    assert (
        "computable_atlas_certificate_enumeration:"
        "generalized_fuchsian_exponent_data_enumerated"
    ) in certificate.blocking_obligations
    assert (
        "computable_atlas_certificate_enumeration:"
        "fuchsian_selector_constants_enumerated"
    ) in certificate.blocking_obligations
    assert (
        "computable_atlas_certificate_enumeration:cauchy_majorants_enumerated"
        in certificate.blocking_obligations
    )


def test_computable_atlas_enumeration_derives_from_pointwise_theorem():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    enumeration = (
        derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
            theorem,
        )
    )
    certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
        certificate_language_soundness_certificate=(
            _complete_certificate_language_soundness()
        ),
        computable_atlas_enumeration_certificate=enumeration,
    )

    assert theorem.proof_certified
    assert enumeration.pointwise_theorem_derived
    assert enumeration.proof_certified
    assert enumeration.finite_target_chart_families == (
        "ordinary_taylor",
        "planar_levi_civita_binary",
        "spatial_ks_binary",
        "total_collision_stop",
    )
    assert enumeration.finite_target_allowed_outcomes == (
        "finite_atlas_reaches_target",
        "unselected_total_collision_before_target",
    )
    assert "computable_atlas_certificate_enumeration" not in (
        certificate.blocking_obligations
    )


def test_computable_atlas_enumeration_derivation_requires_exact_computable_input():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    interval_theorem = replace(
        theorem,
        input_model="interval_box_positive_mass_noncollision",
    )
    enumeration = (
        derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
            interval_theorem,
        )
    )

    assert interval_theorem.proof_certified
    assert not enumeration.proof_certified
    assert not enumeration.pointwise_theorem_derived
    assert enumeration.missing_obligations == (
        "chart_family_words_enumerated",
        "pair_labels_enumerated",
        "rational_domains_enumerated",
        "truncation_orders_enumerated",
        "rational_or_interval_coefficients_enumerated",
        "rational_tail_budgets_enumerated",
        "generalized_fuchsian_exponent_data_enumerated",
        "fuchsian_selector_constants_enumerated",
        "cauchy_majorants_enumerated",
        "transition_witnesses_enumerated",
        "collision_policy_data_enumerated",
        "independent_checker_dovetailed",
        "dovetailing_fairness_certified",
        "finite_target_query_terminates_certified",
    )


def test_pointwise_closed_form_route_certifies_regularized_atlas_for_computable_inputs():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    soundness = _complete_certificate_language_soundness()
    enumeration = _complete_computable_atlas_certificate_enumeration()
    certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
        certificate_language_soundness_certificate=soundness,
        computable_atlas_enumeration_certificate=enumeration,
    )

    assert theorem.proof_certified
    assert soundness.proof_certified
    assert enumeration.proof_certified
    assert certificate.proof_certified
    assert certificate.status == "certified"
    assert certificate.closed_form_certificate.status == (
        "certified_pointwise_regularized_atlas_route"
    )
    assert certificate.closed_form_certificate.atlas_route_certified
    assert "set_valued_constructor_branch_event_completeness" not in (
        certificate.blocking_obligations
    )
    assert certificate.blocking_obligations == ()
    details = {
        detail.requirement: detail
        for detail in certificate.requirement_statuses
    }
    assert details["regularized_locally_finite_atlas"].witness_field == (
        "PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate"
    )


def test_set_valued_constructor_regularized_atlas_route_has_explicit_status():
    theorem = OpenTimeLocallyFiniteAtlasTheoremCertificate(
        finite_target_certificate=SimpleNamespace(
            certified=True,
            outcome_id="finite_chart_chain_reaches_target",
        ),
        compact_interval_certificate=SimpleNamespace(
            certified=True,
            proof_certified=True,
        ),
        exhaustion_family_certificate=SimpleNamespace(
            certified=True,
            proof_certified=True,
        ),
        countable_exhaustion_certificate=SimpleNamespace(
            proof_certified=True,
        ),
        endpoint_regime_partition_required=False,
        obligations=(
            TheoremPipelineObligation(
                obligation="finite_target_atlas_or_stop_theorem",
                certified=True,
                source="test",
                detail="finite chart chain reaches target",
            ),
        ),
        finite_target_completeness_certificate=SimpleNamespace(
            certified=True,
            proof_certified=True,
            missing_obligations=(),
            unaudited_analytic_lemma_ids=(),
            critical_unaudited_analytic_lemma_ids=(),
            analytic_lemma_audit_blockers=(),
        ),
        independent_chart_verifier_certified=True,
        independent_chart_verifier_certificate=SimpleNamespace(certified=True),
    )
    certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
    )

    assert theorem.proof_certified
    assert certificate.proof_certified
    assert certificate.closed_form_certificate.status == (
        "certified_set_valued_constructor_regularized_atlas_route"
    )
    assert certificate.closed_form_certificate.route_summary == (
        "set-valued constructor regularized locally finite atlas route certified for interval boxes"
    )


def test_pointwise_regularized_atlas_closed_form_theorem_feeds_audit_directly():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    soundness = _complete_certificate_language_soundness()
    enumeration = _complete_computable_atlas_certificate_enumeration()
    policy = certify_maximal_classical_total_collision_policy(
        pointwise_open_time_theorem=theorem,
    )
    pointwise_closed_form = certify_pointwise_regularized_atlas_closed_form_theorem(
        pointwise_open_time_theorem=theorem,
        certificate_language_soundness=soundness,
        computable_certificate_enumeration=enumeration,
        maximal_classical_total_collision_policy=policy,
    )
    certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise_closed_form,
    )

    assert policy.proof_certified
    assert pointwise_closed_form.proof_certified
    assert pointwise_closed_form.chart_primitives == (
        "ordinary_taylor",
        "planar_levi_civita_binary",
        "spatial_ks_binary",
        "generalized_fuchsian_puiseux_log_total_stop",
    )
    assert pointwise_closed_form.chart_primitive_scope_certified
    assert pointwise_closed_form.finite_target_certificate_outcomes == (
        "finite_ordinary_lc_ks_chart_chain_reaches_target",
        "finite_ordinary_lc_ks_total_stop_chain_certifies_first_unselected_total_collision",
    )
    assert pointwise_closed_form.finite_target_outcome_scope_certified
    assert not pointwise_closed_form.endpoint_regime_partition_required
    assert "generalized Fuchsian/Puiseux-log total-stop certificates" in (
        pointwise_closed_form.proof_sketch
    )
    assert "Endpoint classification" in pointwise_closed_form.proof_sketch
    assert all(
        isinstance(obligation, TheoremPipelineObligation)
        for obligation in pointwise_closed_form.obligations
    )
    assert pointwise_closed_form.missing_obligations == ()
    assert certificate.proof_certified
    assert certificate.closed_form_certificate.status == (
        "certified_pointwise_regularized_atlas_route"
    )
    details = {
        detail.requirement: detail
        for detail in certificate.requirement_statuses
    }
    assert details["regularized_locally_finite_atlas"].witness_field == (
        "PointwiseRegularizedAtlasClosedFormTheoremCertificate"
    )


def test_pointwise_regularized_atlas_closed_form_theorem_requires_declared_scope():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    soundness = _complete_certificate_language_soundness()
    enumeration = _complete_computable_atlas_certificate_enumeration()
    policy = certify_maximal_classical_total_collision_policy(
        pointwise_open_time_theorem=theorem,
    )
    pointwise_closed_form = replace(
        certify_pointwise_regularized_atlas_closed_form_theorem(
            pointwise_open_time_theorem=theorem,
            certificate_language_soundness=soundness,
            computable_certificate_enumeration=enumeration,
            maximal_classical_total_collision_policy=policy,
        ),
        chart_primitives=("ordinary_taylor",),
        endpoint_regime_partition_required=True,
    )

    assert not pointwise_closed_form.proof_certified
    assert "pointwise_closed_form_chart_primitives" in (
        pointwise_closed_form.missing_obligations
    )
    assert "endpoint_regime_partition_not_required" in (
        pointwise_closed_form.missing_obligations
    )


def test_pointwise_closed_form_public_api_exports_are_available():
    assert (
        three_body_api.PointwiseRegularizedAtlasClosedFormTheoremCertificate
        is not None
    )
    assert three_body_api.MaximalClassicalTotalCollisionPolicyCertificate is not None
    assert three_body_api.certify_pointwise_regularized_atlas_closed_form_theorem
    assert three_body_api.certify_maximal_classical_total_collision_policy
    assert three_body_api.certify_certificate_language_soundness
    assert three_body_api.certify_computable_atlas_certificate_enumeration


def test_pointwise_regularized_atlas_closed_form_theorem_requires_maximal_policy():
    theorem = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    soundness = _complete_certificate_language_soundness()
    enumeration = _complete_computable_atlas_certificate_enumeration()
    selected_policy = certify_maximal_classical_total_collision_policy(
        policy_id="selected_identity_selector",
        pointwise_open_time_theorem=theorem,
    )
    pointwise_closed_form = certify_pointwise_regularized_atlas_closed_form_theorem(
        pointwise_open_time_theorem=theorem,
        certificate_language_soundness=soundness,
        computable_certificate_enumeration=enumeration,
        maximal_classical_total_collision_policy=selected_policy,
    )
    certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise_closed_form,
    )

    assert not selected_policy.proof_certified
    assert not pointwise_closed_form.proof_certified
    assert "maximal_classical_total_collision_policy" in (
        pointwise_closed_form.missing_obligations
    )
    assert not certificate.proof_certified
    assert "maximal_classical_total_collision_policy" in (
        certificate.blocking_obligations
    )


def test_closed_form_audit_consumes_search_refinement_evidence_without_overclaiming():
    masses, positions, velocities = _open_time_spatial_initial_data()
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
        checked_prefix_strategy="ordinary_taylor",
        certificate_search_recursive_branch_refinement_certificate=(
            branch_refinement
        ),
        certificate_search_event_order_refinement_certificate=event_refinement,
        **_open_time_solver_options(),
    )
    atlas_certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
    )

    assert theorem.independent_chart_verifier_certified
    assert theorem.finite_target_completeness_certificate is not None
    assert (
        theorem.finite_target_completeness_certificate
        .certificate_search_completeness
        .certified
    )
    assert "recursive_set_valued_branch_partition_consumption" not in (
        atlas_certificate.blocking_obligations
    )
    assert "event_order_partition_consumption_theorem" not in (
        atlas_certificate.blocking_obligations
    )
    assert "independent_chart_verifier" not in (
        atlas_certificate.blocking_obligations
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        atlas_certificate.blocking_obligations
    )
    assert not any(
        str(obligation).startswith("independent_chart_verifier_arithmetic:")
        for obligation in atlas_certificate.blocking_obligations
    )
    assert "audited_or_machine_checked_open_time_atlas_proof" in (
        atlas_certificate.blocking_obligations
    )
    assert not atlas_certificate.proof_certified


def test_closed_form_audit_consumes_spatial_ks_checked_prefix_without_overclaiming():
    masses, positions, velocities = _open_time_spatial_ks_initial_data()
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
    atlas_certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
    )

    assert theorem.independent_chart_verifier_certified
    assert theorem.independent_chart_verifier_certificate.certified
    assert theorem.independent_chart_verifier_certificate.spatial_ks_binary_chart_count == 1
    assert atlas_certificate.status == "incomplete"
    assert not atlas_certificate.proof_certified
    assert "independent_chart_verifier" not in (
        atlas_certificate.blocking_obligations
    )
    assert "audited_or_machine_checked_open_time_atlas_proof" in (
        atlas_certificate.blocking_obligations
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        atlas_certificate.blocking_obligations
    )


def test_closed_form_audit_consumes_supplied_spatial_ordinary_ks_checked_prefix():
    masses, positions, velocities, atlas = _open_time_spatial_ordinary_ks_handoff_data()
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
        **_open_time_solver_options(),
    )
    atlas_certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
    )

    assert theorem.independent_chart_verifier_certified
    assert theorem.independent_chart_verifier_certificate.certified
    assert theorem.independent_chart_verifier_certificate.ordinary_taylor_chart_count == 2
    assert theorem.independent_chart_verifier_certificate.spatial_ks_binary_chart_count == 1
    assert theorem.independent_chart_verifier_arithmetic_certified
    assert theorem.independent_chart_verifier_arithmetic_blockers == ()
    assert atlas_certificate.status == "incomplete"
    assert not atlas_certificate.proof_certified
    assert "independent_chart_verifier" not in (
        atlas_certificate.blocking_obligations
    )
    assert "audited_or_machine_checked_open_time_atlas_proof" in (
        atlas_certificate.blocking_obligations
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        atlas_certificate.blocking_obligations
    )
    assert not any(
        str(obligation).startswith("independent_chart_verifier_arithmetic:")
        for obligation in atlas_certificate.blocking_obligations
    )


def test_closed_form_audit_consumes_supplied_branch_union_checked_prefix():
    masses, positions, velocities, partition, atlas = (
        _open_time_spatial_close_pair_branch_union_atlas_data()
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
        **_open_time_solver_options(),
    )
    atlas_certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
    )

    assert theorem.independent_chart_verifier_certified
    assert theorem.independent_chart_verifier_certificate.certified
    assert theorem.independent_chart_verifier_certificate.checked_branch_union_count == 1
    assert theorem.independent_chart_verifier_certificate.checked_chart_chain_count == (
        len(partition.branches)
    )
    assert atlas_certificate.status == "incomplete"
    assert not atlas_certificate.proof_certified
    assert "independent_chart_verifier" not in (
        atlas_certificate.blocking_obligations
    )
    assert "audited_or_machine_checked_open_time_atlas_proof" in (
        atlas_certificate.blocking_obligations
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        atlas_certificate.blocking_obligations
    )


def test_closed_form_audit_consumes_stratified_branch_union_checked_prefix():
    masses, positions, velocities, partition, atlas = (
        _open_time_spatial_close_pair_branch_union_atlas_data()
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
        **_open_time_solver_options(),
    )
    atlas_certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
    )
    branch_union_result = (
        theorem.independent_chart_verifier_certificate.branch_union_results[-1]
    )

    assert theorem.independent_chart_verifier_certified
    assert theorem.independent_chart_verifier_certificate.certified
    assert branch_union_result.union_type == "finite_time_stratified_branch_union"
    assert branch_union_result.missing_obligations == ()
    assert atlas_certificate.status == "incomplete"
    assert not atlas_certificate.proof_certified
    assert "independent_chart_verifier" not in (
        atlas_certificate.blocking_obligations
    )
    assert "audited_or_machine_checked_open_time_atlas_proof" in (
        atlas_certificate.blocking_obligations
    )
    assert "set_valued_constructor_branch_event_completeness" in (
        atlas_certificate.blocking_obligations
    )


def test_closed_form_audit_rejects_raw_boolean_constructor_theorem_witness():
    with pytest.raises(TypeError, match="constructor-derived certificate"):
        certify_general_closed_form_solution_target(
            "sundman global series",
            general_theorem_certificate=True,
        )


def test_binary_collision_continuation_witness_reports_granular_blockers():
    binary = certify_binary_collision_continuation_witness(
        regularized_pairs=((0, 1), (1, 2)),
        pair_regularization_charts_certified=True,
        branch_atlas_certified=False,
        projection_back_to_newtonian_certified=True,
        regularized_time_parameter_certified=False,
        witness_source="partial_binary_collision_test",
    )

    assert not binary.all_required_pairs_covered
    assert not binary.all_binary_pairs_regularized_certified
    assert not binary.binary_collision_continuation_certified
    assert binary.missing_binary_obligations == (
        "binary_required_pairs_covered",
        "binary_branch_atlas",
        "binary_collision_time_parameter",
        "binary_collision_continuation",
    )
    details = {detail.obligation: detail for detail in binary.binary_obligation_details}
    assert details["binary_required_pairs_covered"].observed == (
        "regularized_pairs=((0, 1), (1, 2))"
    )
    assert details["binary_branch_atlas"].witness_field == "branch_atlas_certified"


def test_nonzero_angular_triple_exclusion_witness_reports_granular_blockers():
    nonzero = certify_nonzero_angular_momentum_triple_exclusion_witness(
        nonzero_branch_domain_quantified_certified=True,
        centered_angular_momentum_conservation_certified=False,
        triple_collision_zero_angular_momentum_lemma_certified=True,
        positive_lower_bound_predicate_certified=False,
        witness_source="partial_nonzero_angular_triple_exclusion_test",
    )

    assert not nonzero.nonzero_angular_momentum_triple_exclusion_certified
    assert nonzero.missing_nonzero_angular_obligations == (
        "centered_angular_momentum_conservation",
        "nonzero_angular_lower_bound_predicate",
        "nonzero_angular_momentum_triple_exclusion",
    )
    details = {
        detail.obligation: detail
        for detail in nonzero.nonzero_angular_obligation_details
    }
    assert details["centered_angular_momentum_conservation"].required == (
        "centered angular momentum is conserved by the lifted/projected flow"
    )
    assert details["nonzero_angular_lower_bound_predicate"].observed == (
        "positive_lower_bound_predicate_certified=False"
    )


def test_zero_angular_triple_collision_convention_reports_granular_blockers():
    convention = certify_zero_angular_momentum_triple_collision_convention(
        convention_id="unvetted_total_collision_rule",
        regularized_time_parameter_certified=True,
        terminal_collision_value_certified=False,
        continuation_selection_rule_certified=True,
        witness_source="invalid_zero_angular_convention_test",
    )

    assert not convention.convention_id_allowed
    assert not convention.convention_certified
    assert convention.missing_convention_obligations == (
        "zero_angular_triple_convention_id_allowed",
        "zero_angular_triple_terminal_collision_value",
    )
    details = {
        detail.obligation: detail for detail in convention.convention_obligation_details
    }
    assert details["zero_angular_triple_convention_id_allowed"].observed == (
        "convention_id=unvetted_total_collision_rule"
    )
    assert details["zero_angular_triple_terminal_collision_value"].witness_field == (
        "terminal_collision_value_certified"
    )


def test_regularized_second_jet_triple_convention_requires_branch_rule():
    incomplete = certify_zero_angular_momentum_triple_collision_convention(
        convention_id="regularized_second_jet_branch",
        regularized_time_parameter_certified=True,
        terminal_collision_value_certified=True,
        continuation_selection_rule_certified=False,
        witness_source="regularized_second_jet_incomplete_test",
    )
    complete = certify_zero_angular_momentum_triple_collision_convention(
        convention_id="regularized_second_jet_branch",
        regularized_time_parameter_certified=True,
        terminal_collision_value_certified=True,
        continuation_selection_rule_certified=True,
        witness_source="regularized_second_jet_complete_test",
    )

    assert incomplete.convention_id_allowed
    assert not incomplete.convention_certified
    assert incomplete.missing_convention_obligations == (
        "zero_angular_triple_continuation_selection_rule",
    )
    incomplete_details = {
        detail.obligation: detail
        for detail in incomplete.convention_obligation_details
    }
    assert "regularized second-jet branch datum" in (
        incomplete_details[
            "zero_angular_triple_continuation_selection_rule"
        ].required
    )

    assert complete.convention_id_allowed
    assert complete.convention_certified
    assert complete.missing_convention_obligations == ()


def test_collision_continuation_witness_reports_granular_missing_obligations():
    collision = certify_collision_continuation_witness(
        all_binary_pairs_regularized_certified=True,
        binary_projection_back_to_newtonian_certified=True,
        nonzero_angular_momentum_triple_exclusion_certified=True,
        witness_source="partial_collision_test",
    )

    assert not collision.binary_collision_continuation_certified
    assert not collision.triple_collision_continuation_certified
    assert not collision.collision_continuation_certified
    assert collision.missing_collision_obligations == (
        "binary_branch_atlas",
        "binary_collision_time_parameter",
        "zero_angular_momentum_triple_collision_convention",
        "binary_collision_continuation",
        "triple_collision_continuation",
        "collision_continuation",
    )
    details = {detail.obligation: detail for detail in collision.collision_obligation_details}
    assert details["binary_branch_atlas"].required == (
        "Levi-Civita branch atlas covers binary collision branches"
    )
    assert details["zero_angular_momentum_triple_collision_convention"].observed == (
        "zero_angular_momentum_triple_collision_convention_certified=False; "
        "convention=unspecified"
    )


def test_collision_continuation_consumes_typed_binary_blockers():
    binary = certify_binary_collision_continuation_witness(
        regularized_pairs=((0, 1), (0, 2), (1, 2)),
        pair_regularization_charts_certified=True,
        branch_atlas_certified=False,
        projection_back_to_newtonian_certified=True,
        regularized_time_parameter_certified=True,
        witness_source="partial_typed_binary_collision_test",
    )
    collision = certify_collision_continuation_witness(
        all_binary_pairs_regularized_certified=True,
        binary_branch_atlas_certified=True,
        binary_projection_back_to_newtonian_certified=True,
        binary_collision_time_parameter_certified=True,
        binary_collision_continuation_witness=binary,
        nonzero_angular_momentum_triple_exclusion_certified=True,
        zero_angular_momentum_triple_collision_convention_witness=(
            _complete_zero_angular_triple_convention()
        ),
        witness_source="typed_binary_collision_blocker_test",
    )

    assert not collision.binary_collision_continuation_certified
    assert collision.triple_collision_continuation_certified
    assert not collision.collision_continuation_certified
    assert "binary_branch_atlas" in collision.missing_collision_obligations
    assert "binary_collision_continuation" in collision.missing_collision_obligations
    details = {detail.obligation: detail for detail in collision.collision_obligation_details}
    assert details["binary_branch_atlas"].observed == (
        "branch_atlas_certified=False"
    )


def test_collision_continuation_consumes_typed_nonzero_angular_blockers():
    nonzero = certify_nonzero_angular_momentum_triple_exclusion_witness(
        nonzero_branch_domain_quantified_certified=True,
        centered_angular_momentum_conservation_certified=False,
        triple_collision_zero_angular_momentum_lemma_certified=True,
        positive_lower_bound_predicate_certified=True,
        witness_source="partial_typed_nonzero_angular_test",
    )
    collision = certify_collision_continuation_witness(
        binary_collision_continuation_witness=_complete_binary_collision_continuation(),
        nonzero_angular_momentum_triple_exclusion_certified=True,
        nonzero_angular_momentum_triple_exclusion_witness=nonzero,
        zero_angular_momentum_triple_collision_convention_witness=(
            _complete_zero_angular_triple_convention()
        ),
        witness_source="typed_nonzero_angular_blocker_test",
    )

    assert collision.binary_collision_continuation_certified
    assert not collision.triple_collision_continuation_certified
    assert not collision.collision_continuation_certified
    assert "centered_angular_momentum_conservation" in (
        collision.missing_collision_obligations
    )
    assert "nonzero_angular_momentum_triple_exclusion" in (
        collision.missing_collision_obligations
    )
    details = {detail.obligation: detail for detail in collision.collision_obligation_details}
    assert details["nonzero_angular_momentum_triple_exclusion"].observed == (
        "nonzero_angular_momentum_triple_exclusion_certified=False"
    )


def test_collision_continuation_consumes_typed_zero_angular_convention_blockers():
    convention = certify_zero_angular_momentum_triple_collision_convention(
        convention_id="unvetted_total_collision_rule",
        regularized_time_parameter_certified=True,
        terminal_collision_value_certified=True,
        continuation_selection_rule_certified=True,
        witness_source="invalid_collision_convention_test",
    )
    collision = certify_collision_continuation_witness(
        all_binary_pairs_regularized_certified=True,
        binary_branch_atlas_certified=True,
        binary_projection_back_to_newtonian_certified=True,
        binary_collision_time_parameter_certified=True,
        nonzero_angular_momentum_triple_exclusion_certified=True,
        zero_angular_momentum_triple_collision_convention_certified=True,
        triple_collision_continuation_convention="sundman_total_collision_regularization",
        zero_angular_momentum_triple_collision_convention_witness=convention,
        witness_source="typed_invalid_collision_convention_test",
    )

    assert collision.binary_collision_continuation_certified
    assert not collision.triple_collision_continuation_certified
    assert not collision.collision_continuation_certified
    assert collision.effective_triple_collision_continuation_convention == (
        "unvetted_total_collision_rule"
    )
    assert "zero_angular_triple_convention_id_allowed" in (
        collision.missing_collision_obligations
    )
    assert "zero_angular_momentum_triple_collision_convention" in (
        collision.missing_collision_obligations
    )
    details = {detail.obligation: detail for detail in collision.collision_obligation_details}
    assert details["zero_angular_momentum_triple_collision_convention"].observed == (
        "zero_angular_momentum_triple_collision_convention_certified=True; "
        "convention=unvetted_total_collision_rule"
    )


def test_sundman_theorem_witness_consumes_typed_collision_blockers():
    recurrence = _RecurrenceClosure(
        compact_witness=_CompactSundmanWitness(global_series_certified=True),
    )
    scope = certify_general_solution_scope_witness(
        arbitrary_positive_masses_certified=True,
        arbitrary_noncollision_initial_data_certified=True,
        all_real_target_times_certified=True,
        lift_construct_project_verify_certified=True,
        newton_equations_full_interval_certified=True,
        witness_source="complete_scope_for_collision_blocker_test",
    )
    collision = certify_collision_continuation_witness(
        all_binary_pairs_regularized_certified=True,
        binary_projection_back_to_newtonian_certified=True,
        nonzero_angular_momentum_triple_exclusion_certified=True,
        witness_source="partial_collision_for_theorem_test",
    )
    theorem_witness = certify_sundman_general_solution_theorem_witness(
        recurrence_closure=recurrence,
        general_scope_witness=scope,
        collision_witness=collision,
        witness_source="typed_sundman_collision_blocker_test",
    )

    certificate = certify_general_closed_form_solution_target(
        "sundman global series",
        compact_sundman_witness=theorem_witness,
    )

    assert not theorem_witness.global_series_certified
    assert not theorem_witness.collision_continuation_obligations_certified
    assert "binary_branch_atlas" in theorem_witness.missing_global_proof_obligations
    assert "zero_angular_momentum_triple_collision_convention" in (
        theorem_witness.missing_global_proof_obligations
    )
    assert "compact_sundman_global_series" in certificate.missing_requirements
    assert "collision_continuation" in certificate.missing_requirements
    assert "binary_branch_atlas" in certificate.blocking_obligations
    assert "zero_angular_momentum_triple_collision_convention" in certificate.blocking_obligations
    details = {detail.requirement: detail for detail in certificate.missing_requirement_details}
    assert "binary_branch_atlas" in details["compact_sundman_global_series"].blocking_obligations
    assert "triple_collision_continuation" in details["collision_continuation"].blocking_obligations


def test_sundman_theorem_witness_can_certify_top_level_when_all_layers_are_certified():
    recurrence = _RecurrenceClosure(
        compact_witness=_CompactSundmanWitness(global_series_certified=True),
    )
    scope = certify_general_solution_scope_witness(
        arbitrary_positive_masses_certified=True,
        arbitrary_noncollision_initial_data_certified=True,
        all_real_target_times_certified=True,
        lift_construct_project_verify_certified=True,
        newton_equations_full_interval_certified=True,
        witness_source="complete_scope_for_theorem_witness_test",
    )
    collision = certify_collision_continuation_witness(
        binary_collision_continuation_witness=_complete_binary_collision_continuation(),
        nonzero_angular_momentum_triple_exclusion_witness=(
            _complete_nonzero_angular_triple_exclusion()
        ),
        zero_angular_momentum_triple_collision_convention_witness=(
            _complete_zero_angular_triple_convention()
        ),
        witness_source="complete_collision_for_theorem_test",
    )
    theorem_witness = certify_sundman_general_solution_theorem_witness(
        recurrence_closure=recurrence,
        general_scope_witness=scope,
        collision_witness=collision,
        witness_source="typed_sundman_complete_test",
    )

    certificate = certify_general_closed_form_solution_target(
        "sundman global series",
        compact_sundman_witness=theorem_witness,
    )

    assert theorem_witness.global_series_certified
    assert theorem_witness.collision_continuation_obligations_certified
    assert theorem_witness.general_solution_scope_certified
    assert theorem_witness.missing_collision_obligations == ()
    assert theorem_witness.missing_global_proof_obligations == ()
    assert certificate.status == "certified"
    assert certificate.proof_certified
    assert certificate.missing_requirements == ()


def test_general_solution_target_can_be_certified_by_separate_typed_scope_witness():
    compact_witness = _CompactSundmanWitness(
        global_series_certified=True,
        collision_continuation_obligations_certified=True,
    )
    scope = certify_general_solution_scope_witness(
        arbitrary_positive_masses_certified=True,
        arbitrary_noncollision_initial_data_certified=True,
        all_real_target_times_certified=True,
        lift_construct_project_verify_certified=True,
        newton_equations_full_interval_certified=True,
        witness_source="complete_scope_test",
    )

    certificate = certify_general_closed_form_solution_target(
        "sundman",
        compact_sundman_witness=compact_witness,
        general_scope_witness=scope,
    )

    assert certificate.status == "certified"
    assert certificate.proof_certified
    assert certificate.closed_form_certificate.general_solution_certified
    assert certificate.missing_requirements == ()


def test_general_solution_target_can_be_certified_by_full_scope_witness():
    witness = _CompactSundmanWitness(
        global_series_certified=True,
        collision_continuation_obligations_certified=True,
        general_solution_scope_certified=True,
    )

    certificate = certify_general_closed_form_solution_target(
        "sundman",
        compact_sundman_witness=witness,
    )

    assert certificate.status == "certified"
    assert certificate.proof_certified
    assert certificate.closed_form_certificate.general_solution_certified
    assert certificate.missing_requirements == ()
