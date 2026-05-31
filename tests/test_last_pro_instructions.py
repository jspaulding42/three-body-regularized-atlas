from pathlib import Path

import three_body_symmetry as three_body_api

from three_body_symmetry.closed_form import (
    certify_certificate_language_soundness,
    certify_computable_atlas_certificate_enumeration,
    certify_general_closed_form_solution_target,
    certify_maximal_classical_total_collision_policy,
    certify_pointwise_regularized_atlas_closed_form_theorem,
    derive_certificate_language_soundness_from_checker_kernel,
    derive_computable_atlas_certificate_enumeration_from_pointwise_theorem,
)
from three_body_symmetry.certificate_checker import (
    certify_certificate_checker_kernel_support,
    certify_rational_interval_arithmetic_backend_soundness,
)
from three_body_symmetry.finite_target_completeness import (
    certify_finite_target_certificate_search_completeness,
    certify_finite_target_completeness_theorem,
    certify_uniform_margin_branch_refinement_termination,
    certify_uniform_margin_set_valued_constructor_completeness,
    certify_validated_set_valued_constructor_completeness_theorem,
)
from three_body_symmetry.open_time_atlas import (
    certify_pointwise_open_time_locally_finite_atlas_theorem,
)


ROOT = Path(__file__).resolve().parents[1]


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


def test_last_pro_pointwise_closed_form_route_closes_without_interval_box_blocker():
    pointwise_open_time = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    soundness = derive_certificate_language_soundness_from_checker_kernel(
        certify_certificate_checker_kernel_support(
            proof_grade_arithmetic_backend_certificate=(
                certify_rational_interval_arithmetic_backend_soundness()
            ),
        )
    )
    enumeration = derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
        pointwise_open_time,
    )
    certificate = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise_open_time,
        certificate_language_soundness_certificate=soundness,
        computable_atlas_enumeration_certificate=enumeration,
    )

    assert pointwise_open_time.proof_certified
    assert soundness.proof_certified
    assert enumeration.proof_certified
    assert enumeration.pointwise_theorem_derived
    assert certificate.proof_certified
    assert certificate.closed_form_certificate.status == (
        "certified_pointwise_regularized_atlas_route"
    )
    assert "set_valued_constructor_branch_event_completeness" not in (
        certificate.blocking_obligations
    )
    assert certificate.blocking_obligations == ()


def test_last_pro_direct_pointwise_theorem_binds_primitives_outcomes_and_policy():
    pointwise_open_time = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    policy = certify_maximal_classical_total_collision_policy(
        pointwise_open_time_theorem=pointwise_open_time,
    )
    theorem = certify_pointwise_regularized_atlas_closed_form_theorem(
        pointwise_open_time_theorem=pointwise_open_time,
        certificate_language_soundness=derive_certificate_language_soundness_from_checker_kernel(
            certify_certificate_checker_kernel_support(
                proof_grade_arithmetic_backend_certificate=(
                    certify_rational_interval_arithmetic_backend_soundness()
                ),
            )
        ),
        computable_certificate_enumeration=(
            derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
                pointwise_open_time,
            )
        ),
        maximal_classical_total_collision_policy=policy,
    )

    assert theorem.proof_certified
    assert theorem.chart_primitives == (
        "ordinary_taylor",
        "planar_levi_civita_binary",
        "spatial_ks_binary",
        "generalized_fuchsian_puiseux_log_total_stop",
    )
    assert theorem.finite_target_certificate_outcomes == (
        "finite_ordinary_lc_ks_chart_chain_reaches_target",
        "finite_ordinary_lc_ks_total_stop_chain_certifies_first_unselected_total_collision",
    )
    assert not theorem.endpoint_regime_partition_required
    assert policy.proof_certified
    assert (
        three_body_api.PointwiseRegularizedAtlasClosedFormTheoremCertificate
        is not None
    )


def test_last_pro_finish_line_theorem_note_preserves_exact_point_scope():
    theorem_note = (
        ROOT / "docs" / "regularized-locally-finite-atlas-closed-form-theorem.md"
    ).read_text()

    assert "For `d in {2,3}`" in theorem_note
    assert "positive computable masses" in theorem_note
    assert "computable noncollision" in theorem_note
    assert "maximal-classical total-collision stop policy" in theorem_note
    assert "ordinary Taylor charts" in theorem_note
    assert "planar Levi-Civita binary charts" in theorem_note
    assert "spatial KS binary charts" in theorem_note
    assert "generalized Fuchsian/Puiseux-log total-collision stop charts" in (
        theorem_note
    )
    assert "finite ordinary/Levi-Civita/KS chart chain reaches `T`" in (
        theorem_note
    )
    assert "certifies the first unselected total collision" in theorem_note
    assert "before or at `T`" in theorem_note
    assert "Endpoint classification" in theorem_note
    assert "not a theorem prerequisite" in theorem_note
    assert "It does not mean a finite elementary expression" in theorem_note
    assert "arbitrary recursive partition" in theorem_note
    assert "generation remains open" in theorem_note


def test_last_pro_soundness_and_enumeration_require_specific_constructor_gates():
    aggregate_soundness = certify_certificate_language_soundness(
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
    partial_enumeration = _complete_computable_atlas_certificate_enumeration(
        generalized_fuchsian_exponent_data_enumerated=False,
        fuchsian_selector_constants_enumerated=False,
        cauchy_majorants_enumerated=False,
    )

    assert not aggregate_soundness.proof_certified
    assert aggregate_soundness.missing_obligations == (
        "fuchsian_stop_sound",
        "generalized_fuchsian_stop_sound",
    )
    assert not partial_enumeration.proof_certified
    assert partial_enumeration.missing_obligations == (
        "generalized_fuchsian_exponent_data_enumerated",
        "fuchsian_selector_constants_enumerated",
        "cauchy_majorants_enumerated",
    )


def test_last_pro_interval_box_constructor_theorem_stays_separate_and_scoped():
    theorem = certify_finite_target_completeness_theorem(dimension=3)
    raw_search = certify_finite_target_certificate_search_completeness(theorem)

    assert not raw_search.certified
    assert "recursive_set_valued_branch_partition_consumption" in (
        raw_search.missing_obligations
    )
    assert "event_order_partition_consumption_theorem" in (
        raw_search.missing_obligations
    )

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
    assert validated.proof_certified
    assert validated.input_scope_id == "positive_margin_interval_boxes"
    assert not validated.arbitrary_partition_generation_claimed
    assert "arbitrary recursive partition generation remains open" in (
        validated.proof_sketch
    )


def test_last_pro_total_collision_proof_note_and_checker_hygiene_are_locked_down():
    proof_note = (
        ROOT / "docs" / "total-collision-generalized-fuchsian-stop-proof.md"
    ).read_text()
    checker_tests = (ROOT / "tests" / "test_certificate_checker.py").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()

    for section in ("TC1", "TC2", "TC3", "TC4", "TC5", "TC6", "TC7"):
        assert f"## {section}." in proof_note
    assert "Poincare-Dulac resonant polynomial" in proof_note
    assert "(<m,alpha>-alpha_j)c_{j,m,ell}" in proof_note
    assert "W(n,beta) = n + sum_j beta_j alpha_j" in proof_note
    assert "Pi_j:E_j -> Ker(M_j)" in proof_note
    assert "there is no hidden finite row datum" in proof_note
    assert "induction over `(W,row,log-degree)`" in proof_note
    assert "B D + q R_0 <= R_0" in proof_note
    assert "(C_0, Lambda, sigma, p_0, d)" in proof_note
    assert "P(r) = {|tau| <= r_tau" in proof_note
    assert "M_a(rho r)" in proof_note
    assert "||Phi(R_1)-Phi(R_2)|| <= B L_N" in proof_note
    assert "does not require one uniform `W_*` for a whole interval box" in proof_note

    assert "@lru_cache(maxsize=2)" in checker_tests
    assert "_fast_total_collision_generalized_fuchsian_stop_chart_certificate" in (
        checker_tests
    )
    assert "test_fast_reduced_order_generalized_fuchsian_stop_checker_ci_fixture" in (
        checker_tests
    )
    assert checker_tests.count("@pytest.mark.slow") >= 4
    assert "slow: expensive generalized Fuchsian checker paths" in pyproject
