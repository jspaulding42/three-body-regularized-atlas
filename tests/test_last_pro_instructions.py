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
    build_fast_total_collision_generalized_fuchsian_stop_chart_certificate,
    certify_certificate_checker_kernel_support,
    certify_rational_interval_arithmetic_backend_soundness,
)
from three_body_symmetry.finite_target_completeness import (
    SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS,
    certify_finite_target_certificate_search_completeness,
    certify_finite_target_completeness_theorem,
    certify_uniform_margin_branch_refinement_termination,
    certify_uniform_margin_set_valued_constructor_completeness,
    certify_validated_set_valued_constructor_completeness_theorem,
)
from three_body_symmetry.open_time_atlas import (
    certify_pointwise_open_time_locally_finite_atlas_theorem,
)
from three_body_symmetry.public_proof_audit import (
    PUBLIC_REVIEW_ARTIFACT_KIND_MACHINE,
    PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS,
    build_public_tc4_tc6_audit_manifest,
    build_review_ready_total_collision_audit_evidence_bundle,
    certify_public_audit_manifest_resolution,
    certify_public_general_closed_form_solution_target,
    certify_public_regularized_atlas_closed_form_proof,
    certify_public_review_artifact_resolution,
    certify_review_ready_total_collision_audit_package_from_evidence_bundle,
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


def test_last_pro_pointwise_closed_form_route_stays_open_without_audited_lemmas():
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

    assert not pointwise_open_time.proof_certified
    assert "pointwise_open_time_finite_target_theorem_proof" in (
        pointwise_open_time.missing_obligations
    )
    assert soundness.proof_certified
    assert not enumeration.proof_certified
    assert not enumeration.pointwise_theorem_derived
    assert not certificate.proof_certified
    assert certificate.closed_form_certificate.status == (
        "conditional_regularized_atlas_route"
    )
    assert "set_valued_constructor_branch_event_completeness" not in (
        certificate.blocking_obligations
    )
    assert "pointwise_open_time_atlas_proof" in certificate.blocking_obligations
    assert "computable_atlas_certificate_enumeration" in (
        certificate.blocking_obligations
    )

    public_certificate = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise_open_time,
        certificate_language_soundness_certificate=soundness,
        computable_atlas_enumeration_certificate=enumeration,
    )
    assert not public_certificate.internal_proof_certified
    assert not public_certificate.public_proof_certified
    assert "tc4_reduced_hyperbolicity_audited" in (
        public_certificate.public_audit_blockers
    )
    assert three_body_api.PublicGeneralClosedFormSolutionCertificate is not None


def test_last_pro_definition_of_done_is_executable_from_current_constructors():
    finite_target = certify_finite_target_completeness_theorem(dimension=3)
    raw_search = certify_finite_target_certificate_search_completeness(finite_target)
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
    internal_closed_form = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise_open_time,
        certificate_language_soundness_certificate=soundness,
        computable_atlas_enumeration_certificate=enumeration,
    )
    pointwise_closed_form = certify_pointwise_regularized_atlas_closed_form_theorem(
        pointwise_open_time_theorem=pointwise_open_time,
        certificate_language_soundness=soundness,
        computable_certificate_enumeration=enumeration,
        maximal_classical_total_collision_policy=(
            certify_maximal_classical_total_collision_policy(
                pointwise_open_time_theorem=pointwise_open_time,
            )
        ),
    )
    checked_stop_chart = (
        build_fast_total_collision_generalized_fuchsian_stop_chart_certificate()
    )
    resolver = certify_public_review_artifact_resolution(project_root=ROOT)
    bundle = build_review_ready_total_collision_audit_evidence_bundle(
        checked_stop_chart,
    )
    local_tc_package = (
        certify_review_ready_total_collision_audit_package_from_evidence_bundle(
            bundle,
        )
    )
    public_tc_package = (
        certify_review_ready_total_collision_audit_package_from_evidence_bundle(
            bundle,
            public_review_resolution_certificate=resolver,
        )
    )
    raw_string_package = (
        certify_review_ready_total_collision_audit_package_from_evidence_bundle(
            bundle,
            public_review_artifact_ids=(
                PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
            ),
        )
    )
    public_regularized = certify_public_regularized_atlas_closed_form_proof(
        pointwise_closed_form,
        total_collision_audit=public_tc_package,
    )
    public_general = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise_open_time,
        certificate_language_soundness_certificate=soundness,
        computable_atlas_enumeration_certificate=enumeration,
        total_collision_audit=public_tc_package,
    )
    default_public_general = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise_open_time,
        certificate_language_soundness_certificate=soundness,
        computable_atlas_enumeration_certificate=enumeration,
    )
    manifest_resolution = certify_public_audit_manifest_resolution(project_root=ROOT)
    public_manifest = build_public_tc4_tc6_audit_manifest(project_root=ROOT)
    machine_manifest = build_public_tc4_tc6_audit_manifest(
        project_root=ROOT,
        public_review_resolution_certificate=resolver,
    )

    assert not finite_target.proof_certified
    assert not pointwise_open_time.proof_certified
    assert soundness.proof_certified
    assert not enumeration.proof_certified
    assert not internal_closed_form.proof_certified
    assert not pointwise_closed_form.proof_certified
    assert local_tc_package.local_audit_package_certified
    assert not local_tc_package.public_proof_certified
    assert resolver.proof_certified
    assert resolver.artifact_kind == PUBLIC_REVIEW_ARTIFACT_KIND_MACHINE
    assert resolver.machine_checked_public_audit_resolved
    assert all(
        artifact.artifact_exists and artifact.no_placeholder_text
        for artifact in (
            resolver.tc4_artifact,
            resolver.tc5_artifact,
            resolver.tc6_artifact,
        )
    )
    assert public_tc_package.public_proof_certified
    assert public_tc_package.public_review_artifact_manifest_certified
    assert not public_regularized.public_proof_certified
    assert not public_general.public_proof_certified
    assert "internal_general_closed_form_solution_certified" in (
        public_general.public_audit_blockers
    )
    assert raw_string_package.public_review_artifact_manifest_supplied
    assert not raw_string_package.public_proof_certified
    assert "public_review_artifact_verification" in (
        raw_string_package.public_audit_blockers
    )
    assert not default_public_general.internal_proof_certified
    assert not default_public_general.public_proof_certified
    assert "tc4_reduced_hyperbolicity_audited" in (
        default_public_general.public_audit_blockers
    )
    assert not raw_search.certified
    assert raw_search.missing_obligations == (
        "recursive_set_valued_branch_partition_consumption",
        "event_order_partition_consumption_theorem",
        "pointwise_finite_target_theorem_proof_certified",
    )
    assert manifest_resolution.proof_certified
    assert manifest_resolution.fast_ci_artifact_ids
    assert manifest_resolution.slow_ci_artifact_ids
    assert public_manifest["public_closure_status"] == "external_review_open"
    assert machine_manifest["public_closure_status"] == (
        "machine_checked_public_audit_verified"
    )
    assert machine_manifest["public_review_resolution_is_external"] is False


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

    assert not theorem.proof_certified
    assert "pointwise_open_time_atlas_proof" in theorem.missing_obligations
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
    assert not policy.proof_certified
    assert (
        three_body_api.PointwiseRegularizedAtlasClosedFormTheoremCertificate
        is not None
    )


def test_last_pro_finish_line_theorem_note_preserves_exact_point_scope():
    theorem_note = (
        ROOT / "docs" / "regularized-locally-finite-atlas-closed-form-theorem.md"
    ).read_text()
    readme = (ROOT / "README.md").read_text()
    last_pro = (ROOT / "LAST_PRO_INSTRUCTIONS.md").read_text()

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
    assert "proof-grade total-collision entry/stop semantics" not in readme
    assert "regularized-atlas route is internally proof-certified" not in readme
    assert "regularized-atlas route is internally assembled but not proof-certified" in (
        readme
    )
    assert "public proof" in readme
    assert "machine-checked TC4-TC6 public-audit artifact" in readme
    assert "independent external public-review artifacts remain a separate" in (
        readme
    )
    assert "local_audit_package_certified" in readme
    assert "machine_checked_public_audit_verified" in readme
    assert "machine_checked_public_audit_resolved=True" in readme
    assert "public_review_artifact_kind" in readme
    assert "public_review_resolved_by_artifact_resolution" in readme
    assert "public_review_resolution_is_external" in readme
    assert "public_review_artifact_manifest_supplied" in readme
    assert "public_review_artifact_manifest_certified" in readme
    assert "certify_review_ready_total_collision_audit_package" in readme
    assert "public_review_artifact_ids" in readme
    assert "public_review_resolution_certificate" in readme
    assert "conflicting legacy/neutral inputs" in readme
    assert "placeholder external artifact ids" in readme
    assert "independent external review" in readme
    assert "mislabeled as external review" in readme
    assert "concrete TC4-TC6 proof-reference manifest" in readme
    assert "concrete spectrum regression artifact ids" in readme
    assert "constructor-backed stable-selector/Fuchsian-log artifact ids" in readme
    assert "rational-interval backend constructor id" in readme
    assert "concrete generalized-Fuchsian checker regression artifact ids" in readme
    assert "resolves each required proof-reference id" in readme
    assert "expected total-stop checker artifact manifest" in readme
    assert "top-level total-stop checker artifact ids" in readme
    assert "actually covered by the fast and slow CI selections" in readme
    assert "auditable references" in readme
    assert "implementation theorem" in readme
    assert "finite Taylor-model decision arrangements" in readme
    assert "Rational decision constructor update" in readme
    assert "Rational arrangement constructor update" in readme
    assert "finite_rational_decision_stratification_interval_boxes" in readme
    assert "finite_rational_decision_arrangement_interval_boxes" in readme
    assert "quadratic double-root" in readme
    assert "computed polynomial-root" in readme
    assert "matching Taylor-model root brackets are grouped" in readme
    assert "overlapping nonmatching brackets are rejected" in readme
    assert "every source type listed in `SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS`" in readme
    assert "2D/3D halfspace grammars" in readme
    assert "Mixed supported branch/event constructors retain separate" in readme
    assert "constructor-derived branch and event-order input scope ids" in readme
    assert "LAST_PRO completion audit" in last_pro
    assert "same concrete TC4-TC6 section-reference" in last_pro
    assert "concrete machine-check\n   regression artifact ids" in last_pro
    assert "constructor-backed artifact ids" in last_pro
    assert "certify_rational_interval_arithmetic_backend_soundness" in last_pro
    assert "concrete generalized-Fuchsian checker\n   regression artifact ids" in (
        last_pro
    )
    assert "resolves the required TC4-TC6 proof-reference strings" in last_pro
    assert "top-level total-stop checker artifact ids" in last_pro
    assert "machine_checked_public_audit line-audit artifacts" in last_pro
    assert "public_review_artifact_kind=\"machine_checked_public_audit\"" in (
        last_pro
    )
    assert "public_review_resolved_by_artifact_resolution=True" in last_pro
    assert "public_review_resolution_is_external=False" in last_pro
    assert "public_review_artifact_manifest_supplied" in last_pro
    assert "public_review_artifact_manifest_certified" in last_pro
    assert "certify_review_ready_total_collision_audit_package" in last_pro
    assert "public_review_artifact_ids" in last_pro
    assert "public_review_resolution_certificate" in last_pro
    assert "conflicting values" in last_pro
    assert "required TC4/TC5\n  artifact manifests are covered" in last_pro
    assert "split between the fast non-slow checker\n  target" in last_pro
    assert "quadratic double\n                roots" in last_pro
    assert "grouped computed polynomial-root grammar" in last_pro
    assert "rational decisions are supported" in last_pro
    assert "finite rational arrangements" in last_pro
    assert "denominator is interval-certified away from zero" in last_pro
    assert "finite-target regressions now exercise every source type" in last_pro
    assert "2D/3D\n   affine halfspace arrangement grammars" in last_pro
    assert "Mixed branch/event constructor pairs now preserve exact" in last_pro
    assert "constructor input scope ids resolved from the actual recursive" in (
        last_pro
    )
    assert "Task 1, public proof audit objects:" in last_pro
    assert "`local_audit_package_certified`" in last_pro
    assert "machine-check public-audit artifact resolver" in last_pro
    assert "independent external review remains a separate" in last_pro
    assert "placeholder external artifact ids" in last_pro
    assert "production complete-local-audit constructor is implemented" in last_pro
    assert "Task 6, API ergonomics:" in last_pro
    assert "The earlier \"missing item A-F\" list is therefore no longer" in (
        last_pro
    )
    assert "### Missing item A:" not in last_pro
    assert "### Task 1:" not in last_pro
    assert "README.md` still says" not in last_pro


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

    assert not scoped.proof_certified
    assert not validated.proof_certified
    assert "pointwise_finite_target_theorem_proof_certified" in (
        scoped.missing_obligations
    )
    assert validated.input_scope_id == "positive_margin_interval_boxes"
    assert not validated.arbitrary_partition_generation_claimed
    assert "arbitrary recursive partition generation remains open" in (
        validated.proof_sketch
    )
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "RationalDecisionStratification"
    ] == (
        "finite_rational_decision_interval_inputs_with_denominator_exclusion"
    )
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "RationalDecisionArrangement"
    ] == (
        "finite_rational_decision_arrangement_interval_inputs_with_denominator_exclusion"
    )
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "AffineDecisionArrangement"
    ] == "finite_affine_decision_arrangement_interval_inputs"
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "PolynomialDecisionArrangement"
    ] == "finite_polynomial_decision_arrangement_interval_inputs"
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "SturmPolynomialDecisionArrangement"
    ] == "finite_sturm_polynomial_decision_arrangement_interval_inputs"
    assert SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS[
        "TaylorModelDecisionArrangement"
    ] == (
        "finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate"
    )
    assert "finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate" in (
        ROOT / "LAST_PRO_INSTRUCTIONS.md"
    ).read_text()


def test_last_pro_total_collision_proof_note_and_checker_hygiene_are_locked_down():
    proof_note = (
        ROOT / "docs" / "total-collision-generalized-fuchsian-stop-proof.md"
    ).read_text()
    checker_tests = (ROOT / "tests" / "test_certificate_checker.py").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()
    fast_ci = (ROOT / "scripts" / "fast_ci.py").read_text()
    slow_checker = (ROOT / "scripts" / "slow_certificate_checker.py").read_text()

    for section in ("TC1", "TC2", "TC3", "TC4", "TC5", "TC6", "TC7"):
        assert f"## {section}." in proof_note
    assert "centered Sundman inequality" in proof_note
    assert "|L|^2\n = |sum_i m_i q_i wedge v_i|^2" in proof_note
    assert "= 2 I K" in proof_note
    assert "The inequality itself does not use TC2" in proof_note
    assert "|L|^2 <= 2 I K" in proof_note
    assert "K = H + U" in proof_note
    assert "|q_i-q_j| >= d_* sqrt(I)" in proof_note
    assert "U <= C_U / sqrt(I)" in proof_note
    assert "I K = I(H+U) <= |H| I + C_U sqrt(I) -> 0" in proof_note
    assert "Since `L` is conserved on the incoming\npunctured branch" in proof_note
    assert "Fix the candidate tight binary `(1,2)`" in proof_note
    assert "I = alpha |x|^2 + beta |y|^2" in proof_note
    assert "|R_x| <= C |x|/|y|^3 = o(|x|^-2)" in proof_note
    assert "|R_y| <= C |x|/|y|^3 = o(|y|^-2)" in proof_note
    assert "z'' = -mu z/|z|^3 + o(|z|^-2)" in proof_note
    assert "dr/dt = -sqrt(2 mu/r) (1+o(1))" in proof_note
    assert "|z(t)| = (9 mu/2)^(1/3) (t_c-t)^(2/3) (1+o(1))" in proof_note
    assert "|x(t)|/|y(t)| -> ((m_1+m_2)/M)^(1/3) > 0" in proof_note
    assert "pair-distance floor used in TC1" in proof_note
    assert "U(rho s) = rho^(-1) U(s)" in proof_note
    assert "K = (1/2) rho_dot^2 + (1/2) rho^2 ||s_dot||_m^2" in proof_note
    assert "d sigma/dt = rho^(-3/2)" in proof_note
    assert "(1/2)(v^2 + ||w||_m^2) - U(s) = rho H -> 0" in proof_note
    assert "I'' = 4H + 2U = 4H + 2 rho^(-1) U(s)" in proof_note
    assert "rho(t) = (9 U(s_*)/2)^(1/3) (t_c-t)^(2/3)" in proof_note
    assert "s_sigma = w" in proof_note
    assert "w_sigma = Pi_s(grad_m U(s))" in proof_note
    assert "int_{sigma_0}^infinity ||w(sigma)||_m^2 d sigma" in proof_note
    assert "Pi_{s_*}(grad_m U(s_*)) = 0" in proof_note
    assert "lambda = U(s_*) > 0" in proof_note
    assert "The indicial equation is not a separate assumption" in proof_note
    assert "d/dt = -(1/(3 tau^2)) d/dtau" in proof_note
    assert "tau^2 S'' + 2 tau S' - 2S = 9 A(S)" in proof_note
    assert "-2C=9A(C)" in proof_note
    assert "(k^2+k-2)V = 9 DA(C)V" in proof_note
    assert "(k+2)(k-1)=9mu" in proof_note
    assert "mu=-2/9" in proof_note
    assert "beta=(m_1m_2+m_1m_3+m_2m_3)/M^2" in proof_note
    assert "{0,0,4/9,-2/9," in proof_note
    assert "finite mass-metric calculation" in proof_note
    assert "DA(C) C = 4/9 C" in proof_note
    assert "DA(C) J C = -2/9 J C" in proof_note
    assert "chi_Lag(mu)" in proof_note
    assert "tr_shape = 2/9" in proof_note
    assert "det_shape = (27 beta - 8)/81" in proof_note
    assert "0<beta<=1/3" in proof_note
    assert "((m_1-m_2)^2+(m_1-m_3)^2+(m_2-m_3)^2)/2" in proof_note
    assert "L_parallel = -2 L_perp" in proof_note
    assert "4/9 < sigma < 32/9" in proof_note
    assert "Euler's quintic to eliminate the third mass" in proof_note
    assert "tr(L_parallel)" in proof_note
    assert "32/9 - sigma" in proof_note
    assert "sigma - 4/9" in proof_note
    assert "positivity of the eliminated mass" in proof_note
    assert "uniformly on the positive-mass ordered Euler family" in proof_note
    assert "no reduced Euler shape mode has" in proof_note
    assert "Poincare-Dulac resonant polynomial" in proof_note
    assert "dy_j/ds = -alpha_j y_j" in proof_note
    assert "stable manifold theorem" in proof_note
    assert "eventual entry time `S_0`" in proof_note
    assert "Lyapunov-Perron fixed point" in proof_note
    assert "finite-order normalizing map" in proof_note
    assert "homological denominator" in proof_note
    assert "Poincare domain" in proof_note
    assert "F_j^{<W}(s)" in proof_note
    assert "Y_j^{W}(s)=y_j(s)-P_j^{<W}(s)" in proof_note
    assert "a_j = lim_{s->infinity}" in proof_note
    assert "forced_resonant_log_polynomial_j" in proof_note
    assert "variation-of-constants estimate" in proof_note
    assert "Induction over `(W,row,log-degree)`" in proof_note
    assert "(<m,alpha>-alpha_j)c_{j,m,ell}" in proof_note
    assert "W(n,beta) = n + sum_j beta_j alpha_j" in proof_note
    assert "Pi_j:E_j -> Ker(M_j)" in proof_note
    assert "denominator/projector audit is finite" in proof_note
    assert "D_*(W_*) = {|<m,alpha>-alpha_j|" in proof_note
    assert "delta_*(W_*) <= min D_*(W_*)" in proof_note
    assert "Rows whose interval denominator contains zero" in proof_note
    assert "Pi_j^2 = Pi_j" in proof_note
    assert "M_j Q_j = I-Pi_j" in proof_note
    assert "Q_j is a right inverse on the range" in proof_note
    assert "not hiding a choice of complement" in proof_note
    assert "there is no hidden finite row datum" in proof_note
    assert "induction over `(W,row,log-degree)`" in proof_note
    assert "finite log-degree bound is explicit" in proof_note
    assert "I(W_*) = {(omega,j)" in proof_note
    assert "dependency graph" in proof_note
    assert "this dependency graph\nis acyclic" in proof_note
    assert "rho(omega,j) = length of the longest directed path" in proof_note
    assert "deg_log H_j(omega) <=" in proof_note
    assert "L_*(W_*)" in proof_note
    assert "No infinite logarithmic\ntower can appear below `W_*`" in proof_note
    assert "selector extraction algorithm is therefore finite" in proof_note
    assert "topological\nordering of the acyclic retained graph" in proof_note
    assert "project the remaining forcing with I-Pi_j" in proof_note
    assert "record only the final constant kernel component" in proof_note
    assert "a_(omega,j) =" in proof_note
    assert "retained_particular_rows_(omega,j)(s)" in proof_note
    assert "The Cauchy property is a scalar row calculation" in proof_note
    assert "dY/ds + alpha_j Y = exp(-omega s) P_omega(s)" in proof_note
    assert "Y_part = exp(-omega s) B(s)" in proof_note
    assert "B'(s) + (alpha_j-omega) B(s) = P_omega(s)" in proof_note
    assert "b_L = f_L/(alpha_j-omega)" in proof_note
    assert "B'(s)=P_alpha_j(s)" in proof_note
    assert "f_L s^(L+1)/(L+1)" in proof_note
    assert "dR/ds + alpha_j R = O(exp(-(alpha_j+epsilon)s) s^m)" in proof_note
    assert "C' exp(-epsilon S) S^m" in proof_note
    assert "selector_data_fields" in proof_note
    assert "time-change formula is finite in the lifted selector class" in (
        proof_note
    )
    assert "lambda = c tau^2 (1+eta(tau,z))" in proof_note
    assert "s = -2 log tau - log c - log(1+eta(tau,z))" in proof_note
    assert "exp(-omega s) s^ell" in proof_note
    assert "c^omega tau^(2 omega) (1+eta(tau,z))^omega" in proof_note
    assert "preserves the finite exponent table and the\nfinite log-degree bound" in (
        proof_note
    )
    assert "B D + q R_0 <= R_0" in proof_note
    assert "(C_0, Lambda, sigma, p_0, d)" in proof_note
    assert "P(r) = {|tau| <= r_tau" in proof_note
    assert "projection constants are finite chain-rule consequences" in proof_note
    assert "q(tau)=tau^2 S(tau)" in proof_note
    assert "d/dt=-(1/(3 tau^2)) d/dtau" in proof_note
    assert "|delta q| <= C tau^(omega+2)" in proof_note
    assert "|delta v| = |delta q_t|" in proof_note
    assert "delta q_tt =" in proof_note
    assert "projected_residual" in proof_note
    assert "tau^2 S'' + 2 tau S' - 2 S - 9 A(S)" in proof_note
    assert "physical residual\ntail of weight `omega-4`" in proof_note
    assert "projected_physical_residual" in proof_note
    assert "M_a(rho r)" in proof_note
    assert "rational interval enclosures" in proof_note
    assert "expression DAG" in proof_note
    assert "X op Y subset I_op(X,Y)" in proof_note
    assert "denominator exclusion certificate" in proof_note
    assert "B A = I+E" in proof_note
    assert "log_tau_shell" in proof_note
    assert "tau_power_alpha_shell_j" in proof_note
    assert "interval inclusion invariant" in proof_note
    assert "upper(B) upper(L_N) < 1" in proof_note
    assert "proof-grade backend id" in proof_note
    assert "diagnostic estimates rather than certificate-grade constants" in (
        proof_note
    )
    assert "retained_defect_{>W_*}" in proof_note
    assert "M_j(omega) = omega I - A_j" in proof_note
    assert "B = max( max_{W_+ <= omega <= W_inv} B_{j,omega}" in proof_note
    assert "DN_rem" in proof_note
    assert "rho_n = rho_0 sigma^n" in proof_note
    assert "C_0 Lambda^n sigma^(p_0+n d)" in proof_note
    assert "Lambda sigma^d < 1" in proof_note
    assert "||Phi(R_1)-Phi(R_2)|| <= B L_N" in proof_note
    assert "R^(n+1) = Phi(R^(n))" in proof_note
    assert "||R^(1)-R^(0)|| <= B D" in proof_note
    assert "||R^(n+1)-R^(n)|| <= q^n B D" in proof_note
    assert "R = sum_{n>=0} (R^(n+1)-R^(n))" in proof_note
    assert "||R-R^(N)|| <= q^N B D/(1-q)" in proof_note
    assert "q=B L_N" in proof_note
    assert "endpoint and invariant ledgers are finite interval consequences" in (
        proof_note
    )
    assert "q_i-q_j = tau^2(S_i-S_j)" in proof_note
    assert "CM(tau) = sum_i m_i q_i(tau)" in proof_note
    assert "P(tau)  = sum_i m_i q_{i,t}(tau)" in proof_note
    assert "|delta CM| <= sum_i m_i |delta q_i|" in proof_note
    assert "q_t = -(2/(3tau))S - (1/3)S'" in proof_note
    assert "L = sum_i m_i q_i wedge q_{i,t}" in proof_note
    assert "L -> 0" in proof_note
    assert "tau^(-2) [ (2/9)||S||_m^2 - U(S) ]" in proof_note
    assert "U(C) = (2/9)||C||_m^2" in proof_note
    assert "cancel every remaining negative-power row" in proof_note
    assert "finite-energy matching constant `H_ret`" in proof_note
    assert "|S_i-S_j| >= d_shape > 0" in proof_note
    assert "finite Lipschitz bound `L_U`" in proof_note
    assert "finite interval derivative bound\n`L_K`" in proof_note
    assert "|delta H| <= L_U ||delta S|| + L_K" in proof_note
    assert "retained_energy_defect_tail" in proof_note
    assert "finite-energy matching constant" in proof_note
    assert "invariant rows are not sampled diagnostics" in proof_note
    assert "Physical-time containment is equally explicit" in proof_note
    assert "t(tau) = t_c - tau^3" in proof_note
    assert "dt/dtau = -3 tau^2 < 0" in proof_note
    assert "t in [t_c-tau_0^3, t_c)" in proof_note
    assert "rational endpoint\ncomparisons" in proof_note
    assert "no-continuation row is a policy statement" in proof_note
    assert "maximal_classical_stop" in proof_note
    assert "no outgoing selector constants" in proof_note
    assert "separate selector policy and a separate outgoing chart" in proof_note
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
    assert "tests/test_last_pro_instructions.py" in fast_ci
    assert "tests/test_closed_form.py" in fast_ci
    assert "tests/test_public_proof_audit.py" in fast_ci
    assert "PUBLIC_AUDIT_FAST_K" in fast_ci
    assert "PUBLIC_REVIEW_ARTIFACT_HARDENING_K" in fast_ci
    assert "public_total_collision_audit_keeps_tc4_tc6_open_by_default" in fast_ci
    assert "public_audit_manifest_resolver_certifies_local_artifact_manifest" in fast_ci
    assert "public_audit_required_artifacts_are_covered_by_ci_targets" in fast_ci
    assert "public_review_artifact_resolution_certifies_required_tc4_tc6_artifacts" in (
        fast_ci
    )
    assert "public_review_artifact_resolution_rejects_missing_artifact_object" in (
        fast_ci
    )
    assert "public_review_artifact_resolution_rejects_duplicate_artifact_reuse" in (
        fast_ci
    )
    assert "public_review_artifact_resolution_rejects_missing_proof_reference" in (
        fast_ci
    )
    assert "public_review_artifact_resolution_rejects_stale_proof_note_reference" in (
        fast_ci
    )
    assert "public_review_artifact_resolution_rejects_repo_artifacts_labeled_external_review" in (
        fast_ci
    )
    assert "public_review_artifact_resolution_rejects_missing_checked_artifacts" in (
        fast_ci
    )
    assert "public_review_resolver_rejects_stale_local_audit_evidence" in fast_ci
    assert "public_review_resolver_rejects_stale_top_level_proof_reference_manifest" in (
        fast_ci
    )
    assert "review_ready_total_collision_package_accepts_neutral_public_review_artifact_ids" in (
        fast_ci
    )
    assert "review_ready_total_collision_package_rejects_conflicting_public_review_ids" in (
        fast_ci
    )
    assert "public_total_collision_audit_accepts_neutral_public_review_aliases" in (
        fast_ci
    )
    assert "public_total_collision_audit_rejects_conflicting_public_review_alias_ids" in (
        fast_ci
    )
    assert "public_total_collision_audit_rejects_conflicting_public_review_resolver_aliases" in (
        fast_ci
    )
    assert "tests/test_general_solution_theorem.py" in fast_ci
    assert "tests/test_open_time_atlas.py" in fast_ci
    assert "tests/test_zero_angular_entry.py" in fast_ci
    assert "tests/test_obstructions.py" in fast_ci
    assert "PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS" in fast_ci
    assert "PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS" in fast_ci
    assert "TC4_TC5_PUBLIC_AUDIT_ARTIFACT_K" in fast_ci
    assert "FINITE_TIME_ATLAS_HARDENING_K" in fast_ci
    assert "validated_atlas_proof_ledger_rejects_truthy_fake_and_optional_only_entries" in (
        fast_ci
    )
    assert "validated_atlas_rejects_truthy_component_certification_flags" in (
        fast_ci
    )
    assert "validated_atlas_rejects_attribute_compatible_nested_proof_objects" in (
        fast_ci
    )
    assert "finite_time_regime_classifier_rejects_spoofed_obligation_ledgers" in (
        fast_ci
    )
    assert "finite_time_regime_classifier_ignores_fake_proof_ledger_entries" in (
        fast_ci
    )
    assert "ZERO_ANGULAR_SELECTOR_HARDENING_K" in fast_ci
    assert "zero_angular_compact_finite_atlas_requires_constructor_selector_entry" in (
        fast_ci
    )
    assert "positive_energy_homothetic_escape_rejects_fake_majorant_and_selector_entry" in (
        fast_ci
    )
    assert "CONSTRUCTOR_OBLIGATION_HARDENING_K" in fast_ci
    assert "required_constructor_obligations_reject_truthy_attribute_fields" in (
        fast_ci
    )
    assert "compact_finite_atlas_rejects_attribute_compatible_validated_atlas" in (
        fast_ci
    )
    assert "OPEN_TIME_CHECKED_PREFIX_K" in fast_ci
    assert "FINITE_TARGET_FUCHSIAN_STOP_LEDGER_HARDENING_K" in fast_ci
    assert "supplied_generalized_fuchsian_certificates_reject_spoofed_obligation_ledgers" in (
        fast_ci
    )
    assert "supplied_finite_fuchsian_log_stop_chart_rejects_spoofed_obligation_ledger" in (
        fast_ci
    )
    assert "supplied_finite_fuchsian_log_stop_chart_rejects_fake_constructor_inputs" in (
        fast_ci
    )
    assert "open_time_accepts_supplied_branch_union_checked_prefix" in fast_ci
    assert "open_time_accepts_stratified_branch_union_checked_prefix" in fast_ci
    assert "open_time_theorem_surfaces_reject_spoofed_obligation_ledgers" in (
        fast_ci
    )
    assert "total_collision_policy_certificate_requires_exact_policy_flags" in (
        fast_ci
    )
    assert "compact_interval_rejects_attribute_compatible_finite_target_certificates" in (
        fast_ci
    )
    assert '"scripts"' in fast_ci
    assert "regularized_locally_finite_atlas or pointwise" in fast_ci
    assert "sundman_route_rejects_truthy_global_series_and_scope_flags" in fast_ci
    assert "scope_and_sundman_witness_constructors_reject_truthy_flags" in fast_ci
    assert "tests/test_certificate_checker.py" in fast_ci
    assert '"not slow"' in fast_ci
    assert "--maxfail=1" in fast_ci
    assert "tests/test_certificate_checker.py" in slow_checker
    assert '"slow"' in slow_checker
    assert "PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS" in slow_checker
    assert "SLOW_GENERALIZED_FUCHSIAN_CHECKER_K" in slow_checker
    assert "independent_checker_accepts_serialized_generalized_fuchsian_stop_chart" in (
        slow_checker
    )
