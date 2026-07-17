from dataclasses import replace
from functools import lru_cache
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from three_body_symmetry.certificate_checker import (
    BranchUnionCheckResult,
    CertificateCheckObligation,
    CertificateCheckResult,
    ChartChainCheckResult,
    CertificateCheckerKernelSupportCertificate,
    EventIsolationCheckResult,
    IndependentChartVerifierCertificate,
    InitialValueBindingCheckResult,
    OrdinaryAposterioriTubeCheckResult,
    WeightedOrdinaryAposterioriTubeCheckResult,
    OrdinaryEnclosureTransitionCheckResult,
    PlanarLCAposterioriTubeCheckResult,
    OrdinaryToPlanarLCEnclosureTransitionCheckResult,
    PlanarLCToOrdinaryEnclosureTransitionCheckResult,
    PlanarLCExactCollisionAnchorCheckResult,
    PlanarLCTwoSidedCollisionPassageCheckResult,
    ValidatedOrdinaryIVPChainCheckResult,
    ValidatedOrdinaryIVPChartCheckResult,
    TransitionCheckResult,
    attach_independent_chart_verifier,
    certify_certificate_checker_kernel_support,
    certify_rational_interval_arithmetic_backend_soundness,
    check_branch_union,
    check_chart_chain,
    check_event_isolation,
    check_initial_value_problem_binding,
    check_ordinary_aposteriori_tube,
    check_weighted_ordinary_aposteriori_tube,
    check_ordinary_enclosure_transition,
    check_ordinary_to_planar_lc_enclosure_transition,
    check_planar_lc_to_ordinary_enclosure_transition,
    check_planar_lc_exact_collision_anchor,
    check_planar_lc_two_sided_collision_passage,
    check_validated_ordinary_ivp_chain,
    check_validated_ordinary_ivp_chart,
    check_ordinary_chart_transition,
    check_ordinary_taylor_chart,
    check_planar_levi_civita_binary_chart,
    check_planar_lc_aposteriori_tube,
    check_planar_levi_civita_transition,
    check_spatial_ks_binary_chart,
    check_spatial_ks_transition,
    check_total_collision_fuchsian_stop_chart,
    check_total_collision_generalized_fuchsian_stop_chart,
    verify_chart_certificates,
)
from three_body_symmetry.certificate_language import (
    BranchUnionCertificate,
    ChartChainCertificate,
    EventIsolationCertificate,
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    WeightedOrdinaryAposterioriTubeCertificate,
    OrdinaryEnclosureTransitionCertificate,
    OrdinaryToPlanarLCEnclosureTransitionCertificate,
    PlanarLCToOrdinaryEnclosureTransitionCertificate,
    PlanarLCExactCollisionAnchorCertificate,
    PlanarLCTwoSidedCollisionPassageCertificate,
    ValidatedOrdinaryIVPChainCertificate,
    GeneralizedFuchsianRemainderMajorantCertificate,
    TotalCollisionFuchsianStopChartCertificate,
    TotalCollisionGeneralizedFuchsianStopChartCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
    PlanarLCAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
    PrimitiveCauchyTailInputCertificate,
    SpatialKSBinaryChartCertificate,
    ordinary_chart_transition_certificate,
    ordinary_taylor_chart_certificate_from_solution,
    planar_hybrid_chart_chain_certificates_from_solution,
    planar_levi_civita_binary_chart_certificate_from_solution,
    planar_levi_civita_transition_certificate,
    spatial_ks_entry_event_isolation_certificate,
    spatial_ks_binary_chart_certificate_from_solution,
    spatial_ks_rho_exit_event_isolation_certificate,
    spatial_ks_transition_certificate,
    total_collision_fuchsian_stop_chart_certificate_from_branch,
    total_collision_generalized_fuchsian_stop_chart_certificate_from_branch,
)
from three_body_symmetry.binary_chart import (
    RegularizedBinaryCollisionChartState,
    planar_to_regularized_binary_collision_chart,
)
from three_body_symmetry.binary_chart import regularized_binary_collision_chart_to_planar
from three_body_symmetry.binary_series import construct_regularized_binary_taylor_solution
from three_body_symmetry.ks_binary_chart import SpatialKSBinaryChartState, ks_binary_chart_to_spatial
from three_body_symmetry.planar_lc_mass_coefficients import (
    PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID,
)
from three_body_symmetry.ks_binary_chart import spatial_to_ks_binary_chart
from three_body_symmetry.ks_binary_series import construct_spatial_ks_binary_taylor_solution
from three_body_symmetry.ks_binary_series import (
    certify_spatial_ks_binary_rho_exit_event,
    certify_spatial_ordinary_ks_entry_event,
    construct_interval_spatial_ks_binary_taylor_solution,
)
from three_body_symmetry.closed_form import certify_general_closed_form_solution_target
from three_body_symmetry.finite_target_completeness import (
    certify_supplied_generalized_fuchsian_analytic_remainder_majorant,
    certify_supplied_generalized_fuchsian_entry_data,
    certify_supplied_generalized_fuchsian_finite_row_tail_budget,
    certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data,
)
from three_body_symmetry.fuchsian import (
    FiniteFuchsianLogBranch,
    FuchsianLogTerm,
    certify_finite_fuchsian_log_total_collision_isolation,
    construct_fuchsian_selector_continuation,
    derive_finite_fuchsian_log_branch_primitive_cauchy_inputs,
    linearized_acceleration_matrix,
    mass_inner_product,
)
from three_body_symmetry.open_time_atlas import (
    construct_independent_finite_target_checked_atlas,
    construct_independent_validated_atlas_checked_chain,
    construct_open_time_locally_finite_atlas_theorem,
)
from three_body_symmetry.series import construct_interval_taylor_solution, construct_taylor_solution


def test_certificate_check_obligation_rejects_truthy_nonboolean_certification():
    result = CertificateCheckResult(
        certificate_id="manual-result",
        certificate_type="manual",
        checker_id="test",
        obligations=(
            CertificateCheckObligation(
                obligation="raw_truthy_gate",
                certified="yes",
                detail="truthy strings are not checker evidence",
            ),
        ),
        max_coefficient_residual=0.0,
        max_sampled_newton_residual=0.0,
    )

    assert result.obligations[0].certified is False
    assert not result.certified
    assert result.missing_obligations == ("raw_truthy_gate",)


def test_certificate_check_results_reject_spoofed_obligation_ledgers():
    fake_obligation = SimpleNamespace(
        obligation="fake",
        certified=True,
        detail="attribute-compatible spoof",
    )
    check_results = (
        CertificateCheckResult(
            certificate_id="chart-result",
            certificate_type="manual",
            checker_id="test",
            obligations=(fake_obligation,),
            max_coefficient_residual=0.0,
            max_sampled_newton_residual=0.0,
        ),
        TransitionCheckResult(
            transition_id="transition-result",
            transition_type="manual",
            checker_id="test",
            obligations=(fake_obligation,),
            max_position_gap=0.0,
            max_velocity_gap=0.0,
        ),
        EventIsolationCheckResult(
            event_id="event-result",
            event_type="manual",
            checker_id="test",
            obligations=(fake_obligation,),
            root_interval_width=0.0,
        ),
        BranchUnionCheckResult(
            union_id="branch-union-result",
            union_type="manual",
            checker_id="test",
            obligations=(fake_obligation,),
            leaf_count=1,
        ),
        ChartChainCheckResult(
            chain_id="chart-chain-result",
            chain_type="manual",
            checker_id="test",
            obligations=(fake_obligation,),
            chart_count=1,
        ),
    )

    for result in check_results:
        assert not result.certified
        assert any(
            missing.endswith("_obligation_type")
            for missing in result.missing_obligations
        )

    empty = replace(check_results[0], obligations=())
    assert not empty.certified
    assert empty.missing_obligations == ("chart-result_obligations_present",)


def test_proof_grade_backend_and_kernel_reject_truthy_direct_flags():
    backend = certify_rational_interval_arithmetic_backend_soundness()
    assert backend.proof_certified

    truthy_backend = replace(backend, exact_fraction_endpoints="yes")
    assert not truthy_backend.proof_certified
    assert truthy_backend.missing_obligations == ("exact_fraction_endpoints",)

    fake_backend_kernel = CertificateCheckerKernelSupportCertificate(
        checker_id="independent_chart_verifier_v1",
        supported_chart_types=(
            "ordinary_taylor",
            "planar_levi_civita_binary",
            "spatial_ks_binary",
            "total_collision_fuchsian_stop",
            "total_collision_generalized_fuchsian_stop",
        ),
        event_obligation_ids=("event_root_endpoint_signs",),
        branch_union_obligation_ids=("branch_union_leaf_cover",),
        chart_chain_obligation_ids=("chart_chain_transition_count_matches",),
        transition_obligation_ids=("transition_state_continuity",),
        proof_grade_arithmetic_backend_certificate=SimpleNamespace(
            proof_certified=True,
        ),
    )
    assert not fake_backend_kernel.proof_certified
    assert "proof_grade_arithmetic_backend_sound" in (
        fake_backend_kernel.missing_obligations
    )

    with pytest.raises(
        TypeError,
        match="ProofGradeArithmeticBackendCertificate",
    ):
        certify_certificate_checker_kernel_support(
            proof_grade_arithmetic_backend_sound=True,
        )


def test_independent_verifier_rejects_spoofed_result_objects():
    certified_obligation = CertificateCheckObligation(
        obligation="ordinary_taylor_exact_rational_residual_polynomials",
        certified=True,
        detail="fixture",
    )
    real_result = CertificateCheckResult(
        certificate_id="ordinary-result",
        certificate_type="ordinary_taylor",
        checker_id="test",
        obligations=(certified_obligation,),
        max_coefficient_residual=0.0,
        max_sampled_newton_residual=0.0,
    )
    fake_result = SimpleNamespace(
        certificate_id="fake-result",
        certificate_type="ordinary_taylor",
        certified=True,
        obligations=(certified_obligation,),
        missing_obligations=(),
    )

    verifier = IndependentChartVerifierCertificate(
        checker_id="independent_chart_verifier_v1",
        chart_results=(real_result,),
    )
    assert verifier.certified

    spoofed = replace(verifier, chart_results=(fake_result,))
    assert not spoofed.certified
    assert "chart_result_type" in spoofed.missing_obligations
    assert "chart_result_type" in spoofed.proof_grade_arithmetic_blockers


def test_independent_verifier_rejects_subclassed_result_objects():
    class SpoofedCertificateCheckResult(CertificateCheckResult):
        @property
        def certified(self):
            return True

        @property
        def missing_obligations(self):
            return ()

    class SpoofedChartChainCheckResult(ChartChainCheckResult):
        @property
        def certified(self):
            return True

        @property
        def missing_obligations(self):
            return ()

    real_chart_result = check_ordinary_taylor_chart(_ordinary_chart_certificate())
    spoofed_chart_result = SpoofedCertificateCheckResult(
        certificate_id="spoofed-chart-result",
        certificate_type="ordinary_taylor",
        checker_id="test",
        obligations=(),
        max_coefficient_residual=0.0,
        max_sampled_newton_residual=0.0,
    )
    spoofed_chain_result = SpoofedChartChainCheckResult(
        chain_id="spoofed-chart-chain",
        chain_type="regularized_atlas_chart_chain",
        checker_id="test",
        obligations=(),
        chart_count=1,
    )

    spoofed_chart_verifier = IndependentChartVerifierCertificate(
        checker_id="independent_chart_verifier_v1",
        chart_results=(spoofed_chart_result,),
    )
    spoofed_chain_verifier = IndependentChartVerifierCertificate(
        checker_id="independent_chart_verifier_v1",
        chart_results=(real_chart_result,),
        chart_chain_results=(spoofed_chain_result,),
    )

    assert real_chart_result.certified
    assert spoofed_chart_result.certified
    assert spoofed_chain_result.certified
    assert not spoofed_chart_verifier.certified
    assert "chart_result_type" in spoofed_chart_verifier.missing_obligations
    assert "chart_result_type" in (
        spoofed_chart_verifier.proof_grade_arithmetic_blockers
    )
    assert not spoofed_chain_verifier.certified
    assert not spoofed_chain_verifier.proof_grade_finite_atlas_bundle_certified
    assert "chart_chain_result_type" in (
        spoofed_chain_verifier.missing_obligations
    )
    assert "chart_chain_result_type" in (
        spoofed_chain_verifier.proof_grade_finite_atlas_blockers
    )


def _ordinary_chart_certificate():
    masses = np.array([1.0, 0.8, 1.2])
    positions = np.array(
        [
            [0.8, -0.2],
            [-0.4, 0.6],
            [0.1, -0.5],
        ]
    )
    velocities = np.array(
        [
            [0.03, 0.01],
            [-0.02, 0.04],
            [0.01, -0.03],
        ]
    )
    solution = construct_taylor_solution(positions, velocities, masses, order=10)
    return ordinary_taylor_chart_certificate_from_solution(
        solution,
        certificate_id="ordinary-chart-0",
        chart_id="ordinary-0",
        parameter_interval=(-0.02, 0.02),
        coefficient_tolerance=1.0e-11,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )


def _ordinary_chart_chain():
    masses = np.array([1.0, 0.8, 1.2])
    positions = np.array(
        [
            [0.8, -0.2],
            [-0.4, 0.6],
            [0.1, -0.5],
        ]
    )
    velocities = np.array(
        [
            [0.03, 0.01],
            [-0.02, 0.04],
            [0.01, -0.03],
        ]
    )
    first_solution = construct_taylor_solution(positions, velocities, masses, order=10)
    handoff_time = 0.02
    second_solution = construct_taylor_solution(
        first_solution.positions_at(handoff_time),
        first_solution.velocities_at(handoff_time),
        masses,
        order=10,
    )
    first = ordinary_taylor_chart_certificate_from_solution(
        first_solution,
        certificate_id="ordinary-chart-0",
        chart_id="ordinary-0",
        parameter_interval=(0.0, handoff_time),
        physical_time_interval=(0.0, handoff_time),
        coefficient_tolerance=1.0e-11,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    second = ordinary_taylor_chart_certificate_from_solution(
        second_solution,
        certificate_id="ordinary-chart-1",
        chart_id="ordinary-1",
        parameter_interval=(0.0, handoff_time),
        physical_time_interval=(handoff_time, 2.0 * handoff_time),
        coefficient_tolerance=1.0e-11,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    transition = ordinary_chart_transition_certificate(
        transition_id="ordinary-transition-0-1",
        source_chart_id=first.chart_id,
        target_chart_id=second.chart_id,
        handoff_time=handoff_time,
        position_tolerance=1.0e-10,
        velocity_tolerance=1.0e-10,
    )
    return (first, second), (transition,)


def _planar_lc_binary_chart_certificate():
    solution = _planar_lc_binary_solution()
    return planar_levi_civita_binary_chart_certificate_from_solution(
        solution,
        certificate_id="planar-lc-chart-0",
        chart_id="planar-lc-0",
        parameter_interval=(0.0, 1.0e-10),
        coefficient_tolerance=1.0e-11,
        regularized_residual_tolerance=1.0e-8,
        projected_residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
        projection_rho_lower_bound=1.0e-10,
    )


def _planar_lc_binary_solution():
    # Exact dyadic pair ratios keep the proof-facing LC/Newton mass algebra
    # identical to the serialized binary mass system.
    masses = np.array([1.0, 1.0, 1.2])
    positions = np.array(
        [
            [-0.12, 0.025],
            [0.10, -0.015],
            [1.7, 0.9],
        ]
    )
    velocities = np.array(
        [
            [0.04, -0.08],
            [-0.02, 0.06],
            [0.01, 0.03],
        ]
    )
    initial = planar_to_regularized_binary_collision_chart(
        positions,
        velocities,
        masses,
        pair=(0, 1),
    )
    return construct_regularized_binary_taylor_solution(initial, order=12)


def _planar_lc_exact_collision_solution():
    masses = np.array([1.0, 1.0, 1.2])
    pair_mass = float(masses[0] + masses[1])
    initial = RegularizedBinaryCollisionChartState(
        masses=masses,
        pair=(0, 1),
        z=np.array([0.0, 0.0]),
        z_velocity=np.array([np.sqrt(pair_mass / 2.0), 0.0]),
        pair_energy=0.0,
        binary_center=np.array([0.0, 0.0]),
        binary_center_velocity=np.array([0.01, -0.02]),
        third_offset=np.array([2.0, 1.0]),
        third_offset_velocity=np.array([-0.01, 0.015]),
    )
    return construct_regularized_binary_taylor_solution(initial, order=12)


def _planar_lc_exact_collision_chart_certificate():
    solution = _planar_lc_exact_collision_solution()
    return planar_levi_civita_binary_chart_certificate_from_solution(
        solution,
        certificate_id="planar-lc-exact-collision-chart",
        chart_id="planar-lc-exact-collision",
        parameter_interval=(-1.0e-4, 1.0e-4),
        coefficient_tolerance=1.0e-10,
        regularized_residual_tolerance=1.0e-7,
        projected_residual_tolerance=1.0,
        tail_bound=1.0e-9,
        sample_count=8,
        projection_rho_lower_bound=1.0e-12,
    )


def _spatial_ks_binary_chart_certificate():
    solution = _spatial_ks_binary_solution()
    return spatial_ks_binary_chart_certificate_from_solution(
        solution,
        certificate_id="spatial-ks-chart-0",
        chart_id="spatial-ks-0",
        parameter_interval=(0.0, 1.0e-9),
        coefficient_tolerance=1.0e-11,
        regularized_residual_tolerance=1.0e-8,
        projected_residual_tolerance=1.0e-8,
        constraint_tolerance=1.0e-10,
        tail_bound=1.0e-9,
        sample_count=7,
        projection_rho_lower_bound=1.0e-10,
    )


def _spatial_ks_binary_solution():
    masses = np.array([0.8, 1.2, 1.7])
    positions = np.array(
        [
            [-0.30, 0.20, 0.10],
            [0.45, -0.10, 0.35],
            [1.30, 0.90, -0.20],
        ]
    )
    velocities = np.array(
        [
            [0.15, -0.05, 0.04],
            [-0.10, 0.22, -0.03],
            [0.03, -0.08, 0.05],
        ]
    )
    initial = spatial_to_ks_binary_chart(
        positions,
        velocities,
        masses,
        pair=(0, 1),
    )
    return construct_spatial_ks_binary_taylor_solution(initial, order=12)


def _spatial_ks_binary_crossing_solution():
    centered = construct_spatial_ks_binary_taylor_solution(
        _exact_spatial_ks_collision_state(),
        order=40,
    )
    return construct_spatial_ks_binary_taylor_solution(
        centered.state_at(-0.01),
        order=30,
    )


def _exact_spatial_ks_collision_state():
    masses = np.array([0.8, 1.2, 1.7])
    pair_mass = masses[0] + masses[1]
    return SpatialKSBinaryChartState(
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


def _ordinary_ks_entry_event_certificate():
    masses = np.array([1.0, 1.2, 1.5])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.02, 0.01],
            [1.0, 0.8, 0.5],
        ],
    )
    velocities = np.array(
        [
            [0.03, 0.0, 0.0],
            [-0.03, -0.001, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )
    interval = construct_interval_taylor_solution(positions, velocities, masses, order=16)
    event = certify_spatial_ordinary_ks_entry_event(
        interval,
        pair=(0, 1),
        enter_distance=0.198,
        time_upper=0.1,
        coefficient_count=14,
    )
    return spatial_ks_entry_event_isolation_certificate(
        event,
        certificate_id="ordinary-ks-entry-event-certificate",
        event_id="ordinary-ks-entry-event",
    )


def _ks_rho_exit_event_certificate():
    initial = _exact_spatial_ks_collision_state()
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=24)
    event = certify_spatial_ks_binary_rho_exit_event(
        interval,
        exit_rho=1.0e-4,
        s_upper=0.02,
        coefficient_count=20,
    )
    return spatial_ks_rho_exit_event_isolation_certificate(
        event,
        certificate_id="ks-rho-exit-event-certificate",
        event_id="ks-rho-exit-event",
    )


def _total_collision_fuchsian_stop_chart_certificate(
    *,
    scale_coefficient=0.0,
    include_cauchy_inputs=True,
    terms=(),
    residual_tolerance=1.0e-6,
):
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
        scale_coefficient=float(scale_coefficient),
        terms=tuple(terms),
    )
    isolation = certify_finite_fuchsian_log_total_collision_isolation(
        branch,
        radius=0.025,
    )
    cauchy_inputs = None
    tail_bound = 0.0
    if include_cauchy_inputs:
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
        tail_bound = max(
            input_.first_shell_tail_bound
            for input_ in cauchy_inputs.component_inputs.values()
        )
    return total_collision_fuchsian_stop_chart_certificate_from_branch(
        branch,
        isolation,
        certificate_id="fuchsian-total-stop-certificate",
        chart_id="fuchsian-total-stop",
        residual_tolerance=float(residual_tolerance),
        angular_momentum_tolerance=1.0e-12,
        tail_bound=tail_bound,
        sample_count=7,
        cauchy_inputs=cauchy_inputs,
    )


@lru_cache(maxsize=2)
def _supplied_generalized_fuchsian_branch(max_total_degree=4):
    max_total_degree = int(max_total_degree)
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
        max_total_degree=max_total_degree,
        scale_index=(1, 0, 0),
    )
    return continuation.incoming


def _generalized_remainder_majorant_certificate(majorant):
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


@lru_cache(maxsize=1)
def _total_collision_generalized_fuchsian_stop_chart_certificate():
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
    stop_chart = certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        remainder_majorant=majorant,
        residual_tolerance=1.0e-5,
        angular_momentum_tolerance=1.0e-5,
    )
    return total_collision_generalized_fuchsian_stop_chart_certificate_from_branch(
        branch,
        certificate_id="generalized-fuchsian-total-stop-certificate",
        chart_id="generalized-fuchsian-total-stop",
        isolation_radius=entry.radius,
        central_shape_pair_distance_floor=entry.central_shape_pair_distance_floor,
        shape_deviation_bound=entry.shape_deviation_bound,
        shape_pair_distance_floor=entry.shape_pair_distance_floor,
        tau_interval=stop_chart.tau_interval,
        residual_tolerance=stop_chart.residual_tolerance,
        angular_momentum_tolerance=stop_chart.angular_momentum_tolerance,
        tail_bound=stop_chart.tail_bound,
        sample_count=7,
        remainder_majorant=_generalized_remainder_majorant_certificate(majorant),
    )


@lru_cache(maxsize=1)
def _fast_total_collision_generalized_fuchsian_stop_chart_certificate():
    branch = _supplied_generalized_fuchsian_branch(2)
    entry = certify_supplied_generalized_fuchsian_entry_data(
        branch=branch,
        radius=0.02,
        sample_taus=(-0.012, 0.012),
        tolerance=1.0e-4,
        energy_tolerance=1.0e-7,
    )
    finite_rows = certify_supplied_generalized_fuchsian_finite_row_tail_budget(
        entry_certificate=entry,
        retained_total_degree=branch.max_total_degree,
        radius=0.018,
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
            "value": 6,
            "first_jet": 6,
            "lifted_residual": 6,
            "physical_residual": 6,
            "regularized_position_value": 6,
        },
        retained_order_increments={
            "value": 1,
            "first_jet": 1,
            "lifted_residual": 1,
            "physical_residual": 1,
            "regularized_position_value": 1,
        },
        initial_radius=0.014,
        shell_contraction=0.5,
        analytic_disk_fraction=0.2,
    )
    stop_chart = certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        remainder_majorant=majorant,
        residual_tolerance=1.0e-4,
        angular_momentum_tolerance=1.0e-4,
    )
    return total_collision_generalized_fuchsian_stop_chart_certificate_from_branch(
        branch,
        certificate_id="fast-generalized-fuchsian-total-stop-certificate",
        chart_id="fast-generalized-fuchsian-total-stop",
        isolation_radius=entry.radius,
        central_shape_pair_distance_floor=entry.central_shape_pair_distance_floor,
        shape_deviation_bound=entry.shape_deviation_bound,
        shape_pair_distance_floor=entry.shape_pair_distance_floor,
        tau_interval=stop_chart.tau_interval,
        residual_tolerance=stop_chart.residual_tolerance,
        angular_momentum_tolerance=stop_chart.angular_momentum_tolerance,
        tail_bound=stop_chart.tail_bound,
        sample_count=3,
        remainder_majorant=_generalized_remainder_majorant_certificate(majorant),
    )


def _ordinary_to_ks_transition_bundle():
    solution = _spatial_ks_binary_solution()
    positions, velocities = ks_binary_chart_to_spatial(solution.state_at(0.0))
    ordinary_solution = construct_taylor_solution(
        positions,
        velocities,
        solution.masses,
        order=10,
    )
    ordinary = ordinary_taylor_chart_certificate_from_solution(
        ordinary_solution,
        certificate_id="ordinary-ks-entry-chart",
        chart_id="ordinary-ks-entry",
        parameter_interval=(0.0, 0.012),
        physical_time_interval=(0.0, 0.012),
        coefficient_tolerance=1.0e-10,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    ks = spatial_ks_binary_chart_certificate_from_solution(
        solution,
        certificate_id="spatial-ks-entry-chart",
        chart_id="spatial-ks-entry",
        parameter_interval=(0.0, 0.012),
        coefficient_tolerance=1.0e-11,
        regularized_residual_tolerance=1.0e-8,
        projected_residual_tolerance=5.0e-2,
        constraint_tolerance=1.0e-10,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    transition = spatial_ks_transition_certificate(
        transition_id="ordinary-to-ks-entry",
        source_chart_id=ordinary.chart_id,
        target_chart_id=ks.chart_id,
        transition_type="spatial_ordinary_to_ks_decreasing_distance_entry",
        handoff_time=0.0,
        source_parameter=0.0,
        target_parameter=0.0,
        position_tolerance=1.0e-10,
        velocity_tolerance=1.0e-10,
        physical_time_tolerance=1.0e-12,
    )
    return (ordinary, ks), (transition,)


def _ks_to_ordinary_transition_bundle():
    solution = _spatial_ks_binary_solution()
    exit_parameter = 0.006
    handoff_time = solution.physical_time_at(exit_parameter)
    positions, velocities = ks_binary_chart_to_spatial(
        solution.state_at(exit_parameter),
    )
    ordinary_solution = construct_taylor_solution(
        positions,
        velocities,
        solution.masses,
        order=10,
    )
    ks = spatial_ks_binary_chart_certificate_from_solution(
        solution,
        certificate_id="spatial-ks-exit-chart",
        chart_id="spatial-ks-exit",
        parameter_interval=(0.0, exit_parameter),
        coefficient_tolerance=1.0e-11,
        regularized_residual_tolerance=1.0e-8,
        projected_residual_tolerance=5.0e-2,
        constraint_tolerance=1.0e-10,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    ordinary = ordinary_taylor_chart_certificate_from_solution(
        ordinary_solution,
        certificate_id="ordinary-ks-exit-chart",
        chart_id="ordinary-ks-exit",
        parameter_interval=(0.0, 0.01),
        physical_time_interval=(handoff_time, handoff_time + 0.01),
        coefficient_tolerance=1.0e-10,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    transition = spatial_ks_transition_certificate(
        transition_id="ks-to-ordinary-exit",
        source_chart_id=ks.chart_id,
        target_chart_id=ordinary.chart_id,
        transition_type="spatial_ks_to_ordinary_rho_positive_endpoint_projection",
        handoff_time=handoff_time,
        source_parameter=exit_parameter,
        target_parameter=0.0,
        position_tolerance=1.0e-10,
        velocity_tolerance=1.0e-10,
        physical_time_tolerance=1.0e-12,
    )
    return (ks, ordinary), (transition,)


def _ordinary_to_lc_transition_bundle():
    solution = _planar_lc_binary_solution()
    initial_state = solution.state_at(0.0)
    positions, velocities = regularized_binary_collision_chart_to_planar(initial_state)
    ordinary_solution = construct_taylor_solution(
        positions,
        velocities,
        solution.masses,
        order=10,
    )
    ordinary = ordinary_taylor_chart_certificate_from_solution(
        ordinary_solution,
        certificate_id="ordinary-entry-chart",
        chart_id="ordinary-entry",
        parameter_interval=(0.0, 0.012),
        physical_time_interval=(0.0, 0.012),
        coefficient_tolerance=1.0e-6,
        residual_tolerance=1.0e-6,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    lc = planar_levi_civita_binary_chart_certificate_from_solution(
        solution,
        certificate_id="planar-lc-entry-chart",
        chart_id="planar-lc-entry",
        parameter_interval=(0.0, 0.012),
        coefficient_tolerance=1.0e-11,
        regularized_residual_tolerance=1.0e-8,
        projected_residual_tolerance=2.0e-1,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    transition = planar_levi_civita_transition_certificate(
        transition_id="ordinary-to-lc-entry",
        source_chart_id=ordinary.chart_id,
        target_chart_id=lc.chart_id,
        transition_type="ordinary_to_binary_event_handoff",
        handoff_time=0.0,
        source_parameter=0.0,
        target_parameter=0.0,
        position_tolerance=1.0e-10,
        velocity_tolerance=1.0e-10,
        physical_time_tolerance=1.0e-12,
    )
    return (ordinary, lc), (transition,)


def _lc_to_ordinary_transition_bundle():
    solution = _planar_lc_binary_solution()
    exit_parameter = 0.006
    handoff_time = solution.physical_time_at(exit_parameter)
    positions, velocities = regularized_binary_collision_chart_to_planar(
        solution.state_at(exit_parameter),
    )
    ordinary_solution = construct_taylor_solution(
        positions,
        velocities,
        solution.masses,
        order=10,
    )
    lc = planar_levi_civita_binary_chart_certificate_from_solution(
        solution,
        certificate_id="planar-lc-exit-chart",
        chart_id="planar-lc-exit",
        parameter_interval=(0.0, exit_parameter),
        coefficient_tolerance=1.0e-11,
        regularized_residual_tolerance=1.0e-8,
        projected_residual_tolerance=2.0e-1,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    ordinary = ordinary_taylor_chart_certificate_from_solution(
        ordinary_solution,
        certificate_id="ordinary-exit-chart",
        chart_id="ordinary-exit",
        parameter_interval=(0.0, 0.01),
        physical_time_interval=(handoff_time, handoff_time + 0.01),
        coefficient_tolerance=1.0e-6,
        residual_tolerance=1.0e-6,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    transition = planar_levi_civita_transition_certificate(
        transition_id="lc-to-ordinary-exit",
        source_chart_id=lc.chart_id,
        target_chart_id=ordinary.chart_id,
        transition_type="binary_to_ordinary_event_handoff",
        handoff_time=handoff_time,
        source_parameter=exit_parameter,
        target_parameter=0.0,
        position_tolerance=1.0e-10,
        velocity_tolerance=1.0e-10,
        physical_time_tolerance=1.0e-12,
    )
    return (lc, ordinary), (transition,)


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


def test_independent_checker_accepts_serialized_ordinary_taylor_chart():
    certificate = _ordinary_chart_certificate()
    result = check_ordinary_taylor_chart(certificate)
    round_trip = OrdinaryTaylorChartCertificate.from_dict(certificate.to_dict())
    round_trip_result = check_ordinary_taylor_chart(round_trip)

    assert result.certified
    assert result.missing_obligations == ()
    assert result.max_coefficient_residual <= certificate.coefficient_tolerance
    assert result.max_sampled_newton_residual <= certificate.residual_tolerance
    assert "interval_taylor_model_newton_residual" not in result.missing_obligations
    details = {obligation.obligation: obligation for obligation in result.obligations}
    assert details["ordinary_taylor_exact_rational_residual_polynomials"].certified
    assert "exact rational interval arithmetic" in (
        details["ordinary_taylor_exact_rational_residual_polynomials"].detail
    )
    assert round_trip_result.certified
    assert round_trip_result.max_sampled_newton_residual == (
        result.max_sampled_newton_residual
    )


def test_verifier_arithmetic_audit_accepts_chart_only_exact_rational_bundle():
    verifier = verify_chart_certificates((_ordinary_chart_certificate(),))

    assert verifier.certified
    assert verifier.proof_grade_arithmetic_checked_bundle_certified
    assert verifier.proof_grade_arithmetic_blockers == ()
    assert not verifier.proof_grade_finite_atlas_bundle_certified
    assert verifier.proof_grade_finite_atlas_blockers == (
        "finite_atlas_chart_chain_checked",
    )
    assert "ordinary_taylor_exact_rational_residual_polynomials" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )
    assert "interval_taylor_model_newton_residual" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )


def test_ordinary_checker_certification_is_interval_not_sample_gated():
    certificate = replace(_ordinary_chart_certificate(), sample_count=1)
    result = check_ordinary_taylor_chart(certificate)
    obligation_ids = {obligation.obligation for obligation in result.obligations}

    assert result.certified
    assert "sampled_projected_newton_residual" not in obligation_ids
    assert "ordinary_taylor_exact_rational_residual_polynomials" in obligation_ids
    assert "interval_taylor_model_newton_residual" in obligation_ids


def test_independent_checker_rejects_corrupted_taylor_recurrence():
    certificate = _ordinary_chart_certificate()
    velocity = [
        [list(axis_values) for axis_values in degree]
        for degree in certificate.velocity_coefficients
    ]
    velocity[3][1][0] += 0.1
    corrupted = replace(
        certificate,
        velocity_coefficients=tuple(
            tuple(tuple(axis_values) for axis_values in degree)
            for degree in velocity
        ),
    )
    result = check_ordinary_taylor_chart(corrupted)

    assert not result.certified
    assert "ordinary_taylor_coefficient_recurrence" in result.missing_obligations
    assert "ordinary_taylor_exact_rational_residual_polynomials" in (
        result.missing_obligations
    )
    assert "interval_taylor_model_newton_residual" in result.missing_obligations


def test_independent_checker_rejects_excessive_ordinary_taylor_tail_model_bound():
    certificate = _ordinary_chart_certificate()
    inflated_tail = replace(
        certificate,
        tail_bound=10.0 * certificate.residual_tolerance,
    )
    result = check_ordinary_taylor_chart(inflated_tail)

    assert not result.certified
    assert "interval_taylor_model_newton_residual" in result.missing_obligations
    assert "tail_bound_admissible" not in result.missing_obligations


def test_independent_checker_rejects_nonunit_physical_parameter_speed():
    certificate = _ordinary_chart_certificate()
    stretched = replace(
        certificate,
        parameter_interval=(0.0, 0.02),
        physical_time_interval=(0.0, 0.04),
    )
    result = check_ordinary_taylor_chart(stretched)

    assert not result.certified
    assert "ordinary_physical_parameter_unit_speed" in result.missing_obligations
    assert "ordinary_taylor_exact_rational_residual_polynomials" in (
        result.missing_obligations
    )
    assert "interval_taylor_model_newton_residual" in result.missing_obligations


def test_independent_checker_accepts_ordinary_transition_chain():
    charts, transitions = _ordinary_chart_chain()
    result = check_ordinary_chart_transition(transitions[0], charts)
    verifier = verify_chart_certificates(charts, transitions)
    details = {obligation.obligation: obligation for obligation in result.obligations}

    assert result.certified
    assert result.max_position_gap <= transitions[0].position_tolerance
    assert result.max_velocity_gap <= transitions[0].velocity_tolerance
    assert details["transition_exact_rational_state_continuity"].certified
    assert "exact rational arithmetic" in (
        details["transition_exact_rational_state_continuity"].detail
    )
    assert verifier.certified
    assert verifier.checked_certificate_count == 2
    assert len(verifier.transition_results) == 1
    assert verifier.missing_obligations == ()
    assert "transition_exact_rational_state_continuity" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )


def test_initial_value_binding_anchors_ordinary_chart_polynomial():
    chart = _ordinary_chart_certificate()
    binding = InitialValueProblemBindingCertificate(
        binding_id="ivp-binding:ordinary",
        chart_id=chart.chart_id,
        masses=chart.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=chart.position_coefficients[0],
        velocities=chart.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )

    result = check_initial_value_problem_binding(binding, (chart,))

    assert type(result) is InitialValueBindingCheckResult
    assert result.certified
    assert result.missing_obligations == ()


def test_initial_value_binding_rejects_wrong_initial_state():
    chart = _ordinary_chart_certificate()
    positions = [list(row) for row in chart.position_coefficients[0]]
    positions[0][0] += 0.25
    binding = InitialValueProblemBindingCertificate(
        binding_id="ivp-binding:wrong-state",
        chart_id=chart.chart_id,
        masses=chart.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=tuple(tuple(row) for row in positions),
        velocities=chart.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )

    result = check_initial_value_problem_binding(binding, (chart,))

    assert not result.certified
    assert "initial_value_binding_polynomial_state_matches" in (
        result.missing_obligations
    )


def test_ordinary_aposteriori_tube_certifies_exact_solution_enclosure():
    chart = _ordinary_chart_certificate()
    tube = OrdinaryAposterioriTubeCertificate(
        tube_id="ordinary-tube:fixture",
        chart_id=chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=2.0e-2,
        max_defect_bound=1.0e-1,
        max_lipschitz_bound=70.0,
    )

    result = check_ordinary_aposteriori_tube(tube, chart)

    assert type(result) is OrdinaryAposterioriTubeCheckResult
    assert result.certified
    assert result.defect_bound <= tube.max_defect_bound
    assert result.lipschitz_bound <= tube.max_lipschitz_bound
    assert result.gronwall_error_bound <= tube.tube_radius
    assert result.tube_pair_distance_floor > 0.0


def test_ordinary_aposteriori_tube_rejects_inconsistent_initial_error():
    chart = _ordinary_chart_certificate()
    tube = OrdinaryAposterioriTubeCertificate(
        tube_id="ordinary-tube:bad-initial-error",
        chart_id=chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=2.0e-2,
        tube_radius=2.0e-2,
        max_defect_bound=1.0e-1,
        max_lipschitz_bound=70.0,
    )

    result = check_ordinary_aposteriori_tube(tube, chart)

    assert not result.certified
    assert "ordinary_tube_gronwall_self_consistent" in result.missing_obligations


def test_validated_ordinary_ivp_chart_encloses_bound_initial_value_problem():
    chart = _ordinary_chart_certificate()
    binding = InitialValueProblemBindingCertificate(
        binding_id="ivp-binding:validated-ordinary",
        chart_id=chart.chart_id,
        masses=chart.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=chart.position_coefficients[0],
        velocities=chart.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )
    tube = OrdinaryAposterioriTubeCertificate(
        tube_id="ordinary-tube:validated-ordinary",
        chart_id=chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=2.0e-2,
        max_defect_bound=1.0e-1,
        max_lipschitz_bound=70.0,
    )

    result = check_validated_ordinary_ivp_chart(binding, tube, chart)

    assert type(result) is ValidatedOrdinaryIVPChartCheckResult
    assert result.certified
    assert result.missing_obligations == ()


def test_weighted_ordinary_tube_recovers_scalar_case_with_equal_radii():
    chart = _ordinary_chart_certificate()
    result = check_weighted_ordinary_aposteriori_tube(
        WeightedOrdinaryAposterioriTubeCertificate(
            "weighted-ordinary-tube:equal-radii",
            chart.chart_id,
            0.0,
            0.0,
            0.0,
            2.0e-2,
            2.0e-2,
            10.0,
            70.0,
        ),
        chart,
    )

    assert type(result) is WeightedOrdinaryAposterioriTubeCheckResult
    assert result.certified
    assert result.normalized_gronwall_error_bound < 1.0
    assert result.proven_position_error_bound < 2.0e-2
    assert result.proven_velocity_error_bound < 2.0e-2


def test_validated_ordinary_ivp_chart_rejects_uncovered_binding_error():
    chart = _ordinary_chart_certificate()
    positions = [list(row) for row in chart.position_coefficients[0]]
    positions[0][0] += 1.0e-3
    binding = InitialValueProblemBindingCertificate(
        binding_id="ivp-binding:uncovered-error",
        chart_id=chart.chart_id,
        masses=chart.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=tuple(tuple(row) for row in positions),
        velocities=chart.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=1.0e-2,
        velocity_tolerance=0.0,
    )
    tube = OrdinaryAposterioriTubeCertificate(
        tube_id="ordinary-tube:uncovered-error",
        chart_id=chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=2.0e-2,
        max_defect_bound=1.0e-1,
        max_lipschitz_bound=70.0,
    )

    result = check_validated_ordinary_ivp_chart(binding, tube, chart)

    assert result.binding_result.certified
    assert result.tube_result.certified
    assert not result.certified
    assert "validated_ordinary_actual_initial_error_covered" in (
        result.missing_obligations
    )


def test_ordinary_enclosure_transition_continues_exact_ivp_branch():
    charts, _ = _ordinary_chart_chain()
    source, target = charts
    binding = InitialValueProblemBindingCertificate(
        binding_id="ivp-binding:transition-source",
        chart_id=source.chart_id,
        masses=source.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=source.position_coefficients[0],
        velocities=source.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )
    source_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="ordinary-tube:transition-source",
        chart_id=source.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=2.0e-2,
        max_defect_bound=1.0e-1,
        max_lipschitz_bound=70.0,
    )
    source_validation = check_validated_ordinary_ivp_chart(
        binding, source_tube, source
    )
    target_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="ordinary-tube:transition-target",
        chart_id=target.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=2.0e-3,
        tube_radius=3.0e-2,
        max_defect_bound=2.0e-1,
        max_lipschitz_bound=100.0,
    )
    transition = OrdinaryEnclosureTransitionCertificate(
        transition_id="ordinary-enclosure-transition:0-1",
        source_chart_id=source.chart_id,
        target_chart_id=target.chart_id,
        source_parameter=0.02,
        target_parameter=0.0,
        handoff_time=0.02,
        max_time_gap=0.0,
    )

    result = check_ordinary_enclosure_transition(
        transition, source, target, source_validation, target_tube
    )

    assert type(result) is OrdinaryEnclosureTransitionCheckResult
    assert result.certified
    assert result.required_target_initial_error <= target_tube.initial_error_bound
    assert result.missing_obligations == ()


def test_ordinary_enclosure_transition_rejects_uncovered_source_tube_error():
    charts, _ = _ordinary_chart_chain()
    source, target = charts
    binding = InitialValueProblemBindingCertificate(
        binding_id="ivp-binding:transition-reject",
        chart_id=source.chart_id,
        masses=source.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=source.position_coefficients[0],
        velocities=source.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )
    source_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="ordinary-tube:transition-reject-source",
        chart_id=source.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=2.0e-2,
        max_defect_bound=1.0e-1,
        max_lipschitz_bound=70.0,
    )
    source_validation = check_validated_ordinary_ivp_chart(
        binding, source_tube, source
    )
    target_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="ordinary-tube:transition-reject-target",
        chart_id=target.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=1.0e-3,
        tube_radius=3.0e-2,
        max_defect_bound=2.0e-1,
        max_lipschitz_bound=100.0,
    )
    transition = OrdinaryEnclosureTransitionCertificate(
        transition_id="ordinary-enclosure-transition:reject",
        source_chart_id=source.chart_id,
        target_chart_id=target.chart_id,
        source_parameter=0.02,
        target_parameter=0.0,
        handoff_time=0.02,
        max_time_gap=0.0,
    )

    result = check_ordinary_enclosure_transition(
        transition, source, target, source_validation, target_tube
    )

    assert result.target_tube_result.certified
    assert not result.certified
    assert "ordinary_enclosure_transition_target_error_covers_handoff" in (
        result.missing_obligations
    )


def test_validated_ordinary_ivp_chain_certifies_oriented_target_coverage():
    charts, _ = _ordinary_chart_chain()
    source, target = charts
    binding = InitialValueProblemBindingCertificate(
        binding_id="ivp-binding:validated-chain",
        chart_id=source.chart_id,
        masses=source.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=source.position_coefficients[0],
        velocities=source.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )
    tubes = (
        OrdinaryAposterioriTubeCertificate(
            "ordinary-tube:chain-0", source.chart_id, 0.0, 0.0, 2.0e-2, 1.0e-1, 70.0
        ),
        OrdinaryAposterioriTubeCertificate(
            "ordinary-tube:chain-1", target.chart_id, 0.0, 2.0e-3, 3.0e-2, 2.0e-1, 100.0
        ),
    )
    transitions = (
        OrdinaryEnclosureTransitionCertificate(
            "ordinary-enclosure-transition:chain-0-1",
            source.chart_id,
            target.chart_id,
            0.02,
            0.0,
            0.02,
            0.0,
        ),
    )
    chain = ValidatedOrdinaryIVPChainCertificate(
        chain_id="validated-ordinary-chain",
        chart_ids=(source.chart_id, target.chart_id),
        transition_ids=(transitions[0].transition_id,),
        target_physical_time_interval=(0.0, 0.04),
    )

    result = check_validated_ordinary_ivp_chain(
        chain, binding, charts, tubes, transitions
    )

    assert type(result) is ValidatedOrdinaryIVPChainCheckResult
    assert result.certified
    assert result.exact_ivp_enclosure_certified
    assert result.target_state_enclosure_certified
    assert result.target_time == 0.04
    assert len(result.target_position_intervals) == 3
    assert len(result.target_velocity_intervals) == 3
    assert all(
        lower <= upper
        for body in result.target_position_intervals
        for lower, upper in body
    )
    assert result.covered_physical_time_interval == (0.0, 0.04)
    assert result.missing_obligations == ()


def test_validated_ordinary_ivp_chain_rejects_uncovered_target_interval():
    charts, _ = _ordinary_chart_chain()
    source, target = charts
    binding = InitialValueProblemBindingCertificate(
        binding_id="ivp-binding:validated-chain-target-reject",
        chart_id=source.chart_id,
        masses=source.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=source.position_coefficients[0],
        velocities=source.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )
    tubes = (
        OrdinaryAposterioriTubeCertificate(
            "ordinary-tube:chain-reject-0", source.chart_id, 0.0, 0.0, 2.0e-2, 1.0e-1, 70.0
        ),
        OrdinaryAposterioriTubeCertificate(
            "ordinary-tube:chain-reject-1", target.chart_id, 0.0, 2.0e-3, 3.0e-2, 2.0e-1, 100.0
        ),
    )
    transitions = (
        OrdinaryEnclosureTransitionCertificate(
            "ordinary-enclosure-transition:chain-reject",
            source.chart_id,
            target.chart_id,
            0.02,
            0.0,
            0.02,
            0.0,
        ),
    )
    chain = ValidatedOrdinaryIVPChainCertificate(
        chain_id="validated-ordinary-chain-target-reject",
        chart_ids=(source.chart_id, target.chart_id),
        transition_ids=(transitions[0].transition_id,),
        target_physical_time_interval=(0.0, 0.05),
    )

    result = check_validated_ordinary_ivp_chain(
        chain, binding, charts, tubes, transitions
    )

    assert not result.certified
    assert "validated_ordinary_chain_target_interval_covered" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_duplicate_chart_ids_in_bundle():
    charts, _ = _ordinary_chart_chain()
    duplicate = replace(
        charts[1],
        chart_id=charts[0].chart_id,
        certificate_id="ordinary-chart:duplicate-chart-id",
    )

    verifier = verify_chart_certificates((charts[0], duplicate))

    assert not verifier.certified
    assert "bundle_chart_ids_unique" in verifier.missing_obligations


def test_independent_checker_rejects_duplicate_certificate_ids_in_bundle():
    charts, _ = _ordinary_chart_chain()
    duplicate = replace(
        charts[1],
        chart_id="ordinary-chart:duplicate-certificate-id",
        certificate_id=charts[0].certificate_id,
    )

    verifier = verify_chart_certificates((charts[0], duplicate))

    assert not verifier.certified
    assert "bundle_certificate_ids_unique" in verifier.missing_obligations


def test_independent_checker_accepts_serialized_chart_chain_coverage():
    charts, transitions = _ordinary_chart_chain()
    chain = ChartChainCertificate(
        certificate_id="chart-chain:ordinary",
        chain_id="ordinary-chain",
        chain_type="finite_time_chart_chain",
        chart_ids=tuple(chart.chart_id for chart in charts),
        transition_ids=tuple(transition.transition_id for transition in transitions),
        target_physical_time_interval=(
            charts[0].physical_time_interval[0],
            charts[-1].physical_time_interval[1],
        ),
    )
    round_trip = ChartChainCertificate.from_dict(chain.to_dict())
    verifier = verify_chart_certificates(
        charts,
        transitions,
        chart_chains=(round_trip,),
    )
    result = verifier.chart_chain_results[0]
    details = {obligation.obligation: obligation for obligation in result.obligations}

    assert result.certified
    assert result.chart_count == 2
    assert details["chart_chain_exact_rational_time_coverage"].certified
    assert "exact rational interval arithmetic" in (
        details["chart_chain_exact_rational_time_coverage"].detail
    )
    assert verifier.certified
    assert verifier.checked_chart_chain_count == 1
    assert verifier.proof_grade_finite_atlas_bundle_certified
    assert verifier.proof_grade_finite_atlas_blockers == ()
    assert verifier.missing_obligations == ()
    assert "chart_chain_exact_rational_time_coverage" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )
    assert "finite_time_chart_chain_proof_grade_time_coverage_not_implemented" not in (
        verifier.proof_grade_arithmetic_blockers
    )


def test_independent_checker_rejects_chart_chain_missing_transition():
    charts, transitions = _ordinary_chart_chain()
    chain = ChartChainCertificate(
        certificate_id="chart-chain:missing-transition",
        chain_id="missing-transition-chain",
        chain_type="finite_time_chart_chain",
        chart_ids=tuple(chart.chart_id for chart in charts),
        transition_ids=(),
        target_physical_time_interval=(
            charts[0].physical_time_interval[0],
            charts[-1].physical_time_interval[1],
        ),
    )
    verifier = verify_chart_certificates(
        charts,
        transitions,
        chart_chains=(chain,),
    )
    direct = check_chart_chain(
        chain,
        charts,
        transitions,
        verifier.chart_results,
        verifier.transition_results,
    )

    assert not verifier.certified
    assert not direct.certified
    assert "chart_chain_transition_count_matches" in direct.missing_obligations
    assert "chart_chain_transition_count_matches" in verifier.missing_obligations


def test_independent_checker_rejects_chart_chain_target_coverage_gap():
    charts, transitions = _ordinary_chart_chain()
    chain = ChartChainCertificate(
        certificate_id="chart-chain:target-gap",
        chain_id="target-gap-chain",
        chain_type="finite_time_chart_chain",
        chart_ids=tuple(chart.chart_id for chart in charts),
        transition_ids=tuple(transition.transition_id for transition in transitions),
        target_physical_time_interval=(
            charts[0].physical_time_interval[0],
            charts[-1].physical_time_interval[1] + 0.01,
        ),
    )
    verifier = verify_chart_certificates(
        charts,
        transitions,
        chart_chains=(chain,),
    )

    assert not verifier.certified
    assert "chart_chain_target_interval_covered" in verifier.missing_obligations
    assert "chart_chain_exact_rational_time_coverage" in verifier.missing_obligations


def test_independent_checker_accepts_serialized_branch_union():
    charts, transitions = _ordinary_chart_chain()
    branch_union = BranchUnionCertificate(
        certificate_id="branch-union:ordinary-chain",
        union_id="ordinary-chain-union",
        union_type="finite_time_branch_union",
        leaf_response_certificate_ids=tuple(chart.certificate_id for chart in charts),
        leaf_target_intervals=tuple(chart.physical_time_interval for chart in charts),
        aggregate_target_interval=(
            charts[0].physical_time_interval[0],
            charts[-1].physical_time_interval[1],
        ),
        leaf_kinds=("positive_margin_unique_event", "positive_margin_unique_event"),
    )
    round_trip = BranchUnionCertificate.from_dict(branch_union.to_dict())
    verifier = verify_chart_certificates(
        charts,
        transitions,
        branch_unions=(round_trip,),
    )
    result = verifier.branch_union_results[0]
    details = {obligation.obligation: obligation for obligation in result.obligations}

    assert result.certified
    assert result.leaf_count == 2
    assert details["branch_union_exact_rational_interval_aggregation"].certified
    assert "exact rational interval arithmetic" in (
        details["branch_union_exact_rational_interval_aggregation"].detail
    )
    assert verifier.certified
    assert verifier.checked_branch_union_count == 1
    assert verifier.missing_obligations == ()
    assert "branch_union_exact_rational_interval_aggregation" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )
    assert "finite_time_branch_union_proof_grade_interval_aggregation_not_implemented" not in (
        verifier.proof_grade_arithmetic_blockers
    )


def test_independent_checker_accepts_stratified_branch_union_taxonomy():
    charts, transitions = _ordinary_chart_chain()
    branch_union = BranchUnionCertificate(
        certificate_id="branch-union:stratified-chain",
        union_id="stratified-chain-union",
        union_type="finite_time_stratified_branch_union",
        leaf_response_certificate_ids=tuple(chart.certificate_id for chart in charts),
        leaf_target_intervals=tuple(chart.physical_time_interval for chart in charts),
        aggregate_target_interval=(
            charts[0].physical_time_interval[0],
            charts[-1].physical_time_interval[1],
        ),
        leaf_kinds=("positive_margin_unique_event", "separated_binary_entry"),
    )
    verifier = verify_chart_certificates(
        charts,
        transitions,
        branch_unions=(branch_union,),
    )
    result = verifier.branch_union_results[0]

    assert result.certified
    assert "stratified_branch_union_leaf_kinds_supported" not in (
        result.missing_obligations
    )
    assert "stratified_branch_union_unsupported_strata_absent" not in (
        result.missing_obligations
    )


def test_independent_checker_rejects_stratified_branch_union_without_taxonomy():
    charts, _transitions = _ordinary_chart_chain()
    branch_union = BranchUnionCertificate(
        certificate_id="branch-union:stratified-no-taxonomy",
        union_id="stratified-no-taxonomy-union",
        union_type="finite_time_stratified_branch_union",
        leaf_response_certificate_ids=(charts[0].certificate_id,),
        leaf_target_intervals=(charts[0].physical_time_interval,),
        aggregate_target_interval=charts[0].physical_time_interval,
    )
    verifier = verify_chart_certificates(
        (charts[0],),
        branch_unions=(branch_union,),
    )
    result = verifier.branch_union_results[0]

    assert not result.certified
    assert "stratified_branch_union_leaf_kinds_present" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_stratified_branch_union_unsupported_stratum():
    charts, _transitions = _ordinary_chart_chain()
    branch_union = BranchUnionCertificate(
        certificate_id="branch-union:stratified-unsupported",
        union_id="stratified-unsupported-union",
        union_type="finite_time_stratified_branch_union",
        leaf_response_certificate_ids=(charts[0].certificate_id,),
        leaf_target_intervals=(charts[0].physical_time_interval,),
        aggregate_target_interval=charts[0].physical_time_interval,
        leaf_kinds=("unsupported_analytic_stratum",),
    )
    verifier = verify_chart_certificates(
        (charts[0],),
        branch_unions=(branch_union,),
    )
    result = verifier.branch_union_results[0]

    assert not result.certified
    assert "stratified_branch_union_unsupported_strata_absent" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_branch_union_without_checked_leaf_response():
    chart = _ordinary_chart_certificate()
    branch_union = BranchUnionCertificate(
        certificate_id="branch-union:missing-leaf",
        union_id="missing-leaf-union",
        union_type="finite_time_branch_union",
        leaf_response_certificate_ids=(chart.certificate_id, "missing-chart"),
        leaf_target_intervals=(chart.physical_time_interval, chart.physical_time_interval),
        aggregate_target_interval=chart.physical_time_interval,
    )
    result = check_branch_union(branch_union, (check_ordinary_taylor_chart(chart),))
    verifier = verify_chart_certificates((chart,), branch_unions=(branch_union,))

    assert not result.certified
    assert "branch_union_leaf_responses_independently_checked" in (
        result.missing_obligations
    )
    assert not verifier.certified
    assert "branch_union_leaf_responses_independently_checked" in (
        verifier.missing_obligations
    )


def test_independent_checker_rejects_branch_union_target_hull_gap():
    chart = _ordinary_chart_certificate()
    branch_union = BranchUnionCertificate(
        certificate_id="branch-union:hull-gap",
        union_id="hull-gap-union",
        union_type="finite_time_branch_union",
        leaf_response_certificate_ids=(chart.certificate_id,),
        leaf_target_intervals=(chart.physical_time_interval,),
        aggregate_target_interval=(
            chart.physical_time_interval[0],
            chart.physical_time_interval[1] - 0.5 * (
                chart.physical_time_interval[1] - chart.physical_time_interval[0]
            ),
        ),
    )
    result = check_branch_union(branch_union, (check_ordinary_taylor_chart(chart),))

    assert not result.certified
    assert "branch_union_aggregate_contains_leaf_targets" in result.missing_obligations
    assert "branch_union_exact_rational_interval_aggregation" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_corrupted_ordinary_transition():
    charts, transitions = _ordinary_chart_chain()
    second = charts[1]
    position = [
        [list(axis_values) for axis_values in degree]
        for degree in second.position_coefficients
    ]
    position[0][0][0] += 0.01
    corrupted_second = replace(
        second,
        position_coefficients=tuple(
            tuple(tuple(axis_values) for axis_values in degree)
            for degree in position
        ),
    )
    verifier = verify_chart_certificates((charts[0], corrupted_second), transitions)
    transition_result = verifier.transition_results[0]

    assert not verifier.certified
    assert not transition_result.certified
    assert "transition_state_continuity" in transition_result.missing_obligations
    assert "transition_exact_rational_state_continuity" in (
        transition_result.missing_obligations
    )
    assert "transition_state_continuity" in verifier.missing_obligations


def test_independent_checker_rejects_unknown_transition_type():
    charts, transitions = _ordinary_chart_chain()
    unsupported = replace(transitions[0], transition_type="opaque_handoff")
    result = check_ordinary_chart_transition(unsupported, charts)

    assert not result.certified
    assert "transition_type_supported" in result.missing_obligations


def test_independent_checker_accepts_serialized_planar_lc_binary_chart():
    certificate = _planar_lc_binary_chart_certificate()
    result = check_planar_levi_civita_binary_chart(certificate)
    round_trip = PlanarLeviCivitaBinaryChartCertificate.from_dict(
        certificate.to_dict(),
    )
    round_trip_result = check_planar_levi_civita_binary_chart(round_trip)
    verifier = verify_chart_certificates((certificate,))

    assert result.certified
    assert result.missing_obligations == ()
    assert result.max_coefficient_residual <= certificate.coefficient_tolerance
    assert result.max_sampled_newton_residual <= (
        certificate.projected_residual_tolerance
    )
    assert "interval_planar_lc_regularized_rhs_residual" not in (
        result.missing_obligations
    )
    details = {obligation.obligation: obligation for obligation in result.obligations}
    assert details[
        "planar_lc_exact_rational_regularized_residual_polynomials"
    ].certified
    assert "exact rational interval arithmetic" in (
        details[
            "planar_lc_exact_rational_regularized_residual_polynomials"
        ].detail
    )
    assert "interval_projected_newton_residual_away_from_binary_collision" not in (
        result.missing_obligations
    )
    assert round_trip_result.certified
    assert verifier.certified
    assert verifier.planar_levi_civita_binary_chart_count == 1


def test_planar_lc_aposteriori_tube_encloses_lifted_solution():
    chart = _planar_lc_binary_chart_certificate()
    tube = PlanarLCAposterioriTubeCertificate(
        tube_id="planar-lc-tube:regular",
        chart_id=chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=1.0e-8,
        max_defect_bound=1.0e-9,
        max_lipschitz_bound=5.0,
    )

    result = check_planar_lc_aposteriori_tube(tube, chart)

    assert type(result) is PlanarLCAposterioriTubeCheckResult
    assert result.certified
    assert result.checker_id == "independent_planar_lc_aposteriori_tube_checker_v2"
    assert result.mass_arithmetic_kernel_id == (
        PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
    )
    assert "planar_lc_tube_outward_mass_arithmetic_certified" in {
        obligation.obligation for obligation in result.obligations
    }
    assert not replace(
        result,
        mass_arithmetic_kernel_id="wrong-mass-kernel",
    ).certified

    class _ForgedTubeResult(PlanarLCAposterioriTubeCheckResult):
        pass

    assert not _ForgedTubeResult(**result.__dict__).certified
    assert result.lifted_exact_solution_enclosure_certified
    assert result.third_body_distance_floor > 0.0
    assert result.gronwall_error_bound < tube.tube_radius


def test_planar_lc_aposteriori_tube_crosses_exact_binary_collision():
    chart = _planar_lc_exact_collision_chart_certificate()
    legacy_result = check_planar_levi_civita_binary_chart(chart)
    tube = PlanarLCAposterioriTubeCertificate(
        tube_id="planar-lc-tube:exact-collision",
        chart_id=chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=1.0e-6,
        max_defect_bound=1.0e-7,
        max_lipschitz_bound=2.0,
        require_pair_energy_constraint=True,
    )

    result = check_planar_lc_aposteriori_tube(tube, chart)

    assert not legacy_result.certified
    assert "interval_projected_newton_residual_away_from_binary_collision" in (
        legacy_result.missing_obligations
    )
    assert result.certified
    assert result.third_body_distance_floor > 2.0
    assert result.pair_energy_constraint_anchor_certified
    assert result.constrained_newtonian_lift_certified
    assert result.anchor_pair_energy_constraint_residual == 0.0


def test_planar_lc_tube_does_not_promote_approximate_energy_constraint():
    chart = _planar_lc_binary_chart_certificate()
    tube = PlanarLCAposterioriTubeCertificate(
        tube_id="planar-lc-tube:constraint-required",
        chart_id=chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=1.0e-8,
        max_defect_bound=1.0e-9,
        max_lipschitz_bound=5.0,
        require_pair_energy_constraint=True,
    )

    result = check_planar_lc_aposteriori_tube(tube, chart)

    assert not result.pair_energy_constraint_anchor_certified
    assert not result.certified
    assert "planar_lc_tube_pair_energy_constraint_when_required" in (
        result.missing_obligations
    )


def test_planar_lc_positive_anchor_ball_is_not_promoted_to_constrained_family():
    chart = _planar_lc_exact_collision_chart_certificate()
    result = check_planar_lc_aposteriori_tube(
        PlanarLCAposterioriTubeCertificate(
            "planar-lc-tube:positive-anchor-ball", chart.chart_id,
            0.0, 1.0e-12, 1.0e-6, 1.0e-7, 5.0, True,
        ),
        chart,
    )

    assert result.certified
    assert result.pair_energy_constraint_anchor_certified
    assert not result.anchor_is_polynomial_center
    assert not result.constrained_newtonian_lift_certified


def _exact_collision_anchor_bundle():
    chart = _planar_lc_exact_collision_chart_certificate()
    tube = PlanarLCAposterioriTubeCertificate(
        "planar-lc-tube:exact-collision-anchor", chart.chart_id,
        0.0, 0.0, 1.0e-6, 1.0e-7, 2.0, True,
    )
    certificate = PlanarLCExactCollisionAnchorCertificate(
        "planar-lc-collision:0", chart.chart_id, tube.tube_id, 0.0,
    )
    return certificate, tube, chart


def test_planar_lc_exact_collision_anchor_certifies_selected_exact_ivp():
    certificate, tube, chart = _exact_collision_anchor_bundle()
    result = check_planar_lc_exact_collision_anchor(certificate, tube, chart)

    assert type(result) is PlanarLCExactCollisionAnchorCheckResult
    assert result.certified
    assert result.tube_result.certified
    assert result.isolated_binary_collision_certified
    assert result.local_physical_time_strictly_increasing_certified
    assert result.exact_speed_squared == 1.0
    assert result.pair_mass == 2.0
    assert result.mass_ratio_arithmetic_exact
    assert result.collision_physical_time == 0.0


def test_planar_lc_collision_anchor_rejects_positive_initial_error_ball():
    certificate, tube, chart = _exact_collision_anchor_bundle()
    result = check_planar_lc_exact_collision_anchor(
        certificate, replace(tube, initial_error_bound=1.0e-12), chart
    )

    assert result.tube_result.certified
    assert not result.certified
    assert "planar_lc_collision_zero_error_center_anchor" in (
        result.missing_obligations
    )


@pytest.mark.parametrize(
    "field, center, missing",
    [
        ("z_coefficients", (1.0e-12, 0.0), "planar_lc_collision_exact_z_zero"),
        (
            "z_velocity_coefficients",
            (1.0 + 1.0e-12, 0.0),
            "planar_lc_collision_exact_pair_energy_constraint",
        ),
    ],
)
def test_planar_lc_collision_anchor_rejects_noncollision_or_off_constraint_center(
    field, center, missing
):
    certificate, tube, chart = _exact_collision_anchor_bundle()
    coefficients = list(getattr(chart, field))
    coefficients[0] = center
    corrupted = replace(chart, **{field: tuple(coefficients)})
    result = check_planar_lc_exact_collision_anchor(
        certificate, tube, corrupted
    )

    assert not result.certified
    assert missing in result.missing_obligations


def test_two_sided_lc_passage_certifies_one_exact_binary_collision_continuation():
    endpoint = 1.0e-2
    solution = _planar_lc_exact_collision_solution()
    source_chart = replace(
        _planar_lc_exact_collision_chart_certificate(),
        parameter_interval=(-endpoint, endpoint),
    )
    source_tube = PlanarLCAposterioriTubeCertificate(
        "planar-lc-tube:two-sided-passage", source_chart.chart_id,
        0.0, 0.0, 1.0e-5, 1.0, 10.0, True,
    )
    collision = check_planar_lc_exact_collision_anchor(
        PlanarLCExactCollisionAnchorCertificate(
            "planar-lc-collision:two-sided", source_chart.chart_id,
            source_tube.tube_id, 0.0,
        ),
        source_tube,
        source_chart,
    )
    target_charts = []
    target_tubes = []
    for side, parameter, interval in (
        ("left", -endpoint, (-1.0e-12, 0.0)),
        ("right", endpoint, (0.0, 1.0e-12)),
    ):
        positions, velocities = regularized_binary_collision_chart_to_planar(
            solution.state_at(parameter)
        )
        ordinary_solution = construct_taylor_solution(
            positions, velocities, solution.masses, order=12
        )
        chart = ordinary_taylor_chart_certificate_from_solution(
            ordinary_solution,
            certificate_id=f"ordinary-collision-{side}",
            chart_id=f"ordinary-collision-{side}",
            parameter_interval=interval,
            physical_time_interval=interval,
            coefficient_tolerance=1.0,
            residual_tolerance=1.0,
            tail_bound=0.0,
            sample_count=3,
        )
        tube = WeightedOrdinaryAposterioriTubeCertificate(
            f"weighted-ordinary-collision-{side}", chart.chart_id,
            0.0, 5.0e-6, 1.0e-1, 1.0e-5, 1.0, 1.0e3, 1.0e9,
        )
        target_charts.append(chart)
        target_tubes.append(tube)

    result = check_planar_lc_two_sided_collision_passage(
        PlanarLCTwoSidedCollisionPassageCertificate(
            "planar-lc-passage:two-sided",
            source_chart.chart_id,
            collision.collision_id,
            target_charts[0].chart_id,
            target_charts[1].chart_id,
            -endpoint,
            endpoint,
            0.0,
            0.0,
        ),
        source_chart,
        collision,
        target_charts[0],
        target_charts[1],
        target_tubes[0],
        target_tubes[1],
    )

    assert collision.certified
    assert type(result) is PlanarLCTwoSidedCollisionPassageCheckResult
    assert result.certified
    assert result.generalized_binary_collision_continuation_certified
    assert result.left_rho_lower_bound > 0.0
    assert result.right_rho_lower_bound > 0.0
    assert np.isfinite(result.left_time_origin_interval).all()
    assert np.isfinite(result.right_time_origin_interval).all()


def test_ordinary_to_lc_enclosure_transition_binds_exact_source_branch():
    charts, _ = _ordinary_to_lc_transition_bundle()
    ordinary = replace(
        charts[0],
        parameter_interval=(0.0, 1.0e-6),
        physical_time_interval=(0.0, 1.0e-6),
        residual_tolerance=10.0,
    )
    lc = replace(
        charts[1],
        parameter_interval=(0.0, 1.0e-6),
    )
    binding = InitialValueProblemBindingCertificate(
        "ivp-binding:ordinary-to-lc",
        ordinary.chart_id,
        ordinary.masses,
        0.0,
        0.0,
        ordinary.position_coefficients[0],
        ordinary.velocity_coefficients[0],
        0.0,
        0.0,
        0.0,
    )
    ordinary_tube = OrdinaryAposterioriTubeCertificate(
        "ordinary-tube:ordinary-to-lc",
        ordinary.chart_id,
        0.0,
        0.0,
        1.0e-6,
        10.0,
        1.0e6,
    )
    source = check_validated_ordinary_ivp_chart(
        binding, ordinary_tube, ordinary
    )
    lc_tube = PlanarLCAposterioriTubeCertificate(
        tube_id="planar-lc-tube:ordinary-entry",
        chart_id=lc.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=1.0e-5,
        tube_radius=1.0e-1,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )
    transition = OrdinaryToPlanarLCEnclosureTransitionCertificate(
        "ordinary-to-lc-enclosure:entry",
        ordinary.chart_id,
        lc.chart_id,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    result = check_ordinary_to_planar_lc_enclosure_transition(
        transition, ordinary, lc, source, lc_tube
    )

    assert type(result) is OrdinaryToPlanarLCEnclosureTransitionCheckResult
    assert source.certified
    assert result.certified
    assert result.constrained_newtonian_lift_certified
    assert result.physical_time_strictly_monotone_certified
    assert result.entry_lift_rho_lower_bound > 0.0
    assert result.lift_branch_count >= 1
    assert result.max_lift_box_gap <= lc_tube.initial_error_bound


def test_ordinary_to_lc_transition_rejects_tube_smaller_than_lift_atlas():
    charts, _ = _ordinary_to_lc_transition_bundle()
    ordinary = replace(
        charts[0],
        parameter_interval=(0.0, 1.0e-6),
        physical_time_interval=(0.0, 1.0e-6),
        residual_tolerance=10.0,
    )
    lc = replace(charts[1], parameter_interval=(0.0, 1.0e-6))
    binding = InitialValueProblemBindingCertificate(
        "ivp-binding:ordinary-to-lc-reject",
        ordinary.chart_id,
        ordinary.masses,
        0.0,
        0.0,
        ordinary.position_coefficients[0],
        ordinary.velocity_coefficients[0],
        0.0,
        0.0,
        0.0,
    )
    source = check_validated_ordinary_ivp_chart(
        binding,
        OrdinaryAposterioriTubeCertificate(
            "ordinary-tube:ordinary-to-lc-reject",
            ordinary.chart_id,
            0.0,
            0.0,
            1.0e-6,
            10.0,
            1.0e6,
        ),
        ordinary,
    )
    lc_tube = PlanarLCAposterioriTubeCertificate(
        "planar-lc-tube:ordinary-entry-reject",
        lc.chart_id,
        0.0,
        1.0e-10,
        1.0e-1,
        10.0,
        1.0e6,
    )
    transition = OrdinaryToPlanarLCEnclosureTransitionCertificate(
        "ordinary-to-lc-enclosure:reject",
        ordinary.chart_id,
        lc.chart_id,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    result = check_ordinary_to_planar_lc_enclosure_transition(
        transition, ordinary, lc, source, lc_tube
    )

    assert result.target_tube_result.certified
    assert not result.certified
    assert "ordinary_to_lc_target_initial_error_contains_lift_atlas" in (
        result.missing_obligations
    )


def test_ordinary_to_lc_transition_requires_time_inside_14d_initial_ball():
    charts, _ = _ordinary_to_lc_transition_bundle()
    ordinary = replace(
        charts[0],
        parameter_interval=(0.0, 1.0e-6),
        physical_time_interval=(0.0, 1.0e-6),
        residual_tolerance=10.0,
    )
    time_shift = 2.0e-5
    lc = replace(
        charts[1],
        parameter_interval=(0.0, 1.0e-6),
        physical_time_coefficients=(
            charts[1].physical_time_coefficients[0] + time_shift,
        )
        + charts[1].physical_time_coefficients[1:],
    )
    binding = InitialValueProblemBindingCertificate(
        "ivp-binding:ordinary-to-lc-time-radius",
        ordinary.chart_id,
        ordinary.masses,
        0.0,
        0.0,
        ordinary.position_coefficients[0],
        ordinary.velocity_coefficients[0],
        0.0,
        0.0,
        0.0,
    )
    source = check_validated_ordinary_ivp_chart(
        binding,
        OrdinaryAposterioriTubeCertificate(
            "ordinary-tube:ordinary-to-lc-time-radius",
            ordinary.chart_id,
            0.0,
            0.0,
            1.0e-6,
            10.0,
            1.0e6,
        ),
        ordinary,
    )
    target_tube = PlanarLCAposterioriTubeCertificate(
        "planar-lc-tube:ordinary-entry-time-radius",
        lc.chart_id,
        0.0,
        1.0e-5,
        1.0e-1,
        10.0,
        1.0e6,
    )
    transition = OrdinaryToPlanarLCEnclosureTransitionCertificate(
        "ordinary-to-lc-enclosure:time-radius",
        ordinary.chart_id,
        lc.chart_id,
        0.0,
        0.0,
        0.0,
        time_shift,
    )

    result = check_ordinary_to_planar_lc_enclosure_transition(
        transition,
        ordinary,
        lc,
        source,
        target_tube,
    )
    obligations = {
        obligation.obligation: obligation.certified
        for obligation in result.obligations
    }

    assert source.certified
    assert result.target_tube_result.certified
    assert obligations["ordinary_to_lc_handoff_time_matches"]
    assert obligations["ordinary_to_lc_target_initial_error_contains_lift_atlas"]
    assert not obligations[
        "ordinary_to_lc_target_initial_error_contains_physical_time"
    ]
    assert result.target_initial_time_gap == time_shift
    assert result.target_initial_time_gap > target_tube.initial_error_bound
    assert not result.certified


def test_legacy_ordinary_to_lc_rejects_offset_physical_time_parameterization():
    """A permissive raw binding tolerance cannot redefine exact source time."""

    charts, _ = _ordinary_to_lc_transition_bundle()
    source_chart = replace(
        charts[0],
        parameter_interval=(0.0, 1.0e-6),
        physical_time_interval=(2.0e-5, 2.1e-5),
        residual_tolerance=10.0,
    )
    target_chart = replace(
        charts[1],
        parameter_interval=(0.0, 1.0e-6),
        physical_time_coefficients=(
            charts[1].physical_time_coefficients[0] + 2.0e-5,
        )
        + charts[1].physical_time_coefficients[1:],
    )
    source_validation = check_validated_ordinary_ivp_chart(
        InitialValueProblemBindingCertificate(
            "ivp-binding:ordinary-to-lc-offset-time",
            source_chart.chart_id,
            source_chart.masses,
            0.0,
            0.0,
            source_chart.position_coefficients[0],
            source_chart.velocity_coefficients[0],
            2.0e-5,
            0.0,
            0.0,
        ),
        OrdinaryAposterioriTubeCertificate(
            "ordinary-tube:ordinary-to-lc-offset-time",
            source_chart.chart_id,
            0.0,
            0.0,
            1.0e-6,
            10.0,
            1.0e6,
        ),
        source_chart,
    )
    target_tube = PlanarLCAposterioriTubeCertificate(
        "planar-lc-tube:ordinary-entry-offset-time",
        target_chart.chart_id,
        0.0,
        1.0e-5,
        1.0e-1,
        10.0,
        1.0e6,
    )
    result = check_ordinary_to_planar_lc_enclosure_transition(
        OrdinaryToPlanarLCEnclosureTransitionCertificate(
            "ordinary-to-lc-enclosure:offset-time",
            source_chart.chart_id,
            target_chart.chart_id,
            0.0,
            0.0,
            2.0e-5,
            0.0,
        ),
        source_chart,
        target_chart,
        source_validation,
        target_tube,
    )
    obligations = {
        obligation.obligation: obligation.certified
        for obligation in result.obligations
    }

    assert source_validation.certified
    assert source_validation.binding_result.time_gap == 2.0e-5
    assert result.target_tube_result.certified
    assert obligations["ordinary_to_lc_parameters_inside_and_anchor_matches"]
    assert not obligations[
        "ordinary_to_lc_source_physical_time_parameterization_exact"
    ]
    assert not result.certified


@pytest.mark.parametrize("target_initial_error, expected", [(2.0e-3, True), (1.0e-3, False)])
def test_lc_to_ordinary_exit_projects_certified_punctured_enclosure(
    target_initial_error, expected
):
    (ordinary_entry, lc), _ = _ordinary_to_lc_transition_bundle()
    ordinary_entry = replace(
        ordinary_entry,
        parameter_interval=(0.0, 1.0e-6),
        physical_time_interval=(0.0, 1.0e-6),
        residual_tolerance=10.0,
    )
    binding = InitialValueProblemBindingCertificate(
        "ivp-binding:lc-exit-source", ordinary_entry.chart_id,
        ordinary_entry.masses, 0.0, 0.0,
        ordinary_entry.position_coefficients[0],
        ordinary_entry.velocity_coefficients[0], 0.0, 0.0, 0.0,
    )
    source = check_validated_ordinary_ivp_chart(
        binding,
        OrdinaryAposterioriTubeCertificate(
            "ordinary-tube:lc-exit-source", ordinary_entry.chart_id,
            0.0, 0.0, 1.0e-6, 10.0, 1.0e6,
        ),
        ordinary_entry,
    )
    lc_tube = PlanarLCAposterioriTubeCertificate(
        "planar-lc-tube:exit-source", lc.chart_id,
        0.0, 1.0e-5, 1.0e-3, 10.0, 1.0e6,
    )
    entry = check_ordinary_to_planar_lc_enclosure_transition(
        OrdinaryToPlanarLCEnclosureTransitionCertificate(
            "ordinary-to-lc-enclosure:for-exit", ordinary_entry.chart_id,
            lc.chart_id, 0.0, 0.0, 0.0, 0.0,
        ),
        ordinary_entry, lc, source, lc_tube,
    )
    (_, ordinary_exit), _ = _lc_to_ordinary_transition_bundle()
    ordinary_exit = replace(
        ordinary_exit,
        parameter_interval=(0.0, 1.0e-6),
        physical_time_interval=(0.0, 1.0e-6),
    )
    target_tube = OrdinaryAposterioriTubeCertificate(
        "ordinary-tube:lc-exit-target", ordinary_exit.chart_id,
        0.0, target_initial_error, 2.0e-2, 10.0, 1.0e6,
    )
    result = check_planar_lc_to_ordinary_enclosure_transition(
        PlanarLCToOrdinaryEnclosureTransitionCertificate(
            "lc-to-ordinary-enclosure:exit", lc.chart_id,
            ordinary_exit.chart_id, 0.006, 0.0,
        ),
        lc, ordinary_exit, entry, target_tube,
    )

    assert entry.certified
    assert result.certified is expected
    assert result.punctured_newton_projection_certified is expected
    assert result.source_rho_lower_bound > 0.0
    assert result.max_projected_lift_gap > 1.0e-3
    assert np.isfinite(result.physical_time_origin_interval).all()
    if not expected:
        assert "lc_to_ordinary_target_initial_error_contains_projection" in (
            result.missing_obligations
        )


def test_planar_lc_aposteriori_tube_rejects_third_body_collision_domain():
    chart = _planar_lc_exact_collision_chart_certificate()
    zero_offsets = tuple((0.0, 0.0) for _ in chart.third_offset_coefficients)
    corrupted = replace(chart, third_offset_coefficients=zero_offsets)
    tube = PlanarLCAposterioriTubeCertificate(
        tube_id="planar-lc-tube:third-body-collision",
        chart_id=corrupted.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=1.0e-6,
        max_defect_bound=1.0,
        max_lipschitz_bound=1.0e6,
    )

    result = check_planar_lc_aposteriori_tube(tube, corrupted)

    assert not result.certified
    assert "planar_lc_tube_separated_third_body" in result.missing_obligations


def test_planar_lc_checker_certification_is_interval_not_sample_gated():
    certificate = replace(_planar_lc_binary_chart_certificate(), sample_count=1)
    result = check_planar_levi_civita_binary_chart(certificate)
    obligation_ids = {obligation.obligation for obligation in result.obligations}

    assert result.certified
    assert "sampled_physical_time_containment" not in obligation_ids
    assert "interval_physical_time_containment" in obligation_ids
    assert "sampled_regularized_binary_rhs_residual" not in obligation_ids
    assert "sampled_projected_newton_residual_away_from_binary_collision" not in (
        obligation_ids
    )
    assert "planar_lc_exact_rational_regularized_residual_polynomials" in obligation_ids
    assert "interval_planar_lc_regularized_rhs_residual" in obligation_ids
    assert "interval_projected_newton_residual_away_from_binary_collision" in (
        obligation_ids
    )


def test_planar_lc_checker_rejects_unsampled_physical_time_excursion():
    certificate = _planar_lc_binary_chart_certificate()
    lower, upper = certificate.parameter_interval
    magnitude = 1.0e14
    coefficients = list(certificate.physical_time_coefficients)
    coefficients[0] += -magnitude * lower * upper
    coefficients[1] += magnitude * (lower + upper)
    coefficients[2] += -magnitude
    corrupted = replace(
        certificate,
        physical_time_coefficients=tuple(coefficients),
        coefficient_tolerance=1.0e16,
        regularized_residual_tolerance=1.0e16,
        projected_residual_tolerance=1.0e16,
        sample_count=2,
    )
    result = check_planar_levi_civita_binary_chart(corrupted)
    details = {obligation.obligation: obligation for obligation in result.obligations}

    assert not result.certified
    assert "interval_physical_time_containment" in result.missing_obligations
    assert "sampled_physical_time_containment" not in details
    assert "diagnostic_sampled_physical_time_contained=True" in (
        details["interval_physical_time_containment"].detail
    )


def test_independent_checker_rejects_wide_planar_lc_projected_interval_residual():
    certificate = planar_levi_civita_binary_chart_certificate_from_solution(
        _planar_lc_binary_solution(),
        certificate_id="planar-lc-chart-wide",
        chart_id="planar-lc-wide",
        parameter_interval=(0.0, 0.012),
        coefficient_tolerance=1.0e-11,
        regularized_residual_tolerance=1.0e-8,
        projected_residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
        projection_rho_lower_bound=1.0e-10,
    )
    result = check_planar_levi_civita_binary_chart(certificate)

    assert not result.certified
    assert "sampled_projected_newton_residual_away_from_binary_collision" not in (
        result.missing_obligations
    )
    assert "interval_projected_newton_residual_away_from_binary_collision" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_corrupted_planar_lc_binary_recurrence():
    certificate = _planar_lc_binary_chart_certificate()
    z_velocity = [list(degree) for degree in certificate.z_velocity_coefficients]
    z_velocity[3][0] += 0.1
    corrupted = replace(
        certificate,
        z_velocity_coefficients=tuple(tuple(degree) for degree in z_velocity),
    )
    result = check_planar_levi_civita_binary_chart(corrupted)

    assert not result.certified
    assert "planar_lc_regularized_coefficient_recurrence" in (
        result.missing_obligations
    )
    assert "planar_lc_exact_rational_regularized_residual_polynomials" in (
        result.missing_obligations
    )
    assert "interval_planar_lc_regularized_rhs_residual" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_excessive_planar_lc_tail_model_bound():
    certificate = _planar_lc_binary_chart_certificate()
    inflated_tail = replace(
        certificate,
        tail_bound=10.0 * certificate.regularized_residual_tolerance,
    )
    result = check_planar_levi_civita_binary_chart(inflated_tail)

    assert not result.certified
    assert "interval_planar_lc_regularized_rhs_residual" in (
        result.missing_obligations
    )
    assert "tail_bound_admissible" not in result.missing_obligations


def test_independent_checker_accepts_serialized_spatial_ks_binary_chart():
    certificate = _spatial_ks_binary_chart_certificate()
    result = check_spatial_ks_binary_chart(certificate)
    round_trip = SpatialKSBinaryChartCertificate.from_dict(certificate.to_dict())
    round_trip_result = check_spatial_ks_binary_chart(round_trip)
    verifier = verify_chart_certificates((certificate,))

    assert result.certified
    assert result.missing_obligations == ()
    assert result.max_coefficient_residual <= certificate.coefficient_tolerance
    assert result.max_sampled_newton_residual <= (
        certificate.projected_residual_tolerance
    )
    assert "interval_spatial_ks_regularized_rhs_residual" not in (
        result.missing_obligations
    )
    details = {obligation.obligation: obligation for obligation in result.obligations}
    assert details[
        "spatial_ks_exact_rational_regularized_residual_polynomials"
    ].certified
    assert "exact rational interval arithmetic" in (
        details[
            "spatial_ks_exact_rational_regularized_residual_polynomials"
        ].detail
    )
    assert "interval_spatial_ks_projected_newton_residual_away_from_binary_collision" not in (
        result.missing_obligations
    )
    assert round_trip_result.certified
    assert verifier.certified
    assert verifier.spatial_ks_binary_chart_count == 1


def test_spatial_ks_checker_certification_is_interval_not_sample_gated():
    certificate = replace(_spatial_ks_binary_chart_certificate(), sample_count=1)
    result = check_spatial_ks_binary_chart(certificate)
    obligation_ids = {obligation.obligation for obligation in result.obligations}

    assert result.certified
    assert "sampled_physical_time_containment" not in obligation_ids
    assert "interval_physical_time_containment" in obligation_ids
    assert "sampled_spatial_ks_rhs_residual" not in obligation_ids
    assert "sampled_projected_newton_residual_away_from_binary_collision" not in (
        obligation_ids
    )
    assert "spatial_ks_exact_rational_regularized_residual_polynomials" in obligation_ids
    assert "interval_spatial_ks_regularized_rhs_residual" in obligation_ids
    assert "interval_spatial_ks_projected_newton_residual_away_from_binary_collision" in (
        obligation_ids
    )


def test_spatial_ks_checker_rejects_unsampled_physical_time_excursion():
    certificate = _spatial_ks_binary_chart_certificate()
    lower, upper = certificate.parameter_interval
    magnitude = 1.0e12
    coefficients = list(certificate.physical_time_coefficients)
    coefficients[0] += -magnitude * lower * upper
    coefficients[1] += magnitude * (lower + upper)
    coefficients[2] += -magnitude
    corrupted = replace(
        certificate,
        physical_time_coefficients=tuple(coefficients),
        coefficient_tolerance=1.0e16,
        regularized_residual_tolerance=1.0e16,
        projected_residual_tolerance=1.0e16,
        constraint_tolerance=1.0e16,
        sample_count=2,
    )
    result = check_spatial_ks_binary_chart(corrupted)
    details = {obligation.obligation: obligation for obligation in result.obligations}

    assert not result.certified
    assert "interval_physical_time_containment" in result.missing_obligations
    assert "sampled_physical_time_containment" not in details
    assert "diagnostic_sampled_physical_time_contained=True" in (
        details["interval_physical_time_containment"].detail
    )


def test_independent_checker_rejects_wide_spatial_ks_projected_interval_residual():
    certificate = spatial_ks_binary_chart_certificate_from_solution(
        _spatial_ks_binary_solution(),
        certificate_id="spatial-ks-chart-wide",
        chart_id="spatial-ks-wide",
        parameter_interval=(0.0, 0.012),
        coefficient_tolerance=1.0e-11,
        regularized_residual_tolerance=1.0e-8,
        projected_residual_tolerance=1.0e-8,
        constraint_tolerance=1.0e-10,
        tail_bound=1.0e-9,
        sample_count=7,
        projection_rho_lower_bound=1.0e-10,
    )
    result = check_spatial_ks_binary_chart(certificate)

    assert not result.certified
    assert "sampled_projected_newton_residual_away_from_binary_collision" not in (
        result.missing_obligations
    )
    assert "interval_spatial_ks_projected_newton_residual_away_from_binary_collision" in (
        result.missing_obligations
    )


def test_independent_checker_accepts_spatial_ks_projected_punctured_slabs():
    solution = _spatial_ks_binary_crossing_solution()
    parameter_interval = (0.0, 0.02)
    certificate = spatial_ks_binary_chart_certificate_from_solution(
        solution,
        certificate_id="spatial-ks-crossing-chart",
        chart_id="spatial-ks-crossing",
        parameter_interval=parameter_interval,
        coefficient_tolerance=1.0e-10,
        regularized_residual_tolerance=1.0e-8,
        projected_residual_tolerance=1.0e18,
        constraint_tolerance=1.0e-8,
        tail_bound=1.0e-12,
        sample_count=9,
        projection_rho_lower_bound=1.0e-14,
    )
    result = check_spatial_ks_binary_chart(certificate)

    assert result.certified
    assert "interval_spatial_ks_projected_newton_residual_away_from_binary_collision" not in (
        result.missing_obligations
    )
    details = {obligation.obligation: obligation for obligation in result.obligations}
    assert "inf" not in details[
        "interval_spatial_ks_projected_newton_residual_away_from_binary_collision"
    ].detail


def test_independent_checker_rejects_corrupted_spatial_ks_binary_recurrence():
    certificate = _spatial_ks_binary_chart_certificate()
    u_velocity = [list(degree) for degree in certificate.u_velocity_coefficients]
    u_velocity[3][0] += 0.1
    corrupted = replace(
        certificate,
        u_velocity_coefficients=tuple(tuple(degree) for degree in u_velocity),
    )
    result = check_spatial_ks_binary_chart(corrupted)

    assert not result.certified
    assert "spatial_ks_regularized_coefficient_recurrence" in (
        result.missing_obligations
    )
    assert "spatial_ks_exact_rational_regularized_residual_polynomials" in (
        result.missing_obligations
    )
    assert "interval_spatial_ks_regularized_rhs_residual" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_excessive_spatial_ks_tail_model_bound():
    certificate = _spatial_ks_binary_chart_certificate()
    inflated_tail = replace(
        certificate,
        tail_bound=10.0 * certificate.regularized_residual_tolerance,
    )
    result = check_spatial_ks_binary_chart(inflated_tail)

    assert not result.certified
    assert "interval_spatial_ks_regularized_rhs_residual" in (
        result.missing_obligations
    )
    assert "tail_bound_admissible" not in result.missing_obligations


def test_independent_checker_accepts_serialized_total_collision_fuchsian_stop_chart():
    certificate = _total_collision_fuchsian_stop_chart_certificate()
    result = check_total_collision_fuchsian_stop_chart(certificate)
    round_trip = TotalCollisionFuchsianStopChartCertificate.from_dict(
        certificate.to_dict(),
    )
    round_trip_result = check_total_collision_fuchsian_stop_chart(round_trip)
    verifier = verify_chart_certificates((certificate,))

    assert result.certified
    assert result.missing_obligations == ()
    assert result.max_coefficient_residual <= certificate.residual_tolerance
    assert result.max_sampled_newton_residual <= certificate.residual_tolerance
    assert "interval_fuchsian_lifted_residual_on_punctured_shells" not in (
        result.missing_obligations
    )
    assert "interval_fuchsian_projected_residual_on_punctured_shells" not in (
        result.missing_obligations
    )
    assert "interval_total_collision_endpoint_collapse_envelope" not in (
        result.missing_obligations
    )
    assert "interval_center_of_mass_and_linear_momentum_on_punctured_shells" not in (
        result.missing_obligations
    )
    assert "interval_fuchsian_supported_scale_finite_energy_matching" not in (
        result.missing_obligations
    )
    assert certificate.primitive_cauchy_inputs is not None
    assert round_trip_result.certified
    assert verifier.certified
    assert verifier.total_collision_fuchsian_stop_chart_count == 1


def test_fuchsian_stop_checker_certification_is_interval_not_sample_gated():
    certificate = replace(_total_collision_fuchsian_stop_chart_certificate(), sample_count=1)
    result = check_total_collision_fuchsian_stop_chart(certificate)
    obligation_ids = {obligation.obligation for obligation in result.obligations}

    assert result.certified
    assert "sampled_fuchsian_lifted_and_projected_newton_residual" not in obligation_ids
    assert "sampled_zero_angular_momentum_consistency" not in obligation_ids
    assert "sampled_total_collision_scaling" not in obligation_ids
    assert "interval_fuchsian_lifted_residual_on_punctured_shells" in obligation_ids
    assert "interval_fuchsian_projected_residual_on_punctured_shells" in obligation_ids
    assert "interval_zero_angular_momentum_on_punctured_shells" in obligation_ids
    assert "interval_total_collision_endpoint_collapse_envelope" in obligation_ids


def test_fast_reduced_order_generalized_fuchsian_stop_checker_ci_fixture():
    raw_certificate = _fast_total_collision_generalized_fuchsian_stop_chart_certificate()
    certificate = replace(
        raw_certificate,
        residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
    )
    result = check_total_collision_generalized_fuchsian_stop_chart(certificate)

    assert _fast_total_collision_generalized_fuchsian_stop_chart_certificate() is raw_certificate
    assert certificate.max_total_degree == 2
    assert certificate.sample_count == 3
    assert result.certified
    assert result.missing_obligations == ()
    assert result.max_coefficient_residual <= certificate.residual_tolerance
    assert result.max_sampled_newton_residual <= certificate.projected_residual_tolerance
    obligations = {obligation.obligation: obligation for obligation in result.obligations}
    assert "diagnostic_direct_interval_projected_residual=" in obligations[
        "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells"
    ].detail
    assert "slack=1-q" in obligations[
        "generalized_fuchsian_remainder_contraction_factor"
    ].detail
    assert "margin=R-(B*D+q*R)" in obligations[
        "generalized_fuchsian_remainder_self_map"
    ].detail


def test_generalized_fuchsian_projected_residual_direct_interval_is_certification_gated():
    certificate = _fast_total_collision_generalized_fuchsian_stop_chart_certificate()
    result = check_total_collision_generalized_fuchsian_stop_chart(certificate)
    obligations = {obligation.obligation: obligation for obligation in result.obligations}
    projected = obligations[
        "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells"
    ]

    assert not result.certified
    assert "interval_generalized_fuchsian_lifted_residual_on_punctured_shells" not in (
        result.missing_obligations
    )
    assert "generalized_fuchsian_remainder_required_residual_components" not in (
        result.missing_obligations
    )
    assert projected.obligation in result.missing_obligations
    assert not projected.certified
    assert "diagnostic_direct_interval_projected_residual=" in projected.detail
    assert result.max_sampled_newton_residual > certificate.residual_tolerance


def test_generalized_fuchsian_projected_budget_does_not_weaken_lifted_gate():
    raw_certificate = _fast_total_collision_generalized_fuchsian_stop_chart_certificate()
    certificate = replace(
        raw_certificate,
        residual_tolerance=1.0e-8,
        projected_residual_tolerance=2.0e4,
    )
    result = check_total_collision_generalized_fuchsian_stop_chart(certificate)

    assert not result.certified
    assert "interval_generalized_fuchsian_lifted_residual_on_punctured_shells" in (
        result.missing_obligations
    )
    assert "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells" not in (
        result.missing_obligations
    )
    assert result.max_sampled_newton_residual <= certificate.projected_residual_tolerance


def test_generalized_fuchsian_remainder_majorant_picard_tail_formula_is_explicit():
    certificate = _fast_total_collision_generalized_fuchsian_stop_chart_certificate()
    majorant = certificate.remainder_majorant
    assert majorant is not None

    q = majorant.linear_inverse_bound * majorant.nonlinear_lipschitz_bound
    first_step = majorant.linear_inverse_bound * majorant.defect_bound

    assert majorant.contraction_factor == pytest.approx(q, rel=0.0, abs=0.0)
    assert majorant.self_map_bound == pytest.approx(
        first_step + q * majorant.remainder_ball_radius,
        rel=0.0,
        abs=0.0,
    )
    assert majorant.contraction_slack == pytest.approx(1.0 - q, rel=0.0, abs=0.0)
    assert majorant.self_map_margin == pytest.approx(
        majorant.remainder_ball_radius - majorant.self_map_bound,
        rel=0.0,
        abs=0.0,
    )
    assert majorant.picard_first_step_bound == pytest.approx(
        first_step,
        rel=0.0,
        abs=0.0,
    )
    for iteration_count in range(5):
        assert majorant.picard_tail_bound(iteration_count) == pytest.approx(
            q**iteration_count * first_step / (1.0 - q),
            rel=1e-15,
            abs=1e-30,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        majorant.picard_tail_bound(-1)


def test_independent_checker_recomputes_generalized_fuchsian_primitive_tail_bounds():
    certificate = _fast_total_collision_generalized_fuchsian_stop_chart_certificate()
    majorant = certificate.remainder_majorant
    assert majorant is not None

    for _component, primitive in majorant.component_inputs:
        expected_first_shell = (
            primitive.majorant_initial
            * primitive.step_ratio_bound ** (primitive.retained_order_initial + 1)
            / (1.0 - primitive.step_ratio_bound)
        )
        expected_shell_ratio = (
            primitive.majorant_growth
            * primitive.step_ratio_bound ** primitive.retained_order_increment
        )
        assert primitive.first_shell_tail_bound == pytest.approx(
            expected_first_shell,
            rel=1e-14,
            abs=1e-30,
        )
        assert primitive.shell_ratio == pytest.approx(
            expected_shell_ratio,
            rel=1e-14,
            abs=1e-30,
        )

    corrupted_inputs = []
    for component, primitive in majorant.component_inputs:
        if component == "physical_residual":
            primitive = replace(
                primitive,
                first_shell_tail_bound=1.0,
            )
        corrupted_inputs.append((component, primitive))
    corrupted = replace(
        certificate,
        remainder_majorant=replace(
            majorant,
            component_inputs=tuple(corrupted_inputs),
        ),
    )
    result = check_total_collision_generalized_fuchsian_stop_chart(corrupted)

    assert not result.certified
    assert "generalized_fuchsian_remainder_component_inputs_certify" in (
        result.missing_obligations
    )
    assert "generalized_fuchsian_remainder_majorant_certifies" in (
        result.missing_obligations
    )


def test_generalized_fuchsian_projected_residual_weight_shift_is_cubic_time_exact():
    tau = Fraction(1, 7)
    coefficient = Fraction(11, 5)

    for lifted_weight in range(5, 12):
        lifted_residual_numerator = coefficient * tau**lifted_weight
        projected_residual = lifted_residual_numerator / (9 * tau**4)
        expected_projected = Fraction(1, 9) * coefficient * tau ** (
            lifted_weight - 4
        )

        assert projected_residual == expected_projected

    for monomial_power in range(2, 9):
        shape_coefficient = Fraction(3, 10)
        lifted_operator_coefficient = (
            monomial_power * (monomial_power - 1)
            + 2 * monomial_power
            - 2
        )
        lifted_numerator = (
            shape_coefficient
            * lifted_operator_coefficient
            * tau**monomial_power
        )
        projected_from_lifted = lifted_numerator / (9 * tau**4)
        projected_by_chain_rule = (
            shape_coefficient
            * lifted_operator_coefficient
            / 9
            * tau ** (monomial_power - 4)
        )

        assert lifted_operator_coefficient == (monomial_power + 2) * (
            monomial_power - 1
        )
        assert projected_from_lifted == projected_by_chain_rule


@pytest.mark.slow
def test_independent_checker_accepts_serialized_generalized_fuchsian_stop_chart():
    certificate = replace(
        _total_collision_generalized_fuchsian_stop_chart_certificate(),
        residual_tolerance=5.0e3,
    )
    result = check_total_collision_generalized_fuchsian_stop_chart(certificate)
    round_trip = TotalCollisionGeneralizedFuchsianStopChartCertificate.from_dict(
        certificate.to_dict(),
    )
    round_trip_result = check_total_collision_generalized_fuchsian_stop_chart(
        round_trip,
    )
    verifier = verify_chart_certificates((certificate,))

    assert result.certified
    assert result.missing_obligations == ()
    assert result.max_coefficient_residual <= certificate.residual_tolerance
    assert result.max_sampled_newton_residual <= certificate.residual_tolerance
    assert "interval_generalized_fuchsian_lifted_residual_on_punctured_shells" not in (
        result.missing_obligations
    )
    assert "generalized_fuchsian_remainder_required_residual_components" not in (
        result.missing_obligations
    )
    assert (
        "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells"
        not in result.missing_obligations
    )
    assert "interval_generalized_zero_angular_momentum_on_punctured_shells" not in (
        result.missing_obligations
    )
    assert (
        "interval_generalized_center_of_mass_and_linear_momentum_on_punctured_shells"
        not in result.missing_obligations
    )
    assert round_trip_result.certified
    assert verifier.certified
    assert verifier.total_collision_generalized_fuchsian_stop_chart_count == 1


@pytest.mark.slow
def test_generalized_fuchsian_stop_checker_certification_is_interval_not_sample_gated():
    certificate = replace(
        _total_collision_generalized_fuchsian_stop_chart_certificate(),
        sample_count=1,
        residual_tolerance=5.0e3,
    )
    result = check_total_collision_generalized_fuchsian_stop_chart(certificate)
    obligation_ids = {obligation.obligation for obligation in result.obligations}

    assert result.certified
    assert "sampled_generalized_fuchsian_lifted_residual_with_tail" not in obligation_ids
    assert "sampled_generalized_fuchsian_zero_angular_momentum" not in obligation_ids
    assert "sampled_total_collision_scaling" not in obligation_ids
    assert "interval_generalized_fuchsian_lifted_residual_on_punctured_shells" in (
        obligation_ids
    )
    assert "generalized_fuchsian_remainder_required_residual_components" in (
        obligation_ids
    )
    assert (
        "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells"
        in obligation_ids
    )
    assert "interval_generalized_zero_angular_momentum_on_punctured_shells" in (
        obligation_ids
    )
    assert (
        "interval_generalized_center_of_mass_and_linear_momentum_on_punctured_shells"
        in obligation_ids
    )
    assert "generalized_fuchsian_endpoint_collapse_envelope" in obligation_ids


@pytest.mark.slow
def test_independent_checker_rejects_corrupted_generalized_fuchsian_majorant():
    certificate = _total_collision_generalized_fuchsian_stop_chart_certificate()
    assert certificate.remainder_majorant is not None
    corrupted = replace(
        certificate,
        remainder_majorant=replace(
            certificate.remainder_majorant,
            remainder_ball_radius=1.0e-20,
        ),
    )
    result = check_total_collision_generalized_fuchsian_stop_chart(corrupted)
    verifier = verify_chart_certificates((corrupted,))

    assert not result.certified
    assert "generalized_fuchsian_remainder_self_map" in (
        result.missing_obligations
    )
    assert "generalized_fuchsian_remainder_majorant_certifies" in (
        result.missing_obligations
    )
    assert not verifier.certified
    assert verifier.total_collision_generalized_fuchsian_stop_chart_count == 1


@pytest.mark.slow
def test_independent_checker_rejects_generalized_fuchsian_without_projected_tail():
    certificate = _total_collision_generalized_fuchsian_stop_chart_certificate()
    assert certificate.remainder_majorant is not None
    missing_physical = replace(
        certificate,
        remainder_majorant=replace(
            certificate.remainder_majorant,
            component_effective_exponents=tuple(
                item
                for item in certificate.remainder_majorant.component_effective_exponents
                if item[0] != "physical_residual"
            ),
            component_inputs=tuple(
                item
                for item in certificate.remainder_majorant.component_inputs
                if item[0] != "physical_residual"
            ),
        ),
    )
    result = check_total_collision_generalized_fuchsian_stop_chart(missing_physical)

    assert not result.certified
    assert "generalized_fuchsian_remainder_required_residual_components" in (
        result.missing_obligations
    )
    assert (
        "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells"
        in result.missing_obligations
    )


def test_independent_checker_rejects_sample_only_fuchsian_stop_chart():
    certificate = _total_collision_fuchsian_stop_chart_certificate(
        include_cauchy_inputs=False,
    )
    result = check_total_collision_fuchsian_stop_chart(certificate)
    verifier = verify_chart_certificates((certificate,))

    assert not result.certified
    assert "fuchsian_primitive_cauchy_inputs_present" in (
        result.missing_obligations
    )
    assert not verifier.certified
    assert verifier.total_collision_fuchsian_stop_chart_count == 1
    assert not verifier.chart_results[0].certified


def test_independent_checker_rejects_noncollapsing_fuchsian_endpoint_envelope():
    certificate = _total_collision_fuchsian_stop_chart_certificate()
    primitive_inputs = certificate.primitive_cauchy_inputs
    assert primitive_inputs is not None
    component_inputs = []
    for component, input_ in primitive_inputs.component_inputs:
        if component == "value":
            growth = 5.0
            input_ = replace(
                input_,
                majorant_growth=growth,
                shell_ratio=growth
                * input_.step_ratio_bound ** input_.retained_order_increment,
            )
        component_inputs.append((component, input_))
    corrupted = replace(
        certificate,
        primitive_cauchy_inputs=replace(
            primitive_inputs,
            component_inputs=tuple(component_inputs),
        ),
    )
    result = check_total_collision_fuchsian_stop_chart(corrupted)

    assert not result.certified
    assert "fuchsian_primitive_cauchy_inputs_certify" not in (
        result.missing_obligations
    )
    assert "interval_total_collision_endpoint_collapse_envelope" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_nonzero_interval_fuchsian_angular_momentum():
    base = _total_collision_fuchsian_stop_chart_certificate()
    central_shape = np.asarray(base.central_shape, dtype=float)
    rotational_row = np.column_stack((-central_shape[:, 1], central_shape[:, 0]))
    term = FuchsianLogTerm(
        power=1.0,
        coefficients_by_log_power={0: 1.0e-4 * rotational_row},
    )
    certificate = _total_collision_fuchsian_stop_chart_certificate(
        terms=(term,),
        residual_tolerance=1.0e9,
    )
    result = check_total_collision_fuchsian_stop_chart(certificate)

    assert not result.certified
    assert "fuchsian_primitive_cauchy_inputs_certify" not in (
        result.missing_obligations
    )
    assert "interval_zero_angular_momentum_on_punctured_shells" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_nonzero_fuchsian_center_of_mass_drift():
    translation_row = np.ones((3, 2), dtype=float)
    term = FuchsianLogTerm(
        power=1.0,
        coefficients_by_log_power={0: 1.0e-4 * translation_row},
    )
    certificate = _total_collision_fuchsian_stop_chart_certificate(terms=(term,))
    result = check_total_collision_fuchsian_stop_chart(certificate)

    assert not result.certified
    assert "fuchsian_primitive_cauchy_inputs_certify" not in (
        result.missing_obligations
    )
    assert "interval_zero_angular_momentum_on_punctured_shells" not in (
        result.missing_obligations
    )
    assert "interval_center_of_mass_and_linear_momentum_on_punctured_shells" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_stale_fuchsian_scale_energy_row():
    base = _total_collision_fuchsian_stop_chart_certificate()
    central_shape = np.asarray(base.central_shape, dtype=float)
    term = FuchsianLogTerm(
        power=2.0,
        coefficients_by_log_power={0: 1.0e-12 * central_shape},
    )
    certificate = _total_collision_fuchsian_stop_chart_certificate(
        terms=(term,),
        residual_tolerance=1.0e9,
    )
    result = check_total_collision_fuchsian_stop_chart(certificate)

    assert not result.certified
    assert "fuchsian_primitive_cauchy_inputs_certify" not in (
        result.missing_obligations
    )
    assert "interval_center_of_mass_and_linear_momentum_on_punctured_shells" not in (
        result.missing_obligations
    )
    assert "interval_zero_angular_momentum_on_punctured_shells" not in (
        result.missing_obligations
    )
    assert "interval_fuchsian_supported_scale_finite_energy_matching" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_corrupted_total_collision_central_shape():
    certificate = _total_collision_fuchsian_stop_chart_certificate()
    central_shape = [list(row) for row in certificate.central_shape]
    central_shape[1] = list(central_shape[0])
    corrupted = replace(
        certificate,
        central_shape=tuple(tuple(row) for row in central_shape),
    )
    result = check_total_collision_fuchsian_stop_chart(corrupted)

    assert not result.certified
    assert "central_shape_collision_free" in result.missing_obligations
    assert "punctured_total_collision_isolation_data" in result.missing_obligations


def test_independent_checker_rejects_nonpunctured_total_collision_interval():
    certificate = _total_collision_fuchsian_stop_chart_certificate()
    shifted = replace(
        certificate,
        tau_interval=(0.001, 0.025),
        physical_time_interval=(1.0e-9, 0.025**3),
    )
    result = check_total_collision_fuchsian_stop_chart(shifted)

    assert not result.certified
    assert "punctured_tau_interval_straddles_total_collision" in (
        result.missing_obligations
    )
    assert "interval_fuchsian_lifted_residual_on_punctured_shells" in (
        result.missing_obligations
    )
    assert "interval_fuchsian_projected_residual_on_punctured_shells" in (
        result.missing_obligations
    )


def test_independent_checker_rejects_corrupted_fuchsian_stop_residual_sample():
    certificate = _total_collision_fuchsian_stop_chart_certificate(
        scale_coefficient=0.5,
    )
    result = check_total_collision_fuchsian_stop_chart(certificate)

    assert not result.certified
    assert "punctured_total_collision_isolation_data" not in (
        result.missing_obligations
    )
    assert "interval_fuchsian_projected_residual_on_punctured_shells" in (
        result.missing_obligations
    )


def test_independent_checker_accepts_ordinary_to_ks_transition():
    charts, transitions = _ordinary_to_ks_transition_bundle()
    result = check_spatial_ks_transition(transitions[0], charts)
    verifier = verify_chart_certificates(charts, transitions)
    details = {obligation.obligation: obligation for obligation in result.obligations}

    assert result.certified
    assert result.max_position_gap <= transitions[0].position_tolerance
    assert result.max_velocity_gap <= transitions[0].velocity_tolerance
    assert details["transition_exact_rational_state_continuity"].certified
    assert verifier.certified
    assert verifier.ordinary_taylor_chart_count == 1
    assert verifier.spatial_ks_binary_chart_count == 1
    assert len(verifier.transition_results) == 1
    assert "transition_exact_rational_state_continuity" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )


def test_independent_checker_accepts_ks_to_ordinary_transition():
    charts, transitions = _ks_to_ordinary_transition_bundle()
    result = check_spatial_ks_transition(transitions[0], charts)
    verifier = verify_chart_certificates(charts, transitions)
    details = {obligation.obligation: obligation for obligation in result.obligations}

    assert result.certified
    assert result.max_position_gap <= transitions[0].position_tolerance
    assert result.max_velocity_gap <= transitions[0].velocity_tolerance
    assert details["transition_exact_rational_state_continuity"].certified
    assert verifier.certified
    assert verifier.ordinary_taylor_chart_count == 1
    assert verifier.spatial_ks_binary_chart_count == 1
    assert len(verifier.transition_results) == 1
    assert "transition_exact_rational_state_continuity" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )


def test_independent_checker_rejects_corrupted_ks_to_ordinary_transition():
    charts, transitions = _ks_to_ordinary_transition_bundle()
    corrupted = replace(transitions[0], target_parameter=0.003)
    result = check_spatial_ks_transition(corrupted, charts)
    verifier = verify_chart_certificates(charts, (corrupted,))

    assert not result.certified
    assert "transition_physical_time_match" in result.missing_obligations
    assert "transition_state_continuity" in result.missing_obligations
    assert not verifier.certified


def test_independent_checker_accepts_serialized_ordinary_ks_entry_event():
    certificate = _ordinary_ks_entry_event_certificate()
    result = check_event_isolation(certificate)
    round_trip = EventIsolationCertificate.from_dict(certificate.to_dict())
    round_trip_result = check_event_isolation(round_trip)
    verifier = verify_chart_certificates(
        (_ordinary_chart_certificate(),),
        events=(certificate,),
    )

    assert result.certified
    assert result.missing_obligations == ()
    assert result.checker_id == "independent_polynomial_event_isolation_checker_v2"
    details = {obligation.obligation: obligation for obligation in result.obligations}
    assert details["event_exact_rational_interval_arithmetic"].certified
    assert "exact rational interval arithmetic" in (
        details["event_exact_rational_interval_arithmetic"].detail
    )
    assert round_trip_result.certified
    assert verifier.certified
    assert verifier.checked_event_count == 1


def test_verifier_arithmetic_audit_accepts_exact_rational_event_bundle():
    verifier = verify_chart_certificates(
        (_ordinary_chart_certificate(),),
        events=(_ordinary_ks_entry_event_certificate(),),
    )

    assert verifier.certified
    assert verifier.proof_grade_arithmetic_checked_bundle_certified
    assert "event_exact_rational_interval_arithmetic" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )
    assert verifier.proof_grade_arithmetic_blockers == ()


def test_independent_checker_accepts_serialized_ks_rho_exit_event():
    certificate = _ks_rho_exit_event_certificate()
    result = check_event_isolation(certificate)

    assert result.certified
    assert result.missing_obligations == ()
    assert result.root_interval_width > 0.0


def test_independent_checker_rejects_corrupted_event_isolation_signs():
    certificate = _ordinary_ks_entry_event_certificate()
    corrupted_coefficients = list(certificate.coefficient_intervals)
    lower, upper = corrupted_coefficients[0]
    corrupted_coefficients[0] = (lower - 1.0, upper - 1.0)
    corrupted = replace(
        certificate,
        coefficient_intervals=tuple(corrupted_coefficients),
    )
    result = check_event_isolation(corrupted)

    assert not result.certified
    assert "event_root_endpoint_signs" in result.missing_obligations
    assert "event_excludes_earlier_roots" in result.missing_obligations
    assert "event_exact_rational_interval_arithmetic" not in (
        result.missing_obligations
    )


def test_independent_checker_accepts_ordinary_to_lc_transition():
    charts, transitions = _ordinary_to_lc_transition_bundle()
    result = check_planar_levi_civita_transition(transitions[0], charts)
    verifier = verify_chart_certificates(charts, transitions)
    details = {obligation.obligation: obligation for obligation in result.obligations}

    assert result.certified
    assert result.max_position_gap <= transitions[0].position_tolerance
    assert result.max_velocity_gap <= transitions[0].velocity_tolerance
    assert details["transition_exact_rational_state_continuity"].certified
    assert verifier.certified
    assert verifier.ordinary_taylor_chart_count == 1
    assert verifier.planar_levi_civita_binary_chart_count == 1
    assert len(verifier.transition_results) == 1
    assert "transition_exact_rational_state_continuity" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )


def test_independent_checker_accepts_lc_to_ordinary_transition():
    charts, transitions = _lc_to_ordinary_transition_bundle()
    result = check_planar_levi_civita_transition(transitions[0], charts)
    verifier = verify_chart_certificates(charts, transitions)
    details = {obligation.obligation: obligation for obligation in result.obligations}

    assert result.certified
    assert result.max_position_gap <= transitions[0].position_tolerance
    assert result.max_velocity_gap <= transitions[0].velocity_tolerance
    assert details["transition_exact_rational_state_continuity"].certified
    assert verifier.certified
    assert verifier.ordinary_taylor_chart_count == 1
    assert verifier.planar_levi_civita_binary_chart_count == 1
    assert len(verifier.transition_results) == 1
    assert "transition_exact_rational_state_continuity" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )


def test_planar_hybrid_solution_serializes_to_independently_checked_chain():
    solution = _planar_lc_binary_solution()
    exit_parameter = 0.006
    handoff_time = solution.physical_time_at(exit_parameter)
    start_positions, start_velocities = regularized_binary_collision_chart_to_planar(
        solution.state_at(0.0),
    )
    end_positions, end_velocities = regularized_binary_collision_chart_to_planar(
        solution.state_at(exit_parameter),
    )
    start_state = np.concatenate(
        [start_positions.reshape(-1), start_velocities.reshape(-1)],
    )
    handoff_state = np.concatenate(
        [end_positions.reshape(-1), end_velocities.reshape(-1)],
    )
    hybrid_solution = SimpleNamespace(
        masses=solution.masses,
        states=np.vstack([start_state, handoff_state]),
        steps=(
            SimpleNamespace(
                chart="binary",
                pair=solution.pair,
                start_time=0.0,
                physical_step=handoff_time,
                parameter_step=exit_parameter,
                end_time=handoff_time,
                truncation_certificate=SimpleNamespace(
                    computed_order=solution.order,
                    retained_order=solution.order,
                    tail_bound=1.0e-9,
                ),
            ),
            SimpleNamespace(
                chart="ordinary",
                start_time=handoff_time,
                physical_step=0.01,
                parameter_step=0.01,
                end_time=handoff_time + 0.01,
                truncation_certificate=SimpleNamespace(
                    computed_order=10,
                    retained_order=10,
                    tail_bound=1.0e-9,
                ),
            ),
        ),
    )

    charts, transitions, chain = planar_hybrid_chart_chain_certificates_from_solution(
        hybrid_solution,
    )
    verifier = verify_chart_certificates(
        charts,
        transitions=transitions,
        chart_chains=(chain,),
    )

    assert verifier.certified
    assert verifier.planar_levi_civita_binary_chart_count == 1
    assert verifier.ordinary_taylor_chart_count == 1
    assert len(verifier.transition_results) == 1
    assert verifier.checked_chart_chain_count == 1
    assert [chart.chart_type for chart in charts] == [
        "planar_levi_civita_binary",
        "ordinary_taylor",
    ]
    assert transitions[0].transition_type == "binary_to_ordinary_event_handoff"

    atlas_bridge = construct_independent_validated_atlas_checked_chain(
        SimpleNamespace(evaluation=SimpleNamespace(hybrid_solution=hybrid_solution)),
    )
    finite_target_bridge = construct_independent_finite_target_checked_atlas(
        SimpleNamespace(
            validated_atlas=SimpleNamespace(
                proof_certified=True,
                evaluation=SimpleNamespace(hybrid_solution=hybrid_solution),
            ),
        ),
    )

    assert atlas_bridge.certified
    assert finite_target_bridge.certified
    assert finite_target_bridge.checked_chart_chain_count == 1


def test_independent_checker_rejects_corrupted_lc_to_ordinary_transition():
    charts, transitions = _lc_to_ordinary_transition_bundle()
    corrupted = replace(transitions[0], target_parameter=0.003)
    result = check_planar_levi_civita_transition(corrupted, charts)
    verifier = verify_chart_certificates(charts, (corrupted,))

    assert not result.certified
    assert "transition_physical_time_match" in result.missing_obligations
    assert "transition_state_continuity" in result.missing_obligations
    assert not verifier.certified


def test_verifier_certificate_attaches_to_open_time_audit_without_closing_proof():
    charts, transitions = _ordinary_chart_chain()
    verifier = verify_chart_certificates(charts, transitions)
    masses, positions, velocities = _open_time_spatial_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        initial_radius=1.0e-15,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.025,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=42,
    )
    base = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=theorem,
    )
    chart_chain = ChartChainCertificate(
        certificate_id="ordinary-chain-for-open-time-audit",
        chain_id="ordinary-chain-for-open-time-audit",
        chain_type="regularized_atlas_chart_chain",
        chart_ids=tuple(chart.chart_id for chart in charts),
        transition_ids=tuple(transition.transition_id for transition in transitions),
        target_physical_time_interval=(
            charts[0].physical_time_interval[0],
            charts[-1].physical_time_interval[1],
        ),
    )
    finite_atlas_verifier = verify_chart_certificates(
        charts,
        transitions,
        chart_chains=(chart_chain,),
    )
    chart_only_attached = attach_independent_chart_verifier(theorem, verifier)
    chart_only_checked = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=chart_only_attached,
    )
    finite_atlas_attached = attach_independent_chart_verifier(
        theorem,
        finite_atlas_verifier,
    )
    finite_atlas_checked = certify_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=finite_atlas_attached,
    )

    assert verifier.certified
    assert verifier.ordinary_taylor_chart_count == 2
    assert len(verifier.transition_results) == 1
    assert verifier.proof_grade_arithmetic_checked_bundle_certified
    assert not verifier.proof_grade_finite_atlas_bundle_certified
    assert verifier.proof_grade_arithmetic_blockers == ()
    assert finite_atlas_verifier.certified
    assert finite_atlas_verifier.proof_grade_finite_atlas_bundle_certified
    assert "transition_exact_rational_state_continuity" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )
    assert "ordinary_taylor_exact_rational_residual_polynomials" in (
        verifier.proof_grade_arithmetic_obligation_ids
    )
    assert "independent_chart_verifier" in base.blocking_obligations
    assert "independent_chart_verifier" in chart_only_checked.blocking_obligations
    assert "independent_chart_verifier" not in (
        finite_atlas_checked.blocking_obligations
    )
    assert "audited_or_machine_checked_open_time_atlas_proof" in (
        finite_atlas_checked.blocking_obligations
    )
    assert chart_only_checked.status == "incomplete"
    assert not chart_only_checked.proof_certified
    assert finite_atlas_checked.status == "incomplete"
    assert not finite_atlas_checked.proof_certified


def test_attach_independent_chart_verifier_rejects_subclassed_verifier():
    class SpoofedIndependentChartVerifierCertificate(
        IndependentChartVerifierCertificate
    ):
        @property
        def certified(self):
            return True

        @property
        def proof_grade_finite_atlas_bundle_certified(self):
            return True

    charts, transitions = _ordinary_chart_chain()
    real_verifier = verify_chart_certificates(charts, transitions)
    spoofed_verifier = SpoofedIndependentChartVerifierCertificate(
        checker_id=real_verifier.checker_id,
        chart_results=real_verifier.chart_results,
        transition_results=real_verifier.transition_results,
        event_results=real_verifier.event_results,
        branch_union_results=real_verifier.branch_union_results,
        chart_chain_results=real_verifier.chart_chain_results,
    )
    masses, positions, velocities = _open_time_spatial_initial_data()
    theorem = construct_open_time_locally_finite_atlas_theorem(
        masses,
        positions,
        velocities,
        1.0e-4,
        compact_time_rate=1.3,
        exhaustion_prefix_count=1,
        initial_radius=1.0e-15,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.025,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=42,
    )

    assert isinstance(spoofed_verifier, IndependentChartVerifierCertificate)
    assert type(spoofed_verifier) is not IndependentChartVerifierCertificate
    assert spoofed_verifier.certified
    with pytest.raises(TypeError, match="constructor-derived"):
        attach_independent_chart_verifier(theorem, spoofed_verifier)
