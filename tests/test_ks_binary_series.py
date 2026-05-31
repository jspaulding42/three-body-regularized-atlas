from dataclasses import replace

import numpy as np
import pytest

import three_body_symmetry.validated_atlas as validated_atlas_module
from three_body_symmetry.dynamics import (
    accelerations,
    angular_momentum_components,
    energy,
    linear_momentum,
)
from three_body_symmetry.ks_binary_chart import (
    SpatialKSBinaryChartState,
    ks_binary_chart_to_spatial,
    ks_pair_energy_constraint,
    ks_vertical_generator,
    regularized_ks_binary_chart_rhs,
    spatial_accelerations_from_ks_binary_rhs,
    spatial_to_ks_binary_chart,
)
from three_body_symmetry.ks_binary_series import (
    certify_spatial_ks_binary_center_of_mass_motion,
    certify_spatial_ks_binary_centered_angular_momentum_conservation,
    certify_spatial_ks_binary_horizontal_constraint,
    certify_spatial_ks_binary_interval_taylor_equations,
    certify_spatial_ks_binary_linear_momentum_conservation,
    certify_spatial_ks_binary_pair_energy_constraint,
    certify_spatial_ks_binary_rho_exit_event,
    certify_spatial_ks_binary_total_energy_conservation,
    certify_spatial_ks_competing_binary_entry_event,
    certify_spatial_ordinary_ks_entry_event,
    construct_interval_spatial_ks_binary_taylor_solution,
    construct_interval_spatial_ks_binary_taylor_solution_from_intervals,
    construct_spatial_ks_binary_taylor_solution,
    integrate_spatial_ks_binary_reference,
    interval_spatial_ks_binary_chart_state_from_point,
    ks_horizontal_constraint_coefficients,
    ks_pair_energy_constraint_coefficients,
    project_spatial_ks_binary_interval_chart_state_to_physical,
    project_spatial_ks_binary_taylor_endpoint_to_physical,
    regularized_rhs_coefficients,
    spatial_interval_to_ks_binary_chart_state,
    spatial_interval_to_ks_binary_chart_state_atlas,
    spatial_ks_competing_entry_event_to_ks_chart_state,
    spatial_ks_competing_entry_event_to_ks_chart_state_atlas,
    spatial_ordinary_entry_event_to_ks_chart_state,
    spatial_ordinary_entry_event_to_ks_chart_state_atlas,
)
from three_body_symmetry.intervals import FloatInterval, interval_array_series_eval
from three_body_symmetry.series import (
    construct_interval_taylor_solution,
    construct_taylor_solution,
    integrate_reference,
)
from three_body_symmetry.tail_bounds import (
    spatial_ks_binary_interval_segmented_tail_certificate,
    spatial_ks_binary_interval_tail_certificate,
    spatial_ks_binary_tail_certificate,
)
from three_body_symmetry.validated_atlas import (
    certify_next_finite_time_event_set,
    partition_ks_state_by_competing_event_order,
    certify_spatial_ks_to_ordinary_handoff_admissibility,
    validated_atlas_from_spatial_ks_competing_binary_handoff,
    validated_atlas_from_spatial_ks_competing_binary_handoff_to_atlas,
    validated_atlas_from_spatial_ks_event_order_partition,
    validated_atlas_from_spatial_ordinary_ks_handoff,
    validated_atlas_from_spatial_ks_binary_chart,
)


def _spatial_state():
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
    return masses, positions, velocities


def _exact_collision_state():
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


def _state_vector(positions, velocities):
    return np.concatenate([positions.reshape(-1), velocities.reshape(-1)])


def _centered_state_vector(positions, velocities, masses):
    total_mass = float(np.sum(masses))
    center = np.sum(masses[:, None] * positions, axis=0) / total_mass
    center_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    return _state_vector(positions - center, velocities - center_velocity)


def _interval_state_box(positions, velocities, half_width=1e-5):
    state = _state_vector(positions, velocities)
    return tuple(
        (float(value - half_width), float(value + half_width))
        for value in state
    )


def test_spatial_ks_binary_taylor_coefficients_satisfy_rhs_recurrence():
    masses, positions, velocities = _spatial_state()
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    solution = construct_spatial_ks_binary_taylor_solution(initial, order=10)

    for n in range(9):
        rhs = regularized_rhs_coefficients(solution, n)
        assert np.linalg.norm((n + 1) * solution.u[n + 1] - rhs.u[n], ord=np.inf) < 1e-13
        assert np.linalg.norm((n + 1) * solution.u_velocity[n + 1] - rhs.u_velocity[n], ord=np.inf) < 1e-13
        assert abs((n + 1) * solution.pair_energy[n + 1] - rhs.pair_energy[n]) < 1e-13
        assert np.linalg.norm((n + 1) * solution.binary_center[n + 1] - rhs.binary_center[n], ord=np.inf) < 1e-13
        assert (
            np.linalg.norm(
                (n + 1) * solution.binary_center_velocity[n + 1] - rhs.binary_center_velocity[n],
                ord=np.inf,
            )
            < 1e-13
        )
        assert np.linalg.norm((n + 1) * solution.third_offset[n + 1] - rhs.third_offset[n], ord=np.inf) < 1e-13
        assert (
            np.linalg.norm(
                (n + 1) * solution.third_offset_velocity[n + 1] - rhs.third_offset_velocity[n],
                ord=np.inf,
            )
            < 1e-13
        )
        assert abs((n + 1) * solution.physical_time[n + 1] - rhs.physical_time[n]) < 1e-13


def test_spatial_ks_binary_taylor_matches_lifted_reference_integration():
    masses, positions, velocities = _spatial_state()
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    solution = construct_spatial_ks_binary_taylor_solution(initial, order=16)
    s_value = 0.02
    reference = integrate_spatial_ks_binary_reference(initial, s_value)

    assert np.linalg.norm(solution.vector_at(s_value) - reference, ord=np.inf) < 1e-12


def test_spatial_ks_binary_taylor_projection_matches_newtonian_motion():
    masses, positions, velocities = _spatial_state()
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    solution = construct_spatial_ks_binary_taylor_solution(initial, order=18)
    s_value = 0.015
    projected_state = solution.state_at(s_value)
    projected_positions, projected_velocities = ks_binary_chart_to_spatial(projected_state)
    physical_time = solution.physical_time_at(s_value)
    reference = integrate_reference(positions, velocities, masses, physical_time)
    reference_positions = reference[:9].reshape(3, 3)
    reference_velocities = reference[9:].reshape(3, 3)

    assert physical_time > 0.0
    assert np.linalg.norm(projected_positions - reference_positions, ord=np.inf) < 1e-12
    assert np.linalg.norm(projected_velocities - reference_velocities, ord=np.inf) < 1e-12


def test_spatial_ks_binary_taylor_starts_at_exact_binary_collision():
    initial = _exact_collision_state()
    solution = construct_spatial_ks_binary_taylor_solution(initial, order=14)
    s_value = 0.02
    reference = integrate_spatial_ks_binary_reference(initial, s_value)
    constraint = ks_pair_energy_constraint_coefficients(solution, 14)

    assert np.all(np.isfinite(solution.u))
    assert np.all(np.isfinite(solution.u_velocity))
    assert np.all(np.isfinite(solution.pair_energy))
    assert solution.physical_time[1] == 0.0
    assert solution.physical_time[2] == 0.0
    assert abs(solution.physical_time[3] - np.dot(initial.u_velocity, initial.u_velocity) / 3.0) < 1e-14
    assert np.linalg.norm(constraint, ord=np.inf) < 1e-12
    assert np.linalg.norm(solution.vector_at(s_value) - reference, ord=np.inf) < 1e-11


def test_spatial_ks_collision_taylor_projects_to_two_sided_newtonian_branches():
    initial = _exact_collision_state()
    solution = construct_spatial_ks_binary_taylor_solution(initial, order=24)
    constraint = ks_pair_energy_constraint_coefficients(solution, 24)

    assert np.linalg.norm(solution.state_at(0.0).u, ord=np.inf) == 0.0
    assert solution.physical_time_at(0.0) == 0.0
    assert solution.physical_time[3] == (
        np.dot(initial.u_velocity, initial.u_velocity) / 3.0
    )
    assert np.linalg.norm(constraint, ord=np.inf) < 1e-12

    for s_value in (-0.03, -0.02, 0.02, 0.03):
        state = solution.state_at(s_value)
        physical_time = solution.physical_time_at(s_value)
        positions, _velocities = ks_binary_chart_to_spatial(state)
        rhs = regularized_ks_binary_chart_rhs(state)
        projected_acceleration = spatial_accelerations_from_ks_binary_rhs(state, rhs)

        assert np.sign(physical_time) == np.sign(s_value)
        assert state.rho > 0.0
        assert abs(np.linalg.norm(positions[1] - positions[0]) - state.rho) < 1e-18
        assert np.linalg.norm(positions[2] - positions[0]) > 1.5
        assert np.linalg.norm(positions[2] - positions[1]) > 1.5
        assert np.linalg.norm(
            projected_acceleration - accelerations(positions, initial.masses),
            ord=np.inf,
        ) < 2e-8
        assert abs(ks_pair_energy_constraint(state)) < 2e-10


def test_interval_spatial_ks_binary_taylor_contains_point_coefficients_and_values():
    masses, positions, velocities = _spatial_state()
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    point = construct_spatial_ks_binary_taylor_solution(initial, order=10)
    interval_state = interval_spatial_ks_binary_chart_state_from_point(initial)
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=10)
    s_value = 0.012

    assert interval_state.contains_point_state(initial)
    assert interval.contains_point_solution(point)
    assert interval.vector_contains(point.vector_at(s_value), s_value)


def test_interval_spatial_ks_binary_taylor_residual_certificate_closes_recurrence():
    masses, positions, velocities = _spatial_state()
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=10)
    certificate = certify_spatial_ks_binary_interval_taylor_equations(interval)

    assert certificate.certified
    assert certificate.coefficient_count == 10
    assert certificate.pair == (0, 1)
    assert certificate.max_residual_radius < 1e-10


def test_interval_spatial_ks_binary_taylor_handles_exact_collision_start():
    initial = _exact_collision_state()
    point = construct_spatial_ks_binary_taylor_solution(initial, order=10)
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=10)
    certificate = certify_spatial_ks_binary_interval_taylor_equations(interval)

    assert interval.contains_point_solution(point)
    assert interval.vector_contains(point.vector_at(0.02), 0.02)
    assert certificate.certified
    assert certificate.max_residual_radius < 1e-11


def test_spatial_ks_binary_tail_certificate_uses_interval_guard_terms():
    masses, positions, velocities = _spatial_state()
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    certificate = spatial_ks_binary_tail_certificate(
        initial,
        retained_order=8,
        guard_order=8,
        step_size=0.01,
    )

    assert certificate.uses_interval_coefficients
    assert certificate.is_nontrivial
    assert certificate.tail_bound >= certificate.observed_tail >= 0.0
    assert certificate.ratio_bound < 0.02


def test_interval_spatial_ks_binary_invariant_certificates_match_physical_initial_state():
    masses, positions, velocities = _spatial_state()
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=10)
    center = certify_spatial_ks_binary_center_of_mass_motion(interval)
    momentum = certify_spatial_ks_binary_linear_momentum_conservation(interval)
    angular = certify_spatial_ks_binary_centered_angular_momentum_conservation(interval)
    total_energy = certify_spatial_ks_binary_total_energy_conservation(interval)
    projected_positions, projected_velocities = ks_binary_chart_to_spatial(initial)
    state = _state_vector(projected_positions, projected_velocities)
    centered_state = _centered_state_vector(projected_positions, projected_velocities, masses)
    expected_momentum = linear_momentum(state, masses)
    expected_angular = angular_momentum_components(centered_state, masses)
    expected_energy = energy(state, masses)

    assert center.certified
    assert momentum.certified
    assert angular.certified
    assert total_energy.certified
    assert center.max_residual_radius < 1e-11
    assert momentum.max_nonconstant_radius < 1e-10
    assert angular.max_nonconstant_radius < 1e-10
    assert total_energy.max_nonconstant_radius < 1e-10
    assert all(
        coefficient.lower <= value <= coefficient.upper
        for coefficient, value in zip(momentum.linear_momentum_coefficients[0], expected_momentum)
    )
    assert all(
        coefficient.lower <= value <= coefficient.upper
        for coefficient, value in zip(angular.angular_momentum_coefficients[0], expected_angular)
    )
    assert total_energy.energy_coefficients[0].lower <= expected_energy <= total_energy.energy_coefficients[0].upper


def test_interval_spatial_ks_binary_invariant_certificates_handle_exact_collision_start():
    initial = _exact_collision_state()
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=10)
    center = certify_spatial_ks_binary_center_of_mass_motion(interval)
    momentum = certify_spatial_ks_binary_linear_momentum_conservation(interval)
    angular = certify_spatial_ks_binary_centered_angular_momentum_conservation(interval)
    total_energy = certify_spatial_ks_binary_total_energy_conservation(interval)

    assert center.certified
    assert momentum.certified
    assert angular.certified
    assert total_energy.certified
    assert center.max_residual_radius < 1e-12
    assert momentum.max_nonconstant_radius < 1e-12
    assert angular.max_nonconstant_radius < 1e-12
    assert total_energy.max_nonconstant_radius < 1e-12


def test_interval_spatial_ks_binary_constraint_certificates_close_projection_prerequisites():
    masses, positions, velocities = _spatial_state()
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    point = construct_spatial_ks_binary_taylor_solution(initial, order=12)
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=12)
    horizontal_point = ks_horizontal_constraint_coefficients(point, 12)
    pair_point = ks_pair_energy_constraint_coefficients(point, 12)
    horizontal = certify_spatial_ks_binary_horizontal_constraint(interval)
    pair_energy = certify_spatial_ks_binary_pair_energy_constraint(interval)

    assert np.linalg.norm(horizontal_point, ord=np.inf) < 1e-12
    assert np.linalg.norm(pair_point, ord=np.inf) < 1e-12
    assert horizontal.certified
    assert pair_energy.certified
    assert horizontal.constraint_id == "ks_horizontal_constraint"
    assert pair_energy.constraint_id == "ks_pair_energy_constraint"
    assert horizontal.max_radius < 1e-10
    assert pair_energy.max_radius < 1e-10


def test_interval_spatial_ks_binary_constraint_certificates_handle_exact_collision_start():
    initial = _exact_collision_state()
    point = construct_spatial_ks_binary_taylor_solution(initial, order=12)
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=12)
    horizontal = certify_spatial_ks_binary_horizontal_constraint(interval)
    pair_energy = certify_spatial_ks_binary_pair_energy_constraint(interval)

    assert np.linalg.norm(ks_horizontal_constraint_coefficients(point, 12), ord=np.inf) < 1e-12
    assert np.linalg.norm(ks_pair_energy_constraint_coefficients(point, 12), ord=np.inf) < 1e-12
    assert horizontal.certified
    assert pair_energy.certified
    assert horizontal.max_radius < 1e-12
    assert pair_energy.max_radius < 1e-12


def test_interval_spatial_ks_binary_constraint_certificates_reject_bad_lifts():
    masses, positions, velocities = _spatial_state()
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    bad_pair_energy = replace(initial, pair_energy=initial.pair_energy + 0.1)
    bad_horizontal = replace(
        initial,
        u_velocity=initial.u_velocity + 0.1 * ks_vertical_generator(initial.u),
    )

    pair_interval = construct_interval_spatial_ks_binary_taylor_solution(
        bad_pair_energy,
        order=6,
    )
    horizontal_interval = construct_interval_spatial_ks_binary_taylor_solution(
        bad_horizontal,
        order=6,
    )

    assert not certify_spatial_ks_binary_pair_energy_constraint(pair_interval).certified
    assert not certify_spatial_ks_binary_horizontal_constraint(horizontal_interval).certified


def test_spatial_interval_box_lifts_into_certified_ks_binary_branch():
    masses, positions, velocities = _spatial_state()
    state_interval = _interval_state_box(positions, velocities, half_width=1e-5)
    chart = spatial_interval_to_ks_binary_chart_state(
        state_interval,
        masses,
        pair=(0, 1),
        branch="positive_x",
    )
    point = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))

    assert chart.certified
    assert chart.branch_certificate.branch == "positive_x"
    assert chart.contains_physical_point(positions, velocities)
    assert chart.contains_point_state(point)


def test_spatial_interval_box_ks_atlas_covers_overlapping_branches():
    masses, positions, velocities = _spatial_state()
    state_interval = _interval_state_box(positions, velocities, half_width=1e-5)
    atlas = spatial_interval_to_ks_binary_chart_state_atlas(
        state_interval,
        masses,
        pair=(0, 1),
    )

    assert {chart.branch_certificate.branch for chart in atlas} == {
        "positive_x",
        "negative_x",
    }
    assert all(chart.certified for chart in atlas)
    assert all(chart.contains_physical_point(positions, velocities) for chart in atlas)


def test_spatial_interval_box_ks_taylor_closes_residuals_constraints_invariants_and_tail():
    masses, positions, velocities = _spatial_state()
    state_interval = _interval_state_box(positions, velocities, half_width=1e-6)
    chart = spatial_interval_to_ks_binary_chart_state(
        state_interval,
        masses,
        pair=(0, 1),
        branch="positive_x",
    )
    solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        chart,
        order=10,
    )
    residual = certify_spatial_ks_binary_interval_taylor_equations(solution)
    horizontal = certify_spatial_ks_binary_horizontal_constraint(solution)
    pair_energy = certify_spatial_ks_binary_pair_energy_constraint(solution)
    center = certify_spatial_ks_binary_center_of_mass_motion(solution)
    momentum = certify_spatial_ks_binary_linear_momentum_conservation(solution)
    angular = certify_spatial_ks_binary_centered_angular_momentum_conservation(solution)
    total_energy = certify_spatial_ks_binary_total_energy_conservation(solution)
    tail = spatial_ks_binary_interval_tail_certificate(
        chart,
        retained_order=8,
        guard_order=8,
        step_size=0.005,
    )

    assert residual.certified
    assert horizontal.certified
    assert pair_energy.certified
    assert center.certified
    assert momentum.certified
    assert angular.certified
    assert total_energy.certified
    assert tail.is_nontrivial
    assert tail.uses_interval_coefficients


def test_spatial_interval_box_ks_entry_projects_back_to_physical_state_box():
    masses, positions, velocities = _spatial_state()
    state_interval = _interval_state_box(positions, velocities, half_width=1e-6)
    chart = spatial_interval_to_ks_binary_chart_state(
        state_interval,
        masses,
        pair=(0, 1),
        branch="positive_x",
    )
    projection = project_spatial_ks_binary_interval_chart_state_to_physical(chart)

    assert projection.certified
    assert projection.rho.lower > 0.0
    assert projection.projection_domain == "rho_positive_interval"
    assert len(projection.state_interval) == 18
    assert projection.contains_physical_point(positions, velocities)


def test_spatial_interval_box_ks_endpoint_projects_to_ordinary_handoff_box():
    masses, positions, velocities = _spatial_state()
    state_interval = _interval_state_box(positions, velocities, half_width=1e-6)
    chart = spatial_interval_to_ks_binary_chart_state(
        state_interval,
        masses,
        pair=(0, 1),
        branch="positive_x",
    )
    interval = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        chart,
        order=16,
    )
    point_initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    point_solution = construct_spatial_ks_binary_taylor_solution(point_initial, order=16)
    s_value = 0.004
    point_endpoint = point_solution.state_at(s_value)
    point_positions, point_velocities = ks_binary_chart_to_spatial(point_endpoint)
    projection = project_spatial_ks_binary_taylor_endpoint_to_physical(interval, s_value)

    assert projection.certified
    assert projection.physical_time.lower <= point_solution.physical_time_at(s_value) <= projection.physical_time.upper
    assert projection.contains_physical_point(point_positions, point_velocities)


def test_spatial_ks_endpoint_projection_contains_newtonian_reference_handoff():
    masses, positions, velocities = _spatial_state()
    state_interval = _interval_state_box(positions, velocities, half_width=1e-5)
    chart = spatial_interval_to_ks_binary_chart_state(
        state_interval,
        masses,
        pair=(0, 1),
        branch="positive_x",
    )
    interval = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        chart,
        order=18,
    )
    point_initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    point_solution = construct_spatial_ks_binary_taylor_solution(point_initial, order=18)
    s_value = 0.005
    physical_time = point_solution.physical_time_at(s_value)
    reference = integrate_reference(positions, velocities, masses, physical_time)
    reference_positions = reference[:9].reshape(3, 3)
    reference_velocities = reference[9:].reshape(3, 3)
    projection = project_spatial_ks_binary_taylor_endpoint_to_physical(interval, s_value)

    assert projection.certified
    assert projection.contains_physical_point(reference_positions, reference_velocities)


def test_exact_collision_ks_projection_certifies_only_after_rho_positive_exit():
    initial = _exact_collision_state()
    point = construct_spatial_ks_binary_taylor_solution(initial, order=16)
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=16)
    collision_projection = project_spatial_ks_binary_taylor_endpoint_to_physical(
        interval,
        0.0,
    )
    exit_projection = project_spatial_ks_binary_taylor_endpoint_to_physical(
        interval,
        0.02,
    )
    point_positions, point_velocities = ks_binary_chart_to_spatial(point.state_at(0.02))

    assert not collision_projection.certified
    assert collision_projection.missing_obligations == ("rho_positive_interval",)
    assert collision_projection.rho.lower <= 0.0 <= collision_projection.rho.upper
    assert exit_projection.certified
    assert exit_projection.rho.lower > 0.0
    assert exit_projection.contains_physical_point(point_positions, point_velocities)


def test_spatial_ks_rho_exit_event_isolates_exact_collision_exit():
    initial = _exact_collision_state()
    point = construct_spatial_ks_binary_taylor_solution(initial, order=24)
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=24)
    exit_rho = 1.0e-4
    certificate = certify_spatial_ks_binary_rho_exit_event(
        interval,
        exit_rho=exit_rho,
        s_upper=0.02,
        coefficient_count=20,
    )

    assert certificate.certified
    assert certificate.interval_is_isolated
    assert certificate.excludes_earlier_roots
    assert certificate.coefficient_source == "spatial_ks_binary_interval_taylor"
    assert certificate.root_interval[0] <= certificate.root <= certificate.root_interval[1]
    assert abs(point.state_at(certificate.root).rho - exit_rho) < 1e-12


def test_spatial_ks_rho_exit_event_rejects_already_outside_threshold():
    masses, positions, velocities = _spatial_state()
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=12)
    certificate = certify_spatial_ks_binary_rho_exit_event(
        interval,
        exit_rho=0.1 * initial.rho,
        s_upper=0.01,
        coefficient_count=10,
    )

    assert not certificate.certified
    assert certificate.root is None


def test_spatial_ordinary_ks_entry_event_isolates_decreasing_pair_distance():
    masses = np.array([1.0, 1.2, 1.5])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.02, 0.01],
            [1.0, 0.8, 0.5],
        ]
    )
    velocities = np.array(
        [
            [0.03, 0.0, 0.0],
            [-0.03, -0.001, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    interval = construct_interval_taylor_solution(positions, velocities, masses, order=16)
    point = construct_taylor_solution(positions, velocities, masses, order=16)
    enter_distance = 0.198
    certificate = certify_spatial_ordinary_ks_entry_event(
        interval,
        pair=(0, 1),
        enter_distance=enter_distance,
        time_upper=0.1,
        coefficient_count=14,
    )

    point_positions = point.positions_at(certificate.root)

    assert certificate.certified
    assert certificate.interval_is_isolated
    assert certificate.excludes_earlier_roots
    assert certificate.coefficient_source == "spatial_ordinary_interval_taylor"
    assert certificate.root_interval[0] <= certificate.root <= certificate.root_interval[1]
    assert abs(np.linalg.norm(point_positions[1] - point_positions[0]) - enter_distance) < 5e-11


def test_spatial_ordinary_entry_event_lifts_to_certified_ks_branch_atlas():
    masses = np.array([1.0, 1.2, 1.5])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.02, 0.01],
            [1.0, 0.8, 0.5],
        ]
    )
    velocities = np.array(
        [
            [0.03, 0.0, 0.0],
            [-0.03, -0.001, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    interval = construct_interval_taylor_solution(positions, velocities, masses, order=16)
    point = construct_taylor_solution(positions, velocities, masses, order=16)
    certificate = certify_spatial_ordinary_ks_entry_event(
        interval,
        pair=(0, 1),
        enter_distance=0.198,
        time_upper=0.1,
        coefficient_count=14,
    )
    chart = spatial_ordinary_entry_event_to_ks_chart_state(
        interval,
        certificate,
        branch="positive_x",
    )
    atlas = spatial_ordinary_entry_event_to_ks_chart_state_atlas(interval, certificate)
    point_positions = point.positions_at(certificate.root)
    point_velocities = point.velocities_at(certificate.root)

    assert chart.certified
    assert chart.branch_certificate.branch == "positive_x"
    assert chart.contains_physical_point(point_positions, point_velocities)
    assert {member.branch_certificate.branch for member in atlas} == {
        "positive_x",
        "negative_x",
    }
    assert all(member.certified for member in atlas)
    assert all(member.contains_physical_point(point_positions, point_velocities) for member in atlas)


def test_spatial_ordinary_ks_entry_event_rejects_non_crossing_pair():
    masses = np.array([1.0, 1.2, 1.5])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.02, 0.01],
            [1.0, 0.8, 0.5],
        ]
    )
    velocities = np.array(
        [
            [-0.02, 0.0, 0.0],
            [0.02, 0.001, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    interval = construct_interval_taylor_solution(positions, velocities, masses, order=12)
    certificate = certify_spatial_ordinary_ks_entry_event(
        interval,
        pair=(0, 1),
        enter_distance=0.198,
        time_upper=0.001,
        coefficient_count=10,
    )

    assert not certificate.certified
    assert certificate.root is None


def test_spatial_ks_competing_binary_entry_event_isolates_next_pair():
    initial = replace(
        _exact_collision_state(),
        third_offset=np.array([0.05, 0.0, 0.0]),
        third_offset_velocity=np.array([-100.0, 0.0, 0.0]),
    )
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=40)
    point = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    enter_distance = 0.02
    certificate = certify_spatial_ks_competing_binary_entry_event(
        interval,
        pair=(0, 2),
        enter_distance=enter_distance,
        s_upper=0.14,
        coefficient_count=30,
    )

    positions, _velocities = ks_binary_chart_to_spatial(point.state_at(certificate.root))

    assert certificate.certified
    assert certificate.interval_is_isolated
    assert certificate.excludes_earlier_roots
    assert certificate.coefficient_source == "spatial_ks_binary_interval_taylor"
    assert certificate.root_interval[0] <= certificate.root <= certificate.root_interval[1]
    assert abs(np.linalg.norm(positions[2] - positions[0]) - enter_distance) < 2e-8


def test_spatial_ks_competing_binary_entry_event_rejects_selected_pair():
    initial = _exact_collision_state()
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=12)

    with pytest.raises(ValueError, match="competing KS entry pair"):
        certify_spatial_ks_competing_binary_entry_event(
            interval,
            pair=(0, 1),
            enter_distance=0.1,
            s_upper=0.1,
            coefficient_count=10,
        )

    with pytest.raises(ValueError, match="pair must contain two distinct body indices"):
        certify_spatial_ks_competing_binary_entry_event(
            interval,
            pair=(0, 0),
            enter_distance=0.1,
            s_upper=0.1,
            coefficient_count=10,
        )


def test_spatial_ks_competing_binary_entry_event_lifts_to_next_ks_branch_atlas():
    initial = replace(
        _exact_collision_state(),
        third_offset=np.array([0.05, 0.0, 0.0]),
        third_offset_velocity=np.array([-100.0, 0.0, 0.0]),
    )
    interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=40)
    point = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    certificate = certify_spatial_ks_competing_binary_entry_event(
        interval,
        pair=(0, 2),
        enter_distance=0.02,
        s_upper=0.14,
        coefficient_count=30,
    )
    chart = spatial_ks_competing_entry_event_to_ks_chart_state(
        interval,
        certificate,
        branch="positive_x",
    )
    atlas = spatial_ks_competing_entry_event_to_ks_chart_state_atlas(
        interval,
        certificate,
    )
    positions, velocities = ks_binary_chart_to_spatial(point.state_at(certificate.root))

    assert certificate.certified
    assert chart.certified
    assert chart.branch_certificate.branch == "positive_x"
    assert chart.contains_physical_point(positions, velocities)
    assert {member.branch_certificate.branch for member in atlas} == {"positive_x"}
    assert all(member.certified for member in atlas)
    assert all(member.contains_physical_point(positions, velocities) for member in atlas)


def test_spatial_ks_competing_binary_handoff_builds_two_ks_atlas():
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
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    initial_interval = interval_spatial_ks_binary_chart_state_from_point(initial)
    first_interval = construct_interval_spatial_ks_binary_taylor_solution(initial, order=28)
    first_point = construct_spatial_ks_binary_taylor_solution(initial, order=28)
    competing_event = certify_spatial_ks_competing_binary_entry_event(
        first_interval,
        pair=(0, 2),
        enter_distance=0.025,
        s_upper=0.02,
        coefficient_count=20,
    )
    event_positions, event_velocities = ks_binary_chart_to_spatial(
        first_point.state_at(competing_event.root)
    )
    next_point = spatial_to_ks_binary_chart(
        event_positions,
        event_velocities,
        masses,
        pair=(0, 2),
    )
    next_point_solution = construct_spatial_ks_binary_taylor_solution(
        next_point,
        order=28,
    )
    target_s = 1.0e-4
    target_time = (
        first_point.physical_time_at(competing_event.root)
        + next_point_solution.physical_time_at(target_s)
    )
    target_positions, target_velocities = ks_binary_chart_to_spatial(
        next_point_solution.state_at(target_s)
    )
    atlas = validated_atlas_from_spatial_ks_competing_binary_handoff(
        initial_interval,
        competing_pair=(0, 2),
        enter_distance=0.025,
        entry_s_upper=0.02,
        branch="positive_x",
        next_s_endpoint=2.0e-4,
        target_time_after_ks_start_interval=FloatInterval.point(target_time),
        retained_order=20,
        guard_order=8,
    )

    assert atlas.chart_count == 2
    assert atlas.transition_count == 1
    assert [chart.chart_type for chart in atlas.charts] == [
        "spatial_ks_binary",
        "spatial_ks_binary",
    ]
    assert atlas.transitions[0].transition_type == "spatial_ks_to_ks_competing_binary_entry"
    assert atlas.transitions[0].certified
    assert atlas.evaluation.competing_entry_event_certificate.certified
    assert atlas.evaluation.next_ks_state.certified
    assert atlas.evaluation.next_ks_state.pair == (0, 2)
    assert atlas.residual_budget.certified
    assert atlas.invariants.certified
    assert atlas.tail_budget.certified
    assert atlas.target_state_contains(
        np.concatenate([target_positions.reshape(-1), target_velocities.reshape(-1)])
    )
    assert "spatial_ks_competing_entry_event_isolation" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert "spatial_ks_to_ks_transition" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert atlas.evaluation.first_collision_policy_certificate.certified
    assert atlas.evaluation.next_collision_policy_certificate.certified
    assert "spatial_collision_policy_scope" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert atlas.proof_ledger_covers_solution
    assert atlas.proof_certified
    assert atlas.missing_certification_obligations == ()


def test_finite_time_event_set_certifies_target_before_competing_ks_events():
    masses = np.array([0.8, 1.2, 1.7])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [-0.03, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    retained_order = 12
    guard_order = 6
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    interval = construct_interval_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    point = construct_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    first_event = certify_spatial_ks_competing_binary_entry_event(
        interval,
        pair=(0, 2),
        enter_distance=0.025,
        s_upper=0.03,
        coefficient_count=retained_order,
    )
    target_time = 0.5 * point.physical_time_at(first_event.root)

    event_set = certify_next_finite_time_event_set(
        interval,
        target_time_after_start_interval=FloatInterval.point(target_time),
        selected_pair=(0, 1),
        competing_enter_distance=0.025,
        competing_s_upper=0.03,
        retained_order=retained_order,
    )

    assert first_event.certified
    assert event_set.well_formed
    assert event_set.certified
    assert event_set.target_before_all_events
    assert event_set.no_event_before_target_certified
    assert event_set.first_event is None
    assert event_set.missing_obligations == ()
    assert {candidate.pair for candidate in event_set.event_candidates} == {
        (0, 2),
        (1, 2),
    }


def test_finite_time_event_set_selects_unique_first_competing_ks_event():
    masses = np.array([0.8, 1.2, 1.7])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [-0.03, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    retained_order = 12
    guard_order = 6
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    interval = construct_interval_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    point = construct_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    first_event = certify_spatial_ks_competing_binary_entry_event(
        interval,
        pair=(0, 2),
        enter_distance=0.025,
        s_upper=0.03,
        coefficient_count=retained_order,
    )
    target_time = point.physical_time_at(first_event.root) + 1.0e-5

    event_set = certify_next_finite_time_event_set(
        interval,
        target_time_after_start_interval=FloatInterval.point(target_time),
        selected_pair=(0, 1),
        competing_enter_distance=0.025,
        competing_s_upper=0.03,
        retained_order=retained_order,
    )

    assert first_event.certified
    assert event_set.well_formed
    assert event_set.certified
    assert not event_set.target_before_all_events
    assert event_set.unique_first_event_certified
    assert event_set.first_event is not None
    assert event_set.first_event.pair == (0, 2)
    assert event_set.first_event.event_type == "spatial_ks_competing_binary_entry"
    assert event_set.first_event.physical_time_interval.upper < target_time
    assert event_set.missing_obligations == ()


def test_finite_time_event_set_refuses_overlapping_competing_ks_events():
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array(
        [
            [-0.005, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -20.0, 0.0],
        ]
    )
    retained_order = 8
    guard_order = 4
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    interval = construct_interval_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )

    event_set = certify_next_finite_time_event_set(
        interval,
        target_time_after_start_interval=FloatInterval.point(1.0e-3),
        selected_pair=(0, 1),
        competing_enter_distance=0.045,
        competing_s_upper=0.1,
        retained_order=retained_order,
    )

    assert event_set.well_formed
    assert not event_set.certified
    assert not event_set.unique_first_event_certified
    assert event_set.multiple_possible_first_events_requiring_split
    assert event_set.ambiguous_event_order_partition is not None
    assert event_set.ambiguous_event_order_partition.well_formed
    assert event_set.ambiguous_event_order_partition.event_alternative_cover_certified
    assert not event_set.ambiguous_event_order_partition.certified
    assert (
        "state_space_event_order_partition_not_constructed"
        in event_set.ambiguous_event_order_partition.missing_obligations
    )
    assert {
        leaf.pair
        for leaf in event_set.ambiguous_event_order_partition.branch_leaves
    } == {
        (0, 2),
        (1, 2),
    }
    assert all(
        leaf.certified
        and leaf.competing_first_event_ids
        and leaf.physical_time_interval.lower <= leaf.physical_time_interval.upper
        for leaf in event_set.ambiguous_event_order_partition.branch_leaves
    )
    assert {candidate.pair for candidate in event_set.event_candidates} == {
        (0, 2),
        (1, 2),
    }
    assert any(
        obligation.startswith("finite_time_event_order_requires_split")
        for obligation in event_set.missing_obligations
    )


def test_ks_event_order_partition_certifies_overestimated_competing_order():
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array(
        [
            [-0.005, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -20.0, 0.0],
        ]
    )
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    state = interval_spatial_ks_binary_chart_state_from_point(initial)
    third_offset = state.third_offset.copy()
    third_offset[0] = FloatInterval(1.0e-4, 5.0e-4)
    state = replace(state, third_offset=third_offset)
    retained_order = 4
    guard_order = 2
    interval = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        state,
        order=retained_order + guard_order,
    )

    unsplit_event_set = certify_next_finite_time_event_set(
        interval,
        target_time_after_start_interval=FloatInterval.point(1.0e-3),
        selected_pair=(0, 1),
        competing_enter_distance=0.045,
        competing_s_upper=0.1,
        retained_order=retained_order,
    )
    partition = partition_ks_state_by_competing_event_order(
        state,
        target_time_after_start_interval=FloatInterval.point(1.0e-3),
        selected_pair=(0, 1),
        competing_enter_distance=0.045,
        competing_s_upper=0.1,
        retained_order=retained_order,
        guard_order=guard_order,
        max_depth=2,
        max_branches=8,
    )

    assert unsplit_event_set.multiple_possible_first_events_requiring_split
    assert partition.well_formed
    assert partition.certified
    assert partition.split_count == 2
    assert partition.certified_leaf_count == len(partition.leaves)
    assert partition.recursive_bisection_cover_certified
    assert partition.leaf_decisions_certified
    assert {leaf.decision for leaf in partition.leaves} == {"unique_first_event"}
    assert {leaf.first_event_pair for leaf in partition.leaves} == {(1, 2)}
    assert partition.missing_obligations == ()


def test_ks_event_order_partition_consumes_certified_leaves_as_branch_union():
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array(
        [
            [-0.005, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -20.0, 0.0],
        ]
    )
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    state = interval_spatial_ks_binary_chart_state_from_point(initial)
    third_offset = state.third_offset.copy()
    third_offset[0] = FloatInterval(1.0e-4, 5.0e-4)
    state = replace(state, third_offset=third_offset)

    partition = partition_ks_state_by_competing_event_order(
        state,
        target_time_after_start_interval=FloatInterval.point(2.7e-4),
        selected_pair=(0, 1),
        competing_enter_distance=0.045,
        competing_s_upper=0.1,
        retained_order=8,
        guard_order=4,
        max_depth=2,
        max_branches=8,
    )
    atlas = validated_atlas_from_spatial_ks_event_order_partition(
        partition,
        retained_order=8,
        guard_order=4,
        competing_pair_min_distance_required=0.0,
        max_competing_repeats=0,
    )

    assert partition.certified
    assert atlas.proof_certified
    assert atlas.charts[0].chart_type == "spatial_ks_event_order_branch_union"
    assert len(atlas.evaluation.branch_atlases) == len(partition.leaves)
    assert all(member.proof_certified for member in atlas.evaluation.branch_atlases)
    assert atlas.evaluation.branch_partition is partition
    assert atlas.missing_certification_obligations == ()
    assert any(
        entry.name == "finite_time_event_order_branch_union_consumption"
        and entry.certified
        for entry in atlas.proof_ledger.entries
    )


def test_spatial_ks_segmented_tail_certifies_middle_competing_chart():
    masses = np.array([0.8, 1.2, 1.7])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [-0.03, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    retained_order = 12
    guard_order = 6
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    first_interval = construct_interval_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    first_point = construct_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    first_event = certify_spatial_ks_competing_binary_entry_event(
        first_interval,
        pair=(0, 2),
        enter_distance=0.025,
        s_upper=0.03,
        coefficient_count=retained_order,
    )
    middle_state = spatial_ks_competing_entry_event_to_ks_chart_state(
        first_interval,
        first_event,
        branch="negative_x",
    )
    middle_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        middle_state,
        order=retained_order + guard_order,
    )
    second_event = certify_spatial_ks_competing_binary_entry_event(
        middle_solution,
        pair=(1, 2),
        enter_distance=0.025,
        s_upper=0.03,
        coefficient_count=retained_order,
    )
    direct_tail = spatial_ks_binary_interval_tail_certificate(
        middle_state,
        retained_order=retained_order,
        guard_order=guard_order,
        step_size=second_event.root,
    )
    segmented_tail = spatial_ks_binary_interval_segmented_tail_certificate(
        middle_state,
        retained_order=retained_order,
        guard_order=guard_order,
        step_size=second_event.root,
        segment_count=2,
    )
    event_positions, event_velocities = ks_binary_chart_to_spatial(
        first_point.state_at(first_event.root)
    )
    middle_point = spatial_to_ks_binary_chart(
        event_positions,
        event_velocities,
        masses,
        pair=(0, 2),
    )
    middle_point_solution = construct_spatial_ks_binary_taylor_solution(
        middle_point,
        order=retained_order + guard_order,
    )

    assert first_event.certified
    assert second_event.certified
    assert not direct_tail.is_nontrivial
    assert not np.isfinite(direct_tail.tail_bound)
    assert segmented_tail.is_nontrivial
    assert segmented_tail.segment_count == 2
    assert sum(segmented_tail.segment_step_sizes) == pytest.approx(second_event.root)
    assert segmented_tail.local_tail_bound >= segmented_tail.max_step_tail_bound >= 0.0
    assert segmented_tail.propagation_chain_certified
    assert segmented_tail.segment_handoffs_certified
    assert segmented_tail.physical_time_accumulation_certified
    assert segmented_tail.final_state_matches_last_segment
    assert len(segmented_tail.segment_propagation_steps) == segmented_tail.segment_count
    assert all(step.certified for step in segmented_tail.segment_propagation_steps)
    assert all(
        step.equation_residual_certified
        for step in segmented_tail.segment_propagation_steps
    )
    assert all(
        step.projection_constraints_certified
        for step in segmented_tail.segment_propagation_steps
    )
    assert all(
        step.chart_evidence_certified
        for step in segmented_tail.segment_propagation_steps
    )
    assert all(
        left.end_state is right.start_state
        for left, right in zip(
            segmented_tail.segment_propagation_steps,
            segmented_tail.segment_propagation_steps[1:],
        )
    )
    assert segmented_tail.final_state is not None
    assert segmented_tail.final_physical_time_delta is not None
    assert segmented_tail.final_state.certified
    assert segmented_tail.final_state.contains_point_state(
        middle_point_solution.state_at(second_event.root)
    )
    reference_time = middle_point_solution.physical_time_at(second_event.root)
    assert (
        segmented_tail.final_physical_time_delta.lower
        <= reference_time
        <= segmented_tail.final_physical_time_delta.upper
    )


def test_spatial_ks_target_inside_atlas_uses_segmented_tail_when_direct_tail_fails():
    masses = np.array([0.8, 1.2, 1.7])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [-0.03, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    retained_order = 12
    guard_order = 6
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    first_interval = construct_interval_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    first_point = construct_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    first_event = certify_spatial_ks_competing_binary_entry_event(
        first_interval,
        pair=(0, 2),
        enter_distance=0.025,
        s_upper=0.03,
        coefficient_count=retained_order,
    )
    middle_state = spatial_ks_competing_entry_event_to_ks_chart_state(
        first_interval,
        first_event,
        branch="negative_x",
    )
    middle_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        middle_state,
        order=retained_order + guard_order,
    )
    second_event = certify_spatial_ks_competing_binary_entry_event(
        middle_solution,
        pair=(1, 2),
        enter_distance=0.025,
        s_upper=0.03,
        coefficient_count=retained_order,
    )
    direct_tail = spatial_ks_binary_interval_tail_certificate(
        middle_state,
        retained_order=retained_order,
        guard_order=guard_order,
        step_size=second_event.root,
    )
    event_positions, event_velocities = ks_binary_chart_to_spatial(
        first_point.state_at(first_event.root)
    )
    middle_point = spatial_to_ks_binary_chart(
        event_positions,
        event_velocities,
        masses,
        pair=(0, 2),
    )
    middle_point_solution = construct_spatial_ks_binary_taylor_solution(
        middle_point,
        order=retained_order + guard_order,
    )
    target_time = middle_point_solution.physical_time_at(second_event.root)
    target_positions, target_velocities = ks_binary_chart_to_spatial(
        middle_point_solution.state_at(second_event.root)
    )

    atlas = validated_atlas_from_spatial_ks_binary_chart(
        middle_state,
        s_endpoint=1.25 * second_event.root,
        target_time_after_ks_start_interval=FloatInterval.point(target_time),
        retained_order=retained_order,
        guard_order=guard_order,
    )

    assert first_event.certified
    assert second_event.certified
    assert not direct_tail.is_nontrivial
    assert not np.isfinite(direct_tail.tail_bound)
    assert atlas.chart_count == 1
    assert atlas.transition_count == 0
    assert atlas.charts[0].chart_type == "spatial_ks_binary"
    assert atlas.charts[0].tail_certified
    assert np.isfinite(atlas.charts[0].tail_bound)
    assert atlas.tail_budget.certified
    assert atlas.evaluation.ordinary_solution is None
    assert atlas.evaluation.ks_tail_certificate.__class__.__name__ == (
        "SpatialKSSegmentedTailCertificate"
    )
    assert atlas.evaluation.ks_tail_certificate.propagation_chain_certified
    assert atlas.evaluation.endpoint_projection.certified
    assert atlas.target_state_contains(
        np.concatenate([target_positions.reshape(-1), target_velocities.reshape(-1)])
    )
    assert any(
        entry.name == "spatial_ks_target_time_projection"
        and "re-expanded KS tail segments" in entry.detail
        for entry in atlas.proof_ledger.entries
    )


def test_spatial_ks_competing_handoff_uses_segmented_tail_for_first_leg():
    masses = np.array([0.8, 1.2, 1.7])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [-0.03, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    retained_order = 12
    guard_order = 6
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    first_interval = construct_interval_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    first_point = construct_spatial_ks_binary_taylor_solution(
        initial,
        order=retained_order + guard_order,
    )
    first_event = certify_spatial_ks_competing_binary_entry_event(
        first_interval,
        pair=(0, 2),
        enter_distance=0.025,
        s_upper=0.03,
        coefficient_count=retained_order,
    )
    middle_state = spatial_ks_competing_entry_event_to_ks_chart_state(
        first_interval,
        first_event,
        branch="negative_x",
    )
    middle_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        middle_state,
        order=retained_order + guard_order,
    )
    second_event = certify_spatial_ks_competing_binary_entry_event(
        middle_solution,
        pair=(1, 2),
        enter_distance=0.025,
        s_upper=0.03,
        coefficient_count=retained_order,
    )
    direct_tail = spatial_ks_binary_interval_tail_certificate(
        middle_state,
        retained_order=retained_order,
        guard_order=guard_order,
        step_size=second_event.root,
    )
    event_positions, event_velocities = ks_binary_chart_to_spatial(
        first_point.state_at(first_event.root)
    )
    middle_point = spatial_to_ks_binary_chart(
        event_positions,
        event_velocities,
        masses,
        pair=(0, 2),
    )
    middle_point_solution = construct_spatial_ks_binary_taylor_solution(
        middle_point,
        order=retained_order + guard_order,
    )
    next_positions, next_velocities = ks_binary_chart_to_spatial(
        middle_point_solution.state_at(second_event.root)
    )
    next_point = spatial_to_ks_binary_chart(
        next_positions,
        next_velocities,
        masses,
        pair=(1, 2),
    )
    next_point_solution = construct_spatial_ks_binary_taylor_solution(
        next_point,
        order=retained_order + guard_order,
    )
    target_s = 1.0e-4
    target_time = (
        middle_point_solution.physical_time_at(second_event.root)
        + next_point_solution.physical_time_at(target_s)
    )
    target_positions, target_velocities = ks_binary_chart_to_spatial(
        next_point_solution.state_at(target_s)
    )

    atlas = validated_atlas_from_spatial_ks_competing_binary_handoff(
        middle_state,
        competing_pair=(1, 2),
        enter_distance=0.025,
        entry_s_upper=0.03,
        branch="negative_x",
        next_s_endpoint=2.0e-4,
        target_time_after_ks_start_interval=FloatInterval.point(target_time),
        retained_order=retained_order,
        guard_order=guard_order,
    )

    assert first_event.certified
    assert second_event.certified
    assert not direct_tail.is_nontrivial
    assert not np.isfinite(direct_tail.tail_bound)
    assert atlas.proof_certified
    assert atlas.missing_certification_obligations == ()
    assert [chart.chart_type for chart in atlas.charts] == [
        "spatial_ks_binary",
        "spatial_ks_binary",
    ]
    assert atlas.charts[0].tail_certified
    assert np.isfinite(atlas.charts[0].tail_bound)
    assert atlas.evaluation.first_ks_tail_certificate.__class__.__name__ == (
        "SpatialKSSegmentedTailCertificate"
    )
    assert atlas.evaluation.first_ks_tail_certificate.propagation_chain_certified
    assert atlas.tail_budget.certified
    assert atlas.target_state_contains(
        np.concatenate([target_positions.reshape(-1), target_velocities.reshape(-1)])
    )
    assert any(
        entry.name == "local_tail_budget"
        and "re-expanded interval KS subcharts" in entry.detail
        for entry in atlas.proof_ledger.entries
    )


def test_spatial_ks_local_handoff_enters_validated_atlas_surface_without_global_claim():
    masses, positions, velocities = _spatial_state()
    state_interval = _interval_state_box(positions, velocities, half_width=1e-6)
    chart = spatial_interval_to_ks_binary_chart_state(
        state_interval,
        masses,
        pair=(0, 1),
        branch="positive_x",
    )
    point_initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    point_solution = construct_spatial_ks_binary_taylor_solution(point_initial, order=20)
    s_endpoint = 0.004
    ordinary_step = 1.0e-12
    endpoint_state = point_solution.state_at(s_endpoint)
    endpoint_positions, endpoint_velocities = ks_binary_chart_to_spatial(endpoint_state)
    reference = integrate_reference(
        endpoint_positions,
        endpoint_velocities,
        masses,
        ordinary_step,
    )
    atlas = validated_atlas_from_spatial_ks_binary_chart(
        chart,
        s_endpoint=s_endpoint,
        ordinary_step_size=ordinary_step,
        retained_order=12,
        guard_order=6,
    )

    assert atlas.chart_count == 2
    assert atlas.transition_count == 1
    assert [chart.chart_type for chart in atlas.charts] == [
        "spatial_ks_binary",
        "spatial_ordinary_taylor_after_ks",
    ]
    assert atlas.transitions[0].transition_type == (
        "spatial_ks_to_ordinary_rho_positive_endpoint_projection"
    )
    assert atlas.transitions[0].certified
    assert all(chart.certified for chart in atlas.charts)
    assert atlas.residual_budget.certified
    assert atlas.invariants.certified
    assert atlas.tail_budget.certified
    assert atlas.target_state_contains(reference)
    assert not atlas.proof_certified
    assert atlas.proof_ledger.missing_required_obligations == (
        "spatial_ks_entry_event_isolation",
        "finite_time_physical_targeting",
        "spatial_collision_policy_scope",
    )


def test_spatial_ks_exact_collision_exit_event_hands_off_to_ordinary_chart_but_not_theorem():
    initial = _exact_collision_state()
    interval = interval_spatial_ks_binary_chart_state_from_point(initial)
    point_solution = construct_spatial_ks_binary_taylor_solution(initial, order=24)
    exit_rho = 1.0e-4
    ordinary_step = 1.0e-12
    atlas = validated_atlas_from_spatial_ks_binary_chart(
        interval,
        exit_rho=exit_rho,
        s_upper=0.02,
        ordinary_step_size=ordinary_step,
        retained_order=14,
        guard_order=6,
    )
    endpoint_positions, endpoint_velocities = ks_binary_chart_to_spatial(
        point_solution.state_at(atlas.evaluation.exit_event_certificate.root)
    )
    reference = integrate_reference(
        endpoint_positions,
        endpoint_velocities,
        initial.masses,
        ordinary_step,
    )

    assert atlas.charts[0].projection_certified
    assert atlas.charts[1].residual_certified
    assert atlas.transitions[0].certified
    assert atlas.target_state_contains(reference)
    assert atlas.evaluation.endpoint_projection.certified
    assert atlas.evaluation.exit_event_certificate.certified
    assert atlas.evaluation.endpoint_projection.rho.lower > 0.0
    assert not atlas.proof_certified
    assert "spatial_ks_exit_event_isolation" not in atlas.proof_ledger.missing_required_obligations
    assert "spatial_ks_entry_event_isolation" in atlas.proof_ledger.missing_required_obligations


def test_spatial_ks_handoff_admissibility_blocks_rho_positive_but_too_close_exit():
    initial = _exact_collision_state()
    interval = interval_spatial_ks_binary_chart_state_from_point(initial)
    interval_solution = construct_interval_spatial_ks_binary_taylor_solution(initial, order=24)
    exit_certificate = certify_spatial_ks_binary_rho_exit_event(
        interval_solution,
        exit_rho=1.0e-8,
        s_upper=0.01,
        coefficient_count=14,
    )
    projection = project_spatial_ks_binary_taylor_endpoint_to_physical(
        interval_solution,
        exit_certificate.root,
    )
    certificate = certify_spatial_ks_to_ordinary_handoff_admissibility(
        projection,
        retained_order=14,
        requested_post_time_interval=FloatInterval.point(0.0),
        min_pair_distance_required=1.0e-3,
    )

    assert exit_certificate.certified
    assert projection.certified
    assert not certificate.certified
    assert "ordinary_handoff_pair_distance_too_small" in certificate.missing_obligations
    with pytest.raises(ValueError, match="ordinary handoff is not admissible"):
        validated_atlas_from_spatial_ks_binary_chart(
            interval,
            exit_rho=1.0e-8,
            s_upper=0.01,
            ordinary_step_size=1.0e-7,
            retained_order=14,
            guard_order=6,
            ordinary_handoff_min_pair_distance_required=1.0e-3,
        )


def test_spatial_ks_handoff_admissibility_requires_constructed_ordinary_chart_evidence():
    initial = _exact_collision_state()
    interval_solution = construct_interval_spatial_ks_binary_taylor_solution(initial, order=24)
    exit_certificate = certify_spatial_ks_binary_rho_exit_event(
        interval_solution,
        exit_rho=1.0e-4,
        s_upper=0.02,
        coefficient_count=14,
    )
    projection = project_spatial_ks_binary_taylor_endpoint_to_physical(
        interval_solution,
        exit_certificate.root,
    )
    certificate = certify_spatial_ks_to_ordinary_handoff_admissibility(
        projection,
        retained_order=14,
        requested_post_time_interval=FloatInterval.point(1.0e-7),
    )

    assert exit_certificate.certified
    assert projection.certified
    assert not certificate.certified
    assert certificate.ordinary_chart_evidence_required
    assert "ordinary_handoff_constructed_chart_missing" in certificate.missing_obligations
    assert "ordinary_handoff_residual_certificate_missing" in certificate.missing_obligations
    assert "ordinary_handoff_residual_not_certified" in certificate.missing_obligations


def test_spatial_ks_handoff_admissibility_rejects_zero_post_handoff_interval_with_ordinary_evidence():
    initial = _exact_collision_state()
    interval_solution = construct_interval_spatial_ks_binary_taylor_solution(initial, order=24)
    exit_certificate = certify_spatial_ks_binary_rho_exit_event(
        interval_solution,
        exit_rho=1.0e-4,
        s_upper=0.02,
        coefficient_count=14,
    )
    projection = project_spatial_ks_binary_taylor_endpoint_to_physical(
        interval_solution,
        exit_certificate.root,
    )
    ordinary_positions, ordinary_velocities = validated_atlas_module._spatial_interval_state_arrays(
        projection.state_interval,
    )
    ordinary_solution = validated_atlas_module.construct_interval_taylor_solution_from_intervals(
        ordinary_positions,
        ordinary_velocities,
        initial.masses,
        order=20,
    )
    ordinary_residual = validated_atlas_module.certify_ordinary_interval_taylor_equations(
        ordinary_solution,
        coefficient_count=14,
    )
    ordinary_tail = validated_atlas_module.interval_guarded_tail_certificate(
        validated_atlas_module.ordinary_interval_solution_arrays(ordinary_solution),
        14,
        0.0,
    )

    certificate = certify_spatial_ks_to_ordinary_handoff_admissibility(
        projection,
        retained_order=14,
        requested_post_time_interval=FloatInterval.point(0.0),
        ordinary_initial_state_interval=projection.state_interval,
        ordinary_solution=ordinary_solution,
        ordinary_residual=ordinary_residual,
        ordinary_tail=ordinary_tail,
    )

    assert exit_certificate.certified
    assert projection.certified
    assert ordinary_residual.certified
    assert not ordinary_tail.is_nontrivial
    assert certificate.ordinary_chart_evidence_required
    assert not certificate.certified
    assert "post_handoff_time_interval_not_positive" in certificate.missing_obligations
    assert "ordinary_handoff_constructed_chart_missing" not in certificate.missing_obligations
    assert "ordinary_handoff_residual_certificate_missing" not in certificate.missing_obligations


def test_spatial_ks_to_ordinary_handoff_inflates_initial_box_by_ks_tail():
    initial = _exact_collision_state()
    interval = interval_spatial_ks_binary_chart_state_from_point(initial)

    atlas = validated_atlas_from_spatial_ks_binary_chart(
        interval,
        s_endpoint=0.04,
        ordinary_step_size=1.0e-12,
        retained_order=14,
        guard_order=6,
    )

    tail = atlas.charts[0].tail_bound
    ordinary_solution = atlas.evaluation.ordinary_solution
    assert ordinary_solution is not None
    ordinary_initial = np.concatenate(
        [
            ordinary_solution.position[0].reshape(-1),
            ordinary_solution.velocity[0].reshape(-1),
        ]
    )

    assert tail > 0.0
    assert atlas.transitions[0].certified
    for component, (lower, upper) in zip(
        ordinary_initial,
        atlas.evaluation.endpoint_projection.state_interval,
        strict=True,
    ):
        assert component.lower <= lower - 0.5 * tail
        assert component.upper >= upper + 0.5 * tail


def test_spatial_ks_default_ordinary_handoff_budget_is_constructor_derived():
    initial = _exact_collision_state()
    interval = interval_spatial_ks_binary_chart_state_from_point(initial)
    ordinary_step = 1.0e-12

    atlas = validated_atlas_from_spatial_ks_binary_chart(
        interval,
        s_endpoint=0.04,
        ordinary_step_size=ordinary_step,
        retained_order=14,
        guard_order=6,
    )
    certificate = atlas.evaluation.ordinary_handoff_admissibility

    assert certificate is not None
    assert certificate.certified
    assert certificate.min_cauchy_radius == pytest.approx(ordinary_step)
    assert certificate.cauchy_radius >= certificate.min_cauchy_radius
    assert np.isfinite(certificate.max_acceleration_bound)
    assert np.isfinite(certificate.max_residual_bound)
    assert np.isfinite(certificate.max_tail_bound)
    assert certificate.acceleration_bound <= certificate.max_acceleration_bound
    assert certificate.residual_bound <= certificate.max_residual_bound
    assert certificate.tail_bound <= certificate.max_tail_bound


def test_spatial_ks_default_cauchy_budget_keeps_target_in_regularized_chart():
    initial = _exact_collision_state()
    interval = interval_spatial_ks_binary_chart_state_from_point(initial)
    point_solution = construct_spatial_ks_binary_taylor_solution(initial, order=32)
    interval_solution = construct_interval_spatial_ks_binary_taylor_solution(
        initial,
        order=32,
    )
    exit_certificate = certify_spatial_ks_binary_rho_exit_event(
        interval_solution,
        exit_rho=1.0e-4,
        s_upper=0.02,
        coefficient_count=18,
    )
    endpoint_projection = project_spatial_ks_binary_taylor_endpoint_to_physical(
        interval_solution,
        exit_certificate.root,
    )
    target_s = exit_certificate.root + 1.0e-3
    target_time = FloatInterval.point(point_solution.physical_time_at(target_s))
    requested_post_handoff_interval = target_time - endpoint_projection.physical_time
    pre_handoff = certify_spatial_ks_to_ordinary_handoff_admissibility(
        endpoint_projection,
        retained_order=18,
        requested_post_time_interval=requested_post_handoff_interval,
        require_ordinary_chart_evidence=False,
    )

    atlas = validated_atlas_from_spatial_ks_binary_chart(
        interval,
        exit_rho=1.0e-4,
        s_upper=0.02,
        target_time_after_ks_start_interval=target_time,
        retained_order=18,
        guard_order=8,
    )

    assert exit_certificate.certified
    assert endpoint_projection.certified
    assert target_s > exit_certificate.root
    assert pre_handoff.min_cauchy_radius == pytest.approx(
        requested_post_handoff_interval.upper
    )
    assert not pre_handoff.certified
    assert "ordinary_handoff_cauchy_radius_too_small" in pre_handoff.missing_obligations
    assert atlas.chart_count == 1
    assert atlas.charts[0].chart_type == "spatial_ks_binary"
    assert atlas.evaluation.ordinary_solution is None
    assert atlas.evaluation.ordinary_handoff_admissibility is None
    assert atlas.evaluation.endpoint_projection.s_value == pytest.approx(target_s)


def test_spatial_ordinary_after_ks_chart_requires_nested_handoff_certificate():
    initial = _exact_collision_state()
    interval = interval_spatial_ks_binary_chart_state_from_point(initial)
    atlas = validated_atlas_from_spatial_ks_binary_chart(
        interval,
        s_endpoint=0.04,
        ordinary_step_size=1.0e-12,
        retained_order=14,
        guard_order=6,
    )
    stale_atlas = replace(
        atlas,
        evaluation=replace(atlas.evaluation, ordinary_handoff_admissibility=None),
    )

    assert "spatial_ordinary_taylor_after_ks" in {
        chart.chart_type for chart in atlas.charts
    }
    assert atlas.spatial_ks_ordinary_handoff_structure_certified
    assert not stale_atlas.spatial_ks_ordinary_handoff_structure_certified
    assert "spatial_ks_ordinary_handoff_admissibility_structure" in (
        stale_atlas.missing_certification_obligations
    )


def test_spatial_ks_target_before_ordinary_safe_exit_evaluates_inside_ks_chart():
    initial = _exact_collision_state()
    interval = interval_spatial_ks_binary_chart_state_from_point(initial)
    point_solution = construct_spatial_ks_binary_taylor_solution(initial, order=32)
    target_s = 2.0e-3
    target_time = point_solution.physical_time_at(target_s)
    target_positions, target_velocities = ks_binary_chart_to_spatial(
        point_solution.state_at(target_s)
    )

    atlas = validated_atlas_from_spatial_ks_binary_chart(
        interval,
        exit_rho=1.0e-4,
        s_upper=0.02,
        target_time_after_ks_start_interval=FloatInterval.point(target_time),
        retained_order=18,
        guard_order=8,
        ordinary_handoff_min_pair_distance_required=1.0,
    )

    assert atlas.chart_count == 1
    assert atlas.transition_count == 0
    assert atlas.charts[0].chart_type == "spatial_ks_binary"
    assert atlas.evaluation.ordinary_solution is None
    assert atlas.evaluation.endpoint_projection.certified
    assert atlas.evaluation.endpoint_projection.s_value == pytest.approx(target_s)
    assert atlas.evaluation.ordinary_handoff_admissibility is None
    target_positions, target_velocities = ks_binary_chart_to_spatial(
        point_solution.state_at(atlas.evaluation.endpoint_projection.s_value)
    )
    assert atlas.target_state_contains(
        np.concatenate([target_positions.reshape(-1), target_velocities.reshape(-1)])
    )
    assert "ordinary_post_handoff_residuals" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert not atlas.proof_certified


def test_spatial_ks_direct_target_tail_requires_interval_physical_time_containment(
    monkeypatch,
):
    initial = _exact_collision_state()
    interval = interval_spatial_ks_binary_chart_state_from_point(initial)
    retained_order = 18
    guard_order = 8
    target_s = 2.0e-3
    point_solution = construct_spatial_ks_binary_taylor_solution(initial, order=32)
    target_time = FloatInterval.point(point_solution.physical_time_at(target_s))
    interval_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        interval,
        order=retained_order + guard_order,
    )
    direct_tail = validated_atlas_module.spatial_ks_binary_interval_tail_certificate(
        interval,
        retained_order=retained_order,
        guard_order=guard_order,
        step_size=target_s,
    )
    real_project = (
        validated_atlas_module.project_spatial_ks_binary_taylor_endpoint_to_physical
    )

    class EmptySegmentedTail:
        is_nontrivial = False
        final_state = None

    def shifted_physical_time_projection(ks_solution, s_value):
        projection = real_project(ks_solution, s_value)
        return replace(
            projection,
            physical_time=FloatInterval(
                target_time.upper + 1.0,
                target_time.upper + 1.1,
            ),
        )

    monkeypatch.setattr(
        validated_atlas_module,
        "project_spatial_ks_binary_taylor_endpoint_to_physical",
        shifted_physical_time_projection,
    )
    monkeypatch.setattr(
        validated_atlas_module,
        "spatial_ks_binary_interval_segmented_tail_certificate",
        lambda *_args, **_kwargs: EmptySegmentedTail(),
    )
    monkeypatch.setattr(
        validated_atlas_module,
        "_ks_physical_time_roots_for_target_interval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no interval root")),
    )

    assert direct_tail.is_nontrivial
    with pytest.raises(ValueError, match="KS target tail could not be certified"):
        validated_atlas_module._certify_spatial_ks_target_tail(
            interval,
            ks_solution=interval_solution,
            target_s=target_s,
            target_time_after_ks_start_interval=target_time,
            retained_order=retained_order,
            guard_order=guard_order,
        )


def test_spatial_ks_interval_target_tail_brackets_physical_time_interval():
    initial = _exact_collision_state()
    interval = interval_spatial_ks_binary_chart_state_from_point(initial)
    retained_order = 18
    guard_order = 8
    target_s = 2.0e-3
    point_solution = construct_spatial_ks_binary_taylor_solution(initial, order=32)
    center_time = point_solution.physical_time_at(target_s)
    target_time = FloatInterval(center_time - 1.0e-12, center_time + 1.0e-12)
    interval_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        interval,
        order=retained_order + guard_order,
    )

    result = validated_atlas_module._certify_spatial_ks_target_tail(
        interval,
        ks_solution=interval_solution,
        target_s=target_s,
        s_search_upper=0.02,
        target_time_after_ks_start_interval=target_time,
        retained_order=retained_order,
        guard_order=guard_order,
    )

    assert result.tail_certified
    assert result.endpoint_projection.certified
    assert result.endpoint_projection.physical_time.lower <= target_time.lower
    assert result.endpoint_projection.physical_time.upper >= target_time.upper
    assert "endpoint-bracketed" in result.proof_detail


def test_spatial_ks_interval_target_tail_rejects_segmented_point_fallback(
    monkeypatch,
):
    initial = _exact_collision_state()
    interval = interval_spatial_ks_binary_chart_state_from_point(initial)
    retained_order = 18
    guard_order = 8
    target_s = 2.0e-3
    point_solution = construct_spatial_ks_binary_taylor_solution(initial, order=32)
    center_time = point_solution.physical_time_at(target_s)
    target_time = FloatInterval(center_time - 1.0e-12, center_time + 1.0e-12)
    interval_solution = construct_interval_spatial_ks_binary_taylor_solution_from_intervals(
        interval,
        order=retained_order + guard_order,
    )
    real_segmented_tail = spatial_ks_binary_interval_segmented_tail_certificate(
        interval,
        retained_order=retained_order,
        guard_order=guard_order,
        step_size=target_s,
        segment_count=2,
    )

    class FakeSegmentedTail:
        is_nontrivial = True
        final_state = real_segmented_tail.final_state
        final_physical_time_delta = target_time
        local_tail_bound = real_segmented_tail.local_tail_bound

    monkeypatch.setattr(
        validated_atlas_module,
        "_ks_physical_time_roots_for_target_interval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no interval root")),
    )
    monkeypatch.setattr(
        validated_atlas_module,
        "spatial_ks_binary_interval_segmented_tail_certificate",
        lambda *_args, **_kwargs: FakeSegmentedTail(),
    )

    with pytest.raises(ValueError, match="spatial_ks_target_time_endpoint_bracketing"):
        validated_atlas_module._certify_spatial_ks_target_tail(
            interval,
            ks_solution=interval_solution,
            target_s=target_s,
            s_search_upper=0.02,
            target_time_after_ks_start_interval=target_time,
            retained_order=retained_order,
            guard_order=guard_order,
        )


def test_spatial_ordinary_ks_handoff_composes_entry_and_local_exit_chart():
    masses = np.array([1.0, 1.2, 1.5])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.02, 0.01],
            [1.0, 0.8, 0.5],
        ]
    )
    velocities = np.array(
        [
            [0.03, 0.0, 0.0],
            [-0.03, -0.001, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    state_interval = _interval_state_box(positions, velocities, half_width=0.0)
    s_endpoint = 1.0e-4
    ordinary_step = 1.0e-5
    atlas = validated_atlas_from_spatial_ordinary_ks_handoff(
        state_interval,
        masses,
        pair=(0, 1),
        enter_distance=0.198,
        entry_time_upper=0.1,
        branch="positive_x",
        s_endpoint=s_endpoint,
        ordinary_step_size=ordinary_step,
        retained_order=12,
        guard_order=6,
    )

    ordinary_point = construct_taylor_solution(positions, velocities, masses, order=20)
    entry_time = atlas.evaluation.entry_event_certificate.root
    entry_positions = ordinary_point.positions_at(entry_time)
    entry_velocities = ordinary_point.velocities_at(entry_time)
    ks_point = spatial_to_ks_binary_chart(entry_positions, entry_velocities, masses, pair=(0, 1))
    ks_point_solution = construct_spatial_ks_binary_taylor_solution(ks_point, order=20)
    endpoint_positions, endpoint_velocities = ks_binary_chart_to_spatial(
        ks_point_solution.state_at(s_endpoint)
    )
    reference = integrate_reference(endpoint_positions, endpoint_velocities, masses, ordinary_step)

    assert atlas.chart_count == 3
    assert atlas.transition_count == 2
    assert [chart.chart_type for chart in atlas.charts] == [
        "spatial_ordinary_taylor_before_ks",
        "spatial_ks_binary",
        "spatial_ordinary_taylor_after_ks",
    ]
    assert [transition.transition_type for transition in atlas.transitions] == [
        "spatial_ordinary_to_ks_decreasing_distance_entry",
        "spatial_ks_to_ordinary_rho_positive_endpoint_projection",
    ]
    assert atlas.evaluation.entry_event_certificate.certified
    assert atlas.evaluation.entry_ks_state.certified
    assert atlas.evaluation.entry_projection.certified
    assert all(chart.certified for chart in atlas.charts)
    assert all(transition.certified for transition in atlas.transitions)
    assert atlas.residual_budget.certified
    assert atlas.invariants.certified
    assert atlas.tail_budget.certified
    assert atlas.target_state_contains(reference)
    assert not atlas.proof_certified
    assert "spatial_ordinary_ks_entry_event_isolation" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert "spatial_ordinary_to_ks_branch_lift" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert atlas.evaluation.collision_policy_certificate.certified
    assert (
        atlas.evaluation.collision_policy_certificate.min_squared_distance_lower_bound
        > 0.0
    )
    assert "spatial_collision_policy_scope" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert atlas.proof_ledger.missing_required_obligations == (
        "finite_time_physical_targeting",
    )


def test_spatial_ordinary_ks_handoff_can_close_certified_rho_exit_event():
    initial = _exact_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    s_start = -0.04
    s_entry = -0.03
    positions, velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_start)
    )
    entry_positions, _entry_velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_entry)
    )
    entry_time_upper = 1.5 * (
        pre_collision_solution.physical_time_at(s_entry)
        - pre_collision_solution.physical_time_at(s_start)
    )
    enter_distance = float(np.linalg.norm(entry_positions[1] - entry_positions[0]))
    exit_rho = 1.0e-1
    ordinary_step = 1.0e-12
    atlas = validated_atlas_from_spatial_ordinary_ks_handoff(
        _interval_state_box(positions, velocities, half_width=0.0),
        initial.masses,
        pair=(0, 1),
        enter_distance=enter_distance,
        entry_time_upper=entry_time_upper,
        branch="positive_x",
        exit_rho=exit_rho,
        s_upper=0.5,
        ordinary_step_size=ordinary_step,
        retained_order=24,
        guard_order=6,
    )

    ordinary_point = construct_taylor_solution(positions, velocities, initial.masses, order=32)
    entry_time = atlas.evaluation.entry_event_certificate.root
    lifted_entry = spatial_to_ks_binary_chart(
        ordinary_point.positions_at(entry_time),
        ordinary_point.velocities_at(entry_time),
        initial.masses,
        pair=(0, 1),
    )
    ks_point_solution = construct_spatial_ks_binary_taylor_solution(lifted_entry, order=32)
    endpoint_positions, endpoint_velocities = ks_binary_chart_to_spatial(
        ks_point_solution.state_at(atlas.evaluation.exit_event_certificate.root)
    )
    reference = integrate_reference(
        endpoint_positions,
        endpoint_velocities,
        initial.masses,
        ordinary_step,
    )

    assert atlas.evaluation.entry_event_certificate.certified
    assert atlas.evaluation.exit_event_certificate.certified
    assert atlas.evaluation.exit_event_certificate.pre_event_subintervals is not None
    assert atlas.transitions[0].transition_type == "spatial_ordinary_to_ks_decreasing_distance_entry"
    assert atlas.transitions[1].transition_type == (
        "spatial_ks_exit_event_to_ordinary_rho_positive_projection"
    )
    assert all(chart.certified for chart in atlas.charts)
    assert all(transition.certified for transition in atlas.transitions)
    assert atlas.target_state_contains(reference)
    assert not atlas.proof_certified
    assert "spatial_ordinary_ks_entry_event_isolation" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert "spatial_ks_exit_event_isolation" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert atlas.evaluation.collision_policy_certificate.certified
    assert (
        atlas.evaluation.collision_policy_certificate.min_squared_distance_lower_bound
        > 0.0
    )
    assert "spatial_collision_policy_scope" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert atlas.proof_ledger.missing_required_obligations == (
        "finite_time_physical_targeting",
    )


def test_spatial_ordinary_ks_handoff_evaluates_requested_target_time_interval():
    initial = _exact_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    s_start = -0.04
    s_entry = -0.03
    positions, velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_start)
    )
    entry_positions, _entry_velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_entry)
    )
    entry_time_upper = 1.5 * (
        pre_collision_solution.physical_time_at(s_entry)
        - pre_collision_solution.physical_time_at(s_start)
    )
    enter_distance = float(np.linalg.norm(entry_positions[1] - entry_positions[0]))
    exit_rho = 1.0e-1
    seed_atlas = validated_atlas_from_spatial_ordinary_ks_handoff(
        _interval_state_box(positions, velocities, half_width=0.0),
        initial.masses,
        pair=(0, 1),
        enter_distance=enter_distance,
        entry_time_upper=entry_time_upper,
        branch="positive_x",
        exit_rho=exit_rho,
        s_upper=0.5,
        ordinary_step_size=1.0e-12,
        retained_order=24,
        guard_order=6,
    )
    target_time = seed_atlas.evaluation.target_time_interval.upper
    atlas = validated_atlas_from_spatial_ordinary_ks_handoff(
        _interval_state_box(positions, velocities, half_width=0.0),
        initial.masses,
        pair=(0, 1),
        enter_distance=enter_distance,
        entry_time_upper=entry_time_upper,
        branch="positive_x",
        exit_rho=exit_rho,
        s_upper=0.5,
        target_time=target_time,
        retained_order=24,
        guard_order=6,
    )
    ordinary_point = construct_taylor_solution(positions, velocities, initial.masses, order=32)
    entry_time = atlas.evaluation.entry_event_certificate.root
    lifted_entry = spatial_to_ks_binary_chart(
        ordinary_point.positions_at(entry_time),
        ordinary_point.velocities_at(entry_time),
        initial.masses,
        pair=(0, 1),
    )
    ks_point_solution = construct_spatial_ks_binary_taylor_solution(lifted_entry, order=32)
    target_positions, target_velocities = ks_binary_chart_to_spatial(
        ks_point_solution.state_at(atlas.evaluation.ks_evaluation.endpoint_projection.s_value)
    )
    reference = np.concatenate([target_positions.reshape(-1), target_velocities.reshape(-1)])

    assert abs(atlas.evaluation.target_time_interval.lower - target_time) < 1e-18
    assert abs(atlas.evaluation.target_time_interval.upper - target_time) < 1e-18
    assert atlas.evaluation.exit_event_certificate.certified
    assert [chart.chart_type for chart in atlas.charts] == [
        "spatial_ordinary_taylor_before_ks",
        "spatial_ks_binary",
    ]
    assert atlas.target_state_contains(reference)
    assert atlas.evaluation.collision_policy_certificate.certified
    assert (
        atlas.evaluation.collision_policy_certificate.min_squared_distance_lower_bound
        > 0.0
    )
    assert atlas.proof_certified
    assert "finite_time_physical_targeting" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert "spatial_ks_exit_event_isolation" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert "spatial_collision_policy_scope" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert atlas.proof_ledger.missing_required_obligations == ()
    assert atlas.missing_certification_obligations == ()

    missing_ks_target_obligation = replace(
        atlas,
        proof_ledger=replace(
            atlas.proof_ledger,
            entries=tuple(
                entry
                for entry in atlas.proof_ledger.entries
                if entry.name != "spatial_ks_target_inside_regularized_chart"
            ),
        ),
    )
    assert missing_ks_target_obligation.proof_ledger.certified
    assert not missing_ks_target_obligation.proof_ledger_covers_solution
    assert (
        "proof_ledger_coverage:any_of(spatial_ks_target_inside_regularized_chart|ordinary_handoff_admissibility)"
        in missing_ks_target_obligation.missing_certification_obligations
    )
    assert not missing_ks_target_obligation.proof_certified
    missing_physical_targeting = replace(
        atlas,
        proof_ledger=replace(
            atlas.proof_ledger,
            entries=tuple(
                entry
                for entry in atlas.proof_ledger.entries
                if entry.name != "finite_time_physical_targeting"
            ),
        ),
    )
    assert missing_physical_targeting.proof_ledger.certified
    assert not missing_physical_targeting.proof_ledger_covers_solution
    assert any(
        obligation.startswith("proof_ledger_coverage:any_of(target_time|target_containment")
        for obligation in missing_physical_targeting.missing_certification_obligations
    )
    assert not missing_physical_targeting.proof_certified


def test_spatial_ordinary_ks_handoff_stays_in_ks_after_unsafe_ordinary_exit():
    initial = _exact_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    s_start = -0.04
    s_entry = -0.03
    positions, velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_start)
    )
    entry_positions, _entry_velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_entry)
    )
    entry_time_upper = 1.5 * (
        pre_collision_solution.physical_time_at(s_entry)
        - pre_collision_solution.physical_time_at(s_start)
    )
    enter_distance = float(np.linalg.norm(entry_positions[1] - entry_positions[0]))
    exit_rho = 1.0e-1
    s_upper = 0.5
    seed_atlas = validated_atlas_from_spatial_ordinary_ks_handoff(
        _interval_state_box(positions, velocities, half_width=0.0),
        initial.masses,
        pair=(0, 1),
        enter_distance=enter_distance,
        entry_time_upper=entry_time_upper,
        branch="positive_x",
        exit_rho=exit_rho,
        s_upper=s_upper,
        ordinary_step_size=1.0e-12,
        retained_order=24,
        guard_order=6,
    )
    exit_s = seed_atlas.evaluation.exit_event_certificate.root
    target_s = exit_s + 1.0e-4
    ordinary_point = construct_taylor_solution(positions, velocities, initial.masses, order=32)
    entry_time = seed_atlas.evaluation.entry_event_certificate.root
    lifted_entry = spatial_to_ks_binary_chart(
        ordinary_point.positions_at(entry_time),
        ordinary_point.velocities_at(entry_time),
        initial.masses,
        pair=(0, 1),
    )
    ks_point_solution = construct_spatial_ks_binary_taylor_solution(lifted_entry, order=32)
    target_time = entry_time + ks_point_solution.physical_time_at(target_s)
    target_positions, target_velocities = ks_binary_chart_to_spatial(
        ks_point_solution.state_at(target_s)
    )

    atlas = validated_atlas_from_spatial_ordinary_ks_handoff(
        _interval_state_box(positions, velocities, half_width=0.0),
        initial.masses,
        pair=(0, 1),
        enter_distance=enter_distance,
        entry_time_upper=entry_time_upper,
        branch="positive_x",
        exit_rho=exit_rho,
        s_upper=s_upper,
        target_time=target_time,
        retained_order=24,
        guard_order=6,
        ordinary_handoff_min_pair_distance_required=10.0,
    )

    assert target_s > exit_s
    assert [chart.chart_type for chart in atlas.charts] == [
        "spatial_ordinary_taylor_before_ks",
        "spatial_ks_binary",
    ]
    assert atlas.evaluation.ks_evaluation.ordinary_solution is None
    assert atlas.evaluation.ks_evaluation.ordinary_handoff_admissibility is None
    assert atlas.evaluation.exit_event_certificate.certified
    assert atlas.evaluation.ks_evaluation.endpoint_projection.s_value > exit_s
    assert atlas.target_state_contains(
        np.concatenate([target_positions.reshape(-1), target_velocities.reshape(-1)])
    )
    assert "ordinary_handoff_admissibility" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert "spatial_ks_target_inside_regularized_chart" not in (
        atlas.proof_ledger.missing_required_obligations
    )
    assert atlas.proof_certified


def test_spatial_interval_box_ks_atlas_rejects_collision_containing_pair_box():
    masses, positions, velocities = _spatial_state()
    positions = positions.copy()
    positions[1] = positions[0]
    state_interval = _interval_state_box(positions, velocities, half_width=0.1)

    assert (
        spatial_interval_to_ks_binary_chart_state_atlas(
            state_interval,
            masses,
            pair=(0, 1),
        )
        == ()
    )
