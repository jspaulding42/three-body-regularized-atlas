from dataclasses import replace

import numpy as np
import pytest

from three_body_symmetry.compact_dynamics import (
    certify_compactified_dynamics_equations,
    compact_guarded_tail_certificate,
    compact_time_factor_coefficients,
    continue_compactified_solution,
    continue_compactified_solution_to_time,
    construct_compactified_taylor_solution,
    propagate_compact_atlas_error_budget,
    propagate_compact_atlas_interval_enclosure,
    reference_state_at_compact_time,
)
from three_body_symmetry.compact_time import (
    compact_parameter_from_physical_time,
    construct_compactified_time_taylor_solution,
    physical_time_from_compact_parameter,
)
from three_body_symmetry.series import construct_taylor_solution, integrate_reference


def _general_initial_data():
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


def test_compactified_dynamics_coefficients_satisfy_chain_rule_equations():
    masses, positions, velocities = _general_initial_data()
    chart = construct_compactified_taylor_solution(
        positions,
        velocities,
        masses,
        order=12,
        time_rate=1.7,
    )

    certificate = certify_compactified_dynamics_equations(chart)

    assert certificate.certified
    assert certificate.coefficient_count == 12
    assert certificate.max_residual < 1e-12
    assert chart.physical_time[1] == pytest.approx(1.0 / 1.7)
    assert chart.physical_time[2] == 0.0
    assert chart.physical_time[3] == pytest.approx(1.0 / (3.0 * 1.7))


def test_compactified_dynamics_factor_coefficients_are_inverse_time_derivative():
    factor = compact_time_factor_coefficients(8, time_rate=2.0)

    assert np.allclose(factor, np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]))


def test_compactified_dynamics_factor_coefficients_support_nonzero_center():
    center = 0.25
    rate = 1.3
    factor = compact_time_factor_coefficients(8, center=center, time_rate=rate)
    scalar = construct_compactified_time_taylor_solution(center=center, rate=rate, order=9)
    derivative = np.array([(degree + 1) * scalar.coefficients[degree + 1] for degree in range(9)])

    assert np.allclose(factor, derivative)


def test_compactified_dynamics_supports_nonzero_compact_time_center():
    masses, positions, velocities = _general_initial_data()
    center = 0.25
    rate = 1.4
    chart = construct_compactified_taylor_solution(
        positions,
        velocities,
        masses,
        order=16,
        time_rate=rate,
        center=center,
    )
    compact_parameter = center + 0.01

    certificate = certify_compactified_dynamics_equations(chart)
    reference = reference_state_at_compact_time(chart, compact_parameter)
    expected_center_time = physical_time_from_compact_parameter(center, rate=rate)
    expected_target_time = physical_time_from_compact_parameter(compact_parameter, rate=rate)

    assert certificate.certified
    assert chart.center == center
    assert chart.analytic_radius == pytest.approx(0.75)
    assert chart.physical_time_at_u(center) == pytest.approx(expected_center_time, abs=1e-18)
    assert chart.physical_time_delta_at_u(compact_parameter) == pytest.approx(
        expected_target_time - expected_center_time,
        abs=1e-16,
    )
    assert np.linalg.norm(chart.state_at_u(compact_parameter) - reference, ord=np.inf) < 1e-13


def test_compactified_dynamics_matches_ordinary_taylor_under_time_substitution():
    masses, positions, velocities = _general_initial_data()
    compact = construct_compactified_taylor_solution(positions, velocities, masses, order=14, time_rate=1.2)
    ordinary = construct_taylor_solution(positions, velocities, masses, order=18)
    compact_parameter = 0.012
    physical_time = physical_time_from_compact_parameter(compact_parameter, rate=1.2)

    assert compact.physical_time_at_u(compact_parameter) == pytest.approx(physical_time, abs=1e-18)
    assert np.linalg.norm(compact.state_at_u(compact_parameter) - ordinary.state_at(physical_time), ord=np.inf) < 1e-15


def test_compactified_dynamics_matches_reference_integration_at_compact_time():
    masses, positions, velocities = _general_initial_data()
    chart = construct_compactified_taylor_solution(positions, velocities, masses, order=16, time_rate=1.5)
    compact_parameter = 0.018

    reference = reference_state_at_compact_time(chart, compact_parameter)

    assert np.linalg.norm(chart.state_at_u(compact_parameter) - reference, ord=np.inf) < 1e-13


def test_compactified_guarded_tail_certificate_bounds_observed_local_tail():
    masses, positions, velocities = _general_initial_data()
    center = -0.1
    target = -0.065
    retained = construct_compactified_taylor_solution(
        positions,
        velocities,
        masses,
        order=12,
        time_rate=1.2,
        center=center,
    )
    guarded = construct_compactified_taylor_solution(
        positions,
        velocities,
        masses,
        order=18,
        time_rate=1.2,
        center=center,
    )

    certificate = compact_guarded_tail_certificate(guarded, retained_order=12, compact_parameter=target)

    assert certificate.is_nontrivial
    assert certificate.guard_terms == 6
    assert certificate.ratio_bound < 0.1
    assert np.linalg.norm(guarded.state_at_u(target) - retained.state_at_u(target), ord=np.inf) <= certificate.tail_bound


def test_compactified_atlas_continuation_matches_reference_forward():
    masses, positions, velocities = _general_initial_data()
    initial_u = -0.18
    target_u = 0.17
    rate = 1.2

    result = continue_compactified_solution(
        positions,
        velocities,
        masses,
        target_u,
        initial_compact_parameter=initial_u,
        order=16,
        time_rate=rate,
        max_compact_step=0.035,
        radius_fraction=0.2,
    )
    reference = integrate_reference(positions, velocities, masses, result.total_physical_time_delta)

    assert result.certified
    assert len(result.steps) > 1
    assert result.compact_parameters[0] == pytest.approx(initial_u)
    assert result.compact_parameters[-1] == pytest.approx(target_u)
    assert all(step.residual_certificate.certified for step in result.steps)
    assert max(step.residual_certificate.max_residual for step in result.steps) < 1e-8
    assert all(abs(step.compact_step) < step.chart.analytic_radius for step in result.steps)
    assert np.all(np.diff(result.compact_parameters) > 0.0)
    assert np.linalg.norm(result.final_state - reference, ord=np.inf) < 5e-12


def test_compactified_atlas_continuation_carries_local_tail_certificates():
    masses, positions, velocities = _general_initial_data()
    result = continue_compactified_solution(
        positions,
        velocities,
        masses,
        0.17,
        initial_compact_parameter=-0.18,
        order=12,
        guard_order=6,
        time_rate=1.2,
        max_compact_step=0.035,
        radius_fraction=0.2,
    )

    assert result.certified
    assert result.tail_certified
    assert result.chain_certified
    assert result.proof_certified
    for step in result.steps:
        assert step.tail_certificate is not None
        assert step.tail_certificate.is_nontrivial
        assert step.tail_certificate.retained_order == step.chart.order
        assert step.tail_certificate.guard_terms == 6
        guarded = construct_compactified_taylor_solution(
            step.chart.position[0],
            step.chart.velocity[0],
            step.chart.masses,
            order=step.chart.order + 6,
            time_rate=step.chart.time_rate,
            center=step.chart.center,
        )
        local_difference = np.linalg.norm(
            guarded.state_at_u(step.end_compact_parameter) - step.chart.state_at_u(step.end_compact_parameter),
            ord=np.inf,
        )
        assert local_difference <= step.tail_certificate.tail_bound

    broken_compact_parameters = result.compact_parameters.copy()
    broken_compact_parameters[1] = np.nextafter(broken_compact_parameters[1] + 1e-4, np.inf)
    broken_parameters = replace(result, compact_parameters=broken_compact_parameters)
    assert broken_parameters.certified
    assert broken_parameters.tail_certified
    assert not broken_parameters.chain_certified
    assert not broken_parameters.proof_certified

    broken_states = result.states.copy()
    broken_states[1, 0] = np.nextafter(broken_states[1, 0] + 1e-4, np.inf)
    broken_state = replace(result, states=broken_states)
    assert broken_state.certified
    assert broken_state.tail_certified
    assert not broken_state.chain_certified
    assert not broken_state.proof_certified

    broken_rate = replace(result, time_rate=result.time_rate * 1.1)
    assert broken_rate.certified
    assert broken_rate.tail_certified
    assert not broken_rate.chain_certified
    assert not broken_rate.proof_certified


def test_compactified_atlas_propagated_error_budget_covers_endpoint_error():
    masses, positions, velocities = _general_initial_data()
    low_order = continue_compactified_solution(
        positions,
        velocities,
        masses,
        0.17,
        initial_compact_parameter=-0.18,
        order=12,
        guard_order=6,
        time_rate=1.2,
        max_compact_step=0.035,
        radius_fraction=0.2,
    )
    high_order = continue_compactified_solution(
        positions,
        velocities,
        masses,
        0.17,
        initial_compact_parameter=-0.18,
        order=18,
        time_rate=1.2,
        max_compact_step=0.035,
        radius_fraction=0.2,
    )
    reference = integrate_reference(positions, velocities, masses, low_order.total_physical_time_delta)

    budget = low_order.propagated_error_budget()
    direct_budget = propagate_compact_atlas_error_budget(low_order)
    enclosure = low_order.propagated_interval_enclosure()
    direct_enclosure = propagate_compact_atlas_interval_enclosure(low_order)

    assert low_order.proof_certified
    assert len(budget.steps) == len(low_order.steps)
    assert budget.final_bound == direct_budget.final_bound
    assert enclosure.final_radius == budget.final_bound
    assert enclosure.local_tail_bound == budget.local_tail_bound
    assert enclosure.final_state_interval == direct_enclosure.final_state_interval
    assert budget.local_tail_bound == pytest.approx(sum(step.tail_certificate.tail_bound for step in low_order.steps))
    assert budget.final_bound >= budget.local_tail_bound
    assert budget.max_lipschitz_bound >= 1.0
    for index, step in enumerate(enclosure.steps):
        assert step.start_state_contains(low_order.states[index])
        assert step.end_state_contains(low_order.states[index + 1])
    assert enclosure.final_state_contains(low_order.final_state)
    assert enclosure.final_state_contains(high_order.final_state)
    assert enclosure.final_state_contains(reference)
    assert np.linalg.norm(low_order.final_state - high_order.final_state, ord=np.inf) <= budget.final_bound
    assert np.linalg.norm(low_order.final_state - reference, ord=np.inf) <= budget.final_bound


def test_compactified_atlas_targets_positive_physical_time_with_interval_enclosure():
    masses, positions, velocities = _general_initial_data()
    target_time = 0.24
    rate = 1.15

    result = continue_compactified_solution_to_time(
        positions,
        velocities,
        masses,
        target_time,
        order=12,
        guard_order=6,
        time_rate=rate,
        max_compact_step=0.035,
        radius_fraction=0.2,
    )
    reference = integrate_reference(positions, velocities, masses, target_time)
    enclosure = result.propagated_interval_enclosure()

    assert result.proof_certified
    assert result.compact_parameters[0] == pytest.approx(0.0)
    assert result.compact_parameters[-1] == pytest.approx(compact_parameter_from_physical_time(target_time, rate=rate))
    assert result.total_physical_time_delta == pytest.approx(target_time, abs=2e-16)
    assert np.all(np.diff(result.compact_parameters) > 0.0)
    assert enclosure.final_state_contains(result.final_state)
    assert enclosure.final_state_contains(reference)
    assert np.linalg.norm(result.final_state - reference, ord=np.inf) <= result.propagated_error_budget().final_bound


def test_compactified_atlas_targets_negative_physical_time_with_interval_enclosure():
    masses, positions, velocities = _general_initial_data()
    target_time = -0.2
    rate = 1.25

    result = continue_compactified_solution_to_time(
        positions,
        velocities,
        masses,
        target_time,
        order=12,
        guard_order=6,
        time_rate=rate,
        max_compact_step=0.035,
        radius_fraction=0.2,
    )
    reference = integrate_reference(positions, velocities, masses, target_time)
    enclosure = result.propagated_interval_enclosure()

    assert result.proof_certified
    assert result.compact_parameters[0] == pytest.approx(0.0)
    assert result.compact_parameters[-1] == pytest.approx(compact_parameter_from_physical_time(target_time, rate=rate))
    assert result.total_physical_time_delta == pytest.approx(target_time, abs=2e-16)
    assert np.all(np.diff(result.compact_parameters) < 0.0)
    assert enclosure.final_state_contains(result.final_state)
    assert enclosure.final_state_contains(reference)
    assert np.linalg.norm(result.final_state - reference, ord=np.inf) <= result.propagated_error_budget().final_bound


def test_compactified_atlas_continuation_matches_reference_backward():
    masses, positions, velocities = _general_initial_data()
    initial_u = 0.22
    target_u = -0.16
    rate = 1.1

    result = continue_compactified_solution(
        positions,
        velocities,
        masses,
        target_u,
        initial_compact_parameter=initial_u,
        order=16,
        time_rate=rate,
        max_compact_step=0.04,
        radius_fraction=0.2,
    )
    reference = integrate_reference(positions, velocities, masses, result.total_physical_time_delta)

    assert result.certified
    assert len(result.steps) > 1
    assert result.compact_parameters[0] == pytest.approx(initial_u)
    assert result.compact_parameters[-1] == pytest.approx(target_u)
    assert max(step.residual_certificate.max_residual for step in result.steps) < 1e-8
    assert np.all(np.diff(result.compact_parameters) < 0.0)
    assert np.linalg.norm(result.final_state - reference, ord=np.inf) < 8e-12


def test_compactified_atlas_continuation_rejects_invalid_step_controls():
    masses, positions, velocities = _general_initial_data()

    with pytest.raises(ValueError, match="strictly between"):
        continue_compactified_solution(positions, velocities, masses, 1.0)

    with pytest.raises(ValueError, match="positive"):
        continue_compactified_solution(positions, velocities, masses, 0.1, max_compact_step=0.0)

    with pytest.raises(ValueError, match="radius_fraction"):
        continue_compactified_solution(positions, velocities, masses, 0.1, radius_fraction=1.0)

    with pytest.raises(ValueError, match="residual_tolerance"):
        continue_compactified_solution(positions, velocities, masses, 0.1, residual_tolerance=0.0)

    with pytest.raises(ValueError, match="guard_order"):
        continue_compactified_solution(positions, velocities, masses, 0.1, guard_order=-1)

    with pytest.raises(ValueError, match="target_physical_time"):
        continue_compactified_solution_to_time(positions, velocities, masses, np.inf)


def test_compactified_dynamics_rejects_invalid_inputs():
    masses, positions, velocities = _general_initial_data()

    with pytest.raises(ValueError, match="positive finite"):
        construct_compactified_taylor_solution(positions, velocities, masses, order=8, time_rate=0.0)

    with pytest.raises(ValueError, match="strictly between"):
        construct_compactified_taylor_solution(positions, velocities, masses, order=8, center=1.0)

    chart = construct_compactified_taylor_solution(positions, velocities, masses, order=8, center=0.8)
    with pytest.raises(ValueError, match="convergence disk"):
        chart.state_at_u(0.1)

    positions[1] = positions[0]
    with pytest.raises(ValueError, match="collision-free"):
        construct_compactified_taylor_solution(positions, velocities, masses, order=8)
