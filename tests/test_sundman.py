from dataclasses import replace

import numpy as np
import pytest

from three_body_symmetry.global_invariants import (
    center_of_mass_motion_residual_coefficients,
    certify_interval_center_of_mass_motion,
    centered_angular_momentum_series_coefficients,
    certify_interval_centered_angular_momentum_conservation,
    certify_interval_linear_momentum_conservation,
    certify_interval_total_energy_conservation,
    linear_momentum_series_coefficients,
    total_energy_series_coefficients,
)
from three_body_symmetry.intervals import FloatInterval
from three_body_symmetry.series import acceleration_coefficients, integrate_reference
from three_body_symmetry.tail_bounds import sundman_interval_cauchy_majorant_tail_certificate
from three_body_symmetry.sundman import (
    certify_interval_sundman_equations,
    certify_sundman_physical_time_target,
    continue_interval_sundman_to_s,
    continue_interval_sundman_to_time,
    continue_sundman_to_s,
    construct_interval_sundman_taylor_solution,
    construct_interval_sundman_taylor_solution_from_intervals,
    construct_sundman_taylor_solution,
    continue_sundman_to_time,
    pairwise_distance_product,
    reference_state_at_sundman_chart,
    series_vector_product,
    sundman_factor_coefficients,
    sundman_factor_interval_coefficients,
    sundman_factor_interval_over_s_interval,
)


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


def _rotating_planar_initial_data():
    masses = np.array([1.0, 1.3, 0.8])
    positions = np.array(
        [
            [0.9, -0.2],
            [-0.4, 0.7],
            [0.1, -0.8],
        ]
    )
    total_mass = np.sum(masses)
    center = np.sum(masses[:, None] * positions, axis=0) / total_mass
    centered = positions - center
    velocities = 0.4 * np.column_stack((-centered[:, 1], centered[:, 0]))
    return masses, positions, velocities


def _interval_array_subset(inner, outer):
    inner = np.asarray(inner, dtype=object)
    outer = np.asarray(outer, dtype=object)
    if inner.shape != outer.shape:
        return False
    for index in np.ndindex(inner.shape):
        if outer[index].lower > inner[index].lower or inner[index].upper > outer[index].upper:
            return False
    return True


def _interval_arrays_equal(left, right):
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape:
        return False
    for index in np.ndindex(left.shape):
        if left[index].lower != right[index].lower or left[index].upper != right[index].upper:
            return False
    return True


def test_sundman_chart_matches_newtonian_reference_at_projected_time():
    masses, positions, velocities = _general_initial_data()
    chart = construct_sundman_taylor_solution(positions, velocities, masses, order=18)
    s_value = 0.02

    reference = reference_state_at_sundman_chart(chart, s_value)

    assert chart.physical_time_at_s(s_value) > 0.0
    assert np.linalg.norm(chart.state_at_s(s_value) - reference, ord=np.inf) < 1e-12


def test_sundman_coefficients_satisfy_time_rescaled_equations():
    masses, positions, velocities = _general_initial_data()
    chart = construct_sundman_taylor_solution(positions, velocities, masses, order=12)

    for n in range(11):
        factor = sundman_factor_coefficients(chart.position, n)
        acceleration = acceleration_coefficients(chart.position, masses, n)
        q_rhs = series_vector_product(factor, chart.velocity, n)
        v_rhs = series_vector_product(factor, acceleration, n)

        q_scale = max(1.0, np.linalg.norm(q_rhs[n], ord=np.inf))
        v_scale = max(1.0, np.linalg.norm(v_rhs[n], ord=np.inf))
        t_scale = max(1.0, abs(factor[n]))
        assert np.linalg.norm((n + 1) * chart.position[n + 1] - q_rhs[n], ord=np.inf) < q_scale * 1e-13
        assert np.linalg.norm((n + 1) * chart.velocity[n + 1] - v_rhs[n], ord=np.inf) < v_scale * 1e-13
        assert abs((n + 1) * chart.physical_time[n + 1] - factor[n]) < t_scale * 1e-13


def test_sundman_chart_certifies_centered_angular_momentum_conservation():
    masses, positions, velocities = _general_initial_data()
    point = construct_sundman_taylor_solution(positions, velocities, masses, order=12)
    interval = construct_interval_sundman_taylor_solution(positions, velocities, masses, order=12)

    point_coefficients = centered_angular_momentum_series_coefficients(
        point.position,
        point.velocity,
        masses,
        12,
    )
    certificate = certify_interval_centered_angular_momentum_conservation(
        interval.position,
        interval.velocity,
        masses,
        coefficient_count=12,
    )

    assert np.linalg.norm(point_coefficients[1:], ord=np.inf) < 1e-9
    assert certificate.certified
    assert certificate.coefficient_count == 12
    assert certificate.angular_momentum_coefficients[0][0].lower <= point_coefficients[0, 0]
    assert point_coefficients[0, 0] <= certificate.angular_momentum_coefficients[0][0].upper
    for coefficient in certificate.nonconstant_coefficients:
        for component in coefficient:
            assert component.lower <= 0.0 <= component.upper


def test_sundman_chart_certifies_total_energy_conservation():
    masses, positions, velocities = _general_initial_data()
    point = construct_sundman_taylor_solution(positions, velocities, masses, order=12)
    interval = construct_interval_sundman_taylor_solution(positions, velocities, masses, order=12)

    point_coefficients = total_energy_series_coefficients(
        point.position,
        point.velocity,
        masses,
        12,
    )
    certificate = certify_interval_total_energy_conservation(
        interval.position,
        interval.velocity,
        masses,
        coefficient_count=12,
    )

    assert np.linalg.norm(point_coefficients[1:], ord=np.inf) < 1e-9
    assert certificate.certified
    assert certificate.coefficient_count == 12
    assert certificate.energy_coefficients[0].lower <= point_coefficients[0]
    assert point_coefficients[0] <= certificate.energy_coefficients[0].upper
    for coefficient in certificate.nonconstant_coefficients:
        assert coefficient.lower <= 0.0 <= coefficient.upper


def test_sundman_chart_certifies_linear_momentum_conservation():
    masses, positions, velocities = _general_initial_data()
    point = construct_sundman_taylor_solution(positions, velocities, masses, order=12)
    interval = construct_interval_sundman_taylor_solution(positions, velocities, masses, order=12)

    point_coefficients = linear_momentum_series_coefficients(
        point.velocity,
        masses,
        12,
    )
    certificate = certify_interval_linear_momentum_conservation(
        interval.velocity,
        masses,
        coefficient_count=12,
    )

    assert np.linalg.norm(point_coefficients[1:], ord=np.inf) < 1e-9
    assert certificate.certified
    assert certificate.coefficient_count == 12
    for axis in range(point_coefficients.shape[1]):
        assert certificate.linear_momentum_coefficients[0][axis].lower <= point_coefficients[0, axis]
        assert point_coefficients[0, axis] <= certificate.linear_momentum_coefficients[0][axis].upper
    for coefficient in certificate.nonconstant_coefficients:
        for component in coefficient:
            assert component.lower <= 0.0 <= component.upper


def test_sundman_chart_certifies_center_of_mass_motion():
    masses, positions, velocities = _general_initial_data()
    point = construct_sundman_taylor_solution(positions, velocities, masses, order=12)
    interval = construct_interval_sundman_taylor_solution(positions, velocities, masses, order=12)

    point_residual = center_of_mass_motion_residual_coefficients(
        point.position,
        point.velocity,
        point.physical_time,
        masses,
        12,
    )
    certificate = certify_interval_center_of_mass_motion(
        interval.position,
        interval.velocity,
        interval.physical_time,
        masses,
        coefficient_count=12,
    )

    assert np.linalg.norm(point_residual, ord=np.inf) < 1e-9
    assert certificate.certified
    assert certificate.coefficient_count == 12
    for coefficient in certificate.residual_coefficients:
        for component in coefficient:
            assert component.lower <= 0.0 <= component.upper


def test_sundman_time_slows_near_binary_close_approach():
    masses, positions, velocities = _general_initial_data()
    close_positions = positions.copy()
    close_positions[1] = close_positions[0] + np.array([1e-3, 0.0, 0.0])
    s_value = 0.001

    ordinary = construct_sundman_taylor_solution(positions, velocities, masses, order=8)
    close = construct_sundman_taylor_solution(close_positions, velocities, masses, order=8)

    assert pairwise_distance_product(close_positions) < pairwise_distance_product(positions) * 1e-2
    assert close.physical_time_at_s(s_value) < ordinary.physical_time_at_s(s_value) * 1e-2


def test_sundman_continuation_to_physical_time_matches_reference():
    masses, positions, velocities = _general_initial_data()
    t_final = 0.12

    continued = continue_sundman_to_time(positions, velocities, masses, t_final, order=16, max_s_step=0.02)
    reference = integrate_reference(positions, velocities, masses, t_final)

    assert len(continued.steps) > 1
    assert abs(continued.times[-1] - t_final) < 1e-14
    assert np.linalg.norm(continued.final_state - reference, ord=np.inf) < 1e-11


def test_interval_sundman_coefficients_enclose_point_chart():
    masses, positions, velocities = _general_initial_data()

    point = construct_sundman_taylor_solution(positions, velocities, masses, order=10)
    interval = construct_interval_sundman_taylor_solution(positions, velocities, masses, order=10)

    assert interval.contains_point_solution(point)


def test_interval_sundman_factor_coefficients_enclose_point_factor():
    masses, positions, velocities = _general_initial_data()
    point = construct_sundman_taylor_solution(positions, velocities, masses, order=8)
    interval = construct_interval_sundman_taylor_solution(positions, velocities, masses, order=8)

    point_factor = sundman_factor_coefficients(point.position, 6)
    interval_factor = sundman_factor_interval_coefficients(interval.position, 6)

    for value, enclosure in zip(point_factor, interval_factor):
        assert enclosure.lower <= value <= enclosure.upper


def test_interval_sundman_equation_residual_certificate_contains_zero():
    masses, positions, velocities = _general_initial_data()
    position_intervals = np.empty(positions.shape, dtype=object)
    velocity_intervals = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        position_intervals[index] = FloatInterval(positions[index] - 1e-15, positions[index] + 1e-15)
        velocity_intervals[index] = FloatInterval(velocities[index] - 1e-15, velocities[index] + 1e-15)
    interval = construct_interval_sundman_taylor_solution_from_intervals(
        position_intervals,
        velocity_intervals,
        masses,
        order=8,
    )

    certificate = certify_interval_sundman_equations(interval)

    assert certificate.certified
    assert certificate.coefficient_count == interval.order
    assert certificate.position_residual.shape == interval.position[1:].shape
    assert certificate.velocity_residual.shape == interval.velocity[1:].shape
    assert len(certificate.physical_time_residual) == interval.order
    assert certificate.factor_coefficient_source == "sundman_interval"
    assert certificate.acceleration_coefficient_source == "newtonian_interval"
    assert np.isfinite(certificate.max_residual_radius)


def test_interval_sundman_state_contains_point_evaluation():
    masses, positions, velocities = _general_initial_data()
    point = construct_sundman_taylor_solution(positions, velocities, masses, order=12)
    interval = construct_interval_sundman_taylor_solution(positions, velocities, masses, order=12)
    s_value = 0.015
    point_time = point.physical_time_at_s(s_value)
    interval_time = interval.physical_time_at_s(s_value)

    assert interval.state_contains(point.state_at_s(s_value), s_value)
    assert interval_time.lower <= point_time <= interval_time.upper


def test_interval_sundman_from_interval_initial_data_contains_point_chart():
    masses, positions, velocities = _general_initial_data()
    position_intervals = np.empty(positions.shape, dtype=object)
    velocity_intervals = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        position_intervals[index] = FloatInterval(positions[index] - 1e-15, positions[index] + 1e-15)
        velocity_intervals[index] = FloatInterval(velocities[index] - 1e-15, velocities[index] + 1e-15)

    point = construct_sundman_taylor_solution(positions, velocities, masses, order=8)
    interval = construct_interval_sundman_taylor_solution_from_intervals(
        position_intervals,
        velocity_intervals,
        masses,
        order=8,
    )

    assert interval.contains_point_solution(point)


def test_interval_sundman_continuation_to_s_contains_point_continuation():
    masses, positions, velocities = _general_initial_data()
    s_final = 0.045

    point = continue_sundman_to_s(
        positions,
        velocities,
        masses,
        s_final,
        order=14,
        max_s_step=0.015,
    )
    interval = continue_interval_sundman_to_s(
        positions,
        velocities,
        masses,
        s_final,
        order=14,
        max_s_step=0.015,
    )

    assert len(interval.steps) == len(point.steps) == 3
    assert np.allclose(interval.s_values, np.array([0.0, 0.015, 0.03, 0.045]))
    assert interval.certified
    assert not interval.tail_certified
    assert not interval.proof_certified
    assert interval.equation_residual_certified_step_count == len(interval.steps)
    for point_time, time_interval in zip(point.times, interval.time_intervals):
        assert time_interval.lower <= point_time <= time_interval.upper
    for index, step in enumerate(interval.steps):
        assert step.time_monotone_certified
        assert step.equation_residual_certificate.certified
        assert step.equation_residual_certificate.coefficient_count == step.series.order
        assert step.factor_interval.lower > 0.0
        assert step.start_state_contains(point.states[index])
        assert step.end_state_contains(point.states[index + 1])
    assert interval.final_state_contains(point.final_state)


def test_interval_sundman_continuation_from_interval_data_contains_point_path():
    masses, positions, velocities = _general_initial_data()
    position_intervals = np.empty(positions.shape, dtype=object)
    velocity_intervals = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        position_intervals[index] = FloatInterval(positions[index] - 1e-15, positions[index] + 1e-15)
        velocity_intervals[index] = FloatInterval(velocities[index] - 1e-15, velocities[index] + 1e-15)
    s_final = 0.03

    point = continue_sundman_to_s(
        positions,
        velocities,
        masses,
        s_final,
        order=12,
        max_s_step=0.015,
    )
    interval = continue_interval_sundman_to_s(
        position_intervals,
        velocity_intervals,
        masses,
        s_final,
        order=12,
        max_s_step=0.015,
    )

    assert len(interval.steps) == len(point.steps) == 2
    assert interval.certified
    assert not interval.tail_certified
    assert not interval.proof_certified
    assert all(step.time_monotone_certified for step in interval.steps)
    assert interval.final_state_contains(point.final_state)
    assert interval.final_time_interval.lower <= point.times[-1] <= interval.final_time_interval.upper


def test_interval_sundman_continuation_can_carry_cauchy_tail_certificates():
    masses, positions, velocities = _general_initial_data()
    first_limit = 0.5 * sundman_interval_cauchy_majorant_tail_certificate(
        positions,
        velocities,
        masses,
        retained_order=10,
        step_size=0.0,
    ).s_radius

    continued = continue_interval_sundman_to_s(
        positions,
        velocities,
        masses,
        2.0 * first_limit,
        order=10,
        max_s_step=1.0,
        tail_certificate_mode="cauchy",
    )

    assert continued.steps
    assert continued.certified
    assert continued.tail_certified
    assert continued.chain_certified
    assert continued.proof_certified
    assert abs(continued.steps[0].s_step) == pytest.approx(first_limit)
    assert continued.certified_step_count == len(continued.steps)
    assert all(step.truncation_certificate is not None for step in continued.steps)
    assert all(
        step.truncation_certificate.coefficient_source == "sundman_interval_cauchy_majorant"
        for step in continued.steps
    )
    assert all(step.truncation_certificate.uses_interval_initial_state for step in continued.steps)
    assert all(step.truncation_certificate.is_nontrivial for step in continued.steps)
    assert all(step.truncation_certificate.ratio_bound <= 0.5 + 1e-14 for step in continued.steps)
    for step in continued.steps:
        truncated_end = step.series.state_at_s(step.s_step)
        assert _interval_array_subset(truncated_end, step.end_state_interval)
        truncated_time = step.series.physical_time_at_s(step.s_step)
        assert step.physical_step_interval.lower <= truncated_time.lower
        assert truncated_time.upper <= step.physical_step_interval.upper
    for previous, following in zip(continued.steps, continued.steps[1:]):
        assert _interval_arrays_equal(previous.end_state_interval, following.start_state_interval)
    step_bounds = [step.truncation_certificate.tail_bound for step in continued.steps]
    assert continued.local_tail_bound == sum(step_bounds)
    assert continued.max_step_tail_bound == max(step_bounds)

    broken_s_values = continued.s_values.copy()
    broken_s_values[1] = np.nextafter(broken_s_values[1] + 1e-4, np.inf)
    broken_s = replace(continued, s_values=broken_s_values)
    assert broken_s.certified
    assert broken_s.tail_certified
    assert not broken_s.chain_certified
    assert not broken_s.proof_certified

    broken_state = np.array(continued.state_intervals[1], dtype=object).copy()
    broken_entry = broken_state.flat[0]
    broken_state.flat[0] = FloatInterval(
        float(np.nextafter(broken_entry.lower + 0.1, -np.inf)),
        float(np.nextafter(broken_entry.upper + 0.1, np.inf)),
    )
    broken_states = replace(
        continued,
        state_intervals=(continued.state_intervals[0], broken_state, *continued.state_intervals[2:]),
    )
    assert broken_states.certified
    assert broken_states.tail_certified
    assert not broken_states.chain_certified
    assert not broken_states.proof_certified


def test_interval_sundman_records_triple_collision_exclusion_certificate():
    masses, positions, velocities = _rotating_planar_initial_data()
    point_chart = construct_sundman_taylor_solution(positions, velocities, masses, order=12)
    target_time = point_chart.physical_time_at_s(0.006)

    fixed_s = continue_interval_sundman_to_s(
        positions,
        velocities,
        masses,
        0.01,
        order=10,
        max_s_step=0.01,
    )
    target = continue_interval_sundman_to_time(
        positions,
        velocities,
        masses,
        target_time,
        order=10,
        max_s_step=0.01,
    )

    assert fixed_s.certified
    assert fixed_s.triple_collision_excluded
    assert fixed_s.triple_collision_status == "excluded"
    assert fixed_s.triple_collision_exclusion_reason == "centered angular momentum is bounded away from zero"
    assert not fixed_s.triple_collision_undecided
    assert fixed_s.triple_collision_exclusion_certificate.certified
    assert fixed_s.angular_momentum_certified_step_count == len(fixed_s.steps)
    assert all(step.angular_momentum_certified for step in fixed_s.steps)
    assert fixed_s.energy_certified_step_count == len(fixed_s.steps)
    assert all(step.energy_certified for step in fixed_s.steps)
    assert fixed_s.linear_momentum_certified_step_count == len(fixed_s.steps)
    assert all(step.linear_momentum_certified for step in fixed_s.steps)
    assert fixed_s.center_of_mass_certified_step_count == len(fixed_s.steps)
    assert all(step.center_of_mass_certified for step in fixed_s.steps)
    assert target.certified
    assert target.triple_collision_excluded
    assert target.triple_collision_status == "excluded"
    assert target.triple_collision_exclusion_reason == "centered angular momentum is bounded away from zero"
    assert not target.triple_collision_undecided
    assert target.triple_collision_exclusion_certificate.certified
    assert target.angular_momentum_certified_step_count == len(target.steps) + 1
    assert target.target_angular_momentum_certificate.certified
    assert target.energy_certified_step_count == len(target.steps) + 1
    assert target.target_energy_certificate.certified
    assert target.linear_momentum_certified_step_count == len(target.steps) + 1
    assert target.target_linear_momentum_certificate.certified
    assert target.center_of_mass_certified_step_count == len(target.steps) + 1
    assert target.target_center_of_mass_certificate.certified


def test_interval_sundman_time_target_marks_zero_angular_momentum_branch_undecided():
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    velocities = np.zeros_like(positions)
    target_time = 1e-3

    target = continue_interval_sundman_to_time(
        positions,
        velocities,
        masses,
        target_time,
        order=10,
        max_s_step=0.02,
        target_bisections=36,
        tail_certificate_mode="cauchy",
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert target.proof_certified
    assert not target.triple_collision_excluded
    assert target.triple_collision_status == "undecided"
    assert target.triple_collision_exclusion_reason == "centered angular momentum interval contains zero"
    assert target.triple_collision_undecided
    assert target.target_state_contains(reference)


def test_interval_sundman_time_target_can_carry_cauchy_tail_certificates():
    masses, positions, velocities = _general_initial_data()
    first_limit = 0.5 * sundman_interval_cauchy_majorant_tail_certificate(
        positions,
        velocities,
        masses,
        retained_order=10,
        step_size=0.0,
    ).s_radius
    point_s_target = 1.5 * first_limit
    point = construct_sundman_taylor_solution(positions, velocities, masses, order=16)
    target_time = point.physical_time_at_s(point_s_target)

    interval = continue_interval_sundman_to_time(
        positions,
        velocities,
        masses,
        target_time,
        order=10,
        max_s_step=1.0,
        tail_certificate_mode="cauchy",
    )

    assert interval.certified
    assert interval.tail_certified
    assert interval.chain_certified
    assert interval.proof_certified
    assert len(interval.steps) == 1
    assert interval.steps[0].s_step == pytest.approx(first_limit)
    assert interval.target_s_interval.lower <= point_s_target <= interval.target_s_interval.upper
    assert interval.equation_residual_certified_step_count == len(interval.steps) + 1
    assert all(step.equation_residual_certificate.certified for step in interval.steps)
    assert interval.target_equation_residual_certificate.certified
    assert interval.target_equation_residual_certificate.coefficient_count == 10
    assert _interval_arrays_equal(interval.steps[-1].end_state_interval, interval.target_start_state_interval)
    assert interval.target_state_contains(point.state_at_s(point_s_target))
    assert interval.target_truncation_certificate is not None
    assert interval.certified_step_count == len(interval.steps) + 1
    certificates = [step.truncation_certificate for step in interval.steps] + [
        interval.target_truncation_certificate
    ]
    assert all(certificate is not None for certificate in certificates)
    assert all(
        certificate.coefficient_source == "sundman_interval_cauchy_majorant"
        for certificate in certificates
    )
    assert all(certificate.uses_interval_initial_state for certificate in certificates)
    assert all(certificate.is_nontrivial for certificate in certificates)
    assert all(certificate.ratio_bound <= 0.5 + 1e-12 for certificate in certificates)
    step_bounds = [certificate.tail_bound for certificate in certificates]
    assert interval.local_tail_bound == sum(step_bounds)
    assert interval.max_step_tail_bound == max(step_bounds)

    broken_certificate = replace(
        interval.target_certificate,
        start_time_interval=FloatInterval.point(interval.target_time + 1.0),
    )
    broken_target = replace(interval, target_certificate=broken_certificate)
    assert broken_target.certified
    assert broken_target.tail_certified
    assert not broken_target.chain_certified
    assert not broken_target.proof_certified


def test_interval_sundman_negative_time_target_can_carry_cauchy_tail_certificates():
    masses, positions, velocities = _general_initial_data()
    first_limit = 0.5 * sundman_interval_cauchy_majorant_tail_certificate(
        positions,
        velocities,
        masses,
        retained_order=10,
        step_size=0.0,
    ).s_radius
    point_s_target = -1.5 * first_limit
    point = construct_sundman_taylor_solution(positions, velocities, masses, order=16)
    target_time = point.physical_time_at_s(point_s_target)

    interval = continue_interval_sundman_to_time(
        positions,
        velocities,
        masses,
        target_time,
        order=10,
        max_s_step=1.0,
        tail_certificate_mode="cauchy",
    )

    assert interval.certified
    assert interval.tail_certified
    assert interval.chain_certified
    assert interval.proof_certified
    assert len(interval.steps) == 1
    assert interval.steps[0].s_step == pytest.approx(-first_limit)
    assert interval.steps[0].physical_step_interval.upper < 0.0
    assert interval.target_s_interval.lower <= point_s_target <= interval.target_s_interval.upper
    assert interval.equation_residual_certified_step_count == len(interval.steps) + 1
    assert all(step.equation_residual_certificate.certified for step in interval.steps)
    assert interval.target_equation_residual_certificate.certified
    assert _interval_arrays_equal(interval.steps[-1].end_state_interval, interval.target_start_state_interval)
    assert interval.target_state_contains(point.state_at_s(point_s_target))
    assert interval.target_truncation_certificate is not None
    assert interval.certified_step_count == len(interval.steps) + 1
    certificates = [step.truncation_certificate for step in interval.steps] + [
        interval.target_truncation_certificate
    ]
    assert all(certificate is not None for certificate in certificates)
    assert all(certificate.is_nontrivial for certificate in certificates)
    assert all(certificate.ratio_bound <= 0.5 + 1e-12 for certificate in certificates)
    step_bounds = [certificate.tail_bound for certificate in certificates]
    assert interval.local_tail_bound == sum(step_bounds)
    assert interval.max_step_tail_bound == max(step_bounds)


def test_interval_sundman_time_target_adaptively_shortens_uncertified_crossing_steps():
    masses, positions, velocities = _general_initial_data()
    retained_order = 10
    first_limit = 0.5 * sundman_interval_cauchy_majorant_tail_certificate(
        positions,
        velocities,
        masses,
        retained_order=retained_order,
        step_size=0.0,
    ).s_radius
    point_s_target = 3.0 * first_limit
    point = continue_sundman_to_s(
        positions,
        velocities,
        masses,
        point_s_target,
        order=18,
        max_s_step=0.5 * first_limit,
    )

    with pytest.raises(RuntimeError, match="adaptive shortening"):
        continue_interval_sundman_to_time(
            positions,
            velocities,
            masses,
            point.times[-1],
            order=retained_order,
            max_s_step=1.0,
            tail_certificate_mode="cauchy",
            step_shrink_bisections=0,
        )

    interval = continue_interval_sundman_to_time(
        positions,
        velocities,
        masses,
        point.times[-1],
        order=retained_order,
        max_s_step=1.0,
        tail_certificate_mode="cauchy",
    )

    assert interval.proof_certified
    assert interval.target_state_contains(point.final_state)
    assert interval.target_s_interval.lower <= point_s_target <= interval.target_s_interval.upper
    assert len(interval.steps) >= 3
    assert any(0.0 < abs(step.s_step) < 0.75 * first_limit for step in interval.steps)


def test_sundman_physical_time_target_certificate_brackets_point_root():
    masses, positions, velocities = _general_initial_data()
    point = construct_sundman_taylor_solution(positions, velocities, masses, order=16)
    interval = construct_interval_sundman_taylor_solution(positions, velocities, masses, order=16)
    point_root = 0.02
    target_time = point.physical_time_at_s(point_root)

    certificate = certify_sundman_physical_time_target(
        interval,
        target_time,
        FloatInterval(0.0, 0.04),
        max_bisections=32,
    )

    assert certificate.certified
    assert certificate.s_interval.lower <= point_root <= certificate.s_interval.upper
    assert certificate.factor_interval.lower > 0.0
    assert certificate.time_at_lower.upper <= target_time <= certificate.time_at_upper.lower
    assert interval.state_over_interval_contains(point.state_at_s(point_root), certificate.s_interval)


def test_sundman_physical_time_target_certificate_brackets_negative_point_root():
    masses, positions, velocities = _general_initial_data()
    point = construct_sundman_taylor_solution(positions, velocities, masses, order=16)
    interval = construct_interval_sundman_taylor_solution(positions, velocities, masses, order=16)
    point_root = -0.02
    target_time = point.physical_time_at_s(point_root)

    certificate = certify_sundman_physical_time_target(
        interval,
        target_time,
        FloatInterval(-0.04, 0.0),
        max_bisections=32,
    )

    assert certificate.certified
    assert certificate.s_interval.lower <= point_root <= certificate.s_interval.upper
    assert certificate.factor_interval.lower > 0.0
    assert certificate.time_at_lower.upper <= target_time <= certificate.time_at_upper.lower
    assert interval.state_over_interval_contains(point.state_at_s(point_root), certificate.s_interval)


def test_sundman_physical_time_target_certificate_rejects_uncrossed_target():
    masses, positions, velocities = _general_initial_data()
    interval = construct_interval_sundman_taylor_solution(positions, velocities, masses, order=12)

    certificate = certify_sundman_physical_time_target(
        interval,
        10.0,
        FloatInterval(0.0, 0.01),
    )

    assert not certificate.certified
    assert certificate.factor_interval.lower > 0.0


def test_sundman_factor_interval_over_s_interval_is_positive_on_small_chart():
    masses, positions, velocities = _general_initial_data()
    interval = construct_interval_sundman_taylor_solution(positions, velocities, masses, order=10)

    factor = sundman_factor_interval_over_s_interval(interval, FloatInterval(0.0, 0.02))

    assert factor.lower > 0.0


def test_interval_sundman_continuation_to_physical_time_contains_point_path():
    masses, positions, velocities = _general_initial_data()
    point_s_target = 0.032
    point = continue_sundman_to_s(
        positions,
        velocities,
        masses,
        point_s_target,
        order=14,
        max_s_step=0.015,
    )
    target_time = point.times[-1]

    interval = continue_interval_sundman_to_time(
        positions,
        velocities,
        masses,
        target_time,
        order=14,
        max_s_step=0.015,
    )

    assert interval.certified
    assert not interval.tail_certified
    assert not interval.proof_certified
    assert len(interval.steps) == 2
    assert all(step.time_monotone_certified for step in interval.steps)
    assert interval.target_s_interval.lower <= point_s_target <= interval.target_s_interval.upper
    assert interval.target_state_contains(point.final_state)
    assert (
        interval.target_certificate.global_time_at_lower.upper
        <= target_time
        <= interval.target_certificate.global_time_at_upper.lower
    )


def test_interval_sundman_continuation_to_negative_physical_time_contains_point_path():
    masses, positions, velocities = _general_initial_data()
    point_s_target = -0.032
    point = continue_sundman_to_s(
        positions,
        velocities,
        masses,
        point_s_target,
        order=14,
        max_s_step=0.015,
    )
    target_time = point.times[-1]

    interval = continue_interval_sundman_to_time(
        positions,
        velocities,
        masses,
        target_time,
        order=14,
        max_s_step=0.015,
    )

    assert interval.certified
    assert not interval.tail_certified
    assert not interval.proof_certified
    assert all(step.s_step < 0.0 for step in interval.steps)
    assert all(step.physical_step_interval.upper < 0.0 for step in interval.steps)
    assert all(step.time_monotone_certified for step in interval.steps)
    assert interval.target_s_interval.lower <= point_s_target <= interval.target_s_interval.upper
    assert interval.target_state_contains(point.final_state)
    assert (
        interval.target_certificate.global_time_at_lower.upper
        <= target_time
        <= interval.target_certificate.global_time_at_upper.lower
    )


def test_interval_sundman_continuation_to_time_from_interval_data_contains_point_path():
    masses, positions, velocities = _general_initial_data()
    position_intervals = np.empty(positions.shape, dtype=object)
    velocity_intervals = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        position_intervals[index] = FloatInterval(positions[index] - 1e-15, positions[index] + 1e-15)
        velocity_intervals[index] = FloatInterval(velocities[index] - 1e-15, velocities[index] + 1e-15)
    point_s_target = 0.024
    point = continue_sundman_to_s(
        positions,
        velocities,
        masses,
        point_s_target,
        order=12,
        max_s_step=0.015,
    )

    interval = continue_interval_sundman_to_time(
        position_intervals,
        velocity_intervals,
        masses,
        point.times[-1],
        order=12,
        max_s_step=0.015,
    )

    assert interval.certified
    assert not interval.tail_certified
    assert not interval.proof_certified
    assert all(step.time_monotone_certified for step in interval.steps)
    assert interval.target_s_interval.lower <= point_s_target <= interval.target_s_interval.upper
    assert interval.target_state_contains(point.final_state)


def test_interval_sundman_time_target_requires_reaching_step_budget():
    masses, positions, velocities = _general_initial_data()
    point = continue_sundman_to_s(
        positions,
        velocities,
        masses,
        0.04,
        order=10,
        max_s_step=0.02,
    )

    with pytest.raises(RuntimeError, match="not reached"):
        continue_interval_sundman_to_time(
            positions,
            velocities,
            masses,
            point.times[-1],
            order=10,
            max_s_step=0.01,
            max_steps=1,
        )


def test_interval_sundman_continuation_rejects_uncertified_time_monotonicity():
    masses, positions, velocities = _general_initial_data()

    with pytest.raises(RuntimeError, match="monotonicity"):
        continue_interval_sundman_to_s(
            positions,
            velocities,
            masses,
            0.4,
            order=10,
            max_s_step=0.4,
        )
