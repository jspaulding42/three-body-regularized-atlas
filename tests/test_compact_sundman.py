from dataclasses import replace

import numpy as np
import pytest

from three_body_symmetry.compact_sundman import (
    certify_compactified_sundman_accelerated_future_envelope_cross_validation,
    certify_compactified_sundman_accelerated_future_envelope_holdout,
    certify_compactified_sundman_accelerated_global_summability,
    certify_compactified_sundman_accelerated_induction_witness,
    certify_compactified_sundman_future_shell_induction_closure,
    certify_compactified_sundman_scalar_recurrence_induction,
    certify_compactified_sundman_physical_time_target,
    certify_compactified_sundman_compact_domain_cover,
    certify_compactified_sundman_compact_domain_exhaustion_prefix,
    certify_compactified_sundman_conditional_global_summability,
    certify_compactified_sundman_finite_prefix_accelerated_summability,
    certify_compactified_sundman_finite_prefix_envelope_summability,
    certify_compactified_sundman_future_envelope_holdout,
    certify_compactified_sundman_geometric_tail_transfer_global_summability,
    certify_compactified_sundman_tail_transfer_global_summability,
    certify_cauchy_capped_geometric_segment_count_from_state_envelope,
    certify_compactified_sundman_geometric_exhaustion_extension,
    certify_compactified_sundman_geometric_exhaustion_extension_chain,
    certify_compactified_sundman_geometric_exhaustion_schedule,
    certify_compactified_sundman_geometric_schedule_recurrence,
    certify_compactified_sundman_recurrence_template,
    certify_interval_compactified_sundman_equations,
    certify_compactified_sundman_equations,
    compactified_sundman_cauchy_majorant_tail_certificate,
    compactified_sundman_factor_interval_over_w_interval,
    compactified_sundman_guarded_tail_certificate,
    compactified_sundman_interval_cauchy_majorant_tail_certificate,
    continue_compactified_sundman_solution,
    continue_interval_compactified_sundman_solution,
    continue_interval_compactified_sundman_to_time,
    construct_compactified_sundman_centered_exhaustion_prefix,
    construct_compactified_sundman_centered_geometric_exhaustion_prefix,
    extend_compactified_sundman_centered_geometric_exhaustion_chain,
    extend_compactified_sundman_centered_geometric_exhaustion_chain_with_orders,
    extend_compactified_sundman_centered_geometric_exhaustion_prefix,
    construct_interval_compactified_sundman_taylor_solution,
    construct_interval_compactified_sundman_taylor_solution_from_intervals,
    construct_compactified_sundman_taylor_solution,
    construct_sundman_cauchy_radius_state_envelope,
    reference_state_at_compactified_sundman_time,
)
from three_body_symmetry.compact_time import physical_time_from_compact_parameter
from three_body_symmetry.intervals import FloatInterval
from three_body_symmetry.series import integrate_reference
from three_body_symmetry.sundman import construct_sundman_taylor_solution


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


def _interval_box_around(values, radius):
    values = np.asarray(values, dtype=float)
    out = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        out[index] = FloatInterval(
            float(np.nextafter(values[index] - radius, -np.inf)),
            float(np.nextafter(values[index] + radius, np.inf)),
        )
    return out


def test_compactified_sundman_coefficients_satisfy_chain_rule_equations():
    masses, positions, velocities = _general_initial_data()
    chart = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=12,
        sundman_rate=1.4,
    )

    certificate = certify_compactified_sundman_equations(chart)

    assert certificate.certified
    assert certificate.coefficient_count == 12
    assert certificate.max_residual < 1e-12
    assert chart.sundman_time[1] == pytest.approx(1.0 / 1.4)


def test_interval_compactified_sundman_coefficients_enclose_point_chart_and_residuals():
    masses, positions, velocities = _general_initial_data()
    point = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=10,
        sundman_rate=1.3,
        center=0.08,
    )
    interval = construct_interval_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=10,
        sundman_rate=1.3,
        center=0.08,
    )

    certificate = certify_interval_compactified_sundman_equations(interval)
    compact_parameter = 0.095

    assert interval.contains_point_solution(point)
    assert certificate.certified
    assert certificate.coefficient_count == 10
    assert interval.state_contains(point.state_at_w(compact_parameter), compact_parameter)
    physical_time = interval.physical_time_delta_at_w(compact_parameter)
    assert physical_time.lower <= point.physical_time_delta_at_w(compact_parameter) <= physical_time.upper
    sundman_time = interval.sundman_time_at_w(compact_parameter)
    assert sundman_time.lower <= point.sundman_time_at_w(compact_parameter) <= sundman_time.upper


def test_interval_compactified_sundman_accepts_small_initial_boxes():
    masses, positions, velocities = _general_initial_data()
    point = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=8,
        sundman_rate=1.1,
        center=-0.05,
    )
    interval = construct_interval_compactified_sundman_taylor_solution_from_intervals(
        _interval_box_around(positions, 1e-15),
        _interval_box_around(velocities, 1e-15),
        masses,
        order=8,
        sundman_rate=1.1,
        center=-0.05,
    )

    assert interval.contains_point_solution(point)
    assert certify_interval_compactified_sundman_equations(interval).certified
    assert interval.state_contains(point.state_at_w(-0.04), -0.04)


def test_interval_compactified_sundman_time_factor_is_positive_on_compact_interval():
    masses, positions, velocities = _general_initial_data()
    interval = construct_interval_compactified_sundman_taylor_solution_from_intervals(
        _interval_box_around(positions, 1e-15),
        _interval_box_around(velocities, 1e-15),
        masses,
        order=10,
        sundman_rate=1.2,
        center=0.02,
    )

    factor = compactified_sundman_factor_interval_over_w_interval(interval, FloatInterval(0.02, 0.04))

    assert factor.lower > 0.0
    assert factor.upper > factor.lower


def test_compactified_sundman_physical_time_target_certificate_brackets_point_root():
    masses, positions, velocities = _general_initial_data()
    point = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=14,
        sundman_rate=1.2,
    )
    interval = construct_interval_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=14,
        sundman_rate=1.2,
    )
    target_w = 0.024
    target_time = point.physical_time_delta_at_w(target_w)

    certificate = certify_compactified_sundman_physical_time_target(
        interval,
        FloatInterval.point(0.0),
        target_time,
        FloatInterval(0.0, 0.05),
        max_bisections=40,
    )

    assert certificate.certified
    assert certificate.factor_interval.lower > 0.0
    assert certificate.compact_parameter_interval.lower <= target_w <= certificate.compact_parameter_interval.upper
    assert certificate.global_time_at_lower.upper <= target_time <= certificate.global_time_at_upper.lower


def test_compactified_sundman_physical_time_target_certificate_brackets_negative_point_root():
    masses, positions, velocities = _general_initial_data()
    point = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=14,
        sundman_rate=1.2,
    )
    interval = construct_interval_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=14,
        sundman_rate=1.2,
    )
    target_w = -0.022
    target_time = point.physical_time_delta_at_w(target_w)

    certificate = certify_compactified_sundman_physical_time_target(
        interval,
        FloatInterval.point(0.0),
        target_time,
        FloatInterval(-0.05, 0.0),
        max_bisections=40,
    )

    assert certificate.certified
    assert certificate.factor_interval.lower > 0.0
    assert certificate.compact_parameter_interval.lower <= target_w <= certificate.compact_parameter_interval.upper
    assert certificate.global_time_at_lower.upper <= target_time <= certificate.global_time_at_upper.lower


def test_compactified_sundman_matches_sundman_chart_under_substitution():
    masses, positions, velocities = _general_initial_data()
    rate = 1.25
    compact = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=16,
        sundman_rate=rate,
    )
    sundman = construct_sundman_taylor_solution(positions, velocities, masses, order=20)
    compact_parameter = 0.012
    s_value = physical_time_from_compact_parameter(compact_parameter, rate=rate)

    assert compact.sundman_time_at_w(compact_parameter) == pytest.approx(s_value, abs=5e-18)
    assert compact.physical_time_delta_at_w(compact_parameter) == pytest.approx(
        sundman.physical_time_at_s(s_value),
        abs=1e-15,
    )
    assert np.linalg.norm(compact.state_at_w(compact_parameter) - sundman.state_at_s(s_value), ord=np.inf) < 1e-14


def test_compactified_sundman_matches_reference_integration_at_compact_target():
    masses, positions, velocities = _general_initial_data()
    chart = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=16,
        sundman_rate=1.3,
    )
    compact_parameter = 0.018

    reference = reference_state_at_compactified_sundman_time(chart, compact_parameter)

    assert np.linalg.norm(chart.state_at_w(compact_parameter) - reference, ord=np.inf) < 1e-13


def test_compactified_sundman_guarded_tail_certificate_bounds_observed_local_tail():
    masses, positions, velocities = _general_initial_data()
    target = -0.06
    retained = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=12,
        sundman_rate=1.2,
        center=-0.1,
    )
    guarded = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=18,
        sundman_rate=1.2,
        center=-0.1,
    )

    certificate = compactified_sundman_guarded_tail_certificate(
        guarded,
        retained_order=12,
        compact_parameter=target,
    )

    assert certificate.is_nontrivial
    assert certificate.guard_terms == 6
    assert certificate.ratio_bound < 0.2
    local_difference = np.linalg.norm(guarded.state_at_w(target) - retained.state_at_w(target), ord=np.inf)
    assert local_difference <= certificate.tail_bound


def test_compactified_sundman_cauchy_majorant_tail_certificate_bounds_observed_local_tail():
    masses, positions, velocities = _general_initial_data()
    center = -0.1
    target = -0.09
    retained = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=12,
        sundman_rate=1.2,
        center=center,
    )
    higher_order = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=18,
        sundman_rate=1.2,
        center=center,
    )

    certificate = compactified_sundman_cauchy_majorant_tail_certificate(
        retained,
        retained_order=12,
        compact_parameter=target,
    )

    assert certificate.is_nontrivial
    assert certificate.coefficient_source == "compactified_sundman_cauchy_majorant"
    assert certificate.sundman_image_radius <= certificate.sundman_certificate.s_radius
    assert certificate.ratio_bound < 1.0
    local_difference = np.linalg.norm(higher_order.state_at_w(target) - retained.state_at_w(target), ord=np.inf)
    assert local_difference <= certificate.tail_bound


def test_interval_compactified_sundman_cauchy_majorant_bounds_sampled_box_tails():
    masses, positions, velocities = _general_initial_data()
    center = 0.04
    target = 0.05
    retained = construct_interval_compactified_sundman_taylor_solution_from_intervals(
        _interval_box_around(positions, 1e-15),
        _interval_box_around(velocities, 1e-15),
        masses,
        order=12,
        sundman_rate=1.15,
        center=center,
    )

    certificate = compactified_sundman_interval_cauchy_majorant_tail_certificate(
        retained,
        retained_order=12,
        compact_parameter=target,
    )

    assert certificate.is_nontrivial
    assert certificate.coefficient_source == "compactified_sundman_interval_cauchy_majorant"
    assert certificate.uses_interval_initial_state
    assert certificate.sundman_image_radius <= certificate.sundman_certificate.s_radius

    for position_shift, velocity_shift in [
        (0.0, 0.0),
        (3e-16, -2e-16),
        (-4e-16, 5e-16),
    ]:
        sample_positions = positions + position_shift
        sample_velocities = velocities + velocity_shift
        low_order = construct_compactified_sundman_taylor_solution(
            sample_positions,
            sample_velocities,
            masses,
            order=12,
            sundman_rate=1.15,
            center=center,
        )
        high_order = construct_compactified_sundman_taylor_solution(
            sample_positions,
            sample_velocities,
            masses,
            order=18,
            sundman_rate=1.15,
            center=center,
        )
        local_difference = np.linalg.norm(high_order.state_at_w(target) - low_order.state_at_w(target), ord=np.inf)
        assert local_difference <= certificate.tail_bound


def test_compactified_sundman_atlases_can_carry_cauchy_tail_certificates():
    masses, positions, velocities = _general_initial_data()
    target_w = 0.015

    point = continue_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        target_w,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
        tail_certificate_mode="cauchy",
    )
    interval = continue_interval_compactified_sundman_solution(
        _interval_box_around(positions, 1e-15),
        _interval_box_around(velocities, 1e-15),
        masses,
        target_w,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
        tail_certificate_mode="cauchy",
    )

    assert point.proof_certified
    assert point.cauchy_cover_certified
    assert point.cauchy_cover_certificate.step_count == len(point.steps)
    assert point.cauchy_cover_certificate.tail_budget_certified
    assert point.cauchy_cover_certificate.total_tail_bound == pytest.approx(
        sum(step.tail_certificate.tail_bound for step in point.steps)
    )
    assert point.cauchy_cover_certificate.min_retained_order == 10
    assert point.cauchy_cover_certificate.max_retained_order == 10
    assert all(
        step.tail_certificate.coefficient_source == "compactified_sundman_cauchy_majorant"
        for step in point.steps
    )
    assert interval.proof_certified
    assert interval.cauchy_cover_certified
    assert interval.cauchy_cover_certificate.step_count == len(interval.steps)
    assert all(
        step.tail_certificate.coefficient_source == "compactified_sundman_interval_cauchy_majorant"
        for step in interval.steps
    )
    assert interval.final_state_contains(point.final_state)


def test_compactified_sundman_cauchy_mode_caps_oversized_atlas_steps():
    masses, positions, velocities = _general_initial_data()
    point_chart = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=10,
        sundman_rate=1.2,
    )
    radius_certificate = compactified_sundman_cauchy_majorant_tail_certificate(
        point_chart,
        retained_order=10,
        compact_parameter=0.0,
    )
    first_limit = 0.5 * radius_certificate.compact_radius
    target_w = 1.5 * first_limit

    point = continue_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        target_w,
        order=10,
        sundman_rate=1.2,
        max_compact_step=1.0,
        radius_fraction=0.9,
        tail_certificate_mode="cauchy",
    )
    interval = continue_interval_compactified_sundman_solution(
        _interval_box_around(positions, 1e-15),
        _interval_box_around(velocities, 1e-15),
        masses,
        target_w,
        order=10,
        sundman_rate=1.2,
        max_compact_step=1.0,
        radius_fraction=0.9,
        tail_certificate_mode="cauchy",
    )

    assert point.proof_certified
    assert abs(point.steps[0].compact_step) == pytest.approx(first_limit)
    assert point.steps[0].tail_certificate.ratio_bound <= 0.5 + 1e-14
    assert point.cauchy_cover_certified
    assert point.cauchy_cover_certificate.max_step_radius_ratio <= 0.5 + 1e-14
    assert interval.proof_certified
    assert abs(interval.steps[0].compact_step) == pytest.approx(first_limit)
    assert interval.steps[0].tail_certificate.ratio_bound <= 0.5 + 1e-14
    assert interval.cauchy_cover_certified
    assert interval.cauchy_cover_certificate.max_step_radius_ratio <= 0.5 + 1e-14
    assert interval.final_state_contains(point.final_state)


def test_compactified_sundman_cauchy_cover_rejects_guarded_tail_atlases():
    masses, positions, velocities = _general_initial_data()

    guarded = continue_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        0.012,
        order=8,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
        guard_order=4,
    )

    assert guarded.proof_certified
    assert not guarded.cauchy_cover_certified
    assert not guarded.cauchy_cover_certificate.tail_budget_certified
    assert guarded.cauchy_cover_certificate.step_count == len(guarded.steps)


def test_compactified_sundman_cauchy_covers_certify_compact_domain_exhaustion():
    masses, positions, velocities = _general_initial_data()
    left_w = -0.012
    right_w = 0.018

    backward = continue_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        left_w,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
        tail_certificate_mode="cauchy",
    )
    forward = continue_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        right_w,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
        tail_certificate_mode="cauchy",
    )

    domain = certify_compactified_sundman_compact_domain_cover(
        (backward.cauchy_cover_certificate, forward.cauchy_cover_certificate),
        compact_interval=FloatInterval(left_w, right_w),
    )
    incomplete = certify_compactified_sundman_compact_domain_cover(
        (forward.cauchy_cover_certificate,),
        compact_interval=FloatInterval(left_w, right_w),
    )

    assert backward.cauchy_cover_certified
    assert forward.cauchy_cover_certified
    assert domain.certified
    assert domain.cover_count == 2
    assert domain.segment_count == len(backward.steps) + len(forward.steps)
    assert domain.boundary_margin == pytest.approx(1.0 - right_w)
    assert domain.max_uncovered_gap == pytest.approx(0.0)
    assert domain.tail_budget_certified
    assert domain.total_tail_bound == pytest.approx(
        backward.cauchy_cover_certificate.total_tail_bound + forward.cauchy_cover_certificate.total_tail_bound
    )
    assert not incomplete.certified
    assert incomplete.max_uncovered_gap >= abs(left_w)


def test_compactified_sundman_exhaustion_prefix_certifies_nested_compact_domains():
    masses, positions, velocities = _general_initial_data()

    prefix = construct_compactified_sundman_centered_exhaustion_prefix(
        positions,
        velocities,
        masses,
        (0.006, 0.012, 0.018),
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
    )
    reversed_prefix = certify_compactified_sundman_compact_domain_exhaustion_prefix(
        tuple(reversed(prefix.domain_covers))
    )
    duplicate_prefix = certify_compactified_sundman_compact_domain_exhaustion_prefix(
        (prefix.domain_covers[0], prefix.domain_covers[0])
    )
    incomplete_prefix = certify_compactified_sundman_compact_domain_exhaustion_prefix(
        (
            prefix.domain_covers[0],
            certify_compactified_sundman_compact_domain_cover(
                (prefix.domain_covers[-1].covers[1],),
                compact_interval=prefix.domain_covers[-1].compact_interval,
            ),
        )
    )

    assert prefix.certified
    assert prefix.domain_count == 3
    assert prefix.segment_count == sum(domain.segment_count for domain in prefix.domain_covers)
    assert prefix.nested_expanding
    assert prefix.boundary_margins_monotone
    assert prefix.max_uncovered_gap == pytest.approx(0.0)
    assert prefix.remaining_boundary_margin == pytest.approx(1.0 - 0.018)
    assert prefix.max_step_radius_ratio <= 0.5 + 1e-14
    assert prefix.tail_budget_certified
    assert prefix.total_tail_bound == pytest.approx(sum(domain.total_tail_bound for domain in prefix.domain_covers))
    assert len(prefix.incremental_tail_bounds) == prefix.domain_count
    assert prefix.incremental_tail_bounds[0] == pytest.approx(prefix.domain_covers[0].total_tail_bound)
    assert prefix.incremental_total_tail_bound <= prefix.total_tail_bound
    assert not reversed_prefix.certified
    assert not reversed_prefix.nested_expanding
    assert not duplicate_prefix.certified
    assert not duplicate_prefix.nested_expanding
    assert not incomplete_prefix.certified


def test_compactified_sundman_geometric_exhaustion_schedule_certifies_prefix_only():
    masses, positions, velocities = _general_initial_data()

    schedule = construct_compactified_sundman_centered_geometric_exhaustion_prefix(
        positions,
        velocities,
        masses,
        initial_boundary_margin=0.994,
        contraction=0.99,
        prefix_length=3,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
    )
    too_aggressive = certify_compactified_sundman_geometric_exhaustion_schedule(
        schedule.prefix,
        initial_boundary_margin=0.98,
        contraction=0.98,
    )
    invalid_contraction = certify_compactified_sundman_geometric_exhaustion_schedule(
        schedule.prefix,
        initial_boundary_margin=0.994,
        contraction=1.0,
    )

    assert schedule.certified
    assert schedule.prefix_length == 3
    assert schedule.endpoint_limit_certified
    assert schedule.prefix_covers_schedule
    assert schedule.boundary_margins_follow_schedule
    assert schedule.expected_boundary_margins == pytest.approx((0.994, 0.98406, 0.9742194))
    assert schedule.next_expected_boundary_margin == pytest.approx(0.964477206)
    assert schedule.expected_compact_intervals[0].lower == pytest.approx(-0.006)
    assert schedule.expected_compact_intervals[-1].upper == pytest.approx(0.0257806)
    assert schedule.tail_budget_certified
    assert schedule.total_tail_bound == pytest.approx(schedule.prefix.total_tail_bound)
    assert schedule.incremental_total_tail_bound == pytest.approx(schedule.prefix.incremental_total_tail_bound)
    assert not schedule.global_domain_certified
    assert "no induction certificate" in schedule.missing_global_induction_reason
    assert not too_aggressive.certified
    assert not too_aggressive.prefix_covers_schedule
    assert not invalid_contraction.certified
    assert not invalid_contraction.schedule_defined


def test_compactified_sundman_geometric_exhaustion_extension_certifies_next_domain():
    masses, positions, velocities = _general_initial_data()

    schedule = construct_compactified_sundman_centered_geometric_exhaustion_prefix(
        positions,
        velocities,
        masses,
        initial_boundary_margin=0.994,
        contraction=0.99,
        prefix_length=2,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
    )
    extension = extend_compactified_sundman_centered_geometric_exhaustion_prefix(
        schedule,
        positions,
        velocities,
        masses,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
    )
    duplicate_extension = certify_compactified_sundman_geometric_exhaustion_extension(
        schedule,
        schedule.prefix.domain_covers[-1],
    )

    assert schedule.certified
    assert extension.certified
    assert extension.added_index == 2
    assert extension.added_domain.certified
    assert extension.added_domain_matches_schedule
    assert extension.schedule_parameters_preserved
    assert extension.prefix_extended_by_one
    assert extension.expected_added_boundary_margin == pytest.approx(0.9742194)
    assert extension.expected_added_compact_interval.lower == pytest.approx(-0.0257806)
    assert extension.tail_budget_certified
    assert extension.added_tail_bound == pytest.approx(extension.added_domain.total_tail_bound)
    assert extension.added_incremental_tail_bound == pytest.approx(
        extension.added_domain.tail_bound_outside(schedule.prefix.domain_covers[-1].compact_interval)
    )
    assert extension.added_incremental_tail_bound <= extension.added_tail_bound
    assert extension.total_tail_bound == pytest.approx(extension.extended_schedule.total_tail_bound)
    assert extension.incremental_total_tail_bound == pytest.approx(
        extension.extended_schedule.incremental_total_tail_bound
    )
    assert extension.extended_schedule.certified
    assert extension.extended_schedule.prefix_length == 3
    assert extension.extended_schedule.next_expected_boundary_margin == pytest.approx(0.964477206)
    assert not extension.global_domain_certified
    assert "no induction certificate" in extension.missing_global_induction_reason
    assert not duplicate_extension.certified
    assert not duplicate_extension.added_domain_matches_schedule
    assert duplicate_extension.prefix_extended_by_one
    assert not duplicate_extension.extended_schedule.certified


def test_compactified_sundman_geometric_exhaustion_extension_chain_certifies_iterated_extensions():
    masses, positions, velocities = _general_initial_data()

    schedule = construct_compactified_sundman_centered_geometric_exhaustion_prefix(
        positions,
        velocities,
        masses,
        initial_boundary_margin=0.994,
        contraction=0.99,
        prefix_length=1,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
    )
    chain = extend_compactified_sundman_centered_geometric_exhaustion_chain(
        schedule,
        positions,
        velocities,
        masses,
        extension_count=2,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
    )
    disconnected = certify_compactified_sundman_geometric_exhaustion_extension_chain(
        schedule,
        (chain.extensions[0], chain.extensions[0]),
    )

    assert schedule.certified
    assert chain.certified
    assert chain.extension_count == 2
    assert chain.added_indices == (1, 2)
    assert chain.chain_connected
    assert chain.final_schedule.certified
    assert chain.final_schedule.prefix_length == 3
    assert chain.final_schedule.next_expected_boundary_margin == pytest.approx(0.964477206)
    assert chain.tail_budget_certified
    assert chain.added_tail_bounds == pytest.approx(
        tuple(extension.added_tail_bound for extension in chain.extensions)
    )
    assert chain.added_incremental_tail_bounds == pytest.approx(
        tuple(extension.added_incremental_tail_bound for extension in chain.extensions)
    )
    assert chain.added_incremental_segment_counts == (4, 6)
    shell_count_envelope = 2 * (
        np.ceil(max((1.0 - 0.99) * 0.994 / 0.006, (1.0 - 0.99) / (0.2 * 0.99))) + 1
    )
    assert shell_count_envelope == 6
    assert max(chain.added_incremental_segment_counts) <= shell_count_envelope
    assert shell_count_envelope / shell_count_envelope == 1.0
    assert chain.added_max_step_radius_ratios == pytest.approx((0.2312578449794283, 0.23874657382858488))
    assert chain.added_max_state_sup_bounds == pytest.approx((0.8979048047254348, 0.8980701805248219))
    assert chain.added_segment_growth_ratios == pytest.approx((1.5,))
    assert chain.max_added_segment_growth_ratio == pytest.approx(1.5)
    assert chain.added_state_sup_growth_ratios == pytest.approx((1.0001841796574835,))
    assert chain.max_added_state_sup_growth_ratio == pytest.approx(1.0001841796574835)
    assert chain.added_cauchy_tail_denominator_factors == pytest.approx(
        (1.3008262828688506, 1.3136229875894512)
    )
    assert chain.added_cauchy_tail_denominator_growth_ratios == pytest.approx((1.0098373663640765,))
    assert chain.max_cauchy_tail_denominator_growth_ratio == pytest.approx(1.0098373663640765)
    assert chain.uniform_added_retained_orders == (10, 10)
    assert chain.added_retained_order_increments == (0,)
    assert not chain.positive_retained_order_growth_certified
    assert not chain.arithmetic_retained_order_growth_certified
    assert chain.retained_order_increment == 0
    assert np.isinf(chain.cauchy_order_growth_decay_factor_bound)
    assert np.isinf(chain.cauchy_geometry_complexity_decay_factor_bound)
    assert np.isinf(chain.cauchy_majorant_decay_factor_bound)
    assert not chain.finite_cauchy_geometry_decay_candidate_certified
    assert not chain.finite_cauchy_majorant_decay_candidate_certified
    assert chain.total_added_tail_bound == pytest.approx(sum(chain.added_tail_bounds))
    assert chain.total_added_incremental_tail_bound == pytest.approx(sum(chain.added_incremental_tail_bounds))
    assert chain.total_tail_bound == pytest.approx(chain.final_schedule.total_tail_bound)
    assert chain.incremental_total_tail_bound == pytest.approx(chain.final_schedule.incremental_total_tail_bound)
    assert chain.incremental_tail_growth_ratios
    assert chain.max_incremental_tail_growth_ratio > 1.0
    assert not chain.geometric_tail_decay_candidate_certified
    assert not chain.adaptive_order_tail_decay_candidate_certified
    assert np.isinf(chain.geometric_tail_remainder_bound)
    assert np.isinf(chain.adaptive_order_tail_remainder_bound)
    assert not chain.global_domain_certified
    assert "finite chain" in chain.missing_global_induction_reason
    assert not disconnected.certified
    assert not disconnected.chain_connected


def test_compact_atanh_eventual_shell_lower_step_condition_has_uniform_bound():
    initial_boundary_margin = 0.994
    contraction = 0.99
    max_compact_step = 0.006
    alpha = 0.2
    eta = 0.5
    sundman_rate = 1.2

    assert 2.0 * alpha <= eta < 1.0

    threshold_ratio = max_compact_step / (alpha * initial_boundary_margin)
    first_power_index = int(np.ceil(np.log(threshold_ratio) / np.log(contraction)))
    first_shell_index = max(0, first_power_index - 1)

    first_delta_next = initial_boundary_margin * contraction ** (first_shell_index + 1)
    assert alpha * first_delta_next <= max_compact_step
    previous_delta_next = initial_boundary_margin * contraction**first_shell_index
    assert alpha * previous_delta_next > max_compact_step

    uniform_sundman_radius_floor = 2.0 * alpha / (sundman_rate * (1.0 - eta))
    assert uniform_sundman_radius_floor == pytest.approx(2.0 / 3.0)

    for shell_index in (first_shell_index, first_shell_index + 25, first_shell_index + 100):
        delta_next = initial_boundary_margin * contraction ** (shell_index + 1)
        worst_case_center_margin = delta_next
        required_compact_radius = 2.0 * min(max_compact_step, alpha * delta_next)

        assert alpha * delta_next <= max_compact_step
        assert required_compact_radius == pytest.approx(2.0 * alpha * delta_next)
        assert required_compact_radius <= eta * worst_case_center_margin
        required_sundman_radius = required_compact_radius / (
            sundman_rate * (1.0 - eta) * worst_case_center_margin
        )
        assert required_sundman_radius <= uniform_sundman_radius_floor
        assert required_sundman_radius == pytest.approx(uniform_sundman_radius_floor)


def test_sundman_cauchy_radius_state_envelope_lower_bounds_majorant():
    masses, positions, velocities = _general_initial_data()
    envelope = construct_sundman_cauchy_radius_state_envelope(
        positions,
        velocities,
        masses,
    )
    chart = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=8,
        sundman_rate=1.2,
    )
    majorant = compactified_sundman_cauchy_majorant_tail_certificate(
        chart,
        retained_order=8,
        compact_parameter=0.001,
    )
    sundman_majorant = majorant.sundman_certificate

    assert envelope.certified
    assert envelope.position_radius <= sundman_majorant.position_radius
    assert envelope.lower_squared_distance_bound <= sundman_majorant.lower_squared_distance
    assert sundman_majorant.acceleration_bound <= envelope.acceleration_bound * (1.0 + 1.0e-14)
    assert sundman_majorant.sundman_factor_bound <= envelope.sundman_factor_bound * (1.0 + 1.0e-14)
    assert sundman_majorant.velocity_radius <= envelope.velocity_radius_bound * (1.0 + 1.0e-14)
    assert envelope.sundman_radius_lower_bound <= sundman_majorant.s_radius
    assert envelope.certifies_compact_cauchy_radius(
        compact_radius=0.006,
        compact_center=0.0,
        sundman_rate=1.2,
    )


def test_sundman_cauchy_radius_state_envelope_closes_capped_lower_step():
    masses, positions, velocities = _general_initial_data()
    envelope = construct_sundman_cauchy_radius_state_envelope(
        positions,
        velocities,
        masses,
    )
    sundman_rate = 1.2
    eta = 0.5
    alpha_limit = envelope.eventual_shell_alpha_upper_bound(
        sundman_rate=sundman_rate,
        eta=eta,
    )
    alpha = 0.5 * alpha_limit
    initial_boundary_margin = 0.994
    contraction = 0.99

    assert envelope.certifies_margin_only_eventual_lower_step(
        alpha=alpha,
        sundman_rate=sundman_rate,
        eta=eta,
    )
    assert not envelope.certifies_margin_only_eventual_lower_step(
        alpha=float(np.nextafter(alpha_limit, np.inf)),
        sundman_rate=sundman_rate,
        eta=eta,
    )

    for shell_index in (0, 25, 100):
        next_margin = initial_boundary_margin * contraction ** (shell_index + 1)
        center = 1.0 - next_margin
        assert envelope.certifies_cauchy_capped_lower_step(
            compact_center=center,
            next_boundary_margin=next_margin,
            sundman_rate=sundman_rate,
            max_compact_step=1.0,
            alpha=alpha,
        )

    repeated_positions = positions.copy()
    repeated_positions[1] = repeated_positions[0]
    with pytest.raises(ValueError, match="collision-free"):
        construct_sundman_cauchy_radius_state_envelope(
            repeated_positions,
            velocities,
            masses,
        )


def test_state_envelope_closes_cauchy_capped_geometric_segment_count():
    masses, positions, velocities = _general_initial_data()
    envelope = construct_sundman_cauchy_radius_state_envelope(
        positions,
        velocities,
        masses,
    )
    sundman_rate = 1.2
    eta = 0.5
    alpha_limit = envelope.eventual_shell_alpha_upper_bound(
        sundman_rate=sundman_rate,
        eta=eta,
    )
    alpha = 0.5 * alpha_limit
    initial_boundary_margin = 0.994
    contraction = 0.99
    max_compact_step = 1.0
    certificate = certify_cauchy_capped_geometric_segment_count_from_state_envelope(
        envelope,
        initial_boundary_margin=initial_boundary_margin,
        contraction=contraction,
        max_compact_step=max_compact_step,
        alpha=alpha,
        eta=eta,
        sundman_rate=sundman_rate,
    )

    manual_one_sided_bound = np.ceil(
        max(
            (1.0 - contraction) * initial_boundary_margin / max_compact_step,
            (1.0 - contraction) / (alpha * contraction),
        )
    )
    assert certificate.certified
    assert certificate.lower_step_certified
    assert certificate.segment_count_certified
    assert certificate.first_boundary_fraction_shell_index == 0
    assert certificate.required_uniform_sundman_radius_floor == pytest.approx(
        alpha / alpha_limit * envelope.sundman_radius_lower_bound,
    )
    assert certificate.shell_segment_count_bound == int(2 * (manual_one_sided_bound + 1))
    assert certificate.segment_count_growth_ratio_bound == 1.0

    for shell_index in (0, 25, 100):
        next_margin = certificate.next_boundary_margin(shell_index)
        center = 1.0 - next_margin
        assert certificate.lower_step_bound(shell_index) == pytest.approx(
            alpha * next_margin,
        )
        assert certificate.certifies_shell_lower_step(
            shell_index=shell_index,
            compact_center=center,
        )
        assert not certificate.certifies_shell_lower_step(
            shell_index=shell_index,
            compact_center=1.0 - 0.5 * next_margin,
        )

    with pytest.raises(ValueError, match="eventual lower-step"):
        certify_cauchy_capped_geometric_segment_count_from_state_envelope(
            envelope,
            initial_boundary_margin=initial_boundary_margin,
            contraction=contraction,
            max_compact_step=max_compact_step,
            alpha=float(np.nextafter(alpha_limit, np.inf)),
            eta=eta,
            sundman_rate=sundman_rate,
        )


def test_compactified_sundman_adaptive_order_chain_can_certify_finite_tail_decay_candidate():
    masses, positions, velocities = _general_initial_data()

    schedule = construct_compactified_sundman_centered_geometric_exhaustion_prefix(
        positions,
        velocities,
        masses,
        initial_boundary_margin=0.994,
        contraction=0.98,
        prefix_length=1,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.003,
        radius_fraction=0.2,
    )
    chain = extend_compactified_sundman_centered_geometric_exhaustion_chain_with_orders(
        schedule,
        positions,
        velocities,
        masses,
        extension_orders=(12, 14, 16, 18, 20, 22),
        sundman_rate=1.2,
        max_compact_step=0.003,
        radius_fraction=0.2,
    )

    assert schedule.certified
    assert chain.certified
    assert chain.added_min_retained_orders == (12, 14, 16, 18, 20, 22)
    assert chain.added_max_retained_orders == (12, 14, 16, 18, 20, 22)
    assert chain.uniform_added_retained_orders == (12, 14, 16, 18, 20, 22)
    assert chain.added_retained_order_increments == (2, 2, 2, 2, 2)
    assert chain.positive_retained_order_growth_certified
    assert chain.arithmetic_retained_order_growth_certified
    assert chain.retained_order_increment == 2
    assert chain.added_incremental_tail_bounds == pytest.approx(
        (
            7.867186613685335e-12,
            2.415791530303874e-13,
            8.155995305102742e-15,
            4.1069388436210135e-16,
            2.593413491593397e-17,
            1.9771037953579996e-18,
        )
    )
    assert chain.added_incremental_segment_counts == (14, 16, 14, 14, 14, 14)
    assert chain.added_max_step_radius_ratios == pytest.approx(
        (
            0.12039191800270897,
            0.12810378170933012,
            0.13523842380562512,
            0.14268767941362667,
            0.15040183693293374,
            0.15838256783395113,
        )
    )
    assert chain.added_max_state_sup_bounds == pytest.approx(
        (
            0.8980751022534326,
            0.8980281557659211,
            0.9023744251534115,
            0.9525449655441267,
            1.0038344254534057,
            1.0563117231625683,
        )
    )
    assert chain.max_added_step_radius_ratio == pytest.approx(0.15838256783395113)
    assert chain.added_segment_growth_ratios == pytest.approx((16 / 14, 14 / 16, 1.0, 1.0, 1.0))
    assert chain.max_added_segment_growth_ratio == pytest.approx(16 / 14)
    assert chain.added_state_sup_growth_ratios == pytest.approx(
        (
            0.9999477254325461,
            1.0048397918924752,
            1.055598362489258,
            1.053844660110067,
            1.0522768460400826,
        )
    )
    assert chain.max_added_state_sup_growth_ratio == pytest.approx(1.055598362489258)
    assert chain.added_cauchy_tail_denominator_factors == pytest.approx(
        (
            1.1368699543202694,
            1.1469254929909827,
            1.1563881045695619,
            1.1664360536846514,
            1.1770270269770595,
            1.1881883166634584,
        )
    )
    assert chain.added_cauchy_tail_denominator_growth_ratios == pytest.approx(
        (
            1.0088449330835958,
            1.0082504152505167,
            1.0086890803142856,
            1.0090797718905826,
            1.009482611215024,
        )
    )
    assert chain.max_cauchy_tail_denominator_growth_ratio == pytest.approx(1.009482611215024)
    assert chain.cauchy_order_growth_decay_factor_bound == pytest.approx(0.15838256783395113**2)
    assert chain.cauchy_geometry_complexity_decay_factor_bound == pytest.approx(
        (16 / 14) * 0.15838256783395113**2
    )
    assert chain.cauchy_majorant_decay_factor_bound == pytest.approx(
        (16 / 14) * 1.055598362489258 * 1.009482611215024 * 0.15838256783395113**2
    )
    assert chain.finite_prefix_uniform_safety_factor_limit == pytest.approx(2.009088874007347)
    assert chain.geometric_tail_decay_candidate_certified
    assert chain.adaptive_order_tail_decay_candidate_certified
    assert chain.finite_cauchy_geometry_decay_candidate_certified
    assert chain.finite_cauchy_majorant_decay_candidate_certified
    assert chain.max_incremental_tail_growth_ratio < 0.077
    assert chain.geometric_tail_remainder_bound < chain.added_incremental_tail_bounds[-1] * 0.083
    assert chain.adaptive_order_tail_remainder_bound == chain.geometric_tail_remainder_bound
    assert not chain.global_domain_certified

    conditional = certify_compactified_sundman_conditional_global_summability(
        chain,
        max_future_segment_growth_ratio=1.15,
        max_future_state_sup_growth_ratio=1.06,
        max_future_denominator_growth_ratio=1.01,
        max_future_step_radius_ratio=0.16,
    )
    too_tight = certify_compactified_sundman_conditional_global_summability(
        chain,
        max_future_segment_growth_ratio=1.15,
        max_future_state_sup_growth_ratio=1.06,
        max_future_denominator_growth_ratio=1.01,
        max_future_step_radius_ratio=0.15,
    )
    divergent = certify_compactified_sundman_conditional_global_summability(
        chain,
        max_future_segment_growth_ratio=50.0,
        max_future_state_sup_growth_ratio=1.06,
        max_future_denominator_growth_ratio=1.01,
        max_future_step_radius_ratio=0.16,
    )

    assert conditional.uniform_bounds_finite
    assert conditional.observed_prefix_within_uniform_bounds
    assert conditional.conditional_tail_decay_factor_bound == pytest.approx(0.031518464)
    assert conditional.required_retained_order_increment_for_summability == 1
    assert conditional.retained_order_increment_surplus == 1
    assert conditional.retained_order_increment_sufficient
    assert conditional.conditional_global_summability_certified
    uniform_structural_growth = 1.15 * 1.06 * 1.01
    uniform_tail_ratio = uniform_structural_growth * 0.16**chain.retained_order_increment
    uniform_radius_remainder = (
        chain.added_incremental_tail_bounds[-1] * uniform_tail_ratio / (1.0 - uniform_tail_ratio)
    )
    assert uniform_tail_ratio == pytest.approx(conditional.conditional_tail_decay_factor_bound)
    assert uniform_radius_remainder == pytest.approx(
        conditional.conditional_future_tail_remainder_bound
    )
    assert conditional.conditional_future_tail_remainder_bound == pytest.approx(6.434327602736503e-20)
    assert conditional.conditional_total_incremental_tail_bound == pytest.approx(
        chain.incremental_total_tail_bound + conditional.conditional_future_tail_remainder_bound
    )
    assert not conditional.global_domain_certified
    assert "conditional" in conditional.missing_global_induction_reason
    assert not too_tight.observed_prefix_within_uniform_bounds
    assert too_tight.required_retained_order_increment_for_summability == 1
    assert too_tight.retained_order_increment_sufficient
    assert not too_tight.conditional_global_summability_certified
    assert np.isinf(too_tight.conditional_future_tail_remainder_bound)
    assert divergent.observed_prefix_within_uniform_bounds
    assert divergent.conditional_tail_decay_factor_bound > 1.0
    assert divergent.required_retained_order_increment_for_summability == 3
    assert divergent.retained_order_increment_surplus == -1
    assert not divergent.retained_order_increment_sufficient
    assert not divergent.conditional_global_summability_certified

    finite_envelope = certify_compactified_sundman_finite_prefix_envelope_summability(
        chain,
        segment_growth_safety_factor=1.05,
        state_sup_growth_safety_factor=1.05,
        denominator_growth_safety_factor=1.05,
        step_radius_safety_factor=1.05,
    )
    too_loose_finite_envelope = certify_compactified_sundman_finite_prefix_envelope_summability(
        chain,
        segment_growth_safety_factor=2.02,
        state_sup_growth_safety_factor=2.02,
        denominator_growth_safety_factor=2.02,
        step_radius_safety_factor=2.02,
    )

    assert finite_envelope.envelope_source == "finite_prefix"
    assert finite_envelope.observed_prefix_within_uniform_bounds
    assert finite_envelope.future_envelope_growth_factor_bound == pytest.approx(1.4097996371692263)
    assert finite_envelope.max_step_radius_ratio_for_summability == pytest.approx(0.8422117628850202)
    assert finite_envelope.step_radius_summability_slack == pytest.approx(0.6759100666593715)
    assert finite_envelope.cauchy_boundary_step_radius_slack == pytest.approx(0.8336983037743513)
    assert finite_envelope.conditional_tail_decay_factor_bound == pytest.approx(0.038989777090840796)
    assert finite_envelope.required_retained_order_increment_for_summability == 1
    assert finite_envelope.retained_order_increment_surplus == 1
    assert finite_envelope.retained_order_increment_sufficient
    assert finite_envelope.tail_decay_slack == pytest.approx(0.9610102229091592)
    assert finite_envelope.conditional_global_summability_certified
    assert finite_envelope.conditional_future_tail_remainder_bound == pytest.approx(8.021437694295003e-20)
    assert not finite_envelope.global_domain_certified
    assert "finite-prefix-derived envelope" in finite_envelope.missing_global_induction_reason
    assert too_loose_finite_envelope.observed_prefix_within_uniform_bounds
    assert too_loose_finite_envelope.conditional_tail_decay_factor_bound > 1.0
    assert too_loose_finite_envelope.required_retained_order_increment_for_summability > 2
    assert not too_loose_finite_envelope.retained_order_increment_sufficient
    assert too_loose_finite_envelope.tail_decay_slack < 0.0
    assert not too_loose_finite_envelope.conditional_global_summability_certified
    with pytest.raises(ValueError, match="at least 1"):
        certify_compactified_sundman_finite_prefix_envelope_summability(
            chain,
            segment_growth_safety_factor=0.99,
        )


def test_compactified_sundman_deeper_adaptive_prefix_preserves_finite_envelope_margin():
    masses, positions, velocities = _general_initial_data()

    schedule = construct_compactified_sundman_centered_geometric_exhaustion_prefix(
        positions,
        velocities,
        masses,
        initial_boundary_margin=0.994,
        contraction=0.98,
        prefix_length=1,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.003,
        radius_fraction=0.2,
    )
    chain = extend_compactified_sundman_centered_geometric_exhaustion_chain_with_orders(
        schedule,
        positions,
        velocities,
        masses,
        extension_orders=(12, 14, 16, 18, 20, 22, 24, 26),
        sundman_rate=1.2,
        max_compact_step=0.003,
        radius_fraction=0.2,
    )

    assert chain.certified
    assert chain.added_min_retained_orders == (12, 14, 16, 18, 20, 22, 24, 26)
    assert chain.added_retained_order_increments == (2, 2, 2, 2, 2, 2, 2)
    assert chain.added_incremental_segment_counts == (14, 16, 14, 14, 14, 14, 14, 14)
    assert chain.added_incremental_tail_bounds[-2:] == pytest.approx(
        (1.8228680134714622e-19, 2.042653883673186e-20)
    )
    assert chain.max_added_step_radius_ratio == pytest.approx(0.17530001078349408)
    assert chain.max_added_state_sup_growth_ratio == pytest.approx(1.055598362489258)
    assert chain.max_cauchy_tail_denominator_growth_ratio == pytest.approx(1.0104728336231787)
    assert chain.cauchy_majorant_decay_factor_bound == pytest.approx(0.03746098413587541)
    assert chain.cauchy_majorant_transition_factor_bounds == pytest.approx(
        (
            0.018919844476308553,
            0.01621337726721404,
            0.02167848743273517,
            0.02405516724537342,
            0.026646711293221725,
            0.02947998159757684,
            0.03259378382826899,
        )
    )
    assert chain.tail_transfer_growth_factors == pytest.approx(
        (
            1.6230146509170051,
            2.082303477630828,
            2.3228025370089616,
            2.6250957525472924,
            2.8609751620823722,
            3.1275088434351352,
            3.4379913443714156,
        )
    )
    assert chain.max_tail_transfer_growth_factor == pytest.approx(3.4379913443714156)
    assert chain.tail_transfer_growth_ratios == pytest.approx(
        (
            1.2829850158495637,
            1.1154966420417092,
            1.1301415900499185,
            1.08985554500486,
            1.0931618298841732,
            1.0992746996025304,
        )
    )
    assert chain.max_tail_transfer_growth_ratio == pytest.approx(1.2829850158495637)
    assert chain.finite_prefix_uniform_safety_factor_limit == pytest.approx(1.9287865259288886)
    assert chain.finite_cauchy_majorant_decay_candidate_certified

    finite_envelope = certify_compactified_sundman_finite_prefix_envelope_summability(
        chain,
        segment_growth_safety_factor=1.05,
        state_sup_growth_safety_factor=1.05,
        denominator_growth_safety_factor=1.05,
        step_radius_safety_factor=1.05,
    )

    assert finite_envelope.observed_prefix_within_uniform_bounds
    assert finite_envelope.future_envelope_growth_factor_bound == pytest.approx(1.4111825388420483)
    assert finite_envelope.max_step_radius_ratio_for_summability == pytest.approx(0.8417989950458773)
    assert finite_envelope.step_radius_summability_slack == pytest.approx(0.6577339837232086)
    assert finite_envelope.conditional_tail_decay_factor_bound == pytest.approx(0.04781076336572279)
    assert finite_envelope.required_retained_order_increment_for_summability == 1
    assert finite_envelope.retained_order_increment_surplus == 1
    assert finite_envelope.conditional_global_summability_certified
    assert finite_envelope.conditional_future_tail_remainder_bound == pytest.approx(1.0256452994110404e-21)
    assert not finite_envelope.global_domain_certified

    tight_holdout = certify_compactified_sundman_future_envelope_holdout(
        chain,
        training_extension_count=6,
        segment_growth_safety_factor=1.05,
        state_sup_growth_safety_factor=1.05,
        denominator_growth_safety_factor=1.05,
        step_radius_safety_factor=1.05,
    )
    structural_holdout = certify_compactified_sundman_future_envelope_holdout(
        chain,
        training_extension_count=6,
        segment_growth_safety_factor=1.2,
        state_sup_growth_safety_factor=1.2,
        denominator_growth_safety_factor=1.2,
        step_radius_safety_factor=1.2,
    )
    tail_transfer_holdout = certify_compactified_sundman_future_envelope_holdout(
        chain,
        training_extension_count=6,
        segment_growth_safety_factor=1.2,
        state_sup_growth_safety_factor=1.2,
        denominator_growth_safety_factor=1.2,
        step_radius_safety_factor=1.2,
        tail_transfer_safety_factor=1.25,
    )
    predictive_holdout = certify_compactified_sundman_future_envelope_holdout(
        chain,
        training_extension_count=6,
        segment_growth_safety_factor=1.3,
        state_sup_growth_safety_factor=1.3,
        denominator_growth_safety_factor=1.3,
        step_radius_safety_factor=1.3,
    )

    assert tight_holdout.validation_connected
    assert tight_holdout.validation_extension_count == 2
    assert tight_holdout.validation_incremental_tail_bounds == pytest.approx(
        (1.8228680134714622e-19, 2.042653883673186e-20)
    )
    assert tight_holdout.validation_segment_growth_ratios == pytest.approx((1.0, 1.0))
    assert tight_holdout.max_validation_step_radius_ratio == pytest.approx(0.17530001078349408)
    assert tight_holdout.max_validation_tail_growth_ratio == pytest.approx(0.11205714668190181)
    assert tight_holdout.validation_cauchy_majorant_transition_factor_bounds == pytest.approx(
        (0.02947998159757684, 0.03259378382826899)
    )
    assert tight_holdout.validation_tail_transfer_growth_factors == pytest.approx(
        (3.1275088434351352, 3.4379913443714156)
    )
    assert not tight_holdout.validation_diagnostics_within_envelope
    assert not tight_holdout.validation_tail_growth_within_conditional_factor
    assert not tight_holdout.validation_tail_sum_within_conditional_remainder
    assert not tight_holdout.holdout_certified
    assert "diagnostics" in tight_holdout.missing_global_induction_reason

    assert structural_holdout.validation_diagnostics_within_envelope
    assert structural_holdout.conditional_certificate.conditional_tail_decay_factor_bound == pytest.approx(
        0.07601695815509434
    )
    assert structural_holdout.training_tail_transfer_growth_factor_bound == pytest.approx(2.8609751620823722)
    assert not structural_holdout.validation_tail_transfer_within_training_bound
    assert structural_holdout.validation_tail_transfer_growth_ratios == pytest.approx(
        (1.0931618298841732, 1.0992746996025304)
    )
    assert structural_holdout.max_validation_tail_transfer_growth_ratio == pytest.approx(1.0992746996025304)
    assert structural_holdout.training_tail_transfer_growth_ratio_bound == pytest.approx(1.2829850158495637)
    assert structural_holdout.validation_tail_transfer_growth_within_training_ratio_bound
    assert structural_holdout.projected_tail_transfer_growth_bounds == pytest.approx(
        (3.6705882636694604, 4.709309741641185)
    )
    assert structural_holdout.validation_tail_transfer_within_projected_growth_bounds
    assert structural_holdout.projected_tail_transfer_adjusted_validation_ratio_bounds == pytest.approx(
        (0.27902695444394177, 0.3579874015697161)
    )
    assert structural_holdout.projected_validation_tail_bounds == pytest.approx(
        (5.516652506383009e-19, 1.9748920961231148e-19)
    )
    assert structural_holdout.projected_validation_tail_sum_bound == pytest.approx(7.491544602506124e-19)
    assert structural_holdout.validation_tail_growth_within_projected_transfer_bounds
    assert structural_holdout.validation_tail_sum_within_projected_transfer_bounds
    assert structural_holdout.projected_tail_transfer_holdout_certified
    assert not structural_holdout.validation_tail_growth_within_conditional_factor
    assert not structural_holdout.validation_tail_sum_within_conditional_remainder
    assert structural_holdout.tail_transfer_adjusted_conditional_factor_bound == pytest.approx(
        0.21748262917877992
    )
    assert structural_holdout.tail_transfer_adjusted_future_tail_remainder_bound == pytest.approx(
        5.4949033415392385e-19
    )
    assert structural_holdout.validation_tail_growth_within_tail_transfer_adjusted_factor
    assert structural_holdout.validation_tail_sum_within_tail_transfer_adjusted_remainder
    assert not structural_holdout.tail_transfer_adjusted_holdout_certified
    assert not structural_holdout.holdout_certified
    assert "tails exceed" in structural_holdout.missing_global_induction_reason

    assert tail_transfer_holdout.validation_diagnostics_within_envelope
    assert tail_transfer_holdout.validation_tail_transfer_within_training_bound
    assert tail_transfer_holdout.training_tail_transfer_growth_factor_bound == pytest.approx(3.5762189526029653)
    assert tail_transfer_holdout.tail_transfer_adjusted_conditional_factor_bound == pytest.approx(
        0.2718532864734749
    )
    assert tail_transfer_holdout.tail_transfer_adjusted_future_tail_remainder_bound == pytest.approx(
        7.381509172295031e-19
    )
    assert tail_transfer_holdout.validation_tail_growth_within_tail_transfer_adjusted_factor
    assert tail_transfer_holdout.validation_tail_sum_within_tail_transfer_adjusted_remainder
    assert tail_transfer_holdout.tail_transfer_adjusted_holdout_certified
    assert not tail_transfer_holdout.holdout_certified

    tail_transfer_global = certify_compactified_sundman_tail_transfer_global_summability(
        tail_transfer_holdout.conditional_certificate,
        max_future_tail_transfer_growth_factor=tail_transfer_holdout.training_tail_transfer_growth_factor_bound,
        tail_transfer_source="finite_prefix_holdout",
    )
    excessive_tail_transfer_global = certify_compactified_sundman_tail_transfer_global_summability(
        tail_transfer_holdout.conditional_certificate,
        max_future_tail_transfer_growth_factor=20.0,
    )
    geometric_tail_transfer_global = certify_compactified_sundman_geometric_tail_transfer_global_summability(
        tail_transfer_holdout.conditional_certificate,
        initial_tail_transfer_growth_factor_bound=tail_transfer_holdout.training_tail_transfer_growth_factor_bound,
        max_future_tail_transfer_growth_ratio=tail_transfer_holdout.training_tail_transfer_growth_ratio_bound,
        retained_order_increment_growth=1,
        tail_transfer_source="finite_prefix_holdout",
    )
    constant_increment_tail_transfer_global = certify_compactified_sundman_geometric_tail_transfer_global_summability(
        tail_transfer_holdout.conditional_certificate,
        initial_tail_transfer_growth_factor_bound=tail_transfer_holdout.training_tail_transfer_growth_factor_bound,
        max_future_tail_transfer_growth_ratio=tail_transfer_holdout.training_tail_transfer_growth_ratio_bound,
        retained_order_increment_growth=0,
    )

    assert tail_transfer_global.tail_transfer_bound_finite
    assert tail_transfer_global.tail_transfer_source == "finite_prefix_holdout"
    assert tail_transfer_global.tail_transfer_adjusted_tail_decay_factor_bound == pytest.approx(
        0.2718532864734749
    )
    assert tail_transfer_global.tail_transfer_adjusted_future_envelope_growth_factor_bound == pytest.approx(
        7.525880807844004
    )
    assert tail_transfer_global.required_retained_order_increment_for_tail_transfer_summability == 2
    assert tail_transfer_global.retained_order_increment_tail_transfer_surplus == 0
    assert tail_transfer_global.retained_order_increment_tail_transfer_sufficient
    assert tail_transfer_global.max_step_radius_ratio_for_tail_transfer_summability == pytest.approx(
        0.3645199752212293
    )
    assert tail_transfer_global.step_radius_tail_transfer_summability_slack == pytest.approx(
        0.17446089382048796
    )
    assert tail_transfer_global.max_tail_transfer_growth_factor_for_summability == pytest.approx(
        13.154959423129512
    )
    assert tail_transfer_global.tail_transfer_summability_slack == pytest.approx(9.578740470526547)
    assert tail_transfer_global.tail_transfer_adjusted_global_summability_certified
    assert tail_transfer_global.tail_transfer_adjusted_future_tail_remainder_bound == pytest.approx(
        7.381509172295031e-19
    )
    assert not tail_transfer_global.global_domain_certified
    assert "conditional" in tail_transfer_global.missing_global_induction_reason
    assert excessive_tail_transfer_global.tail_transfer_adjusted_tail_decay_factor_bound > 1.0
    assert excessive_tail_transfer_global.tail_transfer_adjusted_future_envelope_growth_factor_bound == pytest.approx(
        42.08847896388593
    )
    assert excessive_tail_transfer_global.required_retained_order_increment_for_tail_transfer_summability == 3
    assert excessive_tail_transfer_global.retained_order_increment_tail_transfer_surplus == -1
    assert not excessive_tail_transfer_global.retained_order_increment_tail_transfer_sufficient
    assert excessive_tail_transfer_global.max_step_radius_ratio_for_tail_transfer_summability == pytest.approx(
        0.1541410753493388
    )
    assert excessive_tail_transfer_global.step_radius_tail_transfer_summability_slack < 0.0
    assert excessive_tail_transfer_global.tail_transfer_summability_slack < 0.0
    assert not excessive_tail_transfer_global.tail_transfer_adjusted_global_summability_certified
    assert np.isinf(excessive_tail_transfer_global.tail_transfer_adjusted_future_tail_remainder_bound)
    assert geometric_tail_transfer_global.tail_transfer_growth_bounds_finite
    assert geometric_tail_transfer_global.tail_transfer_source == "finite_prefix_holdout"
    assert geometric_tail_transfer_global.initial_tail_transfer_adjusted_tail_decay_factor_bound == pytest.approx(
        0.2718532864734749
    )
    assert geometric_tail_transfer_global.tail_transfer_growth_ratio_absorption_factor_bound == pytest.approx(
        0.24384295356328364
    )
    assert geometric_tail_transfer_global.max_future_tail_transfer_growth_ratio_for_order_acceleration == pytest.approx(
        5.261521799589732
    )
    assert geometric_tail_transfer_global.tail_transfer_growth_ratio_slack == pytest.approx(3.978536783740169)
    assert geometric_tail_transfer_global.required_retained_order_increment_growth_for_tail_transfer_ratio == 1
    assert geometric_tail_transfer_global.retained_order_increment_growth_surplus == 0
    assert geometric_tail_transfer_global.retained_order_increment_growth_sufficient
    assert geometric_tail_transfer_global.geometric_tail_transfer_growth_absorbed
    assert geometric_tail_transfer_global.geometric_tail_transfer_global_summability_certified
    assert geometric_tail_transfer_global.geometric_tail_transfer_future_tail_remainder_bound == pytest.approx(
        5.756411320752925e-19
    )
    assert not geometric_tail_transfer_global.global_domain_certified
    assert "conditional" in geometric_tail_transfer_global.missing_global_induction_reason
    assert constant_increment_tail_transfer_global.tail_transfer_growth_ratio_absorption_factor_bound == pytest.approx(
        1.2829850158495637
    )
    assert constant_increment_tail_transfer_global.required_retained_order_increment_growth_for_tail_transfer_ratio == 1
    assert constant_increment_tail_transfer_global.retained_order_increment_growth_surplus == -1
    assert not constant_increment_tail_transfer_global.geometric_tail_transfer_growth_absorbed
    assert not constant_increment_tail_transfer_global.geometric_tail_transfer_global_summability_certified
    assert np.isinf(constant_increment_tail_transfer_global.geometric_tail_transfer_future_tail_remainder_bound)

    assert predictive_holdout.validation_diagnostics_within_envelope
    assert predictive_holdout.validation_tail_growth_within_conditional_factor
    assert predictive_holdout.validation_tail_sum_within_conditional_remainder
    assert predictive_holdout.conditional_certificate.conditional_tail_decay_factor_bound == pytest.approx(
        0.11342819430089159
    )
    assert predictive_holdout.conditional_certificate.conditional_future_tail_remainder_bound == pytest.approx(
        2.5295109996878048e-19
    )
    assert predictive_holdout.holdout_certified
    assert not predictive_holdout.global_domain_certified
    assert "no induction proof" in predictive_holdout.missing_global_induction_reason

    with pytest.raises(ValueError, match="held-out extension"):
        certify_compactified_sundman_future_envelope_holdout(
            chain,
            training_extension_count=chain.extension_count,
        )
    with pytest.raises(ValueError, match="tail_transfer_safety_factor"):
        certify_compactified_sundman_future_envelope_holdout(
            chain,
            training_extension_count=6,
            tail_transfer_safety_factor=0.99,
        )
    with pytest.raises(ValueError, match="tail_transfer_growth_safety_factor"):
        certify_compactified_sundman_future_envelope_holdout(
            chain,
            training_extension_count=6,
            tail_transfer_growth_safety_factor=0.99,
        )


def test_compactified_sundman_accelerated_order_chain_reduces_structural_transition_factors():
    masses, positions, velocities = _general_initial_data()

    schedule = construct_compactified_sundman_centered_geometric_exhaustion_prefix(
        positions,
        velocities,
        masses,
        initial_boundary_margin=0.994,
        contraction=0.98,
        prefix_length=1,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.003,
        radius_fraction=0.2,
    )
    chain = extend_compactified_sundman_centered_geometric_exhaustion_chain_with_orders(
        schedule,
        positions,
        velocities,
        masses,
        extension_orders=(12, 14, 17, 21, 26, 32),
        sundman_rate=1.2,
        max_compact_step=0.003,
        radius_fraction=0.2,
    )

    assert chain.certified
    assert chain.added_min_retained_orders == (12, 14, 17, 21, 26, 32)
    assert chain.added_retained_order_increments == (2, 3, 4, 5, 6)
    assert chain.added_retained_order_increment_growths == (1, 1, 1, 1)
    assert chain.positive_retained_order_growth_certified
    assert not chain.arithmetic_retained_order_growth_certified
    assert chain.retained_order_increment == 0
    assert chain.arithmetic_retained_order_acceleration_certified
    assert chain.retained_order_increment_growth == 1
    assert chain.added_incremental_tail_bounds == pytest.approx(
        (
            7.867186613685335e-12,
            2.415791530303874e-13,
            1.069653150656165e-15,
            1.10453662233113e-18,
            2.620020778462955e-22,
            1.611083174147312e-26,
        )
    )
    geometric_shell_segment_bound = 2 * (
        int(
            np.ceil(
                max(
                    (1.0 - 0.98) * 0.994 / 0.003,
                    (1.0 - 0.98) / (0.2 * 0.98),
                )
            )
        )
        + 1
    )
    assert geometric_shell_segment_bound == 16
    assert max(chain.added_incremental_segment_counts) <= geometric_shell_segment_bound
    cauchy_lower_step_thresholds = []
    cauchy_min_radii_on_added_shells = []
    compact_atanh_lower_bound_margin_ratios = []
    compact_atanh_required_sundman_radii = []
    for extension in chain.extensions:
        previous_interval = extension.previous_schedule.prefix.domain_covers[-1].compact_interval
        threshold = 2.0 * min(0.003, 0.2 * extension.expected_added_boundary_margin)
        shell_radii = []
        for cover in extension.added_domain.covers:
            for start, end, center, radius in zip(
                cover.segment_starts,
                cover.segment_ends,
                cover.disk_centers,
                cover.disk_radii,
            ):
                segment_lower = min(start, end)
                segment_upper = max(start, end)
                contained_in_previous = (
                    previous_interval.lower <= segment_lower
                    and segment_upper <= previous_interval.upper
                )
                if not contained_in_previous:
                    center_margin = 1.0 - abs(center)
                    compact_atanh_lower_bound_margin_ratios.append(threshold / center_margin)
                    compact_atanh_required_sundman_radii.append(
                        threshold / (1.2 * (1.0 - (abs(center) + threshold) ** 2))
                    )
                    shell_radii.append(radius)
        cauchy_lower_step_thresholds.append(threshold)
        cauchy_min_radii_on_added_shells.append(min(shell_radii))
    assert cauchy_lower_step_thresholds == pytest.approx((0.006,) * 6)
    assert min(cauchy_min_radii_on_added_shells) > max(cauchy_lower_step_thresholds)
    assert max(compact_atanh_lower_bound_margin_ratios) < 0.01
    assert max(compact_atanh_required_sundman_radii) < 0.0051
    assert chain.incremental_tail_growth_ratios == pytest.approx(
        (
            0.030707184778119956,
            0.004427754370517298,
            0.0010326119468292748,
            0.0002372054240205624,
            6.149123653486671e-05,
        )
    )
    assert chain.cauchy_majorant_transition_factor_bounds == pytest.approx(
        (
            0.018919844476308553,
            0.0021926715861839805,
            0.0004413691016802925,
            8.18404111174527e-05,
            1.6767686128848877e-05,
        )
    )
    assert chain.max_cauchy_majorant_transition_factor_bound == pytest.approx(0.018919844476308553)
    assert chain.tail_transfer_growth_factors == pytest.approx(
        (
            1.6230146509170051,
            2.0193422482493824,
            2.3395655538598428,
            2.898389936973052,
            3.667246396571723,
        )
    )
    assert chain.tail_transfer_growth_ratios == pytest.approx(
        (
            1.2441922487319825,
            1.1585780250416045,
            1.238858185525622,
            1.2652701935618884,
        )
    )
    radius_drift_factors = tuple(
        (current_radius / previous_radius) ** (previous_order + 1)
        for previous_radius, current_radius, previous_order in zip(
            chain.added_max_step_radius_ratios,
            chain.added_max_step_radius_ratios[1:],
            chain.added_max_retained_orders,
        )
    )
    assert radius_drift_factors == pytest.approx(
        (
            2.2415066086343147,
            2.2546142511998286,
            2.6251537259732824,
            3.184686184883412,
            4.038912652624907,
        )
    )
    assert min(radius_drift_factors) > 2.0
    uniform_radius_tail_envelopes = tuple(
        segment_count
        * state_sup_bound
        * chain.max_added_step_radius_ratio ** (retained_order + 1)
        / (1.0 - chain.max_added_step_radius_ratio)
        for segment_count, state_sup_bound, retained_order in zip(
            chain.added_incremental_segment_counts,
            chain.added_max_state_sup_bounds,
            chain.added_min_retained_orders,
        )
    )
    assert all(
        tail_bound <= envelope
        for tail_bound, envelope in zip(
            chain.added_incremental_tail_bounds,
            uniform_radius_tail_envelopes,
        )
    )
    assert chain.max_tail_transfer_growth_ratio == pytest.approx(1.2652701935618884)
    assert chain.geometric_tail_decay_candidate_certified
    assert not chain.adaptive_order_tail_decay_candidate_certified
    assert chain.accelerated_order_tail_decay_candidate_certified
    assert not chain.finite_cauchy_majorant_decay_candidate_certified
    assert chain.finite_cauchy_majorant_transition_candidate_certified
    assert chain.geometric_tail_remainder_bound == pytest.approx(5.103909566289003e-28)
    assert not chain.global_domain_certified

    accelerated_global = certify_compactified_sundman_finite_prefix_accelerated_summability(
        chain,
        segment_growth_safety_factor=1.05,
        state_sup_growth_safety_factor=1.05,
        denominator_growth_safety_factor=1.05,
        step_radius_safety_factor=1.05,
        tail_transfer_growth_safety_factor=1.05,
    )
    constant_increment_global = certify_compactified_sundman_finite_prefix_accelerated_summability(
        chain,
        segment_growth_safety_factor=1.05,
        state_sup_growth_safety_factor=1.05,
        denominator_growth_safety_factor=1.05,
        step_radius_safety_factor=1.05,
        tail_transfer_growth_safety_factor=1.05,
        retained_order_increment_growth=0,
    )

    assert accelerated_global.uniform_bounds_finite
    assert accelerated_global.future_envelope_growth_factor_bound == pytest.approx(1.4097996371692267)
    assert accelerated_global.projected_next_tail_transfer_growth_factor_bound == pytest.approx(
        4.872060435930914
    )
    assert accelerated_global.observed_prefix_within_accelerated_bounds
    assert accelerated_global.initial_structural_tail_decay_factor_bound == pytest.approx(
        4.959457201990103e-06
    )
    assert accelerated_global.initial_tail_transfer_adjusted_tail_decay_factor_bound == pytest.approx(
        2.4162775217508615e-05
    )
    assert accelerated_global.tail_transfer_growth_ratio_absorption_factor_bound == pytest.approx(
        0.22093740834175174
    )
    assert accelerated_global.required_initial_retained_order_increment_for_summability == 2
    assert accelerated_global.initial_retained_order_increment_surplus == 5
    assert accelerated_global.required_retained_order_increment_growth_for_tail_transfer_ratio == 1
    assert accelerated_global.retained_order_increment_growth_surplus == 0
    assert accelerated_global.accelerated_tail_ratios_shrink
    assert accelerated_global.accelerated_global_summability_certified
    assert accelerated_global.future_tail_ratio_bounds(3) == pytest.approx(
        (2.4162775217508615e-05, 5.33846093490066e-06, 1.1794657234906368e-06)
    )
    assert accelerated_global.future_tail_bounds(3) == pytest.approx(
        (3.8928240593631784e-31, 2.0781689167351736e-36, 2.4511290049128043e-42)
    )
    assert accelerated_global.finite_future_tail_sum_bound(3) == pytest.approx(3.892844841076857e-31)
    assert accelerated_global.accelerated_future_tail_remainder_bound == pytest.approx(
        3.8928448411632886e-31
    )
    accelerated_geometric_majorant = (
        chain.added_incremental_tail_bounds[-1]
        * accelerated_global.initial_tail_transfer_adjusted_tail_decay_factor_bound
        / (1.0 - accelerated_global.initial_tail_transfer_adjusted_tail_decay_factor_bound)
    )
    assert accelerated_global.tail_transfer_growth_ratio_absorption_factor_bound < 1.0
    assert accelerated_global.accelerated_future_tail_remainder_bound <= accelerated_geometric_majorant
    assert not accelerated_global.global_domain_certified
    assert "finite-prefix-derived envelopes" in accelerated_global.missing_global_induction_reason
    missing_induction = certify_compactified_sundman_accelerated_induction_witness(accelerated_global)
    qualitative_only_witness = certify_compactified_sundman_accelerated_induction_witness(
        accelerated_global,
        future_domain_exhaustion_certified=True,
        future_extension_induction_certified=True,
        future_structural_envelopes_certified=True,
        future_tail_transfer_growth_certified=True,
        future_accelerated_order_schedule_certified=True,
    )
    accelerated_witness_bounds = accelerated_global.required_induction_witness_bounds
    accelerated_schedule_witness = accelerated_global.required_domain_schedule_witness
    accelerated_induction_subproofs = {
        "future_segment_growth_bound_certified": True,
        "future_state_sup_growth_bound_certified": True,
        "future_denominator_growth_bound_certified": True,
        "future_step_radius_bound_certified": True,
        "future_initial_tail_transfer_bound_certified": True,
        "future_tail_transfer_growth_ratio_certified": True,
        "future_initial_retained_order_increment_certified": True,
        "future_retained_order_increment_growth_certified": True,
    }
    accelerated_recurrence_subproofs = {
        "segment_growth_recurrence_certified": True,
        "state_sup_growth_recurrence_certified": True,
        "denominator_growth_recurrence_certified": True,
        "step_radius_recurrence_certified": True,
        "initial_tail_transfer_recurrence_certified": True,
        "tail_transfer_growth_ratio_recurrence_certified": True,
        "initial_retained_order_recurrence_certified": True,
        "retained_order_growth_recurrence_certified": True,
    }
    accelerated_scalar_recurrence_bounds = {
        "certified_initial_segment_growth_ratio_bound": (
            accelerated_global.max_future_segment_growth_ratio
        ),
        "certified_recurrent_segment_growth_ratio_bound": (
            accelerated_global.max_future_segment_growth_ratio
        ),
        "certified_initial_state_sup_growth_ratio_bound": (
            accelerated_global.max_future_state_sup_growth_ratio
        ),
        "certified_recurrent_state_sup_growth_ratio_bound": (
            accelerated_global.max_future_state_sup_growth_ratio
        ),
        "certified_initial_denominator_growth_ratio_bound": (
            accelerated_global.max_future_denominator_growth_ratio
        ),
        "certified_recurrent_denominator_growth_ratio_bound": (
            accelerated_global.max_future_denominator_growth_ratio
        ),
        "certified_initial_step_radius_ratio_bound": (
            accelerated_global.max_future_step_radius_ratio
        ),
        "certified_recurrent_step_radius_ratio_bound": (
            accelerated_global.max_future_step_radius_ratio
        ),
        "certified_initial_tail_transfer_growth_factor_bound": (
            accelerated_global.initial_tail_transfer_growth_factor_bound
        ),
        "certified_recurrent_tail_transfer_growth_factor_bound": (
            accelerated_global.initial_tail_transfer_growth_factor_bound
        ),
        "certified_initial_tail_transfer_growth_ratio_bound": (
            accelerated_global.max_future_tail_transfer_growth_ratio
        ),
        "certified_recurrent_tail_transfer_growth_ratio_bound": (
            accelerated_global.max_future_tail_transfer_growth_ratio
        ),
        "certified_initial_retained_order_increment_bound": (
            accelerated_global.initial_retained_order_increment
        ),
        "certified_recurrent_retained_order_increment_bound": (
            accelerated_global.initial_retained_order_increment
        ),
        "certified_initial_retained_order_growth_bound": (
            accelerated_global.retained_order_increment_growth
        ),
        "certified_recurrent_retained_order_growth_bound": (
            accelerated_global.retained_order_increment_growth
        ),
    }
    assert accelerated_witness_bounds == {
        "certified_max_future_segment_growth_ratio": accelerated_global.max_future_segment_growth_ratio,
        "certified_max_future_state_sup_growth_ratio": accelerated_global.max_future_state_sup_growth_ratio,
        "certified_max_future_denominator_growth_ratio": accelerated_global.max_future_denominator_growth_ratio,
        "certified_max_future_step_radius_ratio": accelerated_global.max_future_step_radius_ratio,
        "certified_initial_tail_transfer_growth_factor_bound": (
            accelerated_global.initial_tail_transfer_growth_factor_bound
        ),
        "certified_max_future_tail_transfer_growth_ratio": accelerated_global.max_future_tail_transfer_growth_ratio,
        "certified_initial_retained_order_increment": accelerated_global.initial_retained_order_increment,
        "certified_retained_order_increment_growth": accelerated_global.retained_order_increment_growth,
    }
    assert accelerated_schedule_witness == {
        "certified_schedule_initial_boundary_margin": (
            accelerated_global.chain.final_schedule.initial_boundary_margin
        ),
        "certified_schedule_contraction": accelerated_global.chain.final_schedule.contraction,
        "certified_extension_induction_start_index": accelerated_global.chain.final_schedule.prefix_length,
    }
    schedule_recurrence = certify_compactified_sundman_geometric_schedule_recurrence(
        accelerated_global.chain.final_schedule,
        one_step_extension_template_certified=True,
        witness_source="schedule_recurrence_test_witness",
        **accelerated_schedule_witness,
    )
    weak_schedule_recurrence = certify_compactified_sundman_geometric_schedule_recurrence(
        accelerated_global.chain.final_schedule,
        one_step_extension_template_certified=False,
        witness_source="weak_schedule_recurrence_test_witness",
        **{
            **accelerated_schedule_witness,
            "certified_extension_induction_start_index": (
                accelerated_global.chain.final_schedule.prefix_length + 1
            ),
        },
    )
    recurrence_template = certify_compactified_sundman_recurrence_template(
        accelerated_global,
        structural_envelope_form_certified=True,
        structural_transition_template_certified=True,
        tail_transfer_form_certified=True,
        tail_transfer_growth_template_certified=True,
        retained_order_arithmetic_schedule_certified=True,
        retained_order_growth_template_certified=True,
        witness_source="recurrence_template_test_witness",
    )
    weak_recurrence_template = certify_compactified_sundman_recurrence_template(
        accelerated_global,
        structural_envelope_form_certified=True,
        structural_transition_template_certified=True,
        tail_transfer_form_certified=True,
        tail_transfer_growth_template_certified=False,
        retained_order_arithmetic_schedule_certified=True,
        retained_order_growth_template_certified=False,
        witness_source="weak_recurrence_template_test_witness",
    )
    unscheduled_quantitative_witness = certify_compactified_sundman_accelerated_induction_witness(
        accelerated_global,
        future_domain_exhaustion_certified=True,
        future_extension_induction_certified=True,
        future_structural_envelopes_certified=True,
        future_tail_transfer_growth_certified=True,
        future_accelerated_order_schedule_certified=True,
        **accelerated_witness_bounds,
    )
    too_late_schedule_witness = certify_compactified_sundman_accelerated_induction_witness(
        accelerated_global,
        future_domain_exhaustion_certified=True,
        future_extension_induction_certified=True,
        future_structural_envelopes_certified=True,
        future_tail_transfer_growth_certified=True,
        future_accelerated_order_schedule_certified=True,
        **{
            **accelerated_witness_bounds,
            **accelerated_schedule_witness,
            "certified_extension_induction_start_index": (
                accelerated_global.chain.final_schedule.prefix_length + 1
            ),
        },
    )
    unfactored_structural_witness = certify_compactified_sundman_accelerated_induction_witness(
        accelerated_global,
        future_domain_exhaustion_certified=True,
        future_extension_induction_certified=True,
        future_structural_envelopes_certified=True,
        future_tail_transfer_growth_certified=True,
        future_accelerated_order_schedule_certified=True,
        **accelerated_schedule_witness,
        **accelerated_witness_bounds,
    )
    too_weak_quantitative_witness = certify_compactified_sundman_accelerated_induction_witness(
        accelerated_global,
        future_domain_exhaustion_certified=True,
        future_extension_induction_certified=True,
        future_structural_envelopes_certified=True,
        future_tail_transfer_growth_certified=True,
        future_accelerated_order_schedule_certified=True,
        **{
            **accelerated_witness_bounds,
            **accelerated_schedule_witness,
            **accelerated_induction_subproofs,
            "certified_max_future_step_radius_ratio": (
                1.01 * accelerated_global.max_future_step_radius_ratio
            ),
        },
    )
    summability_witness = certify_compactified_sundman_accelerated_induction_witness(
        accelerated_global,
        future_domain_exhaustion_certified=True,
        future_extension_induction_certified=True,
        future_structural_envelopes_certified=True,
        future_tail_transfer_growth_certified=True,
        future_accelerated_order_schedule_certified=True,
        **accelerated_schedule_witness,
        **accelerated_witness_bounds,
        **accelerated_induction_subproofs,
    )
    opaque_collision_witness = certify_compactified_sundman_accelerated_induction_witness(
        accelerated_global,
        future_domain_exhaustion_certified=True,
        future_extension_induction_certified=True,
        future_structural_envelopes_certified=True,
        future_tail_transfer_growth_certified=True,
        future_accelerated_order_schedule_certified=True,
        collision_continuation_certified=True,
        **accelerated_schedule_witness,
        **accelerated_witness_bounds,
        **accelerated_induction_subproofs,
    )
    binary_only_collision_witness = certify_compactified_sundman_accelerated_induction_witness(
        accelerated_global,
        future_domain_exhaustion_certified=True,
        future_extension_induction_certified=True,
        future_structural_envelopes_certified=True,
        future_tail_transfer_growth_certified=True,
        future_accelerated_order_schedule_certified=True,
        collision_continuation_certified=True,
        binary_collision_continuation_certified=True,
        **accelerated_schedule_witness,
        **accelerated_witness_bounds,
        **accelerated_induction_subproofs,
    )
    complete_abstract_witness = certify_compactified_sundman_accelerated_induction_witness(
        accelerated_global,
        future_domain_exhaustion_certified=True,
        future_extension_induction_certified=True,
        future_structural_envelopes_certified=True,
        future_tail_transfer_growth_certified=True,
        future_accelerated_order_schedule_certified=True,
        collision_continuation_certified=True,
        binary_collision_continuation_certified=True,
        triple_collision_continuation_certified=True,
        witness_source="abstract_test_witness",
        **accelerated_schedule_witness,
        **accelerated_witness_bounds,
        **accelerated_induction_subproofs,
    )
    recurrence_closure = certify_compactified_sundman_future_shell_induction_closure(
        accelerated_global,
        witness_source="abstract_recurrence_test_witness",
        **schedule_recurrence.recurrence_closure_kwargs,
        **recurrence_template.recurrence_closure_kwargs,
        **accelerated_witness_bounds,
        **accelerated_recurrence_subproofs,
    )
    weak_recurrence_closure = certify_compactified_sundman_future_shell_induction_closure(
        accelerated_global,
        domain_exhaustion_recurrence_certified=True,
        scheduled_extension_recurrence_certified=True,
        structural_envelope_recurrence_certified=True,
        tail_transfer_recurrence_certified=True,
        retained_order_schedule_recurrence_certified=True,
        **{
            **accelerated_schedule_witness,
            **accelerated_witness_bounds,
            **accelerated_recurrence_subproofs,
            "denominator_growth_recurrence_certified": False,
            "certified_max_future_step_radius_ratio": (
                1.01 * accelerated_global.max_future_step_radius_ratio
            ),
        },
    )
    weak_schedule_closure = certify_compactified_sundman_future_shell_induction_closure(
        accelerated_global,
        **weak_schedule_recurrence.recurrence_closure_kwargs,
        **recurrence_template.recurrence_closure_kwargs,
        **accelerated_witness_bounds,
        **accelerated_recurrence_subproofs,
    )
    weak_template_closure = certify_compactified_sundman_future_shell_induction_closure(
        accelerated_global,
        **schedule_recurrence.recurrence_closure_kwargs,
        **weak_recurrence_template.recurrence_closure_kwargs,
        **accelerated_witness_bounds,
        **accelerated_recurrence_subproofs,
    )
    scalar_recurrence = certify_compactified_sundman_scalar_recurrence_induction(
        accelerated_global,
        witness_source="scalar_recurrence_test_witness",
        **schedule_recurrence.recurrence_closure_kwargs,
        **recurrence_template.scalar_recurrence_kwargs,
        **accelerated_scalar_recurrence_bounds,
    )
    weak_scalar_recurrence = certify_compactified_sundman_scalar_recurrence_induction(
        accelerated_global,
        **{
            **schedule_recurrence.recurrence_closure_kwargs,
            **recurrence_template.scalar_recurrence_kwargs,
            **accelerated_scalar_recurrence_bounds,
            "certified_recurrent_step_radius_ratio_bound": (
                1.01 * accelerated_global.max_future_step_radius_ratio
            ),
            "certified_recurrent_retained_order_growth_bound": (
                accelerated_global.retained_order_increment_growth - 1
            ),
        },
    )
    weak_template_scalar_recurrence = certify_compactified_sundman_scalar_recurrence_induction(
        accelerated_global,
        **schedule_recurrence.recurrence_closure_kwargs,
        **weak_recurrence_template.scalar_recurrence_kwargs,
        **accelerated_scalar_recurrence_bounds,
    )

    assert not missing_induction.future_shell_induction_certified
    assert not missing_induction.global_domain_certified
    assert np.isinf(missing_induction.global_tail_bound)
    assert "compact-Sundman domain" in missing_induction.missing_global_induction_reason
    assert tuple(
        detail.obligation for detail in missing_induction.global_proof_obligation_details
    ) == tuple(
        obligation for obligation, _certified in missing_induction.global_proof_obligation_statuses
    )
    assert "future_domain_exhaustion" in missing_induction.missing_global_proof_obligations
    assert "quantitative_witness_present" in missing_induction.missing_global_proof_obligations
    assert "binary_collision_continuation" in missing_induction.missing_global_proof_obligations
    assert qualitative_only_witness.quantitative_witness_present is False
    assert not qualitative_only_witness.future_shell_induction_certified
    assert "geometric domain schedule witness" in qualitative_only_witness.missing_global_induction_reason
    assert unscheduled_quantitative_witness.quantitative_witness_present
    assert not unscheduled_quantitative_witness.domain_schedule_witness_present
    assert not unscheduled_quantitative_witness.global_domain_certified
    assert "geometric domain schedule witness" in unscheduled_quantitative_witness.missing_global_induction_reason
    assert "geometric_domain_schedule_witness_present" in (
        unscheduled_quantitative_witness.missing_global_proof_obligations
    )
    assert "quantitative_witness_present" not in unscheduled_quantitative_witness.missing_global_proof_obligations
    unscheduled_details = {
        detail.obligation: detail
        for detail in unscheduled_quantitative_witness.global_proof_obligation_details
    }
    assert "certified_schedule_initial_boundary_margin" in (
        unscheduled_details["geometric_domain_schedule_witness_present"].observed
    )
    assert too_late_schedule_witness.domain_schedule_witness_finite
    assert not too_late_schedule_witness.domain_schedule_witness_matches_accelerated_chain
    assert not too_late_schedule_witness.global_domain_certified
    assert "checked induction base" in too_late_schedule_witness.missing_global_induction_reason
    assert too_late_schedule_witness.missing_global_proof_obligations == (
        "geometric_domain_schedule_matches_checked_prefix",
        "future_segment_count_growth_bound",
        "future_state_supremum_growth_bound",
        "future_cauchy_denominator_growth_bound",
        "future_step_radius_ratio_bound",
        "first_future_tail_transfer_bound",
        "future_tail_transfer_growth_ratio",
        "first_future_retained_order_increment",
        "future_retained_order_increment_growth",
        "collision_continuation",
        "binary_collision_continuation",
        "triple_collision_continuation",
    )
    too_late_details = {
        detail.obligation: detail for detail in too_late_schedule_witness.global_proof_obligation_details
    }
    too_late_schedule_detail = too_late_details[
        "geometric_domain_schedule_matches_checked_prefix"
    ]
    assert too_late_schedule_detail.witness_field == "certified_extension_induction_start_index"
    assert too_late_schedule_detail.comparison == "<="
    assert too_late_schedule_detail.required == accelerated_global.chain.final_schedule.prefix_length
    assert too_late_schedule_detail.observed == accelerated_global.chain.final_schedule.prefix_length + 1
    assert schedule_recurrence.required_domain_schedule_witness == accelerated_schedule_witness
    assert schedule_recurrence.certified_domain_schedule_witness == accelerated_schedule_witness
    assert schedule_recurrence.recurrence_closure_kwargs == {
        "domain_exhaustion_recurrence_certified": True,
        "scheduled_extension_recurrence_certified": True,
        **accelerated_schedule_witness,
    }
    assert schedule_recurrence.domain_exhaustion_recurrence_certified
    assert schedule_recurrence.scheduled_extension_recurrence_certified
    assert schedule_recurrence.schedule_recurrence_certified
    assert schedule_recurrence.missing_schedule_recurrence_obligations == ()
    assert schedule_recurrence.witness_source == "schedule_recurrence_test_witness"
    assert tuple(
        detail.obligation for detail in schedule_recurrence.schedule_recurrence_details
    ) == tuple(
        obligation
        for obligation, _certified in schedule_recurrence.schedule_recurrence_obligation_statuses
    )
    assert all(detail.certified for detail in schedule_recurrence.schedule_recurrence_details)
    assert weak_schedule_recurrence.schedule_witness_finite
    assert not weak_schedule_recurrence.domain_exhaustion_recurrence_certified
    assert not weak_schedule_recurrence.scheduled_extension_recurrence_certified
    assert not weak_schedule_recurrence.schedule_recurrence_certified
    assert weak_schedule_recurrence.missing_schedule_recurrence_obligations == (
        "geometric_schedule_induction_start_index",
        "geometric_schedule_one_step_extension_template",
    )
    weak_schedule_details = {
        detail.obligation: detail
        for detail in weak_schedule_recurrence.schedule_recurrence_details
    }
    assert weak_schedule_details["geometric_schedule_induction_start_index"].comparison == "<="
    assert weak_schedule_details["geometric_schedule_induction_start_index"].observed == (
        accelerated_global.chain.final_schedule.prefix_length + 1
    )
    assert recurrence_template.recurrence_closure_kwargs == {
        "structural_envelope_recurrence_certified": True,
        "tail_transfer_recurrence_certified": True,
        "retained_order_schedule_recurrence_certified": True,
    }
    assert recurrence_template.scalar_recurrence_kwargs == {
        "structural_envelope_template_certified": True,
        "tail_transfer_template_certified": True,
        "retained_order_schedule_template_certified": True,
    }
    assert recurrence_template.structural_envelope_template_certified
    assert recurrence_template.tail_transfer_template_certified
    assert recurrence_template.retained_order_schedule_template_certified
    assert recurrence_template.recurrence_template_certified
    assert recurrence_template.missing_recurrence_template_obligations == ()
    assert recurrence_template.witness_source == "recurrence_template_test_witness"
    assert tuple(
        detail.obligation for detail in recurrence_template.recurrence_template_details
    ) == tuple(
        obligation
        for obligation, _certified in recurrence_template.recurrence_template_obligation_statuses
    )
    assert all(detail.certified for detail in recurrence_template.recurrence_template_details)
    assert weak_recurrence_template.structural_envelope_template_certified
    assert not weak_recurrence_template.tail_transfer_template_certified
    assert not weak_recurrence_template.retained_order_schedule_template_certified
    assert not weak_recurrence_template.recurrence_template_certified
    assert weak_recurrence_template.missing_recurrence_template_obligations == (
        "tail_transfer_growth_template",
        "retained_order_growth_template",
    )
    weak_template_details = {
        detail.obligation: detail
        for detail in weak_recurrence_template.recurrence_template_details
    }
    assert weak_template_details["tail_transfer_growth_template"].witness_field == (
        "tail_transfer_growth_template_certified"
    )
    assert weak_template_details["retained_order_growth_template"].witness_field == (
        "retained_order_growth_template_certified"
    )
    assert unfactored_structural_witness.domain_schedule_witness_matches_accelerated_chain
    assert not unfactored_structural_witness.structural_envelope_subproofs_certified
    assert not unfactored_structural_witness.global_domain_certified
    assert "segment-count growth" in unfactored_structural_witness.missing_global_induction_reason
    assert unfactored_structural_witness.missing_global_proof_obligations[:4] == (
        "future_segment_count_growth_bound",
        "future_state_supremum_growth_bound",
        "future_cauchy_denominator_growth_bound",
        "future_step_radius_ratio_bound",
    )
    assert too_weak_quantitative_witness.quantitative_witness_present
    assert too_weak_quantitative_witness.domain_schedule_witness_matches_accelerated_chain
    assert too_weak_quantitative_witness.structural_envelope_subproofs_certified
    assert too_weak_quantitative_witness.tail_transfer_subproofs_certified
    assert too_weak_quantitative_witness.accelerated_order_schedule_subproofs_certified
    assert too_weak_quantitative_witness.quantitative_witness_finite
    assert not too_weak_quantitative_witness.quantitative_witness_dominates_accelerated_bounds
    assert not too_weak_quantitative_witness.global_domain_certified
    assert "does not dominate" in too_weak_quantitative_witness.missing_global_induction_reason
    assert "quantitative_witness_dominates_accelerated_bounds" in (
        too_weak_quantitative_witness.missing_global_proof_obligations
    )
    too_weak_details = {
        detail.obligation: detail
        for detail in too_weak_quantitative_witness.global_proof_obligation_details
    }
    too_weak_quantitative_detail = too_weak_details[
        "quantitative_witness_dominates_accelerated_bounds"
    ]
    assert too_weak_quantitative_detail.witness_field == "certified_max_future_step_radius_ratio"
    assert too_weak_quantitative_detail.comparison == "<="
    assert too_weak_quantitative_detail.required == pytest.approx(
        accelerated_global.max_future_step_radius_ratio
    )
    assert too_weak_quantitative_detail.observed == pytest.approx(
        1.01 * accelerated_global.max_future_step_radius_ratio
    )
    too_weak_bound_details = {
        detail.witness_field: detail
        for detail in too_weak_quantitative_witness.quantitative_witness_bound_details
    }
    assert not too_weak_bound_details["certified_max_future_step_radius_ratio"].certified
    assert too_weak_bound_details["certified_max_future_segment_growth_ratio"].certified
    assert summability_witness.quantitative_witness_dominates_accelerated_bounds
    assert summability_witness.future_shell_induction_certified
    assert summability_witness.global_domain_certified
    assert summability_witness.global_tail_bound == pytest.approx(
        accelerated_global.accelerated_total_incremental_tail_bound
    )
    assert not summability_witness.global_series_certified
    assert "collision continuation" in summability_witness.missing_global_induction_reason
    assert summability_witness.missing_global_proof_obligations == (
        "collision_continuation",
        "binary_collision_continuation",
        "triple_collision_continuation",
    )
    assert not opaque_collision_witness.collision_continuation_obligations_certified
    assert not opaque_collision_witness.global_series_certified
    assert "binary-collision continuation" in opaque_collision_witness.missing_global_induction_reason
    assert opaque_collision_witness.missing_global_proof_obligations == (
        "binary_collision_continuation",
        "triple_collision_continuation",
    )
    assert not binary_only_collision_witness.collision_continuation_obligations_certified
    assert not binary_only_collision_witness.global_series_certified
    assert "triple-collision" in binary_only_collision_witness.missing_global_induction_reason
    assert binary_only_collision_witness.missing_global_proof_obligations == (
        "triple_collision_continuation",
    )
    assert complete_abstract_witness.collision_continuation_obligations_certified
    assert complete_abstract_witness.global_proof_obligations_certified
    assert complete_abstract_witness.missing_global_proof_obligations == ()
    assert complete_abstract_witness.global_series_certified
    assert complete_abstract_witness.witness_source == "abstract_test_witness"
    assert complete_abstract_witness.missing_global_induction_reason is None
    assert all(detail.certified for detail in complete_abstract_witness.global_proof_obligation_details)
    assert all(detail.certified for detail in complete_abstract_witness.domain_schedule_witness_details)
    assert all(detail.certified for detail in complete_abstract_witness.quantitative_witness_bound_details)
    assert recurrence_closure.future_shell_induction_certified
    assert recurrence_closure.global_domain_certified
    assert recurrence_closure.global_tail_bound == pytest.approx(
        accelerated_global.accelerated_total_incremental_tail_bound
    )
    assert recurrence_closure.missing_recurrence_obligations == ()
    assert recurrence_closure.missing_recurrence_reason is None
    assert tuple(
        detail.obligation for detail in recurrence_closure.recurrence_obligation_details
    ) == tuple(
        obligation for obligation, _certified in recurrence_closure.recurrence_obligation_statuses
    )
    assert all(detail.certified for detail in recurrence_closure.recurrence_obligation_details)
    no_collision_recurrence_witness = recurrence_closure.to_accelerated_induction_witness()
    assert no_collision_recurrence_witness.future_shell_induction_certified
    assert not no_collision_recurrence_witness.global_series_certified
    assert no_collision_recurrence_witness.missing_global_proof_obligations == (
        "collision_continuation",
        "binary_collision_continuation",
        "triple_collision_continuation",
    )
    collision_recurrence_witness = recurrence_closure.to_accelerated_induction_witness(
        collision_continuation_certified=True,
        binary_collision_continuation_certified=True,
        triple_collision_continuation_certified=True,
    )
    assert collision_recurrence_witness.global_series_certified
    assert collision_recurrence_witness.witness_source == "abstract_recurrence_test_witness"
    assert not weak_recurrence_closure.future_shell_induction_certified
    assert not weak_recurrence_closure.global_domain_certified
    assert "denominator_growth_recurrence" in weak_recurrence_closure.missing_recurrence_obligations
    assert "quantitative_witness_dominates_accelerated_bounds" in (
        weak_recurrence_closure.missing_recurrence_obligations
    )
    assert "Cauchy-denominator growth" in weak_recurrence_closure.missing_recurrence_reason
    weak_recurrence_details = {
        detail.obligation: detail
        for detail in weak_recurrence_closure.recurrence_obligation_details
    }
    assert weak_recurrence_details["denominator_growth_recurrence"].witness_field == (
        "denominator_growth_recurrence_certified"
    )
    assert not weak_recurrence_details["denominator_growth_recurrence"].certified
    assert (
        weak_recurrence_details["quantitative_witness_dominates_accelerated_bounds"].witness_field
        == "certified_max_future_step_radius_ratio"
    )
    assert weak_recurrence_details[
        "quantitative_witness_dominates_accelerated_bounds"
    ].observed == pytest.approx(1.01 * accelerated_global.max_future_step_radius_ratio)
    assert not weak_schedule_closure.future_shell_induction_certified
    assert not weak_schedule_closure.global_domain_certified
    assert "domain_exhaustion_recurrence" in weak_schedule_closure.missing_recurrence_obligations
    assert "scheduled_extension_recurrence" in weak_schedule_closure.missing_recurrence_obligations
    assert "geometric_domain_schedule_matches_checked_prefix" in (
        weak_schedule_closure.missing_recurrence_obligations
    )
    assert "compact-Sundman domains" in weak_schedule_closure.missing_recurrence_reason
    assert not weak_template_closure.future_shell_induction_certified
    assert not weak_template_closure.global_domain_certified
    assert "tail_transfer_recurrence" in weak_template_closure.missing_recurrence_obligations
    assert "retained_order_schedule_recurrence" in (
        weak_template_closure.missing_recurrence_obligations
    )
    assert "tail-transfer" in weak_template_closure.missing_recurrence_reason
    assert scalar_recurrence.scalar_recurrence_certified
    assert scalar_recurrence.missing_scalar_recurrence_obligations == ()
    assert tuple(
        detail.obligation for detail in scalar_recurrence.scalar_recurrence_details
    ) == tuple(
        obligation for obligation, _certified in scalar_recurrence.scalar_recurrence_obligation_statuses
    )
    scalar_closure = scalar_recurrence.to_future_shell_induction_closure()
    assert scalar_closure.future_shell_induction_certified
    assert scalar_closure.global_domain_certified
    assert scalar_closure.witness_source == "scalar_recurrence_test_witness"
    assert scalar_closure.missing_recurrence_obligations == ()
    assert scalar_closure.to_accelerated_induction_witness().missing_global_proof_obligations == (
        "collision_continuation",
        "binary_collision_continuation",
        "triple_collision_continuation",
    )
    assert not weak_scalar_recurrence.scalar_recurrence_certified
    assert weak_scalar_recurrence.missing_scalar_recurrence_obligations == (
        "step_radius_recurrence",
        "retained_order_growth_recurrence",
    )
    weak_scalar_details = {
        detail.obligation: detail for detail in weak_scalar_recurrence.scalar_recurrence_details
    }
    assert weak_scalar_details["step_radius_recurrence"].comparison == "<="
    assert weak_scalar_details["step_radius_recurrence"].observed == pytest.approx(
        1.01 * accelerated_global.max_future_step_radius_ratio
    )
    assert weak_scalar_details["retained_order_growth_recurrence"].comparison == ">="
    assert weak_scalar_details["retained_order_growth_recurrence"].observed == (
        accelerated_global.retained_order_increment_growth - 1
    )
    assert not weak_scalar_recurrence.to_future_shell_induction_closure().global_domain_certified
    assert not weak_template_scalar_recurrence.scalar_recurrence_certified
    assert weak_template_scalar_recurrence.missing_scalar_recurrence_obligations == (
        "tail_transfer_template",
        "retained_order_schedule_template",
    )
    weak_template_scalar_details = {
        detail.obligation: detail
        for detail in weak_template_scalar_recurrence.scalar_recurrence_details
    }
    assert weak_template_scalar_details["tail_transfer_template"].witness_field == (
        "tail_transfer_template_certified"
    )
    assert weak_template_scalar_details["retained_order_schedule_template"].witness_field == (
        "retained_order_schedule_template_certified"
    )
    assert not weak_template_scalar_recurrence.to_future_shell_induction_closure().global_domain_certified
    assert not constant_increment_global.observed_prefix_within_accelerated_bounds
    assert constant_increment_global.tail_transfer_growth_ratio_absorption_factor_bound == pytest.approx(
        1.3285337032399829
    )
    assert constant_increment_global.required_retained_order_increment_growth_for_tail_transfer_ratio == 1
    assert constant_increment_global.retained_order_increment_growth_surplus == -1
    assert not constant_increment_global.accelerated_tail_ratios_shrink
    assert not constant_increment_global.accelerated_global_summability_certified
    assert np.isinf(constant_increment_global.accelerated_future_tail_remainder_bound)
    tight_accelerated_holdout = certify_compactified_sundman_accelerated_future_envelope_holdout(
        chain,
        training_extension_count=4,
        segment_growth_safety_factor=1.05,
        state_sup_growth_safety_factor=1.05,
        denominator_growth_safety_factor=1.05,
        step_radius_safety_factor=1.05,
        tail_transfer_growth_safety_factor=1.05,
    )
    accelerated_holdout = certify_compactified_sundman_accelerated_future_envelope_holdout(
        chain,
        training_extension_count=4,
        segment_growth_safety_factor=1.05,
        state_sup_growth_safety_factor=1.05,
        denominator_growth_safety_factor=1.05,
        step_radius_safety_factor=1.12,
        tail_transfer_growth_safety_factor=1.05,
    )

    assert not tight_accelerated_holdout.validation_diagnostics_within_envelope
    assert not tight_accelerated_holdout.holdout_certified
    assert tight_accelerated_holdout.validation_tail_growth_within_projected_accelerated_bounds
    assert tight_accelerated_holdout.validation_tail_sum_within_projected_accelerated_bounds
    assert accelerated_holdout.validation_extension_count == 2
    assert accelerated_holdout.validation_connected
    assert accelerated_holdout.validation_retained_order_increments == (5, 6)
    assert accelerated_holdout.validation_retained_order_acceleration_matches
    assert accelerated_holdout.validation_incremental_tail_bounds == pytest.approx(
        (2.620020778462955e-22, 1.611083174147312e-26)
    )
    assert accelerated_holdout.validation_tail_growth_ratios == pytest.approx(
        (0.0002372054240205624, 6.149123653486671e-05)
    )
    assert accelerated_holdout.validation_cauchy_majorant_transition_factor_bounds == pytest.approx(
        (8.18404111174527e-05, 1.6767686128848877e-05)
    )
    assert accelerated_holdout.validation_tail_transfer_growth_factors == pytest.approx(
        (2.898389936973052, 3.667246396571723)
    )
    assert accelerated_holdout.validation_tail_transfer_growth_ratios == pytest.approx(
        (1.238858185525622, 1.2652701935618884)
    )
    assert accelerated_holdout.projected_tail_transfer_growth_bounds == pytest.approx(
        (3.0564127938884025, 3.992903362435274)
    )
    assert accelerated_holdout.validation_tail_transfer_within_projected_growth_bounds
    assert accelerated_holdout.projected_validation_tail_ratio_bounds == pytest.approx(
        (0.0004488667809333775, 9.371276544208172e-05)
    )
    assert accelerated_holdout.projected_validation_tail_bounds == pytest.approx(
        (4.957897980888001e-22, 4.646183305687278e-26)
    )
    assert accelerated_holdout.projected_validation_tail_sum_bound == pytest.approx(4.958362599218569e-22)
    assert accelerated_holdout.projected_validation_tail_bounds == pytest.approx(
        accelerated_holdout.accelerated_certificate.future_tail_bounds(
            accelerated_holdout.validation_extension_count
        )
    )
    assert accelerated_holdout.projected_validation_tail_sum_bound == pytest.approx(
        accelerated_holdout.accelerated_certificate.finite_future_tail_sum_bound(
            accelerated_holdout.validation_extension_count
        )
    )
    assert accelerated_holdout.validation_diagnostics_within_envelope
    assert accelerated_holdout.validation_tail_growth_within_projected_accelerated_bounds
    assert accelerated_holdout.validation_tail_sum_within_projected_accelerated_bounds
    assert accelerated_holdout.holdout_certified
    assert not accelerated_holdout.global_domain_certified
    assert "no induction proof" in accelerated_holdout.missing_global_induction_reason
    accelerated_cross_validation = certify_compactified_sundman_accelerated_future_envelope_cross_validation(
        chain,
        training_extension_counts=(4, 5),
        segment_growth_safety_factor=1.05,
        state_sup_growth_safety_factor=1.05,
        denominator_growth_safety_factor=1.05,
        step_radius_safety_factor=1.12,
        tail_transfer_growth_safety_factor=1.05,
    )
    tight_accelerated_cross_validation = certify_compactified_sundman_accelerated_future_envelope_cross_validation(
        chain,
        training_extension_counts=(4, 5),
        segment_growth_safety_factor=1.05,
        state_sup_growth_safety_factor=1.05,
        denominator_growth_safety_factor=1.05,
        step_radius_safety_factor=1.05,
        tail_transfer_growth_safety_factor=1.05,
    )

    assert accelerated_cross_validation.split_count == 2
    assert accelerated_cross_validation.training_extension_counts == (4, 5)
    assert accelerated_cross_validation.validation_extension_counts == (2, 1)
    assert accelerated_cross_validation.holdout_certified_flags == (True, True)
    assert accelerated_cross_validation.certified_split_count == 2
    assert accelerated_cross_validation.all_holdouts_certified
    assert accelerated_cross_validation.max_projected_validation_tail_sum_bound == pytest.approx(
        4.958362599218569e-22
    )
    assert accelerated_cross_validation.max_validation_total_incremental_tail_bound == pytest.approx(
        2.6201818867803696e-22
    )
    assert not accelerated_cross_validation.global_domain_certified
    assert "no induction proof" in accelerated_cross_validation.missing_global_induction_reason
    assert tight_accelerated_cross_validation.holdout_certified_flags == (False, False)
    assert tight_accelerated_cross_validation.certified_split_count == 0
    assert not tight_accelerated_cross_validation.all_holdouts_certified
    with pytest.raises(ValueError, match="tail_transfer_growth_safety_factor"):
        certify_compactified_sundman_finite_prefix_accelerated_summability(
            chain,
            tail_transfer_growth_safety_factor=0.99,
        )
    with pytest.raises(ValueError, match="training_extension_counts"):
        certify_compactified_sundman_accelerated_future_envelope_cross_validation(
            chain,
            training_extension_counts=(),
        )
    with pytest.raises(ValueError, match="held-out extension"):
        certify_compactified_sundman_accelerated_future_envelope_holdout(
            chain,
            training_extension_count=chain.extension_count,
        )


def test_interval_compactified_sundman_time_target_cauchy_mode_caps_oversized_steps():
    masses, positions, velocities = _general_initial_data()
    point_chart = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=18,
        sundman_rate=1.2,
    )
    radius_certificate = compactified_sundman_cauchy_majorant_tail_certificate(
        point_chart,
        retained_order=10,
        compact_parameter=0.0,
    )
    first_limit = 0.5 * radius_certificate.compact_radius
    target_w = 1.5 * first_limit
    target_time = point_chart.physical_time_delta_at_w(target_w)

    target = continue_interval_compactified_sundman_to_time(
        _interval_box_around(positions, 1e-15),
        _interval_box_around(velocities, 1e-15),
        masses,
        target_time,
        order=10,
        sundman_rate=1.2,
        max_compact_step=1.0,
        radius_fraction=0.9,
        target_bisections=36,
        tail_certificate_mode="cauchy",
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert target.proof_certified
    assert target.steps
    assert abs(target.steps[0].compact_step) == pytest.approx(first_limit)
    assert target.steps[0].tail_certificate.ratio_bound <= 0.5 + 1e-14
    assert target.cauchy_cover_certified
    assert target.cauchy_cover_certificate.step_count == len(target.steps) + 1
    assert target.cauchy_cover_certificate.max_step_radius_ratio <= 0.5 + 1e-14
    assert target.target_state_contains(reference)


def test_interval_compactified_sundman_time_target_can_use_cauchy_tail_certificates():
    masses, positions, velocities = _general_initial_data()
    point_chart = construct_compactified_sundman_taylor_solution(
        positions,
        velocities,
        masses,
        order=18,
        sundman_rate=1.2,
    )
    target_time = point_chart.physical_time_delta_at_w(0.006)

    target = continue_interval_compactified_sundman_to_time(
        _interval_box_around(positions, 1e-15),
        _interval_box_around(velocities, 1e-15),
        masses,
        target_time,
        order=10,
        sundman_rate=1.2,
        max_compact_step=0.006,
        radius_fraction=0.2,
        target_bisections=36,
        tail_certificate_mode="cauchy",
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert target.proof_certified
    assert target.cauchy_cover_certified
    assert target.target_tail_certificate is not None
    assert target.target_tail_certificate.coefficient_source == "compactified_sundman_interval_cauchy_majorant"
    assert target.target_state_contains(reference)


def test_compactified_sundman_atlas_matches_reference_forward_and_backward():
    masses, positions, velocities = _general_initial_data()
    for initial_w, target_w, sign in [(-0.16, 0.17, 1.0), (0.18, -0.15, -1.0)]:
        result = continue_compactified_sundman_solution(
            positions,
            velocities,
            masses,
            target_w,
            initial_compact_parameter=initial_w,
            order=14,
            guard_order=6,
            sundman_rate=1.1,
            max_compact_step=0.035,
            radius_fraction=0.2,
        )
        reference = integrate_reference(positions, velocities, masses, result.total_physical_time_delta)

        assert result.certified
        assert result.tail_certified
        assert result.chain_certified
        assert result.proof_certified
        assert len(result.steps) > 1
        assert result.compact_parameters[0] == pytest.approx(initial_w)
        assert result.compact_parameters[-1] == pytest.approx(target_w)
        assert np.all(sign * np.diff(result.compact_parameters) > 0.0)
        assert result.total_sundman_time_delta == pytest.approx(
            physical_time_from_compact_parameter(target_w, rate=1.1)
            - physical_time_from_compact_parameter(initial_w, rate=1.1),
            abs=2e-16,
        )
        assert np.linalg.norm(result.final_state - reference, ord=np.inf) < 2e-11
        broken_physical_times = result.physical_times.copy()
        broken_physical_times[1] = np.nextafter(broken_physical_times[1] + 1e-4, np.inf)
        broken = replace(result, physical_times=broken_physical_times)
        assert broken.certified
        assert broken.tail_certified
        assert not broken.chain_certified
        assert not broken.proof_certified


def test_interval_compactified_sundman_atlas_contains_point_atlas_across_steps():
    masses, positions, velocities = _general_initial_data()
    initial_w = -0.06
    target_w = 0.06
    point = continue_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        target_w,
        initial_compact_parameter=initial_w,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.03,
        radius_fraction=0.2,
    )
    higher_order = continue_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        target_w,
        initial_compact_parameter=initial_w,
        order=16,
        sundman_rate=1.15,
        max_compact_step=0.03,
        radius_fraction=0.2,
    )
    interval = continue_interval_compactified_sundman_solution(
        _interval_box_around(positions, 1e-15),
        _interval_box_around(velocities, 1e-15),
        masses,
        target_w,
        initial_compact_parameter=initial_w,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.03,
        radius_fraction=0.2,
        guard_order=6,
    )

    assert interval.certified
    assert interval.tail_certified
    assert interval.chain_certified
    assert interval.proof_certified
    assert interval.local_tail_bound > 0.0
    assert interval.max_step_tail_bound > 0.0
    assert len(interval.steps) == len(point.steps)
    assert np.allclose(interval.compact_parameters, point.compact_parameters)
    assert interval.total_sundman_time_delta == pytest.approx(point.total_sundman_time_delta)
    assert (
        interval.total_physical_time_interval.lower
        <= point.total_physical_time_delta
        <= interval.total_physical_time_interval.upper
    )
    for index, step in enumerate(interval.steps):
        assert step.start_state_contains(point.states[index])
        assert step.end_state_contains(point.states[index + 1])
        assert step.time_monotone_certified
        assert step.factor_interval.lower > 0.0
        assert step.residual_certificate.certified
        assert step.angular_momentum_certificate.certified
        assert step.energy_certificate.certified
        assert step.linear_momentum_certificate.certified
        assert step.center_of_mass_certificate.certified
        assert step.tail_certificate is not None
        assert step.tail_certificate.uses_interval_coefficients
        assert step.tail_certificate.is_nontrivial
    assert interval.final_state_contains(point.final_state)
    assert interval.final_state_contains(higher_order.final_state)

    broken_state = np.array(interval.state_intervals[1], dtype=object).copy()
    broken_entry = broken_state.flat[0]
    broken_state.flat[0] = FloatInterval(
        float(np.nextafter(broken_entry.lower + 0.1, -np.inf)),
        float(np.nextafter(broken_entry.upper + 0.1, np.inf)),
    )
    broken_state_intervals = (
        interval.state_intervals[0],
        broken_state,
        *interval.state_intervals[2:],
    )
    broken_interval = replace(interval, state_intervals=broken_state_intervals)
    assert broken_interval.certified
    assert broken_interval.tail_certified
    assert not broken_interval.chain_certified
    assert not broken_interval.proof_certified


def test_interval_compactified_sundman_to_time_contains_point_target_forward_and_backward():
    masses, positions, velocities = _general_initial_data()
    for target_w in (0.072, -0.066):
        point = continue_compactified_sundman_solution(
            positions,
            velocities,
            masses,
            target_w,
            order=10,
            sundman_rate=1.1,
            max_compact_step=0.03,
            radius_fraction=0.2,
        )
        higher_order = continue_compactified_sundman_solution(
            positions,
            velocities,
            masses,
            target_w,
            order=16,
            sundman_rate=1.1,
            max_compact_step=0.03,
            radius_fraction=0.2,
        )
        target = continue_interval_compactified_sundman_to_time(
            _interval_box_around(positions, 1e-15),
            _interval_box_around(velocities, 1e-15),
            masses,
            point.final_physical_time,
            order=10,
            sundman_rate=1.1,
            max_compact_step=0.03,
            radius_fraction=0.2,
            guard_order=6,
            target_bisections=42,
        )

        assert target.certified
        assert target.tail_certified
        assert target.chain_certified
        assert target.proof_certified
        assert target.local_tail_bound > 0.0
        assert target.max_step_tail_bound > 0.0
        assert target.target_certificate.certified
        assert target.target_certificate.factor_interval.lower > 0.0
        assert target.target_residual_certificate.certified
        assert target.target_angular_momentum_certificate.certified
        assert target.target_energy_certificate.certified
        assert target.target_linear_momentum_certificate.certified
        assert target.target_center_of_mass_certificate.certified
        assert target.angular_momentum_certified_step_count == len(target.steps) + 1
        assert target.energy_certified_step_count == len(target.steps) + 1
        assert target.linear_momentum_certified_step_count == len(target.steps) + 1
        assert target.center_of_mass_certified_step_count == len(target.steps) + 1
        assert target.triple_collision_excluded
        assert target.triple_collision_status == "excluded"
        assert not target.triple_collision_undecided
        assert target.triple_collision_exclusion_certificate.certified
        assert target.triple_collision_exclusion_reason == "centered angular momentum is bounded away from zero"
        assert target.target_tail_certificate is not None
        assert target.target_tail_certificate.uses_interval_coefficients
        assert target.target_compact_parameter_interval.lower <= target_w <= target.target_compact_parameter_interval.upper
        assert (
            target.target_certificate.global_time_at_lower.upper
            <= point.final_physical_time
            <= target.target_certificate.global_time_at_upper.lower
        )
        assert len(target.steps) >= 2
        assert target.target_state_contains(point.final_state)
        assert target.target_state_contains(higher_order.final_state)

        broken_certificate = replace(
            target.target_certificate,
            start_time_interval=FloatInterval.point(target.target_time + 1.0),
        )
        broken_target = replace(target, target_certificate=broken_certificate)
        assert broken_target.certified
        assert broken_target.tail_certified
        assert not broken_target.chain_certified
        assert not broken_target.proof_certified


def test_compactified_sundman_rejects_invalid_inputs():
    masses, positions, velocities = _general_initial_data()

    with pytest.raises(ValueError, match="sundman_rate"):
        construct_compactified_sundman_taylor_solution(positions, velocities, masses, order=8, sundman_rate=0.0)

    with pytest.raises(ValueError, match="distance_power"):
        construct_compactified_sundman_taylor_solution(positions, velocities, masses, order=8, distance_power=0.0)

    with pytest.raises(ValueError, match="strictly between"):
        continue_compactified_sundman_solution(positions, velocities, masses, 1.0)

    with pytest.raises(ValueError, match="guard_order"):
        continue_compactified_sundman_solution(positions, velocities, masses, 0.1, guard_order=-1)

    with pytest.raises(ValueError, match="max_compact_step"):
        continue_interval_compactified_sundman_solution(positions, velocities, masses, 0.1, max_compact_step=0.0)

    with pytest.raises(ValueError, match="target_time"):
        continue_interval_compactified_sundman_to_time(positions, velocities, masses, 0.0)

    positions[1] = positions[0] + np.array([1e-4, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-collision"):
        construct_interval_compactified_sundman_taylor_solution_from_intervals(
            _interval_box_around(positions, 1e-3),
            _interval_box_around(velocities, 1e-15),
            masses,
            order=8,
        )
