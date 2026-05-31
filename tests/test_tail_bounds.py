import numpy as np
import pytest

from three_body_symmetry.binary_chart import (
    RegularizedBinaryCollisionChartState,
    exact_binary_collision_interval_chart,
    planar_interval_to_regularized_binary_collision_chart,
    planar_interval_to_regularized_binary_collision_chart_atlas,
    planar_to_regularized_binary_collision_chart,
)
from three_body_symmetry.binary_series import construct_regularized_binary_taylor_solution
from three_body_symmetry.hybrid import (
    continue_hybrid,
    event_limited_ordinary_taylor_step,
    regularized_binary_taylor_step,
)
from three_body_symmetry.intervals import FloatInterval
from three_body_symmetry.series import construct_interval_taylor_solution, construct_taylor_solution
from three_body_symmetry.sundman import construct_sundman_taylor_solution, continue_sundman_to_s
from three_body_symmetry.tail_bounds import (
    evaluate_coefficients,
    guarded_tail_certificate,
    interval_guarded_tail_certificate,
    ordinary_cauchy_majorant_tail_certificate,
    ordinary_interval_cauchy_majorant_tail_certificate,
    ordinary_interval_union_cauchy_majorant_tail_certificate,
    ordinary_interval_solution_arrays,
    ordinary_solution_arrays,
    ordinary_taylor_tail_certificate,
    regularized_binary_cauchy_majorant_tail_certificate,
    regularized_binary_interval_atlas_cauchy_majorant_tail_certificate,
    regularized_binary_interval_cauchy_majorant_tail_certificate,
    regularized_binary_tail_certificate,
    regularized_binary_solution_arrays,
    sundman_cauchy_majorant_tail_certificate,
    sundman_interval_cauchy_majorant_tail_certificate,
    sundman_solution_arrays,
    sundman_taylor_tail_certificate,
)


def _ordinary_data():
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


def _binary_data():
    masses = np.array([0.8, 1.2, 1.7])
    positions = np.array(
        [
            [-0.30, 0.20],
            [0.45, -0.10],
            [1.30, 0.90],
        ]
    )
    velocities = np.array(
        [
            [0.15, -0.05],
            [-0.10, 0.22],
            [0.03, -0.08],
        ]
    )
    return masses, positions, velocities


def _exact_collision_state():
    masses = np.array([0.8, 1.2, 1.7])
    pair_mass = masses[0] + masses[1]
    return RegularizedBinaryCollisionChartState(
        masses=masses,
        pair=(0, 1),
        z=np.array([0.0, 0.0]),
        z_velocity=np.array([np.sqrt(pair_mass / 2.0), 0.0]),
        pair_energy=-0.3,
        binary_center=np.array([0.0, 0.0]),
        binary_center_velocity=np.array([0.2, -0.1]),
        third_offset=np.array([1.5, 0.25]),
        third_offset_velocity=np.array([-0.03, 0.07]),
    )


def _exact_collision_interval_state():
    initial = _exact_collision_state()
    return exact_binary_collision_interval_chart(
        initial.masses,
        pair=initial.pair,
        z_velocity=(
            (initial.z_velocity[0] - 1e-14, initial.z_velocity[0] + 1e-14),
            (initial.z_velocity[1], initial.z_velocity[1]),
        ),
        pair_energy=(initial.pair_energy - 1e-14, initial.pair_energy + 1e-14),
        binary_center=tuple((value - 1e-14, value + 1e-14) for value in initial.binary_center),
        binary_center_velocity=tuple(
            (value - 1e-14, value + 1e-14) for value in initial.binary_center_velocity
        ),
        third_offset=tuple((value - 1e-14, value + 1e-14) for value in initial.third_offset),
        third_offset_velocity=tuple(
            (value - 1e-14, value + 1e-14) for value in initial.third_offset_velocity
        ),
    )


def _planar_state_interval(positions, velocities, half_width=0.0):
    state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
    return tuple((float(value - half_width), float(value + half_width)) for value in state)


def test_guarded_tail_certificate_bounds_observed_ordinary_tail():
    masses, positions, velocities = _ordinary_data()
    solution = construct_taylor_solution(positions, velocities, masses, order=18)
    arrays = ordinary_solution_arrays(solution)
    step = 0.02

    certificate = guarded_tail_certificate(arrays, retained_order=10, step_size=step)
    full = evaluate_coefficients(arrays, step)
    retained = evaluate_coefficients(arrays, step, max_order=10)

    assert certificate.is_nontrivial
    assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.tail_bound


def test_interval_guarded_tail_certificate_bounds_observed_ordinary_tail():
    masses, positions, velocities = _ordinary_data()
    point_solution = construct_taylor_solution(positions, velocities, masses, order=18)
    interval_solution = construct_interval_taylor_solution(positions, velocities, masses, order=18)
    arrays = ordinary_interval_solution_arrays(interval_solution)
    step = 0.02

    certificate = interval_guarded_tail_certificate(arrays, retained_order=10, step_size=step)
    full = evaluate_coefficients(ordinary_solution_arrays(point_solution), step)
    retained = evaluate_coefficients(ordinary_solution_arrays(point_solution), step, max_order=10)

    assert interval_solution.contains_point_solution(point_solution)
    assert certificate.uses_interval_coefficients
    assert certificate.is_nontrivial
    assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.observed_tail
    assert certificate.observed_tail <= certificate.tail_bound


def test_ordinary_cauchy_majorant_tail_certificate_bounds_observed_tail():
    masses, positions, velocities = _ordinary_data()
    solution = construct_taylor_solution(positions, velocities, masses, order=24)
    arrays = ordinary_solution_arrays(solution)
    step = 0.02

    certificate = ordinary_cauchy_majorant_tail_certificate(
        positions,
        velocities,
        masses,
        retained_order=10,
        step_size=step,
    )
    full = evaluate_coefficients(arrays, step)
    retained = evaluate_coefficients(arrays, step, max_order=10)

    assert certificate.is_nontrivial
    assert certificate.coefficient_source == "cauchy_majorant"
    assert certificate.lower_squared_distance > 0.0
    assert abs(step) < certificate.time_radius
    assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.tail_bound


def test_ordinary_cauchy_majorant_rejects_step_outside_certified_radius():
    masses, positions, velocities = _ordinary_data()

    with pytest.raises(ValueError, match="inside the certified time radius"):
        ordinary_cauchy_majorant_tail_certificate(
            positions,
            velocities,
            masses,
            retained_order=10,
            step_size=1.0,
        )


def test_ordinary_interval_cauchy_majorant_bounds_sampled_box_tails():
    masses, positions, velocities = _ordinary_data()
    state_interval = _planar_state_interval(positions, velocities, half_width=1e-6)
    radius_certificate = ordinary_interval_cauchy_majorant_tail_certificate(
        state_interval,
        masses,
        retained_order=10,
        step_size=0.0,
    )
    step = 0.4 * radius_certificate.time_radius

    certificate = ordinary_interval_cauchy_majorant_tail_certificate(
        state_interval,
        masses,
        retained_order=10,
        step_size=step,
    )

    assert certificate.is_nontrivial
    assert certificate.coefficient_source == "interval_cauchy_majorant"
    assert certificate.uses_interval_initial_state
    assert certificate.lower_squared_distance > 0.0
    assert certificate.ratio_bound == pytest.approx(0.4)

    samples = [
        (positions, velocities),
        (positions + 0.5e-6, velocities - 0.5e-6),
        (positions - 0.5e-6, velocities + 0.5e-6),
    ]
    for sample_positions, sample_velocities in samples:
        solution = construct_taylor_solution(sample_positions, sample_velocities, masses, order=24)
        arrays = ordinary_solution_arrays(solution)
        full = evaluate_coefficients(arrays, step)
        retained = evaluate_coefficients(arrays, step, max_order=10)
        assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.tail_bound


def test_ordinary_interval_union_cauchy_majorant_aggregates_member_bounds():
    masses, positions, velocities = _ordinary_data()
    left_positions = positions.copy()
    right_positions = positions.copy()
    left_positions[0, 0] -= 1e-4
    right_positions[0, 0] += 1e-4
    state_interval_union = (
        _planar_state_interval(left_positions, velocities, half_width=1e-7),
        _planar_state_interval(right_positions, velocities, half_width=1e-7),
    )
    radius_certificate = ordinary_interval_union_cauchy_majorant_tail_certificate(
        state_interval_union,
        masses,
        retained_order=10,
        step_size=0.0,
    )
    step = 0.5 * radius_certificate.time_radius

    certificate = ordinary_interval_union_cauchy_majorant_tail_certificate(
        state_interval_union,
        masses,
        retained_order=10,
        step_size=step,
    )

    member_bounds = [member.tail_bound for member in certificate.member_certificates]
    assert certificate.member_count == 2
    assert certificate.coefficient_source == "interval_cauchy_majorant_union"
    assert certificate.uses_interval_initial_state
    assert certificate.is_nontrivial
    assert certificate.time_radius == min(member.time_radius for member in certificate.member_certificates)
    assert certificate.tail_bound == max(member_bounds)
    assert certificate.ratio_bound <= 0.5 + 1e-14


def test_guarded_tail_certificate_bounds_observed_regularized_binary_tail():
    masses, positions, velocities = _binary_data()
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    solution = construct_regularized_binary_taylor_solution(initial, order=18)
    arrays = regularized_binary_solution_arrays(solution)
    step = 0.02

    certificate = guarded_tail_certificate(arrays, retained_order=10, step_size=step)
    full = evaluate_coefficients(arrays, step)
    retained = evaluate_coefficients(arrays, step, max_order=10)

    assert certificate.is_nontrivial
    assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.tail_bound


def test_regularized_binary_cauchy_majorant_tail_certificate_bounds_observed_tail():
    masses, positions, velocities = _binary_data()
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    solution = construct_regularized_binary_taylor_solution(initial, order=24)
    arrays = regularized_binary_solution_arrays(solution)
    step = 0.002

    certificate = regularized_binary_cauchy_majorant_tail_certificate(
        initial,
        retained_order=10,
        step_size=step,
    )
    full = evaluate_coefficients(arrays, step)
    retained = evaluate_coefficients(arrays, step, max_order=10)

    assert certificate.is_nontrivial
    assert certificate.coefficient_source == "cauchy_majorant"
    assert certificate.third_distance_lower_bound > 0.0
    assert certificate.selected_pair_rho_bound > 0.0
    assert abs(step) < certificate.s_radius
    assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.tail_bound


def test_regularized_binary_interval_cauchy_majorant_bounds_sampled_box_tails():
    masses, positions, velocities = _binary_data()
    state_interval = _planar_state_interval(positions, velocities, half_width=1e-7)
    interval_initial = planar_interval_to_regularized_binary_collision_chart(
        state_interval,
        masses,
        pair=(0, 1),
    )
    radius_certificate = regularized_binary_interval_cauchy_majorant_tail_certificate(
        interval_initial,
        retained_order=10,
        step_size=0.0,
    )
    step = 0.4 * radius_certificate.s_radius

    certificate = regularized_binary_interval_cauchy_majorant_tail_certificate(
        interval_initial,
        retained_order=10,
        step_size=step,
    )

    assert certificate.is_nontrivial
    assert certificate.coefficient_source == "regularized_interval_cauchy_majorant"
    assert certificate.uses_interval_initial_state
    assert certificate.third_distance_lower_bound > 0.0
    assert certificate.selected_pair_rho_bound > 0.0
    assert certificate.ratio_bound == pytest.approx(0.4)

    samples = [
        (positions, velocities),
        (positions + 0.5e-7, velocities - 0.5e-7),
        (positions - 0.5e-7, velocities + 0.5e-7),
    ]
    for sample_positions, sample_velocities in samples:
        initial = planar_to_regularized_binary_collision_chart(sample_positions, sample_velocities, masses, pair=(0, 1))
        assert interval_initial.contains_point(initial)
        solution = construct_regularized_binary_taylor_solution(initial, order=24)
        arrays = regularized_binary_solution_arrays(solution)
        full = evaluate_coefficients(arrays, step)
        retained = evaluate_coefficients(arrays, step, max_order=10)
        assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.tail_bound


def test_regularized_binary_interval_cauchy_majorant_accepts_exact_collision_lift():
    initial = _exact_collision_state()
    interval_initial = _exact_collision_interval_state()
    radius_certificate = regularized_binary_interval_cauchy_majorant_tail_certificate(
        interval_initial,
        retained_order=10,
        step_size=0.0,
    )
    step = 0.25 * radius_certificate.s_radius

    certificate = regularized_binary_interval_cauchy_majorant_tail_certificate(
        interval_initial,
        retained_order=10,
        step_size=step,
    )
    solution = construct_regularized_binary_taylor_solution(initial, order=24)
    arrays = regularized_binary_solution_arrays(solution)
    full = evaluate_coefficients(arrays, step)
    retained = evaluate_coefficients(arrays, step, max_order=10)

    assert interval_initial.contains_point(initial)
    assert interval_initial.branch_certificate.branch == "exact_binary_collision"
    assert certificate.is_nontrivial
    assert certificate.coefficient_source == "regularized_interval_cauchy_majorant"
    assert certificate.uses_interval_initial_state
    assert certificate.third_distance_lower_bound > 0.0
    assert certificate.selected_pair_rho_bound > 0.0
    assert certificate.ratio_bound == pytest.approx(0.25)
    assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.tail_bound


def test_regularized_binary_interval_atlas_cauchy_majorant_bounds_branch_tails():
    masses = np.array([1.0, 1.0, 1.0])
    state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-1.2, -0.8),
        (-0.2, 0.2),
        (3.0, 3.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    atlas = planar_interval_to_regularized_binary_collision_chart_atlas(
        state_interval,
        masses,
        pair=(0, 1),
    )
    radius_certificate = regularized_binary_interval_atlas_cauchy_majorant_tail_certificate(
        atlas,
        retained_order=10,
        step_size=0.0,
    )
    step = 0.4 * radius_certificate.s_radius

    certificate = regularized_binary_interval_atlas_cauchy_majorant_tail_certificate(
        atlas,
        retained_order=10,
        step_size=step,
    )

    assert len(atlas) == 2
    assert certificate.member_count == 2
    assert certificate.coefficient_source == "regularized_interval_cauchy_majorant_atlas"
    assert certificate.uses_interval_initial_state
    assert certificate.is_nontrivial
    assert certificate.tail_bound == max(member.tail_bound for member in certificate.member_certificates)
    assert certificate.ratio_bound <= 0.4 + 1e-14

    velocities = np.zeros((3, 2))
    samples = [
        np.array([[0.0, 0.0], [-1.0, 0.1], [3.0, 0.0]]),
        np.array([[0.0, 0.0], [-1.0, -0.1], [3.0, 0.0]]),
    ]
    for sample_positions in samples:
        initial = planar_to_regularized_binary_collision_chart(sample_positions, velocities, masses, pair=(0, 1))
        assert any(chart.contains_point(initial) for chart in atlas)
        solution = construct_regularized_binary_taylor_solution(initial, order=24)
        arrays = regularized_binary_solution_arrays(solution)
        full = evaluate_coefficients(arrays, step)
        retained = evaluate_coefficients(arrays, step, max_order=10)
        assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.tail_bound


def test_regularized_binary_cauchy_majorant_rejects_step_outside_certified_radius():
    masses, positions, velocities = _binary_data()
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))

    with pytest.raises(ValueError, match="inside the certified s radius"):
        regularized_binary_cauchy_majorant_tail_certificate(
            initial,
            retained_order=10,
            step_size=1.0,
        )


def test_guarded_tail_certificate_bounds_observed_sundman_tail():
    masses, positions, velocities = _ordinary_data()
    solution = construct_sundman_taylor_solution(positions, velocities, masses, order=18)
    arrays = sundman_solution_arrays(solution)
    step = 0.04

    certificate = guarded_tail_certificate(arrays, retained_order=10, step_size=step)
    full = evaluate_coefficients(arrays, step)
    retained = evaluate_coefficients(arrays, step, max_order=10)

    assert certificate.is_nontrivial
    assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.tail_bound


def test_sundman_cauchy_majorant_tail_certificate_bounds_observed_tail():
    masses, positions, velocities = _ordinary_data()
    solution = construct_sundman_taylor_solution(positions, velocities, masses, order=24)
    arrays = sundman_solution_arrays(solution)
    step = 0.015

    certificate = sundman_cauchy_majorant_tail_certificate(
        positions,
        velocities,
        masses,
        retained_order=10,
        step_size=step,
    )
    full = evaluate_coefficients(arrays, step)
    retained = evaluate_coefficients(arrays, step, max_order=10)

    assert certificate.is_nontrivial
    assert certificate.coefficient_source == "cauchy_majorant"
    assert certificate.lower_squared_distance > 0.0
    assert certificate.sundman_factor_bound > 0.0
    assert abs(step) < certificate.s_radius
    assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.tail_bound


def test_sundman_cauchy_radius_obeys_state_envelope_lower_bound():
    masses, positions, velocities = _ordinary_data()

    certificate = sundman_cauchy_majorant_tail_certificate(
        positions,
        velocities,
        masses,
        retained_order=10,
        step_size=0.0,
    )
    pair_distances = [
        float(np.linalg.norm(positions[j] - positions[i]))
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    min_pair_distance = min(pair_distances)
    max_pair_distance = max(pair_distances)
    max_velocity = max(float(np.linalg.norm(velocity)) for velocity in velocities)
    total_mass = float(np.sum(masses))
    position_radius = min_pair_distance / 10.0
    lower_squared_fraction = 14.0 / 25.0
    acceleration_envelope = (
        total_mass
        * (max_pair_distance + min_pair_distance / 5.0)
        / (lower_squared_fraction**1.5 * min_pair_distance**3)
    )
    sundman_factor_envelope = (max_pair_distance + min_pair_distance / 5.0) ** 3
    velocity_radius_envelope = float(np.sqrt(acceleration_envelope * position_radius))
    state_envelope_radius_lower_bound = min(
        position_radius / (sundman_factor_envelope * (max_velocity + velocity_radius_envelope)),
        np.sqrt(position_radius / acceleration_envelope) / sundman_factor_envelope,
    )

    assert state_envelope_radius_lower_bound > 0.012
    assert certificate.s_radius >= state_envelope_radius_lower_bound * (1.0 - 1e-12)


def test_sundman_cauchy_majorant_rejects_step_outside_certified_radius():
    masses, positions, velocities = _ordinary_data()

    with pytest.raises(ValueError, match="inside the certified s radius"):
        sundman_cauchy_majorant_tail_certificate(
            positions,
            velocities,
            masses,
            retained_order=10,
            step_size=1.0,
        )


def test_sundman_interval_cauchy_majorant_bounds_sampled_box_tails():
    masses, positions, velocities = _ordinary_data()
    position_intervals = np.empty(positions.shape, dtype=object)
    velocity_intervals = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        position_intervals[index] = FloatInterval(positions[index] - 1e-6, positions[index] + 1e-6)
        velocity_intervals[index] = FloatInterval(velocities[index] - 1e-6, velocities[index] + 1e-6)
    radius_certificate = sundman_interval_cauchy_majorant_tail_certificate(
        position_intervals,
        velocity_intervals,
        masses,
        retained_order=10,
        step_size=0.0,
    )
    step = 0.4 * radius_certificate.s_radius

    certificate = sundman_interval_cauchy_majorant_tail_certificate(
        position_intervals,
        velocity_intervals,
        masses,
        retained_order=10,
        step_size=step,
    )

    assert certificate.is_nontrivial
    assert certificate.coefficient_source == "sundman_interval_cauchy_majorant"
    assert certificate.uses_interval_initial_state
    assert certificate.lower_squared_distance > 0.0
    assert certificate.sundman_factor_bound > 0.0
    assert certificate.ratio_bound == pytest.approx(0.4)

    samples = [
        (positions, velocities),
        (positions + 0.5e-6, velocities - 0.5e-6),
        (positions - 0.5e-6, velocities + 0.5e-6),
    ]
    for sample_positions, sample_velocities in samples:
        solution = construct_sundman_taylor_solution(sample_positions, sample_velocities, masses, order=24)
        arrays = sundman_solution_arrays(solution)
        full = evaluate_coefficients(arrays, step)
        retained = evaluate_coefficients(arrays, step, max_order=10)
        assert np.linalg.norm(full - retained, ord=np.inf) <= certificate.tail_bound


def test_tail_helpers_use_interval_coefficients():
    masses, positions, velocities = _ordinary_data()
    binary_initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))

    certificates = [
        ordinary_taylor_tail_certificate(
            positions,
            velocities,
            masses,
            retained_order=10,
            guard_order=8,
            step_size=0.02,
        ),
        regularized_binary_tail_certificate(
            binary_initial,
            retained_order=10,
            guard_order=8,
            step_size=0.02,
        ),
        sundman_taylor_tail_certificate(
            positions,
            velocities,
            masses,
            retained_order=10,
            guard_order=8,
            step_size=0.04,
        ),
    ]

    assert all(certificate.uses_interval_coefficients for certificate in certificates)
    assert all(certificate.is_nontrivial for certificate in certificates)


def test_hybrid_steps_can_carry_tail_certificates():
    masses, positions, velocities = _ordinary_data()

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        0.05,
        binary_distance_threshold=0.05,
        max_time_step=0.02,
        ordinary_order=10,
        tail_guard_order=8,
    )

    assert continued.steps
    assert all(step.truncation_certificate is not None for step in continued.steps)
    assert all(step.truncation_certificate.uses_interval_coefficients for step in continued.steps)
    assert all(step.truncation_certificate.is_nontrivial for step in continued.steps)
    step_bounds = [step.truncation_certificate.tail_bound for step in continued.steps]
    assert continued.certified_step_count == len(continued.steps)
    assert continued.local_tail_bound == sum(step_bounds)
    assert continued.max_step_tail_bound == max(step_bounds)
    assert continued.local_tail_bound > 0.0


def test_hybrid_ordinary_steps_can_carry_cauchy_tail_certificates():
    masses, positions, velocities = _ordinary_data()
    first_limit = 0.5 * ordinary_interval_union_cauchy_majorant_tail_certificate(
        (_planar_state_interval(positions, velocities),),
        masses,
        retained_order=10,
        step_size=0.0,
    ).time_radius

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        2.0 * first_limit,
        binary_distance_threshold=0.05,
        max_time_step=1.0,
        ordinary_order=10,
        safety=1.0,
        tail_certificate_mode="cauchy",
    )

    assert continued.steps
    assert {step.chart for step in continued.steps} == {"ordinary"}
    assert continued.steps[0].physical_step == pytest.approx(first_limit)
    assert all(step.truncation_certificate is not None for step in continued.steps)
    assert all(
        step.truncation_certificate.coefficient_source == "interval_cauchy_majorant_union"
        for step in continued.steps
    )
    assert all(step.truncation_certificate.uses_interval_initial_state for step in continued.steps)
    assert all(step.truncation_certificate.is_nontrivial for step in continued.steps)
    assert all(step.truncation_certificate.ratio_bound <= 0.5 + 1e-14 for step in continued.steps)
    step_bounds = [step.truncation_certificate.tail_bound for step in continued.steps]
    assert continued.local_tail_bound == sum(step_bounds)


def test_hybrid_ordinary_entry_event_cauchy_certificate_covers_event_interval():
    masses = np.array([1.0, 1.0, 0.1])
    positions = np.array([[0.0, 0.0], [0.12, 0.0], [5.0, 0.0]])
    velocities = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])
    (
        _end_positions,
        _end_velocities,
        event_time,
        _indicator,
        _event_pair,
        _certificate,
    ) = event_limited_ordinary_taylor_step(
        positions,
        velocities,
        masses,
        0.01,
        order=20,
        binary_enter_distance=0.1199,
    )

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        event_time + 1e-8,
        binary_distance_threshold=0.1199,
        binary_exit_distance=0.13,
        max_time_step=0.01,
        safety=1.0,
        ordinary_order=20,
        binary_order=20,
        tail_certificate_mode="cauchy",
        max_steps=5,
    )

    step = continued.steps[0]
    assert step.chart == "ordinary"
    assert step.event == "enter_binary"
    assert step.event_time_interval is not None
    assert step.truncation_certificate is not None
    assert step.truncation_certificate.coefficient_source == "interval_cauchy_majorant_union"
    event_step_size = max(abs(step.event_time_interval[0]), abs(step.event_time_interval[1]))
    assert step.truncation_certificate.step_size == pytest.approx(event_step_size)
    assert step.truncation_certificate.step_size >= abs(step.physical_step)
    assert step.truncation_certificate.is_nontrivial


def test_hybrid_binary_steps_can_carry_cauchy_tail_certificates():
    masses = np.array([1.0, 1.0, 0.7])
    positions = np.array(
        [
            [0.0, 0.0],
            [1e-3, 0.0],
            [1.0, 0.4],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.02],
            [-0.01, 0.0],
        ]
    )
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    interval_initial = planar_interval_to_regularized_binary_collision_chart(
        _planar_state_interval(positions, velocities),
        masses,
        pair=(0, 1),
    )
    first_limit = 0.5 * regularized_binary_interval_cauchy_majorant_tail_certificate(
        interval_initial,
        retained_order=10,
        step_size=0.0,
    ).s_radius
    physical_target = construct_regularized_binary_taylor_solution(initial, order=10).physical_time_at(first_limit)

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        physical_target,
        binary_distance_threshold=1e-2,
        max_binary_s_step=1.0,
        binary_order=10,
        tail_certificate_mode="cauchy",
        max_steps=3,
    )

    assert len(continued.steps) == 1
    assert continued.steps[0].chart == "binary"
    assert continued.steps[0].parameter_step == pytest.approx(first_limit)
    assert continued.steps[0].truncation_certificate is not None
    assert continued.steps[0].truncation_certificate.coefficient_source == "regularized_interval_cauchy_majorant"
    assert continued.steps[0].truncation_certificate.uses_interval_initial_state
    assert continued.steps[0].truncation_certificate.is_nontrivial
    assert continued.steps[0].truncation_certificate.ratio_bound <= 0.5 + 1e-14
    assert continued.local_tail_bound == continued.steps[0].truncation_certificate.tail_bound


def test_hybrid_binary_exit_event_cauchy_certificate_covers_event_interval():
    masses = np.array([1.0, 1.0, 0.7])
    positions = np.array([[0.0, 0.0], [1e-3, 0.0], [1.0, 0.4]])
    velocities = np.array([[-40.0, 0.0], [40.0, 0.0], [-0.01, 0.0]])
    (
        _end_positions,
        _end_velocities,
        _s_step,
        physical_step,
        _certificate,
    ) = regularized_binary_taylor_step(
        positions,
        velocities,
        masses,
        (0, 1),
        1.0,
        order=30,
        max_s_step=0.05,
        binary_exit_distance=0.00101,
    )

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        physical_step,
        binary_distance_threshold=0.001005,
        binary_exit_distance=0.00101,
        max_binary_s_step=0.05,
        binary_order=30,
        tail_certificate_mode="cauchy",
        max_steps=3,
    )

    assert len(continued.steps) == 1
    step = continued.steps[0]
    assert step.chart == "binary"
    assert step.event == "exit_binary"
    assert step.event_certificate is not None
    assert step.event_certificate.root_enclosure is not None
    assert step.truncation_certificate is not None
    assert step.truncation_certificate.coefficient_source in {
        "regularized_interval_cauchy_majorant",
        "regularized_interval_cauchy_majorant_atlas",
        "cauchy_majorant",
    }
    event_step_size = max(
        abs(step.event_certificate.root_enclosure.interval[0]),
        abs(step.event_certificate.root_enclosure.interval[1]),
    )
    assert step.truncation_certificate.step_size == pytest.approx(event_step_size)
    assert step.truncation_certificate.step_size >= abs(step.parameter_step)
    assert step.truncation_certificate.is_nontrivial


def test_sundman_steps_can_carry_tail_certificates():
    masses, positions, velocities = _ordinary_data()

    continued = continue_sundman_to_s(
        positions,
        velocities,
        masses,
        0.045,
        order=10,
        max_s_step=0.015,
        tail_guard_order=8,
    )

    assert continued.steps
    assert all(step.truncation_certificate is not None for step in continued.steps)
    assert all(step.truncation_certificate.uses_interval_coefficients for step in continued.steps)
    assert all(step.truncation_certificate.is_nontrivial for step in continued.steps)
    step_bounds = [step.truncation_certificate.tail_bound for step in continued.steps]
    assert continued.certified_step_count == len(continued.steps)
    assert continued.local_tail_bound == sum(step_bounds)
    assert continued.max_step_tail_bound == max(step_bounds)
    assert continued.local_tail_bound > 0.0


def test_sundman_steps_can_carry_cauchy_tail_certificates():
    masses, positions, velocities = _ordinary_data()
    first_limit = 0.5 * sundman_cauchy_majorant_tail_certificate(
        positions,
        velocities,
        masses,
        retained_order=10,
        step_size=0.0,
    ).s_radius

    continued = continue_sundman_to_s(
        positions,
        velocities,
        masses,
        2.0 * first_limit,
        order=10,
        max_s_step=1.0,
        tail_certificate_mode="cauchy",
    )

    assert continued.steps
    assert abs(continued.steps[0].s_step) == pytest.approx(first_limit)
    assert all(step.truncation_certificate is not None for step in continued.steps)
    assert all(step.truncation_certificate.coefficient_source == "cauchy_majorant" for step in continued.steps)
    assert all(step.truncation_certificate.is_nontrivial for step in continued.steps)
    assert all(step.truncation_certificate.ratio_bound <= 0.5 + 1e-14 for step in continued.steps)
    step_bounds = [step.truncation_certificate.tail_bound for step in continued.steps]
    assert continued.local_tail_bound == sum(step_bounds)
