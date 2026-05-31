import math

import numpy as np

from three_body_symmetry.continuation import choose_step_size, continue_solution, pairwise_distances
from three_body_symmetry.dynamics import integrate, pack_state
from three_body_symmetry.series import integrate_reference


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


def test_taylor_chart_continuation_matches_reference_integration():
    masses, positions, velocities = _general_initial_data()
    t_final = 0.25

    continued = continue_solution(positions, velocities, masses, t_final, order=18, max_step=0.035)
    reference = integrate_reference(positions, velocities, masses, t_final)

    assert len(continued.steps) > 1
    assert np.linalg.norm(continued.final_state - reference, ord=np.inf) < 1e-11
    assert max(step.truncation_indicator for step in continued.steps) < 1e-19


def test_compact_collision_free_segment_has_finite_taylor_chart_cover():
    masses, positions, velocities = _general_initial_data()
    t_final = 0.3

    continued = continue_solution(
        positions,
        velocities,
        masses,
        t_final,
        order=18,
        max_step=0.025,
    )
    reference = integrate_reference(positions, velocities, masses, t_final)
    sampled_reference = integrate(
        pack_state(positions, velocities),
        t_final,
        masses=masses,
        samples=80,
    )
    reference_distance_floor = min(
        np.min(pairwise_distances(state[:9].reshape(3, 3)))
        for state in sampled_reference.states
    )

    assert len(continued.steps) < 20
    assert continued.times[0] == 0.0
    assert continued.times[-1] == t_final
    assert reference_distance_floor > 0.8
    assert min(step.min_pair_distance for step in continued.steps) > 0.8
    assert max(step.truncation_indicator for step in continued.steps) < 1e-20
    assert np.linalg.norm(continued.final_state - reference, ord=np.inf) < 1e-11

    for previous, current in zip(continued.steps, continued.steps[1:]):
        assert current.start_time == previous.end_time

    for step in continued.steps:
        assert 0.0 < step.step_size <= 0.025
        for local_fraction in (0.0, 0.5, 1.0):
            local_time = local_fraction * step.step_size
            positions_on_chart = step.series.positions_at(local_time)
            assert np.min(pairwise_distances(positions_on_chart)) > (
                0.5 * step.min_pair_distance
            )


def test_lagrange_relative_equilibrium_has_uniform_all_future_taylor_recurrence():
    masses = np.ones(3)
    angular_speed = 3.0 ** (-0.25)
    angles = np.array([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    positions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    velocities = angular_speed * np.stack([-np.sin(angles), np.cos(angles)], axis=1)

    period = 2.0 * math.pi / angular_speed
    target_time = 3.0 * period
    max_step = 0.12
    order = 10
    continued = continue_solution(
        positions,
        velocities,
        masses,
        target_time,
        order=order,
        max_step=max_step,
    )

    pair_distance = math.sqrt(3.0)
    acceleration_norm = angular_speed**2
    uniform_step_candidate = 0.08 * min(
        pair_distance / angular_speed,
        math.sqrt(pair_distance / acceleration_norm),
    )
    assert uniform_step_candidate > max_step
    assert len(continued.steps) > 200
    assert all(
        abs(abs(step.step_size) - max_step) < 1e-14
        for step in continued.steps[:-1]
    )
    assert min(step.min_pair_distance for step in continued.steps) > (
        0.999999999 * pair_distance
    )
    assert max(step.truncation_indicator for step in continued.steps) < 2e-17

    max_speed = 0.0
    max_radius = 0.0
    for state in continued.states:
        state_positions = state[:6].reshape(3, 2)
        state_velocities = state[6:].reshape(3, 2)
        max_speed = max(max_speed, float(np.max(np.linalg.norm(state_velocities, axis=1))))
        max_radius = max(max_radius, float(np.max(np.linalg.norm(state_positions, axis=1))))
    assert max_speed < 1.000000001 * angular_speed
    assert max_radius < 1.000000001

    final_angles = angles + angular_speed * target_time
    exact_positions = np.stack([np.cos(final_angles), np.sin(final_angles)], axis=1)
    exact_velocities = angular_speed * np.stack(
        [-np.sin(final_angles), np.cos(final_angles)],
        axis=1,
    )
    exact_final = np.concatenate([exact_positions.reshape(-1), exact_velocities.reshape(-1)])
    assert np.linalg.norm(continued.final_state - exact_final, ord=np.inf) < 1e-9

    first_step = continued.steps[0]
    one_step_angles = angles + angular_speed * max_step
    exact_one_step_positions = np.stack(
        [np.cos(one_step_angles), np.sin(one_step_angles)],
        axis=1,
    )
    exact_one_step_velocities = angular_speed * np.stack(
        [-np.sin(one_step_angles), np.cos(one_step_angles)],
        axis=1,
    )
    exact_one_step = np.concatenate(
        [
            exact_one_step_positions.reshape(-1),
            exact_one_step_velocities.reshape(-1),
        ]
    )
    assert np.linalg.norm(first_step.series.state_at(max_step) - exact_one_step, ord=np.inf) < 1e-14


def test_periodic_collision_free_orbit_reuses_finite_cyclic_taylor_atlas_all_future():
    masses = np.ones(3)
    angular_speed = 3.0 ** (-0.25)
    angles = np.array([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    positions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    velocities = angular_speed * np.stack([-np.sin(angles), np.cos(angles)], axis=1)
    period = 2.0 * math.pi / angular_speed

    one_period = continue_solution(
        positions,
        velocities,
        masses,
        period,
        order=12,
        max_step=0.12,
    )

    assert len(one_period.steps) < 80
    assert np.linalg.norm(
        one_period.final_state - one_period.states[0],
        ord=np.inf,
    ) < 1e-12
    assert max(step.truncation_indicator for step in one_period.steps) < 1e-20
    assert min(step.min_pair_distance for step in one_period.steps) > (
        0.999999999 * math.sqrt(3.0)
    )

    def cyclic_atlas_state(target_time):
        reduced_time = target_time % period
        if math.isclose(reduced_time, period, rel_tol=0.0, abs_tol=1e-13):
            reduced_time = 0.0
        step_index = int(np.searchsorted(one_period.times, reduced_time, side="right") - 1)
        step_index = min(step_index, len(one_period.steps) - 1)
        step = one_period.steps[step_index]
        local_time = reduced_time - step.start_time
        return step.series.state_at(local_time)

    for cycle_count, phase in (
        (0, 0.07),
        (1, 0.31 * period),
        (3, 0.73 * period),
        (8, period - 1e-6),
    ):
        target_time = cycle_count * period + phase
        final_angles = angles + angular_speed * target_time
        exact_positions = np.stack([np.cos(final_angles), np.sin(final_angles)], axis=1)
        exact_velocities = angular_speed * np.stack(
            [-np.sin(final_angles), np.cos(final_angles)],
            axis=1,
        )
        exact_state = np.concatenate(
            [exact_positions.reshape(-1), exact_velocities.reshape(-1)]
        )

        np.testing.assert_allclose(
            cyclic_atlas_state(target_time),
            exact_state,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            cyclic_atlas_state(target_time + 5.0 * period),
            cyclic_atlas_state(target_time),
            rtol=0.0,
            atol=1e-13,
        )


def test_continuation_error_decreases_with_series_order():
    masses, positions, velocities = _general_initial_data()
    t_final = 0.18
    reference = integrate_reference(positions, velocities, masses, t_final)

    low_order = continue_solution(positions, velocities, masses, t_final, order=6, max_step=0.035)
    high_order = continue_solution(positions, velocities, masses, t_final, order=14, max_step=0.035)

    low_error = np.linalg.norm(low_order.final_state - reference, ord=np.inf)
    high_error = np.linalg.norm(high_order.final_state - reference, ord=np.inf)

    assert high_error < low_error * 1e-4


def test_backward_continuation_reverses_forward_continuation():
    masses, positions, velocities = _general_initial_data()
    forward = continue_solution(positions, velocities, masses, 0.12, order=16, max_step=0.03)
    final_positions, final_velocities = forward.final_positions_velocities()

    backward = continue_solution(final_positions, final_velocities, masses, -0.12, order=16, max_step=0.03)
    initial_state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])

    assert np.linalg.norm(backward.final_state - initial_state, ord=np.inf) < 1e-11


def test_step_choice_shrinks_near_close_approach():
    masses, positions, velocities = _general_initial_data()
    close_positions = positions.copy()
    close_positions[1] = close_positions[0] + np.array([1e-3, 0.0, 0.0])

    ordinary_step = abs(choose_step_size(positions, velocities, masses, 1.0, max_step=0.05))
    close_step = abs(choose_step_size(close_positions, velocities, masses, 1.0, max_step=0.05))

    assert close_step < ordinary_step * 0.1
