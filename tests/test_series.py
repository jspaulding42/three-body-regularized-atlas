import numpy as np

from three_body_symmetry.intervals import FloatInterval
from three_body_symmetry.series import (
    acceleration_coefficients,
    acceleration_interval_coefficients,
    center_of_mass_coefficients,
    construct_interval_taylor_solution,
    construct_interval_taylor_solution_from_intervals,
    construct_taylor_solution,
    integrate_reference,
    linear_momentum_coefficients,
)


def test_taylor_series_matches_direct_integration_for_general_3d_data():
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

    solution = construct_taylor_solution(positions, velocities, masses, order=18)
    time = 0.01

    series_state = solution.state_at(time)
    reference_state = integrate_reference(positions, velocities, masses, time)

    assert np.linalg.norm(series_state - reference_state, ord=np.inf) < 1e-13


def test_series_coefficients_satisfy_newton_equations_by_recurrence():
    masses = np.array([2.0, 1.0, 0.5])
    positions = np.array(
        [
            [1.0, 0.0],
            [-0.2, 0.9],
            [-0.4, -0.7],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.2],
            [-0.1, -0.05],
            [0.3, -0.1],
        ]
    )

    solution = construct_taylor_solution(positions, velocities, masses, order=12)
    acceleration = acceleration_coefficients(solution.position, masses, 11)

    for n in range(11):
        assert np.linalg.norm((n + 1) * solution.position[n + 1] - solution.velocity[n], ord=np.inf) < 1e-14
        assert np.linalg.norm((n + 1) * solution.velocity[n + 1] - acceleration[n], ord=np.inf) < 1e-14


def test_classical_linear_invariants_are_exact_in_series_space():
    masses = np.array([1.0, 1.5, 2.0])
    positions = np.array(
        [
            [0.6, 0.2, -0.1],
            [-0.3, 0.5, 0.4],
            [0.0, -0.7, 0.2],
        ]
    )
    velocities = np.array(
        [
            [0.2, -0.1, 0.05],
            [0.0, 0.08, -0.02],
            [-0.1, 0.02, 0.03],
        ]
    )

    solution = construct_taylor_solution(positions, velocities, masses, order=14)
    center_of_mass = center_of_mass_coefficients(solution)
    momentum = linear_momentum_coefficients(solution)

    expected_com_velocity = momentum[0] / np.sum(masses)
    assert np.linalg.norm(center_of_mass[1] - expected_com_velocity, ord=np.inf) < 1e-15
    assert np.linalg.norm(center_of_mass[2:], ord=np.inf) < 1e-14
    assert np.linalg.norm(momentum[1:], ord=np.inf) < 5e-14


def test_interval_taylor_coefficients_enclose_point_taylor_coefficients():
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

    point_solution = construct_taylor_solution(positions, velocities, masses, order=8)
    interval_solution = construct_interval_taylor_solution(positions, velocities, masses, order=8)

    assert interval_solution.contains_point_solution(point_solution)
    assert np.all(np.isfinite(interval_solution.position_lower))
    assert np.all(np.isfinite(interval_solution.position_upper))
    assert np.all(interval_solution.position_upper >= interval_solution.position_lower)
    assert np.all(interval_solution.velocity_upper >= interval_solution.velocity_lower)


def test_interval_acceleration_coefficients_enclose_point_coefficients():
    masses = np.array([2.0, 1.0, 0.5])
    positions = np.array(
        [
            [1.0, 0.0],
            [-0.2, 0.9],
            [-0.4, -0.7],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.2],
            [-0.1, -0.05],
            [0.3, -0.1],
        ]
    )

    point_solution = construct_taylor_solution(positions, velocities, masses, order=6)
    point_acceleration = acceleration_coefficients(point_solution.position, masses, 5)
    interval_solution = construct_interval_taylor_solution(positions, velocities, masses, order=6)
    interval_acceleration = acceleration_interval_coefficients(interval_solution.position, masses, 5)

    for index in np.ndindex(point_acceleration.shape):
        assert interval_acceleration[index].lower <= point_acceleration[index] <= interval_acceleration[index].upper


def test_interval_taylor_accepts_interval_initial_data():
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
    position_intervals = np.empty_like(positions, dtype=object)
    velocity_intervals = np.empty_like(velocities, dtype=object)
    for index in np.ndindex(positions.shape):
        position_intervals[index] = FloatInterval(positions[index] - 1e-15, positions[index] + 1e-15)
        velocity_intervals[index] = FloatInterval(velocities[index] - 1e-15, velocities[index] + 1e-15)

    point_solution = construct_taylor_solution(positions, velocities, masses, order=6)
    interval_solution = construct_interval_taylor_solution_from_intervals(
        position_intervals,
        velocity_intervals,
        masses,
        order=6,
    )

    assert interval_solution.contains_point_solution(point_solution)
