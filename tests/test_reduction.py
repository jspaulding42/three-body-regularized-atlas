import numpy as np

from three_body_symmetry.intervals import FloatInterval, interval_array_contains_point
from three_body_symmetry.reduction import (
    reconstruct_interval_from_center_of_mass_frame,
    reconstruct_from_center_of_mass_frame,
    reconstruct_sundman_solution_from_center_of_mass,
    reconstruct_taylor_solution_from_center_of_mass,
    reduce_interval_to_center_of_mass_frame,
    reduce_to_center_of_mass_frame,
)
from three_body_symmetry.series import construct_interval_taylor_solution_from_intervals, construct_taylor_solution
from three_body_symmetry.sundman import (
    construct_interval_sundman_taylor_solution_from_intervals,
    construct_sundman_taylor_solution,
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


def _max_abs(values):
    return float(np.max(np.abs(values)))


def _interval_box_around(values, radius):
    values = np.asarray(values, dtype=float)
    out = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        out[index] = FloatInterval(values[index] - radius, values[index] + radius)
    return out


def test_center_of_mass_reduction_certifies_zero_moments_and_round_trips_initial_state():
    masses, positions, velocities = _general_initial_data()

    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    reconstructed_positions, reconstructed_velocities = reconstruct_from_center_of_mass_frame(
        reduced.positions,
        reduced.velocities,
        reduced.center_position,
        reduced.center_velocity,
        time=0.0,
    )

    assert reduced.certificate.certified
    assert np.linalg.norm(np.sum(masses[:, None] * reduced.positions, axis=0), ord=np.inf) < 1e-14
    assert np.linalg.norm(np.sum(masses[:, None] * reduced.velocities, axis=0), ord=np.inf) < 1e-14
    assert np.linalg.norm(reconstructed_positions - positions, ord=np.inf) < 1e-14
    assert np.linalg.norm(reconstructed_velocities - velocities, ord=np.inf) < 1e-14


def test_center_of_mass_reconstruction_commutes_with_ordinary_taylor_lift():
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)

    original_solution = construct_taylor_solution(positions, velocities, masses, order=12)
    reduced_solution = construct_taylor_solution(reduced.positions, reduced.velocities, masses, order=12)
    reconstructed_solution = reconstruct_taylor_solution_from_center_of_mass(
        reduced_solution,
        reduced.center_position,
        reduced.center_velocity,
    )

    assert _max_abs(reconstructed_solution.position - original_solution.position) < 1e-12
    assert _max_abs(reconstructed_solution.velocity - original_solution.velocity) < 1e-12
    for time in (0.0, 0.003, 0.01):
        reduced_positions = reduced_solution.positions_at(time)
        reduced_velocities = reduced_solution.velocities_at(time)
        reconstructed_positions, reconstructed_velocities = reconstruct_from_center_of_mass_frame(
            reduced_positions,
            reduced_velocities,
            reduced.center_position,
            reduced.center_velocity,
            time=time,
        )
        assert np.linalg.norm(reconstructed_positions - original_solution.positions_at(time), ord=np.inf) < 1e-12
        assert np.linalg.norm(reconstructed_velocities - original_solution.velocities_at(time), ord=np.inf) < 1e-12


def test_center_of_mass_reconstruction_commutes_with_sundman_lift():
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)

    original_solution = construct_sundman_taylor_solution(positions, velocities, masses, order=12)
    reduced_solution = construct_sundman_taylor_solution(reduced.positions, reduced.velocities, masses, order=12)
    reconstructed_solution = reconstruct_sundman_solution_from_center_of_mass(
        reduced_solution,
        reduced.center_position,
        reduced.center_velocity,
    )

    assert _max_abs(reduced_solution.physical_time - original_solution.physical_time) < 1e-9
    assert _max_abs(reconstructed_solution.position - original_solution.position) < 1e-9
    assert _max_abs(reconstructed_solution.velocity - original_solution.velocity) < 1e-9
    for s_value in (0.0, 0.003, 0.01):
        time = reduced_solution.physical_time_at_s(s_value)
        reduced_positions = reduced_solution.positions_at_s(s_value)
        reduced_velocities = reduced_solution.velocities_at_s(s_value)
        reconstructed_positions, reconstructed_velocities = reconstruct_from_center_of_mass_frame(
            reduced_positions,
            reduced_velocities,
            reduced.center_position,
            reduced.center_velocity,
            time=time,
        )
        assert np.linalg.norm(reconstructed_positions - original_solution.positions_at_s(s_value), ord=np.inf) < 1e-9
        assert np.linalg.norm(reconstructed_velocities - original_solution.velocities_at_s(s_value), ord=np.inf) < 1e-9


def test_interval_center_of_mass_reduction_contains_point_reduction_and_initial_state():
    masses, positions, velocities = _general_initial_data()
    position_box = _interval_box_around(positions, 1e-12)
    velocity_box = _interval_box_around(velocities, 1e-12)

    point_reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    interval_reduced = reduce_interval_to_center_of_mass_frame(position_box, velocity_box, masses)
    reconstructed_positions, reconstructed_velocities = reconstruct_interval_from_center_of_mass_frame(
        interval_reduced.positions,
        interval_reduced.velocities,
        interval_reduced.center_position,
        interval_reduced.center_velocity,
        time=FloatInterval.point(0.0),
    )

    assert interval_reduced.certificate.certified
    assert interval_reduced.contains_point_reduction(point_reduced)
    assert interval_array_contains_point(reconstructed_positions, positions)
    assert interval_array_contains_point(reconstructed_velocities, velocities)


def test_interval_center_of_mass_reconstruction_contains_ordinary_taylor_path():
    masses, positions, velocities = _general_initial_data()
    position_box = _interval_box_around(positions, 1e-12)
    velocity_box = _interval_box_around(velocities, 1e-12)
    point_reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    interval_reduced = reduce_interval_to_center_of_mass_frame(position_box, velocity_box, masses)

    point_original = construct_taylor_solution(positions, velocities, masses, order=12)
    interval_reduced_solution = construct_interval_taylor_solution_from_intervals(
        interval_reduced.positions,
        interval_reduced.velocities,
        masses,
        order=12,
    )
    time = 0.004
    reconstructed_positions, reconstructed_velocities = reconstruct_interval_from_center_of_mass_frame(
        interval_reduced_solution.positions_at(time),
        interval_reduced_solution.velocities_at(time),
        interval_reduced.center_position,
        interval_reduced.center_velocity,
        time=FloatInterval.point(time),
    )

    assert interval_reduced.contains_point_reduction(point_reduced)
    assert interval_array_contains_point(reconstructed_positions, point_original.positions_at(time))
    assert interval_array_contains_point(reconstructed_velocities, point_original.velocities_at(time))


def test_interval_center_of_mass_reconstruction_contains_sundman_path():
    masses, positions, velocities = _general_initial_data()
    position_box = _interval_box_around(positions, 1e-12)
    velocity_box = _interval_box_around(velocities, 1e-12)
    point_reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    interval_reduced = reduce_interval_to_center_of_mass_frame(position_box, velocity_box, masses)

    point_original = construct_sundman_taylor_solution(positions, velocities, masses, order=12)
    point_reduced_solution = construct_sundman_taylor_solution(point_reduced.positions, point_reduced.velocities, masses, order=12)
    interval_reduced_solution = construct_interval_sundman_taylor_solution_from_intervals(
        interval_reduced.positions,
        interval_reduced.velocities,
        masses,
        order=12,
    )
    s_value = 0.004
    physical_time = interval_reduced_solution.physical_time_at_s(s_value)
    reconstructed_positions, reconstructed_velocities = reconstruct_interval_from_center_of_mass_frame(
        interval_reduced_solution.positions_at_s(s_value),
        interval_reduced_solution.velocities_at_s(s_value),
        interval_reduced.center_position,
        interval_reduced.center_velocity,
        time=physical_time,
    )

    assert interval_reduced.contains_point_reduction(point_reduced)
    assert physical_time.lower <= point_reduced_solution.physical_time_at_s(s_value) <= physical_time.upper
    assert interval_array_contains_point(reconstructed_positions, point_original.positions_at_s(s_value))
    assert interval_array_contains_point(reconstructed_velocities, point_original.velocities_at_s(s_value))
