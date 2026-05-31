import numpy as np

from three_body_symmetry.binary_chart import (
    RegularizedBinaryCollisionChartState,
    exact_binary_collision_interval_chart,
    planar_interval_to_regularized_binary_collision_chart,
    planar_to_regularized_binary_collision_chart,
    planar_accelerations_from_regularized_chart_rhs,
    regularized_binary_collision_chart_rhs,
    regularized_binary_collision_chart_to_planar,
)
from three_body_symmetry.binary_series import (
    construct_interval_regularized_binary_taylor_solution,
    construct_interval_regularized_binary_taylor_solution_from_intervals,
    construct_regularized_binary_taylor_solution,
    integrate_regularized_binary_reference,
    pair_energy_constraint_coefficients,
    regularized_rhs_interval_coefficients,
    regularized_rhs_coefficients,
)
from three_body_symmetry.dynamics import accelerations
from three_body_symmetry.series import construct_taylor_solution, integrate_reference


def _planar_state():
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


def test_regularized_binary_taylor_coefficients_satisfy_rhs_recurrence():
    masses, positions, velocities = _planar_state()
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    solution = construct_regularized_binary_taylor_solution(initial, order=10)

    for n in range(9):
        rhs = regularized_rhs_coefficients(solution, n)
        assert np.linalg.norm((n + 1) * solution.z[n + 1] - rhs.z[n], ord=np.inf) < 1e-13
        assert np.linalg.norm((n + 1) * solution.z_velocity[n + 1] - rhs.z_velocity[n], ord=np.inf) < 1e-13
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


def test_regularized_binary_taylor_matches_lifted_reference_integration():
    masses, positions, velocities = _planar_state()
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    solution = construct_regularized_binary_taylor_solution(initial, order=16)
    s_value = 0.02

    reference = integrate_regularized_binary_reference(initial, s_value)

    assert np.linalg.norm(solution.vector_at(s_value) - reference, ord=np.inf) < 1e-12


def test_regularized_binary_taylor_projection_matches_newtonian_motion():
    masses, positions, velocities = _planar_state()
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    solution = construct_regularized_binary_taylor_solution(initial, order=18)
    s_value = 0.015
    projected_state = solution.state_at(s_value)
    projected_positions, projected_velocities = regularized_binary_collision_chart_to_planar(projected_state)
    physical_time = solution.physical_time_at(s_value)
    reference = integrate_reference(positions, velocities, masses, physical_time)
    reference_positions = reference[:6].reshape(3, 2)
    reference_velocities = reference[6:].reshape(3, 2)

    assert physical_time > 0.0
    assert np.linalg.norm(projected_positions - reference_positions, ord=np.inf) < 1e-12
    assert np.linalg.norm(projected_velocities - reference_velocities, ord=np.inf) < 1e-12


def test_regularized_binary_taylor_starts_at_exact_binary_collision():
    initial = _exact_collision_state()
    solution = construct_regularized_binary_taylor_solution(initial, order=14)
    s_value = 0.02
    reference = integrate_regularized_binary_reference(initial, s_value)
    constraint = pair_energy_constraint_coefficients(solution, 14)

    assert np.all(np.isfinite(solution.z))
    assert np.all(np.isfinite(solution.z_velocity))
    assert np.all(np.isfinite(solution.pair_energy))
    assert solution.physical_time[1] == 0.0
    assert solution.physical_time[2] == 0.0
    assert abs(solution.physical_time[3] - np.dot(initial.z_velocity, initial.z_velocity) / 3.0) < 1e-14
    assert np.linalg.norm(constraint, ord=np.inf) < 1e-12
    assert np.linalg.norm(solution.vector_at(s_value) - reference, ord=np.inf) < 1e-11


def test_exact_binary_collision_chart_projects_to_two_sided_newtonian_branches():
    initial = _exact_collision_state()
    solution = construct_regularized_binary_taylor_solution(initial, order=24)
    constraint = pair_energy_constraint_coefficients(solution, 24)

    assert np.linalg.norm(solution.state_at(0.0).z, ord=np.inf) == 0.0
    assert solution.physical_time_at(0.0) == 0.0
    assert solution.physical_time[3] == (
        np.dot(initial.z_velocity, initial.z_velocity) / 3.0
    )
    assert np.linalg.norm(constraint, ord=np.inf) < 1e-12

    for s_value in (-0.03, -0.02, 0.02, 0.03):
        state = solution.state_at(s_value)
        physical_time = solution.physical_time_at(s_value)
        positions, _velocities = regularized_binary_collision_chart_to_planar(state)
        rhs = regularized_binary_collision_chart_rhs(state)
        projected_acceleration = planar_accelerations_from_regularized_chart_rhs(
            state,
            rhs,
        )

        assert np.sign(physical_time) == np.sign(s_value)
        assert state.rho > 0.0
        assert abs(np.linalg.norm(positions[1] - positions[0]) - state.rho) < 1e-18
        assert np.linalg.norm(positions[2] - positions[0]) > 1.5
        assert np.linalg.norm(positions[2] - positions[1]) > 1.5
        assert np.linalg.norm(
            projected_acceleration - accelerations(positions, initial.masses),
            ord=np.inf,
        ) < 2e-8


def test_separated_binary_chart_glues_to_ordinary_taylor_pieces_on_compact_interval():
    initial = _exact_collision_state()
    binary_solution = construct_regularized_binary_taylor_solution(initial, order=40)
    compact_s_windows = ((-0.05, -0.045), (-0.04, -0.035), (0.035, 0.04), (0.04, 0.045))
    local_residuals = []
    handoff_times = []

    for s_start, s_end in compact_s_windows:
        start_state = binary_solution.state_at(s_start)
        end_state = binary_solution.state_at(s_end)
        start_positions, start_velocities = regularized_binary_collision_chart_to_planar(
            start_state
        )
        end_positions, end_velocities = regularized_binary_collision_chart_to_planar(
            end_state
        )
        delta_t = binary_solution.physical_time_at(s_end) - binary_solution.physical_time_at(
            s_start
        )
        ordinary_solution = construct_taylor_solution(
            start_positions,
            start_velocities,
            initial.masses,
            order=40,
        )
        ordinary_positions = ordinary_solution.positions_at(delta_t)
        ordinary_velocities = ordinary_solution.velocities_at(delta_t)
        min_pair_distance = min(
            np.linalg.norm(start_positions[j] - start_positions[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        third_separations = (
            np.linalg.norm(start_positions[2] - start_positions[0]),
            np.linalg.norm(start_positions[2] - start_positions[1]),
        )

        assert np.sign(binary_solution.physical_time_at(s_start)) == np.sign(s_start)
        assert np.sign(binary_solution.physical_time_at(s_end)) == np.sign(s_end)
        assert delta_t > 0.0
        assert min_pair_distance > 0.0
        assert min(third_separations) > 1.5
        local_residuals.append(
            max(
                np.linalg.norm(ordinary_positions - end_positions, ord=np.inf),
                np.linalg.norm(ordinary_velocities - end_velocities, ord=np.inf),
            )
        )
        handoff_times.append(binary_solution.physical_time_at(s_start))
        handoff_times.append(binary_solution.physical_time_at(s_end))

    assert max(local_residuals) < 5.0e-12
    assert min(handoff_times) < 0.0 < max(handoff_times)
    assert sum(local_residuals) < 1.0e-10


def test_regularized_binary_taylor_constraint_is_preserved_for_noncollision_data():
    masses, positions, velocities = _planar_state()
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    solution = construct_regularized_binary_taylor_solution(initial, order=14)

    assert np.linalg.norm(pair_energy_constraint_coefficients(solution, 14), ord=np.inf) < 1e-12


def test_interval_regularized_binary_taylor_encloses_point_coefficients():
    masses, positions, velocities = _planar_state()
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    point_solution = construct_regularized_binary_taylor_solution(initial, order=8)
    interval_solution = construct_interval_regularized_binary_taylor_solution(initial, order=8)

    assert interval_solution.contains_point_solution(point_solution)


def test_interval_regularized_binary_taylor_accepts_interval_initial_chart():
    masses, positions, velocities = _planar_state()
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
    state_interval = tuple((float(value - 1e-13), float(value + 1e-13)) for value in state)
    interval_initial = planar_interval_to_regularized_binary_collision_chart(
        state_interval,
        masses,
        pair=(0, 1),
    )
    point_solution = construct_regularized_binary_taylor_solution(initial, order=6)
    interval_solution = construct_interval_regularized_binary_taylor_solution_from_intervals(
        interval_initial,
        order=6,
    )

    assert interval_initial.contains_point(initial)
    assert interval_solution.contains_point_solution(point_solution)


def test_interval_regularized_rhs_encloses_point_rhs_coefficients():
    masses, positions, velocities = _planar_state()
    initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    point_solution = construct_regularized_binary_taylor_solution(initial, order=7)
    interval_solution = construct_interval_regularized_binary_taylor_solution(initial, order=7)
    point_rhs = regularized_rhs_coefficients(point_solution, 6)
    interval_rhs = regularized_rhs_interval_coefficients(interval_solution, 6)

    vector_fields = [
        (interval_rhs.z, point_rhs.z),
        (interval_rhs.z_velocity, point_rhs.z_velocity),
        (interval_rhs.binary_center, point_rhs.binary_center),
        (interval_rhs.binary_center_velocity, point_rhs.binary_center_velocity),
        (interval_rhs.third_offset, point_rhs.third_offset),
        (interval_rhs.third_offset_velocity, point_rhs.third_offset_velocity),
    ]
    for interval_field, point_field in vector_fields:
        for index in np.ndindex(point_field.shape):
            assert interval_field[index].lower <= point_field[index] <= interval_field[index].upper
    for index in range(point_rhs.pair_energy.shape[0]):
        assert interval_rhs.pair_energy[index].lower <= point_rhs.pair_energy[index] <= interval_rhs.pair_energy[index].upper
        assert interval_rhs.physical_time[index].lower <= point_rhs.physical_time[index] <= interval_rhs.physical_time[index].upper


def test_interval_regularized_binary_taylor_starts_at_exact_collision():
    initial = _exact_collision_state()
    point_solution = construct_regularized_binary_taylor_solution(initial, order=8)
    interval_solution = construct_interval_regularized_binary_taylor_solution(initial, order=8)

    assert interval_solution.contains_point_solution(point_solution)
    assert interval_solution.physical_time[1].lower <= 0.0 <= interval_solution.physical_time[1].upper
    assert interval_solution.physical_time[2].lower <= 0.0 <= interval_solution.physical_time[2].upper


def test_exact_collision_interval_chart_seeds_interval_taylor_solution():
    initial = _exact_collision_state()
    interval_initial = exact_binary_collision_interval_chart(
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
    point_solution = construct_regularized_binary_taylor_solution(initial, order=8)
    interval_solution = construct_interval_regularized_binary_taylor_solution_from_intervals(
        interval_initial,
        order=8,
    )

    assert interval_initial.branch_certificate.certified
    assert interval_initial.branch_certificate.branch == "exact_binary_collision"
    assert interval_initial.contains_point(initial)
    assert interval_solution.contains_point_solution(point_solution)
    assert interval_solution.physical_time[1].lower <= 0.0 <= interval_solution.physical_time[1].upper
    assert interval_solution.physical_time[2].lower <= 0.0 <= interval_solution.physical_time[2].upper


def test_exact_collision_interval_chart_rejects_uncertified_lift_data():
    initial = _exact_collision_state()

    try:
        exact_binary_collision_interval_chart(
            initial.masses,
            pair=initial.pair,
            z_velocity=((0.1, 0.2), (0.0, 0.0)),
            pair_energy=initial.pair_energy,
            binary_center=initial.binary_center,
            binary_center_velocity=initial.binary_center_velocity,
            third_offset=initial.third_offset,
            third_offset_velocity=initial.third_offset_velocity,
        )
    except ValueError as error:
        assert "2|z_velocity|^2" in str(error)
    else:
        raise AssertionError("uncertified collision velocity interval was accepted")

    try:
        exact_binary_collision_interval_chart(
            initial.masses,
            pair=initial.pair,
            z_velocity=initial.z_velocity,
            pair_energy=initial.pair_energy,
            binary_center=initial.binary_center,
            binary_center_velocity=initial.binary_center_velocity,
            third_offset=((0.0, 0.0), (0.0, 0.0)),
            third_offset_velocity=initial.third_offset_velocity,
        )
    except ValueError as error:
        assert "third body" in str(error)
    else:
        raise AssertionError("unseparated third body interval was accepted")
