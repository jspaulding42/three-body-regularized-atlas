import numpy as np

from three_body_symmetry.dynamics import accelerations
from three_body_symmetry.ks_binary_chart import (
    SpatialKSBinaryChartState,
    construct_ks_chart,
    integrate_relative_kepler_spatial,
    ks_analytic_coordinate_accelerations,
    ks_binary_chart_to_spatial,
    ks_gauge_rotate,
    ks_horizontal_constraint,
    ks_interval_position_atlas,
    ks_interval_position_chart,
    ks_interval_state_atlas,
    ks_interval_state_chart,
    ks_lift_position,
    ks_lift_position_on_branch,
    ks_matrix,
    ks_mu,
    ks_pair_energy_constraint,
    ks_pair_energy_constraint_derivative,
    ks_project,
    ks_regularized_rhs,
    ks_velocity,
    physical_to_ks,
    physical_to_ks_on_branch,
    projected_acceleration_from_ks_lift,
    regularized_ks_binary_chart_rhs,
    regularized_ks_binary_u_acceleration,
    relative_acceleration_from_ks_acceleration,
    spatial_accelerations_from_ks_binary_rhs,
    spatial_to_ks_binary_chart,
)
from three_body_symmetry.levi_civita import (
    lc_square,
    lc_velocity,
    levi_civita_mu,
    projected_acceleration_from_lift,
)


def _spatial_binary_state():
    masses = np.array([0.8, 1.2, 1.7])
    positions = np.array(
        [
            [-0.025, 0.015, 0.01],
            [0.035, -0.02, 0.03],
            [1.2, 0.4, -0.25],
        ]
    )
    velocities = np.array(
        [
            [0.15, -0.08, 0.03],
            [-0.06, 0.11, 0.02],
            [0.01, -0.03, 0.04],
        ]
    )
    return masses, positions, velocities


def test_ks_projection_matches_spatial_relative_kepler_reference():
    mu = 1.7
    relative_position = np.array([0.8, 0.3, 0.4])
    relative_velocity = np.array([-0.2, 0.6, 0.1])
    u0, u_velocity0, energy = physical_to_ks(relative_position, relative_velocity, mu)
    chart = construct_ks_chart(u0, u_velocity0, energy, order=28)

    s_value = 0.03
    physical_time = chart.physical_time_at(s_value)
    reference = integrate_relative_kepler_spatial(
        relative_position,
        relative_velocity,
        mu,
        physical_time,
    )

    assert abs(np.linalg.norm(ks_project(u0)) - np.dot(u0, u0)) < 1e-14
    assert abs(ks_horizontal_constraint(u0, u_velocity0)) < 1e-14
    assert abs(ks_mu(u0, u_velocity0, energy) - mu) < 1e-14
    assert np.linalg.norm(chart.relative_state_at(s_value) - reference, ord=np.inf) < 1e-12


def test_ks_projected_lifted_acceleration_equals_spatial_kepler_force():
    u = np.array([0.7, 0.2, -0.4, 0.3])
    physical_velocity = np.array([0.1, 0.8, -0.35])
    u_velocity = 0.25 * ks_matrix(u).T @ physical_velocity
    energy = -1.0
    mu = ks_mu(u, u_velocity, energy)
    relative_position = ks_project(u)

    projected = projected_acceleration_from_ks_lift(u, u_velocity, energy)
    expected = -mu * relative_position / np.linalg.norm(relative_position) ** 3

    assert mu > 0.0
    assert abs(ks_horizontal_constraint(u, u_velocity)) < 1e-14
    assert np.linalg.norm(projected - expected, ord=np.inf) < 1e-12


def test_ks_planar_slice_agrees_with_levi_civita_chart():
    z = np.array([0.7, 0.2])
    z_velocity = np.array([0.1, 0.8])
    u = np.array([z[0], z[1], 0.0, 0.0])
    u_velocity = np.array([z_velocity[0], z_velocity[1], 0.0, 0.0])
    energy = -1.0

    projected_position = ks_project(u)
    projected_velocity = ks_velocity(u, u_velocity)
    projected_acceleration = projected_acceleration_from_ks_lift(u, u_velocity, energy)

    assert np.linalg.norm(projected_position[:2] - lc_square(z), ord=np.inf) < 1e-14
    assert abs(projected_position[2]) < 1e-14
    assert np.linalg.norm(projected_velocity[:2] - lc_velocity(z, z_velocity), ord=np.inf) < 1e-14
    assert abs(projected_velocity[2]) < 1e-14
    assert abs(ks_mu(u, u_velocity, energy) - levi_civita_mu(z, z_velocity, energy)) < 1e-14
    assert np.linalg.norm(
        projected_acceleration[:2] - projected_acceleration_from_lift(z, z_velocity, energy),
        ord=np.inf,
    ) < 1e-12
    assert abs(projected_acceleration[2]) < 1e-14


def test_ks_gauge_rotation_preserves_projection_velocity_and_constraint():
    mu = 2.3
    relative_position = np.array([0.5, -0.7, 0.4])
    relative_velocity = np.array([0.2, 0.1, -0.6])
    u, u_velocity, energy = physical_to_ks(relative_position, relative_velocity, mu)

    rotated_u = ks_gauge_rotate(u, 0.73)
    rotated_u_velocity = ks_gauge_rotate(u_velocity, 0.73)

    assert np.linalg.norm(ks_project(rotated_u) - relative_position, ord=np.inf) < 1e-14
    assert np.linalg.norm(
        ks_velocity(rotated_u, rotated_u_velocity) - relative_velocity,
        ord=np.inf,
    ) < 1e-14
    assert abs(ks_horizontal_constraint(rotated_u, rotated_u_velocity)) < 1e-14
    assert abs(ks_mu(rotated_u, rotated_u_velocity, energy) - mu) < 1e-14


def test_ks_lift_position_covers_negative_x_branch():
    relative_position = np.array([-1.0, 0.2, -0.3])
    u = ks_lift_position(relative_position)

    assert np.linalg.norm(ks_project(u) - relative_position, ord=np.inf) < 1e-14
    assert abs(np.linalg.norm(relative_position) - np.dot(u, u)) < 1e-14


def test_ks_interval_positive_branch_contains_point_lift():
    relative_position = np.array([0.8, 0.3, 0.4])
    position_box = np.array(
        [
            [0.79, 0.81],
            [0.29, 0.31],
            [0.39, 0.41],
        ]
    )
    chart = ks_interval_position_chart(position_box, branch="positive_x")

    assert chart.certified
    assert chart.branch_certificate.branch == "positive_x"
    assert chart.contains_point(relative_position)
    assert any(atlas_chart.contains_point(relative_position) for atlas_chart in ks_interval_position_atlas(position_box))


def test_ks_interval_negative_branch_covers_negative_x_axis_box():
    relative_position = np.array([-1.0, 0.2, -0.3])
    position_box = np.array(
        [
            [-1.05, -0.95],
            [0.19, 0.21],
            [-0.31, -0.29],
        ]
    )
    positive = ks_interval_position_chart(position_box, branch="positive_x")
    negative = ks_interval_position_chart(position_box, branch="negative_x")
    atlas = ks_interval_position_atlas(position_box)

    assert not positive.certified
    assert positive.branch_certificate.branch == "positive_x"
    assert negative.certified
    assert negative.contains_point(relative_position)
    assert len(atlas) == 1
    assert atlas[0].branch_certificate.branch == "negative_x"
    assert atlas[0].contains_point(relative_position)


def test_ks_interval_atlas_rejects_collision_containing_box():
    position_box = np.array(
        [
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1],
        ]
    )
    atlas = ks_interval_position_atlas(position_box)

    assert atlas == ()


def test_ks_positive_and_negative_position_branches_are_gauge_equivalent():
    relative_position = np.array([0.4, -0.7, 0.5])
    positive = ks_lift_position_on_branch(relative_position, branch="positive_x")
    negative = ks_lift_position_on_branch(relative_position, branch="negative_x")
    vertical = np.array([-positive[3], positive[2], -positive[1], positive[0]])
    rho = np.dot(positive, positive)
    angle = np.arctan2(np.dot(vertical, negative), np.dot(positive, negative))

    assert np.linalg.norm(ks_project(positive) - relative_position, ord=np.inf) < 1e-14
    assert np.linalg.norm(ks_project(negative) - relative_position, ord=np.inf) < 1e-14
    assert abs(np.linalg.norm(relative_position) - rho) < 1e-14
    assert np.linalg.norm(ks_gauge_rotate(positive, angle) - negative, ord=np.inf) < 1e-14


def test_ks_interval_state_lift_contains_point_energy_and_horizontal_constraint():
    mu = 1.7
    relative_position = np.array([0.8, 0.3, 0.4])
    relative_velocity = np.array([-0.2, 0.6, 0.1])
    position_box = np.array(
        [
            [0.79, 0.81],
            [0.29, 0.31],
            [0.39, 0.41],
        ]
    )
    velocity_box = np.array(
        [
            [-0.21, -0.19],
            [0.59, 0.61],
            [0.09, 0.11],
        ]
    )
    point_u, point_u_velocity, point_energy = physical_to_ks_on_branch(
        relative_position,
        relative_velocity,
        mu,
        branch="positive_x",
    )
    chart = ks_interval_state_chart(
        position_box,
        velocity_box,
        mu,
        branch="positive_x",
    )

    assert chart.certified
    assert chart.branch_certificate.branch == "positive_x"
    assert chart.horizontal_constraint_contains_zero
    assert chart.contains_point(relative_position, relative_velocity, mu)
    assert chart.energy.lower <= point_energy <= chart.energy.upper
    assert all(
        interval.lower <= value <= interval.upper
        for interval, value in zip(chart.u, point_u)
    )
    assert all(
        interval.lower <= value <= interval.upper
        for interval, value in zip(chart.u_velocity, point_u_velocity)
    )


def test_ks_interval_state_atlas_covers_both_overlapping_gauge_branches():
    mu = 2.3
    relative_position = np.array([0.4, -0.7, 0.5])
    relative_velocity = np.array([0.2, 0.1, -0.6])
    position_box = np.array(
        [
            [0.39, 0.41],
            [-0.71, -0.69],
            [0.49, 0.51],
        ]
    )
    velocity_box = np.array(
        [
            [0.19, 0.21],
            [0.09, 0.11],
            [-0.61, -0.59],
        ]
    )
    atlas = ks_interval_state_atlas(position_box, velocity_box, mu)

    assert {chart.branch_certificate.branch for chart in atlas} == {
        "positive_x",
        "negative_x",
    }
    assert all(chart.certified for chart in atlas)
    assert all(chart.horizontal_constraint_contains_zero for chart in atlas)
    assert all(chart.contains_point(relative_position, relative_velocity, mu) for chart in atlas)


def test_ks_interval_state_lift_reports_branch_failure_without_boolean_witness():
    mu = 1.0
    position_box = np.array(
        [
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1],
        ]
    )
    velocity_box = np.array(
        [
            [-0.2, 0.2],
            [-0.2, 0.2],
            [-0.2, 0.2],
        ]
    )
    chart = ks_interval_state_chart(
        position_box,
        velocity_box,
        mu,
        branch="positive_x",
    )

    assert not chart.certified
    assert not chart.branch_certificate.certified
    assert chart.branch_certificate.branch == "positive_x"
    assert ks_interval_state_atlas(position_box, velocity_box, mu) == ()


def test_spatial_ks_binary_chart_round_trips_noncollision_state():
    masses, positions, velocities = _spatial_binary_state()
    chart = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    projected_positions, projected_velocities = ks_binary_chart_to_spatial(chart)

    assert abs(ks_pair_energy_constraint(chart)) < 1e-14
    assert abs(ks_horizontal_constraint(chart.u, chart.u_velocity)) < 1e-14
    assert np.linalg.norm(projected_positions - positions, ord=np.inf) < 1e-14
    assert np.linalg.norm(projected_velocities - velocities, ord=np.inf) < 1e-14


def test_spatial_ks_binary_rhs_projects_to_newtonian_accelerations():
    masses, positions, velocities = _spatial_binary_state()
    chart = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    derivative = regularized_ks_binary_chart_rhs(chart)
    projected_acceleration = spatial_accelerations_from_ks_binary_rhs(chart, derivative)

    assert np.linalg.norm(projected_acceleration - accelerations(positions, masses), ord=np.inf) < 1e-12
    assert abs(ks_pair_energy_constraint_derivative(chart, derivative)) < 1e-12


def test_regularized_ks_binary_u_acceleration_agrees_with_physical_split():
    masses, positions, velocities = _spatial_binary_state()
    chart = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    projected_positions, _projected_velocities = ks_binary_chart_to_spatial(chart)
    physical_acceleration = accelerations(projected_positions, masses)
    relative_acceleration = physical_acceleration[1] - physical_acceleration[0]

    projected_split = relative_acceleration_from_ks_acceleration(
        chart.u,
        chart.u_velocity,
        regularized_ks_binary_u_acceleration(chart),
    )

    assert np.linalg.norm(projected_split - relative_acceleration, ord=np.inf) < 1e-12


def test_spatial_ks_regularized_rhs_is_finite_at_exact_binary_collision():
    masses = np.array([0.8, 1.2, 1.7])
    pair_mass = masses[0] + masses[1]
    state = SpatialKSBinaryChartState(
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

    derivative = regularized_ks_binary_chart_rhs(state)

    assert abs(ks_pair_energy_constraint(state)) < 1e-14
    assert np.all(np.isfinite(derivative.u))
    assert np.all(np.isfinite(derivative.u_velocity))
    assert np.all(np.isfinite(derivative.binary_center))
    assert np.all(np.isfinite(derivative.binary_center_velocity))
    assert np.all(np.isfinite(derivative.third_offset))
    assert np.all(np.isfinite(derivative.third_offset_velocity))
    assert derivative.physical_time == 0.0
    assert np.linalg.norm(derivative.u_velocity, ord=np.inf) == 0.0
    assert abs(ks_pair_energy_constraint_derivative(state, derivative)) < 1e-14


def test_spatial_ks_analytic_third_body_fields_have_collision_limit():
    masses = np.array([0.8, 1.2, 1.7])
    pair_mass = masses[0] + masses[1]
    third_offset = np.array([1.3, 0.4, -0.2])

    def relative_perturbation(epsilon):
        state = SpatialKSBinaryChartState(
            masses=masses,
            pair=(0, 1),
            u=np.array([epsilon, 0.0, 0.0, 0.0]),
            u_velocity=np.array([np.sqrt(pair_mass / 2.0), 0.0, 0.0, 0.0]),
            pair_energy=0.0,
            binary_center=np.array([0.0, 0.0, 0.0]),
            binary_center_velocity=np.array([0.0, 0.0, 0.0]),
            third_offset=third_offset,
            third_offset_velocity=np.array([0.0, 0.0, 0.0]),
        )
        return ks_analytic_coordinate_accelerations(state)[2]

    collision_state = SpatialKSBinaryChartState(
        masses=masses,
        pair=(0, 1),
        u=np.zeros(4),
        u_velocity=np.array([np.sqrt(pair_mass / 2.0), 0.0, 0.0, 0.0]),
        pair_energy=0.0,
        binary_center=np.array([0.0, 0.0, 0.0]),
        binary_center_velocity=np.array([0.0, 0.0, 0.0]),
        third_offset=third_offset,
        third_offset_velocity=np.array([0.0, 0.0, 0.0]),
    )
    collision_limit = ks_analytic_coordinate_accelerations(collision_state)[2]

    assert np.linalg.norm(collision_limit, ord=np.inf) == 0.0
    assert np.linalg.norm(relative_perturbation(1e-3) - collision_limit, ord=np.inf) < 1e-5
    assert np.linalg.norm(relative_perturbation(1e-4) - collision_limit, ord=np.inf) < 1e-7


def test_ks_collision_lift_has_finite_regularized_rhs_and_series():
    u0 = np.zeros(4)
    u_velocity0 = np.array([1.0, 0.0, 0.0, 0.0])
    energy = -0.5
    derivative = ks_regularized_rhs(u0, u_velocity0, energy)
    chart = construct_ks_chart(u0, u_velocity0, energy, order=10)

    assert np.all(np.isfinite(derivative.u))
    assert np.all(np.isfinite(derivative.u_velocity))
    assert derivative.physical_time == 0.0
    assert np.all(np.isfinite(chart.u))
    assert np.all(np.isfinite(chart.u_velocity))
    assert np.all(np.isfinite(chart.physical_time))
    assert np.linalg.norm(chart.relative_position_at(0.0), ord=np.inf) == 0.0
    assert abs(chart.physical_time[3] - 1.0 / 3.0) < 1e-14
    assert chart.physical_time_at(0.1) > 0.0
