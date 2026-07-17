from fractions import Fraction

import numpy as np

from three_body_symmetry.binary_chart import (
    BinaryCollisionChartState,
    RegularizedBinaryCollisionChartState,
    pair_energy_constraint,
    pair_energy_constraint_derivative,
    binary_collision_chart_rhs,
    binary_collision_chart_to_planar,
    coordinate_accelerations_from_planar,
    analytic_coordinate_accelerations,
    planar_accelerations_from_chart_rhs,
    planar_accelerations_from_regularized_chart_rhs,
    planar_interval_to_regularized_binary_collision_chart_atlas,
    planar_interval_to_regularized_binary_collision_chart,
    planar_to_binary_collision_chart,
    planar_to_regularized_binary_collision_chart,
    relative_acceleration_from_z_acceleration,
    regularized_binary_collision_chart_rhs,
    regularized_binary_collision_chart_to_planar,
    regularized_z_acceleration,
    z_acceleration_from_relative_acceleration,
)
from three_body_symmetry.dynamics import accelerations
from three_body_symmetry.certificate_checker import (
    _planar_lc_mass_ratio_arithmetic_exact,
)
from three_body_symmetry.intervals import FloatInterval
from three_body_symmetry.levi_civita import lc_square, levi_civita_mu


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


def test_binary_collision_chart_round_trips_planar_state():
    masses, positions, velocities = _planar_state()

    chart = planar_to_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    projected_positions, projected_velocities = binary_collision_chart_to_planar(chart)

    assert np.linalg.norm(projected_positions - positions, ord=np.inf) < 1e-14
    assert np.linalg.norm(projected_velocities - velocities, ord=np.inf) < 1e-14


def test_chart_rhs_projects_to_newtonian_accelerations():
    masses, positions, velocities = _planar_state()
    chart = planar_to_binary_collision_chart(positions, velocities, masses, pair=(0, 1))

    derivative = binary_collision_chart_rhs(chart)
    projected_acceleration = planar_accelerations_from_chart_rhs(chart, derivative)

    assert np.linalg.norm(projected_acceleration - accelerations(positions, masses), ord=np.inf) < 1e-12


def test_relative_acceleration_transform_is_reversible_away_from_collision():
    masses, positions, velocities = _planar_state()
    chart = planar_to_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    _center_acceleration, _third_acceleration, relative_acceleration = coordinate_accelerations_from_planar(chart)

    z_acceleration = z_acceleration_from_relative_acceleration(
        chart.z,
        chart.z_velocity,
        relative_acceleration,
    )
    projected = relative_acceleration_from_z_acceleration(chart.z, chart.z_velocity, z_acceleration)

    assert np.linalg.norm(projected - relative_acceleration, ord=np.inf) < 1e-12


def test_third_body_perturbation_stays_regular_in_collision_scaling():
    masses = np.array([0.8, 1.2, 1.7])
    pair_mass = masses[0] + masses[1]
    energy = -0.3
    binary_center = np.array([0.0, 0.0])
    binary_center_velocity = np.array([0.0, 0.0])
    third_offset = np.array([1.3, 0.4])
    third_offset_velocity = np.array([0.0, 0.0])

    def near_collision_z_acceleration(epsilon):
        z = np.array([epsilon, 0.0])
        rho = float(np.dot(z, z))
        z_velocity = np.array([np.sqrt((pair_mass + energy * rho) / 2.0), 0.0])
        chart = BinaryCollisionChartState(
            masses=masses,
            pair=(0, 1),
            z=z,
            z_velocity=z_velocity,
            binary_center=binary_center,
            binary_center_velocity=binary_center_velocity,
            third_offset=third_offset,
            third_offset_velocity=third_offset_velocity,
        )
        return binary_collision_chart_rhs(chart).z_velocity

    coarse_error = np.linalg.norm(near_collision_z_acceleration(1e-2) - np.array([0.5 * energy * 1e-2, 0.0]))
    fine_error = np.linalg.norm(near_collision_z_acceleration(1e-3) - np.array([0.5 * energy * 1e-3, 0.0]))

    assert np.isfinite(coarse_error)
    assert np.isfinite(fine_error)
    assert fine_error < coarse_error * 0.02


def test_near_collision_binary_chart_matches_isolated_lc_limit_when_third_far():
    masses = np.array([0.8, 1.2, 1.7])
    z = np.array([1e-3, 0.0])
    energy = -0.4
    pair_mass = masses[0] + masses[1]
    rho = float(np.dot(z, z))
    z_velocity = np.array([np.sqrt((pair_mass + energy * rho) / 2.0), 0.0])
    chart = BinaryCollisionChartState(
        masses=masses,
        pair=(0, 1),
        z=z,
        z_velocity=z_velocity,
        binary_center=np.array([0.0, 0.0]),
        binary_center_velocity=np.array([0.0, 0.0]),
        third_offset=np.array([100.0, 50.0]),
        third_offset_velocity=np.array([0.0, 0.0]),
    )

    derivative = binary_collision_chart_rhs(chart)
    expected_mu = levi_civita_mu(z, z_velocity, energy)

    assert abs(expected_mu - pair_mass) < 1e-12
    assert np.linalg.norm(lc_square(z), ord=np.inf) <= 1e-6
    assert np.linalg.norm(derivative.z_velocity - 0.5 * energy * z, ord=np.inf) < 1e-12


def test_regularized_chart_round_trips_planar_state():
    masses, positions, velocities = _planar_state()

    chart = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    projected_positions, projected_velocities = regularized_binary_collision_chart_to_planar(chart)

    assert np.linalg.norm(projected_positions - positions, ord=np.inf) < 1e-14
    assert np.linalg.norm(projected_velocities - velocities, ord=np.inf) < 1e-14
    assert abs(pair_energy_constraint(chart)) < 1e-14


def test_interval_regularized_chart_lift_encloses_point_lift():
    masses, positions, velocities = _planar_state()
    point_chart = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    point_state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
    state_interval = tuple((float(value - 1e-12), float(value + 1e-12)) for value in point_state)

    interval_chart = planar_interval_to_regularized_binary_collision_chart(
        state_interval,
        masses,
        pair=(0, 1),
    )

    assert interval_chart.contains_point(point_chart)
    assert interval_chart.branch_certificate is not None
    assert interval_chart.branch_certificate.certified
    assert interval_chart.branch_certificate.branch == "principal_lower_half"


def test_interval_regularized_chart_lift_records_lc_branch_certificates():
    masses = np.array([1.0, 1.0, 1.0])
    cases = [
        ((-1.2, -0.8), (0.1, 0.2), "principal_upper_half", True),
        ((-1.2, -0.8), (-0.2, -0.1), "principal_lower_half", True),
        ((0.8, 1.2), (-0.1, 0.1), "principal_right_half", True),
        ((-1.2, -0.8), (-0.1, 0.1), "principal_branch_cut_overlap", False),
    ]

    for x_interval, y_interval, branch, certified in cases:
        state_interval = (
            (0.0, 0.0),
            (0.0, 0.0),
            x_interval,
            y_interval,
            (3.0, 3.0),
            (0.0, 0.0),
        ) + ((0.0, 0.0),) * 6

        interval_chart = planar_interval_to_regularized_binary_collision_chart(
            state_interval,
            masses,
            pair=(0, 1),
        )

        assert interval_chart.branch_certificate is not None
        assert interval_chart.branch_certificate.branch == branch
        assert interval_chart.branch_certificate.certified is certified


def test_interval_regularized_chart_atlas_splits_branch_cut_overlap():
    masses = np.array([1.0, 1.0, 1.0])
    state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-1.2, -0.8),
        (-0.2, 0.2),
        (3.0, 3.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    upper_positions = np.array([[0.0, 0.0], [-1.0, 0.1], [3.0, 0.0]])
    lower_positions = np.array([[0.0, 0.0], [-1.0, -0.1], [3.0, 0.0]])
    velocities = np.zeros((3, 2))
    upper_point = planar_to_regularized_binary_collision_chart(upper_positions, velocities, masses, pair=(0, 1))
    lower_point = planar_to_regularized_binary_collision_chart(lower_positions, velocities, masses, pair=(0, 1))

    single_lift = planar_interval_to_regularized_binary_collision_chart(
        state_interval,
        masses,
        pair=(0, 1),
    )
    atlas = planar_interval_to_regularized_binary_collision_chart_atlas(
        state_interval,
        masses,
        pair=(0, 1),
    )

    assert single_lift.branch_certificate is not None
    assert not single_lift.branch_certificate.certified
    assert len(atlas) == 2
    assert {chart.branch_certificate.branch for chart in atlas if chart.branch_certificate is not None} == {
        "atlas_upper_closed_half",
        "atlas_lower_closed_half",
    }
    assert all(chart.branch_certificate is not None and chart.branch_certificate.certified for chart in atlas)
    assert any(chart.contains_point(upper_point) for chart in atlas)
    assert any(chart.contains_point(lower_point) for chart in atlas)


def test_interval_regularized_chart_atlas_uses_one_closed_half_plane_patch():
    masses = np.array([1.0, 1.0, 1.0])
    cases = (
        ((0.0, 0.2), "principal_upper_half"),
        ((-0.2, 0.0), "principal_lower_half"),
        ((0.0, 0.0), "principal_upper_half"),
    )
    for y_interval, expected_branch in cases:
        state_interval = (
            (0.0, 0.0),
            (0.0, 0.0),
            (-1.2, -0.8),
            y_interval,
            (3.0, 3.0),
            (0.0, 0.0),
        ) + ((0.0, 0.0),) * 6

        atlas = planar_interval_to_regularized_binary_collision_chart_atlas(
            state_interval,
            masses,
            pair=(0, 1),
        )

        assert len(atlas) == 1
        assert atlas[0].branch_certificate is not None
        assert atlas[0].branch_certificate.certified
        assert atlas[0].branch_certificate.branch == expected_branch


def test_interval_binary_center_uses_exact_gated_direct_mass_ratios():
    masses = np.array([3.0, 3.0, 1.5])
    assert _planar_lc_mass_ratio_arithmetic_exact(tuple(masses), (0, 1))
    assert Fraction.from_float(1.0 / 6.0) != Fraction(1, 6)
    state_interval = (
        (0.1, 0.2),
        (-0.1, 0.1),
        (1.4, 1.5),
        (0.2, 0.3),
        (3.0, 3.1),
        (0.4, 0.5),
        (0.3, 0.4),
        (-0.2, -0.1),
        (-0.2, -0.1),
        (0.5, 0.6),
        (0.0, 0.1),
        (-0.4, -0.3),
    )

    chart = planar_interval_to_regularized_binary_collision_chart(
        state_interval,
        masses,
        pair=(0, 1),
    )
    expected_center_x = FloatInterval(*state_interval[0]).scale(0.5) + FloatInterval(
        *state_interval[2]
    ).scale(0.5)
    expected_center_velocity_x = FloatInterval(*state_interval[6]).scale(
        0.5
    ) + FloatInterval(*state_interval[8]).scale(0.5)

    assert chart.binary_center[0] == expected_center_x
    assert chart.binary_center_velocity[0] == expected_center_velocity_x


def test_interval_binary_center_encloses_nonrepresentable_exact_mass_ratio():
    masses = np.array([1.0, 2.0, 1.5])
    assert Fraction.from_float(1.0 / 3.0) != Fraction(1, 3)
    positions = ((0.1, -0.1), (1.4, 0.2), (3.0, 0.4))
    velocities = ((0.3, -0.2), (-0.2, 0.5), (0.0, -0.4))
    state_values = tuple(
        value
        for matrix in (positions, velocities)
        for row in matrix
        for value in row
    )
    state_interval = tuple((value, value) for value in state_values)

    chart = planar_interval_to_regularized_binary_collision_chart(
        state_interval,
        masses,
        pair=(0, 1),
    )
    first_mass_q = Fraction.from_float(float(masses[0]))
    second_mass_q = Fraction.from_float(float(masses[1]))
    pair_mass_q = first_mass_q + second_mass_q

    def exact_weighted(first_value: float, second_value: float) -> Fraction:
        return (
            first_mass_q * Fraction.from_float(first_value)
            + second_mass_q * Fraction.from_float(second_value)
        ) / pair_mass_q

    for intervals, values in (
        (chart.binary_center, positions),
        (chart.binary_center_velocity, velocities),
    ):
        for axis in range(2):
            exact = exact_weighted(values[0][axis], values[1][axis])
            assert Fraction.from_float(intervals[axis].lower) <= exact
            assert exact <= Fraction.from_float(intervals[axis].upper)


def test_interval_pair_energy_encloses_nonrepresentable_exact_pair_mass():
    masses = np.array([1.0, 2.0**-53, 1.5])
    pair_mass_q = Fraction.from_float(float(masses[0])) + Fraction.from_float(
        float(masses[1])
    )
    assert Fraction.from_float(float(masses[0] + masses[1])) != pair_mass_q
    state_values = (
        0.0,
        0.0,
        1.0,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    chart = planar_interval_to_regularized_binary_collision_chart(
        tuple((value, value) for value in state_values),
        masses,
        pair=(0, 1),
    )
    exact_pair_energy = -pair_mass_q  # rho=1 and relative velocity is zero.

    assert Fraction.from_float(chart.pair_energy.lower) <= exact_pair_energy
    assert exact_pair_energy <= Fraction.from_float(chart.pair_energy.upper)


def test_regularized_chart_rhs_projects_to_newtonian_accelerations():
    masses, positions, velocities = _planar_state()
    chart = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))

    derivative = regularized_binary_collision_chart_rhs(chart)
    projected_acceleration = planar_accelerations_from_regularized_chart_rhs(chart, derivative)

    assert np.linalg.norm(projected_acceleration - accelerations(positions, masses), ord=np.inf) < 1e-12
    assert abs(pair_energy_constraint_derivative(chart, derivative)) < 1e-12


def test_regularized_z_acceleration_agrees_with_physical_split_away_from_collision():
    masses, positions, velocities = _planar_state()
    old_chart = planar_to_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    new_chart = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
    _center_acceleration, _third_acceleration, relative_acceleration = coordinate_accelerations_from_planar(old_chart)

    projected_split = z_acceleration_from_relative_acceleration(
        new_chart.z,
        new_chart.z_velocity,
        relative_acceleration,
    )

    assert np.linalg.norm(regularized_z_acceleration(new_chart) - projected_split, ord=np.inf) < 1e-12


def test_regularized_rhs_is_finite_at_exact_binary_collision():
    masses = np.array([0.8, 1.2, 1.7])
    pair_mass = masses[0] + masses[1]
    state = RegularizedBinaryCollisionChartState(
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

    derivative = regularized_binary_collision_chart_rhs(state)

    assert abs(pair_energy_constraint(state)) < 1e-14
    assert np.all(np.isfinite(derivative.z))
    assert np.all(np.isfinite(derivative.z_velocity))
    assert np.all(np.isfinite(derivative.binary_center))
    assert np.all(np.isfinite(derivative.binary_center_velocity))
    assert np.all(np.isfinite(derivative.third_offset))
    assert np.all(np.isfinite(derivative.third_offset_velocity))
    assert derivative.physical_time == 0.0
    assert np.linalg.norm(derivative.z_velocity, ord=np.inf) == 0.0
    assert abs(pair_energy_constraint_derivative(state, derivative)) < 1e-14


def test_analytic_third_body_fields_have_collision_limit():
    masses = np.array([0.8, 1.2, 1.7])
    pair_mass = masses[0] + masses[1]
    third_offset = np.array([1.3, 0.4])

    def relative_perturbation(epsilon):
        state = RegularizedBinaryCollisionChartState(
            masses=masses,
            pair=(0, 1),
            z=np.array([epsilon, 0.0]),
            z_velocity=np.array([np.sqrt(pair_mass / 2.0), 0.0]),
            pair_energy=0.0,
            binary_center=np.array([0.0, 0.0]),
            binary_center_velocity=np.array([0.0, 0.0]),
            third_offset=third_offset,
            third_offset_velocity=np.array([0.0, 0.0]),
        )
        return analytic_coordinate_accelerations(state)[2]

    collision_state = RegularizedBinaryCollisionChartState(
        masses=masses,
        pair=(0, 1),
        z=np.array([0.0, 0.0]),
        z_velocity=np.array([np.sqrt(pair_mass / 2.0), 0.0]),
        pair_energy=0.0,
        binary_center=np.array([0.0, 0.0]),
        binary_center_velocity=np.array([0.0, 0.0]),
        third_offset=third_offset,
        third_offset_velocity=np.array([0.0, 0.0]),
    )
    collision_limit = analytic_coordinate_accelerations(collision_state)[2]

    assert np.linalg.norm(collision_limit, ord=np.inf) == 0.0
    assert np.linalg.norm(relative_perturbation(1e-3) - collision_limit, ord=np.inf) < 1e-5
    assert np.linalg.norm(relative_perturbation(1e-4) - collision_limit, ord=np.inf) < 1e-7
