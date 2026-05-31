import numpy as np

from three_body_symmetry.levi_civita import (
    construct_levi_civita_chart,
    integrate_relative_kepler,
    lc_square,
    levi_civita_mu,
    physical_to_levi_civita,
    projected_acceleration_from_lift,
)


def test_levi_civita_projection_matches_relative_kepler_reference():
    mu = 1.7
    relative_position = np.array([0.8, 0.3])
    relative_velocity = np.array([-0.2, 0.6])
    z0, z_velocity0, energy = physical_to_levi_civita(relative_position, relative_velocity, mu)
    chart = construct_levi_civita_chart(z0, z_velocity0, energy, order=28)

    s_value = 0.03
    physical_time = chart.physical_time_at(s_value)
    reference = integrate_relative_kepler(relative_position, relative_velocity, mu, physical_time)

    assert abs(levi_civita_mu(z0, z_velocity0, energy) - mu) < 1e-14
    assert np.linalg.norm(chart.relative_state_at(s_value) - reference, ord=np.inf) < 1e-12


def test_projected_lifted_acceleration_equals_kepler_force():
    z = np.array([0.7, 0.2])
    z_velocity = np.array([0.1, 0.8])
    energy = -1.0
    mu = levi_civita_mu(z, z_velocity, energy)
    relative_position = lc_square(z)

    projected = projected_acceleration_from_lift(z, z_velocity, energy)
    expected = -mu * relative_position / np.linalg.norm(relative_position) ** 3

    assert mu > 0.0
    assert np.linalg.norm(projected - expected, ord=np.inf) < 1e-12


def test_collision_lift_has_finite_series_coefficients():
    z0 = np.array([0.0, 0.0])
    z_velocity0 = np.array([1.0, 0.0])
    energy = -0.5
    chart = construct_levi_civita_chart(z0, z_velocity0, energy, order=10)

    assert np.all(np.isfinite(chart.z))
    assert np.all(np.isfinite(chart.z_velocity))
    assert np.all(np.isfinite(chart.physical_time))
    assert np.linalg.norm(chart.relative_position_at(0.0), ord=np.inf) == 0.0
    assert abs(chart.physical_time[3] - 1.0 / 3.0) < 1e-14
    assert chart.physical_time_at(0.1) > 0.0


def test_parabolic_collision_passes_through_chart_origin():
    energy = 0.0
    z0 = np.array([0.1, 0.0])
    z_velocity0 = np.array([-1.0, 0.0])
    chart = construct_levi_civita_chart(z0, z_velocity0, energy, order=6)

    collision_s = 0.1
    before = chart.relative_position_at(collision_s - 0.01)
    collision = chart.relative_position_at(collision_s)
    after = chart.relative_position_at(collision_s + 0.01)

    assert np.linalg.norm(collision, ord=np.inf) < 1e-14
    assert np.linalg.norm(before, ord=np.inf) > 0.0
    assert np.linalg.norm(after, ord=np.inf) > 0.0
    assert np.all(np.isfinite(chart.z_at(collision_s)))
    assert np.all(np.isfinite(chart.z_velocity_at(collision_s)))

