import numpy as np

from three_body_symmetry.dynamics import (
    accelerations,
    angular_momentum_components,
    angular_momentum_z,
    center_of_mass,
    energy,
    integrate,
    linear_momentum,
    pack_state,
    split_state,
)
from three_body_symmetry.series import integrate_reference


def _spatial_initial_data():
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


def test_pack_split_and_invariants_accept_spatial_three_body_state():
    masses, positions, velocities = _spatial_initial_data()
    state = pack_state(positions, velocities)
    split_positions, split_velocities = split_state(state)
    expected_center = np.average(positions, axis=0, weights=masses)
    expected_momentum = np.sum(masses[:, None] * velocities, axis=0)
    expected_angular_z = np.sum(
        positions[:, 0] * masses * velocities[:, 1]
        - positions[:, 1] * masses * velocities[:, 0]
    )

    assert state.shape == (18,)
    assert np.linalg.norm(split_positions - positions, ord=np.inf) == 0.0
    assert np.linalg.norm(split_velocities - velocities, ord=np.inf) == 0.0
    assert np.linalg.norm(center_of_mass(state, masses) - expected_center, ord=np.inf) < 1e-15
    assert np.linalg.norm(linear_momentum(state, masses) - expected_momentum, ord=np.inf) < 1e-15
    assert abs(angular_momentum_z(state, masses) - expected_angular_z) < 1e-15
    assert np.isfinite(energy(state, masses))


def test_spatial_angular_momentum_components_match_cross_product_convention():
    masses, positions, velocities = _spatial_initial_data()
    state = pack_state(positions, velocities)
    momenta = masses[:, None] * velocities
    axial_vector = np.sum(np.cross(positions, momenta), axis=0)
    expected_components = np.array(
        [axial_vector[2], -axial_vector[1], axial_vector[0]],
        dtype=float,
    )

    components = angular_momentum_components(state, masses)

    assert components.shape == (3,)
    assert np.linalg.norm(components - expected_components, ord=np.inf) < 1e-15
    assert angular_momentum_z(state, masses) == components[0]


def test_spatial_dynamics_integration_matches_dimension_generic_reference():
    masses, positions, velocities = _spatial_initial_data()
    state = pack_state(positions, velocities)
    target_time = 0.02

    orbit = integrate(
        state,
        target_time,
        masses=masses,
        samples=5,
        rtol=1e-12,
        atol=1e-14,
    )
    reference = integrate_reference(
        positions,
        velocities,
        masses,
        target_time,
        rtol=1e-12,
        atol=1e-14,
    )
    energy_values = np.array([energy(sample, masses) for sample in orbit.states])

    assert orbit.final_state.shape == (18,)
    assert np.linalg.norm(orbit.final_state - reference, ord=np.inf) < 1e-12
    assert np.max(np.abs(energy_values - energy_values[0])) < 1e-12


def test_spatial_dynamics_conserves_full_angular_momentum_components():
    masses, positions, velocities = _spatial_initial_data()
    state = pack_state(positions, velocities)
    orbit = integrate(
        state,
        0.04,
        masses=masses,
        samples=7,
        rtol=1e-12,
        atol=1e-14,
    )
    components = np.array([angular_momentum_components(sample, masses) for sample in orbit.states])

    assert np.max(np.abs(components - components[0])) < 1e-12


def test_spatial_dynamics_integration_supports_negative_target_time():
    masses, positions, velocities = _spatial_initial_data()
    state = pack_state(positions, velocities)
    target_time = -0.02

    orbit = integrate(
        state,
        target_time,
        masses=masses,
        samples=5,
        rtol=1e-12,
        atol=1e-14,
    )
    reference = integrate_reference(
        positions,
        velocities,
        masses,
        target_time,
        rtol=1e-12,
        atol=1e-14,
    )

    assert np.all(np.diff(orbit.times) < 0.0)
    assert np.linalg.norm(orbit.final_state - reference, ord=np.inf) < 1e-12


def test_spatial_dynamics_integration_supports_zero_target_time():
    masses, positions, velocities = _spatial_initial_data()
    state = pack_state(positions, velocities)

    orbit = integrate(state, 0.0, masses=masses, samples=3)

    assert np.linalg.norm(orbit.times, ord=np.inf) == 0.0
    assert orbit.states.shape == (3, state.shape[0])
    assert np.linalg.norm(orbit.final_state - state, ord=np.inf) == 0.0


def test_spatial_accelerations_agree_with_pairwise_newton_formula():
    masses, positions, _velocities = _spatial_initial_data()
    expected = np.zeros_like(positions)
    for i in range(3):
        for j in range(i + 1, 3):
            delta = positions[j] - positions[i]
            distance = np.linalg.norm(delta)
            force_shape = delta / distance**3
            expected[i] += masses[j] * force_shape
            expected[j] -= masses[i] * force_shape

    assert np.linalg.norm(accelerations(positions, masses) - expected, ord=np.inf) < 1e-15
