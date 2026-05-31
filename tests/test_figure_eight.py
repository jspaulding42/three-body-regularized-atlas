import numpy as np

from three_body_symmetry.dynamics import (
    angular_momentum_z,
    center_of_mass,
    energy,
    integrate,
    linear_momentum,
    max_periodic_error,
)
from three_body_symmetry.figure_eight import FIGURE_EIGHT_PERIOD, figure_eight_state


def test_figure_eight_closes_and_conserves_energy():
    state0 = figure_eight_state()
    orbit = integrate(state0, FIGURE_EIGHT_PERIOD, samples=501, rtol=1e-11, atol=1e-13)

    assert np.linalg.norm(orbit.final_state - state0, ord=np.inf) < 1e-6

    energies = np.array([energy(state) for state in orbit.states])
    assert np.max(np.abs(energies - energies[0])) < 1e-9


def test_figure_eight_has_zero_global_invariants():
    state0 = figure_eight_state()

    assert np.linalg.norm(center_of_mass(state0), ord=np.inf) < 1e-14
    assert np.linalg.norm(linear_momentum(state0), ord=np.inf) < 1e-14
    assert abs(angular_momentum_z(state0)) < 1e-8


def test_one_third_period_is_body_cycle():
    state0 = figure_eight_state()
    orbit = integrate(state0, FIGURE_EIGHT_PERIOD / 3.0, rtol=1e-11, atol=1e-13)

    positions0 = state0[:6].reshape(3, 2)
    velocities0 = state0[6:].reshape(3, 2)
    cycled = np.concatenate([positions0[[2, 0, 1]].reshape(-1), velocities0[[2, 0, 1]].reshape(-1)])

    assert np.linalg.norm(orbit.final_state - cycled, ord=np.inf) < 1e-6


def test_periodic_error_helper_matches_direct_integration():
    assert max_periodic_error(figure_eight_state(), FIGURE_EIGHT_PERIOD, rtol=1e-11, atol=1e-13) < 1e-6

