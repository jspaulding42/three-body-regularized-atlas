import numpy as np

from three_body_symmetry.dynamics import max_periodic_error
from three_body_symmetry.shooting import refine_figure_eight, residual


def test_shooting_recovers_periodic_orbit_from_rough_symmetric_seed():
    seed = np.array([0.97, -0.243, 0.466, 0.432, 6.326], dtype=float)

    seed_residual = np.linalg.norm(residual(seed, rtol=3e-10, atol=1e-12), ord=np.inf)
    result = refine_figure_eight(seed, max_nfev=25, rtol=3e-10, atol=1e-12)

    assert seed_residual > 1e-3
    assert result.success
    assert result.residual_norm < 1e-7
    assert result.residual_norm < seed_residual * 1e-5
    assert max_periodic_error(result.state, result.period, rtol=1e-10, atol=1e-12) < 1e-7
    assert 6.0 < result.period < 6.7
