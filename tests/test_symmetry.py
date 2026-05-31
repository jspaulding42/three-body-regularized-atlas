import numpy as np

from three_body_symmetry.symmetry import (
    FourierMode,
    allowed_choreography_harmonics,
    choreography_positions,
    evaluate_curve,
)


def test_allowed_harmonics_are_cyclotomic_filter():
    assert allowed_choreography_harmonics(18) == [1, 5, 7, 11, 13, 17]


def test_fourier_filter_enforces_choreography_identities():
    period = 2.0 * np.pi
    modes = [
        FourierMode(1, 1.0 + 0.25j, -0.30 + 0.70j),
        FourierMode(5, -0.20 + 0.40j, 0.15 - 0.10j),
        FourierMode(7, 0.05 - 0.35j, -0.25 + 0.05j),
    ]

    for time in np.linspace(0.0, period, 9, endpoint=False):
        positions = choreography_positions(modes, time, period)
        assert np.linalg.norm(np.sum(positions, axis=0), ord=np.inf) < 1e-12

        q_now = evaluate_curve(modes, np.array([time]), period)[0]
        q_half_turn = evaluate_curve(modes, np.array([time + period / 2.0]), period)[0]
        assert abs(q_now + q_half_turn) < 1e-12

