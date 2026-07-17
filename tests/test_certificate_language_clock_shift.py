from types import SimpleNamespace

import numpy as np

from three_body_symmetry.binary_chart import (
    planar_to_regularized_binary_collision_chart,
    regularized_binary_collision_chart_to_planar,
)
from three_body_symmetry.binary_series import (
    construct_regularized_binary_taylor_solution,
)
from three_body_symmetry.certificate_language import (
    planar_hybrid_chart_chain_certificates_from_solution,
    planar_levi_civita_binary_chart_certificate_from_solution,
)


def _planar_lc_solution():
    masses = np.array([1.0, 1.0, 1.2])
    positions = np.array(
        [
            [-0.12, 0.025],
            [0.10, -0.015],
            [1.7, 0.9],
        ]
    )
    velocities = np.array(
        [
            [0.04, -0.08],
            [-0.02, 0.06],
            [0.01, 0.03],
        ]
    )
    initial = planar_to_regularized_binary_collision_chart(
        positions,
        velocities,
        masses,
        pair=(0, 1),
    )
    return construct_regularized_binary_taylor_solution(initial, order=12)


def test_planar_lc_serializer_applies_nonzero_physical_time_shift():
    solution = _planar_lc_solution()
    parameter_interval = (0.0, 0.004)
    shift = 7.25
    unshifted = planar_levi_civita_binary_chart_certificate_from_solution(
        solution,
        certificate_id="unshifted-planar-lc-chart",
        chart_id="unshifted-planar-lc",
        parameter_interval=parameter_interval,
    )
    shifted = planar_levi_civita_binary_chart_certificate_from_solution(
        solution,
        certificate_id="shifted-planar-lc-chart",
        chart_id="shifted-planar-lc",
        parameter_interval=parameter_interval,
        physical_time_shift=shift,
    )

    assert shifted.physical_time_coefficients[0] == (
        unshifted.physical_time_coefficients[0] + shift
    )
    assert shifted.physical_time_coefficients[1:] == (
        unshifted.physical_time_coefficients[1:]
    )
    assert shifted.physical_time_interval == tuple(
        endpoint + shift for endpoint in unshifted.physical_time_interval
    )


def test_planar_hybrid_serializer_uses_binary_step_global_start_time():
    solution = _planar_lc_solution()
    parameter_step = 0.004
    start_time = 7.25
    physical_step = solution.physical_time_at(parameter_step)
    positions, velocities = regularized_binary_collision_chart_to_planar(
        solution.state_at(0.0)
    )
    start_state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
    step = SimpleNamespace(
        chart="binary",
        pair=solution.pair,
        start_time=start_time,
        physical_step=physical_step,
        parameter_step=parameter_step,
        end_time=start_time + physical_step,
        truncation_certificate=SimpleNamespace(
            computed_order=solution.order,
            retained_order=solution.order,
            tail_bound=1.0e-9,
        ),
    )
    hybrid_solution = SimpleNamespace(
        masses=solution.masses,
        states=np.asarray([start_state]),
        steps=(step,),
    )

    charts, transitions, chain = planar_hybrid_chart_chain_certificates_from_solution(
        hybrid_solution,
        certificate_id_prefix="nonzero-clock",
    )

    assert transitions == ()
    assert len(charts) == 1
    chart = charts[0]
    assert chart.physical_time_coefficients[0] == start_time
    assert chart.physical_time_interval[0] > 0.5 * start_time
    assert chart.physical_time_interval[0] <= start_time
    assert chart.physical_time_interval[1] >= step.end_time
    assert chain.target_physical_time_interval == chart.physical_time_interval
