from decimal import Decimal, ROUND_CEILING, localcontext
from fractions import Fraction

import numpy as np

from three_body_symmetry.certificate_checker import (
    _newton_acceleration_lipschitz_upper,
    check_ordinary_aposteriori_tube,
)
from three_body_symmetry.certificate_language import (
    OrdinaryAposterioriTubeCertificate,
    ordinary_taylor_chart_certificate_from_solution,
)
from three_body_symmetry.series import construct_taylor_solution


def _fraction_upper_float_reference(value: Fraction) -> float:
    candidate = float(value)
    if Fraction.from_float(candidate) < value:
        candidate = float(np.nextafter(candidate, np.inf))
    return candidate


def _exp_upper_reference(value: float) -> float:
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_CEILING
        exact_input = Decimal.from_float(value)
        nearest_exponential = context.exp(exact_input)
        decimal_exponential = context.next_plus(nearest_exponential)
    candidate = float(decimal_exponential)
    if Decimal.from_float(candidate) < decimal_exponential:
        candidate = float(np.nextafter(candidate, np.inf))
    return candidate


def _ordinary_tube_fixture():
    masses = np.array([1.0, 0.8, 1.2])
    positions = np.array(
        [
            [0.8, -0.2],
            [-0.4, 0.6],
            [0.1, -0.5],
        ]
    )
    velocities = np.array(
        [
            [0.03, 0.01],
            [-0.02, 0.04],
            [0.01, -0.03],
        ]
    )
    solution = construct_taylor_solution(positions, velocities, masses, order=10)
    chart = ordinary_taylor_chart_certificate_from_solution(
        solution,
        certificate_id="ordinary-chart:fraction-gronwall",
        chart_id="ordinary:fraction-gronwall",
        parameter_interval=(-0.02, 0.02),
        coefficient_tolerance=1.0e-11,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    tube = OrdinaryAposterioriTubeCertificate(
        tube_id="ordinary-tube:fraction-gronwall",
        chart_id=chart.chart_id,
        anchor_parameter=-0.010481414916324346,
        initial_error_bound=1.0e-6,
        tube_radius=2.0e-2,
        max_defect_bound=1.0e-1,
        max_lipschitz_bound=70.0,
    )
    return chart, tube


def test_newton_lipschitz_prefactor_matches_exact_fraction_reference():
    masses = np.array([0.5, 1.25, 2.0])

    bound = _newton_acceleration_lipschitz_upper(
        masses,
        body_index=0,
        dimension=4,
        pair_distance_floor=0.5,
    )
    mass_sum_q = Fraction.from_float(1.25) + Fraction.from_float(2.0)
    sqrt_dimension_upper = float(np.nextafter(2.0, np.inf))
    derivative_factor_q = (
        Fraction(1) + 3 * Fraction.from_float(sqrt_dimension_upper)
    )
    derivative_factor_upper = _fraction_upper_float_reference(
        derivative_factor_q
    )
    rho_q = Fraction.from_float(0.5)
    outward_bound_q = (
        2
        * mass_sum_q
        * Fraction.from_float(derivative_factor_upper)
        / (rho_q * rho_q * rho_q)
    )
    reference = _fraction_upper_float_reference(outward_bound_q)
    theoretical_bound_q = (
        2 * mass_sum_q * Fraction(7) / (rho_q * rho_q * rho_q)
    )

    assert theoretical_bound_q == 364
    assert bound == reference
    assert Fraction.from_float(bound) >= outward_bound_q > theoretical_bound_q


def test_scalar_ordinary_tube_gronwall_matches_fraction_reference_and_certifies():
    chart, tube = _ordinary_tube_fixture()

    result = check_ordinary_aposteriori_tube(tube, chart)

    anchor_q = Fraction.from_float(tube.anchor_parameter)
    horizon_q = max(
        abs(Fraction.from_float(endpoint) - anchor_q)
        for endpoint in chart.parameter_interval
    )
    mixed_float_horizon = max(
        abs(endpoint - tube.anchor_parameter)
        for endpoint in chart.parameter_interval
    )
    assert Fraction.from_float(mixed_float_horizon) != horizon_q

    lipschitz_q = Fraction.from_float(result.lipschitz_bound)
    exponent_q = lipschitz_q * horizon_q
    exponent_upper = _fraction_upper_float_reference(exponent_q)
    exponential_upper = _exp_upper_reference(exponent_upper)
    exponential_q = Fraction.from_float(exponential_upper)
    composed_bound_q = (
        exponential_q * Fraction.from_float(tube.initial_error_bound)
        + Fraction.from_float(result.defect_bound)
        * (exponential_q - Fraction(1))
        / lipschitz_q
    )
    reference = _fraction_upper_float_reference(composed_bound_q)
    legacy_exponent = _fraction_upper_float_reference(
        lipschitz_q * Fraction.from_float(mixed_float_horizon)
    )
    legacy_exponential = _exp_upper_reference(legacy_exponent)
    legacy_mixed_float_bound = float(
        np.nextafter(
            legacy_exponential * tube.initial_error_bound
            + result.defect_bound
            * (legacy_exponential - 1.0)
            / result.lipschitz_bound,
            np.inf,
        )
    )

    assert result.certified
    assert result.gronwall_error_bound == reference
    assert result.gronwall_error_bound != legacy_mixed_float_bound
    assert Fraction.from_float(result.gronwall_error_bound) >= composed_bound_q
    assert result.gronwall_error_bound < tube.tube_radius
