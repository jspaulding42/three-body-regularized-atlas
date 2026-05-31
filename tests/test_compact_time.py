import pytest

from three_body_symmetry.compact_time import (
    certify_compactified_time_target,
    certify_compactified_time_taylor_evaluation,
    certify_compactified_time_taylor_solution,
    compact_parameter_from_physical_time,
    compactified_time_taylor_tail_bound,
    construct_compactified_time_taylor_solution,
    compact_parameter_interval_from_physical_time_interval,
    inverse_time_derivative_interval,
    physical_time_from_compact_parameter,
    physical_time_interval_from_compact_parameter_interval,
)
from three_body_symmetry.intervals import FloatInterval


def test_compact_time_round_trips_physical_time():
    physical_time = 2.75
    rate = 0.4

    compact = compact_parameter_from_physical_time(physical_time, rate=rate)
    reconstructed = physical_time_from_compact_parameter(compact, rate=rate)
    certificate = certify_compactified_time_target(compact, rate=rate)

    assert -1.0 < compact < 1.0
    assert reconstructed == pytest.approx(physical_time)
    assert certificate.certified
    assert certificate.physical_time_interval.lower <= reconstructed <= certificate.physical_time_interval.upper
    assert (
        certificate.roundtrip_compact_parameter_interval.lower
        <= compact
        <= certificate.roundtrip_compact_parameter_interval.upper
    )
    assert certificate.inverse_derivative_interval.lower > 0.0


def test_compact_time_interval_maps_are_monotone_and_contain_endpoints():
    physical_interval = FloatInterval(-3.0, 4.0)
    rate = 0.25

    compact_interval = compact_parameter_interval_from_physical_time_interval(
        physical_interval,
        rate=rate,
    )
    reconstructed_interval = physical_time_interval_from_compact_parameter_interval(
        compact_interval,
        rate=rate,
    )
    derivative_interval = inverse_time_derivative_interval(compact_interval, rate=rate)

    assert compact_interval.lower < 0.0 < compact_interval.upper
    assert compact_interval.lower < compact_interval.upper
    assert reconstructed_interval.lower <= physical_interval.lower
    assert physical_interval.upper <= reconstructed_interval.upper
    assert derivative_interval.lower > 0.0


def test_compact_time_rejects_singular_parameter_endpoints():
    for value in (-1.0, 1.0):
        with pytest.raises(ValueError, match="strictly between"):
            physical_time_from_compact_parameter(value)

    with pytest.raises(ValueError, match="strictly inside"):
        physical_time_interval_from_compact_parameter_interval(FloatInterval(-0.5, 1.0))


def test_compact_time_rejects_nonpositive_rate():
    with pytest.raises(ValueError, match="positive finite"):
        compact_parameter_from_physical_time(1.0, rate=0.0)


def test_zero_centered_compact_time_taylor_coefficients_satisfy_inverse_ode():
    solution = construct_compactified_time_taylor_solution(center=0.0, rate=0.5, order=12)

    certificate = certify_compactified_time_taylor_solution(solution)

    assert certificate.certified
    assert certificate.coefficient_count == 12
    assert solution.analytic_radius == 1.0
    assert solution.coefficients[0] == 0.0
    assert solution.coefficients[1] == pytest.approx(2.0)
    assert solution.coefficients[2] == 0.0
    assert solution.coefficients[3] == pytest.approx(2.0 / 3.0)


def test_compact_time_taylor_tail_encloses_exact_inverse_time():
    compact = 0.35
    solution = construct_compactified_time_taylor_solution(center=0.0, rate=1.3, order=9)

    enclosure = solution.physical_time_enclosure_at_delta(compact)
    exact = physical_time_from_compact_parameter(compact, rate=1.3)
    certificate = certify_compactified_time_taylor_evaluation(compact, rate=1.3, order=9)

    assert compactified_time_taylor_tail_bound(solution, compact) > 0.0
    assert enclosure.lower <= exact <= enclosure.upper
    assert certificate.certified
    assert certificate.time_enclosure.lower <= exact <= certificate.time_enclosure.upper
    assert certificate.tail_bound == compactified_time_taylor_tail_bound(solution, compact)
