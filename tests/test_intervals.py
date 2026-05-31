import numpy as np

from three_body_symmetry.intervals import (
    FloatInterval,
    RationalInterval,
    interval_coefficients_as_tuples,
    interval_polyder,
    interval_polynomial_eval,
    interval_series_power,
    interval_series_product,
    interval_sign,
    point_interval_coefficients,
    rational_interval_polyder,
    rational_interval_polynomial_eval,
    rational_interval_sign,
)


def test_interval_polynomial_eval_contains_point_value():
    coefficients = np.array([1.0, -3.0, 2.0])
    point = 0.25

    interval = interval_polynomial_eval(coefficients, FloatInterval.point(point))
    expected = np.polynomial.polynomial.polyval(point, coefficients)

    assert interval.lower <= expected <= interval.upper
    assert interval.upper - interval.lower < 1e-14


def test_interval_polynomial_eval_bounds_simple_range():
    coefficients = np.array([1.0, 0.0, -1.0])

    interval = interval_polynomial_eval(coefficients, FloatInterval(0.0, 0.5))

    assert interval.lower <= 0.75
    assert interval.upper >= 1.0
    assert interval_sign(interval) > 0


def test_interval_polynomial_eval_accepts_uncertain_coefficients():
    coefficients = (
        FloatInterval(0.9999999999999998, 1.0000000000000002),
        FloatInterval(-3.000000000000001, -2.999999999999999),
        FloatInterval(1.9999999999999998, 2.0000000000000004),
    )
    point = 0.25
    expected = np.polynomial.polynomial.polyval(
        point,
        np.array([1.0, -3.0, 2.0]),
    )

    interval = interval_polynomial_eval(coefficients, FloatInterval.point(point))

    assert interval.lower <= expected <= interval.upper
    assert interval.upper > interval.lower


def test_interval_polyder_preserves_coefficient_enclosures():
    coefficients = point_interval_coefficients(np.array([1.0, -3.0, 2.0]))

    derivative = interval_polyder(coefficients)
    derivative_intervals = interval_coefficients_as_tuples(derivative)

    assert derivative_intervals[0][0] <= -3.0 <= derivative_intervals[0][1]
    assert derivative_intervals[1][0] <= 4.0 <= derivative_intervals[1][1]


def test_interval_division_and_positive_power_enclose_point_operations():
    numerator = FloatInterval.point(2.0)
    denominator = FloatInterval(3.0, 3.000000000000001)
    quotient = numerator / denominator
    root = FloatInterval(3.999999999999999, 4.000000000000001).positive_power(-1.5)

    assert quotient.lower <= 2.0 / 3.0 <= quotient.upper
    assert root.lower <= 0.125 <= root.upper


def test_interval_series_power_encloses_point_series_power():
    base = point_interval_coefficients(np.array([4.0, 0.5, -0.125]))

    interval_power = interval_series_power(base, -1.5, 2)
    point_power = np.array([4.0, 0.5, -0.125])
    expected = np.array([0.125, -0.0234375, 0.009521484375])

    product = interval_series_product(base, interval_power, 2)
    product_again = interval_series_product(product, interval_power, 2)

    for index, expected_value in enumerate(expected):
        assert interval_power[index].lower <= expected_value <= interval_power[index].upper
    assert product_again[0].lower <= point_power[0] ** -2.0 <= product_again[0].upper


def test_rational_interval_polynomial_eval_uses_exact_endpoint_arithmetic():
    coefficients = (
        RationalInterval.point(1),
        RationalInterval.point(-3),
        RationalInterval.point(2),
    )
    variable = RationalInterval.from_float_interval(0.25, 0.5)

    value = rational_interval_polynomial_eval(coefficients, variable)
    derivative = rational_interval_polynomial_eval(
        rational_interval_polyder(coefficients),
        variable,
    )

    assert value.lower <= 0
    assert value.upper >= 0.375
    assert rational_interval_sign(derivative) == -1
    assert value.lower.denominator != 0
    assert value.upper.denominator != 0


def test_rational_interval_rejects_zero_division_exactly():
    numerator = RationalInterval.point(1)
    denominator = RationalInterval(-1, 1)

    try:
        _ = numerator / denominator
    except ValueError as exc:
        assert "containing zero" in str(exc)
    else:
        raise AssertionError("division by zero-containing rational interval succeeded")
