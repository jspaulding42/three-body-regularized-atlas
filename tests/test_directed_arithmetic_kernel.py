from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

from three_body_symmetry.certificate_checker import (
    _exp_upper,
    _fraction_sqrt_float,
    _interval_ordinary_newton_residual_blocks,
)
from three_body_symmetry.intervals import (
    _directed_negative_half_integer_power,
    interval_array_derivative_coefficients,
)


def test_decimal_exp_and_sqrt_use_strict_neighbor_enclosures_at_exact_values():
    exponential_upper = _exp_upper(0.0)
    sqrt_lower = _fraction_sqrt_float(Fraction(4), upward=False)
    sqrt_upper = _fraction_sqrt_float(Fraction(4), upward=True)

    assert exponential_upper == float(np.nextafter(1.0, np.inf))
    assert sqrt_lower == float(np.nextafter(2.0, -np.inf))
    assert sqrt_upper == float(np.nextafter(2.0, np.inf))
    assert Decimal.from_float(exponential_upper) > Decimal(1)
    assert Decimal.from_float(sqrt_lower) < Decimal(2)
    assert Decimal.from_float(sqrt_upper) > Decimal(2)

    nonzero_input = float(np.nextafter(1.0, np.inf))
    with localcontext() as context:
        context.prec = 250
        high_precision_exponential = context.exp(
            Decimal.from_float(nonzero_input)
        )
    assert Decimal.from_float(_exp_upper(nonzero_input)) >= high_precision_exponential


def test_negative_half_integer_powers_bracket_exact_and_high_precision_values():
    exact_at_four = {
        -0.5: Fraction(1, 2),
        -1.5: Fraction(1, 8),
        -2.5: Fraction(1, 32),
    }
    for exponent, exact in exact_at_four.items():
        lower = _directed_negative_half_integer_power(
            4.0, exponent, upward=False
        )
        upper = _directed_negative_half_integer_power(
            4.0, exponent, upward=True
        )
        assert Fraction.from_float(lower) < exact
        assert exact < Fraction.from_float(upper)

    value = float(np.nextafter(2.0, np.inf))
    base = Decimal.from_float(value)
    with localcontext() as context:
        context.prec = 250
        exact_root = context.sqrt(base)
        for exponent, integer_power in ((-0.5, 0), (-1.5, 1), (-2.5, 2)):
            denominator = exact_root * (base**integer_power)
            reference = Decimal(1) / denominator
            lower = _directed_negative_half_integer_power(
                value, exponent, upward=False
            )
            upper = _directed_negative_half_integer_power(
                value, exponent, upward=True
            )
            assert Decimal.from_float(lower) <= reference
            assert reference <= Decimal.from_float(upper)


def test_interval_array_derivative_encloses_exact_serialized_float_product():
    coefficients = np.zeros((4, 1, 2), dtype=float)
    coefficients[3, 0, 0] = 0.3
    coefficients[3, 0, 1] = 0.1
    exact_down = Fraction(3) * Fraction.from_float(0.3)
    exact_up = Fraction(3) * Fraction.from_float(0.1)
    preprocessed_down = Fraction.from_float(float(3.0 * 0.3))
    preprocessed_up = Fraction.from_float(float(3.0 * 0.1))
    assert preprocessed_down < exact_down
    assert exact_up < preprocessed_up

    derivative = interval_array_derivative_coefficients(coefficients)
    down_enclosure = derivative[2, 0, 0]
    up_enclosure = derivative[2, 0, 1]

    assert Fraction.from_float(down_enclosure.lower) <= exact_down
    assert exact_down <= Fraction.from_float(down_enclosure.upper)
    assert Fraction.from_float(up_enclosure.lower) <= exact_up
    assert exact_up <= Fraction.from_float(up_enclosure.upper)
    assert down_enclosure.lower < down_enclosure.upper
    assert up_enclosure.lower < up_enclosure.upper


def test_ordinary_residual_block_contains_exact_derivative_reference():
    q = np.zeros((4, 3, 2), dtype=float)
    v = np.zeros_like(q)
    q[0] = np.array(((0.0, 0.0), (3.0, 0.0), (0.0, 4.0)))
    q[3, 0, 0] = 0.3
    masses = np.array((1.0, 1.0, 1.0))
    exact_derivative = Fraction(3) * Fraction.from_float(0.3)

    position_residual, _ = _interval_ordinary_newton_residual_blocks(
        q,
        v,
        masses,
        (1.0, 1.0),
    )

    assert Fraction.from_float(position_residual) >= exact_derivative
