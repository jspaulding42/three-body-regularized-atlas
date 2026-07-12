from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

from three_body_symmetry.certificate_checker import (
    _fraction_sqrt_float,
    _ordinary_tube_pair_floor_lower,
    _planar_lc_mass_ratio_arithmetic_exact,
)


def test_fraction_sqrt_bounds_are_directed_on_adversarial_rationals():
    values = (
        Fraction.from_float(float(np.nextafter(1.0, 0.0))),
        Fraction.from_float(float(np.nextafter(2.0, np.inf))),
        Fraction(1, 10**40),
        Fraction(10**40 + 1, 3),
    )
    for value in values:
        lower = _fraction_sqrt_float(value, upward=False)
        upper = _fraction_sqrt_float(value, upward=True)
        with localcontext() as context:
            context.prec = 200
            exact = context.sqrt(
                Decimal(value.numerator) / Decimal(value.denominator)
            )
        assert Decimal.from_float(lower) <= exact
        assert exact <= Decimal.from_float(upper)


def test_weighted_position_tube_floor_is_not_above_exact_expression():
    nominal = float(np.nextafter(1.0e-4, 0.0))
    radius = float(np.nextafter(1.0e-5, np.inf))
    floor = _ordinary_tube_pair_floor_lower(nominal, 2, radius)
    with localcontext() as context:
        context.prec = 200
        exact = Decimal.from_float(nominal) - (
            Decimal(2)
            * context.sqrt(Decimal(2))
            * Decimal.from_float(radius)
        )
    assert Decimal.from_float(floor) <= exact


def test_lc_newton_projection_requires_exact_serialized_mass_ratios():
    assert _planar_lc_mass_ratio_arithmetic_exact((1.0, 1.0, 1.2), (0, 1))
    assert not _planar_lc_mass_ratio_arithmetic_exact((1.0, 0.8, 1.2), (0, 1))
    assert not _planar_lc_mass_ratio_arithmetic_exact((1.0, 2.0, 3.0), (0, 1))
