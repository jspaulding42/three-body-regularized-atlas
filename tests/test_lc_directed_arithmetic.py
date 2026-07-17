from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

from three_body_symmetry.binary_chart import (
    _interval_norm,
    _interval_sqrt_nonnegative,
)
from three_body_symmetry.binary_series import RegularizedBinaryTaylorSolution
from three_body_symmetry.certificate_checker import (
    _gronwall_error_upper_fraction,
    _planar_lc_derivative_intervals,
)
from three_body_symmetry.intervals import (
    FloatInterval,
    directed_nonnegative_sqrt_endpoint,
)


def _decimal_fraction(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def test_directed_nonnegative_sqrt_and_binary_interval_paths_contain_reference():
    exact_lower = directed_nonnegative_sqrt_endpoint(4.0, upward=False)
    exact_upper = directed_nonnegative_sqrt_endpoint(4.0, upward=True)

    assert exact_lower < 2.0 < exact_upper

    value = float(np.nextafter(2.0, np.inf))
    with localcontext() as context:
        context.prec = 250
        reference = context.sqrt(Decimal.from_float(value))
    lower = directed_nonnegative_sqrt_endpoint(value, upward=False)
    upper = directed_nonnegative_sqrt_endpoint(value, upward=True)
    assert Decimal.from_float(lower) <= reference <= Decimal.from_float(upper)

    square_root = _interval_sqrt_nonnegative(FloatInterval.point(value))
    assert Decimal.from_float(square_root.lower) <= reference
    assert reference <= Decimal.from_float(square_root.upper)

    vector = np.asarray(
        [FloatInterval.point(0.1), FloatInterval.point(0.3)],
        dtype=object,
    )
    norm = _interval_norm(vector)
    with localcontext() as context:
        context.prec = 250
        norm_reference = context.sqrt(
            Decimal.from_float(0.1) ** 2 + Decimal.from_float(0.3) ** 2
        )
    assert Decimal.from_float(norm.lower) <= norm_reference
    assert norm_reference <= Decimal.from_float(norm.upper)


def test_all_fourteen_lc_derivative_blocks_enclose_exact_products():
    def vector_block(first: float, second: float) -> np.ndarray:
        block = np.zeros((4, 2), dtype=float)
        block[3] = (first, second)
        return block

    def scalar_block(value: float) -> np.ndarray:
        block = np.zeros(4, dtype=float)
        block[3] = value
        return block

    solution = RegularizedBinaryTaylorSolution(
        masses=np.asarray((1.0, 1.0, 1.0)),
        pair=(0, 1),
        z=vector_block(0.3, 0.1),
        z_velocity=vector_block(0.1, 0.3),
        pair_energy=scalar_block(0.3),
        binary_center=vector_block(0.3, 0.1),
        binary_center_velocity=vector_block(0.1, 0.3),
        third_offset=vector_block(0.3, 0.1),
        third_offset_velocity=vector_block(0.1, 0.3),
        physical_time=scalar_block(0.1),
    )
    derivative = _planar_lc_derivative_intervals(solution, (1.0, 1.0))
    serialized = (
        0.3,
        0.1,
        0.1,
        0.3,
        0.3,
        0.3,
        0.1,
        0.1,
        0.3,
        0.3,
        0.1,
        0.1,
        0.3,
        0.1,
    )

    assert len(derivative) == len(serialized) == 14
    assert Fraction.from_float(float(3.0 * 0.3)) < (
        Fraction(3) * Fraction.from_float(0.3)
    )
    assert Fraction(3) * Fraction.from_float(0.1) < Fraction.from_float(
        float(3.0 * 0.1)
    )
    for interval, coefficient in zip(derivative, serialized):
        reference = Fraction(3) * Fraction.from_float(coefficient)
        assert Fraction.from_float(interval.lower) <= reference
        assert reference <= Fraction.from_float(interval.upper)


def test_fraction_gronwall_composition_dominates_high_precision_reference():
    initial_error = 0.1
    defect = 0.3
    lipschitz = 0.7
    horizon = Fraction.from_float(0.2)
    bound = _gronwall_error_upper_fraction(
        initial_error,
        defect,
        lipschitz,
        horizon,
    )

    with localcontext() as context:
        context.prec = 250
        initial_decimal = Decimal.from_float(initial_error)
        defect_decimal = Decimal.from_float(defect)
        lipschitz_decimal = Decimal.from_float(lipschitz)
        horizon_decimal = _decimal_fraction(horizon)
        exponential = context.exp(lipschitz_decimal * horizon_decimal)
        reference = initial_decimal + (
            defect_decimal * (exponential - Decimal(1)) / lipschitz_decimal
        )
        bound_decimal = _decimal_fraction(bound)

    assert bound_decimal >= reference
    assert _gronwall_error_upper_fraction(
        initial_error,
        defect,
        0.0,
        horizon,
    ) == (
        Fraction.from_float(initial_error)
        + Fraction.from_float(defect) * horizon
    )
