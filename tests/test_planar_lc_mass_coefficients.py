from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import math

import numpy as np
import pytest

from three_body_symmetry.certificate_checker import (
    _planar_lc_mass_ratio_arithmetic_exact,
    _planar_lc_dual_rhs,
    _project_interval_planar_lc_state,
)
from three_body_symmetry.intervals import FloatInterval
from three_body_symmetry.planar_lc_mass_coefficients import (
    PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID,
    PlanarLCMassCoefficientWitness,
    derive_planar_lc_mass_coefficient_witness,
    planar_lc_mass_coefficient_witness_certified,
)


MASS_VALUES = (0.1, 0.2, 0.3)
CANONICAL_PAIRS = ((0, 1), (0, 2), (1, 2))


def _contains_exact(interval: FloatInterval, value: Fraction) -> bool:
    return bool(
        Fraction.from_float(interval.lower)
        <= value
        <= Fraction.from_float(interval.upper)
    )


def _point_lc_state() -> tuple[FloatInterval, ...]:
    values = (
        0.75,
        -0.5,
        0.125,
        0.375,
        -0.25,
        1.0,
        -1.25,
        0.5,
        -0.75,
        2.0,
        0.25,
        -0.5,
        0.625,
        1.25,
    )
    return tuple(FloatInterval.point(value) for value in values)


@pytest.mark.parametrize(
    "pair",
    tuple(
        (first, second) for first in range(3) for second in range(3) if first != second
    ),
)
def test_decimal_mass_witness_is_exact_outward_and_tight_for_every_ordered_pair(pair):
    witness = derive_planar_lc_mass_coefficient_witness(MASS_VALUES, pair)
    mass_q = tuple(Fraction.from_float(value) for value in MASS_VALUES)
    first, second = pair
    third = ({0, 1, 2} - set(pair)).pop()
    pair_q = mass_q[first] + mass_q[second]
    expected = (
        (pair_q, witness.pair_mass_interval),
        (mass_q[second] / pair_q, witness.alpha_interval),
        (mass_q[first] / pair_q, witness.beta_interval),
        (mass_q[third] / pair_q, witness.third_over_pair_interval),
        (
            mass_q[first] * mass_q[third] / pair_q,
            witness.center_first_interval,
        ),
        (
            mass_q[second] * mass_q[third] / pair_q,
            witness.center_second_interval,
        ),
        (
            mass_q[first] + mass_q[first] * mass_q[third] / pair_q,
            witness.offset_first_interval,
        ),
        (
            mass_q[second] + mass_q[second] * mass_q[third] / pair_q,
            witness.offset_second_interval,
        ),
    )

    assert witness.kernel_id == PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
    assert witness.certified
    assert planar_lc_mass_coefficient_witness_certified(witness)
    assert witness.mass_fractions == mass_q
    assert witness.pair == pair
    assert witness.third_index == third
    assert witness.alpha + witness.beta == 1
    for exact, interval in expected:
        assert _contains_exact(interval, exact)
        if interval.lower != interval.upper:
            assert math.nextafter(interval.lower, math.inf) == interval.upper


@pytest.mark.parametrize("pair", CANONICAL_PAIRS)
def test_interval_projection_contains_exact_fraction_projection_for_decimal_masses(
    pair,
):
    state = _point_lc_state()
    positions, velocities, physical_time, rho_interval = (
        _project_interval_planar_lc_state(
            state,
            np.asarray(MASS_VALUES),
            pair,
        )
    )
    values = tuple(Fraction.from_float(interval.lower) for interval in state)
    x, y, wx, wy = values[:4]
    center = values[5:7]
    center_velocity = values[7:9]
    third_offset = values[9:11]
    third_velocity = values[11:13]
    rho = x * x + y * y
    relative_position = (x * x - y * y, 2 * x * y)
    relative_velocity = (
        2 * (x * wx - y * wy) / rho,
        2 * (y * wx + x * wy) / rho,
    )
    witness = derive_planar_lc_mass_coefficient_witness(MASS_VALUES, pair)
    first, second = pair
    third = witness.third_index
    exact_positions = [[Fraction(0), Fraction(0)] for _ in range(3)]
    exact_velocities = [[Fraction(0), Fraction(0)] for _ in range(3)]
    for axis in range(2):
        exact_positions[first][axis] = (
            center[axis] - witness.alpha * relative_position[axis]
        )
        exact_positions[second][axis] = (
            center[axis] + witness.beta * relative_position[axis]
        )
        exact_positions[third][axis] = center[axis] + third_offset[axis]
        exact_velocities[first][axis] = (
            center_velocity[axis] - witness.alpha * relative_velocity[axis]
        )
        exact_velocities[second][axis] = (
            center_velocity[axis] + witness.beta * relative_velocity[axis]
        )
        exact_velocities[third][axis] = center_velocity[axis] + third_velocity[axis]

    for body in range(3):
        for axis in range(2):
            assert _contains_exact(
                positions[body, axis],
                exact_positions[body][axis],
            )
            assert _contains_exact(
                velocities[body, axis],
                exact_velocities[body][axis],
            )
    assert _contains_exact(physical_time, values[13])
    assert _contains_exact(rho_interval, rho)


@pytest.mark.parametrize(
    ("masses", "pair"),
    (
        ((0.1, 0.2), (0, 1)),
        ((0.1, 0.2, 0.3, 0.4), (0, 1)),
        ((0.1, 0.0, 0.3), (0, 1)),
        ((0.1, -0.2, 0.3), (0, 1)),
        ((0.1, float("nan"), 0.3), (0, 1)),
        ((0.1, float("inf"), 0.3), (0, 1)),
        ((0.1, True, 0.3), (0, 1)),
        ((0.1, 1, 0.3), (0, 1)),
        ((0.1, Fraction(1, 5), 0.3), (0, 1)),
        ((0.1, Decimal("0.2"), 0.3), (0, 1)),
        ((0.1, "0.2", 0.3), (0, 1)),
        (np.asarray([0.1, 0.2, 0.3], dtype=np.float64), (0, 1)),
        (np.asarray([[0.1], [0.2], [0.3]]), (0, 1)),
        (np.asarray([0.1, 0.2, 0.3], dtype=np.float32), (0, 1)),
        ((0.1, 0.2, 0.3), (0, 0)),
        ((0.1, 0.2, 0.3), (0, 3)),
        ((0.1, 0.2, 0.3), (True, 1)),
        ((0.1, 0.2, 0.3), (0,)),
        ((0.1, 0.2, 0.3), [0, 1]),
        ((1.0e308, 1.0e308, 1.0), (0, 1)),
    ),
)
def test_mass_witness_rejects_malformed_masses_pairs_and_unbounded_coefficients(
    masses,
    pair,
):
    with pytest.raises(ValueError):
        derive_planar_lc_mass_coefficient_witness(masses, pair)


def test_low_level_projection_and_rhs_reject_malformed_mass_inputs():
    state = _point_lc_state()
    with pytest.raises(ValueError):
        _project_interval_planar_lc_state(state, np.asarray([0.1, 0.0, 0.3]), (0, 1))
    with pytest.raises(ValueError):
        _project_interval_planar_lc_state(
            state,
            np.asarray([[0.1], [0.2], [0.3]]),
            (0, 1),
        )
    with pytest.raises(ValueError):
        _planar_lc_dual_rhs(state, np.asarray([0.1, 0.2, 0.3]), (0, 0))
    with pytest.raises(ValueError):
        _planar_lc_dual_rhs(
            state,
            np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            (0, 1),
        )


def test_adversarial_near_one_pair_sum_has_a_certified_outward_witness():
    masses = (1.0, 3.0 * 2.0**-53, 0.3)
    witness = derive_planar_lc_mass_coefficient_witness(masses, (0, 1))
    exact_pair_mass = Fraction.from_float(masses[0]) + Fraction.from_float(masses[1])

    assert witness.certified
    assert not _planar_lc_mass_ratio_arithmetic_exact(masses, (0, 1))
    assert witness.pair_mass == exact_pair_mass
    assert _contains_exact(witness.pair_mass_interval, exact_pair_mass)
    assert _contains_exact(witness.alpha_interval, witness.alpha)
    assert _contains_exact(witness.beta_interval, witness.beta)


def test_numpy_float64_is_normalized_only_at_the_low_level_checker_boundary():
    masses = np.asarray(MASS_VALUES, dtype=np.float64)
    rhs, third_floor = _planar_lc_dual_rhs(
        _point_lc_state(),
        masses,
        (0, 1),
    )

    assert len(rhs) == 14
    assert math.isfinite(third_floor)
    with pytest.raises(ValueError):
        derive_planar_lc_mass_coefficient_witness(masses, (0, 1))


def test_smallest_subnormal_mass_scale_certifies_but_unenclosable_pair_sum_rejects():
    smallest = math.ulp(0.0)
    witness = derive_planar_lc_mass_coefficient_witness(
        (smallest, smallest, smallest),
        (0, 1),
    )

    assert witness.certified
    assert witness.pair_mass_interval == FloatInterval.point(2.0 * smallest)
    with pytest.raises(ValueError):
        derive_planar_lc_mass_coefficient_witness(
            (float.fromhex("0x1.fffffffffffffp+1023"), 1.0, 1.0),
            (0, 1),
        )


def test_witness_validation_rejects_wrong_identity_interval_kernel_and_subclass():
    witness = derive_planar_lc_mass_coefficient_witness(MASS_VALUES, (0, 1))
    wrong_interval = FloatInterval(
        math.nextafter(witness.alpha_interval.lower, -math.inf),
        witness.alpha_interval.upper,
    )

    assert not replace(witness, kernel_id="wrong_kernel").certified
    assert not replace(witness, alpha=witness.alpha + 1).certified
    assert not replace(witness, alpha_interval=wrong_interval).certified

    class WitnessSubclass(PlanarLCMassCoefficientWitness):
        pass

    subclass = WitnessSubclass(
        **{
            field_name: getattr(witness, field_name)
            for field_name in PlanarLCMassCoefficientWitness.__dataclass_fields__
        }
    )
    assert not subclass.certified
    assert not planar_lc_mass_coefficient_witness_certified(subclass)


@pytest.mark.parametrize("pair", CANONICAL_PAIRS)
def test_decimal_mass_rhs_and_full_interval_jacobian_are_finite_for_all_pairs(pair):
    rhs, third_floor = _planar_lc_dual_rhs(
        _point_lc_state(),
        np.asarray(MASS_VALUES),
        pair,
    )

    assert len(rhs) == 14
    assert math.isfinite(third_floor) and third_floor > 0.0
    for component in rhs:
        assert type(component.value) is FloatInterval
        assert math.isfinite(component.value.lower)
        assert math.isfinite(component.value.upper)
        assert component.value.lower <= component.value.upper
        assert len(component.derivative) == 14
        for derivative in component.derivative:
            assert type(derivative) is FloatInterval
            assert math.isfinite(derivative.lower)
            assert math.isfinite(derivative.upper)
            assert derivative.lower <= derivative.upper
