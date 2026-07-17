"""Exact derived mass coefficients for the planar Levi--Civita kernel.

The serialized masses are binary64 values.  This module treats those values as
exact rationals, derives every mass coefficient used by the checker-side LC
projection and vector field with :class:`fractions.Fraction`, and only then
converts the results to tight outward binary64 intervals.

The module deliberately depends only on the scalar interval layer and the
Python standard library.  In particular, it does not depend on certificates,
the certificate checker, or the numerical LC producer.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable

from .intervals import FloatInterval


PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID = (
    "planar_lc_mass_coefficients_exact_binary64_fraction_outward_v1"
)


@dataclass(frozen=True)
class PlanarLCMassCoefficientWitness:
    """Fresh exact and outward-enclosed coefficients for one ordered LC pair."""

    kernel_id: str
    pair: tuple[int, int]
    third_index: int
    mass_fractions: tuple[Fraction, Fraction, Fraction]
    pair_mass: Fraction
    alpha: Fraction
    beta: Fraction
    third_over_pair: Fraction
    center_first: Fraction
    center_second: Fraction
    offset_first: Fraction
    offset_second: Fraction
    mass_intervals: tuple[FloatInterval, FloatInterval, FloatInterval]
    pair_mass_interval: FloatInterval
    alpha_interval: FloatInterval
    beta_interval: FloatInterval
    third_over_pair_interval: FloatInterval
    center_first_interval: FloatInterval
    center_second_interval: FloatInterval
    offset_first_interval: FloatInterval
    offset_second_interval: FloatInterval

    @property
    def first_mass(self) -> Fraction:
        return self.mass_fractions[self.pair[0]]

    @property
    def second_mass(self) -> Fraction:
        return self.mass_fractions[self.pair[1]]

    @property
    def third_mass(self) -> Fraction:
        return self.mass_fractions[self.third_index]

    @property
    def first_mass_interval(self) -> FloatInterval:
        return self.mass_intervals[self.pair[0]]

    @property
    def second_mass_interval(self) -> FloatInterval:
        return self.mass_intervals[self.pair[1]]

    @property
    def third_mass_interval(self) -> FloatInterval:
        return self.mass_intervals[self.third_index]

    @property
    def certified(self) -> bool:
        """Whether this exact-class witness freshly validates in full."""

        return planar_lc_mass_coefficient_witness_certified(self)


def derive_planar_lc_mass_coefficient_witness(
    masses: Iterable[object],
    pair: tuple[int, int],
) -> PlanarLCMassCoefficientWitness:
    """Derive the exact mass-coefficient witness from serialized binary64s.

    Only exact built-in ``float`` values are accepted.  Numerical callers must
    validate a binary64 container and explicitly normalize its scalars before
    entering this dependency-independent kernel.  Integers, strings,
    arbitrary ``__float__`` objects, NumPy scalars, invalid shapes, non-finite
    values, and nonpositive masses reject rather than being silently rounded
    to binary64.
    """

    if type(pair) is not tuple or len(pair) != 2:
        raise ValueError("pair must be an ordered pair of distinct body indices")
    if any(type(index) is not int for index in pair):
        raise ValueError("pair indices must be integers")
    normalized_pair = pair
    if normalized_pair[0] == normalized_pair[1] or not set(normalized_pair).issubset(
        {0, 1, 2}
    ):
        raise ValueError("pair must contain two distinct indices from {0, 1, 2}")

    try:
        raw_masses = tuple(masses)
    except TypeError as exc:
        raise ValueError("masses must be an iterable of three binary64 values") from exc
    if len(raw_masses) != 3:
        raise ValueError("masses must contain exactly three values")

    mass_values: list[float] = []
    for raw in raw_masses:
        if type(raw) is not float:
            raise ValueError("masses must be finite positive binary64 values")
        value = raw
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("masses must be finite positive binary64 values")
        mass_values.append(value)

    mass_fractions = (
        Fraction.from_float(mass_values[0]),
        Fraction.from_float(mass_values[1]),
        Fraction.from_float(mass_values[2]),
    )
    first, second = normalized_pair
    third = ({0, 1, 2} - {first, second}).pop()
    first_mass = mass_fractions[first]
    second_mass = mass_fractions[second]
    third_mass = mass_fractions[third]
    pair_mass = first_mass + second_mass
    alpha = second_mass / pair_mass
    beta = first_mass / pair_mass
    third_over_pair = third_mass / pair_mass
    center_first = first_mass * third_over_pair
    center_second = second_mass * third_over_pair
    offset_first = first_mass + center_first
    offset_second = second_mass + center_second

    return PlanarLCMassCoefficientWitness(
        kernel_id=PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID,
        pair=normalized_pair,
        third_index=third,
        mass_fractions=mass_fractions,
        pair_mass=pair_mass,
        alpha=alpha,
        beta=beta,
        third_over_pair=third_over_pair,
        center_first=center_first,
        center_second=center_second,
        offset_first=offset_first,
        offset_second=offset_second,
        mass_intervals=(
            tight_fraction_float_interval(mass_fractions[0]),
            tight_fraction_float_interval(mass_fractions[1]),
            tight_fraction_float_interval(mass_fractions[2]),
        ),
        pair_mass_interval=tight_fraction_float_interval(pair_mass),
        alpha_interval=tight_fraction_float_interval(alpha),
        beta_interval=tight_fraction_float_interval(beta),
        third_over_pair_interval=tight_fraction_float_interval(third_over_pair),
        center_first_interval=tight_fraction_float_interval(center_first),
        center_second_interval=tight_fraction_float_interval(center_second),
        offset_first_interval=tight_fraction_float_interval(offset_first),
        offset_second_interval=tight_fraction_float_interval(offset_second),
    )


def planar_lc_mass_coefficient_witness_certified(witness: object) -> bool:
    """Freshly validate the exact identities and every outward enclosure."""

    if type(witness) is not PlanarLCMassCoefficientWitness:
        return False
    try:
        if (
            type(witness.kernel_id) is not str
            or witness.kernel_id != PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
            or type(witness.pair) is not tuple
            or len(witness.pair) != 2
            or any(type(index) is not int for index in witness.pair)
            or witness.pair[0] == witness.pair[1]
            or not set(witness.pair).issubset({0, 1, 2})
            or type(witness.third_index) is not int
            or witness.third_index != ({0, 1, 2} - set(witness.pair)).pop()
            or type(witness.mass_fractions) is not tuple
            or len(witness.mass_fractions) != 3
            or any(
                type(value) is not Fraction or value <= 0
                for value in witness.mass_fractions
            )
            or any(
                not math.isfinite(float(value))
                or Fraction.from_float(float(value)) != value
                for value in witness.mass_fractions
            )
        ):
            return False
        first, second = witness.pair
        third = witness.third_index
        first_mass = witness.mass_fractions[first]
        second_mass = witness.mass_fractions[second]
        third_mass = witness.mass_fractions[third]
        pair_mass = first_mass + second_mass
        third_over_pair = third_mass / pair_mass
        exact_values = (
            witness.mass_fractions[0],
            witness.mass_fractions[1],
            witness.mass_fractions[2],
            pair_mass,
            second_mass / pair_mass,
            first_mass / pair_mass,
            third_over_pair,
            first_mass * third_over_pair,
            second_mass * third_over_pair,
            first_mass + first_mass * third_over_pair,
            second_mass + second_mass * third_over_pair,
        )
        supplied_values = (
            witness.mass_fractions[0],
            witness.mass_fractions[1],
            witness.mass_fractions[2],
            witness.pair_mass,
            witness.alpha,
            witness.beta,
            witness.third_over_pair,
            witness.center_first,
            witness.center_second,
            witness.offset_first,
            witness.offset_second,
        )
        if (
            type(witness.mass_intervals) is not tuple
            or len(witness.mass_intervals) != 3
        ):
            return False
        intervals = (
            *witness.mass_intervals,
            witness.pair_mass_interval,
            witness.alpha_interval,
            witness.beta_interval,
            witness.third_over_pair_interval,
            witness.center_first_interval,
            witness.center_second_interval,
            witness.offset_first_interval,
            witness.offset_second_interval,
        )
        if (
            any(type(value) is not Fraction for value in supplied_values)
            or supplied_values != exact_values
            or any(type(interval) is not FloatInterval for interval in intervals)
            or any(
                type(endpoint) is not float
                for interval in intervals
                for endpoint in (interval.lower, interval.upper)
            )
        ):
            return False
        for exact, interval in zip(exact_values, intervals):
            if not (
                Fraction.from_float(interval.lower)
                <= exact
                <= Fraction.from_float(interval.upper)
            ):
                return False
            fresh = tight_fraction_float_interval(exact)
            if (interval.lower, interval.upper) != (fresh.lower, fresh.upper):
                return False
        return True
    except (AttributeError, OverflowError, TypeError, ValueError, ZeroDivisionError):
        return False


def tight_fraction_float_interval(value: Fraction) -> FloatInterval:
    """Return the tightest binary64 interval enclosing an exact finite rational."""

    if type(value) is not Fraction:
        raise TypeError("tight interval input must be an exact Fraction")
    exact = value
    try:
        candidate = float(exact)
    except OverflowError as exc:
        raise ValueError(
            "mass coefficient is outside the finite binary64 range"
        ) from exc
    if not math.isfinite(candidate):
        raise ValueError("mass coefficient is outside the finite binary64 range")
    candidate_fraction = Fraction.from_float(candidate)
    lower = candidate
    upper = candidate
    if candidate_fraction > exact:
        lower = math.nextafter(candidate, -math.inf)
    elif candidate_fraction < exact:
        upper = math.nextafter(candidate, math.inf)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("mass coefficient has no finite binary64 enclosure")
    return FloatInterval(lower, upper)
