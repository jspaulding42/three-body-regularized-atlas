"""Small outward-rounded interval helpers for scalar polynomial checks."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from functools import lru_cache
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np


@dataclass(frozen=True)
class FloatInterval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("interval endpoints must be finite")
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint cannot exceed upper endpoint")

    @classmethod
    def point(cls, value: float) -> "FloatInterval":
        value = float(value)
        return cls(value, value)

    def __add__(self, other: "FloatInterval") -> "FloatInterval":
        return FloatInterval(
            _round_down(self.lower + other.lower),
            _round_up(self.upper + other.upper),
        )

    def __sub__(self, other: "FloatInterval") -> "FloatInterval":
        return FloatInterval(
            _round_down(self.lower - other.upper),
            _round_up(self.upper - other.lower),
        )

    def __mul__(self, other: "FloatInterval") -> "FloatInterval":
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return FloatInterval(_round_down(min(products)), _round_up(max(products)))

    def as_tuple(self) -> tuple[float, float]:
        return self.lower, self.upper

    def scale(self, factor: float) -> "FloatInterval":
        factor = float(factor)
        products = (self.lower * factor, self.upper * factor)
        return FloatInterval(_round_down(min(products)), _round_up(max(products)))

    def reciprocal(self) -> "FloatInterval":
        if interval_contains_zero(self):
            raise ValueError("cannot divide by an interval containing zero")
        values = (1.0 / self.lower, 1.0 / self.upper)
        return FloatInterval(_round_down(min(values)), _round_up(max(values)))

    def __truediv__(self, other: "FloatInterval") -> "FloatInterval":
        return self * other.reciprocal()

    def positive_power(self, exponent: float) -> "FloatInterval":
        if self.lower <= 0.0:
            raise ValueError("positive power requires a strictly positive interval")
        if exponent in {-0.5, -1.5, -2.5}:
            # These are the Newton/LC force exponents.  Avoid relying on an
            # unspecified libm pow error bound: Decimal.sqrt and directed
            # arithmetic start from the exact binary endpoint values.
            return FloatInterval(
                _directed_negative_half_integer_power(
                    self.upper, exponent, upward=False
                ),
                _directed_negative_half_integer_power(
                    self.lower, exponent, upward=True
                ),
            )
        values = (self.lower**exponent, self.upper**exponent)
        return FloatInterval(_round_down(min(values)), _round_up(max(values)))


@dataclass(frozen=True)
class RationalInterval:
    """Exact rational interval for checker-side polynomial arithmetic."""

    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        lower = _coerce_fraction(self.lower)
        upper = _coerce_fraction(self.upper)
        if lower > upper:
            raise ValueError("interval lower endpoint cannot exceed upper endpoint")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def point(cls, value: object) -> "RationalInterval":
        fraction = _coerce_fraction(value)
        return cls(fraction, fraction)

    @classmethod
    def from_float_interval(
        cls,
        lower: float,
        upper: float | None = None,
    ) -> "RationalInterval":
        if upper is None:
            upper = lower
        return cls(_fraction_from_float(lower), _fraction_from_float(upper))

    def __add__(self, other: "RationalInterval") -> "RationalInterval":
        return RationalInterval(self.lower + other.lower, self.upper + other.upper)

    def __sub__(self, other: "RationalInterval") -> "RationalInterval":
        return RationalInterval(self.lower - other.upper, self.upper - other.lower)

    def __mul__(self, other: "RationalInterval") -> "RationalInterval":
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return RationalInterval(min(products), max(products))

    def scale(self, factor: object) -> "RationalInterval":
        factor = _coerce_fraction(factor)
        products = (self.lower * factor, self.upper * factor)
        return RationalInterval(min(products), max(products))

    def reciprocal(self) -> "RationalInterval":
        if rational_interval_contains_zero(self):
            raise ValueError("cannot divide by an interval containing zero")
        values = (Fraction(1, 1) / self.lower, Fraction(1, 1) / self.upper)
        return RationalInterval(min(values), max(values))

    def __truediv__(self, other: "RationalInterval") -> "RationalInterval":
        return self * other.reciprocal()

    def as_float_interval(self) -> FloatInterval:
        return FloatInterval(_round_down(float(self.lower)), _round_up(float(self.upper)))

    def as_tuple(self) -> tuple[Fraction, Fraction]:
        return self.lower, self.upper


def _coerce_fraction(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value), 1)
    if isinstance(value, Rational) and not isinstance(value, float):
        return Fraction(value)
    if isinstance(value, (float, np.floating)):
        return _fraction_from_float(float(value))
    return Fraction(value)


def _fraction_from_float(value: float) -> Fraction:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("rational interval endpoints must be finite")
    return Fraction.from_float(value)


def _round_down(value: float) -> float:
    return float(np.nextafter(float(value), -np.inf))


def _round_up(value: float) -> float:
    return float(np.nextafter(float(value), np.inf))


@lru_cache(maxsize=131072)
def directed_nonnegative_sqrt_endpoint(value: float, *, upward: bool) -> float:
    """Return a directed binary64 endpoint for ``sqrt(value)``.

    The input is the exact real number represented by the supplied finite
    binary64 value.  ``Decimal.sqrt`` is correctly rounded to nearest rather
    than according to the context direction, so an adjacent 100-digit Decimal
    supplies a strict endpoint before the final directed binary64 conversion.
    """

    if not np.isfinite(value) or value < 0.0:
        raise ValueError("square-root argument must be finite and nonnegative")
    if value == 0.0:
        return 0.0
    with localcontext() as context:
        context.prec = 100
        exact_value = Decimal.from_float(float(value))
        nearest_root = context.sqrt(exact_value)
        decimal_endpoint = (
            context.next_plus(nearest_root)
            if upward
            else context.next_minus(nearest_root)
        )
    candidate = float(decimal_endpoint)
    candidate_decimal = Decimal.from_float(candidate)
    if upward and candidate_decimal < decimal_endpoint:
        candidate = float(np.nextafter(candidate, np.inf))
    elif not upward and candidate_decimal > decimal_endpoint:
        candidate = float(np.nextafter(candidate, -np.inf))
    return candidate


@lru_cache(maxsize=131072)
def _directed_negative_half_integer_power(
    value: float,
    exponent: float,
    *,
    upward: bool,
) -> float:
    """Directed bound for x**(-1/2), x**(-3/2), or x**(-5/2).

    ``Decimal.sqrt`` is correctly rounded to nearest, regardless of the
    context's directed rounding mode.  Its adjacent context numbers therefore
    bracket the exact square root.  An upper reciprocal uses the lower square-
    root neighbor and downward denominator products; a lower reciprocal uses
    the upper neighbor and upward denominator products.  The final division
    and binary64 conversion are directed in the requested result direction.
    """

    if value <= 0.0 or not np.isfinite(value):
        raise ValueError("finite positive power base required")
    powers = {-0.5: 0, -1.5: 1, -2.5: 2}
    if exponent not in powers:
        raise ValueError("unsupported directed half-integer exponent")
    with localcontext() as context:
        context.prec = 100
        base = Decimal.from_float(float(value))
        nearest_root = context.sqrt(base)
        root_lower = context.next_minus(nearest_root)
        root_upper = context.next_plus(nearest_root)
    if root_lower <= 0:
        raise ValueError("decimal precision did not prove a positive sqrt lower bound")

    denominator_direction = ROUND_FLOOR if upward else ROUND_CEILING
    with localcontext() as context:
        context.prec = 100
        context.rounding = denominator_direction
        denominator = root_lower if upward else root_upper
        for _ in range(powers[exponent]):
            denominator = context.multiply(denominator, base)
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_CEILING if upward else ROUND_FLOOR
        result = context.divide(Decimal(1), denominator)
    candidate = float(result)
    candidate_decimal = Decimal.from_float(candidate)
    if upward and candidate_decimal < result:
        candidate = float(np.nextafter(candidate, np.inf))
    elif not upward and candidate_decimal > result:
        candidate = float(np.nextafter(candidate, -np.inf))
    return candidate


def interval_contains_zero(interval: FloatInterval) -> bool:
    return interval.lower <= 0.0 <= interval.upper


def interval_excludes_zero(interval: FloatInterval) -> bool:
    return not interval_contains_zero(interval)


def interval_sign(interval: FloatInterval) -> int:
    if interval.lower > 0.0:
        return 1
    if interval.upper < 0.0:
        return -1
    return 0


def rational_interval_contains_zero(interval: RationalInterval) -> bool:
    return interval.lower <= 0 <= interval.upper


def rational_interval_sign(interval: RationalInterval) -> int:
    if interval.lower > 0:
        return 1
    if interval.upper < 0:
        return -1
    return 0


def point_interval_coefficients(coefficients: np.ndarray) -> tuple[FloatInterval, ...]:
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1:
        raise ValueError("coefficients must be a one-dimensional array")
    return tuple(FloatInterval.point(float(coefficient)) for coefficient in coefficients)


def point_rational_interval_coefficients(
    coefficients: np.ndarray,
) -> tuple[RationalInterval, ...]:
    coefficients = np.asarray(coefficients, dtype=object)
    if coefficients.ndim != 1:
        raise ValueError("coefficients must be a one-dimensional array")
    return tuple(RationalInterval.point(coefficient) for coefficient in coefficients)


def coerce_interval_coefficients(coefficients: np.ndarray | tuple[FloatInterval, ...]) -> tuple[FloatInterval, ...]:
    coefficients = np.asarray(coefficients, dtype=object)
    if coefficients.ndim != 1:
        raise ValueError("coefficients must be a one-dimensional array")
    return tuple(
        coefficient if isinstance(coefficient, FloatInterval) else FloatInterval.point(float(coefficient))
        for coefficient in coefficients
    )


def coerce_rational_interval_coefficients(
    coefficients: np.ndarray | tuple[RationalInterval, ...],
) -> tuple[RationalInterval, ...]:
    coefficients = np.asarray(coefficients, dtype=object)
    if coefficients.ndim != 1:
        raise ValueError("coefficients must be a one-dimensional array")
    return tuple(
        coefficient
        if isinstance(coefficient, RationalInterval)
        else RationalInterval.point(coefficient)
        for coefficient in coefficients
    )


def interval_coefficients_as_tuples(coefficients: tuple[FloatInterval, ...]) -> tuple[tuple[float, float], ...]:
    return tuple(coefficient.as_tuple() for coefficient in coefficients)


def rational_interval_coefficients_as_tuples(
    coefficients: tuple[RationalInterval, ...],
) -> tuple[tuple[Fraction, Fraction], ...]:
    return tuple(coefficient.as_tuple() for coefficient in coefficients)


def interval_polyder(coefficients: np.ndarray | tuple[FloatInterval, ...]) -> tuple[FloatInterval, ...]:
    coefficient_intervals = coerce_interval_coefficients(coefficients)
    if len(coefficient_intervals) <= 1:
        return ()
    return tuple(coefficient_intervals[index].scale(index) for index in range(1, len(coefficient_intervals)))


def interval_array_derivative_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Differentiate serialized float arrays before any binary64 product rounding.

    Output coefficient ``n`` is the exact rational product
    ``(n + 1) * Fraction.from_float(serialized[n + 1])``, enclosed by the
    nearest outward binary64 endpoints.  This prevents a pre-rounded NumPy
    derivative coefficient from being treated as an exact point interval.
    """

    serialized = np.asarray(coefficients, dtype=float)
    if serialized.ndim < 1:
        raise ValueError("coefficients must have a degree axis")
    out = np.empty((max(0, serialized.shape[0] - 1), *serialized.shape[1:]), dtype=object)
    for index in np.ndindex(out.shape):
        degree = index[0] + 1
        exact = Fraction(degree) * Fraction.from_float(
            float(serialized[(degree, *index[1:])])
        )
        candidate = float(exact)
        candidate_q = Fraction.from_float(candidate)
        lower = (
            candidate
            if candidate_q <= exact
            else float(np.nextafter(candidate, -np.inf))
        )
        upper = (
            candidate
            if candidate_q >= exact
            else float(np.nextafter(candidate, np.inf))
        )
        out[index] = FloatInterval(lower, upper)
    return out


def rational_interval_polyder(
    coefficients: np.ndarray | tuple[RationalInterval, ...],
) -> tuple[RationalInterval, ...]:
    coefficient_intervals = coerce_rational_interval_coefficients(coefficients)
    if len(coefficient_intervals) <= 1:
        return ()
    return tuple(
        coefficient_intervals[index].scale(index)
        for index in range(1, len(coefficient_intervals))
    )


def interval_polynomial_eval(
    coefficients: np.ndarray | tuple[FloatInterval, ...],
    variable: FloatInterval,
) -> FloatInterval:
    """Evaluate a power-basis polynomial over an interval with Horner arithmetic.

    Coefficients are ordered as NumPy's ``np.polynomial.polynomial`` helpers
    expect: constant term first.
    """

    coefficients = coerce_interval_coefficients(coefficients)
    result = FloatInterval.point(0.0)
    for coefficient in reversed(coefficients):
        result = result * variable + coefficient
    return result


def rational_interval_polynomial_eval(
    coefficients: np.ndarray | tuple[RationalInterval, ...],
    variable: RationalInterval,
) -> RationalInterval:
    """Evaluate a power-basis polynomial using exact rational intervals."""

    coefficients = coerce_rational_interval_coefficients(coefficients)
    result = RationalInterval.point(0)
    for coefficient in reversed(coefficients):
        result = result * variable + coefficient
    return result


def interval_array_series_eval(coefficients: np.ndarray, variable: FloatInterval) -> np.ndarray:
    """Evaluate every component of an interval Taylor coefficient array."""

    coefficients = np.asarray(coefficients, dtype=object)
    if coefficients.ndim < 1:
        raise ValueError("coefficients must have a degree axis")
    out = np.empty(coefficients.shape[1:], dtype=object)
    for index in np.ndindex(out.shape):
        out[index] = interval_polynomial_eval(coefficients[(slice(None), *index)], variable)
    return out


def interval_array_as_tuples(values: np.ndarray) -> tuple[tuple[float, float], ...]:
    values = np.asarray(values, dtype=object)
    return tuple(
        (value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))).as_tuple()
        for value in values.reshape(-1)
    )


def interval_array_contains_point(intervals: np.ndarray, point: np.ndarray) -> bool:
    intervals = np.asarray(intervals, dtype=object)
    point = np.asarray(point, dtype=float)
    if intervals.shape != point.shape:
        return False
    for index in np.ndindex(intervals.shape):
        value = intervals[index]
        if value.lower > point[index] or point[index] > value.upper:
            return False
    return True


def zero_interval() -> FloatInterval:
    return FloatInterval.point(0.0)


def interval_series_product(
    left: np.ndarray | tuple[FloatInterval, ...],
    right: np.ndarray | tuple[FloatInterval, ...],
    max_degree: int,
) -> tuple[FloatInterval, ...]:
    if max_degree < 0:
        raise ValueError("max_degree cannot be negative")
    left = coerce_interval_coefficients(left)
    right = coerce_interval_coefficients(right)
    out = []
    for n in range(max_degree + 1):
        value = zero_interval()
        first = max(0, n - len(right) + 1)
        last = min(n, len(left) - 1)
        for k in range(first, last + 1):
            value = value + left[k] * right[n - k]
        out.append(value)
    return tuple(out)


def interval_series_add(
    left: tuple[FloatInterval, ...],
    right: tuple[FloatInterval, ...],
) -> tuple[FloatInterval, ...]:
    if len(left) != len(right):
        raise ValueError("interval series must have the same length")
    return tuple(left[index] + right[index] for index in range(len(left)))


def subtract_interval_from_constant_term(
    coefficients: tuple[FloatInterval, ...],
    value: FloatInterval,
) -> tuple[FloatInterval, ...]:
    if not coefficients:
        raise ValueError("coefficient series cannot be empty")
    out = list(coefficients)
    out[0] = out[0] - value
    return tuple(out)


def interval_series_power(
    base: np.ndarray | tuple[FloatInterval, ...],
    exponent: float,
    max_degree: int,
) -> tuple[FloatInterval, ...]:
    """Return interval coefficients enclosing ``base ** exponent``.

    The recurrence mirrors the point Taylor-series implementation and uses
    ``base * y' = exponent * base' * y``.
    """

    if max_degree < 0:
        raise ValueError("max_degree cannot be negative")
    coefficients = list(coerce_interval_coefficients(base))
    if len(coefficients) < max_degree + 1:
        coefficients.extend(zero_interval() for _ in range(max_degree + 1 - len(coefficients)))
    base_coefficients = tuple(coefficients[: max_degree + 1])
    if base_coefficients[0].lower <= 0.0:
        raise ValueError("series power requires a positive constant interval")

    out = [zero_interval() for _ in range(max_degree + 1)]
    out[0] = base_coefficients[0].positive_power(exponent)
    for n in range(1, max_degree + 1):
        right = zero_interval()
        for i in range(1, n + 1):
            right = right + base_coefficients[i].scale(i) * out[n - i]
        right = right.scale(exponent)

        left_known = zero_interval()
        for i in range(1, n):
            left_known = left_known + base_coefficients[i] * out[n - i].scale(n - i)

        out[n] = (right - left_known) / base_coefficients[0].scale(n)
    return tuple(out)
