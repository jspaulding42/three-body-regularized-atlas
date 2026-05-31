"""Analytic compactification of physical time for global-series experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .intervals import FloatInterval, interval_contains_zero, interval_polynomial_eval


@dataclass(frozen=True)
class CompactifiedTimeTargetCertificate:
    """Certificate for the map ``u = tanh(rate * t)``.

    This maps the full physical time line to the bounded parameter interval
    ``-1 < u < 1``. It is not a global three-body solution, but it is the
    analytic scalar compactification needed before a Sundman-style global
    series can be stated on a fixed bounded domain.
    """

    compact_parameter: float
    physical_time: float
    rate: float
    physical_time_interval: FloatInterval
    roundtrip_compact_parameter_interval: FloatInterval
    inverse_derivative_interval: FloatInterval

    @property
    def certified(self) -> bool:
        return bool(
            -1.0 < self.compact_parameter < 1.0
            and self.rate > 0.0
            and self.physical_time_interval.lower <= self.physical_time <= self.physical_time_interval.upper
            and self.roundtrip_compact_parameter_interval.lower
            <= self.compact_parameter
            <= self.roundtrip_compact_parameter_interval.upper
            and self.inverse_derivative_interval.lower > 0.0
        )


@dataclass(frozen=True)
class CompactifiedTimeTaylorSolution:
    """Taylor coefficients for physical time as a function of compact time."""

    center: float
    rate: float
    coefficients: np.ndarray

    @property
    def order(self) -> int:
        return int(self.coefficients.shape[0] - 1)

    @property
    def analytic_radius(self) -> float:
        return float(1.0 - abs(self.center))

    def physical_time_at_delta(self, delta: float) -> float:
        return float(np.polynomial.polynomial.polyval(float(delta), self.coefficients))

    def physical_time_enclosure_at_delta(self, delta: float) -> FloatInterval:
        value = self.physical_time_at_delta(delta)
        tail = compactified_time_taylor_tail_bound(self, delta)
        return FloatInterval(_round_down(value - tail), _round_up(value + tail))

    def physical_time_interval_at_delta(self, delta: FloatInterval) -> FloatInterval:
        return interval_polynomial_eval(tuple(FloatInterval.point(value) for value in self.coefficients), delta)


@dataclass(frozen=True)
class CompactifiedTimeTaylorResidualCertificate:
    """Coefficient residual certificate for the compact-time inverse ODE."""

    center: float
    rate: float
    coefficient_count: int
    residual_coefficients: tuple[float, ...]
    initial_value_residual: float
    analytic_radius: float
    tolerance: float = 1e-12

    @property
    def max_residual(self) -> float:
        return float(
            max(
                [abs(self.initial_value_residual), *[abs(value) for value in self.residual_coefficients]],
                default=0.0,
            )
        )

    @property
    def certified(self) -> bool:
        return bool(self.analytic_radius > 0.0 and self.max_residual <= self.tolerance)


@dataclass(frozen=True)
class CompactifiedTimeTaylorEvaluationCertificate:
    """Tail certificate for evaluating a compact-time Taylor polynomial."""

    compact_parameter: float
    delta: float
    polynomial_time: float
    exact_physical_time: float
    tail_bound: float
    time_enclosure: FloatInterval
    residual_certificate: CompactifiedTimeTaylorResidualCertificate

    @property
    def certified(self) -> bool:
        return bool(
            self.residual_certificate.certified
            and self.tail_bound >= 0.0
            and self.time_enclosure.lower <= self.exact_physical_time <= self.time_enclosure.upper
        )


def compact_parameter_from_physical_time(physical_time: float, *, rate: float = 1.0) -> float:
    """Map physical time to the bounded compact parameter ``(-1, 1)``."""

    rate = _validate_rate(rate)
    return float(np.tanh(rate * float(physical_time)))


def physical_time_from_compact_parameter(compact_parameter: float, *, rate: float = 1.0) -> float:
    """Invert the compact time map."""

    rate = _validate_rate(rate)
    compact_parameter = float(compact_parameter)
    if not -1.0 < compact_parameter < 1.0:
        raise ValueError("compact_parameter must lie strictly between -1 and 1")
    return float(np.arctanh(compact_parameter) / rate)


def compact_parameter_interval_from_physical_time_interval(
    physical_time: FloatInterval,
    *,
    rate: float = 1.0,
) -> FloatInterval:
    """Outward-rounded image of a physical-time interval under ``tanh(rate*t)``."""

    rate = _validate_rate(rate)
    physical_time = _as_interval(physical_time)
    scaled = physical_time.scale(rate)
    return FloatInterval(
        _round_down(np.tanh(scaled.lower)),
        _round_up(np.tanh(scaled.upper)),
    )


def physical_time_interval_from_compact_parameter_interval(
    compact_parameter: FloatInterval,
    *,
    rate: float = 1.0,
) -> FloatInterval:
    """Outward-rounded image of a compact-parameter interval under ``atanh(u)/rate``."""

    rate = _validate_rate(rate)
    compact_parameter = _as_interval(compact_parameter)
    if compact_parameter.lower <= -1.0 or compact_parameter.upper >= 1.0:
        raise ValueError("compact_parameter interval must lie strictly inside (-1, 1)")
    return FloatInterval(
        _round_down(np.arctanh(compact_parameter.lower) / rate),
        _round_up(np.arctanh(compact_parameter.upper) / rate),
    )


def certify_compactified_time_target(
    compact_parameter: float,
    *,
    rate: float = 1.0,
) -> CompactifiedTimeTargetCertificate:
    """Build a round-trip and monotonicity certificate for a compact target."""

    rate = _validate_rate(rate)
    compact_parameter = float(compact_parameter)
    physical_time = physical_time_from_compact_parameter(compact_parameter, rate=rate)
    compact_parameter_interval = FloatInterval(
        _round_down(compact_parameter),
        _round_up(compact_parameter),
    )
    physical_time_interval = physical_time_interval_from_compact_parameter_interval(
        compact_parameter_interval,
        rate=rate,
    )
    roundtrip_interval = compact_parameter_interval_from_physical_time_interval(
        physical_time_interval,
        rate=rate,
    )
    inverse_derivative_interval = inverse_time_derivative_interval(
        compact_parameter_interval,
        rate=rate,
    )
    return CompactifiedTimeTargetCertificate(
        compact_parameter=compact_parameter,
        physical_time=physical_time,
        rate=rate,
        physical_time_interval=physical_time_interval,
        roundtrip_compact_parameter_interval=roundtrip_interval,
        inverse_derivative_interval=inverse_derivative_interval,
    )


def construct_compactified_time_taylor_solution(
    *,
    center: float = 0.0,
    rate: float = 1.0,
    order: int,
) -> CompactifiedTimeTaylorSolution:
    """Construct coefficients for ``t(center + delta)``.

    The coefficients satisfy
    ``(1 - (center + delta)^2) dt/delta = 1 / rate``.
    """

    rate = _validate_rate(rate)
    center = float(center)
    if not -1.0 < center < 1.0:
        raise ValueError("center must lie strictly between -1 and 1")
    if order < 1:
        raise ValueError("order must be at least 1")

    coefficients = np.zeros(order + 1, dtype=float)
    coefficients[0] = physical_time_from_compact_parameter(center, rate=rate)
    p0 = 1.0 - center * center
    p1 = -2.0 * center
    p2 = -1.0
    derivative_coefficients = np.zeros(order, dtype=float)
    for degree in range(order):
        rhs = 1.0 / rate if degree == 0 else 0.0
        previous_1 = derivative_coefficients[degree - 1] if degree - 1 >= 0 else 0.0
        previous_2 = derivative_coefficients[degree - 2] if degree - 2 >= 0 else 0.0
        derivative_coefficients[degree] = (rhs - p1 * previous_1 - p2 * previous_2) / p0
        coefficients[degree + 1] = derivative_coefficients[degree] / float(degree + 1)
    return CompactifiedTimeTaylorSolution(center=center, rate=rate, coefficients=coefficients)


def certify_compactified_time_taylor_solution(
    solution: CompactifiedTimeTaylorSolution,
    *,
    coefficient_count: int | None = None,
    tolerance: float = 1e-12,
) -> CompactifiedTimeTaylorResidualCertificate:
    """Certify the Taylor coefficients satisfy the compact-time inverse ODE."""

    if coefficient_count is None:
        coefficient_count = solution.order
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > solution.order:
        raise ValueError("coefficient_count cannot exceed solution.order")

    p0 = 1.0 - solution.center * solution.center
    p1 = -2.0 * solution.center
    p2 = -1.0
    derivative = np.array(
        [(degree + 1) * solution.coefficients[degree + 1] for degree in range(solution.order)],
        dtype=float,
    )
    residuals = []
    for degree in range(coefficient_count):
        lhs = p0 * derivative[degree]
        if degree - 1 >= 0:
            lhs += p1 * derivative[degree - 1]
        if degree - 2 >= 0:
            lhs += p2 * derivative[degree - 2]
        rhs = 1.0 / solution.rate if degree == 0 else 0.0
        residuals.append(float(lhs - rhs))
    initial = solution.coefficients[0] - physical_time_from_compact_parameter(solution.center, rate=solution.rate)
    return CompactifiedTimeTaylorResidualCertificate(
        center=solution.center,
        rate=solution.rate,
        coefficient_count=coefficient_count,
        residual_coefficients=tuple(residuals),
        initial_value_residual=float(initial),
        analytic_radius=solution.analytic_radius,
        tolerance=float(tolerance),
    )


def compactified_time_taylor_tail_bound(solution: CompactifiedTimeTaylorSolution, delta: float) -> float:
    """Bound the omitted tail for the zero-centered compact-time series.

    Around zero, ``atanh(u) = sum u^(2k+1)/(2k+1)``. A simple geometric bound
    gives a certified enclosure for all ``|u| < 1``.
    """

    if solution.center != 0.0:
        raise ValueError("tail bound is currently implemented for the zero-centered compact-time series")
    radius = abs(float(delta))
    if radius >= 1.0:
        raise ValueError("delta must lie inside the compact-time convergence disk")
    if radius == 0.0:
        return 0.0
    tail = radius ** (solution.order + 1) / (solution.rate * (1.0 - radius))
    return float(np.nextafter(tail, np.inf))


def certify_compactified_time_taylor_evaluation(
    compact_parameter: float,
    *,
    rate: float = 1.0,
    order: int,
) -> CompactifiedTimeTaylorEvaluationCertificate:
    """Certify a zero-centered compact-time Taylor evaluation contains exact time."""

    solution = construct_compactified_time_taylor_solution(center=0.0, rate=rate, order=order)
    compact_parameter = float(compact_parameter)
    if not -1.0 < compact_parameter < 1.0:
        raise ValueError("compact_parameter must lie strictly between -1 and 1")
    polynomial_time = solution.physical_time_at_delta(compact_parameter)
    tail_bound = compactified_time_taylor_tail_bound(solution, compact_parameter)
    time_enclosure = FloatInterval(
        _round_down(polynomial_time - tail_bound),
        _round_up(polynomial_time + tail_bound),
    )
    residual_certificate = certify_compactified_time_taylor_solution(solution)
    exact_physical_time = physical_time_from_compact_parameter(compact_parameter, rate=rate)
    return CompactifiedTimeTaylorEvaluationCertificate(
        compact_parameter=compact_parameter,
        delta=compact_parameter,
        polynomial_time=polynomial_time,
        exact_physical_time=exact_physical_time,
        tail_bound=tail_bound,
        time_enclosure=time_enclosure,
        residual_certificate=residual_certificate,
    )


def inverse_time_derivative_interval(
    compact_parameter: FloatInterval,
    *,
    rate: float = 1.0,
) -> FloatInterval:
    """Return an interval for ``dt/du = 1 / (rate * (1 - u^2))``."""

    rate = _validate_rate(rate)
    compact_parameter = _as_interval(compact_parameter)
    if compact_parameter.lower <= -1.0 or compact_parameter.upper >= 1.0:
        raise ValueError("compact_parameter interval must lie strictly inside (-1, 1)")
    square_lower, square_upper = _interval_square_bounds(compact_parameter)
    denominator = FloatInterval(
        _round_down(rate * (1.0 - square_upper)),
        _round_up(rate * (1.0 - square_lower)),
    )
    if interval_contains_zero(denominator):
        raise ValueError("compact_parameter interval reaches a singular time endpoint")
    return denominator.reciprocal()


def _validate_rate(rate: float) -> float:
    rate = float(rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate must be a positive finite value")
    return rate


def _as_interval(value: FloatInterval | float) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


def _interval_square_bounds(value: FloatInterval) -> tuple[float, float]:
    candidates = (value.lower * value.lower, value.upper * value.upper)
    upper = _round_up(max(candidates))
    if value.lower <= 0.0 <= value.upper:
        lower = 0.0
    else:
        lower = max(0.0, _round_down(min(candidates)))
    return lower, upper


def _round_down(value: float) -> float:
    return float(np.nextafter(float(value), -np.inf))


def _round_up(value: float) -> float:
    return float(np.nextafter(float(value), np.inf))
