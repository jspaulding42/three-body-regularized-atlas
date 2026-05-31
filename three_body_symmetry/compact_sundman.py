"""Compactified Sundman-time Taylor charts.

This module applies the bounded-parameter lift after Sundman regularization:
``w = tanh(rate * s)``.  The constructed charts solve

``dq/dw = g(q) v ds/dw``
``dv/dw = g(q) a(q) ds/dw``
``dt/dw = g(q) ds/dw``

where ``g(q)`` is the Sundman distance product.  This is still a finite atlas
rather than Sundman's global series, but it is the right independent-variable
composition for that path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compact_dynamics import choose_compact_step_size, compact_time_factor_coefficients
from .compact_time import physical_time_from_compact_parameter
from .global_invariants import (
    AngularMomentumConservationCertificate,
    CenterOfMassMotionCertificate,
    EnergyConservationCertificate,
    LinearMomentumConservationCertificate,
    TripleCollisionExclusionCertificate,
    certify_interval_center_of_mass_motion,
    certify_interval_centered_angular_momentum_conservation,
    certify_interval_linear_momentum_conservation,
    certify_interval_total_energy_conservation,
    certify_nonzero_angular_momentum_excludes_triple_collision,
)
from .intervals import (
    FloatInterval,
    interval_array_contains_point,
    interval_array_series_eval,
    interval_polynomial_eval,
    interval_series_product,
    zero_interval,
)
from .series import (
    acceleration_coefficients,
    acceleration_interval_coefficients,
    integrate_reference,
    scalar_series_product,
)
from .sundman import pairwise_distance_product, sundman_factor_coefficients, sundman_factor_interval_coefficients
from .tail_bounds import (
    SundmanCauchyMajorantCertificate,
    SundmanIntervalCauchyMajorantCertificate,
    TailBoundCertificate,
    guarded_tail_certificate,
    interval_guarded_tail_certificate,
    sundman_cauchy_majorant_tail_certificate,
    sundman_interval_cauchy_majorant_tail_certificate,
)


Array = np.ndarray


@dataclass(frozen=True)
class CompactifiedSundmanTaylorSolution:
    """Taylor chart for Sundman-regularized dynamics in compact Sundman time."""

    position: Array
    velocity: Array
    physical_time: Array
    sundman_time: Array
    masses: Array
    sundman_rate: float
    distance_power: float = 1.0
    center: float = 0.0

    @property
    def order(self) -> int:
        return int(self.position.shape[0] - 1)

    @property
    def body_count(self) -> int:
        return int(self.position.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.position.shape[2])

    @property
    def analytic_radius(self) -> float:
        return float(1.0 - abs(self.center))

    def positions_at_w(self, compact_parameter: float) -> Array:
        return _evaluate(self.position, self.delta_from_compact_parameter(compact_parameter))

    def velocities_at_w(self, compact_parameter: float) -> Array:
        return _evaluate(self.velocity, self.delta_from_compact_parameter(compact_parameter))

    def physical_time_delta_at_w(self, compact_parameter: float) -> float:
        return float(_evaluate(self.physical_time[:, None], self.delta_from_compact_parameter(compact_parameter))[0])

    def sundman_time_at_w(self, compact_parameter: float) -> float:
        return float(_evaluate(self.sundman_time[:, None], self.delta_from_compact_parameter(compact_parameter))[0])

    def sundman_time_delta_at_w(self, compact_parameter: float) -> float:
        return self.sundman_time_at_w(compact_parameter) - self.sundman_time_at_w(self.center)

    def state_at_w(self, compact_parameter: float) -> Array:
        return np.concatenate(
            [
                self.positions_at_w(compact_parameter).reshape(-1),
                self.velocities_at_w(compact_parameter).reshape(-1),
            ]
        )

    def delta_from_compact_parameter(self, compact_parameter: float) -> float:
        compact_parameter = float(compact_parameter)
        if not -1.0 < compact_parameter < 1.0:
            raise ValueError("compact_parameter must lie strictly between -1 and 1")
        delta = compact_parameter - self.center
        if abs(delta) >= self.analytic_radius:
            raise ValueError("compact_parameter must lie inside this chart's compact-time convergence disk")
        return float(delta)


@dataclass(frozen=True)
class IntervalCompactifiedSundmanTaylorSolution:
    """Outward-rounded interval coefficients for compactified Sundman dynamics."""

    position: Array
    velocity: Array
    physical_time: Array
    sundman_time: Array
    masses: Array
    sundman_rate: float
    distance_power: float = 1.0
    center: float = 0.0

    @property
    def order(self) -> int:
        return int(self.position.shape[0] - 1)

    @property
    def body_count(self) -> int:
        return int(self.position.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.position.shape[2])

    @property
    def analytic_radius(self) -> float:
        return float(1.0 - abs(self.center))

    def contains_point_solution(self, solution: CompactifiedSundmanTaylorSolution) -> bool:
        if (
            solution.position.shape != self.position.shape
            or solution.velocity.shape != self.velocity.shape
            or solution.physical_time.shape != self.physical_time.shape
            or solution.sundman_time.shape != self.sundman_time.shape
        ):
            return False
        return bool(
            interval_array_contains_point(self.position, solution.position)
            and interval_array_contains_point(self.velocity, solution.velocity)
            and interval_array_contains_point(self.physical_time, solution.physical_time)
            and interval_array_contains_point(self.sundman_time, solution.sundman_time)
        )

    def positions_at_w(self, compact_parameter: float) -> Array:
        return interval_array_series_eval(self.position, FloatInterval.point(self.delta_from_compact_parameter(compact_parameter)))

    def velocities_at_w(self, compact_parameter: float) -> Array:
        return interval_array_series_eval(self.velocity, FloatInterval.point(self.delta_from_compact_parameter(compact_parameter)))

    def physical_time_delta_at_w(self, compact_parameter: float) -> FloatInterval:
        return interval_array_series_eval(
            self.physical_time[:, None],
            FloatInterval.point(self.delta_from_compact_parameter(compact_parameter)),
        )[0]

    def physical_time_delta_over_w_interval(self, compact_interval: FloatInterval) -> FloatInterval:
        return interval_polynomial_eval(self.physical_time, self.delta_interval_from_compact_interval(compact_interval))

    def sundman_time_at_w(self, compact_parameter: float) -> FloatInterval:
        return interval_array_series_eval(
            self.sundman_time[:, None],
            FloatInterval.point(self.delta_from_compact_parameter(compact_parameter)),
        )[0]

    def positions_over_w_interval(self, compact_interval: FloatInterval) -> Array:
        return interval_array_series_eval(self.position, self.delta_interval_from_compact_interval(compact_interval))

    def velocities_over_w_interval(self, compact_interval: FloatInterval) -> Array:
        return interval_array_series_eval(self.velocity, self.delta_interval_from_compact_interval(compact_interval))

    def state_at_w(self, compact_parameter: float) -> Array:
        return np.concatenate(
            [
                self.positions_at_w(compact_parameter).reshape(-1),
                self.velocities_at_w(compact_parameter).reshape(-1),
            ]
        )

    def state_over_w_interval(self, compact_interval: FloatInterval) -> Array:
        return np.concatenate(
            [
                self.positions_over_w_interval(compact_interval).reshape(-1),
                self.velocities_over_w_interval(compact_interval).reshape(-1),
            ]
        )

    def state_contains(self, state: Array, compact_parameter: float) -> bool:
        return interval_array_contains_point(self.state_at_w(compact_parameter), np.asarray(state, dtype=float))

    def state_over_interval_contains(self, state: Array, compact_interval: FloatInterval) -> bool:
        return interval_array_contains_point(self.state_over_w_interval(compact_interval), np.asarray(state, dtype=float))

    def delta_from_compact_parameter(self, compact_parameter: float) -> float:
        compact_parameter = float(compact_parameter)
        if not -1.0 < compact_parameter < 1.0:
            raise ValueError("compact_parameter must lie strictly between -1 and 1")
        delta = compact_parameter - self.center
        if abs(delta) >= self.analytic_radius:
            raise ValueError("compact_parameter must lie inside this chart's compact-time convergence disk")
        return float(delta)

    def delta_interval_from_compact_interval(self, compact_interval: FloatInterval) -> FloatInterval:
        compact_interval = _as_interval(compact_interval)
        _validate_compact_interval_inside_chart(
            compact_interval,
            center=self.center,
            analytic_radius=self.analytic_radius,
        )
        return FloatInterval(
            float(np.nextafter(compact_interval.lower - self.center, -np.inf)),
            float(np.nextafter(compact_interval.upper - self.center, np.inf)),
        )


@dataclass(frozen=True)
class CompactifiedSundmanResidualCertificate:
    """Coefficient residual certificate for compactified Sundman dynamics."""

    coefficient_count: int
    position_residual: Array
    velocity_residual: Array
    physical_time_residual: Array
    sundman_time_residual: Array
    tolerance: float = 1e-11

    @property
    def max_residual(self) -> float:
        return float(
            max(
                np.max(np.abs(self.position_residual)),
                np.max(np.abs(self.velocity_residual)),
                np.max(np.abs(self.physical_time_residual)),
                np.max(np.abs(self.sundman_time_residual)),
            )
        )

    @property
    def certified(self) -> bool:
        return bool(self.max_residual <= self.tolerance)


@dataclass(frozen=True)
class IntervalCompactifiedSundmanResidualCertificate:
    """Interval residual certificate for compactified Sundman dynamics."""

    coefficient_count: int
    position_residual: Array
    velocity_residual: Array
    physical_time_residual: Array
    sundman_time_residual: Array

    @property
    def certified(self) -> bool:
        return bool(
            _interval_array_contains_zero(self.position_residual)
            and _interval_array_contains_zero(self.velocity_residual)
            and _interval_array_contains_zero(self.physical_time_residual)
            and _interval_array_contains_zero(self.sundman_time_residual)
        )


@dataclass(frozen=True)
class CompactifiedSundmanCauchyMajorantCertificate:
    """A priori Cauchy tail bound in bounded compact-Sundman parameter."""

    retained_order: int
    step_size: float
    compact_radius: float
    sundman_image_radius: float
    center: float
    sundman_rate: float
    state_sup_bound: float
    tail_bound: float
    sundman_certificate: SundmanCauchyMajorantCertificate

    @property
    def ratio_bound(self) -> float:
        return abs(self.step_size) / self.compact_radius

    @property
    def coefficient_source(self) -> str:
        return "compactified_sundman_cauchy_majorant"

    @property
    def is_nontrivial(self) -> bool:
        return bool(
            np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
            and self.compact_radius > 0.0
            and self.sundman_image_radius <= self.sundman_certificate.s_radius
            and self.ratio_bound < 1.0
            and self.sundman_certificate.is_nontrivial
        )


@dataclass(frozen=True)
class CompactifiedSundmanIntervalCauchyMajorantCertificate:
    """Interval initial-state Cauchy tail bound in compact-Sundman parameter."""

    retained_order: int
    step_size: float
    compact_radius: float
    sundman_image_radius: float
    center: float
    sundman_rate: float
    state_sup_bound: float
    tail_bound: float
    sundman_certificate: SundmanIntervalCauchyMajorantCertificate

    @property
    def ratio_bound(self) -> float:
        return abs(self.step_size) / self.compact_radius

    @property
    def coefficient_source(self) -> str:
        return "compactified_sundman_interval_cauchy_majorant"

    @property
    def uses_interval_initial_state(self) -> bool:
        return True

    @property
    def is_nontrivial(self) -> bool:
        return bool(
            np.isfinite(self.tail_bound)
            and self.tail_bound >= 0.0
            and self.compact_radius > 0.0
            and self.sundman_image_radius <= self.sundman_certificate.s_radius
            and self.ratio_bound < 1.0
            and self.sundman_certificate.is_nontrivial
        )


@dataclass(frozen=True)
class SundmanCauchyRadiusStateEnvelopeCertificate:
    """State-envelope proof of a Sundman-time Cauchy-radius lower bound."""

    min_pair_distance: float
    max_pair_distance: float
    max_body_speed: float
    total_mass: float
    position_radius: float
    lower_squared_distance_bound: float
    acceleration_bound: float
    sundman_factor_bound: float
    velocity_radius_bound: float
    sundman_radius_lower_bound: float
    distance_power: float = 1.0

    @property
    def certified(self) -> bool:
        return bool(
            self.min_pair_distance > 0.0
            and self.max_pair_distance >= self.min_pair_distance
            and self.max_body_speed >= 0.0
            and self.total_mass > 0.0
            and self.position_radius > 0.0
            and self.lower_squared_distance_bound > 0.0
            and self.acceleration_bound > 0.0
            and self.sundman_factor_bound > 0.0
            and self.velocity_radius_bound > 0.0
            and self.sundman_radius_lower_bound > 0.0
            and self.distance_power > 0.0
        )

    def compact_sundman_image_radius(
        self,
        *,
        compact_radius: float,
        compact_center: float,
        sundman_rate: float,
    ) -> float:
        compact_radius = float(compact_radius)
        compact_center = float(compact_center)
        sundman_rate = float(sundman_rate)
        if compact_radius <= 0.0:
            raise ValueError("compact_radius must be positive")
        if sundman_rate <= 0.0:
            raise ValueError("sundman_rate must be positive")
        boundary = abs(compact_center) + compact_radius
        if boundary >= 1.0:
            raise ValueError("compact disk must stay inside (-1, 1)")
        return float(compact_radius / (sundman_rate * (1.0 - boundary * boundary)))

    def certifies_compact_cauchy_radius(
        self,
        *,
        compact_radius: float,
        compact_center: float,
        sundman_rate: float,
    ) -> bool:
        return bool(
            self.certified
            and self.compact_sundman_image_radius(
                compact_radius=compact_radius,
                compact_center=compact_center,
                sundman_rate=sundman_rate,
            )
            <= self.sundman_radius_lower_bound
        )

    def certifies_cauchy_capped_lower_step(
        self,
        *,
        compact_center: float,
        next_boundary_margin: float,
        sundman_rate: float,
        max_compact_step: float,
        alpha: float,
    ) -> bool:
        next_boundary_margin = float(next_boundary_margin)
        max_compact_step = float(max_compact_step)
        alpha = float(alpha)
        if next_boundary_margin <= 0.0:
            raise ValueError("next_boundary_margin must be positive")
        if max_compact_step <= 0.0:
            raise ValueError("max_compact_step must be positive")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        required_compact_radius = 2.0 * min(
            max_compact_step,
            alpha * next_boundary_margin,
        )
        return self.certifies_compact_cauchy_radius(
            compact_radius=required_compact_radius,
            compact_center=compact_center,
            sundman_rate=sundman_rate,
        )

    def eventual_shell_alpha_upper_bound(self, *, sundman_rate: float, eta: float) -> float:
        sundman_rate = float(sundman_rate)
        eta = float(eta)
        if sundman_rate <= 0.0:
            raise ValueError("sundman_rate must be positive")
        if not 0.0 < eta < 1.0:
            raise ValueError("eta must lie strictly between 0 and 1")
        cauchy_bound = 0.5 * sundman_rate * (1.0 - eta) * self.sundman_radius_lower_bound
        margin_bound = 0.5 * eta
        return float(min(cauchy_bound, margin_bound))

    def certifies_margin_only_eventual_lower_step(
        self,
        *,
        alpha: float,
        sundman_rate: float,
        eta: float,
    ) -> bool:
        alpha = float(alpha)
        return bool(
            alpha > 0.0
            and alpha <= self.eventual_shell_alpha_upper_bound(
                sundman_rate=sundman_rate,
                eta=eta,
            )
        )


@dataclass(frozen=True)
class CompactifiedSundmanCauchyCappedSegmentCountCertificate:
    """Eventual shell segment-count bound from a Cauchy-radius state envelope."""

    state_envelope: SundmanCauchyRadiusStateEnvelopeCertificate
    initial_boundary_margin: float
    contraction: float
    max_compact_step: float
    alpha: float
    eta: float
    sundman_rate: float
    first_boundary_fraction_shell_index: int
    required_uniform_sundman_radius_floor: float
    shell_segment_count_bound: int

    @property
    def schedule_certified(self) -> bool:
        return bool(
            np.isfinite(self.initial_boundary_margin)
            and 0.0 < self.initial_boundary_margin < 1.0
            and np.isfinite(self.contraction)
            and 0.0 < self.contraction < 1.0
            and np.isfinite(self.max_compact_step)
            and self.max_compact_step > 0.0
        )

    @property
    def lower_step_certified(self) -> bool:
        return bool(
            self.state_envelope.certified
            and np.isfinite(self.alpha)
            and self.alpha > 0.0
            and np.isfinite(self.eta)
            and 0.0 < self.eta < 1.0
            and np.isfinite(self.sundman_rate)
            and self.sundman_rate > 0.0
            and 2.0 * self.alpha <= self.eta
            and self.required_uniform_sundman_radius_floor
            <= self.state_envelope.sundman_radius_lower_bound * (1.0 + 1e-12)
            and self.state_envelope.certifies_margin_only_eventual_lower_step(
                alpha=self.alpha,
                sundman_rate=self.sundman_rate,
                eta=self.eta,
            )
        )

    @property
    def segment_count_certified(self) -> bool:
        return bool(
            self.schedule_certified
            and self.lower_step_certified
            and self.first_boundary_fraction_shell_index >= 0
            and self.shell_segment_count_bound > 0
        )

    @property
    def certified(self) -> bool:
        return self.segment_count_certified

    @property
    def segment_count_growth_ratio_bound(self) -> float:
        return 1.0 if self.segment_count_certified else float("inf")

    def next_boundary_margin(self, shell_index: int) -> float:
        shell_index = int(shell_index)
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        return float(self.initial_boundary_margin * self.contraction ** (shell_index + 1))

    def lower_step_bound(self, shell_index: int) -> float:
        delta_next = self.next_boundary_margin(shell_index)
        return float(min(self.max_compact_step, self.alpha * delta_next))

    def certifies_shell_lower_step(self, *, shell_index: int, compact_center: float) -> bool:
        shell_index = int(shell_index)
        compact_center = float(compact_center)
        if shell_index < self.first_boundary_fraction_shell_index:
            return False
        if not np.isfinite(compact_center) or abs(compact_center) >= 1.0:
            return False
        center_margin = 1.0 - abs(compact_center)
        delta_next = self.next_boundary_margin(shell_index)
        return bool(
            center_margin >= delta_next * (1.0 - 1e-12)
            and self.state_envelope.certifies_cauchy_capped_lower_step(
                compact_center=compact_center,
                next_boundary_margin=delta_next,
                sundman_rate=self.sundman_rate,
                max_compact_step=self.max_compact_step,
                alpha=self.alpha,
            )
        )


def construct_sundman_cauchy_radius_state_envelope(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    distance_power: float = 1.0,
) -> SundmanCauchyRadiusStateEnvelopeCertificate:
    """Derive the state-envelope lower bound for the Sundman Cauchy radius."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    distance_power = float(distance_power)
    if positions.ndim != 2 or velocities.shape != positions.shape or positions.shape[0] != 3:
        raise ValueError("positions and velocities must have matching three-body shapes")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    if distance_power <= 0.0:
        raise ValueError("distance_power must be positive")

    pair_distances = [
        float(np.linalg.norm(positions[j] - positions[i]))
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    min_pair_distance = min(pair_distances)
    if min_pair_distance <= 0.0:
        raise ValueError("initial data must be collision-free")
    max_pair_distance = max(pair_distances)
    max_body_speed = float(max(np.linalg.norm(velocity) for velocity in velocities))
    total_mass = float(np.sum(masses))
    position_radius = float(np.nextafter(min_pair_distance / 10.0, 0.0))
    lower_squared_distance_bound = float(
        np.nextafter((14.0 / 25.0) * min_pair_distance**2, 0.0)
    )
    distance_upper_bound = float(
        np.nextafter(max_pair_distance + min_pair_distance / 5.0, np.inf)
    )
    acceleration_bound = float(
        np.nextafter(
            total_mass
            * distance_upper_bound
            / (((14.0 / 25.0) ** 1.5) * min_pair_distance**3),
            np.inf,
        )
    )
    sundman_factor_bound = float(
        np.nextafter(distance_upper_bound ** (3.0 * distance_power), np.inf)
    )
    velocity_radius_bound = float(
        np.nextafter(np.sqrt(acceleration_bound * position_radius), np.inf)
    )
    position_self_map_radius = position_radius / (
        sundman_factor_bound * (max_body_speed + velocity_radius_bound)
    )
    velocity_self_map_radius = (
        np.sqrt(position_radius / acceleration_bound) / sundman_factor_bound
    )
    sundman_radius_lower_bound = float(
        np.nextafter(min(position_self_map_radius, velocity_self_map_radius), 0.0)
    )
    certificate = SundmanCauchyRadiusStateEnvelopeCertificate(
        min_pair_distance=float(min_pair_distance),
        max_pair_distance=float(max_pair_distance),
        max_body_speed=float(max_body_speed),
        total_mass=total_mass,
        position_radius=position_radius,
        lower_squared_distance_bound=lower_squared_distance_bound,
        acceleration_bound=acceleration_bound,
        sundman_factor_bound=sundman_factor_bound,
        velocity_radius_bound=velocity_radius_bound,
        sundman_radius_lower_bound=sundman_radius_lower_bound,
        distance_power=distance_power,
    )
    if not certificate.certified:
        raise ValueError("Sundman Cauchy-radius state envelope is not certified")
    return certificate


def certify_cauchy_capped_geometric_segment_count_from_state_envelope(
    state_envelope: SundmanCauchyRadiusStateEnvelopeCertificate,
    *,
    initial_boundary_margin: float,
    contraction: float,
    max_compact_step: float,
    alpha: float,
    eta: float,
    sundman_rate: float,
) -> CompactifiedSundmanCauchyCappedSegmentCountCertificate:
    """Close the eventual lower-step and segment-count recurrence.

    The proof combines the compact ``atanh`` Cauchy-radius bound with the
    geometric shell segment-count lemma.  It is conditional only on the supplied
    state envelope remaining valid for the future shell centers.
    """

    initial_boundary_margin = float(initial_boundary_margin)
    contraction = float(contraction)
    max_compact_step = float(max_compact_step)
    alpha = float(alpha)
    eta = float(eta)
    sundman_rate = float(sundman_rate)
    if not state_envelope.certified:
        raise ValueError("state envelope must certify")
    if not (
        np.isfinite(initial_boundary_margin)
        and 0.0 < initial_boundary_margin < 1.0
        and np.isfinite(contraction)
        and 0.0 < contraction < 1.0
        and np.isfinite(max_compact_step)
        and max_compact_step > 0.0
        and np.isfinite(alpha)
        and alpha > 0.0
        and np.isfinite(eta)
        and 0.0 < eta < 1.0
        and np.isfinite(sundman_rate)
        and sundman_rate > 0.0
        and 2.0 * alpha <= eta
    ):
        raise ValueError("Cauchy-capped segment-count parameters do not certify")
    if not state_envelope.certifies_margin_only_eventual_lower_step(
        alpha=alpha,
        sundman_rate=sundman_rate,
        eta=eta,
    ):
        raise ValueError("state envelope does not close the eventual lower-step bound")

    threshold_ratio = max_compact_step / (alpha * initial_boundary_margin)
    if threshold_ratio >= contraction:
        first_shell_index = 0
    else:
        first_shell_index = max(
            0,
            int(np.ceil(np.log(threshold_ratio) / np.log(contraction))) - 1,
        )
    required_uniform_sundman_radius_floor = float(
        2.0 * alpha / (sundman_rate * (1.0 - eta))
    )
    one_sided_bound = np.ceil(
        max(
            (1.0 - contraction) * initial_boundary_margin / max_compact_step,
            (1.0 - contraction) / (alpha * contraction),
        )
    )
    shell_segment_count_bound = int(2 * (one_sided_bound + 1))
    certificate = CompactifiedSundmanCauchyCappedSegmentCountCertificate(
        state_envelope=state_envelope,
        initial_boundary_margin=initial_boundary_margin,
        contraction=contraction,
        max_compact_step=max_compact_step,
        alpha=alpha,
        eta=eta,
        sundman_rate=sundman_rate,
        first_boundary_fraction_shell_index=first_shell_index,
        required_uniform_sundman_radius_floor=required_uniform_sundman_radius_floor,
        shell_segment_count_bound=shell_segment_count_bound,
    )
    if not certificate.certified:
        raise ValueError("Cauchy-capped geometric segment count did not certify")
    return certificate


CompactifiedSundmanTailCertificate = (
    TailBoundCertificate
    | CompactifiedSundmanCauchyMajorantCertificate
    | CompactifiedSundmanIntervalCauchyMajorantCertificate
)


@dataclass(frozen=True)
class CompactifiedSundmanCauchyCoverCertificate:
    """Finite cover of accepted compact-Sundman segments by Cauchy disks."""

    start_compact_parameter: float
    end_compact_parameter: float
    segment_starts: tuple[float, ...]
    segment_ends: tuple[float, ...]
    disk_centers: tuple[float, ...]
    disk_radii: tuple[float, ...]
    segment_tail_bounds: tuple[float, ...]
    segment_retained_orders: tuple[int, ...]
    segment_state_sup_bounds: tuple[float, ...]
    coefficient_sources: tuple[str, ...]
    certificate_nontrivial: tuple[bool, ...]

    @property
    def step_count(self) -> int:
        return len(self.segment_starts)

    @property
    def max_step_radius_ratio(self) -> float:
        ratios = []
        for start, end, center, radius in zip(
            self.segment_starts,
            self.segment_ends,
            self.disk_centers,
            self.disk_radii,
        ):
            if not np.isfinite(radius) or radius <= 0.0:
                return float("inf")
            ratios.append(max(abs(start - center), abs(end - center)) / radius)
        return float(max(ratios)) if ratios else float("inf")

    @property
    def total_tail_bound(self) -> float:
        if not self.tail_budget_certified:
            return float("inf")
        return float(sum(self.segment_tail_bounds))

    @property
    def max_tail_bound(self) -> float:
        if not self.tail_budget_certified:
            return float("inf")
        return float(max(self.segment_tail_bounds)) if self.segment_tail_bounds else float("inf")

    @property
    def tail_budget_certified(self) -> bool:
        if len(self.segment_tail_bounds) != self.step_count or self.step_count == 0:
            return False
        return bool(
            self.sources_are_cauchy
            and all(self.certificate_nontrivial)
            and all(np.isfinite(bound) and bound >= 0.0 for bound in self.segment_tail_bounds)
        )

    @property
    def retained_orders_certified(self) -> bool:
        if len(self.segment_retained_orders) != self.step_count or self.step_count == 0:
            return False
        return all(isinstance(order, int) and order >= 0 for order in self.segment_retained_orders)

    @property
    def state_sup_bounds_certified(self) -> bool:
        if len(self.segment_state_sup_bounds) != self.step_count or self.step_count == 0:
            return False
        return all(np.isfinite(bound) and bound >= 0.0 for bound in self.segment_state_sup_bounds)

    @property
    def min_retained_order(self) -> int:
        return min(self.segment_retained_orders) if self.retained_orders_certified else -1

    @property
    def max_retained_order(self) -> int:
        return max(self.segment_retained_orders) if self.retained_orders_certified else -1

    def tail_bound_outside(self, compact_interval: FloatInterval) -> float:
        """Sum segment tail bounds for segments not contained in ``compact_interval``."""

        if not self.tail_budget_certified:
            return float("inf")
        compact_interval = _as_interval(compact_interval)
        total = 0.0
        for start, end, tail_bound in zip(self.segment_starts, self.segment_ends, self.segment_tail_bounds):
            if not _compact_interval_contains_segment(compact_interval, start, end):
                total += tail_bound
        return float(total)

    def segment_count_outside(self, compact_interval: FloatInterval) -> int:
        """Count Cauchy-covered segments not contained in ``compact_interval``."""

        if not self.certified:
            return 0
        compact_interval = _as_interval(compact_interval)
        total = 0
        for start, end in zip(self.segment_starts, self.segment_ends):
            if not _compact_interval_contains_segment(compact_interval, start, end):
                total += 1
        return total

    def max_step_radius_ratio_outside(self, compact_interval: FloatInterval) -> float:
        """Max endpoint-to-Cauchy-radius ratio for segments outside ``compact_interval``."""

        if not self.certified:
            return float("inf")
        compact_interval = _as_interval(compact_interval)
        ratios: list[float] = []
        for start, end, center, radius in zip(
            self.segment_starts,
            self.segment_ends,
            self.disk_centers,
            self.disk_radii,
        ):
            if _compact_interval_contains_segment(compact_interval, start, end):
                continue
            if not np.isfinite(radius) or radius <= 0.0:
                return float("inf")
            ratios.append(max(abs(start - center), abs(end - center)) / radius)
        return float(max(ratios)) if ratios else 0.0

    def max_state_sup_bound_outside(self, compact_interval: FloatInterval) -> float:
        """Max Cauchy state supremum bound for segments outside ``compact_interval``."""

        if not self.certified:
            return float("inf")
        compact_interval = _as_interval(compact_interval)
        bounds: list[float] = []
        for start, end, state_sup_bound in zip(
            self.segment_starts,
            self.segment_ends,
            self.segment_state_sup_bounds,
        ):
            if _compact_interval_contains_segment(compact_interval, start, end):
                continue
            if not np.isfinite(state_sup_bound) or state_sup_bound < 0.0:
                return float("inf")
            bounds.append(state_sup_bound)
        return float(max(bounds)) if bounds else 0.0

    @property
    def sources_are_cauchy(self) -> bool:
        return all(source in _COMPACT_SUNDMAN_CAUCHY_SOURCES for source in self.coefficient_sources)

    @property
    def contiguous(self) -> bool:
        if self.step_count == 0:
            return False
        if not _compact_parameters_close(self.segment_starts[0], self.start_compact_parameter):
            return False
        if not _compact_parameters_close(self.segment_ends[-1], self.end_compact_parameter):
            return False

        direction = _cover_direction(self.start_compact_parameter, self.end_compact_parameter)
        for index, (start, end) in enumerate(zip(self.segment_starts, self.segment_ends)):
            if index > 0 and not _compact_parameters_close(start, self.segment_ends[index - 1]):
                return False
            step_direction = _cover_direction(start, end)
            if step_direction != 0.0 and direction != 0.0 and step_direction != direction:
                return False
        return True

    @property
    def certified(self) -> bool:
        lengths = {
            len(self.segment_starts),
            len(self.segment_ends),
            len(self.disk_centers),
            len(self.disk_radii),
            len(self.segment_tail_bounds),
            len(self.segment_retained_orders),
            len(self.segment_state_sup_bounds),
            len(self.coefficient_sources),
            len(self.certificate_nontrivial),
        }
        if len(lengths) != 1 or self.step_count == 0:
            return False
        if not (
            np.isfinite(self.start_compact_parameter)
            and np.isfinite(self.end_compact_parameter)
            and -1.0 < self.start_compact_parameter < 1.0
            and -1.0 < self.end_compact_parameter < 1.0
        ):
            return False
        if (
            not self.sources_are_cauchy
            or not all(self.certificate_nontrivial)
            or not self.tail_budget_certified
            or not self.retained_orders_certified
            or not self.state_sup_bounds_certified
        ):
            return False
        if not self.contiguous:
            return False

        for start, end, center, radius in zip(
            self.segment_starts,
            self.segment_ends,
            self.disk_centers,
            self.disk_radii,
        ):
            if not (
                np.isfinite(start)
                and np.isfinite(end)
                and np.isfinite(center)
                and np.isfinite(radius)
                and radius > 0.0
            ):
                return False
            if abs(center) + radius >= 1.0:
                return False
            if max(abs(start - center), abs(end - center)) >= radius:
                return False
        return True


@dataclass(frozen=True)
class CompactifiedSundmanCompactDomainCoverCertificate:
    """Finite certified Cauchy cover of a compact subinterval of ``(-1, 1)``."""

    compact_interval: FloatInterval
    covers: tuple[CompactifiedSundmanCauchyCoverCertificate, ...]

    @property
    def cover_count(self) -> int:
        return len(self.covers)

    @property
    def segment_count(self) -> int:
        return sum(cover.step_count for cover in self.covers)

    @property
    def boundary_margin(self) -> float:
        return float(min(self.compact_interval.lower + 1.0, 1.0 - self.compact_interval.upper))

    @property
    def max_step_radius_ratio(self) -> float:
        ratios = [cover.max_step_radius_ratio for cover in self.covers]
        return float(max(ratios)) if ratios else float("inf")

    @property
    def total_tail_bound(self) -> float:
        if not self.tail_budget_certified:
            return float("inf")
        return float(sum(cover.total_tail_bound for cover in self.covers))

    @property
    def max_tail_bound(self) -> float:
        if not self.tail_budget_certified:
            return float("inf")
        return float(max(cover.max_tail_bound for cover in self.covers)) if self.covers else float("inf")

    @property
    def tail_budget_certified(self) -> bool:
        return bool(self.covers and all(cover.tail_budget_certified for cover in self.covers))

    @property
    def min_retained_order(self) -> int:
        if not self.covers or not all(cover.retained_orders_certified for cover in self.covers):
            return -1
        return min(cover.min_retained_order for cover in self.covers)

    @property
    def max_retained_order(self) -> int:
        if not self.covers or not all(cover.retained_orders_certified for cover in self.covers):
            return -1
        return max(cover.max_retained_order for cover in self.covers)

    def tail_bound_outside(self, compact_interval: FloatInterval) -> float:
        if not self.tail_budget_certified:
            return float("inf")
        return float(sum(cover.tail_bound_outside(compact_interval) for cover in self.covers))

    def segment_count_outside(self, compact_interval: FloatInterval) -> int:
        if not self.certified:
            return 0
        return sum(cover.segment_count_outside(compact_interval) for cover in self.covers)

    def max_step_radius_ratio_outside(self, compact_interval: FloatInterval) -> float:
        if not self.certified:
            return float("inf")
        ratios = [cover.max_step_radius_ratio_outside(compact_interval) for cover in self.covers]
        return float(max(ratios)) if ratios else float("inf")

    def max_state_sup_bound_outside(self, compact_interval: FloatInterval) -> float:
        if not self.certified:
            return float("inf")
        bounds = [cover.max_state_sup_bound_outside(compact_interval) for cover in self.covers]
        return float(max(bounds)) if bounds else float("inf")

    @property
    def max_uncovered_gap(self) -> float:
        return _compact_sundman_cover_max_gap(self.covers, self.compact_interval)

    @property
    def certified(self) -> bool:
        if self.compact_interval.lower >= self.compact_interval.upper:
            return False
        if self.boundary_margin <= 0.0:
            return False
        if not self.covers or not all(cover.certified for cover in self.covers) or not self.tail_budget_certified:
            return False
        tolerance = _compact_parameter_tolerance(
            self.compact_interval.lower,
            self.compact_interval.upper,
        )
        return self.max_uncovered_gap <= tolerance


@dataclass(frozen=True)
class CompactifiedSundmanCompactDomainExhaustionPrefixCertificate:
    """Finite prefix of nested compact-domain covers inside compact Sundman time."""

    domain_covers: tuple[CompactifiedSundmanCompactDomainCoverCertificate, ...]

    @property
    def domain_count(self) -> int:
        return len(self.domain_covers)

    @property
    def segment_count(self) -> int:
        return sum(domain.segment_count for domain in self.domain_covers)

    @property
    def compact_intervals(self) -> tuple[FloatInterval, ...]:
        return tuple(domain.compact_interval for domain in self.domain_covers)

    @property
    def boundary_margins(self) -> tuple[float, ...]:
        return tuple(domain.boundary_margin for domain in self.domain_covers)

    @property
    def remaining_boundary_margin(self) -> float:
        return self.boundary_margins[-1] if self.boundary_margins else float("inf")

    @property
    def max_uncovered_gap(self) -> float:
        gaps = [domain.max_uncovered_gap for domain in self.domain_covers]
        return float(max(gaps)) if gaps else float("inf")

    @property
    def max_step_radius_ratio(self) -> float:
        ratios = [domain.max_step_radius_ratio for domain in self.domain_covers]
        return float(max(ratios)) if ratios else float("inf")

    @property
    def total_tail_bound(self) -> float:
        if not self.tail_budget_certified:
            return float("inf")
        return float(sum(domain.total_tail_bound for domain in self.domain_covers))

    @property
    def max_tail_bound(self) -> float:
        if not self.tail_budget_certified:
            return float("inf")
        return float(max(domain.max_tail_bound for domain in self.domain_covers)) if self.domain_covers else float("inf")

    @property
    def tail_budget_certified(self) -> bool:
        return bool(self.domain_covers and all(domain.tail_budget_certified for domain in self.domain_covers))

    @property
    def incremental_tail_bounds(self) -> tuple[float, ...]:
        if not self.tail_budget_certified:
            return ()
        bounds: list[float] = []
        for index, domain in enumerate(self.domain_covers):
            if index == 0:
                bounds.append(domain.total_tail_bound)
            else:
                bounds.append(domain.tail_bound_outside(self.domain_covers[index - 1].compact_interval))
        return tuple(bounds)

    @property
    def incremental_total_tail_bound(self) -> float:
        if not self.tail_budget_certified:
            return float("inf")
        return float(sum(self.incremental_tail_bounds))

    @property
    def max_incremental_tail_bound(self) -> float:
        if not self.tail_budget_certified:
            return float("inf")
        bounds = self.incremental_tail_bounds
        return float(max(bounds)) if bounds else float("inf")

    @property
    def min_retained_order(self) -> int:
        if not self.domain_covers or not all(domain.min_retained_order >= 0 for domain in self.domain_covers):
            return -1
        return min(domain.min_retained_order for domain in self.domain_covers)

    @property
    def max_retained_order(self) -> int:
        if not self.domain_covers or not all(domain.max_retained_order >= 0 for domain in self.domain_covers):
            return -1
        return max(domain.max_retained_order for domain in self.domain_covers)

    @property
    def nested_expanding(self) -> bool:
        if self.domain_count == 0:
            return False
        for previous, current in zip(self.compact_intervals, self.compact_intervals[1:]):
            tolerance = _compact_parameter_tolerance(
                previous.lower,
                previous.upper,
                current.lower,
                current.upper,
            )
            left_expands = current.lower <= previous.lower + tolerance
            right_expands = current.upper >= previous.upper - tolerance
            strictly_expands = (
                current.lower < previous.lower - tolerance
                or current.upper > previous.upper + tolerance
            )
            if not (left_expands and right_expands and strictly_expands):
                return False
        return True

    @property
    def boundary_margins_monotone(self) -> bool:
        for previous, current in zip(self.boundary_margins, self.boundary_margins[1:]):
            tolerance = _compact_parameter_tolerance(previous, current)
            if current > previous + tolerance:
                return False
        return True

    @property
    def certified(self) -> bool:
        return bool(
            self.domain_count > 0
            and all(domain.certified for domain in self.domain_covers)
            and self.tail_budget_certified
            and self.nested_expanding
            and self.boundary_margins_monotone
            and self.remaining_boundary_margin > 0.0
        )


@dataclass(frozen=True)
class CompactifiedSundmanGeometricExhaustionScheduleCertificate:
    """Finite prefix checked against a geometric approach to ``w = +/-1``."""

    prefix: CompactifiedSundmanCompactDomainExhaustionPrefixCertificate
    initial_boundary_margin: float
    contraction: float

    @property
    def prefix_length(self) -> int:
        return self.prefix.domain_count

    @property
    def schedule_defined(self) -> bool:
        return bool(
            np.isfinite(self.initial_boundary_margin)
            and np.isfinite(self.contraction)
            and 0.0 < self.initial_boundary_margin < 1.0
            and 0.0 < self.contraction < 1.0
        )

    @property
    def expected_boundary_margins(self) -> tuple[float, ...]:
        if not self.schedule_defined:
            return ()
        return tuple(
            float(self.initial_boundary_margin * self.contraction**index)
            for index in range(self.prefix_length)
        )

    @property
    def expected_compact_intervals(self) -> tuple[FloatInterval, ...]:
        return tuple(
            FloatInterval(-1.0 + margin, 1.0 - margin)
            for margin in self.expected_boundary_margins
        )

    @property
    def next_expected_boundary_margin(self) -> float:
        if not self.schedule_defined:
            return float("inf")
        return float(self.initial_boundary_margin * self.contraction**self.prefix_length)

    @property
    def total_tail_bound(self) -> float:
        return self.prefix.total_tail_bound

    @property
    def max_tail_bound(self) -> float:
        return self.prefix.max_tail_bound

    @property
    def tail_budget_certified(self) -> bool:
        return self.prefix.tail_budget_certified

    @property
    def incremental_tail_bounds(self) -> tuple[float, ...]:
        return self.prefix.incremental_tail_bounds

    @property
    def incremental_total_tail_bound(self) -> float:
        return self.prefix.incremental_total_tail_bound

    @property
    def max_incremental_tail_bound(self) -> float:
        return self.prefix.max_incremental_tail_bound

    @property
    def min_retained_order(self) -> int:
        return self.prefix.min_retained_order

    @property
    def max_retained_order(self) -> int:
        return self.prefix.max_retained_order

    @property
    def endpoint_limit_certified(self) -> bool:
        return self.schedule_defined

    @property
    def prefix_covers_schedule(self) -> bool:
        if not self.schedule_defined or self.prefix_length == 0:
            return False
        for domain, expected in zip(self.prefix.domain_covers, self.expected_compact_intervals):
            tolerance = _compact_parameter_tolerance(
                domain.compact_interval.lower,
                domain.compact_interval.upper,
                expected.lower,
                expected.upper,
            )
            if domain.compact_interval.lower > expected.lower + tolerance:
                return False
            if domain.compact_interval.upper < expected.upper - tolerance:
                return False
        return True

    @property
    def boundary_margins_follow_schedule(self) -> bool:
        if not self.schedule_defined or self.prefix_length == 0:
            return False
        for actual, expected in zip(self.prefix.boundary_margins, self.expected_boundary_margins):
            tolerance = _compact_parameter_tolerance(actual, expected)
            if actual > expected + tolerance:
                return False
        return True

    @property
    def certified(self) -> bool:
        return bool(
            self.prefix.certified
            and self.tail_budget_certified
            and self.endpoint_limit_certified
            and self.prefix_covers_schedule
            and self.boundary_margins_follow_schedule
        )

    @property
    def global_domain_certified(self) -> bool:
        return False

    @property
    def missing_global_induction_reason(self) -> str | None:
        if not self.certified:
            return "finite prefix does not satisfy the geometric exhaustion schedule"
        return (
            "finite prefix matches the geometric endpoint-exhaustion schedule, "
            "but no induction certificate proves all later domains"
        )


@dataclass(frozen=True)
class CompactifiedSundmanGeometricExhaustionExtensionCertificate:
    """Certified finite extension by the next geometric exhaustion domain."""

    previous_schedule: CompactifiedSundmanGeometricExhaustionScheduleCertificate
    added_domain: CompactifiedSundmanCompactDomainCoverCertificate
    extended_schedule: CompactifiedSundmanGeometricExhaustionScheduleCertificate

    @property
    def added_index(self) -> int:
        return self.previous_schedule.prefix_length

    @property
    def expected_added_boundary_margin(self) -> float:
        return self.previous_schedule.next_expected_boundary_margin

    @property
    def expected_added_compact_interval(self) -> FloatInterval | None:
        margin = self.expected_added_boundary_margin
        if not np.isfinite(margin) or not 0.0 < margin < 1.0:
            return None
        return FloatInterval(-1.0 + margin, 1.0 - margin)

    @property
    def added_domain_matches_schedule(self) -> bool:
        expected = self.expected_added_compact_interval
        if expected is None:
            return False
        tolerance = _compact_parameter_tolerance(
            self.added_domain.compact_interval.lower,
            self.added_domain.compact_interval.upper,
            expected.lower,
            expected.upper,
        )
        return bool(
            self.added_domain.compact_interval.lower <= expected.lower + tolerance
            and self.added_domain.compact_interval.upper >= expected.upper - tolerance
        )

    @property
    def schedule_parameters_preserved(self) -> bool:
        return bool(
            _compact_parameters_close(
                self.previous_schedule.initial_boundary_margin,
                self.extended_schedule.initial_boundary_margin,
            )
            and _compact_parameters_close(
                self.previous_schedule.contraction,
                self.extended_schedule.contraction,
            )
        )

    @property
    def prefix_extended_by_one(self) -> bool:
        if self.extended_schedule.prefix_length != self.previous_schedule.prefix_length + 1:
            return False
        return self.extended_schedule.prefix.domain_covers[-1] == self.added_domain

    @property
    def added_tail_bound(self) -> float:
        return self.added_domain.total_tail_bound

    @property
    def added_incremental_tail_bound(self) -> float:
        if not self.previous_schedule.prefix.domain_covers:
            return self.added_tail_bound
        return self.added_domain.tail_bound_outside(
            self.previous_schedule.prefix.domain_covers[-1].compact_interval
        )

    @property
    def added_min_retained_order(self) -> int:
        return self.added_domain.min_retained_order

    @property
    def added_max_retained_order(self) -> int:
        return self.added_domain.max_retained_order

    @property
    def total_tail_bound(self) -> float:
        return self.extended_schedule.total_tail_bound

    @property
    def incremental_total_tail_bound(self) -> float:
        return self.extended_schedule.incremental_total_tail_bound

    @property
    def tail_budget_certified(self) -> bool:
        return bool(
            self.added_domain.tail_budget_certified
            and self.extended_schedule.tail_budget_certified
            and np.isfinite(self.added_incremental_tail_bound)
            and self.added_incremental_tail_bound >= 0.0
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.previous_schedule.certified
            and self.added_domain.certified
            and self.tail_budget_certified
            and self.added_domain_matches_schedule
            and self.schedule_parameters_preserved
            and self.prefix_extended_by_one
            and self.extended_schedule.certified
        )

    @property
    def global_domain_certified(self) -> bool:
        return False

    @property
    def missing_global_induction_reason(self) -> str | None:
        if not self.certified:
            return "finite extension does not certify the next geometric exhaustion domain"
        return (
            "one more geometric domain is certified, but no induction certificate "
            "proves extension for every later domain"
        )


@dataclass(frozen=True)
class CompactifiedSundmanGeometricExhaustionExtensionChainCertificate:
    """Finite chain of certified geometric exhaustion extensions."""

    initial_schedule: CompactifiedSundmanGeometricExhaustionScheduleCertificate
    extensions: tuple[CompactifiedSundmanGeometricExhaustionExtensionCertificate, ...]

    @property
    def extension_count(self) -> int:
        return len(self.extensions)

    @property
    def final_schedule(self) -> CompactifiedSundmanGeometricExhaustionScheduleCertificate:
        return self.extensions[-1].extended_schedule if self.extensions else self.initial_schedule

    @property
    def added_indices(self) -> tuple[int, ...]:
        return tuple(extension.added_index for extension in self.extensions)

    @property
    def added_tail_bounds(self) -> tuple[float, ...]:
        return tuple(extension.added_tail_bound for extension in self.extensions)

    @property
    def added_incremental_tail_bounds(self) -> tuple[float, ...]:
        return tuple(extension.added_incremental_tail_bound for extension in self.extensions)

    @property
    def added_incremental_segment_counts(self) -> tuple[int, ...]:
        counts: list[int] = []
        for extension in self.extensions:
            previous_domains = extension.previous_schedule.prefix.domain_covers
            if not previous_domains:
                counts.append(extension.added_domain.segment_count)
            else:
                counts.append(extension.added_domain.segment_count_outside(previous_domains[-1].compact_interval))
        return tuple(counts)

    @property
    def added_max_step_radius_ratios(self) -> tuple[float, ...]:
        ratios: list[float] = []
        for extension in self.extensions:
            previous_domains = extension.previous_schedule.prefix.domain_covers
            if not previous_domains:
                ratios.append(extension.added_domain.max_step_radius_ratio)
            else:
                ratios.append(
                    extension.added_domain.max_step_radius_ratio_outside(previous_domains[-1].compact_interval)
                )
        return tuple(ratios)

    @property
    def max_added_step_radius_ratio(self) -> float:
        ratios = self.added_max_step_radius_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def added_max_state_sup_bounds(self) -> tuple[float, ...]:
        bounds: list[float] = []
        for extension in self.extensions:
            previous_domains = extension.previous_schedule.prefix.domain_covers
            if not previous_domains:
                state_bounds = [
                    bound
                    for cover in extension.added_domain.covers
                    for bound in cover.segment_state_sup_bounds
                    if np.isfinite(bound)
                ]
                bounds.append(float(max(state_bounds)) if state_bounds else float("inf"))
            else:
                bounds.append(
                    extension.added_domain.max_state_sup_bound_outside(previous_domains[-1].compact_interval)
                )
        return tuple(bounds)

    @property
    def added_segment_growth_ratios(self) -> tuple[float, ...]:
        counts = self.added_incremental_segment_counts
        if len(counts) < 2:
            return ()
        ratios: list[float] = []
        for previous, current in zip(counts, counts[1:]):
            if previous <= 0:
                return ()
            ratios.append(float(current / previous))
        return tuple(ratios)

    @property
    def max_added_segment_growth_ratio(self) -> float:
        ratios = self.added_segment_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def added_state_sup_growth_ratios(self) -> tuple[float, ...]:
        bounds = self.added_max_state_sup_bounds
        if len(bounds) < 2:
            return ()
        ratios: list[float] = []
        for previous, current in zip(bounds, bounds[1:]):
            if not np.isfinite(previous) or previous <= 0.0 or not np.isfinite(current):
                return ()
            ratios.append(float(current / previous))
        return tuple(ratios)

    @property
    def max_added_state_sup_growth_ratio(self) -> float:
        ratios = self.added_state_sup_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def added_cauchy_tail_denominator_factors(self) -> tuple[float, ...]:
        factors: list[float] = []
        for ratio in self.added_max_step_radius_ratios:
            if not np.isfinite(ratio) or ratio < 0.0 or ratio >= 1.0:
                factors.append(float("inf"))
            else:
                factors.append(float(1.0 / (1.0 - ratio)))
        return tuple(factors)

    @property
    def added_cauchy_tail_denominator_growth_ratios(self) -> tuple[float, ...]:
        factors = self.added_cauchy_tail_denominator_factors
        if len(factors) < 2:
            return ()
        ratios: list[float] = []
        for previous, current in zip(factors, factors[1:]):
            if not np.isfinite(previous) or previous <= 0.0 or not np.isfinite(current):
                return ()
            ratios.append(float(current / previous))
        return tuple(ratios)

    @property
    def max_cauchy_tail_denominator_growth_ratio(self) -> float:
        ratios = self.added_cauchy_tail_denominator_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def cauchy_order_growth_decay_factor_bound(self) -> float:
        if not self.arithmetic_retained_order_growth_certified:
            return float("inf")
        ratio = self.max_added_step_radius_ratio
        if not np.isfinite(ratio) or ratio < 0.0:
            return float("inf")
        return float(ratio**self.retained_order_increment)

    @property
    def cauchy_geometry_complexity_decay_factor_bound(self) -> float:
        if not self.arithmetic_retained_order_growth_certified:
            return float("inf")
        segment_growth = self.max_added_segment_growth_ratio
        order_factor = self.cauchy_order_growth_decay_factor_bound
        if not np.isfinite(segment_growth) or not np.isfinite(order_factor):
            return float("inf")
        return float(segment_growth * order_factor)

    @property
    def cauchy_majorant_decay_factor_bound(self) -> float:
        if not self.arithmetic_retained_order_growth_certified:
            return float("inf")
        factors = (
            self.max_added_segment_growth_ratio,
            self.max_added_state_sup_growth_ratio,
            self.max_cauchy_tail_denominator_growth_ratio,
            self.cauchy_order_growth_decay_factor_bound,
        )
        if not all(np.isfinite(factor) for factor in factors):
            return float("inf")
        result = 1.0
        for factor in factors:
            result *= factor
        return float(result)

    @property
    def cauchy_majorant_transition_factor_bounds(self) -> tuple[float, ...]:
        if not self.positive_retained_order_growth_certified:
            return ()
        factors: list[float] = []
        for segment_growth, state_growth, denominator_growth, step_radius_ratio, order_increment in zip(
            self.added_segment_growth_ratios,
            self.added_state_sup_growth_ratios,
            self.added_cauchy_tail_denominator_growth_ratios,
            self.added_max_step_radius_ratios[1:],
            self.added_retained_order_increments,
        ):
            components = (
                segment_growth,
                state_growth,
                denominator_growth,
                step_radius_ratio,
            )
            if not all(np.isfinite(component) for component in components):
                return ()
            factors.append(
                float(
                    segment_growth
                    * state_growth
                    * denominator_growth
                    * step_radius_ratio**order_increment
                )
            )
        return tuple(factors)

    @property
    def max_cauchy_majorant_transition_factor_bound(self) -> float:
        factors = self.cauchy_majorant_transition_factor_bounds
        return float(max(factors)) if factors else float("inf")

    @property
    def tail_transfer_growth_factors(self) -> tuple[float, ...]:
        structural_factors = self.cauchy_majorant_transition_factor_bounds
        tail_ratios = self.incremental_tail_growth_ratios
        if len(structural_factors) != len(tail_ratios):
            return ()
        factors: list[float] = []
        for tail_ratio, structural_factor in zip(tail_ratios, structural_factors):
            if not np.isfinite(tail_ratio) or not np.isfinite(structural_factor) or structural_factor <= 0.0:
                return ()
            factors.append(float(tail_ratio / structural_factor))
        return tuple(factors)

    @property
    def max_tail_transfer_growth_factor(self) -> float:
        factors = self.tail_transfer_growth_factors
        return float(max(factors)) if factors else float("inf")

    @property
    def tail_transfer_growth_ratios(self) -> tuple[float, ...]:
        factors = self.tail_transfer_growth_factors
        if len(factors) < 2:
            return ()
        ratios: list[float] = []
        for previous, current in zip(factors, factors[1:]):
            if not np.isfinite(previous) or previous <= 0.0 or not np.isfinite(current):
                return ()
            ratios.append(float(current / previous))
        return tuple(ratios)

    @property
    def max_tail_transfer_growth_ratio(self) -> float:
        ratios = self.tail_transfer_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def finite_prefix_uniform_safety_factor_limit(self) -> float:
        """Largest equal safety factor still compatible with finite-prefix summability."""

        if not self.finite_cauchy_majorant_decay_candidate_certified:
            return float("inf")
        factor = self.cauchy_majorant_decay_factor_bound
        exponent = 3 + self.retained_order_increment
        if not np.isfinite(factor) or factor <= 0.0 or exponent <= 0:
            return float("inf")
        tail_limit = factor ** (-1.0 / exponent)
        cauchy_limit = 1.0 / self.max_added_step_radius_ratio
        return float(min(tail_limit, cauchy_limit))

    @property
    def finite_cauchy_geometry_decay_candidate_certified(self) -> bool:
        return bool(
            self.adaptive_order_tail_decay_candidate_certified
            and self.added_segment_growth_ratios
            and self.max_added_step_radius_ratio < 1.0
            and self.cauchy_geometry_complexity_decay_factor_bound < 1.0
        )

    @property
    def finite_cauchy_majorant_decay_candidate_certified(self) -> bool:
        return bool(
            self.finite_cauchy_geometry_decay_candidate_certified
            and self.added_state_sup_growth_ratios
            and self.added_cauchy_tail_denominator_growth_ratios
            and self.cauchy_majorant_decay_factor_bound < 1.0
        )

    @property
    def total_added_tail_bound(self) -> float:
        if not self.tail_budget_certified:
            return float("inf")
        return float(sum(self.added_tail_bounds))

    @property
    def total_added_incremental_tail_bound(self) -> float:
        if not self.tail_budget_certified:
            return float("inf")
        return float(sum(self.added_incremental_tail_bounds))

    @property
    def total_tail_bound(self) -> float:
        return self.final_schedule.total_tail_bound

    @property
    def added_min_retained_orders(self) -> tuple[int, ...]:
        return tuple(extension.added_min_retained_order for extension in self.extensions)

    @property
    def added_max_retained_orders(self) -> tuple[int, ...]:
        return tuple(extension.added_max_retained_order for extension in self.extensions)

    @property
    def uniform_added_retained_orders(self) -> tuple[int, ...]:
        """Return the per-extension order when every added cover uses one retained order."""

        orders: list[int] = []
        for min_order, max_order in zip(self.added_min_retained_orders, self.added_max_retained_orders):
            if min_order < 0 or max_order < 0 or min_order != max_order:
                return ()
            orders.append(min_order)
        return tuple(orders)

    @property
    def added_retained_order_increments(self) -> tuple[int, ...]:
        orders = self.uniform_added_retained_orders
        if len(orders) < 2:
            return ()
        return tuple(current - previous for previous, current in zip(orders, orders[1:]))

    @property
    def added_retained_order_increment_growths(self) -> tuple[int, ...]:
        increments = self.added_retained_order_increments
        if len(increments) < 2:
            return ()
        return tuple(current - previous for previous, current in zip(increments, increments[1:]))

    @property
    def positive_retained_order_growth_certified(self) -> bool:
        increments = self.added_retained_order_increments
        return bool(increments and all(increment > 0 for increment in increments))

    @property
    def arithmetic_retained_order_growth_certified(self) -> bool:
        increments = self.added_retained_order_increments
        return bool(
            self.positive_retained_order_growth_certified
            and all(increment == increments[0] for increment in increments)
        )

    @property
    def arithmetic_retained_order_acceleration_certified(self) -> bool:
        growths = self.added_retained_order_increment_growths
        return bool(
            self.positive_retained_order_growth_certified
            and growths
            and all(growth > 0 for growth in growths)
            and all(growth == growths[0] for growth in growths)
        )

    @property
    def retained_order_increment(self) -> int:
        return self.added_retained_order_increments[0] if self.arithmetic_retained_order_growth_certified else 0

    @property
    def retained_order_increment_growth(self) -> int:
        return (
            self.added_retained_order_increment_growths[0]
            if self.arithmetic_retained_order_acceleration_certified
            else 0
        )

    @property
    def min_retained_order(self) -> int:
        return self.final_schedule.min_retained_order

    @property
    def max_retained_order(self) -> int:
        return self.final_schedule.max_retained_order

    @property
    def incremental_total_tail_bound(self) -> float:
        return self.final_schedule.incremental_total_tail_bound

    @property
    def max_tail_bound(self) -> float:
        return self.final_schedule.max_tail_bound

    @property
    def incremental_tail_growth_ratios(self) -> tuple[float, ...]:
        bounds = self.added_incremental_tail_bounds
        if len(bounds) < 2:
            return ()
        ratios = []
        for previous, current in zip(bounds, bounds[1:]):
            if not np.isfinite(previous) or previous <= 0.0 or not np.isfinite(current):
                return ()
            ratios.append(float(current / previous))
        return tuple(ratios)

    @property
    def max_incremental_tail_growth_ratio(self) -> float:
        ratios = self.incremental_tail_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def geometric_tail_decay_candidate_certified(self) -> bool:
        return bool(self.certified and self.incremental_tail_growth_ratios and self.max_incremental_tail_growth_ratio < 1.0)

    @property
    def adaptive_order_tail_decay_candidate_certified(self) -> bool:
        return bool(
            self.geometric_tail_decay_candidate_certified
            and self.arithmetic_retained_order_growth_certified
        )

    @property
    def accelerated_order_tail_decay_candidate_certified(self) -> bool:
        return bool(
            self.geometric_tail_decay_candidate_certified
            and self.arithmetic_retained_order_acceleration_certified
        )

    @property
    def finite_cauchy_majorant_transition_candidate_certified(self) -> bool:
        return bool(
            self.certified
            and self.positive_retained_order_growth_certified
            and self.cauchy_majorant_transition_factor_bounds
            and self.max_cauchy_majorant_transition_factor_bound < 1.0
        )

    @property
    def geometric_tail_remainder_bound(self) -> float:
        if not self.geometric_tail_decay_candidate_certified:
            return float("inf")
        ratio = self.max_incremental_tail_growth_ratio
        return float(self.added_incremental_tail_bounds[-1] * ratio / (1.0 - ratio))

    @property
    def adaptive_order_tail_remainder_bound(self) -> float:
        if not self.adaptive_order_tail_decay_candidate_certified:
            return float("inf")
        return self.geometric_tail_remainder_bound

    @property
    def tail_budget_certified(self) -> bool:
        return bool(
            self.initial_schedule.tail_budget_certified
            and self.extensions
            and all(extension.tail_budget_certified for extension in self.extensions)
        )

    @property
    def chain_connected(self) -> bool:
        current = self.initial_schedule
        for extension in self.extensions:
            if extension.previous_schedule != current:
                return False
            current = extension.extended_schedule
        return True

    @property
    def certified(self) -> bool:
        return bool(
            self.initial_schedule.certified
            and self.extension_count > 0
            and self.tail_budget_certified
            and self.chain_connected
            and all(extension.certified for extension in self.extensions)
            and self.final_schedule.certified
        )

    @property
    def global_domain_certified(self) -> bool:
        return False

    @property
    def missing_global_induction_reason(self) -> str | None:
        if not self.certified:
            return "finite extension chain is not certified"
        return (
            "finite chain certifies repeated scheduled extensions, but no induction "
            "certificate proves extension for every later domain"
        )


@dataclass(frozen=True)
class CompactifiedSundmanConditionalGlobalSummabilityCertificate:
    """Conditional infinite-tail certificate from a finite adaptive shell chain.

    This is not a global proof by itself.  It records explicit uniform bounds
    that an induction theorem would need to prove for all later shells.
    """

    chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate
    max_future_segment_growth_ratio: float
    max_future_state_sup_growth_ratio: float
    max_future_denominator_growth_ratio: float
    max_future_step_radius_ratio: float
    retained_order_increment: int
    envelope_source: str = "manual"

    @property
    def uniform_bounds_finite(self) -> bool:
        return bool(
            np.isfinite(self.max_future_segment_growth_ratio)
            and np.isfinite(self.max_future_state_sup_growth_ratio)
            and np.isfinite(self.max_future_denominator_growth_ratio)
            and np.isfinite(self.max_future_step_radius_ratio)
            and self.max_future_segment_growth_ratio >= 0.0
            and self.max_future_state_sup_growth_ratio >= 0.0
            and self.max_future_denominator_growth_ratio >= 0.0
            and 0.0 <= self.max_future_step_radius_ratio < 1.0
            and self.retained_order_increment > 0
        )

    @property
    def observed_prefix_within_uniform_bounds(self) -> bool:
        if not self.uniform_bounds_finite or not self.chain.finite_cauchy_majorant_decay_candidate_certified:
            return False
        return bool(
            self.retained_order_increment == self.chain.retained_order_increment
            and self.chain.max_added_segment_growth_ratio <= self.max_future_segment_growth_ratio
            and self.chain.max_added_state_sup_growth_ratio <= self.max_future_state_sup_growth_ratio
            and self.chain.max_cauchy_tail_denominator_growth_ratio <= self.max_future_denominator_growth_ratio
            and self.chain.max_added_step_radius_ratio <= self.max_future_step_radius_ratio
        )

    @property
    def conditional_tail_decay_factor_bound(self) -> float:
        if not self.uniform_bounds_finite:
            return float("inf")
        return float(
            self.max_future_segment_growth_ratio
            * self.max_future_state_sup_growth_ratio
            * self.max_future_denominator_growth_ratio
            * self.max_future_step_radius_ratio**self.retained_order_increment
        )

    @property
    def future_envelope_growth_factor_bound(self) -> float:
        if not self.uniform_bounds_finite:
            return float("inf")
        return float(
            self.max_future_segment_growth_ratio
            * self.max_future_state_sup_growth_ratio
            * self.max_future_denominator_growth_ratio
        )

    @property
    def required_retained_order_increment_for_summability(self) -> int:
        """Smallest positive order increment making the future envelope summable."""

        if not self.uniform_bounds_finite:
            return -1
        growth = self.future_envelope_growth_factor_bound
        step_ratio = self.max_future_step_radius_ratio
        if not np.isfinite(growth) or growth < 0.0:
            return -1
        if growth == 0.0 or step_ratio == 0.0:
            return 1
        if not 0.0 < step_ratio < 1.0:
            return -1
        if growth * step_ratio < 1.0:
            return 1

        quotient = np.log(1.0 / growth) / np.log(step_ratio)
        if not np.isfinite(quotient):
            return -1
        increment = max(1, int(np.floor(quotient)) + 1)
        while growth * step_ratio**increment >= 1.0:
            increment += 1
        while increment > 1 and growth * step_ratio**(increment - 1) < 1.0:
            increment -= 1
        return increment

    @property
    def retained_order_increment_surplus(self) -> int:
        required = self.required_retained_order_increment_for_summability
        if required <= 0:
            return -1
        return int(self.retained_order_increment - required)

    @property
    def retained_order_increment_sufficient(self) -> bool:
        required = self.required_retained_order_increment_for_summability
        return bool(required > 0 and self.retained_order_increment >= required)

    @property
    def max_step_radius_ratio_for_summability(self) -> float:
        if not self.uniform_bounds_finite:
            return float("inf")
        growth = self.future_envelope_growth_factor_bound
        if not np.isfinite(growth) or growth <= 0.0:
            return float("inf")
        return float((1.0 / growth) ** (1.0 / self.retained_order_increment))

    @property
    def step_radius_summability_slack(self) -> float:
        limit = self.max_step_radius_ratio_for_summability
        if not np.isfinite(limit):
            return float("inf")
        return float(limit - self.max_future_step_radius_ratio)

    @property
    def cauchy_boundary_step_radius_slack(self) -> float:
        if not self.uniform_bounds_finite:
            return float("-inf")
        return float(1.0 - self.max_future_step_radius_ratio)

    @property
    def tail_decay_slack(self) -> float:
        if not self.uniform_bounds_finite:
            return float("-inf")
        return float(1.0 - self.conditional_tail_decay_factor_bound)

    @property
    def conditional_global_summability_certified(self) -> bool:
        return bool(
            self.observed_prefix_within_uniform_bounds
            and self.retained_order_increment_sufficient
            and self.chain.added_incremental_tail_bounds
        )

    @property
    def conditional_future_tail_remainder_bound(self) -> float:
        if not self.conditional_global_summability_certified:
            return float("inf")
        factor = self.conditional_tail_decay_factor_bound
        return float(self.chain.added_incremental_tail_bounds[-1] * factor / (1.0 - factor))

    @property
    def conditional_total_incremental_tail_bound(self) -> float:
        if not self.conditional_global_summability_certified:
            return float("inf")
        return float(self.chain.incremental_total_tail_bound + self.conditional_future_tail_remainder_bound)

    @property
    def global_domain_certified(self) -> bool:
        return False

    @property
    def missing_global_induction_reason(self) -> str | None:
        if not self.conditional_global_summability_certified:
            return "the supplied future-shell envelope does not imply a summable Cauchy-majorant tail"
        if self.envelope_source == "finite_prefix":
            return (
                "future shell summability follows from the finite-prefix-derived envelope only if an "
                "induction proof shows that this envelope persists for every later shell"
            )
        return (
            "future shell summability is conditional on proving the supplied uniform "
            "segment, state-bound, denominator, and step-radius envelopes for every later shell"
        )


@dataclass(frozen=True)
class CompactifiedSundmanTailTransferGlobalSummabilityCertificate:
    """Conditional global summability with an explicit future tail-transfer bound."""

    conditional_certificate: CompactifiedSundmanConditionalGlobalSummabilityCertificate
    max_future_tail_transfer_growth_factor: float
    tail_transfer_source: str = "manual"

    @property
    def tail_transfer_bound_finite(self) -> bool:
        return bool(
            np.isfinite(self.max_future_tail_transfer_growth_factor)
            and self.max_future_tail_transfer_growth_factor >= 0.0
        )

    @property
    def tail_transfer_adjusted_tail_decay_factor_bound(self) -> float:
        if not self.tail_transfer_bound_finite:
            return float("inf")
        return float(
            self.conditional_certificate.conditional_tail_decay_factor_bound
            * self.max_future_tail_transfer_growth_factor
        )

    @property
    def tail_transfer_adjusted_future_envelope_growth_factor_bound(self) -> float:
        if not self.tail_transfer_bound_finite:
            return float("inf")
        growth = self.conditional_certificate.future_envelope_growth_factor_bound
        if not np.isfinite(growth):
            return float("inf")
        return float(growth * self.max_future_tail_transfer_growth_factor)

    @property
    def max_tail_transfer_growth_factor_for_summability(self) -> float:
        factor = self.conditional_certificate.conditional_tail_decay_factor_bound
        if not np.isfinite(factor) or factor <= 0.0:
            return float("inf")
        return float(1.0 / factor)

    @property
    def tail_transfer_summability_slack(self) -> float:
        limit = self.max_tail_transfer_growth_factor_for_summability
        if not np.isfinite(limit):
            return float("inf")
        return float(limit - self.max_future_tail_transfer_growth_factor)

    @property
    def required_retained_order_increment_for_tail_transfer_summability(self) -> int:
        if not self.tail_transfer_bound_finite or not self.conditional_certificate.uniform_bounds_finite:
            return -1
        growth = self.tail_transfer_adjusted_future_envelope_growth_factor_bound
        step_ratio = self.conditional_certificate.max_future_step_radius_ratio
        if not np.isfinite(growth) or growth < 0.0:
            return -1
        if growth == 0.0 or step_ratio == 0.0:
            return 1
        if not 0.0 < step_ratio < 1.0:
            return -1
        if growth * step_ratio < 1.0:
            return 1

        quotient = np.log(1.0 / growth) / np.log(step_ratio)
        if not np.isfinite(quotient):
            return -1
        increment = max(1, int(np.floor(quotient)) + 1)
        while growth * step_ratio**increment >= 1.0:
            increment += 1
        while increment > 1 and growth * step_ratio**(increment - 1) < 1.0:
            increment -= 1
        return increment

    @property
    def retained_order_increment_tail_transfer_surplus(self) -> int:
        required = self.required_retained_order_increment_for_tail_transfer_summability
        if required <= 0:
            return -1
        return int(self.conditional_certificate.retained_order_increment - required)

    @property
    def retained_order_increment_tail_transfer_sufficient(self) -> bool:
        required = self.required_retained_order_increment_for_tail_transfer_summability
        return bool(required > 0 and self.conditional_certificate.retained_order_increment >= required)

    @property
    def max_step_radius_ratio_for_tail_transfer_summability(self) -> float:
        if not self.tail_transfer_bound_finite or not self.conditional_certificate.uniform_bounds_finite:
            return float("inf")
        growth = self.tail_transfer_adjusted_future_envelope_growth_factor_bound
        increment = self.conditional_certificate.retained_order_increment
        if not np.isfinite(growth) or growth <= 0.0 or increment <= 0:
            return float("inf")
        return float((1.0 / growth) ** (1.0 / increment))

    @property
    def step_radius_tail_transfer_summability_slack(self) -> float:
        limit = self.max_step_radius_ratio_for_tail_transfer_summability
        if not np.isfinite(limit):
            return float("inf")
        return float(limit - self.conditional_certificate.max_future_step_radius_ratio)

    @property
    def tail_transfer_adjusted_global_summability_certified(self) -> bool:
        return bool(
            self.conditional_certificate.observed_prefix_within_uniform_bounds
            and self.retained_order_increment_tail_transfer_sufficient
            and self.tail_transfer_bound_finite
            and self.conditional_certificate.chain.added_incremental_tail_bounds
        )

    @property
    def tail_transfer_adjusted_future_tail_remainder_bound(self) -> float:
        if not self.tail_transfer_adjusted_global_summability_certified:
            return float("inf")
        factor = self.tail_transfer_adjusted_tail_decay_factor_bound
        tail = self.conditional_certificate.chain.added_incremental_tail_bounds[-1]
        return float(tail * factor / (1.0 - factor))

    @property
    def tail_transfer_adjusted_total_incremental_tail_bound(self) -> float:
        if not self.tail_transfer_adjusted_global_summability_certified:
            return float("inf")
        return float(
            self.conditional_certificate.chain.incremental_total_tail_bound
            + self.tail_transfer_adjusted_future_tail_remainder_bound
        )

    @property
    def global_domain_certified(self) -> bool:
        return False

    @property
    def missing_global_induction_reason(self) -> str | None:
        if not self.tail_transfer_adjusted_global_summability_certified:
            return "the supplied structural envelopes and tail-transfer bound do not imply a summable tail"
        return (
            "tail-transfer-adjusted future summability is conditional on proving the structural "
            "future-shell envelopes and the tail-transfer bound for every later shell"
        )


@dataclass(frozen=True)
class CompactifiedSundmanGeometricTailTransferGlobalSummabilityCertificate:
    """Conditional summability when future tail-transfer factors may grow geometrically.

    If the transfer multiplier can grow by at most ``gamma`` per future shell, a
    retained-order schedule whose transition increments grow by ``q`` per shell
    contributes an extra ``step_radius**q`` decay factor each shell.  The
    condition ``gamma * step_radius**q < 1`` is the finite certificate that this
    acceleration absorbs the geometric transfer growth.
    """

    conditional_certificate: CompactifiedSundmanConditionalGlobalSummabilityCertificate
    initial_tail_transfer_growth_factor_bound: float
    max_future_tail_transfer_growth_ratio: float
    retained_order_increment_growth: int
    tail_transfer_source: str = "manual"

    @property
    def tail_transfer_growth_bounds_finite(self) -> bool:
        return bool(
            np.isfinite(self.initial_tail_transfer_growth_factor_bound)
            and self.initial_tail_transfer_growth_factor_bound >= 0.0
            and np.isfinite(self.max_future_tail_transfer_growth_ratio)
            and self.max_future_tail_transfer_growth_ratio >= 0.0
            and self.retained_order_increment_growth >= 0
        )

    @property
    def initial_tail_transfer_adjusted_tail_decay_factor_bound(self) -> float:
        if not self.tail_transfer_growth_bounds_finite:
            return float("inf")
        return float(
            self.conditional_certificate.conditional_tail_decay_factor_bound
            * self.initial_tail_transfer_growth_factor_bound
        )

    @property
    def tail_transfer_growth_ratio_absorption_factor_bound(self) -> float:
        if not self.tail_transfer_growth_bounds_finite or not self.conditional_certificate.uniform_bounds_finite:
            return float("inf")
        return float(
            self.max_future_tail_transfer_growth_ratio
            * self.conditional_certificate.max_future_step_radius_ratio**self.retained_order_increment_growth
        )

    @property
    def max_future_tail_transfer_growth_ratio_for_order_acceleration(self) -> float:
        if not self.conditional_certificate.uniform_bounds_finite:
            return float("inf")
        step_ratio = self.conditional_certificate.max_future_step_radius_ratio
        growth = self.retained_order_increment_growth
        if growth < 0 or not np.isfinite(step_ratio) or not 0.0 <= step_ratio < 1.0:
            return float("inf")
        if growth == 0:
            return 1.0
        if step_ratio == 0.0:
            return float("inf")
        return float(1.0 / step_ratio**growth)

    @property
    def tail_transfer_growth_ratio_slack(self) -> float:
        limit = self.max_future_tail_transfer_growth_ratio_for_order_acceleration
        if not np.isfinite(limit):
            return float("inf")
        return float(limit - self.max_future_tail_transfer_growth_ratio)

    @property
    def required_retained_order_increment_growth_for_tail_transfer_ratio(self) -> int:
        if not self.tail_transfer_growth_bounds_finite or not self.conditional_certificate.uniform_bounds_finite:
            return -1
        gamma = self.max_future_tail_transfer_growth_ratio
        step_ratio = self.conditional_certificate.max_future_step_radius_ratio
        if not np.isfinite(gamma) or gamma < 0.0:
            return -1
        if gamma < 1.0:
            return 0
        if not 0.0 < step_ratio < 1.0:
            return -1

        quotient = np.log(1.0 / gamma) / np.log(step_ratio)
        if not np.isfinite(quotient):
            return -1
        growth = max(0, int(np.floor(quotient)) + 1)
        while gamma * step_ratio**growth >= 1.0:
            growth += 1
        while growth > 0 and gamma * step_ratio ** (growth - 1) < 1.0:
            growth -= 1
        return growth

    @property
    def retained_order_increment_growth_surplus(self) -> int:
        required = self.required_retained_order_increment_growth_for_tail_transfer_ratio
        if required < 0:
            return -1
        return int(self.retained_order_increment_growth - required)

    @property
    def retained_order_increment_growth_sufficient(self) -> bool:
        required = self.required_retained_order_increment_growth_for_tail_transfer_ratio
        return bool(required >= 0 and self.retained_order_increment_growth >= required)

    @property
    def geometric_tail_transfer_growth_absorbed(self) -> bool:
        factor = self.tail_transfer_growth_ratio_absorption_factor_bound
        return bool(np.isfinite(factor) and factor < 1.0)

    @property
    def geometric_tail_transfer_global_summability_certified(self) -> bool:
        initial_factor = self.initial_tail_transfer_adjusted_tail_decay_factor_bound
        return bool(
            self.conditional_certificate.observed_prefix_within_uniform_bounds
            and self.tail_transfer_growth_bounds_finite
            and np.isfinite(initial_factor)
            and initial_factor < 1.0
            and self.geometric_tail_transfer_growth_absorbed
            and self.retained_order_increment_growth_sufficient
            and self.conditional_certificate.chain.added_incremental_tail_bounds
        )

    @property
    def geometric_tail_transfer_future_tail_remainder_bound(self) -> float:
        if not self.geometric_tail_transfer_global_summability_certified:
            return float("inf")
        base_tail = float(self.conditional_certificate.chain.added_incremental_tail_bounds[-1])
        first_ratio = self.initial_tail_transfer_adjusted_tail_decay_factor_bound
        ratio_growth = self.tail_transfer_growth_ratio_absorption_factor_bound
        if first_ratio == 0.0:
            return 0.0

        term = base_tail
        total = 0.0
        ratio = first_ratio
        for _ in range(10000):
            if not np.isfinite(term) or not np.isfinite(ratio):
                return float("inf")
            term *= ratio
            total += term
            next_ratio = ratio * ratio_growth
            if next_ratio == 0.0:
                return float(total)
            if 0.0 < next_ratio < 0.5:
                return float(total + term * next_ratio / (1.0 - next_ratio))
            if abs(term) < np.finfo(float).tiny:
                return float(total)
            ratio = next_ratio
        return float("inf")

    @property
    def geometric_tail_transfer_total_incremental_tail_bound(self) -> float:
        if not self.geometric_tail_transfer_global_summability_certified:
            return float("inf")
        return float(
            self.conditional_certificate.chain.incremental_total_tail_bound
            + self.geometric_tail_transfer_future_tail_remainder_bound
        )

    @property
    def global_domain_certified(self) -> bool:
        return False

    @property
    def missing_global_induction_reason(self) -> str | None:
        if not self.geometric_tail_transfer_global_summability_certified:
            return (
                "the supplied structural envelopes, geometric transfer-growth bound, and "
                "retained-order acceleration do not imply a summable tail"
            )
        return (
            "geometric tail-transfer summability is conditional on proving the structural "
            "future-shell envelopes, transfer-growth ratio, and accelerated retained-order "
            "schedule for every later shell"
        )


@dataclass(frozen=True)
class CompactifiedSundmanAcceleratedGlobalSummabilityCertificate:
    """Conditional summability for an accelerated future retained-order schedule.

    Future shell transitions are bounded by

    ``growth * step_radius**(d0 + k*q) * transfer0 * gamma**k``

    where ``d0`` is the first future retained-order increment and ``q`` is the
    per-shell increase in that increment.  The proof obligation becomes two
    finite inequalities: the first unseen transition has ratio below one, and
    ``gamma * step_radius**q < 1`` makes all later ratios shrink geometrically.
    """

    chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate
    max_future_segment_growth_ratio: float
    max_future_state_sup_growth_ratio: float
    max_future_denominator_growth_ratio: float
    max_future_step_radius_ratio: float
    initial_retained_order_increment: int
    retained_order_increment_growth: int
    initial_tail_transfer_growth_factor_bound: float = 1.0
    max_future_tail_transfer_growth_ratio: float = 1.0
    envelope_source: str = "manual"

    @property
    def uniform_bounds_finite(self) -> bool:
        return bool(
            np.isfinite(self.max_future_segment_growth_ratio)
            and np.isfinite(self.max_future_state_sup_growth_ratio)
            and np.isfinite(self.max_future_denominator_growth_ratio)
            and np.isfinite(self.max_future_step_radius_ratio)
            and np.isfinite(self.initial_tail_transfer_growth_factor_bound)
            and np.isfinite(self.max_future_tail_transfer_growth_ratio)
            and self.max_future_segment_growth_ratio >= 0.0
            and self.max_future_state_sup_growth_ratio >= 0.0
            and self.max_future_denominator_growth_ratio >= 0.0
            and 0.0 <= self.max_future_step_radius_ratio < 1.0
            and self.initial_retained_order_increment > 0
            and self.retained_order_increment_growth >= 0
            and self.initial_tail_transfer_growth_factor_bound >= 0.0
            and self.max_future_tail_transfer_growth_ratio >= 0.0
        )

    @property
    def future_envelope_growth_factor_bound(self) -> float:
        if not self.uniform_bounds_finite:
            return float("inf")
        return float(
            self.max_future_segment_growth_ratio
            * self.max_future_state_sup_growth_ratio
            * self.max_future_denominator_growth_ratio
        )

    @property
    def projected_next_tail_transfer_growth_factor_bound(self) -> float:
        factors = self.chain.tail_transfer_growth_factors
        if not factors or not np.isfinite(self.max_future_tail_transfer_growth_ratio):
            return float("inf")
        return float(factors[-1] * self.max_future_tail_transfer_growth_ratio)

    @property
    def observed_prefix_within_accelerated_bounds(self) -> bool:
        if not self.uniform_bounds_finite or not self.chain.finite_cauchy_majorant_transition_candidate_certified:
            return False
        if not self.chain.arithmetic_retained_order_acceleration_certified:
            return False
        increments = self.chain.added_retained_order_increments
        if not increments:
            return False
        next_increment = increments[-1] + self.retained_order_increment_growth
        if (
            self.retained_order_increment_growth != self.chain.retained_order_increment_growth
            or self.initial_retained_order_increment < next_increment
        ):
            return False
        if self.chain.tail_transfer_growth_ratios:
            if self.chain.max_tail_transfer_growth_ratio > self.max_future_tail_transfer_growth_ratio:
                return False
            if self.projected_next_tail_transfer_growth_factor_bound > self.initial_tail_transfer_growth_factor_bound:
                return False
        return bool(
            self.chain.max_added_segment_growth_ratio <= self.max_future_segment_growth_ratio
            and self.chain.max_added_state_sup_growth_ratio <= self.max_future_state_sup_growth_ratio
            and self.chain.max_cauchy_tail_denominator_growth_ratio <= self.max_future_denominator_growth_ratio
            and self.chain.max_added_step_radius_ratio <= self.max_future_step_radius_ratio
        )

    @property
    def initial_structural_tail_decay_factor_bound(self) -> float:
        if not self.uniform_bounds_finite:
            return float("inf")
        return float(
            self.future_envelope_growth_factor_bound
            * self.max_future_step_radius_ratio**self.initial_retained_order_increment
        )

    @property
    def initial_tail_transfer_adjusted_tail_decay_factor_bound(self) -> float:
        if not self.uniform_bounds_finite:
            return float("inf")
        return float(
            self.initial_structural_tail_decay_factor_bound
            * self.initial_tail_transfer_growth_factor_bound
        )

    @property
    def tail_transfer_growth_ratio_absorption_factor_bound(self) -> float:
        if not self.uniform_bounds_finite:
            return float("inf")
        return float(
            self.max_future_tail_transfer_growth_ratio
            * self.max_future_step_radius_ratio**self.retained_order_increment_growth
        )

    @property
    def required_initial_retained_order_increment_for_summability(self) -> int:
        if not self.uniform_bounds_finite:
            return -1
        growth = self.future_envelope_growth_factor_bound * self.initial_tail_transfer_growth_factor_bound
        step_ratio = self.max_future_step_radius_ratio
        if not np.isfinite(growth) or growth < 0.0:
            return -1
        if growth == 0.0 or step_ratio == 0.0:
            return 1
        if not 0.0 < step_ratio < 1.0:
            return -1
        if growth * step_ratio < 1.0:
            return 1

        quotient = np.log(1.0 / growth) / np.log(step_ratio)
        if not np.isfinite(quotient):
            return -1
        increment = max(1, int(np.floor(quotient)) + 1)
        while growth * step_ratio**increment >= 1.0:
            increment += 1
        while increment > 1 and growth * step_ratio**(increment - 1) < 1.0:
            increment -= 1
        return increment

    @property
    def initial_retained_order_increment_surplus(self) -> int:
        required = self.required_initial_retained_order_increment_for_summability
        if required <= 0:
            return -1
        return int(self.initial_retained_order_increment - required)

    @property
    def initial_retained_order_increment_sufficient(self) -> bool:
        required = self.required_initial_retained_order_increment_for_summability
        return bool(required > 0 and self.initial_retained_order_increment >= required)

    @property
    def required_retained_order_increment_growth_for_tail_transfer_ratio(self) -> int:
        if not self.uniform_bounds_finite:
            return -1
        gamma = self.max_future_tail_transfer_growth_ratio
        step_ratio = self.max_future_step_radius_ratio
        if not np.isfinite(gamma) or gamma < 0.0:
            return -1
        if gamma < 1.0:
            return 0
        if not 0.0 < step_ratio < 1.0:
            return -1

        quotient = np.log(1.0 / gamma) / np.log(step_ratio)
        if not np.isfinite(quotient):
            return -1
        growth = max(0, int(np.floor(quotient)) + 1)
        while gamma * step_ratio**growth >= 1.0:
            growth += 1
        while growth > 0 and gamma * step_ratio ** (growth - 1) < 1.0:
            growth -= 1
        return growth

    @property
    def retained_order_increment_growth_surplus(self) -> int:
        required = self.required_retained_order_increment_growth_for_tail_transfer_ratio
        if required < 0:
            return -1
        return int(self.retained_order_increment_growth - required)

    @property
    def retained_order_increment_growth_sufficient(self) -> bool:
        required = self.required_retained_order_increment_growth_for_tail_transfer_ratio
        return bool(required >= 0 and self.retained_order_increment_growth >= required)

    @property
    def accelerated_tail_ratios_shrink(self) -> bool:
        factor = self.tail_transfer_growth_ratio_absorption_factor_bound
        return bool(np.isfinite(factor) and factor < 1.0)

    @property
    def accelerated_global_summability_certified(self) -> bool:
        initial_factor = self.initial_tail_transfer_adjusted_tail_decay_factor_bound
        return bool(
            self.observed_prefix_within_accelerated_bounds
            and np.isfinite(initial_factor)
            and initial_factor < 1.0
            and self.initial_retained_order_increment_sufficient
            and self.retained_order_increment_growth_sufficient
            and self.accelerated_tail_ratios_shrink
            and self.chain.added_incremental_tail_bounds
        )

    def future_tail_ratio_bounds(self, count: int) -> tuple[float, ...]:
        """Return the first ``count`` accelerated future tail-ratio bounds."""

        count = int(count)
        if count < 0:
            raise ValueError("count must be nonnegative")
        if not self.uniform_bounds_finite:
            return ()
        ratio = self.initial_tail_transfer_adjusted_tail_decay_factor_bound
        growth = self.tail_transfer_growth_ratio_absorption_factor_bound
        bounds: list[float] = []
        for _ in range(count):
            if not np.isfinite(ratio):
                return ()
            bounds.append(float(ratio))
            ratio *= growth
        return tuple(bounds)

    def future_tail_bounds(self, count: int) -> tuple[float, ...]:
        """Return the first ``count`` accelerated future incremental-tail bounds."""

        ratios = self.future_tail_ratio_bounds(count)
        tails = self.chain.added_incremental_tail_bounds
        if len(ratios) != int(count) or not tails:
            return ()
        current = float(tails[-1])
        bounds: list[float] = []
        for ratio in ratios:
            if not np.isfinite(ratio):
                return ()
            current *= ratio
            bounds.append(float(current))
        return tuple(bounds)

    def finite_future_tail_sum_bound(self, count: int) -> float:
        """Return the finite sum of the first ``count`` future tail bounds."""

        bounds = self.future_tail_bounds(count)
        return float(sum(bounds)) if len(bounds) == int(count) else float("inf")

    @property
    def accelerated_future_tail_remainder_bound(self) -> float:
        if not self.accelerated_global_summability_certified:
            return float("inf")
        base_tail = float(self.chain.added_incremental_tail_bounds[-1])
        first_ratio = self.initial_tail_transfer_adjusted_tail_decay_factor_bound
        ratio_growth = self.tail_transfer_growth_ratio_absorption_factor_bound
        if first_ratio == 0.0:
            return 0.0

        term = base_tail
        total = 0.0
        ratio = first_ratio
        for _ in range(10000):
            if not np.isfinite(term) or not np.isfinite(ratio):
                return float("inf")
            term *= ratio
            total += term
            next_ratio = ratio * ratio_growth
            if next_ratio == 0.0:
                return float(total)
            if 0.0 < next_ratio < 0.5:
                return float(total + term * next_ratio / (1.0 - next_ratio))
            if abs(term) < np.finfo(float).tiny:
                return float(total)
            ratio = next_ratio
        return float("inf")

    @property
    def accelerated_total_incremental_tail_bound(self) -> float:
        if not self.accelerated_global_summability_certified:
            return float("inf")
        return float(self.chain.incremental_total_tail_bound + self.accelerated_future_tail_remainder_bound)

    @property
    def required_induction_witness_bounds(self) -> dict[str, float | int]:
        """Return the quantitative all-future bounds a global witness must prove."""

        return {
            "certified_max_future_segment_growth_ratio": self.max_future_segment_growth_ratio,
            "certified_max_future_state_sup_growth_ratio": self.max_future_state_sup_growth_ratio,
            "certified_max_future_denominator_growth_ratio": self.max_future_denominator_growth_ratio,
            "certified_max_future_step_radius_ratio": self.max_future_step_radius_ratio,
            "certified_initial_tail_transfer_growth_factor_bound": (
                self.initial_tail_transfer_growth_factor_bound
            ),
            "certified_max_future_tail_transfer_growth_ratio": self.max_future_tail_transfer_growth_ratio,
            "certified_initial_retained_order_increment": self.initial_retained_order_increment,
            "certified_retained_order_increment_growth": self.retained_order_increment_growth,
        }

    @property
    def required_domain_schedule_witness(self) -> dict[str, float | int]:
        """Return the geometric schedule data a global domain witness must prove."""

        schedule = self.chain.final_schedule
        return {
            "certified_schedule_initial_boundary_margin": schedule.initial_boundary_margin,
            "certified_schedule_contraction": schedule.contraction,
            "certified_extension_induction_start_index": schedule.prefix_length,
        }

    @property
    def global_domain_certified(self) -> bool:
        return False

    @property
    def missing_global_induction_reason(self) -> str | None:
        if not self.accelerated_global_summability_certified:
            return (
                "the supplied structural envelopes, transfer-growth ratio, and accelerated retained-order "
                "schedule do not imply a summable tail"
            )
        if self.envelope_source == "finite_prefix":
            return (
                "accelerated future-shell summability follows from the finite-prefix-derived envelopes only if "
                "an induction proof shows that those envelopes and the accelerated retained-order schedule "
                "persist for every later shell"
            )
        return (
            "accelerated future-shell summability is conditional on proving the supplied structural envelopes, "
            "transfer-growth ratio, and retained-order acceleration for every later shell"
        )


@dataclass(frozen=True)
class CompactifiedSundmanProofObligationDetail:
    """Actionable local evidence for one global proof obligation."""

    obligation: str
    certified: bool
    required: float | int | str | None = None
    observed: float | int | str | None = None
    comparison: str | None = None
    witness_field: str | None = None


@dataclass(frozen=True)
class CompactifiedSundmanGeometricScheduleRecurrenceCertificate:
    """Typed recurrence witness for the geometric compact-domain schedule.

    A finite prefix only checks the currently constructed compact-Sundman
    domains.  This certificate records the extra induction data needed to turn
    that prefix into the all-future geometric exhaustion used by the global
    compact-Sundman proof ledger.
    """

    schedule: CompactifiedSundmanGeometricExhaustionScheduleCertificate
    one_step_extension_template_certified: bool = False
    witness_source: str = "manual_geometric_schedule_recurrence"
    certified_schedule_initial_boundary_margin: float | None = None
    certified_schedule_contraction: float | None = None
    certified_extension_induction_start_index: float | int | None = None

    @property
    def schedule_witness_present(self) -> bool:
        return bool(
            self.certified_schedule_initial_boundary_margin is not None
            and self.certified_schedule_contraction is not None
            and self.certified_extension_induction_start_index is not None
        )

    @property
    def schedule_witness_finite(self) -> bool:
        if not self.schedule_witness_present:
            return False
        return bool(
            np.isfinite(self.certified_schedule_initial_boundary_margin)
            and np.isfinite(self.certified_schedule_contraction)
            and np.isfinite(self.certified_extension_induction_start_index)
            and 0.0 < self.certified_schedule_initial_boundary_margin < 1.0
            and 0.0 < self.certified_schedule_contraction < 1.0
            and int(self.certified_extension_induction_start_index)
            == self.certified_extension_induction_start_index
            and self.certified_extension_induction_start_index >= 0
        )

    @property
    def initial_boundary_margin_matches_checked_prefix(self) -> bool:
        if not self.schedule_witness_finite:
            return False
        tolerance = _compact_parameter_tolerance(
            self.certified_schedule_initial_boundary_margin,
            self.schedule.initial_boundary_margin,
        )
        return bool(
            abs(
                self.certified_schedule_initial_boundary_margin
                - self.schedule.initial_boundary_margin
            )
            <= tolerance
        )

    @property
    def contraction_matches_checked_prefix(self) -> bool:
        if not self.schedule_witness_finite:
            return False
        tolerance = _compact_parameter_tolerance(
            self.certified_schedule_contraction,
            self.schedule.contraction,
        )
        return bool(
            abs(self.certified_schedule_contraction - self.schedule.contraction)
            <= tolerance
        )

    @property
    def induction_start_index_matches_checked_prefix(self) -> bool:
        if not self.schedule_witness_finite:
            return False
        return bool(
            self.certified_extension_induction_start_index <= self.schedule.prefix_length
        )

    @property
    def endpoint_exhaustion_certified(self) -> bool:
        return bool(self.schedule.schedule_defined and self.schedule.endpoint_limit_certified)

    @property
    def schedule_witness_matches_checked_prefix(self) -> bool:
        return bool(
            self.schedule.certified
            and self.endpoint_exhaustion_certified
            and self.initial_boundary_margin_matches_checked_prefix
            and self.contraction_matches_checked_prefix
            and self.induction_start_index_matches_checked_prefix
        )

    @property
    def domain_exhaustion_recurrence_certified(self) -> bool:
        return self.schedule_witness_matches_checked_prefix

    @property
    def scheduled_extension_recurrence_certified(self) -> bool:
        return bool(
            self.domain_exhaustion_recurrence_certified
            and self.one_step_extension_template_certified
        )

    @property
    def schedule_recurrence_certified(self) -> bool:
        return all(detail.certified for detail in self.schedule_recurrence_details)

    @property
    def required_domain_schedule_witness(self) -> dict[str, float | int]:
        return {
            "certified_schedule_initial_boundary_margin": (
                self.schedule.initial_boundary_margin
            ),
            "certified_schedule_contraction": self.schedule.contraction,
            "certified_extension_induction_start_index": self.schedule.prefix_length,
        }

    @property
    def certified_domain_schedule_witness(self) -> dict[str, float | int | None]:
        return {
            "certified_schedule_initial_boundary_margin": (
                self.certified_schedule_initial_boundary_margin
            ),
            "certified_schedule_contraction": self.certified_schedule_contraction,
            "certified_extension_induction_start_index": (
                self.certified_extension_induction_start_index
            ),
        }

    @property
    def recurrence_closure_kwargs(self) -> dict[str, bool | float | int | None]:
        return {
            "domain_exhaustion_recurrence_certified": (
                self.domain_exhaustion_recurrence_certified
            ),
            "scheduled_extension_recurrence_certified": (
                self.scheduled_extension_recurrence_certified
            ),
            **self.certified_domain_schedule_witness,
        }

    @property
    def schedule_recurrence_obligation_statuses(self) -> tuple[tuple[str, bool], ...]:
        return tuple(
            (detail.obligation, detail.certified)
            for detail in self.schedule_recurrence_details
        )

    @property
    def schedule_recurrence_details(
        self,
    ) -> tuple[CompactifiedSundmanProofObligationDetail, ...]:
        missing_fields = tuple(
            field
            for field in (
                "certified_schedule_initial_boundary_margin",
                "certified_schedule_contraction",
                "certified_extension_induction_start_index",
            )
            if getattr(self, field) is None
        )
        witness_presence = (
            "present" if not missing_fields else "missing: " + ", ".join(missing_fields)
        )
        return (
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_schedule_prefix_certified",
                certified=self.schedule.certified,
                required="finite prefix satisfies the geometric compact-domain schedule",
                observed="certified" if self.schedule.certified else "not certified",
                comparison="==",
                witness_field="schedule",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_schedule_endpoint_limit",
                certified=self.endpoint_exhaustion_certified,
                required="geometric boundary margins converge to zero at both compact endpoints",
                observed=(
                    "certified"
                    if self.endpoint_exhaustion_certified
                    else "not certified"
                ),
                comparison="==",
                witness_field="schedule.endpoint_limit_certified",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_schedule_witness_present",
                certified=self.schedule_witness_present,
                required="all geometric schedule witness fields supplied",
                observed=witness_presence,
                comparison="==",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_schedule_witness_finite",
                certified=self.schedule_witness_finite,
                required="finite margin/contraction in (0, 1) and nonnegative integer start index",
                observed="finite" if self.schedule_witness_finite else "not finite",
                comparison="==",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_schedule_initial_boundary_margin",
                certified=self.initial_boundary_margin_matches_checked_prefix,
                required=float(self.schedule.initial_boundary_margin),
                observed=self.certified_schedule_initial_boundary_margin,
                comparison="approximately equal",
                witness_field="certified_schedule_initial_boundary_margin",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_schedule_contraction",
                certified=self.contraction_matches_checked_prefix,
                required=float(self.schedule.contraction),
                observed=self.certified_schedule_contraction,
                comparison="approximately equal",
                witness_field="certified_schedule_contraction",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_schedule_induction_start_index",
                certified=self.induction_start_index_matches_checked_prefix,
                required=int(self.schedule.prefix_length),
                observed=self.certified_extension_induction_start_index,
                comparison="<=",
                witness_field="certified_extension_induction_start_index",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_schedule_one_step_extension_template",
                certified=self.one_step_extension_template_certified,
                required="template proves the scheduled compact-domain extension step",
                observed=(
                    "certified"
                    if self.one_step_extension_template_certified
                    else "not certified"
                ),
                comparison="==",
                witness_field="one_step_extension_template_certified",
            ),
        )

    @property
    def missing_schedule_recurrence_obligations(self) -> tuple[str, ...]:
        return tuple(
            detail.obligation
            for detail in self.schedule_recurrence_details
            if not detail.certified
        )


@dataclass(frozen=True)
class CompactifiedSundmanRecurrenceTemplateCertificate:
    """Typed evidence for the nonnumeric recurrence templates.

    The scalar recurrence certificate checks arithmetic inequalities.  This
    object records the separate analytic-template obligations that make those
    inequalities meaningful for every future shell: structural envelope form,
    tail-transfer form, and retained-order schedule form.
    """

    accelerated_certificate: CompactifiedSundmanAcceleratedGlobalSummabilityCertificate
    structural_envelope_form_certified: bool = False
    structural_transition_template_certified: bool = False
    tail_transfer_form_certified: bool = False
    tail_transfer_growth_template_certified: bool = False
    retained_order_arithmetic_schedule_certified: bool = False
    retained_order_growth_template_certified: bool = False
    witness_source: str = "manual_recurrence_template"

    @property
    def accelerated_summability_target_certified(self) -> bool:
        return self.accelerated_certificate.accelerated_global_summability_certified

    @property
    def structural_envelope_template_certified(self) -> bool:
        return bool(
            self.accelerated_summability_target_certified
            and self.structural_envelope_form_certified
            and self.structural_transition_template_certified
        )

    @property
    def tail_transfer_template_certified(self) -> bool:
        return bool(
            self.accelerated_summability_target_certified
            and self.tail_transfer_form_certified
            and self.tail_transfer_growth_template_certified
        )

    @property
    def retained_order_schedule_template_certified(self) -> bool:
        return bool(
            self.accelerated_summability_target_certified
            and self.retained_order_arithmetic_schedule_certified
            and self.retained_order_growth_template_certified
        )

    @property
    def recurrence_template_certified(self) -> bool:
        return all(detail.certified for detail in self.recurrence_template_details)

    @property
    def recurrence_closure_kwargs(self) -> dict[str, bool]:
        return {
            "structural_envelope_recurrence_certified": (
                self.structural_envelope_template_certified
            ),
            "tail_transfer_recurrence_certified": self.tail_transfer_template_certified,
            "retained_order_schedule_recurrence_certified": (
                self.retained_order_schedule_template_certified
            ),
        }

    @property
    def scalar_recurrence_kwargs(self) -> dict[str, bool]:
        return {
            "structural_envelope_template_certified": (
                self.structural_envelope_template_certified
            ),
            "tail_transfer_template_certified": self.tail_transfer_template_certified,
            "retained_order_schedule_template_certified": (
                self.retained_order_schedule_template_certified
            ),
        }

    @property
    def recurrence_template_obligation_statuses(self) -> tuple[tuple[str, bool], ...]:
        return tuple(
            (detail.obligation, detail.certified)
            for detail in self.recurrence_template_details
        )

    @property
    def recurrence_template_details(
        self,
    ) -> tuple[CompactifiedSundmanProofObligationDetail, ...]:
        return (
            self._flag_detail(
                "accelerated_summability_target",
                self.accelerated_summability_target_certified,
                "accelerated finite-prefix summability target certified",
                "accelerated_certificate.accelerated_global_summability_certified",
            ),
            self._flag_detail(
                "structural_envelope_form",
                self.structural_envelope_form_certified,
                "template preserves the future structural envelope form",
                "structural_envelope_form_certified",
            ),
            self._flag_detail(
                "structural_transition_template",
                self.structural_transition_template_certified,
                "template converts structural envelopes into segment/state/denominator/step bounds",
                "structural_transition_template_certified",
            ),
            self._flag_detail(
                "tail_transfer_form",
                self.tail_transfer_form_certified,
                "template preserves the tail-transfer multiplier form",
                "tail_transfer_form_certified",
            ),
            self._flag_detail(
                "tail_transfer_growth_template",
                self.tail_transfer_growth_template_certified,
                "template converts tail-transfer growth into initial/growth-ratio bounds",
                "tail_transfer_growth_template_certified",
            ),
            self._flag_detail(
                "retained_order_arithmetic_schedule",
                self.retained_order_arithmetic_schedule_certified,
                "template preserves an arithmetic retained-order schedule",
                "retained_order_arithmetic_schedule_certified",
            ),
            self._flag_detail(
                "retained_order_growth_template",
                self.retained_order_growth_template_certified,
                "template converts the retained-order schedule into initial/growth bounds",
                "retained_order_growth_template_certified",
            ),
        )

    @property
    def missing_recurrence_template_obligations(self) -> tuple[str, ...]:
        return tuple(
            detail.obligation
            for detail in self.recurrence_template_details
            if not detail.certified
        )

    def _flag_detail(
        self,
        obligation: str,
        certified: bool,
        required: str,
        witness_field: str,
    ) -> CompactifiedSundmanProofObligationDetail:
        return CompactifiedSundmanProofObligationDetail(
            obligation=obligation,
            certified=bool(certified),
            required=required,
            observed="certified" if certified else "not certified",
            comparison="==",
            witness_field=witness_field,
        )


@dataclass(frozen=True)
class CompactifiedSundmanAcceleratedInductionWitnessCertificate:
    """Abstract endpoint for upgrading accelerated summability to a global certificate.

    This object does not prove the missing induction theorem by itself. It
    records the exact universal proof obligations needed after finite-prefix
    accelerated summability has been established.  A witness must supply
    quantitative all-future bounds strong enough to imply the accelerated
    envelope used by the summability certificate; boolean proof flags alone are
    not treated as a global theorem.
    """

    accelerated_certificate: CompactifiedSundmanAcceleratedGlobalSummabilityCertificate
    future_domain_exhaustion_certified: bool = False
    future_extension_induction_certified: bool = False
    future_structural_envelopes_certified: bool = False
    future_tail_transfer_growth_certified: bool = False
    future_accelerated_order_schedule_certified: bool = False
    future_segment_growth_bound_certified: bool = False
    future_state_sup_growth_bound_certified: bool = False
    future_denominator_growth_bound_certified: bool = False
    future_step_radius_bound_certified: bool = False
    future_initial_tail_transfer_bound_certified: bool = False
    future_tail_transfer_growth_ratio_certified: bool = False
    future_initial_retained_order_increment_certified: bool = False
    future_retained_order_increment_growth_certified: bool = False
    collision_continuation_certified: bool = False
    binary_collision_continuation_certified: bool = False
    triple_collision_continuation_certified: bool = False
    witness_source: str = "manual"
    certified_schedule_initial_boundary_margin: float | None = None
    certified_schedule_contraction: float | None = None
    certified_extension_induction_start_index: int | None = None
    certified_max_future_segment_growth_ratio: float | None = None
    certified_max_future_state_sup_growth_ratio: float | None = None
    certified_max_future_denominator_growth_ratio: float | None = None
    certified_max_future_step_radius_ratio: float | None = None
    certified_initial_tail_transfer_growth_factor_bound: float | None = None
    certified_max_future_tail_transfer_growth_ratio: float | None = None
    certified_initial_retained_order_increment: int | None = None
    certified_retained_order_increment_growth: int | None = None

    @property
    def quantitative_witness_present(self) -> bool:
        return bool(
            self.certified_max_future_segment_growth_ratio is not None
            and self.certified_max_future_state_sup_growth_ratio is not None
            and self.certified_max_future_denominator_growth_ratio is not None
            and self.certified_max_future_step_radius_ratio is not None
            and self.certified_initial_tail_transfer_growth_factor_bound is not None
            and self.certified_max_future_tail_transfer_growth_ratio is not None
            and self.certified_initial_retained_order_increment is not None
            and self.certified_retained_order_increment_growth is not None
        )

    @property
    def quantitative_witness_finite(self) -> bool:
        if not self.quantitative_witness_present:
            return False
        return bool(
            np.isfinite(self.certified_max_future_segment_growth_ratio)
            and np.isfinite(self.certified_max_future_state_sup_growth_ratio)
            and np.isfinite(self.certified_max_future_denominator_growth_ratio)
            and np.isfinite(self.certified_max_future_step_radius_ratio)
            and np.isfinite(self.certified_initial_tail_transfer_growth_factor_bound)
            and np.isfinite(self.certified_max_future_tail_transfer_growth_ratio)
            and np.isfinite(self.certified_initial_retained_order_increment)
            and np.isfinite(self.certified_retained_order_increment_growth)
            and self.certified_max_future_segment_growth_ratio >= 0.0
            and self.certified_max_future_state_sup_growth_ratio >= 0.0
            and self.certified_max_future_denominator_growth_ratio >= 0.0
            and 0.0 <= self.certified_max_future_step_radius_ratio < 1.0
            and self.certified_initial_tail_transfer_growth_factor_bound >= 0.0
            and self.certified_max_future_tail_transfer_growth_ratio >= 0.0
            and int(self.certified_initial_retained_order_increment)
            == self.certified_initial_retained_order_increment
            and self.certified_initial_retained_order_increment > 0
            and int(self.certified_retained_order_increment_growth)
            == self.certified_retained_order_increment_growth
            and self.certified_retained_order_increment_growth >= 0
        )

    @property
    def domain_schedule_witness_present(self) -> bool:
        return bool(
            self.certified_schedule_initial_boundary_margin is not None
            and self.certified_schedule_contraction is not None
            and self.certified_extension_induction_start_index is not None
        )

    @property
    def domain_schedule_witness_finite(self) -> bool:
        if not self.domain_schedule_witness_present:
            return False
        return bool(
            np.isfinite(self.certified_schedule_initial_boundary_margin)
            and np.isfinite(self.certified_schedule_contraction)
            and np.isfinite(self.certified_extension_induction_start_index)
            and 0.0 < self.certified_schedule_initial_boundary_margin < 1.0
            and 0.0 < self.certified_schedule_contraction < 1.0
            and int(self.certified_extension_induction_start_index)
            == self.certified_extension_induction_start_index
            and self.certified_extension_induction_start_index >= 0
        )

    @property
    def domain_schedule_witness_matches_accelerated_chain(self) -> bool:
        if not self.domain_schedule_witness_finite:
            return False
        schedule = self.accelerated_certificate.chain.final_schedule
        if not schedule.certified or not schedule.endpoint_limit_certified:
            return False
        initial_tolerance = _compact_parameter_tolerance(
            self.certified_schedule_initial_boundary_margin,
            schedule.initial_boundary_margin,
        )
        contraction_tolerance = _compact_parameter_tolerance(
            self.certified_schedule_contraction,
            schedule.contraction,
        )
        return bool(
            abs(self.certified_schedule_initial_boundary_margin - schedule.initial_boundary_margin)
            <= initial_tolerance
            and abs(self.certified_schedule_contraction - schedule.contraction)
            <= contraction_tolerance
            and self.certified_extension_induction_start_index <= schedule.prefix_length
        )

    @property
    def structural_envelope_subproofs_certified(self) -> bool:
        return bool(
            self.future_structural_envelopes_certified
            and self.future_segment_growth_bound_certified
            and self.future_state_sup_growth_bound_certified
            and self.future_denominator_growth_bound_certified
            and self.future_step_radius_bound_certified
        )

    @property
    def tail_transfer_subproofs_certified(self) -> bool:
        return bool(
            self.future_tail_transfer_growth_certified
            and self.future_initial_tail_transfer_bound_certified
            and self.future_tail_transfer_growth_ratio_certified
        )

    @property
    def accelerated_order_schedule_subproofs_certified(self) -> bool:
        return bool(
            self.future_accelerated_order_schedule_certified
            and self.future_initial_retained_order_increment_certified
            and self.future_retained_order_increment_growth_certified
        )

    @property
    def quantitative_witness_dominates_accelerated_bounds(self) -> bool:
        if not self.quantitative_witness_finite or not self.accelerated_certificate.uniform_bounds_finite:
            return False
        accelerated = self.accelerated_certificate
        return bool(
            self.certified_max_future_segment_growth_ratio <= accelerated.max_future_segment_growth_ratio
            and self.certified_max_future_state_sup_growth_ratio
            <= accelerated.max_future_state_sup_growth_ratio
            and self.certified_max_future_denominator_growth_ratio
            <= accelerated.max_future_denominator_growth_ratio
            and self.certified_max_future_step_radius_ratio <= accelerated.max_future_step_radius_ratio
            and self.certified_initial_tail_transfer_growth_factor_bound
            <= accelerated.initial_tail_transfer_growth_factor_bound
            and self.certified_max_future_tail_transfer_growth_ratio
            <= accelerated.max_future_tail_transfer_growth_ratio
            and self.certified_initial_retained_order_increment
            >= accelerated.initial_retained_order_increment
            and self.certified_retained_order_increment_growth
            >= accelerated.retained_order_increment_growth
        )

    @property
    def domain_schedule_witness_details(
        self,
    ) -> tuple[CompactifiedSundmanProofObligationDetail, ...]:
        """Compare the supplied geometric schedule witness with the checked prefix."""

        schedule = self.accelerated_certificate.chain.final_schedule
        boundary_margin_certified = False
        contraction_certified = False
        start_index_certified = False
        if self.domain_schedule_witness_finite:
            boundary_margin_tolerance = _compact_parameter_tolerance(
                self.certified_schedule_initial_boundary_margin,
                schedule.initial_boundary_margin,
            )
            contraction_tolerance = _compact_parameter_tolerance(
                self.certified_schedule_contraction,
                schedule.contraction,
            )
            boundary_margin_certified = bool(
                abs(self.certified_schedule_initial_boundary_margin - schedule.initial_boundary_margin)
                <= boundary_margin_tolerance
            )
            contraction_certified = bool(
                abs(self.certified_schedule_contraction - schedule.contraction)
                <= contraction_tolerance
            )
            start_index_certified = bool(
                self.certified_extension_induction_start_index <= schedule.prefix_length
            )
        return (
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_domain_schedule_initial_boundary_margin",
                certified=boundary_margin_certified,
                required=float(schedule.initial_boundary_margin),
                observed=self.certified_schedule_initial_boundary_margin,
                comparison="approximately equal",
                witness_field="certified_schedule_initial_boundary_margin",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_domain_schedule_contraction",
                certified=contraction_certified,
                required=float(schedule.contraction),
                observed=self.certified_schedule_contraction,
                comparison="approximately equal",
                witness_field="certified_schedule_contraction",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_domain_extension_induction_start_index",
                certified=start_index_certified,
                required=int(schedule.prefix_length),
                observed=self.certified_extension_induction_start_index,
                comparison="<=",
                witness_field="certified_extension_induction_start_index",
            ),
        )

    @property
    def quantitative_witness_bound_details(
        self,
    ) -> tuple[CompactifiedSundmanProofObligationDetail, ...]:
        """Compare each supplied quantitative witness bound with the accelerated target."""

        accelerated = self.accelerated_certificate

        def finite_leq(observed: float | int | None, required: float | int) -> bool:
            return bool(observed is not None and np.isfinite(observed) and observed <= required)

        def finite_geq_integer(observed: float | int | None, required: int) -> bool:
            return bool(
                observed is not None
                and np.isfinite(observed)
                and int(observed) == observed
                and observed >= required
            )

        return (
            CompactifiedSundmanProofObligationDetail(
                obligation="future_segment_count_growth_bound_value",
                certified=finite_leq(
                    self.certified_max_future_segment_growth_ratio,
                    accelerated.max_future_segment_growth_ratio,
                ),
                required=float(accelerated.max_future_segment_growth_ratio),
                observed=self.certified_max_future_segment_growth_ratio,
                comparison="<=",
                witness_field="certified_max_future_segment_growth_ratio",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="future_state_supremum_growth_bound_value",
                certified=finite_leq(
                    self.certified_max_future_state_sup_growth_ratio,
                    accelerated.max_future_state_sup_growth_ratio,
                ),
                required=float(accelerated.max_future_state_sup_growth_ratio),
                observed=self.certified_max_future_state_sup_growth_ratio,
                comparison="<=",
                witness_field="certified_max_future_state_sup_growth_ratio",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="future_cauchy_denominator_growth_bound_value",
                certified=finite_leq(
                    self.certified_max_future_denominator_growth_ratio,
                    accelerated.max_future_denominator_growth_ratio,
                ),
                required=float(accelerated.max_future_denominator_growth_ratio),
                observed=self.certified_max_future_denominator_growth_ratio,
                comparison="<=",
                witness_field="certified_max_future_denominator_growth_ratio",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="future_step_radius_ratio_bound_value",
                certified=finite_leq(
                    self.certified_max_future_step_radius_ratio,
                    accelerated.max_future_step_radius_ratio,
                ),
                required=float(accelerated.max_future_step_radius_ratio),
                observed=self.certified_max_future_step_radius_ratio,
                comparison="<=",
                witness_field="certified_max_future_step_radius_ratio",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="first_future_tail_transfer_bound_value",
                certified=finite_leq(
                    self.certified_initial_tail_transfer_growth_factor_bound,
                    accelerated.initial_tail_transfer_growth_factor_bound,
                ),
                required=float(accelerated.initial_tail_transfer_growth_factor_bound),
                observed=self.certified_initial_tail_transfer_growth_factor_bound,
                comparison="<=",
                witness_field="certified_initial_tail_transfer_growth_factor_bound",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="future_tail_transfer_growth_ratio_value",
                certified=finite_leq(
                    self.certified_max_future_tail_transfer_growth_ratio,
                    accelerated.max_future_tail_transfer_growth_ratio,
                ),
                required=float(accelerated.max_future_tail_transfer_growth_ratio),
                observed=self.certified_max_future_tail_transfer_growth_ratio,
                comparison="<=",
                witness_field="certified_max_future_tail_transfer_growth_ratio",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="first_future_retained_order_increment_value",
                certified=finite_geq_integer(
                    self.certified_initial_retained_order_increment,
                    accelerated.initial_retained_order_increment,
                ),
                required=int(accelerated.initial_retained_order_increment),
                observed=self.certified_initial_retained_order_increment,
                comparison=">=",
                witness_field="certified_initial_retained_order_increment",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="future_retained_order_increment_growth_value",
                certified=finite_geq_integer(
                    self.certified_retained_order_increment_growth,
                    accelerated.retained_order_increment_growth,
                ),
                required=int(accelerated.retained_order_increment_growth),
                observed=self.certified_retained_order_increment_growth,
                comparison=">=",
                witness_field="certified_retained_order_increment_growth",
            ),
        )

    @property
    def future_shell_induction_certified(self) -> bool:
        return bool(
            self.future_domain_exhaustion_certified
            and self.future_extension_induction_certified
            and self.domain_schedule_witness_matches_accelerated_chain
            and self.structural_envelope_subproofs_certified
            and self.tail_transfer_subproofs_certified
            and self.accelerated_order_schedule_subproofs_certified
            and self.quantitative_witness_dominates_accelerated_bounds
        )

    @property
    def global_domain_certified(self) -> bool:
        return bool(
            self.accelerated_certificate.accelerated_global_summability_certified
            and self.future_shell_induction_certified
        )

    @property
    def global_tail_bound(self) -> float:
        if not self.global_domain_certified:
            return float("inf")
        return self.accelerated_certificate.accelerated_total_incremental_tail_bound

    @property
    def collision_continuation_obligations_certified(self) -> bool:
        return bool(
            self.collision_continuation_certified
            and self.binary_collision_continuation_certified
            and self.triple_collision_continuation_certified
        )

    @property
    def global_series_certified(self) -> bool:
        return bool(self.global_domain_certified and self.collision_continuation_obligations_certified)

    @property
    def global_proof_obligation_statuses(self) -> tuple[tuple[str, bool], ...]:
        """Machine-readable ledger of every proof obligation at the global endpoint."""

        return (
            (
                "accelerated_finite_prefix_summability",
                self.accelerated_certificate.accelerated_global_summability_certified,
            ),
            ("future_domain_exhaustion", self.future_domain_exhaustion_certified),
            ("future_scheduled_extension_induction", self.future_extension_induction_certified),
            ("geometric_domain_schedule_witness_present", self.domain_schedule_witness_present),
            ("geometric_domain_schedule_witness_finite", self.domain_schedule_witness_finite),
            (
                "geometric_domain_schedule_matches_checked_prefix",
                self.domain_schedule_witness_matches_accelerated_chain,
            ),
            ("future_structural_envelopes", self.future_structural_envelopes_certified),
            ("future_segment_count_growth_bound", self.future_segment_growth_bound_certified),
            ("future_state_supremum_growth_bound", self.future_state_sup_growth_bound_certified),
            ("future_cauchy_denominator_growth_bound", self.future_denominator_growth_bound_certified),
            ("future_step_radius_ratio_bound", self.future_step_radius_bound_certified),
            ("future_tail_transfer_growth", self.future_tail_transfer_growth_certified),
            ("first_future_tail_transfer_bound", self.future_initial_tail_transfer_bound_certified),
            ("future_tail_transfer_growth_ratio", self.future_tail_transfer_growth_ratio_certified),
            ("future_accelerated_retained_order_schedule", self.future_accelerated_order_schedule_certified),
            ("first_future_retained_order_increment", self.future_initial_retained_order_increment_certified),
            ("future_retained_order_increment_growth", self.future_retained_order_increment_growth_certified),
            ("quantitative_witness_present", self.quantitative_witness_present),
            ("quantitative_witness_finite", self.quantitative_witness_finite),
            (
                "quantitative_witness_dominates_accelerated_bounds",
                self.quantitative_witness_dominates_accelerated_bounds,
            ),
            ("collision_continuation", self.collision_continuation_certified),
            ("binary_collision_continuation", self.binary_collision_continuation_certified),
            ("triple_collision_continuation", self.triple_collision_continuation_certified),
        )

    @property
    def global_proof_obligation_details(
        self,
    ) -> tuple[CompactifiedSundmanProofObligationDetail, ...]:
        """Return obligation statuses with the local evidence needed to discharge failures."""

        statuses = dict(self.global_proof_obligation_statuses)
        missing_schedule_fields = tuple(
            field
            for field in (
                "certified_schedule_initial_boundary_margin",
                "certified_schedule_contraction",
                "certified_extension_induction_start_index",
            )
            if getattr(self, field) is None
        )
        missing_quantitative_fields = tuple(
            field
            for field in (
                "certified_max_future_segment_growth_ratio",
                "certified_max_future_state_sup_growth_ratio",
                "certified_max_future_denominator_growth_ratio",
                "certified_max_future_step_radius_ratio",
                "certified_initial_tail_transfer_growth_factor_bound",
                "certified_max_future_tail_transfer_growth_ratio",
                "certified_initial_retained_order_increment",
                "certified_retained_order_increment_growth",
            )
            if getattr(self, field) is None
        )
        first_failed_schedule_detail = next(
            (detail for detail in self.domain_schedule_witness_details if not detail.certified),
            None,
        )
        first_failed_quantitative_detail = next(
            (detail for detail in self.quantitative_witness_bound_details if not detail.certified),
            None,
        )

        def proof_flag(
            obligation: str,
            *,
            required: str = "proof flag certified",
            witness_field: str | None = None,
        ) -> CompactifiedSundmanProofObligationDetail:
            return CompactifiedSundmanProofObligationDetail(
                obligation=obligation,
                certified=statuses[obligation],
                required=required,
                observed="certified" if statuses[obligation] else "not certified",
                comparison="==",
                witness_field=witness_field,
            )

        schedule_presence = (
            "present" if not missing_schedule_fields else "missing: " + ", ".join(missing_schedule_fields)
        )
        quantitative_presence = (
            "present"
            if not missing_quantitative_fields
            else "missing: " + ", ".join(missing_quantitative_fields)
        )
        schedule_match_required: float | int | str | None = "all supplied schedule fields match"
        schedule_match_observed: float | int | str | None = (
            "all supplied schedule fields match"
            if self.domain_schedule_witness_matches_accelerated_chain
            else "no finite schedule witness"
        )
        schedule_match_comparison: str | None = None
        schedule_match_field: str | None = None
        if first_failed_schedule_detail is not None:
            schedule_match_required = first_failed_schedule_detail.required
            schedule_match_observed = first_failed_schedule_detail.observed
            schedule_match_comparison = first_failed_schedule_detail.comparison
            schedule_match_field = first_failed_schedule_detail.witness_field

        quantitative_domination_required: float | int | str | None = (
            "all quantitative witness comparisons dominate accelerated bounds"
        )
        quantitative_domination_observed: float | int | str | None = (
            "all quantitative witness comparisons dominate accelerated bounds"
            if self.quantitative_witness_dominates_accelerated_bounds
            else "no finite quantitative witness"
        )
        quantitative_domination_comparison: str | None = None
        quantitative_domination_field: str | None = None
        if first_failed_quantitative_detail is not None:
            quantitative_domination_required = first_failed_quantitative_detail.required
            quantitative_domination_observed = first_failed_quantitative_detail.observed
            quantitative_domination_comparison = first_failed_quantitative_detail.comparison
            quantitative_domination_field = first_failed_quantitative_detail.witness_field

        return (
            CompactifiedSundmanProofObligationDetail(
                obligation="accelerated_finite_prefix_summability",
                certified=statuses["accelerated_finite_prefix_summability"],
                required="accelerated global summability certificate",
                observed=(
                    "certified"
                    if statuses["accelerated_finite_prefix_summability"]
                    else "not certified"
                ),
                comparison="==",
                witness_field="accelerated_certificate",
            ),
            proof_flag(
                "future_domain_exhaustion",
                required="universal proof covers every later compact-Sundman domain",
                witness_field="future_domain_exhaustion_certified",
            ),
            proof_flag(
                "future_scheduled_extension_induction",
                required="universal proof constructs every later scheduled extension",
                witness_field="future_extension_induction_certified",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_domain_schedule_witness_present",
                certified=statuses["geometric_domain_schedule_witness_present"],
                required="all geometric schedule witness fields supplied",
                observed=schedule_presence,
                comparison="==",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_domain_schedule_witness_finite",
                certified=statuses["geometric_domain_schedule_witness_finite"],
                required="finite margin/contraction in (0, 1) and nonnegative integer start index",
                observed="finite" if self.domain_schedule_witness_finite else "not finite",
                comparison="==",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="geometric_domain_schedule_matches_checked_prefix",
                certified=statuses["geometric_domain_schedule_matches_checked_prefix"],
                required=schedule_match_required,
                observed=schedule_match_observed,
                comparison=schedule_match_comparison,
                witness_field=schedule_match_field,
            ),
            proof_flag(
                "future_structural_envelopes",
                required="universal proof preserves structural future-shell envelopes",
                witness_field="future_structural_envelopes_certified",
            ),
            proof_flag(
                "future_segment_count_growth_bound",
                required="universal proof certifies segment-count growth bound",
                witness_field="future_segment_growth_bound_certified",
            ),
            proof_flag(
                "future_state_supremum_growth_bound",
                required="universal proof certifies state-supremum growth bound",
                witness_field="future_state_sup_growth_bound_certified",
            ),
            proof_flag(
                "future_cauchy_denominator_growth_bound",
                required="universal proof certifies Cauchy-denominator growth bound",
                witness_field="future_denominator_growth_bound_certified",
            ),
            proof_flag(
                "future_step_radius_ratio_bound",
                required="universal proof certifies step-radius ratio bound",
                witness_field="future_step_radius_bound_certified",
            ),
            proof_flag(
                "future_tail_transfer_growth",
                required="universal proof preserves tail-transfer growth bound",
                witness_field="future_tail_transfer_growth_certified",
            ),
            proof_flag(
                "first_future_tail_transfer_bound",
                required="universal proof certifies first future tail-transfer bound",
                witness_field="future_initial_tail_transfer_bound_certified",
            ),
            proof_flag(
                "future_tail_transfer_growth_ratio",
                required="universal proof certifies future tail-transfer growth-ratio bound",
                witness_field="future_tail_transfer_growth_ratio_certified",
            ),
            proof_flag(
                "future_accelerated_retained_order_schedule",
                required="universal proof preserves accelerated retained-order schedule",
                witness_field="future_accelerated_order_schedule_certified",
            ),
            proof_flag(
                "first_future_retained_order_increment",
                required="universal proof certifies first future retained-order increment",
                witness_field="future_initial_retained_order_increment_certified",
            ),
            proof_flag(
                "future_retained_order_increment_growth",
                required="universal proof certifies retained-order increment-growth rule",
                witness_field="future_retained_order_increment_growth_certified",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="quantitative_witness_present",
                certified=statuses["quantitative_witness_present"],
                required="all quantitative induction witness fields supplied",
                observed=quantitative_presence,
                comparison="==",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="quantitative_witness_finite",
                certified=statuses["quantitative_witness_finite"],
                required="finite nonnegative bounds, step ratio < 1, and integer retained-order increments",
                observed="finite" if self.quantitative_witness_finite else "not finite",
                comparison="==",
            ),
            CompactifiedSundmanProofObligationDetail(
                obligation="quantitative_witness_dominates_accelerated_bounds",
                certified=statuses["quantitative_witness_dominates_accelerated_bounds"],
                required=quantitative_domination_required,
                observed=quantitative_domination_observed,
                comparison=quantitative_domination_comparison,
                witness_field=quantitative_domination_field,
            ),
            proof_flag(
                "collision_continuation",
                required="global collision continuation proof supplied",
                witness_field="collision_continuation_certified",
            ),
            proof_flag(
                "binary_collision_continuation",
                required="binary-collision continuation proof supplied",
                witness_field="binary_collision_continuation_certified",
            ),
            proof_flag(
                "triple_collision_continuation",
                required="triple-collision exclusion or continuation proof supplied",
                witness_field="triple_collision_continuation_certified",
            ),
        )

    @property
    def missing_global_proof_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation
            for obligation, certified in self.global_proof_obligation_statuses
            if not certified
        )

    @property
    def global_proof_obligations_certified(self) -> bool:
        return not self.missing_global_proof_obligations

    @property
    def missing_global_induction_reason(self) -> str | None:
        if not self.accelerated_certificate.accelerated_global_summability_certified:
            return "accelerated finite-prefix summability is not certified"
        if not self.future_domain_exhaustion_certified:
            return "no induction proof covers every later compact-Sundman domain"
        if not self.future_extension_induction_certified:
            return "no induction proof constructs every later scheduled extension"
        if not self.domain_schedule_witness_present:
            return "no geometric domain schedule witness supplies the all-future boundary-margin data"
        if not self.domain_schedule_witness_finite:
            return "geometric domain schedule witness is nonfinite or outside the required ranges"
        if not self.domain_schedule_witness_matches_accelerated_chain:
            return (
                "geometric domain schedule witness does not match the accelerated chain "
                "or its checked induction base"
            )
        if not self.future_structural_envelopes_certified:
            return "no induction proof preserves the structural future-shell envelopes"
        if not self.future_segment_growth_bound_certified:
            return "no induction proof certifies the future segment-count growth bound"
        if not self.future_state_sup_growth_bound_certified:
            return "no induction proof certifies the future state-supremum growth bound"
        if not self.future_denominator_growth_bound_certified:
            return "no induction proof certifies the future Cauchy-denominator growth bound"
        if not self.future_step_radius_bound_certified:
            return "no induction proof certifies the future step-radius ratio bound"
        if not self.future_tail_transfer_growth_certified:
            return "no induction proof preserves the tail-transfer growth bound"
        if not self.future_initial_tail_transfer_bound_certified:
            return "no induction proof certifies the first future tail-transfer bound"
        if not self.future_tail_transfer_growth_ratio_certified:
            return "no induction proof certifies the future tail-transfer growth-ratio bound"
        if not self.future_accelerated_order_schedule_certified:
            return "no induction proof preserves the accelerated retained-order schedule"
        if not self.future_initial_retained_order_increment_certified:
            return "no induction proof certifies the first future retained-order increment"
        if not self.future_retained_order_increment_growth_certified:
            return "no induction proof certifies the retained-order increment-growth rule"
        if not self.quantitative_witness_present:
            return "no quantitative induction witness supplies the accelerated future-shell bounds"
        if not self.quantitative_witness_finite:
            return "quantitative induction witness bounds are nonfinite or outside the required ranges"
        if not self.quantitative_witness_dominates_accelerated_bounds:
            return "quantitative induction witness does not dominate the accelerated future-shell bounds"
        if not self.collision_continuation_certified:
            return "global compact-Sundman summability is certified, but collision continuation is not certified"
        if not self.binary_collision_continuation_certified:
            return "collision continuation is asserted, but binary-collision continuation is not certified"
        if not self.triple_collision_continuation_certified:
            return "collision continuation is asserted, but triple-collision exclusion or continuation is not certified"
        return None


@dataclass(frozen=True)
class CompactifiedSundmanFutureShellInductionClosureCertificate:
    """Typed recurrence-level route into the accelerated induction endpoint.

    This object is one layer above
    ``CompactifiedSundmanAcceleratedInductionWitnessCertificate``.  It does not
    prove the all-future recurrence.  It records the recurrence lemmas that
    would close the future-shell induction and then projects them into the
    existing global proof-obligation ledger.
    """

    accelerated_certificate: CompactifiedSundmanAcceleratedGlobalSummabilityCertificate
    domain_exhaustion_recurrence_certified: bool = False
    scheduled_extension_recurrence_certified: bool = False
    structural_envelope_recurrence_certified: bool = False
    segment_growth_recurrence_certified: bool = False
    state_sup_growth_recurrence_certified: bool = False
    denominator_growth_recurrence_certified: bool = False
    step_radius_recurrence_certified: bool = False
    tail_transfer_recurrence_certified: bool = False
    initial_tail_transfer_recurrence_certified: bool = False
    tail_transfer_growth_ratio_recurrence_certified: bool = False
    retained_order_schedule_recurrence_certified: bool = False
    initial_retained_order_recurrence_certified: bool = False
    retained_order_growth_recurrence_certified: bool = False
    witness_source: str = "manual_recurrence_closure"
    certified_schedule_initial_boundary_margin: float | None = None
    certified_schedule_contraction: float | None = None
    certified_extension_induction_start_index: int | None = None
    certified_max_future_segment_growth_ratio: float | None = None
    certified_max_future_state_sup_growth_ratio: float | None = None
    certified_max_future_denominator_growth_ratio: float | None = None
    certified_max_future_step_radius_ratio: float | None = None
    certified_initial_tail_transfer_growth_factor_bound: float | None = None
    certified_max_future_tail_transfer_growth_ratio: float | None = None
    certified_initial_retained_order_increment: int | None = None
    certified_retained_order_increment_growth: int | None = None

    def to_accelerated_induction_witness(
        self,
        *,
        collision_continuation_certified: bool = False,
        binary_collision_continuation_certified: bool = False,
        triple_collision_continuation_certified: bool = False,
        witness_source: str | None = None,
    ) -> CompactifiedSundmanAcceleratedInductionWitnessCertificate:
        """Project recurrence lemmas into the global compact-Sundman ledger."""

        return CompactifiedSundmanAcceleratedInductionWitnessCertificate(
            accelerated_certificate=self.accelerated_certificate,
            future_domain_exhaustion_certified=bool(
                self.domain_exhaustion_recurrence_certified
            ),
            future_extension_induction_certified=bool(
                self.scheduled_extension_recurrence_certified
            ),
            future_structural_envelopes_certified=bool(
                self.structural_envelope_recurrence_certified
            ),
            future_tail_transfer_growth_certified=bool(
                self.tail_transfer_recurrence_certified
            ),
            future_accelerated_order_schedule_certified=bool(
                self.retained_order_schedule_recurrence_certified
            ),
            future_segment_growth_bound_certified=bool(
                self.segment_growth_recurrence_certified
            ),
            future_state_sup_growth_bound_certified=bool(
                self.state_sup_growth_recurrence_certified
            ),
            future_denominator_growth_bound_certified=bool(
                self.denominator_growth_recurrence_certified
            ),
            future_step_radius_bound_certified=bool(self.step_radius_recurrence_certified),
            future_initial_tail_transfer_bound_certified=bool(
                self.initial_tail_transfer_recurrence_certified
            ),
            future_tail_transfer_growth_ratio_certified=bool(
                self.tail_transfer_growth_ratio_recurrence_certified
            ),
            future_initial_retained_order_increment_certified=bool(
                self.initial_retained_order_recurrence_certified
            ),
            future_retained_order_increment_growth_certified=bool(
                self.retained_order_growth_recurrence_certified
            ),
            collision_continuation_certified=bool(collision_continuation_certified),
            binary_collision_continuation_certified=bool(
                binary_collision_continuation_certified
            ),
            triple_collision_continuation_certified=bool(
                triple_collision_continuation_certified
            ),
            witness_source=str(witness_source or self.witness_source),
            certified_schedule_initial_boundary_margin=(
                self.certified_schedule_initial_boundary_margin
            ),
            certified_schedule_contraction=self.certified_schedule_contraction,
            certified_extension_induction_start_index=(
                self.certified_extension_induction_start_index
            ),
            certified_max_future_segment_growth_ratio=(
                self.certified_max_future_segment_growth_ratio
            ),
            certified_max_future_state_sup_growth_ratio=(
                self.certified_max_future_state_sup_growth_ratio
            ),
            certified_max_future_denominator_growth_ratio=(
                self.certified_max_future_denominator_growth_ratio
            ),
            certified_max_future_step_radius_ratio=(
                self.certified_max_future_step_radius_ratio
            ),
            certified_initial_tail_transfer_growth_factor_bound=(
                self.certified_initial_tail_transfer_growth_factor_bound
            ),
            certified_max_future_tail_transfer_growth_ratio=(
                self.certified_max_future_tail_transfer_growth_ratio
            ),
            certified_initial_retained_order_increment=(
                self.certified_initial_retained_order_increment
            ),
            certified_retained_order_increment_growth=(
                self.certified_retained_order_increment_growth
            ),
        )

    @property
    def accelerated_induction_witness(
        self,
    ) -> CompactifiedSundmanAcceleratedInductionWitnessCertificate:
        return self.to_accelerated_induction_witness()

    @property
    def recurrence_obligation_statuses(self) -> tuple[tuple[str, bool], ...]:
        witness = self.accelerated_induction_witness
        return (
            (
                "accelerated_finite_prefix_summability",
                witness.accelerated_certificate.accelerated_global_summability_certified,
            ),
            ("domain_exhaustion_recurrence", self.domain_exhaustion_recurrence_certified),
            (
                "scheduled_extension_recurrence",
                self.scheduled_extension_recurrence_certified,
            ),
            (
                "geometric_domain_schedule_witness_present",
                witness.domain_schedule_witness_present,
            ),
            (
                "geometric_domain_schedule_witness_finite",
                witness.domain_schedule_witness_finite,
            ),
            (
                "geometric_domain_schedule_matches_checked_prefix",
                witness.domain_schedule_witness_matches_accelerated_chain,
            ),
            (
                "structural_envelope_recurrence",
                self.structural_envelope_recurrence_certified,
            ),
            ("segment_growth_recurrence", self.segment_growth_recurrence_certified),
            ("state_sup_growth_recurrence", self.state_sup_growth_recurrence_certified),
            (
                "denominator_growth_recurrence",
                self.denominator_growth_recurrence_certified,
            ),
            ("step_radius_recurrence", self.step_radius_recurrence_certified),
            ("tail_transfer_recurrence", self.tail_transfer_recurrence_certified),
            (
                "initial_tail_transfer_recurrence",
                self.initial_tail_transfer_recurrence_certified,
            ),
            (
                "tail_transfer_growth_ratio_recurrence",
                self.tail_transfer_growth_ratio_recurrence_certified,
            ),
            (
                "retained_order_schedule_recurrence",
                self.retained_order_schedule_recurrence_certified,
            ),
            (
                "initial_retained_order_recurrence",
                self.initial_retained_order_recurrence_certified,
            ),
            (
                "retained_order_growth_recurrence",
                self.retained_order_growth_recurrence_certified,
            ),
            ("quantitative_witness_present", witness.quantitative_witness_present),
            ("quantitative_witness_finite", witness.quantitative_witness_finite),
            (
                "quantitative_witness_dominates_accelerated_bounds",
                witness.quantitative_witness_dominates_accelerated_bounds,
            ),
        )

    @property
    def recurrence_obligation_details(
        self,
    ) -> tuple[CompactifiedSundmanProofObligationDetail, ...]:
        witness = self.accelerated_induction_witness
        global_details = {
            detail.obligation: detail for detail in witness.global_proof_obligation_details
        }

        def recurrence_flag(
            obligation: str,
            certified: bool,
            *,
            required: str,
            witness_field: str,
        ) -> CompactifiedSundmanProofObligationDetail:
            return CompactifiedSundmanProofObligationDetail(
                obligation=obligation,
                certified=bool(certified),
                required=required,
                observed="certified" if certified else "not certified",
                comparison="==",
                witness_field=witness_field,
            )

        return (
            global_details["accelerated_finite_prefix_summability"],
            recurrence_flag(
                "domain_exhaustion_recurrence",
                self.domain_exhaustion_recurrence_certified,
                required="recurrence proves every later compact-Sundman domain is reached",
                witness_field="domain_exhaustion_recurrence_certified",
            ),
            recurrence_flag(
                "scheduled_extension_recurrence",
                self.scheduled_extension_recurrence_certified,
                required="recurrence constructs the next scheduled domain from the previous one",
                witness_field="scheduled_extension_recurrence_certified",
            ),
            global_details["geometric_domain_schedule_witness_present"],
            global_details["geometric_domain_schedule_witness_finite"],
            global_details["geometric_domain_schedule_matches_checked_prefix"],
            recurrence_flag(
                "structural_envelope_recurrence",
                self.structural_envelope_recurrence_certified,
                required="recurrence preserves the structural envelope template",
                witness_field="structural_envelope_recurrence_certified",
            ),
            recurrence_flag(
                "segment_growth_recurrence",
                self.segment_growth_recurrence_certified,
                required="recurrence proves the segment-count growth bound",
                witness_field="segment_growth_recurrence_certified",
            ),
            recurrence_flag(
                "state_sup_growth_recurrence",
                self.state_sup_growth_recurrence_certified,
                required="recurrence proves the state-supremum growth bound",
                witness_field="state_sup_growth_recurrence_certified",
            ),
            recurrence_flag(
                "denominator_growth_recurrence",
                self.denominator_growth_recurrence_certified,
                required="recurrence proves the Cauchy-denominator growth bound",
                witness_field="denominator_growth_recurrence_certified",
            ),
            recurrence_flag(
                "step_radius_recurrence",
                self.step_radius_recurrence_certified,
                required="recurrence proves the step-radius ratio bound",
                witness_field="step_radius_recurrence_certified",
            ),
            recurrence_flag(
                "tail_transfer_recurrence",
                self.tail_transfer_recurrence_certified,
                required="recurrence preserves the tail-transfer bound template",
                witness_field="tail_transfer_recurrence_certified",
            ),
            recurrence_flag(
                "initial_tail_transfer_recurrence",
                self.initial_tail_transfer_recurrence_certified,
                required="recurrence proves the first future tail-transfer bound",
                witness_field="initial_tail_transfer_recurrence_certified",
            ),
            recurrence_flag(
                "tail_transfer_growth_ratio_recurrence",
                self.tail_transfer_growth_ratio_recurrence_certified,
                required="recurrence proves the tail-transfer growth-ratio bound",
                witness_field="tail_transfer_growth_ratio_recurrence_certified",
            ),
            recurrence_flag(
                "retained_order_schedule_recurrence",
                self.retained_order_schedule_recurrence_certified,
                required="recurrence preserves the accelerated retained-order schedule",
                witness_field="retained_order_schedule_recurrence_certified",
            ),
            recurrence_flag(
                "initial_retained_order_recurrence",
                self.initial_retained_order_recurrence_certified,
                required="recurrence proves the first future retained-order increment",
                witness_field="initial_retained_order_recurrence_certified",
            ),
            recurrence_flag(
                "retained_order_growth_recurrence",
                self.retained_order_growth_recurrence_certified,
                required="recurrence proves the retained-order increment-growth rule",
                witness_field="retained_order_growth_recurrence_certified",
            ),
            global_details["quantitative_witness_present"],
            global_details["quantitative_witness_finite"],
            global_details["quantitative_witness_dominates_accelerated_bounds"],
        )

    @property
    def missing_recurrence_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation
            for obligation, certified in self.recurrence_obligation_statuses
            if not certified
        )

    @property
    def future_shell_induction_certified(self) -> bool:
        return self.accelerated_induction_witness.future_shell_induction_certified

    @property
    def global_domain_certified(self) -> bool:
        return self.accelerated_induction_witness.global_domain_certified

    @property
    def global_tail_bound(self) -> float:
        return self.accelerated_induction_witness.global_tail_bound

    @property
    def missing_recurrence_reason(self) -> str | None:
        if not self.missing_recurrence_obligations:
            return None
        first_missing = self.missing_recurrence_obligations[0]
        reasons = {
            "accelerated_finite_prefix_summability": (
                "accelerated finite-prefix summability is not certified"
            ),
            "domain_exhaustion_recurrence": (
                "no recurrence proof exhausts all later compact-Sundman domains"
            ),
            "scheduled_extension_recurrence": (
                "no recurrence proof constructs every later scheduled extension"
            ),
            "structural_envelope_recurrence": (
                "no recurrence proof preserves the structural envelope template"
            ),
            "segment_growth_recurrence": (
                "no recurrence proof certifies segment-count growth"
            ),
            "state_sup_growth_recurrence": (
                "no recurrence proof certifies state-supremum growth"
            ),
            "denominator_growth_recurrence": (
                "no recurrence proof certifies Cauchy-denominator growth"
            ),
            "step_radius_recurrence": (
                "no recurrence proof certifies step-radius ratio growth"
            ),
            "tail_transfer_recurrence": (
                "no recurrence proof preserves the tail-transfer template"
            ),
            "initial_tail_transfer_recurrence": (
                "no recurrence proof certifies the first future tail-transfer bound"
            ),
            "tail_transfer_growth_ratio_recurrence": (
                "no recurrence proof certifies tail-transfer growth-ratio control"
            ),
            "retained_order_schedule_recurrence": (
                "no recurrence proof preserves the accelerated retained-order schedule"
            ),
            "initial_retained_order_recurrence": (
                "no recurrence proof certifies the first future retained-order increment"
            ),
            "retained_order_growth_recurrence": (
                "no recurrence proof certifies retained-order increment-growth"
            ),
        }
        return reasons.get(
            first_missing,
            self.accelerated_induction_witness.missing_global_induction_reason,
        )


@dataclass(frozen=True)
class CompactifiedSundmanScalarRecurrenceInductionCertificate:
    """Executable scalar inequalities for future-shell recurrence subproofs.

    This certificate handles the arithmetic part of the recurrence closure: the
    proposed base and recurrent transition bounds must dominate the accelerated
    future-shell envelope.  It still relies on separate template proofs for
    domain exhaustion, scheduled extension, structural envelope form, tail
    transfer form, and retained-order schedule form.
    """

    accelerated_certificate: CompactifiedSundmanAcceleratedGlobalSummabilityCertificate
    domain_exhaustion_recurrence_certified: bool = False
    scheduled_extension_recurrence_certified: bool = False
    structural_envelope_template_certified: bool = False
    tail_transfer_template_certified: bool = False
    retained_order_schedule_template_certified: bool = False
    witness_source: str = "manual_scalar_recurrence_induction"
    certified_schedule_initial_boundary_margin: float | None = None
    certified_schedule_contraction: float | None = None
    certified_extension_induction_start_index: int | None = None
    certified_initial_segment_growth_ratio_bound: float | None = None
    certified_recurrent_segment_growth_ratio_bound: float | None = None
    certified_initial_state_sup_growth_ratio_bound: float | None = None
    certified_recurrent_state_sup_growth_ratio_bound: float | None = None
    certified_initial_denominator_growth_ratio_bound: float | None = None
    certified_recurrent_denominator_growth_ratio_bound: float | None = None
    certified_initial_step_radius_ratio_bound: float | None = None
    certified_recurrent_step_radius_ratio_bound: float | None = None
    certified_initial_tail_transfer_growth_factor_bound: float | None = None
    certified_recurrent_tail_transfer_growth_factor_bound: float | None = None
    certified_initial_tail_transfer_growth_ratio_bound: float | None = None
    certified_recurrent_tail_transfer_growth_ratio_bound: float | None = None
    certified_initial_retained_order_increment_bound: int | None = None
    certified_recurrent_retained_order_increment_bound: int | None = None
    certified_initial_retained_order_growth_bound: int | None = None
    certified_recurrent_retained_order_growth_bound: int | None = None

    def to_future_shell_induction_closure(
        self,
    ) -> CompactifiedSundmanFutureShellInductionClosureCertificate:
        """Project checked scalar recurrence inequalities into the closure object."""

        return CompactifiedSundmanFutureShellInductionClosureCertificate(
            accelerated_certificate=self.accelerated_certificate,
            domain_exhaustion_recurrence_certified=self.domain_exhaustion_recurrence_certified,
            scheduled_extension_recurrence_certified=(
                self.scheduled_extension_recurrence_certified
            ),
            structural_envelope_recurrence_certified=(
                self.structural_envelope_template_certified
            ),
            segment_growth_recurrence_certified=self.segment_growth_recurrence_certified,
            state_sup_growth_recurrence_certified=self.state_sup_growth_recurrence_certified,
            denominator_growth_recurrence_certified=(
                self.denominator_growth_recurrence_certified
            ),
            step_radius_recurrence_certified=self.step_radius_recurrence_certified,
            tail_transfer_recurrence_certified=self.tail_transfer_template_certified,
            initial_tail_transfer_recurrence_certified=(
                self.initial_tail_transfer_recurrence_certified
            ),
            tail_transfer_growth_ratio_recurrence_certified=(
                self.tail_transfer_growth_ratio_recurrence_certified
            ),
            retained_order_schedule_recurrence_certified=(
                self.retained_order_schedule_template_certified
            ),
            initial_retained_order_recurrence_certified=(
                self.initial_retained_order_recurrence_certified
            ),
            retained_order_growth_recurrence_certified=(
                self.retained_order_growth_recurrence_certified
            ),
            witness_source=self.witness_source,
            certified_schedule_initial_boundary_margin=(
                self.certified_schedule_initial_boundary_margin
            ),
            certified_schedule_contraction=self.certified_schedule_contraction,
            certified_extension_induction_start_index=(
                self.certified_extension_induction_start_index
            ),
            certified_max_future_segment_growth_ratio=self._upper_observed_bound(
                self.certified_initial_segment_growth_ratio_bound,
                self.certified_recurrent_segment_growth_ratio_bound,
            ),
            certified_max_future_state_sup_growth_ratio=self._upper_observed_bound(
                self.certified_initial_state_sup_growth_ratio_bound,
                self.certified_recurrent_state_sup_growth_ratio_bound,
            ),
            certified_max_future_denominator_growth_ratio=self._upper_observed_bound(
                self.certified_initial_denominator_growth_ratio_bound,
                self.certified_recurrent_denominator_growth_ratio_bound,
            ),
            certified_max_future_step_radius_ratio=self._upper_observed_bound(
                self.certified_initial_step_radius_ratio_bound,
                self.certified_recurrent_step_radius_ratio_bound,
            ),
            certified_initial_tail_transfer_growth_factor_bound=(
                self._upper_observed_bound(
                    self.certified_initial_tail_transfer_growth_factor_bound,
                    self.certified_recurrent_tail_transfer_growth_factor_bound,
                )
            ),
            certified_max_future_tail_transfer_growth_ratio=self._upper_observed_bound(
                self.certified_initial_tail_transfer_growth_ratio_bound,
                self.certified_recurrent_tail_transfer_growth_ratio_bound,
            ),
            certified_initial_retained_order_increment=self._lower_observed_bound(
                self.certified_initial_retained_order_increment_bound,
                self.certified_recurrent_retained_order_increment_bound,
            ),
            certified_retained_order_increment_growth=self._lower_observed_bound(
                self.certified_initial_retained_order_growth_bound,
                self.certified_recurrent_retained_order_growth_bound,
            ),
        )

    @property
    def scalar_recurrence_obligation_statuses(self) -> tuple[tuple[str, bool], ...]:
        return tuple((detail.obligation, detail.certified) for detail in self.scalar_recurrence_details)

    @property
    def scalar_recurrence_details(
        self,
    ) -> tuple[CompactifiedSundmanProofObligationDetail, ...]:
        accelerated = self.accelerated_certificate
        return (
            self._proof_flag_detail(
                "domain_exhaustion_recurrence",
                self.domain_exhaustion_recurrence_certified,
                "domain recurrence proof supplied",
                "domain_exhaustion_recurrence_certified",
            ),
            self._proof_flag_detail(
                "scheduled_extension_recurrence",
                self.scheduled_extension_recurrence_certified,
                "scheduled extension recurrence proof supplied",
                "scheduled_extension_recurrence_certified",
            ),
            self._proof_flag_detail(
                "structural_envelope_template",
                self.structural_envelope_template_certified,
                "structural envelope recurrence template proof supplied",
                "structural_envelope_template_certified",
            ),
            self._upper_bound_detail(
                "segment_growth_recurrence",
                self.certified_initial_segment_growth_ratio_bound,
                self.certified_recurrent_segment_growth_ratio_bound,
                accelerated.max_future_segment_growth_ratio,
                "certified_initial_segment_growth_ratio_bound",
                "certified_recurrent_segment_growth_ratio_bound",
            ),
            self._upper_bound_detail(
                "state_sup_growth_recurrence",
                self.certified_initial_state_sup_growth_ratio_bound,
                self.certified_recurrent_state_sup_growth_ratio_bound,
                accelerated.max_future_state_sup_growth_ratio,
                "certified_initial_state_sup_growth_ratio_bound",
                "certified_recurrent_state_sup_growth_ratio_bound",
            ),
            self._upper_bound_detail(
                "denominator_growth_recurrence",
                self.certified_initial_denominator_growth_ratio_bound,
                self.certified_recurrent_denominator_growth_ratio_bound,
                accelerated.max_future_denominator_growth_ratio,
                "certified_initial_denominator_growth_ratio_bound",
                "certified_recurrent_denominator_growth_ratio_bound",
            ),
            self._upper_bound_detail(
                "step_radius_recurrence",
                self.certified_initial_step_radius_ratio_bound,
                self.certified_recurrent_step_radius_ratio_bound,
                accelerated.max_future_step_radius_ratio,
                "certified_initial_step_radius_ratio_bound",
                "certified_recurrent_step_radius_ratio_bound",
            ),
            self._proof_flag_detail(
                "tail_transfer_template",
                self.tail_transfer_template_certified,
                "tail-transfer recurrence template proof supplied",
                "tail_transfer_template_certified",
            ),
            self._upper_bound_detail(
                "initial_tail_transfer_recurrence",
                self.certified_initial_tail_transfer_growth_factor_bound,
                self.certified_recurrent_tail_transfer_growth_factor_bound,
                accelerated.initial_tail_transfer_growth_factor_bound,
                "certified_initial_tail_transfer_growth_factor_bound",
                "certified_recurrent_tail_transfer_growth_factor_bound",
            ),
            self._upper_bound_detail(
                "tail_transfer_growth_ratio_recurrence",
                self.certified_initial_tail_transfer_growth_ratio_bound,
                self.certified_recurrent_tail_transfer_growth_ratio_bound,
                accelerated.max_future_tail_transfer_growth_ratio,
                "certified_initial_tail_transfer_growth_ratio_bound",
                "certified_recurrent_tail_transfer_growth_ratio_bound",
            ),
            self._proof_flag_detail(
                "retained_order_schedule_template",
                self.retained_order_schedule_template_certified,
                "retained-order schedule recurrence template proof supplied",
                "retained_order_schedule_template_certified",
            ),
            self._lower_integer_detail(
                "initial_retained_order_recurrence",
                self.certified_initial_retained_order_increment_bound,
                self.certified_recurrent_retained_order_increment_bound,
                accelerated.initial_retained_order_increment,
                "certified_initial_retained_order_increment_bound",
                "certified_recurrent_retained_order_increment_bound",
            ),
            self._lower_integer_detail(
                "retained_order_growth_recurrence",
                self.certified_initial_retained_order_growth_bound,
                self.certified_recurrent_retained_order_growth_bound,
                accelerated.retained_order_increment_growth,
                "certified_initial_retained_order_growth_bound",
                "certified_recurrent_retained_order_growth_bound",
            ),
        )

    @property
    def scalar_recurrence_certified(self) -> bool:
        return all(detail.certified for detail in self.scalar_recurrence_details)

    @property
    def missing_scalar_recurrence_obligations(self) -> tuple[str, ...]:
        return tuple(detail.obligation for detail in self.scalar_recurrence_details if not detail.certified)

    @property
    def segment_growth_recurrence_certified(self) -> bool:
        return self._detail_certified("segment_growth_recurrence")

    @property
    def state_sup_growth_recurrence_certified(self) -> bool:
        return self._detail_certified("state_sup_growth_recurrence")

    @property
    def denominator_growth_recurrence_certified(self) -> bool:
        return self._detail_certified("denominator_growth_recurrence")

    @property
    def step_radius_recurrence_certified(self) -> bool:
        return self._detail_certified("step_radius_recurrence")

    @property
    def initial_tail_transfer_recurrence_certified(self) -> bool:
        return self._detail_certified("initial_tail_transfer_recurrence")

    @property
    def tail_transfer_growth_ratio_recurrence_certified(self) -> bool:
        return self._detail_certified("tail_transfer_growth_ratio_recurrence")

    @property
    def initial_retained_order_recurrence_certified(self) -> bool:
        return self._detail_certified("initial_retained_order_recurrence")

    @property
    def retained_order_growth_recurrence_certified(self) -> bool:
        return self._detail_certified("retained_order_growth_recurrence")

    def _detail_certified(self, obligation: str) -> bool:
        return next(
            detail.certified for detail in self.scalar_recurrence_details if detail.obligation == obligation
        )

    def _proof_flag_detail(
        self,
        obligation: str,
        certified: bool,
        required: str,
        witness_field: str,
    ) -> CompactifiedSundmanProofObligationDetail:
        return CompactifiedSundmanProofObligationDetail(
            obligation=obligation,
            certified=bool(certified),
            required=required,
            observed="certified" if certified else "not certified",
            comparison="==",
            witness_field=witness_field,
        )

    def _upper_bound_detail(
        self,
        obligation: str,
        initial_observed: float | None,
        recurrent_observed: float | None,
        required: float,
        initial_field: str,
        recurrent_field: str,
    ) -> CompactifiedSundmanProofObligationDetail:
        observed_bound = self._upper_observed_bound(initial_observed, recurrent_observed)
        certified = bool(
            observed_bound is not None
            and np.isfinite(observed_bound)
            and np.isfinite(required)
            and observed_bound <= required
        )
        return CompactifiedSundmanProofObligationDetail(
            obligation=obligation,
            certified=certified,
            required=float(required),
            observed=observed_bound,
            comparison="<=",
            witness_field=f"{initial_field}, {recurrent_field}",
        )

    def _lower_integer_detail(
        self,
        obligation: str,
        initial_observed: int | None,
        recurrent_observed: int | None,
        required: int,
        initial_field: str,
        recurrent_field: str,
    ) -> CompactifiedSundmanProofObligationDetail:
        observed_bound = self._lower_observed_bound(initial_observed, recurrent_observed)
        certified = bool(
            observed_bound is not None
            and np.isfinite(observed_bound)
            and int(observed_bound) == observed_bound
            and observed_bound >= required
        )
        return CompactifiedSundmanProofObligationDetail(
            obligation=obligation,
            certified=certified,
            required=int(required),
            observed=observed_bound,
            comparison=">=",
            witness_field=f"{initial_field}, {recurrent_field}",
        )

    def _upper_observed_bound(
        self,
        initial_observed: float | None,
        recurrent_observed: float | None,
    ) -> float | None:
        if initial_observed is None or recurrent_observed is None:
            return None
        return float(max(initial_observed, recurrent_observed))

    def _lower_observed_bound(
        self,
        initial_observed: int | None,
        recurrent_observed: int | None,
    ) -> int | None:
        if initial_observed is None or recurrent_observed is None:
            return None
        return int(min(initial_observed, recurrent_observed))


@dataclass(frozen=True)
class CompactifiedSundmanFutureEnvelopeHoldoutCertificate:
    """Validate a prefix-derived future-shell envelope on held-out finite shells.

    This is still finite evidence, not a global induction proof.  It separates
    envelope coverage of shell diagnostics from actual held-out tail growth.
    """

    training_chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate
    validation_extensions: tuple[CompactifiedSundmanGeometricExhaustionExtensionCertificate, ...]
    conditional_certificate: CompactifiedSundmanConditionalGlobalSummabilityCertificate
    tail_transfer_safety_factor: float = 1.0
    tail_transfer_growth_safety_factor: float = 1.0

    @property
    def validation_extension_count(self) -> int:
        return len(self.validation_extensions)

    @property
    def validation_connected(self) -> bool:
        current = self.training_chain.final_schedule
        for extension in self.validation_extensions:
            if extension.previous_schedule != current:
                return False
            current = extension.extended_schedule
        return bool(self.validation_extensions)

    @property
    def validation_extensions_certified(self) -> bool:
        return bool(
            self.validation_extensions
            and all(extension.certified for extension in self.validation_extensions)
        )

    @property
    def validation_tail_budget_certified(self) -> bool:
        return bool(
            self.validation_extensions
            and all(extension.tail_budget_certified for extension in self.validation_extensions)
        )

    @property
    def validation_incremental_tail_bounds(self) -> tuple[float, ...]:
        return tuple(extension.added_incremental_tail_bound for extension in self.validation_extensions)

    @property
    def validation_total_incremental_tail_bound(self) -> float:
        if not self.validation_tail_budget_certified:
            return float("inf")
        return float(sum(self.validation_incremental_tail_bounds))

    @property
    def validation_incremental_segment_counts(self) -> tuple[int, ...]:
        counts: list[int] = []
        for extension in self.validation_extensions:
            previous_domains = extension.previous_schedule.prefix.domain_covers
            if not previous_domains:
                counts.append(extension.added_domain.segment_count)
            else:
                counts.append(extension.added_domain.segment_count_outside(previous_domains[-1].compact_interval))
        return tuple(counts)

    @property
    def validation_max_step_radius_ratios(self) -> tuple[float, ...]:
        ratios: list[float] = []
        for extension in self.validation_extensions:
            previous_domains = extension.previous_schedule.prefix.domain_covers
            if not previous_domains:
                ratios.append(extension.added_domain.max_step_radius_ratio)
            else:
                ratios.append(
                    extension.added_domain.max_step_radius_ratio_outside(previous_domains[-1].compact_interval)
                )
        return tuple(ratios)

    @property
    def max_validation_step_radius_ratio(self) -> float:
        ratios = self.validation_max_step_radius_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def validation_max_state_sup_bounds(self) -> tuple[float, ...]:
        bounds: list[float] = []
        for extension in self.validation_extensions:
            previous_domains = extension.previous_schedule.prefix.domain_covers
            if not previous_domains:
                state_bounds = [
                    bound
                    for cover in extension.added_domain.covers
                    for bound in cover.segment_state_sup_bounds
                    if np.isfinite(bound)
                ]
                bounds.append(float(max(state_bounds)) if state_bounds else float("inf"))
            else:
                bounds.append(
                    extension.added_domain.max_state_sup_bound_outside(previous_domains[-1].compact_interval)
                )
        return tuple(bounds)

    @staticmethod
    def _growth_ratios(previous: float, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or not np.isfinite(previous) or previous <= 0.0:
            return ()
        ratios: list[float] = []
        last = float(previous)
        for value in values:
            if not np.isfinite(value):
                return ()
            ratios.append(float(value / last))
            last = float(value)
        return tuple(ratios)

    @property
    def validation_segment_growth_ratios(self) -> tuple[float, ...]:
        training_counts = self.training_chain.added_incremental_segment_counts
        if not training_counts:
            return ()
        return self._growth_ratios(
            float(training_counts[-1]),
            tuple(float(count) for count in self.validation_incremental_segment_counts),
        )

    @property
    def max_validation_segment_growth_ratio(self) -> float:
        ratios = self.validation_segment_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def validation_state_sup_growth_ratios(self) -> tuple[float, ...]:
        training_bounds = self.training_chain.added_max_state_sup_bounds
        if not training_bounds:
            return ()
        return self._growth_ratios(training_bounds[-1], self.validation_max_state_sup_bounds)

    @property
    def max_validation_state_sup_growth_ratio(self) -> float:
        ratios = self.validation_state_sup_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def validation_cauchy_tail_denominator_factors(self) -> tuple[float, ...]:
        factors: list[float] = []
        for ratio in self.validation_max_step_radius_ratios:
            if not np.isfinite(ratio) or ratio < 0.0 or ratio >= 1.0:
                factors.append(float("inf"))
            else:
                factors.append(float(1.0 / (1.0 - ratio)))
        return tuple(factors)

    @property
    def validation_cauchy_tail_denominator_growth_ratios(self) -> tuple[float, ...]:
        training_factors = self.training_chain.added_cauchy_tail_denominator_factors
        if not training_factors:
            return ()
        return self._growth_ratios(training_factors[-1], self.validation_cauchy_tail_denominator_factors)

    @property
    def max_validation_cauchy_tail_denominator_growth_ratio(self) -> float:
        ratios = self.validation_cauchy_tail_denominator_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def validation_retained_orders(self) -> tuple[int, ...]:
        orders: list[int] = []
        for extension in self.validation_extensions:
            min_order = extension.added_min_retained_order
            max_order = extension.added_max_retained_order
            if min_order < 0 or min_order != max_order:
                return ()
            orders.append(min_order)
        return tuple(orders)

    @property
    def validation_retained_order_increments(self) -> tuple[int, ...]:
        training_orders = self.training_chain.uniform_added_retained_orders
        validation_orders = self.validation_retained_orders
        if not training_orders or not validation_orders:
            return ()
        orders = (training_orders[-1],) + validation_orders
        return tuple(current - previous for previous, current in zip(orders, orders[1:]))

    @property
    def validation_retained_order_growth_matches(self) -> bool:
        increments = self.validation_retained_order_increments
        return bool(
            increments
            and self.conditional_certificate.retained_order_increment > 0
            and all(increment == self.conditional_certificate.retained_order_increment for increment in increments)
        )

    @property
    def validation_diagnostics_within_envelope(self) -> bool:
        certificate = self.conditional_certificate
        return bool(
            certificate.conditional_global_summability_certified
            and self.validation_connected
            and self.validation_extensions_certified
            and self.validation_retained_order_growth_matches
            and self.max_validation_segment_growth_ratio <= certificate.max_future_segment_growth_ratio
            and self.max_validation_state_sup_growth_ratio <= certificate.max_future_state_sup_growth_ratio
            and self.max_validation_cauchy_tail_denominator_growth_ratio
            <= certificate.max_future_denominator_growth_ratio
            and self.max_validation_step_radius_ratio <= certificate.max_future_step_radius_ratio
        )

    @property
    def validation_tail_growth_ratios(self) -> tuple[float, ...]:
        training_tails = self.training_chain.added_incremental_tail_bounds
        if not training_tails:
            return ()
        return self._growth_ratios(training_tails[-1], self.validation_incremental_tail_bounds)

    @property
    def max_validation_tail_growth_ratio(self) -> float:
        ratios = self.validation_tail_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def validation_cauchy_majorant_transition_factor_bounds(self) -> tuple[float, ...]:
        if not self.validation_retained_order_growth_matches:
            return ()
        factors: list[float] = []
        for segment_growth, state_growth, denominator_growth, step_radius_ratio, order_increment in zip(
            self.validation_segment_growth_ratios,
            self.validation_state_sup_growth_ratios,
            self.validation_cauchy_tail_denominator_growth_ratios,
            self.validation_max_step_radius_ratios,
            self.validation_retained_order_increments,
        ):
            components = (
                segment_growth,
                state_growth,
                denominator_growth,
                step_radius_ratio,
            )
            if not all(np.isfinite(component) for component in components):
                return ()
            factors.append(
                float(
                    segment_growth
                    * state_growth
                    * denominator_growth
                    * step_radius_ratio**order_increment
                )
            )
        return tuple(factors)

    @property
    def validation_tail_transfer_growth_factors(self) -> tuple[float, ...]:
        structural_factors = self.validation_cauchy_majorant_transition_factor_bounds
        tail_ratios = self.validation_tail_growth_ratios
        if len(structural_factors) != len(tail_ratios):
            return ()
        factors: list[float] = []
        for tail_ratio, structural_factor in zip(tail_ratios, structural_factors):
            if not np.isfinite(tail_ratio) or not np.isfinite(structural_factor) or structural_factor <= 0.0:
                return ()
            factors.append(float(tail_ratio / structural_factor))
        return tuple(factors)

    @property
    def max_validation_tail_transfer_growth_factor(self) -> float:
        factors = self.validation_tail_transfer_growth_factors
        return float(max(factors)) if factors else float("inf")

    @property
    def validation_tail_transfer_growth_ratios(self) -> tuple[float, ...]:
        training_factors = self.training_chain.tail_transfer_growth_factors
        if not training_factors:
            return ()
        return self._growth_ratios(training_factors[-1], self.validation_tail_transfer_growth_factors)

    @property
    def max_validation_tail_transfer_growth_ratio(self) -> float:
        ratios = self.validation_tail_transfer_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def training_tail_transfer_growth_factor_bound(self) -> float:
        factor = self.training_chain.max_tail_transfer_growth_factor
        safety_factor = self.tail_transfer_safety_factor
        if not np.isfinite(factor) or not np.isfinite(safety_factor) or safety_factor < 1.0:
            return float("inf")
        return float(factor * safety_factor)

    @property
    def validation_tail_transfer_within_training_bound(self) -> bool:
        bound = self.training_tail_transfer_growth_factor_bound
        return bool(
            self.validation_tail_transfer_growth_factors
            and np.isfinite(bound)
            and self.max_validation_tail_transfer_growth_factor <= bound
        )

    @property
    def training_tail_transfer_growth_ratio_bound(self) -> float:
        ratio = self.training_chain.max_tail_transfer_growth_ratio
        safety_factor = self.tail_transfer_growth_safety_factor
        if not np.isfinite(ratio) or not np.isfinite(safety_factor) or safety_factor < 1.0:
            return float("inf")
        return float(ratio * safety_factor)

    @property
    def validation_tail_transfer_growth_within_training_ratio_bound(self) -> bool:
        bound = self.training_tail_transfer_growth_ratio_bound
        return bool(
            self.validation_tail_transfer_growth_ratios
            and np.isfinite(bound)
            and self.max_validation_tail_transfer_growth_ratio <= bound
        )

    @property
    def projected_tail_transfer_growth_bounds(self) -> tuple[float, ...]:
        training_factors = self.training_chain.tail_transfer_growth_factors
        growth_ratio = self.training_tail_transfer_growth_ratio_bound
        if not training_factors or not np.isfinite(growth_ratio):
            return ()
        bounds: list[float] = []
        current = float(training_factors[-1])
        for _ in self.validation_extensions:
            current *= growth_ratio
            bounds.append(float(current))
        return tuple(bounds)

    @property
    def validation_tail_transfer_within_projected_growth_bounds(self) -> bool:
        projected = self.projected_tail_transfer_growth_bounds
        validation = self.validation_tail_transfer_growth_factors
        return bool(
            projected
            and len(projected) == len(validation)
            and all(value <= bound for value, bound in zip(validation, projected))
        )

    @property
    def projected_tail_transfer_adjusted_validation_ratio_bounds(self) -> tuple[float, ...]:
        projected = self.projected_tail_transfer_growth_bounds
        factor = self.conditional_certificate.conditional_tail_decay_factor_bound
        if not projected or not np.isfinite(factor):
            return ()
        return tuple(float(factor * transfer) for transfer in projected)

    @property
    def projected_validation_tail_bounds(self) -> tuple[float, ...]:
        ratio_bounds = self.projected_tail_transfer_adjusted_validation_ratio_bounds
        training_tails = self.training_chain.added_incremental_tail_bounds
        if not ratio_bounds or not training_tails:
            return ()
        bounds: list[float] = []
        current = float(training_tails[-1])
        for ratio_bound in ratio_bounds:
            if not np.isfinite(ratio_bound):
                return ()
            current *= ratio_bound
            bounds.append(float(current))
        return tuple(bounds)

    @property
    def projected_validation_tail_sum_bound(self) -> float:
        bounds = self.projected_validation_tail_bounds
        return float(sum(bounds)) if bounds else float("inf")

    @property
    def validation_tail_growth_within_projected_transfer_bounds(self) -> bool:
        ratio_bounds = self.projected_tail_transfer_adjusted_validation_ratio_bounds
        tail_ratios = self.validation_tail_growth_ratios
        return bool(
            ratio_bounds
            and len(ratio_bounds) == len(tail_ratios)
            and all(tail_ratio <= bound for tail_ratio, bound in zip(tail_ratios, ratio_bounds))
        )

    @property
    def validation_tail_sum_within_projected_transfer_bounds(self) -> bool:
        bound = self.projected_validation_tail_sum_bound
        return bool(
            self.validation_tail_budget_certified
            and np.isfinite(bound)
            and self.validation_total_incremental_tail_bound <= bound
        )

    @property
    def tail_transfer_adjusted_conditional_factor_bound(self) -> float:
        factor = self.conditional_certificate.conditional_tail_decay_factor_bound
        transfer = self.training_tail_transfer_growth_factor_bound
        if not np.isfinite(factor) or not np.isfinite(transfer):
            return float("inf")
        return float(factor * transfer)

    @property
    def tail_transfer_adjusted_future_tail_remainder_bound(self) -> float:
        factor = self.tail_transfer_adjusted_conditional_factor_bound
        training_tails = self.training_chain.added_incremental_tail_bounds
        if not training_tails or not np.isfinite(factor) or factor >= 1.0:
            return float("inf")
        return float(training_tails[-1] * factor / (1.0 - factor))

    @property
    def validation_tail_growth_within_tail_transfer_adjusted_factor(self) -> bool:
        factor = self.tail_transfer_adjusted_conditional_factor_bound
        return bool(
            self.validation_tail_budget_certified
            and self.validation_tail_growth_ratios
            and np.isfinite(factor)
            and self.max_validation_tail_growth_ratio <= factor
        )

    @property
    def validation_tail_sum_within_tail_transfer_adjusted_remainder(self) -> bool:
        remainder = self.tail_transfer_adjusted_future_tail_remainder_bound
        return bool(
            self.validation_tail_budget_certified
            and np.isfinite(remainder)
            and self.validation_total_incremental_tail_bound <= remainder
        )

    @property
    def validation_tail_growth_within_conditional_factor(self) -> bool:
        factor = self.conditional_certificate.conditional_tail_decay_factor_bound
        return bool(
            self.validation_tail_budget_certified
            and self.validation_tail_growth_ratios
            and np.isfinite(factor)
            and self.max_validation_tail_growth_ratio <= factor
        )

    @property
    def validation_tail_sum_within_conditional_remainder(self) -> bool:
        remainder = self.conditional_certificate.conditional_future_tail_remainder_bound
        return bool(
            self.validation_tail_budget_certified
            and np.isfinite(remainder)
            and self.validation_total_incremental_tail_bound <= remainder
        )

    @property
    def holdout_certified(self) -> bool:
        return bool(
            self.validation_diagnostics_within_envelope
            and self.validation_tail_growth_within_conditional_factor
            and self.validation_tail_sum_within_conditional_remainder
        )

    @property
    def tail_transfer_adjusted_holdout_certified(self) -> bool:
        return bool(
            self.validation_diagnostics_within_envelope
            and self.validation_tail_transfer_within_training_bound
            and self.validation_tail_growth_within_tail_transfer_adjusted_factor
            and self.validation_tail_sum_within_tail_transfer_adjusted_remainder
        )

    @property
    def projected_tail_transfer_holdout_certified(self) -> bool:
        return bool(
            self.validation_diagnostics_within_envelope
            and self.validation_tail_transfer_growth_within_training_ratio_bound
            and self.validation_tail_transfer_within_projected_growth_bounds
            and self.validation_tail_growth_within_projected_transfer_bounds
            and self.validation_tail_sum_within_projected_transfer_bounds
        )

    @property
    def global_domain_certified(self) -> bool:
        return False

    @property
    def missing_global_induction_reason(self) -> str | None:
        if not self.holdout_certified:
            if not self.validation_diagnostics_within_envelope:
                return "held-out shell diagnostics do not stay inside the prefix-derived future envelope"
            return "held-out shell tails exceed the conditional tail-growth prediction"
        return (
            "held-out shells validate the prefix-derived envelope, but no induction proof "
            "shows that the envelope persists for every later shell"
        )


@dataclass(frozen=True)
class CompactifiedSundmanAcceleratedFutureEnvelopeHoldoutCertificate:
    """Validate an accelerated finite-prefix envelope on held-out finite shells."""

    training_chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate
    validation_extensions: tuple[CompactifiedSundmanGeometricExhaustionExtensionCertificate, ...]
    accelerated_certificate: CompactifiedSundmanAcceleratedGlobalSummabilityCertificate

    @property
    def validation_extension_count(self) -> int:
        return len(self.validation_extensions)

    @property
    def validation_connected(self) -> bool:
        current = self.training_chain.final_schedule
        for extension in self.validation_extensions:
            if extension.previous_schedule != current:
                return False
            current = extension.extended_schedule
        return bool(self.validation_extensions)

    @property
    def validation_extensions_certified(self) -> bool:
        return bool(
            self.validation_extensions
            and all(extension.certified for extension in self.validation_extensions)
        )

    @property
    def validation_tail_budget_certified(self) -> bool:
        return bool(
            self.validation_extensions
            and all(extension.tail_budget_certified for extension in self.validation_extensions)
        )

    @property
    def validation_incremental_tail_bounds(self) -> tuple[float, ...]:
        return tuple(extension.added_incremental_tail_bound for extension in self.validation_extensions)

    @property
    def validation_total_incremental_tail_bound(self) -> float:
        if not self.validation_tail_budget_certified:
            return float("inf")
        return float(sum(self.validation_incremental_tail_bounds))

    @property
    def validation_incremental_segment_counts(self) -> tuple[int, ...]:
        counts: list[int] = []
        for extension in self.validation_extensions:
            previous_domains = extension.previous_schedule.prefix.domain_covers
            if not previous_domains:
                counts.append(extension.added_domain.segment_count)
            else:
                counts.append(extension.added_domain.segment_count_outside(previous_domains[-1].compact_interval))
        return tuple(counts)

    @property
    def validation_max_step_radius_ratios(self) -> tuple[float, ...]:
        ratios: list[float] = []
        for extension in self.validation_extensions:
            previous_domains = extension.previous_schedule.prefix.domain_covers
            if not previous_domains:
                ratios.append(extension.added_domain.max_step_radius_ratio)
            else:
                ratios.append(
                    extension.added_domain.max_step_radius_ratio_outside(previous_domains[-1].compact_interval)
                )
        return tuple(ratios)

    @property
    def max_validation_step_radius_ratio(self) -> float:
        ratios = self.validation_max_step_radius_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def validation_max_state_sup_bounds(self) -> tuple[float, ...]:
        bounds: list[float] = []
        for extension in self.validation_extensions:
            previous_domains = extension.previous_schedule.prefix.domain_covers
            if not previous_domains:
                state_bounds = [
                    bound
                    for cover in extension.added_domain.covers
                    for bound in cover.segment_state_sup_bounds
                    if np.isfinite(bound)
                ]
                bounds.append(float(max(state_bounds)) if state_bounds else float("inf"))
            else:
                bounds.append(
                    extension.added_domain.max_state_sup_bound_outside(previous_domains[-1].compact_interval)
                )
        return tuple(bounds)

    @staticmethod
    def _growth_ratios(previous: float, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or not np.isfinite(previous) or previous <= 0.0:
            return ()
        ratios: list[float] = []
        last = float(previous)
        for value in values:
            if not np.isfinite(value):
                return ()
            ratios.append(float(value / last))
            last = float(value)
        return tuple(ratios)

    @property
    def validation_segment_growth_ratios(self) -> tuple[float, ...]:
        training_counts = self.training_chain.added_incremental_segment_counts
        if not training_counts:
            return ()
        return self._growth_ratios(
            float(training_counts[-1]),
            tuple(float(count) for count in self.validation_incremental_segment_counts),
        )

    @property
    def max_validation_segment_growth_ratio(self) -> float:
        ratios = self.validation_segment_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def validation_state_sup_growth_ratios(self) -> tuple[float, ...]:
        training_bounds = self.training_chain.added_max_state_sup_bounds
        if not training_bounds:
            return ()
        return self._growth_ratios(training_bounds[-1], self.validation_max_state_sup_bounds)

    @property
    def max_validation_state_sup_growth_ratio(self) -> float:
        ratios = self.validation_state_sup_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def validation_cauchy_tail_denominator_factors(self) -> tuple[float, ...]:
        factors: list[float] = []
        for ratio in self.validation_max_step_radius_ratios:
            if not np.isfinite(ratio) or ratio < 0.0 or ratio >= 1.0:
                factors.append(float("inf"))
            else:
                factors.append(float(1.0 / (1.0 - ratio)))
        return tuple(factors)

    @property
    def validation_cauchy_tail_denominator_growth_ratios(self) -> tuple[float, ...]:
        training_factors = self.training_chain.added_cauchy_tail_denominator_factors
        if not training_factors:
            return ()
        return self._growth_ratios(training_factors[-1], self.validation_cauchy_tail_denominator_factors)

    @property
    def max_validation_cauchy_tail_denominator_growth_ratio(self) -> float:
        ratios = self.validation_cauchy_tail_denominator_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def validation_retained_orders(self) -> tuple[int, ...]:
        orders: list[int] = []
        for extension in self.validation_extensions:
            min_order = extension.added_min_retained_order
            max_order = extension.added_max_retained_order
            if min_order < 0 or min_order != max_order:
                return ()
            orders.append(min_order)
        return tuple(orders)

    @property
    def validation_retained_order_increments(self) -> tuple[int, ...]:
        training_orders = self.training_chain.uniform_added_retained_orders
        validation_orders = self.validation_retained_orders
        if not training_orders or not validation_orders:
            return ()
        orders = (training_orders[-1],) + validation_orders
        return tuple(current - previous for previous, current in zip(orders, orders[1:]))

    @property
    def validation_retained_order_acceleration_matches(self) -> bool:
        increments = self.validation_retained_order_increments
        if not increments:
            return False
        expected = tuple(
            self.accelerated_certificate.initial_retained_order_increment
            + index * self.accelerated_certificate.retained_order_increment_growth
            for index in range(len(increments))
        )
        return increments == expected

    @property
    def validation_diagnostics_within_envelope(self) -> bool:
        certificate = self.accelerated_certificate
        return bool(
            certificate.accelerated_global_summability_certified
            and self.validation_connected
            and self.validation_extensions_certified
            and self.validation_retained_order_acceleration_matches
            and self.max_validation_segment_growth_ratio <= certificate.max_future_segment_growth_ratio
            and self.max_validation_state_sup_growth_ratio <= certificate.max_future_state_sup_growth_ratio
            and self.max_validation_cauchy_tail_denominator_growth_ratio
            <= certificate.max_future_denominator_growth_ratio
            and self.max_validation_step_radius_ratio <= certificate.max_future_step_radius_ratio
        )

    @property
    def validation_tail_growth_ratios(self) -> tuple[float, ...]:
        training_tails = self.training_chain.added_incremental_tail_bounds
        if not training_tails:
            return ()
        return self._growth_ratios(training_tails[-1], self.validation_incremental_tail_bounds)

    @property
    def max_validation_tail_growth_ratio(self) -> float:
        ratios = self.validation_tail_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def validation_cauchy_majorant_transition_factor_bounds(self) -> tuple[float, ...]:
        if not self.validation_retained_order_acceleration_matches:
            return ()
        factors: list[float] = []
        for segment_growth, state_growth, denominator_growth, step_radius_ratio, order_increment in zip(
            self.validation_segment_growth_ratios,
            self.validation_state_sup_growth_ratios,
            self.validation_cauchy_tail_denominator_growth_ratios,
            self.validation_max_step_radius_ratios,
            self.validation_retained_order_increments,
        ):
            components = (
                segment_growth,
                state_growth,
                denominator_growth,
                step_radius_ratio,
            )
            if not all(np.isfinite(component) for component in components):
                return ()
            factors.append(
                float(
                    segment_growth
                    * state_growth
                    * denominator_growth
                    * step_radius_ratio**order_increment
                )
            )
        return tuple(factors)

    @property
    def validation_tail_transfer_growth_factors(self) -> tuple[float, ...]:
        structural_factors = self.validation_cauchy_majorant_transition_factor_bounds
        tail_ratios = self.validation_tail_growth_ratios
        if len(structural_factors) != len(tail_ratios):
            return ()
        factors: list[float] = []
        for tail_ratio, structural_factor in zip(tail_ratios, structural_factors):
            if not np.isfinite(tail_ratio) or not np.isfinite(structural_factor) or structural_factor <= 0.0:
                return ()
            factors.append(float(tail_ratio / structural_factor))
        return tuple(factors)

    @property
    def max_validation_tail_transfer_growth_factor(self) -> float:
        factors = self.validation_tail_transfer_growth_factors
        return float(max(factors)) if factors else float("inf")

    @property
    def validation_tail_transfer_growth_ratios(self) -> tuple[float, ...]:
        training_factors = self.training_chain.tail_transfer_growth_factors
        if not training_factors:
            return ()
        return self._growth_ratios(training_factors[-1], self.validation_tail_transfer_growth_factors)

    @property
    def max_validation_tail_transfer_growth_ratio(self) -> float:
        ratios = self.validation_tail_transfer_growth_ratios
        return float(max(ratios)) if ratios else float("inf")

    @property
    def projected_tail_transfer_growth_bounds(self) -> tuple[float, ...]:
        training_factors = self.training_chain.tail_transfer_growth_factors
        growth_ratio = self.accelerated_certificate.max_future_tail_transfer_growth_ratio
        if not training_factors or not np.isfinite(growth_ratio):
            return ()
        bounds: list[float] = []
        current = float(training_factors[-1])
        for _ in self.validation_extensions:
            current *= growth_ratio
            bounds.append(float(current))
        return tuple(bounds)

    @property
    def validation_tail_transfer_within_projected_growth_bounds(self) -> bool:
        projected = self.projected_tail_transfer_growth_bounds
        validation = self.validation_tail_transfer_growth_factors
        return bool(
            projected
            and len(projected) == len(validation)
            and all(value <= bound for value, bound in zip(validation, projected))
        )

    @property
    def projected_validation_tail_ratio_bounds(self) -> tuple[float, ...]:
        return self.accelerated_certificate.future_tail_ratio_bounds(self.validation_extension_count)

    @property
    def projected_validation_tail_bounds(self) -> tuple[float, ...]:
        if self.accelerated_certificate.chain != self.training_chain:
            return ()
        return self.accelerated_certificate.future_tail_bounds(self.validation_extension_count)

    @property
    def projected_validation_tail_sum_bound(self) -> float:
        bounds = self.projected_validation_tail_bounds
        return float(sum(bounds)) if bounds else float("inf")

    @property
    def validation_tail_growth_within_projected_accelerated_bounds(self) -> bool:
        ratio_bounds = self.projected_validation_tail_ratio_bounds
        tail_ratios = self.validation_tail_growth_ratios
        return bool(
            ratio_bounds
            and len(ratio_bounds) == len(tail_ratios)
            and all(tail_ratio <= bound for tail_ratio, bound in zip(tail_ratios, ratio_bounds))
        )

    @property
    def validation_tail_sum_within_projected_accelerated_bounds(self) -> bool:
        bound = self.projected_validation_tail_sum_bound
        return bool(
            self.validation_tail_budget_certified
            and np.isfinite(bound)
            and self.validation_total_incremental_tail_bound <= bound
        )

    @property
    def holdout_certified(self) -> bool:
        return bool(
            self.validation_diagnostics_within_envelope
            and self.validation_tail_transfer_within_projected_growth_bounds
            and self.validation_tail_growth_within_projected_accelerated_bounds
            and self.validation_tail_sum_within_projected_accelerated_bounds
        )

    @property
    def global_domain_certified(self) -> bool:
        return False

    @property
    def missing_global_induction_reason(self) -> str | None:
        if not self.holdout_certified:
            return "the accelerated finite-prefix envelope does not cover the held-out shells"
        return (
            "accelerated finite-prefix envelope covers the held-out shells, but no induction proof "
            "shows that the same accelerated envelope persists for every later shell"
        )


@dataclass(frozen=True)
class CompactifiedSundmanAcceleratedFutureEnvelopeCrossValidationCertificate:
    """Validate accelerated finite-prefix envelopes across multiple train/holdout splits."""

    chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate
    holdouts: tuple[CompactifiedSundmanAcceleratedFutureEnvelopeHoldoutCertificate, ...]

    @property
    def split_count(self) -> int:
        return len(self.holdouts)

    @property
    def training_extension_counts(self) -> tuple[int, ...]:
        return tuple(holdout.training_chain.extension_count for holdout in self.holdouts)

    @property
    def validation_extension_counts(self) -> tuple[int, ...]:
        return tuple(holdout.validation_extension_count for holdout in self.holdouts)

    @property
    def holdout_certified_flags(self) -> tuple[bool, ...]:
        return tuple(holdout.holdout_certified for holdout in self.holdouts)

    @property
    def certified_split_count(self) -> int:
        return sum(1 for certified in self.holdout_certified_flags if certified)

    @property
    def all_holdouts_certified(self) -> bool:
        return bool(self.holdouts and all(self.holdout_certified_flags))

    @property
    def max_projected_validation_tail_sum_bound(self) -> float:
        bounds = [
            holdout.projected_validation_tail_sum_bound
            for holdout in self.holdouts
            if np.isfinite(holdout.projected_validation_tail_sum_bound)
        ]
        return float(max(bounds)) if bounds else float("inf")

    @property
    def max_validation_total_incremental_tail_bound(self) -> float:
        bounds = [
            holdout.validation_total_incremental_tail_bound
            for holdout in self.holdouts
            if np.isfinite(holdout.validation_total_incremental_tail_bound)
        ]
        return float(max(bounds)) if bounds else float("inf")

    @property
    def global_domain_certified(self) -> bool:
        return False

    @property
    def missing_global_induction_reason(self) -> str | None:
        if not self.all_holdouts_certified:
            return "at least one accelerated finite-prefix holdout split is not certified"
        return (
            "accelerated finite-prefix envelopes validate all requested finite holdout splits, "
            "but no induction proof shows that the same envelopes persist for every later shell"
        )


@dataclass(frozen=True)
class CompactifiedSundmanAtlasStep:
    """One compactified-Sundman Taylor chart in an atlas continuation."""

    start_compact_parameter: float
    end_compact_parameter: float
    chart: CompactifiedSundmanTaylorSolution
    residual_certificate: CompactifiedSundmanResidualCertificate
    truncation_indicator: float
    tail_certificate: CompactifiedSundmanTailCertificate | None = None

    @property
    def compact_step(self) -> float:
        return float(self.end_compact_parameter - self.start_compact_parameter)

    @property
    def physical_time_step(self) -> float:
        return self.chart.physical_time_delta_at_w(self.end_compact_parameter)

    @property
    def sundman_time_step(self) -> float:
        return self.chart.sundman_time_delta_at_w(self.end_compact_parameter)

    @property
    def certified(self) -> bool:
        return bool(
            self.residual_certificate.certified
            and self.chart.center == self.start_compact_parameter
            and abs(self.compact_step) < self.chart.analytic_radius
        )

    @property
    def tail_certified(self) -> bool:
        return bool(self.tail_certificate is not None and self.tail_certificate.is_nontrivial)

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and self.tail_certified)


@dataclass(frozen=True)
class CompactifiedSundmanAtlasSolution:
    """Projected states produced by compactified-Sundman atlas continuation."""

    masses: Array
    sundman_rate: float
    distance_power: float
    compact_parameters: Array
    states: Array
    physical_times: Array
    sundman_times: Array
    steps: tuple[CompactifiedSundmanAtlasStep, ...]

    @property
    def final_state(self) -> Array:
        return self.states[-1]

    @property
    def final_physical_time(self) -> float:
        return float(self.physical_times[-1])

    @property
    def final_sundman_time(self) -> float:
        return float(self.sundman_times[-1])

    @property
    def certified(self) -> bool:
        return bool(all(step.certified for step in self.steps))

    @property
    def tail_certified(self) -> bool:
        return bool(self.steps and all(step.tail_certified for step in self.steps))

    @property
    def chain_certified(self) -> bool:
        return _compactified_sundman_point_atlas_chain_certified(
            self.compact_parameters,
            self.states,
            self.physical_times,
            self.sundman_times,
            self.steps,
        )

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and self.tail_certified and self.chain_certified)

    @property
    def cauchy_cover_certificate(self) -> CompactifiedSundmanCauchyCoverCertificate:
        return certify_compactified_sundman_cauchy_cover(self.steps)

    @property
    def cauchy_cover_certified(self) -> bool:
        return self.cauchy_cover_certificate.certified

    @property
    def total_physical_time_delta(self) -> float:
        return self.final_physical_time

    @property
    def total_sundman_time_delta(self) -> float:
        return self.final_sundman_time - self.sundman_times[0]


@dataclass(frozen=True)
class IntervalCompactifiedSundmanAtlasStep:
    """One set-valued compactified-Sundman chart in an interval atlas."""

    start_compact_parameter: float
    end_compact_parameter: float
    chart: IntervalCompactifiedSundmanTaylorSolution
    start_state_interval: Array
    end_state_interval: Array
    physical_time_step_interval: FloatInterval
    factor_interval: FloatInterval
    sundman_time_step: float
    residual_certificate: IntervalCompactifiedSundmanResidualCertificate
    angular_momentum_certificate: AngularMomentumConservationCertificate
    energy_certificate: EnergyConservationCertificate
    linear_momentum_certificate: LinearMomentumConservationCertificate
    center_of_mass_certificate: CenterOfMassMotionCertificate
    tail_certificate: CompactifiedSundmanTailCertificate | None = None

    @property
    def compact_step(self) -> float:
        return float(self.end_compact_parameter - self.start_compact_parameter)

    @property
    def certified(self) -> bool:
        return bool(
            self.residual_certificate.certified
            and self.angular_momentum_certificate.certified
            and self.energy_certificate.certified
            and self.linear_momentum_certificate.certified
            and self.center_of_mass_certificate.certified
            and self.time_monotone_certified
            and self.chart.center == self.start_compact_parameter
            and abs(self.compact_step) < self.chart.analytic_radius
        )

    @property
    def time_monotone_certified(self) -> bool:
        return bool(self.factor_interval.lower > 0.0)

    @property
    def tail_certified(self) -> bool:
        return bool(self.tail_certificate is not None and self.tail_certificate.is_nontrivial)

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and self.tail_certified)

    def start_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.start_state_interval, np.asarray(state, dtype=float))

    def end_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.end_state_interval, np.asarray(state, dtype=float))


@dataclass(frozen=True)
class IntervalCompactifiedSundmanAtlasSolution:
    """Set-valued compactified-Sundman atlas built from interval charts."""

    masses: Array
    sundman_rate: float
    distance_power: float
    compact_parameters: Array
    state_intervals: tuple[Array, ...]
    physical_time_intervals: tuple[FloatInterval, ...]
    sundman_times: Array
    steps: tuple[IntervalCompactifiedSundmanAtlasStep, ...]

    @property
    def final_state_interval(self) -> Array:
        return self.state_intervals[-1]

    @property
    def final_physical_time_interval(self) -> FloatInterval:
        return self.physical_time_intervals[-1]

    @property
    def final_sundman_time(self) -> float:
        return float(self.sundman_times[-1])

    @property
    def certified(self) -> bool:
        return bool(all(step.certified for step in self.steps))

    @property
    def tail_certified(self) -> bool:
        return bool(self.steps and all(step.tail_certified for step in self.steps))

    @property
    def chain_certified(self) -> bool:
        return _compactified_sundman_interval_atlas_chain_certified(
            self.compact_parameters,
            self.state_intervals,
            self.physical_time_intervals,
            self.sundman_times,
            self.steps,
        )

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and self.tail_certified and self.chain_certified)

    @property
    def cauchy_cover_certificate(self) -> CompactifiedSundmanCauchyCoverCertificate:
        return certify_compactified_sundman_cauchy_cover(self.steps)

    @property
    def cauchy_cover_certified(self) -> bool:
        return self.cauchy_cover_certificate.certified

    @property
    def local_tail_bound(self) -> float:
        return float(
            sum(0.0 if step.tail_certificate is None else step.tail_certificate.tail_bound for step in self.steps)
        )

    @property
    def max_step_tail_bound(self) -> float:
        bounds = [
            step.tail_certificate.tail_bound
            for step in self.steps
            if step.tail_certificate is not None
        ]
        return float(max(bounds)) if bounds else 0.0

    @property
    def total_physical_time_interval(self) -> FloatInterval:
        return self.final_physical_time_interval

    @property
    def total_sundman_time_delta(self) -> float:
        return self.final_sundman_time - self.sundman_times[0]

    def final_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.final_state_interval, np.asarray(state, dtype=float))


@dataclass(frozen=True)
class CompactifiedSundmanPhysicalTimeTargetCertificate:
    """Certified physical-time bracket in compactified Sundman parameter."""

    target_time: float
    chart_start_compact_parameter: float
    compact_parameter_interval: FloatInterval
    start_time_interval: FloatInterval
    local_time_at_lower: FloatInterval
    local_time_at_upper: FloatInterval
    global_time_at_lower: FloatInterval
    global_time_at_upper: FloatInterval
    factor_interval: FloatInterval
    bisections: int
    time_tail_bound: float = 0.0

    @property
    def certified(self) -> bool:
        return bool(
            self.factor_interval.lower > 0.0
            and self.global_time_at_lower.upper <= self.target_time <= self.global_time_at_upper.lower
        )

    @property
    def width(self) -> float:
        return self.compact_parameter_interval.upper - self.compact_parameter_interval.lower


@dataclass(frozen=True)
class IntervalCompactifiedSundmanTimeTargetSolution:
    """Set-valued compactified-Sundman continuation to a physical-time target."""

    masses: Array
    target_time: float
    sundman_rate: float
    distance_power: float
    compact_parameters: Array
    physical_time_intervals: tuple[FloatInterval, ...]
    sundman_times: Array
    state_intervals: tuple[Array, ...]
    steps: tuple[IntervalCompactifiedSundmanAtlasStep, ...]
    target_start_state_interval: Array
    target_certificate: CompactifiedSundmanPhysicalTimeTargetCertificate
    target_state_interval: Array
    target_residual_certificate: IntervalCompactifiedSundmanResidualCertificate
    target_angular_momentum_certificate: AngularMomentumConservationCertificate
    target_energy_certificate: EnergyConservationCertificate
    target_linear_momentum_certificate: LinearMomentumConservationCertificate
    target_center_of_mass_certificate: CenterOfMassMotionCertificate
    target_tail_certificate: CompactifiedSundmanTailCertificate | None = None
    triple_collision_exclusion_certificate: TripleCollisionExclusionCertificate | None = None

    @property
    def target_compact_parameter_interval(self) -> FloatInterval:
        return self.target_certificate.compact_parameter_interval

    @property
    def certified(self) -> bool:
        return bool(
            self.target_certificate.certified
            and self.target_residual_certificate.certified
            and self.target_angular_momentum_certificate.certified
            and self.target_energy_certificate.certified
            and self.target_linear_momentum_certificate.certified
            and self.target_center_of_mass_certificate.certified
            and all(step.certified for step in self.steps)
        )

    @property
    def tail_certified(self) -> bool:
        return bool(
            self.target_tail_certificate is not None
            and self.target_tail_certificate.is_nontrivial
            and all(step.tail_certified for step in self.steps)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and self.tail_certified and self.chain_certified)

    @property
    def chain_certified(self) -> bool:
        return _compactified_sundman_time_target_chain_certified(self)

    @property
    def cauchy_cover_certificate(self) -> CompactifiedSundmanCauchyCoverCertificate:
        return certify_compactified_sundman_cauchy_cover(
            self.steps,
            target_interval=self.target_certificate.compact_parameter_interval,
            target_tail_certificate=self.target_tail_certificate,
            target_center=self.target_certificate.chart_start_compact_parameter,
        )

    @property
    def cauchy_cover_certified(self) -> bool:
        return self.cauchy_cover_certificate.certified

    @property
    def triple_collision_excluded(self) -> bool:
        return bool(
            self.triple_collision_exclusion_certificate is not None
            and self.triple_collision_exclusion_certificate.certified
        )

    @property
    def triple_collision_status(self) -> str:
        if self.triple_collision_exclusion_certificate is None:
            return "missing"
        return self.triple_collision_exclusion_certificate.status

    @property
    def triple_collision_exclusion_reason(self) -> str | None:
        if self.triple_collision_exclusion_certificate is None:
            return None
        return self.triple_collision_exclusion_certificate.reason

    @property
    def triple_collision_undecided(self) -> bool:
        return self.triple_collision_status == "undecided"

    @property
    def angular_momentum_certified_step_count(self) -> int:
        return sum(step.angular_momentum_certificate.certified for step in self.steps) + int(
            self.target_angular_momentum_certificate.certified
        )

    @property
    def energy_certified_step_count(self) -> int:
        return sum(step.energy_certificate.certified for step in self.steps) + int(
            self.target_energy_certificate.certified
        )

    @property
    def linear_momentum_certified_step_count(self) -> int:
        return sum(step.linear_momentum_certificate.certified for step in self.steps) + int(
            self.target_linear_momentum_certificate.certified
        )

    @property
    def center_of_mass_certified_step_count(self) -> int:
        return sum(step.center_of_mass_certificate.certified for step in self.steps) + int(
            self.target_center_of_mass_certificate.certified
        )

    @property
    def local_tail_bound(self) -> float:
        total = 0.0
        for step in self.steps:
            if step.tail_certificate is not None:
                total += step.tail_certificate.tail_bound
        if self.target_tail_certificate is not None:
            total += self.target_tail_certificate.tail_bound
        return float(total)

    @property
    def max_step_tail_bound(self) -> float:
        bounds = [
            step.tail_certificate.tail_bound
            for step in self.steps
            if step.tail_certificate is not None
        ]
        if self.target_tail_certificate is not None:
            bounds.append(self.target_tail_certificate.tail_bound)
        return float(max(bounds)) if bounds else 0.0

    def target_state_contains(self, state: Array) -> bool:
        return interval_array_contains_point(self.target_state_interval, np.asarray(state, dtype=float))


def _compactified_sundman_float_boundaries_match(left: float, right: float) -> bool:
    left = float(left)
    right = float(right)
    if not (np.isfinite(left) and np.isfinite(right)):
        return False
    tolerance = 1.0e-10 * max(1.0, abs(left), abs(right))
    return bool(abs(left - right) <= tolerance)


def _compactified_sundman_float_arrays_match(left: Array, right: Array) -> bool:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        return False
    if left_array.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(left_array))), float(np.max(np.abs(right_array))))
    return bool(
        np.all(np.isfinite(left_array))
        and np.all(np.isfinite(right_array))
        and np.all(np.abs(left_array - right_array) <= 1.0e-10 * scale)
    )


def _compactified_sundman_intervals_match(left: object, right: object) -> bool:
    return bool(_as_interval(left).as_tuple() == _as_interval(right).as_tuple())


def _compactified_sundman_interval_arrays_match(left: Array, right: Array) -> bool:
    left_array = np.asarray(left, dtype=object)
    right_array = np.asarray(right, dtype=object)
    if left_array.shape != right_array.shape:
        return False
    return bool(
        all(
            _compactified_sundman_intervals_match(left_array[index], right_array[index])
            for index in np.ndindex(left_array.shape)
        )
    )


def _compactified_sundman_point_atlas_chain_certified(
    compact_parameters: Array,
    states: Array,
    physical_times: Array,
    sundman_times: Array,
    steps: tuple[CompactifiedSundmanAtlasStep, ...],
) -> bool:
    step_count = len(steps)
    compact_parameters = np.asarray(compact_parameters, dtype=float)
    states = np.asarray(states, dtype=float)
    physical_times = np.asarray(physical_times, dtype=float)
    sundman_times = np.asarray(sundman_times, dtype=float)
    if (
        compact_parameters.shape != (step_count + 1,)
        or physical_times.shape != (step_count + 1,)
        or sundman_times.shape != (step_count + 1,)
        or states.ndim != 2
        or states.shape[0] != step_count + 1
    ):
        return False
    if step_count == 0:
        return True
    if not (
        np.all(np.isfinite(compact_parameters))
        and np.all(np.isfinite(physical_times))
        and np.all(np.isfinite(sundman_times))
        and np.all(np.isfinite(states))
    ):
        return False
    for index, step in enumerate(steps):
        if not (
            _compactified_sundman_float_boundaries_match(
                compact_parameters[index],
                step.start_compact_parameter,
            )
            and _compactified_sundman_float_boundaries_match(
                compact_parameters[index + 1],
                step.end_compact_parameter,
            )
            and _compactified_sundman_float_arrays_match(
                states[index],
                step.chart.state_at_w(step.start_compact_parameter),
            )
            and _compactified_sundman_float_arrays_match(
                states[index + 1],
                step.chart.state_at_w(step.end_compact_parameter),
            )
            and _compactified_sundman_float_boundaries_match(
                physical_times[index] + step.physical_time_step,
                physical_times[index + 1],
            )
            and _compactified_sundman_float_boundaries_match(
                sundman_times[index] + step.sundman_time_step,
                sundman_times[index + 1],
            )
        ):
            return False
    return True


def _compactified_sundman_interval_atlas_chain_certified(
    compact_parameters: Array,
    state_intervals: tuple[Array, ...],
    physical_time_intervals: tuple[FloatInterval, ...],
    sundman_times: Array,
    steps: tuple[IntervalCompactifiedSundmanAtlasStep, ...],
) -> bool:
    step_count = len(steps)
    compact_parameters = np.asarray(compact_parameters, dtype=float)
    sundman_times = np.asarray(sundman_times, dtype=float)
    if (
        compact_parameters.shape != (step_count + 1,)
        or sundman_times.shape != (step_count + 1,)
        or len(state_intervals) != step_count + 1
        or len(physical_time_intervals) != step_count + 1
    ):
        return False
    if step_count == 0:
        return True
    if not (np.all(np.isfinite(compact_parameters)) and np.all(np.isfinite(sundman_times))):
        return False
    for index, step in enumerate(steps):
        if not (
            _compactified_sundman_float_boundaries_match(
                compact_parameters[index],
                step.start_compact_parameter,
            )
            and _compactified_sundman_float_boundaries_match(
                compact_parameters[index + 1],
                step.end_compact_parameter,
            )
            and _compactified_sundman_interval_arrays_match(
                state_intervals[index],
                step.start_state_interval,
            )
            and _compactified_sundman_interval_arrays_match(
                state_intervals[index + 1],
                step.end_state_interval,
            )
            and _compactified_sundman_intervals_match(
                _as_interval(physical_time_intervals[index]) + step.physical_time_step_interval,
                physical_time_intervals[index + 1],
            )
            and _compactified_sundman_float_boundaries_match(
                sundman_times[index] + step.sundman_time_step,
                sundman_times[index + 1],
            )
        ):
            return False
    return True


def _compactified_sundman_time_target_chain_certified(
    solution: IntervalCompactifiedSundmanTimeTargetSolution,
) -> bool:
    if not _compactified_sundman_interval_atlas_chain_certified(
        solution.compact_parameters,
        solution.state_intervals,
        solution.physical_time_intervals,
        solution.sundman_times,
        solution.steps,
    ):
        return False
    if (
        len(solution.compact_parameters) == 0
        or not solution.state_intervals
        or not solution.physical_time_intervals
    ):
        return False
    return bool(
        _compactified_sundman_float_boundaries_match(
            solution.compact_parameters[-1],
            solution.target_certificate.chart_start_compact_parameter,
        )
        and _compactified_sundman_interval_arrays_match(
            solution.state_intervals[-1],
            solution.target_start_state_interval,
        )
        and _compactified_sundman_intervals_match(
            solution.physical_time_intervals[-1],
            solution.target_certificate.start_time_interval,
        )
    )


_COMPACT_SUNDMAN_CAUCHY_SOURCES = frozenset(
    {
        "compactified_sundman_cauchy_majorant",
        "compactified_sundman_interval_cauchy_majorant",
    }
)


def certify_compactified_sundman_cauchy_cover(
    steps: tuple[CompactifiedSundmanAtlasStep | IntervalCompactifiedSundmanAtlasStep, ...],
    *,
    target_interval: FloatInterval | None = None,
    target_tail_certificate: CompactifiedSundmanTailCertificate | None = None,
    target_center: float | None = None,
) -> CompactifiedSundmanCauchyCoverCertificate:
    """Certify that accepted compact-Sundman segments lie inside Cauchy disks."""

    segment_starts: list[float] = []
    segment_ends: list[float] = []
    disk_centers: list[float] = []
    disk_radii: list[float] = []
    segment_tail_bounds: list[float] = []
    segment_retained_orders: list[int] = []
    segment_state_sup_bounds: list[float] = []
    coefficient_sources: list[str] = []
    certificate_nontrivial: list[bool] = []

    for step in steps:
        _append_compact_sundman_cauchy_cover_segment(
            segment_starts,
            segment_ends,
            disk_centers,
            disk_radii,
            segment_tail_bounds,
            segment_retained_orders,
            segment_state_sup_bounds,
            coefficient_sources,
            certificate_nontrivial,
            start=step.start_compact_parameter,
            end=step.end_compact_parameter,
            certificate=step.tail_certificate,
        )

    if target_interval is not None:
        if target_center is None:
            raise ValueError("target_center is required when target_interval is provided")
        target_interval = _as_interval(target_interval)
        target_center = float(target_center)
        lower_distance = abs(target_interval.lower - target_center)
        upper_distance = abs(target_interval.upper - target_center)
        target_end = target_interval.lower if lower_distance >= upper_distance else target_interval.upper
        _append_compact_sundman_cauchy_cover_segment(
            segment_starts,
            segment_ends,
            disk_centers,
            disk_radii,
            segment_tail_bounds,
            segment_retained_orders,
            segment_state_sup_bounds,
            coefficient_sources,
            certificate_nontrivial,
            start=target_center,
            end=target_end,
            certificate=target_tail_certificate,
        )

    if segment_starts:
        start_compact_parameter = segment_starts[0]
        end_compact_parameter = segment_ends[-1]
    else:
        start_compact_parameter = float("nan")
        end_compact_parameter = float("nan")

    return CompactifiedSundmanCauchyCoverCertificate(
        start_compact_parameter=float(start_compact_parameter),
        end_compact_parameter=float(end_compact_parameter),
        segment_starts=tuple(segment_starts),
        segment_ends=tuple(segment_ends),
        disk_centers=tuple(disk_centers),
        disk_radii=tuple(disk_radii),
        segment_tail_bounds=tuple(segment_tail_bounds),
        segment_retained_orders=tuple(segment_retained_orders),
        segment_state_sup_bounds=tuple(segment_state_sup_bounds),
        coefficient_sources=tuple(coefficient_sources),
        certificate_nontrivial=tuple(certificate_nontrivial),
    )


def certify_compactified_sundman_compact_domain_cover(
    covers: tuple[CompactifiedSundmanCauchyCoverCertificate, ...],
    *,
    compact_interval: FloatInterval,
) -> CompactifiedSundmanCompactDomainCoverCertificate:
    """Certify that finite Cauchy-covered paths exhaust a compact ``w`` interval."""

    return CompactifiedSundmanCompactDomainCoverCertificate(
        compact_interval=_as_interval(compact_interval),
        covers=tuple(covers),
    )


def certify_compactified_sundman_compact_domain_exhaustion_prefix(
    domain_covers: tuple[CompactifiedSundmanCompactDomainCoverCertificate, ...],
) -> CompactifiedSundmanCompactDomainExhaustionPrefixCertificate:
    """Certify a finite nested prefix of compact-Sundman domain covers."""

    return CompactifiedSundmanCompactDomainExhaustionPrefixCertificate(domain_covers=tuple(domain_covers))


def certify_compactified_sundman_geometric_exhaustion_schedule(
    prefix: CompactifiedSundmanCompactDomainExhaustionPrefixCertificate,
    *,
    initial_boundary_margin: float,
    contraction: float,
) -> CompactifiedSundmanGeometricExhaustionScheduleCertificate:
    """Check a finite prefix against a geometric endpoint-exhaustion schedule."""

    return CompactifiedSundmanGeometricExhaustionScheduleCertificate(
        prefix=prefix,
        initial_boundary_margin=float(initial_boundary_margin),
        contraction=float(contraction),
    )


def certify_compactified_sundman_geometric_schedule_recurrence(
    schedule: CompactifiedSundmanGeometricExhaustionScheduleCertificate,
    *,
    one_step_extension_template_certified: bool = False,
    witness_source: str = "manual_geometric_schedule_recurrence",
    certified_schedule_initial_boundary_margin: float | None = None,
    certified_schedule_contraction: float | None = None,
    certified_extension_induction_start_index: float | int | None = None,
) -> CompactifiedSundmanGeometricScheduleRecurrenceCertificate:
    """Record induction data for the all-future geometric domain schedule."""

    return CompactifiedSundmanGeometricScheduleRecurrenceCertificate(
        schedule=schedule,
        one_step_extension_template_certified=bool(one_step_extension_template_certified),
        witness_source=str(witness_source),
        certified_schedule_initial_boundary_margin=certified_schedule_initial_boundary_margin,
        certified_schedule_contraction=certified_schedule_contraction,
        certified_extension_induction_start_index=certified_extension_induction_start_index,
    )


def certify_compactified_sundman_recurrence_template(
    accelerated_certificate: CompactifiedSundmanAcceleratedGlobalSummabilityCertificate,
    *,
    structural_envelope_form_certified: bool = False,
    structural_transition_template_certified: bool = False,
    tail_transfer_form_certified: bool = False,
    tail_transfer_growth_template_certified: bool = False,
    retained_order_arithmetic_schedule_certified: bool = False,
    retained_order_growth_template_certified: bool = False,
    witness_source: str = "manual_recurrence_template",
) -> CompactifiedSundmanRecurrenceTemplateCertificate:
    """Record nonnumeric recurrence-template evidence for future shells."""

    return CompactifiedSundmanRecurrenceTemplateCertificate(
        accelerated_certificate=accelerated_certificate,
        structural_envelope_form_certified=bool(structural_envelope_form_certified),
        structural_transition_template_certified=bool(
            structural_transition_template_certified
        ),
        tail_transfer_form_certified=bool(tail_transfer_form_certified),
        tail_transfer_growth_template_certified=bool(
            tail_transfer_growth_template_certified
        ),
        retained_order_arithmetic_schedule_certified=bool(
            retained_order_arithmetic_schedule_certified
        ),
        retained_order_growth_template_certified=bool(
            retained_order_growth_template_certified
        ),
        witness_source=str(witness_source),
    )


def certify_compactified_sundman_geometric_exhaustion_extension(
    previous_schedule: CompactifiedSundmanGeometricExhaustionScheduleCertificate,
    added_domain: CompactifiedSundmanCompactDomainCoverCertificate,
) -> CompactifiedSundmanGeometricExhaustionExtensionCertificate:
    """Certify that one added domain extends the geometric exhaustion prefix."""

    extended_prefix = certify_compactified_sundman_compact_domain_exhaustion_prefix(
        (*previous_schedule.prefix.domain_covers, added_domain)
    )
    extended_schedule = certify_compactified_sundman_geometric_exhaustion_schedule(
        extended_prefix,
        initial_boundary_margin=previous_schedule.initial_boundary_margin,
        contraction=previous_schedule.contraction,
    )
    return CompactifiedSundmanGeometricExhaustionExtensionCertificate(
        previous_schedule=previous_schedule,
        added_domain=added_domain,
        extended_schedule=extended_schedule,
    )


def certify_compactified_sundman_geometric_exhaustion_extension_chain(
    initial_schedule: CompactifiedSundmanGeometricExhaustionScheduleCertificate,
    extensions: tuple[CompactifiedSundmanGeometricExhaustionExtensionCertificate, ...],
) -> CompactifiedSundmanGeometricExhaustionExtensionChainCertificate:
    """Certify a finite chain of geometric exhaustion extensions."""

    return CompactifiedSundmanGeometricExhaustionExtensionChainCertificate(
        initial_schedule=initial_schedule,
        extensions=tuple(extensions),
    )


def certify_compactified_sundman_conditional_global_summability(
    chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate,
    *,
    max_future_segment_growth_ratio: float,
    max_future_state_sup_growth_ratio: float,
    max_future_denominator_growth_ratio: float,
    max_future_step_radius_ratio: float,
    retained_order_increment: int | None = None,
    envelope_source: str = "manual",
) -> CompactifiedSundmanConditionalGlobalSummabilityCertificate:
    """Check that proposed future-shell envelopes would imply summability."""

    if retained_order_increment is None:
        retained_order_increment = chain.retained_order_increment
    return CompactifiedSundmanConditionalGlobalSummabilityCertificate(
        chain=chain,
        max_future_segment_growth_ratio=float(max_future_segment_growth_ratio),
        max_future_state_sup_growth_ratio=float(max_future_state_sup_growth_ratio),
        max_future_denominator_growth_ratio=float(max_future_denominator_growth_ratio),
        max_future_step_radius_ratio=float(max_future_step_radius_ratio),
        retained_order_increment=int(retained_order_increment),
        envelope_source=str(envelope_source),
    )


def certify_compactified_sundman_tail_transfer_global_summability(
    conditional_certificate: CompactifiedSundmanConditionalGlobalSummabilityCertificate,
    *,
    max_future_tail_transfer_growth_factor: float,
    tail_transfer_source: str = "manual",
) -> CompactifiedSundmanTailTransferGlobalSummabilityCertificate:
    """Check summability under a structural envelope plus a tail-transfer bound."""

    return CompactifiedSundmanTailTransferGlobalSummabilityCertificate(
        conditional_certificate=conditional_certificate,
        max_future_tail_transfer_growth_factor=float(max_future_tail_transfer_growth_factor),
        tail_transfer_source=str(tail_transfer_source),
    )


def certify_compactified_sundman_geometric_tail_transfer_global_summability(
    conditional_certificate: CompactifiedSundmanConditionalGlobalSummabilityCertificate,
    *,
    initial_tail_transfer_growth_factor_bound: float,
    max_future_tail_transfer_growth_ratio: float,
    retained_order_increment_growth: int,
    tail_transfer_source: str = "manual",
) -> CompactifiedSundmanGeometricTailTransferGlobalSummabilityCertificate:
    """Check summability when tail-transfer factors have geometric growth."""

    return CompactifiedSundmanGeometricTailTransferGlobalSummabilityCertificate(
        conditional_certificate=conditional_certificate,
        initial_tail_transfer_growth_factor_bound=float(initial_tail_transfer_growth_factor_bound),
        max_future_tail_transfer_growth_ratio=float(max_future_tail_transfer_growth_ratio),
        retained_order_increment_growth=int(retained_order_increment_growth),
        tail_transfer_source=str(tail_transfer_source),
    )


def certify_compactified_sundman_accelerated_global_summability(
    chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate,
    *,
    max_future_segment_growth_ratio: float,
    max_future_state_sup_growth_ratio: float,
    max_future_denominator_growth_ratio: float,
    max_future_step_radius_ratio: float,
    initial_retained_order_increment: int,
    retained_order_increment_growth: int,
    initial_tail_transfer_growth_factor_bound: float = 1.0,
    max_future_tail_transfer_growth_ratio: float = 1.0,
    envelope_source: str = "manual",
) -> CompactifiedSundmanAcceleratedGlobalSummabilityCertificate:
    """Check summability for a future accelerated retained-order schedule."""

    return CompactifiedSundmanAcceleratedGlobalSummabilityCertificate(
        chain=chain,
        max_future_segment_growth_ratio=float(max_future_segment_growth_ratio),
        max_future_state_sup_growth_ratio=float(max_future_state_sup_growth_ratio),
        max_future_denominator_growth_ratio=float(max_future_denominator_growth_ratio),
        max_future_step_radius_ratio=float(max_future_step_radius_ratio),
        initial_retained_order_increment=int(initial_retained_order_increment),
        retained_order_increment_growth=int(retained_order_increment_growth),
        initial_tail_transfer_growth_factor_bound=float(initial_tail_transfer_growth_factor_bound),
        max_future_tail_transfer_growth_ratio=float(max_future_tail_transfer_growth_ratio),
        envelope_source=str(envelope_source),
    )


def certify_compactified_sundman_accelerated_induction_witness(
    accelerated_certificate: CompactifiedSundmanAcceleratedGlobalSummabilityCertificate,
    *,
    future_domain_exhaustion_certified: bool = False,
    future_extension_induction_certified: bool = False,
    future_structural_envelopes_certified: bool = False,
    future_tail_transfer_growth_certified: bool = False,
    future_accelerated_order_schedule_certified: bool = False,
    future_segment_growth_bound_certified: bool = False,
    future_state_sup_growth_bound_certified: bool = False,
    future_denominator_growth_bound_certified: bool = False,
    future_step_radius_bound_certified: bool = False,
    future_initial_tail_transfer_bound_certified: bool = False,
    future_tail_transfer_growth_ratio_certified: bool = False,
    future_initial_retained_order_increment_certified: bool = False,
    future_retained_order_increment_growth_certified: bool = False,
    collision_continuation_certified: bool = False,
    binary_collision_continuation_certified: bool = False,
    triple_collision_continuation_certified: bool = False,
    witness_source: str = "manual",
    certified_schedule_initial_boundary_margin: float | None = None,
    certified_schedule_contraction: float | None = None,
    certified_extension_induction_start_index: int | None = None,
    certified_max_future_segment_growth_ratio: float | None = None,
    certified_max_future_state_sup_growth_ratio: float | None = None,
    certified_max_future_denominator_growth_ratio: float | None = None,
    certified_max_future_step_radius_ratio: float | None = None,
    certified_initial_tail_transfer_growth_factor_bound: float | None = None,
    certified_max_future_tail_transfer_growth_ratio: float | None = None,
    certified_initial_retained_order_increment: int | None = None,
    certified_retained_order_increment_growth: int | None = None,
) -> CompactifiedSundmanAcceleratedInductionWitnessCertificate:
    """Record the universal induction witness needed for a global accelerated series."""

    return CompactifiedSundmanAcceleratedInductionWitnessCertificate(
        accelerated_certificate=accelerated_certificate,
        future_domain_exhaustion_certified=bool(future_domain_exhaustion_certified),
        future_extension_induction_certified=bool(future_extension_induction_certified),
        future_structural_envelopes_certified=bool(future_structural_envelopes_certified),
        future_tail_transfer_growth_certified=bool(future_tail_transfer_growth_certified),
        future_accelerated_order_schedule_certified=bool(future_accelerated_order_schedule_certified),
        future_segment_growth_bound_certified=bool(future_segment_growth_bound_certified),
        future_state_sup_growth_bound_certified=bool(future_state_sup_growth_bound_certified),
        future_denominator_growth_bound_certified=bool(future_denominator_growth_bound_certified),
        future_step_radius_bound_certified=bool(future_step_radius_bound_certified),
        future_initial_tail_transfer_bound_certified=bool(future_initial_tail_transfer_bound_certified),
        future_tail_transfer_growth_ratio_certified=bool(future_tail_transfer_growth_ratio_certified),
        future_initial_retained_order_increment_certified=bool(
            future_initial_retained_order_increment_certified
        ),
        future_retained_order_increment_growth_certified=bool(
            future_retained_order_increment_growth_certified
        ),
        collision_continuation_certified=bool(collision_continuation_certified),
        binary_collision_continuation_certified=bool(binary_collision_continuation_certified),
        triple_collision_continuation_certified=bool(triple_collision_continuation_certified),
        witness_source=str(witness_source),
        certified_schedule_initial_boundary_margin=certified_schedule_initial_boundary_margin,
        certified_schedule_contraction=certified_schedule_contraction,
        certified_extension_induction_start_index=certified_extension_induction_start_index,
        certified_max_future_segment_growth_ratio=certified_max_future_segment_growth_ratio,
        certified_max_future_state_sup_growth_ratio=certified_max_future_state_sup_growth_ratio,
        certified_max_future_denominator_growth_ratio=certified_max_future_denominator_growth_ratio,
        certified_max_future_step_radius_ratio=certified_max_future_step_radius_ratio,
        certified_initial_tail_transfer_growth_factor_bound=certified_initial_tail_transfer_growth_factor_bound,
        certified_max_future_tail_transfer_growth_ratio=certified_max_future_tail_transfer_growth_ratio,
        certified_initial_retained_order_increment=certified_initial_retained_order_increment,
        certified_retained_order_increment_growth=certified_retained_order_increment_growth,
    )


def certify_compactified_sundman_future_shell_induction_closure(
    accelerated_certificate: CompactifiedSundmanAcceleratedGlobalSummabilityCertificate,
    *,
    domain_exhaustion_recurrence_certified: bool = False,
    scheduled_extension_recurrence_certified: bool = False,
    structural_envelope_recurrence_certified: bool = False,
    segment_growth_recurrence_certified: bool = False,
    state_sup_growth_recurrence_certified: bool = False,
    denominator_growth_recurrence_certified: bool = False,
    step_radius_recurrence_certified: bool = False,
    tail_transfer_recurrence_certified: bool = False,
    initial_tail_transfer_recurrence_certified: bool = False,
    tail_transfer_growth_ratio_recurrence_certified: bool = False,
    retained_order_schedule_recurrence_certified: bool = False,
    initial_retained_order_recurrence_certified: bool = False,
    retained_order_growth_recurrence_certified: bool = False,
    witness_source: str = "manual_recurrence_closure",
    certified_schedule_initial_boundary_margin: float | None = None,
    certified_schedule_contraction: float | None = None,
    certified_extension_induction_start_index: int | None = None,
    certified_max_future_segment_growth_ratio: float | None = None,
    certified_max_future_state_sup_growth_ratio: float | None = None,
    certified_max_future_denominator_growth_ratio: float | None = None,
    certified_max_future_step_radius_ratio: float | None = None,
    certified_initial_tail_transfer_growth_factor_bound: float | None = None,
    certified_max_future_tail_transfer_growth_ratio: float | None = None,
    certified_initial_retained_order_increment: int | None = None,
    certified_retained_order_increment_growth: int | None = None,
) -> CompactifiedSundmanFutureShellInductionClosureCertificate:
    """Record recurrence lemmas that would close the accelerated shell induction."""

    return CompactifiedSundmanFutureShellInductionClosureCertificate(
        accelerated_certificate=accelerated_certificate,
        domain_exhaustion_recurrence_certified=bool(domain_exhaustion_recurrence_certified),
        scheduled_extension_recurrence_certified=bool(
            scheduled_extension_recurrence_certified
        ),
        structural_envelope_recurrence_certified=bool(
            structural_envelope_recurrence_certified
        ),
        segment_growth_recurrence_certified=bool(segment_growth_recurrence_certified),
        state_sup_growth_recurrence_certified=bool(state_sup_growth_recurrence_certified),
        denominator_growth_recurrence_certified=bool(denominator_growth_recurrence_certified),
        step_radius_recurrence_certified=bool(step_radius_recurrence_certified),
        tail_transfer_recurrence_certified=bool(tail_transfer_recurrence_certified),
        initial_tail_transfer_recurrence_certified=bool(
            initial_tail_transfer_recurrence_certified
        ),
        tail_transfer_growth_ratio_recurrence_certified=bool(
            tail_transfer_growth_ratio_recurrence_certified
        ),
        retained_order_schedule_recurrence_certified=bool(
            retained_order_schedule_recurrence_certified
        ),
        initial_retained_order_recurrence_certified=bool(
            initial_retained_order_recurrence_certified
        ),
        retained_order_growth_recurrence_certified=bool(
            retained_order_growth_recurrence_certified
        ),
        witness_source=str(witness_source),
        certified_schedule_initial_boundary_margin=certified_schedule_initial_boundary_margin,
        certified_schedule_contraction=certified_schedule_contraction,
        certified_extension_induction_start_index=certified_extension_induction_start_index,
        certified_max_future_segment_growth_ratio=certified_max_future_segment_growth_ratio,
        certified_max_future_state_sup_growth_ratio=certified_max_future_state_sup_growth_ratio,
        certified_max_future_denominator_growth_ratio=certified_max_future_denominator_growth_ratio,
        certified_max_future_step_radius_ratio=certified_max_future_step_radius_ratio,
        certified_initial_tail_transfer_growth_factor_bound=(
            certified_initial_tail_transfer_growth_factor_bound
        ),
        certified_max_future_tail_transfer_growth_ratio=(
            certified_max_future_tail_transfer_growth_ratio
        ),
        certified_initial_retained_order_increment=certified_initial_retained_order_increment,
        certified_retained_order_increment_growth=certified_retained_order_increment_growth,
    )


def certify_compactified_sundman_scalar_recurrence_induction(
    accelerated_certificate: CompactifiedSundmanAcceleratedGlobalSummabilityCertificate,
    *,
    domain_exhaustion_recurrence_certified: bool = False,
    scheduled_extension_recurrence_certified: bool = False,
    structural_envelope_template_certified: bool = False,
    tail_transfer_template_certified: bool = False,
    retained_order_schedule_template_certified: bool = False,
    witness_source: str = "manual_scalar_recurrence_induction",
    certified_schedule_initial_boundary_margin: float | None = None,
    certified_schedule_contraction: float | None = None,
    certified_extension_induction_start_index: int | None = None,
    certified_initial_segment_growth_ratio_bound: float | None = None,
    certified_recurrent_segment_growth_ratio_bound: float | None = None,
    certified_initial_state_sup_growth_ratio_bound: float | None = None,
    certified_recurrent_state_sup_growth_ratio_bound: float | None = None,
    certified_initial_denominator_growth_ratio_bound: float | None = None,
    certified_recurrent_denominator_growth_ratio_bound: float | None = None,
    certified_initial_step_radius_ratio_bound: float | None = None,
    certified_recurrent_step_radius_ratio_bound: float | None = None,
    certified_initial_tail_transfer_growth_factor_bound: float | None = None,
    certified_recurrent_tail_transfer_growth_factor_bound: float | None = None,
    certified_initial_tail_transfer_growth_ratio_bound: float | None = None,
    certified_recurrent_tail_transfer_growth_ratio_bound: float | None = None,
    certified_initial_retained_order_increment_bound: int | None = None,
    certified_recurrent_retained_order_increment_bound: int | None = None,
    certified_initial_retained_order_growth_bound: int | None = None,
    certified_recurrent_retained_order_growth_bound: int | None = None,
) -> CompactifiedSundmanScalarRecurrenceInductionCertificate:
    """Record arithmetic induction inequalities for future-shell recurrence."""

    return CompactifiedSundmanScalarRecurrenceInductionCertificate(
        accelerated_certificate=accelerated_certificate,
        domain_exhaustion_recurrence_certified=bool(domain_exhaustion_recurrence_certified),
        scheduled_extension_recurrence_certified=bool(
            scheduled_extension_recurrence_certified
        ),
        structural_envelope_template_certified=bool(structural_envelope_template_certified),
        tail_transfer_template_certified=bool(tail_transfer_template_certified),
        retained_order_schedule_template_certified=bool(
            retained_order_schedule_template_certified
        ),
        witness_source=str(witness_source),
        certified_schedule_initial_boundary_margin=certified_schedule_initial_boundary_margin,
        certified_schedule_contraction=certified_schedule_contraction,
        certified_extension_induction_start_index=certified_extension_induction_start_index,
        certified_initial_segment_growth_ratio_bound=(
            certified_initial_segment_growth_ratio_bound
        ),
        certified_recurrent_segment_growth_ratio_bound=(
            certified_recurrent_segment_growth_ratio_bound
        ),
        certified_initial_state_sup_growth_ratio_bound=(
            certified_initial_state_sup_growth_ratio_bound
        ),
        certified_recurrent_state_sup_growth_ratio_bound=(
            certified_recurrent_state_sup_growth_ratio_bound
        ),
        certified_initial_denominator_growth_ratio_bound=(
            certified_initial_denominator_growth_ratio_bound
        ),
        certified_recurrent_denominator_growth_ratio_bound=(
            certified_recurrent_denominator_growth_ratio_bound
        ),
        certified_initial_step_radius_ratio_bound=certified_initial_step_radius_ratio_bound,
        certified_recurrent_step_radius_ratio_bound=(
            certified_recurrent_step_radius_ratio_bound
        ),
        certified_initial_tail_transfer_growth_factor_bound=(
            certified_initial_tail_transfer_growth_factor_bound
        ),
        certified_recurrent_tail_transfer_growth_factor_bound=(
            certified_recurrent_tail_transfer_growth_factor_bound
        ),
        certified_initial_tail_transfer_growth_ratio_bound=(
            certified_initial_tail_transfer_growth_ratio_bound
        ),
        certified_recurrent_tail_transfer_growth_ratio_bound=(
            certified_recurrent_tail_transfer_growth_ratio_bound
        ),
        certified_initial_retained_order_increment_bound=(
            certified_initial_retained_order_increment_bound
        ),
        certified_recurrent_retained_order_increment_bound=(
            certified_recurrent_retained_order_increment_bound
        ),
        certified_initial_retained_order_growth_bound=(
            certified_initial_retained_order_growth_bound
        ),
        certified_recurrent_retained_order_growth_bound=(
            certified_recurrent_retained_order_growth_bound
        ),
    )


def certify_compactified_sundman_finite_prefix_envelope_summability(
    chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate,
    *,
    segment_growth_safety_factor: float = 1.0,
    state_sup_growth_safety_factor: float = 1.0,
    denominator_growth_safety_factor: float = 1.0,
    step_radius_safety_factor: float = 1.0,
    retained_order_increment: int | None = None,
) -> CompactifiedSundmanConditionalGlobalSummabilityCertificate:
    """Derive future-shell envelopes from the finite prefix plus safety factors.

    The result is still conditional.  The derived envelope is useful only after
    a separate induction proof shows that later shells obey it.
    """

    for factor_name, factor in (
        ("segment_growth_safety_factor", segment_growth_safety_factor),
        ("state_sup_growth_safety_factor", state_sup_growth_safety_factor),
        ("denominator_growth_safety_factor", denominator_growth_safety_factor),
        ("step_radius_safety_factor", step_radius_safety_factor),
    ):
        if not np.isfinite(factor) or factor < 1.0:
            raise ValueError(f"{factor_name} must be finite and at least 1")
    return certify_compactified_sundman_conditional_global_summability(
        chain,
        max_future_segment_growth_ratio=chain.max_added_segment_growth_ratio * segment_growth_safety_factor,
        max_future_state_sup_growth_ratio=chain.max_added_state_sup_growth_ratio * state_sup_growth_safety_factor,
        max_future_denominator_growth_ratio=(
            chain.max_cauchy_tail_denominator_growth_ratio * denominator_growth_safety_factor
        ),
        max_future_step_radius_ratio=chain.max_added_step_radius_ratio * step_radius_safety_factor,
        retained_order_increment=retained_order_increment,
        envelope_source="finite_prefix",
    )


def certify_compactified_sundman_finite_prefix_accelerated_summability(
    chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate,
    *,
    segment_growth_safety_factor: float = 1.0,
    state_sup_growth_safety_factor: float = 1.0,
    denominator_growth_safety_factor: float = 1.0,
    step_radius_safety_factor: float = 1.0,
    tail_transfer_growth_safety_factor: float = 1.0,
    initial_tail_transfer_safety_factor: float = 1.0,
    initial_retained_order_increment: int | None = None,
    retained_order_increment_growth: int | None = None,
) -> CompactifiedSundmanAcceleratedGlobalSummabilityCertificate:
    """Derive an accelerated future-shell envelope from a finite prefix.

    The first future transfer bound is projected from the latest observed
    transfer multiplier by the safety-adjusted observed transfer-growth ratio.
    Like the non-accelerated finite-prefix helper, the returned certificate is
    conditional until an induction proof shows that the derived envelopes hold
    for every later shell.
    """

    for factor_name, factor in (
        ("segment_growth_safety_factor", segment_growth_safety_factor),
        ("state_sup_growth_safety_factor", state_sup_growth_safety_factor),
        ("denominator_growth_safety_factor", denominator_growth_safety_factor),
        ("step_radius_safety_factor", step_radius_safety_factor),
        ("tail_transfer_growth_safety_factor", tail_transfer_growth_safety_factor),
        ("initial_tail_transfer_safety_factor", initial_tail_transfer_safety_factor),
    ):
        if not np.isfinite(factor) or factor < 1.0:
            raise ValueError(f"{factor_name} must be finite and at least 1")
    if retained_order_increment_growth is None:
        retained_order_increment_growth = chain.retained_order_increment_growth
    if initial_retained_order_increment is None:
        increments = chain.added_retained_order_increments
        initial_retained_order_increment = (
            increments[-1] + retained_order_increment_growth if increments else retained_order_increment_growth
        )

    transfer_growth_ratio = chain.max_tail_transfer_growth_ratio * tail_transfer_growth_safety_factor
    transfer_factors = chain.tail_transfer_growth_factors
    projected_transfer = (
        transfer_factors[-1] * transfer_growth_ratio * initial_tail_transfer_safety_factor
        if transfer_factors
        else float("inf")
    )
    return certify_compactified_sundman_accelerated_global_summability(
        chain,
        max_future_segment_growth_ratio=chain.max_added_segment_growth_ratio * segment_growth_safety_factor,
        max_future_state_sup_growth_ratio=chain.max_added_state_sup_growth_ratio * state_sup_growth_safety_factor,
        max_future_denominator_growth_ratio=(
            chain.max_cauchy_tail_denominator_growth_ratio * denominator_growth_safety_factor
        ),
        max_future_step_radius_ratio=chain.max_added_step_radius_ratio * step_radius_safety_factor,
        initial_retained_order_increment=initial_retained_order_increment,
        retained_order_increment_growth=retained_order_increment_growth,
        initial_tail_transfer_growth_factor_bound=projected_transfer,
        max_future_tail_transfer_growth_ratio=transfer_growth_ratio,
        envelope_source="finite_prefix",
    )


def certify_compactified_sundman_accelerated_future_envelope_holdout(
    chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate,
    *,
    training_extension_count: int,
    segment_growth_safety_factor: float = 1.0,
    state_sup_growth_safety_factor: float = 1.0,
    denominator_growth_safety_factor: float = 1.0,
    step_radius_safety_factor: float = 1.0,
    tail_transfer_growth_safety_factor: float = 1.0,
    initial_tail_transfer_safety_factor: float = 1.0,
    initial_retained_order_increment: int | None = None,
    retained_order_increment_growth: int | None = None,
) -> CompactifiedSundmanAcceleratedFutureEnvelopeHoldoutCertificate:
    """Derive an accelerated envelope from a prefix and validate held-out shells."""

    training_extension_count = int(training_extension_count)
    if training_extension_count < 3:
        raise ValueError("training_extension_count must include at least three extensions")
    if training_extension_count >= chain.extension_count:
        raise ValueError("training_extension_count must leave at least one held-out extension")
    training_chain = certify_compactified_sundman_geometric_exhaustion_extension_chain(
        chain.initial_schedule,
        chain.extensions[:training_extension_count],
    )
    accelerated_certificate = certify_compactified_sundman_finite_prefix_accelerated_summability(
        training_chain,
        segment_growth_safety_factor=segment_growth_safety_factor,
        state_sup_growth_safety_factor=state_sup_growth_safety_factor,
        denominator_growth_safety_factor=denominator_growth_safety_factor,
        step_radius_safety_factor=step_radius_safety_factor,
        tail_transfer_growth_safety_factor=tail_transfer_growth_safety_factor,
        initial_tail_transfer_safety_factor=initial_tail_transfer_safety_factor,
        initial_retained_order_increment=initial_retained_order_increment,
        retained_order_increment_growth=retained_order_increment_growth,
    )
    return CompactifiedSundmanAcceleratedFutureEnvelopeHoldoutCertificate(
        training_chain=training_chain,
        validation_extensions=tuple(chain.extensions[training_extension_count:]),
        accelerated_certificate=accelerated_certificate,
    )


def certify_compactified_sundman_accelerated_future_envelope_cross_validation(
    chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate,
    *,
    training_extension_counts: tuple[int, ...],
    segment_growth_safety_factor: float = 1.0,
    state_sup_growth_safety_factor: float = 1.0,
    denominator_growth_safety_factor: float = 1.0,
    step_radius_safety_factor: float = 1.0,
    tail_transfer_growth_safety_factor: float = 1.0,
    initial_tail_transfer_safety_factor: float = 1.0,
    retained_order_increment_growth: int | None = None,
) -> CompactifiedSundmanAcceleratedFutureEnvelopeCrossValidationCertificate:
    """Validate accelerated finite-prefix envelopes on multiple held-out suffixes."""

    training_extension_counts = tuple(int(count) for count in training_extension_counts)
    if not training_extension_counts:
        raise ValueError("training_extension_counts must be nonempty")
    holdouts = tuple(
        certify_compactified_sundman_accelerated_future_envelope_holdout(
            chain,
            training_extension_count=count,
            segment_growth_safety_factor=segment_growth_safety_factor,
            state_sup_growth_safety_factor=state_sup_growth_safety_factor,
            denominator_growth_safety_factor=denominator_growth_safety_factor,
            step_radius_safety_factor=step_radius_safety_factor,
            tail_transfer_growth_safety_factor=tail_transfer_growth_safety_factor,
            initial_tail_transfer_safety_factor=initial_tail_transfer_safety_factor,
            retained_order_increment_growth=retained_order_increment_growth,
        )
        for count in training_extension_counts
    )
    return CompactifiedSundmanAcceleratedFutureEnvelopeCrossValidationCertificate(
        chain=chain,
        holdouts=holdouts,
    )


def certify_compactified_sundman_future_envelope_holdout(
    chain: CompactifiedSundmanGeometricExhaustionExtensionChainCertificate,
    *,
    training_extension_count: int,
    segment_growth_safety_factor: float = 1.0,
    state_sup_growth_safety_factor: float = 1.0,
    denominator_growth_safety_factor: float = 1.0,
    step_radius_safety_factor: float = 1.0,
    tail_transfer_safety_factor: float = 1.0,
    tail_transfer_growth_safety_factor: float = 1.0,
    retained_order_increment: int | None = None,
) -> CompactifiedSundmanFutureEnvelopeHoldoutCertificate:
    """Derive an envelope from an earlier prefix and validate later finite shells."""

    training_extension_count = int(training_extension_count)
    if training_extension_count < 2:
        raise ValueError("training_extension_count must include at least two extensions")
    if training_extension_count >= chain.extension_count:
        raise ValueError("training_extension_count must leave at least one held-out extension")
    tail_transfer_safety_factor = float(tail_transfer_safety_factor)
    if not np.isfinite(tail_transfer_safety_factor) or tail_transfer_safety_factor < 1.0:
        raise ValueError("tail_transfer_safety_factor must be finite and at least 1")
    tail_transfer_growth_safety_factor = float(tail_transfer_growth_safety_factor)
    if not np.isfinite(tail_transfer_growth_safety_factor) or tail_transfer_growth_safety_factor < 1.0:
        raise ValueError("tail_transfer_growth_safety_factor must be finite and at least 1")
    training_chain = certify_compactified_sundman_geometric_exhaustion_extension_chain(
        chain.initial_schedule,
        chain.extensions[:training_extension_count],
    )
    conditional_certificate = certify_compactified_sundman_finite_prefix_envelope_summability(
        training_chain,
        segment_growth_safety_factor=segment_growth_safety_factor,
        state_sup_growth_safety_factor=state_sup_growth_safety_factor,
        denominator_growth_safety_factor=denominator_growth_safety_factor,
        step_radius_safety_factor=step_radius_safety_factor,
        retained_order_increment=retained_order_increment,
    )
    return CompactifiedSundmanFutureEnvelopeHoldoutCertificate(
        training_chain=training_chain,
        validation_extensions=chain.extensions[training_extension_count:],
        conditional_certificate=conditional_certificate,
        tail_transfer_safety_factor=tail_transfer_safety_factor,
        tail_transfer_growth_safety_factor=tail_transfer_growth_safety_factor,
    )


def _construct_compactified_sundman_centered_domain_cover(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    center: float,
    compact_radius: float,
    order: int,
    sundman_rate: float,
    distance_power: float,
    max_compact_step: float,
    radius_fraction: float,
    residual_tolerance: float,
) -> CompactifiedSundmanCompactDomainCoverCertificate:
    center = _validate_center(center)
    compact_radius = float(compact_radius)
    if not np.isfinite(compact_radius) or compact_radius <= 0.0:
        raise ValueError("compact radius must be a positive finite value")
    lower = center - compact_radius
    upper = center + compact_radius
    if lower <= -1.0 or upper >= 1.0:
        raise ValueError("centered compact domain must lie strictly inside (-1, 1)")
    backward = continue_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        lower,
        initial_compact_parameter=center,
        order=order,
        sundman_rate=sundman_rate,
        distance_power=distance_power,
        max_compact_step=max_compact_step,
        radius_fraction=radius_fraction,
        residual_tolerance=residual_tolerance,
        tail_certificate_mode="cauchy",
    )
    forward = continue_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        upper,
        initial_compact_parameter=center,
        order=order,
        sundman_rate=sundman_rate,
        distance_power=distance_power,
        max_compact_step=max_compact_step,
        radius_fraction=radius_fraction,
        residual_tolerance=residual_tolerance,
        tail_certificate_mode="cauchy",
    )
    return certify_compactified_sundman_compact_domain_cover(
        (backward.cauchy_cover_certificate, forward.cauchy_cover_certificate),
        compact_interval=FloatInterval(lower, upper),
    )


def construct_compactified_sundman_centered_exhaustion_prefix(
    positions: Array,
    velocities: Array,
    masses: Array,
    compact_radii: tuple[float, ...],
    *,
    initial_compact_parameter: float = 0.0,
    order: int = 16,
    sundman_rate: float = 1.0,
    distance_power: float = 1.0,
    max_compact_step: float = 0.04,
    radius_fraction: float = 0.25,
    residual_tolerance: float = 1e-8,
) -> CompactifiedSundmanCompactDomainExhaustionPrefixCertificate:
    """Build certified nested compact-domain covers centered at one ``w`` value."""

    if not compact_radii:
        raise ValueError("compact_radii must contain at least one positive radius")
    center = _validate_center(initial_compact_parameter)
    previous_radius = 0.0
    domain_covers: list[CompactifiedSundmanCompactDomainCoverCertificate] = []
    for compact_radius in compact_radii:
        compact_radius = float(compact_radius)
        if not np.isfinite(compact_radius) or compact_radius <= 0.0:
            raise ValueError("compact radii must be positive finite values")
        if compact_radius <= previous_radius:
            raise ValueError("compact radii must be strictly increasing")
        domain_covers.append(
            _construct_compactified_sundman_centered_domain_cover(
                positions,
                velocities,
                masses,
                center=center,
                compact_radius=compact_radius,
                order=order,
                sundman_rate=sundman_rate,
                distance_power=distance_power,
                max_compact_step=max_compact_step,
                radius_fraction=radius_fraction,
                residual_tolerance=residual_tolerance,
            )
        )
        previous_radius = compact_radius

    return certify_compactified_sundman_compact_domain_exhaustion_prefix(tuple(domain_covers))


def extend_compactified_sundman_centered_geometric_exhaustion_prefix(
    schedule: CompactifiedSundmanGeometricExhaustionScheduleCertificate,
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    initial_compact_parameter: float = 0.0,
    order: int = 16,
    sundman_rate: float = 1.0,
    distance_power: float = 1.0,
    max_compact_step: float = 0.04,
    radius_fraction: float = 0.25,
    residual_tolerance: float = 1e-8,
) -> CompactifiedSundmanGeometricExhaustionExtensionCertificate:
    """Construct and certify the next domain in a centered geometric prefix."""

    center = _validate_center(initial_compact_parameter)
    if not _compact_parameters_close(center, 0.0):
        raise ValueError("geometric endpoint-exhaustion prefixes are centered at w = 0")
    if not schedule.schedule_defined:
        raise ValueError("schedule must have a valid geometric definition")
    next_margin = schedule.next_expected_boundary_margin
    if not np.isfinite(next_margin) or not 0.0 < next_margin < 1.0:
        raise ValueError("next geometric boundary margin must lie strictly between 0 and 1")
    added_domain = _construct_compactified_sundman_centered_domain_cover(
        positions,
        velocities,
        masses,
        center=center,
        compact_radius=1.0 - next_margin,
        order=order,
        sundman_rate=sundman_rate,
        distance_power=distance_power,
        max_compact_step=max_compact_step,
        radius_fraction=radius_fraction,
        residual_tolerance=residual_tolerance,
    )
    return certify_compactified_sundman_geometric_exhaustion_extension(schedule, added_domain)


def extend_compactified_sundman_centered_geometric_exhaustion_chain(
    schedule: CompactifiedSundmanGeometricExhaustionScheduleCertificate,
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    extension_count: int,
    initial_compact_parameter: float = 0.0,
    order: int = 16,
    sundman_rate: float = 1.0,
    distance_power: float = 1.0,
    max_compact_step: float = 0.04,
    radius_fraction: float = 0.25,
    residual_tolerance: float = 1e-8,
) -> CompactifiedSundmanGeometricExhaustionExtensionChainCertificate:
    """Construct and certify several consecutive geometric exhaustion extensions."""

    if extension_count <= 0:
        raise ValueError("extension_count must be positive")
    current_schedule = schedule
    extensions: list[CompactifiedSundmanGeometricExhaustionExtensionCertificate] = []
    for _ in range(extension_count):
        extension = extend_compactified_sundman_centered_geometric_exhaustion_prefix(
            current_schedule,
            positions,
            velocities,
            masses,
            initial_compact_parameter=initial_compact_parameter,
            order=order,
            sundman_rate=sundman_rate,
            distance_power=distance_power,
            max_compact_step=max_compact_step,
            radius_fraction=radius_fraction,
            residual_tolerance=residual_tolerance,
        )
        extensions.append(extension)
        current_schedule = extension.extended_schedule
    return certify_compactified_sundman_geometric_exhaustion_extension_chain(schedule, tuple(extensions))


def extend_compactified_sundman_centered_geometric_exhaustion_chain_with_orders(
    schedule: CompactifiedSundmanGeometricExhaustionScheduleCertificate,
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    extension_orders: tuple[int, ...],
    initial_compact_parameter: float = 0.0,
    sundman_rate: float = 1.0,
    distance_power: float = 1.0,
    max_compact_step: float = 0.04,
    radius_fraction: float = 0.25,
    residual_tolerance: float = 1e-8,
) -> CompactifiedSundmanGeometricExhaustionExtensionChainCertificate:
    """Construct consecutive geometric extensions with per-extension Taylor orders."""

    if not extension_orders:
        raise ValueError("extension_orders must contain at least one order")
    current_schedule = schedule
    extensions: list[CompactifiedSundmanGeometricExhaustionExtensionCertificate] = []
    for order in extension_orders:
        order = int(order)
        if order < 1:
            raise ValueError("extension orders must be positive")
        extension = extend_compactified_sundman_centered_geometric_exhaustion_prefix(
            current_schedule,
            positions,
            velocities,
            masses,
            initial_compact_parameter=initial_compact_parameter,
            order=order,
            sundman_rate=sundman_rate,
            distance_power=distance_power,
            max_compact_step=max_compact_step,
            radius_fraction=radius_fraction,
            residual_tolerance=residual_tolerance,
        )
        extensions.append(extension)
        current_schedule = extension.extended_schedule
    return certify_compactified_sundman_geometric_exhaustion_extension_chain(schedule, tuple(extensions))


def construct_compactified_sundman_centered_geometric_exhaustion_prefix(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    initial_boundary_margin: float,
    contraction: float,
    prefix_length: int,
    initial_compact_parameter: float = 0.0,
    order: int = 16,
    sundman_rate: float = 1.0,
    distance_power: float = 1.0,
    max_compact_step: float = 0.04,
    radius_fraction: float = 0.25,
    residual_tolerance: float = 1e-8,
) -> CompactifiedSundmanGeometricExhaustionScheduleCertificate:
    """Build a finite centered prefix following a geometric boundary-margin schedule."""

    center = _validate_center(initial_compact_parameter)
    if not _compact_parameters_close(center, 0.0):
        raise ValueError("geometric endpoint-exhaustion prefixes are centered at w = 0")
    initial_boundary_margin = float(initial_boundary_margin)
    contraction = float(contraction)
    if prefix_length <= 0:
        raise ValueError("prefix_length must be positive")
    if not (
        np.isfinite(initial_boundary_margin)
        and np.isfinite(contraction)
        and 0.0 < initial_boundary_margin < 1.0
        and 0.0 < contraction < 1.0
    ):
        raise ValueError("geometric schedule requires 0 < initial_boundary_margin < 1 and 0 < contraction < 1")
    compact_radii = tuple(
        float(1.0 - initial_boundary_margin * contraction**index)
        for index in range(prefix_length)
    )
    prefix = construct_compactified_sundman_centered_exhaustion_prefix(
        positions,
        velocities,
        masses,
        compact_radii,
        initial_compact_parameter=center,
        order=order,
        sundman_rate=sundman_rate,
        distance_power=distance_power,
        max_compact_step=max_compact_step,
        radius_fraction=radius_fraction,
        residual_tolerance=residual_tolerance,
    )
    return certify_compactified_sundman_geometric_exhaustion_schedule(
        prefix,
        initial_boundary_margin=initial_boundary_margin,
        contraction=contraction,
    )

def construct_compactified_sundman_taylor_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
    sundman_rate: float = 1.0,
    distance_power: float = 1.0,
    center: float = 0.0,
) -> CompactifiedSundmanTaylorSolution:
    """Construct a local chart around compact Sundman parameter ``center``."""

    if order < 1:
        raise ValueError("order must be at least 1")
    sundman_rate = _validate_rate(sundman_rate)
    distance_power = _validate_distance_power(distance_power)
    center = _validate_center(center)
    positions, velocities, masses = _validate_initial_data(positions, velocities, masses)
    body_count, dimension = positions.shape

    q = np.zeros((order + 1, body_count, dimension), dtype=float)
    v = np.zeros_like(q)
    physical_time = np.zeros(order + 1, dtype=float)
    sundman_time = np.zeros(order + 1, dtype=float)
    q[0] = positions
    v[0] = velocities
    sundman_time[0] = physical_time_from_compact_parameter(center, rate=sundman_rate)

    compact_factor = compact_time_factor_coefficients(order, center=center, time_rate=sundman_rate)
    for degree in range(order):
        distance_factor = sundman_factor_coefficients(q, degree, distance_power=distance_power)
        combined_factor = scalar_series_product(compact_factor[: degree + 1], distance_factor, degree)
        acceleration = acceleration_coefficients(q, masses, degree)
        q_rhs = _series_vector_product(combined_factor, v, degree)
        v_rhs = _series_vector_product(combined_factor, acceleration, degree)
        scale = 1.0 / float(degree + 1)
        q[degree + 1] = q_rhs[degree] * scale
        v[degree + 1] = v_rhs[degree] * scale
        physical_time[degree + 1] = combined_factor[degree] * scale
        sundman_time[degree + 1] = compact_factor[degree] * scale

    return CompactifiedSundmanTaylorSolution(
        position=q,
        velocity=v,
        physical_time=physical_time,
        sundman_time=sundman_time,
        masses=masses,
        sundman_rate=sundman_rate,
        distance_power=distance_power,
        center=center,
    )


def construct_interval_compactified_sundman_taylor_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
    sundman_rate: float = 1.0,
    distance_power: float = 1.0,
    center: float = 0.0,
) -> IntervalCompactifiedSundmanTaylorSolution:
    """Construct interval compactified-Sundman coefficients from point data."""

    positions, velocities, masses = _validate_initial_data(positions, velocities, masses)
    return construct_interval_compactified_sundman_taylor_solution_from_intervals(
        _interval_array_from_points(positions),
        _interval_array_from_points(velocities),
        masses,
        order=order,
        sundman_rate=sundman_rate,
        distance_power=distance_power,
        center=center,
    )


def construct_interval_compactified_sundman_taylor_solution_from_intervals(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    order: int,
    sundman_rate: float = 1.0,
    distance_power: float = 1.0,
    center: float = 0.0,
) -> IntervalCompactifiedSundmanTaylorSolution:
    """Construct interval compactified-Sundman coefficients from interval data."""

    if order < 1:
        raise ValueError("order must be at least 1")
    sundman_rate = _validate_rate(sundman_rate)
    distance_power = _validate_distance_power(distance_power)
    center = _validate_center(center)
    positions, velocities, masses = _validate_interval_initial_data(positions, velocities, masses)
    body_count, dimension = positions.shape

    q = _interval_zeros((order + 1, body_count, dimension))
    v = _interval_zeros((order + 1, body_count, dimension))
    physical_time = _interval_zeros((order + 1,))
    sundman_time = _interval_zeros((order + 1,))
    q[0] = positions
    v[0] = velocities
    sundman_time[0] = FloatInterval.point(physical_time_from_compact_parameter(center, rate=sundman_rate))

    compact_factor = tuple(
        FloatInterval.point(value)
        for value in compact_time_factor_coefficients(order, center=center, time_rate=sundman_rate)
    )
    for degree in range(order):
        distance_factor = sundman_factor_interval_coefficients(q, degree, distance_power=distance_power)
        combined_factor = interval_series_product(compact_factor[: degree + 1], distance_factor, degree)
        acceleration = acceleration_interval_coefficients(q, masses, degree)
        q_rhs = _interval_series_vector_product(combined_factor, v, degree)
        v_rhs = _interval_series_vector_product(combined_factor, acceleration, degree)
        scale = 1.0 / float(degree + 1)
        for index in np.ndindex((body_count, dimension)):
            q[(degree + 1, *index)] = q_rhs[(degree, *index)].scale(scale)
            v[(degree + 1, *index)] = v_rhs[(degree, *index)].scale(scale)
        physical_time[degree + 1] = combined_factor[degree].scale(scale)
        sundman_time[degree + 1] = compact_factor[degree].scale(scale)

    return IntervalCompactifiedSundmanTaylorSolution(
        position=q,
        velocity=v,
        physical_time=physical_time,
        sundman_time=sundman_time,
        masses=masses,
        sundman_rate=sundman_rate,
        distance_power=distance_power,
        center=center,
    )


def continue_interval_compactified_sundman_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    target_compact_parameter: float,
    *,
    initial_compact_parameter: float = 0.0,
    order: int = 12,
    sundman_rate: float = 1.0,
    distance_power: float = 1.0,
    max_compact_step: float = 0.03,
    radius_fraction: float = 0.2,
    guard_order: int = 0,
    tail_certificate_mode: str = "guarded",
) -> IntervalCompactifiedSundmanAtlasSolution:
    """Propagate an interval state box through compactified-Sundman charts."""

    if order < 1:
        raise ValueError("order must be at least 1")
    sundman_rate = _validate_rate(sundman_rate)
    distance_power = _validate_distance_power(distance_power)
    current_w = _validate_center(initial_compact_parameter)
    target_w = _validate_center(target_compact_parameter)
    if max_compact_step <= 0.0:
        raise ValueError("max_compact_step must be positive")
    if not 0.0 < radius_fraction < 1.0:
        raise ValueError("radius_fraction must lie strictly between 0 and 1")
    if guard_order < 0:
        raise ValueError("guard_order cannot be negative")
    if tail_certificate_mode not in {"guarded", "cauchy"}:
        raise ValueError("tail_certificate_mode must be 'guarded' or 'cauchy'")

    current_positions, current_velocities, masses = _validate_interval_initial_data(positions, velocities, masses)
    current_state_interval = _interval_state(current_positions, current_velocities)
    compact_parameters = [current_w]
    state_intervals = [current_state_interval]
    physical_time_intervals = [FloatInterval.point(0.0)]
    sundman_times = [physical_time_from_compact_parameter(current_w, rate=sundman_rate)]
    steps: list[IntervalCompactifiedSundmanAtlasStep] = []

    while abs(target_w - current_w) > 10.0 * np.finfo(float).eps:
        compact_step = choose_compact_step_size(
            current_w,
            target_w,
            max_compact_step=max_compact_step,
            radius_fraction=radius_fraction,
        )
        computed_chart = construct_interval_compactified_sundman_taylor_solution_from_intervals(
            current_positions,
            current_velocities,
            masses,
            order=order + (guard_order if tail_certificate_mode == "guarded" else 0),
            sundman_rate=sundman_rate,
            distance_power=distance_power,
            center=current_w,
        )
        chart = _truncate_interval_chart(computed_chart, order)
        if tail_certificate_mode == "cauchy":
            radius_certificate = compactified_sundman_interval_cauchy_majorant_tail_certificate(
                chart,
                retained_order=order,
                compact_parameter=current_w,
            )
            compact_step = _cap_compact_step_to_cauchy_radius(
                compact_step,
                compact_radius=radius_certificate.compact_radius,
            )
        next_w = current_w + compact_step
        residual = certify_interval_compactified_sundman_equations(chart)
        (
            angular_momentum_certificate,
            energy_certificate,
            linear_momentum_certificate,
            center_of_mass_certificate,
        ) = _certify_interval_compactified_sundman_invariants(chart, coefficient_count=order)
        factor_interval = compactified_sundman_factor_interval_over_w_interval(
            chart,
            FloatInterval(min(current_w, next_w), max(current_w, next_w)),
        )
        if factor_interval.lower <= 0.0:
            raise RuntimeError("compactified Sundman time monotonicity is not certified on interval step")
        if tail_certificate_mode == "cauchy":
            tail_certificate = compactified_sundman_interval_cauchy_majorant_tail_certificate(
                chart,
                retained_order=order,
                compact_parameter=next_w,
            )
        else:
            tail_certificate = (
                compactified_sundman_interval_guarded_tail_certificate(
                    computed_chart,
                    retained_order=order,
                    compact_parameter=next_w,
                )
                if guard_order > 0
                else None
            )
        tail_bound = 0.0 if tail_certificate is None else tail_certificate.tail_bound
        next_state_interval = _inflate_interval_array(chart.state_at_w(next_w), tail_bound)
        physical_step = _inflate_interval(chart.physical_time_delta_at_w(next_w), tail_bound)
        next_time_interval = physical_time_intervals[-1] + physical_step
        next_sundman_time = physical_time_from_compact_parameter(next_w, rate=sundman_rate)
        steps.append(
            IntervalCompactifiedSundmanAtlasStep(
                start_compact_parameter=current_w,
                end_compact_parameter=next_w,
                chart=chart,
                start_state_interval=current_state_interval,
                end_state_interval=next_state_interval,
                physical_time_step_interval=physical_step,
                factor_interval=factor_interval,
                sundman_time_step=next_sundman_time - sundman_times[-1],
                residual_certificate=residual,
                angular_momentum_certificate=angular_momentum_certificate,
                energy_certificate=energy_certificate,
                linear_momentum_certificate=linear_momentum_certificate,
                center_of_mass_certificate=center_of_mass_certificate,
                tail_certificate=tail_certificate,
            )
        )
        current_w = next_w
        current_positions, current_velocities = _split_state_interval(
            next_state_interval,
            body_count=current_positions.shape[0],
            dimension=current_positions.shape[1],
        )
        current_state_interval = next_state_interval
        compact_parameters.append(current_w)
        state_intervals.append(current_state_interval)
        physical_time_intervals.append(next_time_interval)
        sundman_times.append(next_sundman_time)

    return IntervalCompactifiedSundmanAtlasSolution(
        masses=masses,
        sundman_rate=sundman_rate,
        distance_power=distance_power,
        compact_parameters=np.array(compact_parameters, dtype=float),
        state_intervals=tuple(state_intervals),
        physical_time_intervals=tuple(physical_time_intervals),
        sundman_times=np.array(sundman_times, dtype=float),
        steps=tuple(steps),
    )


def certify_compactified_sundman_physical_time_target(
    chart: IntervalCompactifiedSundmanTaylorSolution,
    start_time_interval: FloatInterval,
    target_time: float,
    compact_bounds: FloatInterval,
    *,
    max_bisections: int = 60,
    time_tail_bound: float = 0.0,
) -> CompactifiedSundmanPhysicalTimeTargetCertificate:
    """Bracket a physical-time target inside one compactified-Sundman chart.

    The certificate is proof-grade when the compactified time factor is
    positive over the returned compact interval and the outward-rounded endpoint
    time intervals force every represented trajectory to cross ``target_time``.
    """

    target_time = float(target_time)
    start_time_interval = _as_interval(start_time_interval)
    compact_bounds = _as_interval(compact_bounds)
    if max_bisections < 0:
        raise ValueError("max_bisections cannot be negative")
    if time_tail_bound < 0.0:
        raise ValueError("time_tail_bound cannot be negative")
    if compact_bounds.lower >= compact_bounds.upper:
        raise ValueError("compact_bounds must have positive width")
    _validate_compact_interval_inside_chart(
        compact_bounds,
        center=chart.center,
        analytic_radius=chart.analytic_radius,
    )

    low = compact_bounds.lower
    high = compact_bounds.upper

    def time_with_tail(compact_parameter: float) -> FloatInterval:
        return _inflate_interval(chart.physical_time_delta_at_w(compact_parameter), time_tail_bound)

    def make_certificate(bisections: int) -> CompactifiedSundmanPhysicalTimeTargetCertificate:
        compact_interval = FloatInterval(low, high)
        local_lower = time_with_tail(low)
        local_upper = time_with_tail(high)
        global_lower = start_time_interval + local_lower
        global_upper = start_time_interval + local_upper
        return CompactifiedSundmanPhysicalTimeTargetCertificate(
            target_time=target_time,
            chart_start_compact_parameter=chart.center,
            compact_parameter_interval=compact_interval,
            start_time_interval=start_time_interval,
            local_time_at_lower=local_lower,
            local_time_at_upper=local_upper,
            global_time_at_lower=global_lower,
            global_time_at_upper=global_upper,
            factor_interval=compactified_sundman_factor_interval_over_w_interval(chart, compact_interval),
            bisections=bisections,
            time_tail_bound=float(time_tail_bound),
        )

    certificate = make_certificate(0)
    if not certificate.certified:
        return certificate

    for bisection in range(1, max_bisections + 1):
        midpoint = (low + high) / 2.0
        midpoint_time = start_time_interval + time_with_tail(midpoint)
        if midpoint_time.upper <= target_time:
            low = midpoint
        elif midpoint_time.lower >= target_time:
            high = midpoint
        else:
            break

        next_certificate = make_certificate(bisection)
        if not next_certificate.certified:
            break
        certificate = next_certificate

    return certificate


def continue_interval_compactified_sundman_to_time(
    positions: Array,
    velocities: Array,
    masses: Array,
    target_time: float,
    *,
    initial_compact_parameter: float = 0.0,
    order: int = 12,
    sundman_rate: float = 1.0,
    distance_power: float = 1.0,
    max_compact_step: float = 0.03,
    radius_fraction: float = 0.2,
    max_steps: int = 10000,
    target_bisections: int = 60,
    step_shrink_bisections: int = 60,
    guard_order: int = 0,
    tail_certificate_mode: str = "guarded",
) -> IntervalCompactifiedSundmanTimeTargetSolution:
    """Propagate compactified-Sundman interval charts to a physical-time target."""

    target_time = float(target_time)
    if target_time == 0.0:
        raise ValueError("target_time must be nonzero")
    if order < 1:
        raise ValueError("order must be at least 1")
    sundman_rate = _validate_rate(sundman_rate)
    distance_power = _validate_distance_power(distance_power)
    current_w = _validate_center(initial_compact_parameter)
    if max_compact_step <= 0.0:
        raise ValueError("max_compact_step must be positive")
    if not 0.0 < radius_fraction < 1.0:
        raise ValueError("radius_fraction must lie strictly between 0 and 1")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if target_bisections < 0:
        raise ValueError("target_bisections cannot be negative")
    if step_shrink_bisections < 0:
        raise ValueError("step_shrink_bisections cannot be negative")
    if guard_order < 0:
        raise ValueError("guard_order cannot be negative")
    if tail_certificate_mode not in {"guarded", "cauchy"}:
        raise ValueError("tail_certificate_mode must be 'guarded' or 'cauchy'")

    current_positions, current_velocities, masses = _validate_interval_initial_data(positions, velocities, masses)
    triple_collision_exclusion_certificate = certify_nonzero_angular_momentum_excludes_triple_collision(
        current_positions,
        current_velocities,
        masses,
    )
    direction = float(np.sign(target_time))
    current_state_interval = _interval_state(current_positions, current_velocities)
    current_time_interval = FloatInterval.point(0.0)
    compact_parameters = [current_w]
    state_intervals = [current_state_interval]
    physical_time_intervals = [current_time_interval]
    sundman_times = [physical_time_from_compact_parameter(current_w, rate=sundman_rate)]
    steps: list[IntervalCompactifiedSundmanAtlasStep] = []

    for _step_index in range(max_steps):
        compact_step = _choose_directed_compact_step(
            current_w,
            direction,
            max_compact_step=max_compact_step,
            radius_fraction=radius_fraction,
        )
        computed_chart = construct_interval_compactified_sundman_taylor_solution_from_intervals(
            current_positions,
            current_velocities,
            masses,
            order=order + (guard_order if tail_certificate_mode == "guarded" else 0),
            sundman_rate=sundman_rate,
            distance_power=distance_power,
            center=current_w,
        )
        chart = _truncate_interval_chart(computed_chart, order)
        if tail_certificate_mode == "cauchy":
            radius_certificate = compactified_sundman_interval_cauchy_majorant_tail_certificate(
                chart,
                retained_order=order,
                compact_parameter=current_w,
            )
            compact_step = _cap_compact_step_to_cauchy_radius(
                compact_step,
                compact_radius=radius_certificate.compact_radius,
            )
        residual = certify_interval_compactified_sundman_equations(chart)
        if not residual.certified:
            raise RuntimeError("compactified Sundman residuals are not certified")
        (
            angular_momentum_certificate,
            energy_certificate,
            linear_momentum_certificate,
            center_of_mass_certificate,
        ) = _certify_interval_compactified_sundman_invariants(chart, coefficient_count=order)
        if not angular_momentum_certificate.certified:
            raise RuntimeError("compactified Sundman centered angular momentum conservation is not certified")
        if not energy_certificate.certified:
            raise RuntimeError("compactified Sundman total-energy conservation is not certified")
        if not linear_momentum_certificate.certified:
            raise RuntimeError("compactified Sundman linear momentum conservation is not certified")
        if not center_of_mass_certificate.certified:
            raise RuntimeError("compactified Sundman center-of-mass motion is not certified")

        for _shrink_index in range(step_shrink_bisections + 1):
            next_w = current_w + compact_step
            compact_bounds = FloatInterval(min(current_w, next_w), max(current_w, next_w))
            if tail_certificate_mode == "cauchy":
                try:
                    step_tail_certificate = compactified_sundman_interval_cauchy_majorant_tail_certificate_over_w_interval(
                        chart,
                        retained_order=order,
                        compact_interval=compact_bounds,
                    )
                except ValueError:
                    if _shrink_index >= step_shrink_bisections:
                        raise
                    compact_step *= 0.5
                    continue
            else:
                step_tail_certificate = (
                    compactified_sundman_interval_guarded_tail_certificate_over_w_interval(
                        computed_chart,
                        retained_order=order,
                        compact_interval=compact_bounds,
                    )
                    if guard_order > 0
                        else None
                )
            step_tail_bound = 0.0 if step_tail_certificate is None else step_tail_certificate.tail_bound
            if not np.isfinite(step_tail_bound):
                if _shrink_index >= step_shrink_bisections:
                    raise RuntimeError(
                        "compactified Sundman tail bound stayed nonfinite after adaptive shortening"
                    )
                compact_step *= 0.5
                continue
            target_certificate = certify_compactified_sundman_physical_time_target(
                chart,
                current_time_interval,
                target_time,
                compact_bounds,
                max_bisections=target_bisections,
                time_tail_bound=step_tail_bound,
            )
            if target_certificate.certified:
                if tail_certificate_mode == "cauchy":
                    target_tail_certificate = compactified_sundman_interval_cauchy_majorant_tail_certificate_over_w_interval(
                        chart,
                        retained_order=order,
                        compact_interval=target_certificate.compact_parameter_interval,
                    )
                else:
                    target_tail_certificate = (
                        compactified_sundman_interval_guarded_tail_certificate_over_w_interval(
                            computed_chart,
                            retained_order=order,
                            compact_interval=target_certificate.compact_parameter_interval,
                        )
                        if guard_order > 0
                        else None
                    )
                target_tail_bound = (
                    0.0 if target_tail_certificate is None else target_tail_certificate.tail_bound
                )
                if not np.isfinite(target_tail_bound):
                    if _shrink_index >= step_shrink_bisections:
                        raise RuntimeError(
                            "compactified Sundman target tail bound stayed nonfinite after adaptive shortening"
                        )
                    compact_step *= 0.5
                    continue
                target_state_interval = _inflate_interval_array(
                    chart.state_over_w_interval(target_certificate.compact_parameter_interval),
                    target_tail_bound,
                )
                return IntervalCompactifiedSundmanTimeTargetSolution(
                    masses=masses,
                    target_time=target_time,
                    sundman_rate=sundman_rate,
                    distance_power=distance_power,
                    compact_parameters=np.array(compact_parameters, dtype=float),
                    physical_time_intervals=tuple(physical_time_intervals),
                    sundman_times=np.array(sundman_times, dtype=float),
                    state_intervals=tuple(state_intervals),
                    steps=tuple(steps),
                    target_start_state_interval=current_state_interval,
                    target_certificate=target_certificate,
                    target_state_interval=target_state_interval,
                    target_residual_certificate=residual,
                    target_angular_momentum_certificate=angular_momentum_certificate,
                    target_energy_certificate=energy_certificate,
                    target_linear_momentum_certificate=linear_momentum_certificate,
                    target_center_of_mass_certificate=center_of_mass_certificate,
                    target_tail_certificate=target_tail_certificate,
                    triple_collision_exclusion_certificate=triple_collision_exclusion_certificate,
                )

            factor_interval = compactified_sundman_factor_interval_over_w_interval(chart, compact_bounds)
            if factor_interval.lower <= 0.0:
                raise RuntimeError("compactified Sundman time monotonicity is not certified on interval step")

            next_state_interval = _inflate_interval_array(chart.state_at_w(next_w), step_tail_bound)
            physical_step = _inflate_interval(chart.physical_time_delta_at_w(next_w), step_tail_bound)
            next_time_interval = current_time_interval + physical_step
            target_inside_uncertified_step = (
                (direction > 0.0 and target_time <= next_time_interval.upper)
                or (direction < 0.0 and target_time >= next_time_interval.lower)
            )
            if not target_inside_uncertified_step:
                break
            compact_step *= 0.5
        else:
            raise RuntimeError(
                "target physical time remained inside an uncertified compactified-Sundman step "
                "after adaptive shortening"
            )

        next_sundman_time = physical_time_from_compact_parameter(next_w, rate=sundman_rate)
        steps.append(
            IntervalCompactifiedSundmanAtlasStep(
                start_compact_parameter=current_w,
                end_compact_parameter=next_w,
                chart=chart,
                start_state_interval=current_state_interval,
                end_state_interval=next_state_interval,
                physical_time_step_interval=physical_step,
                factor_interval=factor_interval,
                sundman_time_step=next_sundman_time - sundman_times[-1],
                residual_certificate=residual,
                angular_momentum_certificate=angular_momentum_certificate,
                energy_certificate=energy_certificate,
                linear_momentum_certificate=linear_momentum_certificate,
                center_of_mass_certificate=center_of_mass_certificate,
                tail_certificate=step_tail_certificate,
            )
        )
        current_w = next_w
        current_positions, current_velocities = _split_state_interval(
            next_state_interval,
            body_count=current_positions.shape[0],
            dimension=current_positions.shape[1],
        )
        current_state_interval = next_state_interval
        current_time_interval = next_time_interval
        compact_parameters.append(current_w)
        state_intervals.append(current_state_interval)
        physical_time_intervals.append(current_time_interval)
        sundman_times.append(next_sundman_time)

    raise RuntimeError("target physical time was not reached before max_steps")


def continue_compactified_sundman_solution(
    positions: Array,
    velocities: Array,
    masses: Array,
    target_compact_parameter: float,
    *,
    initial_compact_parameter: float = 0.0,
    order: int = 16,
    sundman_rate: float = 1.0,
    distance_power: float = 1.0,
    max_compact_step: float = 0.04,
    radius_fraction: float = 0.25,
    residual_tolerance: float = 1e-8,
    guard_order: int = 0,
    tail_certificate_mode: str = "guarded",
) -> CompactifiedSundmanAtlasSolution:
    """Continue by chaining compactified-Sundman Taylor charts."""

    if order < 1:
        raise ValueError("order must be at least 1")
    sundman_rate = _validate_rate(sundman_rate)
    distance_power = _validate_distance_power(distance_power)
    current_w = _validate_center(initial_compact_parameter)
    target_w = _validate_center(target_compact_parameter)
    if max_compact_step <= 0.0:
        raise ValueError("max_compact_step must be positive")
    if not 0.0 < radius_fraction < 1.0:
        raise ValueError("radius_fraction must lie strictly between 0 and 1")
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive")
    if guard_order < 0:
        raise ValueError("guard_order cannot be negative")
    if tail_certificate_mode not in {"guarded", "cauchy"}:
        raise ValueError("tail_certificate_mode must be 'guarded' or 'cauchy'")
    current_positions, current_velocities, masses = _validate_initial_data(positions, velocities, masses)

    compact_parameters = [current_w]
    physical_times = [0.0]
    sundman_times = [physical_time_from_compact_parameter(current_w, rate=sundman_rate)]
    states = [np.concatenate([current_positions.reshape(-1), current_velocities.reshape(-1)])]
    steps: list[CompactifiedSundmanAtlasStep] = []

    while abs(target_w - current_w) > 10.0 * np.finfo(float).eps:
        compact_step = choose_compact_step_size(
            current_w,
            target_w,
            max_compact_step=max_compact_step,
            radius_fraction=radius_fraction,
        )
        computed_chart = construct_compactified_sundman_taylor_solution(
            current_positions,
            current_velocities,
            masses,
            order=order + (guard_order if tail_certificate_mode == "guarded" else 0),
            sundman_rate=sundman_rate,
            distance_power=distance_power,
            center=current_w,
        )
        chart = _truncate_chart(computed_chart, order)
        if tail_certificate_mode == "cauchy":
            radius_certificate = compactified_sundman_cauchy_majorant_tail_certificate(
                chart,
                retained_order=order,
                compact_parameter=current_w,
            )
            compact_step = _cap_compact_step_to_cauchy_radius(
                compact_step,
                compact_radius=radius_certificate.compact_radius,
            )
        next_w = current_w + compact_step
        residual = certify_compactified_sundman_equations(chart, tolerance=residual_tolerance)
        indicator = compactified_sundman_truncation_indicator(chart, next_w)
        if tail_certificate_mode == "cauchy":
            tail_certificate = compactified_sundman_cauchy_majorant_tail_certificate(
                chart,
                retained_order=order,
                compact_parameter=next_w,
            )
        else:
            tail_certificate = (
                compactified_sundman_guarded_tail_certificate(
                    computed_chart,
                    retained_order=order,
                    compact_parameter=next_w,
                )
                if guard_order > 0
                else None
            )
        next_state = chart.state_at_w(next_w)
        body_count, dimension = current_positions.shape
        current_positions = next_state[: body_count * dimension].reshape(body_count, dimension)
        current_velocities = next_state[body_count * dimension :].reshape(body_count, dimension)
        steps.append(
            CompactifiedSundmanAtlasStep(
                start_compact_parameter=current_w,
                end_compact_parameter=next_w,
                chart=chart,
                residual_certificate=residual,
                truncation_indicator=indicator,
                tail_certificate=tail_certificate,
            )
        )
        current_w = next_w
        compact_parameters.append(current_w)
        physical_times.append(physical_times[-1] + chart.physical_time_delta_at_w(current_w))
        sundman_times.append(physical_time_from_compact_parameter(current_w, rate=sundman_rate))
        states.append(next_state)

    return CompactifiedSundmanAtlasSolution(
        masses=masses,
        sundman_rate=sundman_rate,
        distance_power=distance_power,
        compact_parameters=np.array(compact_parameters, dtype=float),
        states=np.vstack(states),
        physical_times=np.array(physical_times, dtype=float),
        sundman_times=np.array(sundman_times, dtype=float),
        steps=tuple(steps),
    )


def compactified_sundman_factor_interval_over_w_interval(
    chart: IntervalCompactifiedSundmanTaylorSolution,
    compact_interval: FloatInterval,
) -> FloatInterval:
    """Evaluate an interval enclosure of ``dt/dw`` over a compact-Sundman interval."""

    compact_interval = _as_interval(compact_interval)
    _validate_compact_interval_inside_chart(
        compact_interval,
        center=chart.center,
        analytic_radius=chart.analytic_radius,
    )

    max_degree = chart.order
    compact_factor = tuple(
        FloatInterval.point(value)
        for value in compact_time_factor_coefficients(
            max_degree,
            center=chart.center,
            time_rate=chart.sundman_rate,
        )
    )
    distance_factor = sundman_factor_interval_coefficients(
        chart.position,
        max_degree,
        distance_power=chart.distance_power,
    )
    combined_factor = interval_series_product(compact_factor, distance_factor, max_degree)
    delta_interval = FloatInterval(
        float(np.nextafter(compact_interval.lower - chart.center, -np.inf)),
        float(np.nextafter(compact_interval.upper - chart.center, np.inf)),
    )
    return interval_polynomial_eval(combined_factor, delta_interval)


def _certify_interval_compactified_sundman_invariants(
    chart: IntervalCompactifiedSundmanTaylorSolution,
    *,
    coefficient_count: int,
) -> tuple[
    AngularMomentumConservationCertificate,
    EnergyConservationCertificate,
    LinearMomentumConservationCertificate,
    CenterOfMassMotionCertificate,
]:
    angular_momentum_certificate = certify_interval_centered_angular_momentum_conservation(
        chart.position,
        chart.velocity,
        chart.masses,
        coefficient_count=coefficient_count,
    )
    energy_certificate = certify_interval_total_energy_conservation(
        chart.position,
        chart.velocity,
        chart.masses,
        coefficient_count=coefficient_count,
    )
    linear_momentum_certificate = certify_interval_linear_momentum_conservation(
        chart.velocity,
        chart.masses,
        coefficient_count=coefficient_count,
    )
    center_of_mass_certificate = certify_interval_center_of_mass_motion(
        chart.position,
        chart.velocity,
        chart.physical_time,
        chart.masses,
        coefficient_count=coefficient_count,
    )
    return (
        angular_momentum_certificate,
        energy_certificate,
        linear_momentum_certificate,
        center_of_mass_certificate,
    )


def certify_interval_compactified_sundman_equations(
    chart: IntervalCompactifiedSundmanTaylorSolution,
    *,
    coefficient_count: int | None = None,
) -> IntervalCompactifiedSundmanResidualCertificate:
    """Certify interval residuals for compactified Sundman equations."""

    if coefficient_count is None:
        coefficient_count = chart.order
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > chart.order:
        raise ValueError("coefficient_count cannot exceed chart.order")

    max_degree = coefficient_count - 1
    compact_factor = tuple(
        FloatInterval.point(value)
        for value in compact_time_factor_coefficients(
            max_degree,
            center=chart.center,
            time_rate=chart.sundman_rate,
        )
    )
    distance_factor = sundman_factor_interval_coefficients(
        chart.position,
        max_degree,
        distance_power=chart.distance_power,
    )
    combined_factor = interval_series_product(compact_factor, distance_factor, max_degree)
    acceleration = acceleration_interval_coefficients(chart.position, chart.masses, max_degree)
    position_rhs = _interval_series_vector_product(combined_factor, chart.velocity, max_degree)
    velocity_rhs = _interval_series_vector_product(combined_factor, acceleration, max_degree)

    position_residual = _interval_zeros(position_rhs.shape)
    velocity_residual = _interval_zeros(velocity_rhs.shape)
    physical_time_residual = _interval_zeros((coefficient_count,))
    sundman_time_residual = _interval_zeros((coefficient_count,))
    for degree in range(coefficient_count):
        scale = float(degree + 1)
        for index in np.ndindex(chart.position.shape[1:]):
            position_residual[(degree, *index)] = _as_interval(chart.position[(degree + 1, *index)]).scale(
                scale
            ) - _as_interval(position_rhs[(degree, *index)])
            velocity_residual[(degree, *index)] = _as_interval(chart.velocity[(degree + 1, *index)]).scale(
                scale
            ) - _as_interval(velocity_rhs[(degree, *index)])
        physical_time_residual[degree] = _as_interval(chart.physical_time[degree + 1]).scale(scale) - combined_factor[
            degree
        ]
        sundman_time_residual[degree] = _as_interval(chart.sundman_time[degree + 1]).scale(scale) - compact_factor[
            degree
        ]

    return IntervalCompactifiedSundmanResidualCertificate(
        coefficient_count=coefficient_count,
        position_residual=position_residual,
        velocity_residual=velocity_residual,
        physical_time_residual=physical_time_residual,
        sundman_time_residual=sundman_time_residual,
    )


def compactified_sundman_cauchy_majorant_tail_certificate(
    chart: CompactifiedSundmanTaylorSolution,
    *,
    retained_order: int,
    compact_parameter: float,
    compact_radius: float | None = None,
) -> CompactifiedSundmanCauchyMajorantCertificate:
    """Build an a priori Cauchy tail certificate in compact Sundman time.

    The certificate transports the existing Sundman-time Cauchy disk through
    ``s = atanh(w) / rate``.  The compact disk is accepted only when a
    derivative bound for ``atanh`` maps the whole disk into the certified
    Sundman disk and keeps the disk away from the singular endpoints
    ``w = +/-1``.
    """

    delta = chart.delta_from_compact_parameter(compact_parameter)
    sundman_certificate = sundman_cauchy_majorant_tail_certificate(
        chart.position[0],
        chart.velocity[0],
        chart.masses,
        retained_order=retained_order,
        step_size=0.0,
        distance_power=chart.distance_power,
    )
    compact_radius = _choose_compact_sundman_cauchy_radius(
        center=chart.center,
        sundman_rate=chart.sundman_rate,
        sundman_radius=sundman_certificate.s_radius,
        step_size=delta,
        compact_radius=compact_radius,
    )
    sundman_image_radius = _compact_sundman_cauchy_image_radius(
        center=chart.center,
        sundman_rate=chart.sundman_rate,
        compact_radius=compact_radius,
    )
    center_s = physical_time_from_compact_parameter(chart.center, rate=chart.sundman_rate)
    state_sup_bound = max(
        sundman_certificate.state_sup_bound,
        abs(center_s) + sundman_image_radius,
    )
    ratio = abs(delta) / compact_radius
    tail_bound = float(np.nextafter(state_sup_bound * ratio ** (retained_order + 1) / (1.0 - ratio), np.inf))
    return CompactifiedSundmanCauchyMajorantCertificate(
        retained_order=int(retained_order),
        step_size=float(delta),
        compact_radius=float(compact_radius),
        sundman_image_radius=float(sundman_image_radius),
        center=float(chart.center),
        sundman_rate=float(chart.sundman_rate),
        state_sup_bound=float(state_sup_bound),
        tail_bound=float(tail_bound),
        sundman_certificate=sundman_certificate,
    )


def compactified_sundman_interval_cauchy_majorant_tail_certificate(
    chart: IntervalCompactifiedSundmanTaylorSolution,
    *,
    retained_order: int,
    compact_parameter: float,
    compact_radius: float | None = None,
) -> CompactifiedSundmanIntervalCauchyMajorantCertificate:
    """Build an interval Cauchy tail certificate in compact Sundman time."""

    delta = chart.delta_from_compact_parameter(compact_parameter)
    return compactified_sundman_interval_cauchy_majorant_tail_certificate_over_w_interval(
        chart,
        retained_order=retained_order,
        compact_interval=FloatInterval.point(chart.center + delta),
        compact_radius=compact_radius,
    )


def compactified_sundman_interval_cauchy_majorant_tail_certificate_over_w_interval(
    chart: IntervalCompactifiedSundmanTaylorSolution,
    *,
    retained_order: int,
    compact_interval: FloatInterval,
    compact_radius: float | None = None,
) -> CompactifiedSundmanIntervalCauchyMajorantCertificate:
    """Build an interval Cauchy tail certificate covering a compact interval."""

    compact_interval = _as_interval(compact_interval)
    _validate_compact_interval_inside_chart(
        compact_interval,
        center=chart.center,
        analytic_radius=chart.analytic_radius,
    )
    step_size = max(
        abs(compact_interval.lower - chart.center),
        abs(compact_interval.upper - chart.center),
    )
    sundman_certificate = sundman_interval_cauchy_majorant_tail_certificate(
        chart.position[0],
        chart.velocity[0],
        chart.masses,
        retained_order=retained_order,
        step_size=0.0,
        distance_power=chart.distance_power,
    )
    compact_radius = _choose_compact_sundman_cauchy_radius(
        center=chart.center,
        sundman_rate=chart.sundman_rate,
        sundman_radius=sundman_certificate.s_radius,
        step_size=step_size,
        compact_radius=compact_radius,
    )
    sundman_image_radius = _compact_sundman_cauchy_image_radius(
        center=chart.center,
        sundman_rate=chart.sundman_rate,
        compact_radius=compact_radius,
    )
    center_s_interval = _as_interval(chart.sundman_time[0])
    center_s_bound = max(abs(center_s_interval.lower), abs(center_s_interval.upper))
    state_sup_bound = max(
        sundman_certificate.state_sup_bound,
        center_s_bound + sundman_image_radius,
    )
    ratio = step_size / compact_radius
    tail_bound = float(np.nextafter(state_sup_bound * ratio ** (retained_order + 1) / (1.0 - ratio), np.inf))
    return CompactifiedSundmanIntervalCauchyMajorantCertificate(
        retained_order=int(retained_order),
        step_size=float(step_size),
        compact_radius=float(compact_radius),
        sundman_image_radius=float(sundman_image_radius),
        center=float(chart.center),
        sundman_rate=float(chart.sundman_rate),
        state_sup_bound=float(state_sup_bound),
        tail_bound=float(tail_bound),
        sundman_certificate=sundman_certificate,
    )


def compactified_sundman_interval_guarded_tail_certificate(
    chart: IntervalCompactifiedSundmanTaylorSolution,
    *,
    retained_order: int,
    compact_parameter: float,
) -> TailBoundCertificate:
    """Build an interval guard-term tail certificate for a compactified-Sundman chart."""

    delta = chart.delta_from_compact_parameter(compact_parameter)
    arrays = [chart.position, chart.velocity, chart.physical_time[:, None], chart.sundman_time[:, None]]
    certificate = interval_guarded_tail_certificate(arrays, retained_order=retained_order, step_size=delta)
    scale = max(1.0, *(_interval_array_abs_bound(array) for array in arrays))
    roundoff_slack = 64.0 * np.finfo(float).eps * scale
    return TailBoundCertificate(
        retained_order=certificate.retained_order,
        computed_order=certificate.computed_order,
        step_size=certificate.step_size,
        first_omitted_term=certificate.first_omitted_term,
        observed_tail=certificate.observed_tail,
        ratio_bound=certificate.ratio_bound,
        tail_bound=float(np.nextafter(certificate.tail_bound + roundoff_slack, np.inf)),
        coefficient_source=certificate.coefficient_source,
    )


def compactified_sundman_interval_guarded_tail_certificate_over_w_interval(
    chart: IntervalCompactifiedSundmanTaylorSolution,
    *,
    retained_order: int,
    compact_interval: FloatInterval,
) -> TailBoundCertificate:
    """Guarded tail certificate covering a compactified-Sundman interval."""

    compact_interval = _as_interval(compact_interval)
    _validate_compact_interval_inside_chart(
        compact_interval,
        center=chart.center,
        analytic_radius=chart.analytic_radius,
    )
    step_size = max(
        abs(compact_interval.lower - chart.center),
        abs(compact_interval.upper - chart.center),
    )
    arrays = [chart.position, chart.velocity, chart.physical_time[:, None], chart.sundman_time[:, None]]
    certificate = interval_guarded_tail_certificate(arrays, retained_order=retained_order, step_size=step_size)
    scale = max(1.0, *(_interval_array_abs_bound(array) for array in arrays))
    roundoff_slack = 64.0 * np.finfo(float).eps * scale
    return TailBoundCertificate(
        retained_order=certificate.retained_order,
        computed_order=certificate.computed_order,
        step_size=certificate.step_size,
        first_omitted_term=certificate.first_omitted_term,
        observed_tail=certificate.observed_tail,
        ratio_bound=certificate.ratio_bound,
        tail_bound=float(np.nextafter(certificate.tail_bound + roundoff_slack, np.inf)),
        coefficient_source=certificate.coefficient_source,
    )


def certify_compactified_sundman_equations(
    chart: CompactifiedSundmanTaylorSolution,
    *,
    coefficient_count: int | None = None,
    tolerance: float = 1e-11,
) -> CompactifiedSundmanResidualCertificate:
    """Certify coefficient residuals for compactified Sundman equations."""

    if coefficient_count is None:
        coefficient_count = chart.order
    if coefficient_count < 1:
        raise ValueError("coefficient_count must be positive")
    if coefficient_count > chart.order:
        raise ValueError("coefficient_count cannot exceed chart.order")

    max_degree = coefficient_count - 1
    compact_factor = compact_time_factor_coefficients(
        max_degree,
        center=chart.center,
        time_rate=chart.sundman_rate,
    )
    distance_factor = sundman_factor_coefficients(
        chart.position,
        max_degree,
        distance_power=chart.distance_power,
    )
    combined_factor = scalar_series_product(compact_factor, distance_factor, max_degree)
    acceleration = acceleration_coefficients(chart.position, chart.masses, max_degree)
    position_rhs = _series_vector_product(combined_factor, chart.velocity, max_degree)
    velocity_rhs = _series_vector_product(combined_factor, acceleration, max_degree)

    position_residual = np.zeros_like(position_rhs)
    velocity_residual = np.zeros_like(velocity_rhs)
    physical_time_residual = np.zeros(coefficient_count, dtype=float)
    sundman_time_residual = np.zeros(coefficient_count, dtype=float)
    for degree in range(coefficient_count):
        scale = float(degree + 1)
        position_residual[degree] = scale * chart.position[degree + 1] - position_rhs[degree]
        velocity_residual[degree] = scale * chart.velocity[degree + 1] - velocity_rhs[degree]
        physical_time_residual[degree] = scale * chart.physical_time[degree + 1] - combined_factor[degree]
        sundman_time_residual[degree] = scale * chart.sundman_time[degree + 1] - compact_factor[degree]

    return CompactifiedSundmanResidualCertificate(
        coefficient_count=coefficient_count,
        position_residual=position_residual,
        velocity_residual=velocity_residual,
        physical_time_residual=physical_time_residual,
        sundman_time_residual=sundman_time_residual,
        tolerance=float(tolerance),
    )


def compactified_sundman_guarded_tail_certificate(
    chart: CompactifiedSundmanTaylorSolution,
    *,
    retained_order: int,
    compact_parameter: float,
) -> TailBoundCertificate:
    """Build a guard-term tail certificate for a compactified-Sundman chart."""

    delta = chart.delta_from_compact_parameter(compact_parameter)
    arrays = [chart.position, chart.velocity, chart.physical_time[:, None], chart.sundman_time[:, None]]
    certificate = guarded_tail_certificate(arrays, retained_order=retained_order, step_size=delta)
    scale = max(1.0, *(float(np.max(np.abs(array))) for array in arrays))
    roundoff_slack = 64.0 * np.finfo(float).eps * scale
    return TailBoundCertificate(
        retained_order=certificate.retained_order,
        computed_order=certificate.computed_order,
        step_size=certificate.step_size,
        first_omitted_term=certificate.first_omitted_term,
        observed_tail=certificate.observed_tail,
        ratio_bound=certificate.ratio_bound,
        tail_bound=float(np.nextafter(certificate.tail_bound + roundoff_slack, np.inf)),
        coefficient_source=certificate.coefficient_source,
    )


def compactified_sundman_truncation_indicator(
    chart: CompactifiedSundmanTaylorSolution,
    compact_parameter: float,
) -> float:
    """A local truncation proxy from the highest retained compact-Sundman term."""

    delta = chart.delta_from_compact_parameter(compact_parameter)
    power = abs(delta) ** chart.order
    position_tail = np.linalg.norm(chart.position[-1].reshape(-1), ord=np.inf) * power
    velocity_tail = np.linalg.norm(chart.velocity[-1].reshape(-1), ord=np.inf) * power
    physical_time_tail = abs(chart.physical_time[-1]) * power
    sundman_time_tail = abs(chart.sundman_time[-1]) * power
    return float(max(position_tail, velocity_tail, physical_time_tail, sundman_time_tail))


def reference_state_at_compactified_sundman_time(
    chart: CompactifiedSundmanTaylorSolution,
    compact_parameter: float,
) -> Array:
    """Numerically integrate Newtonian time reached by a compactified-Sundman chart."""

    return integrate_reference(
        chart.position[0],
        chart.velocity[0],
        chart.masses,
        chart.physical_time_delta_at_w(compact_parameter),
    )


def _truncate_chart(
    chart: CompactifiedSundmanTaylorSolution,
    order: int,
) -> CompactifiedSundmanTaylorSolution:
    if order < 0 or order > chart.order:
        raise ValueError("order must be inside the chart coefficient range")
    return CompactifiedSundmanTaylorSolution(
        position=chart.position[: order + 1].copy(),
        velocity=chart.velocity[: order + 1].copy(),
        physical_time=chart.physical_time[: order + 1].copy(),
        sundman_time=chart.sundman_time[: order + 1].copy(),
        masses=chart.masses.copy(),
        sundman_rate=chart.sundman_rate,
        distance_power=chart.distance_power,
        center=chart.center,
    )


def _truncate_interval_chart(
    chart: IntervalCompactifiedSundmanTaylorSolution,
    order: int,
) -> IntervalCompactifiedSundmanTaylorSolution:
    if order < 0 or order > chart.order:
        raise ValueError("order must be inside the chart coefficient range")
    return IntervalCompactifiedSundmanTaylorSolution(
        position=chart.position[: order + 1].copy(),
        velocity=chart.velocity[: order + 1].copy(),
        physical_time=chart.physical_time[: order + 1].copy(),
        sundman_time=chart.sundman_time[: order + 1].copy(),
        masses=chart.masses.copy(),
        sundman_rate=chart.sundman_rate,
        distance_power=chart.distance_power,
        center=chart.center,
    )


def _series_vector_product(scalar: Array, vector: Array, max_degree: int) -> Array:
    out = np.zeros_like(vector[: max_degree + 1])
    for degree in range(max_degree + 1):
        for index in range(degree + 1):
            out[degree] += scalar[index] * vector[degree - index]
    return out


def _interval_series_vector_product(scalar: tuple[FloatInterval, ...], vector: Array, max_degree: int) -> Array:
    out = _interval_zeros(vector[: max_degree + 1].shape)
    for degree in range(max_degree + 1):
        for coefficient_index in range(degree + 1):
            for index in np.ndindex(vector.shape[1:]):
                out[(degree, *index)] = out[(degree, *index)] + scalar[coefficient_index] * _as_interval(
                    vector[(degree - coefficient_index, *index)]
                )
    return out


def _evaluate(coefficients: Array, value: float) -> Array:
    out = np.zeros(coefficients.shape[1:], dtype=float)
    for coefficient in coefficients[::-1]:
        out = out * float(value) + coefficient
    return out


def _validate_rate(rate: float) -> float:
    rate = float(rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("sundman_rate must be a positive finite value")
    return rate


def _validate_distance_power(distance_power: float) -> float:
    distance_power = float(distance_power)
    if not np.isfinite(distance_power) or distance_power <= 0.0:
        raise ValueError("distance_power must be a positive finite value")
    return distance_power


def _validate_center(center: float) -> float:
    center = float(center)
    if not np.isfinite(center) or not -1.0 < center < 1.0:
        raise ValueError("center must lie strictly between -1 and 1")
    return center


def _compact_sundman_cover_segment_intervals(
    covers: tuple[CompactifiedSundmanCauchyCoverCertificate, ...],
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for cover in covers:
        for start, end in zip(cover.segment_starts, cover.segment_ends):
            lower = min(start, end)
            upper = max(start, end)
            if np.isfinite(lower) and np.isfinite(upper) and lower <= upper:
                intervals.append((float(lower), float(upper)))
    intervals.sort(key=lambda item: item[0])
    return intervals


def _compact_sundman_cover_max_gap(
    covers: tuple[CompactifiedSundmanCauchyCoverCertificate, ...],
    compact_interval: FloatInterval,
) -> float:
    cursor = float(compact_interval.lower)
    max_gap = 0.0
    for lower, upper in _compact_sundman_cover_segment_intervals(covers):
        if upper < compact_interval.lower or lower > compact_interval.upper:
            continue
        lower = max(lower, compact_interval.lower)
        upper = min(upper, compact_interval.upper)
        if upper < lower:
            continue
        if lower > cursor and not _compact_parameters_close(lower, cursor):
            max_gap = max(max_gap, lower - cursor)
        if upper > cursor:
            cursor = upper
        if cursor >= compact_interval.upper or _compact_parameters_close(cursor, compact_interval.upper):
            return max_gap
    if compact_interval.upper > cursor and not _compact_parameters_close(compact_interval.upper, cursor):
        max_gap = max(max_gap, compact_interval.upper - cursor)
    return float(max_gap)


def _compact_interval_contains_segment(
    compact_interval: FloatInterval,
    start: float,
    end: float,
) -> bool:
    compact_interval = _as_interval(compact_interval)
    lower = min(float(start), float(end))
    upper = max(float(start), float(end))
    tolerance = _compact_parameter_tolerance(compact_interval.lower, compact_interval.upper, lower, upper)
    return bool(
        lower >= compact_interval.lower - tolerance
        and upper <= compact_interval.upper + tolerance
    )


def _append_compact_sundman_cauchy_cover_segment(
    segment_starts: list[float],
    segment_ends: list[float],
    disk_centers: list[float],
    disk_radii: list[float],
    segment_tail_bounds: list[float],
    segment_retained_orders: list[int],
    segment_state_sup_bounds: list[float],
    coefficient_sources: list[str],
    certificate_nontrivial: list[bool],
    *,
    start: float,
    end: float,
    certificate: CompactifiedSundmanTailCertificate | None,
) -> None:
    segment_starts.append(float(start))
    segment_ends.append(float(end))
    if certificate is None:
        disk_centers.append(float("nan"))
        disk_radii.append(float("nan"))
        segment_tail_bounds.append(float("nan"))
        segment_retained_orders.append(-1)
        segment_state_sup_bounds.append(float("nan"))
        coefficient_sources.append("missing")
        certificate_nontrivial.append(False)
        return

    disk_centers.append(float(getattr(certificate, "center", float("nan"))))
    disk_radii.append(float(getattr(certificate, "compact_radius", float("nan"))))
    segment_tail_bounds.append(float(getattr(certificate, "tail_bound", float("nan"))))
    segment_retained_orders.append(int(getattr(certificate, "retained_order", -1)))
    segment_state_sup_bounds.append(float(getattr(certificate, "state_sup_bound", float("nan"))))
    coefficient_sources.append(str(getattr(certificate, "coefficient_source", "unknown")))
    certificate_nontrivial.append(bool(getattr(certificate, "is_nontrivial", False)))


def _compact_parameters_close(left: float, right: float) -> bool:
    return abs(left - right) <= _compact_parameter_tolerance(left, right)


def _compact_parameter_tolerance(*values: float) -> float:
    scale = max(1.0, *(abs(value) for value in values))
    return float(128.0 * np.finfo(float).eps * scale)


def _cover_direction(start: float, end: float) -> float:
    if _compact_parameters_close(start, end):
        return 0.0
    return float(np.sign(end - start))


def _validate_compact_interval_inside_chart(
    compact_interval: FloatInterval,
    *,
    center: float,
    analytic_radius: float,
) -> None:
    if compact_interval.lower <= -1.0 or compact_interval.upper >= 1.0:
        raise ValueError("compact_interval must lie strictly between -1 and 1")
    if (
        abs(compact_interval.lower - center) >= analytic_radius
        or abs(compact_interval.upper - center) >= analytic_radius
    ):
        raise ValueError("compact_interval must lie inside this chart's compact-time convergence disk")


def _compact_sundman_cauchy_image_radius(
    *,
    center: float,
    sundman_rate: float,
    compact_radius: float,
) -> float:
    """Bound ``|atanh(w) - atanh(center)| / rate`` on a compact disk."""

    center = _validate_center(center)
    sundman_rate = _validate_rate(sundman_rate)
    compact_radius = float(compact_radius)
    if not np.isfinite(compact_radius) or compact_radius <= 0.0:
        raise ValueError("compact_radius must be positive")
    boundary = abs(center) + compact_radius
    if boundary >= 1.0:
        raise ValueError("compact_radius must keep the compact disk away from +/-1")
    denominator = 1.0 - boundary * boundary
    return float(compact_radius / (sundman_rate * denominator))


def _choose_compact_sundman_cauchy_radius(
    *,
    center: float,
    sundman_rate: float,
    sundman_radius: float,
    step_size: float,
    compact_radius: float | None,
) -> float:
    center = _validate_center(center)
    sundman_rate = _validate_rate(sundman_rate)
    sundman_radius = float(sundman_radius)
    step_size = abs(float(step_size))
    if not np.isfinite(sundman_radius) or sundman_radius <= 0.0:
        raise ValueError("sundman_radius must be positive")
    if not np.isfinite(step_size):
        raise ValueError("step_size must be finite")
    analytic_margin = 1.0 - abs(center)
    if step_size >= analytic_margin:
        raise ValueError("step_size must lie inside the compact chart disk")

    def image_radius(radius: float) -> float:
        return _compact_sundman_cauchy_image_radius(
            center=center,
            sundman_rate=sundman_rate,
            compact_radius=radius,
        )

    if compact_radius is not None:
        compact_radius = float(compact_radius)
        if not np.isfinite(compact_radius) or compact_radius <= step_size:
            raise ValueError("compact_radius must be finite and larger than the requested step")
        if compact_radius >= analytic_margin:
            raise ValueError("compact_radius must keep the compact disk away from +/-1")
        if image_radius(compact_radius) > sundman_radius:
            raise ValueError("compact_radius maps outside the certified Sundman disk")
        return float(compact_radius)

    lower = step_size
    if lower > 0.0 and image_radius(lower) >= sundman_radius:
        raise ValueError("step_size maps outside the certified Sundman disk")
    upper = float(np.nextafter(analytic_margin, 0.0))
    if upper <= lower:
        raise ValueError("no compact Cauchy radius remains beyond the requested step")
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if image_radius(middle) <= sundman_radius:
            lower = middle
        else:
            upper = middle
    if lower <= step_size:
        raise ValueError("no compact Cauchy radius remains beyond the requested step")
    return float(lower)


def _cap_compact_step_to_cauchy_radius(
    compact_step: float,
    *,
    compact_radius: float,
) -> float:
    compact_step = float(compact_step)
    compact_radius = float(compact_radius)
    if not np.isfinite(compact_step) or compact_step == 0.0:
        raise ValueError("compact_step must be a nonzero finite value")
    if not np.isfinite(compact_radius) or compact_radius <= 0.0:
        raise ValueError("compact_radius must be positive")
    capped_magnitude = min(abs(compact_step), 0.5 * compact_radius)
    if capped_magnitude <= 10.0 * np.finfo(float).eps:
        raise RuntimeError("compactified Sundman Cauchy step underflowed")
    return float(np.sign(compact_step) * capped_magnitude)


def _choose_directed_compact_step(
    current_w: float,
    direction: float,
    *,
    max_compact_step: float,
    radius_fraction: float,
) -> float:
    if direction == 0.0:
        raise ValueError("direction cannot be zero")
    boundary_margin = 1.0 - current_w if direction > 0.0 else current_w + 1.0
    if boundary_margin <= 0.0:
        raise RuntimeError("compactified Sundman parameter reached the boundary before target time")
    analytic_margin = 1.0 - abs(current_w)
    magnitude = min(max_compact_step, radius_fraction * analytic_margin, 0.5 * boundary_margin)
    if magnitude <= 10.0 * np.finfo(float).eps:
        raise RuntimeError("compactified Sundman step underflowed before target time")
    return float(np.sign(direction) * magnitude)


def _as_interval(value: object) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


def _interval_zeros(shape: tuple[int, ...]) -> Array:
    out = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        out[index] = zero_interval()
    return out


def _interval_array_from_points(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    out = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        out[index] = FloatInterval.point(values[index])
    return out


def _inflate_interval(value: FloatInterval, radius: float) -> FloatInterval:
    radius = float(radius)
    if radius < 0.0:
        raise ValueError("radius cannot be negative")
    return FloatInterval(
        float(np.nextafter(value.lower - radius, -np.inf)),
        float(np.nextafter(value.upper + radius, np.inf)),
    )


def _inflate_interval_array(values: Array, radius: float) -> Array:
    values = np.asarray(values, dtype=object)
    out = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        out[index] = _inflate_interval(_as_interval(values[index]), radius)
    return out


def _interval_state(positions: Array, velocities: Array) -> Array:
    return np.concatenate([np.asarray(positions, dtype=object).reshape(-1), np.asarray(velocities, dtype=object).reshape(-1)])


def _split_state_interval(state: Array, *, body_count: int, dimension: int) -> tuple[Array, Array]:
    state = np.asarray(state, dtype=object)
    coordinate_count = body_count * dimension
    expected_count = 2 * coordinate_count
    if state.shape != (expected_count,):
        raise ValueError("state interval does not match the requested body count and dimension")
    positions = np.array(state[:coordinate_count], dtype=object).reshape((body_count, dimension))
    velocities = np.array(state[coordinate_count:], dtype=object).reshape((body_count, dimension))
    return positions, velocities


def _interval_array_contains_zero(values: Array) -> bool:
    values = np.asarray(values, dtype=object)
    for index in np.ndindex(values.shape):
        interval = _as_interval(values[index])
        if not interval.lower <= 0.0 <= interval.upper:
            return False
    return True


def _interval_array_abs_bound(values: Array) -> float:
    values = np.asarray(values, dtype=object)
    bound = 0.0
    for index in np.ndindex(values.shape):
        interval = _as_interval(values[index])
        bound = max(bound, abs(interval.lower), abs(interval.upper))
    return float(bound)


def _interval_square(value: FloatInterval) -> FloatInterval:
    squares = (value.lower * value.lower, value.upper * value.upper)
    upper = float(np.nextafter(max(squares), np.inf))
    if value.lower <= 0.0 <= value.upper:
        lower = 0.0
    else:
        lower = max(0.0, float(np.nextafter(min(squares), -np.inf)))
    return FloatInterval(lower, upper)


def _validate_initial_data(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 2 or positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shape (body_count, dimension)")
    if positions.shape[0] != 3 or masses.shape != (3,):
        raise ValueError("expected three bodies and three masses")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive")
    if pairwise_distance_product(positions) == 0.0:
        raise ValueError("initial data must be collision-free")
    return positions, velocities, masses


def _validate_interval_initial_data(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=object)
    velocities = np.asarray(velocities, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 2:
        raise ValueError("positions must have shape (body_count, dimension)")
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    if positions.shape[0] != 3 or masses.shape != (3,):
        raise ValueError("expected three bodies and three masses")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be positive")

    interval_positions = np.empty(positions.shape, dtype=object)
    interval_velocities = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        interval_positions[index] = _as_interval(positions[index])
        interval_velocities[index] = _as_interval(velocities[index])

    for first in range(3):
        for second in range(first + 1, 3):
            distance_squared = FloatInterval.point(0.0)
            for axis in range(positions.shape[1]):
                distance_squared = distance_squared + _interval_square(
                    interval_positions[second, axis] - interval_positions[first, axis]
                )
            if distance_squared.lower <= 0.0:
                raise ValueError("interval initial data must certify non-collision")
    return interval_positions, interval_velocities, masses
