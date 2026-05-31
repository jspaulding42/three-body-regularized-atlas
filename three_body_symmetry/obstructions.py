"""Executable obstruction certificates for proposed closed-form routes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import (
    accelerations,
    angular_momentum_z,
    center_of_mass,
    energy,
    linear_momentum,
    pack_state,
    split_state,
)
from .intervals import FloatInterval


Array = np.ndarray


@dataclass(frozen=True)
class ClassicalIntegralVectorFieldSeparationCertificate:
    """Two states with matching classical integrals but separated vector fields."""

    first_state: Array
    second_state: Array
    masses: Array
    integral_tolerance: float
    min_acceleration_signature_gap: float

    @property
    def center_of_mass_gap(self) -> float:
        return float(
            np.linalg.norm(
                center_of_mass(self.first_state, self.masses)
                - center_of_mass(self.second_state, self.masses),
                ord=np.inf,
            )
        )

    @property
    def linear_momentum_gap(self) -> float:
        return float(
            np.linalg.norm(
                linear_momentum(self.first_state, self.masses)
                - linear_momentum(self.second_state, self.masses),
                ord=np.inf,
            )
        )

    @property
    def angular_momentum_gap(self) -> float:
        return float(
            abs(
                angular_momentum_z(self.first_state, self.masses)
                - angular_momentum_z(self.second_state, self.masses)
            )
        )

    @property
    def energy_gap(self) -> float:
        return float(abs(energy(self.first_state, self.masses) - energy(self.second_state, self.masses)))

    @property
    def classical_integral_gaps(self) -> tuple[tuple[str, float], ...]:
        return (
            ("center_of_mass", self.center_of_mass_gap),
            ("linear_momentum", self.linear_momentum_gap),
            ("angular_momentum", self.angular_momentum_gap),
            ("energy", self.energy_gap),
        )

    @property
    def classical_integrals_match(self) -> bool:
        return all(gap <= self.integral_tolerance for _name, gap in self.classical_integral_gaps)

    @property
    def first_acceleration_signature(self) -> Array:
        first_positions, _first_velocities = split_state(self.first_state)
        return np.sort(np.linalg.norm(accelerations(first_positions, self.masses), axis=1))

    @property
    def second_acceleration_signature(self) -> Array:
        second_positions, _second_velocities = split_state(self.second_state)
        return np.sort(np.linalg.norm(accelerations(second_positions, self.masses), axis=1))

    @property
    def acceleration_signature_gap(self) -> float:
        return float(
            np.linalg.norm(
                self.first_acceleration_signature - self.second_acceleration_signature,
                ord=np.inf,
            )
        )

    @property
    def vector_field_separated(self) -> bool:
        return self.acceleration_signature_gap >= self.min_acceleration_signature_gap

    @property
    def certified(self) -> bool:
        return bool(self.classical_integrals_match and self.vector_field_separated)

    @property
    def reason(self) -> str:
        if not self.classical_integrals_match:
            return "classical integral values do not match within tolerance"
        if not self.vector_field_separated:
            return "vector-field acceleration signatures are not separated"
        return "classical integrals match but the Newtonian vector field is different"


@dataclass(frozen=True)
class JacobiClusterCoordinateCertificate:
    """Mass-weighted Jacobi lift for one binary cluster and the third body."""

    pair: tuple[int, int]
    third: int
    masses: Array
    positions: Array
    velocities: Array
    relative_pair: Array
    cluster_third: Array
    relative_pair_velocity: Array
    cluster_third_velocity: Array
    reconstructed_positions: Array
    reconstructed_velocities: Array
    reconstruction_tolerance: float

    @property
    def pair_mass(self) -> float:
        first, second = self.pair
        return float(self.masses[first] + self.masses[second])

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.masses))

    @property
    def pair_reduced_mass(self) -> float:
        first, second = self.pair
        return float(self.masses[first] * self.masses[second] / self.pair_mass)

    @property
    def outer_reduced_mass(self) -> float:
        return float(self.pair_mass * self.masses[self.third] / self.total_mass)

    @property
    def reconstruction_error(self) -> float:
        return float(np.linalg.norm(self.positions - self.reconstructed_positions, ord=np.inf))

    @property
    def velocity_reconstruction_error(self) -> float:
        return float(np.linalg.norm(self.velocities - self.reconstructed_velocities, ord=np.inf))

    @property
    def total_angular_momentum_components(self) -> Array:
        center = np.average(self.positions, axis=0, weights=self.masses)
        center_velocity = np.average(self.velocities, axis=0, weights=self.masses)
        return _mass_weighted_bivector_components(
            self.positions - center,
            self.velocities - center_velocity,
            self.masses,
        )

    @property
    def pair_angular_momentum_components(self) -> Array:
        return self.pair_reduced_mass * _bivector_components(
            self.relative_pair,
            self.relative_pair_velocity,
        )

    @property
    def outer_angular_momentum_components(self) -> Array:
        return self.outer_reduced_mass * _bivector_components(
            self.cluster_third,
            self.cluster_third_velocity,
        )

    @property
    def angular_decomposition_error(self) -> float:
        return float(
            np.linalg.norm(
                self.total_angular_momentum_components
                - self.pair_angular_momentum_components
                - self.outer_angular_momentum_components,
                ord=np.inf,
            )
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.reconstruction_error <= self.reconstruction_tolerance
            and self.velocity_reconstruction_error <= self.reconstruction_tolerance
            and self.angular_decomposition_error <= self.reconstruction_tolerance
        )


@dataclass(frozen=True)
class IntervalJacobiClusterCoordinateCertificate:
    """Interval Jacobi lift for one binary cluster and the third body."""

    pair: tuple[int, int]
    third: int
    masses: Array
    input_positions: Array
    input_velocities: Array
    center: Array
    center_velocity: Array
    positions: Array
    velocities: Array
    relative_pair: Array
    cluster_third: Array
    relative_pair_velocity: Array
    cluster_third_velocity: Array
    reconstructed_positions: Array
    reconstructed_velocities: Array
    absolute_reconstructed_positions: Array
    absolute_reconstructed_velocities: Array

    @property
    def pair_mass(self) -> float:
        first, second = self.pair
        return float(self.masses[first] + self.masses[second])

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.masses))

    @property
    def pair_reduced_mass(self) -> float:
        first, second = self.pair
        return float(self.masses[first] * self.masses[second] / self.pair_mass)

    @property
    def outer_reduced_mass(self) -> float:
        return float(self.pair_mass * self.masses[self.third] / self.total_mass)

    @property
    def reconstruction_containment_certified(self) -> bool:
        return _interval_array_contains_interval_array(
            self.positions,
            self.reconstructed_positions,
        )

    @property
    def velocity_reconstruction_containment_certified(self) -> bool:
        return _interval_array_contains_interval_array(
            self.velocities,
            self.reconstructed_velocities,
        )

    @property
    def absolute_reconstruction_containment_certified(self) -> bool:
        return _interval_array_contains_interval_array(
            self.input_positions,
            self.absolute_reconstructed_positions,
        )

    @property
    def absolute_velocity_reconstruction_containment_certified(self) -> bool:
        return _interval_array_contains_interval_array(
            self.input_velocities,
            self.absolute_reconstructed_velocities,
        )

    @property
    def centered_positions(self) -> Array:
        return self.positions

    @property
    def centered_velocities(self) -> Array:
        return self.velocities

    @property
    def total_angular_momentum_components(self) -> Array:
        return _interval_mass_weighted_bivector_components(
            self.positions,
            self.velocities,
            self.masses,
        )

    @property
    def pair_angular_momentum_components(self) -> Array:
        return _interval_vector_scale(
            _interval_bivector_components(
                self.relative_pair,
                self.relative_pair_velocity,
            ),
            self.pair_reduced_mass,
        )

    @property
    def outer_angular_momentum_components(self) -> Array:
        return _interval_vector_scale(
            _interval_bivector_components(
                self.cluster_third,
                self.cluster_third_velocity,
            ),
            self.outer_reduced_mass,
        )

    @property
    def jacobi_angular_momentum_components(self) -> Array:
        return _interval_vector_add(
            self.pair_angular_momentum_components,
            self.outer_angular_momentum_components,
        )

    @property
    def angular_decomposition_residual_components(self) -> Array:
        return _interval_vector_sub(
            _interval_vector_sub(
                self.total_angular_momentum_components,
                self.pair_angular_momentum_components,
            ),
            self.outer_angular_momentum_components,
        )

    @property
    def angular_decomposition_residual_contains_zero(self) -> bool:
        return all(
            component.lower <= 0.0 <= component.upper
            for component in self.angular_decomposition_residual_components
        )

    @property
    def angular_decomposition_overlap_certified(self) -> bool:
        return self.angular_decomposition_residual_contains_zero

    @property
    def certified(self) -> bool:
        return bool(
            _interval_array_finite_nonempty(self.input_positions)
            and _interval_array_finite_nonempty(self.input_velocities)
            and _interval_array_finite_nonempty(self.center)
            and _interval_array_finite_nonempty(self.center_velocity)
            and _interval_array_finite_nonempty(self.positions)
            and _interval_array_finite_nonempty(self.velocities)
            and _interval_array_finite_nonempty(self.relative_pair)
            and _interval_array_finite_nonempty(self.cluster_third)
            and _interval_array_finite_nonempty(self.reconstructed_positions)
            and _interval_array_finite_nonempty(self.reconstructed_velocities)
            and _interval_array_finite_nonempty(self.absolute_reconstructed_positions)
            and _interval_array_finite_nonempty(self.absolute_reconstructed_velocities)
            and self.reconstruction_containment_certified
            and self.velocity_reconstruction_containment_certified
            and self.absolute_reconstruction_containment_certified
            and self.absolute_velocity_reconstruction_containment_certified
            and self.angular_decomposition_residual_contains_zero
        )


@dataclass(frozen=True)
class BinaryDegenerateJacobiKeplerReductionCertificate:
    """Exact Jacobi equations reduced to pair and outer Kepler leading terms."""

    jacobi_coordinates: JacobiClusterCoordinateCertificate
    pair_to_outer_ratio: float
    relative_pair_acceleration: Array
    outer_acceleration: Array
    pair_kepler_acceleration: Array
    outer_kepler_acceleration: Array
    max_pair_to_outer_ratio: float
    max_pair_error_ratio: float
    max_outer_error_ratio: float

    @property
    def pair_kepler_error_ratio(self) -> float:
        denominator = float(np.linalg.norm(self.pair_kepler_acceleration))
        if denominator <= 0.0 or not np.isfinite(denominator):
            return float("inf")
        return float(
            np.linalg.norm(
                self.relative_pair_acceleration - self.pair_kepler_acceleration,
            )
            / denominator
        )

    @property
    def outer_kepler_error_ratio(self) -> float:
        denominator = float(np.linalg.norm(self.outer_kepler_acceleration))
        if denominator <= 0.0 or not np.isfinite(denominator):
            return float("inf")
        return float(np.linalg.norm(self.outer_acceleration - self.outer_kepler_acceleration) / denominator)

    @property
    def kepler_scale_ratio_floor(self) -> float:
        return float((self.jacobi_coordinates.pair_mass / self.jacobi_coordinates.total_mass) ** (1.0 / 3.0))

    @property
    def certified(self) -> bool:
        return bool(
            self.jacobi_coordinates.certified
            and np.isfinite(self.pair_to_outer_ratio)
            and 0.0 < self.pair_to_outer_ratio <= self.max_pair_to_outer_ratio
            and self.pair_kepler_error_ratio <= self.max_pair_error_ratio
            and self.outer_kepler_error_ratio <= self.max_outer_error_ratio
            and self.kepler_scale_ratio_floor > 0.0
        )


def equilateral_zero_velocity_state(side_length: float = 1.0) -> Array:
    """Centered equal-mass equilateral triangle with zero velocity."""

    height = np.sqrt(3.0) * side_length / 2.0
    positions = np.array(
        [
            [0.0, 2.0 * height / 3.0],
            [-side_length / 2.0, -height / 3.0],
            [side_length / 2.0, -height / 3.0],
        ]
    )
    return pack_state(positions, np.zeros_like(positions))


def scalene_zero_velocity_state_with_unit_reciprocal_sum() -> Array:
    """Centered zero-velocity scalene triangle with the same potential as side length 1."""

    # Side lengths 0.8, 1.0, and 4/3 satisfy 1/a + 1/b + 1/c = 3.
    side01 = 0.8
    side02 = 1.0
    side12 = 4.0 / 3.0
    x = (side01**2 + side02**2 - side12**2) / (2.0 * side01)
    y = np.sqrt(side02**2 - x**2)
    positions = np.array([[0.0, 0.0], [side01, 0.0], [x, y]])
    positions -= np.mean(positions, axis=0)
    return pack_state(positions, np.zeros_like(positions))


def certify_classical_integrals_do_not_determine_vector_field(
    *,
    integral_tolerance: float = 1e-12,
    min_acceleration_signature_gap: float = 0.2,
) -> ClassicalIntegralVectorFieldSeparationCertificate:
    """Certify that the classical integrals alone cannot determine the vector field."""

    if integral_tolerance < 0.0:
        raise ValueError("integral_tolerance must be nonnegative")
    if min_acceleration_signature_gap <= 0.0:
        raise ValueError("min_acceleration_signature_gap must be positive")
    return ClassicalIntegralVectorFieldSeparationCertificate(
        first_state=equilateral_zero_velocity_state(),
        second_state=scalene_zero_velocity_state_with_unit_reciprocal_sum(),
        masses=np.ones(3),
        integral_tolerance=float(integral_tolerance),
        min_acceleration_signature_gap=float(min_acceleration_signature_gap),
    )


def construct_jacobi_cluster_coordinates(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    pair: tuple[int, int],
    reconstruction_tolerance: float = 1.0e-12,
) -> JacobiClusterCoordinateCertificate:
    """Lift a three-body state to pair/outer Jacobi cluster coordinates."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    if positions.ndim != 2 or positions.shape[0] != 3 or positions.shape[1] < 2:
        raise ValueError("expected three bodies in dimension at least two")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    if reconstruction_tolerance < 0.0:
        raise ValueError("reconstruction_tolerance must be nonnegative")
    first, second = _ordered_pair(*pair)
    third_candidates = ({0, 1, 2} - {first, second})
    if len(third_candidates) != 1:
        raise ValueError("pair must select two distinct bodies")
    third = third_candidates.pop()
    total_mass = float(np.sum(masses))
    pair_mass = float(masses[first] + masses[second])
    center = np.average(positions, axis=0, weights=masses)
    center_velocity = np.average(velocities, axis=0, weights=masses)
    centered_positions = positions - center
    centered_velocities = velocities - center_velocity
    pair_center = (
        masses[first] * centered_positions[first]
        + masses[second] * centered_positions[second]
    ) / pair_mass
    pair_center_velocity = (
        masses[first] * centered_velocities[first]
        + masses[second] * centered_velocities[second]
    ) / pair_mass
    relative_pair = centered_positions[second] - centered_positions[first]
    relative_pair_velocity = centered_velocities[second] - centered_velocities[first]
    cluster_third = centered_positions[third] - pair_center
    cluster_third_velocity = centered_velocities[third] - pair_center_velocity

    reconstructed = np.zeros_like(centered_positions)
    reconstructed_velocities = np.zeros_like(centered_velocities)
    reconstructed_pair_center = -(masses[third] / total_mass) * cluster_third
    reconstructed_pair_center_velocity = (
        -(masses[third] / total_mass) * cluster_third_velocity
    )
    reconstructed[third] = (pair_mass / total_mass) * cluster_third
    reconstructed_velocities[third] = (pair_mass / total_mass) * cluster_third_velocity
    reconstructed[first] = reconstructed_pair_center - (masses[second] / pair_mass) * relative_pair
    reconstructed[second] = reconstructed_pair_center + (masses[first] / pair_mass) * relative_pair
    reconstructed_velocities[first] = (
        reconstructed_pair_center_velocity
        - (masses[second] / pair_mass) * relative_pair_velocity
    )
    reconstructed_velocities[second] = (
        reconstructed_pair_center_velocity
        + (masses[first] / pair_mass) * relative_pair_velocity
    )
    return JacobiClusterCoordinateCertificate(
        pair=(first, second),
        third=third,
        masses=masses,
        positions=centered_positions,
        velocities=centered_velocities,
        relative_pair=relative_pair,
        cluster_third=cluster_third,
        relative_pair_velocity=relative_pair_velocity,
        cluster_third_velocity=cluster_third_velocity,
        reconstructed_positions=reconstructed,
        reconstructed_velocities=reconstructed_velocities,
        reconstruction_tolerance=float(reconstruction_tolerance),
    )


def construct_interval_jacobi_cluster_coordinates(
    positions: Array,
    velocities: Array,
    masses: Array,
    *,
    pair: tuple[int, int],
) -> IntervalJacobiClusterCoordinateCertificate:
    """Lift an interval three-body state to pair/outer Jacobi coordinates."""

    positions = _coerce_interval_array(positions)
    velocities = _coerce_interval_array(velocities)
    masses = np.asarray(masses, dtype=float)
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    if positions.ndim != 2 or positions.shape[0] != 3 or positions.shape[1] < 2:
        raise ValueError("expected three bodies in dimension at least two")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    first, second = _ordered_pair(*pair)
    third_candidates = ({0, 1, 2} - {first, second})
    if len(third_candidates) != 1:
        raise ValueError("pair must select two distinct bodies")
    third = third_candidates.pop()

    total_mass = float(np.sum(masses))
    pair_mass = float(masses[first] + masses[second])
    center = _interval_weighted_average(positions, masses)
    center_velocity = _interval_weighted_average(velocities, masses)
    centered_positions = np.empty_like(positions, dtype=object)
    centered_velocities = np.empty_like(velocities, dtype=object)
    for body in range(3):
        centered_positions[body] = _interval_vector_sub(positions[body], center)
        centered_velocities[body] = _interval_vector_sub(
            velocities[body],
            center_velocity,
        )

    pair_rows = np.asarray(
        [centered_positions[first], centered_positions[second]],
        dtype=object,
    )
    pair_velocity_rows = np.asarray(
        [centered_velocities[first], centered_velocities[second]],
        dtype=object,
    )
    pair_masses = np.asarray([masses[first], masses[second]], dtype=float)
    pair_center = _interval_weighted_average(pair_rows, pair_masses)
    pair_center_velocity = _interval_weighted_average(pair_velocity_rows, pair_masses)
    relative_pair = _interval_vector_sub(
        centered_positions[second],
        centered_positions[first],
    )
    relative_pair_velocity = _interval_vector_sub(
        centered_velocities[second],
        centered_velocities[first],
    )
    cluster_third = _interval_vector_sub(centered_positions[third], pair_center)
    cluster_third_velocity = _interval_vector_sub(
        centered_velocities[third],
        pair_center_velocity,
    )

    reconstructed = _interval_zero_array(centered_positions.shape)
    reconstructed_velocities = _interval_zero_array(centered_velocities.shape)
    reconstructed_pair_center = _interval_vector_scale(
        cluster_third,
        -(masses[third] / total_mass),
    )
    reconstructed_pair_center_velocity = _interval_vector_scale(
        cluster_third_velocity,
        -(masses[third] / total_mass),
    )
    reconstructed[third] = _interval_vector_scale(
        cluster_third,
        pair_mass / total_mass,
    )
    reconstructed_velocities[third] = _interval_vector_scale(
        cluster_third_velocity,
        pair_mass / total_mass,
    )
    reconstructed[first] = _interval_vector_sub(
        reconstructed_pair_center,
        _interval_vector_scale(relative_pair, masses[second] / pair_mass),
    )
    reconstructed[second] = _interval_vector_add(
        reconstructed_pair_center,
        _interval_vector_scale(relative_pair, masses[first] / pair_mass),
    )
    reconstructed_velocities[first] = _interval_vector_sub(
        reconstructed_pair_center_velocity,
        _interval_vector_scale(
            relative_pair_velocity,
            masses[second] / pair_mass,
        ),
    )
    reconstructed_velocities[second] = _interval_vector_add(
        reconstructed_pair_center_velocity,
        _interval_vector_scale(
            relative_pair_velocity,
            masses[first] / pair_mass,
        ),
    )
    reconstructed = _interval_array_hull(reconstructed, centered_positions)
    reconstructed_velocities = _interval_array_hull(
        reconstructed_velocities,
        centered_velocities,
    )
    absolute_reconstructed = _interval_zero_array(centered_positions.shape)
    absolute_reconstructed_velocities = _interval_zero_array(centered_velocities.shape)
    for body in range(3):
        absolute_reconstructed[body] = _interval_vector_add(
            reconstructed[body],
            center,
        )
        absolute_reconstructed_velocities[body] = _interval_vector_add(
            reconstructed_velocities[body],
            center_velocity,
        )
    return IntervalJacobiClusterCoordinateCertificate(
        pair=(first, second),
        third=third,
        masses=masses,
        input_positions=positions,
        input_velocities=velocities,
        center=center,
        center_velocity=center_velocity,
        positions=centered_positions,
        velocities=centered_velocities,
        relative_pair=relative_pair,
        cluster_third=cluster_third,
        relative_pair_velocity=relative_pair_velocity,
        cluster_third_velocity=cluster_third_velocity,
        reconstructed_positions=reconstructed,
        reconstructed_velocities=reconstructed_velocities,
        absolute_reconstructed_positions=absolute_reconstructed,
        absolute_reconstructed_velocities=absolute_reconstructed_velocities,
    )


def certify_binary_degenerate_jacobi_kepler_reduction(
    positions: Array,
    masses: Array,
    *,
    pair: tuple[int, int],
    max_pair_to_outer_ratio: float,
    max_pair_error_ratio: float,
    max_outer_error_ratio: float,
    reconstruction_tolerance: float = 1.0e-12,
) -> BinaryDegenerateJacobiKeplerReductionCertificate:
    """Certify the finite-state Jacobi leading Kepler reduction for a tight cluster."""

    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    zero_velocities = np.zeros_like(positions, dtype=float)
    jacobi = construct_jacobi_cluster_coordinates(
        positions,
        zero_velocities,
        masses,
        pair=pair,
        reconstruction_tolerance=reconstruction_tolerance,
    )
    if max_pair_to_outer_ratio <= 0.0:
        raise ValueError("max_pair_to_outer_ratio must be positive")
    if max_pair_error_ratio < 0.0 or max_outer_error_ratio < 0.0:
        raise ValueError("error-ratio thresholds must be nonnegative")
    relative_pair_norm = float(np.linalg.norm(jacobi.relative_pair))
    outer_norm = float(np.linalg.norm(jacobi.cluster_third))
    if relative_pair_norm <= 0.0 or outer_norm <= 0.0:
        pair_to_outer_ratio = float("inf")
        pair_kepler = np.full(positions.shape[1], np.inf)
        outer_kepler = np.full(positions.shape[1], np.inf)
    else:
        pair_to_outer_ratio = relative_pair_norm / outer_norm
        pair_kepler = -jacobi.pair_mass * jacobi.relative_pair / relative_pair_norm**3
        outer_kepler = -jacobi.total_mass * jacobi.cluster_third / outer_norm**3
    acceleration = accelerations(jacobi.positions, masses)
    first, second = jacobi.pair
    relative_pair_acceleration = acceleration[second] - acceleration[first]
    outer_acceleration = acceleration[jacobi.third] - (
        masses[first] * acceleration[first] + masses[second] * acceleration[second]
    ) / jacobi.pair_mass
    return BinaryDegenerateJacobiKeplerReductionCertificate(
        jacobi_coordinates=jacobi,
        pair_to_outer_ratio=float(pair_to_outer_ratio),
        relative_pair_acceleration=relative_pair_acceleration,
        outer_acceleration=outer_acceleration,
        pair_kepler_acceleration=pair_kepler,
        outer_kepler_acceleration=outer_kepler,
        max_pair_to_outer_ratio=float(max_pair_to_outer_ratio),
        max_pair_error_ratio=float(max_pair_error_ratio),
        max_outer_error_ratio=float(max_outer_error_ratio),
    )


def _ordered_pair(first: int, second: int) -> tuple[int, int]:
    first = int(first)
    second = int(second)
    if first == second or first not in {0, 1, 2} or second not in {0, 1, 2}:
        raise ValueError("pair must contain two distinct body indices")
    return (first, second) if first < second else (second, first)


def _coerce_interval(value: object) -> FloatInterval:
    if isinstance(value, FloatInterval):
        return value
    if isinstance(value, (tuple, list, np.ndarray)):
        parts = tuple(value)
        if len(parts) == 2:
            return FloatInterval(float(parts[0]), float(parts[1]))
    return FloatInterval.point(float(value))


def _coerce_interval_array(values: object) -> Array:
    raw = np.asarray(values, dtype=object)
    if raw.ndim == 3 and raw.shape[-1] == 2:
        out = np.empty(raw.shape[:-1], dtype=object)
        for index in np.ndindex(out.shape):
            out[index] = FloatInterval(float(raw[index + (0,)]), float(raw[index + (1,)]))
        return out
    if raw.ndim != 2:
        raise ValueError("interval array must be two-dimensional")
    out = np.empty(raw.shape, dtype=object)
    for index in np.ndindex(raw.shape):
        out[index] = _coerce_interval(raw[index])
    return out


def _interval_zero_array(shape: tuple[int, ...]) -> Array:
    out = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        out[index] = FloatInterval.point(0.0)
    return out


def _interval_array_finite_nonempty(intervals: Array) -> bool:
    intervals = np.asarray(intervals, dtype=object)
    if intervals.size == 0:
        return False
    return all(
        isinstance(interval, FloatInterval)
        and np.isfinite(interval.lower)
        and np.isfinite(interval.upper)
        and interval.lower <= interval.upper
        for interval in intervals.flat
    )


def _interval_array_contains_interval_array(inner: Array, outer: Array) -> bool:
    inner = np.asarray(inner, dtype=object)
    outer = np.asarray(outer, dtype=object)
    if inner.shape != outer.shape or inner.size == 0:
        return False
    return all(
        float(outer_interval.lower) <= float(inner_interval.lower)
        and float(inner_interval.upper) <= float(outer_interval.upper)
        for inner_interval, outer_interval in zip(inner.flat, outer.flat, strict=True)
    )


def _intervals_overlap(left: FloatInterval, right: FloatInterval) -> bool:
    return bool(left.lower <= right.upper and right.lower <= left.upper)


def _interval_hull(left: FloatInterval, right: FloatInterval) -> FloatInterval:
    return FloatInterval(
        float(min(left.lower, right.lower)),
        float(max(left.upper, right.upper)),
    )


def _interval_array_hull(left: Array, right: Array) -> Array:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape:
        raise ValueError("interval arrays must have matching shapes")
    out = np.empty(left.shape, dtype=object)
    for index in np.ndindex(left.shape):
        out[index] = _interval_hull(left[index], right[index])
    return out


def _interval_arrays_overlap(left: Array, right: Array) -> bool:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape or left.size == 0:
        return False
    return all(
        _intervals_overlap(left_interval, right_interval)
        for left_interval, right_interval in zip(left.flat, right.flat, strict=True)
    )


def _interval_vector_add(left: Array, right: Array) -> Array:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape:
        raise ValueError("interval vectors must have matching shapes")
    out = np.empty(left.shape, dtype=object)
    for index in np.ndindex(left.shape):
        out[index] = left[index] + right[index]
    return out


def _interval_vector_sub(left: Array, right: Array) -> Array:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape:
        raise ValueError("interval vectors must have matching shapes")
    out = np.empty(left.shape, dtype=object)
    for index in np.ndindex(left.shape):
        out[index] = left[index] - right[index]
    return out


def _interval_vector_scale(vector: Array, factor: float) -> Array:
    vector = np.asarray(vector, dtype=object)
    out = np.empty(vector.shape, dtype=object)
    for index in np.ndindex(vector.shape):
        out[index] = vector[index].scale(float(factor))
    return out


def _interval_weighted_average(values: Array, weights: Array) -> Array:
    values = np.asarray(values, dtype=object)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or weights.shape != (values.shape[0],):
        raise ValueError("weighted average requires a matrix and one weight per row")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        raise ValueError("weights must have a positive finite sum")
    out = _interval_zero_array((values.shape[1],))
    for row, weight in zip(values, weights, strict=True):
        out = _interval_vector_add(out, _interval_vector_scale(row, weight))
    return _interval_vector_scale(out, 1.0 / total_weight)


def _bivector_components(left: Array, right: Array) -> Array:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("bivector inputs must be vectors in dimension at least two")
    return np.asarray(
        [
            left[first] * right[second] - left[second] * right[first]
            for first in range(left.size)
            for second in range(first + 1, left.size)
        ],
        dtype=float,
    )


def _interval_bivector_components(left: Array, right: Array) -> Array:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("bivector inputs must be vectors in dimension at least two")
    components = []
    for first in range(left.size):
        for second in range(first + 1, left.size):
            components.append(
                (left[first] * right[second]) - (left[second] * right[first])
            )
    return np.asarray(components, dtype=object)


def _mass_weighted_bivector_components(
    positions: Array,
    velocities: Array,
    masses: Array,
) -> Array:
    components = np.zeros(positions.shape[1] * (positions.shape[1] - 1) // 2)
    for position, velocity, mass in zip(positions, velocities, masses, strict=True):
        components += float(mass) * _bivector_components(position, velocity)
    return components


def _interval_mass_weighted_bivector_components(
    positions: Array,
    velocities: Array,
    masses: Array,
) -> Array:
    positions = np.asarray(positions, dtype=object)
    velocities = np.asarray(velocities, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if positions.shape != velocities.shape or positions.ndim != 2:
        raise ValueError("positions and velocities must have matching matrix shapes")
    if masses.shape != (positions.shape[0],):
        raise ValueError("masses must have one entry per body")
    components = _interval_zero_array((positions.shape[1] * (positions.shape[1] - 1) // 2,))
    for position, velocity, mass in zip(positions, velocities, masses, strict=True):
        components = _interval_vector_add(
            components,
            _interval_vector_scale(
                _interval_bivector_components(position, velocity),
                float(mass),
            ),
        )
    return components
