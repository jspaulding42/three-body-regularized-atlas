"""Fuchsian selector constructors for zero-angular total-collision branches."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Mapping

import numpy as np

from .dynamics import accelerations
from .event_recurrence import PrimitiveCauchyTailInput, construct_primitive_cauchy_tail_input

Array = np.ndarray
MultiIndex = tuple[int, ...]


def mass_inner_product(masses: Array, left: Array, right: Array) -> float:
    """Mass-weighted Euclidean inner product for shape arrays."""

    masses = np.asarray(masses, dtype=float)
    return float(np.sum(masses[:, None] * left * right))


def acceleration_derivative_apply(positions: Array, masses: Array, perturbation: Array) -> Array:
    """Apply the derivative of the Newtonian acceleration map."""

    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    perturbation = np.asarray(perturbation, dtype=float)
    derivative = np.zeros_like(positions, dtype=float)
    for body in range(positions.shape[0]):
        for other in range(positions.shape[0]):
            if body == other:
                continue
            relative_position = positions[other] - positions[body]
            relative_perturbation = perturbation[other] - perturbation[body]
            distance = float(np.linalg.norm(relative_position))
            derivative[body] += masses[other] * (
                relative_perturbation / distance**3
                - 3.0
                * relative_position
                * np.dot(relative_position, relative_perturbation)
                / distance**5
            )
    return derivative


def linearized_acceleration_matrix(positions: Array, masses: Array) -> Array:
    """Dense matrix for the linearized acceleration at a central shape."""

    positions = np.asarray(positions, dtype=float)
    matrix = np.zeros((positions.size, positions.size), dtype=float)
    for coordinate_index in range(positions.size):
        perturbation = np.zeros_like(positions, dtype=float)
        perturbation.reshape(-1)[coordinate_index] = 1.0
        matrix[:, coordinate_index] = acceleration_derivative_apply(
            positions,
            masses,
            perturbation,
        ).reshape(-1)
    return matrix


def multi_indices_with_total(total_degree: int, dimension: int) -> tuple[MultiIndex, ...]:
    """All multi-indices of a fixed total degree."""

    if dimension == 1:
        return ((int(total_degree),),)
    indices: list[MultiIndex] = []
    for first in range(total_degree + 1):
        for tail in multi_indices_with_total(total_degree - first, dimension - 1):
            indices.append((first, *tail))
    return tuple(indices)


def bounded_multi_indices(maximum: MultiIndex) -> tuple[MultiIndex, ...]:
    """All multi-indices bounded coordinatewise by ``maximum``."""

    return tuple(itertools.product(*(range(limit + 1) for limit in maximum)))


def add_multi_index(left: MultiIndex, right: MultiIndex) -> MultiIndex:
    return tuple(left_value + right_value for left_value, right_value in zip(left, right))


def sub_multi_index(left: MultiIndex, right: MultiIndex) -> MultiIndex:
    return tuple(left_value - right_value for left_value, right_value in zip(left, right))


def multi_index_leq(left: MultiIndex, right: MultiIndex) -> bool:
    return all(left_value <= right_value for left_value, right_value in zip(left, right))


def _scalar_multi_series_product(
    left: Mapping[MultiIndex, float],
    right: Mapping[MultiIndex, float],
    maximum: MultiIndex,
) -> dict[MultiIndex, float]:
    product: dict[MultiIndex, float] = {}
    for left_index, left_value in left.items():
        for right_index, right_value in right.items():
            combined = add_multi_index(left_index, right_index)
            if multi_index_leq(combined, maximum):
                product[combined] = product.get(combined, 0.0) + left_value * right_value
    return product


def _scalar_multi_series_power_one_plus(
    tail: Mapping[MultiIndex, float],
    exponent: float,
    maximum: MultiIndex,
) -> dict[MultiIndex, float]:
    zero_index = tuple(0 for _ in maximum)
    result: dict[MultiIndex, float] = {zero_index: 1.0}
    term: dict[MultiIndex, float] = {zero_index: 1.0}
    binomial_coefficient = 1.0
    for power in range(1, sum(maximum) + 1):
        term = _scalar_multi_series_product(term, tail, maximum)
        if not term:
            break
        binomial_coefficient *= (exponent - power + 1.0) / power
        for index, value in term.items():
            result[index] = result.get(index, 0.0) + binomial_coefficient * value
    return result


def multivariate_acceleration_coefficient(
    shape_coefficients: Mapping[MultiIndex, Array],
    masses: Array,
    target: MultiIndex,
) -> Array:
    """Coefficient of ``A(Phi(x))`` at a multivariable Fuchsian index."""

    zero_shape = np.zeros_like(next(iter(shape_coefficients.values())), dtype=float)
    zero_index = tuple(0 for _ in target)
    acceleration_coefficient = np.zeros_like(zero_shape)
    indices = bounded_multi_indices(target)
    masses = np.asarray(masses, dtype=float)
    for body in range(zero_shape.shape[0]):
        for other in range(zero_shape.shape[0]):
            if body == other:
                continue
            relative = {
                index: (
                    shape_coefficients.get(index, zero_shape)[other]
                    - shape_coefficients.get(index, zero_shape)[body]
                )
                for index in indices
            }
            squared_distance: dict[MultiIndex, float] = {}
            for index in indices:
                squared_distance[index] = sum(
                    float(np.dot(relative[left], relative[sub_multi_index(index, left)]))
                    for left in indices
                    if multi_index_leq(left, index)
                )
            if squared_distance[zero_index] <= 0.0:
                raise ValueError("central shape has a collision")
            normalized_tail = {
                index: value / squared_distance[zero_index]
                for index, value in squared_distance.items()
                if index != zero_index
            }
            inverse_distance_cubed = _scalar_multi_series_power_one_plus(
                normalized_tail,
                -1.5,
                target,
            )
            inverse_distance_cubed = {
                index: squared_distance[zero_index] ** (-1.5) * value
                for index, value in inverse_distance_cubed.items()
            }
            pair_force = sum(
                relative[left]
                * inverse_distance_cubed.get(sub_multi_index(target, left), 0.0)
                for left in indices
                if multi_index_leq(left, target)
            )
            acceleration_coefficient[body] += masses[other] * pair_force
    return acceleration_coefficient


@dataclass(frozen=True)
class FuchsianShapeBranch:
    """Finite Fuchsian shape recurrence for ``q=tau^2 Phi``."""

    masses: Array
    central_shape: Array
    powers: tuple[float, ...]
    coefficients: Mapping[MultiIndex, Array]
    selected_indices: frozenset[MultiIndex]
    solved_indices: tuple[MultiIndex, ...]
    derivative_matrix: Array
    recurrence_residual_norms: Mapping[MultiIndex, float]
    singular_value_floors: Mapping[MultiIndex, float]
    max_total_degree: int
    scale_index: MultiIndex | None = None
    scale_coefficient: float | None = None

    @property
    def zero_index(self) -> MultiIndex:
        return tuple(0 for _ in self.powers)

    @property
    def inertia(self) -> float:
        return mass_inner_product(self.masses, self.central_shape, self.central_shape)

    @property
    def finite_energy_limit(self) -> float:
        if self.scale_coefficient is None:
            return float("nan")
        return float((10.0 / 9.0) * self.scale_coefficient * self.inertia)

    @property
    def max_recurrence_residual_norm(self) -> float:
        if not self.recurrence_residual_norms:
            return 0.0
        return float(max(self.recurrence_residual_norms.values()))

    @property
    def min_solved_singular_value_floor(self) -> float:
        floors = [
            value
            for index, value in self.singular_value_floors.items()
            if index in self.solved_indices
        ]
        if not floors:
            return float("inf")
        return float(min(floors))

    @property
    def recurrence_certified(self) -> bool:
        return (
            np.isfinite(self.max_recurrence_residual_norm)
            and self.max_recurrence_residual_norm < 1e-10
            and self.min_solved_singular_value_floor > 1e-10
        )

    def exponent(self, index: MultiIndex) -> float:
        return float(sum(coordinate * power for coordinate, power in zip(index, self.powers)))

    def shape_at_radius(self, radius: float, *, max_total_degree: int | None = None) -> Array:
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        shape = np.zeros_like(self.central_shape, dtype=float)
        for index, coefficient in self.coefficients.items():
            if max_total_degree is not None and sum(index) > max_total_degree:
                continue
            shape += coefficient * radius ** self.exponent(index)
        return shape

    def radius_derivatives(self, radius: float) -> tuple[Array, Array, Array]:
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        shape = np.zeros_like(self.central_shape, dtype=float)
        first = np.zeros_like(self.central_shape, dtype=float)
        second = np.zeros_like(self.central_shape, dtype=float)
        for index, coefficient in self.coefficients.items():
            exponent = self.exponent(index)
            shape += coefficient * radius**exponent
            if index != self.zero_index:
                first += exponent * coefficient * radius ** (exponent - 1.0)
                second += (
                    exponent
                    * (exponent - 1.0)
                    * coefficient
                    * radius ** (exponent - 2.0)
                )
        return shape, first, second

    def tau_derivatives(self, signed_tau: float) -> tuple[Array, Array, Array]:
        signed_tau = float(signed_tau)
        if signed_tau == 0.0:
            raise ValueError("tau must be nonzero")
        radius = abs(signed_tau)
        sign = 1.0 if signed_tau > 0.0 else -1.0
        shape, radius_first, radius_second = self.radius_derivatives(radius)
        return shape, sign * radius_first, radius_second

    def shape_equation_residual_at_tau(self, signed_tau: float) -> Array:
        shape, first, second = self.tau_derivatives(signed_tau)
        return (
            signed_tau**2 * second
            + 2.0 * signed_tau * first
            - 2.0 * shape
            - 9.0 * accelerations(shape, self.masses)
        )

    def scaled_newton_residual_at_tau(self, signed_tau: float) -> Array:
        return self.shape_equation_residual_at_tau(signed_tau) / 9.0

    def newton_residual_at_tau(self, signed_tau: float) -> Array:
        signed_tau = float(signed_tau)
        if signed_tau == 0.0:
            raise ValueError("tau must be nonzero")
        return self.shape_equation_residual_at_tau(signed_tau) / (9.0 * signed_tau**4)

    def positions_at_tau(self, signed_tau: float) -> Array:
        shape = self.shape_at_radius(abs(float(signed_tau)))
        return float(signed_tau) ** 2 * shape

    def velocities_at_tau(self, signed_tau: float) -> Array:
        signed_tau = float(signed_tau)
        if signed_tau == 0.0:
            raise ValueError("tau must be nonzero")
        shape, first, _second = self.tau_derivatives(signed_tau)
        return (2.0 * signed_tau * shape + signed_tau**2 * first) / (
            3.0 * signed_tau**2
        )

    def state_at_tau(self, signed_tau: float) -> Array:
        return np.concatenate(
            [
                self.positions_at_tau(signed_tau).reshape(-1),
                self.velocities_at_tau(signed_tau).reshape(-1),
            ]
        )

    def centered_angular_momentum_scalar_at_tau(self, signed_tau: float) -> float:
        shape, first, _second = self.tau_derivatives(signed_tau)
        angular_factor = sum(
            mass
            * (shape_row[0] * first_row[1] - shape_row[1] * first_row[0])
            for mass, shape_row, first_row in zip(self.masses, shape, first)
        )
        return float(signed_tau**2 * angular_factor / 3.0)

    def regularized_position_derivative(
        self,
        signed_regularized_time: float,
        derivative_order: int,
    ) -> Array:
        signed_regularized_time = float(signed_regularized_time)
        if signed_regularized_time == 0.0:
            raise ValueError("regularized time must be nonzero")
        radius = abs(signed_regularized_time)
        sign = 1.0 if signed_regularized_time > 0.0 else -1.0
        derivative = np.zeros_like(self.central_shape, dtype=float)
        for index, coefficient in self.coefficients.items():
            power = self.exponent(index) + 2.0
            falling_factorial = 1.0
            for offset in range(derivative_order):
                falling_factorial *= power - offset
            derivative += (
                sign**derivative_order
                * falling_factorial
                * coefficient
                * radius ** (power - derivative_order)
            )
        return derivative


@dataclass(frozen=True)
class FuchsianSelectorContinuation:
    """Two-sided Fuchsian continuation built by a selector rule."""

    incoming: FuchsianShapeBranch
    outgoing: FuchsianShapeBranch

    @property
    def selected_indices(self) -> frozenset[MultiIndex]:
        return self.incoming.selected_indices | self.outgoing.selected_indices

    @property
    def selector_gaps(self) -> Mapping[MultiIndex, float]:
        gaps: dict[MultiIndex, float] = {}
        for index in self.selected_indices:
            gaps[index] = float(
                np.linalg.norm(
                    self.incoming.coefficients[index] - self.outgoing.coefficients[index],
                    ord=np.inf,
                )
            )
        return gaps

    @property
    def max_selector_gap(self) -> float:
        if not self.selector_gaps:
            return 0.0
        return float(max(self.selector_gaps.values()))

    @property
    def energy_gap(self) -> float:
        return float(abs(self.incoming.finite_energy_limit - self.outgoing.finite_energy_limit))

    @property
    def identity_selector_certified(self) -> bool:
        return (
            self.incoming.recurrence_certified
            and self.outgoing.recurrence_certified
            and self.max_selector_gap < 1e-12
            and np.isfinite(self.energy_gap)
            and self.energy_gap < 1e-12
        )


@dataclass(frozen=True)
class FuchsianLogRowSolution:
    """Triangular solution of one resonant Fuchsian-log row."""

    power: float
    row_operator: Array
    masses: Array
    body_shape: tuple[int, int]
    kernel_basis: tuple[Array, ...]
    forcing_by_log_power: Mapping[int, Array]
    coefficients_by_log_power: Mapping[int, Array]
    selector_by_basis: tuple[float, ...]

    @property
    def max_log_power(self) -> int:
        return max(self.coefficients_by_log_power)

    def _zero_vector(self) -> Array:
        return np.zeros(self.body_shape[0] * self.body_shape[1], dtype=float)

    def coefficient_vector(self, log_power: int) -> Array:
        coefficient = self.coefficients_by_log_power.get(log_power)
        if coefficient is None:
            return self._zero_vector()
        return coefficient.reshape(-1)

    def forcing_vector(self, log_power: int) -> Array:
        forcing = self.forcing_by_log_power.get(log_power)
        if forcing is None:
            return self._zero_vector()
        return forcing.reshape(-1)

    def equation_residual_vector(self, log_power: int) -> Array:
        carry = (
            self.row_operator @ self.coefficient_vector(log_power)
            + (log_power + 1.0)
            * (2.0 * self.power + 1.0)
            * self.coefficient_vector(log_power + 1)
            + (log_power + 2.0)
            * (log_power + 1.0)
            * self.coefficient_vector(log_power + 2)
            - self.forcing_vector(log_power)
        )
        return carry

    def equation_residual_norm(self, log_power: int) -> float:
        return float(np.linalg.norm(self.equation_residual_vector(log_power), ord=np.inf))

    @property
    def max_equation_residual_norm(self) -> float:
        powers = set(self.forcing_by_log_power) | set(self.coefficients_by_log_power)
        powers.update(power - 1 for power in self.coefficients_by_log_power if power > 0)
        if not powers:
            return 0.0
        return float(max(self.equation_residual_norm(power) for power in powers if power >= 0))

    @property
    def recovered_selector_by_basis(self) -> tuple[float, ...]:
        constant = self.coefficients_by_log_power.get(0)
        if constant is None:
            return tuple(0.0 for _basis in self.kernel_basis)
        return tuple(
            mass_inner_product(self.masses, constant, basis)
            for basis in self.kernel_basis
        )

    @property
    def row_certified(self) -> bool:
        return (
            np.isfinite(self.max_equation_residual_norm)
            and self.max_equation_residual_norm < 1e-10
        )

    def polynomial_at_log(self, logarithm: float) -> Array:
        polynomial = np.zeros(self.body_shape, dtype=float)
        for log_power, coefficient in self.coefficients_by_log_power.items():
            polynomial += coefficient * float(logarithm) ** log_power
        return polynomial

    def log_derivatives_at_log(self, logarithm: float) -> tuple[Array, Array, Array]:
        polynomial = np.zeros(self.body_shape, dtype=float)
        first = np.zeros(self.body_shape, dtype=float)
        second = np.zeros(self.body_shape, dtype=float)
        logarithm = float(logarithm)
        for log_power, coefficient in self.coefficients_by_log_power.items():
            polynomial += coefficient * logarithm**log_power
            if log_power >= 1:
                first += log_power * coefficient * logarithm ** (log_power - 1)
            if log_power >= 2:
                second += (
                    log_power
                    * (log_power - 1)
                    * coefficient
                    * logarithm ** (log_power - 2)
                )
        return polynomial, first, second


@dataclass(frozen=True)
class FuchsianLogBranch:
    """One resonant log row projected by ``q=tau^2 S``, ``t=tau^3``."""

    masses: Array
    central_shape: Array
    row_solution: FuchsianLogRowSolution
    scale_coefficient: float = 0.0

    @property
    def inertia(self) -> float:
        return mass_inner_product(self.masses, self.central_shape, self.central_shape)

    @property
    def finite_energy_limit(self) -> float:
        return float((10.0 / 9.0) * self.scale_coefficient * self.inertia)

    def radius_derivatives(self, radius: float) -> tuple[Array, Array, Array]:
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        logarithm = np.log(radius)
        polynomial, log_first, log_second = self.row_solution.log_derivatives_at_log(
            logarithm
        )
        power = self.row_solution.power
        correction = radius**power * polynomial
        correction_first = radius ** (power - 1.0) * (
            power * polynomial + log_first
        )
        correction_second = radius ** (power - 2.0) * (
            power * (power - 1.0) * polynomial
            + (2.0 * power - 1.0) * log_first
            + log_second
        )
        shape = (
            self.central_shape
            + self.scale_coefficient * radius**2 * self.central_shape
            + correction
        )
        first = (
            2.0 * self.scale_coefficient * radius * self.central_shape
            + correction_first
        )
        second = (
            2.0 * self.scale_coefficient * self.central_shape
            + correction_second
        )
        return shape, first, second

    def tau_derivatives(self, signed_tau: float) -> tuple[Array, Array, Array]:
        signed_tau = float(signed_tau)
        if signed_tau == 0.0:
            raise ValueError("tau must be nonzero")
        sign = 1.0 if signed_tau > 0.0 else -1.0
        shape, first, second = self.radius_derivatives(abs(signed_tau))
        return shape, sign * first, second

    def positions_at_tau(self, signed_tau: float) -> Array:
        shape, _first, _second = self.tau_derivatives(signed_tau)
        return float(signed_tau) ** 2 * shape

    def velocities_at_tau(self, signed_tau: float) -> Array:
        signed_tau = float(signed_tau)
        shape, first, _second = self.tau_derivatives(signed_tau)
        return (2.0 * signed_tau * shape + signed_tau**2 * first) / (
            3.0 * signed_tau**2
        )

    def state_at_tau(self, signed_tau: float) -> Array:
        return np.concatenate(
            [
                self.positions_at_tau(signed_tau).reshape(-1),
                self.velocities_at_tau(signed_tau).reshape(-1),
            ]
        )

    def shape_projection_identity_residual(self, signed_tau: float) -> Array:
        shape, first, second = self.tau_derivatives(signed_tau)
        positions = self.positions_at_tau(signed_tau)
        projected_acceleration = (
            signed_tau**2 * second + 2.0 * signed_tau * first - 2.0 * shape
        ) / (9.0 * signed_tau**4)
        shape_residual = (
            signed_tau**2 * second
            + 2.0 * signed_tau * first
            - 2.0 * shape
            - 9.0 * accelerations(shape, self.masses)
        )
        projected_residual = projected_acceleration - accelerations(
            positions,
            self.masses,
        )
        return 9.0 * signed_tau**4 * projected_residual - shape_residual

    def centered_angular_momentum_scalar_at_tau(self, signed_tau: float) -> float:
        shape, first, _second = self.tau_derivatives(signed_tau)
        angular_factor = sum(
            mass
            * (shape_row[0] * first_row[1] - shape_row[1] * first_row[0])
            for mass, shape_row, first_row in zip(self.masses, shape, first)
        )
        return float(signed_tau**2 * angular_factor / 3.0)

    def recovered_selector_by_basis(self, radius: float) -> tuple[float, ...]:
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        shape, _first, _second = self.radius_derivatives(radius)
        remainder = (
            shape
            - self.central_shape
            - self.scale_coefficient * radius**2 * self.central_shape
        )
        logarithm = np.log(radius)
        return tuple(
            mass_inner_product(self.masses, remainder, basis)
            / radius**self.row_solution.power
            - sum(
                mass_inner_product(self.masses, coefficient, basis)
                * logarithm**log_power
                for log_power, coefficient in self.row_solution.coefficients_by_log_power.items()
                if log_power > 0
            )
            for basis in self.row_solution.kernel_basis
        )


@dataclass(frozen=True)
class FuchsianLogTerm:
    """One finite row ``sigma^power sum_l H_l (log sigma)^l``."""

    power: float
    coefficients_by_log_power: Mapping[int, Array]
    selector_basis: tuple[Array, ...] = ()

    @property
    def body_shape(self) -> tuple[int, int]:
        return next(iter(self.coefficients_by_log_power.values())).shape

    def coefficient(self, log_power: int) -> Array:
        coefficient = self.coefficients_by_log_power.get(log_power)
        if coefficient is None:
            return np.zeros(self.body_shape, dtype=float)
        return coefficient

    def log_derivatives_at_log(self, logarithm: float) -> tuple[Array, Array, Array]:
        polynomial = np.zeros(self.body_shape, dtype=float)
        first = np.zeros(self.body_shape, dtype=float)
        second = np.zeros(self.body_shape, dtype=float)
        logarithm = float(logarithm)
        for log_power, coefficient in self.coefficients_by_log_power.items():
            polynomial += coefficient * logarithm**log_power
            if log_power >= 1:
                first += log_power * coefficient * logarithm ** (log_power - 1)
            if log_power >= 2:
                second += (
                    log_power
                    * (log_power - 1)
                    * coefficient
                    * logarithm ** (log_power - 2)
                )
        return polynomial, first, second

    def radius_derivatives(self, radius: float) -> tuple[Array, Array, Array]:
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        logarithm = np.log(radius)
        polynomial, log_first, log_second = self.log_derivatives_at_log(logarithm)
        correction = radius**self.power * polynomial
        first = radius ** (self.power - 1.0) * (
            self.power * polynomial + log_first
        )
        second = radius ** (self.power - 2.0) * (
            self.power * (self.power - 1.0) * polynomial
            + (2.0 * self.power - 1.0) * log_first
            + log_second
        )
        return correction, first, second

    @classmethod
    def from_row_solution(cls, row_solution: FuchsianLogRowSolution) -> "FuchsianLogTerm":
        return cls(
            power=row_solution.power,
            coefficients_by_log_power=row_solution.coefficients_by_log_power,
            selector_basis=row_solution.kernel_basis,
        )


@dataclass(frozen=True)
class FiniteFuchsianLogBranch:
    """Finite Fuchsian-log shape branch projected by ``q=tau^2S``."""

    masses: Array
    central_shape: Array
    scale_coefficient: float
    terms: tuple[FuchsianLogTerm, ...]

    @property
    def inertia(self) -> float:
        return mass_inner_product(self.masses, self.central_shape, self.central_shape)

    @property
    def finite_energy_limit(self) -> float:
        return float((10.0 / 9.0) * self.scale_coefficient * self.inertia)

    def radius_derivatives(self, radius: float) -> tuple[Array, Array, Array]:
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        shape = self.central_shape + self.scale_coefficient * radius**2 * self.central_shape
        first = 2.0 * self.scale_coefficient * radius * self.central_shape
        second = 2.0 * self.scale_coefficient * self.central_shape
        for term in self.terms:
            term_value, term_first, term_second = term.radius_derivatives(radius)
            shape += term_value
            first += term_first
            second += term_second
        return shape, first, second

    def tau_derivatives(self, signed_tau: float) -> tuple[Array, Array, Array]:
        signed_tau = float(signed_tau)
        if signed_tau == 0.0:
            raise ValueError("tau must be nonzero")
        sign = 1.0 if signed_tau > 0.0 else -1.0
        shape, first, second = self.radius_derivatives(abs(signed_tau))
        return shape, sign * first, second

    def positions_at_tau(self, signed_tau: float) -> Array:
        shape, _first, _second = self.tau_derivatives(signed_tau)
        return float(signed_tau) ** 2 * shape

    def velocities_at_tau(self, signed_tau: float) -> Array:
        signed_tau = float(signed_tau)
        shape, first, _second = self.tau_derivatives(signed_tau)
        return (2.0 * signed_tau * shape + signed_tau**2 * first) / (
            3.0 * signed_tau**2
        )

    def state_at_tau(self, signed_tau: float) -> Array:
        return np.concatenate(
            [
                self.positions_at_tau(signed_tau).reshape(-1),
                self.velocities_at_tau(signed_tau).reshape(-1),
            ]
        )

    def shape_projection_identity_residual(self, signed_tau: float) -> Array:
        shape, first, second = self.tau_derivatives(signed_tau)
        positions = self.positions_at_tau(signed_tau)
        projected_acceleration = (
            signed_tau**2 * second + 2.0 * signed_tau * first - 2.0 * shape
        ) / (9.0 * signed_tau**4)
        shape_residual = (
            signed_tau**2 * second
            + 2.0 * signed_tau * first
            - 2.0 * shape
            - 9.0 * accelerations(shape, self.masses)
        )
        projected_residual = projected_acceleration - accelerations(
            positions,
            self.masses,
        )
        return 9.0 * signed_tau**4 * projected_residual - shape_residual

    def centered_angular_momentum_scalar_at_tau(self, signed_tau: float) -> float:
        shape, first, _second = self.tau_derivatives(signed_tau)
        angular_factor = sum(
            mass
            * (shape_row[0] * first_row[1] - shape_row[1] * first_row[0])
            for mass, shape_row, first_row in zip(self.masses, shape, first)
        )
        return float(signed_tau**2 * angular_factor / 3.0)

    def _term_value_at_radius(self, term_index: int, radius: float) -> Array:
        value, _first, _second = self.terms[term_index].radius_derivatives(radius)
        return value

    def recovered_selectors_for_term(self, term_index: int, radius: float) -> tuple[float, ...]:
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        term = self.terms[term_index]
        if not term.selector_basis:
            return ()
        shape, _first, _second = self.radius_derivatives(radius)
        remainder = (
            shape
            - self.central_shape
            - self.scale_coefficient * radius**2 * self.central_shape
        )
        for other_index in range(len(self.terms)):
            if other_index != term_index:
                remainder -= self._term_value_at_radius(other_index, radius)
        logarithm = np.log(radius)
        return tuple(
            mass_inner_product(self.masses, remainder, basis) / radius**term.power
            - sum(
                mass_inner_product(self.masses, coefficient, basis)
                * logarithm**log_power
                for log_power, coefficient in term.coefficients_by_log_power.items()
                if log_power > 0
            )
            for basis in term.selector_basis
        )


@dataclass(frozen=True)
class FiniteFuchsianLogContinuation:
    """Two-sided finite Fuchsian-log continuation."""

    incoming: FiniteFuchsianLogBranch
    outgoing: FiniteFuchsianLogBranch

    @property
    def energy_gap(self) -> float:
        return float(abs(self.incoming.finite_energy_limit - self.outgoing.finite_energy_limit))

    def max_selector_gap_at_radius(self, radius: float) -> float:
        gaps: list[float] = []
        for term_index in range(min(len(self.incoming.terms), len(self.outgoing.terms))):
            incoming = self.incoming.recovered_selectors_for_term(term_index, radius)
            outgoing = self.outgoing.recovered_selectors_for_term(term_index, radius)
            gaps.extend(abs(left - right) for left, right in zip(incoming, outgoing))
        if not gaps:
            return 0.0
        return float(max(gaps))

    @property
    def identity_selector_certified(self) -> bool:
        if not (
            _finite_fuchsian_log_branch_data_certified(self.incoming)
            and _finite_fuchsian_log_branch_data_certified(self.outgoing)
            and _finite_fuchsian_log_branch_domains_match(self.incoming, self.outgoing)
            and np.isfinite(self.energy_gap)
            and self.energy_gap < 1e-12
            and len(self.incoming.terms) == len(self.outgoing.terms)
            and abs(
                float(self.incoming.scale_coefficient)
                - float(self.outgoing.scale_coefficient)
            )
            < 1e-12
        ):
            return False
        for incoming, outgoing in zip(self.incoming.terms, self.outgoing.terms):
            if not np.isclose(incoming.power, outgoing.power):
                return False
            if set(incoming.coefficients_by_log_power) != set(outgoing.coefficients_by_log_power):
                return False
            for log_power in incoming.coefficients_by_log_power:
                if np.linalg.norm(
                    incoming.coefficients_by_log_power[log_power]
                    - outgoing.coefficients_by_log_power[log_power],
                    ord=np.inf,
                ) >= 1e-12:
                    return False
            if len(incoming.selector_basis) != len(outgoing.selector_basis):
                return False
            for incoming_basis, outgoing_basis in zip(
                incoming.selector_basis,
                outgoing.selector_basis,
            ):
                if (
                    np.linalg.norm(incoming_basis - outgoing_basis, ord=np.inf)
                    >= 1e-12
                ):
                    return False
        return True


def derive_identity_finite_fuchsian_log_continuation_from_incoming_branch(
    incoming_branch: FiniteFuchsianLogBranch,
) -> FiniteFuchsianLogContinuation:
    """Derive identity-selector continuation from one finite Fuchsian-log germ."""

    if not isinstance(incoming_branch, FiniteFuchsianLogBranch):
        raise TypeError("incoming_branch must be a FiniteFuchsianLogBranch")
    if not _finite_fuchsian_log_branch_data_certified(incoming_branch):
        raise ValueError("incoming finite Fuchsian-log branch does not certify")
    outgoing_terms = tuple(
        FuchsianLogTerm(
            power=float(term.power),
            coefficients_by_log_power={
                int(log_power): np.asarray(coefficient, dtype=float).copy()
                for log_power, coefficient in term.coefficients_by_log_power.items()
            },
            selector_basis=tuple(
                np.asarray(basis, dtype=float).copy()
                for basis in term.selector_basis
            ),
        )
        for term in incoming_branch.terms
    )
    outgoing_branch = FiniteFuchsianLogBranch(
        masses=np.asarray(incoming_branch.masses, dtype=float).copy(),
        central_shape=np.asarray(incoming_branch.central_shape, dtype=float).copy(),
        scale_coefficient=float(incoming_branch.scale_coefficient),
        terms=outgoing_terms,
    )
    continuation = FiniteFuchsianLogContinuation(
        incoming=incoming_branch,
        outgoing=outgoing_branch,
    )
    if not continuation.identity_selector_certified:
        raise ValueError("derived finite Fuchsian-log identity continuation did not certify")
    return continuation


def _finite_fuchsian_log_branch_data_certified(
    branch: FiniteFuchsianLogBranch,
) -> bool:
    if not isinstance(branch, FiniteFuchsianLogBranch):
        return False
    try:
        masses = np.asarray(branch.masses, dtype=float).reshape(-1)
        central_shape = np.asarray(branch.central_shape, dtype=float)
        scale_coefficient = float(branch.scale_coefficient)
    except (TypeError, ValueError):
        return False
    if not (
        masses.ndim == 1
        and central_shape.ndim == 2
        and central_shape.shape[0] == masses.size
        and masses.size >= 2
        and np.all(np.isfinite(masses))
        and np.all(masses > 0.0)
        and np.all(np.isfinite(central_shape))
        and np.isfinite(scale_coefficient)
        and _minimum_pair_distance(central_shape) > 0.0
        and np.isfinite(branch.finite_energy_limit)
    ):
        return False
    for term in branch.terms:
        if not _finite_fuchsian_log_term_data_certified(term, central_shape.shape):
            return False
    return True


def _finite_fuchsian_log_term_data_certified(
    term: FuchsianLogTerm,
    body_shape: tuple[int, int],
) -> bool:
    if not isinstance(term, FuchsianLogTerm):
        return False
    try:
        power = float(term.power)
    except (TypeError, ValueError):
        return False
    if not (
        np.isfinite(power)
        and power > 0.0
        and term.coefficients_by_log_power
    ):
        return False
    for log_power, coefficient in term.coefficients_by_log_power.items():
        try:
            log_power_index = int(log_power)
        except (TypeError, ValueError):
            return False
        coefficient_array = np.asarray(coefficient, dtype=float)
        if not (
            log_power_index == log_power
            and log_power_index >= 0
            and coefficient_array.shape == body_shape
            and np.all(np.isfinite(coefficient_array))
        ):
            return False
    for basis in term.selector_basis:
        basis_array = np.asarray(basis, dtype=float)
        if not (
            basis_array.shape == body_shape
            and np.all(np.isfinite(basis_array))
            and np.linalg.norm(basis_array, ord=np.inf) > 0.0
        ):
            return False
    return True


def _finite_fuchsian_log_branch_domains_match(
    left: FiniteFuchsianLogBranch,
    right: FiniteFuchsianLogBranch,
    *,
    tolerance: float = 1e-12,
) -> bool:
    try:
        left_masses = np.asarray(left.masses, dtype=float)
        right_masses = np.asarray(right.masses, dtype=float)
        left_shape = np.asarray(left.central_shape, dtype=float)
        right_shape = np.asarray(right.central_shape, dtype=float)
    except (TypeError, ValueError):
        return False
    if left_masses.shape != right_masses.shape or left_shape.shape != right_shape.shape:
        return False
    scale = max(
        1.0,
        float(np.linalg.norm(left_masses, ord=np.inf)),
        float(np.linalg.norm(right_masses, ord=np.inf)),
        float(np.linalg.norm(left_shape, ord=np.inf)),
        float(np.linalg.norm(right_shape, ord=np.inf)),
    )
    return bool(
        np.linalg.norm(left_masses - right_masses, ord=np.inf)
        <= tolerance * scale
        and np.linalg.norm(left_shape - right_shape, ord=np.inf)
        <= tolerance * scale
    )


@dataclass(frozen=True)
class FiniteFuchsianLogPrimitiveCauchyInputs:
    """Primitive shell Cauchy inputs derived from a finite Fuchsian-log branch."""

    initial_radius: float
    shell_contraction: float
    analytic_disk_fraction: float
    log_growth_factor: float
    component_derivative_orders: Mapping[str, int]
    component_multipliers: Mapping[str, float]
    component_inputs: Mapping[str, PrimitiveCauchyTailInput]

    @property
    def certified(self) -> bool:
        return bool(
            np.isfinite(self.initial_radius)
            and self.initial_radius > 0.0
            and np.isfinite(self.shell_contraction)
            and 0.0 < self.shell_contraction < 1.0
            and np.isfinite(self.analytic_disk_fraction)
            and 0.0 < self.analytic_disk_fraction < 1.0
            and np.isfinite(self.log_growth_factor)
            and self.log_growth_factor > 1.0
            and self.component_inputs
            and all(input_.certified for input_ in self.component_inputs.values())
        )

    def component_input(self, component: str) -> PrimitiveCauchyTailInput:
        return self.component_inputs[str(component)]

    def component_majorant_bound(self, component: str, shell_index: int) -> float:
        return self.component_input(component).majorant_initial * (
            self.component_input(component).majorant_growth ** int(shell_index)
        )


@dataclass(frozen=True)
class FiniteFuchsianLogTotalCollisionIsolationCertificate:
    """Punctured no-binary-collision radius for a finite Fuchsian-log branch."""

    radius: float
    central_shape_pair_distance_floor: float
    shape_deviation_bound: float
    shape_pair_distance_floor: float
    dimension: int

    @property
    def certified(self) -> bool:
        return bool(
            np.isfinite(self.radius)
            and 0.0 < self.radius < 1.0
            and np.isfinite(self.central_shape_pair_distance_floor)
            and self.central_shape_pair_distance_floor > 0.0
            and np.isfinite(self.shape_deviation_bound)
            and self.shape_deviation_bound >= 0.0
            and np.isfinite(self.shape_pair_distance_floor)
            and self.shape_pair_distance_floor > 0.0
            and self.dimension >= 1
        )

    def physical_pair_distance_floor(self, signed_tau: float) -> float:
        signed_tau = float(signed_tau)
        if signed_tau == 0.0 or abs(signed_tau) > self.radius:
            raise ValueError("tau must be punctured and inside the certified radius")
        return float(signed_tau**2 * self.shape_pair_distance_floor)


@dataclass(frozen=True)
class FiniteFuchsianLogCompactTimeIsolationCertificate:
    """Projection of a local Fuchsian-log isolation radius to compact time."""

    tau_isolation: FiniteFuchsianLogTotalCollisionIsolationCertificate
    event_physical_time: float
    compact_time_rate: float
    event_compact_parameter: float
    compact_lower: float
    compact_upper: float
    compact_isolation_radius: float
    shell_lower: float | None = None
    shell_upper: float | None = None
    shell_isolation_fraction: float | None = None

    @property
    def certified(self) -> bool:
        shell_ok = True
        if self.shell_lower is not None or self.shell_upper is not None:
            shell_ok = bool(
                self.shell_lower is not None
                and self.shell_upper is not None
                and self.shell_isolation_fraction is not None
                and np.isfinite(self.shell_lower)
                and np.isfinite(self.shell_upper)
                and self.shell_lower < self.event_compact_parameter < self.shell_upper
                and 0.0 < self.shell_isolation_fraction <= 1.0
            )
        return bool(
            self.tau_isolation.certified
            and np.isfinite(self.event_physical_time)
            and np.isfinite(self.compact_time_rate)
            and self.compact_time_rate > 0.0
            and -1.0 < self.compact_lower < self.event_compact_parameter < self.compact_upper < 1.0
            and np.isfinite(self.compact_isolation_radius)
            and self.compact_isolation_radius > 0.0
            and shell_ok
        )

    @property
    def compact_interval_width(self) -> float:
        return float(self.compact_upper - self.compact_lower)


def certify_finite_fuchsian_log_compact_time_isolation(
    tau_isolation: FiniteFuchsianLogTotalCollisionIsolationCertificate,
    *,
    event_physical_time: float = 0.0,
    compact_time_rate: float = 1.0,
    shell_interval: tuple[float, float] | None = None,
) -> FiniteFuchsianLogCompactTimeIsolationCertificate:
    """Project ``tau``-isolation through ``t=t0+tau^3`` and ``u=tanh(lambda t)``."""

    if not tau_isolation.certified:
        raise ValueError("tau isolation certificate must certify")
    event_physical_time = float(event_physical_time)
    compact_time_rate = float(compact_time_rate)
    if not np.isfinite(event_physical_time):
        raise ValueError("event_physical_time must be finite")
    if not np.isfinite(compact_time_rate) or compact_time_rate <= 0.0:
        raise ValueError("compact_time_rate must be positive")
    physical_radius = tau_isolation.radius**3
    lower_time = event_physical_time - physical_radius
    upper_time = event_physical_time + physical_radius
    event_compact = float(np.tanh(compact_time_rate * event_physical_time))
    compact_lower = float(np.tanh(compact_time_rate * lower_time))
    compact_upper = float(np.tanh(compact_time_rate * upper_time))
    compact_radius = float(min(event_compact - compact_lower, compact_upper - event_compact))

    shell_lower: float | None = None
    shell_upper: float | None = None
    shell_fraction: float | None = None
    if shell_interval is not None:
        shell_lower = float(shell_interval[0])
        shell_upper = float(shell_interval[1])
        if not (
            np.isfinite(shell_lower)
            and np.isfinite(shell_upper)
            and -1.0 < shell_lower < shell_upper < 1.0
        ):
            raise ValueError("shell interval must lie inside (-1, 1)")
        if not shell_lower < event_compact < shell_upper:
            raise ValueError("event compact parameter must lie inside the shell interval")
        shell_width = shell_upper - shell_lower
        boundary_radius = min(event_compact - shell_lower, shell_upper - event_compact)
        shell_fraction = float(min(compact_radius, boundary_radius) / shell_width)
        if not np.isfinite(shell_fraction) or shell_fraction <= 0.0:
            raise ValueError("compact shell isolation fraction did not certify")
        shell_fraction = min(shell_fraction, 1.0)

    certificate = FiniteFuchsianLogCompactTimeIsolationCertificate(
        tau_isolation=tau_isolation,
        event_physical_time=event_physical_time,
        compact_time_rate=compact_time_rate,
        event_compact_parameter=event_compact,
        compact_lower=compact_lower,
        compact_upper=compact_upper,
        compact_isolation_radius=compact_radius,
        shell_lower=shell_lower,
        shell_upper=shell_upper,
        shell_isolation_fraction=shell_fraction,
    )
    if not certificate.certified:
        raise ValueError("compact-time total-collision isolation did not certify")
    return certificate


def certify_finite_fuchsian_log_total_collision_isolation(
    branch: FiniteFuchsianLogBranch,
    *,
    radius: float,
) -> FiniteFuchsianLogTotalCollisionIsolationCertificate:
    """Certify a punctured total-collision chart has no binary collisions.

    If ``S(tau)=C+Delta(tau)`` and ``|Delta_i|`` is small enough compared with
    the minimum pair distance of ``C``, then every pair distance in the shape
    stays positive for ``0<|tau|<=radius``.  Since ``q=tau^2 S``, all physical
    pair distances vanish only at ``tau=0`` inside that radius.
    """

    radius = float(radius)
    if not np.isfinite(radius) or not 0.0 < radius < 1.0:
        raise ValueError("radius must lie in (0, 1)")
    central_floor = _minimum_pair_distance(branch.central_shape)
    if central_floor <= 0.0:
        raise ValueError("central shape must be collision-free")
    deviation_bound = _finite_fuchsian_log_shape_deviation_bound(branch, radius)
    body_deviation_bound = np.sqrt(branch.central_shape.shape[1]) * deviation_bound
    shape_floor = central_floor - 2.0 * body_deviation_bound
    certificate = FiniteFuchsianLogTotalCollisionIsolationCertificate(
        radius=radius,
        central_shape_pair_distance_floor=central_floor,
        shape_deviation_bound=deviation_bound,
        shape_pair_distance_floor=float(shape_floor),
        dimension=int(branch.central_shape.shape[1]),
    )
    if not certificate.certified:
        raise ValueError("finite Fuchsian-log branch is not isolated at this radius")
    return certificate


def derive_finite_fuchsian_log_branch_primitive_cauchy_inputs(
    branch: FiniteFuchsianLogBranch,
    *,
    initial_radius: float,
    shell_contraction: float,
    analytic_disk_fraction: float,
    log_growth_factor: float,
    step_ratio_bounds: Mapping[str, float],
    retained_order_initials: Mapping[str, int],
    retained_order_increments: Mapping[str, int],
    component_derivative_orders: Mapping[str, int] | None = None,
    component_multipliers: Mapping[str, float] | None = None,
) -> FiniteFuchsianLogPrimitiveCauchyInputs:
    """Derive primitive Cauchy inputs for finite Fuchsian-log total-collision charts.

    The bound is for punctured shells centered at radii
    ``initial_radius * shell_contraction**n``.  On the analytic disk
    ``|tau-tau_n| <= analytic_disk_fraction*tau_n``, every finite row
    ``tau^omega P(log tau)`` is bounded by a geometric envelope; polynomial
    log growth is absorbed into ``log_growth_factor**n``.
    """

    initial_radius = float(initial_radius)
    shell_contraction = float(shell_contraction)
    analytic_disk_fraction = float(analytic_disk_fraction)
    log_growth_factor = float(log_growth_factor)
    if initial_radius <= 0.0 or not np.isfinite(initial_radius):
        raise ValueError("initial_radius must be positive")
    if not np.isfinite(shell_contraction) or not 0.0 < shell_contraction < 1.0:
        raise ValueError("shell_contraction must lie in (0, 1)")
    if not np.isfinite(analytic_disk_fraction) or not 0.0 < analytic_disk_fraction < 1.0:
        raise ValueError("analytic_disk_fraction must lie in (0, 1)")
    if not np.isfinite(log_growth_factor) or log_growth_factor <= 1.0:
        raise ValueError("log_growth_factor must be greater than one")

    if component_derivative_orders is None:
        derivative_orders = {
            "value": 0,
            "first_jet": 1,
            "lifted_residual": 0,
            "physical_residual": 0,
        }
    else:
        derivative_orders = {
            str(key): int(value)
            for key, value in component_derivative_orders.items()
        }
    multipliers = {component: 1.0 for component in derivative_orders}
    if component_multipliers is not None:
        multipliers.update({str(key): float(value) for key, value in component_multipliers.items()})

    component_inputs: dict[str, PrimitiveCauchyTailInput] = {}
    for component, derivative_order in derivative_orders.items():
        if component not in step_ratio_bounds:
            raise ValueError(f"missing step ratio bound for {component}")
        if component not in retained_order_initials:
            raise ValueError(f"missing retained-order initial for {component}")
        if component not in retained_order_increments:
            raise ValueError(f"missing retained-order increment for {component}")
        if derivative_order < 0:
            raise ValueError("component derivative orders must be nonnegative")
        multiplier = float(multipliers[component])
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError("component multipliers must be finite and nonnegative")
        majorant_initial, majorant_growth = _finite_fuchsian_log_branch_derivative_envelope(
            branch,
            derivative_order=derivative_order,
            initial_radius=initial_radius,
            shell_contraction=shell_contraction,
            analytic_disk_fraction=analytic_disk_fraction,
            log_growth_factor=log_growth_factor,
        )
        component_inputs[component] = construct_primitive_cauchy_tail_input(
            majorant_initial=multiplier * majorant_initial,
            majorant_growth=majorant_growth,
            step_ratio_bound=float(step_ratio_bounds[component]),
            retained_order_initial=int(retained_order_initials[component]),
            retained_order_increment=int(retained_order_increments[component]),
        )

    certificate = FiniteFuchsianLogPrimitiveCauchyInputs(
        initial_radius=initial_radius,
        shell_contraction=shell_contraction,
        analytic_disk_fraction=analytic_disk_fraction,
        log_growth_factor=log_growth_factor,
        component_derivative_orders=derivative_orders,
        component_multipliers=multipliers,
        component_inputs=component_inputs,
    )
    if not certificate.certified:
        raise ValueError("finite Fuchsian-log primitive Cauchy inputs did not certify")
    return certificate


def _finite_fuchsian_log_shape_deviation_bound(
    branch: FiniteFuchsianLogBranch,
    radius: float,
) -> float:
    radius = float(radius)
    if not np.isfinite(radius) or not 0.0 < radius < 1.0:
        raise ValueError("radius must lie in (0, 1)")
    deviation = abs(float(branch.scale_coefficient)) * radius**2 * _coefficient_sup_norm(
        branch.central_shape
    )
    for term in branch.terms:
        if term.power <= 0.0:
            raise ValueError("Fuchsian-log isolation requires positive row powers")
        for log_power, coefficient in term.coefficients_by_log_power.items():
            if int(log_power) < 0:
                raise ValueError("log powers must be nonnegative")
            deviation += _coefficient_sup_norm(coefficient) * _log_monomial_abs_sup_on_radius(
                term.power,
                int(log_power),
                radius,
            )
    return float(deviation)


def _log_monomial_abs_sup_on_radius(power: float, log_power: int, radius: float) -> float:
    power = float(power)
    log_power = int(log_power)
    radius = float(radius)
    if power <= 0.0:
        raise ValueError("power must be positive")
    if log_power < 0:
        raise ValueError("log_power must be nonnegative")
    if not 0.0 < radius < 1.0:
        raise ValueError("radius must lie in (0, 1)")
    if log_power == 0:
        return float(radius**power)
    lower_log = -np.log(radius)
    maximizing_log = max(lower_log, log_power / power)
    return float(np.exp(-power * maximizing_log) * maximizing_log**log_power)


def _finite_fuchsian_log_branch_derivative_envelope(
    branch: FiniteFuchsianLogBranch,
    *,
    derivative_order: int,
    initial_radius: float,
    shell_contraction: float,
    analytic_disk_fraction: float,
    log_growth_factor: float,
) -> tuple[float, float]:
    contributions: list[tuple[float, float]] = []
    central_norm = _coefficient_sup_norm(branch.central_shape)
    if derivative_order == 0:
        contributions.append((central_norm, 1.0))
    scale_coefficients = {0: float(branch.scale_coefficient) * branch.central_shape}
    scale_coefficients, scale_power = _differentiate_log_polynomial_coefficients(
        2.0,
        scale_coefficients,
        derivative_order,
    )
    if scale_coefficients:
        contributions.append(
            _log_polynomial_shell_envelope(
                scale_power,
                scale_coefficients,
                initial_radius=initial_radius,
                shell_contraction=shell_contraction,
                analytic_disk_fraction=analytic_disk_fraction,
                log_growth_factor=log_growth_factor,
            )
        )
    for term in branch.terms:
        coefficients, power = _differentiate_log_polynomial_coefficients(
            term.power,
            term.coefficients_by_log_power,
            derivative_order,
        )
        if coefficients:
            contributions.append(
                _log_polynomial_shell_envelope(
                    power,
                    coefficients,
                    initial_radius=initial_radius,
                    shell_contraction=shell_contraction,
                    analytic_disk_fraction=analytic_disk_fraction,
                    log_growth_factor=log_growth_factor,
                )
            )
    if not contributions:
        return 0.0, 0.0
    return (
        float(sum(initial for initial, _growth in contributions)),
        float(max(growth for _initial, growth in contributions)),
    )


def _differentiate_log_polynomial_coefficients(
    power: float,
    coefficients_by_log_power: Mapping[int, Array],
    derivative_order: int,
) -> tuple[dict[int, Array], float]:
    coefficients = {
        int(log_power): np.asarray(coefficient, dtype=float)
        for log_power, coefficient in coefficients_by_log_power.items()
    }
    current_power = float(power)
    for _ in range(int(derivative_order)):
        next_coefficients: dict[int, Array] = {}
        for log_power, coefficient in coefficients.items():
            next_coefficients[log_power] = (
                next_coefficients.get(log_power, np.zeros_like(coefficient))
                + current_power * coefficient
            )
            if log_power > 0:
                next_coefficients[log_power - 1] = (
                    next_coefficients.get(log_power - 1, np.zeros_like(coefficient))
                    + log_power * coefficient
                )
        coefficients = {
            log_power: coefficient
            for log_power, coefficient in next_coefficients.items()
            if _coefficient_sup_norm(coefficient) > 1e-15
        }
        current_power -= 1.0
    return coefficients, current_power


def _log_polynomial_shell_envelope(
    power: float,
    coefficients_by_log_power: Mapping[int, Array],
    *,
    initial_radius: float,
    shell_contraction: float,
    analytic_disk_fraction: float,
    log_growth_factor: float,
) -> tuple[float, float]:
    log_base = abs(np.log(initial_radius)) - np.log(1.0 - analytic_disk_fraction)
    shell_log_step = abs(np.log(shell_contraction))
    coefficient_sum = 0.0
    for log_power, coefficient in coefficients_by_log_power.items():
        coefficient_sum += _coefficient_sup_norm(coefficient) * _log_power_exponential_prefactor(
            int(log_power),
            log_base=log_base,
            shell_log_step=shell_log_step,
            log_growth_factor=log_growth_factor,
        )
    if coefficient_sum == 0.0:
        return 0.0, 0.0
    if power >= 0.0:
        radius_initial = (initial_radius * (1.0 + analytic_disk_fraction)) ** power
    else:
        radius_initial = (initial_radius * (1.0 - analytic_disk_fraction)) ** power
    return (
        float(coefficient_sum * radius_initial),
        float(shell_contraction**power * log_growth_factor),
    )


def _log_power_exponential_prefactor(
    degree: int,
    *,
    log_base: float,
    shell_log_step: float,
    log_growth_factor: float,
) -> float:
    degree = int(degree)
    if degree <= 0:
        return 1.0
    exponential_rate = np.log(log_growth_factor) / shell_log_step
    return float(
        np.exp(exponential_rate * log_base)
        * (degree / (np.e * exponential_rate)) ** degree
    )


def _coefficient_sup_norm(coefficient: Array) -> float:
    coefficient = np.asarray(coefficient, dtype=float)
    if coefficient.size == 0:
        return 0.0
    return float(np.max(np.abs(coefficient)))


def _minimum_pair_distance(positions: Array) -> float:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2:
        return 0.0
    distances = [
        float(np.linalg.norm(positions[j] - positions[i]))
        for i in range(positions.shape[0])
        for j in range(i + 1, positions.shape[0])
    ]
    return float(min(distances)) if distances else 0.0


def _polynomial_product(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    product = [0.0 for _ in range(len(left) + len(right) - 1)]
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            product[left_degree + right_degree] += left_value * right_value
    return tuple(product)


def _polynomial_power(polynomial: tuple[float, ...], power: int) -> tuple[float, ...]:
    if power < 0:
        raise ValueError("polynomial power must be nonnegative")
    result = (1.0,)
    for _ in range(power):
        result = _polynomial_product(result, polynomial)
    return result


def _polynomial_add_scaled(
    left: tuple[float, ...],
    right: tuple[float, ...],
    scale: float,
) -> tuple[float, ...]:
    length = max(len(left), len(right))
    result = [0.0 for _ in range(length)]
    for index, value in enumerate(left):
        result[index] += value
    for index, value in enumerate(right):
        result[index] += scale * value
    while len(result) > 1 and abs(result[-1]) < 1e-15:
        result.pop()
    return tuple(result)


@dataclass(frozen=True)
class StableLogMode:
    """One stable normal-form mode converted to a Fuchsian-log polynomial."""

    index: int
    stable_rate: float
    radial_decay: float
    selector: float
    polynomial_in_time: tuple[float, ...]
    forcing_polynomial_in_time: tuple[float, ...] = ()

    @property
    def fuchsian_power(self) -> float:
        return float(2.0 * self.stable_rate / self.radial_decay)

    @property
    def log_conversion(self) -> float:
        return float(-2.0 / self.radial_decay)

    @property
    def log_coefficients(self) -> tuple[float, ...]:
        return tuple(
            coefficient * self.log_conversion**degree
            for degree, coefficient in enumerate(self.polynomial_in_time)
        )

    def polynomial_at_time(self, parameter: float) -> float:
        parameter = float(parameter)
        return float(
            sum(
                coefficient * parameter**degree
                for degree, coefficient in enumerate(self.polynomial_in_time)
            )
        )

    def polynomial_at_log(self, logarithm: float) -> float:
        logarithm = float(logarithm)
        return float(
            sum(
                coefficient * logarithm**degree
                for degree, coefficient in enumerate(self.log_coefficients)
            )
        )

    def value_at_time(self, parameter: float) -> float:
        return float(np.exp(-self.stable_rate * parameter) * self.polynomial_at_time(parameter))

    def value_at_tau(self, tau: float) -> float:
        tau = float(tau)
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        return float(tau**self.fuchsian_power * self.polynomial_at_log(np.log(tau)))

    def raw_quotient_at_tau(self, tau: float) -> float:
        tau = float(tau)
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        return float(self.value_at_tau(tau) / tau**self.fuchsian_power)

    def recovered_selector_at_tau(self, tau: float) -> float:
        tau = float(tau)
        logarithm = np.log(tau)
        forced = sum(
            coefficient * logarithm**degree
            for degree, coefficient in enumerate(self.log_coefficients)
            if degree > 0
        )
        return float(self.raw_quotient_at_tau(tau) - forced)


@dataclass(frozen=True)
class StableResonanceTerm:
    """A monomial forcing term in a triangular stable normal form."""

    coupling: float
    powers: tuple[int, ...]


@dataclass(frozen=True)
class StableLogSelectorChain:
    """Finite triangular stable normal-form selector chain."""

    radial_decay: float
    stable_rates: tuple[float, ...]
    modes: Mapping[int, StableLogMode]
    resonant_terms: Mapping[int, tuple[StableResonanceTerm, ...]]

    @property
    def certified(self) -> bool:
        if self.radial_decay <= 0.0 or not np.isfinite(self.radial_decay):
            return False
        for index, mode in self.modes.items():
            if index < 0 or index >= len(self.stable_rates):
                return False
            if not np.isclose(mode.stable_rate, self.stable_rates[index]):
                return False
            if not np.isclose(mode.polynomial_in_time[0], mode.selector):
                return False
        return True

    def mode(self, index: int) -> StableLogMode:
        return self.modes[int(index)]

    def recovered_selectors_at_tau(self, tau: float) -> dict[int, float]:
        return {
            index: mode.recovered_selector_at_tau(tau)
            for index, mode in self.modes.items()
        }


def construct_stable_log_selector_chain(
    *,
    radial_decay: float,
    stable_rates: tuple[float, ...],
    source_amplitudes: Mapping[int, float],
    resonant_rows: Mapping[int, tuple[StableResonanceTerm, ...]],
    selectors: Mapping[int, float],
    resonance_tolerance: float = 1e-12,
) -> StableLogSelectorChain:
    """Construct finite Fuchsian-log selector polynomials from a triangular normal form."""

    radial_decay = float(radial_decay)
    if radial_decay <= 0.0 or not np.isfinite(radial_decay):
        raise ValueError("radial_decay must be positive")
    stable_rates = tuple(float(rate) for rate in stable_rates)
    modes: dict[int, StableLogMode] = {}
    for index, amplitude in sorted(source_amplitudes.items()):
        index = int(index)
        if index < 0 or index >= len(stable_rates):
            raise ValueError("source index is out of range")
        modes[index] = StableLogMode(
            index=index,
            stable_rate=stable_rates[index],
            radial_decay=radial_decay,
            selector=float(amplitude),
            polynomial_in_time=(float(amplitude),),
        )

    normalized_rows = {
        int(index): tuple(terms)
        for index, terms in resonant_rows.items()
    }
    for target_index in sorted(normalized_rows):
        if target_index in modes:
            raise ValueError("target row already supplied as a source")
        if target_index not in selectors:
            raise ValueError("missing selector for resonant row")
        if target_index < 0 or target_index >= len(stable_rates):
            raise ValueError("target index is out of range")
        forcing: tuple[float, ...] = (0.0,)
        for term in normalized_rows[target_index]:
            if len(term.powers) != len(stable_rates):
                raise ValueError("resonance power vector has wrong length")
            resonant_rate = sum(
                power * rate
                for power, rate in zip(term.powers, stable_rates)
            )
            if not np.isclose(
                resonant_rate,
                stable_rates[target_index],
                rtol=0.0,
                atol=resonance_tolerance,
            ):
                raise ValueError("forcing monomial is not resonant with target")
            monomial = (1.0,)
            for source_index, power in enumerate(term.powers):
                if power == 0:
                    continue
                if source_index not in modes:
                    raise ValueError("resonant row is not triangular")
                if source_index >= target_index:
                    raise ValueError("resonant row uses a non-earlier mode")
                monomial = _polynomial_product(
                    monomial,
                    _polynomial_power(modes[source_index].polynomial_in_time, power),
                )
            forcing = _polynomial_add_scaled(forcing, monomial, float(term.coupling))
        integrated = [float(selectors[target_index])]
        integrated.extend(
            coefficient / float(degree + 1)
            for degree, coefficient in enumerate(forcing)
        )
        while len(integrated) > 1 and abs(integrated[-1]) < 1e-15:
            integrated.pop()
        modes[target_index] = StableLogMode(
            index=target_index,
            stable_rate=stable_rates[target_index],
            radial_decay=radial_decay,
            selector=float(selectors[target_index]),
            polynomial_in_time=tuple(integrated),
            forcing_polynomial_in_time=forcing,
        )

    chain = StableLogSelectorChain(
        radial_decay=radial_decay,
        stable_rates=stable_rates,
        modes=modes,
        resonant_terms=normalized_rows,
    )
    if not chain.certified:
        raise ValueError("stable log selector chain did not certify")
    return chain


def construct_finite_fuchsian_log_branch_from_stable_chain(
    *,
    masses: Array,
    central_shape: Array,
    scale_coefficient: float,
    chain: StableLogSelectorChain,
    mode_shapes: Mapping[int, Array],
    minimum_non_scale_power: float = 1.0,
) -> FiniteFuchsianLogBranch:
    """Project a finite stable selector chain into a finite Fuchsian-log branch."""

    if not chain.certified:
        raise ValueError("stable selector chain did not certify")
    masses = np.asarray(masses, dtype=float)
    central_shape = np.asarray(central_shape, dtype=float)
    if masses.ndim != 1 or central_shape.ndim != 2 or central_shape.shape[0] != masses.size:
        raise ValueError("central_shape must have one row per mass")
    if np.any(masses <= 0.0) or not np.all(np.isfinite(masses)):
        raise ValueError("masses must be positive and finite")
    if not np.all(np.isfinite(central_shape)):
        raise ValueError("central_shape must be finite")
    minimum_non_scale_power = float(minimum_non_scale_power)
    if not np.isfinite(minimum_non_scale_power):
        raise ValueError("minimum_non_scale_power must be finite")
    central_norm = mass_inner_product(masses, central_shape, central_shape)
    if not np.isfinite(central_norm) or central_norm <= 0.0:
        raise ValueError("central_shape must be nonzero")
    central_unit = central_shape / np.sqrt(central_norm)
    terms: list[FuchsianLogTerm] = []
    for index in sorted(chain.modes):
        if index not in mode_shapes:
            raise ValueError(f"missing mode shape for stable mode {index}")
        mode = chain.mode(index)
        if mode.fuchsian_power <= minimum_non_scale_power:
            raise ValueError("non-scale Fuchsian powers must be greater than one")
        basis = np.asarray(mode_shapes[index], dtype=float)
        if basis.shape != central_shape.shape:
            raise ValueError("mode shape does not match central_shape")
        if not np.all(np.isfinite(basis)):
            raise ValueError("mode shape must be finite")
        norm = np.sqrt(mass_inner_product(masses, basis, basis))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("mode shape must be nonzero")
        basis = basis / norm
        central_component = mass_inner_product(masses, basis, central_unit)
        if abs(central_component) > 1e-10:
            raise ValueError("mode shape is not mass-orthogonal to central_shape")
        coefficients = {
            log_power: coefficient * basis
            for log_power, coefficient in enumerate(mode.log_coefficients)
            if abs(coefficient) > 1e-15 or log_power == 0
        }
        terms.append(
            FuchsianLogTerm(
                power=mode.fuchsian_power,
                coefficients_by_log_power=coefficients,
                selector_basis=(basis,),
            )
        )
    return FiniteFuchsianLogBranch(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=float(scale_coefficient),
        terms=tuple(terms),
    )


def _mass_project_onto_basis(
    masses: Array,
    vector: Array,
    basis: tuple[Array, ...],
    body_shape: tuple[int, int],
) -> Array:
    projection = np.zeros(body_shape, dtype=float)
    shaped = vector.reshape(body_shape)
    for basis_vector in basis:
        projection += mass_inner_product(masses, shaped, basis_vector) * basis_vector
    return projection.reshape(-1)


def construct_fuchsian_log_row_solution(
    *,
    power: float,
    row_operator: Array,
    masses: Array,
    kernel_basis: tuple[Array, ...],
    forcing_by_log_power: Mapping[int, Array],
    selector_by_basis: tuple[float, ...] | None = None,
) -> FuchsianLogRowSolution:
    """Solve one resonant Fuchsian-log row by descending log degree."""

    power = float(power)
    row_operator = np.asarray(row_operator, dtype=float)
    masses = np.asarray(masses, dtype=float)
    kernel_basis = tuple(np.asarray(basis, dtype=float) for basis in kernel_basis)
    if not kernel_basis:
        raise ValueError("kernel_basis must be nonempty")
    body_shape = kernel_basis[0].shape
    if selector_by_basis is None:
        selector_by_basis = tuple(0.0 for _basis in kernel_basis)
    if len(selector_by_basis) != len(kernel_basis):
        raise ValueError("selector_by_basis must match kernel_basis")
    for basis in kernel_basis:
        if basis.shape != body_shape:
            raise ValueError("all kernel basis arrays must have the same shape")
        residual = row_operator @ basis.reshape(-1)
        if np.linalg.norm(residual, ord=np.inf) > 1e-9:
            raise ValueError("kernel_basis vector is not in the row-operator kernel")

    normalized_basis: list[Array] = []
    for basis in kernel_basis:
        candidate = basis.copy()
        for previous in normalized_basis:
            candidate -= mass_inner_product(masses, candidate, previous) * previous
        norm = np.sqrt(mass_inner_product(masses, candidate, candidate))
        if norm <= 1e-12:
            raise ValueError("kernel_basis is linearly dependent")
        normalized_basis.append(candidate / norm)
    kernel_basis = tuple(normalized_basis)

    forcing = {
        int(log_power): np.asarray(value, dtype=float).reshape(body_shape)
        for log_power, value in forcing_by_log_power.items()
    }
    max_forcing = max(forcing, default=0)
    coefficients: dict[int, Array] = {
        log_power: np.zeros(body_shape, dtype=float)
        for log_power in range(max_forcing + 2)
    }

    for log_power in range(max_forcing, -1, -1):
        next_one = coefficients.get(log_power + 1, np.zeros(body_shape)).reshape(-1)
        next_two = coefficients.get(log_power + 2, np.zeros(body_shape)).reshape(-1)
        rhs_before_kernel = (
            forcing.get(log_power, np.zeros(body_shape)).reshape(-1)
            - (log_power + 1.0) * (2.0 * power + 1.0) * next_one
            - (log_power + 2.0) * (log_power + 1.0) * next_two
        )
        kernel_rhs = _mass_project_onto_basis(
            masses,
            rhs_before_kernel,
            kernel_basis,
            body_shape,
        )
        coefficients[log_power + 1] = coefficients[log_power + 1] + (
            kernel_rhs.reshape(body_shape)
            / ((log_power + 1.0) * (2.0 * power + 1.0))
        )
        rhs = rhs_before_kernel - kernel_rhs
        range_solution = np.linalg.lstsq(row_operator, rhs, rcond=None)[0]
        range_solution -= _mass_project_onto_basis(
            masses,
            range_solution,
            kernel_basis,
            body_shape,
        )
        coefficients[log_power] = range_solution.reshape(body_shape)

    selector = np.zeros(body_shape, dtype=float)
    for selector_value, basis in zip(selector_by_basis, kernel_basis):
        selector += float(selector_value) * basis
    coefficients[0] = coefficients[0] + selector

    coefficients = {
        log_power: coefficient
        for log_power, coefficient in coefficients.items()
        if np.linalg.norm(coefficient, ord=np.inf) > 1e-15
        or log_power <= max_forcing + 1
    }
    solution = FuchsianLogRowSolution(
        power=power,
        row_operator=row_operator,
        masses=masses,
        body_shape=body_shape,
        kernel_basis=kernel_basis,
        forcing_by_log_power=forcing,
        coefficients_by_log_power=coefficients,
        selector_by_basis=tuple(float(value) for value in selector_by_basis),
    )
    if not solution.row_certified:
        raise ValueError("Fuchsian-log row did not certify its triangular equations")
    return solution


def construct_fuchsian_shape_branch(
    *,
    masses: Array,
    central_shape: Array,
    powers: tuple[float, ...],
    selected_coefficients: Mapping[MultiIndex, Array],
    max_total_degree: int,
    scale_index: MultiIndex | None = None,
    singular_value_floor: float = 1e-10,
) -> FuchsianShapeBranch:
    """Construct a finite nonresonant Fuchsian shape recurrence."""

    masses = np.asarray(masses, dtype=float)
    central_shape = np.asarray(central_shape, dtype=float)
    powers = tuple(float(power) for power in powers)
    dimension = len(powers)
    zero_index = tuple(0 for _ in powers)
    coefficients: dict[MultiIndex, Array] = {zero_index: central_shape}
    selected_indices = frozenset(tuple(index) for index in selected_coefficients)
    for index, coefficient in selected_coefficients.items():
        index = tuple(index)
        if len(index) != dimension:
            raise ValueError("selected index dimension does not match powers")
        if sum(index) > max_total_degree:
            raise ValueError("selected index is above max_total_degree")
        coefficients[index] = np.asarray(coefficient, dtype=float)

    derivative_matrix = linearized_acceleration_matrix(central_shape, masses)
    residual_norms: dict[MultiIndex, float] = {}
    singular_floors: dict[MultiIndex, float] = {}
    solved_indices: list[MultiIndex] = []

    for total_degree in range(1, max_total_degree + 1):
        for index in multi_indices_with_total(total_degree, dimension):
            weighted_exponent = sum(coordinate * power for coordinate, power in zip(index, powers))
            multiplier = ((weighted_exponent + 2.0) * (weighted_exponent - 1.0)) / 9.0
            operator = multiplier * np.eye(central_shape.size) - derivative_matrix
            trial_coefficients = dict(coefficients)
            trial_coefficients[index] = np.zeros_like(central_shape)
            known_term = multivariate_acceleration_coefficient(
                trial_coefficients,
                masses,
                index,
            ).reshape(-1)
            singular_values = np.linalg.svd(operator, compute_uv=False)
            singular_floors[index] = float(np.min(singular_values))
            if index in selected_indices:
                residual = operator @ coefficients[index].reshape(-1) - known_term
                residual_norms[index] = float(np.linalg.norm(residual, ord=np.inf))
                continue
            if singular_floors[index] <= singular_value_floor:
                raise ValueError(f"Fuchsian recurrence is resonant at index {index}")
            solved = np.linalg.solve(operator, known_term).reshape(central_shape.shape)
            coefficients[index] = solved
            solved_indices.append(index)
            residual = operator @ solved.reshape(-1) - known_term
            residual_norms[index] = float(np.linalg.norm(residual, ord=np.inf))

    scale_coefficient = None
    if scale_index is not None:
        scale_index = tuple(scale_index)
        if scale_index not in coefficients:
            raise ValueError("scale_index is not present in coefficients")
        inertia = mass_inner_product(masses, central_shape, central_shape)
        scale_coefficient = mass_inner_product(
            masses,
            coefficients[scale_index],
            central_shape,
        ) / inertia
        scale_residual = coefficients[scale_index] - scale_coefficient * central_shape
        if np.linalg.norm(scale_residual, ord=np.inf) > 1e-10:
            raise ValueError("scale_index coefficient is not parallel to central_shape")

    branch = FuchsianShapeBranch(
        masses=masses,
        central_shape=central_shape,
        powers=powers,
        coefficients=coefficients,
        selected_indices=selected_indices,
        solved_indices=tuple(solved_indices),
        derivative_matrix=derivative_matrix,
        recurrence_residual_norms=residual_norms,
        singular_value_floors=singular_floors,
        max_total_degree=int(max_total_degree),
        scale_index=scale_index,
        scale_coefficient=scale_coefficient,
    )
    if not branch.recurrence_certified:
        raise ValueError("constructed Fuchsian branch did not certify recurrence")
    return branch


def construct_fuchsian_selector_continuation(
    *,
    masses: Array,
    central_shape: Array,
    powers: tuple[float, ...],
    incoming_selected_coefficients: Mapping[MultiIndex, Array],
    max_total_degree: int,
    outgoing_selected_coefficients: Mapping[MultiIndex, Array] | None = None,
    scale_index: MultiIndex | None = None,
) -> FuchsianSelectorContinuation:
    """Construct incoming and outgoing branches from Fuchsian selector data."""

    if outgoing_selected_coefficients is None:
        outgoing_selected_coefficients = incoming_selected_coefficients
    incoming = construct_fuchsian_shape_branch(
        masses=masses,
        central_shape=central_shape,
        powers=powers,
        selected_coefficients=incoming_selected_coefficients,
        max_total_degree=max_total_degree,
        scale_index=scale_index,
    )
    outgoing = construct_fuchsian_shape_branch(
        masses=masses,
        central_shape=central_shape,
        powers=powers,
        selected_coefficients=outgoing_selected_coefficients,
        max_total_degree=max_total_degree,
        scale_index=scale_index,
    )
    return FuchsianSelectorContinuation(incoming=incoming, outgoing=outgoing)


def derive_identity_fuchsian_selector_continuation_from_incoming_branch(
    incoming_branch: FuchsianShapeBranch,
) -> FuchsianSelectorContinuation:
    """Derive the identity-selector continuation from a certified incoming branch.

    This is the local zero-angular entry bridge for the finite nonresonant
    Fuchsian class: the incoming branch already carries the selected fractional
    and scale coefficients.  The identity rule reuses exactly those selected
    rows to construct the outgoing branch, so no separate selector table can be
    supplied by hand at theorem level.
    """

    if not isinstance(incoming_branch, FuchsianShapeBranch):
        raise TypeError("incoming_branch must be a FuchsianShapeBranch")
    if not incoming_branch.recurrence_certified:
        raise ValueError("incoming Fuchsian branch is not recurrence-certified")
    if incoming_branch.scale_index is None:
        raise ValueError(
            "incoming Fuchsian branch must expose a scale_index to certify finite-energy matching"
        )
    selected_indices = set(incoming_branch.selected_indices)
    selected_indices.add(incoming_branch.scale_index)
    selected_coefficients = {
        index: np.asarray(incoming_branch.coefficients[index], dtype=float).copy()
        for index in selected_indices
    }
    outgoing = construct_fuchsian_shape_branch(
        masses=incoming_branch.masses,
        central_shape=incoming_branch.central_shape,
        powers=incoming_branch.powers,
        selected_coefficients=selected_coefficients,
        max_total_degree=incoming_branch.max_total_degree,
        scale_index=incoming_branch.scale_index,
    )
    continuation = FuchsianSelectorContinuation(
        incoming=incoming_branch,
        outgoing=outgoing,
    )
    if not continuation.identity_selector_certified:
        raise ValueError("derived identity Fuchsian selector continuation did not certify")
    return continuation
