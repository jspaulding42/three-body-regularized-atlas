"""Constructive zero-angular total-collision branch helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import accelerations, pack_state


Array = np.ndarray


@dataclass(frozen=True)
class HomotheticTotalCollisionScalarMajorantCertificate:
    """Rouche/Cauchy majorant for the homothetic collision scalar ``u(z)``."""

    branch: "HomotheticTotalCollisionBranch"
    z_radius: float
    u_radius: float
    energy_alpha: float
    perturbation_eta: float
    linear_boundary_bound: float
    base_nonlinear_bound: float
    energy_perturbation_bound: float
    cauchy_majorant: float
    source: str = "homothetic_total_collision_scalar_rouche_majorant"

    @property
    def rouche_margin(self) -> float:
        return float(
            self.linear_boundary_bound
            - self.base_nonlinear_bound
            - self.energy_perturbation_bound
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.branch.certified_scaled_central_configuration
            and np.isfinite(self.z_radius)
            and self.z_radius > 0.0
            and np.isfinite(self.u_radius)
            and 0.0 < self.u_radius < 1.0
            and np.isfinite(self.perturbation_eta)
            and self.perturbation_eta < 1.0
            and self.rouche_margin > 0.0
            and np.isfinite(self.cauchy_majorant)
            and self.cauchy_majorant > 0.0
        )

    def scalar_tail_bound(
        self,
        *,
        max_abs_tau: float,
        retained_order: int,
    ) -> float:
        """Cauchy tail for ``u(tau^2)`` after the retained ``z`` degree."""

        max_abs_tau = float(max_abs_tau)
        retained_order = int(retained_order)
        if max_abs_tau < 0.0:
            raise ValueError("max_abs_tau must be nonnegative")
        if retained_order < 0:
            raise ValueError("retained_order must be nonnegative")
        if not self.certified:
            return float("inf")
        theta = max_abs_tau**2 / self.z_radius
        if not 0.0 <= theta < 1.0:
            return float("inf")
        return float(
            self.cauchy_majorant
            * theta ** (retained_order + 1)
            / (1.0 - theta)
        )

    def derivative_combination_tail_bound(
        self,
        *,
        max_abs_tau: float,
        retained_order: int,
    ) -> float:
        """Cauchy tail for ``u(z)+z u'(z)`` after the retained ``z`` degree."""

        max_abs_tau = float(max_abs_tau)
        retained_order = int(retained_order)
        if max_abs_tau < 0.0:
            raise ValueError("max_abs_tau must be nonnegative")
        if retained_order < 0:
            raise ValueError("retained_order must be nonnegative")
        if not self.certified:
            return float("inf")
        theta = max_abs_tau**2 / self.z_radius
        if not 0.0 <= theta < 1.0:
            return float("inf")
        return float(
            self.cauchy_majorant
            * theta ** (retained_order + 1)
            * ((retained_order + 2) - (retained_order + 1) * theta)
            / (1.0 - theta) ** 2
        )


def homothetic_energy_series_coefficients(
    energy_per_inertia: float,
    order: int = 3,
) -> Array:
    """Construct the scalar homothetic total-collision energy series."""

    order = int(order)
    if order < 0:
        raise ValueError("order must be nonnegative")
    energy_per_inertia = float(energy_per_inertia)
    coefficients = np.zeros(order + 1, dtype=float)
    inverse = np.zeros(order + 1, dtype=float)
    coefficients[0] = 1.0
    inverse[0] = 1.0
    alpha = (9.0 / 2.0) * energy_per_inertia
    for degree in range(1, order + 1):
        derivative_square_lower = sum(
            (left_degree + 1)
            * coefficients[left_degree]
            * (degree - left_degree + 1)
            * coefficients[degree - left_degree]
            for left_degree in range(1, degree)
        )
        inverse_lower = sum(
            coefficients[left_degree] * inverse[degree - left_degree]
            for left_degree in range(1, degree)
        )
        forcing = alpha if degree == 1 else 0.0
        coefficients[degree] = (
            forcing - inverse_lower - derivative_square_lower
        ) / (2.0 * degree + 3.0)
        inverse[degree] = -coefficients[degree] - inverse_lower
    return coefficients


def certify_homothetic_total_collision_scalar_majorant(
    branch: "HomotheticTotalCollisionBranch",
    *,
    z_radius: float,
    u_radius: float,
) -> HomotheticTotalCollisionScalarMajorantCertificate:
    """Derive an analytic scalar-tail majorant for homothetic collision.

    With ``q=tau^2 u(z)Q`` and ``z=tau^2``, the energy equation is

    ``(u + z u')^2 = 1/u + alpha z``, where
    ``alpha=(9/2) energy_per_inertia``.

    Equivalently, the collision-time integral gives an analytic equation
    ``H(u,z)=0`` with ``H(1,0)=0`` and linear part ``(3/2)(u-1)``.  This
    certificate checks a conservative Rouche inequality on
    ``|u-1|=u_radius`` and returns a Cauchy majorant for the unique analytic
    scalar branch in ``|z|<z_radius``.
    """

    if not isinstance(branch, HomotheticTotalCollisionBranch):
        raise TypeError("branch must be a HomotheticTotalCollisionBranch")
    z_radius = float(z_radius)
    u_radius = float(u_radius)
    if z_radius <= 0.0:
        raise ValueError("z_radius must be positive")
    if not 0.0 < u_radius < 1.0:
        raise ValueError("u_radius must lie in (0, 1)")
    energy_alpha = 4.5 * float(branch.energy_per_inertia)
    perturbation_eta = abs(energy_alpha) * z_radius * (1.0 + u_radius)
    if perturbation_eta < 1.0:
        energy_perturbation_bound = (
            (1.0 + u_radius) ** 1.5
            * perturbation_eta
            / (2.0 * (1.0 - perturbation_eta) ** 1.5)
        )
    else:
        energy_perturbation_bound = float("inf")
    linear_boundary_bound = 1.5 * u_radius
    base_nonlinear_bound = 0.375 * u_radius**2 / np.sqrt(1.0 - u_radius)
    return HomotheticTotalCollisionScalarMajorantCertificate(
        branch=branch,
        z_radius=z_radius,
        u_radius=u_radius,
        energy_alpha=float(energy_alpha),
        perturbation_eta=float(perturbation_eta),
        linear_boundary_bound=float(linear_boundary_bound),
        base_nonlinear_bound=float(base_nonlinear_bound),
        energy_perturbation_bound=float(energy_perturbation_bound),
        cauchy_majorant=float(1.0 + u_radius),
    )


def _series_product(left: Array, right: Array, order: int) -> Array:
    out = np.zeros(order + 1, dtype=float)
    for degree in range(order + 1):
        out[degree] = sum(left[k] * right[degree - k] for k in range(degree + 1))
    return out


def _series_inverse(coefficients: Array, order: int) -> Array:
    inverse = np.zeros(order + 1, dtype=float)
    inverse[0] = 1.0 / coefficients[0]
    for degree in range(1, order + 1):
        inverse[degree] = -sum(
            coefficients[k] * inverse[degree - k]
            for k in range(1, degree + 1)
        ) / coefficients[0]
    return inverse


@dataclass(frozen=True)
class HomotheticTotalCollisionBranch:
    """Regularized homothetic total-collision branch in cubic time."""

    quadratic_coefficient: Array
    masses: Array
    energy_per_inertia: float = 0.0
    order: int = 8

    @property
    def coefficients(self) -> Array:
        return homothetic_energy_series_coefficients(
            self.energy_per_inertia,
            self.order,
        )

    @property
    def inertia(self) -> float:
        return float(
            np.sum(
                np.asarray(self.masses, dtype=float)[:, None]
                * np.asarray(self.quadratic_coefficient, dtype=float) ** 2
            )
        )

    @property
    def second_regularized_jet(self) -> Array:
        return 2.0 * np.asarray(self.quadratic_coefficient, dtype=float)

    @property
    def fourth_regularized_jet(self) -> Array:
        return 24.0 * self.coefficients[1] * np.asarray(
            self.quadratic_coefficient,
            dtype=float,
        )

    @property
    def expected_fourth_regularized_jet(self) -> Array:
        return (108.0 / 5.0) * self.energy_per_inertia * np.asarray(
            self.quadratic_coefficient,
            dtype=float,
        )

    @property
    def central_configuration_residual(self) -> Array:
        return accelerations(self.quadratic_coefficient, self.masses) + (
            2.0 / 9.0
        ) * self.quadratic_coefficient

    @property
    def certified_scaled_central_configuration(self) -> bool:
        scale = max(1.0, np.linalg.norm(self.quadratic_coefficient, ord=np.inf))
        return bool(
            self.inertia > 0.0
            and np.linalg.norm(self.central_configuration_residual, ord=np.inf)
            <= 1.0e-12 * scale
        )

    def radial_u(self, regularized_time: float) -> float:
        z = float(regularized_time) ** 2
        return float(
            sum(coefficient * z**degree for degree, coefficient in enumerate(self.coefficients))
        )

    def radial_factor(self, regularized_time: float) -> float:
        tau = float(regularized_time)
        return float(tau**2 * self.radial_u(tau))

    def radial_velocity_factor(self, regularized_time: float) -> float:
        tau = float(regularized_time)
        if tau == 0.0:
            raise ValueError("physical velocity is singular at total collision")
        z = tau**2
        derivative_combination = sum(
            (degree + 1.0) * coefficient * z**degree
            for degree, coefficient in enumerate(self.coefficients)
        )
        return float((2.0 / (3.0 * tau)) * derivative_combination)

    def positions_at_tau(self, regularized_time: float) -> Array:
        return self.radial_factor(regularized_time) * self.quadratic_coefficient

    def velocities_at_tau(self, regularized_time: float) -> Array:
        return self.radial_velocity_factor(regularized_time) * self.quadratic_coefficient

    def state_at_tau(self, regularized_time: float) -> Array:
        return pack_state(
            self.positions_at_tau(regularized_time),
            self.velocities_at_tau(regularized_time),
        )

    def energy_recurrence_residual_coefficients(self) -> Array:
        order = int(self.order)
        coefficients = self.coefficients
        derivative_combination = np.array(
            [(degree + 1.0) * coefficients[degree] for degree in range(order + 1)],
            dtype=float,
        )
        left = _series_product(derivative_combination, derivative_combination, order)
        right = _series_inverse(coefficients, order)
        if order >= 1:
            right[1] += (9.0 / 2.0) * self.energy_per_inertia
        return left - right

    def newton_residual_at_tau(self, regularized_time: float) -> Array:
        tau = float(regularized_time)
        if tau == 0.0:
            raise ValueError("Newton residual is only defined on punctured branches")
        positions = self.positions_at_tau(tau)
        radial = self.radial_factor(tau)
        expected_acceleration = -(2.0 / 9.0) * self.quadratic_coefficient / radial**2
        return accelerations(positions, self.masses) - expected_acceleration


def construct_scaled_homothetic_total_collision_branch(
    quadratic_coefficient: Array,
    masses: Array | None = None,
    *,
    energy_per_inertia: float = 0.0,
    order: int = 8,
    require_scaled_central: bool = True,
) -> HomotheticTotalCollisionBranch:
    """Construct a homothetic branch from a coefficient with A(Q)=-(2/9)Q."""

    quadratic_coefficient = np.asarray(quadratic_coefficient, dtype=float)
    masses = np.ones(3, dtype=float) if masses is None else np.asarray(masses, dtype=float)
    if quadratic_coefficient.ndim != 2 or quadratic_coefficient.shape[0] != 3:
        raise ValueError("quadratic_coefficient must have shape (3, dimension)")
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must have shape (3,) and be positive")
    branch = HomotheticTotalCollisionBranch(
        quadratic_coefficient=quadratic_coefficient,
        masses=masses,
        energy_per_inertia=float(energy_per_inertia),
        order=int(order),
    )
    if require_scaled_central and not branch.certified_scaled_central_configuration:
        raise ValueError("quadratic_coefficient must satisfy A(Q)=-(2/9)Q")
    return branch


def construct_homothetic_total_collision_branch(
    central_configuration: Array,
    central_lambda: float,
    masses: Array | None = None,
    *,
    energy_per_inertia: float = 0.0,
    order: int = 8,
) -> HomotheticTotalCollisionBranch:
    """Scale a central configuration and construct its homothetic branch."""

    central_configuration = np.asarray(central_configuration, dtype=float)
    central_lambda = float(central_lambda)
    if central_lambda <= 0.0:
        raise ValueError("central_lambda must be positive")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    return construct_scaled_homothetic_total_collision_branch(
        scale_factor * central_configuration,
        masses,
        energy_per_inertia=energy_per_inertia,
        order=order,
    )
