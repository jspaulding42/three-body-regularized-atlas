"""Finite-jet entry certificates for zero-angular total-collision selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np

from .dynamics import accelerations
from .triple_collision import HomotheticTotalCollisionBranch


Array = np.ndarray


@dataclass(frozen=True)
class CubicTimeEntryObligation:
    """One algebraic obligation in a cubic-time total-collision entry check."""

    obligation: str
    certified: bool
    detail: str
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "certified", self.certified is True)
        object.__setattr__(
            self,
            "required",
            self.required if type(self.required) is bool else True,
        )


def _cubic_time_obligation_ledger_certified(
    obligations: tuple[Any, ...],
) -> bool:
    return bool(
        obligations
        and any(
            isinstance(obligation, CubicTimeEntryObligation)
            and obligation.required
            for obligation in obligations
        )
        and all(
            isinstance(obligation, CubicTimeEntryObligation)
            for obligation in obligations
        )
        and all(
            obligation.certified is True
            for obligation in obligations
            if obligation.required
        )
    )


def _cubic_time_obligation_ledger_missing(
    obligations: tuple[Any, ...],
    *,
    ledger_name: str,
) -> tuple[str, ...]:
    missing: list[str] = []
    if not obligations:
        missing.append(f"{ledger_name}_obligations_present")
    if obligations and not any(
        isinstance(obligation, CubicTimeEntryObligation)
        and obligation.required
        for obligation in obligations
    ):
        missing.append(f"{ledger_name}_required_obligation_present")
    for obligation in obligations:
        if not isinstance(obligation, CubicTimeEntryObligation):
            missing.append(f"{ledger_name}_obligation_type")
            continue
        if obligation.required and obligation.certified is not True:
            missing.append(obligation.obligation)
    return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class CubicTimeLeadingJetEntryCertificate:
    """Necessary leading coefficient theorem for cubic-time total collision.

    For a branch written in regularized time as
    ``q(tau)=tau^2 C+O(tau^3)``, with physical time ``t=tau^3``, the
    coefficient of ``tau^-4`` in ``d^2q/dt^2-A(q)`` is
    ``-(2/9)C-A(C)``.  A Newtonian total-collision germ with this cubic-time
    expansion must therefore have ``A(C)=-(2/9)C`` after center-of-mass
    reduction.
    """

    masses: tuple[float, ...]
    quadratic_coefficient: Array
    center_of_mass_residual: float
    central_equation_residual: float
    minimum_pair_distance: float
    tolerance: float
    obligations: tuple[CubicTimeEntryObligation, ...]
    theorem_id: str = "cubic_time_total_collision_leading_jet_is_central"
    statement: str = (
        "A cubic-time total-collision germ q=tau^2 C+O(tau^3), t=tau^3, "
        "has collision-free centered leading shape C satisfying A(C)=-(2/9)C."
    )
    proof_sketch: str = (
        "Substitute q=tau^2 C+O(tau^3) and t=tau^3 into Newton's equation. "
        "The physical acceleration contributes -(2/9)C tau^-4, while the "
        "Newtonian force contributes A(C) tau^-4 by homogeneity.  Vanishing "
        "of the tau^-4 coefficient gives A(C)=-(2/9)C.  The center-of-mass "
        "tau^2 coefficient is zero in the centered frame."
    )

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and _cubic_time_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _cubic_time_obligation_ledger_missing(
            self.obligations,
            ledger_name="cubic_time_leading_jet_entry",
        )

    @property
    def arbitrary_fuchsian_entry_theorem_claimed(self) -> bool:
        return False


@dataclass(frozen=True)
class CubicTimeCubicJetKernelCertificate:
    """Necessary cubic coefficient theorem for cubic-time total collision.

    Once ``C`` satisfies the leading central equation, the coefficient of
    ``tau^-3`` in ``d^2q/dt^2-A(q)`` for
    ``q(tau)=tau^2 C+tau^3 D+O(tau^4)`` is ``-DA(C)D``.  Thus the cubic row
    must lie in the kernel of the linearized acceleration operator at ``C``.
    """

    leading_certificate: CubicTimeLeadingJetEntryCertificate
    cubic_coefficient: Array
    linearized_acceleration: Array
    kernel_residual: float
    tolerance: float
    obligations: tuple[CubicTimeEntryObligation, ...]
    theorem_id: str = "cubic_time_total_collision_cubic_jet_kernel_condition"
    statement: str = (
        "For q=tau^2 C+tau^3 D+O(tau^4), t=tau^3, after the leading central "
        "equation is satisfied, the cubic coefficient D must satisfy DA(C)D=0."
    )
    proof_sketch: str = (
        "The tau^3 row has zero direct physical-acceleration coefficient "
        "because p(p-3)/9=0 at p=3.  The force expansion contributes "
        "DA(C)D tau^-3.  The tau^-3 coefficient in Newton's equation is "
        "therefore -DA(C)D, so it must vanish."
    )

    @property
    def certified(self) -> bool:
        return bool(
            isinstance(
                self.leading_certificate,
                CubicTimeLeadingJetEntryCertificate,
            )
            and self.leading_certificate.certified is True
            and self.statement
            and self.proof_sketch
            and _cubic_time_obligation_ledger_certified(self.obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not isinstance(
            self.leading_certificate,
            CubicTimeLeadingJetEntryCertificate,
        ):
            missing.append("cubic_time_leading_jet_entry_certificate_type")
        elif self.leading_certificate.certified is not True:
            missing.extend(self.leading_certificate.missing_obligations)
        missing.extend(
            _cubic_time_obligation_ledger_missing(
                self.obligations,
                ledger_name="cubic_time_cubic_jet_kernel",
            )
        )
        return tuple(dict.fromkeys(missing))

    @property
    def arbitrary_fuchsian_entry_theorem_claimed(self) -> bool:
        return False


@dataclass(frozen=True)
class FiniteJetSelectorSpec:
    """One finite regularized-jet coordinate used as selector data."""

    name: str
    degree: int
    basis: Array
    reference_coefficients: Array | None = None


@dataclass(frozen=True)
class FiniteJetSelectorCoordinateCertificate:
    """Recovered selector coordinate for one finite regularized jet."""

    name: str
    degree: int
    regularized_power: int
    recovered_value: float
    selected_value: float
    selector_gap: float
    basis_norm: float

    @property
    def certified(self) -> bool:
        return bool(
            self.name
            and self.degree >= 0
            and self.regularized_power == self.degree + 2
            and np.isfinite(self.recovered_value)
            and np.isfinite(self.selected_value)
            and np.isfinite(self.selector_gap)
            and np.isfinite(self.basis_norm)
            and self.basis_norm > 0.0
        )


@dataclass(frozen=True)
class FiniteJetIdentitySelectorEntryCertificate:
    """Finite one-sided jets select a two-sided zero-angular identity branch."""

    masses: tuple[float, ...]
    incoming_energy_limit: float
    selected_energy_limit: float
    energy_gap: float
    coordinate_certificates: tuple[FiniteJetSelectorCoordinateCertificate, ...]
    required_regularized_jet_order: int
    sample_taus: tuple[float, ...]
    newton_residual_bounds: tuple[float, ...]
    angular_momentum_bounds: tuple[float, ...]
    tolerance: float
    incoming_central_coefficient: Array | None = None
    selected_central_coefficient: Array | None = None

    @property
    def max_selector_gap(self) -> float:
        return float(max((c.selector_gap for c in self.coordinate_certificates), default=float("inf")))

    @property
    def max_newton_residual(self) -> float:
        return float(max(self.newton_residual_bounds, default=float("inf")))

    @property
    def max_angular_momentum(self) -> float:
        return float(max(self.angular_momentum_bounds, default=float("inf")))

    @property
    def certified(self) -> bool:
        return bool(
            len(self.masses) == 3
            and all(np.isfinite(mass) and mass > 0.0 for mass in self.masses)
            and self.coordinate_certificates
            and _finite_jet_coordinate_certificates_certified(
                self.coordinate_certificates,
            )
            and np.isfinite(self.incoming_energy_limit)
            and np.isfinite(self.selected_energy_limit)
            and np.isfinite(self.energy_gap)
            and self.required_regularized_jet_order
            == max(certificate.regularized_power for certificate in self.coordinate_certificates)
            and self.sample_taus
            and all(np.isfinite(tau) and tau != 0.0 for tau in self.sample_taus)
            and all(np.isfinite(bound) for bound in self.newton_residual_bounds)
            and all(np.isfinite(bound) for bound in self.angular_momentum_bounds)
            and np.isfinite(self.tolerance)
            and self.tolerance > 0.0
            and self.max_selector_gap <= self.tolerance
            and self.energy_gap <= self.tolerance
            and self.max_newton_residual <= self.tolerance
            and self.max_angular_momentum <= self.tolerance
            and _optional_central_coefficient_certified(
                self.incoming_central_coefficient,
            )
            and _optional_central_coefficient_certified(
                self.selected_central_coefficient,
            )
        )


@dataclass(frozen=True)
class FiniteJetSelectedBranchCertificate:
    """Constructor-derived finite zero-angular branch from selector coordinates."""

    masses: tuple[float, ...]
    coefficients: Array
    selector_specs: tuple[FiniteJetSelectorSpec, ...]
    selected_values: tuple[tuple[str, float], ...]
    selector_operator_residual_bounds: tuple[tuple[str, float], ...]
    selector_gram_singular_value_floors: tuple[tuple[int, float], ...]
    coefficient_residual_bounds: tuple[float, ...]
    sample_taus: tuple[float, ...]
    newton_residual_bounds: tuple[float, ...]
    angular_momentum_bounds: tuple[float, ...]
    tolerance: float
    source: str = "finite_jet_selected_branch_constructor"

    @property
    def max_coefficient_residual(self) -> float:
        return float(max(self.coefficient_residual_bounds, default=float("inf")))

    @property
    def max_selector_operator_residual(self) -> float:
        return float(
            max((bound for _name, bound in self.selector_operator_residual_bounds), default=float("inf"))
        )

    @property
    def min_selector_gram_singular_value_floor(self) -> float:
        return float(
            min((floor for _degree, floor in self.selector_gram_singular_value_floors), default=0.0)
        )

    @property
    def max_newton_residual(self) -> float:
        return float(max(self.newton_residual_bounds, default=float("inf")))

    @property
    def max_angular_momentum(self) -> float:
        return float(max(self.angular_momentum_bounds, default=float("inf")))

    @property
    def required_regularized_jet_order(self) -> int:
        return max((int(spec.degree) + 2 for spec in self.selector_specs), default=0)

    @property
    def certified(self) -> bool:
        coefficients = np.asarray(self.coefficients, dtype=float)
        return bool(
            len(self.masses) == 3
            and all(np.isfinite(mass) and mass > 0.0 for mass in self.masses)
            and coefficients.ndim == 3
            and coefficients.shape[0] >= 3
            and coefficients.shape[1] == 3
            and np.all(np.isfinite(coefficients))
            and _minimum_pair_distance(coefficients[0]) > 0.0
            and self.selector_specs
            and _finite_jet_selector_specs_certified(
                self.selector_specs,
                coefficient_shape=coefficients.shape[1:],
            )
            and {name for name, _value in self.selected_values}
            == {str(spec.name) for spec in self.selector_specs}
            and len(self.selector_operator_residual_bounds) == len(self.selector_specs)
            and all(
                np.isfinite(bound)
                for _name, bound in self.selector_operator_residual_bounds
            )
            and self.selector_gram_singular_value_floors
            and all(
                np.isfinite(floor) and floor > self.tolerance
                for _degree, floor in self.selector_gram_singular_value_floors
            )
            and self.sample_taus
            and all(np.isfinite(tau) and tau != 0.0 for tau in self.sample_taus)
            and all(np.isfinite(bound) for bound in self.coefficient_residual_bounds)
            and all(np.isfinite(bound) for bound in self.newton_residual_bounds)
            and all(np.isfinite(bound) for bound in self.angular_momentum_bounds)
            and np.isfinite(self.tolerance)
            and self.tolerance > 0.0
            and self.max_selector_operator_residual <= self.tolerance
            and self.max_coefficient_residual <= self.tolerance
            and self.max_newton_residual <= self.tolerance
            and self.max_angular_momentum <= self.tolerance
        )


@dataclass(frozen=True)
class FiniteJetDerivedIdentitySelectorEntryCertificate:
    """Incoming finite jets determine the selected outgoing identity branch."""

    incoming_coefficients: Array
    selected_branch: FiniteJetSelectedBranchCertificate
    entry_certificate: FiniteJetIdentitySelectorEntryCertificate
    recovered_selector_values: tuple[tuple[str, float], ...]
    source: str = "finite_jet_identity_selector_from_incoming_constructor"

    @property
    def certified(self) -> bool:
        return bool(
            isinstance(self.selected_branch, FiniteJetSelectedBranchCertificate)
            and self.selected_branch.certified is True
            and isinstance(
                self.entry_certificate,
                FiniteJetIdentitySelectorEntryCertificate,
            )
            and self.entry_certificate.certified is True
            and self.recovered_selector_values == self.selected_branch.selected_values
        )

    @property
    def identity_selector_certified(self) -> bool:
        return self.certified

    @property
    def max_newton_residual(self) -> float:
        return max(
            self.selected_branch.max_newton_residual,
            self.entry_certificate.max_newton_residual,
        )

    @property
    def max_angular_momentum(self) -> float:
        return max(
            self.selected_branch.max_angular_momentum,
            self.entry_certificate.max_angular_momentum,
        )


def certify_cubic_time_total_collision_leading_jet(
    *,
    masses: object,
    quadratic_coefficient: object,
    tolerance: float = 1.0e-8,
) -> CubicTimeLeadingJetEntryCertificate:
    """Certify the necessary leading row of a cubic-time total collision.

    This constructor does not produce a continuation branch.  It checks the
    algebraic coefficient condition forced by Newton's equation at order
    ``tau^-4`` for a supplied leading row ``C``.
    """

    masses_array, leading = _validate_total_collision_leading_shape(
        masses,
        quadratic_coefficient,
    )
    tolerance = _positive_finite_tolerance(tolerance)
    center = _mass_weighted_center(masses_array, leading)
    center_residual = float(np.linalg.norm(center, ord=np.inf))
    pair_distance = _minimum_pair_distance(leading)
    central_residual = float(
        np.linalg.norm(
            accelerations(leading, masses_array) + (2.0 / 9.0) * leading,
            ord=np.inf,
        )
    )
    obligations = (
        CubicTimeEntryObligation(
            obligation="positive_three_body_masses",
            certified=bool(
                masses_array.shape == (3,)
                and np.all(np.isfinite(masses_array))
                and np.all(masses_array > 0.0)
            ),
            detail=f"masses={tuple(float(mass) for mass in masses_array)}",
        ),
        CubicTimeEntryObligation(
            obligation="finite_collision_free_quadratic_shape",
            certified=bool(
                leading.ndim == 2
                and leading.shape[0] == 3
                and leading.shape[1] in (2, 3)
                and np.all(np.isfinite(leading))
                and pair_distance > tolerance
            ),
            detail=f"minimum_pair_distance={pair_distance!r}",
        ),
        CubicTimeEntryObligation(
            obligation="centered_quadratic_total_collision_shape",
            certified=bool(center_residual <= tolerance),
            detail=f"center_of_mass_residual={center_residual!r}",
        ),
        CubicTimeEntryObligation(
            obligation="central_configuration_normalization_A_plus_two_ninths_C",
            certified=bool(central_residual <= tolerance),
            detail=f"central_equation_residual={central_residual!r}",
        ),
        CubicTimeEntryObligation(
            obligation="continuation_not_claimed",
            certified=True,
            detail=(
                "this is a necessary finite-jet entry check only; it does not "
                "derive arbitrary Fuchsian-log entry data or a continuation"
            ),
            required=False,
        ),
    )
    return CubicTimeLeadingJetEntryCertificate(
        masses=tuple(float(mass) for mass in masses_array),
        quadratic_coefficient=leading,
        center_of_mass_residual=center_residual,
        central_equation_residual=central_residual,
        minimum_pair_distance=pair_distance,
        tolerance=tolerance,
        obligations=obligations,
    )


def certify_cubic_time_total_collision_cubic_jet_kernel(
    *,
    masses: object,
    quadratic_coefficient: object,
    cubic_coefficient: object,
    leading_certificate: CubicTimeLeadingJetEntryCertificate | None = None,
    tolerance: float = 1.0e-8,
) -> CubicTimeCubicJetKernelCertificate:
    """Certify the necessary cubic row kernel condition ``DA(C)D=0``."""

    masses_array, leading = _validate_total_collision_leading_shape(
        masses,
        quadratic_coefficient,
    )
    cubic = np.asarray(cubic_coefficient, dtype=float)
    if cubic.shape != leading.shape:
        raise ValueError("cubic_coefficient must match quadratic_coefficient shape")
    if not np.all(np.isfinite(cubic)):
        raise ValueError("cubic_coefficient must be finite")
    tolerance = _positive_finite_tolerance(tolerance)
    if leading_certificate is None:
        leading_certificate = certify_cubic_time_total_collision_leading_jet(
            masses=masses_array,
            quadratic_coefficient=leading,
            tolerance=tolerance,
        )
    leading_matches_inputs = _cubic_time_leading_certificate_matches_inputs(
        leading_certificate,
        masses_array,
        leading,
        tolerance=tolerance,
    )
    linearized = _acceleration_derivative_apply(leading, masses_array, cubic)
    kernel_residual = float(np.linalg.norm(linearized, ord=np.inf))
    cubic_norm = float(np.linalg.norm(cubic, ord=np.inf))
    obligations = (
        CubicTimeEntryObligation(
            obligation="leading_cubic_time_entry_certificate",
            certified=bool(leading_certificate.certified and leading_matches_inputs),
            detail=(
                "missing="
                + ",".join(leading_certificate.missing_obligations)
                if not leading_certificate.certified
                else f"{leading_certificate.theorem_id}; matches_inputs={leading_matches_inputs}"
            ),
        ),
        CubicTimeEntryObligation(
            obligation="finite_cubic_coefficient",
            certified=bool(np.all(np.isfinite(cubic))),
            detail=f"cubic_norm={cubic_norm!r}",
        ),
        CubicTimeEntryObligation(
            obligation="linearized_cubic_jet_kernel_condition",
            certified=bool(kernel_residual <= tolerance),
            detail=f"kernel_residual={kernel_residual!r}",
        ),
        CubicTimeEntryObligation(
            obligation="continuation_not_claimed",
            certified=True,
            detail=(
                "this checks the forced cubic finite-jet condition only; "
                "arbitrary Fuchsian-log entry remains a separate theorem"
            ),
            required=False,
        ),
    )
    return CubicTimeCubicJetKernelCertificate(
        leading_certificate=leading_certificate,
        cubic_coefficient=cubic,
        linearized_acceleration=linearized,
        kernel_residual=kernel_residual,
        tolerance=tolerance,
        obligations=obligations,
    )


def recover_finite_jet_selector_coordinates(
    *,
    masses: object,
    coefficients: object,
    selector_specs: tuple[FiniteJetSelectorSpec, ...],
) -> dict[str, float]:
    """Recover finite selector coordinates from regularized shape coefficients."""

    masses_array, coefficients_array = _validate_finite_jet_inputs(masses, coefficients)
    recovered: dict[str, float] = {}
    for spec in selector_specs:
        recovered[spec.name] = _selector_coordinate(
            masses_array,
            coefficients_array,
            spec,
        )
    return recovered


def construct_finite_jet_selected_branch(
    *,
    masses: object,
    central_coefficient: object,
    selector_specs: tuple[FiniteJetSelectorSpec, ...],
    selected_values: Mapping[str, float],
    order: int,
    sample_taus: tuple[float, ...] = (-0.04, 0.04),
    tolerance: float = 1.0e-8,
) -> FiniteJetSelectedBranchCertificate:
    """Solve the finite regularized jet recurrence with selected resonant rows.

    The expansion is ``q(tau)=sum_d C_d tau^(d+2)``.  At degree ``d`` the
    Newton equation gives

    ``(((d+2)(d-1))/9 I - DA(C_0)) C_d = known_d``.

    Nonresonant rows are solved by least squares.  Selector specs impose the
    finite resonant coordinates by mass-orthogonal projection after subtracting
    any forced reference row.  Certification then checks the coefficient
    recurrence, punctured Newton residual, and zero angular momentum directly.
    """

    masses_array = np.asarray(masses, dtype=float).reshape(-1)
    central = np.asarray(central_coefficient, dtype=float)
    if masses_array.shape != (3,):
        raise ValueError("finite zero-angular branch currently expects three masses")
    if np.any(masses_array <= 0.0) or not np.all(np.isfinite(masses_array)):
        raise ValueError("masses must be positive and finite")
    if central.ndim != 2 or central.shape[0] != 3:
        raise ValueError("central_coefficient must have shape (3, dimension)")
    if not np.all(np.isfinite(central)):
        raise ValueError("central_coefficient must be finite")
    if _minimum_pair_distance(central) <= 0.0:
        raise ValueError("central coefficient must be collision-free")
    order = int(order)
    if order < 2:
        raise ValueError("order must include at least quadratic, cubic, and quartic rows")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if not selector_specs:
        raise ValueError("at least one selector spec is required")
    selector_values = {str(name): float(value) for name, value in dict(selected_values).items()}
    for spec in selector_specs:
        if str(spec.name) not in selector_values:
            raise ValueError(f"missing selected value for selector {spec.name!r}")
        if int(spec.degree) < 0 or int(spec.degree) > order:
            raise ValueError("selector degree is outside the requested order")
        if not np.isfinite(selector_values[str(spec.name)]):
            raise ValueError("selector values must be finite")

    coefficients = np.zeros((order + 1, *central.shape), dtype=float)
    coefficients[0] = central
    derivative_matrix = _acceleration_derivative_matrix(central, masses_array)
    dimension = central.size
    for degree in range(1, order + 1):
        trial = coefficients.copy()
        trial[degree] = 0.0
        known_term = _shape_acceleration_series(trial, masses_array, degree)[
            degree
        ].reshape(-1)
        multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
        operator = multiplier * np.eye(dimension) - derivative_matrix
        solved, *_ = np.linalg.lstsq(operator, known_term, rcond=None)
        coefficients[degree] = solved.reshape(central.shape)
        degree_specs = tuple(spec for spec in selector_specs if int(spec.degree) == degree)
        if degree_specs:
            coefficients[degree] = _apply_selector_rows(
                masses_array,
                coefficients,
                degree,
                degree_specs,
                selector_values,
            )

    selector_operator_residuals = _selector_operator_residual_bounds(
        derivative_matrix,
        selector_specs,
        central.shape,
    )
    selector_gram_floors = _selector_gram_singular_value_floors(
        masses_array,
        selector_specs,
    )
    residual_bounds = _coefficient_recurrence_residual_bounds(
        masses_array,
        coefficients,
    )
    sample_taus = tuple(float(tau) for tau in sample_taus)
    if not sample_taus:
        raise ValueError("at least one punctured sample tau is required")
    newton_bounds = tuple(
        _selected_newton_residual_bound(masses_array, coefficients, tau)
        for tau in sample_taus
    )
    angular_bounds = tuple(
        _selected_angular_momentum_bound(masses_array, coefficients, tau)
        for tau in sample_taus
    )
    certificate = FiniteJetSelectedBranchCertificate(
        masses=tuple(float(mass) for mass in masses_array),
        coefficients=coefficients,
        selector_specs=tuple(selector_specs),
        selected_values=tuple(
            (str(spec.name), float(selector_values[str(spec.name)]))
            for spec in selector_specs
        ),
        selector_operator_residual_bounds=selector_operator_residuals,
        selector_gram_singular_value_floors=selector_gram_floors,
        coefficient_residual_bounds=residual_bounds,
        sample_taus=sample_taus,
        newton_residual_bounds=newton_bounds,
        angular_momentum_bounds=angular_bounds,
        tolerance=tolerance,
    )
    if not certificate.certified:
        raise ValueError("finite-jet selected branch did not certify")
    return certificate


def derive_finite_jet_identity_selector_entry_from_incoming(
    *,
    masses: object,
    incoming_coefficients: object,
    selector_specs: tuple[FiniteJetSelectorSpec, ...],
    order: int | None = None,
    sample_taus: tuple[float, ...] = (-0.04, 0.04),
    tolerance: float = 1.0e-8,
) -> FiniteJetDerivedIdentitySelectorEntryCertificate:
    """Recover selector coordinates from incoming finite jets and continue them.

    This is the executable finite-jet bridge for the zero-angular selector
    convention: finite incoming regularized coefficients determine the selected
    outgoing branch by the identity rule.  The returned certificate includes
    both the constructor-derived selected branch and the identity-entry check.
    """

    masses_array, incoming = _validate_finite_jet_inputs(masses, incoming_coefficients)
    if order is None:
        order = incoming.shape[0] - 1
    order = int(order)
    if order >= incoming.shape[0]:
        raise ValueError("order cannot exceed incoming coefficient order")
    recovered = recover_finite_jet_selector_coordinates(
        masses=masses_array,
        coefficients=incoming,
        selector_specs=selector_specs,
    )
    selected_branch = construct_finite_jet_selected_branch(
        masses=masses_array,
        central_coefficient=incoming[0],
        selector_specs=selector_specs,
        selected_values=recovered,
        order=order,
        sample_taus=sample_taus,
        tolerance=tolerance,
    )
    entry = certify_finite_jet_identity_selector_entry(
        masses=masses_array,
        incoming_coefficients=incoming[: order + 1],
        selected_coefficients=selected_branch.coefficients,
        selector_specs=selector_specs,
        sample_taus=sample_taus,
        tolerance=tolerance,
    )
    certificate = FiniteJetDerivedIdentitySelectorEntryCertificate(
        incoming_coefficients=incoming[: order + 1],
        selected_branch=selected_branch,
        entry_certificate=entry,
        recovered_selector_values=tuple(
            (str(spec.name), float(recovered[str(spec.name)]))
            for spec in selector_specs
        ),
    )
    if not certificate.certified:
        raise ValueError("finite-jet identity selector entry did not certify")
    return certificate


def certify_finite_jet_identity_selector_entry(
    *,
    masses: object,
    incoming_coefficients: object,
    selected_coefficients: object,
    selector_specs: tuple[FiniteJetSelectorSpec, ...],
    sample_taus: tuple[float, ...] = (-0.04, 0.04),
    tolerance: float = 1.0e-8,
) -> FiniteJetIdentitySelectorEntryCertificate:
    """Certify that finite one-sided selector jets define the selected branch.

    The coefficients are for a regularized total-collision expansion
    ``q(tau)=sum_d C_d tau^(d+2)``.  Each selector coordinate is recovered from
    an incoming coefficient row after subtracting a constructor-supplied forced
    reference row.  The selected branch is certified when the same coordinates
    appear in the selected coefficient rows, the finite energy limit is
    preserved, and the selected punctured branch satisfies Newton's equations
    with zero angular momentum at the sampled regularized times.
    """

    masses_array, incoming = _validate_finite_jet_inputs(masses, incoming_coefficients)
    selected_masses, selected = _validate_finite_jet_inputs(masses_array, selected_coefficients)
    if not np.array_equal(masses_array.shape, selected_masses.shape):
        raise ValueError("selected masses shape mismatch")
    if incoming.shape != selected.shape:
        raise ValueError("incoming and selected coefficients must have matching shape")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if not selector_specs:
        raise ValueError("at least one selector spec is required")
    coordinate_certificates = []
    for spec in selector_specs:
        recovered_value = _selector_coordinate(masses_array, incoming, spec)
        selected_value = _selector_coordinate(masses_array, selected, spec)
        basis = np.asarray(spec.basis, dtype=float)
        coordinate_certificates.append(
            FiniteJetSelectorCoordinateCertificate(
                name=str(spec.name),
                degree=int(spec.degree),
                regularized_power=int(spec.degree) + 2,
                recovered_value=float(recovered_value),
                selected_value=float(selected_value),
                selector_gap=float(abs(recovered_value - selected_value)),
                basis_norm=float(_mass_inner_product(masses_array, basis, basis)),
            )
        )
    sample_taus = tuple(float(tau) for tau in sample_taus)
    if not sample_taus:
        raise ValueError("at least one punctured sample tau is required")
    newton_bounds = tuple(
        _selected_newton_residual_bound(masses_array, selected, tau)
        for tau in sample_taus
    )
    angular_bounds = tuple(
        _selected_angular_momentum_bound(masses_array, selected, tau)
        for tau in sample_taus
    )
    incoming_energy = _shape_branch_energy_limit(masses_array, incoming)
    selected_energy = _shape_branch_energy_limit(masses_array, selected)
    certificate = FiniteJetIdentitySelectorEntryCertificate(
        masses=tuple(float(mass) for mass in masses_array),
        incoming_energy_limit=float(incoming_energy),
        selected_energy_limit=float(selected_energy),
        energy_gap=float(abs(incoming_energy - selected_energy)),
        coordinate_certificates=tuple(coordinate_certificates),
        required_regularized_jet_order=max(
            certificate.regularized_power for certificate in coordinate_certificates
        ),
        sample_taus=sample_taus,
        newton_residual_bounds=newton_bounds,
        angular_momentum_bounds=angular_bounds,
        tolerance=tolerance,
        incoming_central_coefficient=incoming[0].copy(),
        selected_central_coefficient=selected[0].copy(),
    )
    if not certificate.certified:
        raise ValueError("finite-jet identity selector entry did not certify")
    return certificate


def certify_homothetic_finite_jet_identity_selector_entry(
    branch: HomotheticTotalCollisionBranch,
    *,
    sample_taus: tuple[float, ...] = (-0.04, 0.04),
    tolerance: float = 1.0e-8,
) -> FiniteJetIdentitySelectorEntryCertificate:
    """Derive finite identity-selector data from a homothetic branch.

    The homothetic branch has regularized expansion
    ``q(tau)=tau^2 u(tau^2) Q``.  The finite selector coordinate used here is
    the quartic scale/energy row, recovered by projecting the degree-two shape
    coefficient onto the scaled central configuration ``Q``.
    """

    if not isinstance(branch, HomotheticTotalCollisionBranch):
        raise TypeError("branch must be a HomotheticTotalCollisionBranch")
    masses = np.asarray(branch.masses, dtype=float)
    coefficients = np.zeros(
        (int(branch.order) + 1, *np.asarray(branch.quadratic_coefficient).shape),
        dtype=float,
    )
    for radial_degree, radial_coefficient in enumerate(branch.coefficients):
        coefficient_degree = 2 * radial_degree
        if coefficient_degree < coefficients.shape[0]:
            coefficients[coefficient_degree] = (
                radial_coefficient * np.asarray(branch.quadratic_coefficient, dtype=float)
            )
    selector_spec = FiniteJetSelectorSpec(
        name="homothetic_energy_scale",
        degree=2,
        basis=np.asarray(branch.quadratic_coefficient, dtype=float),
    )
    return certify_finite_jet_identity_selector_entry(
        masses=masses,
        incoming_coefficients=coefficients,
        selected_coefficients=coefficients,
        selector_specs=(selector_spec,),
        sample_taus=sample_taus,
        tolerance=tolerance,
    )


def _validate_finite_jet_inputs(masses: object, coefficients: object) -> tuple[Array, Array]:
    masses_array = np.asarray(masses, dtype=float).reshape(-1)
    coefficients_array = np.asarray(coefficients, dtype=float)
    if masses_array.shape != (3,):
        raise ValueError("finite zero-angular entry currently expects three masses")
    if np.any(masses_array <= 0.0) or not np.all(np.isfinite(masses_array)):
        raise ValueError("masses must be positive and finite")
    if coefficients_array.ndim != 3 or coefficients_array.shape[1] != 3:
        raise ValueError("coefficients must have shape (order, 3, dimension)")
    if not np.all(np.isfinite(coefficients_array)):
        raise ValueError("coefficients must be finite")
    if _minimum_pair_distance(coefficients_array[0]) <= 0.0:
        raise ValueError("central coefficient must be collision-free")
    return masses_array, coefficients_array


def _validate_total_collision_leading_shape(
    masses: object,
    leading_shape: object,
) -> tuple[Array, Array]:
    masses_array = np.asarray(masses, dtype=float).reshape(-1)
    leading = np.asarray(leading_shape, dtype=float)
    if masses_array.shape != (3,):
        raise ValueError("cubic-time total-collision entry expects three masses")
    if np.any(masses_array <= 0.0) or not np.all(np.isfinite(masses_array)):
        raise ValueError("masses must be positive and finite")
    if leading.ndim != 2 or leading.shape[0] != 3 or leading.shape[1] not in (2, 3):
        raise ValueError("quadratic_coefficient must have shape (3, 2) or (3, 3)")
    if not np.all(np.isfinite(leading)):
        raise ValueError("quadratic_coefficient must be finite")
    return masses_array, leading


def _cubic_time_leading_certificate_matches_inputs(
    certificate: CubicTimeLeadingJetEntryCertificate,
    masses: Array,
    leading_shape: Array,
    *,
    tolerance: float,
) -> bool:
    try:
        certificate_masses = np.asarray(certificate.masses, dtype=float).reshape(-1)
        certificate_shape = np.asarray(
            certificate.quadratic_coefficient,
            dtype=float,
        )
    except (AttributeError, TypeError, ValueError):
        return False
    if certificate_masses.shape != masses.shape or certificate_shape.shape != leading_shape.shape:
        return False
    scale = max(
        1.0,
        float(np.linalg.norm(masses, ord=np.inf)),
        float(np.linalg.norm(certificate_masses, ord=np.inf)),
        float(np.linalg.norm(leading_shape, ord=np.inf)),
        float(np.linalg.norm(certificate_shape, ord=np.inf)),
    )
    return bool(
        np.allclose(certificate_masses, masses, rtol=0.0, atol=tolerance * scale)
        and np.allclose(certificate_shape, leading_shape, rtol=0.0, atol=tolerance * scale)
    )


def _positive_finite_tolerance(tolerance: float) -> float:
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    return tolerance


def _mass_weighted_center(masses: Array, positions: Array) -> Array:
    return np.sum(masses[:, None] * positions, axis=0) / float(np.sum(masses))


def _optional_central_coefficient_certified(coefficient: object | None) -> bool:
    if coefficient is None:
        return True
    coefficient_array = np.asarray(coefficient, dtype=float)
    return bool(
        coefficient_array.ndim == 2
        and coefficient_array.shape[0] == 3
        and np.all(np.isfinite(coefficient_array))
        and _minimum_pair_distance(coefficient_array) > 0.0
    )


def _finite_jet_coordinate_certificates_certified(
    coordinate_certificates: tuple[object, ...],
) -> bool:
    return bool(
        coordinate_certificates
        and all(
            isinstance(certificate, FiniteJetSelectorCoordinateCertificate)
            and certificate.certified is True
            for certificate in coordinate_certificates
        )
    )


def _finite_jet_selector_specs_certified(
    selector_specs: tuple[object, ...],
    *,
    coefficient_shape: tuple[int, ...],
) -> bool:
    for spec in selector_specs:
        if not isinstance(spec, FiniteJetSelectorSpec):
            return False
        if not str(spec.name) or int(spec.degree) < 0:
            return False
        basis = np.asarray(spec.basis, dtype=float)
        if basis.shape != coefficient_shape or not np.all(np.isfinite(basis)):
            return False
        if spec.reference_coefficients is not None:
            reference = np.asarray(spec.reference_coefficients, dtype=float)
            reference_is_row = reference.shape == coefficient_shape
            reference_is_full_coefficient_array = (
                reference.ndim == len(coefficient_shape) + 1
                and reference.shape[1:] == coefficient_shape
                and int(spec.degree) < reference.shape[0]
            )
            if (
                not (reference_is_row or reference_is_full_coefficient_array)
                or not np.all(np.isfinite(reference))
            ):
                return False
    return bool(selector_specs)


def _selector_coordinate(
    masses: Array,
    coefficients: Array,
    spec: FiniteJetSelectorSpec,
) -> float:
    degree = int(spec.degree)
    if degree < 0 or degree >= coefficients.shape[0]:
        raise ValueError("selector degree is outside the coefficient array")
    basis = np.asarray(spec.basis, dtype=float)
    if basis.shape != coefficients.shape[1:]:
        raise ValueError("selector basis shape must match one coefficient row")
    reference = _reference_row(spec, coefficients.shape)
    denominator = _mass_inner_product(masses, basis, basis)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("selector basis must have positive mass norm")
    return float(_mass_inner_product(masses, coefficients[degree] - reference, basis) / denominator)


def _reference_row(spec: FiniteJetSelectorSpec, coefficient_shape: tuple[int, ...]) -> Array:
    degree = int(spec.degree)
    if spec.reference_coefficients is None:
        return np.zeros(coefficient_shape[1:], dtype=float)
    reference = np.asarray(spec.reference_coefficients, dtype=float)
    if reference.shape == coefficient_shape[1:]:
        return reference
    if reference.shape == coefficient_shape:
        return reference[degree]
    raise ValueError("reference coefficients must be a row or full coefficient array")


def _shape_series_state(coefficients: Array, tau: float) -> tuple[Array, Array]:
    tau = float(tau)
    if tau == 0.0:
        raise ValueError("state is singular at total collision")
    positions = sum(
        coefficient * tau ** (degree + 2)
        for degree, coefficient in enumerate(coefficients)
    )
    velocities = sum(
        ((degree + 2.0) / 3.0) * coefficient * tau ** (degree - 1)
        for degree, coefficient in enumerate(coefficients)
    )
    return np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)


def _cubic_time_polynomial_acceleration(coefficients: Array, tau: float) -> Array:
    tau = float(tau)
    acceleration = np.zeros_like(coefficients[0], dtype=float)
    for degree, coefficient in enumerate(coefficients):
        power = degree + 2.0
        acceleration += (
            power * (power - 3.0) / 9.0
        ) * coefficient * tau ** (power - 6.0)
    return acceleration


def _selected_newton_residual_bound(masses: Array, coefficients: Array, tau: float) -> float:
    positions, _velocities = _shape_series_state(coefficients, tau)
    residual = _cubic_time_polynomial_acceleration(coefficients, tau) - accelerations(
        positions,
        masses,
    )
    return float(np.linalg.norm(residual, ord=np.inf))


def _selected_angular_momentum_bound(masses: Array, coefficients: Array, tau: float) -> float:
    positions, velocities = _shape_series_state(coefficients, tau)
    dimension = positions.shape[1]
    components = []
    for left in range(dimension):
        for right in range(left + 1, dimension):
            components.append(
                float(
                    np.sum(
                        masses
                        * (
                            positions[:, left] * velocities[:, right]
                            - positions[:, right] * velocities[:, left]
                        )
                    )
                )
            )
    return float(max((abs(component) for component in components), default=0.0))


def _shape_branch_energy_limit(masses: Array, coefficients: Array) -> float:
    if coefficients.shape[0] < 3:
        raise ValueError("energy limit requires at least quadratic, cubic, and quartic rows")
    central_shape = coefficients[0]
    cubic_shape = coefficients[1]
    quartic_shape = coefficients[2]
    linearized_cubic = _acceleration_derivative_apply(
        central_shape,
        masses,
        cubic_shape,
    )
    return float(
        0.5 * _mass_inner_product(masses, cubic_shape, cubic_shape)
        - 0.5 * _mass_inner_product(masses, cubic_shape, linearized_cubic)
        + (10.0 / 9.0) * _mass_inner_product(masses, central_shape, quartic_shape)
    )


def _acceleration_derivative_apply(positions: Array, masses: Array, perturbation: Array) -> Array:
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


def _mass_inner_product(masses: Array, left: Array, right: Array) -> float:
    return float(np.sum(masses[:, None] * left * right))


def _acceleration_derivative_matrix(positions: Array, masses: Array) -> Array:
    matrix = np.zeros((positions.size, positions.size), dtype=float)
    for coordinate_index in range(positions.size):
        perturbation = np.zeros_like(positions, dtype=float)
        perturbation.reshape(-1)[coordinate_index] = 1.0
        matrix[:, coordinate_index] = _acceleration_derivative_apply(
            positions,
            masses,
            perturbation,
        ).reshape(-1)
    return matrix


def _series_product(left: Array, right: Array, order: int) -> Array:
    product = np.zeros(order + 1, dtype=float)
    for degree in range(order + 1):
        product[degree] = sum(left[k] * right[degree - k] for k in range(degree + 1))
    return product


def _series_power_one_plus(series: Array, exponent: float, order: int) -> Array:
    result = np.zeros(order + 1, dtype=float)
    result[0] = 1.0
    term = np.zeros(order + 1, dtype=float)
    term[0] = 1.0
    binomial_coefficient = 1.0
    for power in range(1, order + 1):
        term = _series_product(term, series, order)
        binomial_coefficient *= (exponent - power + 1.0) / power
        result += binomial_coefficient * term
    return result


def _shape_acceleration_series(
    shape_coefficients: Array,
    masses: Array,
    order: int,
) -> Array:
    coefficients = np.asarray(shape_coefficients, dtype=float)
    acceleration_coefficients = np.zeros(
        (order + 1, coefficients.shape[1], coefficients.shape[2]),
        dtype=float,
    )
    for body in range(coefficients.shape[1]):
        for other in range(coefficients.shape[1]):
            if body == other:
                continue
            relative_coefficients = coefficients[: order + 1, other, :] - coefficients[
                : order + 1, body, :
            ]
            squared_distance = np.zeros(order + 1, dtype=float)
            for degree in range(order + 1):
                squared_distance[degree] = sum(
                    float(
                        np.dot(
                            relative_coefficients[left_degree],
                            relative_coefficients[degree - left_degree],
                        )
                    )
                    for left_degree in range(degree + 1)
                )
            if squared_distance[0] <= 0.0:
                raise ValueError("central coefficient has a collision")
            normalized_tail = np.zeros(order + 1, dtype=float)
            normalized_tail[1:] = squared_distance[1:] / squared_distance[0]
            inverse_distance_cubed = squared_distance[0] ** (-1.5) * _series_power_one_plus(
                normalized_tail,
                -1.5,
                order,
            )
            pair_force = np.zeros((order + 1, coefficients.shape[2]), dtype=float)
            for degree in range(order + 1):
                pair_force[degree] = sum(
                    relative_coefficients[left_degree]
                    * inverse_distance_cubed[degree - left_degree]
                    for left_degree in range(degree + 1)
                )
            acceleration_coefficients[:, body, :] += masses[other] * pair_force
    return acceleration_coefficients


def _apply_selector_rows(
    masses: Array,
    coefficients: Array,
    degree: int,
    selector_specs: tuple[FiniteJetSelectorSpec, ...],
    selected_values: Mapping[str, float],
) -> Array:
    row = np.asarray(coefficients[degree], dtype=float).copy()
    bases = tuple(np.asarray(spec.basis, dtype=float) for spec in selector_specs)
    for basis in bases:
        if basis.shape != row.shape:
            raise ValueError("selector basis shape must match one coefficient row")
    gram = np.array(
        [
            [_mass_inner_product(masses, left, right) for right in bases]
            for left in bases
        ],
        dtype=float,
    )
    rhs = []
    for spec, basis in zip(selector_specs, bases, strict=True):
        reference = _reference_row(spec, coefficients.shape)
        norm = _mass_inner_product(masses, basis, basis)
        rhs.append(
            float(selected_values[str(spec.name)]) * norm
            - _mass_inner_product(masses, row - reference, basis)
        )
    correction_coefficients, *_ = np.linalg.lstsq(gram, np.asarray(rhs), rcond=None)
    for coefficient, basis in zip(correction_coefficients, bases, strict=True):
        row += float(coefficient) * basis
    return row


def _selector_operator_residual_bounds(
    derivative_matrix: Array,
    selector_specs: tuple[FiniteJetSelectorSpec, ...],
    coefficient_row_shape: tuple[int, ...],
) -> tuple[tuple[str, float], ...]:
    residuals = []
    dimension = int(np.prod(coefficient_row_shape))
    for spec in selector_specs:
        degree = int(spec.degree)
        multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
        operator = multiplier * np.eye(dimension) - derivative_matrix
        basis = np.asarray(spec.basis, dtype=float)
        if basis.shape != coefficient_row_shape:
            raise ValueError("selector basis shape must match one coefficient row")
        residuals.append(
            (
                str(spec.name),
                float(np.linalg.norm(operator @ basis.reshape(-1), ord=np.inf)),
            )
        )
    return tuple(residuals)


def _selector_gram_singular_value_floors(
    masses: Array,
    selector_specs: tuple[FiniteJetSelectorSpec, ...],
) -> tuple[tuple[int, float], ...]:
    floors = []
    for degree in sorted({int(spec.degree) for spec in selector_specs}):
        specs = tuple(spec for spec in selector_specs if int(spec.degree) == degree)
        bases = tuple(np.asarray(spec.basis, dtype=float) for spec in specs)
        gram = np.array(
            [
                [_mass_inner_product(masses, left, right) for right in bases]
                for left in bases
            ],
            dtype=float,
        )
        singular_values = np.linalg.svd(gram, compute_uv=False)
        floors.append((degree, float(np.min(singular_values))))
    return tuple(floors)


def _coefficient_recurrence_residual_bounds(
    masses: Array,
    coefficients: Array,
) -> tuple[float, ...]:
    order = coefficients.shape[0] - 1
    acceleration_coefficients = _shape_acceleration_series(coefficients, masses, order)
    residuals = []
    for degree in range(order + 1):
        multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
        residual = multiplier * coefficients[degree] - acceleration_coefficients[degree]
        residuals.append(float(np.linalg.norm(residual, ord=np.inf)))
    return tuple(residuals)


def _minimum_pair_distance(positions: Array) -> float:
    return float(
        min(
            np.linalg.norm(positions[j] - positions[i])
            for i in range(positions.shape[0])
            for j in range(i + 1, positions.shape[0])
        )
    )
